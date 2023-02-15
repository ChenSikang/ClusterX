from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import dgl
from odc import ODC
from openselfsup.third_party import clustering as _clustering
from torch.utils.data import ConcatDataset, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from PotentialNet import PotentialNetParallel, GraphThreshold
from data_utils import DockingDataset
from tqdm import tqdm
from scipy import stats
from mmcv.utils import print_log
from mmcv.runner import HOOKS, Hook
import torch.distributed as dist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.optim import Adam, lr_scheduler, SGD
from torch.nn import init
from torch import nn
import torch
import pandas as pd
import numpy as np
import random
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
# NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = "0"


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def collate_fn(batch):
    batch = [x for x in batch if x is not None]

    graphs, graphs3, idx = map(list, zip(*batch))

    g = dgl.batch(graphs)
    # g1 = dgl.batch(graphs1)
    # g2 = dgl.batch(graphs2)
    g3 = dgl.batch(graphs3)

    # g.lable_index = idx

    return g, g3, idx


class Extractor(object):
    def __init__(self,
                 dataset,
                 batch_size,
                 dist_mode=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = DataLoaderX(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=True)
        self.dist_mode = dist_mode
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, model, g, g3, idx):
        backbone_feat = model(g, g3, idx, mode='extract')
        last_layer_feat = model.neck([backbone_feat])[0]
        # last_layer_feat = model.neck([backbone_feat[-1]])[0]
        # print('last_layer_feat.size',last_layer_feat.size()) [xx, 64]
        last_layer_feat = last_layer_feat.view(1, -1)
        return last_layer_feat.cpu()

    def __call__(self, model):
        def func(x, y, z): return self._forward_func(model, x, y, z)
        feats = nondist_forward_collect(func, self.data_loader,
                                        len(self.data_loader.dataset) - len(self.data_loader.dataset) % self.batch_size)['feature']
        return feats


def nondist_forward_collect(func, data_loader, length):
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        length (int): Expected length of output arrays.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    results = []
    my_len = []
    # errors = []
    for i, batch in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            bg, bg3, idx = batch
            # print('idx',idx)
            # print(g, g3)
            # print(g3.num_edges())
            g, g3 = bg.to(device), bg3.to(device)
            result = func(g, g3, idx)

        # results.append(dict(feature=result.numpy().tolist()))

        my_len.append(result.numpy().shape[1])  # 1
        results.append(result.numpy().tolist())

        # results.append(result.tolist())

        # if np.any(np.isnan(result.numpy)):
        #     errors.append([bg[0], bg3[0]])
        # else:
        #     my_len.append(result.numpy().shape[0])
        #     results.append(result.numpy().tolist())

    # results = sum(results,[])
    # np.save(f'./error.npy', errors)
    # print('len(errors)', len(errors))
    # print('result', result.tolist())
    print('最后一个result形状', result.shape, len(results[0]))
    print('results长度', len(results))

    # my_len = [len(i[0]) for k in range(len(results)) for i in results[k]]

    max_num = max(my_len)
    print('max_num', max_num)
    new_results = []
    # zerolist = [0]*64
    for result in results:
        if len(result[0]) < max_num:
            for j in range(max_num - len(result[0])):
                result[0].append(0)
                # result.append(zerolist)
        # print('len(result)', len(result))
        new_results.append(dict(feature=result))
    print('补零后results维度', np.array(results).shape)

    # print('len(new_results),len(new_results[0])', len(
    #     new_results), len(new_results[0]))

    results_all = {}
    for k in new_results[0].keys():
        results_all[k] = np.concatenate(
            [np.array(batch[k]) for batch in new_results], axis=0).astype(np.float32)
        print('results_all[k].dtype', results_all['feature'].dtype)
        print('results_all[k].shape', results_all['feature'].shape)
        print('length', length)
        assert results_all[k].shape[0] == length

    # print(results[0]['feature'].numpy().shape)
    # print(len(results))
    # print(len(results[0].keys()))
    # results_all = {}
    # for k in new_result[0].keys():
    #     print(k)
    #     results_all[k] = np.concatenate(
    #         [batch[k] for batch in new_result], axis=0)
    #     print('results_all[k]',results_all[k])
    #     print('results_all[k].shape',results_all[k].shape)
    #     print('length',length)
    #     assert results_all[k].shape[0] == length
    return results_all


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def collate_fn_none_filter(batch):
    return [x for x in batch if x is not None]


def odc_evaluate(new_labels, num_classes):
    hist = np.bincount(
        new_labels, minlength=num_classes)
    empty_cls = (hist == 0).sum()
    minimal_cls_size, maximal_cls_size = hist.min(), hist.max()
    tqdm.write("ODC_eval:empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
        empty_cls.item(), minimal_cls_size.item(),
        maximal_cls_size.item()))


@HOOKS.register_module()
class DeepClusterHook():
    def __init__(
            self,
            extractor,
            clustering,
            unif_sampling,
            reweight,
            reweight_pow,
            init_memory=False,  # for ODC
            initial=True,
            data_loaders=None):
        self.extractor = Extractor(**extractor)
        self.clustering_type = clustering.pop('type')
        self.clustering_cfg = clustering
        self.unif_sampling = unif_sampling
        self.reweight = reweight
        self.reweight_pow = reweight_pow
        self.init_memory = init_memory
        self.initial = initial
        self.data_loaders = data_loaders

    def deepcluster(self, model):
        # step 1: get features
        model.eval()
        features = self.extractor(model)
        model.train()

        # step 2: get labels
        clustering_algo = _clustering.__dict__[self.clustering_type](
            **self.clustering_cfg)
        # Features are normalized during clustering
        clustering_algo.cluster(features, verbose=True)
        assert isinstance(clustering_algo.labels, np.ndarray)
        new_labels = clustering_algo.labels.astype(np.int64)
        self.evaluate(new_labels)

        new_labels_list = list(new_labels)

        # step 3: assign new labels
        self.data_loaders[0].dataset.assign_labels(new_labels_list)

        # step 4 (a): set uniform sampler
        if self.unif_sampling:
            self.data_loaders[0].sampler.set_uniform_indices(
                new_labels_list, self.clustering_cfg['k'])

        # step 4 (b): set loss reweight
        if self.reweight:
            model.set_reweight(new_labels, self.reweight_pow)

        # step 5: randomize classifier
        model.head.init_weights(init_linear='normal')

        # step 6: init memory for ODC
        if self.init_memory:
            model.memory_bank.init_memory(features, new_labels)

    def evaluate(self, new_labels):
        hist = np.bincount(new_labels, minlength=self.clustering_cfg['k'])
        empty_cls = (hist == 0).sum()
        minimal_cls_size, maximal_cls_size = hist.min(), hist.max()
        print(
            "Hook_eval:empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
                empty_cls.item(), minimal_cls_size.item(),
                maximal_cls_size.item())
        )


def train(args):
    # only for init

    iteration = 0
    # train_dataloader = DataLoader(  # merges data objects from a torch_geometric.data.dataset to a python list
    #     DockingDataset(data_file=data_dir),
    #     batch_size=args.batch_size,  # 8
    #     shuffle=False,
    #     drop_last=True,
    # )  # just to keep batch sizes even, since shuffling is used
    train_dataloader = DataLoaderX(
        DockingDataset(data_file=args.data_dir),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
    )
    print(f'total sample: {len(train_dataloader.dataset.ligands)}')

    model = ODC(node_feat_size=args.node_feat_size,
                edge_feat_size=args.edge_feat_size_3d,
                num_layers=args.num_layers,
                graph_feat_size=args.graph_feat_size,
                outdim_g3=args.outdim_g3,
                dropout=args.dropout,
                num_classes=args.num_classes,
                min_class=args.min_class,
                feat_dim=75392,
                total_samples=len(train_dataloader.dataset) - len(train_dataloader.dataset) % args.batch_size)

    dc_hook = DeepClusterHook(
        extractor=dict(
            dataset=DockingDataset(data_file=args.data_dir),
            batch_size=args.batch_size),
        clustering=dict(type='Kmeans', k=args.num_classes, pca_dim=-1),
        unif_sampling=False,
        reweight=True,
        reweight_pow=0.5,
        init_memory=True,
        initial=True,
        data_loaders=[train_dataloader, ]
    )

    model.train()
    model.to(0)
    tqdm.write(str(model))
    tqdm.write("{} trainable parameters.".format(sum(p.numel()
               for p in model.parameters() if p.requires_grad)))
    tqdm.write("{} total parameters.".format(sum(p.numel()
               for p in model.parameters())))

    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001,
    #                 nesterov=False)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

    if dc_hook.initial:
        dc_hook.deepcluster(model)

    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(train_dataloader):

            g, g3, idx = batch
            g, g3 = g.to(device), g3.to(device)

            optimizer.zero_grad()

            # data = [x[2] for x in batch]
            loss = model(g, g3, idx)
            tqdm.write('cluster changing ratio {}, CEloss {}, acc:{}'.
                       format(loss['change_ratio'], loss['loss'], loss['acc']))
            loss['loss'].backward()
            iteration += 1
            if iteration % args.centroids_update_interval == 0:

                # # 尝试更新样本-----无效
                # idx = batch.label_index
                # idx = idx % model.total_samples
                # idx = torch.LongTensor(idx)
                # x = model.forward_backbone(batch)
                # feature = model.neck(x)
                # model.memory_bank.update_samples_memory(idx, feature[0].detach())

                model.memory_bank.update_centroids_memory()
            if iteration % args.deal_with_small_clusters_interval == 0:
                model.memory_bank.deal_with_small_clusters()
            model.set_reweight()
            if iteration % args.evaluate_interval == 0:
                new_labels = model.memory_bank.label_bank
                if new_labels.is_cuda:
                    new_labels = new_labels.cpu()
                odc_evaluate(new_labels.numpy(), model.memory_bank.num_classes)
        if epoch % args.save_label_interval == 0:
            new_labels = model.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            np.save(
                "{}/cluster_epoch_{}.npy".format(args.work_dir,
                                                 epoch),
                new_labels.numpy())


def validate(model, val_dataloader):
    model.eval()

    y_true = []
    y_pred = []
    pdbid_list = []
    pose_list = []

    for batch in tqdm(val_dataloader):
        data = [x[2] for x in batch if x is not None]
        y_ = model(data)
        y = torch.cat([x[2].y for x in batch])

        pdbid_list.extend([x[0] for x in batch])
        pose_list.extend([x[1] for x in batch])
        y_true.append(y.cpu().data.numpy())
        y_pred.append(y_.cpu().data.numpy())

    y_true = np.concatenate(y_true).reshape(-1, 1)
    y_pred = np.concatenate(y_pred).reshape(-1, 1)

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
    spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

    tqdm.write(
        str(
            "r2: {}\tmae: {}\tmse: {}\tpearsonr: {}\t spearmanr: {}".format(
                r2, mae, mse, pearsonr, spearmanr
            )
        )
    )
    model.train()
    return {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "pearsonr": pearsonr,
        "spearmanr": spearmanr,
        "y_true": y_true,
        "y_pred": y_pred,
        "pdbid": pdbid_list,
        "pose": pose_list,
    }


def checkpoint_model(model, epoch, step, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    model.train()

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
    }

    torch.save(checkpoint_dict, output_path)

    # return the computed metrics so it can be used to update the training loop
    return checkpoint_dict


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('start!')
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--data_dir', type=str,
                        default='/home/sikang/cluster-AE/dataset/output/test_ign_data_3D.npy')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_iter', type=int, default=10000)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/sikang/cluster-AE/ODC/saveMod/')
    parser.add_argument('--centroids_update_interval', type=int, default=1)
    parser.add_argument(
        '--deal_with_small_clusters_interval', type=int, default=1)
    parser.add_argument('--evaluate_interval', type=int, default=1)
    parser.add_argument('--save_label_interval', type=int, default=2)
    parser.add_argument('--dcHook_interval', type=int, default=2)
    parser.add_argument('--work_dir', type=str,
                        default='/home/sikang/cluster-AE/ODC/work_dir/')
    parser.add_argument('--num_classes', type=int, default=75)
    parser.add_argument('--min_class', type=int, default=100)
    # both acsf feature and basic atom feature
    parser.add_argument('--node_feat_size', type=int, default=40)
    parser.add_argument('--edge_feat_size_2d', type=int, default=12)
    parser.add_argument('--edge_feat_size_3d', type=int, default=21)
    parser.add_argument('--graph_feat_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3,
                        help='the number of intra-molecular layers')
    parser.add_argument('--outdim_g3', type=int, default=200,
                        help='the output dim of inter-molecular layers')
    parser.add_argument('--dropout', type=float,
                        default=0.25, help='dropout ratio')

    args = parser.parse_args()
    print(args)
    train(args)

    # # '利用SSE选择k'
    # SSE = []  # 存放每次结果的误差平方和
    # for i in range(1, 8): #尝试要聚成的类数
    #     estimator = KMeans(n_clusters=i)  # 构造聚类器
    #     estimator.fit(np.array(mdl[['0', '1', '2','3','4','5','6','7']]))
    #     SSE.append(estimator.inertia_)
    # X = range(1, 8) #跟k值要一样
    # plt.xlabel('i')
    # plt.ylabel('SSE')
    # plt.plot(X, SSE, 'o-')
    # plt.show() #画出图
