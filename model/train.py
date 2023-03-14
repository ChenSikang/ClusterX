import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import dgl
from torch import nn
from torch.nn import init
from torch.optim import Adam, lr_scheduler, SGD
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import DockingDataset, collate, clustering_performance_evaluation, rand_hyperparams, set_random_seed
from openselfsup.third_party import clustering as _clustering
from odc import ODC
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from ray import tune
matplotlib.use('agg')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
# NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = "0"


class Extractor(object):
    def __init__(self,
                 dataset,
                 batch_size,
                 dist_mode=False):
        self.dataset = dataset
        self.data_loader = DataLoader(
            dataset=DockingDataset(data_file=config['args'].data_dir),
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=False,
            drop_last=False,
            pin_memory=True,)
        self.dist_mode = dist_mode
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, model, lg, bg1, bg2, idx):
        batch_num_nodes = bg1.batch_num_nodes()
        backbone_feat = model(lg, bg1, bg2, idx, mode='extract')
        # last_layer_feat = model.neck([backbone_feat[-1]])[0]
        last_layer_feat = model.neck([backbone_feat], batch_num_nodes)[0]
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1)
        return dict(feature=last_layer_feat.cpu())

    def __call__(self, model):
        def func(x, y, z, w): return self._forward_func(model, x, y, z, w)
        feats = nondist_forward_collect(func, self.data_loader,
                                        len(self.dataset))['feature']
        return feats


def split_features(features, batch_num_nodes):
    node_nums = torch.cumsum(batch_num_nodes, dim=0)
    ligand_idx = [list(range(node_nums[0]))]  # first ligand
    for i in range(1, len(node_nums)):  # the rest of ligands in the batch
        ligand_idx.append(list(range(node_nums[i-1], node_nums[i])))

    # print('features.size()', features.size())
    # print('ligand_idx', ligand_idx)
    return [[features[i, ].numpy().tolist()] for i in ligand_idx]


def nondist_forward_collect(func, data_loader, length):
    results = []
    for i, batch in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            lg, bg1, bg2, idx = batch
            lg, bg1, bg2 = lg.to(device), bg1.to(device), bg2 .to(device)
            result = func(lg, bg1, bg2, idx)
        results.append(result)

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


def nondist_forward_collect_v0(func, data_loader, length):
    results, my_len = [], []
    for i, batch in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            lg, bg1, bg2, idx = batch
            batch_num_nodes = lg.batch_num_nodes()
            lg, bg1, bg2 = lg.to(device), bg1.to(device), bg2 .to(device)
            result = func(lg, bg1, bg2, idx)
            features = split_features(result, batch_num_nodes)
        for feature in features:
            my_len.append(len(feature[0]))
            results.append(feature)

    # print('最后一个feature形状', np.array(feature).shape)
    # print('results形状', len(results), np.array(results[0]).shape)

    max_num = max(my_len)
    # print('max_num', max_num)
    new_results = []
    zerolist = [0]*config['args'].n_rows_fc[-1]
    for result in results:
        if len(result[0]) < max_num:
            for j in range(max_num - len(result[0])):
                result[0].append(zerolist)
        new_results.append(dict(feature=result))
    # print('补零后results维度', len(results), np.array(results[0]).shape)

    results_all = {}
    for k in new_results[0].keys():
        results_all[k] = np.concatenate(
            [np.array(batch[k]) for batch in new_results], axis=0).astype(np.float32)
        print('results_all[k]形状, length', results_all['feature'].shape, length)
        assert results_all[k].shape[0] == length

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
    tqdm.write("empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
        empty_cls.item(), minimal_cls_size.item(),
        maximal_cls_size.item()))


class DeepClusterHook():
    def __init__(
            self,
            extractor,
            clustering,
            num_classes,
            unif_sampling,
            reweight,
            reweight_pow,
            init_memory=False,  # for ODC
            initial=True,
            data_loaders=None):
        self.extractor = Extractor(**extractor)
        self.clustering_type = clustering.pop('type')
        self.clustering_cfg = clustering
        self.num_classes = num_classes
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

        X_tsne = TSNE(n_components=2,random_state=33).fit_transform(features)
        X_pca = PCA(n_components=2).fit_transform(features)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=20, c=new_labels_list, label="t-SNE")
        plt.legend()
        plt.subplot(122)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=20, c=new_labels_list, label="PCA")
        plt.legend()
        plt.savefig(f'/home/sikang/cluster-AE/odc-pnet/run/result/notume_tsne-pca_k={self.num_classes}.png', dpi=300)

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
            "empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
                empty_cls.item(), minimal_cls_size.item(),
                maximal_cls_size.item())
        )


def train(config):
    # only for init
    # if config['f_spatial'] < config['f_gather']:
    #     config['f_spatial']=config['f_gather']

    iteration = 0
    train_dataloader = DataLoader(dataset=DockingDataset(data_file=config['args'].data_dir),
                                  batch_size=config['args'].batch_size,
                                  collate_fn=collate,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  )
    print(f'total sample: {len(train_dataloader.dataset.ligands)}')

    model = ODC(pretrain_path=config['args'].pretrain_path,
                total_samples=len(train_dataloader.dataset),
                num_classes=config['args'].num_classes,
                min_cluster=config['args'].min_cluster,
                n_etypes=len(config['args'].distance_bins)+5,
                f_in=config['args'].f_in,
                f_bond=config['args'].f_bond,
                f_spatial=config['args'].f_spatial,
                f_gather=config['args'].f_gather,
                n_rows_fc=config['args'].n_rows_fc,
                n_bond_conv_steps=config['args'].n_bond_conv_steps,
                n_spatial_conv_steps=config['args'].n_spatial_conv_steps,
                dropouts=config['args'].dropouts,
                momentum=0.5)

    dc_hook = DeepClusterHook(
        extractor=dict(
            batch_size=config['args'].batch_size,
            dataset=DockingDataset(data_file=config['args'].data_dir)
        ),
        clustering=dict(type='Kmeans', k=config['args'].num_classes, pca_dim=12),
        num_classes=config['args'].num_classes,
        unif_sampling=False,
        reweight=True,
        reweight_pow=0.5,
        init_memory=True,
        initial=True,
        data_loaders=[train_dataloader, ]
    )

    model.train()
    model.to(device)
    tqdm.write(str(model))
    tqdm.write(
        "{} trainable parameters."
        .format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    tqdm.write(
        "{} total parameters."
        .format(sum(p.numel() for p in model.parameters())))

    optimizer = SGD(model.parameters(), lr=config['lr'], momentum=0.9,
                    weight_decay=config['args'].weight_decay, nesterov=False)
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if dc_hook.initial:
        dc_hook.deepcluster(model)

    losses, cls_loss, rec_loss = [], [], []
    NMI = np.zeros(config['args'].epochs)
    for epoch in range(config['args'].epochs):
        losses = []
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()

            lg, bg1, bg2, idx = batch
            lg, bg1, bg2 = lg.to(device), bg1.to(device), bg2 .to(device)

            loss = model(lg, bg1, bg2, idx)
            losses.append(loss['loss'].cpu().detach().numpy())
            cls_loss.append(loss['cls_loss'])
            rec_loss.append(loss['rec_loss'])
            # tqdm.write('loss {}, cls_loss {}, rec_loss {}'.
            #            format(loss['loss'], loss['cls_loss'], loss['rec_loss']))
            # tqdm.write('cluster changing ratio: {}, ACC: {}'.
            #            format(loss['change_ratio'], loss['acc']))
            loss['loss'].backward()
            iteration += 1
            if iteration % config['args'].centroids_update_interval == 0:
                model.memory_bank.update_centroids_memory()
            # if iteration % config['args'].deal_with_small_clusters_interval == 0:
            #     model.memory_bank.deal_with_small_clusters()
            model.set_reweight()
            if iteration % config['args'].evaluate_interval == 0:
                new_labels = model.memory_bank.label_bank
                if new_labels.is_cuda:
                    new_labels = new_labels.cpu()
                odc_evaluate(new_labels.numpy(), model.memory_bank.num_classes)
        
        mean_loss = np.mean(losses)
        tune.report(my_loss=mean_loss)

        if epoch % config['args'].save_label_interval == 0:
            new_labels = model.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            np.save(
                "{}/cluster_epoch_{}.npy".format(config['args'].work_dir, epoch),
                new_labels.numpy())

        y_target = model.memory_bank.label_bank.cpu().detach()
        if epoch>0:
            p_labels = np.load("{}/cluster_epoch_{}.npy".format(config['args'].work_dir, epoch-1))
        else:
            p_labels = np.load("{}/cluster_epoch_{}.npy".format(config['args'].work_dir, epoch))
        NMI[epoch] = metrics.adjusted_mutual_info_score(p_labels, y_target)
        tqdm.write('epoch {}/{}, NMI: {}'.
                   format(epoch+1, config['args'].epochs, NMI[epoch]))

    checkpoint_dict = {'stage1': model.stage_1_model.state_dict(
    ), 'stage2': model.stage_2_model.state_dict(), 'neck': model.neck.state_dict()}
    torch.save(
        checkpoint_dict, f"/home/sikang/cluster-AE/odc-pnet/train/train_epoch{epoch}.pkl")
    
    clusters = np.load("{}/cluster_epoch_{}.npy".format(config['args'].work_dir, epoch))
    
    return clusters

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='train model')

    # parser.add_argument('--data_dir', type=str,
    #                     default='/home/sikang/cluster-AE/dataset/output/pn_data.npy')
    parser.add_argument('--data_dir', type=str,
                        default='/home/sikang/cluster-AE/dataset/output/pn_testset_notume.npy')
    parser.add_argument('--pretrain_path', type=str,
                        default='/home/sikang/cluster-AE/odc-pnet/pretrain/pretrain_trial0_epoch1999.pkl')
    parser.add_argument('--work_dir', type=str,
                        default='/home/sikang/cluster-AE/odc-pnet/work_dir')
    parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_iter', type=int, default=10000)
    parser.add_argument('--checkpoint_dir', type=str, default='./saveMod')

    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--dropouts', type=list, default=[0.25, 0.25, 0.25])
    parser.add_argument('--weight_decay', type=int, default=1e-05)

    parser.add_argument('--max_num_neighbors', type=int, default=5)
    parser.add_argument('--distance_bins', type=list,
                        default=[1.5, 2.5, 3.5, 4.5])
    parser.add_argument('--f_in', type=int, default=44)
    # has to be larger than f_in
    parser.add_argument('--f_bond', type=int, default=48)
    parser.add_argument('--f_gather', type=int, default=48)
    # better to be the same as f_gather
    parser.add_argument('--f_spatial', type=int, default=48)
    parser.add_argument('--n_rows_fc', type=list, default=[48, 24])
    parser.add_argument('--n_bond_conv_steps', type=int, default=2)
    parser.add_argument('--n_spatial_conv_steps', type=int, default=1)

    parser.add_argument('--centroids_update_interval', type=int, default=1)
    parser.add_argument(
        '--deal_with_small_clusters_interval', type=int, default=1)
    parser.add_argument('--evaluate_interval', type=int, default=1)
    parser.add_argument('--save_label_interval', type=int, default=1)
    parser.add_argument('--dcHook_interval', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--min_cluster', type=int, default=1)

    args = parser.parse_args()
    print(args)
    train(args)

