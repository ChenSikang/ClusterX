import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import dgl
import torch
import random
import pickle
import os.path as osp
import multiprocessing as mp
import numpy as np
import numpy.random as nrd
import pandas as pd
from tqdm import tqdm
from glob import glob
from functools import partial
from itertools import accumulate
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import get_dist_info
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader, DataListLoader, InMemoryDataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7

        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))

        target = labels.view(y_pred.size())

        ce = -1*torch.log(y_pred+eps)*target
        floss = torch.pow((1-y_pred), self.gamma)*ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)

        return torch.mean(floss)


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def clustering_performance_evaluation(X, y_preds, y_trues, num_class=1):
    X = X.cpu().detach()
    y_preds = [i.index(max(i)) for i in y_preds.tolist()]
    y_trues = [i for i in y_trues.tolist()]
    label = [i for i in range(num_class)]

    accuracy = metrics.accuracy_score(y_trues, y_preds)
    # roc_auc = metrics.roc_auc_score(y_trues, y_preds, average='micro',
    #                                 multi_class='ovr', labels=label)
    NMI = metrics.adjusted_mutual_info_score(y_trues, y_preds)
    ARI = metrics.adjusted_rand_score(y_trues, y_preds)
    SC = metrics.silhouette_score(X, y_preds)
    CHI = metrics.calinski_harabasz_score(X, y_preds)

    return accuracy, NMI, ARI, SC, CHI


def classification_performance_evaluation(y_preds, y_trues):
    y_preds = [i.index(max(i))
               for y_pred in y_preds for i in y_pred.tolist()]
    y_trues = [i for y_true in y_trues for i in y_true.tolist()]

    accuracy = metrics.accuracy_score(y_trues, y_preds)
    precision = metrics.precision_score(
        y_trues, y_preds, average='macro')
    recall = metrics.recall_score(y_trues, y_preds, average='macro')
    f1_macro = metrics.f1_score(y_trues, y_preds, average='macro')
    f1_micro = metrics.f1_score(y_trues, y_preds, average='micro')

    return accuracy, precision, recall, f1_macro, f1_micro


def draw_confusion_matrix(y_preds, y_trues, draw=False, num_class=1):
    # y_pred = ['2','2','3','1','4'] # 类似的格式
    # y_true = ['0','1','2','3','4'] # 类似的格式
    y_preds = [i.index(max(i))
               for y_pred in y_preds for i in y_pred.tolist()]
    y_trues = [i for y_true in y_trues for i in y_true.tolist()]
    label = [i for i in range(num_class)]
    C = confusion_matrix(y_trues, y_preds, labels=label)

    plt.matshow(C, cmap=plt.cm.Reds)
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(
                i, j), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    if draw:
        plt.savefig('./result/pretrain-confusion_matrix.png')


class DockingDataset(Dataset):
    def __init__(self, data_file):
        super(DockingDataset, self).__init__()
        self.data_file = data_file
        self.data_dict = {}
        self.data_dict = np.load(data_file, allow_pickle=True).item()
        # self.data_dict = pickle.load(open(data_file, 'rb')).item()
        self.ligands = list(self.data_dict.keys())
        self.labels = [-1 for _ in range(len(self.ligands))]

    def __len__(self):
        return len(self.ligands)

    def __getitem__(self, item):
        (target, pose) = self.ligands[item]
        ligand_bigraph = self.data_dict[(target, pose)]['ligand_bigraph']
        complex_bigraph = self.data_dict[(target, pose)]['complex_graph']
        complex_knn_graph = self.data_dict[(target, pose)]['complex_knn_graph']
        label_index = item % len(self.ligands)

        return ligand_bigraph, complex_bigraph, complex_knn_graph, label_index

    def assign_labels(self, labels):
        # assert len(self.labels) == len(labels), \
        #     "Inconsistent lenght of asigned labels, \
        #     {} vs {}".format(len(self.labels), len(labels))
        self.labels[:len(labels)] = labels[:]


class DockingDataset_labeled(Dataset):
    def __init__(self, data_file):
        super(DockingDataset_labeled, self).__init__()
        self.data_file = data_file
        self.data_dict = {}
        # self.data_dict = pickle.load(open(data_file, 'rb')).item()
        self.data_dict = data_file
        self.ligands = list(self.data_dict.keys())
        self.targets = list(set([i[0] for i in self.ligands]))
        self.labels = [-1 for _ in range(len(self.ligands))]

    def __len__(self):
        return len(self.ligands)

    def __getitem__(self, item):
        (target, pose) = self.ligands[item]
        ligand_bigraph = self.data_dict[(target, pose)]['ligand_bigraph']
        complex_bigraph = self.data_dict[(target, pose)]['complex_graph']
        complex_knn_graph = self.data_dict[(target, pose)]['complex_knn_graph']
        label_index = item % len(self.ligands)
        label = self.data_dict[(target, pose)]['label']
        # label = label2onehot(label)

        return ligand_bigraph, complex_bigraph, complex_knn_graph, label_index, label


def label2onehot(labels, num_class=97):
    l = [[i] for i in range(num_class)]
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(l)
    labels = onehot_encoder.transform(labels.reshape(-1, 1)).toarray()
    labels = torch.from_numpy(labels)
    return labels


def collate(data):
    lg, bg1, bg2, idx = map(list, zip(*data))
    lg = dgl.batch(lg)
    bg1 = dgl.batch(bg1)
    bg2 = dgl.batch(bg2)

    return lg, bg1, bg2, idx


def collate_pretrain(data):
    lg, bg1, bg2, idx, labels = map(list, zip(*data))
    lg = dgl.batch(lg)
    bg1 = dgl.batch(bg1)
    bg2 = dgl.batch(bg2)
    labels = [torch.tensor([i]).float() for i in labels]
    labels = torch.stack(labels, dim=0)

    return lg, bg1, bg2, idx, labels


def data_preprocessing(data):
    data = DockingDataset_labeled(data_file=data)
    lg, bg1, bg2, idx, labels = map(list, zip(*data))
    labels_mean = torch.mean(torch.tensor(labels).float())
    labels_std = torch.std(torch.tensor(labels).float())

    return labels_mean, labels_std


def rand_hyperparams():
    """ Randomly generate a set of hyperparameters.
    Returns a dictionary of randomized hyperparameters.
    """
    hyper_params = {}
    hyper_params['f_bond'] = nrd.randint(70, 120)
    hyper_params['f_gather'] = nrd.randint(80, 129)
    hyper_params['f_spatial'] = nrd.randint(hyper_params['f_gather'], 129)
    hyper_params['n_bond_conv_steps'] = nrd.randint(1, 3)
    hyper_params['n_spatial_conv_steps'] = nrd.randint(1, 2)
    hyper_params['wd'] = nrd.choice([1e-7, 1e-5])
    hyper_params['dropouts'] = [nrd.choice([0, 0.25, 0.4]) for i in range(3)]
    hyper_params['n_rows_fc'] = [nrd.choice([16])]
    hyper_params['max_num_neighbors'] = nrd.randint(3, 13)
    return hyper_params


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def split_datasets(data, train_ratio):
    random.seed(0)
    num = len(data.keys())
    val_ratio = (1-train_ratio)/2
    all_index = set(range(num))
    train_index = random.sample(all_index, int(train_ratio*num))
    val_index = random.sample(all_index - set(train_index), int(val_ratio*num))
    # test_index = all_index - set(train_index) - set(val_index)

    # print(f'训练集样本:{len(train_index)}, 验证集样本:{len(val_index)}')

    index = 0
    train_set, val_set, test_set = dict(), dict(), dict()
    for key, value in dict.items(data):
        if index in train_index:
            train_set[key] = value
        elif index in val_index:
            val_set[key] = value
        else:
            test_set[key] = value
        index += 1
    return train_set, val_set, test_set


if __name__ == '__main__':

