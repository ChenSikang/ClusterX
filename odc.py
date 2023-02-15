from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import HOOKS, Hook
from mmcv.utils import print_log
from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from openselfsup.utils import print_log
from openselfsup.models.necks import NonLinearNeckV0
from openselfsup.models.heads import ClsHead
from openselfsup.models.memories import ODCMemory
from torch_geometric.data import DataListLoader, DataLoader
from IGN import ModifiedAttentiveFPPredictorV2, DTIConvGraph3Layer, FC, EdgeWeightAndSum_V2
from torch_geometric.utils import to_dense_batch, add_self_loops, remove_self_loops, normalized_cut, dense_to_sparse, \
    is_undirected, to_undirected, contains_self_loops


class ODC(nn.Module):

    def __init__(self,
                 node_feat_size=40,
                 edge_feat_size=21,
                 num_layers=3,
                 graph_feat_size=256,
                 outdim_g3=200,
                 dropout=0.25,
                 total_samples=48348,
                 feat_dim=64,
                 num_classes=75,
                 min_class=100):
        super(ODC, self).__init__()
        # odc
        self.neck = FCNeck(
            in_channels=200,  # 1024
            hid_channels=128,  # 512
            out_channels=64,  # 256
            dropout=dropout,
            with_avg_pool=True
        )
        self.head = ClsHead(
            with_avg_pool=False,
            in_channels=64,  # 256
            num_classes=num_classes
        )
        self.memory_bank = ODCMemory(
            length=total_samples,
            feat_dim=feat_dim,  # 256
            momentum=0.5,
            num_classes=num_classes,
            min_cluster=min_class,
            debug=False
        )
        # set reweight tensors
        self.num_classes = num_classes
        self.loss_weight = torch.ones((self.num_classes,),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()
        self.total_samples = total_samples

        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(
            node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(
            graph_feat_size+1, outdim_g3, dropout)

        # read out
        # self.readout = EdgeWeightAndSum_V2(outdim_g3)

    def init_weights(self):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        self.neck.init_weights(init_linear='kaiming')
        self.head.init_weights(init_linear='normal')

    def forward_backbone(self, g, g3):
        atom_feats = g.ndata.pop('h')
        bond_feats = g.edata.pop('e')
        atom_feats = self.cov_graph(g, atom_feats, bond_feats)
        bond_feats3 = g3.edata['e']
        bond_feats3 = self.noncov_graph(g3, atom_feats, bond_feats3)
        # readouts, weights = self.readout(g3, bond_feats3)
        # print('bond_feats3.size():', bond_feats3.size())

        return bond_feats3  # .view(1,-1)

        # batch_size = data.batch.max().item() + 1
        # if torch.cuda.is_available():
        #     data.x = data.x.cuda()
        #     data.edge_attr = data.edge_attr.cuda()
        #     data.edge_index = data.edge_index.cuda()
        #     data.batch = data.batch.cuda()

        # # make sure that we have undirected graph
        # if not is_undirected(data.edge_index):
        #     data.edge_index = to_undirected(data.edge_index)

        # # make sure that nodes can propagate messages to themselves
        # if not contains_self_loops(data.edge_index):
        #     data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr.view(-1))

        # # covalent_propagation
        # # add self loops to enable self propagation
        # covalent_edge_index, covalent_edge_attr = self.covalent_neighbor_threshold(data.edge_index, data.edge_attr)
        # non_covalent_edge_index, non_covalent_edge_attr = self.non_covalent_neighbor_threshold(data.edge_index,
        #                                                                                        data.edge_attr)

        # # covalent_propagation and non_covalent_propagation
        # covalent_x = self.covalent_propagation(data.x, covalent_edge_index, covalent_edge_attr)
        # non_covalent_x = self.non_covalent_propagation(covalent_x, non_covalent_edge_index, non_covalent_edge_attr)

        # # zero out the protein features then do ligand only gather...hacky sure but it gets the job done
        # non_covalent_ligand_only_x = non_covalent_x
        # non_covalent_ligand_only_x[data.x[:, 16] == -1] = 0
        # pool_x = self.global_add_pool(non_covalent_ligand_only_x, data.batch)  # [batch, 1, non_covalent_gather_width]
        # out = self.output(pool_x)
        # out = out.view(batch_size, -1)
        # return [out, ]

    def forward_train(self, g, g3, idx, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # forward & backward
        # idx = g.label_index
        idx = idx[0] % self.total_samples
        print('idx:',self.memory_bank.label_bank[idx])
        idx = torch.LongTensor(idx)
        print('lable:',self.memory_bank.label_bank[idx].numpy)
        x = self.forward_backbone(g, g3)

        ######################
        feature = self.neck([x])
        print('feature size', feature[0].size())
        # feature = x
        ######################

        outs = self.head(feature)
        print('head outs的维度', torch.tensor(outs[0], device='cpu').size())
        if self.memory_bank.label_bank.is_cuda:
            loss_inputs = (outs, self.memory_bank.label_bank[idx])
        else:
            loss_inputs = (outs, self.memory_bank.label_bank[idx].cuda())
        losses = self.head.loss(*loss_inputs)
        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
        losses['change_ratio'] = change_ratio

        return losses

    def forward_test(self, g, g3, **kwargs):
        x = self.forward_backbone(data)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, g, g3, idx, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(g, g3, idx, **kwargs)
        elif mode == 'test':
            return self.forward_test(g, g3, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(g, g3)
        else:
            raise Exception("No such mode: {}".format(mode))

    def set_reweight(self, labels=None, reweight_pow=0.5):
        """Loss re-weighting.

        Re-weighting the loss according to the number of samples in each class.

        Args:
            labels (numpy.ndarray): Label assignments. Default: None.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        if labels is None:
            if self.memory_bank.label_bank.is_cuda:
                labels = self.memory_bank.label_bank.cpu().numpy()
            else:
                labels = self.memory_bank.label_bank.numpy()
        ##################
        print("lables:", labels)
        ##################
        hist = np.bincount(
            labels, minlength=self.num_classes).astype(np.float32)
        inv_hist = (1. / (hist + 1e-5)) ** reweight_pow
        weight = inv_hist / inv_hist.sum()
        self.loss_weight.copy_(torch.from_numpy(weight))
        self.head.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)


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


class FCNeck(nn.Module):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 dropout,
                 with_avg_pool=True):
        super(FCNeck, self).__init__()

        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.bn0 = nn.BatchNorm1d(hid_channels)
        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.leakyrelu = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.leakyrelu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        return [x]


if __name__ == "__main__":
    net = ODC(in_channels=1, out_channels=10)
