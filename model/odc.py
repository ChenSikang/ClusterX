from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from mmcv.cnn import kaiming_init, normal_init
from PotentialNet import PotentialNet, CustomizedGatedGraphConv, sum_ligand_features, ligand_gather
from utils import clustering_performance_evaluation, _init_weights, build_norm_layer, accuracy
from sklearn.cluster import KMeans
from sklearn import metrics

class ODC(nn.Module):

    def __init__(self,
                 pretrain_path,
                 total_samples,
                 num_classes=10,
                 min_cluster=20,
                 f_in=44,
                 f_bond=48,
                 f_spatial=48,
                 f_gather=48,
                 n_etypes=9,
                 n_bond_conv_steps=2,
                 n_spatial_conv_steps=1,
                 n_rows_fc=[48, 24],
                 dropouts=[0.25, 0.25, 0.25],
                 momentum=0.5):
        super(ODC, self).__init__()

        # odc
        self.neck = NonLinearNeckV0(
            f_in=f_gather,
            n_row=n_rows_fc,
            dropout=dropouts[2]
        )
        self.head = ClsHead(
            with_avg_pool=False,
            in_channels=n_rows_fc[-1],
            num_classes=num_classes
        )
        self.memory_bank = ODCMemory(
            length=total_samples,
            feat_dim=n_rows_fc[-1],
            momentum=momentum,
            num_classes=num_classes,
            min_cluster=min_cluster,
            debug=False
        )

        # set reweight tensors
        self.num_classes = num_classes
        self.loss_weight = torch.ones((self.num_classes,),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()
        self.total_samples = total_samples

        # potentialNet
        self.stage_1_model = CustomizedGatedGraphConv(in_feats=f_in,
                                                      out_feats=f_bond,
                                                      f_gather=f_gather,
                                                      n_etypes=5,
                                                      n_steps=n_bond_conv_steps,
                                                      dropout=dropouts[0])
        self.stage_2_model = CustomizedGatedGraphConv(in_feats=f_gather,
                                                      out_feats=f_spatial,
                                                      f_gather=f_gather,
                                                      n_etypes=n_etypes,
                                                      n_steps=n_spatial_conv_steps,
                                                      dropout=dropouts[1])

        check_point_dict = torch.load(
            pretrain_path)
        self.stage_1_model.load_state_dict(check_point_dict['stage1'])
        self.stage_2_model.load_state_dict(check_point_dict['stage2'])

        # self.pnet = PotentialNet(n_etypes=n_etypes,
        #                          f_in=f_in,
        #                          f_bond=f_bond,
        #                          f_spatial=f_spatial,
        #                          f_gather=f_gather,
        #                          n_rows_fc=n_rows_fc,
        #                          n_bond_conv_steps=n_bond_conv_steps,
        #                          n_spatial_conv_steps=n_spatial_conv_steps,
        #                          out_size=num_classes,
        #                          dropouts=dropouts)
        # self.pnet.load_state_dict(
        #     torch.load(pretrain_path, map_location='cpu'))

    # def init_weights(self):
    #     """Initialize the weights of model.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Default: None.
    #     """
    #     self.neck.init_weights(init_linear='kaiming')
    #     self.head.init_weights(init_linear='normal')

    def forward_backbone(self, bg1, bg2):
        bigraph, knn_graph = bg1, bg2
        batch_num_nodes = bigraph.batch_num_nodes()
        h = self.stage_1_model(graph=bigraph, feat=bigraph.ndata['h'])
        h = self.stage_2_model(graph=knn_graph, feat=h)
        z = ligand_gather(h, batch_num_nodes=batch_num_nodes)
        z = F.normalize(z, p=2, dim=1)
        A_pred = self.dot_product_decoder(z)
        # _, A_pred, h = self.pnet(bigraph, knn_graph)

        return A_pred, h

    def forward_train(self, lg, bg1, bg2, idx, **kwargs):
        # forward & backward
        # idx = data.label_index
        # idx = idx % self.total_samples
        idx = [i % self.total_samples for i in idx]

        idx = torch.LongTensor(idx)
        adj_label = lg.adjacency_matrix().to_dense().cuda()
        batch_num_nodes = bg1.batch_num_nodes()
        A_pred, x = self.forward_backbone(bg1, bg2)
        feature = self.neck([x], batch_num_nodes)
        outs = self.head(feature)
        if self.memory_bank.label_bank.is_cuda:
            loss_inputs = (feature[0], outs, self.memory_bank.label_bank[idx])
        else:
            loss_inputs = (feature[0], outs,
                           self.memory_bank.label_bank[idx].cuda())

        # print('标签idx', idx)
        # print('backbone输出维度', x.size())
        # print('backbone输出', x)
        # print('neck输出维度', feature[0].size())
        # print('neck输出', feature[0])
        # print('head输出维度', outs[0].size())
        # print('head输出', outs[0])
        # print('标签维度', self.memory_bank.label_bank[idx].size())
        # print('标签', self.memory_bank.label_bank[idx])

        losses = self.head.loss(*loss_inputs)
        losses["rec_loss"] = F.binary_cross_entropy(
            A_pred.view(-1), adj_label.view(-1))
        losses["loss"] = 10 * losses["cls_loss"] + losses["rec_loss"]
        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
        losses['change_ratio'] = change_ratio

        return losses

    def forward_test(self, data, **kwargs):
        x = self.forward_backbone(data)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, lg, bg1, bg2, idx, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(lg, bg1, bg2, idx, **kwargs)
        elif mode == 'test':
            return self.forward_test(bg1, bg2, idx, **kwargs)
        elif mode == 'extract':
            _, z = self.forward_backbone(bg1, bg2)
            return z
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
        hist = np.bincount(
            labels, minlength=self.num_classes).astype(np.float32)
        inv_hist = (1. / (hist + 1e-5)) ** reweight_pow
        weight = inv_hist / inv_hist.sum()
        self.loss_weight.copy_(torch.from_numpy(weight))
        self.head.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)

    def dot_product_decoder(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class NonLinearNeckV0(nn.Module):
    def __init__(self,
                 f_in,
                 n_row,
                 dropout):
        super(NonLinearNeckV0, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(f_in, n_row[0]))
        for i in range(1, len(n_row)):
            self.layers.append(nn.Linear(n_row[i-1], n_row[i]))
        self.dropout = nn.Dropout(p=dropout)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x, batch_num_nodes):
        assert len(x) == 1
        x = x[0]
        # x = x.view(x.size(0), -1)
        x = sum_ligand_features(x, batch_num_nodes)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        return [x]


class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, x, cls_score, labels):
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        # print('cls_score形状', cls_score[0].size())
        # print('labels形状', labels.size())
        losses['cls_loss'] = self.criterion(cls_score[0], labels)
        # losses['acc'], losses['NMI'], losses['ARI'], losses['SC'], losses['CHI'] = clustering_performance_evaluation(
        #     x, cls_score[0], labels)
        y_preds = [i.index(max(i)) for i in cls_score[0].tolist()]
        y_trues = [i for i in labels.tolist()]
        losses['acc'] = metrics.accuracy_score(y_trues, y_preds)
        return losses


class ODCMemory(nn.Module):
    """Memory modules for ODC.

    Args:
        length (int): Number of features stored in samples memory.
        feat_dim (int): Dimension of stored features.
        momentum (float): Momentum coefficient for updating features.
        num_classes (int): Number of clusters.
        min_cluster (int): Minimal cluster size.
    """

    def __init__(self, length, feat_dim, momentum, num_classes, min_cluster,
                 **kwargs):
        super(ODCMemory, self).__init__()
        self.feature_bank = torch.zeros((length, feat_dim),
                                        dtype=torch.float32)
        self.label_bank = torch.zeros((length, ), dtype=torch.long)
        self.centroids = torch.zeros((num_classes, feat_dim),
                                     dtype=torch.float32).cuda()
        self.kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20)
        self.feat_dim = feat_dim
        self.initialized = False
        self.momentum = momentum
        self.num_classes = num_classes
        self.min_cluster = min_cluster
        self.debug = kwargs.get('debug', False)

    def init_memory(self, feature, label):
        """Initialize memory modules."""
        self.initialized = True
        self.label_bank.copy_(torch.from_numpy(label).long())
        # make sure no empty clusters
        assert (np.bincount(label, minlength=self.num_classes) != 0).all()

        feature = feature / \
            (np.linalg.norm(feature, axis=1, keepdims=True) + 1e-10)
        # feature /= (np.linalg.norm(feature, axis=1).reshape(-1, 1) + 1e-10)
        self.feature_bank.copy_(torch.from_numpy(feature))
        centroids = self._compute_centroids()
        self.centroids.copy_(centroids)
        # dist.broadcast(self.centroids, 0)

    def _compute_centroids_ind(self, cinds):
        """Compute a few centroids."""
        num = len(cinds)
        centroids = torch.zeros(
            (num, self.feat_dim), dtype=torch.float32)
        for i, c in enumerate(cinds):
            ind = np.where(self.label_bank.numpy() == c)[0]
            centroids[i, :] = self.feature_bank[ind, :].mean(dim=0)
        return centroids

    def _compute_centroids(self):
        """Compute all non-empty centroids."""
        l = self.label_bank.numpy()
        argl = np.argsort(l)
        sortl = l[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(l))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        centroids = self.centroids.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(dim=0)
        return centroids

    def update_samples_memory(self, ind, feature):
        """Update samples memory."""
        assert self.initialized
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        ind = ind.cpu()

        feature_old = self.feature_bank[ind, ...].cuda()

        feature_new = (1 - self.momentum) * feature_old + \
            self.momentum * feature_norm

        feature_norm = feature_new / (
            feature_new.norm(dim=1).view(-1, 1) + 1e-10)
        self.feature_bank[ind, ...] = feature_norm.cpu()

        # compute new labels
        similarity_to_centroids = torch.mm(self.centroids,
                                           feature_norm.permute(1, 0))  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        newlabel_cpu = newlabel.cpu()
        change_ratio = (newlabel_cpu !=
                        self.label_bank[ind]).sum().float().cuda() \
            / float(newlabel_cpu.shape[0])
        self.label_bank[ind] = newlabel_cpu.clone()  # copy to cpu
        return change_ratio

    def deal_with_small_clusters(self):
        """Deal with small clusters."""
        # check empty class
        hist = np.bincount(self.label_bank.numpy(), minlength=self.num_classes)
        small_clusters = np.where(hist < self.min_cluster)[0].tolist()
        if self.debug:
            print("mincluster: {}, num of small class: {}".format(
                hist.min(), len(small_clusters)))
        if len(small_clusters) == 0:
            return
        # re-assign samples in small clusters to make them empty
        for s in small_clusters:
            ind = np.where(self.label_bank.numpy() == s)[0]
            if len(ind) > 0:
                inclusion = torch.from_numpy(
                    np.setdiff1d(
                        np.arange(self.num_classes),
                        np.array(small_clusters),
                        assume_unique=True)).cuda()
                target_ind = torch.mm(
                    self.centroids[inclusion, :],
                    self.feature_bank[ind, :].cuda().permute(
                        1, 0)).argmax(dim=0)
                target = inclusion[target_ind]
                self.label_bank[ind] = torch.from_numpy(target.cpu().numpy())
        # deal with empty cluster
        self._redirect_empty_clusters(small_clusters)

    def update_centroids_memory(self, cinds=None):
        """Update centroids memory."""

        if self.debug:
            print("updating centroids ...")
        if cinds is None:
            center = self._compute_centroids()
            self.centroids.copy_(center)
        else:
            center = self._compute_centroids_ind(cinds)
            self.centroids[
                torch.LongTensor(cinds).cuda(), :] = center.cuda()

    def _partition_max_cluster(self, max_cluster):
        """Partition the largest cluster into two sub-clusters."""
        max_cluster_inds = np.where(self.label_bank == max_cluster)[0]

        assert len(max_cluster_inds) >= 2
        max_cluster_features = self.feature_bank[max_cluster_inds, :]
        if np.any(np.isnan(max_cluster_features.numpy())):
            raise Exception("Has nan in features.")
        kmeans_ret = self.kmeans.fit(max_cluster_features)
        sub_cluster1_ind = max_cluster_inds[kmeans_ret.labels_ == 0]
        sub_cluster2_ind = max_cluster_inds[kmeans_ret.labels_ == 1]
        if not (len(sub_cluster1_ind) > 0 and len(sub_cluster2_ind) > 0):
            print(
                "Warning: kmeans partition fails, resort to random partition.")
            sub_cluster1_ind = np.random.choice(
                max_cluster_inds, len(max_cluster_inds) // 2, replace=False)
            sub_cluster2_ind = np.setdiff1d(
                max_cluster_inds, sub_cluster1_ind, assume_unique=True)
        return sub_cluster1_ind, sub_cluster2_ind

    def _redirect_empty_clusters(self, empty_clusters):
        """Re-direct empty clusters."""
        for e in empty_clusters:
            assert (self.label_bank != e).all().item(), \
                "Cluster #{} is not an empty cluster.".format(e)
            max_cluster = np.bincount(
                self.label_bank, minlength=self.num_classes).argmax().item()
            # gather partitioning indices
            sub_cluster1_ind, sub_cluster2_ind = self._partition_max_cluster(
                max_cluster)
            size1 = torch.LongTensor([len(sub_cluster1_ind)]).cuda()
            size2 = torch.LongTensor([len(sub_cluster2_ind)]).cuda()
            sub_cluster1_ind_tensor = torch.from_numpy(
                sub_cluster1_ind).long().cuda()
            sub_cluster2_ind_tensor = torch.from_numpy(
                sub_cluster2_ind).long().cuda()

            # reassign samples in partition #2 to the empty class
            self.label_bank[sub_cluster2_ind] = e
            # update centroids of max_cluster and e
            self.update_centroids_memory([max_cluster, e])


if __name__ == "__main__":
    net = ODC(in_channels=1, out_channels=10)
