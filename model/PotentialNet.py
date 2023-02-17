import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init


class PotentialNet(nn.Module):
    def __init__(self,
                 f_in,
                 f_bond,
                 f_spatial,
                 f_gather,
                 n_etypes,
                 n_bond_conv_steps,
                 n_spatial_conv_steps,
                 n_rows_fc,
                 out_size,
                 dropouts
                 ):
        super(PotentialNet, self).__init__()
        self.stage_1_model = CustomizedGatedGraphConv(in_feats=f_in,
                                                      out_feats=f_bond,
                                                      f_gather=f_gather,
                                                      n_etypes=5,
                                                      n_steps=n_bond_conv_steps,
                                                      dropout=dropouts[0])
        self.stage_2_model = CustomizedGatedGraphConv(in_feats=f_gather,
                                                      out_feats=f_spatial,
                                                      f_gather=f_gather,
                                                      n_etypes=n_etypes,  # num_distance_bins + 5 covalent types
                                                      n_steps=n_spatial_conv_steps,
                                                      dropout=dropouts[1])
        self.stage_3_model = StagedFCNN(f_in=f_gather,
                                        n_row=n_rows_fc,
                                        out_size=out_size,
                                        dropout=dropouts[2])

    def forward(self, bigraph, knn_graph):
        batch_num_nodes = bigraph.batch_num_nodes()
        # print("batch_num_nodes", batch_num_nodes)
        h = self.stage_1_model(graph=bigraph, feat=bigraph.ndata['h'])
        # print('stage1 output', h)
        # print('stage1 output size', h.size())
        h = self.stage_2_model(graph=knn_graph, feat=h)
        # print('stage2 output', h)
        # print('stage2 output size', h.size())
        x = self.stage_3_model(batch_num_nodes=batch_num_nodes, features=h)
        # print('x', x)
        # print('x.size()', x.size())  # torch.Size([200, 97])

        z = ligand_gather(h, batch_num_nodes=batch_num_nodes)
        # print('gather output', h)
        # print('gather output size', h.size())
        z = F.normalize(z, p=2, dim=1)
        # print('final output', z)
        # print('final output size', z.size())
        A_pred = self.dot_product_decoder(z)

        return x, A_pred, h

    def dot_product_decoder(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


def sum_ligand_features(h, batch_num_nodes):
    """
    Compute the sum of only ligand features `h` according to the batch information `batch_num_nodes`.
    """
    node_nums = torch.cumsum(batch_num_nodes, dim=0)
    B = int(len(batch_num_nodes) / 2)  # actual batch size
    ligand_idx = [list(range(node_nums[0]))]  # first ligand
    for i in range(2, len(node_nums), 2):  # the rest of ligands in the batch
        ligand_idx.append(list(range(node_nums[i-1], node_nums[i])))
    # print('ligand_idx', ligand_idx)
    # sum over each ligand
    return torch.cat([h[i, ].sum(0, keepdim=True) for i in ligand_idx]).to(device=h.device)


def ligand_gather(h, batch_num_nodes):
    """
    Compute the sum of only ligand features `h` according to the batch information `batch_num_nodes`.
    """
    node_nums = torch.cumsum(batch_num_nodes, dim=0)
    B = int(len(batch_num_nodes) / 2)  # actual batch size
    ligand_idx = [list(range(node_nums[0]))]  # first ligand
    for i in range(2, len(node_nums), 2):  # the rest of ligands in the batch
        ligand_idx.append(list(range(node_nums[i-1], node_nums[i])))
    # print('ligand_idx', ligand_idx)
    # sum over each ligand
    return torch.cat([h[i, ] for i in ligand_idx]).to(device=h.device)


class StagedFCNN(nn.Module):
    def __init__(self,
                 f_in,
                 n_row,
                 out_size,
                 dropout):
        super(StagedFCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(f_in, n_row[0]))
        for i in range(1, len(n_row)):
            self.layers.append(nn.Linear(n_row[i-1], n_row[i]))
        self.out_layer = nn.Linear(n_row[-1], out_size)

    def forward(self, batch_num_nodes, features):
        x = sum_ligand_features(features, batch_num_nodes)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        return x


class CustomizedGatedGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 f_gather,
                 n_steps,
                 n_etypes,
                 dropout,
                 bias=True):
        super(CustomizedGatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.i_nn = nn.Linear(in_features=(
            in_feats + out_feats), out_features=f_gather)
        self.j_nn = nn.Linear(in_features=out_feats, out_features=f_gather)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def forward(self, graph, feat):
        with graph.local_scope():
            assert graph.is_homogeneous, \
                "not a homogeneous graph; convert it with to_homogeneous " \
                "and pass in the edge type as argument"
            assert graph.edata['e'].shape[1] <= self._n_etypes, \
                "edge type indices {} out of range [0, {})".format(
                    graph.edata['e'].shape, self._n_etypes)
            zero_pad = feat.new_zeros(
                (feat.shape[0], self._out_feats - feat.shape[1]))
            h = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                graph.ndata['h'] = h
                for i in range(self._n_etypes):
                    eids = graph.edata['e'][:, i].nonzero(
                        as_tuple=False).view(-1).type(graph.idtype)
                    if len(eids) > 0:
                        graph.apply_edges(
                            lambda edges: {
                                'W_e*h': self.linears[i](edges.src['h'])},
                            eids
                        )
                graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
                a = graph.ndata.pop('a')  # (N, D)
                h = self.gru(a, h)

            h = self.dropout(h)
            h = torch.mul(
                torch.sigmoid(self.i_nn(torch.cat((h, feat), dim=1))),
                self.j_nn(h))
            return h
