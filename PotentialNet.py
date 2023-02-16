import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch.nn import BatchNorm1d
from torch_geometric.nn import (
    GlobalAttention,
    global_add_pool,  # Returns batch-wise graph-level-outputs by adding node features across the node dimension, so that for a single graph
    NNConv,
    avg_pool_x,
    avg_pool,
    max_pool_x,
    # GatedGraphConv,
)  # NOTE: maybe use the default version of GatedGraphConv, and make a PNET wrapper?
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch, add_self_loops, remove_self_loops, normalized_cut, dense_to_sparse, \
    is_undirected, to_undirected, contains_self_loops
# from torch_scatter import scatter as scatter_
# sum/mean/max values with same index  torch_scatter.scatter(src: torch.Tensor, index: torch.Tensor, dim: int = - 1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, reduce: str = 'sum')
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.nn.inits import uniform, reset
from torch_geometric.data import Batch
from torch.nn import init
from torch_sparse import coalesce


# from torch_geometric.utils.num_nodes import maybe_num_nodes


class PotentialNetAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(PotentialNetAttention, self).__init__()
        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        return torch.nn.Softmax(dim=1)(self.net_i(torch.cat([h_i, h_j], dim=1))) * self.net_j(
            h_j)  # 按行(feature/channel)SoftMax,行和为1


class GraphThreshold(torch.nn.Module):
    def __init__(self, t):
        super(GraphThreshold, self).__init__()
        if torch.cuda.is_available():
            self.t = nn.Parameter(t, requires_grad=True).cuda()  # Threshold trainable !
        else:
            self.t = nn.Parameter(t, requires_grad=True)

    def filter_adj(self, row, col, edge_attr, mask):
        mask = mask.squeeze()
        return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

    def maybe_num_nodes(self, index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def forward(self, edge_index, edge_attr):
        # # N = maybe_num_nodes(edge_index, None)
        # row, col = edge_index
        # mask = edge_attr <= self.t
        # row, col, edge_attr = self.filter_adj(row, col, edge_attr, mask)
        # edge_index = torch.cat([row, col], dim=0)
        # return edge_index, edge_attr
        N = self.maybe_num_nodes(edge_index, None)
        row, col = edge_index
        mask = edge_attr <= self.t
        row, col, edge_attr = self.filter_adj(row, col, edge_attr, mask)
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0
        )
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        return edge_index, edge_attr


class PotentialNetPropagation(torch.nn.Module):
    def __init__(
            self,
            feat_size=20,
            gather_width=64,
            k=2,
            neighbor_threshold=None,
            output_pool_result=False,
            bn_track_running_stats=False,
    ):
        super(PotentialNetPropagation, self).__init__()
        assert neighbor_threshold is not None

        self.neighbor_threshold = neighbor_threshold
        self.bn_track_running_stats = bn_track_running_stats
        self.edge_attr_size = 1                  #为什么不放到参数里？？？

        self.k = k  #门控网络参数
        self.gather_width = gather_width
        self.feat_size = feat_size
        self.edge_network_nn = nn.Sequential(
            nn.Linear(self.edge_attr_size, int(self.feat_size / 2)),
            nn.Softsign(),
            nn.Linear(int(self.feat_size / 2), self.feat_size),
            nn.Softsign(),
        )

        self.edge_network = NNConv(
            self.feat_size,
            self.edge_attr_size * self.feat_size,
            nn=self.edge_network_nn,  # 意义何在？
            root_weight=True,
            aggr="add",
        )
        # self.gate = GatedGraphConv(self.feat_size, self.k)
        self.gate = GatedGraphConv(
            self.feat_size, self.k, edge_network=self.edge_network
        )

        self.attention = PotentialNetAttention(
            net_i=nn.Sequential(
                nn.Linear(self.feat_size * 2, self.feat_size),
                nn.Softsign(),
                nn.Linear(self.feat_size, self.gather_width),  # 为什么不是1而是self.gather_width？
                nn.Softsign(),
            ),
            net_j=nn.Sequential(
                nn.Linear(self.feat_size, self.gather_width),
                nn.Softsign()
            )
        )

        self.output_pool_result = output_pool_result
        if self.output_pool_result:
            self.global_add_pool = global_add_pool

    def forward(self, data, edge_index, edge_attr):
        # propagtion
        h_0 = data
        h_1 = self.gate(h_0, edge_index, edge_attr)
        h_1 = self.attention(h_1, h_0)  # 全局注意力 ？？？？？？？？？？？？？？？？？？softmax

        return h_1


class PotentialNetFullyConnected(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PotentialNetFullyConnected, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / 1.5)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 1.5), int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 2), out_channels),
        )

    def forward(self, data, return_hidden_feature=False):
        if return_hidden_feature:
            return self.output[:-2](data), self.output[:-4](data), self.output(data)
        else:
            return self.output(data)


class PotentialNetParallel(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            covalent_gather_width=128,  # 16
            non_covalent_gather_width=64,  # 12
            covalent_k=1,  # 2
            non_covalent_k=1,  # 2
            covalent_neighbor_threshold=None,  # 1.5
            non_covalent_neighbor_threshold=None,  # 4.5
            always_return_hidden_feature=False,
    ):
        super(PotentialNetParallel, self).__init__()

        assert (covalent_neighbor_threshold is not None and non_covalent_neighbor_threshold is not None)

        if torch.cuda.is_available():
            self.covalent_neighbor_threshold = GraphThreshold(
                torch.ones(1).cuda() * covalent_neighbor_threshold)  # covalent
            self.non_covalent_neighbor_threshold = GraphThreshold(
                torch.ones(1).cuda() * non_covalent_neighbor_threshold)  # cov + noncov
        else:
            self.covalent_neighbor_threshold = GraphThreshold(torch.ones(1) * covalent_neighbor_threshold)
            self.non_covalent_neighbor_threshold = GraphThreshold(torch.ones(1) * non_covalent_neighbor_threshold)

        self.always_return_hidden_feature = always_return_hidden_feature
        self.global_add_pool = global_add_pool

        self.covalent_propagation = PotentialNetPropagation(
            feat_size=in_channels,
            gather_width=covalent_gather_width,
            neighbor_threshold=self.covalent_neighbor_threshold,
            k=covalent_k,
        )

        self.non_covalent_propagation = PotentialNetPropagation(
            feat_size=covalent_gather_width,
            gather_width=non_covalent_gather_width,
            neighbor_threshold=self.non_covalent_neighbor_threshold,
            k=non_covalent_k,
        )

        self.output = PotentialNetFullyConnected(non_covalent_gather_width, out_channels)

    def forward(self, data, return_hidden_feature=False):
        # import pdb
        # pdb.set_trace()
        if torch.cuda.is_available():
            data.x = data.x.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.edge_index = data.edge_index.cuda()
            data.batch = data.batch.cuda()

        # make sure that we have undirected graph
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)

        # make sure that nodes can propagate messages to themselves
        if not contains_self_loops(data.edge_index):
            data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr.view(-1))

        # covalent_propagation
        # add self loops to enable self propagation
        covalent_edge_index, covalent_edge_attr = self.covalent_neighbor_threshold(data.edge_index, data.edge_attr)
        non_covalent_edge_index, non_covalent_edge_attr = self.non_covalent_neighbor_threshold(data.edge_index,
                                                                                               data.edge_attr)

        # covalent_propagation and non_covalent_propagation
        covalent_x = self.covalent_propagation(data.x, covalent_edge_index, covalent_edge_attr)
        non_covalent_x = self.non_covalent_propagation(covalent_x, non_covalent_edge_index, non_covalent_edge_attr)

        # zero out the protein features then do ligand only gather...hacky sure but it gets the job done
        non_covalent_ligand_only_x = non_covalent_x
        non_covalent_ligand_only_x[data.x[:, 16] == -1] = 0
        pool_x = self.global_add_pool(non_covalent_ligand_only_x, data.batch)  # [batch, 1, non_covalent_gather_width]

        # fully connected and output layers
        if return_hidden_feature or self.always_return_hidden_feature:
            # return prediction and atomistic features (covalent result, non-covalent result, pool result)

            avg_covalent_x, _ = avg_pool_x(data.batch, covalent_x, data.batch)
            avg_non_covalent_x, _ = avg_pool_x(data.batch, non_covalent_x, data.batch)

            fc0_x, fc1_x, output_x = self.output(pool_x, return_hidden_feature=True)

            return avg_covalent_x, avg_non_covalent_x, pool_x, fc0_x, fc1_x, output_x
        else:
            return self.output(pool_x)

# TODO: is this defined correctly for batches?
class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}
        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}
        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})
    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, out_channels, num_layers, edge_network, aggr="add", bias=True):
        super(GatedGraphConv, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.edge_network = (
            edge_network  # TODO: make into a list of neural networks for each edge_attr
        )

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    # TODO: remove none defautl for edge_attr
    def forward(self, x, edge_index, edge_attr):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        assert h.size(1) <= self.out_channels

        # if input size is < out_channels, pad input with 0s to equal the sizes
        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])

            # source = h[edge_index[0].long()]
            # sink = h[edge_index[1].long()]
            # edge_feat = torch.cat([sink, edge_attr.view(-1, 1)], dim=1)

            # # TODO: edge_feat is defined over number of edges, while h is defined over num nodes dimension, convert back to node space by summing the features for all neighbors for each node
            # h += self.edge_network(h, edge_index, edge_attr)  # TODO: find a better network than NNConv

            # m = torch.matmul(h, self.weight[i])

            m = self.propagate(edge_index=edge_index, x=h, aggr="add")
            h = self.rnn(m, h)

        return h

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out

    def __repr__(self):
        return "{}({}, num_layers={})".format(
            self.__class__.__name__, self.out_channels, self.num_layers
        )

class PotentialNetAutoEncoder(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            covalent_gather_width=128,
            non_covalent_gather_width=64,
            covalent_k=1,
            non_covalent_k=1,
            covalent_neighbor_threshold=None,
            non_covalent_neighbor_threshold=None,
            always_return_hidden_feature=False,
    ):
        super(PotentialNetAutoEncoder, self).__init__()
    
        self.pnet_conv = PotentialNetParallel(
            in_channels,
            out_channels,
            covalent_gather_width,
            non_covalent_gather_width,
            covalent_k,
            non_covalent_k,
            covalent_neighbor_threshold,
            non_covalent_neighbor_threshold,
            always_return_hidden_feature,
    )
        self.decoder_fc = nn.Sequential(
            nn.Linear(out_channels, int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 2), int(in_channels / 1.5)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 1.5), in_channels),
        )

    def forward(self, data):
        h = self.pnet_conv(data)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.decode(z)
        return A_pred, z

    ###什么decode
    def decode(self, Z):
        Z = self.decoder_fc(Z)
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred