import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

from torch.nn import Sequential, ReLU
from torch_geometric.nn import (
    global_add_pool,
    Linear,
    global_mean_pool,
    GINEConv,
    SAGEConv,
    global_max_pool,
    MessagePassing,
    global_sort_pool,
)
from torch_geometric.utils import add_self_loops, degree

from typing import Callable, Union

from torch import Tensor

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


# class AttentionModule(nn.Module):
#     def __init__(self, input_dim):
#         super(AttentionModule, self).__init__()
#         self.query_layer = nn.Linear(input_dim, input_dim)
#         self.key_layer = nn.Linear(input_dim, input_dim)
#         self.value_layer = nn.Linear(input_dim, input_dim)
#
#     def forward(self, query, keys, values):
#         query = self.query_layer(query)
#         keys = self.key_layer(keys)
#         values = self.value_layer(values)
#         attention_scores = torch.bmm(query, keys.transpose(1, 2))
#         attention_weights = torch.softmax(attention_scores, dim=-1)
#         print(attention_weights, attention_scores.shape)
#         exit()
#         weighted_sum = torch.matmul(attention_weights, values)
#         weighted_sum += query
#
#         return weighted_sum

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, query, keys):
        atten_scores_list = []
        for key in keys:
            atten_scores = torch.sum(query * key, dim=0)
            # print(atten_scores)
            max_scores = atten_scores.max()
            min_scores = atten_scores.min()
            norm_scores = (atten_scores - min_scores) / (max_scores - min_scores + 1e-6)
            atten_scores = self.linear(norm_scores)
            atten_scores_list.append(atten_scores)

        stacked_attention_scores = torch.stack(atten_scores_list, dim=-1)
        atten_weights = F.softmax(stacked_attention_scores, dim=-1)

        atten_rep = torch.sum(torch.stack(keys, dim=-1) * atten_weights, dim=-1)
        fina_atten_rep = atten_rep + query

        return fina_atten_rep

class EdgePathNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        cutoff,
        y,
        n_classes,
        device,
        residuals=False,
        encode_distances=False,
        use_edge_attr=False,
        num_embeddings_v=None,
        num_embeddings_e=None,
        readout="sum",
        path_agg="sum",
        dropout=0,
    ):
        """
        Main class for PathNN that also uses edge features. Edge discrete features are first encoded using an embedding or ogb's BondEncoder, and
        then directly used as input to the lstm. Distance encoding can be used together with edge features.
        """
        super(EdgePathNN, self).__init__()
        self.cutoff = cutoff
        self.device = device
        self.residuals = residuals
        self.dropout = dropout
        self.encode_distances = encode_distances
        self.attention_module = AttentionModule(hidden_dim)
        self.y = y  

        # AtomEncoder if not ZINC
        if not num_embeddings_v:
            self.feature_encoder = AtomEncoder(hidden_dim)
        else:
            self.feature_encoder = nn.Embedding(num_embeddings_v, hidden_dim)
        # EdgeEncoder if not ZINC
        if not num_embeddings_e:
            self.bond_encoder = BondEncoder(hidden_dim)
        else:
            self.bond_encoder = nn.Embedding(num_embeddings_e, hidden_dim)

        # Distance Encoding + LSTM parameters for DE + Edge
        if encode_distances:
            self.distance_encoder = nn.Embedding(cutoff, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 3,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )

        conv_class = EdgePathConv

        # 1 MLP per conv layer
        self.convs = nn.ModuleList([])
        for i in range(self.cutoff - 1):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                ReLU(),
            )
            if path_agg == "sum":
                bn = nn.BatchNorm1d(hidden_dim)
            else:
                bn = None

            self.convs.append(
                conv_class(
                    hidden_dim,
                    self.lstm,
                    mlp,
                    bn,
                    residuals=self.residuals,
                    path_agg=path_agg,
                    dropout=self.dropout,
                )
            )

        self.use_edge_attr = use_edge_attr
        self.hidden_dim = hidden_dim

        # 2-layers MLP for prediction
        self.linear1 = Linear(hidden_dim, hidden_dim)
        # self.linear1 = Linear(hidden_dim * cutoff, hidden_dim)
        self.linear2 = Linear(hidden_dim, n_classes)

        if readout == "sum":
            self.pooling = global_add_pool
        elif readout == "mean":
            self.pooling = global_mean_pool
        self.reset_parameters()

    def reset_parameters(self):

        if hasattr(self.feature_encoder, "atom_embedding_list"):
            for emb in self.feature_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.feature_encoder.weight.data)

        if hasattr(self, "bond_encoder"):
            if hasattr(self.bond_encoder, "bond_embedding_list"):
                for emb in self.bond_encoder.bond_embedding_list:
                    nn.init.xavier_uniform_(emb.weight.data)
            else:
                nn.init.xavier_uniform_(self.bond_encoder.weight.data)
        for conv in self.convs:
            conv.reset_parameters()
        self.lstm.reset_parameters()
        if hasattr(self, "edge_pos_encoder"):
            nn.init.xavier_uniform_(self.edge_pos_encoder.weight.data)
        if hasattr(self, "distance_encoder"):
            nn.init.xavier_uniform_(self.distance_encoder.weight.data)

        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, data):
        # Save the result of each convolution at every step using a list
        conv_results = []
        # Map node initial features to d-dim space
        # [n_nodes, hidden_size]
        W_0 = self.feature_encoder(data.x.long())
        # conv_results.append(W)
        # Map initial edge features to d-dim space
        # [n_edges, hidden_size]
        edge_attr = self.bond_encoder(data.edge_attr.long())

        for i in range(self.cutoff - 1):
            # Distance encoding
            if self.encode_distances:
                # [n_paths, path_length, hidden_size]
                dist_emb = self.distance_encoder(getattr(data, f"sp_dists_{i+2}"))
            else:
                dist_emb = None
            # Get edge feature with respect to paths
            # [n_paths, path_length]
            edge_indices = getattr(data, f"edge_indices_{i+2}")
            # [n_paths, path_length-1, hidden_size]
            edge_attr_in = edge_attr[edge_indices]

            # Update node representations
            # [n_nodes, hidden_size]

            if i == 0:
                W_1 = self.convs[i](
                    W_0, getattr(data, f"path_{i+2}"), edge_attr_in, dist_emb=dist_emb
                )
            else:
                W = (1 - self.y) * W_0 - self.y * torch.sum(torch.stack(conv_results, dim=-1), dim=-1)
                W_1 = self.convs[i](
                    W, getattr(data, f"path_{i + 2}"), edge_attr_in, dist_emb=dist_emb
                )
            # else:
            #     W_1 = self.convs[i](
            #         W_1, getattr(data, f"path_{i + 2}"), edge_attr_in, dist_emb=dist_emb
            #     )

            conv_results.append(W_1)
        # get attention result
        out = self.attention_module(W_0, conv_results)

        # Redout and predict
        # [n_graphs, hidden_size]
        # out = torch.cat(conv_results, dim=1)
        out = self.pooling(out, data.batch)
        out = F.relu(self.linear1(out))
        out = F.dropout(out, training=self.training, p=self.dropout)
        return self.linear2(out)
        # return out


# class AttentionPooling(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(AttentionPooling, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim =hidden_dim
#
#     def atten(self, k, q):
#         atten_scores = torch.matmul(k, q.t())
#         atten_weights = torch.softmax(atten_scores, dim=0)
#         weight_k = torch.matmul(atten_weights.t(), k)
#         return weight_k
#
#     def forward(self, h, paths, x):
#         x_out = torch.zeros(self.input_dim, self.hidden_dim, device=x.device)
#         idx = paths[:, -1]
#         h = h.squeeze(0)
#         unique_val = torch.unique(idx)
#         for i in range(unique_val.size(0)):
#             indcies = torch.nonzero(idx == unique_val[i]).squeeze(1)
#             h_i = torch.zeros(indcies.size(0), self.hidden_dim, device=x.device)
#             for j in range(indcies.size(0)):
#                 h_i[j] = h[indcies[j]].unsqueeze(0)
#             x_out[i] = self.atten(h_i, x[unique_val[i]].unsqueeze(0))
#         return x_out



class EdgePathConv(torch.nn.Module):
    r""" """

    def __init__(
        self,
        hidden_dim,
        rnn: Callable,
        mlp: Callable,
        batch_norm: Callable,
        residuals=True,
        path_agg="sum",
        dropout=0.5,
    ):
        super(EdgePathConv, self).__init__()
        self.nn = mlp
        self.rnn = rnn
        self.bn = batch_norm
        if self.bn is None:
            self.bn = nn.Identity()

        self.hidden_dim = hidden_dim
        self.residuals = residuals
        self.dropout = dropout
        self.path_agg = path_agg

        if path_agg == "sum":
            self.path_pooling = scatter_add
        elif path_agg == "mean":
            self.path_pooling = scatter_mean
        # if path_agg == 'mean':
        #     self.path_pooling = AttentionPooling

        self.reset_parameters()

    def reset_parameters(self):
        if self.nn is not None:
            for c in self.nn.children():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()
        if hasattr(self.bn, "reset_parameters"):
            self.bn.reset_parameters()

  
    def forward(self, x: Union[Tensor, OptPairTensor], paths, edge_attr, dist_emb=None):

        # Adding a vector of zeros to first edge features entry as there are no relevant edge at first sequence input
        # [n_paths, path_length, hidden_size]
        edge_attr = torch.cat(
            [
                torch.zeros(paths.size(0), 1, self.hidden_dim, device=x.device),
                edge_attr,
            ],
            dim=1,
        )
        # [n_paths, path_length, hidden_size]
        x_cat = x[paths]
        # Distance Encoding
        if dist_emb is not None:
            # [n_paths, path_length, hidden_size * 3]
            x_cat = torch.cat([x_cat, dist_emb, edge_attr], dim=-1)
        else:
            # [n_paths, path_length, hidden_size * 2]
            x_cat = torch.cat([x_cat, edge_attr], dim=-1)

        # Applying dropout to lstm input
        x_cat = F.dropout(x_cat, training=self.training, p=self.dropout)
        # Path representations
        # [1, n_paths, hidden_size]
        _, (h, _) = self.rnn(x_cat)
        # Summing paths to get intermediate node representations (right side Eq 2 in the paper)
        # [n_nodes, hidden_size]
        h = self.path_pooling(
            h.squeeze(0),
            paths[:, -1],
            dim=0,
            out=torch.zeros(x.size(0), self.hidden_dim, device=x.device),
        )

        # atten_pool = AttentionPooling(x.size(0), self.hidden_dim)
        # h = atten_pool(h, paths, x)

        # Applying residuals connections (left side Eq 2 in the paper) + BN
        if self.residuals:
            h = self.bn(x + h)
        else:
            h = self.bn(h)

        # Dropout before MLP
        h = F.dropout(h, training=self.training, p=self.dropout)
        # 2-layers MLP for phi function
        # [n_nodes, hidden_size]
        h = self.nn(h)

        return h
