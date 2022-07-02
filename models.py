import numpy as np
import scipy.sparse as sp
from sparsesvd import sparsesvd
from tqdm.notebook import tqdm

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, negative_sampling
import torch.nn.functional as F


class GF_CF(object):
    def __init__(self, train_matrix):
        self.train_matrix = train_matrix

    def fit(self, dim=16):
        R = self.train_matrix
        rowsum = np.array(R.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        R_norm = d_mat @ R

        colsum = np.array(R.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1 / d_inv)
        R_norm = R_norm @ d_mat
        self.R_norm = R_norm.tocsc()
        ut, s, self.vt = sparsesvd(self.R_norm, dim)

    def predict(self, new_ratings: sp.coo_matrix):
        R_norm = self.R_norm
        # new_ratings = new_ratings.todense()
        U_2 = new_ratings @ (R_norm.T @ R_norm)
        U_1 = new_ratings @  (self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv)
        predict = U_2 + U_1
        predict[np.isnan(predict)] = 0
        return predict

    def recommend_top_k(self, interactions: sp.coo_matrix, k=20):
        """
        Предполагается нумерация айтемов с 0, иначе не будет работать argsort
        """
        ranks = self.predict(interactions)
        ranks[interactions.nonzero()] = -1e5  # exclude seen items
        return np.asarray(np.argsort(-ranks, axis=1)[:, :k])

class SimpleProp(MessagePassing):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def forward(self, x, edge_index, size):
        return self.propagate(edge_index, x=x, size=size)

    def message(self, x_j):
        return x_j


class LGCN_E(torch.nn.Module):
    def __init__(self, n_users, emb_dim, alphas=(1, 1), normalize=False, **kwargs):
        super().__init__()
        self.m = n_users
        self.normalize = normalize
        self.alphas = alphas
        self.U_0 = torch.nn.Embedding(self.m, emb_dim)
        self.layer_1 = SimpleProp(**kwargs)  # E_1 = R.T @ U
        self.layer_2 = SimpleProp(**kwargs)  # U_2 = R @ E_1
        self.layer_3 = SimpleProp(**kwargs)  # E_3 = R @ U_2

    def get_embeddings(self, edge_index, n=None):
        """
        Propagate messages through the graph and return fused final embeddings of users ans items: U, E.

        Parameters:
        ----------
        edge_index : Tensor
            Edge tensor specifying the connectivity of the graph
        n : int (optional)
            Number of items

        Returns:
        -------
            U : Tensor
                User embeddings of shape (m, emb_dim)
            E : Tensor
                Item embeddings of shape (n, emb_dim)
        """
        if n is None:
            n = edge_index[1].max().item() + 1
        m = self.m

        if self.normalize:
            inv_sqrt_user_degrees = degree(edge_index[0]).pow(-0.5)
            inv_sqrt_item_degrees = degree(edge_index[1]).pow(-0.5)
            # inv_norm = inv_sqrt_user_degrees * inv_sqrt_item_degrees
        else:
            inv_sqrt_item_degrees = torch.Tensor(1)
            inv_sqrt_user_degrees = torch.Tensor(1)

        E_1 = self.layer_1.forward(self.U_0.weight, edge_index=edge_index, size=(m, n)) * inv_sqrt_item_degrees.view(-1, 1)
        U_2 = self.layer_2.forward(E_1, edge_index[[1, 0], :], size=(n, m)) * inv_sqrt_user_degrees.view(-1, 1)
        E_3 = self.layer_3.forward(U_2, edge_index, size=(m, n)) * inv_sqrt_item_degrees.view(-1, 1)

        E = 0.5 * (E_1 + E_3)
        U = 0.5 * (self.U_0.weight + U_2)
        return U, E

    def forward(self, edge_index, edge_label_index, n_items=None):
        """
        Computes rankings for pairs of nodes using learned user embeddings.

        Parameters
        ----------
        edge_index: Tensor
            Edge tensor specifying the connectivity of the graph
        edge_label_index: Tensor, optional
            Edge tensor specifying the node pairs for which to compute rankings or probabilities
        n_items: int (optional)
            Number of items

        Returns
        -------
        scores : Tensor
            Scores of edges of shape (edge_index_label, ).
        """
        if edge_label_index is None:
            edge_label_index = edge_index

        U, E = self.get_embeddings(edge_index, n=n_items)
        src = U[edge_label_index[0]]
        dst = E[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
