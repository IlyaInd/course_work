import pandas as pd
import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

class InteractionMatrixDataset:
    def __init__(self, path, **kwargs):
        self.path = path
        self.df = self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        df = pd.read_csv(self.path, **kwargs)

        df.columns = ('user_id', 'item_id', 'rating')
        df['rating'] = 1
        all_possible_connections = df['user_id'].nunique() * df['item_id'].nunique()
        print(f"Sparsity = {100 * round(1 - df.shape[0] / all_possible_connections, 5)}%")
        n_users, n_items = df[['user_id', 'item_id']].nunique()
        print(f"Users: {n_users}")
        print(f"Items: {n_items}")
        return df

    def train_test_split(self, test_ratio=0.25, observed_ratio=0.1):
        df = self.df
        all_users = self.df['user_id'].unique()
        test_size = int(len(all_users) * test_ratio)

        # Choose users for test (i.e. cold start)
        test_users = np.random.choice(a=all_users, size=test_size, replace=False)
        df_train = df[~df['user_id'].isin(test_users)]
        df_test = df[df['user_id'].isin(test_users)]
        df_test = df_test[df_test['item_id'].isin(df_train['item_id'].unique())]

        # Encode items id
        encoder = LabelEncoder()
        df_train['user_id'] = encoder.fit_transform(df_train['user_id'])
        df_train['item_id'] = encoder.fit_transform(df_train['item_id'])
        df_test['item_id'] = encoder.transform(df_test['item_id'])

        # Split test interactions into observed and future items
        test_observed, test_future = train_test_split(df_test, test_size=1-observed_ratio, random_state=42)
        common_users = np.intersect1d(test_observed['user_id'], test_future['user_id'])
        test_observed = test_observed[test_observed['user_id'].isin(common_users)]
        test_future = test_future[test_future['user_id'].isin(common_users)]
        test_observed['user_id'] = encoder.fit_transform(test_observed['user_id'])
        test_future['user_id'] = encoder.transform(test_future['user_id'])

        self._df_train = df_train
        self._test_observed = test_observed
        self._test_future = test_future
        self.n_items = df_train['item_id'].nunique()
        self.n_train_users = df_train['user_id'].nunique()
        self.n_test_users = test_observed['user_id'].nunique()

    def build_sparse_interaction_matrix(self):
        # n_train_users = self.df_train['user_id'].nunique()
        self.train_interactions = sp.csr_matrix((self._df_train['rating'], (self._df_train['user_id'], self._df_train['item_id'])))
        self.observed_interactions = sp.csr_matrix((self._test_observed['rating'],
                                                    (self._test_observed['user_id'], self._test_observed['item_id'])),
                                                   shape=(self._test_observed['user_id'].nunique(), self.n_items)
                                                   )
        self.future_interactions = sp.csr_matrix((self._test_future['rating'],
                                                  (self._test_future['user_id'], self._test_future['item_id'])),
                                                 shape=(self._test_future['user_id'].nunique(), self.n_items)
                                                 )


class GraphDataset(InteractionMatrixDataset):
    def build_interaction_graph(self):
        if not hasattr(self, '_df_train'):
            raise AttributeError('Perform train_test_split() before build_interaction_graph()')

        pos_edges = torch.LongTensor(self._df_train[['item_id', 'user_id']].values.T)

        test_observed_edges = torch.LongTensor(self._test_observed[['item_id', 'user_id']].values.T)
        test_observed_edges[1] += self.n_train_users

        test_future_edges = torch.LongTensor(self._test_future[['item_id', 'user_id']].values.T)
        test_future_edges[1] += self.n_train_users

        self.pos_edges = pos_edges
        self.test_observed_edges = test_observed_edges
        self.test_future_edges  = test_future_edges
        self.n_test_users = len(test_observed_edges[1].unique())

    def train_val_split(self, val_ratio=0.1):
        """
        Split edges so that all users presence both in train in validation splits.
        Cut specified ratio of edges for each user, providing balanced partitioning.

        Parameters:
        ----------
        val_ratio: float
            Share of validation edges in all positive edges of the given graph

        Returns:
        -------
        train_edges: LongTensor
        val_edges: LongTensor
        """
        torch.manual_seed(42)
        val_edges = []
        train_edges = []
        for user in self.pos_edges[0].unique():
            edges = self.pos_edges[:, self.pos_edges[0] == user]
            n_items = edges.shape[1]
            edges = edges[:, torch.randperm(n_items)]
            if n_items == 1:
                thr = 1
            else:
                thr = int(n_items * (1-val_ratio))
            train_edges.append(edges[:, :thr])
            val_edges.append(edges[:, thr:])

        val_edges = torch.cat(val_edges, dim=1)
        train_edges = torch.cat(train_edges, dim=1)
        self.train_edges = train_edges
        self.val_edges = val_edges