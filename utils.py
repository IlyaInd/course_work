import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torch_geometric.utils import negative_sampling
from tqdm.notebook import tqdm
import torch.nn.functional as F

def recall_at_k(pred_items: np.array, true_items: np.array):
    assert pred_items.shape[0] == true_items.shape[0]
    recall_list = []
    for i_pred, i_true in zip(pred_items, true_items):
        try:
            hits = len(np.intersect1d(i_pred, i_true))
            recall = hits / len(i_true)
            recall_list.append(recall)
        except ZeroDivisionError:
            print(len(recall_list), i_true)
    return np.mean(recall_list)


def precision_at_k(pred_items: np.array, true_items: np.array):
    assert pred_items.shape[0] == true_items.shape[0]
    precision_list = []
    for i_pred, i_true in zip(pred_items, true_items):
        try:
            hits = len(np.intersect1d(i_pred, i_true))
            prec = hits / len(i_pred)
            precision_list.append(prec)
        except ZeroDivisionError:
            print(len(precision_list), i_true)
    return np.mean(precision_list)


def ndcg_at_k(ranks, recs, interactions):
    assert ranks.shape[0] == recs.shape[0] == interactions.shape[0]
    scores_list = []
    for u in range(interactions.shape[0]):
        score = ndcg_score(interactions[u, recs[u]].toarray(), [ranks[u]])
        scores_list.append(score)
    return np.mean(scores_list)


def get_k_core(df, core=20):
    df = df.clone()
    for _ in tqdm(range(50)):
        gpb = df.groupby('user_id')['rating'].count()
        if gpb.min() < core:
            print('user min = ', gpb.min())
            core_k_users = gpb[gpb >= core].index.values
            df = df[df['user_id'].isin(core_k_users)]
            gpb = df.groupby('item_id')['rating'].count()
            core_k_items = gpb[gpb >= core].index.values
            df = df[df['item_id'].isin(core_k_items)]
        else:
            return df


def train(model, opt, dataset, train_edges, device, n_epoch):
    """TODO: refactor function and build it up as models or trainer method"""
    for i in tqdm(range(n_epoch)):
        opt.zero_grad()
        # ==== TRAIN ====
        neg_train_edges = negative_sampling(train_edges, num_neg_samples=train_edges.shape[1],
                                            num_nodes=(model.m, dataset.n_train_users)).to(device)

        train_scores = model.forward(edge_index=train_edges, edge_label_index=train_edges, n_items=dataset.n_train_users)
        neg_scores = model.forward(edge_index=train_edges, edge_label_index=neg_train_edges, n_items=dataset.n_train_users)
        train_loss = torch.mean(F.softplus(neg_scores - train_scores))
        train_loss.backward()
        opt.step()
    return model, opt
        # train_acc = (train_scores > neg_scores).float().mean()


        # ==== VALIDATION ====
        # neg_val_edges = negative_sampling(pos_edges, num_neg_samples=val_edges.shape[1],
        #                                   num_nodes=(model.m, dataset.n_train_users)).to(device)
        # val_scores = model.forward(edge_index=train_edges, edge_label_index=val_edges, n_items=dataset.n_train_users)
        # neg_val_scores = model.forward(edge_index=train_edges, edge_label_index=neg_val_edges, n_items=dataset.n_train_users)
        # val_loss = torch.mean(F.softplus(neg_val_scores - val_scores))
        # val_acc = (val_scores > neg_val_scores).float().mean()


        # ==== TEST ====
        # neg_test_edges = negative_sampling(_tfe, num_nodes=(model.m, dataset._test_observed['user_id'].nunique()))
        # neg_test_edges[1] += dataset.n_train_users
        # test_scores = model.forward(edge_index=torch.cat([pos_edges, test_observed_edges], dim=1),
        #                             edge_label_index=test_future_edges, n_items=test_observed_edges[1].max().item() + 1)
        # neg_test_scores = model.forward(edge_index=torch.cat([pos_edges, test_observed_edges], dim=1),
        #                                edge_label_index=neg_test_edges, n_items=test_observed_edges[1].max().item() + 1)
        # test_loss = torch.mean(F.softplus(neg_test_scores - test_scores))
        # test_acc = (test_scores > neg_test_scores).float().mean()


        # === LOGGING ====
        # wandb.log({'loss': {'train': train_loss, 'val': val_loss, 'test': test_loss},
        #            'acc': {'train': train_acc, 'val': val_acc, 'test': test_acc}})