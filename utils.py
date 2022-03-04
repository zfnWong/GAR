import datetime
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import tensorflow as tf
# import torch
import random
from tqdm import tqdm


def set_seed(seed):
    # initial setting
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    tf.set_random_seed(seed)


def df_get_neighbors(input_df, obj, max_num):
    """
    Get users' neighboring items.
    return:
        nei_array - [max_num, neighbor array], use 0 to pad users which have no neighbors.
    """
    group = tuple(input_df.groupby(obj))
    keys, values = zip(*group)  # key: obj scalar, values: neighbor array

    keys = np.array(keys, dtype=np.int64)
    opp_obj = 'item' if obj == 'user' else 'user'
    values = list(map(lambda x: x[opp_obj].values, values))
    values.append(0)
    values = np.array(values, dtype=np.object)

    nei_array = np.zeros((max_num, ), dtype=np.object)
    nei_array[keys] = values[:-1]
    return nei_array


def neighbor_to_neighbor(begin_array, end_array):
    two_hop_nei_array = []
    for i, neighbors in tqdm(enumerate(begin_array)):
        if hasattr(neighbors, 'shape'):
            two_hop_neighbors_list = end_array[neighbors].tolist()
            two_hop_neighbors_set = set(np.concatenate(two_hop_neighbors_list, axis=0)) - {i}
            two_hop_neighbors_array = np.array(list(two_hop_neighbors_set), dtype=np.int64)
        else:
            two_hop_neighbors_array = 0
        two_hop_nei_array.append(two_hop_neighbors_array)

    return np.array(two_hop_nei_array, dtype=np.object)


class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.last_time = time.time()
        self.run_time = 0.0

    def logging(self, message):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = '' if message is None else message
        print("{} {} {:.0f}s | {}".format(current_time,
                                          self._name,
                                          self.run_time,
                                          message))

    def tic(self):
        self.last_time = time.time()
        return self

    def toc(self):
        self.run_time += time.time() - self.last_time
        return self


def bpr_neg_samp(uni_users, support_dict, item_array, neg_rate):
    """
    param:
        uni_users - user array in training data
        dict - {uid: array[items]}
        n_users - sample n users
        neg_num - n of sample pairs for a user.
        item_array - sample item in this array.

    return:
        ret_array - [uid pos_iid neg_iid] * n_records
    """
    pos_items = []
    for user in uni_users:
        # pos sampling
        pos_candidates = support_dict[user]
        # if not hasattr(pos_candidates, 'shape'):
        #     continue
        pos_item = np.random.choice(pos_candidates, 1)
        pos_items.append(pos_item)

    pos_items = np.array(pos_items, dtype=np.int64).flatten()
    neg_items = np.random.choice(item_array, (len(uni_users) * neg_rate, 1), replace=True)
    ret = np.tile(np.stack([uni_users, pos_items], axis=1), [1, neg_rate]).reshape((-1, 2))
    ret = np.concatenate([ret, neg_items], axis=1)
    return ret


def negative_sampling(pos_user_array, pos_item_array, neg, warm_item):
    """
    Args:
        pos_user_array: users in train interactions
        pos_item_array: items in train interactions
        neg: num of negative samples
        warm_item: train item set

    Returns:
        user: concat pos users and neg ones
        item: concat pos item and neg ones
        target: scores of both pos interactions and neg ones
    """
    user_pos = pos_user_array.reshape((-1))
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    user_array = np.concatenate([user_pos, user_neg], axis=0)
    item_pos = pos_item_array.reshape((-1))
    item_neg = np.random.choice(warm_item, size=neg * pos_user_array.shape[0], replace=True).reshape((-1))
    item_array = np.concatenate([item_pos, item_neg], axis=0)
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    target_array = np.concatenate([target_pos, target_neg], axis=0)
    random_idx = np.random.permutation(user_array.shape[0])
    return user_array[random_idx], item_array[random_idx], target_array[random_idx]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx