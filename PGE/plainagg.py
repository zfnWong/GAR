"""
Aggregate the stacked embedding together to constrain the space
"""
import time
import random
import pickle
import argparse
import warnings
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
import pandas as pd


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--sorted', action='store_true', default=False,
                    help='Whether use sorting.')
parser.add_argument('--dataset', type=str, default="CiteULike",
                    help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/",
                    help='Director of the dataset.')
parser.add_argument('--trajdir', type=str, default="../traj/",
                    help='Director of the dataset.')
parser.add_argument('--samp_meth', type=str, default='sepdot',
                    help='Sampling method.')
parser.add_argument('--samp_size', nargs='?', default='[25,25]',
                    help='Sampling size')
parser.add_argument('--emb', type=str, default='bprmf',
                    help='Emebdding method')
parser.add_argument('--n_jobs', type=int, default=4,
                    help='Multiprocessing number.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random Seed.')
parser.add_argument('--max_nei', type=int, default=256,
                    help='Multiprocessing number.')
parser.add_argument('--recompute', action='store_true', default=False,
                    help='Whether recompute the trajectory list.')
args, _ = parser.parse_known_args()

args.samp_size = eval(args.samp_size)
args.sep_samp = args.samp_size
#  [0, 1, 25, 25] = [0, 1, 26, 51]
args.samp_pos = np.cumsum([0, 1] + args.samp_size)


def set_seed(seed):
    print('Unfolder Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_adj_element(l):
    adj_map = num_nodes + np.zeros((l[1] - l[0], args.max_nei), dtype=np.int)  # (batch, max_nei) ele=mask
    sub_adj = RAW_ADJ[l[0]: l[1]]
    for v in range(l[0], l[1]):
        neighbors = np.nonzero(sub_adj[v - l[0], :])[1]
        len_neighbors = len(neighbors)
        if len_neighbors > args.max_nei:
            if args.sorted:
                P = NVS[v]
                Q = NVS[neighbors]
                weight_sort = np.argsort(-np.sum(P * Q, axis=-1))
                neighbors = neighbors[weight_sort[:args.max_nei]]
            else:
                neighbors = np.random.choice(neighbors, args.max_nei, replace=False)
            adj_map[v - l[0]] = neighbors
        else:
            adj_map[v - l[0], :len_neighbors] = neighbors
    return adj_map


def compute_adjlist_parallel(batch=50):
    index_list = [[ind, min(ind + batch, num_nodes)] for ind in range(0, num_nodes, batch)]
    with Pool(args.n_jobs) as pool:
        adj_list = pool.map(compute_adj_element, index_list)
    adj_list.append(num_nodes + np.zeros((1, args.max_nei), dtype=np.int))
    adj_map = np.vstack(adj_list)
    return adj_map


def get_traj_child(parent, sample_num=0):
    """
    If sample_num == 0 return all the trajectory, else return truncated list
    """
    traj_list = []
    for p in parent:
        neigh = np.unique(ADJ_TAB[p].reshape([-1]))
        # if len(neigh) > 1:
        #     neigh = neigh[neigh != num_nodes]
        neigh = np.random.choice(neigh, min(args.max_nei, len(neigh)), replace=False)
        t_array = np.hstack([p * np.ones((len(neigh), 1)).astype(np.int), neigh.reshape([-1, 1])])
        traj_list.append(t_array)
    # traj_array = np.unique(np.vstack(traj_list), axis=0)
    traj_array = np.vstack(traj_list)
    if traj_array.shape[0] > 1:
        traj_array = traj_array[traj_array[:, -1] != num_nodes]
    if sample_num:
        traj_array = traj_array[
            np.random.choice(
                list(range(traj_array.shape[0])), min(sample_num, traj_array.shape[0]), replace=False)]
    return traj_array


def get_gun_traj(idx):
    """
    Get the trajectory list of a given node under the naive gun setting.
    return: [root node, trajectory list]
    """
    traj_list = [np.array(idx), []]
    whole_trajs = np.unique(ADJ_TAB[idx])
    whole_trajs = get_traj_child(whole_trajs, 0)
    traj_list[1] = whole_trajs
    return traj_list


def sepdot_samp_traj(idx):
    traj = get_gun_traj(idx)
    n_samp1 = args.sep_samp[0]
    n_samp2 = args.sep_samp[1]
    cen_node = traj[0]
    traj_list = traj[1]
    traj_vec = np.array([cen_node])

    # Blank Graph
    if np.sum(traj_list[:, -1] != num_nodes) == 0:
        traj_vec = np.hstack([
            traj_vec,
            np.ones(n_samp1).astype(int) * cen_node,
            np.ones(n_samp2).astype(int) * cen_node])
        sub_emb = np.vstack([EMB[traj_vec[args.samp_pos[_]:args.samp_pos[_ + 1]]].mean(0, keepdims=True) for _ in
                             range(len(args.samp_pos) - 1)])
        return sub_emb

    # Sample of First-ord neis
    traj_list = traj_list[traj_list[:, -1] != num_nodes]
    idxes = np.unique(traj_list[:, 0].reshape([-1]))
    if len(idxes) > 1:
        # idxes = idxes[idxes != cen_node]
        idxes = idxes[idxes != num_nodes]
    if len(idxes) > n_samp1:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes1 = idxes[traj_rank[-n_samp1:]]
    else:
        extra_idxes = np.random.choice(idxes, n_samp1 - len(idxes))
        idxes1 = np.hstack([idxes, extra_idxes])
    traj_vec = np.hstack([traj_vec, idxes1])

    # Sample of second-ord neis
    idxes = np.unique(traj_list[:, -1].reshape([-1]))
    if len(idxes) > 1:
        idxes = idxes[idxes != cen_node]
    if len(idxes) > n_samp2:
        P = NVS[cen_node]
        Q = NVS[idxes]
        traj_sim = np.sum(P * Q, axis=1)
        traj_rank = np.argsort(traj_sim)
        idxes2 = idxes[traj_rank[-n_samp2:]]
    else:
        extra_idxes = np.random.choice(idxes, n_samp2 - len(idxes))
        idxes2 = np.hstack([idxes, extra_idxes])
    traj_vec = np.hstack([traj_vec, idxes2])
    sub_emb = np.vstack([EMB[traj_vec[
                                  args.samp_pos[_]:args.samp_pos[_ + 1]]].mean(0, keepdims=True) for _ in
                         range(len(args.samp_pos) - 1)])
    return sub_emb


def compute_adj(input_df):
    """
    generate user-item sparse adjacent matrix.
    | 0      user-item |
    | item-user      0 |
    """
    cols = input_df.user.values
    rows = input_df.item.values + para_dict['user_num']
    values = np.ones(len(cols))

    adj = sp.csr_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    return adj


set_seed(args.seed)
print(args.dataset)
para_dict = pickle.load(open(args.datadir + args.dataset + '/convert_dict.pkl', 'rb'))
num_nodes = para_dict['item_num'] + para_dict['user_num']
train_list = pd.read_csv(args.datadir + args.dataset + '/warm_emb.csv', dtype=np.int)
RAW_ADJ = compute_adj(train_list)
EMB = np.load(args.datadir + args.dataset + '/{}.npy'.format(args.emb))
null_feature = np.zeros((1, EMB.shape[1]))
EMB = np.concatenate([EMB, null_feature], axis=0)
ADJ_TAB = compute_adjlist_parallel()
NVS = EMB
node_list = list(range(num_nodes))
print('Init Global Variables!')

t0 = time.time()
with Pool(args.n_jobs) as pool:
    embs = np.stack(pool.map(eval(args.samp_meth + '_samp_traj'), node_list))
t1 = time.time()
np.save("%s%s/%s_%d-%d_u(%s)_i(%s)_plain.npy" % (
    args.datadir, args.dataset, args.emb, args.samp_size[0], args.samp_size[1],
    args.samp_meth, args.samp_meth), embs)
print("%s%s/%s_%d-%d_u(%s)_i(%s)_plain.npy" % (
    args.datadir, args.dataset, args.emb, args.samp_size[0], args.samp_size[1],
    args.samp_meth, args.samp_meth))
