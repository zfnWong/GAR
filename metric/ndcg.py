import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
from tqdm import tqdm
import tensorflow as tf
import torch


def init(args):
    global Ks, BATCH_SIZE, LOG_ARANGE, max_K, DEVICE, N_JOBS
    print('ndcg init for %s' % args.dataset)
    BATCH_SIZE = args.test_batch_us
    Ks = args.Ks
    max_K = max(Ks)
    LOG_ARANGE = np.log2(np.arange(max_K + 2) + 1e-9)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_JOBS = args.n_jobs


def open_pool(func, obj, n_jobs):
    with Pool(n_jobs) as p:
        ret = p.map(func, obj)
    return ret


def test(get_topk, get_user_rating, ts_nei, ts_user, item_array, pos_dict=None, masked_items=None, showbar=False, val=True):
    """
    :param get_user_rating:
    :param ts_nei:
    :param ts_user:
    :param item_array:
    :param pos_dict:
    :param masked_items: mask 掉所有 user 与固定的一些 item 计算的 scores
    :param showbar:
    :param val:
    :return:
    """
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}
    rating_list = []
    groundTrue_list = []
    auc_record = []

    if showbar:
        rg = tqdm(range(0, len(ts_user), BATCH_SIZE))
    else:
        rg = range(0, len(ts_user), BATCH_SIZE)
    for beg in rg:
        end = min(beg + BATCH_SIZE, len(ts_user))
        batch_user = ts_user[beg:end]
        rating_all_item = get_user_rating(batch_user, item_array)

        # ================== exclude =======================
        def get_exclude_pair(u_pair):
            # 找出 user 在全集但不在当前测试集中的 item
            pos_item = np.array(list(set(pos_dict[u_pair[0]]) - set(ts_nei[u_pair[0]])), dtype=np.int64)
            pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
            return np.stack([pos_user, pos_item], axis=1)

        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)
        exclude_pair = list(map(get_exclude_pair, batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)
        rating_all_item[exclude_pair[:, 0], exclude_pair[:, 1]] = -1e10  # 不能直接用 warm_item 原因在这
        if masked_items is not None:
            rating_all_item[:, masked_items] = -1e10
        # ===================================================
        groundTrue = list(ts_nei[batch_user])
        batch_gt_pair = tuple(zip(list(rating_all_item), groundTrue))
        if N_JOBS > 1:
            aucs = open_pool(AUC, batch_gt_pair, N_JOBS)
        else:
            aucs = list(map(AUC, batch_gt_pair))
        auc_record.extend(aucs)
        if not val:
            top_item_index = get_topk(rating_all_item, max_K)
            rating_list.append(top_item_index)
            groundTrue_list.append(groundTrue)

    results['auc'] = np.mean(auc_record)
    if val:
        return results['auc']
    X = zip(rating_list, groundTrue_list)
    pre_results = list(map(test_one_batch, X))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    n_ts_user = float(len(ts_user))
    results['recall'] /= n_ts_user
    results['precision'] /= n_ts_user
    results['ndcg'] /= n_ts_user
    return results


def test_pt(get_user_rating, ts_nei, ts_user, item_array, pos_dict=None, masked_items=None, showbar=False, val=True):
    """
    :param get_user_rating:
    :param ts_nei:
    :param ts_user:
    :param item_array:
    :param pos_dict:
    :param masked_items: mask 掉所有 user 与固定的一些 item 计算的 scores
    :param showbar:
    :param val:
    :return:
    """
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}
    rating_list = []
    groundTrue_list = []
    auc_record = []

    if showbar:
        rg = tqdm(range(0, len(ts_user), BATCH_SIZE))
    else:
        rg = range(0, len(ts_user), BATCH_SIZE)

    top_k_res = None
    for beg in rg:
        end = min(beg + BATCH_SIZE, len(ts_user))
        batch_user = ts_user[beg:end]
        rating_all_item = get_user_rating(batch_user, item_array)

        # ================== exclude =======================
        def get_exclude_pair(u_pair):
            # 找出 user 在全集但不在当前测试集中的 item
            pos_item = np.array(list(set(pos_dict[u_pair[0]]) - set(ts_nei[u_pair[0]])), dtype=np.int64)
            pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
            return np.stack([pos_user, pos_item], axis=1)

        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)
        exclude_pair = list(map(get_exclude_pair, batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)
        rating_all_item[exclude_pair[:, 0], exclude_pair[:, 1]] = -1e10  # 不能直接用 warm_item 原因在这
        if masked_items is not None:
            rating_all_item[:, masked_items] = -1e10
        # ===================================================
        groundTrue = list(ts_nei[batch_user.tolist()])
        batch_gt_pair = tuple(zip(list(rating_all_item.cpu().data.numpy()), groundTrue))
        if N_JOBS > 1:
            aucs = open_pool(AUC, batch_gt_pair, N_JOBS)
        else:
            aucs = list(map(AUC, batch_gt_pair))
        auc_record.extend(aucs)
        if not val:
            top_k_res = torch.topk(rating_all_item, k=max_K, out=top_k_res)  # 排过序的
            rating_list.append(top_k_res[1].cpu().data.numpy())
            groundTrue_list.append(groundTrue)

    results['auc'] = np.mean(auc_record)
    if val:
        return results['auc']
    X = zip(rating_list, groundTrue_list)
    pre_results = list(map(test_one_batch, X))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    n_ts_user = float(len(ts_user))
    results['recall'] /= n_ts_user
    results['precision'] /= n_ts_user
    results['ndcg'] /= n_ts_user
    return results


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        r.append(pred)
    return np.array(r, dtype=np.float)


def AUC(input_data):
    """
        design for a single user
        Args:
            input_data - (all_item_scores, groundTrue)
    """
    all_item_scores = input_data[0]
    groundTrue = input_data[1]
    label_all = np.zeros((len(all_item_scores),))
    label_all[groundTrue] = 1
    mask_array = all_item_scores > -1e9
    masked_label = label_all[mask_array]
    masked_scores = all_item_scores[mask_array]
    return roc_auc_score(masked_label, masked_scores)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precision = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precision}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / LOG_ARANGE[2:k + 2], axis=1)
    dcg = pred_data * (1. / LOG_ARANGE[2:k + 2])
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X):
    sorted_items = X[0]
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in Ks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


