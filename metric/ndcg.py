import numpy as np


def init(args):
    global Ks, TEST_BATCH_SIZE, LOG_ARANGE, max_K, DEVICE
    print('Init for %s' % args.dataset)
    TEST_BATCH_SIZE = args.test_batch_us
    Ks = args.Ks
    max_K = max(Ks)
    LOG_ARANGE = np.log2(np.arange(max_K + 2) + 1e-9)


def test(get_topk, get_user_rating, ts_nei, ts_user, exclude_pair_cnt, masked_items=None, val=True):
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}
    rating_list = []
    score_list = []
    groundTrue_list = []
    batch_size = TEST_BATCH_SIZE

    for i, beg in enumerate(range(0, len(ts_user), batch_size)):
        end = min(beg + batch_size, len(ts_user))
        batch_user = ts_user[beg:end]
        rating_all_item = get_user_rating(batch_user)

        # ================== exclude =======================
        exclude_pair = exclude_pair_cnt[0][exclude_pair_cnt[1][i]:exclude_pair_cnt[1][i + 1]]
        rating_all_item[exclude_pair[:, 0], exclude_pair[:, 1]] = -1e10

        if masked_items is not None:
            rating_all_item[:, masked_items] = -1e10
        # ===================================================
        groundTrue = list(ts_nei[batch_user])
        top_scores, top_item_index = get_topk(rating_all_item, max_K)

        score_list.append(top_scores)
        rating_list.append(top_item_index)
        groundTrue_list.append(groundTrue)

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
    return results, np.concatenate(score_list, axis=0)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        r.append(pred)
    return np.array(r, dtype=np.float32)


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


