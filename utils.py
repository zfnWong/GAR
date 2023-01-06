import time
import numpy as np
import tensorflow as tf
import random
import os

def set_seed_tf(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.begin_time = time.time()
        self.last_time = time.time()
        self.current_time = time.time()
        self.stage_time = 0.0
        self.run_time = 0.0

    def logging(self, message):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.update()
        message = '' if message is None else message
        print("{} {} {:.0f}s {:.0f}s | {}".format(current_time,
                                                  self._name,
                                                  self.run_time,
                                                  self.stage_time,
                                                  message))

    def update(self):
        self.current_time = time.time()

        self.stage_time = self.current_time - self.last_time
        self.last_time = self.current_time
        self.run_time = self.current_time - self.begin_time
        return self


def bpr_neg_samp(uni_users, n_users, support_dict, item_array):
    """
    param:
        uni_users - unique users in training data
        dict - {uid: array[items]}
        n_users - sample n users
        neg_num - n of sample pairs for a user.
        item_array - sample item in this array.

    return:
        ret_array - [uid pos_iid neg_iid] * n_records
    """
    pos_items = []
    users = np.random.choice(uni_users, size=n_users, replace=True)
    for user in users:
        pos_candidates = support_dict[user]
        pos_item = random.choice(pos_candidates)
        pos_items.append(pos_item)

    pos_items = np.array(pos_items, dtype=np.int64).flatten()
    neg_items = np.random.choice(item_array, len(users), replace=True)
    ret = np.stack([users, pos_items, neg_items], axis=1)
    return ret

