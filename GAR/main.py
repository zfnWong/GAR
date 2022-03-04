import os
import sys

sys.path.append("..")
from metric import ndcg
import utils
import time
import uuid
import pickle
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from pprint import pprint
import random
import scipy.sparse as sp
from gar_mlp import GARMLP
from gar_gnn import GARGNN


def set_seed_tf(seed):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/", help='Director of the dataset.')
parser.add_argument('--user_samp', type=str, default='sepdot', help='Sampling method for user.')
parser.add_argument('--item_samp', type=str, default='sepdot', help='Sampling method for item.')
parser.add_argument('--samp_size', type=str, default='[25,25]', help='Sampling size.')
parser.add_argument('--embed_meth', type=str, default='bprmf', help='Emebdding method')
parser.add_argument('--batch_size', type=int, default=1024, help='Normal batch size.')
parser.add_argument('--out_start', type=int, default=0, help='Validation per training batch.')
parser.add_argument('--Ks', nargs='?', default='[20,50,100]', help='Output sizes of every layer')
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--test_batch_us', type=int, default=100)
parser.add_argument('--n_jobs', type=int, default=4, help='Multiprocessing number.')

parser.add_argument('--agg_meth', type=str, default='plain', choices=['plain', 'rec', 'none'])
parser.add_argument('--gan_model', type=str, default='gargnn')
parser.add_argument('--val_interval', type=int, default=20)
parser.add_argument('--patience', type=int, default=20, help='Early stop patience.')
parser.add_argument('--gan_epoch', type=int, default=10000)
parser.add_argument('--restore_all', type=str, default="")
parser.add_argument('--pretrain_gnn', type=int, default=0, choices=[0, 1])
parser.add_argument('--cold_test', type=int, default=1, choices=[0, 1])

parser.add_argument('--d_lr', type=float, default=1e-3, help='Multiprocessing number.')
parser.add_argument('--g_lr', type=float, default=1e-3, help='Multiprocessing number.')
parser.add_argument('--gen_lys', type=str, default='[200,200]', help='MLP layer num of stack machine.')
parser.add_argument('--real_lys', type=str, default='[256,256,1]', help='MLP layer num of stack machine.')
parser.add_argument('--gen_act', type=str, default='tanh', help='activation function of generator')
parser.add_argument('--real_act', type=str, default='relu', help='activation function of discriminator')
parser.add_argument('--gen_dropout', type=float, default=0.1, help='Drop Rate.')
parser.add_argument('--real_dropout', type=float, default=0.5, help='Drop Rate.')
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--gp_coe', type=float, default=0.9)
parser.add_argument('--sim_coe', type=float, default=0.1,
                    help='similarity coefficient for generative to approximate real')
parser.add_argument('--d_step', type=int, default=1)
parser.add_argument('--g_step', type=int, default=1)
parser.add_argument('--d_batch', type=int, default=1024)
parser.add_argument('--g_batch', type=int, default=1024)
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.samp_size = eval(args.samp_size)
args.gen_lys = eval(args.gen_lys)
args.real_lys = eval(args.real_lys)
args.Ks = eval(args.Ks)
args.gan_model = args.gan_model.upper()
set_seed_tf(args.seed)
pprint(vars(args))

timer = utils.Timer(name='main')
ndcg.init(args)
train_data = pd.read_csv(args.datadir + args.dataset + '/warm_map.csv', dtype=np.int64).values
train_batch = [(begin, min(begin + args.batch_size, len(train_data)))
               for begin in range(0, len(train_data) - args.batch_size, args.batch_size)]
content_data = np.load(args.datadir + args.dataset + '/item_content.npy')
para_dict = pickle.load(open(args.datadir + args.dataset + '/convert_dict.pkl', 'rb'))
save_path = './model_save/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_file = save_path + args.dataset + args.gan_model + args.embed_meth + str(args.samp_size[0]) + str(
    args.samp_size[1])
if len(args.restore_all) <= 1:
    param_file = str(uuid.uuid4())[:4]
else:
    param_file = args.restore_all
save_file += param_file
args.param_file = param_file
timer.toc().tic().logging('Model will be stored in ' + save_file)

t0 = time.time()
emb_path = ""
if args.agg_meth == 'plain':
    emb_path = "%s%s/%s_%d-%d_u(sepdot)_i(sepdot)_plain.npy" % (
        args.datadir, args.dataset, args.embed_meth, args.samp_size[0], args.samp_size[1],
    )
elif args.agg_meth == 'rec':
    emb_path = "%s%s/%s_%d-%d_u(rand)_i(rand)_rec.npy" % (
        args.datadir, args.dataset, args.embed_meth, args.samp_size[0], args.samp_size[1],
    )
elif args.agg_meth == 'none':
    emb_path = os.path.join(args.datadir, args.dataset, args.embed_meth + '.npy')

USER_NUM = para_dict['user_num']
ITEM_NUM = para_dict['item_num']
emb = np.load(emb_path)
user_emb = emb[:USER_NUM]
item_emb = emb[USER_NUM:]
patience_count = 0
va_auc_max = 0
time_plot_list = []
train_time = 0
val_time = 0
stop_flag = 0
batch_count = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
gan = eval(args.gan_model)(sess, args, emb.shape[-1], content_data.shape[-1])
saver = tf.train.Saver()
d_loss, g_loss = [], []
timer.toc().tic().logging("Training GAN model...")
for epoch in range(1, args.gan_epoch + 1):
    t_epoch_begin = time.time()

    # build train pair
    np.random.shuffle(train_data)
    neg_train_data = np.random.choice(para_dict['warm_item'], size=(len(train_data), 1), replace=True)
    pair_train_data = np.concatenate([train_data, neg_train_data], axis=1)

    for beg, end in train_batch:
        batch_count += 1
        t_train_begin = time.time()
        for step in range(args.d_step):
            batch_lbs = pair_train_data[beg: end]
            d_loss = gan.train_d(user_emb[batch_lbs[:, 0]],
                                 item_emb[batch_lbs[:, 1]],
                                 item_emb[batch_lbs[:, 2]],
                                 content_data[batch_lbs[:, 1]])
        for step in range(args.g_step):
            batch_lbs = pair_train_data[beg: end]
            g_loss = gan.train_g(user_emb[batch_lbs[:, 0]],
                                 item_emb[batch_lbs[:, 1]],
                                 content_data[batch_lbs[:, 1]])
        t_train_end = time.time()
        train_time += t_train_end - t_train_begin

        # Validation
        if (batch_count % args.val_interval == 0) or (batch_count < args.out_start):
            t_val_begin = time.time()
            num_val = batch_count // args.val_interval
            if args.pretrain_gnn:
                va_auc = ndcg.test(gan.get_ranked_rating,
                                   lambda u, i: gan.get_user_rating(u, i, user_emb, item_emb),
                                   ts_nei=para_dict['warm_val_user_nb'],
                                   ts_user=para_dict['warm_val_sub_user'],
                                   item_array=para_dict['item_array'],
                                   masked_items=para_dict['cold_item'],
                                   pos_dict=para_dict['pos_user_nb'],
                                   val=True)
            else:
                gen_emb = sess.run(gan.gen_emb, feed_dict={gan.content: content_data,
                                                           gan.g_training: False})
                if args.cold_test:
                    va_res = ndcg.test(gan.get_ranked_rating,
                                       lambda u, i: gan.get_user_rating(u, i, user_emb, gen_emb),
                                       ts_nei=para_dict['cold_val_user_nb'],
                                       ts_user=para_dict['cold_val_sub_user'],
                                       item_array=para_dict['item_array'],
                                       masked_items=para_dict['warm_item'],
                                       pos_dict=para_dict['pos_user_nb'],
                                       val=False
                                       )
                    va_auc = va_res['auc']
                else:
                    va_auc = ndcg.test(gan.get_ranked_rating,
                                       lambda u, i: gan.get_user_rating(u, i, user_emb, item_emb),
                                       ts_nei=para_dict['warm_val_user_nb'],
                                       ts_user=para_dict['warm_val_sub_user'],
                                       item_array=para_dict['item_array'],
                                       masked_items=para_dict['cold_item'],
                                       pos_dict=para_dict['pos_user_nb'],
                                       val=True,
                                       )
            if va_auc > va_auc_max:
                va_auc_max = va_auc
                saver.save(sess, save_file)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    stop_flag = 1
                    break
            t_val_end = time.time()
            val_time += t_val_end - t_val_begin
            print("D:", *d_loss)
            print("G:", *g_loss)
            timer.toc().tic().logging('Epo%d(%d/%d) VA_auc:%.4f|BestVA_auc:%.4f|Tr:%.2fs|Val:%.2fs' %
                                      (epoch, patience_count, args.patience,
                                       va_auc, va_auc_max, train_time, val_time))
            if args.cold_test:
                time_plot_list.append([args.gan_model, epoch, num_val, train_time, val_time,
                                       va_auc, va_res['precision'][0], va_res['recall'][0], va_res['ndcg'][0]])

    if stop_flag:
        break
timer.toc().tic().logging("Finish training GAN model.")

if args.gan_epoch > 0:
    train_file = './train/'
    if not os.path.exists(train_file):
        os.makedirs(train_file)
    plot_file_name = train_file + '%s_%s_%s_%s.csv' % (
        args.dataset, args.gan_model, args.embed_meth, args.agg_meth)
    df = pd.DataFrame(time_plot_list,
                      columns=['model', 'epoch', 'num_val', 'tr_time', 'val_time', 'auc', 'pre', 'rec', 'ndcg'])
    df.to_csv(plot_file_name, index=False)

saver.restore(sess, save_file)
print('#' * 30, args.dataset, args.embed_meth, args.gan_model, args.agg_meth, args.samp_size, '#' * 30)
# cold recommendation
gen_emb = sess.run(gan.gen_emb, feed_dict={gan.content: content_data,
                                           gan.g_training: False})
cold_res = ndcg.test(gan.get_ranked_rating,
                     lambda u, i: gan.get_user_rating(u, i, user_emb, gen_emb),
                     ts_nei=para_dict['cold_test_user_nb'],
                     ts_user=para_dict['cold_test_sub_user'],
                     item_array=para_dict['item_array'],
                     masked_items=para_dict['warm_item'],
                     pos_dict=para_dict['pos_user_nb'],
                     val=False)
timer.toc().tic().logging(
    'Cold-start recommendation result@{}: AUC, PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
        args.Ks[0], cold_res['auc'], cold_res['precision'][0], cold_res['recall'][0], cold_res['ndcg'][0]))
# warm recommendation
warm_res = ndcg.test(gan.get_ranked_rating,
                     lambda u, i: gan.get_user_rating(u, i, user_emb, item_emb),
                     ts_nei=para_dict['warm_test_user_nb'],
                     ts_user=para_dict['warm_test_sub_user'],
                     item_array=para_dict['item_array'],
                     masked_items=para_dict['cold_item'],
                     pos_dict=para_dict['pos_user_nb'],
                     val=False,
                     )
timer.toc().tic().logging("Warm recommendation result@{}: AUC, PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], warm_res['auc'], warm_res['precision'][0], warm_res['recall'][0], warm_res['ndcg'][0]))
# hybrid recommendation
hybrid_emb = np.copy(item_emb)
hybrid_emb[para_dict['cold_item']] = gen_emb[para_dict['cold_item']]
hybrid_res = ndcg.test(gan.get_ranked_rating,
                       lambda u, i: gan.get_user_rating(u, i, user_emb, hybrid_emb),
                       ts_nei=para_dict['hybrid_test_user_nb'],
                       ts_user=para_dict['hybrid_test_sub_user'],
                       item_array=para_dict['item_array'],
                       masked_items=None,
                       pos_dict=para_dict['pos_user_nb'],
                       val=False,
                       )
timer.toc().tic().logging("Hybrid recommendation result@{}: AUC, PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], hybrid_res['auc'], hybrid_res['precision'][0], hybrid_res['recall'][0], hybrid_res['ndcg'][0]))

sess.close()
result_file = './result/'
if not os.path.exists(result_file):
    os.makedirs(result_file)
with open(result_file + 'GCGAN-%s.txt' % args.dataset, 'a') as f:
    f.write(str(vars(args)))
    f.write(' | %.4f ' % cold_res['auc'])
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (cold_res['precision'][i], cold_res['recall'][i], cold_res['ndcg'][i]))
    f.write('| %.4f ' % (warm_res['auc']))
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (warm_res['precision'][i], warm_res['recall'][i], warm_res['ndcg'][i]))
    f.write('| %.4f ' % (hybrid_res['auc']))
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (hybrid_res['precision'][i], hybrid_res['recall'][i], hybrid_res['ndcg'][i]))
    f.write('\n')
