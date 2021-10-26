from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import math
import copy
import itertools
import logging
import pickle
import psutil
import datetime
import tensorflow as tf
import scipy.sparse as sp

from utils import *
from models import GeniePath, GAT, GCN
from cython_sampler import BanditMPSampler, BanditSampler, ThanosExp3, BanditLinearSampler, Thanos
from scipy.sparse.linalg import norm as sparsenorm
from ogb.nodeproppred import Evaluator
from utils import OurEvaluator
from log_utils import TBXWrapper
tb_logger = TBXWrapper()

epsilon = 1e-6

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'Cora', 'Dataset string.')
flags.DEFINE_string('rootlogdir', './log', 'rootdir for log folder')
flags.DEFINE_string('logdir', 'log_tmp', 'Dataset string.')
flags.DEFINE_string('logger_name', 'run', 'Dataset string.')
flags.DEFINE_string('model', 'GCN', 'Name of Model')
flags.DEFINE_string('sampler', 'BanditSampler', 'Name of Smapler')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('sample_interval', -1, 'Delta T')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('neighbor_limit', 10, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batchsize', 256, 'Batch size.')
flags.DEFINE_integer('residual', 1, 'Residual.')
flags.DEFINE_float('eta', 0.4, 'Eta.')
flags.DEFINE_float('delta', 0.01, 'Delta.')
flags.DEFINE_float('max_reward', 1.0, 'Max reward.')
flags.DEFINE_integer('num_proc', 12, 'Number of process.')
flags.DEFINE_integer('test_interval', 0, 'Test Interval')
flags.DEFINE_bool('laplacian', False, 'Use Laplacian matrix.')
flags.DEFINE_bool('realbs', True, 'Use real reward for BS')
flags.DEFINE_bool('add_selfloop', False, 'Add Self Loop while Preprocess Data?')
flags.DEFINE_bool('plotreward', False, 'Whether to plot the reward')
flags.DEFINE_bool('shuffle', True, 'Whether to shuffle the order of batch')
flags.DEFINE_bool('normalizeD', False, 'Whether to shuffle the order of batch')
flags.DEFINE_integer('repeat_test_sampling', 1, 'Residual.')



def get_logger_addr(rootdir='/efs/GNNBanditLog/log'):
    interval = FLAGS.sample_interval
    realR = "real" if FLAGS.realbs else "woh2"
    if FLAGS.eta == 1.:
        sampler_name = 'GraphSage'
    else:
        if interval < 0:
            if FLAGS.sampler not in ['ThanosExp3']:
                sampler_name = "%s-%s"%(FLAGS.sampler, realR)
            else:
                sampler_name = FLAGS.sampler
        elif FLAGS.sampler == 'ThanosExp3':
            sampler_name = "%s-%d"%('ThanosExp3', interval)
        elif FLAGS.sampler == 'Thanos':
            sampler_name = "%s-%d"%('Thanos', interval)
        else:
            sampler_name = 'REXP3-%s-%d'%(FLAGS.realbs, interval) 
    name = "{sampler}_{model}-{hidden1}_LP-{laplacian}_SL-{add_selfloop}_NL-{NL:d}_BS-{BS:d}_E-{epochs:d}_eta-{eta:.2f}_delta-{delta:.2f}_MaxR-{maxR:.1f}_lr-{lr:.3f}_dropout-{dropout:.2f}_WD-{weight_decay:.5f}_Re-{repeat_test_sampling}_TInt-{test_interval}_{raw_name}"\
            .format(
                sampler = sampler_name,
                NL = FLAGS.neighbor_limit,
                BS = FLAGS.batchsize,
                epochs = FLAGS.epochs,
                eta = FLAGS.eta, 
                delta = FLAGS.delta,
                maxR = FLAGS.max_reward, 
                lr = FLAGS.learning_rate, 
                raw_name = FLAGS.logger_name,
                model = FLAGS.model,
                hidden1 = FLAGS.hidden1, 
                dropout = FLAGS.dropout, 
                weight_decay = FLAGS.weight_decay, 
                add_selfloop = FLAGS.add_selfloop,
                test_interval = FLAGS.test_interval ,
                laplacian = FLAGS.laplacian, 
                repeat_test_sampling = FLAGS.repeat_test_sampling, 
            )
    abslogdir = os.path.join(FLAGS.rootlogdir, FLAGS.logdir)
    No = get_run_number(name, FLAGS)
    logger_addr = os.path.join(abslogdir, No + "_" + name)
    return logger_addr, No +"_"+ name


def get_DeltaT(d, epoch_len, max_th=1e100):
    DeltaT = {
        0: epoch_len,
        -1: -1,
    }
    if d>0:
        return d
    else:
        return DeltaT[d]

def get_run_number(name, opt, rootdir='/efs/GNNBanditLog/log'):
    log_dir = os.path.join(opt.rootlogdir, opt.logdir)
    if not os.path.exists(log_dir):
        os.mkdir( log_dir )
    os_list = os.listdir(log_dir) 
    for item in os_list:
        if name in item:
            return item.split('_')[0]
    return str(len(os_list) + 1)

def plot_reward(tb_logger, outs_2, sampler, src_list, dst_list, global_step):
    if FLAGS.plotreward and FLAGS.sampler in ['ThanosExp3', 'Thanos']:
        rewards = outs_2
    elif FLAGS.plotreward and FLAGS.sampler in ['BanditSampler', 'BanditLinearSampler']:
        rewards = np.zeros_like(outs_2)
        idx = 0
        for (src, dst) in zip(src_list, dst_list):
            prob = sampler.get_sample_probs(src, dst)
            rewards[idx] = outs_2[idx] / (prob**2) 
            idx += 1
    tb_logger.log_value('Mean_Rewards', np.mean(rewards), step=global_step)
    tb_logger.log_value('Max_Rewards', np.max(rewards), step=global_step)
    tb_logger.log_value('Min_Rewards', np.min(rewards), step=global_step)


def iterate_minibatches(inputs, batchsize, shuffle=False):
    assert inputs is not None
    num_samples = inputs[-1].shape[0]
    nodes = inputs[0]
    labels = inputs[1]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    if num_samples <= batchsize:
        yield np.array(nodes, np.int32), labels
    else:
        num_batch = int(math.ceil(float(num_samples) / batchsize))
        for idx in range(num_batch):
            if (idx+1)*batchsize < num_samples:
                excerpt = indices[idx*batchsize:(idx+1)*batchsize]
            else:
                excerpt = indices[idx*batchsize:]
                excerpt = np.concatenate((excerpt,indices[:(idx+1)*batchsize-num_samples]), axis=0)
            yield np.array(nodes[excerpt], np.int32), labels[excerpt]


def gen_subgraph(sampler, selected_nodes, adj, num_layer=2, neighbor_limit=10, degs=None):
    edges = sampler.sample_graph(selected_nodes)
    edges = sorted(edges, key=lambda element: (element[0], element[1]))

    expand_list = set()
    for (src, dst) in edges:
        expand_list.add(src)
        expand_list.add(dst)
    for nod in selected_nodes:
        expand_list.add(nod)
    expand_list = list(expand_list)
    expand_list = sorted(expand_list)

    node_map = {}
    inverse_node_map = {}
    m_id = 0
    for nod in expand_list:
        node_map[nod] = m_id
        inverse_node_map[m_id] = nod
        m_id += 1

    src_list = []
    dst_list = []
    n2n_indices_batch=[]
    n2n_values_batch=[]
    
    sample_degree = {}
    for dst in set([e[1] for e in edges]):
        sample_degree[dst] = 0
    for (src, dst) in edges:
        sample_degree[dst] += 1

    for (src, dst) in edges:
        n2n_indices_batch.append([node_map[src], node_map[dst]])
        src_list.append(src)
        dst_list.append(dst)
        n2n_values_batch.append(adj[src, dst])
    n2n_indices_batch = np.array(n2n_indices_batch)
    n2n_values_batch = np.array(n2n_values_batch)

    left_indices_batch = [None]*len(n2n_indices_batch)
    left_values_batch = np.ones(len(n2n_indices_batch))
    right_indices_batch = [None]*len(n2n_indices_batch)
    right_values_batch = np.ones(len(n2n_indices_batch))
    ii = 0
    for n1, n2 in n2n_indices_batch:
        left_indices_batch[ii] = [ii, n1]
        right_indices_batch[ii] = [ii, n2]
        ii += 1

    node_indices_batch = []
    node_values_batch = np.ones(len(selected_nodes))
    ii = 0
    for nod in selected_nodes:
        node_indices_batch.append([ii, node_map[nod]])
        ii += 1
    node_indices_batch = np.array(node_indices_batch)
    node_values_batch = np.array(node_values_batch)

    n2n = tf.SparseTensorValue(n2n_indices_batch, n2n_values_batch, [m_id, m_id])
    left = tf.SparseTensorValue(left_indices_batch, left_values_batch, [len(left_indices_batch), m_id])
    right = tf.SparseTensorValue(right_indices_batch, right_values_batch, [len(right_indices_batch), m_id])
    node_select = tf.SparseTensorValue(node_indices_batch, node_values_batch, [len(node_indices_batch), m_id])
    return expand_list, n2n, left, right, node_select, src_list, dst_list, node_map


def gen_fullgraph(selected_nodes, adj, n2n_values_batch):
    adj_coo = adj.tocoo()
    m_id = adj.shape[0]

    n_edges = len(adj_coo.row)
    n2n_indices_batch=np.concatenate(
            [adj_coo.row[:,np.newaxis], adj_coo.col[:,np.newaxis]], axis=1)
    n2n_values_batch = adj_coo.data

    left_indices_batch = np.concatenate(
            [np.arange(n_edges)[:,np.newaxis],
             adj_coo.row[:,np.newaxis]], axis=1)
    left_values_batch = np.ones(len(n2n_indices_batch))
    right_indices_batch = np.concatenate(
            [np.arange(n_edges)[:,np.newaxis],
             adj_coo.col[:,np.newaxis]], axis=1)
    right_values_batch = np.ones(len(n2n_indices_batch))

    node_indices_batch = np.concatenate(
            [np.arange(len(selected_nodes))[:,np.newaxis],
             selected_nodes[:,np.newaxis]], axis=1)
    node_values_batch = np.ones(len(selected_nodes))

    n2n = tf.SparseTensorValue(n2n_indices_batch, n2n_values_batch, [m_id, m_id])
    left = tf.SparseTensorValue(left_indices_batch, left_values_batch, [len(left_indices_batch), m_id])
    right = tf.SparseTensorValue(right_indices_batch, right_values_batch, [len(right_indices_batch), m_id])
    node_select = tf.SparseTensorValue(node_indices_batch, node_values_batch, [len(node_indices_batch), m_id])
    return n2n, left, right, node_select

def get_node_select_for_trainfeed(selected_nodes, m_id):
    node_indices_batch = []
    node_values_batch = np.ones(len(selected_nodes))
    ii = 0
    for nod in selected_nodes:
        node_indices_batch.append([ii, nod])
        ii += 1
    node_indices_batch = np.array(node_indices_batch)
    node_select = tf.SparseTensorValue(node_indices_batch, node_values_batch, [len(node_indices_batch), m_id])
    return node_select


def construct_feed_dict(n_nd, features, node_select, support, left, right, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['node_select']: node_select})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['left']: left})
    feed_dict.update({placeholders['right']: right})
    feed_dict.update({placeholders['n_nd']: n_nd})
    return feed_dict


def convert_support(support, left, right):
    indices = []
    for i in range(len(support[0])):
        indices.append([left[support[0][i][0]], right[support[0][i][1]]])
    return (np.array(indices), support[1], support[2])


def main():
    logger_addr, logger_name = get_logger_addr()
    tb_logger.configure(logger_addr, flush_secs=5, opt=FLAGS)
    rootlogging = logging.getLogger()
    for h in rootlogging.handlers:
        rootlogging.removeHandler(h)
    logging.basicConfig(
        filename=os.path.join(logger_addr, "log.txt"),
        filemode='w',
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(logger_addr)
    No = logger_name.split('_')[0]
    process = psutil.Process()
    
    data_name = "_".join(FLAGS.dataset.split("-"))
    adj_full, features, train_nodes, y_train, \
        valid_nodes, y_valid, test_nodes, y_test = load_data(FLAGS.dataset, flag=FLAGS)
    if FLAGS.laplacian:
        adj_full = adj_full.tocoo()
        d = adj_full.sum(0).A ** -0.5
        d[np.isinf(d)] = 0
        D = sp.diags(d, [0])
        adj_full = D * adj_full * D
        adj_full = adj_full.tolil()
        print("Use Laplacian")
    print("Finish loading data.")

    # adj_train equals adj_full in ogb dataset
    adj_train = adj_full

    evaluator = OurEvaluator(name = FLAGS.dataset)
    Eval_Keys = {
        'ogbn-proteins': 'rocauc',
        'ogbn-products': 'acc',
        'ogbn-arxiv': 'acc',
        'Cora': 'acc', 
        'Pubmed': 'acc',
        'CoraFull': 'acc', 
        'Reddit': 'acc',
        'chameleon': 'acc',
        'cornell': 'acc',
        'squirrel': 'acc',
        'film': 'acc',
        'texas': 'acc',
        'wisconsin': 'acc',
        'flickr': 'acc',
        'yelp': 'rocauc',
    }
    eval_key = Eval_Keys[FLAGS.dataset]

    if FLAGS.sampler == 'BanditSampler':
        sampler = BanditSampler()
    elif FLAGS.sampler == 'BanditMPSampler':
        sampler = BanditMPSampler()
    elif FLAGS.sampler == 'ThanosExp3':
        sampler = ThanosExp3()
    elif FLAGS.sampler == 'BanditLinearSampler':
        sampler = BanditLinearSampler()
    elif FLAGS.sampler == "Thanos":
        sampler = Thanos()
    else:
        print("Input the Right Sampler")
        return

    sampler.init(adj_train)
    print("Finish init %s."%(FLAGS.sampler))

    n2n_values = np.ones(adj_full.count_nonzero(), dtype=np.float32)

    feature_dim = features.shape[-1]
    label_dim = y_train.shape[-1]
    numNode = adj_full.shape[0]

    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None,feature_dim)),
        'node_select': tf.sparse_placeholder(tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(None,label_dim)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'left': tf.sparse_placeholder(tf.float32),
        'right': tf.sparse_placeholder(tf.float32),
        'n_nd': tf.placeholder(tf.int32, shape=[]),
    }

    # Define task type
    task_type_dict = {
        "ogbn-proteins": "multi-label",
        "ogbn-products": "exclusive-label",
        "ogbn-arxiv": "exclusive-label",
        "Cora": "exclusive-label",
        "Pubmed": "exclusive-label",
        "CoraFull": "exclusive-label",
        "Reddit": "exclusive-label", 
        'chameleon': "exclusive-label",
        'cornell': "exclusive-label",
        'squirrel': "exclusive-label",
        'film': "exclusive-label",
        'texas': "exclusive-label",
        'wisconsin': "exclusive-label",
        'flickr': "exclusive-label",
        'yelp': "multi-label",
    }
    task_type = task_type_dict[FLAGS.dataset]

    # Create model
    if FLAGS.model == 'GCN':
        model = GCN(task_type, placeholders, input_dim=features.shape[-1], label_dim=label_dim)
    elif FLAGS.model == 'GAT':
        model = GAT(task_type, placeholders, input_dim=features.shape[-1], label_dim=label_dim)
    elif FLAGS.model == 'GeniePath':
        model = GeniePath(task_type, placeholders, input_dim=features.shape[-1], label_dim=label_dim)
    else:
        raise ValueError("Invalid Model Name: %s\n"%(FLAGS.model))

    # Initialize session
    sess = tf.Session()

    # Construct val feed dictionary
    support, left, right, node_select = gen_fullgraph(valid_nodes, adj_full, n2n_values)
    val_feed_dict = construct_feed_dict(
            adj_full.shape[0], features, node_select, support, left, right, y_valid, placeholders)
    val_feed_dict.update({placeholders['dropout']: 0.})

    # Construct test feed dictionary
    support, left, right, node_select = gen_fullgraph(test_nodes, adj_full, n2n_values)
    test_feed_dict = construct_feed_dict(
            adj_full.shape[0], features, node_select, support, left, right, y_test, placeholders)
    test_feed_dict.update({placeholders['dropout']: 0.})

    # Construct full-graph train feed dictionary
    support, left, right, node_select = gen_fullgraph(train_nodes, adj_full, n2n_values)
    train_feed_dict = construct_feed_dict(
            adj_full.shape[0], features, node_select, support, left, right, y_train, placeholders)
    train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Define model evaluation function
    def evaluate(labels, feed_dict):
        outs = sess.run([model.outputs], feed_dict=feed_dict)
        preds = outs[0].tolist()
        eval_true = np.array(labels)
        eval_pred = np.array(preds)

        # evaluate
        if task_type == "exclusive-label":
            eval_true = np.argmax(eval_true, axis=1).reshape([-1,1])
            eval_pred = np.argmax(eval_pred, axis=1).reshape([-1,1])
        eval_res = evaluator.eval({"y_true": eval_true, "y_pred": eval_pred})[eval_key]

        return eval_res

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if not os.path.exists("./save_models"):
        os.mkdir("./save_models")

    train_true = []
    train_pred = []
    best_va = 0

    global_step = 0
    epoch_len = int(math.ceil(float(y_train.shape[0]) / FLAGS.batchsize))
    DeltaT = get_DeltaT(d=FLAGS.sample_interval, epoch_len=epoch_len)
    degs = adj_train.sum(0).A[0]

    # Train model
    for epoch in range(FLAGS.epochs):
        step = 0
        train_losses = []
        tic = time.time()

        if not os.path.exists("./save_models/%s" % FLAGS.logdir ):
            os.mkdir("./save_models/%s" % FLAGS.logdir)
        saver.save(sess, "./save_models/{}.ckpt".format(os.path.join(FLAGS.logdir, logger_name)))

        stime = datetime.datetime.now()
        for batch in iterate_minibatches(
                [train_nodes, y_train], batchsize=FLAGS.batchsize, shuffle=FLAGS.shuffle):
            batch_nodes, y_batch = batch
            step += 1
            global_step += 1

            # Reset the sampler
            if DeltaT > 0 and global_step % DeltaT == 0:
                sampler.reset_parallel()
                logging.info("[Step %d] Reseted the Sampler"%global_step)

            subgraph_nodes, support, left, right, node_select, src_list, dst_list, node_map = \
                gen_subgraph(sampler, batch_nodes, adj_train, neighbor_limit=FLAGS.neighbor_limit, degs=degs)

            features_inputs = features[subgraph_nodes, :]

            # Construct feed dictionary
            feed_dict = construct_feed_dict(
                    len(node_map), features_inputs, node_select, support, left, right, y_batch, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            if FLAGS.sampler in ['ThanosExp3', 'Thanos']:
                run_list = [model.opt_op, model.loss, model.cosine_reward_addedge, model.outputs]
            else:
                if FLAGS.realbs:
                    run_list = [model.opt_op, model.loss, model.real_bs_reward, model.outputs]
                else:
                    run_list = [model.opt_op, model.loss, model.sparse_attention_l0, model.outputs]
            outs = sess.run(run_list, feed_dict=feed_dict)
            train_losses.append(outs[1])
            tb_logger.log_value('loss', outs[1], step=global_step)
            logging.info("[%s] %s | Epoch: %d | Step: %d | Loss %.3f "%(No, FLAGS.dataset, epoch, global_step, outs[1]))
            tb_logger.log_value("rss", process.memory_info().rss, step=global_step)
            if FLAGS.plotreward:
                plot_reward(tb_logger, outs[2], sampler, src_list, dst_list, global_step)
                batch_node_select = get_node_select_for_trainfeed(batch_nodes, m_id=numNode)
                train_feed_dict.update({placeholders['node_select']: batch_node_select})
                train_feed_dict.update({placeholders['labels']: y_batch})
                n2npool_h0_full = sess.run([model.n2npool_l0_selected_nodes], feed_dict=train_feed_dict)[0]
                n2npool_h0_sub = sess.run([model.n2npool_l0_selected_nodes], feed_dict=feed_dict)[0]
                dist_vector = np.sum( (n2npool_h0_full - n2npool_h0_sub)**2, axis=1)
                if FLAGS.normalizeD:
                    full_square_norm = np.sum( (n2npool_h0_full)**2, axis=1 )
                    dist_with_true = np.mean( dist_vector / (full_square_norm + 1e-12) )
                else:
                    dist_with_true = np.mean( dist_vector )

                tb_logger.log_value("dist_with_true", dist_with_true, step=global_step)

            # Update sample probs
            if FLAGS.eta < 1:
                sampler.update(np.array(src_list, dtype=np.int32), np.array(dst_list, dtype=np.int32), outs[2])

            train_true.extend(y_batch.tolist())
            train_pred.extend(outs[3].tolist())

            if FLAGS.test_interval == 0 and step == 1 \
                    or FLAGS.test_interval != 0 and global_step % FLAGS.test_interval == 0:
                subgraph_nodes, support, left, right, node_select, src_list, dst_list, node_map = \
                    gen_subgraph(sampler, np.array(test_nodes, np.int32), adj_train, neighbor_limit=FLAGS.neighbor_limit, degs=degs)
                features_inputs = features[subgraph_nodes, :]
                # Construct feed dictionary
                test_feed_dict_infer = construct_feed_dict(
                        len(node_map), features_inputs, node_select, support, left, right, y_train, placeholders)
                test_feed_dict_infer.update({placeholders['dropout']: 0.})

                eval_te = evaluate(y_test, test_feed_dict_infer)
                tb_logger.log_value("Eva_Test", eval_te, step=global_step)
                logging.info("Epoch: %d | Step: %d | %s_test %.4f "%(
                        epoch+1, global_step, eval_key, eval_te) )
                tb_logger.save_log()

        etime = datetime.datetime.now()
        tb_logger.log_value("Epoch_Time", (etime-stime).total_seconds(), step=epoch)

        # Compute Train eval
        if task_type == "exclusive-label":
            train_true = np.argmax(train_true, axis=1).reshape([-1,1])
            train_pred = np.argmax(train_pred, axis=1).reshape([-1,1])
        eval_tr = evaluator.eval({"y_true": np.array(train_true), "y_pred": np.array(train_pred)})[eval_key]
        train_true = []
        train_pred = []

        # Valid
        subgraph_nodes, support, left, right, node_select, src_list, dst_list, node_map = \
            gen_subgraph(sampler, np.array(valid_nodes, np.int32), adj_train, neighbor_limit=FLAGS.neighbor_limit, degs=degs)
        features_inputs = features[subgraph_nodes, :]
        # Construct feed dictionary
        val_feed_dict_infer = construct_feed_dict(
                len(node_map), features_inputs, node_select, support, left, right, y_valid, placeholders)
        val_feed_dict_infer.update({placeholders['dropout']: 0.})

        eval_va = evaluate(y_valid, val_feed_dict_infer)
        tb_logger.log_value('Eva_Train', eval_tr, step=global_step)
        tb_logger.log_value('Eva_Val', eval_va, step=global_step)
        logging.info("Epoch: %d | Acc_val: %.5f | Acc_train: %.5f  "%(
                        epoch+1, eval_va, eval_tr) )
        tb_logger.save_log()

        if eval_va > best_va:
            best_va = eval_va
            if not os.path.exists("./save_models/%s" % FLAGS.logdir ):
                os.mkdir("./save_models/%s" % FLAGS.logdir )
            saver.save(sess, "./save_models/{}.ckpt".format( os.path.join(FLAGS.logdir, logger_name)) )
            tb_logger.log_value("Best_Val", best_va, step=epoch)
            tb_logger.log_value("Best_Val_Epoch", epoch, step=epoch)
            tb_logger.save_log()
            sampler_object = sampler.topy()
            with open("./save_models/{}.pkl".format(os.path.join(FLAGS.logdir, logger_name)), 'wb') as f:
                pickle.dump(sampler_object, f)
            logging.info("Saved the Sampler")

            if epoch >= 0  and FLAGS.test_interval >= 0: 
                logging.info("Start Testing: Epoch %d | Step %d | Best_Val %.5f"%(epoch, global_step, best_va))
                eva_true = np.array(y_test)
                eva_pred = np.zeros_like(eva_true)
                for k in range(FLAGS.repeat_test_sampling):
                    subgraph_nodes, support, left, right, node_select, src_list, dst_list, node_map = \
                        gen_subgraph(sampler, np.array(test_nodes, np.int32), adj_train, neighbor_limit=FLAGS.neighbor_limit, degs=degs)
                    features_inputs = features[subgraph_nodes, :]
                    # Construct feed dictionary
                    test_feed_dict_infer = construct_feed_dict(
                            len(node_map), features_inputs, node_select, support, left, right, y_train, placeholders)
                    test_feed_dict_infer.update({placeholders['dropout']: 0.})
                    outs = sess.run([model.outputs], feed_dict=test_feed_dict_infer)
                    eva_pred = eva_pred + np.array(outs[0].tolist())
                eva_pred = eva_pred / FLAGS.repeat_test_sampling
                if task_type == "exclusive-label":
                    eva_true = np.argmax(eva_true, axis=1).reshape([-1,1])
                    eva_pred = np.argmax(eva_pred, axis=1).reshape([-1,1])
                eval_te = evaluator.eval({"y_true": eva_true, "y_pred": eva_pred})[eval_key]

                tb_logger.log_value("Test_Best_Val", eval_te, step=epoch)
                logging.info("Epoch: %d | Step: %d | Test_Best_Val %.5f "%(
                        epoch+1, global_step, eval_te) )
                tb_logger.save_log()

    # Testing
    saver.restore(sess, "./save_models/{}.ckpt".format(os.path.join(FLAGS.logdir, logger_name)) )
    logging.info("Restored the best validiation model.")
    
    # Restore the sampler
    sampler.frompy(*sampler_object)
    logging.info("Restored the sampler")
    subgraph_nodes, support, left, right, node_select, src_list, dst_list, node_map = \
        gen_subgraph(sampler, np.array(test_nodes, np.int32), adj_train, neighbor_limit=FLAGS.neighbor_limit, degs=degs)
    features_inputs = features[subgraph_nodes, :]
    # Construct feed dictionary
    test_feed_dict_infer = construct_feed_dict(
            len(node_map), features_inputs, node_select, support, left, right, y_train, placeholders)
    test_feed_dict_infer.update({placeholders['dropout']: 0.})

    eval_te = evaluate(y_test, test_feed_dict_infer)
    tb_logger.log_value("Test_Best_Val", eval_te, step=epoch)
    logging.info("Final Test Acc: %.5f "%(eval_te))
    tb_logger.save_log()

    eval_te_full = evaluate(y_test, test_feed_dict)
    tb_logger.log_value("Final_Test_Full", eval_te_full, step=epoch)
    tb_logger.save_log()
    logging.info("Final Test Full: {:.5f}".format(eval_te_full))


if __name__=="__main__":
    main()
