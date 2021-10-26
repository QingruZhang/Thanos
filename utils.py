import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from scipy.sparse.linalg.eigen.arpack import eigsh
import os
import sys
import time
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import norm as sparsenorm
from scipy.linalg import qr
from sklearn.metrics import f1_score
from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import Evaluator
import networkx as nx
import dgl
import ogb
import torch
import random


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    if len(shape) == 2:
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    elif len(shape) == 1:
        init_range = np.sqrt(6.0 / shape[0])
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.get_variable(initializer=initial, name=name)


def calc_f1(task_type, y_true, y_pred):
    if task_type == "multi-label":
        #y_pred[y_pred > 0.5] = 1
        #y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.] = 1
        y_pred[y_pred <= 0.] = 0
        return f1_score(y_true, y_pred, average="micro")
    else:
        y_pred = np.argmax(y_pred, 1)
        y_true = np.argmax(y_true, 1)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        return f1_micro


def accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, 1)
    y_true = np.argmax(y_true, 1)
    return np.sum(y_pred == y_true) / float(len(y_true))

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess(name, root, dataset, add_selfloop=True):
    splitted_idx = dataset.get_idx_split()
    graph = dataset[0][0]
    labels = dataset[0][1]

    # create gnn_bs path
    dir_gnn_bs = os.path.join(root, "gnn_bs")
    if not os.path.exists(dir_gnn_bs):
        os.mkdir(dir_gnn_bs)

    # num_node = len(graph['node_feat'])
    num_node = graph['num_nodes']
    edge_index = graph['edge_index']
    edge_feat = graph['edge_feat']
    num_edge = len(edge_index[0])
    # feature_dim = len(graph['node_feat'][0])
    train_indices = splitted_idx['train']

    # add self-loop edges
    edge_index_list = []
    edge_index_list.append(edge_index[0].tolist())
    edge_index_list.append(edge_index[1].tolist())
    if add_selfloop:
        for node in range(num_node):
            edge_index_list[0].append(node)
            edge_index_list[1].append(node)
            num_edge += 1

    # adj_full
    adj_full = sp.coo_matrix((np.ones(num_edge), (edge_index_list[0], edge_index_list[1])),
                             shape=(num_node, num_node))
    sp.save_npz("{}/adj_full.npz".format(dir_gnn_bs), sp.csr_matrix(adj_full))

    # node feats
    node_feat = None
    if name == "ogbn-proteins":
        # average edge feats to get node feats
        node_feat_list = []
        adj_sum = adj_full.sum(axis=1)
        print("generate node feats...")
        for i in tqdm(range(edge_feat.shape[1])):
            edge_feat_i = sp.coo_matrix((edge_feat[:,i], (edge_index[0], edge_index[1])),
                                        shape=(num_node, num_node))
            node_feat_list.append(edge_feat_i.sum(axis=1)/adj_sum)
        node_feat = np.concatenate(node_feat_list, axis=1)
    elif name in ["ogbn-products", "ogbn-arxiv"]:
        node_feat = graph['node_feat']
    else:
        print("unknown data name: {}".format(name))
        sys.exit(0)

    # labels
    if name in ["ogbn-products", "ogbn-arxiv"]:
        label_idx = labels.reshape([-1])
        labels = np.zeros((label_idx.size, label_idx.max()+1))
        labels[np.arange(label_idx.size), label_idx] = 1
    np.save(open("{}/labels.npy".format(dir_gnn_bs), "wb"), labels)

    # feats
    np.save(open("{}/feats.npy".format(dir_gnn_bs), "wb"), node_feat)

    # splitted_idx
    np.save(open("{}/splitted_idx.npy".format(dir_gnn_bs), "wb"), splitted_idx)

def load_OGBdataset(name, flag=None):
    dir_name = "_".join(name.split("-"))
    root = os.path.join("/efs/GNNBanditLog/dataset", dir_name)
    add_selfloop = False if flag is not None and not flag.add_selfloop else True
    root = os.path.join(root, "add_selfloop" if add_selfloop else "no_selfloop" )
    dir_gnn_bs = os.path.join(root, "gnn_bs")

    if not os.path.exists(root):
        os.mkdir(root)

    if not os.path.exists(dir_gnn_bs):
    # if True:
        dataset = NodePropPredDataset(name)
        print("data preprocess...")
        preprocess(name, root, dataset, add_selfloop=add_selfloop)
        adj_full = sp.load_npz('{}/adj_full.npz'.format(dir_gnn_bs)).astype(np.bool)
    else:
        adj_full = sp.load_npz('{}/adj_full.npz'.format(dir_gnn_bs)).astype(np.bool)
    splitted_idx = np.load('{}/splitted_idx.npy'.format(dir_gnn_bs), allow_pickle=True).item()
    feats = np.load('{}/feats.npy'.format(dir_gnn_bs))
    labels = np.load('{}/labels.npy'.format(dir_gnn_bs))

    adj_full = adj_full.tolil()
    train_nodes = splitted_idx['train']
    y_train = labels[train_nodes]
    valid_nodes = splitted_idx['valid']
    y_valid = labels[valid_nodes]
    test_nodes = splitted_idx['test']
    y_test = labels[test_nodes]

    return adj_full, feats, train_nodes, y_train, \
           valid_nodes, y_valid, test_nodes, y_test

def get_test_val_train_nodes(labels, num_test=20, num_val=30):
    num_label = np.max(labels)+1
    test_nodes = []
    val_nodes = []
    for i in range(num_label):
        indexs = np.where(labels==i)[0]
        if len(indexs) <= num_test + num_val:
            continue
        else:
            test_nodes.append(indexs[:num_test])
            val_nodes.append(indexs[num_test:num_test+num_val])
    test_np = np.concatenate(test_nodes)
    val_np = np.concatenate(val_nodes)
    train_mask = np.full(labels.shape, True, dtype=bool)
    for i in range(labels.shape[0]):
        if i in test_np or i in val_np:
            train_mask[i] = False
    train_np = np.arange(labels.shape[0])[train_mask]
    return test_np, val_np, train_np

def load_CoraFull(name='CoraFull', flag=None):
    if name == "CoraFull":
        dataset = dgl.data.CoraFullDataset()
    graph = dataset[0]

    labels_np = graph.ndata['label'].numpy()
    feat = graph.ndata['feat'].numpy()
    adj = graph.adj(scipy_fmt='coo').tolil()
    labels = np.zeros((labels_np.size, labels_np.max()+1))
    labels[np.arange(labels_np.size), labels_np] = 1.
    
    nodes = np.arange(feat.shape[0])
    test_nodes, valid_nodes, train_nodes \
        = get_test_val_train_nodes(labels=labels_np)
    y_train = labels[train_nodes]
    y_test = labels[test_nodes]
    y_valid = labels[valid_nodes]

    return adj, feat, train_nodes, y_train,\
            valid_nodes, y_valid, test_nodes, y_test 


def load_DGLdataset(name, flag=None):
    if name == 'Cora':
        dataset = dgl.data.CoraGraphDataset()
    elif name == 'Pubmed':
        dataset = dgl.data.PubmedGraphDataset()
    elif name == 'Reddit':
        dataset = dgl.data.RedditDataset()
    elif name == 'CoraFull':
        return load_CoraFull(name, flag=flag)
    else:
        raise ValueError("unknown data name: {}".format(name))
    graph = dataset[0]
    
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    extend_mask = ~ ( val_mask | test_mask )
    labels_np = graph.ndata['label'].numpy()
    feat = graph.ndata['feat'].numpy()

    adj = graph.adj(scipy_fmt='coo').tolil()

    nodes = np.arange(feat.shape[0])
    # train_nodes = nodes[train_mask]
    train_nodes = nodes[train_mask] if name in ['Reddit'] else nodes[extend_mask]
    valid_nodes = nodes[val_mask]
    test_nodes = nodes[test_mask] 
    labels = np.zeros((labels_np.size, labels_np.max()+1))
    labels[np.arange(labels_np.size), labels_np] = 1.
    # y_train = labels[train_mask]
    y_train = labels[train_mask] if name in ['Reddit'] else labels[extend_mask]
    y_valid = labels[val_mask]
    y_test = labels[test_mask]

    return adj, feat, train_nodes, y_train,\
            valid_nodes, y_valid, test_nodes, y_test

def load_heteDataset(name, split_ratio = [0.6, 0.2, 0.2], bi_direction = True, flag = None):
    graph_adjacency_list_file_path = os.path.join('hete_data', name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('hete_data', name, f'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    g = dgl.DGLGraph(adj)

    if bi_direction:
        src, dst = g.edges()
        g.add_edges(dst, src)

    if flag.add_selfloop:
        g = dgl.add_self_loop(g)
        print("Add Self Loop")
    else:
        g = dgl.remove_self_loop(g)
        print("Remove Self Loop")

    features = preprocess_features(features)

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)

    N = g.num_nodes()
    all_idx = np.arange(N)
    random.shuffle(all_idx)

    train_ratio, val_ratio, test_ratio = split_ratio

    train_idx = torch.tensor(all_idx[: int(N * train_ratio)])
    val_idx = torch.tensor(all_idx[int(N * train_ratio): int(N * (train_ratio + val_ratio))])
    test_idx = torch.tensor(all_idx[int(N * (train_ratio + val_ratio)):])

    adj = g.adj(scipy_fmt='coo').tolil()

    labels_np = labels
    labels = np.zeros((labels_np.size, labels_np.max()+1))
    labels[np.arange(labels_np.size), labels_np] = 1.

    y_train = torch.FloatTensor(labels[train_idx])
    y_valid = torch.FloatTensor(labels[val_idx])
    y_test = torch.FloatTensor(labels[test_idx])

    return adj, features, train_idx, y_train, val_idx, y_valid, test_idx, y_test
 
def load_SAINTdataset(prefix):
    # Borrowed from https://github.com/GraphSAINT/GraphSAINT/blob/master/graphsaint/utils.py.
    # Download the flickr and yelp datasets from this Google Drive link:
    # https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz
    adj_full = sp.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    train_nodes = np.array(role['tr'])
    valid_nodes = np.array(role['va'])
    test_nodes = np.array(role['te'])
    y_train_np = np.array([class_map[i] for i in train_nodes])
    y_valid_np = np.array([class_map[i] for i in valid_nodes])
    y_test_np = np.array([class_map[i] for i in test_nodes])

    if prefix == 'flickr':
        y_train = np.zeros((y_train_np.size, y_train_np.max()+1))
        y_train[np.arange(y_train_np.size), y_train_np] = 1.
        y_valid = np.zeros((y_valid_np.size, y_valid_np.max()+1))
        y_valid[np.arange(y_valid_np.size), y_valid_np] = 1. 
        y_test = np.zeros((y_test_np.size, y_test_np.max()+1))
        y_test[np.arange(y_test_np.size), y_test_np] = 1.
    else:
        y_train = y_train_np
        y_valid = y_valid_np
        y_test = y_test_np
    return adj_full.tolil(), feats, train_nodes, y_train, valid_nodes, y_valid, test_nodes, y_test

def ContextualSBM(n, d, Lambda, p, mu, sigma=1, split_ratio=[0.6, 0.2, 0.2]):
    # n = 800 #number of nodes
    # d = 5 # average degree
    # Lambda = 1 # parameters
    # p = 1000 # feature dim
    # mu = 1 # mean of Gaussian
    # sigma = 1 # Variance scale difference
    gamma = n/p

    c_in = d + np.sqrt(d)*Lambda
    c_out = d - np.sqrt(d)*Lambda
    y = np.ones(n)
    y[int(n/2)+1:] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i]*y[j] > 0:
                Flip = np.random.binomial(1, c_in/n)
            else:
                Flip = np.random.binomial(1, c_out/n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    # u = np.random.normal(0, 1/np.sqrt(p), [1, p])
    u = (np.random.binomial(1, 0.5, (1, p))*2-1)/np.sqrt(p)
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu/n)*y[i]*u + Z/np.sqrt(p) 
        x[i] = x[i]* sigma**((y[i]+1)/2)

    edges = np.array(edge_index)
    values = np.ones(edges.shape[1])
    adj = sp.coo_matrix((values, (edges[0], edges[1])), shape=(n, n))
    feat = torch.tensor(x, dtype=torch.float32)
    labels_np = np.array((y + 1) // 2, dtype=np.int64)

    # split train/val/test
    all_idx = np.arange(n)
    random.shuffle(all_idx)

    train_ratio, val_ratio, test_ratio = split_ratio

    train_idx = torch.tensor(all_idx[: int(n * train_ratio)])
    val_idx = torch.tensor(all_idx[int(n * train_ratio): int(n * (train_ratio + val_ratio))])
    test_idx = torch.tensor(all_idx[int(n * (train_ratio + val_ratio)):])

    labels = np.zeros((labels_np.size, labels_np.max()+1))
    labels[np.arange(labels_np.size), labels_np] = 1.

    y_train = torch.FloatTensor(labels[train_idx])
    y_valid = torch.FloatTensor(labels[val_idx])
    y_test = torch.FloatTensor(labels[test_idx])

    return adj.tolil(), feat, train_idx, y_train, val_idx, y_valid, test_idx, y_test, labels_np


def load_data(name, flag=None):
    if name in ['ogbn-proteins', 'ogbn-products', 'ogbn-arxiv']:
        return load_OGBdataset(name, flag=flag)
    elif name in ['Cora','Pubmed', 'Reddit', 'CoraFull']:
        return load_DGLdataset(name, flag=flag)
    elif name in ['chameleon', 'cornell', 'squirrel', 'film', 'texas', 'wisconsin']:
        return load_heteDataset(name, flag=flag)
    elif name in ['yelp', 'flickr']:
        return load_SAINTdataset(name)
    elif name in ['cSBM']:
        return ContextualSBM(n=flag.cSBM_n, d=flag.cSBM_d, Lambda=flag.cSBM_lam,
                 p=flag.cSBM_p, mu=flag.cSBM_mu, sigma=flag.cSBM_sigma)


class OurEvaluator(Evaluator):
    def __init__(self,name):
        self.name = name
        self.add_info = {
            'Cora': {'num_tasks': 1, 'eval_metric':'acc'},
            'Pubmed': {'num_tasks':1, 'eval_metric':'acc'},
            'CoraFull': {'num_tasks': 1, 'eval_metric':'acc'},
            'Reddit': {'num_tasks':1, 'eval_metric':'acc'},
            'chameleon': {'num_tasks':1, 'eval_metric':'acc'},
            'cornell': {'num_tasks':1, 'eval_metric':'acc'},
            'squirrel': {'num_tasks':1, 'eval_metric':'acc'},
            'film': {'num_tasks':1, 'eval_metric':'acc'},
            'texas': {'num_tasks':1, 'eval_metric':'acc'},
            'wisconsin': {'num_tasks':1, 'eval_metric':'acc'},
            'flickr': {'num_tasks':1, 'eval_metric':'acc'},
            'yelp': {'num_tasks':100, 'eval_metric':'rocauc'},
            'flickr': {'num_tasks':1, 'eval_metric':'acc'},
            'cSBM': {'num_tasks':1, 'eval_metric':'acc'},
        }

        dir_name = os.path.dirname(ogb.nodeproppred.__file__)
        meta_info = pd.read_csv(os.path.join(dir_name, 'master.csv'), index_col = 0)

        if self.name in meta_info:
            self.num_tasks = int(meta_info[self.name]['num tasks'])
            self.eval_metric = meta_info[self.name]['eval metric']
        elif self.name in self.add_info.keys():
            self.num_tasks = self.add_info[self.name]['num_tasks']
            self.eval_metric = self.add_info[self.name]['eval_metric']   
        else:
            raise ValueError('Invalid dataset name {}.\n'.format(self.name))     
