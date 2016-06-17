from autograd import grad
from collections import defaultdict
from graphfp.layers import FingerprintLayer, LinearRegressionLayer,\
    GraphConvLayer
from graphfp.utils import initialize_network
from pyflatten import flatten
from graphfp.binary_matrix_utils import to_sparse_format
from scipy.sparse import csr_matrix
from graphfp.optimizers import adam
from time import time
from memory_profiler import profile

import autograd.numpy as np
import pickle as pkl
import pandas as pd
import gc


def open_csv_file():
    """
    Open the data file that contains the HIV protease data, and select just
    the FPV column.
    """
    df = pd.read_csv('../data/hiv_data/hiv-protease-data-expanded.csv',
                     index_col=0)
    df = df.dropna(subset=['FPV'])
    return df


def load_feat_array():
    """Open the numpy array of all graphs' data."""
    return np.load('../data/feat_array.npy')


def unpickle_data(path):
    """
    Open the pickles that contain the graph information and node-nbr
    information.
    """
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def preprocess_data(df, nodes_nbrs, graph_idxs, graph_nodes, graph_array):
    intersect = set(df['seqid'].values).intersection(graph_idxs.keys())
    # Get a reduced list of graph_idxs.
    graph_idxs_reduced = dict()
    graph_nodes_reduced = dict()
    for g in intersect:
        graph_idxs_reduced[g] = graph_idxs[g]
        graph_nodes_reduced[g] = graph_nodes[g]
    # return intersect, graph_idxs_reduced, graph_nodes_reduced

    # Initialize a zero-matrix.
    idxs = np.concatenate([i for i in graph_idxs_reduced.values()])
    graph_arr_fin = np.zeros(shape=graph_array[idxs].shape)

    # Initialize empty maps of graph indices from the old to the new.
    nodes_oldnew = dict()  # {old_idx: new_idx}.
    nodes_newold = dict()  # {new_idx: old_idx}

    # Re-assign reduced graphs to the zero-matrix.
    curr_idx = 0
    for seqid, idxs in sorted(graph_idxs_reduced.items()):
        for idx in idxs:
            nodes_oldnew[idx] = curr_idx
            nodes_newold[curr_idx] = idx
            graph_arr_fin[curr_idx] = graph_array[idx]
            curr_idx += 1

    nodes_nbrs_fin = filter_and_reindex_nodes_nbrs(nodes_nbrs, nodes_oldnew)
    graph_idxs_fin = filter_and_reindex_graph_idxs(graph_idxs, nodes_oldnew)
    graph_nodes_fin = filter_and_reindex_graph_nodes(graph_nodes, nodes_oldnew)

    return graph_arr_fin, nodes_nbrs_fin, graph_idxs_fin, graph_nodes_fin


def filter_and_reindex_nodes_nbrs(nodes_nbrs, nodes_oldnew):
    """
    - nodes_nbrs:   a dictionary of nodes and their neighbors.
    - nodes_oldnew: a dictionary mapping old node indices to their new node
                    indices.
    """
    nodes_nbrs_fin = defaultdict(list)

    for node, nbrs in sorted(nodes_nbrs.items()):
        if node in nodes_oldnew.keys():
            for nbr in nbrs:
                nodes_nbrs_fin[nodes_oldnew[node]].append(nodes_oldnew[nbr])
    return nodes_nbrs_fin


def filter_and_reindex_graph_idxs(graph_idxs, nodes_oldnew):
    """
    - graph_idxs: a dictionary of graphs and their original indices.
    - nodes_oldnew: a dictionary mapping old node indices to their new node
                    indices.
    """
    graph_idxs_fin = defaultdict(list)
    for seqid, nodes in sorted(graph_idxs.items()):
        for node in nodes:
            if node in nodes_oldnew.keys():
                graph_idxs_fin[seqid].append(nodes_oldnew[node])
    return graph_idxs_fin


def filter_and_reindex_graph_nodes(graph_nodes, nodes_oldnew):
    """
    - graph_nodes: a dictionary mapping graphs to their dictionary mapping
                   indices to node names.
    - nodes_oldnew: a dictionary mapping old node indices to their new node
                    indices.
    """
    graph_nodes_fin = defaultdict(dict)
    for seqid, idx_node in sorted(graph_nodes.items()):
        for old_idx, node_name in idx_node.items():
            if old_idx in nodes_oldnew.keys():
                graph_nodes_fin[seqid][nodes_oldnew[old_idx]] = node_name
    return graph_nodes_fin


def predict(wb_struct, inputs, nodes_nbrs_compressed, graph_idxs, layers):
    curr_inputs = inputs

    for i, layer in enumerate(layers):
        wb = wb_struct['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb, curr_inputs,
                                         nodes_nbrs_compressed, graph_idxs)
    return curr_inputs


def get_actual(graph_idxs, df, preds):
    """
    Returns the actual data for those protein sequences.
    """
    sorted_graphs = sorted(graph_idxs.keys())
    # print(sorted_graphs)
    sorted_resistances = df[df['seqid'].isin(sorted_graphs)]\
        .set_index('seqid')\
        .ix[sorted_graphs]['FPV']\
        .values
    actual = sorted_resistances.reshape(preds.shape)
    return actual


@profile
def main():
    print('Opening CSV file...')
    df = open_csv_file()
    print('Loading feature array...')
    graph_array = load_feat_array()
    print('Opening graph_idxs...')
    graph_idxs = unpickle_data('../data/graph_idxs.pkl')
    print('Opening graph_nodes...')
    graph_nodes = unpickle_data('../data/graph_nodes.pkl')
    print('Opening nodes_nbrs...')
    nodes_nbrs = unpickle_data('../data/nodes_nbrs.pkl')

    # Check data
    print('Doing data checks...')
    assert df.shape == (6660, 13)
    assert len(graph_array) == 659895
    assert len(graph_idxs) == len(graph_nodes)
    assert len(nodes_nbrs) == len(graph_array)

    print('Preprocessing data...')
    pp_data = preprocess_data(df, nodes_nbrs, graph_idxs, graph_nodes,
                              graph_array)
    graph_arr, nodes_nbrs, graph_idxs, graph_nodes = pp_data

    assert graph_arr.shape[0] == len(nodes_nbrs)
    assert len(graph_idxs) == len(graph_nodes)

    print('Setting up neural net.')
    layers = [GraphConvLayer(weights_shape=(36, 36),
                             biases_shape=(1, 36)),
              FingerprintLayer(weights_shape=(36, 36),
                               biases_shape=(1, 36)),
              LinearRegressionLayer(weights_shape=(36, 1),
                                    biases_shape=(1, 1)),
              ]
    print(layers)

    print('Initializing network...')
    wb = initialize_network(layers_spec=layers)
    wb_vect, unflattener = flatten(wb)
    print('Network initialized. Weights & biases:')
    print(wb)

    node_rows, node_cols, ones = to_sparse_format(nodes_nbrs)

    nodes_nbrs_compressed = csr_matrix((ones, (node_rows, node_cols)),
                                       shape=(len(nodes_nbrs),
                                              len(nodes_nbrs)))

    train_losses = []
    preds_iter = []
    actual_iter = []

    print('Defining train loss function.')

    def train_loss(wb_vect, unflattener, i):
        wb_struct = unflattener(wb_vect)
        preds = predict(wb_struct, graph_arr, nodes_nbrs_compressed,
                        graph_idxs, layers)
        graph_scores = get_actual(graph_idxs, df, preds)
        mse = np.mean(np.power(preds - graph_scores, 2))

        train_losses.append(mse)
        preds_iter.append(preds)
        actual_iter.append(graph_scores)
        gc.collect()
        return mse

    traingrad = grad(train_loss)

    training_losses = []

    print('Defining callback function...')

    def callback(wb, i):
        start = time()
        tl = train_loss(*flatten(wb))
        if i % 1 == 0:
            print(tl, time() - start)
        training_losses.append(tl)
        gc.collect()

    print('Training neural network.')
    wb_vect, unflattener = adam(traingrad, wb, callback=callback, num_iters=20)

    print(wb_vect)

if __name__ == '__main__':
    main()
