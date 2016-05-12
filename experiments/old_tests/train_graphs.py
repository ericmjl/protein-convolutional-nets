"""
In this script, we test the ability of a one convolutional layer network to
learn drug resistance value.

Without doing train/test splitting and cross-validation, this will always be
an overfitting of the data, hence the script name.
"""

from pin import pin
from joblib import Parallel, delayed
from graphfp.layers import FingerprintLayer, LinearRegressionLayer,\
    GraphConvLayer
from graphfp.utils import initialize_network, batch_sample, train_test_split
from graphfp.optimizers import adam
from autograd import grad
from graphfp.flatten import flatten
from time import time

import json
import os
import pandas as pd
import numpy as np
import random as rnd
import sys
import pickle as pkl


def make_protein_graphs(project, seqid):
    """
    Custom function for this script to parallelize the making of protein
    graphs over individual cores.
    """
    p = pin.ProteinInteractionNetwork('../data/batch_models/{0}/model_01.pdb'
                                      .format(project))
    p.graph['project'] = project
    p.graph['input_shape'] = p.nodes(data=True)[0][1]['features'].shape
    p.graph['seqid'] = seqid
    return p


def train_loss(wb_vect, unflattener, cv=False, batch=False, batch_size=4,
               debug=False):
    """
    Training loss is MSE.

    We pass in a flattened parameter vector and its unflattener.
    """
    wb_struct = unflattener(wb_vect)

    if batch and not cv:
        batch_size = batch_size
    else:
        batch_size = len(graphs)

    if cv and not batch:
        samp_graphs, samp_inputs = batch_sample(test_graphs, input_shape,
                                                batch_size=len(test_graphs))
    else:
        samp_graphs, samp_inputs = batch_sample(graphs, input_shape,
                                                batch_size)

    preds = predict(wb_struct, samp_inputs, samp_graphs)
    graph_ids = [g.graph['seqid'] for g in samp_graphs]
    graph_scores = fpv_data.set_index('seqid').ix[graph_ids]['FPV'].values.\
        reshape(preds.shape)
    assert preds.shape == graph_scores.shape

    mse = np.mean(np.power(preds - graph_scores, 2))

    if debug:
        print(graph_ids)
        print('Predictions:')
        print(preds)
        print('Mean: {0}'.format(np.mean(preds)))
        print('')
        print('Actual')
        print(graph_scores)
        print('Mean: {0}'.format(np.mean(graph_scores)))
        print('')
        print('Difference')
        print(preds - graph_scores)
        print('Mean Squared Error: {0}'.format(mse))
        print('')

    return mse


def predict(wb_struct, inputs, graphs):
    """
    Makes predictions by running the forward pass over all of the layers.

    Parameters:
    ===========
    - wb_struct: a dictionary of weights and biases stored for each layer.
    - inputs: the input data matrix. should be one row per graph.
    - graphs: a list of all graphs.
    """
    curr_inputs = inputs

    for i, layer in enumerate(layers):
        wb = wb_struct['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb, curr_inputs, graphs)
    return curr_inputs


def callback(wb, i):
    """
    Any function you want to run at each iteration of the optimization.
    """
    wb_vect, wb_unflattener = flatten(wb)
    print('Iteration: {0}'.format(i))
    print('Training Loss: ')

    # Record training set train_loss
    tl = train_loss(wb_vect, wb_unflattener, batch=False)
    print(tl)
    trainloss.append(tl)

    # Record the preds vs. actual for the training set.
    batch_size = len(graphs)
    samp_graphs, samp_inputs = batch_sample(graphs, input_shape,
                                            batch_size)

    preds = predict(wb, samp_inputs, samp_graphs)
    graph_ids = [g.graph['seqid'] for g in samp_graphs]
    graph_scores = fpv_data.set_index('seqid').ix[graph_ids]['FPV'].\
        values.reshape(preds.shape)

    preds_vs_actual[i] = dict()
    preds_vs_actual[i]['preds'] = preds
    preds_vs_actual[i]['actual'] = graph_scores

    # Record cross-validation train_loss
    cv_tl = train_loss(wb_vect, wb_unflattener, cv=True)
    trainloss_cv.append(cv_tl)

    # Record the preds vs. actual for the test set
    batch_size = len(test_graphs)
    samp_graphs, samp_inputs = batch_sample(test_graphs, input_shape,
                                            batch_size)
    preds_cv = predict(wb, samp_inputs, samp_graphs)
    graph_ids = [g.graph['seqid'] for g in samp_graphs]
    graph_scores_cv = fpv_data.set_index('seqid').ix[graph_ids]['FPV'].\
        values.reshape(preds_cv.shape)

    preds_vs_actual_cv[i] = dict()
    preds_vs_actual_cv[i]['preds'] = preds_cv
    preds_vs_actual_cv[i]['actual'] = graph_scores_cv

    print('cross-validated training loss: {0}'.format(cv_tl))

    # Report on the expected time left.
    time_elapsed = time() - start
    print('Total time: {0} min {1} sec'.format(int(time_elapsed / 60),
          time_elapsed % 60))
    time_left = (num_iters - i + 1) * (time_elapsed / (i + 1))
    print('Expected time left: {0} min {1} sec'.format(int(time_left / 60),
          time_left % 60))

    print('')

if __name__ == '__main__':
    n_graphs = int(sys.argv[1])
    num_iters = int(sys.argv[2])

    # Open the batch_summary JSON file.
    with open('../data/batch_summary.json') as f:
        model_data = json.load(f)

    # Open the protease data
    protease_data = pd.read_csv(
        '../data/hiv_data/hiv-protease-data-expanded.csv', index_col=0)

    # Drop NaN values on only the NFV column, and then log10 transform.
    fpv_data = protease_data.dropna(subset=['FPV'])[['FPV', 'seqid']]
    fpv_data['FPV'] = fpv_data['FPV'].apply(np.log10)

    # Grab out a sample of projects, based on the seqid.
    proj_titles = {c['title']: c['code'] for c in model_data['projects']}
    projs_interest = dict()
    while len(projs_interest) < n_graphs:
        seqid = rnd.choice(list(fpv_data['seqid']))
        if seqid in proj_titles.keys():
            projs_interest[seqid] = proj_titles[seqid]

    # Make graphs in memory.
    print('Making graphs...')
    gr_start = time()
    all_graphs = Parallel(n_jobs=-1)(delayed(make_protein_graphs)(proj, seqid)
                                     for seqid, proj in projs_interest.items())
    gr_end = time()
    print('Graphs took {0} seconds to finish.'.format(gr_end - gr_start))
    start = time()  # start recording time.

    graphs, test_graphs = train_test_split(all_graphs, test_fraction=0.2)

    # Define the gradient function
    grad_tl = grad(train_loss)

    # Specify neural network shape and meta-parameters.
    input_shape = graphs[0].graph['input_shape']
    print('input shape: {0}'.format(input_shape))
    layers = [GraphConvLayer((input_shape[1], input_shape[1])),
              FingerprintLayer(input_shape[1]),
              LinearRegressionLayer((input_shape[1], 1)),
              ]

    # Initialize the neural network's weights and biases, as well as an empty
    # containers for holding the training losses.
    wb_all = initialize_network(input_shape, graphs, layers)
    trainloss = list()
    preds_vs_actual = dict()
    trainloss_cv = list()
    preds_vs_actual_cv = dict()

    # Train the neural network on the data.
    wb_vect, wb_unflattener = adam(grad_tl, wb_all, callback=callback,
                                   num_iters=num_iters)
    wb_all = wb_unflattener(wb_vect)

    # Write training losses and weights/biases to disk.
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    handle = '{0}-graphs_{1}-iters'.format(n_graphs, num_iters)

    with open('outputs/{0}_trainloss.pkl'.format(handle), 'wb') as f:
        pkl.dump(trainloss, f)

    with open('outputs/{0}_wbs.pkl'.format(handle), 'wb') as f:
        pkl.dump(wb_all, f)

    with open('outputs/{0}_predsactual.pkl'.format(handle), 'wb') as f:
        pkl.dump(preds_vs_actual, f)

    with open('outputs/{0}_trainloss_cv.pkl'.format(handle), 'wb') as f:
        pkl.dump(trainloss_cv, f)

    with open('outputs/{0}_predsactual_cv.pkl'.format(handle), 'wb') as f:
        pkl.dump(preds_vs_actual_cv, f)
