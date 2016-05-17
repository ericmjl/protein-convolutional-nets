"""
Author: Eric J. Ma

Date: 17 May 2016

Purpose:
This script reads in all of the `model_01.pdb` files under each project,
creates the graph representation of that graph in memory, and appends all of
the nodes' features to a list that is eventually converted into a large numpy
array. This array is accompanied by a graph_idxs dictionary and
nodes_nbrs_idxs dictionary.

- graph_idxs: the graph seqids are the keys, and the graph nodes form a list
              of values.
- nodes_nbrs_idxs: each node's idx is a key, and itself and its node's
                   neighbors' idxs form a list of values.
"""

import json
import os.path as op
import sys
import autograd.numpy as np
import pickle as pkl

from collections import defaultdict
from pin import pin
from tqdm import tqdm

# Open the batch_models.json file.
with open('../data/batch_summary.json', 'r') as f:
    batch_summary = json.load(f)

models_path = '../data/batch_models'

# Open each PDB file.
feat_list = list()

idx = 0
graph_idxs = defaultdict(list)
nodes_nbrs = defaultdict(list)
for project in tqdm(batch_summary['projects']):
    try:
        # print(project['code'], idx)
        # print(sys.getsizeof(feat_list))
        proj_dir = op.join(models_path, project['code'])
        pdb_path = op.join(proj_dir, 'model_01.pdb')

        p = pin.ProteinInteractionNetwork(pdb_path)

        # Annotate each node with an idx number.
        for n, d in p.nodes(data=True):
            p.node[n]['idx'] = idx
            feat_list.append(d['features'])
            idx += 1

        # Now loop over each node again and figure out its neighbors.
        for n, d in p.nodes(data=True):
            graph_idxs[project['code']].append(d['idx'])
            nodes_nbrs[d['idx']].append(d['idx'])
            for nbr in p.neighbors(n):
                nodes_nbrs[d['idx']].append(p.node[nbr]['idx'])
            print(nodes_nbrs[d['idx']])
    except:
        print('Did not make graph for {0}'.format(project['code']))

# Save the array along with the graph_idxs and nodes_nbrs dict to disk.
feat_array = np.vstack(feat_list)
np.save('../data/feat_array.npy', feat_array)

with open('../data/nodes_nbrs.pkl', 'wb') as f:
    pkl.dump(nodes_nbrs, f)

with open('../data/graph_idxs.pkl', 'wb') as f:
    pkl.dump(graph_idxs, f)
