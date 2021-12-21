from operator import itemgetter

import torch
import numpy as np
import networkx as nx
from scipy import stats

from src.const import *
from src.utils import *

from tqdm.auto import tqdm

##### unique_counts
class UniqueCount(object):
    def __init__(self, dtype, is_weight, rounding_fn=None, device="cpu"):
        self.dtype_range = DTYPE_RANGE
        self.rounding_fn = rounding_fn

        # a dict of all unique value and its index
        self.all_uniques = {}
        if is_weight:
            for i in range(self.dtype_range):
                for j in range(self.dtype_range):
                    unique = i / (j + MIN_VAL)
                    if self.rounding_fn is not None:
                        unique = self.rounding_fn.computeScalar(unique)
                    self.all_uniques[unique] = 0
        else:
            for unique in range(self.dtype_range):
                self.all_uniques[unique] = 0

        # key is unique value, value is index of unique_counts
        for i, key in enumerate(self.all_uniques):
            self.all_uniques[key] = i

        self.unique_counts = torch.zeros((LINESIZE, LINESIZE, len(self.all_uniques)),
                dtype=int, device=device)

    def update(self, target_idx, base_idx, data):
        unique, counts = torch.unique(data, return_counts=True)
        unique = unique.tolist()
        unique_indices = itemgetter(*unique)(self.all_uniques)
        self.unique_counts[target_idx, base_idx, unique_indices] = self.unique_counts[target_idx, base_idx, unique_indices] + counts

    def get(self, target_idx, base_idx):
        unique_counts = self.unique_counts[target_idx, base_idx, :].cpu().numpy()

        uniques_to_return = []
        counts_to_return = []
        for idx, count in enumerate(unique_counts):
            if (count != 0):
                counts_to_return.append(count)
                for unique, i in self.all_uniques.items():
                    if idx == i:
                        uniques_to_return.append(unique)
        return uniques_to_return, counts_to_return

def compute_entropy_by_weight(data, rounding_fn=None, batch_size=65536, device="cpu"):
    # init unique_counts
    # target_col_idx, base_col_idx, unique_idx
    unique_count = UniqueCount(data.dtype, True, rounding_fn, device)

    # init entropy_array
    entropy_array = {}
    for idx in range(LINESIZE):
        entropy_array[idx] = {}
    
    p_bar = tqdm(total = len(data), desc="Computing weight entropy", ncols=150)
    for minibatch in iter_batch(data, batch_size):
        minibatch = minibatch.float().to(device)

#         # change zero value to one
#         # zero cannot be denominator, so change it into the closest value which is one
#         minibatch[minibatch == 0] = 1
        for target_col_idx in range(LINESIZE):
            # target = weight * base
            weight_data = torch.unsqueeze(minibatch[:, target_col_idx], -1) / (minibatch + MIN_VAL)
            
            if rounding_fn is not None:
                weight_data = rounding_fn(weight_data)
                
            for base_col_idx in range(LINESIZE):
                col_weights = weight_data[:, base_col_idx]
                unique_count.update(target_col_idx, base_col_idx, col_weights)
        p_bar.update(len(minibatch)) 
    p_bar.close()

    for target_col_idx in range(LINESIZE):
        for base_col_idx in range(LINESIZE):
            unique, counts = unique_count.get(target_col_idx, base_col_idx)
            col_entropy = stats.entropy(counts, base=2)
            entropy_array[target_col_idx][base_col_idx] = {
                'entropy' : col_entropy,
                'unique'  : unique,
                'counts'  : counts,
            }
            
    return entropy_array

def compute_entropy_by_symbols(data, batch_size=65536, device="cpu"):
    # init unique_counts
    # target_col_idx, base_col_idx, unique_idx
    unique_count = UniqueCount(data.dtype, False, None, device)

    entropy_array = {}
    for idx in range(LINESIZE):
        entropy_array[idx] = {}
        
    p_bar = tqdm(total = len(data), desc="Computing symbol entropy", ncols=150)
    for minibatch in iter_batch(data, batch_size):
        minibatch = minibatch.to(device)
        for col_idx in range(LINESIZE):
            col_symbols = minibatch[:, col_idx]
            unique_count.update(0, col_idx, col_symbols)
        p_bar.update(len(minibatch))
    p_bar.close()

    for col_idx in range(LINESIZE):
        unique, counts = unique_count.get(0, col_idx)
        col_entropy = stats.entropy(counts, base=2)
        entropy_array[0][col_idx] = {
            'entropy' : col_entropy,
            'unique'  : unique,
            'counts'  : counts,
        }
    return entropy_array

## Converts to fully connected indirected graph
#  and execute MST algorithm
def find_tree_height(tree, root_idx):
    if not nx.is_tree(tree):
        print("The given graph is not a tree")
        return -1

    max_level = 0
    for node_idx in tree.nodes():
        level = nx.shortest_path_length(tree, source=root_idx, target=node_idx)
        if level > max_level:
            max_level = level

    return max_level

def make_table(weight_entropy_array, symbol_entropy_array, top=LINESIZE, order='decreasing'):
    # assign weight entropy array and high_prob_weight array
    weight_entropy = np.zeros((LINESIZE, LINESIZE))
    high_prob_weight = np.zeros((LINESIZE, LINESIZE))        
    for target_idx in range(LINESIZE):
        for base_idx in range(LINESIZE):
            weight_entropy[target_idx, base_idx] = weight_entropy_array[target_idx][base_idx]['entropy']

            unique = weight_entropy_array[target_idx][base_idx]['unique']
            counts = weight_entropy_array[target_idx][base_idx]['counts']
            high_prob_weight[target_idx, base_idx] = unique[counts.argmax()]
    
    # assign symbol entropy array
    symbol_entropy = np.zeros(LINESIZE)
    for symbol_idx in range(LINESIZE):
        symbol_entropy[symbol_idx] = symbol_entropy_array[0][symbol_idx]['entropy']
        
    # weight_entropy array to graph
    G = nx.Graph()
    for target_idx in range(LINESIZE):
        for base_idx in range(LINESIZE):
            G.add_edge(target_idx, base_idx,
                       entropy=weight_entropy[target_idx, base_idx])

    # graph to Minimum Spanning Tree
    MST = nx.minimum_spanning_tree(G, weight='entropy')
    
    # converts to base_idx table, weight table
    base_idx_table = np.zeros(LINESIZE, dtype=int)
    weight_table = np.zeros(LINESIZE)    
    
    ## assign root index candidates
    if order == 'increasing':
        root_candidates = symbol_entropy.argsort()
    elif order == 'decreasing':
        root_candidates = symbol_entropy.argsort()[::-1]
    else:
        assert False, 'order is not valid'
    root_candidates = root_candidates[:top]

    ## find optimal root index
    min_height = 99999
    root_idx = 0
    for root_cand in root_candidates:
        height = find_tree_height(MST, root_cand)
        if height < min_height:
            min_height = height
            root_idx = root_cand

    ## assign element to table
    base_idx_table[root_idx] = -1
    edges = nx.dfs_edges(MST, source=root_idx)
    for e in edges:
        base_idx, target_idx = e
        base_idx_table[target_idx] = base_idx
        weight_table[target_idx] = high_prob_weight[target_idx, base_idx]

    return {
        'graph'          : G,
        'mst'            : MST,
        'root_idx'       : root_idx,
        'min_height'     : min_height,
        'base_idx_table' : base_idx_table,
        'weight_table'   : weight_table
    }
