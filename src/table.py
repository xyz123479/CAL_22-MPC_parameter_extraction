import numpy as np
import networkx as nx
from scipy import stats

from src.const import *

##### Weight entropy
class power2:
    def __init__(self, prec=256):
        self.prec = prec
        
    def __call__(self, data):
        sign_array = np.sign(data)
        powered_array = np.exp2(np.round(np.log2(np.abs(data))))

        powered_array[powered_array < 1 / self.prec] = 0
        powered_array[powered_array > self.prec] = self.prec

        return powered_array * sign_array

class rounding:
    def __init__(self, decimal=-4):
        self.decimal = decimal
        
    def __call__(self, data):
        return np.around(data, decimals=self.decimal)
    
class quantizing:
    def __init__(self, prec=64):
        self.prec = prec
        self.bins = np.arange(0, 256, 1/prec)
        
    def __call__(self, data):
        indices = np.digitize(data, self.bins)
        return self.bins[indices-1]

def compute_entropy_by_weight(data, rounding_fn=None, p_bar=None):
    # change zero value to one
    # zero cannot be denominator, so change it into the closest value which is one
    data[data == 0] = 1
    data = data.astype(float)
    
    entropy_array = {}
    for idx in range(LINESIZE):
        entropy_array[idx] = {}
    
    for target_col_idx in range(LINESIZE):
        # target = weight * base
        weight_data = np.expand_dims(data[:, target_col_idx], -1) / data
        
        if rounding_fn is not None:
            weight_data = rounding_fn(weight_data)
            
        for base_col_idx in range(LINESIZE):
            col_weights = weight_data[:, base_col_idx]
            unique, counts = np.unique(col_weights, return_counts=True)
            col_entropy = stats.entropy(counts, base=2)
            
            entropy_array[target_col_idx][base_col_idx] = {
                'entropy' : col_entropy,
                'unique'  : unique,
                'counts'  : counts,
            }
            if p_bar is not None:
                p_bar.update(1)
            
    return entropy_array

def compute_entropy_by_symbols(data, p_bar=None):
    entropy_array = {}
    entropy_array[0] = {}
        
    for col_idx in range(LINESIZE):
        col_symbols = data[:, col_idx]
        unique, counts = np.unique(col_symbols, return_counts=True)
        col_entropy = stats.entropy(counts, base=2)

        entropy_array[0][col_idx] = {
            'entropy' : col_entropy,
            'unique'  : unique,
            'counts'  : counts,
        }
        if p_bar is not None:
            p_bar.update(1)

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
