import os

import numpy as np
from tqdm.auto import tqdm

import argparse

from src import *

parser = argparse.ArgumentParser('MBWD: General Pattern Search')
parser.add_argument('dataset', type=str, help='Dataset path')
parser.add_argument('label', type=str, help='Cluster label path with respect to the given dataset')
parser.add_argument('output', type=str, help='Save directory path of filters json file')
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(e)
    print('Invalid Usage:')
    print(parser.print_help())

## functions
def compute_weight_entropy(data_classes):
    entropy_arrays = {}
    p_bar = tqdm(total = NUM_TYPES * LINESIZE * LINESIZE, desc='Computing weight entropy', ncols=150)
    for selected_cluster in range(1, NUM_CLUSTERS - 1):
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_weight(selected_cluster_data, rounding_fn=power2(), p_bar=p_bar)
        entropy_arrays[selected_cluster] = entropy_array
    p_bar.close()
    return entropy_arrays

def compute_symbol_entropy(data_classes):
    entropy_arrays = {}
    p_bar = tqdm(total = NUM_TYPES * LINESIZE, desc='Computing symbol entropy', ncols=150)
    for selected_cluster in range(1, NUM_CLUSTERS - 1):
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_symbols(selected_cluster_data, p_bar)
        entropy_arrays[selected_cluster] = entropy_array
    p_bar.close()
    return entropy_arrays

def make_filters(weight_entropy_arrays, symbol_entropy_arrays):
    compression_filters = {}
    for selected_cluster in range(1, NUM_CLUSTERS - 1):
        weight_entropy_array = weight_entropy_arrays[selected_cluster]
        symbol_entropy_array = symbol_entropy_arrays[selected_cluster]
        
        compression_filter = make_table(weight_entropy_array, symbol_entropy_array)
        compression_filters[selected_cluster] = {
            'root_idx' : compression_filter['root_idx'],
            'base_idx_table' : compression_filter['base_idx_table'],
            'weight_table' : compression_filter['weight_table'],
        }
    return compression_filters

def generate_scan_path(data_classes, compression_filters):
    # make residue
    residue_array = {}
    for selected_cluster in range(1, NUM_CLUSTERS - 1):
        residue_array[selected_cluster] = compute_residue(
            data_classes[selected_cluster], compression_filters[selected_cluster],
            selected_cluster).astype(np.uint8)
    # find scan route
    scan_index_array = {}
    for selected_cluster in range(1, NUM_CLUSTERS - 1):
        residue = residue_array[selected_cluster]
        DBP = BPX(residue, consecutive_xor=True)
        scan_index = phi_scan(DBP, selected_cluster)
        scan_index_array[selected_cluster] = scan_index
    return scan_index_array

def load_data(dataset_path, label_path):
    # dataset
    data, labels = get_data_and_label(dataset_path, label_path)
    data_classes = sort_lines_by_class(data, labels)
    return data_classes

def main(args):
    # path
    dataset_path = args.dataset
    label_path = args.label
    output_dir_path = args.output

    # load data
    data_classes = load_data(dataset_path, label_path)
    
    # compute entropy
    weight_entropy_arrays = compute_weight_entropy(data_classes)
    symbol_entropy_arrays = compute_symbol_entropy(data_classes)
    # generate filters with MST algorithm
    compression_filters = make_filters(weight_entropy_arrays, symbol_entropy_arrays)
    print('Filters are generated')

    # zero scan path generation
    scan_path = generate_scan_path(data_classes, compression_filters)
    print('Scan path generated')

    # config generation
    compressor_config = make_config(compression_filters, scan_path)
    with open(output_dir_path + '/compressor_config.json', 'w') as json_f:
        json.dump(compressor_config, json_f, cls=NpEncoder)
    print('Config is converted to json format')
    
if __name__ == '__main__':
    main(args)
    






