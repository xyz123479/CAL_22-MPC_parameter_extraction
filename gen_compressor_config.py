import os

import numpy as np
from tqdm.auto import tqdm
import pickle

import argparse
from datetime import datetime

from src import *

BATCH_SIZE=65536
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser('VPC: General Pattern Search. Config generator')
parser.add_argument('dataset', type=str, help='Dataset path')
parser.add_argument('label', type=str, help='Cluster label path with respect to the given dataset')
parser.add_argument('output', type=str, help='Save directory path of filters json file')
parser.add_argument('-f', '--filter-path', type=str, required=False, help='Compression filter path. If it is given, weight/symbol entropy computation will be skipped.')
parser.add_argument('-s', '--scan-path', type=str, required=False, help='Scan path. If it is given, scan path searching will be skipped.')
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(e)
    print('Invalid Usage:')
    print(parser.print_help())

## functions
def load_data(dataset_path, label_path):
    # dataset
    data, labels = get_data_and_label(dataset_path, label_path)
    data_classes = sort_lines_by_class(data, labels)

    # check linesize before start
    for num_class in data_classes:
        if (len(data_classes[num_class]) != 0):
            linesize = data_classes[num_class].shape[1]
            assert (linesize == LINESIZE)

    # if class is empty, fill a dummy dataline
    for num_class in data_classes:
        if (len(data_classes[num_class]) == 0):
            data_classes[num_class] = np.zeros(shape=(1, linesize), dtype=np.uint8)

    # converts ndarray to tensor
    for num_class in data_classes:
        data_classes[num_class] = torch.from_numpy(data_classes[num_class])
    return data_classes

def compute_weight_entropy(data_classes):
    entropy_arrays = {}
    for selected_cluster in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_weight(selected_cluster_data, rounding_fn=power2(),
                batch_size=BATCH_SIZE, device=DEVICE)
        entropy_arrays[selected_cluster] = entropy_array
    return entropy_arrays

def compute_symbol_entropy(data_classes):
    entropy_arrays = {}
    for selected_cluster in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_symbols(selected_cluster_data,
                batch_size=BATCH_SIZE, device=DEVICE)
        entropy_arrays[selected_cluster] = entropy_array
    return entropy_arrays

def make_filters(weight_entropy_arrays, symbol_entropy_arrays):
    compression_filters = {}
    for selected_cluster in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        weight_entropy_array = weight_entropy_arrays[selected_cluster]
        symbol_entropy_array = symbol_entropy_arrays[selected_cluster]
        
        compression_filter = make_table(weight_entropy_array, symbol_entropy_array)
        compression_filters[selected_cluster] = {
            'root_idx' : compression_filter['root_idx'],
            'base_idx_table' : compression_filter['base_idx_table'],
            'weight_table' : compression_filter['weight_table'],
        }
    return compression_filters

def generate_residue(data_classes, compression_filters):
    # make residue
    residue_arrays = {}
    for selected_cluster in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        residue_array = compute_residue(
            data_classes[selected_cluster], compression_filters[selected_cluster],
            batch_size=BATCH_SIZE, device=DEVICE)
        residue_arrays[selected_cluster] = residue_array
    return residue_arrays

def generate_scan_path(residue_arrays):
    # find scan route
    scan_index_array = {}
    for selected_cluster in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        residue = residue_arrays[selected_cluster]
        DBP = BPX(residue, consecutive_xor=True)
        scan_index = phi_scan(DBP, batch_size=BATCH_SIZE, device=DEVICE)
        scan_index_array[selected_cluster] = scan_index
    return scan_index_array

def main(args):
    today = datetime.today().strftime("%y%m%d")

    # path
    dataset_path = args.dataset
    label_path = args.label
    output_dir_path = args.output
    compression_filter_path = os.path.join(output_dir_path,
            "%s_compression_filters.pickle" %(today))
    scan_result_path = os.path.join(output_dir_path,
            "%s_scan_path.pickle" %(today))

    # load data
    print("Loading dataset...")
    data_classes = load_data(dataset_path, label_path)
    
    if args.filter_path is None:
        # compute entropy
        weight_entropy_arrays = compute_weight_entropy(data_classes)
        symbol_entropy_arrays = compute_symbol_entropy(data_classes)
        # generate filters with MST algorithm
        compression_filters = make_filters(weight_entropy_arrays, symbol_entropy_arrays)
        # save
        with open (compression_filter_path, "wb") as f:
            pickle.dump(compression_filters, f)
            print("Filters are saved")
        print('Filters are generated')
    else:
        with open (args.filter_path, "rb") as f:
            compression_filters = pickle.load(f)
            print("Filters are loaded")

    if args.scan_path is None:
        # residue
        residue_arrays = generate_residue(data_classes, compression_filters)

        # zero ordering path search
        scan_index_array = generate_scan_path(residue_arrays)

        # save
        with open (scan_result_path, "wb") as f:
            pickle.dump(scan_index_array, f)
            print("Scan path are saved")
        print('Scan path generated')
    else:
        with open (args.scan_path, "rb") as f:
            scan_index_array = pickle.load(f)
            print("Scan path are loaded")

    # config generation
    compressor_config = make_config(compression_filters, scan_index_array)
    json_output_path = os.path.join(output_dir_path,
            "%s_compressor_config.json" %(today))
    with open(json_output_path, 'w') as json_f:
        json.dump(compressor_config, json_f, cls=NpEncoder)
    print('Config is converted to json format')
    
if __name__ == '__main__':
    main(args)
    

