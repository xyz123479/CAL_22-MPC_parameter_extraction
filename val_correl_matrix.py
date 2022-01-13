import os

import torch
import numpy as np

from tqdm import trange
from tqdm.auto import tqdm
import pickle

import argparse
from datetime import datetime

from src import *

BATCH_SIZE=2**18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_DTYPE=7

parser = argparse.ArgumentParser('VPC: Dtype Value Correlation Matrix.')
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
def load_data(dataset_path, label_path):
    data = np.load(dataset_path)
    data = torch.from_numpy(data)

    assert data.size(1) == LINESIZE

    label = np.load(label_path)
    label = torch.from_numpy(label)

    data_classes = {}
    for selected_class in range(NUM_DTYPE):
        data_classes[selected_class] = data[label == selected_class]
    return data_classes

def compute_weight_entropy(data_classes):
    entropy_arrays = {}
    outer_loop_desc = "Computing Weight Entropy"
    for selected_cluster in trange(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1,
            ncols=TQDM_COLS, position=0, desc=outer_loop_desc.rjust(TQDM_DESC_LEN)):
        inner_loop_desc = "%2d / %2d" %(selected_cluster+1, NUM_CLUSTERS-1)
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_weight(selected_cluster_data, rounding_fn=power2_MINVAL(),
                batch_size=BATCH_SIZE, device=DEVICE,
                desc=inner_loop_desc.rjust(TQDM_DESC_LEN))
        entropy_arrays[selected_cluster] = entropy_array
    return entropy_arrays

def compute_difference_entropy(data_classes):
    entropy_arrays = {}
    outer_loop_desc = "Computing Difference Entropy"
    for selected_cluster in trange(NUM_DTYPE,
            ncols=TQDM_COLS, position=0, desc=outer_loop_desc.rjust(TQDM_DESC_LEN)):
        inner_loop_desc = "%2d / %2d" %(selected_cluster+1, NUM_DTYPE)
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_difference(selected_cluster_data,
                batch_size=BATCH_SIZE, device=DEVICE,
                desc=inner_loop_desc.rjust(TQDM_DESC_LEN))
        entropy_arrays[selected_cluster] = entropy_array
    return entropy_arrays

def compute_ratio_entropy(data_classes):
    entropy_arrays = {}
    outer_loop_desc = "Computing Ratio Entropy"
    for selected_cluster in trange(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1,
            ncols=TQDM_COLS, position=0, desc=outer_loop_desc.rjust(TQDM_DESC_LEN)):
        inner_loop_desc = "%2d / %2d" %(selected_cluster+1, NUM_CLUSTERS-1)
        selected_cluster_data = data_classes[selected_cluster]
        entropy_array = compute_entropy_by_ratio(selected_cluster_data,
                batch_size=BATCH_SIZE, device=DEVICE,
                desc=inner_loop_desc.rjust(TQDM_DESC_LEN))
        entropy_arrays[selected_cluster] = entropy_array
    return entropy_arrays

def main(args):
    today = datetime.today().strftime("%y%m%d")

    # path
    dataset_path = args.dataset
    label_path = args.label
    output_dir_path = args.output
    diff_entropy_arr_path = os.path.join(output_dir_path,
            "%s_diff_entropys.pickle" %(today))

    # load data
    print("Loading dataset...")
    data_classes = load_data(dataset_path, label_path)
    
    # compute entropy
    print("Computing entropy...")
#     diff_entropy_arrays = compute_difference_entropy(data_classes)
    val_entropy_arrays = compute_weight_entropy(data_classes)

    # save
    with open(diff_entropy_arr_path, "wb") as f:
        pickle.dump(val_entropy_arrays, f)



    
if __name__ == '__main__':
    main(args)
    


