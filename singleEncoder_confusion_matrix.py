
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from src import *

import argparse

COMPSIZELIMIT = 280

parser = argparse.ArgumentParser('VPC: Filter Classification Analysis')
parser.add_argument('dataset', type=str, help='Dataset path')
parser.add_argument('label', type=str, help='Cluster label path with respect to the given dataset')
parser.add_argument('filter', type=str, help='Filter path')
parser.add_argument('output', type=str, help='Save directory path of the proximity matrix')
args = parser.parse_args()

def get_confusion_matrix(compressor, dataset, label):
    # key   : data type selection
    # value : compressor selection
    result_cnt = {}
    for num_modules in range(compressor.num_modules):
        result_cnt[num_modules] = { -1: 0 }
        for num_modules_ in range(compressor.num_modules):
            result_cnt[num_modules][num_modules_] = 0

    # count
    p_bar = tqdm(total=len(dataset), ncols=150)
    for idx, dataline in enumerate(dataset):
        result = compressor(dataline)
        compressor_select = result['selected_class']
        datatype_select = label[idx]

        result_cnt[datatype_select][compressor_select] += 1
        p_bar.update(1)
    p_bar.close()

    return result_cnt

def main(args):
    # path
    dataset_path = args.dataset
    label_path = args.label
    filter_path = args.filter
    output_dir_path = args.output

    dataset, label = get_data_and_label(dataset_path, label_path)
    assert(len(dataset) == len(label))

    compressor = SingleEncoderCompressor(filter_path)
    result_cnt = get_confusion_matrix(compressor, dataset, label)

    df = pd.DataFrame(result_cnt)
    print(df)

    df.to_csv(output_dir_path + '/comp_accuracy-prox_matrix.csv')

