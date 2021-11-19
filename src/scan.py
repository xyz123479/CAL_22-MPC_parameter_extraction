import numpy as np

from tqdm.auto import tqdm
from src.const import *

SCANNED_SYMBOL_SIZE = 16

def BPX(data, consecutive_xor=True):
    '''
    prediction root = bit_plane[:, :, 0]
    xor base = bit_plane[:, 0, :] / consecutive xor base = bit_plane[:, :-1, :]
    '''
    block_size = data.shape[1]
    symbol_size = data.dtype.itemsize
    
    num_blocks = len(data)
    
    binary_data = np.unpackbits(data, axis=1).reshape(num_blocks, block_size, symbol_size * 8)
    bit_plane = np.moveaxis(binary_data, 1, -1)
    
    if consecutive_xor:
        bit_plane[:, 1:, 1:] = bit_plane[:, 1:, 1:] ^ bit_plane[:, :-1, 1:] # consecutive xor
    else:
        bit_plane[:, 1:, 1:] = bit_plane[:, 1:, 1:] ^ np.expand_dims(bit_plane[:, 0, 1:], 1) # base xor
    return bit_plane

def phi_scan(data, sel_cluster):
    num_lines = len(data)
    rows = data.shape[1]
    cols = data.shape[2]

    index_checklist = np.zeros(shape=(rows, cols), dtype=int)
    scanned_rows = []
    scanned_cols = []

    description = 'Computing scan route-%2d/%2d' %(sel_cluster, NUM_CLUSTERS)
    p_bar = tqdm(total = rows * cols, desc=description, ncols=150)

    # start index
    one_count_table = np.count_nonzero(data, axis=0)
    zero_count_table = num_lines - one_count_table
    start_index = np.unravel_index(np.argmax(zero_count_table, axis=None), zero_count_table.shape)

    index_checklist[start_index] = 1
    scanned_rows.append(start_index[0])  # scanned row
    scanned_cols.append(start_index[1])  # scanned col
    p_bar.update(1)

    # scan index
    prev_index = start_index
    for _ in range(rows * cols - 1):
        base_zero_data = data[data[:, prev_index[0], prev_index[1]] == 0]
        num_lines = len(base_zero_data)
        one_count_table = np.count_nonzero(base_zero_data, axis=0)
        zero_count_table = num_lines - one_count_table

        sorted_index = np.unravel_index(np.argsort(zero_count_table, axis=None)[::-1], zero_count_table.shape)
        for i in range(len(sorted_index[0])):
            index = (sorted_index[0][i], sorted_index[1][i])
            if index_checklist[index] == 0:
                index_checklist[index] = 1
                scanned_rows.append(index[0])  # scanned row
                scanned_cols.append(index[1])  # scanned col
                prev_index = index
                break
        p_bar.update(1)

    p_bar.close()
    return (scanned_rows, scanned_cols)
    
