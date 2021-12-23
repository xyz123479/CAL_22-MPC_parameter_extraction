import numpy as np
import torch

from tqdm.auto import tqdm
from src.const import *
from src.utils import *

def BPX(data, consecutive_xor=True):
    '''
    prediction root = bit_plane[:, :, 0]
    xor base = bit_plane[:, 0, :] / consecutive xor base = bit_plane[:, :-1, :]
    '''
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    binary_data = np.unpackbits(data, axis=1).reshape(len(data), LINESIZE, DTYPE_SIZE)
    bit_plane = np.moveaxis(binary_data, 1, -1)
    
    if consecutive_xor:
        bit_plane[:, 1:, 1:] = bit_plane[:, 1:, 1:] ^ bit_plane[:, :-1, 1:] # consecutive xor
    else:
        bit_plane[:, 1:, 1:] = bit_plane[:, 1:, 1:] ^ np.expand_dims(bit_plane[:, 0, 1:], 1) # base xor

    bit_plane = torch.from_numpy(bit_plane)
    return bit_plane

def search_idx(data, batch_size, device, desc):
    num_lines = len(data)
    rows = data.shape[1]
    cols = data.shape[2]

    p_bar = tqdm(total=num_lines, desc=desc, ncols=TQDM_COLS, leave=False, position=2)
    total_one_count_table = torch.zeros(size=(rows, cols), dtype=int, device=device)
    for minibatch in iter_batch(data, batch_size):
        minibatch = minibatch.to(device)
        one_count_table = torch.count_nonzero(minibatch, dim=0)
        total_one_count_table = total_one_count_table + one_count_table
        p_bar.update(len(minibatch))
    p_bar.close()
    total_zero_count_table = num_lines - total_one_count_table
    total_zero_count_table = total_zero_count_table.cpu().numpy()

    sorted_index = np.unravel_index(
            np.argsort(total_zero_count_table, axis=None)[::-1], total_zero_count_table.shape)
    return sorted_index

def phi_scan(data,
        batch_size=65536, device="cpu",
        desc="Searching High Zero Probability Order"):
    num_lines = len(data)
    rows = data.shape[1]
    cols = data.shape[2]

    index_checklist = np.zeros(shape=(rows, cols), dtype=int)
    scanned_rows = []
    scanned_cols = []

    p_bar = tqdm(total=rows*cols, desc=desc, ncols=TQDM_COLS, leave=False, position=1)

    # start idx
    inner_loop_desc = "%3d / %3d" %(1, rows*cols)
    sorted_index = search_idx(data, batch_size, device,
            desc=inner_loop_desc.rjust(TQDM_DESC_LEN))
    start_index = (sorted_index[0][0], sorted_index[1][0])

    # start idx check
    index_checklist[start_index] = 1
    scanned_rows.append(start_index[0])  # scanned row
    scanned_cols.append(start_index[1])  # scanned col
    p_bar.update(1)

    # high zero prob route search
    prev_index = start_index
    data_subset = data
    for curr in range(1, rows * cols):
        inner_loop_desc = "%3d / %3d" %(curr, rows*cols)
        data_subset = data_subset[data_subset[:, prev_index[0], prev_index[1]] == 0]
        sorted_next_index_candidates = search_idx(data_subset, batch_size, device,
                desc=inner_loop_desc.rjust(TQDM_DESC_LEN))
        for i in range(len(sorted_next_index_candidates[0])):
            index = (sorted_next_index_candidates[0][i], sorted_next_index_candidates[1][i])
            if index_checklist[index] == 0:
                index_checklist[index] = 1
                scanned_rows.append(index[0])  # scanned row
                scanned_cols.append(index[1])  # scanned col
                prev_index = index
                break
        p_bar.update(1)
    p_bar.close()
    return scanned_rows, scanned_cols

