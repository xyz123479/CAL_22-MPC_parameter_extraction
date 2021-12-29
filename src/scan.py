import numpy as np
import torch
import cupy as cp

from tqdm.auto import tqdm
from src.const import *
from src.utils import *

def BPX(data, consecutive_xor=True,
        batch_size=65536, device="cpu"):
    '''
    prediction root = bit_plane[:, :, 0]
    xor base = bit_plane[:, 0, :] / consecutive xor base = bit_plane[:, :-1, :]
    '''
#     if isinstance(data, torch.Tensor):
#         data = data.cpu().numpy()
    bitplanes = [] 
    for minibatch in iter_batch(data, batch_size):
        minibatch = minibatch.to(device)

        # unpackbits whether it is on gpu or cpu
        if minibatch.is_cuda:
            minibatch = cp.asarray(minibatch)
            binary_batch = cp.unpackbits(minibatch)
        else:
            minibatch = minibatch.numpy()
            binary_batch = np.unpackbits(minibatch, axis=1)
        binary_batch = torch.as_tensor(binary_batch)

        binary_batch = binary_batch.reshape(len(minibatch), LINESIZE, DTYPE_SIZE)
        bitplane = torch.movedim(binary_batch, 1, -1)

        if consecutive_xor:
            bitplane[:, 1:, 1:] = bitplane[:, 1:, 1:] ^ bitplane[:, :-1, 1:]
        else:
            bitplane[:, 1:, 1:] = bitplane[:, 1:, 1:] ^ torch.unsqueeze(bitplane[:, 0, 1:], 1)
        bitplanes.append(bitplane.cpu())
    bitplanes = torch.concat(bitplanes, dim=0)
    return bitplanes

def search_idx(data, prev_index=None,
        batch_size=65536, device="cpu",
        p_bar=None):
    num_lines = len(data)
    rows = data.shape[1]
    cols = data.shape[2]

#     p_bar = tqdm(total=num_lines, desc=desc, ncols=TQDM_COLS, leave=False, position=2)
    total_one_count_table = torch.zeros(size=(rows, cols), dtype=int, device=device)
    for minibatch in iter_batch(data, batch_size):
        minibatch = minibatch.to(device)
        if prev_index is not None:
            minibatch = minibatch[minibatch[:, prev_index[0], prev_index[1]] == 0]
        one_count_table = torch.count_nonzero(minibatch, dim=0)
        total_one_count_table = total_one_count_table + one_count_table
        if p_bar is not None:
            p_bar.update(len(minibatch))
#     p_bar.close()
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

    p_bar = tqdm(total=rows*cols*num_lines, desc=desc, ncols=TQDM_COLS, leave=False, position=1)

    # start idx
#     inner_loop_desc = "%3d / %3d" %(1, rows*cols)
    sorted_index = search_idx(data,
            batch_size=batch_size, device=device,
            p_bar=p_bar)
    start_index = (sorted_index[0][0], sorted_index[1][0])

    # start idx check
    index_checklist[start_index] = 1
    scanned_rows.append(start_index[0])  # scanned row
    scanned_cols.append(start_index[1])  # scanned col
#     p_bar.update(1)

    # high zero prob route search
    prev_index = start_index
    data_subset = data
    for curr in range(1, rows * cols):
#         inner_loop_desc = "%3d / %3d" %(curr, rows*cols)
#         data_subset = data[data[:, prev_index[0], prev_index[1]] == 0]
        sorted_next_index_candidates = search_idx(data, prev_index=prev_index,
                batch_size=batch_size, device=device,
                p_bar=p_bar)
        for i in range(len(sorted_next_index_candidates[0])):
            index = (sorted_next_index_candidates[0][i], sorted_next_index_candidates[1][i])
            if index_checklist[index] == 0:
                index_checklist[index] = 1
                scanned_rows.append(index[0])  # scanned row
                scanned_cols.append(index[1])  # scanned col
                prev_index = index
                break
    p_bar.close()
    return scanned_rows, scanned_cols

