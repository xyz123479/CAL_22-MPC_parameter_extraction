import numpy as np

from tqdm.auto import tqdm

from src.const import *

def predict(line, base_idx_table, weight_table, root_idx):
    pred_line = []
    
    root = line[root_idx]
    pred_line.append(root)
    for target_idx in range(len(line)):
        if target_idx == root_idx:
            continue
            
        weight = weight_table[target_idx]
        base_idx = base_idx_table[target_idx]
        base = line[base_idx]
        pred = (weight * base).astype(line.dtype)
        
        pred_line.append(pred)
    return np.array(pred_line)

def compute_residue(data, compression_table, sel_cluster):
    root_idx = compression_table['root_idx']
    base_idx_table = compression_table['base_idx_table']
    weight_table = compression_table['weight_table']

    residue_list = []

    description = 'Computing residue-%2d/%2d' %(sel_cluster, NUM_CLUSTERS)
    p_bar = tqdm(total=len(data), desc=description, ncols=150)
    for line in data:
        pred = predict(line, base_idx_table, weight_table, root_idx)
        
        line_without_root = np.concatenate((line[:root_idx], line[root_idx+1:]))
        residue = np.concatenate((np.expand_dims(line[root_idx], 0),
                                  line_without_root - pred[1:]))
        residue_list.append(residue)
        p_bar.update(1)
    p_bar.close()

    residue_list = np.array(residue_list)
    return residue_list


