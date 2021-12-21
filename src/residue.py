import numpy as np

from tqdm.auto import tqdm

from src.const import *
from src.utils import *

def get_predict_matrix(root_idx, base_idx_table, weight_table, device="cpu"):
    predict_mat = torch.zeros((LINESIZE, LINESIZE), device=device)

    # pred[i] = symbol[base_idx] * weight
    #   base_idx = base_idx_table[i], weight = weight_table[i]
    for i in range(LINESIZE):
        base_idx = base_idx_table[i]
        weight = weight_table[i]
        predict_mat[base_idx, i] = weight

    return predict_mat

def batched_compute_residue(data, compression_table, batch_size=65536, device="cpu"):
    root_idx = compression_table['root_idx']
    base_idx_table = compression_table['base_idx_table']
    weight_table = compression_table['weight_table']

    predict_mat = get_predict_matrix(root_idx, base_idx_table, weight_table, device)

    residues = []
    p_bar = tqdm(total = len(data), desc="Computing residue", ncols=150)
    for minibatch in iter_batch(data, batch_size):
        minibatch = minibatch.to(device)

        # batched matrix[Bx1xN] x broadcasted matrix[NxN]
        #   = batched matrix[Bx1xN]
        lines = torch.unsqueeze(lines, dim=1)
        preds = torch.matmul(lines, predict_mat)
        preds = preds.squeeze().type(minibatch.dtype)

        residue = minibatch - preds
        
        # root is moved to the front
        roots = torch.unsqueeze(minibatch[:, root_idx], dim=-1)
        residue = torch.concat(
                (roots, residue[:, :root_idx], residue[:, root_idx+1:]), dim=1)
        residues.append(residue.cpu())
        p_bar.update(len(minibatch))
    p_bar.close()

    residues = torch.concat(residues, dim=0)
    return residues










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


