import torch

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

def compute_residue(data, compression_table, batch_size=65536, device="cpu"):
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
        lines = torch.unsqueeze(minibatch, dim=1).float()
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


