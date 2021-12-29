import numpy as np
import torch

from tqdm.auto import tqdm
# from numba import njit
# from numba.typed import List

from src.const import *
from src.utils import *

# get dataset witout allzero and allwordsame
def get_data(dataset_path, remove_allzeros=True, remove_allwordsame=True):
    data = np.load(dataset_path).astype(np.uint8)

    @njit
    def exclude_all_word_same(data):
        new_data = List()
        for d in data:
            if not((d[0::4]==d[0] and d[1::4]==d[1] and d[2::4]==d[2] and d[3::4]==d[3]).all()):
                new_data.append(d)
        return new_data

    # exclude all-zeros lines
    if (remove_allzeros):
        non_zero_indices = np.where(data.any(axis=1))[0]
        data = data[non_zero_indices]

    # exclude all-word same lines
    if (remove_allwordsame):
        data = list(exclude_all_word_same(data))

    return data

# get dataset without allzero
# and labels #0 for all-zeros #1 for others
def get_data_and_label(dataset_path, label_path,
        batch_size=65536, device="cpu",
        desc="Selecting labels"):
    """
    Before:
        Bool            = 0,
        Char            = 1,
        Short           = 2,
        Int             = 3,
        Long            = 4,
        Float           = 5,
        Double          = 6
    After:
        All-Zeros       = 0,
        All-WordSame    = 1,
        Bool/Char       = 2,
        Short           = 3,
        Int/Long        = 4,
        Float           = 5,
        Double          = 6,
        Uncomp          = 7
    """
    data = np.load(dataset_path).astype(np.uint8)
    labels = np.load(label_path)

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    newlabels = []
    p_bar = tqdm(total=len(labels), desc=desc, ncols=TQDM_COLS)
    for minibatch, minibatch_labels in zip(iter_batch(data, batch_size), iter_batch(labels, batch_size)):
        minibatch = minibatch.to(device)
        minibatch_labels = minibatch_labels.to(device)
        minibatch_newlabels = torch.zeros_like(minibatch_labels)

        # BE CAUTIOUS!
        #  - NEW_LABELS CAN BE OVERWRITTEN!! CONSIDER THE PRIORITY

        # bool/char
        bool_char_indices = (minibatch_labels == 0) | (minibatch_labels == 1)
        minibatch_newlabels[bool_char_indices] = 2
        # short
        short_indices = (minibatch_labels == 2)
        minibatch_newlabels[short_indices] = 3
        # int/long
        int_long_indices = (minibatch_labels == 3) | (minibatch_labels == 4)
        minibatch_newlabels[int_long_indices] = 4
        # float
        float_indices = (minibatch_labels == 5)
        minibatch_newlabels[float_indices] = 5
        # double
        double_indices = (minibatch_labels == 6)
        minibatch_newlabels[double_indices] = 6

        # all-wordsame
        all_word_same_indices = ( (minibatch[:, 0::4] == torch.unsqueeze(minibatch[:, 0], -1)).all(dim=1)
                                & (minibatch[:, 1::4] == torch.unsqueeze(minibatch[:, 1], -1)).all(dim=1)
                                & (minibatch[:, 2::4] == torch.unsqueeze(minibatch[:, 2], -1)).all(dim=1)
                                & (minibatch[:, 3::4] == torch.unsqueeze(minibatch[:, 3], -1)).all(dim=1))
        minibatch_newlabels[all_word_same_indices] = 1
        # all-zero
        all_zero_indices = (minibatch == 0).all(dim=1)
        minibatch_newlabels[all_zero_indices] = 0

        newlabels.append(minibatch_newlabels.cpu())
        p_bar.update(len(minibatch))
    p_bar.close()

    newlabels = torch.concat(newlabels, dim=0)
    return data, newlabels

def sort_lines_by_class(data, labels):
    data_classes = {}
    for selected_class in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        data_classes[selected_class] = data[labels == selected_class]
    return data_classes

