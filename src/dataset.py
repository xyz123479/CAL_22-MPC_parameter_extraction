import numpy as np

from numba import njit
from numba.typed import List

from src.const import *

# get dataset witout allzero
def get_data(dataset_path, remove_allzeros=True):
    data = np.load(dataset_path).astype(np.uint8)

    # exclude all-zeros
    @njit
    def exclude_allZeros(data):
        new_data = List()
        for d in data:
            if not((d == 0).all()):
                new_data.append(d)
        return new_data

    if (remove_allzeros):
        new_data = list(exclude_allZeros(data))
    else:
        new_data = data
    return np.array(new_data)

# get dataset without allzero
# and labels #0 for all-zeros #1 for others
def get_data_and_label(dataset_path, label_path):
    data = np.load(dataset_path).astype(np.uint8)
    labels = np.load(label_path)
    
    @njit
    def get_labels(data, labels):
        new_labels = List()
        for i, d in enumerate(data):
            if (d == 0).all():
                new_labels.append(0)
            else:
                new_labels.append(labels[i] + 1)
        return new_labels
    labels = np.array(get_labels(data, labels))
    return data, labels

def sort_lines_by_class(lines, labels):
    data_classes = {}
    for selected_class in range(1, NUM_CLUSTERS - 1):
        data_classes[selected_class] = np.array(sort_lines_numba(lines, labels, selected_class), dtype=np.uint8)
    return data_classes

@njit
def sort_lines_numba(lines, labels, selected_labels):
    sorted_lines = List()
    for idx, line in enumerate(lines):
        if(labels[idx] == selected_labels):
            sorted_lines.append(line)
    return list(sorted_lines)

