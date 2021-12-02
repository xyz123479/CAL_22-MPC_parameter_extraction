import numpy as np
import pickle

import json

import os

from src.const import *

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def weight_PredModule(root_idx, base_idx_table, weight_table,
                      scan_table, consecutive_xor=True):
    module_spec = {}

    # module name
    module_spec["name"] = "PredComp"

    # submodules
    module_spec["submodules"] = {}
    submodules = module_spec["submodules"]

    # residue module spec
    submodules["ResidueModule"] = {
        "PredictorModule": {
            "name": "WeightBasePredictor",

            "LineSize": LINESIZE,
            "RootIndex": root_idx,
            "BaseIndexTable": base_idx_table,
            "WeightTable": weight_table,
        }
    }

    # xor module spec
    submodules["XORModule"] = {
        "consecutiveXOR": consecutive_xor,
    }

    # scan module spec
    scan_table_size = len(scan_table[0])
    submodules["ScanModule"] = {
        "TableSize": scan_table_size,

        "Rows": scan_table[0],
        "Cols": scan_table[1],
    }

    # FPC module spec
    submodules["FPCModule"] = {
        "num_modules" : 6,

        0: {
            "name": "ZerosPattern",

            "encodingBitsZRLE": 3,
            "encodingBitsZero": 4,
        },

        1: {
            "name": "SingleOnePattern",

            "encodingBits": 3,
        },

        2: {
            "name": "TwoConsecutiveOnesPattern",

            "encodingBits": 4,
        },

        3: {
            "name": "MaskingPattern",

            "encodingBits": 4,
            "maskingVector": [0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2],
        },

        4: {
            "name": "MaskingPattern",

            "encodingBits": 4,
            "maskingVector": [2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0],
        },

        5: {
            "name": "UncompressedPattern",

            "encodingBits": 1,
        },
    }

    return module_spec

def make_config(compression_tables, scan_tables):
    num_modules = NUM_CLUSTERS - 1
    
    # base config
    config = {
        "overview": {
            "num_modules": num_modules,
            "lineSize": LINESIZE,
        },
        
        "modules": {
            0: {
                "name": "AllZero",
            },
            1: {
                "name": "AllWordSame",
            },
        }
    }
    
    # add PredCompModules
    for selected_module in range(NUM_FIRST_CLUSTER, NUM_CLUSTERS - 1):
        root_idx = compression_tables[selected_module]['root_idx']
        base_idx_table = compression_tables[selected_module]['base_idx_table']
        weight_table = compression_tables[selected_module]['weight_table']
        scan_table = scan_tables[selected_module]
        
        module_spec = weight_PredModule(root_idx, base_idx_table, weight_table, scan_table)
        config["modules"][selected_module] = module_spec
    
    return config
