import numpy as np
from math import log2, ceil

from copy import deepcopy

import os
import time

from tqdm.auto import tqdm

import json
from src import *

# compressor
class SingleEncoderCompressor():
    def __init__(self, json_path):
        self.symbolsize = 8 # bit / symbol
        self.encoding_bits = {}
        self.comp_modules = {}
        self.__parse_config(json_path)
        self.fpc_module = FPC_Module()
    
    def __parse_config(self, json_path):
        with open(json_path, 'r') as j:
            config = json.load(j)
        
        # parse overview
        overview = config['overview']
        self.linesize = overview['lineSize'] # [num of symbols]
        self.num_modules = overview['num_modules']
        self.uncompressed_linesize = self.linesize * self.symbolsize # bits
        if ('encoding_bits' not in overview.keys()):
            encoding_bits = ceil(log2(self.num_modules + 1))
            for num_module in range(-1, self.num_modules):
                self.encoding_bits[num_module] = encoding_bits
        else:
            encoding_bits = overview['encoding_bits']
            for num_module in range(-1, self.num_modules):
                self.encoding_bits[num_module] = encoding_bits[num_module + 1]
        
        # parse modules
        modules = config['modules']
        
        for num_module in modules.keys():
            if modules[num_module]['name'] == 'AllZero':
                self.comp_modules[int(num_module)] = AllZero_Module(num_module, self.linesize, self.symbolsize)
                
            elif modules[num_module]['name'] == 'ByteplaneAllSame' or modules[num_module] == 'AllWordSame':
                self.comp_modules[int(num_module)] = AllWordSame_Module(num_module, self.linesize, self.symbolsize)
                
            elif modules[num_module]['name'] == 'PredComp':
                submodules = modules[num_module]['submodules']
                
                # residue module
                residue_cfg = submodules['ResidueModule']
                predictor_cfg = residue_cfg['PredictorModule']
                if predictor_cfg['name'] == 'WeightBasePredictor':
                    pred_module = WeightBasePred(predictor_cfg)
                elif predictor_cfg['name'] == 'DifferenceBasePredictor':
                    pred_module = DiffBasePred(predictor_cfg)
                elif predictor_cfg['name'] == 'OneBasePredictor':
                    pred_module = OneBasePred()
                elif predictor_cfg['name'] == 'ConsecutiveBasePredictor':
                    pred_module = ConsecutivePred()
                else:
                    print("\"%s\" prediction module is not supported!" %(predictor_cfg['name']))
                    assert(False)
                residue_module = Residue_Module(pred_module)
                
                # bpx module
                xor_cfg = submodules['XORModule']
                is_consec_xor = xor_cfg['consecutiveXOR']
                
                scan_cfg = submodules['ScanModule']
                scanned_symbolsize = 16
                row_indices = scan_cfg['Rows']
                col_indices = scan_cfg['Cols']
                
                bpx_module = BPX_Module(self.linesize, self.symbolsize,
                                        is_consec_xor, [row_indices, col_indices], scanned_symbolsize)
                
                self.comp_modules[int(num_module)] = PredComp_Module(num_module, self.linesize, self.symbolsize,
                                                                residue_module, bpx_module)
                
            else:
                print("\"%s\" compression module is not supported!" %(modules[num_module]['name']))
                assert(False)
        
    def __call__(self, dataline):
        assert(self.linesize == len(dataline))
    
        # pass line to all comp_module
        ## allzero
        size, codeword = self.comp_modules[0](dataline)
        if size == 0:
            result = {
                'original_size' : self.uncompressed_linesize,
#                 'compressed_size' : size + self.encoding_bits[0],
                'compressed_size' : size + 3,
                'codeword' : codeword,
                'selected_class' : 0,
            }
            return result
        
        ## others
        maxConsecZeros = 0
        selectedComp = -1
#         scanned_arrays = {}
        for number in range(1, NUM_CLUSTERS-1):
            scanned_array = self.comp_modules[number](dataline)
#             scanned_arrays[number] = scanned_array
            
            consecZeros = 0
            for row in scanned_array:
                if (row == 0).all():
                    consecZeros += 1
                else:
                    break    
            if maxConsecZeros <= consecZeros:
                selected_scanned_array = scanned_array
                selectedComp = number
                maxConsecZeros = consecZeros
        size, codeword = self.fpc_module(selected_scanned_array)
        if size >= 256:
            size = self.uncompressed_linesize
            codeword = np.unpackbits(dataline)
            selectedComp = NUM_CLUSTERS-1
        
        result = {
            'original_size' : self.uncompressed_linesize,
            'compressed_size' : size + 3,
            'codeword' : codeword,
            'selected_class' : selectedComp,
        }
        return result

