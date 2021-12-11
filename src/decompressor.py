import numpy as np
from math import log2, ceil

from copy import deepcopy

import os
import time

from tqdm.auto import tqdm

import json

from collections import OrderedDict

# decompressor
class Decompressor():
    def __init__(self, json_path, scanned_symbolsize=16):
        self.symbolsize = 8 # bit / symbol
        self.scanned_symbolsize = scanned_symbolsize
        self.decomp_modules = {}
        self.__parse_conf(json_path)
        
    def __call__(self, result):
        codeword = result['codeword']
        selected_class = result['selected_class']
        
        if(selected_class == -1):
            decomp_line = np.packbits(codeword)
        else:
            decomp_line = self.decomp_modules[selected_class](codeword)
        return decomp_line
        
    def __parse_conf(self, json_path):
        with open(json_path, 'r') as j:
            config = json.load(j)
        
        # parse overview
        overview = config['overview']
        self.linesize = overview['lineSize'] # [num of symbols]
        self.num_modules = overview['num_modules']
        self.uncompressed_linesize = self.linesize * self.symbolsize # bits

        # parse modules
        modules = config['modules']
        
        for num_module in modules.keys():
            if modules[num_module]['name'] == 'AllZero':
                self.decomp_modules[int(num_module)] = DeAllZero_Module(num_module, self.linesize, self.symbolsize)
                
            elif modules[num_module]['name'] == 'ByteplaneAllSame' or modules[num_module]['name'] == 'AllWordSame':
                self.decomp_modules[int(num_module)] = DeAllWordSame_Module(num_module, self.linesize, self.symbolsize)
                
            elif modules[num_module]['name'] == 'PredComp':
                submodules = modules[num_module]['submodules']
                
                # deresidue module
                residue_cfg = submodules['ResidueModule']
                predictor_cfg = residue_cfg['PredictorModule']
                if predictor_cfg['name'] == 'WeightBasePredictor':
                    deresidue_module = DeResidueWeightBase(predictor_cfg, self.linesize)
#                 elif predictor_cfg['name'] == 'DifferenceBasePredictor':
#                     pred_module = DiffBasePred(predictor_cfg)
#                 elif predictor_cfg['name'] == 'OneBasePredictor':
#                     pred_module = OneBasePred()
#                 elif predictor_cfg['name'] == 'ConsecutiveBasePredictor':
#                     pred_module = ConsecutivePred()
                else:
                    print("\"%s\" prediction module is not supported!" %(predictor_cfg['name']))
                    assert(False)

                # debpx module
                xor_cfg = submodules['XORModule']
                is_consec_xor = xor_cfg['consecutiveXOR']
                
                scan_cfg = submodules['ScanModule']
                scanned_symbolsize = 16
                row_indices = scan_cfg['Rows']
                col_indices = scan_cfg['Cols']
                
                debpx_module = DeDBX_Module(self.linesize, self.symbolsize,
                                        is_consec_xor, [row_indices, col_indices], scanned_symbolsize)
                
                # defpc module
                defpc_module = DeFPC_Module(self.scanned_symbolsize)
                
                self.decomp_modules[int(num_module)] = DePredComp_Module(num_module, self.linesize, self.symbolsize,
                                                                    deresidue_module, debpx_module, defpc_module)
                
            else:
                print("\"%s\" compression module is not supported!" %(modules[num_module]['name']))
                assert(False)
    
# deComp Module
class DeComp_Module():
    def __init__(self, number, linesize, symbolsize):
        self.number = number
        self.linesize = linesize # [num of symbols]
        self.symbolsize = symbolsize # bit / symbol
        self.uncompressed_linesize = linesize * symbolsize # bits
        
class DeAllZero_Module(DeComp_Module):
    def __init__(self, number, linesize, symbolsize):
        super(DeAllZero_Module, self).__init__(number, linesize, symbolsize)
        
    def __call__(self, codeword):
        return np.zeros(shape=(self.linesize), dtype=np.uint8)

class DeAllWordSame_Module(DeComp_Module):
    def __init__(self, number, linesize, symbolsize, byteplane_format=False):
        super(DeAllWordSame_Module, self).__init__(number, linesize, symbolsize)
        self.byteplane_format = byteplane_format
        
    def __call__(self, codeword):
        codeword = np.packbits(codeword)
        if self.byteplane_format:
            MSByte0 = np.full(shape=(self.linesize//4), fill_value=codeword[0], dtype=np.uint8)
            MSByte1 = np.full(shape=(self.linesize//4), fill_value=codeword[1], dtype=np.uint8)
            MSByte2 = np.full(shape=(self.linesize//4), fill_value=codeword[2], dtype=np.uint8)
            LSByte  = np.full(shape=(self.linesize//4), fill_value=codeword[3], dtype=np.uint8)
            line = np.concatenate((MSByte0, MSByte1, MSByte2, LSByte), axis=0)
        else:
            line = np.zeros(shape=(self.linesize), dtype=np.uint8)
            line[0::4] = codeword[0]
            line[1::4] = codeword[1]
            line[2::4] = codeword[2]
            line[3::4] = codeword[3]
        return line

class DePredComp_Module(DeComp_Module):
    def __init__(self, number, linesize, symbolsize, deresidue_module, debpx_module, defpc_module):
        super(DePredComp_Module, self).__init__(number, linesize, symbolsize)
        self.deresidue_module = deresidue_module
        self.debpx_module = debpx_module
        self.defpc_module = defpc_module
        
    def __call__(self, codeword):
        scanned_array = self.defpc_module(codeword)
        residue_line = self.debpx_module(scanned_array)
        line = self.deresidue_module(residue_line)
        return line

# dePrediction compression module
## deResdiue module & dePrediction module
class DeResidueWeightBase():
    def __init__(self, cfgs, linesize):
        self.linesize = linesize
        self.__makeDeResidueTable(cfgs)
        
    def __call__(self, root_permuted_residue_line, debug=False):
        # currently root is in front of the 'residue line'
        # root has to be moved where it originally placed which is 'root_idx'
        residue_line = []
        residue_line.extend(root_permuted_residue_line[1:self.root_idx+1])
        residue_line.extend([root_permuted_residue_line[0]])
        residue_line.extend(root_permuted_residue_line[self.root_idx+1:])
        
        line = np.zeros_like(residue_line, dtype=np.uint8)
        preds = np.zeros_like(residue_line, dtype=np.uint8)
        line[self.root_idx] = residue_line[self.root_idx]
        for level in self.target_idx_table.keys():
            if level == -1:
                continue
            for target_idx in self.target_idx_table[level]:
                pred = line[self.base_idx_table[target_idx]] * self.weight_table[target_idx]
                line[target_idx] = residue_line[target_idx] + pred
                preds[target_idx] = pred
        if debug:
            return line, preds
        else:
            return line
    
    def __makeDeResidueTable(self, cfgs):
        self.root_idx       = cfgs['RootIndex']
        self.base_idx_table = cfgs['BaseIndexTable']
        self.weight_table   = cfgs['WeightTable']
        
        # construct a target idx tree with cfg level by level
        markerTable = np.zeros_like(self.base_idx_table, dtype=int)
        markerTable[self.root_idx] = 1
        
        self.target_idx_table = OrderedDict()
        self.target_idx_table[-1] = [self.root_idx]
        
        level = 0
        while(not(markerTable == 1).all()):
            self.target_idx_table[level] = []
            for target_idx in range(len(self.base_idx_table)):
                if markerTable[target_idx] == 1:
                    continue
                base_idx = self.base_idx_table[target_idx]
                if base_idx in self.target_idx_table[level - 1]:
                    self.target_idx_table[level].append(target_idx)
                    markerTable[target_idx] = 1
            level += 1
            
            
            assert(level <= self.linesize)

## deFPC module
class DeFPC_Module():
    def __init__(self, scanned_symbolsize):
        self.scanned_symbolsize = scanned_symbolsize
        # zrle, zero, singleOne, twoConsecOnes, frontHalfZeros, backHalfZeros, Uncomp
        self.encoding_bits = ([0,1,0],
                              [0,0,1,1],
                              [0,1,1],
                              [0,0,0,0],
                              [0,0,0,1],
                              [0,0,1,0],
                              [1])
        self.codeword_len = (4,
                             0,
                             4,
                             4,
                             8,
                             8,
                             16)
        
    def __call__(self, compressed_line):
        scanned_array = self.DeFPC_fn(compressed_line)
        return scanned_array
    
    def DeFPC_fn(self, compressed_line):
        scanned_array = np.empty(shape=(0, self.scanned_symbolsize), dtype=np.uint8)
        
        while(compressed_line):
            for idx in range(len(self.encoding_bits)):
                encoding_bits = self.encoding_bits[idx]
                encoding_bit_len = len(encoding_bits)
                codeword_len = self.codeword_len[idx]
                if(compressed_line[:encoding_bit_len] == encoding_bits):
                    pattern = compressed_line[:encoding_bit_len + codeword_len]
                    del compressed_line[:encoding_bit_len + codeword_len]
                    scanned_array = self.__depatternize(pattern, idx, scanned_array)
                    break
                    
        return scanned_array
    
    def __depatternize(self, pattern, idx, scanned_array):
        # zrle
        if(idx == 0):
            p = pattern[-self.codeword_len[idx]:]
            run_length = sum(v<<i for i, v in enumerate(p[::-1])) + 1
            depattern = np.zeros(shape=(run_length, self.scanned_symbolsize), dtype=np.uint8)
        # zero
        elif(idx == 1):
            depattern = np.zeros(shape=(1, self.scanned_symbolsize), dtype=np.uint8)
        # single one
        elif(idx == 2):
            p = pattern[-self.codeword_len[idx]:]
            position = sum(v<<i for i, v in enumerate(p[::-1]))
            depattern = np.zeros(shape=(1, self.scanned_symbolsize), dtype=np.uint8)
            depattern[0, position] = 1
        # two consec ones
        elif(idx == 3):
            p = pattern[-self.codeword_len[idx]:]
            position = sum(v<<i for i, v in enumerate(p[::-1]))
            depattern = np.zeros(shape=(1, self.scanned_symbolsize), dtype=np.uint8)
            depattern[0, position]     = 1
            depattern[0, position + 1] = 1
        # front half zeros
        elif(idx == 4):
            backhalf = pattern[-self.codeword_len[idx]:]
            depattern = np.zeros(shape=(1, self.scanned_symbolsize), dtype=np.uint8)
            depattern[0, self.scanned_symbolsize//2:] = backhalf
        # back half zeros
        elif(idx == 5):
            fronthalf = pattern[-self.codeword_len[idx]:]
            depattern = np.zeros(shape=(1, self.scanned_symbolsize), dtype=np.uint8)
            depattern[0, :self.scanned_symbolsize//2] = fronthalf
        # uncompressed
        elif(idx == 6):
            depattern = np.zeros(shape=(1, self.scanned_symbolsize), dtype=np.uint8)
            depattern[0, :] = pattern[-self.codeword_len[idx]:]
            
        scanned_array = np.concatenate((scanned_array, depattern), axis=0)
        return scanned_array
    
## deDBX module
class DeDBX_Module():
    def __init__(self, linesize, symbolsize, consecutive_xor, scan_index, scanned_symbolsize):
        self.linesize = linesize # [num of symbols]
        self.symbolsize = symbolsize # bit / symbol
        
        # bpx
        self.consecutive_xor = consecutive_xor
        
        # scan
        self.scan_index = scan_index
        self.scanned_symbolsize = scanned_symbolsize # bit / symbol

    def __call__(self, scanned_array, debug=False):
        BPX_array = self.__descan(scanned_array)
        if debug:
            temp0 = deepcopy(BPX_array)
        bitplane_array = self.__dexor(BPX_array)
        if debug:
            temp1 = deepcopy(bitplane_array)
        binarized_array = deepcopy(np.transpose(bitplane_array))
        binarized_line = binarized_array.reshape(-1)
        line = np.packbits(binarized_line)
        if debug:
            return temp0, temp1, line
        else:
            return line

    def __dexor(self, bitplane_array):
        if self.consecutive_xor:
            for i in range(1, len(bitplane_array)):
                bitplane_array[i, 1:] = bitplane_array[i, 1:] ^ bitplane_array[i-1, 1:]
        else:
            bitplane_array[1:, 1:] = bitplane_array[1:, 1:] ^ bitplane_array[0, 1:]
        return bitplane_array
    
    def __descan(self, scanned_array):
        BPX_array = np.zeros_like(scanned_array)
        BPX_array = BPX_array.reshape(self.symbolsize, self.linesize)
        scanned_line = scanned_array.reshape(-1)
        BPX_array[self.scan_index] = scanned_line
        return BPX_array
    
