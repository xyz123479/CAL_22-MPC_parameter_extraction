import numpy as np
from math import log2, ceil

from copy import deepcopy

import os
import time

from tqdm.auto import tqdm

import json

from src import *

# compressor
class Compressor():
    def __init__(self, json_path):
        self.symbolsize = 8 # bit / symbol
        self.encoding_bits = {}
        self.comp_modules = {}
        self.__parse_config(json_path)
    
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
                
                # fpc module
                self.fpc_module = FPC_Module()
                
                self.comp_modules[int(num_module)] = PredComp_Module(num_module, self.linesize, self.symbolsize,
                                                                    residue_module, bpx_module)
                
            else:
                print("\"%s\" compression module is not supported!" %(modules[num_module]['name']))
                assert(False)
        
    def __call__(self, dataline):
        assert(self.linesize == len(dataline))
    
        uncompressed_codeword = np.unpackbits(dataline)
        selected_class = -1
        
        min_comp_size = self.uncompressed_linesize

        # allzero
        size, codeword = self.comp_modules[0](dataline)
        if size == 0:
            compressed_size = (self.encoding_bits[selected_class] + min_comp_size)
            result = {
                'original_size' : self.uncompressed_linesize,
                'compressed_size' : compressed_size,
                'codeword' : codeword,
                'selected_class' : 0,
            }
            return result

        # not all zero
        else:
            # pass dataline to all comp_module
            max_num_scanned_zero = 0
            sel_scanned_array = np.ones(shape=(SCANNED_SYMBOL_SIZE, SCANNED_SYMBOL_SIZE), dtype=np.uint8)
            for number in range(1, NUM_CLUSTERS-1):
                scanned_array = self.comp_modules[number](dataline)
                num_scanned_zero = (SCANNED_SYMBOL_SIZE*SCANNED_SYMBOL_SIZE) - np.count_nonzero(scanned_array)
                if max_num_scanned_zero < num_scanned_zero:
                    max_num_scanned_zero = num_scanned_zero
                    sel_scanned_array = scanned_array
                    selected_class = number
            size, codeword = self.fpc_module(sel_scanned_array)
            if (size >= self.uncompressed_linesize):
                size = self.uncompressed_linesize
                selected_class = -1
                codeword = uncompressed_codeword
            compressed_size = (self.encoding_bits[selected_class] + min_comp_size)
        
            result = {
                'original_size' : self.uncompressed_linesize,
                'compressed_size' : compressed_size,
                'codeword' : codeword,
                'selected_class' : selected_class,
            }
            return result

# compression module
class Comp_Module():
    def __init__(self, number, linesize, symbolsize):
        self.number = number
        self.linesize = linesize # [num of symbols]
        self.symbolsize = symbolsize # bit / symbol
        self.uncompressed_linesize = linesize * symbolsize # bits
        
class AllZero_Module(Comp_Module):
    def __init__(self, number, linesize, symbolsize):
        super(AllZero_Module, self).__init__(number, linesize, symbolsize)
        
    def __call__(self, line):
        is_allZeros = (line == 0).all()

        if is_allZeros:
            return 0, []
        else:
            return self.uncompressed_linesize, np.unpackbits(line)

class AllWordSame_Module(Comp_Module):
    def __init__(self, number, linesize, symbolsize, byteplane_format=False):
        super(AllWordSame_Module, self).__init__(number, linesize, symbolsize)
        self.byteplane_format = byteplane_format
        
    def __call__(self, line):
        codeword = []
        if self.byteplane_format:
            is_byteplaneSame = (    (line[                    :(self.linesize//4)  ] == line[0]).all()
                                and (line[(self.linesize//4)  :(self.linesize//4)*2] == line[self.linesize//4]).all()
                                and (line[(self.linesize//4)*2:(self.linesize//4)*3] == line[(self.linesize//4)*2]).all()
                                and (line[(self.linesize//4)*3:                    ] == line[(self.linesize//4)*3]).all())
            codeword.extend([line[0], line[self.linesize//4], line[(self.linesize//4)*2], line[(self.linesize//4)*3]])
        else:
            is_byteplaneSame = (   (line[3::4] == line[3]).all()
                               and (line[2::4] == line[2]).all()
                               and (line[1::4] == line[1]).all()
                               and (line[0::4] == line[0]).all())
            codeword.extend(line[0:4])
        
        if is_byteplaneSame:
            return self.symbolsize * 4, np.unpackbits(codeword)
        else:
            return self.uncompressed_linesize, np.unpackbits(line)

class PredComp_Module(Comp_Module):
    def __init__(self, number, linesize, symbolsize, residue_module, bpx_module):
        super(PredComp_Module, self).__init__(number, linesize, symbolsize)
        self.residue_module = residue_module
        self.bpx_module = bpx_module
        
    def __call__(self, line):
        residue_line = self.residue_module(line)
        scanned_array = self.bpx_module(residue_line)
#         compressed_size, codeword = self.fpc_module(scanned_array)
        return scanned_array
    
# prediction compression module
## residue module & prediction module
class Residue_Module():
    '''
    residue_line : [root, residues]
    '''
    def __init__(self, pred_module):
        self.root_idx = pred_module.root_idx
        self.pred_module = pred_module
        
    def __call__(self, line):
        pred_line = self.pred_module(line)

        root = line[self.root_idx]
        no_root_line = np.concatenate((line[:self.root_idx], line[self.root_idx+1:]))
        residue_line = np.concatenate((np.expand_dims(root, 0), no_root_line - pred_line[1:]))
        return residue_line

class WeightBasePred():
    def __init__(self, cfgs):
        self.root_idx       = cfgs['RootIndex']
        self.base_idx_table = cfgs['BaseIndexTable']
        self.weight_table   = cfgs['WeightTable']
        
        self.linesize = len(self.base_idx_table)
        
    def __call__(self, line):
        pred_line = []
        
        root = line[self.root_idx]
        pred_line.append(root)
        for target_idx in range(self.linesize):
            if target_idx == self.root_idx:
                continue
                
            weight   = self.weight_table[target_idx]
            base_idx = self.base_idx_table[target_idx]
            base     = line[base_idx]
            pred     = (weight * base).astype(line.dtype)
            
            pred_line.append(pred)
        pred_line = np.array(pred_line)
        return pred_line
    
class DiffBasePred():
    def __init__(self, cfgs):
        self.root_idx       = cfgs['RootIndex']
        self.base_idx_table = cfgs['BaseIndexTable']
        self.diff_table     = cfgs['DifferenceTable']
        
        self.linesize = len(self.base_idx_table)
        
    def __call__(self, line):
        pred_line = []
        
        root = line[self.root_idx]
        pred_line.append(root)
        for target_idx in range(self.linesize):
            if target_idx == self.root_idx:
                continue
                
            diff     = self.diff_table[target_idx]
            base_idx = self.base_idx_table[target_idx]
            base     = line[base_idx]
            pred     = (diff + base).astype(line.dtype)
            
            pred_line.append(pred)
        pred_line = np.array(pred_line)
        return pred_line
    
class OneBasePred():
    def __init__(self):
        self.root_idx = 0
        
    def __call__(self, line):
        pred_line = np.full_like(line, line[0], dtype=line.dtype)
        return pred_line
    
class ConsecutivePred():
    def __init__(self):
        self.root_idx = 0
    
    def __call__(self, line):
        pred_line = np.concatenate((np.expand_dims(line[0], 0), line[:-1]))
        return pred_line

## BPX module
class BPX_Module():
    def __init__(self, linesize, symbolsize, consecutive_xor, scan_index, scanned_symbolsize):
        self.linesize = linesize # [num of symbols]
        self.symbolsize = symbolsize # bit / symbol
        
        # bpx
        self.consecutive_xor = consecutive_xor
        
        # scan
        self.scan_index = (scan_index[0], scan_index[1])
        self.scanned_symbolsize = scanned_symbolsize # bit / symbol
        
    def __call__(self, line):
        binarized_line = np.unpackbits(line.astype(np.uint8))
        binarized_array = binarized_line.reshape(self.linesize, self.symbolsize)
        bitplane_array = deepcopy(np.transpose(binarized_array))
        
        self.bitplane_array = deepcopy(bitplane_array)
        
        BPX_array = self.__xor(bitplane_array)
        scanned_array = self.__scan(BPX_array)

        self.BPX_array = deepcopy(BPX_array)

        return scanned_array
        
    def __xor(self, bitplane_array):
        '''
        first row : xor base
        first col : residue root
        '''
        if self.consecutive_xor:
            bitplane_array[1:, 1:] = bitplane_array[1:, 1:] ^ bitplane_array[:-1, 1:]
        else:
            bitplane_array[1:, 1:] = bitplane_array[1:, 1:] ^ bitplane_array[0, 1:]
        return bitplane_array
    
    def __scan(self, BPX_array):
        scanned_line = BPX_array[self.scan_index]
        scanned_array = scanned_line.reshape(-1, self.scanned_symbolsize)
        return scanned_array
        
## FPC module
class FPC_Module():
    def __init__(self):
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
        
    def __call__(self, scanned_array):
        compressed_size, codewords = self.FPC_fn(scanned_array)
        return compressed_size, codewords
    
    def FPC_fn(self, arr):
        zrle = 0
        run_length = 0
        compressed_size = 0

        codewords = []
        for row in arr:
            if (row == 0).all():
                zrle = 1 if zrle == 0 else zrle
                run_length += 1
            else:
                if zrle == 1:
                    # 0. ZRLE
                    if run_length > 1:
                        compressed_size += len(self.encoding_bits[0]) + self.codeword_len[0]
                        codewords.extend(self.encoding_bits[0])  # encoding bits
                        run_length_bin = '{0:04b}'.format(run_length - 1)
                        codewords.extend(list(map(int, run_length_bin)))  # codeword
                        zrle = 0
                        run_length = 0

                    # 1. Zero Symbol
                    else:
                        compressed_size += len(self.encoding_bits[1]) + self.codeword_len[1]
                        codewords.extend(self.encoding_bits[1])  # encoding bits
                        zrle = 0
                        run_length = 0

                # 2. Single One
                idx = np.where(row == 1)[0]
                if (row == 1).sum() == 1:
                    compressed_size += len(self.encoding_bits[2]) + self.codeword_len[2]
                    codewords.extend(self.encoding_bits[2])  # encoding bits
                    where_one = '{0:04b}'.format(idx[0])
                    codewords.extend(list(map(int, where_one)))  # codeword
                    continue

                # 3. Two Consecutive Ones
                if (row == 1).sum() == 2 and (idx[1] - idx[0]) == 1:
                    compressed_size += len(self.encoding_bits[3]) + self.codeword_len[3]
                    codewords.extend(self.encoding_bits[3])  # encoding bits
                    where_one = '{0:04b}'.format(idx[0])
                    codewords.extend(list(map(int, where_one)))  # codeword
                    continue

                # 4. Front Half Zeros
                if (row[:8] == 0).all():
                    compressed_size += len(self.encoding_bits[4]) + self.codeword_len[4]
                    codewords.extend(self.encoding_bits[4])  # encoding bits
                    codewords.extend(row[8:])  # codeword
                    continue

                # 5. Back Half Zeros
                if (row[8:] == 0).all():
                    compressed_size += len(self.encoding_bits[5]) + self.codeword_len[5]
                    codewords.extend(self.encoding_bits[5])  # encoding bits
                    codewords.extend(row[:8])  # codeword
                    continue

                # 6. Uncompressed
                compressed_size += len(self.encoding_bits[6]) + self.codeword_len[6]
                codewords.extend(self.encoding_bits[6])  # encoding bits
                codewords.extend(row)  # codeword

        # last row zrle check
        if zrle == 1:
            # 0. ZRLE
            if run_length > 1:
                compressed_size += len(self.encoding_bits[0]) + self.codeword_len[0]
                codewords.extend(self.encoding_bits[0])  # encoding bits
                run_length_bin = '{0:04b}'.format(run_length - 1)
                codewords.extend(list(map(int, run_length_bin)))  # codeword

            # 1. Zero Symbol
            else:
                compressed_size += len(self.encoding_bits[1]) + self.codeword_len[1]
                codewords.extend(self.encoding_bits[1])  # encoding bits

        assert(compressed_size == len(codewords))
        return compressed_size, codewords
    
## encoder module
class Encoder_Module():
    def __init__(self):
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
        
    def __call__(self, scanned_array):
        return self.Encoder_fn(scanned_array)
    
    def Encoder_fn(self, scanned_arr):
        isZeros = []
        sizeList = []
        codewordList = []
        ## BPEncoder
        for scanned_line in scanned_arr:
            isZero, size, codeword = self.BPEncoder_fn(scanned_line)
            isZeros.append(isZero), sizeList.append(size), codewordList.append(codeword)

        ## ZRLE module
        zrl_start_idx = -1
        run_length = 0
        for i, isZero in enumerate(isZeros):
            if isZero == 1:
                zrl_start_idx = i if zrl_start_idx == -1 else zrl_start_idx
                run_length += 1
                
                sizeList[i] = 0
                codewordList[i] = [0] * 17
            else:
                if (zrl_start_idx != -1):
                    # zrle
                    if (run_length > 1):
                        size = len(self.encoding_bits[0]) + self.codeword_len[0]
                        
                        codeword = []
                        dummybits = [0] * 10
                        codeword.extend(self.encoding_bits[0])
                        run_length_bin = '{:04b}'.format(run_length - 1)
                        codeword.extend(list(map(int, run_length_bin)))
                        codeword.extend(dummybits)
                        
                        sizeList[zrl_start_idx] = size
                        codewordList[zrl_start_idx] = codeword
                    # zero
                    else:
                        size = len(self.encoding_bits[1]) + self.codeword_len[1]
                        
                        codeword = []
                        dummybits = [0] * 13
                        codeword.extend(self.encoding_bits[1])
                        codeword.extend(dummybits)
                        
                        sizeList[zrl_start_idx] = size
                        codewordList[zrl_start_idx] = codeword
                    
                zrl_start_idx = -1
                run_length = 0
        
        if (zrl_start_idx != -1):
            # zrle
            if (run_length > 1):
                size = len(self.encoding_bits[0]) + self.codeword_len[0]

                codeword = []
                dummybits = [0] * 10
                codeword.extend(self.encoding_bits[0])
                run_length_bin = '{:04b}'.format(run_length - 1)
                codeword.extend(list(map(int, run_length_bin)))
                codeword.extend(dummybits)

                sizeList[zrl_start_idx] = size
                codewordList[zrl_start_idx] = codeword
            # zero
            else:
                size = len(self.encoding_bits[1]) + self.codeword_len[1]

                codeword = []
                dummybits = [0] * 13
                codeword.extend(self.encoding_bits[1])
                codeword.extend(dummybits)

                sizeList[zrl_start_idx] = size
                codewordList[zrl_start_idx] = codeword
                    
        compressed_size = np.sum(sizeList)
        return compressed_size, sizeList, codewordList
        
    def BPEncoder_fn(self, scanned_line):
        isZero_o = 0
        if (scanned_line == 0).all():
            isZero_o = 1
            size_o = 0x1f
            codeword_o = [1] * 17
        else:
            # 2. Single One
            idx = np.where(scanned_line == 1)[0]
            if (scanned_line == 1).sum() == 1:
                # size_o
                size_o = len(self.encoding_bits[2]) + self.codeword_len[2]

                # codeword_o
                dummybits = [0] * 10
                codeword_o = []
                codeword_o.extend(self.encoding_bits[2])  # encoding bits
                where_one = '{0:04b}'.format(idx[0])
                codeword_o.extend(list(map(int, where_one)))  # codeword
                codeword_o.extend(dummybits)

            # 3. Two Consecutive Ones
            elif (scanned_line == 1).sum() == 2 and (idx[1] - idx[0]) == 1:
                # size_o
                size_o = len(self.encoding_bits[3]) + self.codeword_len[3]

                # codeword_o
                dummybits = [0] * 9
                codeword_o = []
                codeword_o.extend(self.encoding_bits[3])  # encoding bits
                where_one = '{0:04b}'.format(idx[0])
                codeword_o.extend(list(map(int, where_one)))  # codeword
                codeword_o.extend(dummybits)

            # 4. Front Half Zeros
            elif (scanned_line[:8] == 0).all():
                # size_o
                size_o = len(self.encoding_bits[4]) + self.codeword_len[4]

                # codeword_o
                dummybits = [0] * 5
                codeword_o = []
                codeword_o.extend(self.encoding_bits[4])  # encoding bits
                codeword_o.extend(scanned_line[8:])  # codeword
                codeword_o.extend(dummybits)

            # 5. Back Half Zeros
            elif (scanned_line[8:] == 0).all():
                # size_o
                size_o = len(self.encoding_bits[5]) + self.codeword_len[5]

                # codeword_o
                dummybits = [0] * 5
                codeword_o = []
                codeword_o.extend(self.encoding_bits[5])  # encoding bits
                codeword_o.extend(scanned_line[:8])  # codeword
                codeword_o.extend(dummybits)
            
            # 6. Uncompressed
            else:
                # size_o
                size_o = len(self.encoding_bits[6]) + self.codeword_len[6]
                
                # codeword_o
                codeword_o = []
                codeword_o.extend(self.encoding_bits[6])
                codeword_o.extend(scanned_line)
                
        return isZero_o, size_o, codeword_o
        
        
        
        
        
        
        
        
        
        


