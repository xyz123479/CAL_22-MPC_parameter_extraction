import torch

from src.const import *

##### utils
def iter_batch(iterable, n=1):
    l = len(iterable)
    for idx in range(0, l, n):
        yield iterable[idx:min(idx + n, l)]

##### rounding functions
class power2:
    def __init__(self, prec=256):
        self.prec = prec
        
    def __call__(self, data):
        sign_array = torch.sign(data)
        powered_array = torch.exp2(torch.round(torch.log2(torch.abs(data) + MIN_VAL)))

        powered_array[powered_array < 1 / self.prec] = 0
        powered_array[powered_array > self.prec] = self.prec

        return powered_array * sign_array

    def computeScalar(self, data):
        sign = 1 if data >= 0 else -1
        powered = np.exp2(np.round(np.log2(np.abs(data) + MIN_VAL)))

        powered = 0 if powered < 1 / self.prec else powered
        powered = self.prec if powered > self.prec else powered

        return sign * powered

class rounding:
    def __init__(self, decimal=-4):
        self.decimal = decimal
        
    def __call__(self, data):
        return torch.round(data * 10**self.decimal) / (10**self.decimal)
    
class quantizing:
    def __init__(self, prec=64):
        self.prec = prec
        self.bins = torch.arange(0, DTYPE_RANGE, 1/prec)
        
    def __call__(self, data):
        indices = torch.bucketize(data, self.bins)
        return self.bins[indices-1].to(data.device)

