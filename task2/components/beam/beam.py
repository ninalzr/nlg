import sys, os
#sys.path.insert(0, '../../..')

from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Beam():
    def __init__(self, decoder, alpha = 0.7):        
        self.decoder = decoder
        self.score = 0.
        self.sequence = []        
        self.alpha = alpha
        self.running = True
        
    def normalized_score(self):
        return self.score / math.pow(len(self.sequence), self.alpha)
        
    def ended(self):
        return False if self.sequence[-1] != self.decoder.end_symbol_id else True
            