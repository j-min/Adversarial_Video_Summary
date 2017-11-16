import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def apply_weight_norm(module):
    """Recursively apply weight norm to children of given module"""
    if isinstance(module, nn.Linear):
        weight_norm(module, 'weight')
    if isinstance(module, (nn.RNNCell, nn.GRUCell, nn.LSTMCell)):
        weight_norm(module, 'weight_ih')
        weight_norm(module, 'weight_hh')
    if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        for i in range(module.num_layers):
            weight_norm(module, f'weight_ih_l{i}')
            weight_norm(module, f'weight_hh_l{i}')
            if module.bidirectional:
                weight_norm(module, f'weight_ih_l{i}_reverse')
                weight_norm(module, f'weight_hh_l{i}_reverse')
