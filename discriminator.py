# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:49:00 2017

@author: j-min
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class cLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2):
        """Discriminator LSTM"""
        super().__init.__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, batch, input_size]
        Return:
            last_h: [batch_size, hidden_size]
        """
        
        # output: seq_len, batch, hidden_size * num_directions
        # h_n, c_n: num_layers * num_directions, batch, hidden_size
        output, (h_n, c_n) = self.lstm(features, init_hidden)
        
        # [batch_size, hidden_size]
        last_h = h_n[-1]
        
        return last_h
    
class Disciriminator(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2):
        """Discriminator: cLSTM + output projection to scalar"""
        super().__init__()
        self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
        self.out = nn.Linear(1024, 1)
        
    def forward(self, features):
        """
        Args:
            features: [seq_len, batch_size, input_size]
        Return:
            Probability to be original feature from CNN
            prob: [batch_size, 1]
        """
        
        # [batch_size, hidden_size]
        h = self.cLSTM(features)
        
        # [batch_size, 1]
        prob = self.out(h)
        
        return prob
        
    
if __name__ == '__main__':
    batch_size = 3
    max_seq_len = 200
    batch_length = [200, 100, 10]
    demo_video1 = Variable(torch.randn([200, 1024])).cuda()
    demo_video2 = Variable(torch.randn([200, 1024])).cuda()
    demo_video3 = Variable(torch.randn([200, 1024])).cuda()
    
    # max_len, batch size, dim
    batch_videos = torch.stack([demo_video1, demo_video2, demo_video3], dim=1)
    
    batch = pack_padded_sequence(batch_videos, lengths=batch_length)
    
    pass