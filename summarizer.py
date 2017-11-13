# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from lstmcell import StackedLSTMCell
from feature_extraction import ResNetFeature

class sLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, num_layers=2):
        """Scoring LSTM"""
        super().__init.__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequence(
                nn.Linear(2*hidden_size, 1),
                nn.Sigmoid())
        
    def score(self, features):
        """
        Args:
            features: (PackedSequence)
                data [total seq size in batch, 2048]
                batch_size

        Return:
            scores (PackedSequence)
                data [total seq size in batch, 1]
                batch_size
        """
        # batch_sizes for PackedSequence
        batch_sizes = features.batch_sizes
        
        scores = self.out(features.data)
        
        scores = PackedSequence(scores, batch_sizes)
        
        return scores
        
    def forward(self, features, init_hidden=None):
        """
        Args:
            pool5 features
            features (PackedSequence)
                data [total seq size in batch, 2048]
            
        Return:
            scores (PackedSequence)
                data[total seq size in batch, 1]
        """
        
        # features: PackedSequence
        # [total seq size in batch, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)
        
        # [total seq size in batchh, 1]
        scores = self.score(features)
        
        return scores
    
class eLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2):
        """Encoder LSTM"""
        super().__init.__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
                
    def forward(self, frame_features, batch_lengths):
        """
        Args:
            frame_features: [seq_len, batch_size, 2*hidden_size]
            batch_length: [batch_size]
        Return:
            last hidden
                h_last [2, batch_size, hidden_size]
                c_last [2, batch_size, hidden_size]
        """
        
        _, (h_last, c_last) = self.lstm(frame_features)
        
        return (h_last, c_last)

class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super().__init.__()
        
        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size, None)        
        self.out = nn.Linear(hidden_size, input_size)
        
    def forward(self, batch_length, init_hidden):
        
        max_len = max(batch_length)
        
        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)
        
        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden # (h_0, c_0): last state of eLSTM
        
        out_features = []
        for i in range(max_len):
            # last_h_c: [2, batch_size, hidden_size] (h from last layer)
            # h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
            last_h_c, (h, c) = self.lstm_cell(x, (h, c))
            last_h = last_h_c[0]
            x = self.out(last_h)
            out_features.append(last_h)
    
        # [max_seq_len, batch_size, hidden_size]
        # reverse
        out_features.reverse()
        
        return out_features
    
class Autoencoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)
        
    def forward(self, features, batch_length):
        """
        Args:
            features: (PackedSequence)
                data [total seq in batch, 2*1024]
        Return:
            features (Variable)
                [max_seq_len, batch_size, hidden_size]
        
        """
        h, c = self.e_lstm(features)
        
        # [max_seq_len, batch_size, hidden_size]
        features = self.d_lstm(batch_length=batch_length, init_hidden=(h, c))
        
        return h, features
        


class Summarizer(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, num_layers=2):
        super().__init__()
        self.resnet = ResNetFeature()
        
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.autoencoder = Autoencoder(input_size, 2*hidden_size, num_layers)
        
    def apply_weights(self, scores, image_features):
        """Weight score on image features"""
        weighted_features = image_features.data * scores
        return PackedSequence(weighted_features, image_features.batch_sizes)
    
    def uniform_score(self):
        pass        

        
    def forward(self, images, batch_length, score='score'):
        # [max_frame_len, batch, 2048]
        image_features = self.resnet(images)
        
        # TODO
        # convert to packedsequence
        
        # PackedSequence[max_frame_len, batch, 1]
        scores = self.s_lstm(image_features, batch_length)
        
        weighted_features = self.apply_weights(image_features, scores)
        
         = self.autoencoder(weighted_features, batch_length)
        
        return scores, weighted_features, , 
        
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

        
        
        
        

