# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from lstmcell import StackedLSTMCell


class sLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid())

    def score(self, features):
        """
        Args:
            features: [seq_len, 1, 2048]
        Return:
            scores: [seq_len, 1]
        """

        scores = self.out(features.squeeze(1))

        return scores

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 2048] (pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.score(features)

        return scores


class eLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, 2*hidden_size]
        Return:
            last hidden
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """

        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len (int)
            init_hidden
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick"""
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).cuda()

        return mu + epsilon * std

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, 2048]
        Return:
            h: [2=num_layers, 1, 2048]
            decoded_features: [seq_len, 1, 2048]
        """
        seq_len = features.size(0)

        # [num_layers, 1, hidden_size]
        h, c = self.e_lstm(features)

        h_mu, h_log_variance = torch.chunk(h, chunks=2, dim=2)

        h = self.reparameterize(h_mu, h_log_variance)

        self.d_lstm.flatten_parameters()

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, 2 * hidden_size, num_layers)

    def forward(self, image_features, uniform=False):
        # Apply weights
        if not uniform:
            # [seq_len, 1]
            scores = self.s_lstm(image_features)

            # [seq_len, 1, 2048]
            weighted_features = image_features * scores
        else:
            weighted_features = image_features

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features


if __name__ == '__main__':

    pass
