# -*- coding: utf-8 -*-
import torch.nn as nn


class cLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, input_size]
        Return:
            last_h: [1, hidden_size]
        """
        self.lstm.flatten_parameters()

        # output: seq_len, batch, hidden_size * num_directions
        # h_n, c_n: num_layers * num_directions, batch_size, hidden_size
        output, (h_n, c_n) = self.lstm(features, init_hidden)

        # [batch_size, hidden_size]
        last_h = h_n[-1]

        return last_h


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator: cLSTM + output projection to probability"""
        super().__init__()
        self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h : [1, hidden_size]
                Last h from top layer of discriminator
            prob: [1=batch_size, 1]
                Probability to be original feature from CNN
        """

        # [1, hidden_size]
        h = self.cLSTM(features)

        # [1]
        prob = self.out(h).squeeze()

        return h, prob


if __name__ == '__main__':

    pass
