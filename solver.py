# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms

from summarizer import Summarizer
from discriminator import Discriminator

from tqdm import tqdm

class Solver(object):
    def __init__(self, config=None, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        
    def build(self):
        self.summarizer = Summarizer(self.config)
        self.discriminator = Discriminator(self.config)
        
    def reconstruction_loss(self):
        pass
    
    def prior_loss(self):
        pass
    
    def sparsity_loss(self):
        pass
    
    def gan_loss(self, real_features, fake_features, uniform_features):
        real_prob = self.discriminator(real_features)
        fake_prob = self.discriminator(fake_features)
        uniform_prob = self.discriminator(uniform_features)
        
        gan_loss = torch.log(real_prob) + \
            torch.log(1 - fake_prob) + \
            torch.log(1 - uniform_prob)

        return gan_loss
        
    def train(self):
        
        for epoch_i in range(self.config.n_epochs):
            for batch_i, images in tqdm(enumerate(self.data_loader)):
                
                images = Variable(images).cuda()
                
                real_features, fake_features = self.summarizer(images)
                
                self.discriminator(real_features, fake_features)
                
                
        
        pass
    
    def evalulation(self):
        pass
    
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
    
    solver = Solver()
    solver.build()
    solver.

        