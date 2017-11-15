# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from summarizer import Summarizer
from discriminator import Discriminator
from feature_extraction import ResNetFeature
from tensorboard import TensorboardWriter
from tqdm import tqdm


class Solver(object):
    def __init__(self, config=None, data_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.data_loader = data_loader

    def build(self):
        self.summarizer = Summarizer(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers)
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers)

        if self.config.mode == 'train':
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.s_lstm.parameters())
                + list(self.summarizer.vae.e_lstm.parameters()))
            self.d_optimizer = optim.Adam(self.summarizer.vae.d_lstm.parameters())
            self.c_optimizer = optim.Adam(self.discriminator.parameters())

            # Overview Parameters
            print('Model Parameters')
            for name, param in self.summarizer.named_parameters():
                print('\t' + name + '\t', list(param.size()))
            for name, param in self.discriminator.named_parameters():
                print('\t' + name + '\t', list(param.size()))

            # Tensorboard
            self.writer = TensorboardWriter(self.config.logdir)

        else:
            self.resnet = ResNetFeature()

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.sum((h_origin - h_fake).pow(2))

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + -log_variance.exp() + mu.pow(2) + log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return F.mse_loss((torch.mean(scores) - self.config.summary_rate))

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""

        gan_loss = torch.log(original_prob) \
            + torch.log(1 - fake_prob) \
            + torch.log(1 - uniform_prob)

        return gan_loss

    def train(self):
        for epoch_i in range(self.config.n_epochs):
            for batch_i, image_features in enumerate(tqdm(self.data_loader, ncols=80)):

                # batch_size: 1
                # [1, seq_len, 2048]
                original_features = Variable(image_features).cuda()

                # [seq_len, 1, 2048]
                original_features = original_features.transpose(0, 1)

                #---- Train sLSTM, eLSTM ----#
                scores, h_mu, h_log_variance, generated_features = self.summarizer(
                    original_features)
                _, _, _, uniform_features = self.summarizer(
                    original_features, uniform=True)

                # Forward propagation
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
                prior_loss = self.prior_loss(h_mu, h_log_variance)
                sparsity_loss = self.sparsity_loss(scores)

                s_e_loss = reconstruction_loss + prior_loss + sparsity_loss

                self.s_e_optimizer.zero_grad()
                s_e_loss.backward()
                self.s_e_optimizer.step()

                self.writer.update_loss(s_e_loss, batch_i, 's_e_loss')

                #---- Train dLSTM ----#

                # Forward propagation
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
                gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)

                d_loss = reconstruction_loss + gan_loss

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                self.writer.update_loss(d_loss, batch_i, 'd_loss')

                #---- Train cLSTM ----#

                # Forward propagation
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                c_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)

                self.c_optimizer.zero_grad()
                c_loss.backward()
                self.c_optimizer.step()

                self.writer.update_loss(c_loss, batch_i, 'c_loss')

    def evalulation(self):
        # [seq_len, batch=1, 2048]
        # images = self.resnet(images)

        pass

    def pretrain(self):
        pass


if __name__ == '__main__':
    pass
