import torch
from torch import nn as nn
import torch.nn.functional as F


class CVAE(torch.nn.Module):
    def __init__(self, input_size, conditioning_size, latent_size, encoder_hidden_size, prior_hidden_size,
                 decoder_hidden_size):
        super(CVAE, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.conditioning_size = conditioning_size

        # Encoder
        # Variational posterior Q(z|x, c)
        self.Q_fc1 = nn.Linear(input_size + conditioning_size, encoder_hidden_size)
        self.Q_activation_fn = nn.ReLU()
        self.Q_fc21 = nn.Linear(encoder_hidden_size, latent_size)
        self.Q_fc22 = nn.Linear(encoder_hidden_size, latent_size)

        # Prior
        self.P_fc1 = nn.Linear(conditioning_size, prior_hidden_size)
        self.P_activation_fn = nn.ReLU()
        self.P_fc21 = nn.Linear(prior_hidden_size, latent_size)
        self.P_fc22 = nn.Linear(prior_hidden_size, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size + conditioning_size, decoder_hidden_size)
        self.prior_activation_fn = nn.ReLU()
        self.fc4 = nn.Linear(decoder_hidden_size, input_size)

    def encode(self, s, c):
        """
        Encode slate for user
        :param s:  input slate encoding
        :param c: conditioning variable
        :return mean, log var of Q - params of z distribution
        """
        input = torch.cat([s, c], 1)
        h = self.Q_fc1(input)
        h = self.Q_activation_fn(h)
        mean = self.Q_fc21(h)
        log_var = self.Q_fc22(h)

        return mean, log_var, h

    def compute_prior(self, c):
        h = self.P_fc1(c)
        h = self.P_activation_fn(h)
        mean = self.P_fc21(h)
        log_var = self.P_fc22(h)

        return mean, log_var

    def decode(self, z, c):
        """
        :param z: latent variable
        :param c: conditioning variable
        :return x: recommendation slate
        """
        input = torch.cat([z, c], 1)
        h = self.fc3(input)
        h = self.prior_activation_fn(h)
        s_reconstruction = self.fc4(h)

        return s_reconstruction, h

    def reparametrize(self, mean, log_var):
        #  sample random number, same size, device as 'mean'
        epsilon = torch.randn_like(mean)

        # log_var unconstrained, get variance from it
        std = torch.exp(0.5 * log_var)

        # element-wise multiplication since covariance is diagonal
        return std * epsilon + mean

    def predict(self, c):
        P_mean, P_log_var = self.compute_prior(c)
        z = self.reparametrize(P_mean, P_log_var)
        s_recon, _ = self.decode(z, c)

        return s_recon

    def forward(self, s, c):
        Q_mean, Q_log_var, h_encoder = self.encode(s, c)
        P_mean, P_log_var = self.compute_prior(c)
        z = self.reparametrize(Q_mean, Q_log_var)
        s_reconstruction, h_decoder = self.decode(z, c)

        return s_reconstruction, Q_mean, Q_log_var, h_encoder, P_mean, P_log_var, h_decoder