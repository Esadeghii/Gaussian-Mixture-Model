import torch
from utils.model import Model
import pandas as pd
import os
import numpy as np

class SequenceModel(Model):
    ALPHABET = ['A', 'C', 'G', 'T']

    def __init__(
        self,
        n_chars=4,
        seq_len=10,
        bidirectional=True,
        batch_size=32,
        hidden_layers=1,
        hidden_size=32,
        lin_dim=16,
        emb_dim=19,  # Ensure emb_dim is correctly set to 19
        dropout=0,
        beta=0.007,   # KLD weight
        gamma=1.0,  # VMIM weight
        continuous_code_dim=5,  # Dimension of continuous latent code
        discrete_code_dim=5,    # Dimension of discrete latent code
        num_categories=10,       # Number of categories for discrete latent code
        file_path='data-and-cleaning/seq_with_dis_normalized.csv',
        label_file_path='data-and-cleaning/supercleanGMMFilteredClusterd.xlsx'
    ):
        super(SequenceModel, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.lin_dim = lin_dim
        self.batch_size = batch_size
        self.beta = beta
        self.gamma = gamma
        self.continuous_code_dim = continuous_code_dim
        self.discrete_code_dim = discrete_code_dim
        self.num_categories = num_categories
        self.file_path = file_path
        self.label_file_path = label_file_path

        self.emb_lstm = torch.nn.LSTM(
            input_size=n_chars,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.latent_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * seq_len * 2, lin_dim),
            torch.nn.ReLU()
        )

        self.latent_mean = torch.nn.Linear(lin_dim, emb_dim)
        self.latent_log_std = torch.nn.Linear(lin_dim, emb_dim)

        self.dec_lin = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU()
        )

        self.dec_lstm = torch.nn.LSTM(
            input_size=1,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
            hidden_size=hidden_size
        )

        self.dec_final = torch.nn.Linear(hidden_size * 2 * self.emb_dim, n_chars * seq_len)

        # Ensure latent_code_fc outputs the correct dimension
        self.latent_code_fc = torch.nn.Linear(hidden_size * 2 * self.emb_dim, 2 * emb_dim)

        self.xavier_initialization()

    def encode(self, x):
        hidden, _ = self.emb_lstm(x.float())
        hidden = self.latent_linear(torch.flatten(hidden, 1))
        z_mean = self.latent_mean(hidden)
        z_log_std = self.latent_log_std(hidden)
        return torch.distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))

    def decode(self, z, c=None):
        hidden = self.dec_lin(z)
        hidden, _ = self.dec_lstm(hidden.view(-1, self.emb_dim, 1))
        hidden_flat = torch.flatten(hidden, 1)

        out = self.dec_final(hidden_flat).view(-1, self.seq_len, self.n_chars)

        if c is not None:
            # Ensure the latent_code dimension is twice the emb_dim
            latent_code = self.latent_code_fc(hidden_flat)
            if latent_code.size(1) != 2 * self.emb_dim:
                raise ValueError(f"Expected latent_code dimension to be {2 * self.emb_dim}, but got {latent_code.size(1)}")

            mu = latent_code[:, :self.emb_dim]
            log_sigma = latent_code[:, self.emb_dim:2 * self.emb_dim]
            return out, mu, log_sigma
        return out

    def reparametrize(self, dist):
        sample = dist.rsample()
        prior = torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        prior_sample = prior.sample()
        return sample, prior_sample, prior

    def forward(self, x):
        latent_dist = self.encode(x)
        latent_sample, prior_sample, prior = self.reparametrize(latent_dist)
        df = pd.read_csv(self.file_path)
        selected_columns = [f' dis {i}' for i in range(64)]
        df = df[selected_columns]

        # Convert DataFrame to tensor
        c = torch.tensor(df.values, dtype=torch.float32).to(x.device)
        
        output, mu, log_sigma = self.decode(latent_sample, c=c)
        
        return output, latent_dist, prior, latent_sample, prior_sample, latent_dist.loc, torch.log(latent_dist.scale), mu, log_sigma

    def __repr__(self):
        return 'SequenceVAE' + self.trainer_config

    def vmim_loss(self, c, mu, log_sigma, epsilon=1e-8):
        assert c.shape == mu.shape == log_sigma.shape, "Shapes of c, mu, and log_sigma must match"
        sigma = torch.exp(log_sigma) + epsilon
        log_p_c_given_theta = -0.5 * ((c - mu)**2 / sigma**2 + 2 * torch.log(sigma) + torch.log(2 * torch.tensor(np.pi)))
        #regularization = 0.01 * torch.sum(mu**2 + sigma**2)
        return -log_p_c_given_theta.mean()# + regularization