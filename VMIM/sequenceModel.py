import torch
import torch.nn.functional as F
import pandas as pd
from utils.model import Model

class DataLoader:
    def __init__(self, file_path='data-and-cleaning/seq_with_dis.csv'):
        self.file_path = file_path

    def load_spectra(self, device=torch.device('cpu')):
        df = pd.read_csv(self.file_path)
        #print("Columns in CSV file:", df.columns)  # Print column names for debugging
        selected_columns = [f' dis {i}' for i in range(64)]
        if not all(col in df.columns for col in selected_columns):
            raise KeyError(f"Some of the required columns are not in the DataFrame: {selected_columns}")
        df = df[selected_columns]
        spectra = torch.tensor(df.values, dtype=torch.float32).to(device)
        return spectra

class SequenceModel(Model):
    ALPHABET = ['A','C','G','T']
    def __init__(
        self,
        n_chars = 4,
        seq_len = 10,
        bidirectional = True,
        batch_size = 32,
        hidden_layers = 1,
        hidden_size = 32,
        lin_dim = 16,
        emb_dim = 10,
        dropout = 0,
        file_path='data-and-cleaning/seq_with_dis.csv'
        #lstm_dropout = 0
    ):
        super(SequenceModel,self).__init__()
        self.n_chars = n_chars          #Number of DNA Bases, 4
        self.seq_len = seq_len          #Number of bases in a sequence, 10
        self.hidden_size = hidden_size  #Number of features hidden in the LSTM
        self.emb_dim = emb_dim          #Width of the embeded dimention
        self.lin_dim = lin_dim          #Width of the linear layer to transform the input and output of the latent space
        self.batch_size = batch_size    #Number of sequences in a batch
        
        # Load spectra data
        self.data_loader = DataLoader(file_path)
        self.spectra = self.data_loader.load_spectra()
        
        #The encoder
        self.emb_lstm = torch.nn.LSTM(
            input_size=n_chars, 
            hidden_size=hidden_size, 
            num_layers=hidden_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = bidirectional
        )

        self.latent_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*seq_len*2,lin_dim),
            torch.nn.ReLU()
        )
        
        self.latent_mean = torch.nn.Linear(lin_dim, emb_dim)
        self.latent_log_std = torch.nn.Linear(lin_dim, emb_dim)

        self.dec_lin = torch.nn.Sequential(
            torch.nn.Linear(emb_dim + 64, emb_dim),
            torch.nn.ReLU()
        )

        #the decoder
        self.dec_lstm = torch.nn.LSTM(
            input_size = 1,
            num_layers = hidden_layers,
            dropout = dropout,
            batch_first = True,
            bidirectional = bidirectional,
            hidden_size = hidden_size
        )

        self.dec_final = torch.nn.Linear(hidden_size*emb_dim*2, n_chars*seq_len)
        
        self.xavier_initialization()
    
    #endoder forward pass, takes one hot encoded sequences x and returns q(x|z)
    def encode(self, x):
        hidden, _ = self.emb_lstm(x.float()) # the _ contains unnecessary hidden and cell state info https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        hidden = self.latent_linear(torch.flatten(hidden, 1))
        z_mean = self.latent_mean(hidden)
        z_log_std = self.latent_log_std(hidden)
        return torch.distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
    
    #decoder forward pass, takes a latent sample z and returns x^hat encoded sequences
    def decode(self, z, c):
        z_c = torch.cat((z, c), dim=1)
        hidden = self.dec_lin(z_c)
        hidden, _ = self.dec_lstm(hidden.view(-1,self.emb_dim,1))
        out = self.dec_final(torch.flatten(hidden, 1))
        return out.view(-1,self.seq_len,self.n_chars)
    
    #reparameterization trick for backwards pass
    def reparametrize(self, dist):
        sample = dist.rsample()
        prior = torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        prior_sample = prior.sample()
        return sample, prior_sample, prior
    
    #full forward pass for entire model
    def forward(self, x):
        latent_dist = self.encode(x)   
        latent_sample, prior_sample, prior = self.reparametrize(latent_dist)
        c = self.spectra.to(x.device)
        output = self.decode(latent_sample, c).view(-1,self.seq_len,self.n_chars)
        return output, latent_dist, prior, latent_sample, prior_sample, latent_dist.loc, torch.log(latent_dist.scale)

    def mutual_information_loss(self, x_reconstructed, c, bins=30):
        x_reconstructed_flat = x_reconstructed.view(x_reconstructed.size(0), -1)
        c_flat = c.view(c.size(0), -1)
        
        joint_hist = torch.histc(x_reconstructed_flat + c_flat, bins=bins)
        x_hist = torch.histc(x_reconstructed_flat, bins=bins)
        c_hist = torch.histc(c_flat, bins=bins)
        
        p_joint = joint_hist / torch.sum(joint_hist)
        p_x = x_hist / torch.sum(x_hist)
        p_c = c_hist / torch.sum(c_hist)
        
        p_joint = p_joint[p_joint > 0]  # Avoid log(0)
        p_x = p_x[p_x > 0]
        p_c = p_c[p_c > 0]
        
        mi = torch.sum(p_joint * torch.log(p_joint / (p_x[:, None] * p_c[None, :])))
        
        return mi

    def __repr__(self):
        return 'SequenceVAE' + self.trainer_config
