import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

import torch.nn as nn


__all__ = ['Encoder','Decoder','Autoencoder','VariationalEncoder','VariationalAutoencoder']

class Encoder(nn.Module):
    def __init__(self, in_dims, latent_dims):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dims, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dims),
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.linear1(x))
#         return self.linear2(x)
        x = self.model(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_dims,latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, in_dims)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z

    
class Autoencoder(nn.Module):
    def __init__(self,in_dims, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_dims,latent_dims)
        self.decoder = Decoder(in_dims,latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)   

class VariationalEncoder(nn.Module):
    def __init__(self, in_dims,latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(in_dims, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
#         z = mu + sigma*self.N.sample(mu.shape)
        z = mu + sigma*self.N.sample(mu.shape).to(x.device) # no need to hack
        z = mu + sigma*torch.randn(mu.shape, device=x.device) # faster
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, in_dims,latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_dims,latent_dims)
        self.decoder = Decoder(in_dims,latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)