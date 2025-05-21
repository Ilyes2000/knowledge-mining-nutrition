import torch
import torch.nn as nn

class ConditionalVAE(nn.Module):
    """
    VAE conditionnel pour données tabulaires.
    """
    def __init__(self, input_dim, cond_dim, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_dec = nn.Linear(latent_dim + cond_dim, 128)
        self.fc_out = nn.Linear(128, input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x,c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        h = torch.relu(self.fc_dec(torch.cat([z,c], dim=1)))
        return self.fc_out(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparam(mu, logvar)
        return self.decode(z, c), mu, logvar
