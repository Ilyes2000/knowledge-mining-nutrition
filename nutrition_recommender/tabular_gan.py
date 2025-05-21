import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

class TabularGANAugmentor:
    def __init__(self, latent_dim=16, hidden_dim=64, epochs=100,
                 batch_size=256, checkpoint_interval=20):
        self.zd, self.hd = latent_dim, hidden_dim
        self.epochs, self.bs = epochs, batch_size
        self.ckpt_int = checkpoint_interval

    def _build(self, dim):
        class G(nn.Module):
            def __init__(self, zd, out, hd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(zd, hd), nn.ReLU(inplace=True),
                    nn.Linear(hd, hd), nn.ReLU(inplace=True),
                    nn.Linear(hd, out)
                )
            def forward(self,z): return self.net(z)

        class D(nn.Module):
            def __init__(self, inp, hd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(inp, hd), nn.LeakyReLU(0.2,inplace=True),
                    nn.Linear(hd, hd//2), nn.LeakyReLU(0.2,inplace=True),
                    nn.Linear(hd//2, 1)
                )
            def forward(self,x): return self.net(x).view(-1,1)

        self.G = G(self.zd, dim, self.hd)
        self.D = D(dim, self.hd)

    def fit(self, df):
        data = torch.tensor(df.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=self.bs, shuffle=True, drop_last=True)
        self._build(df.shape[1])
        optG = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5,0.999))
        optD = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5,0.999))
        loss_fn = nn.BCEWithLogitsLoss()

        for ep in range(1, self.epochs+1):
            ld_tot, lg_tot = 0., 0.
            for (real_batch,) in loader:
                b = real_batch.size(0)
                # Discriminateur
                optD.zero_grad()
                # réels
                lr = loss_fn(self.D(real_batch), torch.ones(b,1))
                # faux
                z  = torch.randn(b, self.zd)
                fake = self.G(z).detach()
                lf = loss_fn(self.D(fake), torch.zeros(b,1))
                ld = 0.5*(lr+lf)
                ld.backward(); optD.step()
                ld_tot += ld.item()
                # Générateur
                optG.zero_grad()
                z2  = torch.randn(b, self.zd)
                gen = self.G(z2)
                lg  = loss_fn(self.D(gen), torch.ones(b,1))
                lg.backward(); optG.step()
                lg_tot += lg.item()

            if ep%self.ckpt_int==0 or ep==self.epochs:
                print(f"Epoch {ep}/{self.epochs} | "
                      f"Loss_D: {ld_tot/len(loader):.4f} | "
                      f"Loss_G: {lg_tot/len(loader):.4f}")

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.zd)
            return self.G(z).cpu().numpy()
    def _save_checkpoint(self, epoch: int):
        gen_path = os.path.join(self.checkpoint_dir, f"generator_epoch{epoch}.pth")
        dis_path = os.path.join(self.checkpoint_dir, f"discriminator_epoch{epoch}.pth")
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), dis_path)
