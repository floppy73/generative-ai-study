import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, img_size, latent_size):
        super().__init__()
        self.img_size = img_size
        self.latent_size = latent_size

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(2),  # 16*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(2),  # 8*8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(2),  # 4*4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(128, latent_size)
        self.sigma_layer = nn.Linear(128, latent_size)

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu_layer(x)
        sigma_log = self.sigma_layer(x)
        sigma = torch.exp(0.5 * sigma_log)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_size, img_size):
        super().__init__()
        self.latent_size = latent_size
        self.img_size = img_size

        self.layers = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)

        return x
    

class VAE(nn.Module):
    def __init__(self, img_size, latent_size):
        super().__init__()
        self.encoder = Encoder(img_size, latent_size)
        self.decoder = Decoder(latent_size, img_size)
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        
        return(L1 + L2) / batch_size 


def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z