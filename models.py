"""Convolutional autoencoder models (Section 4.1 of the research paper).

Shape flow for a 28x28 MNIST image:
  encoder:  1x28x28 -> 32x14x14 -> 64x7x7 -> 128x4x4 -> flatten (2048) -> FC -> z (d)
  decoder:  z (d) -> FC -> 128x4x4 -> 64x7x7 -> 32x14x14 -> 1x28x28 (Sigmoid)
"""

import torch
import torch.nn as nn

FEATURE_SHAPE = (128, 4, 4)
FEATURE_SIZE = 128 * 4 * 4


def conv_block(in_ch, out_ch):
    """Stride-2 Conv2D + BatchNorm + ReLU: halves the spatial size (rounding up)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def deconv_block(in_ch, out_ch, output_padding):
    """Stride-2 ConvTranspose2D + BatchNorm + ReLU: roughly doubles the spatial size."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                           output_padding=output_padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def conv_stack():
    """Conv 32 -> 64 -> 128 (stride 2 each), flattened to FEATURE_SIZE features."""
    return nn.Sequential(
        conv_block(1, 32),    # 28 -> 14
        conv_block(32, 64),   # 14 -> 7
        conv_block(64, 128),  # 7  -> 4
        nn.Flatten(),
    )


class Encoder(nn.Module):
    """Conv stack followed by an FC layer to `out_dim`."""

    def __init__(self, out_dim):
        super().__init__()
        self.conv = conv_stack()
        self.fc = nn.Linear(FEATURE_SIZE, out_dim)

    def forward(self, x):
        return self.fc(self.conv(x))


class Decoder(nn.Module):
    """FC from the latent code, then ConvTranspose 128 -> 64 -> 32 -> 1 with a Sigmoid."""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, FEATURE_SIZE), nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(
            deconv_block(128, 64, output_padding=0),  # 4  -> 7
            deconv_block(64, 32, output_padding=1),   # 7  -> 14
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1,
                               output_padding=1),     # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, *FEATURE_SHAPE))


class Autoencoder(nn.Module):
    """Undercomplete autoencoder: deterministic latent code of size `latent_dim`."""

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    """Variational autoencoder: the bottleneck branches into two FC heads giving
    mu(x) and log sigma^2(x); z is sampled with the reparameterisation trick."""

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = conv_stack()
        self.fc_mu = nn.Linear(FEATURE_SIZE, latent_dim)
        self.fc_logvar = nn.Linear(FEATURE_SIZE, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        """Return (mu, logvar) of q(z|x)."""
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I), so gradients flow through mu and sigma."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decoder(z), mu, logvar
