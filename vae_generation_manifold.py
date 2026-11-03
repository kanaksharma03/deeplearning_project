"""Sample new digits from VAE prior and construct a 2D latent manifold grid.

This script:
  1. Generates new digit images by sampling z ~ N(0, I) from the trained d=16 VAE.
  2. Trains a separate 2D VAE (d=2) on MNIST.
  3. Constructs a continuous 2D latent manifold grid (20x20) demonstrating smooth
     morphing and continuous transitions across digit classes in latent space.
  4. Plots the 2D latent scatter of the test set for d=2, directly linking the
     coordinate space to the manifold grid.

Outputs (in outputs/):
  outputs/vae_generated_samples_d16.png  - Novel digits sampled from d=16 prior
  outputs/vae_d2_best.pt                 - Weights of the trained d=2 VAE
  outputs/vae_generated_samples_d2.png   - Novel digits sampled from d=2 prior
  outputs/vae_latent_manifold_2d.png     - 20x20 continuous latent space manifold grid
  outputs/vae_d2_latent_scatter.png      - Test set directly in 2D latent space
"""

import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri
import torch
import torch.nn.functional as F

from mnist_data import get_dataloaders
from models import VAE

OUT_DIR = "outputs"
SEED = 42
LR = 1e-3
BETA = 1.0
D2_EPOCHS = 35
D2_PATIENCE = 7
N_PIXELS = 28 * 28

DIGIT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader` for training or validation."""
    training = optimizer is not None
    model.train(training)
    total_loss, recon_sum, kl_sum, count = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for images, _ in loader:
            x = images.to(device, non_blocking=True)
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar) if training else mu
            x_hat = model.decoder(z)

            recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = recon + BETA * kl

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = x.size(0)
            total_loss += loss.item() * n
            recon_sum += recon.item() * n
            kl_sum += kl.item() * n
            count += n

    return {
        "loss": total_loss / count,
        "recon": recon_sum / count,
        "kl": kl_sum / count
    }


def train_d2_vae(train_loader, val_loader, device, checkpoint_path):
    """Train a VAE with latent_dim=2."""
    print("\n" + "=" * 60)
    print("Training 2D Latent VAE (d = 2) for Manifold Visualization")
    print("=" * 60)

    model = VAE(latent_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    waited = 0

    for epoch in range(1, D2_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)

        improved = va["loss"] < best_val_loss
        if improved:
            best_val_loss = va["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
        else:
            waited += 1

        print(f"epoch {epoch:2d}/{D2_EPOCHS}  train {tr['loss']:6.2f} (rec {tr['recon']:5.2f} kl {tr['kl']:5.2f})"
              f"  val {va['loss']:6.2f} (rec {va['recon']:5.2f} kl {va['kl']:5.2f})"
              f"  {time.time() - t0:4.1f}s{'  *best' if improved else ''}")

        if waited >= D2_PATIENCE:
            print(f"Early stopping triggered after {waited} epochs without improvement.")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
    print(f"Restored and saved best d=2 VAE weights from epoch {best_epoch} (val loss {best_val_loss:.2f}) to {checkpoint_path}")
    return model


@torch.no_grad()
def generate_random_samples(model, latent_dim, device, out_path, num_rows=8, num_cols=8, seed=SEED):
    """Sample z ~ N(0, I) and decode to synthesize new digit images."""
    model.eval()
    total_samples = num_rows * num_cols
    torch.manual_seed(seed)

    # Sample z ~ N(0, I)
    z = torch.randn(total_samples, latent_dim, device=device)
    generated = model.decoder(z).cpu()  # shape (N, 1, 28, 28)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.1, num_rows * 1.1))
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            ax = axes[i, j]
            ax.imshow(generated[idx, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.suptitle(f"Newly Generated Digits Sampled from VAE Prior z ~ N(0, I)  (d = {latent_dim})",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved generated samples grid to: {out_path}")


@torch.no_grad()
def plot_latent_manifold_2d(model, device, out_path, grid_size=20):
    """Create a 2D continuous latent manifold grid using inverse Gaussian CDF."""
    model.eval()

    # Linearly spaced percentiles between 2% and 98% mapped via inverse Gaussian CDF (probit)
    # This matches the unit Gaussian prior density p(z) ~ N(0, I)
    percentiles = np.linspace(0.02, 0.98, grid_size)
    z1_vals = ndtri(percentiles)
    z2_vals = ndtri(percentiles)

    canvas = np.zeros((grid_size * 28, grid_size * 28), dtype=np.float32)

    # Grid iteration: z2 along vertical axis (inverted so positive is top), z1 along horizontal
    for i, z2 in enumerate(reversed(z2_vals)):
        for j, z1 in enumerate(z1_vals):
            z = torch.tensor([[z1, z2]], dtype=torch.float32, device=device)
            digit = model.decoder(z).squeeze().cpu().numpy()
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray", origin="upper", vmin=0, vmax=1)
    
    # Tick marks showing latent coordinates
    tick_indices = np.linspace(14, grid_size * 28 - 14, 7)
    tick_labels = [f"{v:.1f}" for v in np.linspace(z1_vals[0], z1_vals[-1], 7)]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(list(reversed(tick_labels)), fontsize=10)

    ax.set_xlabel("Latent Variable $z_1$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Latent Variable $z_2$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title("Learned 2D Latent Manifold Grid ($d = 2$ VAE)\nSmooth Semantic Transitions Across Hand-Written Digits",
                 fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved 2D latent manifold grid to: {out_path}")


@torch.no_grad()
def plot_d2_latent_scatter(model, test_loader, device, out_path):
    """Plot test set latent coordinates (mu_1, mu_2) colored by digit class."""
    model.eval()
    all_mus, all_labels = [], []

    for images, labels in test_loader:
        images = images.to(device)
        mu, _ = model.encode(images)
        all_mus.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())

    mus = np.concatenate(all_mus, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    plt.figure(figsize=(9, 7.5))
    for digit in range(10):
        mask = (labels == digit)
        plt.scatter(
            mus[mask, 0],
            mus[mask, 1],
            c=DIGIT_COLORS[digit],
            label=str(digit),
            alpha=0.6,
            s=12,
            edgecolors="none"
        )

    plt.xlabel("Latent Variable $z_1$", fontsize=12, fontweight="bold")
    plt.ylabel("Latent Variable $z_2$", fontsize=12, fontweight="bold")
    plt.title("Test Set Encoded in Direct 2D Latent Space ($d = 2$ VAE)", fontsize=14, fontweight="bold", pad=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Digit Class", fontsize=10, title_fontsize=11, markerscale=2.5, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved d=2 latent scatter plot to: {out_path}")


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()

    # 1. Sample z ~ N(0, I) from trained d=16 VAE
    vae_16_path = os.path.join(OUT_DIR, "vae_best.pt")
    if os.path.exists(vae_16_path):
        print("\nLoading trained d=16 VAE for prior sampling...")
        vae_16 = VAE(latent_dim=16).to(device)
        vae_16.load_state_dict(torch.load(vae_16_path, map_location=device))
        samples_d16_path = os.path.join(OUT_DIR, "vae_generated_samples_d16.png")
        generate_random_samples(vae_16, 16, device, samples_d16_path, num_rows=8, num_cols=8)
    else:
        print(f"Warning: {vae_16_path} not found. Skipping d=16 sample generation.")

    # 2. Train or load d=2 VAE
    vae_d2_path = os.path.join(OUT_DIR, "vae_d2_best.pt")
    if os.path.exists(vae_d2_path):
        print(f"\nFound existing d=2 checkpoint at {vae_d2_path}. Loading...")
        vae_d2 = VAE(latent_dim=2).to(device)
        vae_d2.load_state_dict(torch.load(vae_d2_path, map_location=device))
    else:
        vae_d2 = train_d2_vae(train_loader, val_loader, device, vae_d2_path)

    # 3. Sample novel digits from d=2 VAE prior
    samples_d2_path = os.path.join(OUT_DIR, "vae_generated_samples_d2.png")
    generate_random_samples(vae_d2, 2, device, samples_d2_path, num_rows=8, num_cols=8)

    # 4. Generate 2D continuous latent manifold grid (20x20)
    manifold_path = os.path.join(OUT_DIR, "vae_latent_manifold_2d.png")
    plot_latent_manifold_2d(vae_d2, device, manifold_path, grid_size=20)

    # 5. Scatter plot of test set in direct 2D latent space
    scatter_d2_path = os.path.join(OUT_DIR, "vae_d2_latent_scatter.png")
    plot_d2_latent_scatter(vae_d2, test_loader, device, scatter_d2_path)

    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print(f"1. d=16 Prior Samples: {samples_d16_path}")
    print(f"2. d=2 Model Weights : {vae_d2_path}")
    print(f"3. d=2 Prior Samples : {samples_d2_path}")
    print(f"4. 2D Manifold Grid  : {manifold_path}")
    print(f"5. 2D Latent Scatter : {scatter_d2_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
