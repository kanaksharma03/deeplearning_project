"""Train the variational autoencoder (Step 4 of the experimental plan).

Loss per image:  L = recon + beta * KL
  recon = sum over the 784 pixels of (x - x_hat)^2        (squared-error / Gaussian likelihood)
  KL    = -1/2 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)   (closed form, q(z|x) vs N(0, I))
Both terms are summed per image and averaged over the batch. Using a per-pixel *mean*
for the reconstruction term instead would make it ~784x smaller than the KL term and
the model would collapse to the prior.

Posterior-collapse monitoring (every epoch, on the validation set):
  - KL per latent dimension (a collapsed dimension has KL ~ 0: q(z_j|x) = N(0, 1) for every x)
  - active units: dimensions whose mean mu_j varies across inputs, Var_x[mu_j] > 0.01
    (Burda et al., 2016). Collapse = few or no active units and total KL near 0.

Outputs (in outputs/):
  vae_best.pt                - weights with the lowest validation loss
  vae_loss_curves.png        - total / reconstruction / KL loss and active units per epoch
  vae_latent_kl.png          - KL per latent dimension at the best epoch
  vae_reconstructions.png    - test images, VAE reconstructions (z = mu), plain-AE reconstructions
"""

import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from mnist_data import get_dataloaders
from models import VAE, Autoencoder
from train_ae import LR, OUT_DIR, SEED

LATENT_DIM = 16
BETA = 1.0
MIN_EPOCHS = 30
MAX_EPOCHS = 100
PATIENCE = 10
ACTIVE_THRESHOLD = 0.01
N_PIXELS = 28 * 28


def kl_per_dim(mu, logvar):
    """Closed-form KL(N(mu, sigma^2) || N(0, 1)) for each image and latent dimension."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader` (training if an optimizer is given). z is always sampled
    with the reparameterisation trick so the validation loss is the same objective.

    Returns a dict of per-image averages (total, recon, kl, mse per pixel), the KL of
    each latent dimension and the number of active units."""
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    kl_dims = torch.zeros(model.latent_dim, device=device)
    mus = []
    count = 0
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            x = images.to(device, non_blocking=True)
            mu, logvar = model.encode(x)
            x_hat = model.decoder(model.reparameterize(mu, logvar))

            recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
            kl_d = kl_per_dim(mu, logvar)
            kl = kl_d.sum(dim=1).mean()
            loss = recon + BETA * kl
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = x.size(0)
            sums["total"] += loss.item() * n
            sums["recon"] += recon.item() * n
            sums["kl"] += kl.item() * n
            kl_dims += kl_d.detach().sum(dim=0)
            mus.append(mu.detach())
            count += n

    stats = {k: v / count for k, v in sums.items()}
    stats["mse"] = stats["recon"] / N_PIXELS
    stats["kl_dims"] = (kl_dims / count).cpu()
    stats["active"] = int((torch.cat(mus).var(dim=0) > ACTIVE_THRESHOLD).sum())
    return stats


def train(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {f"{split}_{k}": [] for split in ("train", "val")
               for k in ("total", "recon", "kl", "active")}
    best_val, best_state, best_epoch, best_stats, waited = float("inf"), None, 0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        for split, s in (("train", tr), ("val", va)):
            for k in ("total", "recon", "kl", "active"):
                history[f"{split}_{k}"].append(s[k])

        improved = va["total"] < best_val
        if improved:
            best_val, best_epoch, best_stats, waited = va["total"], epoch, va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
        print(f"epoch {epoch:3d}  train {tr['total']:6.2f} (rec {tr['recon']:5.2f} kl {tr['kl']:5.2f})"
              f"  val {va['total']:6.2f} (rec {va['recon']:5.2f} kl {va['kl']:5.2f})"
              f"  active {va['active']:2d}/{LATENT_DIM}  {time.time() - start:4.1f}s"
              f"{'  *best' if improved else ''}")
        if va["kl"] < 0.5 or va["active"] <= 2:
            print("  WARNING: possible posterior collapse (KL near 0 / few active units)")
        if epoch >= MIN_EPOCHS and waited >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs.")
            break

    model.load_state_dict(best_state)
    print(f"Restored best weights from epoch {best_epoch} (val loss {best_val:.2f}).")
    return history, best_epoch, best_stats


def plot_losses(history, best_epoch, path):
    epochs = range(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [("total", f"Total loss (recon + {BETA:g}·KL)"),
              ("recon", "Reconstruction (sum of squared errors per image)"),
              ("kl", "KL divergence per image (nats)"),
              ("active", f"Active latent units (Var[μ] > {ACTIVE_THRESHOLD})")]
    for ax, (key, title) in zip(axes.flat, panels):
        if key == "active":
            ax.plot(epochs, history["val_active"], "s-", color="C1", label="Validation")
            ax.set_ylim(0, LATENT_DIM + 1)
        else:
            ax.plot(epochs, history[f"train_{key}"], "o-", ms=3, label="Training")
            ax.plot(epochs, history[f"val_{key}"], "s-", ms=3, label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Variational autoencoder (d = {LATENT_DIM}, β = {BETA:g})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_kl_dims(kl_dims, active_var, path):
    dims = range(LATENT_DIM)
    colors = ["C0" if v > ACTIVE_THRESHOLD else "C3" for v in active_var]
    plt.figure(figsize=(8, 4))
    plt.bar(dims, kl_dims, color=colors)
    plt.xticks(dims)
    plt.xlabel("Latent dimension j")
    plt.ylabel("Mean KL (nats)")
    plt.title("KL per latent dimension, validation set (red = inactive / collapsed)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate_test(model, test_loader, device):
    """Per-image MSE / PSNR of reconstructions decoded from z = mu."""
    model.eval()
    mses = []
    for images, _ in test_loader:
        x = images.to(device)
        x_hat, _, _ = model(x)
        mses.append(((x_hat - x) ** 2).flatten(1).mean(dim=1))
    mse = torch.cat(mses)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return mse, psnr


@torch.no_grad()
def plot_reconstructions(model, ae, test_loader, device, path, n=10):
    model.eval()
    x = next(iter(test_loader))[0][:n].to(device)
    rows = [("Original", x), ("VAE (z = μ)", model(x)[0])]
    if ae is not None:
        rows.append(("Plain AE", ae.eval()(x)))
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2 + 1.4, 1.35 * len(rows) + 0.5))
    for r, (title, batch) in enumerate(rows):
        for i in range(n):
            ax = axes[r, i]
            ax.imshow(batch[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(title, fontsize=8)
    fig.suptitle("Variational autoencoder: test-set reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    model = VAE(LATENT_DIM).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  beta={BETA}  "
          f"epochs {MIN_EPOCHS}-{MAX_EPOCHS}, patience {PATIENCE}")

    history, best_epoch, best_stats = train(model, train_loader, val_loader, device)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "vae_best.pt"))

    # Collapse check at the best epoch: per-dimension KL and variance of mu over the val set.
    model.eval()
    with torch.no_grad():
        mu = torch.cat([model.encode(x.to(device))[0] for x, _ in val_loader])
    active_var = mu.var(dim=0).cpu()
    kl_dims = best_stats["kl_dims"]
    print("\nPosterior-collapse check (validation set, best epoch):")
    print("  dim      :", " ".join(f"{j:5d}" for j in range(LATENT_DIM)))
    print("  KL (nats):", " ".join(f"{v:5.2f}" for v in kl_dims))
    print("  Var[mu]  :", " ".join(f"{v:5.2f}" for v in active_var))
    n_active = int((active_var > ACTIVE_THRESHOLD).sum())
    print(f"  active units: {n_active}/{LATENT_DIM}, total KL {kl_dims.sum():.2f} nats"
          f" -> {'no posterior collapse' if n_active > 2 and kl_dims.sum() > 0.5 else 'COLLAPSE'}")

    mse, psnr = evaluate_test(model, test_loader, device)
    print(f"\nTest set, reconstruction from z = mu: MSE {mse.mean():.5f} ± {mse.std():.5f}"
          f", PSNR {psnr.mean():.2f} ± {psnr.std():.2f} dB")

    ae = None
    ae_path = os.path.join(OUT_DIR, "ae_best.pt")
    if os.path.exists(ae_path):
        ae = Autoencoder(LATENT_DIM).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device))

    plot_losses(history, best_epoch, os.path.join(OUT_DIR, "vae_loss_curves.png"))
    plot_kl_dims(kl_dims, active_var, os.path.join(OUT_DIR, "vae_latent_kl.png"))
    plot_reconstructions(model, ae, test_loader, device,
                         os.path.join(OUT_DIR, "vae_reconstructions.png"))
    print(f"saved figures and weights to {OUT_DIR}/")


if __name__ == "__main__":
    main()
