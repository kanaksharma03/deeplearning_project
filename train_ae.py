"""Train the undercomplete autoencoder on clean MNIST (Step 2 of the experimental plan).

Outputs (in outputs/):
  ae_best.pt              - weights with the lowest validation loss
  ae_loss_curves.png      - training / validation MSE per epoch
  ae_reconstructions.png  - test images (top) and their reconstructions (bottom)
"""

import copy
import os
import time

import matplotlib
matplotlib.use("Agg")  # save figures to file without opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from mnist_data import get_dataloaders
from models import Autoencoder

LATENT_DIM = 16
LR = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5  # stop after this many epochs without a new best validation loss
SEED = 42
OUT_DIR = "outputs"


def run_epoch(model, loader, loss_fn, device, optimizer=None, corrupt=None, noise_seed=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.

    corrupt: optional fn(images, generator) -> noisy images. The model sees the
             corrupted batch as input, while the loss is always computed against the
             clean images (denoising objective). Noise is drawn fresh for every batch.
    noise_seed: if given, the noise is drawn from a generator seeded with it, so an
             evaluation pass sees exactly the same noisy images every epoch.
    Returns the mean per-pixel MSE over all images."""
    training = optimizer is not None
    model.train(training)
    generator = None
    if corrupt is not None and noise_seed is not None:
        generator = torch.Generator(device=device).manual_seed(noise_seed)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            inputs = images if corrupt is None else corrupt(images, generator)
            loss = loss_fn(model(inputs), images)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * images.size(0)
            count += images.size(0)
    return total / count


def train(model, train_loader, val_loader, device, corrupt=None, max_epochs=MAX_EPOCHS):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, waited = float("inf"), None, 0, 0

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer, corrupt)
        val_loss = run_epoch(model, val_loader, loss_fn, device, corrupt=corrupt, noise_seed=SEED)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
        print(f"epoch {epoch:2d}/{max_epochs}  train {train_loss:.5f}  val {val_loss:.5f}"
              f"  {time.time() - start:4.1f}s{'  *best' if improved else ''}")
        if waited >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs.")
            break

    model.load_state_dict(best_state)
    print(f"Restored best weights from epoch {best_epoch} (val MSE {best_val:.5f}).")
    return history, best_epoch


def plot_losses(history, best_epoch, path,
                title="Undercomplete autoencoder (d = 16): loss curves"):
    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train"], "o-", label="Training")
    plt.plot(epochs, history["val"], "s-", label="Validation")
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def plot_reconstructions(model, test_loader, device, path, n=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:n].to(device)
    recon = model(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2 + 1.2, 2.8))
    for row, (batch, title) in enumerate([(images, "Original"), (recon, "Reconstructed")]):
        for i in range(n):
            ax = axes[row, i]
            ax.imshow(batch[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(title, fontsize=9)
    fig.suptitle("Undercomplete autoencoder: test-set reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    model = Autoencoder(LATENT_DIM).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    history, best_epoch = train(model, train_loader, val_loader, device)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "ae_best.pt"))

    test_mse = run_epoch(model, test_loader, nn.MSELoss(), device)
    print(f"test MSE {test_mse:.5f}")

    plot_losses(history, best_epoch, os.path.join(OUT_DIR, "ae_loss_curves.png"))
    plot_reconstructions(model, test_loader, device, os.path.join(OUT_DIR, "ae_reconstructions.png"))
    print(f"saved figures and weights to {OUT_DIR}/")


if __name__ == "__main__":
    main()
