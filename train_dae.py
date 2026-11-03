"""Train the denoising autoencoder (Step 3 of the experimental plan).

Same backbone as the undercomplete autoencoder (models.Autoencoder, d = 16) and the
same training loop (train_ae.train), but every batch is corrupted on the fly with
Gaussian noise (sigma = 0.3) before it goes into the network, while the loss is
computed against the clean image.

Evaluation: the test set is corrupted once with a fixed seed, and reconstructions of
those noisy images are compared with the clean ground truth (per-image MSE and PSNR).
For reference the same numbers are reported for the noisy input itself and, if
outputs/ae_best.pt exists, for the plain autoencoder given the same noisy images.

Outputs (in outputs/):
  dae_best.pt            - weights with the lowest validation loss
  dae_loss_curves.png    - training / validation MSE per epoch
  dae_denoising.png      - clean / noisy / plain-AE output / DAE output for test images
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mnist_data import add_noise, get_dataloaders
from models import Autoencoder
from train_ae import LATENT_DIM, OUT_DIR, SEED, plot_losses, train

NOISE_TYPE = "gaussian"
SIGMA = 0.3
EPOCHS = 30
TEST_NOISE_SEED = 123


def corrupt(images, generator=None):
    return add_noise(images, NOISE_TYPE, SIGMA, generator)


def per_image_metrics(output, clean):
    """Per-image MSE and PSNR (dB, MAX_I = 1.0) for batches of shape (N, 1, 28, 28)."""
    mse = ((output - clean) ** 2).flatten(1).mean(dim=1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return mse, psnr


@torch.no_grad()
def evaluate(models, test_loader, device):
    """Corrupt the test set with a fixed seed, run each model on the noisy images and
    return {name: (mse per image, psnr per image)} against the clean images."""
    for m in models.values():
        m.eval()
    generator = torch.Generator(device=device).manual_seed(TEST_NOISE_SEED)
    results = {name: ([], []) for name in ["Noisy input", *models]}
    for images, _ in test_loader:
        clean = images.to(device)
        noisy = corrupt(clean, generator)
        outputs = {"Noisy input": noisy, **{name: m(noisy) for name, m in models.items()}}
        for name, out in outputs.items():
            mse, psnr = per_image_metrics(out, clean)
            results[name][0].append(mse)
            results[name][1].append(psnr)
    return {name: (torch.cat(m), torch.cat(p)) for name, (m, p) in results.items()}


@torch.no_grad()
def plot_denoising(models, test_loader, device, path, n=10):
    clean = next(iter(test_loader))[0][:n].to(device)
    noisy = corrupt(clean, torch.Generator(device=device).manual_seed(TEST_NOISE_SEED))
    rows = [("Clean", clean), (f"Noisy (σ={SIGMA})", noisy)]
    rows += [(name, m.eval()(noisy)) for name, m in models.items()]
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
    fig.suptitle("Denoising noisy test images (Gaussian noise, σ = 0.3)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    dae = Autoencoder(LATENT_DIM).to(device)
    print(f"training DAE: {NOISE_TYPE} noise, sigma={SIGMA}, up to {EPOCHS} epochs")
    history, best_epoch = train(dae, train_loader, val_loader, device,
                                corrupt=corrupt, max_epochs=EPOCHS)
    torch.save(dae.state_dict(), os.path.join(OUT_DIR, "dae_best.pt"))
    plot_losses(history, best_epoch, os.path.join(OUT_DIR, "dae_loss_curves.png"),
                title=f"Denoising autoencoder (d = 16, Gaussian σ = {SIGMA}): loss curves")

    models = {}
    ae_path = os.path.join(OUT_DIR, "ae_best.pt")
    if os.path.exists(ae_path):
        ae = Autoencoder(LATENT_DIM).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        models["Plain AE"] = ae
    models["Denoising AE"] = dae

    results = evaluate(models, test_loader, device)
    print(f"\nTest set ({len(test_loader.dataset)} images), noisy input vs clean ground truth:")
    print(f"{'':16}{'MSE (mean ± std)':>24}{'PSNR dB (mean ± std)':>26}")
    for name, (mse, psnr) in results.items():
        print(f"{name:16}{mse.mean():>13.5f} ± {mse.std():.5f}"
              f"{psnr.mean():>15.2f} ± {psnr.std():.2f}")

    plot_denoising(models, test_loader, device, os.path.join(OUT_DIR, "dae_denoising.png"))
    print(f"\nsaved figures and weights to {OUT_DIR}/")


if __name__ == "__main__":
    main()
