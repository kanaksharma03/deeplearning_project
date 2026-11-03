"""Extract latent codes from the test set for each model and project to 2D using PCA and t-SNE.

Models evaluated:
  1. Plain Undercomplete Autoencoder (AE, d = 16)
  2. Denoising Autoencoder (DAE, d = 16)
  3. Variational Autoencoder (VAE, d = 16, z = mu)

Generates:
  outputs/latent_pca_comparison.png         - Side-by-side PCA projections
  outputs/latent_tsne_comparison.png        - Side-by-side t-SNE projections
  outputs/latent_projections_comparison.png - 2x3 grid (PCA top row, t-SNE bottom row)
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mnist_data import get_dataloaders
from models import Autoencoder, VAE

OUT_DIR = "outputs"
LATENT_DIM = 16
SEED = 42

# 10 distinct, accessible colors for digits 0-9
DIGIT_COLORS = [
    "#1f77b4",  # 0: blue
    "#ff7f0e",  # 1: orange
    "#2ca02c",  # 2: green
    "#d62728",  # 3: red
    "#9467bd",  # 4: purple
    "#8c564b",  # 5: brown
    "#e377c2",  # 6: pink
    "#7f7f7f",  # 7: gray
    "#bcbd22",  # 8: olive
    "#17becf",  # 9: cyan
]


@torch.no_grad()
def extract_latents(model, test_loader, device, is_vae=False, max_samples=None):
    """Extract latent representations and true labels from the test loader.

    For VAE, the mean vector mu(x) is used as the deterministic representation.
    """
    model.eval()
    latents, labels = [], []
    collected = 0

    for images, targets in test_loader:
        images = images.to(device)
        if is_vae:
            mu, _ = model.encode(images)
            z = mu
        else:
            z = model.encode(images)

        latents.append(z.cpu().numpy())
        labels.append(targets.numpy())
        collected += images.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    if max_samples is not None and len(latents) > max_samples:
        latents = latents[:max_samples]
        labels = labels[:max_samples]

    return latents, labels


def project_pca(latents, random_state=SEED):
    """Project latent vectors to 2D with PCA."""
    pca = PCA(n_components=2, random_state=random_state)
    z_pca = pca.fit_transform(latents)
    var_ratio = pca.explained_variance_ratio_
    return z_pca, var_ratio


def project_tsne(latents, perplexity=30.0, random_state=SEED):
    """Project latent vectors to 2D with t-SNE."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        n_jobs=-1
    )
    z_tsne = tsne.fit_transform(latents)
    return z_tsne


def plot_side_by_side(projections_dict, labels, method_name, out_path, extra_info=None):
    """Plot 3 models side by side for a given projection method (PCA or t-SNE)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=False, sharey=False)

    for idx, (model_name, coords) in enumerate(projections_dict.items()):
        ax = axes[idx]
        for digit in range(10):
            mask = (labels == digit)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=DIGIT_COLORS[digit],
                label=str(digit),
                alpha=0.55,
                s=12,
                edgecolors="none"
            )

        subtitle = f"{model_name}"
        if extra_info and model_name in extra_info:
            subtitle += f" ({extra_info[model_name]})"
        ax.set_title(subtitle, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel(f"{method_name} Component 1", fontsize=11)
        ax.set_ylabel(f"{method_name} Component 2", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=9)

    # Shared legend on right
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        title="Digit Class",
        title_fontsize=11,
        fontsize=10,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        markerscale=2.5,
        frameon=True,
        framealpha=0.9
    )

    plt.suptitle(
        f"2D Latent Space Projection ({method_name}) Across Autoencoder Architectures (d = {LATENT_DIM})",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0.02, 0.03, 0.92, 0.93])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {method_name} side-by-side plot to: {out_path}")


def plot_combined_grid(pca_dict, tsne_dict, labels, pca_var_info, out_path):
    """Plot a 2x3 grid: Top row = PCA, Bottom row = t-SNE across Plain AE, DAE, and VAE."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False, sharey=False)
    models = list(pca_dict.keys())

    for col, model_name in enumerate(models):
        # Row 0: PCA
        ax_pca = axes[0, col]
        pca_coords = pca_dict[model_name]
        for digit in range(10):
            mask = (labels == digit)
            ax_pca.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=DIGIT_COLORS[digit],
                label=str(digit),
                alpha=0.55,
                s=10,
                edgecolors="none"
            )
        var_str = pca_var_info.get(model_name, "")
        ax_pca.set_title(f"PCA: {model_name}\n({var_str})", fontsize=12, fontweight="bold")
        ax_pca.set_xlabel("PC 1", fontsize=10)
        ax_pca.set_ylabel("PC 2", fontsize=10)
        ax_pca.grid(True, linestyle="--", alpha=0.3)
        ax_pca.tick_params(labelsize=8)

        # Row 1: t-SNE
        ax_tsne = axes[1, col]
        tsne_coords = tsne_dict[model_name]
        for digit in range(10):
            mask = (labels == digit)
            ax_tsne.scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=DIGIT_COLORS[digit],
                label=str(digit),
                alpha=0.55,
                s=10,
                edgecolors="none"
            )
        ax_tsne.set_title(f"t-SNE: {model_name}", fontsize=12, fontweight="bold")
        ax_tsne.set_xlabel("t-SNE 1", fontsize=10)
        ax_tsne.set_ylabel("t-SNE 2", fontsize=10)
        ax_tsne.grid(True, linestyle="--", alpha=0.3)
        ax_tsne.tick_params(labelsize=8)

    # Shared legend on right
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        title="Digit Class",
        title_fontsize=12,
        fontsize=11,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        markerscale=2.5,
        frameon=True,
        framealpha=0.9
    )

    plt.suptitle(
        f"Comparative 2D Latent Space Projections (PCA vs t-SNE, d = {LATENT_DIM})",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0.02, 0.02, 0.92, 0.94])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved combined PCA/t-SNE 2x3 grid to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract latent codes and plot 2D PCA & t-SNE projections.")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of test samples to project (default: 5000 for clear visualization, max: 10000)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE (default: 30.0)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    _, _, test_loader = get_dataloaders()
    n_samples = args.samples
    print(f"Extracting latents for {n_samples} test images...")

    # Load models
    models_to_eval = [
        ("Undercomplete AE", Autoencoder(LATENT_DIM), os.path.join(OUT_DIR, "ae_best.pt"), False),
        ("Denoising AE", Autoencoder(LATENT_DIM), os.path.join(OUT_DIR, "dae_best.pt"), False),
        ("Variational AE", VAE(LATENT_DIM), os.path.join(OUT_DIR, "vae_best.pt"), True),
    ]

    latents_by_model = {}
    test_labels = None

    for name, model_cls, checkpoint_path, is_vae in models_to_eval:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")

        model = model_cls.to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded weights for {name} from {checkpoint_path}")

        t0 = time.time()
        z, y = extract_latents(model, test_loader, device, is_vae=is_vae, max_samples=n_samples)
        print(f"  Extracted latent matrix shape: {z.shape} in {time.time() - t0:.2f}s")

        latents_by_model[name] = z
        if test_labels is None:
            test_labels = y

    # Compute PCA projections
    print("\nComputing PCA projections...")
    pca_projections = {}
    pca_var_info = {}
    for name, z in latents_by_model.items():
        t0 = time.time()
        z_pca, var_ratio = project_pca(z, random_state=SEED)
        pca_projections[name] = z_pca
        var_pct = var_ratio * 100
        var_info = f"PC1: {var_pct[0]:.1f}%, PC2: {var_pct[1]:.1f}%, Total: {var_pct.sum():.1f}%"
        pca_var_info[name] = var_info
        print(f"  {name:16}: Explained variance = {var_info} ({time.time() - t0:.2f}s)")

    # Compute t-SNE projections
    print("\nComputing t-SNE projections (perplexity={args.perplexity})...")
    tsne_projections = {}
    for name, z in latents_by_model.items():
        t0 = time.time()
        z_tsne = project_tsne(z, perplexity=args.perplexity, random_state=SEED)
        tsne_projections[name] = z_tsne
        print(f"  {name:16}: t-SNE completed in {time.time() - t0:.2f}s")

    # Generate plots
    print("\nRendering visualization plots...")
    pca_out = os.path.join(OUT_DIR, "latent_pca_comparison.png")
    tsne_out = os.path.join(OUT_DIR, "latent_tsne_comparison.png")
    grid_out = os.path.join(OUT_DIR, "latent_projections_comparison.png")

    plot_side_by_side(pca_projections, test_labels, "PCA", pca_out, extra_info=pca_var_info)
    plot_side_by_side(tsne_projections, test_labels, "t-SNE", tsne_out)
    plot_combined_grid(pca_projections, tsne_projections, test_labels, pca_var_info, grid_out)

    print("\nAll latent projections successfully generated and saved to outputs/!")


if __name__ == "__main__":
    main()
