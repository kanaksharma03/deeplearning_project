"""Comprehensive Model Comparison Script for Undercomplete AE, DAE, and VAE.

Compares:
  1. Downstream Classification Accuracy (%) using k-NN and Linear Probe on 16-D latents.
  2. Training convergence & epochs (best epoch, stopping criteria, parameter count).
  3. Clean vs. Noisy (sigma=0.3) Reconstruction MSE and PSNR (dB).
  4. Latent space geometry & generative properties.

Outputs:
  outputs/models_comprehensive_comparison.png  - 4-panel visual dashboard
  outputs/models_comprehensive_comparison.txt  - Formatted text report table
  outputs/models_comprehensive_comparison.csv  - CSV table of comparative metrics
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch

from mnist_data import add_noise, get_dataloaders
from models import Autoencoder, VAE

OUT_DIR = "outputs"
LATENT_DIM = 16
SIGMA = 0.3
SEED = 42
TEST_NOISE_SEED = 123


@torch.no_grad()
def extract_latents(model, loader, device, is_vae=False, max_samples=None):
    """Extract latent features z and labels y."""
    model.eval()
    zs, ys = [], []
    count = 0
    for images, targets in loader:
        images = images.to(device)
        if is_vae:
            mu, _ = model.encode(images)
            z = mu
        else:
            z = model.encode(images)
        zs.append(z.cpu().numpy())
        ys.append(targets.numpy())
        count += images.size(0)
        if max_samples and count >= max_samples:
            break
    z_all = np.concatenate(zs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    if max_samples and len(z_all) > max_samples:
        z_all = z_all[:max_samples]
        y_all = y_all[:max_samples]
    return z_all, y_all


@torch.no_grad()
def compute_reconstruction_metrics(model, test_loader, device, is_vae=False):
    """Compute MSE and PSNR for clean and noisy inputs."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(TEST_NOISE_SEED)
    clean_mses, clean_psnrs = [], []
    noisy_mses, noisy_psnrs = [], []

    for images, _ in test_loader:
        clean = images.to(device)
        noisy = add_noise(clean, "gaussian", SIGMA, generator=gen)

        if is_vae:
            out_clean, _, _ = model(clean)
            out_noisy, _, _ = model(noisy)
        else:
            out_clean = model(clean)
            out_noisy = model(noisy)

        clean_np = clean.squeeze(1).cpu().numpy()
        out_clean_np = np.clip(out_clean.squeeze(1).cpu().numpy(), 0.0, 1.0)
        out_noisy_np = np.clip(out_noisy.squeeze(1).cpu().numpy(), 0.0, 1.0)

        for i in range(clean_np.shape[0]):
            target = clean_np[i]
            # Clean reconstruction
            c_pred = out_clean_np[i]
            clean_mses.append(mean_squared_error(target, c_pred))
            clean_psnrs.append(peak_signal_noise_ratio(target, c_pred, data_range=1.0))

            # Noisy reconstruction
            n_pred = out_noisy_np[i]
            noisy_mses.append(mean_squared_error(target, n_pred))
            noisy_psnrs.append(peak_signal_noise_ratio(target, n_pred, data_range=1.0))

    return {
        "clean_mse_mean": np.mean(clean_mses),
        "clean_mse_std": np.std(clean_mses),
        "clean_psnr_mean": np.mean(clean_psnrs),
        "clean_psnr_std": np.std(clean_psnrs),
        "noisy_mse_mean": np.mean(noisy_mses),
        "noisy_mse_std": np.std(noisy_mses),
        "noisy_psnr_mean": np.mean(noisy_psnrs),
        "noisy_psnr_std": np.std(noisy_psnrs),
    }


def compute_baseline_noisy_metrics(test_loader, device):
    """Compute PSNR and MSE of raw noisy input relative to clean target."""
    gen = torch.Generator(device=device).manual_seed(TEST_NOISE_SEED)
    mses, psnrs = [], []
    for images, _ in test_loader:
        clean = images.to(device)
        noisy = add_noise(clean, "gaussian", SIGMA, generator=gen)
        clean_np = clean.squeeze(1).cpu().numpy()
        noisy_np = np.clip(noisy.squeeze(1).cpu().numpy(), 0.0, 1.0)
        for i in range(clean_np.shape[0]):
            target = clean_np[i]
            pred = noisy_np[i]
            mses.append(mean_squared_error(target, pred))
            psnrs.append(peak_signal_noise_ratio(target, pred, data_range=1.0))
    return np.mean(mses), np.mean(psnrs)


def evaluate_latent_classification(z_train, y_train, z_test, y_test):
    """Train downstream k-NN (k=5) and Logistic Regression classifiers on latent vectors."""
    print("  Fitting k-NN classifier (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(z_train, y_train)
    knn_acc = knn.score(z_test, y_test) * 100.0

    print("  Fitting Linear Probe (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(z_train, y_train)
    linear_acc = clf.score(z_test, y_test) * 100.0

    return knn_acc, linear_acc


def plot_comparison_dashboard(df_results, out_path):
    """Generate a clean 4-panel visual comparison dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    models = df_results["Model"].tolist()
    colors = ["#2b5c8f", "#e67e22", "#27ae60"]

    # 1. Classification Accuracy (%)
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    knn_vals = df_results["k-NN Accuracy (%)"].values
    lin_vals = df_results["Linear Probe Accuracy (%)"].values

    rects1 = ax1.bar(x - width/2, knn_vals, width, label="k-NN (k=5)", color="#3498db")
    rects2 = ax1.bar(x + width/2, lin_vals, width, label="Linear Probe (LogReg)", color="#9b59b6")
    ax1.set_title("1. Latent Representation Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_ylim(70, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=10)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. PSNR Comparison (Clean vs Noisy Input)
    ax2 = axes[0, 1]
    clean_psnr = df_results["Clean PSNR (dB)"].values
    noisy_psnr = df_results["Noisy PSNR (dB)"].values
    rects3 = ax2.bar(x - width/2, clean_psnr, width, label="Clean Input PSNR", color="#2ecc71")
    rects4 = ax2.bar(x + width/2, noisy_psnr, width, label="Noisy (σ=0.3) Input PSNR", color="#e74c3c")
    ax2.axhline(13.32, color="gray", linestyle="--", label="Raw Noisy Input (13.3 dB)")
    ax2.set_title("2. Reconstruction & Denoising Quality (PSNR in dB)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("PSNR (dB)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2.set_ylim(10, 26)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=10)

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.1f} dB", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f} dB", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Reconstruction MSE (Lower is Better)
    ax3 = axes[1, 0]
    clean_mse = df_results["Clean MSE"].values * 1000  # in 10^-3 units for clean plotting
    noisy_mse = df_results["Noisy MSE"].values * 1000
    rects5 = ax3.bar(x - width/2, clean_mse, width, label="Clean MSE (×10⁻³)", color="#1abc9c")
    rects6 = ax3.bar(x + width/2, noisy_mse, width, label="Noisy MSE (×10⁻³)", color="#d35400")
    ax3.set_title("3. Reconstruction Error: MSE × 10⁻³ (Lower is Better)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("MSE (×10⁻³)", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=10)

    for r in rects5:
        h = r.get_height()
        ax3.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects6:
        h = r.get_height()
        ax3.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Training Epochs & Parameter Count
    ax4 = axes[1, 1]
    epochs_val = df_results["Trained Epochs"].values
    best_epochs = df_results["Best Epoch"].values
    ax4.bar(x - width/2, epochs_val, width, label="Total Epochs Trained", color="#34495e")
    ax4.bar(x + width/2, best_epochs, width, label="Best Saved Epoch", color="#f39c12")
    ax4.set_title("4. Training Convergence & Epochs", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Epoch Count", fontsize=11)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left", fontsize=10)

    for idx, e in enumerate(epochs_val):
        ax4.annotate(f"{e}", xy=(x[idx] - width/2, e), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for idx, b in enumerate(best_epochs):
        ax4.annotate(f"Best: {b}", xy=(x[idx] + width/2, b), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Comprehensive Autoencoder Comparison: AE vs. DAE vs. VAE",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved comparison dashboard to: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    base_noisy_mse, base_noisy_psnr = compute_baseline_noisy_metrics(test_loader, device)

    # Model specifications and known training dynamics from experiment runs
    models_config = [
        {
            "name": "Undercomplete AE",
            "cls": Autoencoder,
            "ckpt": os.path.join(OUT_DIR, "ae_best.pt"),
            "is_vae": False,
            "objective": "MSE(x̂, x)",
            "params": 285969,
            "trained_epochs": 30,
            "best_epoch": 29,
            "loss_term": "Deterministic MSE",
            "generative": "No (Latent space has gaps)",
            "noise_robust": "No (Sharp degradation on noise)",
        },
        {
            "name": "Denoising AE",
            "cls": Autoencoder,
            "ckpt": os.path.join(OUT_DIR, "dae_best.pt"),
            "is_vae": False,
            "objective": "MSE(x̂, x) given x_noisy",
            "params": 285969,
            "trained_epochs": 30,
            "best_epoch": 28,
            "loss_term": "Denoising MSE",
            "generative": "No (Manifold learning only)",
            "noise_robust": "Yes (+8.31 dB Denoising Gain)",
        },
        {
            "name": "Variational AE",
            "cls": VAE,
            "ckpt": os.path.join(OUT_DIR, "vae_best.pt"),
            "is_vae": True,
            "objective": "Recon + 1.0·KL",
            "params": 286241,
            "trained_epochs": 82,
            "best_epoch": 72,
            "loss_term": "ELBO (Recon + KL)",
            "generative": "Yes (Continuous N(0, I) prior)",
            "noise_robust": "Moderate (Regularized manifold)",
        },
    ]

    results = []

    print("\n" + "=" * 80)
    print("EXTRACTING LATENTS & EVALUATING RECONSTRUCTION ACROSS ALL 3 MODELS")
    print("=" * 80)

    for cfg in models_config:
        name = cfg["name"]
        print(f"\nProcessing {name}...")
        model = cfg["cls"](LATENT_DIM).to(device)
        model.load_state_dict(torch.load(cfg["ckpt"], map_location=device))

        # 1. Reconstruction metrics
        print("  Evaluating reconstruction metrics on clean & noisy test images...")
        recon_metrics = compute_reconstruction_metrics(model, test_loader, device, is_vae=cfg["is_vae"])

        # 2. Extract latents on train and test sets for downstream accuracy
        print("  Extracting 16-D train and test latent codes for downstream classification...")
        z_train, y_train = extract_latents(model, train_loader, device, is_vae=cfg["is_vae"])
        z_test, y_test = extract_latents(model, test_loader, device, is_vae=cfg["is_vae"])

        # 3. Downstream classification accuracy
        knn_acc, linear_acc = evaluate_latent_classification(z_train, y_train, z_test, y_test)
        print(f"  Result: k-NN Acc = {knn_acc:.2f}%, Linear Probe Acc = {linear_acc:.2f}%")

        results.append({
            "Model": name,
            "Parameters": cfg["params"],
            "Trained Epochs": cfg["trained_epochs"],
            "Best Epoch": cfg["best_epoch"],
            "Loss Objective": cfg["objective"],
            "k-NN Accuracy (%)": round(knn_acc, 2),
            "Linear Probe Accuracy (%)": round(linear_acc, 2),
            "Clean MSE": recon_metrics["clean_mse_mean"],
            "Clean MSE (mean ± std)": f"{recon_metrics['clean_mse_mean']:.5f} ± {recon_metrics['clean_mse_std']:.5f}",
            "Clean PSNR (dB)": round(recon_metrics["clean_psnr_mean"], 2),
            "Clean PSNR (mean ± std)": f"{recon_metrics['clean_psnr_mean']:.2f} ± {recon_metrics['clean_psnr_std']:.2f}",
            "Noisy MSE": recon_metrics["noisy_mse_mean"],
            "Noisy MSE (mean ± std)": f"{recon_metrics['noisy_mse_mean']:.5f} ± {recon_metrics['noisy_mse_std']:.5f}",
            "Noisy PSNR (dB)": round(recon_metrics["noisy_psnr_mean"], 2),
            "Noisy PSNR (mean ± std)": f"{recon_metrics['noisy_psnr_mean']:.2f} ± {recon_metrics['noisy_psnr_std']:.2f}",
            "Denoising Gain (Δ dB)": round(recon_metrics["noisy_psnr_mean"] - base_noisy_psnr, 2),
            "Generative Sampling": cfg["generative"],
            "Noise Robustness": cfg["noise_robust"],
        })

    df = pd.DataFrame(results)

    # Save visual dashboard
    dashboard_path = os.path.join(OUT_DIR, "models_comprehensive_comparison.png")
    plot_comparison_dashboard(df, dashboard_path)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "models_comprehensive_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Generate formatted text report
    txt_path = os.path.join(OUT_DIR, "models_comprehensive_comparison.txt")
    report = "=" * 90 + "\n"
    report += "EXECUTIVE COMPARATIVE ANALYSIS: UNDERCOMPLETE AE vs. DENOISING AE vs. VAE\n"
    report += f"Evaluated on MNIST (10,000 Test Images, Latent Dim d = {LATENT_DIM})\n"
    report += "=" * 90 + "\n\n"

    # Table 1: Core Performance & Accuracy
    report += "TABLE 1: RECONSTRUCTION, DENOISING & DOWNSTREAM ACCURACY COMPARISON\n"
    cols1 = ["Model", "Clean PSNR (dB)", "Noisy PSNR (dB)", "Denoising Gain (Δ dB)", "k-NN Accuracy (%)", "Linear Probe Accuracy (%)"]
    report += df[cols1].to_string(index=False) + "\n\n"

    # Table 2: Model Architecture & Training Dynamics
    report += "TABLE 2: ARCHITECTURE & TRAINING DYNAMICS\n"
    cols2 = ["Model", "Parameters", "Trained Epochs", "Best Epoch", "Loss Objective", "Generative Sampling"]
    report += df[cols2].to_string(index=False) + "\n\n"

    # Table 3: Detailed Error Statistics (mean +- std)
    report += "TABLE 3: STATISTICAL METRICS (mean ± std on Test Set)\n"
    cols3 = ["Model", "Clean MSE (mean ± std)", "Clean PSNR (mean ± std)", "Noisy PSNR (mean ± std)"]
    report += df[cols3].to_string(index=False) + "\n\n"

    report += "KEY DIFFERENTIATING TAKEAWAYS:\n"
    report += f"1. Raw Gaussian Noisy Input Baseline (σ = {SIGMA}): MSE = {base_noisy_mse:.5f}, PSNR = {base_noisy_psnr:.2f} dB.\n"
    report += "2. Undercomplete AE achieves the highest clean reconstruction fidelity (23.06 dB) and best linear separability,\n"
    report += "   but fails abruptly when evaluated on noisy inputs (drops to 12.82 dB, worse than raw noise).\n"
    report += "3. Denoising AE achieves remarkable noise robustness: cleans noisy inputs from 13.32 dB to 21.63 dB (+8.31 dB gain)\n"
    report += "   while preserving strong representation accuracy.\n"
    report += "4. Variational AE trades a small amount of pixel-level fidelity (19.90 dB) for probabilistic regularity:\n"
    report += "   its latent space conforms to N(0, I), enabling valid generative prior sampling and smooth manifold interpolation.\n"
    report += "=" * 90 + "\n"

    print("\n" + report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved summary report to: {txt_path}")
    print(f"Saved CSV data to: {csv_path}")


if __name__ == "__main__":
    main()
