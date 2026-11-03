"""Quantitative comparison across all three models using scikit-image (Step 5).

Prompt 5:
  Using scikit-image, compute per-image MSE and PSNR on the test set for all three
  trained models (Undercomplete AE, DAE on noisy inputs, VAE). Report mean +- std
  for each in a single comparison table.

Outputs:
  outputs/quantitative_comparison.txt - Formatted comparison table
  outputs/quantitative_comparison.csv - CSV format of all results
"""

import os
import time

import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import torch

from mnist_data import add_noise, get_dataloaders
from models import Autoencoder, VAE

OUT_DIR = "outputs"
LATENT_DIM = 16
SIGMA = 0.3
SEED = 42
TEST_NOISE_SEED = 123


@torch.no_grad()
def evaluate_models(device):
    print("Loading test data...")
    _, _, test_loader = get_dataloaders()
    test_size = len(test_loader.dataset)
    print(f"Test set size: {test_size} images")

    # Load all 3 models
    ae_path = os.path.join(OUT_DIR, "ae_best.pt")
    dae_path = os.path.join(OUT_DIR, "dae_best.pt")
    vae_path = os.path.join(OUT_DIR, "vae_best.pt")

    for path, name in [(ae_path, "Undercomplete AE"), (dae_path, "Denoising AE"), (vae_path, "VAE")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint for {name}: {path}")

    ae = Autoencoder(LATENT_DIM).to(device).eval()
    ae.load_state_dict(torch.load(ae_path, map_location=device))

    dae = Autoencoder(LATENT_DIM).to(device).eval()
    dae.load_state_dict(torch.load(dae_path, map_location=device))

    vae = VAE(LATENT_DIM).to(device).eval()
    vae.load_state_dict(torch.load(vae_path, map_location=device))

    print("Evaluating models using scikit-image metrics (MSE, PSNR)...")

    # Fixed generator for consistent test set noise
    generator = torch.Generator(device=device).manual_seed(TEST_NOISE_SEED)

    # Scenarios to record:
    # 1. Undercomplete AE (Clean Input)
    # 2. Undercomplete AE (Noisy Input, sigma=0.3)
    # 3. DAE (Noisy Input, sigma=0.3) -> Core Prompt 5 requirement
    # 4. VAE (Clean Input, z=mu) -> Core Prompt 5 requirement
    # 5. Baseline: Raw Noisy Input (sigma=0.3)
    records = {
        "Undercomplete AE (clean input)": {"mse": [], "psnr": []},
        "Undercomplete AE (noisy input, σ=0.3)": {"mse": [], "psnr": []},
        "Denoising AE (noisy input, σ=0.3)": {"mse": [], "psnr": []},
        "Variational AE (clean input, z=μ)": {"mse": [], "psnr": []},
        "Baseline Noisy Input (σ=0.3)": {"mse": [], "psnr": []},
    }

    t0 = time.time()
    for batch_idx, (images, _) in enumerate(test_loader):
        clean_batch = images.to(device)
        noisy_batch = add_noise(clean_batch, "gaussian", SIGMA, generator=generator)

        # Run models
        ae_clean_out = ae(clean_batch)
        ae_noisy_out = ae(noisy_batch)
        dae_noisy_out = dae(noisy_batch)
        vae_clean_out, _, _ = vae(clean_batch)  # in eval mode z = mu

        # Convert to numpy (N, 28, 28) with range [0, 1]
        clean_np = clean_batch.squeeze(1).cpu().numpy()
        noisy_np = noisy_batch.squeeze(1).cpu().numpy()
        ae_clean_np = ae_clean_out.squeeze(1).cpu().numpy()
        ae_noisy_np = ae_noisy_out.squeeze(1).cpu().numpy()
        dae_noisy_np = dae_noisy_out.squeeze(1).cpu().numpy()
        vae_clean_np = vae_clean_out.squeeze(1).cpu().numpy()

        outputs = {
            "Undercomplete AE (clean input)": ae_clean_np,
            "Undercomplete AE (noisy input, σ=0.3)": ae_noisy_np,
            "Denoising AE (noisy input, σ=0.3)": dae_noisy_np,
            "Variational AE (clean input, z=μ)": vae_clean_np,
            "Baseline Noisy Input (σ=0.3)": noisy_np,
        }

        # Compute per-image metrics with scikit-image
        for i in range(clean_np.shape[0]):
            target = clean_np[i]
            for key, pred_batch in outputs.items():
                pred = np.clip(pred_batch[i], 0.0, 1.0)
                m = mean_squared_error(target, pred)
                # peak_signal_noise_ratio with data_range=1.0
                p = peak_signal_noise_ratio(target, pred, data_range=1.0)
                records[key]["mse"].append(m)
                records[key]["psnr"].append(p)

    print(f"Processed {test_size} test images across all configurations in {time.time() - t0:.2f}s")

    # Aggregate results into mean +- std
    summary_data = []
    for model_name, metrics in records.items():
        mse_arr = np.array(metrics["mse"])
        psnr_arr = np.array(metrics["psnr"])
        summary_data.append({
            "Model / Condition": model_name,
            "MSE Mean": mse_arr.mean(),
            "MSE Std": mse_arr.std(),
            "PSNR Mean (dB)": psnr_arr.mean(),
            "PSNR Std (dB)": psnr_arr.std(),
            "MSE (mean ± std)": f"{mse_arr.mean():.5f} ± {mse_arr.std():.5f}",
            "PSNR dB (mean ± std)": f"{psnr_arr.mean():.2f} ± {psnr_arr.std():.2f}",
        })

    df = pd.DataFrame(summary_data)

    # Focus table for the 3 main models specified in Prompt 5
    prompt5_names = [
        "Undercomplete AE (clean input)",
        "Denoising AE (noisy input, σ=0.3)",
        "Variational AE (clean input, z=μ)",
    ]
    df_prompt5 = df[df["Model / Condition"].isin(prompt5_names)].copy()

    # Format text output
    header = "=" * 80 + "\n"
    header += "STEP 5: QUANTITATIVE RECONSTRUCTION & DENOISING COMPARISON (scikit-image)\n"
    header += f"Evaluated on {test_size} MNIST test images | data_range = 1.0\n"
    header += "=" * 80 + "\n\n"

    table_p5 = df_prompt5[["Model / Condition", "MSE (mean ± std)", "PSNR dB (mean ± std)"]].to_string(index=False)
    table_full = df[["Model / Condition", "MSE (mean ± std)", "PSNR dB (mean ± std)"]].to_string(index=False)

    full_report = header + "--- Core Prompt 5 Comparison (Three Models) ---\n" + table_p5 + "\n\n"
    full_report += "--- Full Benchmark Context (Including Baselines) ---\n" + table_full + "\n"

    print("\n" + full_report)

    # Save to outputs
    txt_path = os.path.join(OUT_DIR, "quantitative_comparison.txt")
    csv_path = os.path.join(OUT_DIR, "quantitative_comparison.csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    df.to_csv(csv_path, index=False)
    print(f"Saved reports to {txt_path} and {csv_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    evaluate_models(device)


if __name__ == "__main__":
    main()
