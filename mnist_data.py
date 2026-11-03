"""MNIST data pipeline and noise models for the autoencoder project.

Implements Section 3 of the research paper:
  3.1  Preprocessing  - pixels scaled to [0, 1], 10% of train held out for validation,
                        mini-batches of 128 shuffled every epoch.
  3.2  Noise models   - Gaussian, salt-and-pepper and masking noise for the DAE,
                        applied on-the-fly to each batch.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_ROOT = "data"
BATCH_SIZE = 128
VAL_FRACTION = 0.1
SEED = 42

# Parameter ranges from Table 3.2 of the paper.
NOISE_RANGES = {
    "gaussian": (0.1, 0.5),         # sigma
    "salt_and_pepper": (0.05, 0.2),  # p, fraction of pixels corrupted
    "masking": (0.1, 0.3),          # p, fraction of pixels zeroed
}


def get_datasets(root=DATA_ROOT, val_fraction=VAL_FRACTION, seed=SEED):
    """Return (train, val, test) MNIST datasets with pixels in [0, 1]."""
    # ToTensor converts uint8 [0, 255] images to float32 [0, 1] tensors of shape (1, 28, 28).
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    # Fixed generator so the train/val split is identical across runs and models.
    train, val = random_split(full_train, [n_train, n_val],
                              generator=torch.Generator().manual_seed(seed))
    return train, val, test


def get_dataloaders(batch_size=BATCH_SIZE, root=DATA_ROOT, val_fraction=VAL_FRACTION,
                    seed=SEED, num_workers=0):
    """Return (train_loader, val_loader, test_loader).

    num_workers defaults to 0 because worker processes on Windows require the
    calling script to be guarded by `if __name__ == "__main__":`.
    """
    train, val, test = get_datasets(root, val_fraction, seed)
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    test_loader = DataLoader(test, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def add_noise(images, noise_type, param, generator=None):
    """Return a corrupted copy of `images` (a tensor with values in [0, 1]).

    noise_type:
      "gaussian"        - x + eps, eps ~ N(0, sigma^2); param = sigma in [0.1, 0.5]
      "salt_and_pepper" - each pixel set to 0 or 1 (equal chance) with probability p;
                          param = p in [0.05, 0.2]
      "masking"         - each pixel set to 0 with probability p; param = p in [0.1, 0.3]

    The output is clipped to [0, 1]. The input tensor is not modified. Pass a
    torch.Generator (on the same device as `images`) for reproducible noise.
    """
    if noise_type not in NOISE_RANGES:
        raise ValueError(f"Unknown noise_type {noise_type!r}; "
                         f"expected one of {list(NOISE_RANGES)}")
    low, high = NOISE_RANGES[noise_type]
    if not low <= param <= high:
        raise ValueError(f"{noise_type} param must be in [{low}, {high}] "
                         f"(paper Table 3.2), got {param}")

    def rand(fn):
        return fn(images.shape, generator=generator, device=images.device, dtype=images.dtype)

    if noise_type == "gaussian":
        noisy = images + param * rand(torch.randn)
    elif noise_type == "salt_and_pepper":
        noisy = images.clone()
        corrupt = rand(torch.rand) < param
        salt = rand(torch.rand) < 0.5
        noisy[corrupt & salt] = 1.0
        noisy[corrupt & ~salt] = 0.0
    else:  # masking
        keep = rand(torch.rand) >= param
        noisy = images * keep

    return noisy.clamp(0.0, 1.0)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"{name:5}: {len(loader.dataset):6} images, {len(loader):3} batches")

    images, labels = next(iter(train_loader))
    print("batch:", tuple(images.shape), images.dtype,
          f"range [{images.min():.1f}, {images.max():.1f}]")
    for noise_type, param in [("gaussian", 0.3), ("salt_and_pepper", 0.1), ("masking", 0.2)]:
        noisy = add_noise(images, noise_type, param)
        print(f"{noise_type:16} param={param}: range [{noisy.min():.2f}, {noisy.max():.2f}], "
              f"pixels changed {(noisy != images).float().mean():.1%}")
