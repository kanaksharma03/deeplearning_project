# Image Denoising and Reconstruction Using Autoencoders

A comparative study of **Undercomplete Autoencoder, Denoising Autoencoder (DAE), and Variational Autoencoder (VAE)** for image reconstruction, noise removal, and image generation.

## Overview
This project implements and compares three convolutional autoencoder architectures to reconstruct images and remove noise. It explores different noise models and evaluates reconstruction quality using MSE and PSNR.

## Models Implemented
- **Undercomplete Autoencoder:** Image compression and reconstruction.
- **Denoising Autoencoder:** Reconstructs clean images from noisy inputs.
- **Variational Autoencoder:** Learns probabilistic latent representations and generates new images.

## Dataset
- MNIST
- Fashion-MNIST
- CIFAR-10

## Noise Techniques
- Gaussian Noise
- Salt-and-Pepper Noise
- Masking Noise

## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Scikit-image
- Google Colab / Jupyter Notebook

## Evaluation Metrics
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- PCA and t-SNE for latent space visualization

## Key Findings
- Undercomplete AE provides baseline image reconstruction.
- Denoising AE improves reconstruction from noisy inputs.
- VAE enables smooth latent space representation and image generation.

## Team Members – Group 5

| Name | Roll Number |

| Kanak Sharma | FET-BAML-2024-28-073 |
| T. Saranya Patra | FET-BAML-2024-28-074 |
| Archana Sahoo | FET-BAML-2024-28-075 |
| Varsha Sharma | FET-BAML-2024-28-076 |

**Project Guide:** Yugansh Varshney Sir

**Institution:** Sri Sri University  
**Department:** Computer Science and Engineering  
**Academic Year:** 2024–2028

## References
1. Vincent et al. (2008) – Denoising Autoencoders.
2. Kingma & Welling (2014) – Auto-Encoding Variational Bayes.
3. Goodfellow et al. (2016) – Deep Learning.
4. Paszke et al. (2019) – PyTorch.
