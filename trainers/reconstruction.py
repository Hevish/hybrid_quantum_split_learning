"""
Reconstruction-attack pipeline for image datasets.

1. Load a **pre-trained** client encoder checkpoint.
2. Train an adversary generator to invert the encoder's activations.
3. Evaluate reconstruction quality (MSE, cosine distance, DSSIM)
   under varying levels of Laplacian noise.
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from metrics import compute_metrics_batch


def run_reconstruction_attack(
    encoder: nn.Module,
    adversary_init: nn.Module,
    train_loader,
    test_loader,
    *,
    epochs: int = 200,
    lr: float = 0.001,
    momentum: float = 0.9,
    device: torch.device,
    noise_means: list[float] | None = None,
    noise_scales: list[float] | None = None,
    results_path: str = "reconstruction_results.xlsx",
    include_lsd: bool = False,
):
    """Train the adversary and evaluate reconstruction under noise.

    Parameters
    ----------
    encoder : nn.Module
        Pre-trained (frozen) client encoder.
    adversary_init : nn.Module
        Un-trained adversary generator.
    train_loader, test_loader : DataLoader
        Image data used to train and evaluate the adversary.
    noise_means, noise_scales : list[float]
        Grid of Laplacian noise parameters to sweep at test time.
    """
    if noise_means is None:
        noise_means = [0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi]
    if noise_scales is None:
        noise_scales = [0.01, 0.1, 0.5, 1.0]

    encoder = encoder.to(device)
    encoder.eval()
    adv_model = copy.deepcopy(adversary_init).to(device)
    opt = optim.SGD(adv_model.parameters(), lr=lr, momentum=momentum)
    recon_loss = nn.L1Loss()

    # ── Phase 1: train adversary (noise-free) ────────────────────────
    print("Training adversary model …")
    adv_model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            opt.zero_grad()
            with torch.no_grad():
                activations = encoder(inputs)
            act = Variable(activations.clone(), requires_grad=True)
            reconstructed = adv_model(act)
            loss = recon_loss(reconstructed, inputs)
            loss.backward()
            opt.step()
            running += loss.item()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs}  loss={running:.4f}")

    # ── Phase 2: evaluate under varying noise ────────────────────────
    print("\nEvaluating reconstruction under Laplacian noise …")
    adv_model.eval()
    results = []

    # Grab a test batch
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)

    with torch.no_grad():
        clean_act = encoder(test_images)

    for scale in noise_scales:
        for loc in noise_means:
            noise = torch.distributions.Laplace(loc, scale).sample(
                clean_act.shape).to(device)
            noisy_act = clean_act + noise
            with torch.no_grad():
                recon = adv_model(noisy_act)
            m = compute_metrics_batch(
                test_images.cpu().numpy(), recon.cpu().numpy(),
                include_lsd=include_lsd,
            )
            row = {
                "Mean": loc, "Scale": scale,
                "MSE": m["mse"][0], "CosDist": m["cos_dist"][0],
                "DSSIM": m["dssim"][0],
            }
            if include_lsd:
                row["LSD"] = m["lsd"][0]
            results.append(row)
            print(f"  μ={loc:.3f} σ={scale:.3f}  "
                  f"MSE={row['MSE']:.4f}  CosDist={row['CosDist']:.4f}  "
                  f"DSSIM={row['DSSIM']:.4f}")

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    pd.DataFrame(results).to_excel(results_path, index=False,
                                   engine="openpyxl")
    print(f"\nResults saved → {results_path}")
