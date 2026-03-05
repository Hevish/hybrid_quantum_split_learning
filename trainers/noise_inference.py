"""
Noise-robustness evaluation for split learning.

1. Train encoder + decoder **noise-free** across K folds.
2. At inference, inject Laplacian noise into the split activations
   and record accuracy / F1 under a grid of (mean, scale) parameters.

Used in the audio (Speech Commands) experiments to study how Laplacian
noise on smashed data affects model performance.
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


def train_and_evaluate_with_noise(
    encoder_init: nn.Module,
    decoder_init: nn.Module,
    data_loader_fn,
    *,
    num_folds: int = 5,
    epochs: int = 50,
    lr: float = 0.0001,
    device: torch.device,
    noise_means: list[float] | None = None,
    noise_scales: list[float] | None = None,
    results_dir: str = "results/results_with_noise",
    tag: str = "classical",
):
    """Train split models noise-free, then evaluate under Laplacian noise.

    Parameters
    ----------
    data_loader_fn : callable(fold) -> (train_loader, test_loader)
    noise_means, noise_scales : lists of floats for noise grid
    """
    if noise_means is None:
        noise_means = [0, 2.5 * torch.pi, 3 * torch.pi,
                       3.5 * torch.pi, 4 * torch.pi]
    if noise_scales is None:
        noise_scales = [0.01, 0.1, 0.5, 1, 2]

    criterion = nn.CrossEntropyLoss()
    client_models = []
    server_models = []

    # -- Phase 1: train noise-free ------------------------------------
    print("Phase 1: Training encoder/decoder noise-free ...")
    for fold in range(1, num_folds + 1):
        encoder = copy.deepcopy(encoder_init).to(device)
        decoder = copy.deepcopy(decoder_init).to(device)
        opt_enc = optim.Adam(encoder.parameters(), lr=lr)
        opt_dec = optim.Adam(decoder.parameters(), lr=lr)

        train_loader, _ = data_loader_fn(fold)
        print(f"  Fold {fold}:")

        for epoch in range(1, epochs + 1):
            encoder.train()
            decoder.train()
            running_loss = 0.0
            correct = total = 0

            for X, Y in train_loader:
                X, Y = X.to(device), Y.to(device)
                opt_enc.zero_grad()
                client_out = encoder(X)
                client_fx = client_out.clone().detach().requires_grad_(True)

                opt_dec.zero_grad()
                server_out = decoder(client_fx)
                loss = criterion(server_out, Y)
                loss.backward()

                opt_dec.step()
                client_out.backward(client_fx.grad.clone().detach())
                opt_enc.step()

                running_loss += loss.item()
                _, pred = torch.max(server_out.data, 1)
                total += Y.size(0)
                correct += (pred == Y).sum().item()

            if epoch % 10 == 0 or epoch == epochs:
                print(f"    Epoch {epoch}/{epochs}  loss={running_loss / len(train_loader):.3f}  "
                      f"acc={100 * correct / total:.1f}%")

        client_models.append(encoder)
        server_models.append(decoder)

    # -- Phase 2: evaluate under noise grid ---------------------------
    print("\nPhase 2: Evaluating with Laplacian noise ...")
    os.makedirs(results_dir, exist_ok=True)

    for mean in noise_means:
        for scale in noise_scales:
            print(f"  mean={mean:.4f}  scale={scale}")
            for fold in range(1, num_folds + 1):
                _, test_loader = data_loader_fn(fold)
                enc = client_models[fold - 1]
                dec = server_models[fold - 1]
                enc.eval()
                dec.eval()

                all_pred, all_true = [], []
                running_test_loss = 0.0

                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        activations = enc(images)
                        noise = torch.distributions.Laplace(
                            mean, scale).sample(activations.shape).to(device)
                        noisy_act = ((noise + activations).detach()
                                     - activations).detach() + activations
                        out = dec(noisy_act)
                        loss = criterion(out, labels)
                        running_test_loss += loss.item()

                        _, pred = torch.max(out.data, 1)
                        all_pred.extend(pred.cpu().numpy())
                        all_true.extend(labels.cpu().numpy())

                acc = np.mean(np.array(all_pred) == np.array(all_true))
                f1 = f1_score(all_true, all_pred, average="weighted")
                test_loss = running_test_loss / len(test_loader)

                row = pd.DataFrame([{
                    "Epoch": epochs, "F1-score": f1,
                    "Accuracy": acc, "Loss": test_loss,
                }])
                path = os.path.join(
                    results_dir,
                    f"{tag}_fold_{fold}_laplace_{mean}_{scale}.xlsx")
                row.to_excel(path, index=False, engine="openpyxl")

                print(f"    Fold {fold}  acc={100*acc:.1f}%  f1={100*f1:.1f}%")
