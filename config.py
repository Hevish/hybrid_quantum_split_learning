"""
Configuration for Hybrid Quantum Split Learning (HQSL) experiments.

Centralises all hyperparameters, dataset-specific settings, and device /
seed management so that every script in this project shares one source
of truth.
"""

import os
import random
import numpy as np
import torch

# ── Reproducibility ──────────────────────────────────────────────────────
DEFAULT_SEED = 150993


def set_seed(seed: int = DEFAULT_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


# ── Device ───────────────────────────────────────────────────────────────
def get_device(gpu_id: int = 0) -> torch.device:
    """Return a CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        print(f"GPU runtime selected: {torch.cuda.get_device_name(gpu_id)}")
        return torch.device(f"cuda:{gpu_id}")
    print("No GPU runtime, running on CPU mode")
    return torch.device("cpu")


# ── Shared training hyper-parameters ─────────────────────────────────────
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_FOLDS = 5

# ── Dataset-specific configurations ──────────────────────────────────────
# Each entry stores everything that differs between datasets.

DATASET_CONFIGS = {
    # --- Tabular datasets (binary classification, BCELoss) ---------------
    "botnet_dga": {
        "type": "tabular",
        "input_dim": 7,
        "num_classes": 1,              # binary (sigmoid output)
        "encoder_hidden": [32, 16],
        "split_dim": 3,               # encoder output / server input
        "server_pre_quantum": [],      # no extra layers before qlayer
        "server_post_quantum": [64, 32, 4],
        "epochs": 100,
        "data_dir_template": "folder{fold}",  # relative to dataset root
    },
    "breast_cancer": {
        "type": "tabular",
        "input_dim": 7,
        "num_classes": 1,
        "encoder_hidden": [32, 16],
        "split_dim": 3,
        "server_pre_quantum": [],
        "server_post_quantum": [64, 32, 4],
        "epochs": 100,
        "data_dir_template": "folder{fold}",
    },
    # --- Image datasets (multi-class classification, CrossEntropyLoss) ---
    "mnist": {
        "type": "image",
        "in_channels": 1,
        "num_classes": 10,
        "split_dim": 3,               # features sent from client to server
        "server_post_quantum": [64, 32],
        "epochs": 50,
        "data_dir": "data_kfold",
        "data_dir_stratified": "data_kfold_stratified",
    },
    "fmnist": {
        "type": "image",
        "in_channels": 1,
        "num_classes": 10,
        "split_dim": 3,
        "server_post_quantum": [64, 32],
        "epochs": 50,
        "data_dir": "data_kfold",
        "data_dir_stratified": "data_kfold_stratified",
    },
    # --- Audio dataset (spectrogram-based binary classification) ----------
    "audio": {
        "type": "audio",
        "in_channels": 1,
        "num_classes": 2,
        "split_dim": 3,
        "server_post_quantum": [64, 32],
        "epochs": 50,
        "data_dir": "data",          # contains fold_1/ … fold_5/
        "seed": 61096,
    },
}

# ── Quantum-circuit defaults ─────────────────────────────────────────────
N_QUBITS = 2
N_LAYERS = 1           # single layer for tabular / N-client image
N_LAYERS_IMAGE = 3     # deeper circuit for image split (kfold) experiments
