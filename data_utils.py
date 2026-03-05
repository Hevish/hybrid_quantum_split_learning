"""
Data-loading utilities for all datasets.

* **Tabular** (botnet_dga, breast_cancer): CSV files in ``folder{fold}/``
  directories.
* **Image** (MNIST, FMNIST): Pre-saved PyTorch DataLoader ``.pt`` files.
* **Audio** (Speech Commands spectrograms): ImageFolder with
  ``fold_{k}/train/`` and ``fold_{k}/test/`` subdirectories.

For multi-client experiments, the training set is split among N clients
using stratified sampling.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms


# ═════════════════════════════════════════════════════════════════════════
#  Tabular datasets
# ═════════════════════════════════════════════════════════════════════════

def load_tabular_fold(data_root: str, fold: int):
    """Load train/test splits for one K-fold iteration.

    Returns
    -------
    x_train, y_train, x_test, y_test : np.ndarray
    """
    train_path = os.path.join(data_root, f"folder{fold}", "train.csv")
    test_path = os.path.join(data_root, f"folder{fold}", "test.csv")
    train_set = pd.read_csv(train_path, header=None, encoding="utf-8")
    test_set = pd.read_csv(test_path, header=None, encoding="utf-8")
    x_train = train_set.iloc[:, :-1].to_numpy()
    y_train = train_set.iloc[:, -1].to_numpy()
    x_test = test_set.iloc[:, :-1].to_numpy()
    y_test = test_set.iloc[:, -1].to_numpy()
    return x_train, y_train, x_test, y_test


def load_tabular_fold_multi_client(data_root: str, fold: int, n_clients: int):
    """Load a fold and partition the training set among *n_clients*.

    Returns
    -------
    client_x : list[np.ndarray]   – one array per client
    client_y : list[np.ndarray]
    x_test, y_test : np.ndarray
    """
    x_train, y_train, x_test, y_test = load_tabular_fold(data_root, fold)
    chunk = len(x_train) // n_clients + 1
    client_x, client_y = [], []
    for start in range(0, len(x_train), chunk):
        end = start + chunk
        client_x.append(x_train[start:end])
        client_y.append(y_train[start:end])
    return client_x, client_y, x_test, y_test


# ═════════════════════════════════════════════════════════════════════════
#  Image datasets  (pre-saved DataLoaders)
# ═════════════════════════════════════════════════════════════════════════

def load_image_fold(data_root: str, fold: int):
    """Load pre-saved train/val DataLoaders for one fold.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    train_path = os.path.join(data_root, f"train_fold_{fold}.pt")
    val_path = os.path.join(data_root, f"val_fold_{fold}.pt")
    train_loader = torch.load(train_path)
    val_loader = torch.load(val_path)
    return train_loader, val_loader


def load_image_fold_multi_client(data_root: str, fold: int, n_clients: int,
                                 batch_size: int = 32):
    """Load a fold and split training data among *n_clients* (stratified).

    Returns
    -------
    train_loaders : list[DataLoader]  – one per client
    val_loader : DataLoader
    """
    train_path = os.path.join(data_root, f"train_fold_{fold}.pt")
    val_path = os.path.join(data_root, f"val_fold_{fold}.pt")
    train_loader = torch.load(train_path)
    val_loader = torch.load(val_path)

    # Flatten batched DataLoader into tensors for stratified splitting
    all_data, all_labels = [], []
    for batch_data, batch_labels in train_loader:
        all_data.append(batch_data)
        all_labels.append(batch_labels)
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    sss = StratifiedShuffleSplit(
        n_splits=n_clients, test_size=1 / n_clients, random_state=42
    )
    train_loaders = []
    for idx, _ in sss.split(all_data, all_labels):
        subset = TensorDataset(all_data[idx], all_labels[idx])
        train_loaders.append(DataLoader(subset, batch_size=batch_size,
                                        shuffle=True))
    return train_loaders, val_loader


# ═════════════════════════════════════════════════════════════════════════
#  Audio dataset  (Speech Commands spectrograms via ImageFolder)
# ═════════════════════════════════════════════════════════════════════════

_AUDIO_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def load_audio_fold(data_root: str, fold: int, batch_size: int = 32):
    """Load train/test DataLoaders for one audio fold.

    Expects ``data_root/fold_{fold}/train/`` and ``…/test/`` in ImageFolder
    layout (class sub-directories).

    Returns
    -------
    train_loader, test_loader : DataLoader
    """
    fold_path = os.path.join(data_root, f"fold_{fold}")
    train_ds = datasets.ImageFolder(os.path.join(fold_path, "train"),
                                    transform=_AUDIO_TRANSFORM)
    test_ds = datasets.ImageFolder(os.path.join(fold_path, "test"),
                                   transform=_AUDIO_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_audio_fold_multi_client(data_root: str, fold: int, n_clients: int,
                                 batch_size: int = 32):
    """Load an audio fold and partition training data among *n_clients*.

    Returns
    -------
    client_loaders : list[DataLoader]
    test_loader : DataLoader
    """
    fold_path = os.path.join(data_root, f"fold_{fold}")
    train_ds = datasets.ImageFolder(os.path.join(fold_path, "train"),
                                    transform=_AUDIO_TRANSFORM)
    test_ds = datasets.ImageFolder(os.path.join(fold_path, "test"),
                                   transform=_AUDIO_TRANSFORM)

    total = len(train_ds)
    indices = torch.randperm(total).tolist()
    split_size = total // n_clients

    client_loaders = []
    for i in range(n_clients):
        start = i * split_size
        end = start + split_size
        subset = Subset(train_ds, indices[start:end])
        client_loaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader
