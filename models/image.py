"""
Neural-network models for **image** datasets (MNIST, Fashion-MNIST).

The client runs a small CNN encoder; the server runs either a classical or
hybrid (quantum) decoder.  Centralized variants combine both into one model.
"""

import torch
import torch.nn as nn

from quantum_circuit import create_quantum_layer


# -- CNN Encoder (client side) -------------------------------------------------

class ImageEncoder(nn.Module):
    """Two-layer CNN that produces a low-dimensional feature vector."""

    def __init__(self, in_channels: int = 1, split_dim: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, split_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -- Classical server decoder --------------------------------------------------

class ImageClassicalServer(nn.Module):
    """Purely classical server network for multi-class classification."""

    def __init__(self, split_dim: int = 3, num_classes: int = 10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(split_dim, 2), nn.ReLU(),
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# -- Hybrid server decoder -----------------------------------------------------

class ImageHybridServer(nn.Module):
    """Hybrid server: quantum layer -> classical head."""

    def __init__(self, n_qubits: int = 2, n_layers: int = 1,
                 num_classes: int = 10,
                 device: torch.device | None = None):
        super().__init__()
        qlayer = create_quantum_layer(n_qubits, n_layers, device)
        self.model = nn.Sequential(
            qlayer,
            nn.Linear(n_qubits, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# -- Centralized classical -----------------------------------------------------

class ImageCentralizedClassical(nn.Module):
    """Full CNN without split (classical baseline)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 2)
        self.head = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.head(x)
        return x


# -- Centralized hybrid --------------------------------------------------------

class ImageCentralizedHybrid(nn.Module):
    """Full CNN + quantum layer (hybrid baseline)."""

    def __init__(self, in_channels: int = 1, n_qubits: int = 2,
                 n_layers: int = 1, num_classes: int = 10,
                 device: torch.device | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, n_qubits)
        qlayer = create_quantum_layer(n_qubits, n_layers, device)
        self.head = nn.Sequential(
            qlayer,
            nn.Linear(n_qubits, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.head(x)
        return x


# -- Laplacian noise layer (for inference-time noise experiments) ---------------

class LaplacianNoiseLayer(nn.Module):
    """Adds Laplacian noise to the activation tensor."""

    def __init__(self, loc: float = 0.0, scale: float = 0.1):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        noise = torch.distributions.Laplace(self.loc, self.scale).sample(x.shape)
        return x + noise.to(x.device)
