"""
Neural-network models for **tabular** datasets (botnet_dga, breast_cancer).

All four variants share the same backbone:
  Encoder  →  [quantum layer or classical bottleneck]  →  Decoder head

The encoder is always on the client side; the decoder (server) comes in a
classical or hybrid (quantum) flavour.
"""

import torch
import torch.nn as nn

from quantum_circuit import create_quantum_layer


# ── Encoder (client) ─────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """MLP encoder that maps raw features to a low-dimensional split vector."""

    def __init__(self, input_dim: int = 7, hidden: list[int] | None = None,
                 split_dim: int = 3):
        super().__init__()
        hidden = hidden or [32, 16]
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, split_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# ── Classical decoder (server) ───────────────────────────────────────────

class TabularClassicalDecoder(nn.Module):
    """Purely classical server-side decoder for binary classification."""

    def __init__(self, split_dim: int = 3, post_dims: list[int] | None = None):
        super().__init__()
        post_dims = post_dims or [64, 32, 4]
        layers = [nn.Linear(split_dim, 2), nn.ReLU()]
        prev = 2
        for dim in post_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            if dim >= 32:
                layers.append(nn.Dropout(0.5))
            prev = dim
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# ── Hybrid decoder (server) ─────────────────────────────────────────────

class TabularHybridDecoder(nn.Module):
    """Hybrid server decoder: quantum layer followed by classical head."""

    def __init__(self, n_qubits: int = 2, n_layers: int = 1,
                 post_dims: list[int] | None = None,
                 device: torch.device | None = None):
        super().__init__()
        post_dims = post_dims or [64, 32, 4]
        qlayer = create_quantum_layer(n_qubits, n_layers, device)
        layers = [qlayer]
        prev = n_qubits
        for dim in post_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            if dim >= 32:
                layers.append(nn.Dropout(0.5))
            prev = dim
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# ── Centralized models (encoder + decoder in one) ───────────────────────

class TabularCentralizedClassical(nn.Module):
    """End-to-end classical model (no split)."""

    def __init__(self, input_dim: int = 7, split_dim: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, split_dim), nn.ReLU(),
            nn.Linear(split_dim, 2), nn.ReLU(),
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class TabularCentralizedHybrid(nn.Module):
    """End-to-end hybrid model (no split)."""

    def __init__(self, input_dim: int = 7, n_qubits: int = 2,
                 n_layers: int = 1, device: torch.device | None = None):
        super().__init__()
        qlayer = create_quantum_layer(n_qubits, n_layers, device)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_qubits + 1), nn.ReLU(),  # 3 inputs for 2-qubit DRU
            qlayer,
            nn.Linear(n_qubits, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
