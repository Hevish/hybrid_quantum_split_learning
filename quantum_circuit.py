"""
Quantum circuit factory for HQSL experiments.

Provides a single function `create_quantum_layer` that builds and returns
a PennyLane TorchLayer implementing the 2-qubit data-reuploading ansatz
used throughout the paper.
"""

import numpy as np
import pennylane as qml
import torch


def create_quantum_layer(
    n_qubits: int = 2,
    n_layers: int = 1,
    device: torch.device | None = None,
):
    """Create a PennyLane TorchLayer with a data-reuploading circuit.

    The circuit expects ``n_qubits + 1`` classical inputs (for the 2-qubit
    default that is 3 inputs) which are re-uploaded into every layer
    together with trainable RZ / RY / RZ rotations.  Consecutive layers are
    connected by CZ entangling gates.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 2).
    n_layers : int
        Number of variational layers (default 1).
    device : torch.device, optional
        Torch device to place the layer on.  If ``None`` the layer is left
        on CPU.

    Returns
    -------
    qml.qnn.TorchLayer
        A differentiable quantum layer that can be used inside
        ``nn.Sequential`` or any PyTorch module.
    """
    n_inputs = n_qubits + 1  # data-reuploading: 3 inputs for 2 qubits

    dev = qml.device("default.qubit", wires=n_qubits)
    weight_shapes = {"weights": (n_layers, n_qubits * 3)}

    @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RZ(inputs[0], wires=q)
                qml.RY(inputs[1], wires=q)
                qml.RZ(inputs[2], wires=q)
                qml.RZ(weights[layer][3 * q], wires=q)
                qml.RY(weights[layer][3 * q + 1], wires=q)
                qml.RZ(weights[layer][3 * q + 2], wires=q)
            if layer < n_layers - 1:
                qml.CZ(wires=[0, 1])
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    if device is not None and device.type == "cuda":
        qlayer = qlayer.cuda(device)
    return qlayer
