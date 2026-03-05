# HQSL: Hybrid Quantum Split Learning

A clean, modular implementation of **Hybrid Quantum Split Learning (HQSL)** 
designed for anyone who wants to experiment with quantum-enhanced split learning
across tabular, image, and audio datasets.

## Paper

This repository complements the following paper:

> **A Hybrid Quantum Neural Network for Split Learning**
>
> Published in *Quantum Machine Intelligence* (Springer, 2025).
>
> - Springer: https://link.springer.com/article/10.1007/s42484-025-00295-z
> - arXiv: https://arxiv.org/pdf/2409.16593

If you're looking to reproduce the results, extend the framework to new
datasets, or simply explore how parameterised quantum circuits can be
integrated into a split learning pipeline, this codebase is a good
starting point.

> **Note:** This repository was made public following its publication.

## Project Structure

```
hybrid_quantum_split_learning/
|-- config.py                 # Seeds, device, hyperparameters, dataset configs
|-- quantum_circuit.py        # 2-qubit data-loading PennyLane circuit
|-- data_utils.py             # Data loaders (tabular + image + audio, single/multi-client)
|-- metrics.py                # Reconstruction metrics (MSE, cosine dist, DSSIM, LSD)
|-- run_experiment.py         # Main CLI entry point
|
|-- models/
|   |-- tabular.py            # Encoder/Decoder/Centralized for tabular data
|   |-- image.py              # CNN Encoder/Server/Centralized for images & audio
|   `-- adversary.py          # Adversary generator for reconstruction attacks
|
|-- trainers/
|   |-- centralized.py        # Centralized training loop (K-fold)
|   |-- split_single.py       # Split learning, 1 client <-> 1 server
|   |-- split_multi.py        # Split learning, N clients <-> 1 server
|   |-- noise_inference.py    # Train noise-free, test with Laplacian noise
|   `-- reconstruction.py     # Reconstruction attack pipeline
|
`-- scripts/
    |-- run_tabular.sh        # Example: tabular dataset experiments
    |-- run_image.sh          # Example: image dataset experiments
    |-- run_audio.sh          # Example: audio dataset experiments
    `-- run_reconstruction.sh # Example: reconstruction attacks
```

## Datasets

| Dataset       | Type    | Input       | Classes | Loss     |
|---------------|---------|-------------|---------|----------|
| `botnet_dga`  | Tabular | 7 features  | 2       | BCE      |
| `breast_cancer` | Tabular | 7 features | 2       | BCE      |
| `mnist`       | Image   | 1x28x28     | 10      | CrossEntropy |
| `fmnist`      | Image   | 1x28x28     | 10      | CrossEntropy |
| `audio`       | Audio   | 1x28x28 spectrogram | 2 | CrossEntropy |

## Experiment Modes

- **`centralized`** -- Standard training, no split (baseline)
- **`split`** -- Split learning: client encoder <-> server decoder, K-fold CV
- **`split_multi`** -- Split learning with N clients, round-robin training
- **`reconstruction`** -- Adversary reconstructs inputs from split activations
- **`noise_inference`** -- Train noise-free, evaluate under Laplacian noise grid (audio)

Each mode supports `--variant classical` or `--variant hybrid` (quantum layer on server).

## Quick Start

```bash
# Install dependencies
pip install torch pennylane scikit-learn pandas openpyxl skimage

# Run centralized classical on botnet_dga
python run_experiment.py --dataset botnet_dga --mode centralized --variant classical

# Run split hybrid on MNIST
python run_experiment.py --dataset mnist --mode split --variant hybrid

# Run multi-client split (10 clients) on FMNIST
python run_experiment.py --dataset fmnist --mode split_multi --variant hybrid --n_clients 10

# Reconstruction attack
python run_experiment.py --dataset mnist --mode reconstruction --variant classical \
    --encoder_checkpoint results/mnist/split/encoder_classical.pt

# Audio: split hybrid
python run_experiment.py --dataset audio --mode split --variant hybrid

# Audio: noise-robustness evaluation
python run_experiment.py --dataset audio --mode noise_inference --variant classical

# Custom Laplacian noise grid (reconstruction or noise_inference)
python run_experiment.py --dataset mnist --mode reconstruction --variant classical \
    --encoder_checkpoint results/mnist/split/encoder_classical.pt \
    --noise_means 0 3.14 6.28 --noise_scales 0.01 0.1 0.5 1.0
```

## CLI Arguments

| Argument | Description |
|---|---|
| `--dataset` | `botnet_dga`, `breast_cancer`, `mnist`, `fmnist`, `audio` |
| `--mode` | `centralized`, `split`, `split_multi`, `reconstruction`, `noise_inference` |
| `--variant` | `classical` or `hybrid` |
| `--n_clients` | Number of clients (for `split_multi`; default 2) |
| `--seed` | Random seed (default 150993) |
| `--gpu` | GPU index (default 0) |
| `--epochs` | Override epoch count from config |
| `--n_layers` | Override quantum circuit layers |
| `--results_dir` | Custom results directory |
| `--data_root` | Root path for data directories |
| `--encoder_checkpoint` | Path to encoder `.pt` file (reconstruction mode) |
| `--recon_epochs` | Adversary training epochs (default 200) |
| `--noise_means` | Laplacian noise mean(s) for noise grid (e.g. `0 3.14 6.28`) |
| `--noise_scales` | Laplacian noise scale(s) for noise grid (e.g. `0.01 0.1 1.0`) |

## Quantum Circuit

2-qubit **efficient data-loading** ansatz with 3 stages per layer:

1. RX(input_1) on each qubit -> trainable RZ -> CZ entanglement
2. RX(input_2) on each qubit -> trainable RY -> CZ entanglement
3. RX(input_3) on each qubit -> trainable RZ

Each layer has **6 trainable parameters** and accepts **3 input dimensions**.

- Returns `qml.qnn.TorchLayer` for seamless PyTorch integration

## Data Layout

The data loaders expect pre-split K-fold data on disk:

- **Tabular**: `folder{fold}/train_{fold}.csv`, `folder{fold}/test_{fold}.csv`
- **Image**: `data_kfold/fold{fold}/train_images.pt`, `data_kfold/fold{fold}/train_labels.pt`, etc.
- **Multi-client image**: Uses `StratifiedShuffleSplit` to partition training data at runtime
- **Audio**: `data/fold_{k}/train/{class}/` and `data/fold_{k}/test/{class}/` (ImageFolder layout with spectrogram PNGs)
