#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Reconstruction attack experiments (image datasets only)
# ─────────────────────────────────────────────────────────────

# After training a split model, use the saved encoder checkpoint
# to run reconstruction attacks.

# MNIST classical
python run_experiment.py --dataset mnist --mode reconstruction --variant classical \
    --encoder_checkpoint results/mnist/split/encoder_classical_split_fold0.pt \
    --data_root . --recon_epochs 200 --gpu 0

# MNIST hybrid
python run_experiment.py --dataset mnist --mode reconstruction --variant hybrid \
    --encoder_checkpoint results/mnist/split/encoder_hybrid_split_fold0.pt \
    --data_root . --recon_epochs 200 --gpu 0

# FMNIST classical
python run_experiment.py --dataset fmnist --mode reconstruction --variant classical \
    --encoder_checkpoint results/fmnist/split/encoder_classical_split_fold0.pt \
    --data_root . --recon_epochs 200 --gpu 0

# FMNIST hybrid
python run_experiment.py --dataset fmnist --mode reconstruction --variant hybrid \
    --encoder_checkpoint results/fmnist/split/encoder_hybrid_split_fold0.pt \
    --data_root . --recon_epochs 200 --gpu 0
