#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Audio dataset (Speech Commands spectrograms) experiments
# ─────────────────────────────────────────────────────────────

# Split classical (K-fold)
python run_experiment.py --dataset audio --mode split --variant classical \
    --data_root . --gpu 0

# Split hybrid (K-fold)
python run_experiment.py --dataset audio --mode split --variant hybrid \
    --data_root . --gpu 0

# Multi-client sweep
for N in 2 3 4 5 10 20 50 100; do
    python run_experiment.py --dataset audio --mode split_multi --variant classical \
        --n_clients $N --data_root . --gpu 0
    python run_experiment.py --dataset audio --mode split_multi --variant hybrid \
        --n_clients $N --data_root . --gpu 0
done

# Noise-robustness evaluation
python run_experiment.py --dataset audio --mode noise_inference --variant classical \
    --data_root . --gpu 0

python run_experiment.py --dataset audio --mode noise_inference --variant hybrid \
    --data_root . --gpu 0

# Reconstruction attack
python run_experiment.py --dataset audio --mode reconstruction --variant classical \
    --encoder_checkpoint models/encoder_SL_audio.pt \
    --data_root . --gpu 0

python run_experiment.py --dataset audio --mode reconstruction --variant hybrid \
    --encoder_checkpoint models/encoder_HQSL_audio.pt \
    --data_root . --gpu 0
