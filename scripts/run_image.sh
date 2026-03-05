#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Example shell scripts for image dataset experiments
# ─────────────────────────────────────────────────────────────

# ── MNIST ────────────────────────────────────────────────────
# Centralized classical
python run_experiment.py --dataset mnist --mode centralized --variant classical \
    --data_root . --seed 150993 --gpu 0

# Centralized hybrid
python run_experiment.py --dataset mnist --mode centralized --variant hybrid \
    --data_root . --seed 150993 --gpu 0

# Split classical (K-fold, 3 qlayers for hybrid)
python run_experiment.py --dataset mnist --mode split --variant classical \
    --data_root . --seed 150993 --gpu 0

# Split hybrid (K-fold, 3 qlayers)
python run_experiment.py --dataset mnist --mode split --variant hybrid \
    --data_root . --seed 150993 --gpu 0

# Split N-client sweep
for N in 2 5 10 20 50 100; do
    python run_experiment.py --dataset mnist --mode split_multi --variant classical \
        --n_clients $N --data_root . --seed 150993 --gpu 0
    python run_experiment.py --dataset mnist --mode split_multi --variant hybrid \
        --n_clients $N --data_root . --seed 150993 --gpu 0
done

# ── FMNIST ───────────────────────────────────────────────────
for MODE in centralized split; do
    for V in classical hybrid; do
        python run_experiment.py --dataset fmnist --mode $MODE --variant $V \
            --data_root . --seed 150993 --gpu 0
    done
done

for N in 2 5 10 20 50 100; do
    python run_experiment.py --dataset fmnist --mode split_multi --variant classical \
        --n_clients $N --data_root . --seed 150993 --gpu 0
    python run_experiment.py --dataset fmnist --mode split_multi --variant hybrid \
        --n_clients $N --data_root . --seed 150993 --gpu 0
done
