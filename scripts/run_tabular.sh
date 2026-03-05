#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Example shell scripts for running HQSL experiments
# ─────────────────────────────────────────────────────────────

# ── Tabular datasets (botnet_dga, breast_cancer) ────────────
# Centralized classical
python run_experiment.py --dataset botnet_dga --mode centralized --variant classical \
    --data_root . --seed 150993 --gpu 0

# Centralized hybrid
python run_experiment.py --dataset botnet_dga --mode centralized --variant hybrid \
    --data_root . --seed 150993 --gpu 0

# Split classical (1 client)
python run_experiment.py --dataset botnet_dga --mode split --variant classical \
    --data_root . --seed 150993 --gpu 0

# Split hybrid (1 client)
python run_experiment.py --dataset botnet_dga --mode split --variant hybrid \
    --data_root . --seed 150993 --gpu 0

# Split N=10 classical clients
python run_experiment.py --dataset botnet_dga --mode split_multi --variant classical \
    --n_clients 10 --data_root . --seed 150993 --gpu 0

# Split N=10 hybrid clients
python run_experiment.py --dataset botnet_dga --mode split_multi --variant hybrid \
    --n_clients 10 --data_root . --seed 150993 --gpu 0
