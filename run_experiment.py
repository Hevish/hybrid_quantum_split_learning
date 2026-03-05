#!/usr/bin/env python
"""
Unified entry point for all HQSL experiments.

Usage
-----
# Centralized classical, botnet_dga
python run_experiment.py --dataset botnet_dga --mode centralized --variant classical

# Split hybrid, breast_cancer, all 5 folds
python run_experiment.py --dataset breast_cancer --mode split --variant hybrid

# Split classical, FMNIST, N=10 clients
python run_experiment.py --dataset fmnist --mode split_multi --variant classical --n_clients 10

# Split hybrid, MNIST, N=50 clients
python run_experiment.py --dataset mnist --mode split_multi --variant hybrid --n_clients 50

# Audio: split hybrid
python run_experiment.py --dataset audio --mode split --variant hybrid

# Audio: noise-robustness evaluation
python run_experiment.py --dataset audio --mode noise_inference --variant classical

# Reconstruction attack on MNIST (classical encoder)
python run_experiment.py --dataset mnist --mode reconstruction --variant classical \\
    --encoder_checkpoint path/to/encoder.pt

# Custom Laplacian noise grid (applies to reconstruction & noise_inference)
python run_experiment.py --dataset audio --mode noise_inference --variant hybrid \\
    --noise_means 0 3.14 6.28 --noise_scales 0.01 0.1 1.0
"""

import argparse
import copy
import os
import sys

import torch

from config import (
    BATCH_SIZE, DATASET_CONFIGS, LEARNING_RATE, NUM_FOLDS,
    N_LAYERS, N_QUBITS,
    get_device, set_seed,
)

# -- Model factories ------------------------------------------------------

def _build_tabular_models(cfg, variant, device):
    from models.tabular import (
        TabularCentralizedClassical, TabularCentralizedHybrid,
        TabularEncoder, TabularClassicalDecoder, TabularHybridDecoder,
    )
    input_dim = cfg["input_dim"]
    split_dim = cfg["split_dim"]
    if variant == "classical":
        centralized = TabularCentralizedClassical(input_dim, split_dim)
        encoder = TabularEncoder(input_dim, cfg["encoder_hidden"], split_dim)
        decoder = TabularClassicalDecoder(split_dim, cfg["server_post_quantum"])
    else:
        centralized = TabularCentralizedHybrid(
            input_dim, N_QUBITS, N_LAYERS, device)
        encoder = TabularEncoder(input_dim, cfg["encoder_hidden"], split_dim)
        decoder = TabularHybridDecoder(
            N_QUBITS, N_LAYERS, cfg["server_post_quantum"], device)
    return centralized, encoder, decoder


def _build_image_models(cfg, variant, device, n_layers=None):
    from models.image import (
        ImageCentralizedClassical, ImageCentralizedHybrid,
        ImageEncoder, ImageClassicalServer, ImageHybridServer,
    )
    nc = cfg["num_classes"]
    sd = cfg["split_dim"]
    if n_layers is None:
        n_layers = N_LAYERS
    if variant == "classical":
        centralized = ImageCentralizedClassical(cfg["in_channels"], nc)
        encoder = ImageEncoder(cfg["in_channels"], sd)
        decoder = ImageClassicalServer(sd, nc)
    else:
        centralized = ImageCentralizedHybrid(
            cfg["in_channels"], N_QUBITS, n_layers, nc, device)
        encoder = ImageEncoder(cfg["in_channels"], sd)
        decoder = ImageHybridServer(N_QUBITS, n_layers, nc, device)
    return centralized, encoder, decoder


# -- Data-loader factories ------------------------------------------------

def _tabular_data_fn(data_root):
    from data_utils import load_tabular_fold
    return lambda fold: load_tabular_fold(data_root, fold)


def _tabular_multi_data_fn(data_root, n_clients):
    from data_utils import load_tabular_fold_multi_client
    return lambda fold: load_tabular_fold_multi_client(
        data_root, fold, n_clients)


def _image_data_fn(data_root):
    from data_utils import load_image_fold
    return lambda fold: load_image_fold(data_root, fold)


def _image_multi_data_fn(data_root, n_clients):
    from data_utils import load_image_fold_multi_client
    return lambda fold: load_image_fold_multi_client(
        data_root, fold, n_clients, BATCH_SIZE)


# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run HQSL experiments (centralized / split / multi-client)")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--mode", required=True,
                        choices=["centralized", "split", "split_multi",
                                 "reconstruction", "noise_inference"])
    parser.add_argument("--variant", required=True,
                        choices=["classical", "hybrid"])
    parser.add_argument("--n_clients", type=int, default=2,
                        help="Number of clients (split_multi mode)")
    parser.add_argument("--seed", type=int, default=150993)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs from config")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Override quantum circuit layers")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=".",
                        help="Root dir containing folder1/, data_kfold/ etc.")
    # Noise parameters (for reconstruction & noise_inference modes)
    parser.add_argument("--noise_means", type=float, nargs="+", default=None,
                        help="Laplacian noise mean(s), e.g. --noise_means 0 3.14 6.28")
    parser.add_argument("--noise_scales", type=float, nargs="+", default=None,
                        help="Laplacian noise scale(s), e.g. --noise_scales 0.01 0.1 0.5 1.0")
    # Reconstruction-specific
    parser.add_argument("--encoder_checkpoint", type=str, default=None)
    parser.add_argument("--recon_epochs", type=int, default=200)

    args = parser.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    seed = cfg.get("seed", args.seed)
    set_seed(seed)
    device = get_device(args.gpu)
    is_audio = cfg["type"] == "audio"

    epochs = args.epochs or cfg["epochs"]
    results_dir = args.results_dir or os.path.join(
        "results", args.dataset, args.mode)
    tag = f"{args.variant}_{args.mode}"
    task = "binary" if cfg["type"] == "tabular" else "multiclass"
    is_tabular = cfg["type"] == "tabular"

    n_layers = args.n_layers if args.n_layers is not None else N_LAYERS

    # Build models
    if is_tabular:
        centralized, encoder, decoder = _build_tabular_models(
            cfg, args.variant, device)
    else:
        # image and audio share the same CNN architectures
        centralized, encoder, decoder = _build_image_models(
            cfg, args.variant, device, n_layers)

    # -- Helper: pick the right data loader factory ------------------
    def _get_loader_fn():
        if is_tabular:
            return _tabular_data_fn(args.data_root)
        if is_audio:
            from data_utils import load_audio_fold
            data_dir = os.path.join(args.data_root, cfg["data_dir"])
            return lambda fold: load_audio_fold(data_dir, fold, BATCH_SIZE)
        # image (mnist / fmnist)
        data_dir = os.path.join(args.data_root, cfg["data_dir"])
        return _image_data_fn(data_dir)

    def _get_multi_loader_fn(n):
        if is_tabular:
            return _tabular_multi_data_fn(args.data_root, n)
        if is_audio:
            from data_utils import load_audio_fold_multi_client
            data_dir = os.path.join(args.data_root, cfg["data_dir"])
            return lambda fold: load_audio_fold_multi_client(
                data_dir, fold, n, BATCH_SIZE)
        data_dir = os.path.join(args.data_root,
                                cfg.get("data_dir_stratified",
                                        cfg["data_dir"]))
        return _image_multi_data_fn(data_dir, n)

    # -- Centralized --------------------------------------------------
    if args.mode == "centralized":
        from trainers.centralized import train_centralized
        train_centralized(
            centralized, _get_loader_fn(),
            num_folds=NUM_FOLDS, epochs=epochs, lr=LEARNING_RATE,
            batch_size=BATCH_SIZE, device=device, task=task,
            results_dir=results_dir, tag=tag,
        )

    # -- Split (single client) ---------------------------------------
    elif args.mode == "split":
        from trainers.split_single import train_split
        train_split(
            encoder, decoder, _get_loader_fn(),
            num_folds=NUM_FOLDS, epochs=epochs, lr=LEARNING_RATE,
            batch_size=BATCH_SIZE, device=device, task=task,
            results_dir=results_dir, tag=tag,
            save_checkpoints=True,
        )

    # -- Split (N clients) -------------------------------------------
    elif args.mode == "split_multi":
        from trainers.split_multi import train_split_multi
        N = args.n_clients
        train_split_multi(
            encoder, decoder, _get_multi_loader_fn(N),
            n_clients=N, num_folds=N + 1, epochs=epochs, lr=LEARNING_RATE,
            batch_size=BATCH_SIZE, device=device, task=task,
            results_dir=results_dir, tag=tag,
        )

    # -- Noise inference (train clean, test with noise) ---------------
    elif args.mode == "noise_inference":
        from trainers.noise_inference import train_and_evaluate_with_noise
        train_and_evaluate_with_noise(
            encoder, decoder, _get_loader_fn(),
            num_folds=NUM_FOLDS, epochs=epochs, lr=LEARNING_RATE,
            device=device, results_dir=results_dir, tag=tag,
            noise_means=args.noise_means,
            noise_scales=args.noise_scales,
        )

    # -- Reconstruction attack ----------------------------------------
    elif args.mode == "reconstruction":
        from models.adversary import AdversaryGenerator
        from models.image import ImageEncoder
        from trainers.reconstruction import run_reconstruction_attack

        if args.encoder_checkpoint is None:
            print("ERROR: --encoder_checkpoint is required for "
                  "reconstruction mode")
            sys.exit(1)

        enc = ImageEncoder(cfg["in_channels"], cfg["split_dim"]).to(device)
        enc.load_state_dict(torch.load(args.encoder_checkpoint,
                                       map_location=device))
        print(f"Loaded encoder from {args.encoder_checkpoint}")

        adv = AdversaryGenerator(input_nc=cfg["split_dim"])

        # Load pre-saved train/test loaders for reconstruction
        if is_audio:
            from data_utils import load_audio_fold
            data_dir = os.path.join(args.data_root, cfg["data_dir"])
            train_loader, test_loader = load_audio_fold(
                data_dir, 1, BATCH_SIZE)
        else:
            train_loader = torch.load(
                os.path.join(args.data_root, "train_loader.pt"))
            test_loader = torch.load(
                os.path.join(args.data_root, "test_loader.pt"))

        run_reconstruction_attack(
            enc, adv, train_loader, test_loader,
            epochs=args.recon_epochs, device=device,
            results_path=os.path.join(results_dir,
                                      f"reconstruction_{args.variant}.xlsx"),
            include_lsd=is_audio,
            noise_means=args.noise_means,
            noise_scales=args.noise_scales,
        )


if __name__ == "__main__":
    main()
