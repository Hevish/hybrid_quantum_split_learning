"""
Split-learning training loop -- single client, single server.

The client (encoder) and server (decoder) pass gradients across the *split
point*:  the client sends activations forward, the server back-propagates
loss gradients to the client.
"""

import copy
import os
from statistics import mean

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_split(
    encoder_init: torch.nn.Module,
    decoder_init: torch.nn.Module,
    data_loader_fn,
    *,
    num_folds: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    task: str,              # "binary" or "multiclass"
    results_dir: str,
    tag: str = "split",
    save_checkpoints: bool = False,
):
    """K-fold split-learning training (1 client -> 1 server).

    Parameters
    ----------
    encoder_init, decoder_init : nn.Module
        Initial models (deep-copied per fold).
    data_loader_fn : callable
        ``fn(fold)`` -> tabular ``(x_tr, y_tr, x_te, y_te)``
                      or image ``(train_loader, val_loader)``.
    save_checkpoints : bool
        If True, save encoder state_dict after each fold (useful for
        reconstruction-attack experiments).
    """
    criterion = (torch.nn.BCELoss() if task == "binary"
                 else torch.nn.CrossEntropyLoss())

    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold}/{num_folds} =====")
        encoder = copy.deepcopy(encoder_init).to(device)
        decoder = copy.deepcopy(decoder_init).to(device)
        opt_enc = torch.optim.Adam(encoder.parameters(), lr=lr)
        opt_dec = torch.optim.Adam(decoder.parameters(), lr=lr)
        data = data_loader_fn(fold)
        records = []

        for epoch in range(1, epochs + 1):
            encoder.train(); decoder.train()
            if task == "binary":
                _split_train_binary(encoder, decoder, opt_enc, opt_dec,
                                    criterion, data[0], data[1], batch_size,
                                    device)
            else:
                _split_train_multiclass(encoder, decoder, opt_enc, opt_dec,
                                        criterion, data[0], device)

            encoder.eval(); decoder.eval()
            row = _evaluate_split(encoder, decoder, criterion, data,
                                  batch_size, device, task, epoch)
            records.append(row)
            _print_epoch(epoch, epochs, row, task)

        _save_results(records, results_dir, f"{tag}_fold_{fold}.xlsx")

        if save_checkpoints:
            ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(encoder.state_dict(),
                       os.path.join(ckpt_dir, f"encoder_fold_{fold}.pt"))
            print(f"  Encoder checkpoint saved -> {ckpt_dir}")


# =========================================================================
#  Internal helpers
# =========================================================================

def _forward_backward_split(encoder, decoder, opt_enc, opt_dec, criterion,
                            inputs, targets, device):
    """One mini-batch forward/backward through the split point."""
    client_out = encoder(inputs)
    # Detach and enable grad so we can propagate gradients back across split
    server_in = client_out.clone().detach().requires_grad_(True)
    opt_enc.zero_grad()
    opt_dec.zero_grad()
    server_out = decoder(server_in)
    server_out = server_out.flatten() if targets.dim() == 1 and server_out.dim() > 1 and server_out.size(-1) == 1 else server_out
    loss = criterion(server_out, targets)
    loss.backward()
    # Back-propagate gradient through split
    dfx = server_in.grad.clone().detach()
    opt_dec.step()
    client_out.backward(dfx)
    opt_enc.step()


def _split_train_binary(enc, dec, oe, od, crit, x, y, bs, dev):
    for i in range(0, len(x), bs):
        inp = torch.tensor(x[i:i+bs], dtype=torch.float32, device=dev)
        tgt = torch.tensor(y[i:i+bs], dtype=torch.float32, device=dev)
        _forward_backward_split(enc, dec, oe, od, crit, inp, tgt, dev)


def _split_train_multiclass(enc, dec, oe, od, crit, loader, dev):
    for inputs, labels in loader:
        inputs, labels = inputs.to(dev), labels.to(dev)
        _forward_backward_split(enc, dec, oe, od, crit, inputs, labels, dev)


@torch.no_grad()
def _evaluate_split(encoder, decoder, criterion, data, bs, dev, task, epoch):
    if task == "binary":
        x_te, y_te = data[2], data[3]
        preds, trues, losses = [], [], []
        for i in range(0, len(x_te), bs):
            inp = torch.tensor(x_te[i:i+bs], dtype=torch.float32, device=dev)
            tgt = torch.tensor(y_te[i:i+bs], dtype=torch.float32, device=dev)
            out = decoder(encoder(inp)).flatten()
            losses.append(criterion(out, tgt).item() * inp.size(0))
            preds.extend(torch.round(out).cpu().numpy())
            trues.extend(tgt.cpu().numpy())
        cm = confusion_matrix(trues, preds)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(trues, preds)
        return {"Epoch": epoch, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "Accuracy": acc, "Loss": mean(losses)}
    else:
        val = data[1]
        preds, trues, total_loss = [], [], 0.0
        for images, labels in val:
            images, labels = images.to(dev), labels.to(dev)
            out = decoder(encoder(images))
            total_loss += criterion(out, labels).item()
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="weighted")
        return {"Epoch": epoch, "Accuracy": acc,
                "Loss": total_loss / max(len(val), 1), "F1-score": f1}


def _print_epoch(ep, total, row, task):
    acc = row["Accuracy"]
    loss = row["Loss"]
    extra = f", F1={row['F1-score']:.4f}" if task == "multiclass" else ""
    print(f"  Epoch {ep}/{total}: Acc={acc:.4f}, Loss={loss:.4f}{extra}")


def _save_results(records, results_dir, filename):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    pd.DataFrame(records).to_excel(path, index=False, engine="openpyxl")
    print(f"  Results saved -> {path}")
