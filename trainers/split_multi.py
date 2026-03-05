"""
Split-learning training loop — **N clients**, single server.

All N clients share one encoder architecture (same weights) and one
server decoder.  Clients train round-robin within each global epoch.
"""

import copy
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_split_multi(
    encoder_init: torch.nn.Module,
    decoder_init: torch.nn.Module,
    data_loader_fn,
    *,
    n_clients: int,
    num_folds: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    task: str,
    results_dir: str,
    tag: str = "split_multi",
):
    """K-fold split-learning with N clients (round-robin).

    Parameters
    ----------
    data_loader_fn : callable
        ``fn(fold)`` → tabular ``(client_x, client_y, x_test, y_test)``
                      or image ``(train_loaders, val_loader)``
        where ``client_x`` / ``train_loaders`` is a list of length N.
    """
    criterion = (torch.nn.BCELoss() if task == "binary"
                 else torch.nn.CrossEntropyLoss())

    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold}/{num_folds}  ({n_clients} clients) =====")
        encoder = copy.deepcopy(encoder_init).to(device)
        decoder = copy.deepcopy(decoder_init).to(device)
        opt_enc = torch.optim.Adam(encoder.parameters(), lr=lr)
        opt_dec = torch.optim.Adam(decoder.parameters(), lr=lr)
        data = data_loader_fn(fold)
        records = []

        for epoch in range(1, epochs + 1):
            # ── Round-robin training across clients ──────────────────
            for c in range(n_clients):
                encoder.train(); decoder.train()
                if task == "binary":
                    _train_one_client_binary(
                        encoder, decoder, opt_enc, opt_dec, criterion,
                        data[0][c], data[1][c], batch_size, device,
                    )
                else:
                    _train_one_client_multiclass(
                        encoder, decoder, opt_enc, opt_dec, criterion,
                        data[0][c], device,
                    )

            # ── Evaluation on the shared test set ────────────────────
            encoder.eval(); decoder.eval()
            row = _eval(encoder, decoder, criterion, data, batch_size,
                        device, task, epoch)
            records.append(row)
            _print_epoch(epoch, epochs, row, task)

        _save(records, results_dir,
              f"{tag}_{n_clients}clients_fold_{fold}.xlsx")
        # Only run first fold for multi-client (mirrors original behaviour)
        break


# ═════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════════════

def _split_step(encoder, decoder, opt_enc, opt_dec, criterion,
                inputs, targets):
    client_out = encoder(inputs)
    server_in = client_out.clone().detach().requires_grad_(True)
    opt_enc.zero_grad(); opt_dec.zero_grad()
    server_out = decoder(server_in)
    if targets.dim() == 1 and server_out.dim() > 1 and server_out.size(-1) == 1:
        server_out = server_out.flatten()
    loss = criterion(server_out, targets)
    loss.backward()
    dfx = server_in.grad.clone().detach()
    opt_dec.step()
    client_out.backward(dfx)
    opt_enc.step()


def _train_one_client_binary(enc, dec, oe, od, crit, x, y, bs, dev):
    for i in range(0, len(x), bs):
        inp = torch.tensor(x[i:i+bs], dtype=torch.float32, device=dev)
        tgt = torch.tensor(y[i:i+bs], dtype=torch.float32, device=dev)
        _split_step(enc, dec, oe, od, crit, inp, tgt)


def _train_one_client_multiclass(enc, dec, oe, od, crit, loader, dev):
    for inputs, labels in loader:
        inputs, labels = inputs.to(dev), labels.to(dev)
        _split_step(enc, dec, oe, od, crit, inputs, labels)


@torch.no_grad()
def _eval(encoder, decoder, criterion, data, bs, dev, task, epoch):
    if task == "binary":
        x_te, y_te = data[2], data[3]
        preds, trues = [], []
        for i in range(0, len(x_te), bs):
            inp = torch.tensor(x_te[i:i+bs], dtype=torch.float32, device=dev)
            tgt = torch.tensor(y_te[i:i+bs], dtype=torch.float32, device=dev)
            out = decoder(encoder(inp)).flatten()
            preds.extend(torch.round(out).cpu().numpy())
            trues.extend(tgt.cpu().numpy())
        cm = confusion_matrix(trues, preds)
        tn, fp, fn, tp = cm.ravel()
        acc = (tn + tp) / (tn + tp + fn + fp)
        return {"Epoch": epoch, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "Accuracy": acc}
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
    loss = row.get("Loss", "")
    extra = f", F1={row['F1-score']:.4f}" if "F1-score" in row else ""
    loss_str = f", Loss={loss:.4f}" if isinstance(loss, float) else ""
    print(f"  Epoch {ep}/{total}: Acc={acc:.4f}{loss_str}{extra}")


def _save(records, results_dir, filename):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    pd.DataFrame(records).to_excel(path, index=False, engine="openpyxl")
    print(f"  Results saved → {path}")
