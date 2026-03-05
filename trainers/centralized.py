"""
Centralized training loop (no split).

Works for both tabular (BCELoss, binary) and image (CrossEntropyLoss,
multi-class) datasets.
"""

import copy
from statistics import mean

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_centralized(
    model_init: torch.nn.Module,
    data_loader_fn,
    *,
    num_folds: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    task: str,              # "binary" or "multiclass"
    results_dir: str,
    tag: str = "centralized",
):
    """Run K-fold centralized training and save per-epoch results.

    Parameters
    ----------
    model_init : nn.Module
        Un-trained model (will be ``deepcopy``-ed for every fold).
    data_loader_fn : callable
        ``data_loader_fn(fold)`` returns data for that fold.
        - tabular: ``(x_train, y_train, x_test, y_test)`` as numpy arrays.
        - image:   ``(train_loader, val_loader)`` as DataLoaders.
    task : str
        ``"binary"``  → BCELoss, sigmoid output, accuracy via rounding.
        ``"multiclass"`` → CrossEntropyLoss, argmax output, weighted F1.
    """
    criterion = (torch.nn.BCELoss() if task == "binary"
                 else torch.nn.CrossEntropyLoss())

    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold}/{num_folds} =====")
        model = copy.deepcopy(model_init).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        data = data_loader_fn(fold)
        records = []

        for epoch in range(1, epochs + 1):
            # ── Training ─────────────────────────────────────────────
            model.train()
            if task == "binary":
                _train_binary_epoch(model, optimizer, criterion,
                                    data[0], data[1], batch_size, device)
            else:
                _train_multiclass_epoch(model, optimizer, criterion,
                                        data[0], device)

            # ── Evaluation ───────────────────────────────────────────
            model.eval()
            row = _evaluate(model, criterion, data, batch_size, device, task,
                            epoch)
            records.append(row)
            _print_epoch(epoch, epochs, row, task)

        _save_results(records, results_dir, f"{tag}_fold_{fold}.xlsx", task)


# ── Helpers ──────────────────────────────────────────────────────────────

def _train_binary_epoch(model, optimizer, criterion, x, y, bs, device):
    for i in range(0, len(x), bs):
        inp = torch.tensor(x[i:i+bs], dtype=torch.float32, device=device)
        tgt = torch.tensor(y[i:i+bs], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        out = model(inp).flatten()
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()


def _train_multiclass_epoch(model, optimizer, criterion, loader, device):
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate(model, criterion, data, bs, device, task, epoch):
    if task == "binary":
        x_test, y_test = data[2], data[3]
        preds, trues, losses = [], [], []
        for i in range(0, len(x_test), bs):
            inp = torch.tensor(x_test[i:i+bs], dtype=torch.float32,
                               device=device)
            tgt = torch.tensor(y_test[i:i+bs], dtype=torch.float32,
                               device=device)
            out = model(inp).flatten()
            losses.append(criterion(out, tgt).item() * inp.size(0))
            preds.extend(torch.round(out).cpu().numpy())
            trues.extend(tgt.cpu().numpy())
        cm = confusion_matrix(trues, preds)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(trues, preds)
        return {"Epoch": epoch, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "Accuracy": acc, "Loss": mean(losses)}
    else:
        val_loader = data[1]
        preds, trues, total_loss = [], [], 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            total_loss += criterion(out, labels).item()
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="weighted")
        loss = total_loss / len(val_loader)
        return {"Epoch": epoch, "Accuracy": acc, "Loss": loss,
                "F1-score": f1}


def _print_epoch(epoch, total, row, task):
    acc = row["Accuracy"]
    loss = row["Loss"]
    extra = f", F1={row['F1-score']:.4f}" if task == "multiclass" else ""
    print(f"  Epoch {epoch}/{total}: Acc={acc:.4f}, Loss={loss:.4f}{extra}")


def _save_results(records, results_dir, filename, task):
    import os
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(records)
    path = os.path.join(results_dir, filename)
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"  Results saved → {path}")
