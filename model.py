"""
=============================================================================
Deep Forward Network – MR Gaze Classification
=============================================================================
Assignment : A1  |  Task : Binary classification (gazing label 0 / 1)
Input      : 64-dimensional voice-embedding vectors from an MR system
=============================================================================

ARCHITECTURE OVERVIEW
---------------------
GazeNet is a deep residual feedforward network:

  Input (64)
    └─ BatchNorm1d
        ├─ Block-0 : Linear(64→256)  + BN + GELU + Dropout(0.35)  [+ skip]
        ├─ Block-1 : Linear(256→128) + BN + GELU + Dropout(0.35)  [+ skip]
        ├─ Block-2 : Linear(128→64)  + BN + GELU + Dropout(0.35)  [+ skip]
        └─ Head    : Linear(64→2)

REGULARISATION CHOICES
----------------------
  • BatchNorm    – normalises layer inputs, reduces internal covariate shift,
                   acts as a mild regulariser (adds noise during training).
  • Dropout(0.35)– randomly zeroes activations to prevent co-adaptation.
  • L2 (AdamW)  – weight decay on non-bias/BN params discourages large weights.
  • Residual     – shortcut connections help gradient flow and reduce overfitting
                   by allowing the network to learn residual mappings.
  • Early stopping (patience=40) – halts training before overfitting begins.
  • Gradient clipping (norm=5.0) – prevents exploding gradients.

OPTIMISATION
------------
  • Optimiser : AdamW  (lr=8e-4, weight_decay=3e-4)
  • Scheduler : CosineAnnealingWarmRestarts  (T_0=40, T_mult=2)
  • Loss      : CrossEntropyLoss with class weights to handle imbalance
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import ast
import re
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.utils.data      import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim           import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection        import StratifiedKFold
from sklearn.preprocessing          import StandardScaler
from sklearn.metrics                import (f1_score,
                                            roc_auc_score, classification_report,
                                            confusion_matrix)
from sklearn.utils.class_weight     import compute_class_weight


SEED = 42

def set_seed(seed: int = SEED):
    """Fix all random sources for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Data Loading and Preprocessing

def safe_parse_vector(raw: str) -> list | None:
    cleaned = re.sub(r'\bnan\b', '0.0', raw)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return None


def load_training_data(path: str,
                       feature_col: str = "input_ids",
                       label_col:   str = "label_id") -> tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(path)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    parsed   = [safe_parse_vector(v) for v in df[feature_col]]
    good_idx = [i for i, p in enumerate(parsed) if p is not None]

    X = np.array([parsed[i] for i in good_idx], dtype=np.float32)
    y = df[label_col].values[good_idx].astype(np.int64)
    return X, y


def load_test_data(path: str,
                   feature_col: str = "input_ids") -> np.ndarray:

    df     = pd.read_csv(path)
    parsed = [safe_parse_vector(v) for v in df[feature_col]]
    X      = np.array([p if p is not None else [0.0]*64
                       for p in parsed], dtype=np.float32)
    return X


def make_loader(X: np.ndarray,
                y: np.ndarray | None = None,
                batch_size: int       = 128,
                shuffle:    bool      = False,
                oversample: bool      = False) -> DataLoader:

    Xt = torch.tensor(X, dtype=torch.float32)

    if y is not None:
        yt = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(Xt, yt)

        if oversample:
            counts    = np.bincount(y)
            weights   = 1.0 / counts[y]
            sampler   = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.float32),
                num_samples=len(y), replacement=True)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        ds = TensorDataset(Xt)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


#Architecture
class ResidualBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.proj = nn.Linear(in_dim, out_dim, bias=False) \
                    if in_dim != out_dim else nn.Identity()

        nn.init.kaiming_normal_(self.block[0].weight, nonlinearity="linear")
        nn.init.zeros_(self.block[0].bias)
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.proj(x)


class GazeNet(nn.Module):

    def __init__(self,
                 input_dim:   int   = 64,
                 hidden_dims: tuple = (256, 128, 64),
                 dropout:     float = 0.35,
                 num_classes: int   = 2):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        blocks  = []
        in_dim  = input_dim
        for h in hidden_dims:
            blocks.append(ResidualBlock(in_dim, h, dropout))
            in_dim = h
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)          
        for block in self.blocks:
            x = block(x)              
        return self.head(x)           


#Training Utilities

class EarlyStopping:

    def __init__(self, patience: int = 40, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.best_state = None         

    def step(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss              
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_criterion(y_train: np.ndarray) -> nn.CrossEntropyLoss:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=w_tensor)


def build_optimizer_scheduler(model: nn.Module,
                               lr:           float = 8e-4,
                               weight_decay: float = 3e-4,
                               T_0:          int   = 40,
                               T_mult:       int   = 2):
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and "bias" not in n
                       and "bn" not in n and "input_bn" not in n]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ("bias" in n
                       or "bn" in n or "input_bn" in n)]

    optimizer = AdamW(
        [{"params": decay_params,    "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)
    return optimizer, scheduler


#Training and Evaluation Loops

def train_one_epoch(model:     nn.Module,
                    loader:    DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    model.train()
    total_loss = correct = n = 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        n          += len(y_b)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model:     nn.Module,
             loader:    DataLoader,
             criterion: nn.Module) -> dict:
    model.eval()
    total_loss = correct = n = 0
    all_probs, all_labels = [], []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        logits    = model(X_b)
        loss      = criterion(logits, y_b)

        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        n          += len(y_b)
        all_probs .append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_labels.append(y_b.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs > 0.5).astype(int)

    return {
        "loss": total_loss / n,
        "acc":  correct / n,
        "auc":  roc_auc_score(labels, probs),
        "f1":   f1_score(labels, preds, zero_division=0),
        "probs":  probs,
        "labels": labels,
        "preds":  preds,
    }


def train_model(model:        nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                criterion:    nn.Module,
                n_epochs:     int   = 300,
                lr:           float = 8e-4,
                weight_decay: float = 3e-4,
                patience:     int   = 40,
                verbose:      bool  = True) -> dict:
    optimizer, scheduler = build_optimizer_scheduler(model, lr, weight_decay)
    stopper              = EarlyStopping(patience=patience)
    history = {k: [] for k in
               ["train_loss","val_loss","train_acc","val_acc","val_auc","val_f1"]}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics     = evaluate(model, val_loader, criterion)

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"]  .append(val_metrics["loss"])
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(val_metrics["acc"])
        history["val_auc"]   .append(val_metrics["auc"])
        history["val_f1"]    .append(val_metrics["f1"])

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Ep {epoch:>4} | "
                  f"tr_loss={tr_loss:.4f}  va_loss={val_metrics['loss']:.4f} | "
                  f"va_acc={val_metrics['acc']:.4f}  "
                  f"va_auc={val_metrics['auc']:.4f}  "
                  f"va_f1={val_metrics['f1']:.4f}")

        if stopper.step(val_metrics["loss"], model):
            if verbose:
                print(f"  ↳ Early stopping triggered at epoch {epoch}. "
                      f"Best val_loss={-stopper.best_score:.4f}")
            break

    stopper.restore(model)   
    return history


# Main Training Pipeline 

def main_train(train_csv:   str = "A1-training.csv",
               pretest_csv: str = "A1-testing.csv",
               weights_out: str = "gazenet_weights.pth"):
    print("=" * 65)
    print("  GazeNet – MR Gaze Classification")
    print("=" * 65)
    print(f"  Device : {DEVICE}\n")

    X_train, y_train = load_training_data(train_csv)
    X_pre,   y_pre   = load_training_data(pretest_csv)   
    INPUT_DIM        = X_train.shape[1]                   

    print(f"  Training samples : {len(X_train)}")
    print(f"  Pre-test samples : {len(X_pre)}")
    print(f"  Input dimension  : {INPUT_DIM}")
    print(f"  Class dist (train) class-0={np.sum(y_train==0)}  "
          f"class-1={np.sum(y_train==1)}\n")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_pre   = scaler.transform(X_pre).astype(np.float32)

    print("─" * 65)
    print("  5-Fold Stratified Cross-Validation")
    print("─" * 65)
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {"acc": [], "auc": [], "f1": []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        tr_loader = make_loader(X_tr, y_tr, batch_size=128,
                                shuffle=False, oversample=True)
        va_loader = make_loader(X_va, y_va, batch_size=256)

        fold_model = GazeNet(input_dim=INPUT_DIM,
                             hidden_dims=(256, 128, 64),
                             dropout=0.35).to(DEVICE)
        criterion  = build_criterion(y_tr)

        print(f"\n  Fold {fold}/5")
        history = train_model(fold_model, tr_loader, va_loader,
                              criterion, n_epochs=300, patience=40,
                              verbose=True)

        metrics = evaluate(fold_model, va_loader, criterion)
        cv_scores["acc"].append(metrics["acc"])
        cv_scores["auc"].append(metrics["auc"])
        cv_scores["f1"] .append(metrics["f1"])
        print(f"  → Fold {fold} final:  "
              f"acc={metrics['acc']:.4f}  "
              f"auc={metrics['auc']:.4f}  "
              f"f1={metrics['f1']:.4f}")

    print("\n" + "─" * 65)
    print("  Cross-Validation Summary")
    print("─" * 65)
    for metric, vals in cv_scores.items():
        print(f"  {metric.upper():6s} : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n" + "─" * 65)
    print("  Final Model Training (all train data, validated on pre-test)")
    print("─" * 65)

    tr_loader  = make_loader(X_train, y_train, batch_size=128,
                             shuffle=False, oversample=True)
    pre_loader = make_loader(X_pre,   y_pre,   batch_size=256)
    criterion  = build_criterion(y_train)

    final_model = GazeNet(input_dim=INPUT_DIM,
                          hidden_dims=(256, 128, 64),
                          dropout=0.35).to(DEVICE)

    total_params = sum(p.numel() for p in final_model.parameters()
                       if p.requires_grad)
    print(f"\n  Total trainable parameters : {total_params:,}")
    print()

    history = train_model(final_model, tr_loader, pre_loader,
                          criterion, n_epochs=300, patience=40, verbose=True)

    metrics = evaluate(final_model, pre_loader, criterion)
    print("\n" + "=" * 65)
    print("  PRE-TEST SET EVALUATION")
    print("=" * 65)
    print(f"  Accuracy : {metrics['acc']:.4f}")
    print(f"  ROC-AUC  : {metrics['auc']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print()
    print(classification_report(metrics["labels"], metrics["preds"],
                                target_names=["Not Gazing (0)", "Gazing (1)"]))

    cm = confusion_matrix(metrics["labels"], metrics["preds"])
    print("  Confusion Matrix:")
    print(f"  {cm}")

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "scaler_mean":      torch.tensor(scaler.mean_,    dtype=torch.float32),
        "scaler_scale":     torch.tensor(scaler.scale_,   dtype=torch.float32),
        "input_dim":        INPUT_DIM,
        "hidden_dims":      (256, 128, 64),
        "dropout":          0.35,
        "cv_acc_mean":      float(np.mean(cv_scores["acc"])),
        "cv_auc_mean":      float(np.mean(cv_scores["auc"])),
        "pretest_acc":      float(metrics["acc"]),
        "pretest_auc":      float(metrics["auc"]),
    }, weights_out)
    print(f"\n  ✓ Weights saved → {weights_out}")

    return final_model, scaler, history, metrics


#Inference 

def predict(test_input_csv:  str = "test_input.csv",
            test_output_csv: str = "test_output.csv",
            weights_path:    str = "gazenet_weights.pth"):

    ckpt       = torch.load(weights_path, map_location=DEVICE)
    input_dim  = ckpt["input_dim"]
    hidden_dims = ckpt["hidden_dims"]
    dropout    = ckpt["dropout"]

    model = GazeNet(input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler_mean  = ckpt["scaler_mean"] .numpy()
    scaler_scale = ckpt["scaler_scale"].numpy()

    X_test = load_test_data(test_input_csv)
    X_test = ((X_test - scaler_mean) / scaler_scale).astype(np.float32)

    loader = make_loader(X_test, batch_size=256)
    preds  = []

    with torch.no_grad():
        for (X_b,) in loader:
            X_b    = X_b.to(DEVICE)
            logits = model(X_b)
            preds .extend(logits.argmax(1).cpu().numpy().tolist())

    out_df = pd.DataFrame({"Y": preds})
    out_df.to_csv(test_output_csv, index=False)
    print(f"Predictions saved to {test_output_csv}  "
          f"(n={len(preds)}, class-0={preds.count(0)}, class-1={preds.count(1)})")


#Entry Point
if __name__ == "__main__":

    if os.path.exists("test_input.csv") and not os.path.exists("A1-training.csv"):
        print("Inference mode detected (test_input.csv found).")
        predict(test_input_csv  = "test_input.csv",
                test_output_csv = "test_output.csv",
                weights_path    = "gazenet_weights.pth")

    else:
        final_model, scaler, history, metrics = main_train(
            train_csv   = "A1-training.csv",
            pretest_csv = "A1-testing.csv",
            weights_out = "gazenet_weights.pth",
        )

        if os.path.exists("test_input.csv"):
            predict()