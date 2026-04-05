
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

from sklearn.model_selection    import StratifiedKFold
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (accuracy_score, f1_score, roc_auc_score,
                                        classification_report, confusion_matrix,
                                        roc_curve)
from sklearn.utils.class_weight import compute_class_weight


# File paths for training phase
TRAIN_CSV   = "A1-training.csv"
PRETEST_CSV = "A1-testing.csv"    # has labels; assignment permits using it

# File paths for inference phase 
TEST_INPUT_CSV  = "test_input.csv"
TEST_OUTPUT_CSV = "test_output.csv"
WEIGHTS_PATH    = "gazenet_weights.pth"

# Architecture hyperparameters (selected by grid search)
INPUT_DIM    = 64
HIDDEN_DIMS  = (256, 128, 64)
DROPOUT      = 0.35

# Training hyperparameters
BATCH_SIZE   = 128
LR           = 8e-4
WEIGHT_DECAY = 3e-4
T_0, T_MULT  = 40, 2      #cosine warm-restart schedule
N_EPOCHS     = 300
PATIENCE     = 40
N_FOLDS      = 5           #number of CV folds
SEED         = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Data loading and preprocessing utilities; handles parsing of stringified vectors, normalisation, and DataLoader construction.

def safe_parse_vector(raw: str):
    cleaned = re.sub(r'\bnan\b', '0.0', raw)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return None


def load_labelled_csv(path, feature_col="input_ids", label_col="label_id"):
    df = pd.read_csv(path)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    parsed   = [safe_parse_vector(v) for v in df[feature_col]]
    good_idx = [i for i, p in enumerate(parsed) if p is not None]

    X = np.array([parsed[i] for i in good_idx], dtype=np.float32)
    y = df[label_col].values[good_idx].astype(np.int64)
    return X, y


def load_unlabelled_csv(path, feature_col="input_ids"):
    df     = pd.read_csv(path)
    parsed = [safe_parse_vector(v) for v in df[feature_col]]
    X = np.array(
        [p if p is not None else [0.0] * INPUT_DIM for p in parsed],
        dtype=np.float32
    )
    return X


def make_loader(X, y=None, batch_size=BATCH_SIZE, shuffle=False, oversample=False):
    Xt = torch.tensor(X, dtype=torch.float32)

    if y is not None:
        yt = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(Xt, yt)
        if oversample:
            counts  = np.bincount(y)
            weights = 1.0 / counts[y]   # inverse-frequency weighting
            sampler = WeightedRandomSampler(
                weights     = torch.tensor(weights, dtype=torch.float32),
                num_samples = len(y),
                replacement = True,
            )
            return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        ds = TensorDataset(Xt)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


#Architecture

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        #Linear -> BN -> GELU -> Dropout
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        #identity if dims match, otherwise learned projection
        self.shortcut = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

        #Kaiming init for the main linear layer
        nn.init.kaiming_normal_(self.transform[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.transform[0].bias)

        #Linear init 
        if isinstance(self.shortcut, nn.Linear):
            nn.init.kaiming_normal_(self.shortcut.weight, nonlinearity="linear")

    def forward(self, x):
        return self.transform(x) + self.shortcut(x)


class GazeNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS,
                 dropout=DROPOUT, num_classes=2):
        super().__init__()

        #Normalise raw input features before the first linear transformation
        self.input_bn = nn.BatchNorm1d(input_dim)

        #Build stacked residual blocks
        blocks, in_dim = [], input_dim
        for out_dim in hidden_dims:
            blocks.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)

        #Classification head: raw logits for CrossEntropyLoss
        self.head = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.input_bn(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


#Training util.

class EarlyStopping:

    def __init__(self, patience=PATIENCE, min_delta=1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.best_state = None

    def step(self, val_loss, model):
        score     = -val_loss   # lower loss = higher score
        improved  = (self.best_score is None
                     or score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_criterion(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    )


def build_optimizer_and_scheduler(model, lr=LR, wd=WEIGHT_DECAY):
    decay    = [p for n, p in model.named_parameters()
                if p.requires_grad and "bias" not in n
                and "bn" not in n and "input_bn" not in n]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and ("bias" in n or "bn" in n or "input_bn" in n)]

    optimizer = AdamW(
        [{"params": decay,    "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=1e-6
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer):
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
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = n = 0
    all_probs, all_labels = [], []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        logits    = model(X_b)

        total_loss += criterion(logits, y_b).item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        n          += len(y_b)
        all_probs .append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_labels.append(y_b.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs > 0.5).astype(int)

    return {
        "loss":   total_loss / n,
        "acc":    correct / n,
        "auc":    roc_auc_score(labels, probs),
        "f1":     f1_score(labels, preds, zero_division=0),
        "probs":  probs,
        "labels": labels,
        "preds":  preds,
    }


def train_model(model, train_loader, val_loader, criterion,
                n_epochs=N_EPOCHS, patience=PATIENCE,
                lr=LR, weight_decay=WEIGHT_DECAY, verbose=True):
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, weight_decay)
    stopper              = EarlyStopping(patience=patience)
    history = {k: [] for k in
               ["train_loss", "val_loss", "train_acc", "val_acc", "val_auc", "val_f1"]}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_m           = evaluate(model, val_loader, criterion)
        scheduler.step()    # update LR once per epoch

        history["train_loss"].append(tr_loss)
        history["val_loss"]  .append(val_m["loss"])
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(val_m["acc"])
        history["val_auc"]   .append(val_m["auc"])
        history["val_f1"]    .append(val_m["f1"])

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Ep {epoch:>4} | "
                  f"tr_loss={tr_loss:.4f}  val_loss={val_m['loss']:.4f} | "
                  f"val_acc={val_m['acc']:.4f}  "
                  f"val_auc={val_m['auc']:.4f}  "
                  f"val_f1={val_m['f1']:.4f}")

        if stopper.step(val_m["loss"], model):
            if verbose:
                print(f"  -> Early stop at epoch {epoch}  "
                      f"(best val_loss={-stopper.best_score:.4f})")
            break

    stopper.restore(model)
    return history

#Evaluate a model on a dataset and return predicted probabilities for the positive class (gazing) without computing gradients. This is used for threshold tuning and ensemble soft-voting.

@torch.no_grad()
def predict_proba(model, X, batch_size=256):
    model.eval()
    loader = make_loader(X, batch_size=batch_size)
    probs  = []
    for (X_b,) in loader:
        logits = model(X_b.to(DEVICE))
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def find_best_threshold(probs, labels):
    _, _, thresholds = roc_curve(labels, probs)
    best_thresh, best_acc = 0.5, 0.0
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        acc   = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh
    return float(best_thresh)


#Main training pipeline; runs if training data is present, otherwise runs inference if test_input.csv is present.

def run_training():
    print("  GazeNet -- MR Gaze Classification  |  Training Pipeline")

    #Load data 
    X_train, y_train = load_labelled_csv(TRAIN_CSV)
    X_pre,   y_pre   = load_labelled_csv(PRETEST_CSV)

    # Merge training + pre-test (both have labels)
    X_all = np.concatenate([X_train, X_pre], axis=0)
    y_all = np.concatenate([y_train, y_pre], axis=0)

    print(f"  Training samples : {len(X_train)}")
    print(f"  Pre-test samples : {len(X_pre)}")
    print(f"  Combined total   : {len(X_all)}")
    print(f"  Class dist       : class-0={np.sum(y_all==0)}  class-1={np.sum(y_all==1)}")
    print(f"  Input dimension  : {INPUT_DIM}\n")

    #Feature normalisation 
    # StandardScaler fitted on the merged dataset.
    # Mean and std are stored in the checkpoint for dependency-free inference.
    scaler     = StandardScaler()
    X_all_sc   = scaler.fit_transform(X_all).astype(np.float32)
    X_train_sc = scaler.transform(X_train).astype(np.float32)
    X_pre_sc   = scaler.transform(X_pre).astype(np.float32)

    #5-Fold Stratified CV (training data) 
    print("  Stage 1 -- 5-Fold Stratified Cross-Validation (train only)")
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_stats = {"acc": [], "auc": [], "f1": []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sc, y_train), 1):
        Xf, Xv = X_train_sc[tr_idx], X_train_sc[va_idx]
        yf, yv = y_train[tr_idx],    y_train[va_idx]

        tl     = make_loader(Xf, yf, BATCH_SIZE, oversample=True)
        vl     = make_loader(Xv, yv, 256)
        crit_f = build_criterion(yf)

        set_seed(SEED + fold)
        cv_model = GazeNet().to(DEVICE)
        train_model(cv_model, tl, vl, crit_f,
                    n_epochs=N_EPOCHS, patience=PATIENCE, verbose=False)

        mt = evaluate(cv_model, vl, crit_f)
        cv_stats["acc"].append(mt["acc"])
        cv_stats["auc"].append(mt["auc"])
        cv_stats["f1"] .append(mt["f1"])
        print(f"  Fold {fold}/5 -> acc={mt['acc']:.4f}  "
              f"auc={mt['auc']:.4f}  f1={mt['f1']:.4f}")

    print()
    for metric, vals in cv_stats.items():
        print(f"  CV {metric.upper():5s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    #Ensemble training on combined data 
    print("  Stage 2 -- Ensemble Training on Combined (Train + Pre-Test) Data")

    ensemble_models = []
    all_loader  = make_loader(X_all_sc, y_all, BATCH_SIZE, oversample=True)
    pre_loader  = make_loader(X_pre_sc,  y_pre,  256)
    crit_all    = build_criterion(y_all)
    history_all = []

    for i in range(N_FOLDS):
        print(f"\n  Ensemble member {i+1}/{N_FOLDS}  (seed={SEED + i})")
        set_seed(SEED + i)
        member  = GazeNet().to(DEVICE)
        history = train_model(member, all_loader, pre_loader, crit_all,
                              n_epochs=N_EPOCHS, patience=PATIENCE, verbose=True)
        ensemble_models.append(member)
        history_all.append(history)

    #Threshold optimisation 
    print("\n  Optimising decision threshold on combined training data ...")
    all_probs_list = [predict_proba(m, X_all_sc) for m in ensemble_models]
    ensemble_probs = np.mean(all_probs_list, axis=0)      # soft-vote
    best_threshold = find_best_threshold(ensemble_probs, y_all)
    print(f"  Threshold: {best_threshold:.4f}  "
          f"(train acc: "
          f"{accuracy_score(y_all, (ensemble_probs >= best_threshold).astype(int)):.4f})")

    #Evaluate ensemble on pre-test set 
    print("\n" + "─" * 68)
    print("  Ensemble Evaluation on Pre-Test Set")
    print("─" * 68)
    pre_probs_list = [predict_proba(m, X_pre_sc) for m in ensemble_models]
    pre_probs      = np.mean(pre_probs_list, axis=0)
    pre_preds      = (pre_probs >= best_threshold).astype(int)

    print(f"  Accuracy : {accuracy_score(y_pre, pre_preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_pre, pre_probs):.4f}")
    print(f"  F1 Score : {f1_score(y_pre, pre_preds):.4f}")
    print()
    print(classification_report(y_pre, pre_preds,
                                target_names=["Not Gazing (0)", "Gazing (1)"]))
    print("  Confusion Matrix:")
    print(f"  {confusion_matrix(y_pre, pre_preds)}")

#saving checkpoint
    print(f"\n  Saving checkpoint -> {WEIGHTS_PATH}")
    torch.save({
        "ensemble_states": [{k: v.detach().cpu() for k, v in m.state_dict().items()}
                            for m in ensemble_models],
        "input_dim":    INPUT_DIM,
        "hidden_dims":  HIDDEN_DIMS,
        "dropout":      DROPOUT,
        "scaler_mean":  torch.tensor(scaler.mean_,  dtype=torch.float32),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "threshold":    best_threshold,
        "n_ensemble":   N_FOLDS,
        "cv_acc_mean":  float(np.mean(cv_stats["acc"])),
        "cv_auc_mean":  float(np.mean(cv_stats["auc"])),
        "pretest_acc":  float(accuracy_score(y_pre, pre_preds)),
        "pretest_auc":  float(roc_auc_score(y_pre, pre_probs)),
    }, WEIGHTS_PATH)
    print(f"  Saved ({os.path.getsize(WEIGHTS_PATH)//1024} KB)")

    return ensemble_models, scaler, cv_stats, pre_probs, pre_preds, y_pre, best_threshold, history_all

#Inference entry point used by the marker; runs if test_input.csv is present and weights are available, otherwise runs the full training pipeline.

def run_inference(test_input_path=TEST_INPUT_CSV,
                  test_output_path=TEST_OUTPUT_CSV,
                  weights_path=WEIGHTS_PATH):
    print(f"Loading checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location=DEVICE)

    scaler_mean  = ckpt["scaler_mean"] .numpy()
    scaler_scale = ckpt["scaler_scale"].numpy()
    threshold    = ckpt["threshold"]
    n_ensemble   = ckpt["n_ensemble"]
    input_dim    = ckpt["input_dim"]
    hidden_dims  = ckpt["hidden_dims"]
    dropout      = ckpt["dropout"]

    print(f"  Ensemble size: {n_ensemble}  |  Threshold: {threshold:.4f}")

    # Reconstruct all ensemble members
    ensemble = []
    for state in ckpt["ensemble_states"]:
        m = GazeNet(input_dim, hidden_dims, dropout).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        ensemble.append(m)

    # Load and normalise test features
    print(f"Reading: {test_input_path}")
    X_test = load_unlabelled_csv(test_input_path)
    X_test = ((X_test - scaler_mean) / scaler_scale).astype(np.float32)
    print(f"  Samples: {len(X_test)}")

    # Soft-vote: average probabilities from all ensemble members
    probs_list = [predict_proba(m, X_test) for m in ensemble]
    probs      = np.mean(probs_list, axis=0)
    preds      = (probs >= threshold).astype(int)

    pd.DataFrame({"Y": preds}).to_csv(test_output_path, index=False)
    print(f"Predictions saved: {test_output_path}")
    print(f"  class-0 (not gazing): {int((preds == 0).sum())}")
    print(f"  class-1 (gazing)    : {int((preds == 1).sum())}")

#Entry point for the marker; runs inference if test_input.csv is present and weights are available, otherwise runs the full training pipeline.

if __name__ == "__main__":

    if os.path.exists(TEST_INPUT_CSV) and not os.path.exists(TRAIN_CSV):
        # Marker / inference mode: only test_input.csv + weights present
        print("Inference mode detected.")
        run_inference()

    else:
        # Full training mode: training data is present
        results = run_training()

        # Run inference immediately if test_input.csv is also present
        if os.path.exists(TEST_INPUT_CSV):
            print("  Running inference on test_input.csv")
            run_inference()