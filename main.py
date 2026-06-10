

"""
Title: Leave-One-Subject-Out (LOSO) Cross-Validation for SigWavNet
Description: Trains and evaluates the SigWavNet dysarthria-severity model on the
TORGO dataset using Leave-One-Subject-Out cross-validation. Every speaker (the
``source`` column) is held out as the test set exactly once while the model is
trained on the remaining speakers. The folds are enumerated with scikit-learn's
``LeaveOneGroupOut`` and orchestrated sequentially with joblib (``n_jobs=1``) so
that only one fold uses the single available GPU at a time.

For every fold we log accuracy, precision, recall, F1 (macro and weighted) and a
confusion matrix annotated with the held-out subject. After all folds finish we
log the aggregated metrics, the overall (pooled) confusion matrix and a
subject-level accuracy table.
"""

import os
import json
import copy
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe: write figures to disk instead of a display

from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import *          # utility functions, classes, and globals (incl. `device`, `severityclasses`)
from model import *          # model definition
from custom_layers import *  # custom layer definitions


# Data loading runs in the main process (see utils.py for the rationale).
num_workers = 0
pin_memory = False

# Root folder for all LOSO experiment artefacts.
torgo_experiments_folder = "experiments"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(log_dir):
    """
    Creates a logger that writes both to ``loso.log`` inside ``log_dir`` and to
    the console.

    Parameters:
    - log_dir (str): directory in which to create the log file.

    Returns:
    - A configured ``logging.Logger`` instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("loso")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()       # avoid duplicate handlers on re-runs
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(log_dir, "loso.log"), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


# -----------------------------------------------------------------------------
# Configuration / model factory
# -----------------------------------------------------------------------------
def build_config(severityclasses):
    """Fixed configuration for a single LOSO trial."""
    wt = pywt.Wavelet('db10')  # wavelet used to initialise the learnable kernels
    return {
        "n_input": 1,
        "hidden_dim": 32,
        "n_layers": 3,
        "n_output": len(severityclasses),
        "weight_decay": 1e-4,
        "level": 4,
        "trainHT": True,
        "initHT": 0.8,
        "kernTrainable": True,
        "kernelInit": np.array(wt.filter_bank[0]),
        "alpha": 10,
        "mode": "PerLayer",
        "lr": 1e-4,
        "batch_size": 4,
        "num_splits": 5,
    }


def build_model(config):
    """Instantiates a SigWavNet model from a configuration dictionary."""
    return SigWavNet(
        n_input=config['n_input'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_output=config['n_output'],
        inputSize=None,
        kernelInit=config['kernelInit'],
        kernTrainable=config['kernTrainable'],
        level=config['level'],
        kernelsConstraint=config['mode'],
        initHT=config['initHT'],
        trainHT=config['trainHT'],
        alpha=config['alpha'],
    )


def _class_weights(labels):
    """
    Inverse-frequency class weights aligned to the global ``severityclasses``
    order, so the focal-loss alpha vector always has ``n_output`` entries even
    when a training fold happens to be missing a class.
    """
    total = len(labels)
    weights = []
    for cls in severityclasses:
        count = int(np.sum(np.asarray(labels) == cls))
        weights.append(total / count if count > 0 else 0.0)
    return weights


# -----------------------------------------------------------------------------
# Training (one fold)
# -----------------------------------------------------------------------------
def train_SigWavNet(config, data, max_num_epochs, device, logger, checkpoint_path):
    """
    Trains the SigWavNet model for a single LOSO fold.

    A stratified train/validation split of the *training* speakers (the first
    fold of an internal StratifiedKFold) is used to monitor training; the model
    state with the lowest validation loss is kept and returned.

    Parameters:
    - config: configuration dictionary for the model and training.
    - data: training DataFrame (path/label/source columns) for this fold.
    - max_num_epochs: number of epochs to train.
    - device: torch device to train on.
    - logger: logger to report progress to.
    - checkpoint_path: where to persist the best model for this fold.

    Returns:
    - The trained model loaded with its best (lowest validation loss) weights.
    """

    criterion = FocalLoss(alpha=torch.FloatTensor(_class_weights(data['label'])), gamma=2)

    model = build_model(config).to(device)

    optimiser = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)

    # Single stratified train/validation split of the training speakers.
    train_loader, val_loader = get_dataloaders(
        data, batch_size=config["batch_size"], num_splits=config["num_splits"])[0]

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, max_num_epochs + 1):

        # ---- Training ----
        model.train()
        right = 0
        h = model.init_hidden(config["batch_size"])
        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            h = [i.data for i in h]  # detach hidden states

            output, h = model(inputs, h)
            right += nr_of_right(get_probable_idx(output), target)

            loss = criterion(output.squeeze(), target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        train_acc = 100. * right / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        right = 0
        val_loss = 0.0
        val_steps = 0
        h = model.init_hidden(config["batch_size"])
        with torch.no_grad():
            for inputs, target in val_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                h = [i.data for i in h]

                output, h = model(inputs, h)
                right += nr_of_right(get_probable_idx(output), target)

                val_loss += criterion(output.squeeze(), target).item()
                val_steps += 1

        val_acc = 100. * right / len(val_loader.dataset)
        val_loss = val_loss / max(val_steps, 1)
        scheduler.step()

        logger.info(
            f"    epoch {epoch:>3}/{max_num_epochs} | "
            f"train acc {train_acc:5.1f}% | val loss {val_loss:.4f} | val acc {val_acc:5.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"    best model saved to {checkpoint_path} (val loss {best_val_loss:.4f})")

    return model


# -----------------------------------------------------------------------------
# Evaluation (one fold)
# -----------------------------------------------------------------------------
def evaluate(model, batch_size, data, train_mean, train_std, device):
    """
    Evaluates the model on a held-out test DataFrame.

    Normalisation uses the *training* statistics (passed in) to avoid peeking at
    the held-out speaker, and the resampling matches the training pipeline
    (16 kHz -> 8 kHz). The GRU hidden state is re-initialised per batch using the
    actual batch size, so no test samples are dropped (``drop_last=False``).

    Returns:
    - Tuple of (y_true, y_pred) as lists of class indices.
    """
    model.eval()

    transform = MyTransformPipeline(
        input_freq=16000, resample_freq=8000, train_mean=train_mean, train_std=train_std)
    transform.to(device)

    test_set = MyDataset(
        data['path'].reset_index(drop=True),
        data['label'].reset_index(drop=True),
        transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            # Independent utterances: start from a zeroed hidden state sized to
            # this batch (handles a smaller final batch gracefully).
            h = model.init_hidden(inputs.size(0))
            output, _ = model(inputs, h)

            pred = get_probable_idx(output)
            y_true.extend(np.atleast_1d(target.cpu().numpy()).tolist())
            y_pred.extend(np.atleast_1d(pred.view(-1).cpu().numpy()).tolist())

    return y_true, y_pred


# -----------------------------------------------------------------------------
# Metrics / artefact helpers
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Computes accuracy plus macro/weighted precision, recall and F1."""
    labels = list(range(len(severityclasses)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, out_path, title):
    """Saves an integer-count confusion matrix heatmap over all severity classes."""
    labels = list(range(len(severityclasses)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure(figsize=(8, 6))
    df_cm = pd.DataFrame(cm, index=severityclasses, columns=severityclasses)
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def log_confusion_matrix(logger, cm, prefix="    "):
    """Logs a confusion matrix as an aligned text table."""
    header = "true\\pred".ljust(12) + "".join(c.rjust(12) for c in severityclasses)
    logger.info(prefix + header)
    for i, cls in enumerate(severityclasses):
        row = cls.ljust(12) + "".join(str(int(v)).rjust(12) for v in cm[i])
        logger.info(prefix + row)


# -----------------------------------------------------------------------------
# One LOSO fold (the unit of work handed to joblib)
# -----------------------------------------------------------------------------
def run_loso_fold(fold_idx, n_folds, train_ds, test_ds, subject,
                  config, max_num_epochs, device, run_dir, logger):
    """
    Runs a single LOSO fold: train on ``train_ds`` and evaluate on ``test_ds``
    (all recordings of the held-out ``subject``). Logs and persists per-fold
    artefacts and returns a result dictionary.
    """
    test_label = sorted(test_ds['label'].unique())
    logger.info("=" * 80)
    logger.info(
        f"FOLD {fold_idx}/{n_folds} | held-out subject: {subject} "
        f"| severity: {test_label} | train={len(train_ds)} utts / "
        f"{train_ds['source'].nunique()} subjects | test={len(test_ds)} utts"
    )

    # Normalisation statistics from the training speakers only (no test leakage).
    train_mean, train_std = compute_precise_mean_std(train_ds['path'])

    checkpoint_path = os.path.join(run_dir, "checkpoints", f"SigWavNet_fold{fold_idx:02d}_{subject}.pt")
    model = train_SigWavNet(config, train_ds, max_num_epochs, device, logger, checkpoint_path)

    y_true, y_pred = evaluate(model, config["batch_size"], test_ds, train_mean, train_std, device)

    metrics = compute_metrics(y_true, y_pred)
    metrics["subject"] = subject
    metrics["n_test"] = len(y_true)

    # Per-fold confusion matrix (annotated with the held-out subject).
    cm_path = os.path.join(run_dir, "confusion_matrices", f"cm_fold{fold_idx:02d}_{subject}.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    cm = save_confusion_matrix(
        y_true, y_pred, cm_path, title=f"Fold {fold_idx} - held-out {subject}")

    # Per-fold predictions.
    pred_df = pd.DataFrame({
        "subject": subject,
        "path": test_ds['path'].reset_index(drop=True),
        "true": [index_to_severity(i) for i in y_true],
        "pred": [index_to_severity(i) for i in y_pred],
    })
    pred_path = os.path.join(run_dir, "predictions", f"pred_fold{fold_idx:02d}_{subject}.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    # ---- Per-fold logging ----
    logger.info(
        f"  [FOLD {fold_idx} | {subject}] "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"precision(macro/weighted)={metrics['precision_macro']:.4f}/{metrics['precision_weighted']:.4f} | "
        f"recall(macro/weighted)={metrics['recall_macro']:.4f}/{metrics['recall_weighted']:.4f} | "
        f"f1(macro/weighted)={metrics['f1_macro']:.4f}/{metrics['f1_weighted']:.4f}"
    )
    logger.info(f"  [FOLD {fold_idx} | {subject}] classification report:\n" +
                classification_report(
                    y_true, y_pred,
                    labels=list(range(len(severityclasses))),
                    target_names=severityclasses,
                    zero_division=0))
    logger.info(f"  [FOLD {fold_idx} | {subject}] confusion matrix (saved to {cm_path}):")
    log_confusion_matrix(logger, cm)

    # Subject-level accuracy at the end of this trial (each fold == one subject).
    logger.info(
        f"  [FOLD {fold_idx} | {subject}] SUBJECT-LEVEL ACCURACY: "
        f"{metrics['accuracy'] * 100:.2f}% ({int(round(metrics['accuracy'] * len(y_true)))}/{len(y_true)})"
    )

    return {
        "fold": fold_idx,
        "subject": subject,
        "severity": test_label[0] if len(test_label) == 1 else "/".join(test_label),
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def main(max_num_epochs=5, n_jobs=1):
    """
    Runs LOSO cross-validation over every speaker in the TORGO severity dataset.

    Parameters:
    - max_num_epochs: number of epochs to train per fold.
    - n_jobs: joblib worker count. Kept at 1 by default because the folds share a
      single GPU (sequential execution in the main process keeps CUDA happy).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(torgo_experiments_folder, f"loso_{timestamp}")
    logger = setup_logger(run_dir)

    logger.info(f"Using device: {device}")
    logger.info(f"Run directory: {os.path.abspath(run_dir)}")

    # Load data (TORGO dysarthria severity).
    data, fold_classes = load_data(TORGO_ROOT)
    config = build_config(fold_classes)

    logger.info(
        f"SigWavNet LOSO | severity classes: {severityclasses} | "
        f"n_output: {config['n_output']} | total utterances: {len(data)} | "
        f"subjects: {sorted(data['source'].unique())}"
    )

    # LeaveOneGroupOut: one fold per speaker (the `source`/subject column).
    logo = LeaveOneGroupOut()
    groups = data['source'].values
    splits = list(logo.split(data, data['label'], groups=groups))
    n_folds = len(splits)
    logger.info(f"LeaveOneGroupOut produced {n_folds} folds (one per subject).")

    # Build the per-fold (train, test, subject) payloads up front.
    fold_payloads = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_ds = data.iloc[train_idx].reset_index(drop=True)
        test_ds = data.iloc[test_idx].reset_index(drop=True)
        subject = test_ds['source'].iloc[0]
        fold_payloads.append((fold_idx, train_ds, test_ds, subject))

    # Sequential orchestration with joblib (single GPU -> n_jobs=1).
    logger.info(f"Orchestrating {n_folds} folds with joblib (n_jobs={n_jobs}).")
    results = Parallel(n_jobs=n_jobs, backend="sequential")(
        delayed(run_loso_fold)(
            fold_idx, n_folds, train_ds, test_ds, subject,
            config, max_num_epochs, device, run_dir, logger,
        )
        for (fold_idx, train_ds, test_ds, subject) in fold_payloads
    )

    # -------------------------------------------------------------------------
    # Aggregate across folds
    # -------------------------------------------------------------------------
    results = sorted(results, key=lambda r: r["fold"])

    pooled_true, pooled_pred = [], []
    subject_accuracy = {}
    per_fold_rows = []
    for r in results:
        pooled_true.extend(r["y_true"])
        pooled_pred.extend(r["y_pred"])
        subject_accuracy[r["subject"]] = r["metrics"]["accuracy"]
        per_fold_rows.append({
            "fold": r["fold"],
            "subject": r["subject"],
            "severity": r["severity"],
            "n_test": r["metrics"]["n_test"],
            **{k: r["metrics"][k] for k in (
                "accuracy", "precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted")},
        })

    overall = compute_metrics(pooled_true, pooled_pred)

    # Macro-average over subjects (each subject weighted equally), with std.
    fold_acc = np.array([r["metrics"]["accuracy"] for r in results])
    fold_f1m = np.array([r["metrics"]["f1_macro"] for r in results])

    logger.info("#" * 80)
    logger.info("LOSO CROSS-VALIDATION SUMMARY")
    logger.info("#" * 80)

    logger.info(
        "POOLED (micro, over all utterances): "
        f"accuracy={overall['accuracy']:.4f} | "
        f"precision(macro/weighted)={overall['precision_macro']:.4f}/{overall['precision_weighted']:.4f} | "
        f"recall(macro/weighted)={overall['recall_macro']:.4f}/{overall['recall_weighted']:.4f} | "
        f"f1(macro/weighted)={overall['f1_macro']:.4f}/{overall['f1_weighted']:.4f}"
    )
    logger.info(
        "MEAN OVER SUBJECTS (each subject weighted equally): "
        f"accuracy={fold_acc.mean():.4f} +/- {fold_acc.std():.4f} | "
        f"f1_macro={fold_f1m.mean():.4f} +/- {fold_f1m.std():.4f}"
    )

    # Subject-level accuracy table.
    logger.info("-" * 80)
    logger.info("SUBJECT-LEVEL ACCURACY")
    logger.info("  " + "subject".ljust(10) + "severity".ljust(12) + "n_test".rjust(8) + "accuracy".rjust(12))
    for r in results:
        logger.info(
            "  " + r["subject"].ljust(10) + str(r["severity"]).ljust(12) +
            str(r["metrics"]["n_test"]).rjust(8) +
            f"{r['metrics']['accuracy'] * 100:.2f}%".rjust(12)
        )

    # Pooled confusion matrix (over all held-out subjects).
    logger.info("-" * 80)
    pooled_cm_path = os.path.join(run_dir, "confusion_matrices", "cm_pooled.png")
    pooled_cm = save_confusion_matrix(
        pooled_true, pooled_pred, pooled_cm_path, title="LOSO pooled confusion matrix")
    logger.info(f"POOLED CONFUSION MATRIX (saved to {pooled_cm_path}):")
    log_confusion_matrix(logger, pooled_cm)

    logger.info("POOLED classification report:\n" +
                classification_report(
                    pooled_true, pooled_pred,
                    labels=list(range(len(severityclasses))),
                    target_names=severityclasses,
                    zero_division=0))

    # -------------------------------------------------------------------------
    # Persist tabular + JSON summaries
    # -------------------------------------------------------------------------
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_csv = os.path.join(run_dir, "per_fold_metrics.csv")
    per_fold_df.to_csv(per_fold_csv, index=False)

    # Combined predictions across all folds.
    combined = pd.concat([
        pd.DataFrame({
            "fold": r["fold"],
            "subject": r["subject"],
            "true": [index_to_severity(i) for i in r["y_true"]],
            "pred": [index_to_severity(i) for i in r["y_pred"]],
        }) for r in results
    ], ignore_index=True)
    combined.to_csv(os.path.join(run_dir, "all_predictions.csv"), index=False)

    summary = {
        "timestamp": timestamp,
        "device": str(device),
        "max_num_epochs": max_num_epochs,
        "n_folds": n_folds,
        "severity_classes": severityclasses,
        "pooled_metrics": overall,
        "mean_over_subjects": {
            "accuracy_mean": float(fold_acc.mean()),
            "accuracy_std": float(fold_acc.std()),
            "f1_macro_mean": float(fold_f1m.mean()),
            "f1_macro_std": float(fold_f1m.std()),
        },
        "subject_accuracy": subject_accuracy,
        "pooled_confusion_matrix": pooled_cm.tolist(),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("-" * 80)
    logger.info(f"Saved per-fold metrics -> {per_fold_csv}")
    logger.info(f"Saved summary          -> {os.path.join(run_dir, 'summary.json')}")
    logger.info("LOSO cross-validation complete.")

    return summary


if __name__ == "__main__":
    main(max_num_epochs=5, n_jobs=1)
