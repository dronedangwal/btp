"""
Training script for CSI gesture recognition using csi_network_inc_res.

Data layout expected:
  data_model/
    train_antennas_still,scroll,flip,type/   <- directory of <n>.txt sample files
    labels_train_antennas_still,scroll,flip,type.txt
    val_antennas_still,scroll,flip,type/
    labels_val_antennas_still,scroll,flip,type.txt
    test_antennas_still,scroll,flip,type/
    labels_test_antennas_still,scroll,flip,type.txt

Each <n>.txt is a 2-D matrix (rows = Doppler bins, cols = time steps).
Labels file: numpy array (allow_pickle=True) with integer labels 0-3.

Outputs (all saved to RESULTS_DIR):
  best_model.keras               -- best checkpoint by val_accuracy
  metrics.txt                    -- accuracy, precision, recall, F1 (per-class + macro)
  metrics.json                   -- same data, machine-readable
  confusion_matrix.png           -- normalised confusion matrix heatmap
  loss_curves.png                -- train vs val loss
  accuracy_curves.png            -- train vs val accuracy
  confidence_distribution.png   -- softmax confidence for correct vs incorrect
"""

import argparse
import os
import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from network_utility import csi_network_inc_res

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT       = "data_model67"
SUFFIX          = "antennas_still,scroll,flip,type"
CLASS_NAMES     = ["still", "scroll", "flip", "type"]
NUM_CLASSES     = len(CLASS_NAMES)
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
RESULTS_DIR     = "results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.keras")
# ---------------------------------------------------------------------------


def load_split(split: str):
    """Load all samples and labels for one split (train / val / test)."""
    sample_dir = os.path.join(DATA_ROOT, f"{split}_{SUFFIX}")
    label_file = os.path.join(DATA_ROOT, f"labels_{split}_{SUFFIX}.txt")

    labels = np.array(np.load(label_file, allow_pickle=True))

    txt_files = sorted(
        glob.glob(os.path.join(sample_dir, "*.txt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )

    if len(txt_files) != len(labels):
        raise ValueError(
            f"[{split}] Found {len(txt_files)} sample files but "
            f"{len(labels)} labels -- they must match."
        )

    samples = []
    for f in txt_files:
        s = np.load(f, allow_pickle=True).astype(np.float32)
        s = np.squeeze(s)
        if s.ndim == 1:
            s = s[np.newaxis, :]
        samples.append(s)

    data = np.stack(samples, axis=0).astype(np.float32)
    print(f"[{split}] data: {data.shape}, labels: {labels.shape}")
    return data, labels


def make_dataset(data: np.ndarray, labels: np.ndarray, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data), reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, out_dir: str):
    """Save loss and accuracy learning curves as separate PNGs."""
    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["loss"],     label="Train")
    ax.plot(epochs, history["val_loss"], label="Val", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved loss_curves.png")

    # Accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["accuracy"],     label="Train")
    ax.plot(epochs, history["val_accuracy"], label="Val", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Validation Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved accuracy_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names: list, out_dir: str):
    """Save a row-normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = range(len(class_names))
    ax.set_xticks(list(ticks)); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(list(ticks)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion_matrix.png")


def plot_confidence_distribution(y_true, y_pred, confidences, out_dir: str):
    """
    Histogram of max-softmax confidence split by correct / incorrect predictions.
    A well-calibrated model should be confident when right, uncertain when wrong.
    """
    correct   = confidences[y_true == y_pred]
    incorrect = confidences[y_true != y_pred]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(correct,   bins=bins, alpha=0.65, label=f"Correct (n={len(correct)})",
            color="steelblue", edgecolor="white")
    ax.hist(incorrect, bins=bins, alpha=0.65, label=f"Incorrect (n={len(incorrect)})",
            color="tomato",    edgecolor="white")
    ax.set_xlabel("Confidence (max softmax probability)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_distribution.png"), dpi=150)
    plt.close(fig)
    print("Saved confidence_distribution.png")


def save_metrics(y_true, y_pred, out_dir: str):
    """Write accuracy, per-class and macro precision/recall/F1 to txt and json."""
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    lines = [
        "=" * 54,
        f"  Overall accuracy : {acc:.4f}",
        "=" * 54,
        f"{'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>9}",
        "-" * 54,
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(
            f"{name:<12} {precision[i]:>7.4f} {recall[i]:>7.4f} "
            f"{f1[i]:>7.4f} {int(support[i]):>9}"
        )
    lines += [
        "-" * 54,
        f"{'macro avg':<12} {macro_p:>7.4f} {macro_r:>7.4f} {macro_f1:>7.4f}",
        f"{'weighted avg':<12} {weighted_p:>7.4f} {weighted_r:>7.4f} {weighted_f1:>7.4f}",
        "=" * 54,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as fh:
        fh.write(report + "\n")
    print("Saved metrics.txt")

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "accuracy": float(acc),
            "per_class": {
                name: {
                    "precision": float(precision[i]),
                    "recall":    float(recall[i]),
                    "f1":        float(f1[i]),
                    "support":   int(support[i]),
                }
                for i, name in enumerate(CLASS_NAMES)
            },
            "macro":    {"precision": float(macro_p),    "recall": float(macro_r),    "f1": float(macro_f1)},
            "weighted": {"precision": float(weighted_p), "recall": float(weighted_r), "f1": float(weighted_f1)},
        }, fh, indent=2)
    print("Saved metrics.json")

def parse_args():
    p = argparse.ArgumentParser(description="Model training")
    p.add_argument("--data_dir",   required=True,
                   help="Root directory which contains training, validation and testing samples and labels")
    p.add_argument("--results_dir",  required=True,
                   help="Directory in which the model behavior metrics are noted")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--activity_tags",      type=str, default="still,scroll,flip,type")
    p.add_argument("--suffix",      type=str, default="antennas_still,scroll,flip,type")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global DATA_ROOT, SUFFIX, CLASS_NAMES, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, RESULTS_DIR, BEST_MODEL_PATH
    DATA_ROOT       = args.data_dir
    SUFFIX          = args.suffix
    CLASS_NAMES     = args.activity_tags.split(",")
    NUM_CLASSES     = len(CLASS_NAMES)
    BATCH_SIZE      = args.batch_size
    EPOCHS          = args.epochs
    LEARNING_RATE   = args.lr
    RESULTS_DIR     = args.results_dir
    BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.keras")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # Normalise with train statistics only (no leakage into val/test)
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0,  keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    assert X_train.ndim == 4, (
        f"Expected 4D array (N, H, W, 1), got {X_train.shape}. "
        "Check that each .txt file is a plain 2-D matrix."
    )

    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")

    # ------------------------------------------------------------------
    # 2. Datasets
    # ------------------------------------------------------------------
    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   shuffle=False)
    test_ds  = make_dataset(X_test,  y_test,  shuffle=False)

    # ------------------------------------------------------------------
    # 3. Build & compile
    # ------------------------------------------------------------------
    model = csi_network_inc_res(input_shape, NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # ------------------------------------------------------------------
    # 4. Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=10, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max",
            factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_training_curves(history.history, RESULTS_DIR)

    # ------------------------------------------------------------------
    # 6. Evaluate best model on test set
    # ------------------------------------------------------------------
    print(f"\nLoading best model from '{BEST_MODEL_PATH}'...")
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss : {test_loss:.4f}")
    print(f"Test acc  : {test_acc:.4f}")

    # Logits -> softmax probabilities -> predicted class + confidence
    logits      = best_model.predict(test_ds, verbose=0)
    probs       = tf.nn.softmax(logits).numpy()
    y_pred      = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)

    # ------------------------------------------------------------------
    # 7. Save all metrics and plots
    # ------------------------------------------------------------------
    save_metrics(y_test, y_pred, RESULTS_DIR)
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, RESULTS_DIR)
    plot_confidence_distribution(y_test, y_pred, confidences, RESULTS_DIR)

    print(f"\nAll results saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
