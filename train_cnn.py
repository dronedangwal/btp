"""
train_cnn.py
────────────────────────────────────────────────────────────────────────────
CNN classifier for CSI activity recognition.

Expected directory layout (both train_dir and test_dir):
    <train_dir>/
        A/
            labels.csv          <- columns: second, label
            2.png
            3.png
            ...
        B/
            labels.csv
            ...

Labels recognised: Still, Typing, Scrolling, Flipping
Rows with label "Ignore" or "Unlabeled" are automatically skipped.

Usage
─────
    python train_cnn.py
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DIR   = Path("harsh_model_data/train_data")
TEST_DIR    = Path("harsh_model_data/val_data")
OUTPUT_DIR  = Path("harsh_model_data/output2")           # checkpoints, plots, reports

LABELS      = ["Still", "Typing", "Scrolling", "Flipping"]
SKIP_LABELS = {"Ignore", "Unlabeled", "Unknown"}

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-3
LR_STEP     = 10                       # decay LR every N epochs
LR_GAMMA    = 0.5                      # LR decay factor
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                        # DataLoader workers
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Loads spectrogram PNGs matched to per-second labels.

    Each sub-directory of `root_dir` is expected to contain:
      - labels.csv  (columns: second, label)
      - PNG files named  <N>.png

    Rows whose label is in SKIP_LABELS are excluded.
    """

    def __init__(
        self,
        root_dir: Path,
        label2idx: dict[str, int],
        transform=None,
    ):
        self.root_dir  = Path(root_dir)
        self.label2idx = label2idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []   # (image_path, class_idx)

        self._build_index()

    def _build_index(self):
        missing_imgs = 0
        skipped_rows = 0

        for rec_dir in sorted(self.root_dir.iterdir()):
            if not rec_dir.is_dir():
                continue

            label_csv = rec_dir / "labels.csv"
            if not label_csv.exists():
                print(f"  [warn] no labels.csv in {rec_dir.name}, skipping")
                continue

            labels_df = pd.read_csv(label_csv)

            # Build a map: second (int) → label string
            sec2label: dict[int, str] = dict(
                zip(labels_df["second"].astype(int), labels_df["label"].astype(str))
            )

            # Build a map: second (int) → PNG path
            sec2img: dict[int, Path] = {}
            for img_path in rec_dir.glob("*.png"):
                m = re.search(r"\d+", img_path.name)
                if m:
                    sec2img[int(m.group())] = img_path

            for second, label in sec2label.items():
                if label in SKIP_LABELS:
                    skipped_rows += 1
                    continue
                if label not in self.label2idx:
                    skipped_rows += 1
                    continue
                if second not in sec2img:
                    missing_imgs += 1
                    continue

                self.samples.append((sec2img[second], self.label2idx[label]))

        print(f"  Loaded {len(self.samples)} samples from {self.root_dir.name}")
        if skipped_rows:
            print(f"    → {skipped_rows} rows skipped (Ignore/Unlabeled/unknown label)")
        if missing_imgs:
            print(f"    → {missing_imgs} rows skipped (PNG not found)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label_idx


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms():
    """
    Train: light augmentation (colour jitter handles CSI amplitude variability across environments).
    Val/Test: deterministic resize + normalise only.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_tf, eval_tf


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """We use pretrained MobileNet and fine tune it."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Replace classifier head only
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Class-weighted loss ───────────────────────────────────────────────────────

def compute_class_weights(dataset: SpectrogramDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weighting to handle class imbalance."""
    counts = torch.zeros(num_classes)
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    # avoid division by zero for unseen classes
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_training_curves(history: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"],
    ):
        for key, label in zip(metric, ["Train", "Val"]):
            ax.plot(history[key], label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved training curves → {path}")


def plot_confusion_matrix(
    y_true, y_pred, class_names: list[str], output_dir: Path
):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Fraction")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=11)

    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix  → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label2idx = {lbl: i for i, lbl in enumerate(LABELS)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    num_classes = len(LABELS)

    print(f"\n{'═'*58}")
    print(f"  Device : {DEVICE}")
    print(f"  Classes: {LABELS}")
    print(f"{'═'*58}\n")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_tf, eval_tf = get_transforms()

    print("Building datasets ...")
    train_dataset = SpectrogramDataset(TRAIN_DIR, label2idx, transform=train_tf)
    test_dataset  = SpectrogramDataset(TEST_DIR,  label2idx, transform=eval_tf)

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found in {TRAIN_DIR}")
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test samples found in {TEST_DIR}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Class distribution summary
    print("\nClass distribution (train):")
    counts = pd.Series([idx2label[l] for _, l in train_dataset.samples]).value_counts()
    for lbl, cnt in counts.items():
        print(f"  {lbl:<12} {cnt:>5} samples")

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model = build_model(num_classes).to(DEVICE)

    # class_weights = compute_class_weights(train_dataset, num_classes).to(DEVICE)
    # criterion     = nn.CrossEntropyLoss(weight=class_weights)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler     = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {NUM_EPOCHS} epochs ...\n")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    best_ckpt     = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_ckpt)
            flag = "  ← best"

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}"
              f"  train loss={tr_loss:.4f}  acc={tr_acc:.3f}"
              f"  │  val loss={va_loss:.4f}  acc={va_acc:.3f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}{flag}")

    # ── Final evaluation on best checkpoint ───────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  Loading best checkpoint (val acc={best_val_acc:.3f}) ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Test accuracy: {test_acc:.4f}\n")

    report = classification_report(
        all_labels, all_preds,
        target_names=LABELS,
        digits=4,
    )
    print(report)

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(report)
    print(f"  Saved report            → {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history, OUTPUT_DIR)
    plot_confusion_matrix(all_labels, all_preds, LABELS, OUTPUT_DIR)

    print(f"\n{'═'*58}")
    print(f"  All outputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
