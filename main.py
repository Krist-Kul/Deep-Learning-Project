"""
Transfer Learning for Pneumonia Detection from Chest X-ray Images
using Deep Convolutional Neural Networks

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Run data_download.py first to fetch the dataset:
    python data_download.py
"""

import os
import time
import copy
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from tqdm import tqdm

from data_download import DEST_PATH as DEFAULT_DATA_ROOT

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CONFIG = {
    "model_name":   "resnet50",      # resnet50 | densenet121 | efficientnet_b0 | vgg16
    "num_classes":  2,
    "batch_size":   32,
    "num_epochs":   20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "momentum":     0.9,
    "step_size":    7,               # LR scheduler step
    "gamma":        0.1,             # LR decay factor
    "img_size":     224,
    "num_workers":  4,
    "freeze_layers": True,           # freeze pretrained backbone
    "output_dir":   "outputs",
    "checkpoint":   "outputs/best_model.pth",
    "seed":         42,

    # Early stopping
    "early_stopping_patience":  5,   # epochs to wait before stopping
    "early_stopping_min_delta": 1e-4,# minimum improvement to reset patience

    # Fine-tuning (optional, runs after initial training)
    "finetune":           False,
    "finetune_epochs":    5,
    "finetune_lr":        1e-5,
    "finetune_unfreeze":  True,      # unfreeze full backbone for fine-tuning
}

CLASSES = ["NORMAL", "PNEUMONIA"]

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Data Transforms & Loaders
# ──────────────────────────────────────────────

def get_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def get_dataloaders(data_root: Path, cfg: dict):
    train_tf, val_tf = get_transforms(cfg["img_size"])

    image_datasets = {
        "train": datasets.ImageFolder(data_root / "train", train_tf),
        "val":   datasets.ImageFolder(data_root / "val",   val_tf),
        "test":  datasets.ImageFolder(data_root / "test",  val_tf),
    }

    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        for split, ds in image_datasets.items()
    }

    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names   = image_datasets["train"].classes
    print(f"Classes : {class_names}")
    print(f"Sizes   : {dataset_sizes}")
    return dataloaders, dataset_sizes, class_names


# ──────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, freeze: bool) -> nn.Module:
    """Return a pretrained model with replaced classification head."""
    weights_map = {
        "resnet50":        models.ResNet50_Weights.IMAGENET1K_V2,
        "densenet121":     models.DenseNet121_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "vgg16":           models.VGG16_Weights.IMAGENET1K_V1,
    }

    if model_name == "resnet50":
        model = models.resnet50(weights=weights_map[model_name])
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=weights_map[model_name])
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_map[model_name])
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "vgg16":
        model = models.vgg16(weights=weights_map[model_name])
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model   : {model_name}  |  trainable params: {trainable:,} / {total:,}")
    return model


# ──────────────────────────────────────────────
# Training  (with early stopping)
# ──────────────────────────────────────────────

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                scheduler, device, cfg, writer, phase_prefix=""):
    """
    Train for up to cfg['num_epochs'] epochs with early stopping.

    Early stopping monitors val accuracy and halts when there has been no
    improvement >= early_stopping_min_delta for early_stopping_patience epochs.
    """
    best_weights      = copy.deepcopy(model.state_dict())
    best_acc          = 0.0
    history           = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    patience          = cfg.get("early_stopping_patience", 5)
    min_delta         = cfg.get("early_stopping_min_delta", 1e-4)
    epochs_no_improve = 0

    for epoch in range(cfg["num_epochs"]):
        print(f"\n{phase_prefix}Epoch {epoch + 1}/{cfg['num_epochs']}  {'─' * 40}")

        early_stop = False

        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            running_loss     = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=phase, leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                loop.set_postfix(loss=loss.item())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            tb_step = epoch   # caller may offset this externally if needed
            writer.add_scalar(f"{phase_prefix}Loss/{phase}", epoch_loss, tb_step)
            writer.add_scalar(f"{phase_prefix}Accuracy/{phase}", epoch_acc, tb_step)

            print(f"  {phase:5s}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc >= best_acc + min_delta:
                    best_acc          = epoch_acc
                    best_weights      = copy.deepcopy(model.state_dict())
                    torch.save(best_weights, cfg["checkpoint"])
                    print(f"  ✓ Best model saved  (val acc = {best_acc:.4f})")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
                    if epochs_no_improve >= patience:
                        print(f"\n  Early stopping triggered after {patience} "
                              f"epochs without improvement.")
                        early_stop = True

        if early_stop:
            break

    print(f"\n{phase_prefix}Training complete. Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model, history


# ──────────────────────────────────────────────
# Fine-tuning  (optional)
# ──────────────────────────────────────────────

def finetune_model(model, dataloaders, dataset_sizes, criterion,
                   device, cfg, writer, base_history: dict):
    """
    Unfreeze the full backbone and train with a much smaller learning rate.
    Returns the model and merged history dict.
    """
    print("\n── Fine-tuning ─────────────────────────────────────────────")
    print(f"  Unfreezing all layers and training with lr={cfg['finetune_lr']:.0e}")

    if cfg.get("finetune_unfreeze", True):
        for p in model.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    ft_cfg = {
        **cfg,
        "num_epochs": cfg["finetune_epochs"],
        "learning_rate": cfg["finetune_lr"],
    }

    ft_optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["finetune_lr"],
        weight_decay=cfg["weight_decay"],
    )
    ft_scheduler = optim.lr_scheduler.StepLR(
        ft_optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
    )

    model, ft_history = train_model(
        model, dataloaders, dataset_sizes,
        criterion, ft_optimizer, ft_scheduler,
        device, ft_cfg, writer, phase_prefix="[FT] ",
    )

    # Merge histories so plots cover the full run
    merged = {k: base_history[k] + ft_history[k] for k in base_history}
    return model, merged


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(model, dataloader, device, class_names, out_dir: str):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    print("\n── Classification Report ──────────────────────")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {out_dir}/confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=150)
    plt.close()
    print(f"ROC curve saved          → {out_dir}/roc_curve.png")

    return {
        "auc": auc,
        "f1":  f1_score(all_labels, all_preds),
        "confusion_matrix": cm.tolist(),
    }


# ──────────────────────────────────────────────
# Plotting History
# ──────────────────────────────────────────────

def plot_history(history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_history.png", dpi=150)
    plt.close()
    print(f"Training history saved   → {out_dir}/training_history.png")


# ──────────────────────────────────────────────
# Compute class weights (handle imbalance)
# ──────────────────────────────────────────────

def compute_class_weights(dataset, device):
    labels = [s[1] for s in dataset.samples]
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pneumonia Detection via Transfer Learning"
    )
    parser.add_argument("--model",      default=CONFIG["model_name"],
                        choices=["resnet50", "densenet121", "efficientnet_b0", "vgg16"])
    parser.add_argument("--epochs",     type=int,   default=CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--no_freeze",  action="store_true",
                        help="Train all layers (no feature freezing)")
    parser.add_argument("--eval_only",  action="store_true",
                        help="Skip training; evaluate checkpoint on test set")
    parser.add_argument("--patience",   type=int,   default=CONFIG["early_stopping_patience"],
                        help="Early stopping patience (epochs)")
    parser.add_argument("--finetune",   action="store_true",
                        help="Run optional fine-tuning phase after initial training")
    parser.add_argument("--finetune_epochs", type=int, default=CONFIG["finetune_epochs"],
                        help="Number of fine-tuning epochs")
    parser.add_argument("--finetune_lr",     type=float, default=CONFIG["finetune_lr"],
                        help="Learning rate for fine-tuning")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides
    cfg = CONFIG.copy()
    cfg["model_name"]          = args.model
    cfg["num_epochs"]          = args.epochs
    cfg["batch_size"]          = args.batch_size
    cfg["learning_rate"]       = args.lr
    cfg["freeze_layers"]       = not args.no_freeze
    cfg["early_stopping_patience"] = args.patience
    cfg["finetune"]            = args.finetune
    cfg["finetune_epochs"]     = args.finetune_epochs
    cfg["finetune_lr"]         = args.finetune_lr

    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──
    data_root = DEFAULT_DATA_ROOT
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_root}. Run `python data_download.py` first."
        )
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_root, cfg)

    # ── Model ──
    model = build_model(cfg["model_name"], cfg["num_classes"], cfg["freeze_layers"])
    model = model.to(device)

    # ── Loss (weighted for class imbalance) ──
    class_weights = compute_class_weights(
        dataloaders["train"].dataset, device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser & Scheduler ──
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
    )

    writer = SummaryWriter(log_dir=f"{cfg['output_dir']}/runs")

    if not args.eval_only:
        # ── Train ──
        model, history = train_model(
            model, dataloaders, dataset_sizes,
            criterion, optimizer, scheduler,
            device, cfg, writer,
        )

        # ── Fine-tuning (optional) ──
        if cfg["finetune"]:
            model, history = finetune_model(
                model, dataloaders, dataset_sizes,
                criterion, device, cfg, writer, history,
            )

        plot_history(history, cfg["output_dir"])
        with open(f"{cfg['output_dir']}/history.json", "w") as f:
            json.dump(history, f, indent=2)
    else:
        print(f"Loading checkpoint: {cfg['checkpoint']}")
        model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))

    # ── Evaluate on test set ──
    print("\n── Test Set Evaluation ─────────────────────────")
    metrics = evaluate(
        model, dataloaders["test"], device,
        class_names, cfg["output_dir"]
    )
    with open(f"{cfg['output_dir']}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    writer.close()
    print("\nDone. All outputs saved in:", cfg["output_dir"])


if __name__ == "__main__":
    main()
