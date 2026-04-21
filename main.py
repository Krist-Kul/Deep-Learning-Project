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
from torch.utils.data import DataLoader, Subset
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
    "model_name":   "densenet121",      # resnet50 | densenet121 | efficientnet_b0 | vgg16
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

    # Fine-tuning (runs after initial training by default)
    "finetune":           True,
    "finetune_epochs":    10,
    "finetune_lr":        1e-5,
    "finetune_unfreeze":  True,      # unfreeze full backbone for fine-tuning

    # Active learning (optional, replaces the normal training loop)
    "al":                    False,
    "al_rounds":             5,        # number of query/train rounds
    "al_initial_size":       200,      # samples in the initial labeled pool
    "al_query_size":         100,      # samples added per round
    "al_epochs_per_round":   10,        # epochs trained each round
    "al_strategy":           "entropy",# entropy | least_confidence | margin | random
}

CLASSES = ["NORMAL", "PNEUMONIA"]

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    pin = torch.cuda.is_available()   # pin_memory only helps on CUDA
    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["num_workers"],
            pin_memory=pin,
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
# Active Learning  (optional)
# ──────────────────────────────────────────────

def compute_uncertainty(model, dataloader, device, strategy: str) -> np.ndarray:
    """Return per-sample uncertainty scores; higher = more uncertain."""
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc=f"Query[{strategy}]", leave=False):
            inputs = inputs.to(device)
            probs = torch.softmax(model(inputs), dim=1)
            if strategy == "entropy":
                s = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            elif strategy == "least_confidence":
                s = 1.0 - probs.max(dim=1).values
            elif strategy == "margin":
                top2 = probs.topk(2, dim=1).values
                s = -(top2[:, 0] - top2[:, 1])
            elif strategy == "random":
                s = torch.rand(inputs.size(0), device=device)
            else:
                raise ValueError(f"Unknown AL strategy: {strategy}")
            scores.append(s.cpu())
    return torch.cat(scores).numpy()


def active_learning(model, train_dataset, score_dataset, val_loader,
                    val_size, device, cfg, writer):
    """
    Pool-based active learning on a fully-labeled training set.

    1. Split the training pool into an initial labeled subset (random) and an
       unlabeled pool.
    2. Each round: train the model on the current labeled set for
       `al_epochs_per_round` epochs.
    3. Score the unlabeled pool with `al_strategy` and move the top-k samples
       to the labeled set.
    4. Model state persists across rounds.
    """
    rng  = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(train_dataset))

    labeled = list(perm[:cfg["al_initial_size"]])
    pool    = list(perm[cfg["al_initial_size"]:])

    rounds   = cfg["al_rounds"]
    query_k  = cfg["al_query_size"]
    strategy = cfg["al_strategy"]

    history = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
        "al_round":   [], "al_labeled": [],
    }

    for rnd in range(rounds):
        print(f"\n── AL Round {rnd + 1}/{rounds}  "
              f"labeled={len(labeled)}, pool={len(pool)} ──")

        labeled_subset = Subset(train_dataset, labeled)
        loaders = {
            "train": DataLoader(labeled_subset, batch_size=cfg["batch_size"],
                                shuffle=True, num_workers=cfg["num_workers"],
                                pin_memory=torch.cuda.is_available()),
            "val":   val_loader,
        }
        sizes = {"train": len(labeled_subset), "val": val_size}

        # Class weights from current labeled subset (handles imbalance)
        subset_labels = [train_dataset.samples[i][1] for i in labeled]
        counts = np.bincount(subset_labels, minlength=cfg["num_classes"])
        w = 1.0 / np.maximum(counts, 1).astype(np.float32)
        w = w / w.sum() * len(counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32).to(device)
        )

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
        )

        round_cfg = {**cfg, "num_epochs": cfg["al_epochs_per_round"]}
        model, rnd_hist = train_model(
            model, loaders, sizes, criterion, optimizer, scheduler,
            device, round_cfg, writer, phase_prefix=f"[AL r{rnd + 1}] ",
        )

        for k in ("train_loss", "val_loss", "train_acc", "val_acc"):
            history[k].extend(rnd_hist[k])
        history["al_round"].append(rnd + 1)
        history["al_labeled"].append(len(labeled))

        writer.add_scalar("AL/labeled_size", len(labeled), rnd + 1)
        writer.add_scalar("AL/val_acc_round", rnd_hist["val_acc"][-1], rnd + 1)

        # Skip querying after the final round
        if rnd < rounds - 1 and pool:
            pool_subset = Subset(score_dataset, pool)
            pool_loader = DataLoader(
                pool_subset, batch_size=cfg["batch_size"], shuffle=False,
                num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available(),
            )
            scores   = compute_uncertainty(model, pool_loader, device, strategy)
            k        = min(query_k, len(pool))
            top_pos  = np.argsort(scores)[-k:]
            queried  = {pool[i] for i in top_pos}
            labeled.extend(queried)
            pool = [i for i in pool if i not in queried]
            print(f"  Queried {k} samples via '{strategy}'; "
                  f"labeled={len(labeled)}, pool={len(pool)}")

    print(f"\nActive learning complete. Final labeled size: {len(labeled)}")
    return model, history


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
    parser.add_argument("--finetune",   action=argparse.BooleanOptionalAction, default=True,
                        help="Run optional fine-tuning phase after initial training "
                             "(default: enabled; disable with --no-finetune)")
    parser.add_argument("--finetune_epochs", type=int, default=CONFIG["finetune_epochs"],
                        help="Number of fine-tuning epochs")
    parser.add_argument("--finetune_lr",     type=float, default=CONFIG["finetune_lr"],
                        help="Learning rate for fine-tuning")
    parser.add_argument("--al", action="store_true",
                        help="Run active learning instead of the normal training loop")
    parser.add_argument("--al_rounds", type=int, default=CONFIG["al_rounds"],
                        help="Number of active-learning rounds")
    parser.add_argument("--al_initial_size", type=int, default=CONFIG["al_initial_size"],
                        help="Initial labeled pool size")
    parser.add_argument("--al_query_size", type=int, default=CONFIG["al_query_size"],
                        help="Samples added per active-learning round")
    parser.add_argument("--al_epochs_per_round", type=int,
                        default=CONFIG["al_epochs_per_round"],
                        help="Epochs trained per active-learning round")
    parser.add_argument("--al_strategy", default=CONFIG["al_strategy"],
                        choices=["entropy", "least_confidence", "margin", "random"],
                        help="Uncertainty strategy for querying new samples")
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
    cfg["al"]                  = args.al
    cfg["al_rounds"]           = args.al_rounds
    cfg["al_initial_size"]     = args.al_initial_size
    cfg["al_query_size"]       = args.al_query_size
    cfg["al_epochs_per_round"] = args.al_epochs_per_round
    cfg["al_strategy"]         = args.al_strategy

    if args.al and args.eval_only:
        raise ValueError("--al cannot be combined with --eval_only")

    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("  Apple Silicon GPU (Metal Performance Shaders)")

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
        if cfg["al"]:
            # ── Active Learning ──
            _, al_val_tf = get_transforms(cfg["img_size"])
            score_dataset = datasets.ImageFolder(data_root / "train", al_val_tf)
            print(f"Active learning: strategy={cfg['al_strategy']}, "
                  f"rounds={cfg['al_rounds']}, init={cfg['al_initial_size']}, "
                  f"query={cfg['al_query_size']}, epochs/round={cfg['al_epochs_per_round']}")
            model, history = active_learning(
                model, dataloaders["train"].dataset, score_dataset,
                dataloaders["val"], dataset_sizes["val"],
                device, cfg, writer,
            )
        else:
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
