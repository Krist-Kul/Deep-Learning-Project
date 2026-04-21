"""
Evaluate a trained checkpoint on a new dataset.

Default layout expected:
    ./test/test/<class_name>/*.jpg

Usage:
    python test.py
    python test.py --data ./test/test --checkpoint outputs/best_model.pth
    python test.py --model densenet121 --data ./test/test
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
)
from tqdm import tqdm

from main import build_model, get_transforms, CONFIG


def infer_num_classes(state_dict: dict) -> int:
    """Infer output dimension from the last Linear layer in a state_dict."""
    last_linear_out = None
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            last_linear_out = v.shape[0]
    if last_linear_out is None:
        raise RuntimeError("Could not infer num_classes from checkpoint.")
    return last_linear_out


def build_binary_target_map(dataset_classes, model_classes):
    """
    Map dataset class indices to model class indices.

    Rule: class name 'NORMAL' (case-insensitive) -> model index for NORMAL,
    every other dataset class -> the remaining model index (PNEUMONIA-like).
    Only used when the dataset has more classes than the model.
    """
    model_idx = {name.upper(): i for i, name in enumerate(model_classes)}
    if "NORMAL" not in model_idx:
        raise RuntimeError(
            f"Binary remap expects a NORMAL class in the model; got {model_classes}"
        )
    normal_idx = model_idx["NORMAL"]
    other_idx = 1 - normal_idx  # assumes 2-class model
    return [
        normal_idx if name.upper() == "NORMAL" else other_idx
        for name in dataset_classes
    ]


def evaluate(model, dataloader, device, class_names, out_dir: Path,
             label_remap=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            if label_remap is not None:
                labels = torch.tensor(
                    [label_remap[int(y)] for y in labels], dtype=torch.long
                )
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    print("\n── Classification Report ──────────────────────")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    num_classes = len(class_names)

    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"ROC-AUC Score: {auc:.4f}")
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "test_roc_curve.png", dpi=150)
        plt.close()
        print(f"ROC curve saved          → {out_dir / 'test_roc_curve.png'}")
        metrics["auc"] = auc
        metrics["f1"] = f1_score(all_labels, all_preds, zero_division=0)
    else:
        metrics["f1_macro"] = f1_score(
            all_labels, all_preds, average="macro", zero_division=0
        )

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "test_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {out_dir / 'test_confusion_matrix.png'}")

    metrics["accuracy"] = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_names"] = list(class_names)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Test trained model on a new dataset")
    p.add_argument("--data", default="./test/test",
                   help="Path to an ImageFolder-style dataset root")
    p.add_argument("--checkpoint", default=CONFIG["checkpoint"],
                   help="Path to model .pth checkpoint")
    p.add_argument("--model", default=CONFIG["model_name"],
                   choices=["resnet50", "densenet121", "efficientnet_b0", "vgg16"])
    p.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    p.add_argument("--img_size", type=int, default=CONFIG["img_size"])
    p.add_argument("--output_dir", default=CONFIG["output_dir"])
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Data root  : {data_root}")
    print(f"Checkpoint : {ckpt_path}")

    _, eval_tf = get_transforms(args.img_size)
    dataset = datasets.ImageFolder(data_root, eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset classes: {dataset.classes}  (n={len(dataset)})")

    state_dict = torch.load(ckpt_path, map_location=device)
    num_classes = infer_num_classes(state_dict)
    print(f"Model output classes (from checkpoint): {num_classes}")

    model = build_model(args.model, num_classes, freeze=False)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Decide evaluation labels
    label_remap = None
    if len(dataset.classes) == num_classes:
        class_names = dataset.classes
    elif num_classes == 2 and len(dataset.classes) > 2:
        # Binary model vs. multi-class dataset → remap to NORMAL / non-NORMAL
        model_classes = ["NORMAL", "PNEUMONIA"]
        label_remap = build_binary_target_map(dataset.classes, model_classes)
        class_names = model_classes
        print(
            "Dataset has more classes than the model — remapping to binary:\n"
            + "\n".join(
                f"  {orig:15s} → {model_classes[label_remap[i]]}"
                for i, orig in enumerate(dataset.classes)
            )
        )
    else:
        raise RuntimeError(
            f"Class count mismatch: dataset has {len(dataset.classes)} classes "
            f"({dataset.classes}) but model has {num_classes}. "
            "No automatic remap available for this combination."
        )

    out_dir = Path(args.output_dir)
    metrics = evaluate(model, loader, device, class_names, out_dir,
                       label_remap=label_remap)

    metrics_path = out_dir / "new_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
