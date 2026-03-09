# Transfer Learning for Pneumonia Detection from Chest X-ray Images
### Using Deep Convolutional Neural Networks · PyTorch · CUDA

---

## Overview

This project applies **transfer learning** with pretrained CNNs (ResNet-50, DenseNet-121, EfficientNet-B0, VGG-16) to classify chest X-ray images as **NORMAL** or **PNEUMONIA**. It is designed as an educational deep learning project demonstrating best practices for medical image classification.

| Component | Description |
|-----------|-------------|
| `data_download.py` | Standalone dataset downloader — saves to `data/chest_xray/` |
| `main.py`  | Full training, early stopping, fine-tuning & evaluation pipeline |
| `app.py`   | Gradio web demo for inference |
| `main.ipynb` | Interactive notebook walkthrough |
| `requirements.txt` | Python dependencies |
| `data/` | Downloaded dataset (auto-created by `data_download.py`) |
| `outputs/` | Saved checkpoints, plots, metrics (auto-created during training) |

---

## Dataset

**Chest X-Ray Images (Pneumonia)** – Paul Mooney, Kaggle
<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

| Split | NORMAL | PNEUMONIA |
|-------|--------|-----------|
| Train | 1,341  | 3,875     |
| Val   | 8      | 8         |
| Test  | 234    | 390       |

The dataset is downloaded once via `data_download.py` and stored at `data/chest_xray/` inside the repository (symlinked from the kagglehub cache to avoid duplicating disk usage).

---

## Architecture

```
ImageNet Pretrained Backbone  ──►  Frozen Feature Extractor  ──►  Custom Head
         │                                                             │
   ResNet-50 / DenseNet-121                                  Dropout(0.5)
   EfficientNet-B0 / VGG-16                                  Linear → 2 classes
```

- **Loss**: Cross-Entropy with class weights (handles class imbalance)
- **Optimiser**: AdamW
- **Scheduler**: StepLR
- **Augmentation**: Random crop, horizontal flip, rotation, colour jitter
- **Early Stopping**: halts training when val accuracy stops improving
- **Fine-tuning** (optional): unfreezes full backbone after initial training

---

## Environment Setup

### 1 · Create & activate conda environment

```bash
conda create -n pneumonia-cnn python=3.10 -y
conda activate pneumonia-cnn
```

### 2 · Install PyTorch (CUDA 11.8 – adjust for your CUDA version)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> Check your CUDA version: `nvidia-smi`
> Find the right command at: <https://pytorch.org/get-started/locally/>

### 3 · Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 · Configure Kaggle API

```bash
# Place your kaggle.json token in:
~/.kaggle/kaggle.json          # Linux / macOS
%USERPROFILE%\.kaggle\kaggle.json   # Windows

chmod 600 ~/.kaggle/kaggle.json    # Linux / macOS only
```

Get your token from: <https://www.kaggle.com/settings/account> → *Create New Token*

---

## Usage

### Step 1 · Download the dataset

Run this **once** before training. Safe to re-run — skips download if already present.

```bash
python data_download.py
```

The dataset is placed at `data/chest_xray/` with the following structure:

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Step 2 · Train

```bash
python main.py
```

**Common options:**

```bash
# Change backbone model
python main.py --model densenet121

# Custom epochs and batch size
python main.py --epochs 30 --batch_size 16

# Adjust early stopping patience (default: 5 epochs)
python main.py --patience 8

# Run optional fine-tuning phase after initial training
python main.py --finetune

# Fine-tuning with custom settings
python main.py --finetune --finetune_epochs 10 --finetune_lr 5e-6

# Unfreeze all backbone layers from the start (no feature extraction phase)
python main.py --no_freeze

# Skip training, evaluate saved checkpoint on test set
python main.py --eval_only
```

**All CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `resnet50` | `resnet50` / `densenet121` / `efficientnet_b0` / `vgg16` |
| `--epochs` | `20` | Max training epochs (may stop earlier via early stopping) |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate |
| `--patience` | `5` | Early stopping patience (epochs without improvement) |
| `--no_freeze` | False | Train all backbone layers from epoch 1 |
| `--eval_only` | False | Evaluate checkpoint without training |
| `--finetune` | False | Run fine-tuning phase after initial training |
| `--finetune_epochs` | `5` | Number of fine-tuning epochs |
| `--finetune_lr` | `1e-5` | Learning rate during fine-tuning |

### Training pipeline

```
Initial training (frozen backbone)
        │
        ▼
  Early stopping monitors val accuracy
  → saves best checkpoint automatically
  → halts if no improvement for `patience` epochs
        │
        ▼
[Optional] Fine-tuning (--finetune)
  → unfreezes full backbone
  → trains with lower LR (finetune_lr)
  → early stopping applied here too
        │
        ▼
Test set evaluation + plots
```

### Monitor training with TensorBoard

```bash
tensorboard --logdir outputs/runs
```

### Run the Gradio demo

```bash
# Download data and train first, then:
python app.py
# Open http://localhost:7860
```

### Open the notebook

```bash
jupyter notebook main.ipynb
```

The notebook mirrors the full pipeline with interactive cells:

| Section | Content |
|---------|---------|
| 1 | Imports & device setup |
| 2 | Dataset download (`data_download.py`) |
| 3 | Dataset exploration & class distribution |
| 4 | Data transforms & loaders |
| 5 | Visualise augmented training batch |
| 6 | Build transfer learning model |
| 7 | Training with early stopping |
| 8 | Training curves |
| 9 | Fine-tuning *(optional — set `ENABLE_FINETUNE = True`)* |
| 10 | Test set evaluation |
| 11 | Confusion matrix & ROC curve |
| 12 | Single-image inference |
| 13 | Save model checkpoint |

---

## Outputs

After training, the `outputs/` directory contains:

```
outputs/
├── best_model.pth          # Best checkpoint (by val accuracy)
├── training_history.png    # Loss & accuracy curves (covers fine-tuning if run)
├── confusion_matrix.png    # Test set confusion matrix
├── roc_curve.png           # ROC curve with AUC score
├── history.json            # Raw training metrics (JSON)
├── test_metrics.json       # Test AUC, F1, confusion matrix (JSON)
└── runs/                   # TensorBoard logs
```

---

## Results (expected baseline – ResNet-50, 20 epochs)

| Metric | Value |
|--------|-------|
| Test Accuracy | ~92–95% |
| ROC-AUC | ~0.96–0.98 |
| F1-score (Pneumonia) | ~0.94–0.96 |

*Actual results vary with GPU, random seed, and hyperparameters.*

---

## Project Structure

```
deep-learning-project/
├── data_download.py    # Dataset downloader (run once before training)
├── main.py             # Training, early stopping, fine-tuning & evaluation
├── app.py              # Gradio inference demo
├── main.ipynb          # Notebook walkthrough
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # Dataset directory (created by data_download.py)
│   └── chest_xray/
└── outputs/            # Auto-created during training
```

---

## Key Concepts Demonstrated

- **Transfer Learning** – reusing ImageNet features for medical imaging
- **Feature Extraction vs Fine-tuning** – staged training with `--finetune`
- **Early Stopping** – automatic halt when validation accuracy plateaus
- **Class Imbalance Handling** – weighted cross-entropy loss
- **Data Augmentation** – preventing overfitting on small datasets
- **Evaluation Metrics** – accuracy, F1, ROC-AUC, confusion matrix
- **Learning Rate Scheduling** – StepLR decay
- **TensorBoard Logging** – real-time training visualisation

---

## Disclaimer

This project is for **educational purposes only**. The model is not validated for clinical use and should not be used to diagnose any medical condition.

---

*Deep Learning Course Project · PyTorch · CUDA · Transfer Learning*
