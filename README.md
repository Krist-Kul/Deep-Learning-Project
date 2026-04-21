# Pneumonia Detection from Chest X-ray Images
### Deep Learning Project · PyTorch · CUDA / MPS · Transfer Learning

> A deep learning project built with **PyTorch** using **transfer learning** on pretrained ImageNet CNNs (ResNet-50, DenseNet-121, EfficientNet-B0, VGG-16). Accelerated on **NVIDIA CUDA** or **Apple MPS** (falls back to CPU when neither is available).

---

## Quick Start

### Step 1 · Clone the repository

```bash
# SSH
git clone --recursive git@github.com:Krist-Kul/deep-learning-project.git

# HTTPS
git clone --recursive https://github.com/Krist-Kul/deep-learning-project.git

cd deep-learning-project
```

### Step 2 · Create the environment and install dependencies

```bash
conda create -n pneumonia-cnn python=3.10 -y
conda activate dl
pip install -r requirements.txt
```

> **Note:** For NVIDIA GPUs, install the CUDA-enabled PyTorch wheel first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```
> On Apple Silicon (M1/M2/M3) the default `pip install torch` already enables MPS.

### Step 3 · Run the program

You have two paths. Pick one.

#### Path A · Use the pre-trained model (fastest — no training needed)

1. Download the trained checkpoint:
   <https://drive.google.com/file/d/1diDFJoxht0NTGWxnR6anSKekc6D6Qfbl/view?usp=sharing>
2. Move it into `outputs/` and rename it to `best_model.pth` (if needed):
   ```
   outputs/best_model.pth
   ```
3. Launch the Gradio demo:
   ```bash
   python app.py
   ```
4. Open <http://localhost:7860> and upload a chest X-ray image.

#### Path B · Train from scratch

1. Download the dataset (requires Kaggle API credentials at `~/.kaggle/kaggle.json`):
   ```bash
   python data_download.py
   ```
2. Train:
   ```bash
   python main.py
   ```
3. After training completes, launch the demo or run evaluation:
   ```bash
   python app.py
   python main.py --eval_only
   ```

---

## Default Run

Running `python main.py` with no flags trains a **DenseNet-121** (ImageNet-pretrained, backbone frozen, custom classifier head = `Dropout(0.5) → Linear(1024, 2)`) and then runs a fine-tuning phase automatically.

| Hyperparameter | Default | Notes |
|----------------|---------|-------|
| `model_name` | `densenet121` | Backbone (ImageNet-pretrained) |
| `num_classes` | `2` | NORMAL / PNEUMONIA |
| `img_size` | `224` | Input resolution |
| `batch_size` | `32` | Mini-batch size |
| `num_epochs` | `20` | Max epochs (early stopping may cut short) |
| `learning_rate` | `1e-4` | AdamW lr (frozen-backbone phase) |
| `weight_decay` | `1e-4` | AdamW L2 regularisation |
| `step_size` / `gamma` | `7` / `0.1` | StepLR: decay lr ×0.1 every 7 epochs |
| `freeze_layers` | `True` | Backbone frozen during initial training |
| `early_stopping_patience` | `5` | Stop after 5 epochs without val-acc gain |
| `early_stopping_min_delta` | `1e-4` | Min val-acc improvement to reset patience |
| `finetune` | **`True`** | Unfreezes full backbone and retrains (default ON) |
| `finetune_epochs` | `10` | Fine-tuning epochs |
| `finetune_lr` | `1e-5` | Fine-tuning AdamW lr |
| `seed` | `42` | Reproducibility |

**Active-learning defaults** (only used with `--al`):

| Hyperparameter | Default |
|----------------|---------|
| `al_strategy` | `entropy` |
| `al_initial_size` | `200` |
| `al_query_size` | `100` |
| `al_rounds` | `5` |
| `al_epochs_per_round` | `10` |

Augmentations on the training set: `Resize → RandomCrop(224) → RandomHorizontalFlip → RandomRotation(10°) → ColorJitter(brightness=0.2, contrast=0.2) → Normalize(ImageNet mean/std)`. Val & test get only `Resize(224) → Normalize`.

Loss: **weighted cross-entropy** (weights computed from the training-set class frequencies to counter imbalance).

---

## CLI Arguments (`main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `densenet121` | `resnet50` / `densenet121` / `efficientnet_b0` / `vgg16` |
| `--epochs` | `20` | Max training epochs (early stopping may halt sooner) |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate |
| `--patience` | `5` | Early stopping patience (epochs without val improvement) |
| `--no_freeze` | False | Train all backbone layers from epoch 1 |
| `--eval_only` | False | Skip training; evaluate saved checkpoint on test set |
| `--finetune` / `--no-finetune` | True | Run fine-tuning phase after initial training (default on) |
| `--finetune_epochs` | `10` | Number of fine-tuning epochs |
| `--finetune_lr` | `1e-5` | Learning rate during fine-tuning |
| `--al` | False | Run pool-based active learning instead of full training |
| `--al_strategy` | `entropy` | `entropy` / `least_confidence` / `margin` / `random` |
| `--al_initial_size` | `200` | Initial labeled pool size |
| `--al_query_size` | `100` | Samples added per round |
| `--al_rounds` | `5` | Number of query rounds |
| `--al_epochs_per_round` | `10` | Training epochs per AL round |

**Examples:**

```bash
# Different backbone
python main.py --model resnet50 --finetune_epochs 10 --finetune_lr 5e-6

# Disable fine-tuning (it is on by default)
python main.py --no-finetune

# Active learning with entropy sampling
python main.py --al --al_strategy entropy \
               --al_initial_size 200 --al_query_size 100 --al_rounds 5

# Evaluate an existing checkpoint only
python main.py --eval_only
```

---

## Training Pipeline

```
Dataset (data/chest_xray)
        │
        ▼
Transforms & DataLoaders  (augment train / normalize val + test)
        │
        ▼
Transfer-learning model   (pretrained backbone + custom head)
        │
        ▼
Weighted Cross-Entropy    (class imbalance handled)
        │
        ▼
Initial training (frozen backbone)
        │
        ▼
Early stopping on val accuracy
  → saves outputs/best_model.pth when val acc improves
        │
        ▼
[Optional] Fine-tuning (--finetune)
  → unfreezes full backbone
  → small LR (finetune_lr)
        │
        ▼
[Optional] Active Learning (--active_learning)
  → seed → train → score pool → query top-k → repeat
        │
        ▼
Test set evaluation
  → classification report, ROC-AUC, F1
  → confusion matrix + ROC curve PNGs
  → test_metrics.json
```

Monitor with TensorBoard:
```bash
tensorboard --logdir outputs/runs
```

---

## Project Structure

```
deep-learning-project/
├── data_download.py    # Dataset downloader (run once before training)
├── main.py             # Training, early stopping, fine-tuning, active learning, evaluation
├── test.py             # Evaluate a checkpoint on an external dataset
├── app.py              # Gradio inference demo
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # Dataset (created by data_download.py)
│   └── chest_xray/
│       ├── train/{NORMAL,PNEUMONIA}/
│       ├── val/{NORMAL,PNEUMONIA}/
│       └── test/{NORMAL,PNEUMONIA}/
├── test/               # Optional external test images (for test.py)
└── outputs/            # Auto-created during training
    ├── best_model.pth
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── history.json
    ├── test_metrics.json
    ├── al_round*.pth
    ├── al_final.pth
    ├── al_history.json
    └── runs/           # TensorBoard logs
```

---

## Key Concepts

- **Transfer Learning** — reuse ImageNet-pretrained CNN features for medical imaging
- **CUDA / MPS acceleration** — automatic device selection (`cuda` → `mps` → `cpu`)
- **Feature Extraction vs Fine-tuning** — staged training, freeze backbone then unfreeze
- **Active Learning** — pool-based uncertainty sampling (entropy / least-confidence / margin) to train with fewer labels
- **Early Stopping** — halt when validation accuracy plateaus
- **Class Imbalance Handling** — weighted cross-entropy loss
- **Data Augmentation** — random crop, horizontal flip, rotation, color jitter
- **Evaluation Metrics** — accuracy, F1, ROC-AUC, confusion matrix
- **Learning Rate Scheduling** — StepLR decay
- **TensorBoard Logging** — real-time training visualisation

---

## Disclaimer

This project is for **educational purposes only**. The model is **not** a medical device, is **not validated for clinical use**, and must not be used to diagnose any medical condition. Always consult a qualified medical professional.

---

*Deep Learning Course Project · PyTorch · CUDA / MPS · Transfer Learning*
