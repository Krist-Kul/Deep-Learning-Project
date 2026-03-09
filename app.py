"""
Gradio demo – Pneumonia Detection from Chest X-ray Images
Run:  python app.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import models, transforms

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

CLASSES       = ["NORMAL", "PNEUMONIA"]
CHECKPOINT    = "outputs/best_model.pth"
MODEL_NAME    = "resnet50"          # must match what was trained
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

def load_model(model_name: str, checkpoint_path: str, num_classes: int = 2):
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return preprocess(pil_img).unsqueeze(0).to(DEVICE)


# ──────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────

model_loaded = False
model        = None

if os.path.exists(CHECKPOINT):
    try:
        model        = load_model(MODEL_NAME, CHECKPOINT)
        model_loaded = True
        print(f"Model loaded from {CHECKPOINT} on {DEVICE}")
    except Exception as e:
        print(f"Warning: could not load checkpoint – {e}")
else:
    print(f"Checkpoint not found at {CHECKPOINT}. Train first with: python main.py")


# ──────────────────────────────────────────────
# Prediction function
# ──────────────────────────────────────────────

def predict(image: Image.Image):
    if not model_loaded:
        return (
            {"NORMAL": 0.0, "PNEUMONIA": 0.0},
            "⚠️ No trained model found. Run `python main.py` first to train the model."
        )

    if image is None:
        return {"NORMAL": 0.0, "PNEUMONIA": 0.0}, "Please upload a chest X-ray image."

    try:
        tensor = preprocess_image(image)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        result     = {cls: float(p) for cls, p in zip(CLASSES, probs)}
        pred_class = CLASSES[int(np.argmax(probs))]
        confidence = float(np.max(probs)) * 100

        if pred_class == "PNEUMONIA":
            verdict = (
                f"🔴 **PNEUMONIA DETECTED**\n\n"
                f"Confidence: **{confidence:.1f}%**\n\n"
                f"The model predicts signs of pneumonia in this chest X-ray. "
                f"Please consult a qualified medical professional for diagnosis."
            )
        else:
            verdict = (
                f"🟢 **NORMAL**\n\n"
                f"Confidence: **{confidence:.1f}%**\n\n"
                f"The model does not detect signs of pneumonia. "
                f"Always confirm with a medical professional."
            )

        return result, verdict

    except Exception as e:
        return {"NORMAL": 0.0, "PNEUMONIA": 0.0}, f"Error during inference: {e}"


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

DESCRIPTION = """
# Pneumonia Detection from Chest X-ray Images

This demo uses a **Transfer Learning** model (ResNet-50 pretrained on ImageNet,
fine-tuned on the [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset) to classify chest X-rays as **NORMAL** or **PNEUMONIA**.

> ⚠️ **Disclaimer**: This tool is for educational purposes only and is **not** a
> medical device. Do not use it for clinical decision-making.
"""

with gr.Blocks(title="Pneumonia Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-ray Image",
                height=350,
            )
            predict_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                num_top_classes=2,
                label="Prediction Probabilities",
            )
            verdict_output = gr.Markdown(label="Verdict")

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, verdict_output],
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, verdict_output],
    )

    gr.Examples(
        examples=[],          # add sample image paths here after training
        inputs=image_input,
    )

    gr.Markdown(
        "---\n"
        "**Model**: ResNet-50 | **Dataset**: Chest X-Ray Images (Pneumonia) – Kaggle | "
        "**Framework**: PyTorch"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
