"""
Gradio demo – Chest X-ray Classifier

Supports any checkpoint produced by main.py (binary NORMAL/PNEUMONIA or a
multi-class model such as COVID19/NORMAL/PNEUMONIA/TUBERCULOSIS). The number
of output classes is inferred from the checkpoint.

Run:
    python app.py
    python app.py --model densenet121 --checkpoint outputs/best_model.pth
    python app.py --classes COVID19 NORMAL PNEUMONIA TUBERCULOSIS
"""

import os
import argparse

import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from main import build_model

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_CLASSES = {
    2: ["NORMAL", "PNEUMONIA"],
    4: ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"],
}


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

def infer_num_classes(state_dict: dict) -> int:
    """Read the output dimension of the last Linear layer in a state_dict."""
    last = None
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            last = v.shape[0]
    if last is None:
        raise RuntimeError("Could not infer num_classes from checkpoint.")
    return last


def load_model(model_name: str, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    num_classes = infer_num_classes(state)
    model = build_model(model_name, num_classes, freeze=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, num_classes


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
# Prediction
# ──────────────────────────────────────────────

def make_predict(model, classes):
    empty = {c: 0.0 for c in classes}

    def _predict(image: Image.Image):
        if image is None:
            return empty, "Please upload a chest X-ray image."
        try:
            tensor = preprocess_image(image)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

            result  = {c: float(p) for c, p in zip(classes, probs)}
            top_i   = int(np.argmax(probs))
            top_cls = classes[top_i]
            conf    = float(probs[top_i]) * 100

            if top_cls.upper() == "NORMAL":
                verdict = (
                    f"🟢 **NORMAL**\n\n"
                    f"Confidence: **{conf:.1f}%**\n\n"
                    f"The model does not detect abnormalities. "
                    f"Always confirm with a medical professional."
                )
            else:
                verdict = (
                    f"🔴 **{top_cls.upper()} DETECTED**\n\n"
                    f"Confidence: **{conf:.1f}%**\n\n"
                    f"The model predicts signs of {top_cls} in this chest X-ray. "
                    f"Please consult a qualified medical professional for diagnosis."
                )
            return result, verdict
        except Exception as e:
            return empty, f"Error during inference: {e}"

    return _predict


def missing_model_predict(checkpoint_path: str):
    def _predict(_image):
        return (
            {},
            f"⚠️ No trained model found at `{checkpoint_path}`. "
            f"Train first with `python main.py`."
        )
    return _predict


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def build_ui(predict_fn, classes, model_name: str, checkpoint_path: str,
             ready: bool):
    title = "Chest X-ray Classifier"
    class_list = ", ".join(classes) if classes else "unknown (no model loaded)"
    description = (
        f"# {title}\n\n"
        f"Transfer-learning model ({model_name}) classifying chest X-rays into: "
        f"**{class_list}**.\n\n"
        f"> ⚠️ **Disclaimer**: Educational purposes only — not a medical device. "
        f"Do not use for clinical decision-making."
    )

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(description)
        if not ready:
            gr.Markdown(
                f"⚠️ No checkpoint at `{checkpoint_path}`. "
                f"Run `python main.py` to train first."
            )

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
                    num_top_classes=min(max(len(classes), 1), 4),
                    label="Prediction Probabilities",
                )
                verdict_output = gr.Markdown(label="Verdict")

        predict_btn.click(
            predict_fn, inputs=image_input,
            outputs=[label_output, verdict_output],
        )
        image_input.change(
            predict_fn, inputs=image_input,
            outputs=[label_output, verdict_output],
        )

        gr.Markdown(
            f"---\n**Model**: {model_name} | "
            f"**Classes**: {len(classes)} | "
            f"**Framework**: PyTorch"
        )

    return demo


# ──────────────────────────────────────────────
# CLI + entrypoint
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chest X-ray Classifier demo")
    p.add_argument("--model", default="densenet121",
                   choices=["resnet50", "densenet121", "efficientnet_b0", "vgg16"],
                   help="Architecture used when the checkpoint was trained")
    p.add_argument("--checkpoint", default="outputs/best_model.pth",
                   help="Path to the trained .pth checkpoint")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Class names in training-index order. "
                        "Defaults: 2→NORMAL/PNEUMONIA, 4→COVID19/NORMAL/PNEUMONIA/TUBERCULOSIS")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    return p.parse_args()


def resolve_classes(num_classes: int, user_classes):
    if user_classes:
        if len(user_classes) != num_classes:
            raise ValueError(
                f"--classes has {len(user_classes)} names but model outputs "
                f"{num_classes} logits."
            )
        return list(user_classes)
    if num_classes in DEFAULT_CLASSES:
        return DEFAULT_CLASSES[num_classes]
    return [f"class_{i}" for i in range(num_classes)]


def main():
    args = parse_args()

    if os.path.exists(args.checkpoint):
        model, num_classes = load_model(args.model, args.checkpoint)
        classes = resolve_classes(num_classes, args.classes)
        print(f"Loaded {args.model} ({num_classes} classes: {classes}) "
              f"from {args.checkpoint} on {DEVICE}")
        predict_fn = make_predict(model, classes)
        ready = True
    else:
        print(f"Checkpoint not found at {args.checkpoint}. "
              f"Train first with: python main.py")
        classes = args.classes or DEFAULT_CLASSES[2]
        predict_fn = missing_model_predict(args.checkpoint)
        ready = False

    demo = build_ui(predict_fn, classes, args.model, args.checkpoint, ready)
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
