"""Flask frontend for lung cancer classification and conditional sample generation."""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, url_for
from PIL import Image
import torch
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier import SimpleCNN  # noqa: E402
from data import build_transforms, discover_class_names  # noqa: E402
from gan import ConditionalGenerator  # noqa: E402


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "LC25000 lung histopathology subset"
DATA_TRAIN_DIR = PROJECT_ROOT / "data" / "lung_cancer" / "train"
CLASSIFIER_PATH = PROJECT_ROOT / "checkpoints" / "lung_classifier.pth"
GENERATOR_PATH = PROJECT_ROOT / "checkpoints" / "lung_dcgan.pth"
METRICS_PATH = PROJECT_ROOT / "checkpoints" / "lung_classifier_metrics.json"
STATIC_GENERATED_DIR = Path(__file__).resolve().parent / "static" / "generated"
STATIC_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

_classifier_cache: tuple[SimpleCNN, dict] | None = None
_generator_cache: tuple[ConditionalGenerator, dict] | None = None


def checkpoint_status(path: Path) -> dict:
    exists = path.exists()
    return {
        "exists": exists,
        "path": path,
        "size_mb": path.stat().st_size / (1024 * 1024) if exists else None,
    }


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    report = metrics.get("classification_report", {})
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": report.get("macro avg", {}).get("f1-score"),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
    }


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_classifier() -> tuple[SimpleCNN, dict]:
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {CLASSIFIER_PATH}")

    checkpoint = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
    class_names = checkpoint.get("class_names") or discover_class_names(DATA_TRAIN_DIR)
    model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    metadata = {
        "class_names": class_names,
        "image_size": checkpoint.get("image_size", 128),
    }
    _classifier_cache = (model, metadata)
    return _classifier_cache


def load_generator() -> tuple[ConditionalGenerator, dict]:
    global _generator_cache
    if _generator_cache is not None:
        return _generator_cache
    if not GENERATOR_PATH.exists():
        raise FileNotFoundError(f"Missing GAN checkpoint: {GENERATOR_PATH}")

    checkpoint = torch.load(GENERATOR_PATH, map_location=DEVICE)
    class_names = checkpoint.get("class_names") or discover_class_names(DATA_TRAIN_DIR)
    model = ConditionalGenerator(
        nz=checkpoint.get("nz", 100),
        ngf=checkpoint.get("ngf", 64),
        nc=3,
        num_classes=len(class_names),
        image_size=checkpoint.get("image_size", 128),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    metadata = {
        "class_names": class_names,
        "image_size": checkpoint.get("image_size", 128),
        "nz": checkpoint.get("nz", 100),
    }
    _generator_cache = (model, metadata)
    return _generator_cache


@app.route("/")
def index():
    class_names = discover_class_names(DATA_TRAIN_DIR)
    return render_template(
        "home.html",
        active_page="home",
        dataset_name=DATASET_NAME,
        data_train_dir=DATA_TRAIN_DIR,
        device=DEVICE,
        class_names=class_names,
        classifier_status=checkpoint_status(CLASSIFIER_PATH),
        generator_status=checkpoint_status(GENERATOR_PATH),
        metrics=load_metrics(),
    )


@app.route("/classify", methods=["GET", "POST"])
def classify():
    result = None
    error = None
    preview = None
    class_names = discover_class_names(DATA_TRAIN_DIR)
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Choose an image before submitting."
        else:
            try:
                model, metadata = load_classifier()
                image = Image.open(file.stream).convert("RGB")
                preview = image_to_data_url(image.copy())
                transform = build_transforms(
                    image_size=metadata["image_size"],
                    augment=False,
                    normalize=True,
                )
                tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(tensor)
                    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

                class_names = metadata["class_names"]
                ranked = sorted(
                    [
                        {"class_name": class_names[idx], "probability": float(prob)}
                        for idx, prob in enumerate(probabilities)
                    ],
                    key=lambda row: row["probability"],
                    reverse=True,
                )
                result = {"prediction": ranked[0]["class_name"], "probabilities": ranked}
            except Exception as exc:  # Keep the UI usable when checkpoints are missing.
                error = str(exc)
    return render_template(
        "classify.html",
        active_page="classify",
        result=result,
        error=error,
        preview=preview,
        class_names=class_names,
        dataset_name=DATASET_NAME,
        classifier_status=checkpoint_status(CLASSIFIER_PATH),
        metrics=load_metrics(),
        device=DEVICE,
    )


@app.route("/generate", methods=["GET", "POST"])
def generate():
    error = None
    image_url = None
    selected_class = request.form.get("class_name", "all")
    class_names = discover_class_names(DATA_TRAIN_DIR)
    try:
        generator, metadata = load_generator()
        class_names = metadata["class_names"]
        if selected_class != "all" and selected_class not in class_names:
            selected_class = "all"
        if request.method == "POST":
            with torch.no_grad():
                if selected_class == "all":
                    labels = (torch.arange(36, device=DEVICE) % len(class_names)).long()
                    nrow = 6
                else:
                    class_idx = class_names.index(selected_class)
                    labels = torch.full((36,), class_idx, dtype=torch.long, device=DEVICE)
                    nrow = 6
                noise = torch.randn(labels.size(0), metadata["nz"], 1, 1, device=DEVICE)
                images = generator(noise, labels).cpu()
            output_file = STATIC_GENERATED_DIR / "latest_samples.png"
            save_image(images, output_file, normalize=True, nrow=nrow)
            image_url = url_for("static", filename="generated/latest_samples.png")
    except Exception as exc:
        error = str(exc)

    return render_template(
        "generate.html",
        active_page="generate",
        error=error,
        image_url=image_url,
        class_names=class_names,
        selected_class=selected_class,
        dataset_name=DATASET_NAME,
        generator_status=checkpoint_status(GENERATOR_PATH),
        device=DEVICE,
    )


if __name__ == "__main__":
    app.run(debug=True)
