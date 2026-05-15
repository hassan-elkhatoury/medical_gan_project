"""Flask frontend for COVID-19 radiography classification and sample generation."""

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
DATASET_NAME = "COVID-19 Radiography Database"
DATA_TRAIN_DIR = PROJECT_ROOT / "data" / "covid_radiography" / "train"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CLASSIFIER_PATHS = [
    CHECKPOINT_DIR / "covid_classifier.pth",
    CHECKPOINT_DIR / "lung_classifier.pth",
]
GENERATOR_PATHS = [
    CHECKPOINT_DIR / "covid_dcgan.pth",
    CHECKPOINT_DIR / "generator.pth",
    CHECKPOINT_DIR / "lung_dcgan.pth",
]
METRICS_PATHS = [
    CHECKPOINT_DIR / "covid_classifier_metrics.json",
    CHECKPOINT_DIR / "lung_classifier_metrics.json",
]
STATIC_GENERATED_DIR = Path(__file__).resolve().parent / "static" / "generated"
STATIC_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
DATASET_SAMPLE_FILES = [
    {
        "file": "dataset_samples/sample_01.jpg",
        "label": "Dataset X-ray 01",
        "note": "public COVID-19 radiography sample",
    },
    {
        "file": "dataset_samples/sample_02.jpg",
        "label": "Dataset X-ray 02",
        "note": "public COVID-19 radiography sample",
    },
    {
        "file": "dataset_samples/sample_03.jpg",
        "label": "Dataset X-ray 03",
        "note": "public COVID-19 radiography sample",
    },
]

_classifier_cache: tuple[SimpleCNN, dict] | None = None
_generator_cache: tuple[ConditionalGenerator, dict] | None = None


def resolve_checkpoint_path(paths: list[Path], kind: str) -> Path:
    for path in paths:
        if path.exists():
            return path

    expected = "\n".join(f"  - {path}" for path in paths)
    available_paths = sorted(CHECKPOINT_DIR.glob("*.pth")) if CHECKPOINT_DIR.exists() else []
    available = "\n".join(f"  - {path}" for path in available_paths) or "  - no .pth files found"
    raise FileNotFoundError(
        f"Missing {kind} checkpoint.\n"
        f"Expected one of:\n{expected}\n"
        f"Available:\n{available}"
    )


def checkpoint_status(paths: list[Path]) -> dict:
    path = next((candidate for candidate in paths if candidate.exists()), paths[0])
    exists = path.exists()
    return {
        "exists": exists,
        "path": path,
        "filename": path.name,
        "expected_names": [candidate.name for candidate in paths],
        "size_mb": path.stat().st_size / (1024 * 1024) if exists else None,
    }


def load_metrics() -> dict | None:
    metrics_path = next((path for path in METRICS_PATHS if path.exists()), None)
    if metrics_path is None:
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    report = metrics.get("classification_report", {})
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": report.get("macro avg", {}).get("f1-score"),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
        "filename": metrics_path.name,
    }


def dataset_samples() -> list[dict]:
    samples = []
    static_root = Path(app.static_folder or "")
    for sample in DATASET_SAMPLE_FILES:
        if (static_root / sample["file"]).exists():
            samples.append(
                {
                    **sample,
                    "url": url_for("static", filename=sample["file"]),
                }
            )
    return samples


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_classifier() -> tuple[SimpleCNN, dict]:
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache
    classifier_path = resolve_checkpoint_path(CLASSIFIER_PATHS, "classifier")
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
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
    generator_path = resolve_checkpoint_path(GENERATOR_PATHS, "GAN")
    checkpoint = torch.load(generator_path, map_location=DEVICE)
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
        classifier_status=checkpoint_status(CLASSIFIER_PATHS),
        generator_status=checkpoint_status(GENERATOR_PATHS),
        metrics=load_metrics(),
        sample_images=dataset_samples(),
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
        classifier_status=checkpoint_status(CLASSIFIER_PATHS),
        metrics=load_metrics(),
        device=DEVICE,
        sample_images=dataset_samples(),
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
        generator_status=checkpoint_status(GENERATOR_PATHS),
        device=DEVICE,
        sample_images=dataset_samples(),
    )


if __name__ == "__main__":
    app.run(debug=True)
