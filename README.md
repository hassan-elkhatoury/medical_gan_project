# HAM10000 DCGAN and Classifier

This project downloads HAM10000, prepares train/test folders, trains a class-conditional DCGAN, generates class-organized synthetic dermoscopic images, trains a CNN classifier with real plus synthetic images, and serves both models through Flask.

## Environment

Use Python 3.10, 3.11, or 3.12 for the smoothest PyTorch install experience.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you need a CUDA-specific PyTorch wheel, install the matching command from the PyTorch website first, then run the remaining requirements.

## Dataset

Place your Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.

```powershell
python download_dataset.py --output_dir data/ham10000_raw
python prepare_dataset.py --raw_dir data/ham10000_raw --output_dir data/ham10000 --test_size 0.2 --overwrite
```

Prepared data will be written to:

```text
data/ham10000/train/<class_name>/*.jpg
data/ham10000/test/<class_name>/*.jpg
```

## Train Models

Train the conditional DCGAN:

```powershell
python train_gan.py --train_dir data/ham10000/train --epochs 50 --batch_size 64 --image_size 128
```

Generate synthetic images, matching the number of real training images per class:

```powershell
python generate_synthetic.py --checkpoint checkpoints/ham10000_dcgan.pth --match_real_dir data/ham10000/train --out_dir synthetic_images
```

Train the augmented classifier:

```powershell
python train_classifier.py --real_train_dir data/ham10000/train --synthetic_train_dir synthetic_images --test_dir data/ham10000/test --epochs 20
```

Train a baseline without synthetic images:

```powershell
python train_classifier.py --real_train_dir data/ham10000/train --synthetic_train_dir none --test_dir data/ham10000/test --out_name baseline_classifier.pth --epochs 20
```

Evaluate a saved classifier:

```powershell
python evaluate.py --checkpoint checkpoints/skin_classifier.pth --test_dir data/ham10000/test
```

## Outputs

The scripts create these main artifacts:

```text
checkpoints/ham10000_dcgan.pth
checkpoints/generator.pth
checkpoints/skin_classifier.pth
skin_classifier.pth
generated_samples/
synthetic_images/
checkpoints/*_metrics.json
checkpoints/*_history.json
```

## Frontend

Run the Flask app after the checkpoints are trained:

```powershell
$env:FLASK_APP = "frontend/app.py"
flask run --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000/classify](http://127.0.0.1:5000/classify) to upload an image for classification, or [http://127.0.0.1:5000/generate](http://127.0.0.1:5000/generate) to refresh synthetic sample grids.

The app selects CUDA when available and falls back to CPU automatically.
