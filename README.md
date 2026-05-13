# Medical GAN Project

This project trains a conditional DCGAN for HAM10000 skin-lesion image generation, trains a CNN classifier, and serves both through a local Flask app.

## Run Locally

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Required trained files:

```text
checkpoints/ham10000_dcgan.pth
checkpoints/skin_classifier.pth
```

Start Flask:

```bash
.venv/bin/flask --app frontend/app.py run --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000/classify
http://127.0.0.1:5000/generate
```

## Train Or Change The Models

Download and prepare HAM10000:

```bash
python download_dataset.py --output_dir data/ham10000_raw
python prepare_dataset.py --raw_dir data/ham10000_raw --output_dir data/ham10000 --test_size 0.2 --overwrite
```

Train the GAN:

```bash
python train_gan.py --train_dir data/ham10000/train --epochs 50 --batch_size 32 --image_size 64 --num_workers 2
```

Generate synthetic images:

```bash
python generate_synthetic.py --checkpoint checkpoints/ham10000_dcgan.pth --match_real_dir data/ham10000/train --out_dir synthetic_images --samples_per_class 20 --batch_size 16
```

Train the classifier:

```bash
python train_classifier.py --real_train_dir data/ham10000/train --synthetic_train_dir synthetic_images --test_dir data/ham10000/test --epochs 20 --batch_size 32 --num_workers 2
```

## What To Change

- Better generated images: increase `--epochs` in `train_gan.py`.
- Faster experiments: keep `--image_size 64` and smaller `--batch_size`.
- Better quality experiments: use more GAN epochs, then replace `checkpoints/ham10000_dcgan.pth`.
- Better classifier results: retrain `train_classifier.py`, then replace `checkpoints/skin_classifier.pth`.
- Different Flask port: change `--port 5000` in the run command.

If a checkpoint is missing, the Flask app shows the missing-file error so you know what to fix.
