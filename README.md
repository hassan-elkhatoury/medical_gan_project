# Medical GAN Project

This project trains a conditional DCGAN for COVID-19 chest X-ray generation, trains a CNN classifier, and serves both through a local Flask app.

The default dataset target is the Kaggle COVID-19 Radiography Database:

- `covid`
- `lung_opacity`
- `normal`
- `viral_pneumonia`

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
checkpoints/covid_dcgan.pth
checkpoints/covid_classifier.pth
```

Start Flask:

```bash
.venv/bin/flask --app frontend/app.py run --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/classify
http://127.0.0.1:5000/generate
```

## Train Or Change The Models

Download and prepare the COVID-19 Radiography Database:

```bash
python download_dataset.py --output_dir data/covid_radiography_raw
python prepare_dataset.py --raw_dir data/covid_radiography_raw --output_dir data/covid_radiography --test_size 0.2 --overwrite
```

Train the GAN:

```bash
python train_gan.py --train_dir data/covid_radiography/train --epochs 50 --batch_size 32 --image_size 64 --num_workers 2
```

Generate synthetic images:

```bash
python generate_synthetic.py --checkpoint checkpoints/covid_dcgan.pth --match_real_dir data/covid_radiography/train --out_dir synthetic_covid_images --samples_per_class 100 --batch_size 16
```

Train the classifier:

```bash
python train_classifier.py --real_train_dir data/covid_radiography/train --synthetic_train_dir synthetic_covid_images --test_dir data/covid_radiography/test --epochs 20 --batch_size 32 --num_workers 2
```

Evaluate the classifier:

```bash
python evaluate.py --checkpoint checkpoints/covid_classifier.pth --test_dir data/covid_radiography/test
```

## What To Change

- Better generated images: increase `--epochs` in `train_gan.py`.
- Faster experiments: keep `--image_size 64`, use smaller `--batch_size`, or pass `--max_images_per_class` to `prepare_dataset.py`.
- Better quality experiments: use more GAN epochs, then replace `checkpoints/covid_dcgan.pth`.
- Better classifier results: retrain `train_classifier.py`, then replace `checkpoints/covid_classifier.pth`.
- Different Flask port: change `--port 5000` in the run command.

If a checkpoint is missing, the Flask app shows the missing-file error so you know what to train next.
