# COVID-19 Radiography DCGAN Experiment Report

## Dataset

The project now targets the Kaggle COVID-19 Radiography Database. The preparation script creates an 80/20 stratified train/test split with four chest X-ray classes:

- covid
- lung_opacity
- normal
- viral_pneumonia

Images are resized by the training transforms and normalized to [-1, 1]. Mask folders from the dataset are ignored because this project uses image classification and GAN generation, not segmentation.

## GAN Training

Model: class-conditional DCGAN with label embeddings.

Recommended command:

```powershell
python train_gan.py --train_dir data/covid_radiography/train --epochs 50 --batch_size 32 --image_size 64
```

Generated sample grids are saved in `generated_samples/`. The final checkpoint is saved at `checkpoints/covid_dcgan.pth`.

## Synthetic Image Generation

Recommended command:

```powershell
python generate_synthetic.py --checkpoint checkpoints/covid_dcgan.pth --match_real_dir data/covid_radiography/train --out_dir synthetic_covid_images
```

Synthetic images are organized by class in `synthetic_covid_images/<class_name>/`.

## Classifier Training

Model: `SimpleCNN` with four convolutional blocks and a fully connected classifier head.

Augmented model:

```powershell
python train_classifier.py --real_train_dir data/covid_radiography/train --synthetic_train_dir synthetic_covid_images --test_dir data/covid_radiography/test --epochs 20
```

Baseline model:

```powershell
python train_classifier.py --real_train_dir data/covid_radiography/train --synthetic_train_dir none --test_dir data/covid_radiography/test --out_name baseline_covid_classifier.pth --epochs 20
```

## Evaluation

Metrics are written to `checkpoints/covid_classifier_metrics.json` and include accuracy, precision, recall, F1-score, and a confusion matrix.

Fill in after training:

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Real only baseline | TBD | TBD | TBD |
| Real + synthetic | TBD | TBD | TBD |

## Observations

Fill in after visual inspection and evaluation:

- Synthetic image quality:
- Classes that improved:
- Classes that regressed:
- Common confusions:
- Training or hardware constraints:
