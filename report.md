# Lung Cancer DCGAN Experiment Report

## Dataset

The project now targets the lung subset of the LC25000 lung and colon histopathology dataset. The preparation script creates an 80/20 stratified train/test split with three lung tissue classes:

- lung_adenocarcinoma
- lung_benign_tissue
- lung_squamous_cell_carcinoma

Images are resized by the training transforms and normalized to [-1, 1].

## GAN Training

Model: class-conditional DCGAN with label embeddings.

Recommended command:

```powershell
python train_gan.py --train_dir data/lung_cancer/train --epochs 50 --batch_size 32 --image_size 64
```

Generated sample grids are saved in `generated_samples/`. The final checkpoint is saved at `checkpoints/lung_dcgan.pth`.

## Synthetic Image Generation

Recommended command:

```powershell
python generate_synthetic.py --checkpoint checkpoints/lung_dcgan.pth --match_real_dir data/lung_cancer/train --out_dir synthetic_lung_images
```

Synthetic images are organized by class in `synthetic_lung_images/<class_name>/`.

## Classifier Training

Model: `SimpleCNN` with four convolutional blocks and a fully connected classifier head.

Augmented model:

```powershell
python train_classifier.py --real_train_dir data/lung_cancer/train --synthetic_train_dir synthetic_lung_images --test_dir data/lung_cancer/test --epochs 20
```

Baseline model:

```powershell
python train_classifier.py --real_train_dir data/lung_cancer/train --synthetic_train_dir none --test_dir data/lung_cancer/test --out_name baseline_lung_classifier.pth --epochs 20
```

## Evaluation

Metrics are written to `checkpoints/lung_classifier_metrics.json` and include accuracy, precision, recall, F1-score, and a confusion matrix.

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
