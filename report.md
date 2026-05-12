# HAM10000 DCGAN Experiment Report

## Dataset

HAM10000 was prepared from Kaggle metadata into an 80/20 stratified train/test split with seven lesion classes:

- actinic_keratoses
- basal_cell_carcinoma
- benign_keratosis
- dermatofibroma
- melanoma
- melanocytic_nevi
- vascular_lesions

Images are resized to 128 x 128 and normalized to [-1, 1].

## GAN Training

Model: class-conditional DCGAN with label embeddings.

Recommended command:

```powershell
python train_gan.py --train_dir data/ham10000/train --epochs 50 --batch_size 64 --image_size 128
```

Generated sample grids are saved in `generated_samples/`. The final checkpoint is saved at `checkpoints/ham10000_dcgan.pth`.

## Synthetic Image Generation

Recommended command:

```powershell
python generate_synthetic.py --checkpoint checkpoints/ham10000_dcgan.pth --match_real_dir data/ham10000/train --out_dir synthetic_images
```

Synthetic images are organized by class in `synthetic_images/<class_name>/`.

## Classifier Training

Model: `SimpleCNN` with four convolutional blocks and a fully connected classifier head.

Augmented model:

```powershell
python train_classifier.py --real_train_dir data/ham10000/train --synthetic_train_dir synthetic_images --test_dir data/ham10000/test --epochs 20
```

Baseline model:

```powershell
python train_classifier.py --real_train_dir data/ham10000/train --synthetic_train_dir none --test_dir data/ham10000/test --out_name baseline_classifier.pth --epochs 20
```

## Evaluation

Metrics are written to `checkpoints/skin_classifier_metrics.json` and include accuracy, precision, recall, F1-score, and a confusion matrix.

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
