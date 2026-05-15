# Presentation Plan: COVID-19 Radiography GAN Project

## Slide 1: Project Title

**Slide content**

- COVID-19 Chest X-ray Generation and Classification
- Deep learning workflow using a Conditional DCGAN and CNN
- Dataset: Kaggle COVID-19 Radiography Database
- Final interface: local Flask web application

**What to say**

"This project builds a complete medical image deep learning pipeline. It prepares COVID-19 chest X-ray data, trains a conditional GAN to generate synthetic images, trains a CNN classifier, evaluates the classifier, and exposes the models through a Flask web app."

---

## Slide 2: Problem Statement

**Slide content**

- Medical image models need enough labeled examples.
- Class imbalance can reduce classification quality.
- Synthetic images can be used as extra training data.
- The system predicts the radiography class of a chest X-ray image.

**What to say**

"The main problem is how to train an image classifier when medical data is limited or imbalanced. This project explores whether generated chest X-ray images can support the classifier training process."

---

## Slide 3: Project Objectives

**Slide content**

- Download and prepare the COVID-19 Radiography Database.
- Train a class-conditional DCGAN.
- Generate class-organized synthetic chest X-ray images.
- Train a CNN classifier with real images and optional synthetic images.
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.
- Build a local Flask app for classification and generation.

**What to say**

"The project has two model parts: a generator for creating synthetic radiography images and a classifier for predicting the image class. Around those models, the project includes data preparation, evaluation, and a small web interface."

---

## Slide 4: Dataset Used

**Slide content**

- Dataset: COVID-19 Radiography Database from Kaggle.
- Source dataset contains image folders and mask folders.
- This project uses only radiography image files.
- Mask folders are ignored because this is not a segmentation project.
- Four classes:
  - `covid`
  - `lung_opacity`
  - `normal`
  - `viral_pneumonia`

**What to say**

"The actual dataset used by the code is the COVID-19 Radiography Database, not HAM10000. The project focuses on chest X-ray image generation and classification across four radiography categories."

---

## Slide 5: Data Preparation Pipeline

**Slide content**

- `download_dataset.py` downloads and extracts the Kaggle dataset.
- `prepare_dataset.py` creates a clean folder dataset.
- Class names are normalized automatically.
- Train/test split is stratified.
- Final structure:

```text
data/covid_radiography/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

**What to say**

"The preparation script searches the extracted dataset, skips mask images, detects the correct class from the path, and creates train and test folders. Stratification keeps the class distribution more consistent in both splits."

---

## Slide 6: Image Preprocessing

**Slide content**

- Images are loaded as RGB.
- Training transforms resize images.
- GAN and classifier use normalized tensors in `[-1, 1]`.
- Training augmentation includes:
  - horizontal flip
  - vertical flip with low probability
  - rotation
  - small color jitter

**What to say**

"The models do not use raw image files directly. Each image is resized, converted to a tensor, and normalized. During training, augmentation is used to make the models less sensitive to small visual variations."

---

## Slide 7: Overall Architecture

**Slide content**

```text
COVID-19 Radiography Database
      |
      v
Prepared train/test folders
      |
      +--------------------+
      |                    |
      v                    v
Conditional DCGAN       Real train images
      |                    |
      v                    |
Synthetic images ---------+
      |
      v
SimpleCNN classifier
      |
      v
Evaluation + Flask app
```

**What to say**

"The project starts with prepared data. The GAN learns from the real training set and generates extra samples. The classifier can then be trained with real images only or with real plus synthetic images. The final checkpoints are used by the Flask app."

---

## Slide 8: Conditional DCGAN

**Slide content**

- File: `gan.py`
- Models:
  - `ConditionalGenerator`
  - `ConditionalDiscriminator`
- Generator input:
  - random noise vector
  - class label embedding
- Generator output:
  - synthetic RGB chest X-ray image
- Discriminator input:
  - image
  - matching class label

**What to say**

"This is a conditional GAN, so the model can generate images for a requested class. The class label is embedded and combined with the noise for the generator, and the discriminator also receives label information."

---

## Slide 9: GAN Training

**Slide content**

- File: `train_gan.py`
- Loss: binary cross entropy.
- Optimizer: Adam.
- Default latent noise size: `100`.
- Supported GAN image sizes: `64` or `128`.
- Checkpoints:

```text
checkpoints/covid_dcgan.pth
checkpoints/generator.pth
```

- Sample grids:

```text
generated_samples/gan_epoch_*.png
```

**What to say**

"During GAN training, the discriminator learns to separate real and fake images, while the generator learns to produce images that the discriminator classifies as real. The script saves both checkpoints and visual sample grids."

---

## Slide 10: Synthetic Image Generation

**Slide content**

- File: `generate_synthetic.py`
- Loads `checkpoints/covid_dcgan.pth`.
- Creates generated images by class.
- Output structure:

```text
synthetic_covid_images/
  covid/
  lung_opacity/
  normal/
  viral_pneumonia/
```

- Can generate:
  - a fixed number per class
  - or the same number as real training images

**What to say**

"After the GAN is trained, the generator is used as a data creation tool. The script writes generated PNG files into folders with the same class names as the real dataset, so the classifier can read them using the same dataset class."

---

## Slide 11: CNN Classifier

**Slide content**

- File: `classifier.py`
- Model: `SimpleCNN`
- Input: RGB chest X-ray image.
- Default classifier image size: `128 x 128`.
- Output: one score for each of the four classes.
- Architecture:
  - 4 convolution blocks
  - batch normalization
  - ReLU
  - max pooling
  - fully connected classification head

**What to say**

"The classifier is intentionally lightweight. It has four convolution blocks to learn visual features, then a fully connected head that predicts the class."

---

## Slide 12: Classifier Training

**Slide content**

- File: `train_classifier.py`
- Training data:
  - real training images
  - optional synthetic images
- Validation split from the training set: default `15%`.
- Loss: `CrossEntropyLoss`.
- Optimizer: Adam.
- Best model selected by validation accuracy.

**What to say**

"The classifier can be trained in two modes. For a baseline, it can use only real images. For the augmented experiment, it uses real images plus GAN-generated images. The script keeps the best validation checkpoint before testing."

---

## Slide 13: Evaluation

**Slide content**

- Test data uses real images only.
- Metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrix
- Output files:

```text
checkpoints/covid_classifier_metrics.json
generated_samples/covid_classifier_confusion_matrix.png
generated_samples/covid_classifier_curves.png
```

**What to say**

"Testing is performed on the real test set, not on generated images. This is important because the goal is to know whether the model performs better on real chest X-rays."

---

## Slide 14: Baseline vs Synthetic Experiment

**Slide content**

| Experiment | Training data | Test data |
| --- | --- | --- |
| Baseline CNN | Real images only | Real test images |
| Augmented CNN | Real + synthetic images | Real test images |

**What to say**

"The project supports a fair comparison. The only difference between the two experiments is whether generated images are included in the training set. Both are evaluated on real test images."

---

## Slide 15: Flask Web App

**Slide content**

- Backend file: `frontend/app.py`
- Loads:

```text
checkpoints/covid_dcgan.pth
checkpoints/covid_classifier.pth
```

- Pages:
  - home dashboard
  - image classification
  - synthetic image generation

**What to say**

"The Flask app makes the trained models interactive. It checks whether checkpoints exist, displays metrics when available, lets the user classify an uploaded image, and lets the user generate a grid of synthetic samples."

---

## Slide 16: Home Dashboard

**Slide content**

- Shows dataset name.
- Shows current device: CPU or CUDA.
- Lists supported classes.
- Shows whether generator and classifier checkpoints exist.
- Displays accuracy, macro F1, and weighted F1 when metrics exist.

**What to say**

"The home page is a status page for the whole project. It tells us whether the trained models are ready and whether evaluation metrics are available."

---

## Slide 17: Classification Page

**Slide content**

- User uploads a chest X-ray image.
- Flask applies the same preprocessing transform.
- `SimpleCNN` predicts probabilities.
- Output shows:
  - predicted class
  - probability for each class
  - uploaded image preview

**What to say**

"The classification page is where the user tests the trained CNN. The backend converts the uploaded image into the same normalized tensor format used during training."

---

## Slide 18: Generation Page

**Slide content**

- User chooses:
  - all classes
  - or one specific class
- Conditional generator creates `36` images.
- App saves and displays a `6 x 6` sample grid.
- Output file:

```text
frontend/static/generated/latest_samples.png
```

**What to say**

"The generation page uses the trained GAN. If all classes are selected, the grid cycles through class labels. If one class is selected, all generated samples use that class label."

---

## Slide 19: Main Commands

**Slide content**

```bash
python download_dataset.py --output_dir data/covid_radiography_raw
python prepare_dataset.py --raw_dir data/covid_radiography_raw --output_dir data/covid_radiography --test_size 0.2 --overwrite
python train_gan.py --train_dir data/covid_radiography/train --epochs 50 --batch_size 32 --image_size 64
python generate_synthetic.py --checkpoint checkpoints/covid_dcgan.pth --match_real_dir data/covid_radiography/train --out_dir synthetic_covid_images
python train_classifier.py --real_train_dir data/covid_radiography/train --synthetic_train_dir synthetic_covid_images --test_dir data/covid_radiography/test --epochs 20
python evaluate.py --checkpoint checkpoints/covid_classifier.pth --test_dir data/covid_radiography/test
flask --app frontend/app.py run --host 127.0.0.1 --port 5000
```

**What to say**

"These are the main commands that reproduce the project workflow: data, GAN training, synthetic generation, classifier training, evaluation, and the web application."

---

## Slide 20: Limitations

**Slide content**

- Educational prototype, not a clinical diagnostic tool.
- GAN images can be blurry or unrealistic if training is short.
- `SimpleCNN` is lightweight and can be improved.
- Results depend on dataset quality and class balance.
- Medical use would require expert validation and stronger testing.

**What to say**

"The project demonstrates the workflow, but it is not ready for medical diagnosis. Generated medical images must be evaluated carefully, and classification results require rigorous validation."

---

## Slide 21: Future Improvements

**Slide content**

- Train the GAN for more epochs.
- Compare more baseline and augmented experiments.
- Use stronger classifiers such as ResNet, DenseNet, or EfficientNet.
- Add Grad-CAM explainability.
- Display confusion matrix and training curves in the web app.
- Add model version information to the dashboard.

**What to say**

"Future work can improve both the machine learning side and the application side. Stronger pretrained classifiers and explainability would make the experiment more useful and easier to interpret."

---

## Slide 22: Conclusion

**Slide content**

- Prepared the COVID-19 Radiography Database.
- Trained a conditional DCGAN for class-controlled image generation.
- Generated synthetic chest X-ray images.
- Trained and evaluated a CNN classifier.
- Built a Flask app for classification and generation.

**What to say**

"The final project is a full deep learning workflow for COVID-19 chest radiography images. It connects data preparation, generative modeling, classification, evaluation, and deployment into one coherent system."
