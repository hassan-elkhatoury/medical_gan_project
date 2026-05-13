# Presentation Slide Guide

## Theme

Use a modern medical-tech style:

- Background: white or very light gray `#F7F9FC`
- Main text: navy `#18324A`
- Accent: teal `#13A89E`
- Secondary accent: coral `#E76F51`
- Font: Inter, Aptos, Calibri, or Poppins
- Style: clean, visual, low text, strong diagrams

Avoid long paragraphs. Each slide should communicate one idea.

## Slide 1: Title

Title: **AI-Powered Skin Lesion Classification Using GAN-Generated Images**

Subtitle: **HAM10000 Dataset | DCGAN | CNN Classifier | Flask Web App**

Layout:

- Large title on the left or centered.
- Background can be a faded dermoscopic image grid or subtle medical AI visual.
- Put name, course, and date at the bottom.

Speaker point:

This project uses generative AI and classification models to recognize seven skin lesion conditions from dermoscopic images.

## Slide 2: Project Motivation

Title: **Why This Project Matters**

Layout:

- Left side: three short points with icons.
- Right side: dermoscopic image or doctor/AI illustration.

Content:

- Skin lesion classification is visually challenging.
- Medical datasets are often imbalanced.
- Synthetic images can increase training diversity.

Speaker point:

Some lesion classes have fewer examples, making it difficult for a classifier to learn them well.

## Slide 3: Project Objective

Title: **Project Objective**

Layout:

- Use a horizontal pipeline diagram.

Pipeline:

```text
Download HAM10000 -> Prepare Data -> Train DCGAN -> Generate Synthetic Images -> Train CNN -> Web App
```

Main text:

Generate synthetic dermoscopic images for each lesion class and use them with real images to train a CNN classifier.

Speaker point:

The project is not only a model training task. It includes dataset preparation, GAN generation, classifier evaluation, and frontend deployment.

## Slide 4: Dataset

Title: **HAM10000 Dataset**

Layout:

- Display a 7-image grid if available, one sample per class.
- Put dataset facts on the right.

Classes:

- Actinic keratoses
- Basal cell carcinoma
- Benign keratosis
- Dermatofibroma
- Melanoma
- Melanocytic nevi
- Vascular lesions

Facts:

- Dataset: HAM10000 / Skin Cancer MNIST
- Input: dermoscopic skin lesion images
- Split: 80% train, 20% test
- Image size: `128 x 128`

Speaker point:

The metadata file maps each image ID to one of seven diagnosis labels.

## Slide 5: Data Preparation

Title: **Data Preparation Pipeline**

Layout:

- Horizontal process diagram.
- Small folder structure on the right.

Process:

```text
Kaggle download -> Extract files -> Read metadata CSV -> Map labels -> Split data -> Resize and normalize
```

Folder structure:

```text
data/ham10000/
  train/class_name/
  test/class_name/
```

Technical details:

- Stratified train/test split
- Images resized to `128 x 128`
- Pixel values normalized to `[-1, 1]`
- Reproducibility through fixed random seeds

Speaker point:

The folder structure makes the dataset easy to load with PyTorch.

## Slide 6: GAN Concept

Title: **DCGAN: Generating Synthetic Lesion Images**

Layout:

- Two-column diagram.
- Left: generator.
- Right: discriminator.

Generator:

```text
Random noise + class label -> Synthetic lesion image
```

Discriminator:

```text
Image + class label -> Real or fake
```

Speaker point:

The generator and discriminator improve through competition. The generator tries to create realistic images, while the discriminator tries to detect fake ones.

## Slide 7: GAN Architecture

Title: **Conditional DCGAN Architecture**

Layout:

- Use stacked block diagrams instead of code.

Generator architecture:

```text
Noise vector
+ Label embedding
-> Transposed convolution blocks
-> Tanh output
-> RGB image
```

Discriminator architecture:

```text
RGB image
+ Label map
-> Strided convolution blocks
-> Sigmoid output
-> Real/fake probability
```

Training details:

- Optimizer: Adam
- Learning rate: `0.0002`
- Betas: `(0.5, 0.999)`
- Batch size: `64` recommended
- Epochs: `50-100` recommended for final training

Speaker point:

The GAN is conditional, so one model can generate images for specific lesion classes.

## Slide 8: Synthetic Image Generation

Title: **Generated Synthetic Samples**

Layout:

- Large image grid in the center.
- Use an actual generated sample file from `generated_samples/` if available.
- Add small labels or caption below the grid.

Content:

After training, random noise vectors are sampled and passed through the generator to create new synthetic images.

Speaker point:

Early samples may look blurry, but quality improves with longer GPU training.

## Slide 9: Classifier Model

Title: **CNN Classifier**

Layout:

- Left: CNN architecture diagram.
- Right: probability bar chart mockup.

Architecture:

```text
Input image
-> Conv Block 1
-> Conv Block 2
-> Conv Block 3
-> Conv Block 4
-> Fully Connected Layers
-> 7-class output
```

Training details:

- Loss: cross-entropy
- Optimizer: Adam
- Augmentation: flips, rotations, color jitter
- Training data: real + synthetic images
- Test data: real images only

Speaker point:

The classifier predicts the most likely lesion class and outputs probabilities for all seven classes.

## Slide 10: Experiment Design

Title: **Baseline vs GAN-Augmented Training**

Layout:

- Comparison table.

Table:

| Model | Training Data | Test Data |
| --- | --- | --- |
| Baseline CNN | Real images only | Real test images |
| Augmented CNN | Real + synthetic images | Real test images |

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Speaker point:

The baseline shows classifier performance without GAN images. The augmented model tests whether synthetic images improve performance.

## Slide 11: Evaluation Results

Title: **Evaluation Results**

Layout:

- Left: metric cards.
- Right: confusion matrix.

Metric cards:

- Accuracy: `TBD`
- Macro F1: `TBD`
- Weighted F1: `TBD`
- Best class: `TBD`

Use results from:

```text
checkpoints/skin_classifier_metrics.json
```

Speaker point:

The model is evaluated on held-out real images only, which gives a more realistic test of generalization.

## Slide 12: Frontend Application

Title: **Interactive Web App**

Layout:

- Two screenshots side by side.
- Left: classifier page.
- Right: generate samples page.

Classifier page:

- Upload image
- View predicted class
- View probability bars

Generate page:

- Select lesion class
- Refresh generated samples
- View generated grid

Speaker point:

The Flask frontend makes the trained models easy to interact with, even for users who do not run Python commands.

## Slide 13: Live Demo

Title: **Live Demo**

Layout:

- Very minimal slide with four steps.

Demo flow:

1. Open frontend link.
2. Upload a lesion image.
3. View predicted class and probabilities.
4. Generate synthetic samples.

Speaker point:

This slide should stay on screen while doing the live demo.

## Slide 14: Challenges

Title: **Challenges Encountered**

Layout:

- Three vertical cards.

Challenge 1: Dataset imbalance

Some classes have many fewer examples than others.

Challenge 2: GAN training time

High-quality images require long training on GPU.

Challenge 3: Medical reliability

This is an educational prototype and not a clinical diagnostic tool.

Speaker point:

The biggest technical challenge is training a GAN long enough to produce realistic medical images.

## Slide 15: Conclusion

Title: **Conclusion**

Layout:

- Clean final slide with three or four takeaway points.

Takeaways:

- Built a full AI pipeline from dataset preparation to deployment.
- Used a conditional DCGAN to generate class-specific synthetic images.
- Trained a CNN classifier using real and synthetic dermoscopic images.
- Created a frontend for image classification and sample generation.

Final statement:

This project demonstrates how generative AI can support medical image classification workflows when used carefully and responsibly.

## Optional Backup Slide: Commands Used

Title: **Main Commands**

Use only if the audience needs implementation details.

```powershell
python download_dataset.py --output_dir data/ham10000_raw
python prepare_dataset.py --raw_dir data/ham10000_raw --output_dir data/ham10000 --test_size 0.2 --overwrite
python train_gan.py --train_dir data/ham10000/train --epochs 50 --batch_size 64 --image_size 128
python generate_synthetic.py --checkpoint checkpoints/ham10000_dcgan.pth --match_real_dir data/ham10000/train --out_dir synthetic_images
python train_classifier.py --real_train_dir data/ham10000/train --synthetic_train_dir synthetic_images --test_dir data/ham10000/test --epochs 20
flask --app frontend/app.py run --host 0.0.0.0 --port 5000
```

## Optional Backup Slide: Responsible Use

Title: **Responsible Use**

Content:

- This system is a research and learning prototype.
- It should not be used as a medical diagnosis tool.
- Real clinical use requires expert validation, larger datasets, regulatory review, and careful bias testing.

