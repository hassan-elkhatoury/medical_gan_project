# Contenu des Slides - Projet GAN Médical COVID-19

## Slide 1 - Titre

**Génération et classification de radiographies thoraciques COVID-19 avec GAN et CNN**

Projet basé sur le dataset **COVID-19 Radiography Database** de Kaggle.

Le projet utilise deux modèles :

- un **Conditional DCGAN** pour générer des images synthétiques de radiographies ;
- un **CNN** pour classifier les radiographies en plusieurs catégories.

**À dire oralement :**  
Dans ce projet, nous avons construit une chaîne complète de deep learning : téléchargement et préparation des données, entraînement d'un GAN, génération d'images synthétiques, entraînement d'un classificateur, évaluation, puis application web Flask.

---

## Slide 2 - Idée générale du projet

Le but est de classifier des images de radiographies thoraciques et d'étudier l'utilisation d'images synthétiques générées par GAN.

Problème principal :

- les modèles de deep learning ont besoin de beaucoup d'exemples ;
- les datasets médicaux peuvent être limités ou déséquilibrés ;
- certaines classes peuvent être plus difficiles à apprendre.

Notre solution :

- préparer les vraies images du dataset ;
- entraîner un GAN conditionnel pour générer des images par classe ;
- ajouter les images synthétiques à l'entraînement du classificateur CNN ;
- évaluer le modèle sur des images réelles de test.

**À dire oralement :**  
Le GAN ne remplace pas les données réelles. Il sert à créer des exemples supplémentaires pour tester si cela peut aider le classificateur.

---

## Slide 3 - Plan de la présentation

Dans cette présentation, nous allons expliquer le projet étape par étape.

Plan :

- contexte et objectif ;
- dataset COVID-19 Radiography Database ;
- préparation des données ;
- architecture générale ;
- entraînement du Conditional DCGAN ;
- génération d'images synthétiques ;
- entraînement du CNN ;
- évaluation ;
- application web Flask ;
- limites et améliorations.

**À dire oralement :**  
Nous allons suivre le même ordre que le projet réel : les données, le GAN, les images synthétiques, le CNN, l'évaluation, puis l'application.

---

## Slide 4 - Dataset utilisé

Le projet utilise le dataset **COVID-19 Radiography Database**.

Il contient des radiographies thoraciques organisées par catégorie.

Classes utilisées par le code :

- `covid` ;
- `lung_opacity` ;
- `normal` ;
- `viral_pneumonia`.

Les dossiers de masques sont ignorés, car ce projet ne fait pas de segmentation.

**À dire oralement :**  
Le projet actuel ne travaille pas sur HAM10000 ou des lésions de peau. Il travaille sur des radiographies thoraciques COVID-19 avec quatre classes.

---

## Slide 5 - Préparation des données

Avant l'entraînement, les données sont préparées avec deux scripts :

```text
download_dataset.py -> téléchargement Kaggle
prepare_dataset.py  -> création train/test
```

Structure finale :

```text
data/covid_radiography/
  train/
    covid/
    lung_opacity/
    normal/
    viral_pneumonia/
  test/
    covid/
    lung_opacity/
    normal/
    viral_pneumonia/
```

La séparation train/test est stratifiée avec `test_size = 0.2`.

**À dire oralement :**  
La préparation transforme le dataset brut en dossiers faciles à lire avec PyTorch. Chaque classe a son propre dossier dans train et dans test.

---

## Slide 6 - Prétraitement des images

Pendant l'entraînement, les images sont :

- ouvertes en RGB ;
- redimensionnées ;
- converties en tenseurs PyTorch ;
- normalisées entre `-1` et `1`.

Augmentations utilisées pendant l'entraînement :

- retournement horizontal ;
- retournement vertical avec faible probabilité ;
- rotation ;
- léger changement de luminosité, contraste et saturation.

**À dire oralement :**  
Le prétraitement donne au modèle des images dans un format constant. Les augmentations aident le modèle à mieux généraliser.

---

## Slide 7 - Pipeline complet du projet

Voici le déroulement complet :

```text
1. Télécharger le dataset Kaggle
2. Préparer les dossiers train/test
3. Entraîner le Conditional DCGAN
4. Générer des images synthétiques
5. Entraîner le CNN avec images réelles ou réelles + synthétiques
6. Évaluer sur les vraies images de test
7. Tester avec l'application Flask
```

**À dire oralement :**  
Le projet n'est pas seulement un modèle. C'est une chaîne complète depuis les données jusqu'à une interface web utilisable.

---

## Slide 8 - Architecture générale

```text
COVID-19 Radiography Database
      |
      v
Préparation des données
      |
      +--------------------+
      |                    |
      v                    v
Conditional DCGAN       Images réelles
      |                    |
      v                    |
Images synthétiques ------+
      |
      v
CNN SimpleCNN
      |
      v
Évaluation + Flask
```

**À dire oralement :**  
Le GAN apprend à générer des images par classe. Ensuite, le CNN peut être entraîné avec les images réelles seulement ou avec les images réelles plus les images générées.

---

## Slide 9 - C'est quoi un GAN ?

Un GAN est un modèle génératif composé de deux réseaux :

- **Generator** : crée des images synthétiques ;
- **Discriminator** : essaie de distinguer les vraies images des fausses.

Principe :

```text
Bruit aléatoire -> Generator -> Image synthétique
Image réelle ou synthétique -> Discriminator -> Vrai/Faux
```

**À dire oralement :**  
Le GAN fonctionne comme une compétition entre deux modèles. Le generator apprend à créer des images plus réalistes, et le discriminator apprend à les détecter.

---

## Slide 10 - Notre Conditional DCGAN

Le projet utilise un **Conditional DCGAN**.

Fichiers principaux :

- `gan.py`
- `train_gan.py`

Entrées du generator :

- vecteur de bruit aléatoire ;
- label de classe.

Sortie du generator :

- image RGB synthétique de radiographie thoracique.

**À dire oralement :**  
Le mot "conditional" signifie que nous pouvons demander au GAN de générer une classe précise, par exemple `covid` ou `normal`.

---

## Slide 11 - Generator

Le generator transforme un bruit aléatoire et un label en image.

Étapes simples :

```text
Bruit + embedding de classe
-> convolutions transposées
-> image RGB
-> Tanh
```

Paramètres importants :

- bruit latent : `100` ;
- nombre de canaux : `3` ;
- image GAN supportée : `64 x 64` ou `128 x 128`.

**À dire oralement :**  
Au début, le bruit ne contient aucune information visuelle. Le generator apprend progressivement à transformer ce bruit en image qui ressemble à une radiographie.

---

## Slide 12 - Discriminator

Le discriminator reçoit une image et le label associé.

Étapes simples :

```text
Image + label de classe
-> convolutions
-> probabilité vrai/faux
```

Il apprend avec :

- images réelles du dataset ;
- images synthétiques créées par le generator.

**À dire oralement :**  
Le discriminator force le generator à s'améliorer. Si les images générées sont mauvaises, le discriminator les reconnaît facilement.

---

## Slide 13 - Entraînement du GAN

Script utilisé :

```text
train_gan.py
```

Paramètres principaux :

- loss : `BCELoss` ;
- optimizer : `Adam` ;
- learning rate : `0.0002` ;
- batch size recommandé : `32` ou `64` ;
- epochs recommandés : `50`.

Checkpoints sauvegardés :

```text
checkpoints/covid_dcgan.pth
checkpoints/generator.pth
```

Exemples générés :

```text
generated_samples/gan_epoch_*.png
```

**À dire oralement :**  
Pendant l'entraînement, le projet sauvegarde des grilles d'images pour suivre visuellement la progression du GAN.

---

## Slide 14 - Génération des images synthétiques

Après l'entraînement, le script suivant utilise le generator :

```text
generate_synthetic.py
```

Il crée des images organisées par classe :

```text
synthetic_covid_images/
  covid/
  lung_opacity/
  normal/
  viral_pneumonia/
```

Deux modes sont possibles :

- générer un nombre fixe d'images par classe ;
- générer le même nombre que les images réelles disponibles.

**À dire oralement :**  
Les images synthétiques sont sauvegardées comme un dataset normal. Cela permet au CNN de les utiliser avec le même code que les vraies images.

---

## Slide 15 - Notre classificateur CNN

Le classificateur s'appelle `SimpleCNN`.

Fichier :

```text
classifier.py
```

Entrée :

- image RGB de radiographie ;
- taille par défaut du classifier : `128 x 128`.

Sortie :

- une prédiction parmi les 4 classes ;
- une probabilité pour chaque classe.

**À dire oralement :**  
Le CNN est la partie qui prend la décision finale. Il regarde l'image et prédit sa catégorie.

---

## Slide 16 - Architecture du SimpleCNN

Architecture simplifiée :

```text
Image
-> bloc convolution 1
-> bloc convolution 2
-> bloc convolution 3
-> bloc convolution 4
-> couche linéaire
-> dropout
-> sortie 4 classes
```

Chaque bloc contient :

- convolution ;
- batch normalization ;
- ReLU ;
- max pooling.

**À dire oralement :**  
Le modèle est volontairement simple pour rester compréhensible et rapide à entraîner. Il peut être remplacé plus tard par ResNet ou EfficientNet.

---

## Slide 17 - Entraînement du CNN

Script utilisé :

```text
train_classifier.py
```

Le CNN peut être entraîné avec :

- images réelles seulement ;
- images réelles + images synthétiques.

Paramètres principaux :

- loss : `CrossEntropyLoss` ;
- optimizer : `Adam` ;
- validation interne : `15%` ;
- meilleur modèle choisi selon l'accuracy de validation.

**À dire oralement :**  
Le projet permet de comparer un modèle baseline avec un modèle augmenté par des images générées par GAN.

---

## Slide 18 - Expérience réalisée

Comparaison prévue :

| Modèle | Données d'entraînement | Données de test |
| --- | --- | --- |
| CNN baseline | Images réelles seulement | Images réelles |
| CNN augmenté | Images réelles + synthétiques | Images réelles |

Point important :

- le test se fait sur des images réelles ;
- les images synthétiques servent seulement à l'entraînement.

**À dire oralement :**  
Cette comparaison permet de voir si les images synthétiques aident réellement le classificateur sur des données réelles.

---

## Slide 19 - Évaluation

Les métriques utilisées :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- matrice de confusion.

Fichiers générés :

```text
checkpoints/covid_classifier_metrics.json
generated_samples/covid_classifier_confusion_matrix.png
generated_samples/covid_classifier_curves.png
```

Résultats à remplir après entraînement :

| Modèle | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| CNN baseline | TBD | TBD | TBD |
| CNN avec images GAN | TBD | TBD | TBD |

**À dire oralement :**  
La matrice de confusion montre quelles classes sont bien reconnues et quelles classes sont confondues.

---

## Slide 20 - Application web Flask

Le projet contient une application web locale.

Fichier principal :

```text
frontend/app.py
```

L'application charge :

```text
checkpoints/covid_dcgan.pth
checkpoints/covid_classifier.pth
```

Pages principales :

- accueil ;
- classification ;
- génération.

**À dire oralement :**  
L'application permet de tester les modèles sans utiliser directement les scripts Python.

---

## Slide 21 - Page accueil

La page accueil affiche :

- nom du dataset ;
- device utilisé : CPU ou CUDA ;
- liste des classes ;
- état du checkpoint GAN ;
- état du checkpoint classifier ;
- métriques si elles existent.

**À dire oralement :**  
La page d'accueil sert de tableau de bord. Elle indique rapidement si les modèles entraînés sont disponibles.

---

## Slide 22 - Page classification

Fonctionnement :

```text
Uploader une image
-> prétraitement
-> SimpleCNN
-> classe prédite + probabilités
```

La page affiche :

- aperçu de l'image uploadée ;
- classe prédite ;
- probabilité pour chaque classe.

**À dire oralement :**  
La classification utilise le même type de prétraitement que pendant l'entraînement, ce qui garde une cohérence entre training et utilisation.

---

## Slide 23 - Page génération

Fonctionnement :

```text
Choisir une classe
-> générer 36 images
-> afficher une grille 6 x 6
```

Options :

- générer toutes les classes ;
- générer une seule classe.

Fichier affiché :

```text
frontend/static/generated/latest_samples.png
```

**À dire oralement :**  
Cette page montre directement le résultat du GAN. On peut choisir une classe et observer les images synthétiques produites.

---

## Slide 24 - Commandes importantes

Commandes principales :

```bash
python download_dataset.py --output_dir data/covid_radiography_raw
python prepare_dataset.py --raw_dir data/covid_radiography_raw --output_dir data/covid_radiography --test_size 0.2 --overwrite
python train_gan.py --train_dir data/covid_radiography/train --epochs 50 --batch_size 32 --image_size 64
python generate_synthetic.py --checkpoint checkpoints/covid_dcgan.pth --match_real_dir data/covid_radiography/train --out_dir synthetic_covid_images
python train_classifier.py --real_train_dir data/covid_radiography/train --synthetic_train_dir synthetic_covid_images --test_dir data/covid_radiography/test --epochs 20
python evaluate.py --checkpoint checkpoints/covid_classifier.pth --test_dir data/covid_radiography/test
flask --app frontend/app.py run --host 127.0.0.1 --port 5000
```

**À dire oralement :**  
Ces commandes représentent le workflow complet : données, GAN, images synthétiques, classifier, évaluation et application.

---

## Slide 25 - Ce que nous avons construit

Dans ce projet, nous avons construit :

- un script de téléchargement du dataset Kaggle ;
- un script de préparation train/test ;
- un Conditional DCGAN ;
- un générateur d'images synthétiques ;
- un CNN de classification ;
- un script d'évaluation ;
- une application web Flask.

**À dire oralement :**  
Le projet montre comment connecter génération d'images et classification dans une seule chaîne deep learning.

---

## Slide 26 - Difficultés rencontrées

Les principales difficultés :

- l'entraînement d'un GAN peut être long ;
- les images générées peuvent être floues au début ;
- les résultats dépendent du nombre d'epochs ;
- les classes médicales peuvent être difficiles à distinguer ;
- la puissance GPU influence fortement le temps d'entraînement.

**À dire oralement :**  
Le GAN est souvent la partie la plus difficile, car il doit apprendre une distribution visuelle réaliste.

---

## Slide 27 - Limites du projet

Ce projet est un prototype éducatif.

Il ne doit pas être utilisé pour faire un vrai diagnostic médical.

Limites :

- validation clinique absente ;
- modèle CNN simple ;
- qualité des images synthétiques variable ;
- dataset limité à une source ;
- résultats à interpréter avec prudence.

**À dire oralement :**  
Dans le domaine médical, un modèle doit être validé par des experts et testé beaucoup plus largement avant une utilisation réelle.

---

## Slide 28 - Améliorations possibles

Pour améliorer le projet, on peut :

- entraîner le GAN plus longtemps ;
- comparer plusieurs configurations real-only vs real + synthetic ;
- utiliser un classifier plus puissant comme ResNet, DenseNet ou EfficientNet ;
- ajouter Grad-CAM pour l'explicabilité ;
- afficher la matrice de confusion dans l'application web ;
- ajouter l'historique d'entraînement dans le dashboard.

**À dire oralement :**  
La version actuelle prouve le concept. Les prochaines étapes serviraient à améliorer la précision, l'interprétation et l'interface.

---

## Slide 29 - Mini démo

Déroulement proposé :

1. Ouvrir l'application Flask.
2. Vérifier les checkpoints sur la page accueil.
3. Aller à la page classification.
4. Uploader une radiographie.
5. Lire la classe prédite et les probabilités.
6. Aller à la page génération.
7. Choisir une classe.
8. Générer une grille d'images synthétiques.

**À dire oralement :**  
Cette démo montre les deux parties importantes du projet : classifier une radiographie et générer de nouvelles images synthétiques.

---

## Slide 30 - Conclusion

Résumé :

- nous avons préparé le dataset COVID-19 Radiography Database ;
- nous avons entraîné un Conditional DCGAN ;
- nous avons généré des images synthétiques par classe ;
- nous avons entraîné un CNN avec données réelles et optionnellement synthétiques ;
- nous avons évalué le modèle ;
- nous avons créé une application Flask.

Conclusion finale :

**Les GAN peuvent aider à augmenter les données, mais leur utilisation en médecine doit rester prudente, mesurée et validée.**

**À dire oralement :**  
Ce travail montre comment les méthodes génératives peuvent soutenir un système de classification d'images médicales, tout en gardant une séparation claire entre prototype éducatif et usage clinique.
