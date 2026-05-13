# Contenu des Slides - Projet Deep Learning GAN Médical

## Slide 1 - Titre

**Classification des lésions de la peau avec Deep Learning et GAN**

Projet basé sur le dataset HAM10000.

Ce projet utilise deux modèles :

- un GAN pour générer de nouvelles images médicales synthétiques ;
- un CNN pour classifier les images de lésions cutanées.

**À dire oralement :**  
Dans ce projet, nous avons construit une chaîne complète de deep learning : préparation des données, génération d'images, entraînement d'un classificateur, évaluation, puis une petite application web.

---

## Slide 2 - Idée générale du projet

Le but est d'aider un modèle à mieux reconnaître différents types de lésions de la peau.

Problème principal :

- certaines classes ont beaucoup d'images ;
- d'autres classes ont très peu d'images ;
- cela peut rendre l'apprentissage déséquilibré.

Notre solution :

- utiliser les vraies images du dataset ;
- entraîner un GAN pour créer des images synthétiques ;
- ajouter ces images synthétiques à l'entraînement du classificateur.

**À dire oralement :**  
Le GAN ne remplace pas les données réelles. Il sert à ajouter plus d'exemples pour aider le modèle à apprendre plus de variations.

---

## Slide 3 - Plan de la présentation

Dans cette présentation, nous allons expliquer le projet étape par étape.

Plan :

- contexte et objectif du projet ;
- dataset HAM10000 ;
- préparation des données ;
- entraînement du GAN ;
- génération d'images synthétiques ;
- entraînement du classificateur CNN ;
- évaluation des résultats ;
- application web et démonstration.

**À dire oralement :**  
Nous allons suivre le même ordre que le projet réel : d'abord les données, ensuite la génération d'images, puis la classification et enfin l'application.

---

## Slide 4 - Dataset utilisé

Nous avons utilisé le dataset **HAM10000**.

Il contient des images dermoscopiques de lésions cutanées.

Le projet travaille avec 7 classes :

- actinic_keratoses ;
- basal_cell_carcinoma ;
- benign_keratosis ;
- dermatofibroma ;
- melanoma ;
- melanocytic_nevi ;
- vascular_lesions.

Les images sont organisées dans des dossiers par classe.

**À dire oralement :**  
Chaque image appartient à une classe. Le modèle apprend à associer une image avec sa bonne catégorie.

---

## Slide 5 - Préparation des données

Avant l'entraînement, les données doivent être préparées.

Étapes principales :

```text
Dataset brut -> Lecture des métadonnées -> Création des classes -> Train/Test -> Images prêtes
```

Dans le projet :

- les images sont séparées en entraînement et test ;
- la séparation est stratifiée pour garder les classes représentées ;
- les images sont redimensionnées en `128 x 128` ;
- les pixels sont normalisés entre `-1` et `1`.

Structure finale :

```text
data/ham10000/
  train/
    class_name/
  test/
    class_name/
```

**À dire oralement :**  
La préparation est importante parce que le modèle a besoin de données propres, bien organisées et dans le même format.

---

## Slide 6 - Pipeline complet du projet

Voici le déroulement complet :

```text
1. Préparer les images réelles
2. Entraîner le GAN
3. Générer des images synthétiques
4. Entraîner le CNN avec images réelles + synthétiques
5. Évaluer le modèle
6. Tester avec une application web
```

**À dire oralement :**  
Le projet n'est pas seulement un modèle. C'est une chaîne complète depuis les données jusqu'à une interface utilisable.

---

## Slide 7 - C'est quoi un GAN ?

Un GAN est un modèle génératif.

Il contient deux parties :

- **Generator** : crée de fausses images ;
- **Discriminator** : essaie de distinguer les vraies images des fausses.

Principe :

```text
Generator -> crée une image
Discriminator -> vérifie si elle semble vraie ou fausse
```

Avec le temps, le generator apprend à créer des images plus réalistes.

**À dire oralement :**  
On peut voir le GAN comme une compétition entre deux réseaux. L'un crée, l'autre juge.

---

## Slide 8 - Notre modèle GAN

Nous avons utilisé un **Conditional DCGAN**.

Cela veut dire que le GAN reçoit aussi la classe demandée.

Exemple :

```text
Bruit aléatoire + classe melanoma -> image synthétique melanoma
```

Dans le code :

- le generator utilise un vecteur de bruit et une classe ;
- le discriminator reçoit une image et sa classe ;
- le modèle peut générer des images pour les 7 classes.

**À dire oralement :**  
Le mot "conditional" signifie que nous pouvons contrôler la classe de l'image générée.

---

## Slide 9 - Architecture du Generator

Le generator transforme un petit vecteur aléatoire en image.

Étapes simples :

```text
Bruit + label de classe
-> couches de convolution transposée
-> image RGB 128 x 128
```

Sortie :

- une image couleur ;
- format `128 x 128` ;
- valeurs normalisées avec `Tanh`.

**À dire oralement :**  
Au début, le bruit ne contient pas d'image. Le generator apprend progressivement à transformer ce bruit en image médicale plausible.

---

## Slide 10 - Architecture du Discriminator

Le discriminator reçoit une image et prédit si elle est vraie ou générée.

Étapes simples :

```text
Image + label de classe
-> couches de convolution
-> probabilité vrai/faux
```

Il apprend avec deux types d'images :

- images réelles du dataset ;
- images synthétiques créées par le generator.

**À dire oralement :**  
Le discriminator force le generator à s'améliorer, car il devient de plus en plus difficile à tromper.

---

## Slide 11 - Entraînement du GAN

Pendant l'entraînement :

- le discriminator apprend à reconnaître les vraies et fausses images ;
- le generator apprend à produire des images qui semblent réelles ;
- les deux pertes sont suivies : `Loss_D` et `Loss_G`.

Paramètres principaux du projet :

- image size : `128` ;
- bruit latent : `100` ;
- optimizer : `Adam` ;
- learning rate : `0.0002` ;
- batch size recommandé : `64` ;
- epochs recommandés : `50`.

Les exemples générés sont sauvegardés dans :

```text
generated_samples/
```

**À dire oralement :**  
Un GAN demande souvent beaucoup de temps d'entraînement, surtout pour des images médicales.

---

## Slide 12 - Génération des images synthétiques

Après l'entraînement, nous utilisons le generator pour créer de nouvelles images.

Dans le projet, les images générées sont organisées par classe :

```text
synthetic_images/
  melanoma/
  basal_cell_carcinoma/
  ...
```

Deux possibilités :

- générer un nombre fixe d'images par classe ;
- générer le même nombre que les images réelles disponibles.

**À dire oralement :**  
Ces images synthétiques sont ensuite utilisées comme données supplémentaires pour entraîner le classificateur.

---

## Slide 13 - Utilisation du CNN dans notre projet

Dans notre projet, le CNN est utilisé pour classifier les images de lésions cutanées.

Entrée du modèle :

- une image RGB de taille `128 x 128`.

Sortie du modèle :

- une prédiction parmi les 7 classes ;
- une probabilité pour chaque classe.

Le CNN est entraîné avec deux types de données :

- les images réelles du dataset HAM10000 ;
- les images synthétiques générées par le GAN.

Objectif :

- vérifier si l'ajout des images générées améliore la classification.

**À dire oralement :**  
Le CNN est la partie qui prend la décision finale. Le GAN sert à enrichir les données, puis le CNN apprend avec ces données pour prédire la classe.

---

## Slide 14 - Notre classificateur CNN

Le modèle s'appelle `SimpleCNN`.

Architecture simplifiée :

```text
Image
-> bloc convolution 1
-> bloc convolution 2
-> bloc convolution 3
-> bloc convolution 4
-> couches fully connected
-> sortie 7 classes
```

Chaque bloc contient :

- convolution ;
- batch normalization ;
- ReLU ;
- max pooling.

La dernière partie contient :

- une couche linéaire ;
- dropout ;
- une sortie avec 7 scores.

**À dire oralement :**  
Même si le modèle est simple, il permet de tester clairement l'effet des images synthétiques.

---

## Slide 15 - Entraînement du CNN

Le CNN est entraîné avec :

- images réelles ;
- images synthétiques générées par le GAN ;
- augmentation de données.

Augmentations utilisées :

- retournement horizontal ;
- retournement vertical léger ;
- rotation ;
- léger changement de couleur.

Le modèle utilise :

- loss : `CrossEntropyLoss` ;
- optimizer : `Adam` ;
- validation interne ;
- meilleur modèle choisi selon la validation.

**À dire oralement :**  
Les augmentations aident le modèle à mieux généraliser, car une lésion peut apparaître avec différentes orientations ou luminosités.

---

## Slide 16 - Expérience réalisée

Nous pouvons comparer deux entraînements :

| Modèle | Données d'entraînement | Données de test |
| --- | --- | --- |
| Baseline CNN | Images réelles seulement | Images réelles |
| CNN augmenté | Images réelles + synthétiques | Images réelles |

Le test se fait seulement sur des images réelles.

**À dire oralement :**  
On teste toujours sur des vraies images pour vérifier si les images synthétiques aident réellement le modèle.

---

## Slide 17 - Évaluation

Les métriques utilisées :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- matrice de confusion.

Les résultats sont sauvegardés dans :

```text
checkpoints/skin_classifier_metrics.json
generated_samples/skin_classifier_confusion_matrix.png
generated_samples/skin_classifier_curves.png
```

Résultats à remplir après entraînement :

| Modèle | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| CNN baseline | TBD | TBD | TBD |
| CNN avec images GAN | TBD | TBD | TBD |

**À dire oralement :**  
La matrice de confusion permet de voir quelles classes sont faciles ou difficiles à distinguer.

---

## Slide 18 - Application web

Nous avons aussi une interface web avec Flask.

Elle contient deux pages principales :

- page classification ;
- page génération.

Page classification :

```text
Uploader une image -> prédiction -> probabilités par classe
```

Page génération :

```text
Choisir une classe -> générer des images synthétiques
```

**À dire oralement :**  
L'application permet de tester le modèle sans manipuler directement le code Python.

---

## Slide 19 - Ce que nous avons construit

Dans ce projet, nous avons construit :

- un script de préparation du dataset ;
- un Conditional DCGAN ;
- un générateur d'images synthétiques ;
- un CNN de classification ;
- un script d'évaluation ;
- des visualisations ;
- une application web Flask.

**À dire oralement :**  
Le projet montre comment combiner génération d'images et classification dans une même chaîne deep learning.

---

## Slide 20 - Difficultés rencontrées

Les principales difficultés :

- les images médicales sont complexes ;
- les classes du dataset ne sont pas équilibrées ;
- le GAN peut produire des images floues au début ;
- l'entraînement peut être long ;
- les résultats dépendent beaucoup du nombre d'epochs et de la puissance GPU.

**À dire oralement :**  
Le GAN est la partie la plus difficile, car il doit apprendre une distribution visuelle réaliste.

---

## Slide 21 - Limites du projet

Ce projet est un prototype éducatif.

Il ne doit pas être utilisé pour faire un vrai diagnostic médical.

Limites :

- dataset limité ;
- modèle CNN simple ;
- qualité des images synthétiques variable ;
- pas de validation clinique ;
- besoin de tests plus avancés.

**À dire oralement :**  
Dans le domaine médical, il faut toujours une validation par des experts avant une utilisation réelle.

---

## Slide 22 - Améliorations possibles

Pour améliorer le projet, on peut :

- entraîner plus longtemps le GAN ;
- utiliser un modèle plus puissant pour la classification ;
- essayer ResNet ou EfficientNet ;
- équilibrer mieux les classes ;
- comparer plusieurs méthodes d'augmentation ;
- ajouter plus de métriques ;
- améliorer l'interface web.

**À dire oralement :**  
La version actuelle prouve le concept. Les prochaines étapes serviraient à améliorer la précision et la robustesse.

---

## Slide 23 - Conclusion

Ce projet montre une utilisation complète du deep learning pour des images médicales.

Résumé :

- nous avons préparé le dataset HAM10000 ;
- nous avons entraîné un GAN conditionnel ;
- nous avons généré des images synthétiques ;
- nous avons entraîné un CNN avec les données réelles et synthétiques ;
- nous avons évalué le modèle ;
- nous avons créé une application web simple.

Conclusion finale :

**Les GAN peuvent aider à augmenter les données, mais leur utilisation en médecine doit rester prudente et bien évaluée.**

**À dire oralement :**  
Le point important est que l'IA peut aider dans l'analyse d'images médicales, mais elle doit être testée avec rigueur.

---

## Slide 24 - Mini démo

Déroulement proposé :

1. Ouvrir la page web.
2. Aller à la page classification.
3. Uploader une image de lésion.
4. Lire la classe prédite et les probabilités.
5. Aller à la page génération.
6. Choisir une classe.
7. Générer des images synthétiques.

**À dire oralement :**  
Cette démo montre les deux parties importantes du projet : classifier une image et générer de nouvelles images.

---

## Slide 25 - Message final

Le projet combine deux idées importantes :

- **générer** des images avec un GAN ;
- **classifier** des images avec un CNN.

La chaîne finale est :

```text
Données réelles -> GAN -> Images synthétiques -> CNN -> Prédiction -> Interface web
```

**À dire oralement :**  
Ce travail montre comment les méthodes génératives peuvent être utilisées pour soutenir un système de classification d'images médicales.
