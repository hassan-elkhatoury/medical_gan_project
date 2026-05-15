# Présentation détaillée - Projet Deep Learning GAN Médical COVID-19

## Direction artistique générale

La présentation doit suivre le style de la première slide fournie :

- fond bleu nuit / noir, très contrasté ;
- accents cyan et bleu électrique ;
- grande typographie blanche, très lisible ;
- mots importants en bleu lumineux, par exemple **GAN**, **CNN**, **COVID-19** ;
- ambiance médicale IA : radiographie thoracique, grille technique, lignes réseau, points lumineux ;
- contenu simple, peu chargé, avec 3 à 5 idées maximum par slide ;
- chaque slide doit ressembler à une vraie slide de soutenance, pas à une page de rapport.

Structure visuelle recommandée :

- en haut à gauche : petit label de section, par exemple `DEEP LEARNING MODULE`, `DATASET`, `GAN`, `CNN` ;
- au centre gauche : titre principal ;
- sous le titre : sous-titre ou idée clé ;
- à droite : espace visuel ou schéma simple ;
- en bas : ligne fine cyan, compteur de slide, ou rappel du pipeline ;
- arrière-plan : gradient sombre + légers points réseau + glow cyan discret.

Palette :

```text
Fond principal : #020617 / #030712
Bleu profond : #071B3A
Cyan : #22D3EE
Bleu électrique : #60A5FA
Texte principal : #F8FAFC
Texte secondaire : #B7C9E2
```

Typographie :

- titre : très grand, gras, blanc ;
- mots clés : bleu lumineux ;
- bullets : courts, espacés, faciles à lire ;
- notes orales : hors slide ou en panneau séparé.

---

## Slide 1 - Titre

### Contenu affiché

**DEEP LEARNING MODULE**

**Génération et classification de radiographies thoraciques COVID-19 avec GAN et CNN**

Projet basé sur le dataset **COVID-19 Radiography Database**

Badges :

- GAN
- CNN
- Flask

Équipe :

- HASSAN EL KHATOURY
- FAHD CHAFAI
- MOHSSINE ECHLAIHI

### Apparence de la slide

- Même style que l'image d'inspiration.
- À gauche : gros titre sur 3 lignes.
- Les mots **GAN** et **CNN** doivent être en bleu lumineux.
- À droite : grande radiographie thoracique en bleu, avec une zone de détection encadrée en cyan.
- En arrière-plan : réseau de neurones discret en haut, vagues de points numériques en bas.
- En bas : bande avec les noms des membres, séparés par des lignes verticales fines.

### À dire oralement

Dans ce projet, nous avons construit une chaîne complète de deep learning : téléchargement et préparation des données, entraînement d'un GAN, génération d'images synthétiques, entraînement d'un classificateur, évaluation, puis application web Flask.

---

## Slide 2 - Idée générale du projet

### Contenu affiché

**Objectif :** classifier des radiographies thoraciques et étudier l'utilisation d'images synthétiques générées par GAN.

**Problème**

- Les modèles deep learning ont besoin de beaucoup d'exemples.
- Les datasets médicaux peuvent être limités.
- Certaines classes sont difficiles à apprendre.

**Solution**

- Préparer les vraies images.
- Générer des images synthétiques avec un Conditional GAN.
- Entraîner un CNN.
- Tester sur des images réelles.

### Apparence de la slide

- Slide divisée en deux grandes colonnes.
- Colonne gauche : `PROBLÈME`, bordure rouge/orange très discrète.
- Colonne droite : `SOLUTION`, bordure cyan/bleue.
- Au centre : petite flèche lumineuse allant de problème vers solution.
- Fond sombre avec un léger motif de grille.

### À dire oralement

Le GAN ne remplace pas les données réelles. Il sert à créer des exemples supplémentaires pour tester si cela peut aider le classificateur.

---

## Slide 3 - Plan de la présentation

### Contenu affiché

1. Contexte et objectif
2. Dataset
3. Préparation des données
4. Architecture générale
5. Conditional DCGAN
6. Images synthétiques
7. CNN classifier
8. Évaluation
9. Application Flask
10. Limites et améliorations

### Apparence de la slide

- Timeline verticale au centre ou à gauche.
- Chaque étape dans une capsule sombre avec numéro cyan.
- Une ligne lumineuse relie les étapes.
- À droite : mini résumé du pipeline `Dataset -> GAN -> CNN -> Flask`.

### À dire oralement

Nous allons suivre le même ordre que le projet réel : les données, le GAN, les images synthétiques, le CNN, l'évaluation, puis l'application.

---

## Slide 4 - Dataset utilisé

### Contenu affiché

Dataset : **COVID-19 Radiography Database**

Classes utilisées :

- `covid`
- `lung_opacity`
- `normal`
- `viral_pneumonia`

Note : les dossiers de masques sont ignorés, car le projet ne fait pas de segmentation.

### Apparence de la slide

- Quatre cartes alignées en grille 2 x 2.
- Chaque carte contient le nom d'une classe.
- Couleur dominante : bleu/cyan.
- À droite ou en arrière-plan : silhouette abstraite de radiographie thoracique.
- Ajouter un petit label : `4 classes de radiographies`.

### À dire oralement

Le projet travaille sur des radiographies thoraciques COVID-19 avec quatre classes : covid, lung_opacity, normal et viral_pneumonia.

---

## Slide 5 - Préparation des données

### Contenu affiché

Scripts :

```text
download_dataset.py
prepare_dataset.py
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

Split test : `0.2`

### Apparence de la slide

- À gauche : deux blocs de scripts avec icône terminal.
- À droite : arbre de dossiers dans un panneau style terminal.
- Fond sombre, texte monospace cyan/blanc.
- Ajouter une petite étiquette : `Train/Test stratifié`.

### À dire oralement

La préparation transforme le dataset brut en dossiers faciles à lire avec PyTorch. Chaque classe a son propre dossier dans train et dans test.

---

## Slide 6 - Prétraitement des images

### Contenu affiché

Prétraitement :

- ouvrir les images en RGB ;
- redimensionner ;
- convertir en tenseurs PyTorch ;
- normaliser entre `-1` et `1`.

Augmentations :

- horizontal flip ;
- vertical flip faible ;
- rotation ;
- changement léger de luminosité, contraste et saturation.

### Apparence de la slide

- Schéma horizontal : `Image brute -> Resize -> Tensor -> Normalisation`.
- Les étapes sont dans des blocs lumineux connectés par flèches.
- En bas : petite section `Augmentations` avec quatre badges.
- Style propre, pas trop de texte.

### À dire oralement

Le prétraitement donne au modèle des images dans un format constant. Les augmentations aident le modèle à mieux généraliser.

---

## Slide 7 - Pipeline complet du projet

### Contenu affiché

1. Télécharger le dataset Kaggle
2. Préparer les dossiers train/test
3. Entraîner le Conditional DCGAN
4. Générer des images synthétiques
5. Entraîner le CNN
6. Évaluer sur les vraies images de test
7. Tester avec Flask

### Apparence de la slide

- Pipeline horizontal avec sept étapes.
- Chaque étape est une capsule avec numéro.
- Les flèches sont cyan avec effet glow.
- Le fond contient une ligne ondulée de points numériques en bas.

### À dire oralement

Le projet n'est pas seulement un modèle. C'est une chaîne complète depuis les données jusqu'à une interface web utilisable.

---

## Slide 8 - Architecture générale

### Contenu affiché

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
Images synthétiques -------> CNN SimpleCNN
                              |
                              v
                         Évaluation + Flask
```

### Apparence de la slide

- Grand flowchart au centre.
- Les blocs principaux doivent être grands et lisibles.
- `Conditional DCGAN` en bleu électrique.
- `CNN SimpleCNN` en cyan.
- Montrer clairement que les images réelles peuvent aller directement vers le CNN.

### À dire oralement

Le GAN apprend à générer des images par classe. Ensuite, le CNN peut être entraîné avec les images réelles seulement ou avec les images réelles plus les images générées.

---

## Slide 9 - C'est quoi un GAN ?

### Contenu affiché

Un GAN contient deux réseaux :

- **Generator** : crée des images synthétiques.
- **Discriminator** : distingue les vraies images des fausses.

Principe :

```text
Bruit -> Generator -> Image synthétique
Image réelle ou synthétique -> Discriminator -> Vrai/Faux
```

### Apparence de la slide

- Deux grands blocs face à face : `Generator` et `Discriminator`.
- Entre les deux : `compétition`.
- Utiliser une ligne lumineuse entre les deux blocs.
- En bas : mini phrase `Le Generator s'améliore grâce au Discriminator`.

### À dire oralement

Le GAN fonctionne comme une compétition entre deux modèles. Le generator apprend à créer des images plus réalistes, et le discriminator apprend à les détecter.

---

## Slide 10 - Notre Conditional DCGAN

### Contenu affiché

Fichiers :

- `gan.py`
- `train_gan.py`

Entrées du Generator :

- vecteur de bruit aléatoire ;
- label de classe.

Sortie :

- radiographie thoracique synthétique RGB.

### Apparence de la slide

- Schéma simple : `Bruit + Label -> Generator -> Image synthétique`.
- Mettre `Conditional` en badge bleu.
- À droite : carte expliquant que le label permet de choisir la classe.
- Exemples de labels : `covid`, `normal`.

### À dire oralement

Le mot conditional signifie que nous pouvons demander au GAN de générer une classe précise, par exemple covid ou normal.

---

## Slide 11 - Generator

### Contenu affiché

Flux :

```text
Bruit + embedding de classe
-> convolutions transposées
-> image RGB
-> Tanh
```

Paramètres :

- bruit latent : `100`
- canaux : `3`
- taille : `64 x 64` ou `128 x 128`

### Apparence de la slide

- Diagramme de couches qui s'élargissent de gauche à droite.
- Départ : petit vecteur `z + label`.
- Arrivée : grand bloc `Image RGB`.
- Couleurs : début violet/bleu, fin cyan.

### À dire oralement

Au début, le bruit ne contient aucune information visuelle. Le generator apprend progressivement à transformer ce bruit en image qui ressemble à une radiographie.

---

## Slide 12 - Discriminator

### Contenu affiché

Flux :

```text
Image + label de classe
-> convolutions
-> probabilité vrai/faux
```

Apprentissage avec :

- images réelles ;
- images synthétiques.

### Apparence de la slide

- À gauche : bloc `Image + Label`.
- Au centre : pile de couches convolutionnelles.
- À droite : deux sorties `Real` et `Fake`.
- Utiliser des couleurs différentes pour réel et faux.

### À dire oralement

Le discriminator force le generator à s'améliorer. Si les images générées sont mauvaises, le discriminator les reconnaît facilement.

---

## Slide 13 - Entraînement du GAN

### Contenu affiché

Script : `train_gan.py`

Paramètres :

- `BCELoss`
- `Adam`
- learning rate : `0.0002`
- batch size : `32` ou `64`
- epochs recommandés : `50`

Checkpoints :

```text
checkpoints/covid_dcgan.pth
checkpoints/generator.pth
```

### Apparence de la slide

- Style tableau de bord d'entraînement.
- À gauche : paramètres d'entraînement.
- À droite : courbes Loss_G et Loss_D en exemple visuel.
- En bas : chemin des checkpoints dans une barre terminal.
- Marquer les courbes comme `Exemple visuel`.

### À dire oralement

Pendant l'entraînement, le projet sauvegarde des grilles d'images pour suivre visuellement la progression du GAN.

---

## Slide 14 - Génération des images synthétiques

### Contenu affiché

Script : `generate_synthetic.py`

Sortie :

```text
synthetic_covid_images/
  covid/
  lung_opacity/
  normal/
  viral_pneumonia/
```

Modes :

- nombre fixe par classe ;
- même nombre que les images réelles.

### Apparence de la slide

- À gauche : bloc `Generator entraîné`.
- À droite : quatre dossiers de classes.
- Utiliser des flèches vers les dossiers.
- Style terminal/folder tree, très lisible.

### À dire oralement

Les images synthétiques sont sauvegardées comme un dataset normal. Cela permet au CNN de les utiliser avec le même code que les vraies images.

---

## Slide 15 - Notre classificateur CNN

### Contenu affiché

Classificateur : `SimpleCNN`

Fichier : `classifier.py`

Entrée :

- radiographie RGB ;
- taille par défaut : `128 x 128`.

Sortie :

- prédiction parmi 4 classes ;
- probabilité pour chaque classe.

### Apparence de la slide

- Schéma : `Radiographie -> SimpleCNN -> Probabilités`.
- À droite : barres de probabilité en exemple visuel.
- Ne pas inventer de vraies performances ; écrire `Exemple visuel`.

### À dire oralement

Le CNN est la partie qui prend la décision finale. Il regarde l'image et prédit sa catégorie.

---

## Slide 16 - Architecture du SimpleCNN

### Contenu affiché

```text
Image
-> Conv block 1
-> Conv block 2
-> Conv block 3
-> Conv block 4
-> Linear layer
-> Dropout
-> Output 4 classes
```

Chaque bloc contient :

- convolution ;
- batch normalization ;
- ReLU ;
- max pooling.

### Apparence de la slide

- Diagramme de couches CNN de gauche à droite.
- Les quatre blocs convolutionnels sont similaires.
- La sortie `4 classes` doit être mise en évidence en cyan.
- Ajouter une petite note : `modèle simple, rapide à entraîner`.

### À dire oralement

Le modèle est volontairement simple pour rester compréhensible et rapide à entraîner. Il peut être remplacé plus tard par ResNet ou EfficientNet.

---

## Slide 17 - Entraînement du CNN

### Contenu affiché

Script : `train_classifier.py`

Deux modes :

- images réelles seulement ;
- images réelles + images synthétiques.

Paramètres :

- `CrossEntropyLoss`
- `Adam`
- validation interne : `15%`
- meilleur modèle selon l'accuracy de validation.

### Apparence de la slide

- Deux colonnes :
  - `Baseline`
  - `Augmenté avec GAN`
- Mettre les paramètres dans une petite carte centrale.
- Ajouter une flèche vers `Best checkpoint`.

### À dire oralement

Le projet permet de comparer un modèle baseline avec un modèle augmenté par des images générées par GAN.

---

## Slide 18 - Expérience réalisée

### Contenu affiché

| Modèle | Données d'entraînement | Données de test |
| --- | --- | --- |
| CNN baseline | Images réelles seulement | Images réelles |
| CNN augmenté | Images réelles + synthétiques | Images réelles |

Important :

- le test se fait uniquement sur des images réelles ;
- les images synthétiques servent seulement à l'entraînement.

### Apparence de la slide

- Grand tableau central avec deux lignes.
- Mettre `Images réelles` en bleu clair.
- Mettre `Images synthétiques` en cyan.
- Ajouter un avertissement discret : `Test réel uniquement`.

### À dire oralement

Cette comparaison permet de voir si les images synthétiques aident réellement le classificateur sur des données réelles.

---

## Slide 19 - Évaluation

### Contenu affiché

Métriques :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- matrice de confusion.

Fichiers :

```text
checkpoints/covid_classifier_metrics.json
generated_samples/covid_classifier_confusion_matrix.png
generated_samples/covid_classifier_curves.png
```

Résultats :

```text
CNN baseline : TBD
CNN avec images GAN : TBD
```

### Apparence de la slide

- Style dashboard.
- À gauche : cartes métriques.
- À droite : matrice de confusion en exemple visuel avec `TBD`.
- En bas : fichiers générés en petit bloc terminal.
- Écrire clairement : `Résultats à remplir après entraînement`.

### À dire oralement

La matrice de confusion montre quelles classes sont bien reconnues et quelles classes sont confondues.

---

## Slide 20 - Application web Flask

### Contenu affiché

Fichier : `frontend/app.py`

L'application charge :

```text
checkpoints/covid_dcgan.pth
checkpoints/covid_classifier.pth
```

Pages :

- accueil ;
- classification ;
- génération.

### Apparence de la slide

- Mockup d'application web sombre.
- Barre latérale avec `Home`, `Classification`, `Generation`.
- Au centre : trois cartes représentant les pages.
- Style dashboard médical IA.

### À dire oralement

L'application permet de tester les modèles sans utiliser directement les scripts Python.

---

## Slide 21 - Page accueil

### Contenu affiché

La page accueil affiche :

- nom du dataset ;
- device utilisé : CPU ou CUDA ;
- liste des classes ;
- statut checkpoint GAN ;
- statut checkpoint classifier ;
- métriques si disponibles.

### Apparence de la slide

- Dashboard avec cartes de statut.
- Utiliser des badges `OK` / `Missing`.
- Afficher `CPU/CUDA` comme un indicateur technique.
- Garder beaucoup d'espace pour ne pas surcharger.

### À dire oralement

La page d'accueil sert de tableau de bord. Elle indique rapidement si les modèles entraînés sont disponibles.

---

## Slide 22 - Page classification

### Contenu affiché

```text
Uploader une image
-> prétraitement
-> SimpleCNN
-> classe prédite + probabilités
```

Affichage :

- aperçu de l'image uploadée ;
- classe prédite ;
- probabilités par classe.

### Apparence de la slide

- À gauche : bloc upload.
- Au centre : flèche vers `SimpleCNN`.
- À droite : carte de résultat avec barres de probabilités.
- Marquer les probabilités comme `Exemple visuel` si aucun résultat réel.

### À dire oralement

La classification utilise le même type de prétraitement que pendant l'entraînement, ce qui garde une cohérence entre training et utilisation.

---

## Slide 23 - Page génération

### Contenu affiché

```text
Choisir une classe
-> générer 36 images
-> afficher une grille 6 x 6
```

Options :

- toutes les classes ;
- une seule classe.

Fichier :

```text
frontend/static/generated/latest_samples.png
```

### Apparence de la slide

- À gauche : menu de choix de classe.
- À droite : grille 6 x 6 stylisée.
- En haut : titre `Synthetic generation`.
- Ne pas montrer de fausses images comme résultat réel ; utiliser une grille placeholder si nécessaire.

### À dire oralement

Cette page montre directement le résultat du GAN. On peut choisir une classe et observer les images synthétiques produites.

---

## Slide 24 - Commandes importantes

### Contenu affiché

```bash
python download_dataset.py --output_dir data/covid_radiography_raw
python prepare_dataset.py --raw_dir data/covid_radiography_raw --output_dir data/covid_radiography --test_size 0.2 --overwrite
python train_gan.py --train_dir data/covid_radiography/train --epochs 50 --batch_size 32 --image_size 64
python generate_synthetic.py --checkpoint checkpoints/covid_dcgan.pth --match_real_dir data/covid_radiography/train --out_dir synthetic_covid_images
python train_classifier.py --real_train_dir data/covid_radiography/train --synthetic_train_dir synthetic_covid_images --test_dir data/covid_radiography/test --epochs 20
python evaluate.py --checkpoint checkpoints/covid_classifier.pth --test_dir data/covid_radiography/test
flask --app frontend/app.py run --host 127.0.0.1 --port 5000
```

### Apparence de la slide

- Plein écran style terminal.
- Fond noir, texte cyan/vert clair.
- Les commandes longues peuvent être plus petites.
- Ajouter un titre discret : `Workflow commands`.

### À dire oralement

Ces commandes représentent le workflow complet : données, GAN, images synthétiques, classifier, évaluation et application.

---

## Slide 25 - Ce que nous avons construit

### Contenu affiché

- script de téléchargement Kaggle ;
- script de préparation train/test ;
- Conditional DCGAN ;
- générateur d'images synthétiques ;
- CNN classifier ;
- script d'évaluation ;
- application Flask.

### Apparence de la slide

- Checklist de sept cartes.
- Chaque carte a une coche cyan.
- Mettre `Full pipeline` comme label central.
- Style très positif, slide de synthèse.

### À dire oralement

Le projet montre comment connecter génération d'images et classification dans une seule chaîne deep learning.

---

## Slide 26 - Difficultés rencontrées

### Contenu affiché

- L'entraînement d'un GAN peut être long.
- Les images générées peuvent être floues au début.
- Les résultats dépendent du nombre d'epochs.
- Les classes médicales peuvent être difficiles à distinguer.
- La puissance GPU influence fortement le temps d'entraînement.

### Apparence de la slide

- Cartes `Challenge`.
- Icône ou symbole d'avertissement discret.
- Couleurs : cyan + jaune léger pour attirer l'attention.
- Garder le texte court.

### À dire oralement

Le GAN est souvent la partie la plus difficile, car il doit apprendre une distribution visuelle réaliste.

---

## Slide 27 - Limites du projet

### Contenu affiché

Ce projet est un prototype éducatif.

Il ne doit pas être utilisé pour faire un vrai diagnostic médical.

Limites :

- pas de validation clinique ;
- modèle CNN simple ;
- qualité synthétique variable ;
- dataset limité à une source ;
- résultats à interpréter avec prudence.

### Apparence de la slide

- Panneau d'avertissement médical.
- Bordure jaune/cyan.
- Mettre la phrase `Prototype éducatif` en grand.
- Ajouter le disclaimer clairement en bas.

### À dire oralement

Dans le domaine médical, un modèle doit être validé par des experts et testé beaucoup plus largement avant une utilisation réelle.

---

## Slide 28 - Améliorations possibles

### Contenu affiché

- Entraîner le GAN plus longtemps.
- Comparer real-only vs real + synthetic.
- Utiliser ResNet, DenseNet ou EfficientNet.
- Ajouter Grad-CAM.
- Afficher la matrice de confusion dans l'app.
- Ajouter l'historique d'entraînement.

### Apparence de la slide

- Roadmap futur en 5 ou 6 étapes.
- Flèche vers la droite.
- Couleurs : bleu vers cyan pour montrer l'évolution.
- Les modèles `ResNet`, `DenseNet`, `EfficientNet` peuvent être dans des badges.

### À dire oralement

La version actuelle prouve le concept. Les prochaines étapes serviraient à améliorer la précision, l'interprétation et l'interface.

---

## Slide 29 - Mini démo

### Contenu affiché

1. Ouvrir l'application Flask.
2. Vérifier les checkpoints.
3. Aller à la page classification.
4. Uploader une radiographie.
5. Lire la classe prédite.
6. Aller à la page génération.
7. Choisir une classe.
8. Générer une grille synthétique.

### Apparence de la slide

- Timeline de démo en 8 étapes.
- Chaque étape courte dans une capsule.
- Mettre `Démo live` en badge cyan.
- Ajouter un rappel : `Préparer les checkpoints avant la soutenance`.

### À dire oralement

Cette démo montre les deux parties importantes du projet : classifier une radiographie et générer de nouvelles images synthétiques.

---

## Slide 30 - Conclusion

### Contenu affiché

Résumé :

- dataset COVID-19 préparé ;
- Conditional DCGAN entraîné ;
- images synthétiques générées ;
- CNN entraîné et évalué ;
- application Flask créée.

Conclusion :

**Les GAN peuvent aider à augmenter les données, mais leur utilisation en médecine doit rester prudente, mesurée et validée.**

### Apparence de la slide

- Slide finale très propre.
- Grand titre `Conclusion`.
- Pipeline final en une ligne : `Dataset -> GAN -> Synthetic images -> CNN -> Flask`.
- En bas : noms des membres de l'équipe.
- Fond similaire à la slide 1, mais plus minimal.

### À dire oralement

Ce travail montre comment les méthodes génératives peuvent soutenir un système de classification d'images médicales, tout en gardant une séparation claire entre prototype éducatif et usage clinique.
