window.PRESENTATION_SLIDES = [
  {
    id: 1,
    kicker: "Deep Learning Module",
    title: "Génération et classification de radiographies thoraciques COVID-19 avec GAN et CNN",
    subtitle: "Projet basé sur le dataset COVID-19 Radiography Database",
    tags: ["GAN", "CNN", "Flask"],
    visual: "hero-xray",
    bullets: [
      "Chaîne complète de deep learning médical",
      "Génération d'images synthétiques avec Conditional DCGAN",
      "Classification de radiographies avec CNN"
    ],
    note: "Dans ce projet, nous avons construit une chaîne complète de deep learning : téléchargement et préparation des données, entraînement d'un GAN, génération d'images synthétiques, entraînement d'un classificateur, évaluation, puis application web Flask."
  },
  {
    id: 2,
    kicker: "Objectif",
    title: "Idée générale du projet",
    subtitle: "Classifier les radiographies et étudier l'apport des images synthétiques",
    tags: ["Problème", "Solution"],
    visual: "problem-solution",
    bullets: [
      "Les modèles deep learning ont besoin de beaucoup d'exemples",
      "Les datasets médicaux peuvent être limités ou déséquilibrés",
      "Le GAN génère des images par classe pour enrichir l'entraînement"
    ],
    note: "Le GAN ne remplace pas les données réelles. Il sert à créer des exemples supplémentaires pour tester si cela peut aider le classificateur."
  },
  {
    id: 3,
    kicker: "Plan",
    title: "Plan de la présentation",
    subtitle: "Une progression de la donnée brute jusqu'à l'application web",
    tags: ["Roadmap"],
    visual: "roadmap",
    bullets: [
      "Dataset, préparation et prétraitement",
      "Conditional DCGAN et génération d'images",
      "CNN classifier, évaluation et application Flask"
    ],
    note: "Nous allons suivre le même ordre que le projet réel : les données, le GAN, les images synthétiques, le CNN, l'évaluation, puis l'application."
  },
  {
    id: 4,
    kicker: "Dataset",
    title: "Dataset utilisé",
    subtitle: "COVID-19 Radiography Database depuis Kaggle",
    tags: ["Kaggle", "4 classes"],
    visual: "class-cards",
    bullets: [
      "Classes : covid, lung_opacity, normal, viral_pneumonia",
      "Les dossiers de masques sont ignorés",
      "Le projet fait classification et génération, pas segmentation"
    ],
    note: "Le projet travaille sur des radiographies thoraciques COVID-19 avec quatre classes : covid, lung_opacity, normal et viral_pneumonia."
  },
  {
    id: 5,
    kicker: "Données",
    title: "Préparation des données",
    subtitle: "Téléchargement Kaggle puis création des dossiers train/test",
    tags: ["download_dataset.py", "prepare_dataset.py"],
    visual: "folder-tree",
    bullets: [
      "download_dataset.py télécharge le dataset",
      "prepare_dataset.py crée les dossiers par classe",
      "Séparation stratifiée avec test_size = 0.2"
    ],
    note: "La préparation transforme le dataset brut en dossiers faciles à lire avec PyTorch. Chaque classe a son propre dossier dans train et dans test."
  },
  {
    id: 6,
    kicker: "Prétraitement",
    title: "Prétraitement des images",
    subtitle: "Mettre toutes les images dans un format compatible PyTorch",
    tags: ["RGB", "Tensor", "Normalize"],
    visual: "preprocessing",
    bullets: [
      "Ouverture des images en RGB",
      "Resize puis conversion en tenseurs",
      "Normalisation entre -1 et 1",
      "Augmentations : flip, rotation, couleur"
    ],
    note: "Le prétraitement donne au modèle des images dans un format constant. Les augmentations aident le modèle à mieux généraliser."
  },
  {
    id: 7,
    kicker: "Workflow",
    title: "Pipeline complet du projet",
    subtitle: "Du dataset Kaggle jusqu'à la démonstration Flask",
    tags: ["Pipeline"],
    visual: "pipeline",
    bullets: [
      "Télécharger et préparer le dataset",
      "Entraîner le GAN puis générer les images",
      "Entraîner, évaluer et tester le CNN avec Flask"
    ],
    note: "Le projet n'est pas seulement un modèle. C'est une chaîne complète depuis les données jusqu'à une interface web utilisable."
  },
  {
    id: 8,
    kicker: "Architecture",
    title: "Architecture générale",
    subtitle: "Images réelles et synthétiques alimentent le classifier",
    tags: ["DCGAN", "SimpleCNN"],
    visual: "architecture",
    bullets: [
      "Le Conditional DCGAN apprend sur les images réelles",
      "Il produit des images synthétiques organisées par classe",
      "Le SimpleCNN apprend avec réel seul ou réel + synthétique"
    ],
    note: "Le GAN apprend à générer des images par classe. Ensuite, le CNN peut être entraîné avec les images réelles seulement ou avec les images réelles plus les images générées."
  },
  {
    id: 9,
    kicker: "GAN",
    title: "C'est quoi un GAN ?",
    subtitle: "Deux réseaux en compétition",
    tags: ["Generator", "Discriminator"],
    visual: "gan-battle",
    bullets: [
      "Le Generator crée des images synthétiques",
      "Le Discriminator distingue vrai et faux",
      "La compétition améliore progressivement le Generator"
    ],
    note: "Le GAN fonctionne comme une compétition entre deux modèles. Le generator apprend à créer des images plus réalistes, et le discriminator apprend à les détecter."
  },
  {
    id: 10,
    kicker: "Conditional DCGAN",
    title: "Notre Conditional DCGAN",
    subtitle: "Générer une classe précise grâce au label",
    tags: ["gan.py", "train_gan.py"],
    visual: "conditional",
    bullets: [
      "Entrées : bruit aléatoire + label de classe",
      "Sortie : image RGB synthétique",
      "On peut demander une classe précise : covid, normal, etc."
    ],
    note: "Le mot conditional signifie que nous pouvons demander au GAN de générer une classe précise, par exemple covid ou normal."
  },
  {
    id: 11,
    kicker: "Generator",
    title: "Generator",
    subtitle: "Transformer un vecteur latent en radiographie synthétique",
    tags: ["Latent 100", "Tanh"],
    visual: "generator",
    bullets: [
      "Bruit + embedding de classe",
      "Convolutions transposées",
      "Image RGB en sortie",
      "Taille supportée : 64x64 ou 128x128"
    ],
    note: "Au début, le bruit ne contient aucune information visuelle. Le generator apprend progressivement à transformer ce bruit en image qui ressemble à une radiographie."
  },
  {
    id: 12,
    kicker: "Discriminator",
    title: "Discriminator",
    subtitle: "Vérifier si une image semble réelle ou générée",
    tags: ["Real", "Fake"],
    visual: "discriminator",
    bullets: [
      "Entrée : image + label de classe",
      "Convolutions successives",
      "Sortie : probabilité vrai/faux",
      "Apprend avec images réelles et synthétiques"
    ],
    note: "Le discriminator force le generator à s'améliorer. Si les images générées sont mauvaises, le discriminator les reconnaît facilement."
  },
  {
    id: 13,
    kicker: "Training",
    title: "Entraînement du GAN",
    subtitle: "Script train_gan.py",
    tags: ["BCELoss", "Adam", "0.0002"],
    visual: "training-chart",
    bullets: [
      "Loss : BCELoss",
      "Optimizer : Adam",
      "Batch size : 32 ou 64",
      "Epochs recommandés : 50"
    ],
    note: "Pendant l'entraînement, le projet sauvegarde des grilles d'images pour suivre visuellement la progression du GAN."
  },
  {
    id: 14,
    kicker: "Synthétique",
    title: "Génération des images synthétiques",
    subtitle: "Script generate_synthetic.py",
    tags: ["synthetic_covid_images"],
    visual: "synthetic-folders",
    bullets: [
      "Images générées par classe",
      "Nombre fixe d'images ou même nombre que le réel",
      "Structure compatible avec l'entraînement du CNN"
    ],
    note: "Les images synthétiques sont sauvegardées comme un dataset normal. Cela permet au CNN de les utiliser avec le même code que les vraies images."
  },
  {
    id: 15,
    kicker: "CNN",
    title: "Notre classificateur CNN",
    subtitle: "SimpleCNN pour classifier les radiographies",
    tags: ["classifier.py", "4 classes"],
    visual: "classifier",
    bullets: [
      "Entrée : radiographie RGB 128x128",
      "Sortie : prédiction parmi 4 classes",
      "Probabilité calculée pour chaque classe"
    ],
    note: "Le CNN est la partie qui prend la décision finale. Il regarde l'image et prédit sa catégorie."
  },
  {
    id: 16,
    kicker: "Architecture CNN",
    title: "Architecture du SimpleCNN",
    subtitle: "Quatre blocs convolutionnels puis classification",
    tags: ["Conv", "BatchNorm", "ReLU", "MaxPool"],
    visual: "cnn-layers",
    bullets: [
      "Conv block 1 à 4",
      "Batch normalization + ReLU + max pooling",
      "Linear layer + Dropout",
      "Sortie : 4 classes"
    ],
    note: "Le modèle est volontairement simple pour rester compréhensible et rapide à entraîner. Il peut être remplacé plus tard par ResNet ou EfficientNet."
  },
  {
    id: 17,
    kicker: "Training CNN",
    title: "Entraînement du CNN",
    subtitle: "Script train_classifier.py",
    tags: ["CrossEntropy", "Validation 15%"],
    visual: "training-comparison",
    bullets: [
      "Option 1 : images réelles seulement",
      "Option 2 : images réelles + synthétiques",
      "Meilleur modèle choisi avec l'accuracy de validation"
    ],
    note: "Le projet permet de comparer un modèle baseline avec un modèle augmenté par des images générées par GAN."
  },
  {
    id: 18,
    kicker: "Expérience",
    title: "Expérience réalisée",
    subtitle: "Comparer baseline et augmentation par GAN",
    tags: ["Baseline", "Augmented"],
    visual: "comparison-table",
    bullets: [
      "CNN baseline : réel seulement",
      "CNN augmenté : réel + synthétique",
      "Le test doit rester sur des images réelles"
    ],
    note: "Cette comparaison permet de voir si les images synthétiques aident réellement le classificateur sur des données réelles."
  },
  {
    id: 19,
    kicker: "Évaluation",
    title: "Évaluation",
    subtitle: "Métriques et résultats à remplir après entraînement",
    tags: ["Accuracy", "F1-score", "Confusion matrix"],
    visual: "metrics",
    bullets: [
      "Accuracy, precision, recall, F1-score",
      "Matrice de confusion",
      "Résultats réels non inventés : TBD"
    ],
    note: "La matrice de confusion montre quelles classes sont bien reconnues et quelles classes sont confondues."
  },
  {
    id: 20,
    kicker: "Application",
    title: "Application web Flask",
    subtitle: "Interface locale pour tester les modèles",
    tags: ["frontend/app.py", "Flask"],
    visual: "flask-app",
    bullets: [
      "Charge covid_dcgan.pth",
      "Charge covid_classifier.pth",
      "Pages : accueil, classification, génération"
    ],
    note: "L'application permet de tester les modèles sans utiliser directement les scripts Python."
  },
  {
    id: 21,
    kicker: "Dashboard",
    title: "Page accueil",
    subtitle: "Tableau de bord du projet",
    tags: ["Status", "Metrics"],
    visual: "dashboard",
    bullets: [
      "Nom du dataset et device CPU/CUDA",
      "Liste des classes",
      "Statut des checkpoints",
      "Métriques si disponibles"
    ],
    note: "La page d'accueil sert de tableau de bord. Elle indique rapidement si les modèles entraînés sont disponibles."
  },
  {
    id: 22,
    kicker: "Classification",
    title: "Page classification",
    subtitle: "Uploader une radiographie et lire la prédiction",
    tags: ["Upload", "Probabilités"],
    visual: "upload-prediction",
    bullets: [
      "Upload image puis prétraitement",
      "Passage dans SimpleCNN",
      "Classe prédite + probabilités"
    ],
    note: "La classification utilise le même type de prétraitement que pendant l'entraînement, ce qui garde une cohérence entre training et utilisation."
  },
  {
    id: 23,
    kicker: "Génération",
    title: "Page génération",
    subtitle: "Créer une grille de radiographies synthétiques",
    tags: ["36 images", "6 x 6"],
    visual: "generation-grid",
    bullets: [
      "Choisir toutes les classes ou une seule classe",
      "Générer 36 images",
      "Afficher une grille 6 x 6"
    ],
    note: "Cette page montre directement le résultat du GAN. On peut choisir une classe et observer les images synthétiques produites."
  },
  {
    id: 24,
    kicker: "Commandes",
    title: "Commandes importantes",
    subtitle: "Workflow complet du projet",
    tags: ["Terminal"],
    visual: "terminal",
    bullets: [
      "download_dataset.py et prepare_dataset.py",
      "train_gan.py puis generate_synthetic.py",
      "train_classifier.py, evaluate.py, Flask"
    ],
    note: "Ces commandes représentent le workflow complet : données, GAN, images synthétiques, classifier, évaluation et application."
  },
  {
    id: 25,
    kicker: "Livrables",
    title: "Ce que nous avons construit",
    subtitle: "Un projet complet et exécutable",
    tags: ["Built"],
    visual: "checklist",
    bullets: [
      "Scripts Kaggle et préparation train/test",
      "Conditional DCGAN et générateur synthétique",
      "CNN classifier, évaluation et Flask app"
    ],
    note: "Le projet montre comment connecter génération d'images et classification dans une seule chaîne deep learning."
  },
  {
    id: 26,
    kicker: "Challenges",
    title: "Difficultés rencontrées",
    subtitle: "Les points les plus coûteux du projet",
    tags: ["GPU", "GAN"],
    visual: "challenges",
    bullets: [
      "L'entraînement d'un GAN peut être long",
      "Les images générées peuvent être floues au début",
      "Les classes médicales peuvent être difficiles à distinguer"
    ],
    note: "Le GAN est souvent la partie la plus difficile, car il doit apprendre une distribution visuelle réaliste."
  },
  {
    id: 27,
    kicker: "Limites",
    title: "Limites du projet",
    subtitle: "Prototype éducatif, pas outil médical réel",
    tags: ["Disclaimer"],
    visual: "warning",
    bullets: [
      "Pas de validation clinique",
      "SimpleCNN reste un modèle léger",
      "Qualité synthétique variable",
      "Résultats à interpréter avec prudence"
    ],
    note: "Ce projet est un prototype éducatif. Il ne doit pas être utilisé pour établir un diagnostic médical réel sans validation clinique et expertise médicale."
  },
  {
    id: 28,
    kicker: "Future work",
    title: "Améliorations possibles",
    subtitle: "Rendre le système plus robuste et plus explicable",
    tags: ["ResNet", "Grad-CAM"],
    visual: "improvements",
    bullets: [
      "Entraîner le GAN plus longtemps",
      "Tester ResNet, DenseNet ou EfficientNet",
      "Ajouter Grad-CAM et historique d'entraînement"
    ],
    note: "La version actuelle prouve le concept. Les prochaines étapes serviraient à améliorer la précision, l'interprétation et l'interface."
  },
  {
    id: 29,
    kicker: "Démo",
    title: "Mini démo",
    subtitle: "Scénario de démonstration devant la classe",
    tags: ["Flask", "Demo"],
    visual: "demo-flow",
    bullets: [
      "Ouvrir Flask et vérifier les checkpoints",
      "Uploader une radiographie et lire la prédiction",
      "Générer une grille synthétique par classe"
    ],
    note: "Cette démo montre les deux parties importantes du projet : classifier une radiographie et générer de nouvelles images synthétiques."
  },
  {
    id: 30,
    kicker: "Conclusion",
    title: "Conclusion",
    subtitle: "GAN + CNN + Flask dans une chaîne deep learning complète",
    tags: ["GAN", "CNN", "Flask"],
    visual: "conclusion",
    bullets: [
      "Dataset COVID-19 préparé",
      "Conditional DCGAN entraîné",
      "Images synthétiques générées par classe",
      "CNN entraîné et évalué",
      "Application Flask créée"
    ],
    note: "Les GAN peuvent aider à augmenter les données, mais leur utilisation en médecine doit rester prudente, mesurée et validée."
  }
];

window.TEAM_MEMBERS = [
  "HASSAN EL KHATOURY",
  "FAHD CHAFAI",
  "MOHSSINE ECHLAIHI"
];
