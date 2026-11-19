
# Reconnaissance des emotions faciales sur FER-2013 : comparaison CNN vs Vision Transformer (MobileViT)

Projet réalisé en binôme dans `reconnaisance_des_emotions` autour de la classification des expressions faciales sur le jeu de données **FER-2013** (c'est un jeu de données de référence pour la reconnaissance automatique des expressions faciales, créé par Pierre-Luc Carrier et Aaron Courville ), comparant une approche convolutionnelle locale (`CNN_Modi_et_al.ipynb`, `48×48`) et une approche basée sur les *Vision Transformers* / **MobileViT** (`vision_transformer_et_comparaison.ipynb`, `128×128`, inspirée des travaux sur les *Vision Transformers* type Xue et al.), avec une comparaison directe et test temps réel via webcam.

## Auteurs (Binôme)
- **RANDRIANJAFY Voahanginiaina Roberte** 
- **RAJAOVELONA Hidekela Fanaperantsoa**

---

## Les objectifs et la méthode utilisée

### Objectifs
- Comparer deux familles d'architectures sur la classification des expressions faciales (**FER-2013**) :
  - **CNN (`CNN_Modi_et_al.ipynb`)** : extraction de caractéristiques locales sur des images en **48×48**.
  - **Vision Transformer avec pooling attentionnel (`vision_transformer_et_comparaison.ipynb`)** : modélisation par auto-attention (*self-attention*) et agrégation par **pooling attentionnel** sur les représentations faciales, inspirée de Xue et al.
- Évaluer les modèles à l'aide des courbes d'apprentissage (`loss` / `accuracy`), des **matrices de confusion**, de la **précision (`Accuracy`)**, du **score F1** et du **temps d'inférence par image**.
- Exporter les figures en haute résolution (`resultats_cnn_emotion.png`, `resultats_vit_emotion.png`, `comparaison_cnn_vs_vit.png`) et tester l'inférence faciale en direct via webcam (`CascadeClassifier`).

### Méthode utilisée
- **CNN (`CNN_Modi_et_al.ipynb`) :** prétraitement des images en `48×48`, entraînement avec `AdamW` et `CrossEntropyLoss`, sauvegarde des meilleurs poids dans `best_cnn_emotion.pth` et export des visualisations dans `resultats_cnn_emotion.png`.
- **Vision Transformer avec pooling attentionnel (`vision_transformer_et_comparaison.ipynb`) :** extraction des représentations via *Vision Transformer* / *self-attention* suivie d'un module de **pooling attentionnel** pondérant les *tokens* / caractéristiques avant la couche de classification, entraînement / *fine-tuning* sur **Google Colab**, sauvegarde dans `best_vit_emotion.pth`, export de `resultats_vit_emotion.png` et comparaison directe (`comparaison_cnn_vs_vit.png`).
- **Inférence webcam :** détection des visages avec `cv2.CascadeClassifier` (`opencv-python<5`) couplée au modèle chargé pour prédire l'émotion en temps réel.
- **Fichiers présents dans le dépôt :** `CNN_Modi_et_al.ipynb`, `vision_transformer_et_comparaison.ipynb`, `best_cnn_emotion.pth`, `best_vit_emotion.pth`, `resultats_cnn_emotion.png`, `resultats_vit_emotion.png`, `comparaison_cnn_vs_vit.png`, `dataset`, `'transformers(Xue_et_al).pdf'`.

---

## Les étapes d’installation et d’exécution du projet

### Bibliothèques utiles
- `torch` et `torchvision` : entraînement, inférence, modules d'attention / pooling et gestion des jeux de données PyTorch.
- `timm` : chargement des architectures *Vision Transformer* de base.
- `scikit-learn` : calcul des métriques (`Accuracy`, `F1-score`, matrice de confusion, rapport de classification).
- `matplotlib` et `numpy` : calcul numérique et génération des graphiques (`300 dpi`).
- `jupyterlab` : exécution interactive des notebooks `.ipynb` en local.
- `opencv-python<5` : détection faciale via `cv2.CascadeClassifier` et gestion du flux webcam (la version inférieure à 5 est recommandée pour conserver la compatibilité attendue avec `CascadeClassifier`).
### Installation
```bash
git clone https://github.com/Hidekela/FacEmoRec.git
cd FacEmoRec

# Créer et activer l'environnement virtuel Python
python3 -m venv .venv
source .venv/bin/activate

# Installer les bibliothèques nécessaires
pip install torch torchvision timm scikit-learn matplotlib numpy jupyterlab "opencv-python<5"
```
Vérification rapide
Graphiques des résultats
Résultats du CNN (48×48)
Résultats du Vision Transformer avec pooling attentionnel
Comparaison directe CNN vs Vision Transformer
Brève analyse

    Apport du Vision Transformer et du pooling attentionnel : l'auto-attention permet de capturer des relations globales entre différentes zones du visage, et le pooling attentionnel pondère dynamiquement les représentations les plus informatives pour la classification des émotions, là où le CNN (48×48) s'appuie sur des filtres convolutifs locaux.

    Compromis performance / temps de calcul : l'architecture CNN reste plus légère et rapide par image, tandis que le Vision Transformer avec pooling attentionnel apporte une modélisation plus riche au prix d'un coût de calcul supérieur.

    Erreurs typiques sur FER-2013 : les matrices de confusion (resultats_cnn_emotion.png et resultats_vit_emotion.png) reflètent les confusions classiques du jeu de données entre expressions visuellement proches (comme peur / surprise ou tristesse / neutre).

Les instructions nécessaires pour reproduire les résultats

    Vérifier le jeu de données : assurez-vous que le dossier dataset/ est présent et organisé selon les chemins attendus par les notebooks (train / test ou validation).

    Reproduire les résultats du CNN :

        Ouvrez CNN_Modi_et_al.ipynb.

        Exécutez les cellules d'entraînement et d'évaluation afin de générer best_cnn_emotion.pth et resultats_cnn_emotion.png.

    Reproduire les résultats du Vision Transformer avec pooling attentionnel et la comparaison :

        Ouvrez vision_transformer_et_comparaison.ipynb.

        Exécutez les cellules d'entraînement et d'évaluation du Vision Transformer équipé du pooling attentionnel pour produire best_vit_emotion.pth et resultats_vit_emotion.png.

        Exécutez la cellule de comparaison finale pour afficher les métriques (précision, score F1, temps d'inférence) et sauvegarder comparaison_cnn_vs_vit.png sans réentraîner le CNN.

    Reproduire le test sur webcam :

        Lancez la cellule dédiée à la webcam en vérifiant que le modèle choisi, sa résolution d'entrée et le fichier .pth correspondant sont cohérents avec cv2.CascadeClassifier.Bash

python -c "import cv2, torch, timm; print('OK:', cv2.__version__, torch.__version__, timm.__version__)"

Bash

jupyter lab

Sélectionnez le noyau Python associé à .venv (Kernel > Restart Kernel si besoin), puis ouvrez CNN_Modi_et_al.ipynb ou vision_transformer_et_comparaison.ipynb et exécutez les cellules.
```
## Les résultats obtenus, accompagnés d’une brève analyse
| Modèle | Notebook | Résolution | meilleur precision | Fichiers associés |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CNN** | `CNN_Modi_et_al.ipynb` | 48×48 | `[62.13%]`| `best_cnn_emotion.pth`, `resultats_cnn_emotion.png` |
| **ViT + Attention Pooling** | `vision_transformer_et_comparaison.ipynb` | `[128x128]` | `[66.35%]` | `best_vit_emotion.pth`, `resultats_vit_emotion.png` |
```  

## Brève analyse

  Apport du Vision Transformer et du pooling attentionnel : l'auto-attention permet de capturer des relations globales entre différentes zones du visage, et le pooling attentionnel pondère dynamiquement les représentations les plus informatives pour la classification des émotions, là où le CNN (48×48) s'appuie sur des filtres convolutifs locaux.

  Compromis performance / temps de calcul : l'architecture CNN reste plus légère et rapide par image, tandis que le Vision Transformer avec pooling attentionnel apporte une modélisation plus riche au prix d'un coût de calcul supérieur.

   Erreurs typiques sur FER-2013 : les matrices de confusion (resultats_cnn_emotion.png et resultats_vit_emotion.png) reflètent les confusions classiques du jeu de données entre expressions visuellement proches (comme peur / surprise ou tristesse / neutre).

## Les instructions nécessaires pour reproduire les résultats

   Vérifier le jeu de données : assurez-vous que le dossier dataset/ est présent et organisé selon les chemins attendus par les notebooks (train / test ou validation).

  Reproduire les résultats du CNN :

        Ouvrez CNN_Modi_et_al.ipynb.

        Exécutez les cellules d'entraînement et d'évaluation afin de générer best_cnn_emotion.pth et resultats_cnn_emotion.png.

  Reproduire les résultats du Vision Transformer avec pooling attentionnel et la comparaison :

        Ouvrez vision_transformer_et_comparaison.ipynb.

        Exécutez les cellules d'entraînement et d'évaluation du Vision Transformer équipé du pooling attentionnel pour produire best_vit_emotion.pth et resultats_vit_emotion.png.

        Exécutez la cellule de comparaison finale pour afficher les métriques (précision, score F1, temps d'inférence) et sauvegarder comparaison_cnn_vs_vit.png sans réentraîner le CNN.

  ### Reproduire le test sur webcam :

        Lancez la cellule dédiée à la webcam en vérifiant le modèle choisi, sa résolution d'entrée et le fichier .pth correspondant sont cohérents avec cv2.CascadeClassifier.
