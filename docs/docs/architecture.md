# Architecture du projet WasteNet MLOps

## 1. Presentation du projet

WasteNet est un systeme de classification d'images de déchets en 6 classes :
**cardboard, glass, metal, paper, plastic, trash**.

Le projet suit une demarche MLOps complete : du versionnement des donnees au deploiement
d'une API de prediction, en passant par le suivi des experimentations et la detection de
derive (drift) en production.

**Stack technique** : Python 3.10, PyTorch + timm, DVC, MLflow, FastAPI, Docker,
Evidently AI, GitHub Actions.

```
Dataset : Garbage Classification (Kaggle) — 2 527 images, 6 classes
Modele  : ConvNeXt Tiny (pre-entraine ImageNet-21k -> ImageNet-1k)
Val acc : 96.6% | Test acc : 91.4% | Test F1 macro : 90.3%
```

---

## 2. Architecture globale

![Architecture globale](architecture-globale.png)

L'architecture s'organise en 4 couches communicantes :

| Couche | Role | Technologies |
|---|---|---|
| **Data** | Acquisition, nettoyage, versionnement et split des donnees | KaggleHub, DVC |
| **Training** | Entrainement, tracking des experimentations et registre de modeles | PyTorch, timm, MLflow |
| **Deployment** | Exposition du modele via API REST, conteneurisation | FastAPI, Docker |
| **Monitoring** | Detection de drift des predictions et des embeddings en production | Evidently AI |

Un pipeline CI/CD (GitHub Actions) automatise le linting, les tests et la construction
de l'image Docker a chaque push sur `main`.

---

## 3. Pipeline MLOps detaille

### 3.1 Data pipeline (DVC)

```
Kaggle Dataset
     |
     v
data/raw/  (2 527 images, 6 dossiers classe)
     |
     |-- prepare (deduplication intra-classe + inter-classe)
     |       |
     |       v
     |   data/interim/cleaned/  (2 521 images, doublons supprimes)
     |
     |-- split (repartition train/val/test via fichiers .txt fournis)
             |
             v
         data/processed/
              |-- train/  (1 764 images)
              |-- val/    (327 images)
              |-- test/   (430 images)
```

**Particularites** :
- Suppression des duplicatas en 2 passes : intra-classe (garder la 1ere occurrence) puis
  inter-classe (supprimer toutes les occurrences si label ambigü)
- Les splits sont predefinis par des fichiers `.txt` fournis avec le dataset
  (pas de split aleatoire)

### 3.2 Training pipeline

```
data/processed/train/ + val/
           |
           v
    ConvNeXt Tiny (timm)
    pretrained=True
           |
           v
    Strategy : linear_probe
    (backbone gele, tete seule entrainee)
           |
           v
    Optimizer : Adam (lr=0.001)
    Scheduler : ReduceLROnPlateau (patience=5, factor=0.3)
    Early stopping : patience=10
    Max epochs : 50
           |
           v
    MLflow Experiment Tracking
    -> logged: hyperparams, train/val loss + accuracy par epoch
    -> registered model alias "champion" si meilleur val_acc
           |
           v
    models/model.pth + model_metadata.json
```

**Strategies de fine-tuning testees** (3) :

| Strategie | Backbone | Tete | Nb runs |
|---|---|---|---|
| linear_probe | gele | entrainee | 4+ |
| partial_finetune | 75% gele, 25% entraine (lr*0.1) | entrainee (lr) | 3+ |
| full_finetune | 3 blocs lr*0.01 / lr*0.1 / lr | entrainee (lr) | 4+ |

### 3.3 Evaluation pipeline

```
models/model.pth + data/processed/test/
           |
           v
    Chargement du modele + metadata
    (architecture, nb classes, input size)
           |
           v
    Inference sur le jeu de test
           |
           v
    Metriques :
    - test_accuracy
    - test_f1_macro
    - f1 par classe
    - matrice de confusion
           |
           v
    MLflow : logue sous le meme run d'entrainement
    Fichiers : metrics/eval_metrics.json, metrics/confusion_matrix.csv
```

### 3.4 Pipeline d'inference (API)

```
Client HTTP
     |
     | POST /predict (image/jpeg)
     v
FastAPI (port 8000)
     |
     |-- 1. Chargement du modele @champion depuis MLflow registry
     |-- 2. Preprocessing (timm transforms)
     |-- 3. forward_features -> embedding (pooling)
     |       + forward_head -> logits -> softmax
     |-- 4. Extraction des proprietes visuelles (brightness, blur, RGB means)
     |-- 5. Log dans predictions.jsonl (timestamp + scores + embedding + image props)
     |
     v
Reponse JSON :
{
  "predicted_class": "plastic",
  "confidence": 0.93,
  "scores": { "cardboard": 0.01, "glass": 0.02, ... },
  "inference_ms": 14.2
}
```

**Endpoints** :

| Endpoint | Methode | Description |
|---|---|---|
| `/health` | GET | Health check, verifie que le modele est charge |
| `/predict` | POST | Classification d'une image uploadee |
| `/monitoring/report` | GET | Redirection vers le dashboard Evidently (port 8001) |

### 3.5 Pipeline de monitoring

```
predictions.jsonl (production)
     +
reference.parquet (training set, pre-calcule)
     |
     v
Evidently Report Engine
     |
     |-- Report 1 : Drift des features scalaires
     |   - predicted_class  -> chi-squared test
     |   - confidence       -> K-S test
     |   - brightness       -> K-S test
     |   - blur_score       -> K-S test
     |   - r_mean, g_mean, b_mean -> K-S test
     |
     |-- Report 2 : Drift des embeddings CNN
     |   - Domaine classifier (ROC-AUC, seuil 0.55)
     |   - Necessite >= 20 echantillons production
     |
     v
Evidently Workspace -> Dashboard UI (port 8001)
```

**Panels du dashboard** :
- Compteurs : share of drifted features, embedding ROC-AUC, missing values
- Series temporelles : share of drifted features, embedding ROC-AUC
- Detail par signal : predicted_class, confidence, brightness, blur, R/G/B means

---

## 4. Carte des ports

| Service | Port | URL |
|---|---|---|
| FastAPI (inference) | 8000 | `http://localhost:8000/docs` |
| MLflow UI | 5000 | `http://localhost:5000` |
| Evidently UI | 8001 | `http://localhost:8001` |

---

## 5. Justification des choix techniques

### 5.1 PyTorch + timm (vs TensorFlow, JAX)

**Role** : Framework de Deep Learning pour l'entrainement et l'inference du modele de
classification d'images.

**Alternatives** :
- TensorFlow / Keras
- JAX + Flax

**Pourquoi ce choix** :
- La bibliotheque `timm` (PyTorch Image Models) donne acces a plus de 1000 architectures
  pre-entrainees, dont ConvNeXt, ResNet, EfficientNet, ViT, etc.
- PyTorch est le framework dominant dans la recherche en vision par ordinateur
  (majorite des papiers publies en PyTorch)
- `timm` fournit les transformations de donnees et les configurations de preprocessing
  specifiques a chaque architecture (taille d'entree, normalisation)
- Ecosysteme mature pour le deploiement (TorchScript, ONNX, MLflow)

**Limites** : Pas de support natif pour TPU (contrairement a JAX). Pas pertinent ici car
l'entrainement se fait sur GPU local ou CPU.

---

### 5.2 ConvNeXt Tiny (vs ResNet18, EfficientNet, MobileNet)

**Role** : Architecture du modele de classification.

**Alternatives evaluees** :
- ResNet18 (modele par defaut du template)
- EfficientNet-B0
- MobileNetV3
- ConvNeXt Tiny

**Pourquoi ce choix** :
- Meilleur compromis precision / cout de calcul parmi les architectures testees
  (96.6% val acc, 91.4% test acc)
- Pre-entraine sur ImageNet-21k puis fine-tune sur ImageNet-1k, ce qui donne des
  caracteristiques visuelles tres generales et transferables
- Design moderne avec convolutions 7x7 + GELU + LayerNorm (inspire des transformers)
- Taille du modele : 111 Mo, raisonnable pour un deploiement CPU en inference
- Linear probe avec ConvNeXt surpasse les autres architectures meme en full finetune

**Limites** : Plus lent en inference qu'un MobileNet. Ne fait pas de difference sur un
petit dataset comme celui-ci (2527 images), mais la marge de progression est plus grande
si le dataset s'agrandit.

---

### 5.3 MLflow (vs W&B, Neptune)

**Role** : Tracking des experimentations et Registre de modeles.

**Pourquoi ce choix** :
- Open source et auto-hebergeable (fichier SQLite local ou serveur distant)
- Model Registry natif avec alias (`@champion`) et gestion des versions
- Integration native avec DVC (les deux outils partagent la meme philosophie
  legere et declarative)
- Possibilite d'utiliser DagsHub comme remote tracking (optionnel)
- API simple : `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.pytorch.log_model()`
- Pas de cout recurrent (contrairement a W&B/Neptune en equipe)

**Limites** : UI pas trop riche (pas de comparaison visuelle de runs, pas de rapports
automatiques). L'UI suffit pour un projet de cette taille.

---

### 5.4 DVC

**Role** : Pipeline de donnees et versionnement des jeux de donnees et des modeles.

**Pourquoi ce choix** :
- Versionne les donnees et les modeles sur Git (via des fichiers `.dvc` et un remote
  storage type S3/GDrive/DagsHub)
- Pas besoin d'infrastructure distribuee : le pipeline s'execute en local avec `dvc repro`
- Detection intelligente des changements : `dvc repro` ne rejoue que les stages dont les
  dependances ont change (code, donnees, parametres)
- Integre le tracking des metriques DVC dans l'UI MLflow si desire
- Courbe d'apprentissage faible par rapport a Airflow ou Kubeflow

**Limites** : Pas de parallelisation native des stages (execution sequentielle). Pas
d'orchestrateur distribue. Convient pour un projet de cette echelle mais ne passerait
pas a l'echelle d'une equipe de 20 data scientists.

---

### 5.5 FastAPI 

**Role** : Framework web pour l'API de prediction.

**Alternatives** :
- Flask
- Django REST Framework
- FastAPI

**Pourquoi ce choix** :
- Performance asynchrone native (uvicorn, bascule automatique sync/async)
- Documentation interactive automatique (Swagger UI, ReDoc) via l'endpoint `/docs`
- Validation des entrees avec Pydantic (type hints, validation automatique)
- Gestion native des `UploadFile` pour les images
- Support du lifespan pour le chargement du modele au demarrage
- API tres performante pour l'inference (temps de reponse < 20ms)

**Limites** : Moins d'ecosysteme que Flask (plugins, extensions). Pas bloquant pour une
API a 2 endpoints.

---

### 5.6 Evidently AI 

**Role** : Detection de derive (data drift, model drift) en production.

**Pourquoi ce choix** :
- Open source, auto-hebergeable (workspace local, pas de SaaS obligatoire)
- Metriques de drift completes : colonnes scalaires, embeddings (classifieur de domaine),
  valeurs manquantes
- Dashboard integre (`evidently ui`) sans besoin d'outil externe
- Integration simple : le script `run_drift.py` produit un rapport et l'ajoute au workspace
- Support des donnees tabulaires avec `ColumnMapping` pour les embeddings

**Limites** : Ne gere pas le drift des donnees d'entree brutes (images). Les metriques
visuelles (brightness, blur) sont un proxy, pas une detection de drift semantique.

---

### 5.7 Linear probe (vs full finetune)

**Role** : Strategie de fine-tuning pour le petit dataset.

**Pourquoi ce choix** :
- Le dataset est petit (1 764 images d'entrainement) : un full finetune risquerait le
  surapprentissage
- Linear probe = backbone gele + tete entrainee. Cela preserve les caracteristiques
  generales apprises sur ImageNet-21k
- Etonnamment efficace : 96.6% val acc en linear probe contre des scores comparables
  en full finetune mais avec plus de risque d'overfitting
- Entrainement plus rapide (moins de gradients a calculer) et moins de consommation GPU
- Les tests montrent que les 3 strategies donnent des resultats proches, ce qui indique
  que le backbone ConvNeXt est deja tres performant pour cette tache

**Limites** : Potentiellement sous-optimal si le domaine cible est tres different
d'ImageNet. La classification d'ordures restant proche de la photographie classique,
le transfert est efficace.

---

### 5.8 Docker (deploiement)

**Role** : Conteneurisation de l'API pour le deploiement.

**Pourquoi ce choix** :
- Image basee sur `python:3.10-slim` (leger : ~150 Mo sans PyTorch)
- Installation CPU de PyTorch (pas de CUDA dans le conteneur, l'inference CPU est
  suffisante pour un demarrage a froid)
- Build multi-stage possible (non implemente, le modele est charge depuis MLflow registry
  au demarrage, pas inclus dans l'image)
- Compatible Railway (variable d'environnement `$PORT` pour le port dynamique)
- Cache Docker via GitHub Actions (type `gha`) pour les builds CI


---

## 6. Resultats experimentaux

### 6.1 Metriques finales

| Metrique | Valeur |
|---|---|
| Best validation accuracy | 96.6% |
| Validation loss (best) | 0.1286 |
| Test accuracy | 91.4% |
| Test F1 macro | 90.3% |
| Epochs realisees | 27 / 50 |
| Taille du modele | 111 Mo |

### 6.2 Architecture retenue

```
Modele    : ConvNeXt Tiny (convnext_tiny.in12k_ft_in1k)
Strategie : linear_probe
Scheduler : ReduceLROnPlateau (patience=5, factor=0.3)
Batch size: 32
Learning rate: 0.001
```

### 6.3 Statistiques du dataset

| Split | Images |
|---|---|
| Train | 1 764 |
| Validation | 327 |
| Test | 430 |
| Total (apres dedup) | 2 521 |
| Total (brut) | 2 527 |
| Doublons intra-classe supprimes | ~4 |
| Doublons inter-classe supprimes | ~2 |

---