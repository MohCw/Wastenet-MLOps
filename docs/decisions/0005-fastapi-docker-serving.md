# ADR-0005 — FastAPI + Docker sur Railway

## Contexte

Le modèle doit être exposé via une API HTTP simple, déployable sur une plateforme cloud à
faible coût, et facile à conteneuriser. Le modèle évolue (nouveau `@champion`) indépendamment
du code de service.

## Décision

- **FastAPI + Uvicorn** pour l'API d'inférence (`/predict`, `/health`), avec documentation
  Swagger générée automatiquement.
- **Docker** (base `python:3.10-slim`, PyTorch **CPU**) pour la portabilité.
- Déploiement sur **Railway**, avec le modèle **tiré du registre MLflow au runtime** plutôt
  qu'embarqué dans l'image.

## Conséquences

- **+** Image légère : ni `model.pth`, ni données, ni `mlflow.db` dans l'image (exclus par
  `.dockerignore`). Le conteneur ne copie que `garbage_classification/`, `api/` et
  `monitoring/static/`.
- **+** Mise à jour du modèle sans rebuild : il suffit de promouvoir un nouveau `@champion`.
- **+** Compatibilité Railway : `CMD` en *shell form* pour résoudre `${PORT}` au runtime ;
  volume monté sur `/app/logs` pour persister `predictions.jsonl`.
- **−** Démarrage plus lent (téléchargement du modèle depuis DagsHub au `lifespan`) et
  dépendance réseau au démarrage.
- **−** Inférence CPU (latence plus élevée qu'en GPU) — acceptable pour une démo.

## Références

- [Déploiement](../deployment.md) (diagramme, variables d'environnement, ports).
- Chargement du modèle → [ADR-0003](0003-mlflow-tracking-registry.md).
