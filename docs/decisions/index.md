# Décisions d'architecture (ADR)

Cette section justifie les **choix techniques majeurs** du projet. Chaque ADR
(*Architecture Decision Record*) suit un format court : **Contexte → Décision → Conséquences**.

| ADR | Décision | Composant concerné |
|---|---|---|
| [0001](0001-modele-convnext-timm.md) | Modèle ConvNeXt Tiny via timm | [Entraînement](../runtime.md) |
| [0002](0002-dvc-versioning-dagshub.md) | DVC + remote DagsHub | [Pipeline](../runtime.md) |
| [0003](0003-mlflow-tracking-registry.md) | MLflow tracking + registry `@champion` | [Entraînement](../runtime.md) |
| [0004](0004-strategie-partial-finetune.md) | Stratégie `partial_finetune` | [Entraînement](../runtime.md) |
| [0005](0005-fastapi-docker-serving.md) | FastAPI + Docker sur Railway | [Déploiement](../deployment.md) |
| [0006](0006-evidently-drift-monitoring.md) | Monitoring Evidently (HTML statique) | [Monitoring](../monitoring.md) |
| [0007](0007-optuna-grid-search-tuning.md) | Tuning Optuna GridSampler | [Entraînement](../runtime.md) |
| [0008](0008-methodologie-tuning-incrementale.md) | Méthodologie d'expérimentation incrémentale | Transverse |

!!! note "Statut"
    Toutes les ADR ci-dessous sont **acceptées** et reflètent l'état actuel du projet.
