# ADR-0003 — MLflow tracking + registry (`@champion`)

## Contexte

La sélection du meilleur modèle repose sur de **nombreuses expériences** (architectures,
stratégies, hyperparamètres). Il faut comparer ces runs de façon fiable, garder une trace du
code/paramètres de chacun, et désigner sans ambiguïté le modèle servi en production.

## Décision

Utiliser **MLflow** pour le suivi d'expériences **et** le registre de modèles, hébergé sur
**DagsHub** (`https://dagshub.com/MohCw/Wastenet-MLOps.mlflow`) :

- **Tracking** : chaque run journalise paramètres, métriques par époque, et l'artefact modèle.
  Expériences séparées `architecture-search` (comparaison de modèles) et `tune-lr-scheduler`
  (recherche d'hyperparamètres).
- **Registry** : le modèle est enregistré sous `garbage-classifier`. L'**alias `@champion`** est
  promu automatiquement par `train.py` quand la val_acc dépasse le champion courant.
- **Consommation** : l'API charge `models:/garbage-classifier@champion` au démarrage — le code
  de service ne référence jamais une version figée.

## Conséquences

- **+** Découplage total entre « quel modèle est le meilleur » (résolu par l'alias) et le code
  de l'API → mise à jour du modèle sans redéploiement de code.
- **+** Traçabilité : en committant `params.yaml` avant chaque run, le hash git est capturé par
  MLflow, reliant métriques ↔ code ↔ données.
- **+** Tableau de bord en ligne (DagsHub) pour comparer les runs.

## Références

- [Déploiement](../deployment.md) (chargement au runtime).
- Recherche d'hyperparamètres → [ADR-0007](0007-optuna-grid-search-tuning.md).
