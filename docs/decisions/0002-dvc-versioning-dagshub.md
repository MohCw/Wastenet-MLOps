# ADR-0002 — DVC + remote DagsHub

## Contexte

Les données (images) et les modèles (`.pth`) sont des artefacts **volumineux et binaires**,
inadaptés au versioning git direct. Or la reproductibilité exige de relier précisément un
modèle à la version exacte des données et du code qui l'ont produit.

## Décision

Utiliser **DVC** pour :

1. **Versionner** données et modèles hors de git (git ne suit que des fichiers `.dvc` / `dvc.lock`
   légers, pointant vers le contenu réel).
2. **Orchestrer le pipeline** (`dvc.yaml`) : `prepare → split → train → evaluate`, avec
   ré-exécution sélective des seules étapes dont les dépendances ont changé.
3. **Stocker** le cache sur un **remote DagsHub partagé** : les artefacts sont récupérables via `dvc pull` depuis n'importe quelle machine.

## Conséquences

- **+** `dvc repro` reconstruit tout le pipeline de façon déterministe ; `dvc.lock` fige les
  hash des sorties → un modèle est exactement reproductible/récupérable.
- **+** Le dépôt git reste léger ; les gros artefacts vivent sur DagsHub.
- **+** Bonne séparation des responsabilités : git pour le code, DVC pour les données/modèles.
- **−** Étape d'apprentissage supplémentaire et discipline requise : `dvc.lock` **doit** être
  committé avec les métriques, sinon le modèle devient irrécupérable.

## Références

- Workflow détaillé : [Vue d'exécution](../runtime.md).
- DagsHub héberge aussi le suivi MLflow → [ADR-0003](0003-mlflow-tracking-registry.md).
