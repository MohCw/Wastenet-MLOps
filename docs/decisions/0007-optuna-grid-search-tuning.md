# ADR-0007 — Tuning des hyperparamètres avec Optuna (GridSampler)

## Contexte

Une fois l'architecture et la stratégie figées (`convnext_tiny` + `partial_finetune`), il reste
à régler les hyperparamètres « nuisance » `lr` et `scheduler`. L'espace de recherche est
**petit** (2 hyperparamètres) et le budget de calcul **limité** (laptop, RTX 3060 6 Go).

## Décision

Utiliser **Optuna** avec :

- **`GridSampler`** (et non TPE) : grille exhaustive de **9 points**
  `lr ∈ {1e-4, 3e-4, 1e-3} × scheduler ∈ {none, cosine, plateau}`.
  Le `lr` est exploré en **échelle log** (grille « 1-3-10 », demi-décade), ancré sur le défaut
  `1e-3` et sondé vers le bas — ce qui garde la tête dans sa bande Adam usuelle (1e-4–1e-3) et
  le backbone (`×0,1`) dans la bande de fine-tuning.
- **`MedianPruner(n_startup_trials=3, n_warmup_steps=5)`** : coupe tôt les combinaisons faibles
  (comparaison à la médiane des trials à la même époque ; warmup pour laisser cosine/plateau
  démarrer).
- **Logging MLflow manuel** par trial → courbes par époque + nom de run
  custom (`..._lr1e-4_cosine`) ; expérience séparée `tune-lr-scheduler`. Le tuning
  **n'enregistre pas** de modèle (sélection sur validation uniquement).

## Conséquences

- **+** Plus **interprétable** et moins coûteux qu'un TPE non convergé (le TPE vise 100–1000
  trials). 9 trials + pruning ≈ 45 min–1 h 30 de GPU.
- **+** Résultat reproductible et exhaustif sur la grille choisie.
- **Résultat** : meilleur `(lr, scheduler) = (0,001, cosine)` ; 5 trials sur 9 élagués ;
  `best_val_acc = 98,17 %` (`metrics/tune_best.json`).
- **−** Une grille manuelle ne couvre que les points prédéfinis ; un échantillonnage bayésien
  serait préférable sur un espace plus grand.

## Suite

Après tuning : reporter le meilleur couple dans `params.yaml`, puis **un seul `dvc repro`**
entraîne, enregistre/promeut le champion et évalue sur le test.

## Références

- Méthodologie globale → [ADR-0008](0008-methodologie-tuning-incrementale.md).
