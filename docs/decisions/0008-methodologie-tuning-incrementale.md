# ADR-0008 — Méthodologie d'expérimentation incrémentale

## Contexte

L'espace de recherche est large (architecture, profondeur de fine-tuning, learning rate,
scheduler) et le dataset est petit. Dans ce régime, l'écart de performance entre deux
configurations est souvent du même ordre que la variance d'un entraînement à l'autre. Tester des
configurations au hasard risque alors de conclure sur du bruit plutôt que sur un effet réel. Il
faut donc une méthode qui isole l'effet de chaque facteur.

## Décision

Adopter la **stratégie de tuning incrémentale** du
[Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) :

- Classer les hyperparamètres en **scientifiques** (étudiés, un seul par round), **fixes**, et
  **nuisance** (réglés mais pas étudiés).
- Chaque **round** a **un objectif** et fait varier **un seul** hyperparamètre scientifique, le
  reste constant.
- N'**adopter** un changement que sur **preuve solide** ; sinon, garder l'option la plus simple
  (rasoir d'Occam).

Application concrète (ordre des rounds) :

| Round | Question | Gagnant |
|---|---|---|
| 1 | Profondeur de fine-tuning (sur resnet18) ? | `partial_finetune` |
| 2 | Une meilleure architecture aide-t-elle ? | `convnext_tiny.in12k_ft_in1k` |
| 3 | La stratégie change-t-elle sur ConvNeXt (partial vs full) ? | `partial_finetune` |
| 4–5 | Réglage `lr × scheduler` | automatisé via [Optuna](0007-optuna-grid-search-tuning.md) |

HP **fixes** : split committé, `epochs=30`, `batch=32`, augmentation timm, `seed=42`.

## Conséquences

- **+** Chaque décision est **isolée et attribuable** (un seul facteur change à la fois).
- **+** Méthode **reconnue et traçable** : critères de sélection fixés *a priori*, chaque round
  documenté avec ses sources → conclusions reproductibles, pas *ad hoc*.
- **+** Évite le sur-engineering : on ne retient pas une complexité qui ne gagne pas clairement
  (cf. décision partial-vs-full, [ADR-0004](0004-strategie-partial-finetune.md)).
- **−** Plus lent qu'une recherche tous azimuts ; suppose une discipline de commit/log stricte.

## Références

- Google Research, *Deep Learning Tuning Playbook* (sections *incremental tuning strategy* /
  *designing the next round*).
- Décisions liées : [ADR-0001](0001-modele-convnext-timm.md),
  [ADR-0004](0004-strategie-partial-finetune.md), [ADR-0007](0007-optuna-grid-search-tuning.md).
