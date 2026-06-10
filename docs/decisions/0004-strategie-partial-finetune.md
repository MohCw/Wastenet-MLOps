# ADR-0004 — Stratégie `partial_finetune`

## Contexte

Avec un backbone pré-entraîné fort et un **petit dataset**, la profondeur de fine-tuning est un
arbitrage clé : geler trop (linear probe) sous-exploite le modèle ; tout dégeler (full
finetune) risque de **distordre les features pré-entraînées** et de sur-apprendre.

Trois stratégies sont implémentées (découpage des paramètres par **position**, sans nom de
couche codé en dur, donc valable pour n'importe quel modèle timm) :

| Stratégie | Comportement | Learning rates |
|---|---|---|
| `linear_probe` | gèle tout le backbone, entraîne la tête | tête : `lr` ; backbone : gelé |
| `partial_finetune` | gèle les premiers 75 % du backbone | tête : `lr` ; queue backbone : `0,1 × lr` |
| `full_finetune` | entraîne toutes les couches | 3 groupes : `0,01 × lr`, `0,1 × lr`, `lr` |

## Décision

Adopter **`partial_finetune`** avec scheduler **cosine + warmup** comme stratégie du champion.

Sur ConvNeXt, partial et full étaient **statistiquement indiscernables** (écart < 1 pp,
signaux contradictoires : partial meilleur en `best_val_acc`, full légèrement meilleur en
`test_f1_macro` / `f1_trash`). La décision a été tranchée par des critères pré-engagés :

1. **Métrique de sélection** fixée *a priori* = `best_val_acc` → favorise partial.
2. **Théorie petit-data** : le full fine-tuning distord les features pré-entraînées et
   sur-apprend.
3. **Coût** : partial converge plus vite (75 % du backbone gelé).

## Conséquences

- **+** Bon compromis précision/coût, convergence rapide (early stop à 17 époques).
- **+** Régularisation implicite (gel) bénéfique sur petit dataset.


## Références

- Kumar et al., *Fine-Tuning can Distort Pretrained Features…*, ICLR 2022 (arXiv:2202.10054).
- *Surgical Fine-Tuning*, ICLR 2023 (arXiv:2210.11466).
- Méthodologie de décision → [ADR-0008](0008-methodologie-tuning-incrementale.md).
