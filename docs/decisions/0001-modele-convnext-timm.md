# ADR-0001 — Modèle ConvNeXt Tiny via timm

## Contexte

La tâche est une classification d'images sur un **petit dataset** (~2 527 images, 6 classes).
Il faut un backbone performant en transfert, sans sur-dimensionner par rapport aux données
disponibles. On veut aussi pouvoir comparer plusieurs architectures de façon équitable.

## Décision

Utiliser **`convnext_tiny.in12k_ft_in1k`** (pré-entraîné ImageNet-12k puis fine-tuné
ImageNet-1k), chargé via la bibliothèque **timm**.

Le choix résulte d'une comparaison sur **trois familles** d'architectures (plutôt que trois variantes d'un même modèle) :

| Modèle | Rôle | Params | IN-1k top-1 |
|---|---|---|---|
| `resnet18.tv_in1k` | baseline à battre | ~11,7 M | ~69,8 % |
| `convnext_tiny.in12k_ft_in1k` | **plafond de précision** (favori en transfert petit-data) | ~28,6 M | ~84,2 % |
| `mobilenetv4_conv_medium.e500_r224_in1k` | efficacité / edge | ~9,7 M | ~79,1 % |

Les **Vision Transformers** ont été écartés : trop gourmands en données pour ce volume.

timm est retenu car il expose des centaines de backbones pré-entraînés avec une API uniforme
(`create_model`, `pretrained_cfg`, `create_transform`), ce qui rend le code agnostique à
l'architecture et facilite la comparaison.

## Conséquences

- **+** Excellente précision en transfert (val_acc 98,17 %, test_acc 95,35 %).
- **+** Code d'entraînement et d'inférence indépendant de l'architecture (changer `model_arch`
  dans `params.yaml` suffit) ; les transforms d'inférence sont dérivées du `pretrained_cfg`.

## Références

- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), arXiv:2201.03545.
- *Battle of the Backbones*, NeurIPS 2023.
- Voir aussi [ADR-0004](0004-strategie-partial-finetune.md) (stratégie de fine-tuning) et
  [ADR-0008](0008-methodologie-tuning-incrementale.md) (méthodologie de comparaison).
