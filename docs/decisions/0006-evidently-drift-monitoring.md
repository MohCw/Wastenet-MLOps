# ADR-0006 — Monitoring de drift Evidently (HTML statique)

## Contexte

Un modèle en production peut se dégrader si les images entrantes s'éloignent de la distribution
d'entraînement (changement d'éclairage, de qualité, de mix de classes). Il faut **détecter ce
drift**. 

## Décision

Utiliser **Evidently AI** pour la détection de drift, avec deux niveaux :

1. **Drift scalaire** : test par colonne — χ² (classe prédite) et Kolmogorov–Smirnov
   (confiance, luminosité, netteté, moyennes RGB).
2. **Drift d'embeddings** : méthode du *domain classifier* (seuil ROC-AUC 0,55), sur les
   embeddings du backbone journalisés à l'inférence.

**Mode de service — Option A (statique)** : `run_drift.py` génère des **rapports HTML
autonomes** dans `monitoring/static/`, committés dans git et servis par FastAPI
(`StaticFiles`) sous `/monitoring`. Pas de serveur Evidently live en production.

## Conséquences

- **+** Simple et honnête pour une démo : un seul service, un seul port, pas d'infrastructure
  de monitoring supplémentaire.
- **+** Réutilise « gratuitement » l'embedding déjà calculé à l'inférence (une seule passe
  backbone).
- **−** Rapports = **instantanés** : rafraîchir nécessite de relancer `run_drift.py` en local,
  committer et redéployer (pas de temps réel).
- **−** Le drift d'embeddings n'est calculé qu'à partir de **≥ 20** prédictions.
- En **dev local**, le serveur interactif reste disponible (`evidently ui` sur le port 8001).

## Références

- [Monitoring](../monitoring.md) (signaux suivis, URLs, rafraîchissement).
