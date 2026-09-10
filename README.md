# Détection précoce de dépression

Application de Machine Learning qui prédit le risque de dépression à partir de données comportementales (sommeil, pression académique/professionnelle, habitudes de vie, etc.), avec une interface Streamlit pour tester le modèle.

## Contexte

La dépression est un trouble de santé mentale majeur qui affecte le fonctionnement cognitif, émotionnel et comportemental des individus. Elle peut être diagnostiquée lors d'un entretien clinique mené par un professionnel de santé, parfois complété par un questionnaire psychométrique. Certaines données comportementales sont mesurables et peuvent présenter des corrélations significatives avec un état dépressif.

## Objectif

Explorer, nettoyer et modéliser un jeu de données comportementales pour prédire précocement un risque de dépression, en comparant plusieurs modèles de Machine Learning afin de retenir le plus performant, puis l'exposer via une application Streamlit.

## Prérequis

- [Python](https://www.python.org/) ≥ 3.10

## Installation

```bash
git clone https://github.com/CapucineGG/early-depression-detection.git
cd early-depression-detection
python3 -m venv venv
source venv/bin/activate  # sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### Lancer l'application

```bash
streamlit run main_app.py
```

### Régénérer les modèles depuis zéro (optionnel)

Les modèles entraînés sont déjà fournis dans `models/`, l'app fonctionne directement après l'installation. Si tu veux relancer le pipeline complet :

```bash
# 1. Ouvrir et exécuter 01_exploration_nettoyage.ipynb (Run All)
# 2. Ouvrir et exécuter 02_comparaison_modeles.ipynb (Run All)
python3 03_entrainement_modele_final.py
```

## Structure du projet

```
early-depression-detection/
├── data/
│   └── depression_dataset.csv           # dataset source
├── models/
│   ├── lr_students.joblib               # modèle final : étudiants
│   └── lr_employee.joblib               # modèle final : employés
├── 01_exploration_nettoyage.ipynb       # exploration + nettoyage des données
├── 02_comparaison_modeles.ipynb         # comparaison de plusieurs modèles
├── 03_entrainement_modele_final.py      # entraînement du modèle retenu
├── main_app.py                          # application Streamlit
├── requirements.txt
└── README.md
```

## Démarche

1. **Exploration et nettoyage** (`01_exploration_nettoyage.ipynb`) : traitement des valeurs manquantes, correction des catégories, séparation du dataset en deux populations (étudiants / employés), qui n'ont pas les mêmes variables pertinentes.
2. **Comparaison de modèles** (`02_comparaison_modeles.ipynb`) : test de 6 modèles supervisés (régression logistique, random forest, arbre de décision, k-NN, AdaBoost, gradient boosting) + un essai non supervisé (KMeans). La régression logistique s'est démarquée comme le modèle le plus performant.
3. **Modèle final** (`03_entrainement_modele_final.py`) : sélection des features les plus corrélées, recherche des meilleurs hyperparamètres par GridSearchCV, évaluation (matrice de confusion, courbe ROC), sauvegarde du modèle.

## Résultats

<!-- Remplace ce tableau par les vraies métriques affichées à la fin de l'exécution de 03_entrainement_modele_final.py (classification_report) -->

| Population | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Étudiants | à compléter | à compléter | à compléter | à compléter |
| Employés | à compléter | à compléter | à compléter | à compléter |

Le scoring a été optimisé sur le **recall** plutôt que l'accuracy : dans un contexte de santé, un faux positif (personne non déprimée signalée à tort) est préférable à un faux négatif (personne déprimée non détectée).

## Données

Dataset : [Depression Survey/Dataset for Analysis](https://www.kaggle.com/datasets/sumansharmadataworld/depression-surveydataset-for-analysis) par Suman Sharma sur Kaggle, sous licence CC0 (domaine public).

Il s'agit de réponses à un sondage anonyme volontaire mené entre janvier et juin 2023 (facteurs liés au mode de vie et à la démographie), sans évaluation clinique.

Note : les colonnes `Name` et `City` du dataset original ont été retirées dès le nettoyage, n'étant pas pertinentes pour la prédiction.

## Limites et axes d'amélioration

- Les prédictions dépendent entièrement de la qualité et de la représentativité du dataset d'entraînement
- Le clustering non supervisé (KMeans) ne s'est pas avéré exploitable pour ce problème
- Piste future : tester d'autres modèles sur le pipeline final (pas seulement la régression logistique), ou du rééquilibrage de classes si le dataset est déséquilibré

## Licence

Ce projet est sous licence MIT — voir le fichier [LICENSE](LICENSE).

## Auteur

Projet initialement réalisé en binôme, repris et retravaillé par **Capucine Gombert-Gillmann** — [GitHub](https://github.com/CapucineGG)