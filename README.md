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
git clone https://github.com/CapucineGG/Detection_precoce_depression.git
cd Detection_precoce_depression
python3 -m venv venv
source venv/bin/activate  # sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### Lancer l'application

Depuis le dossier du projet, avec le venv activé (`source venv/bin/activate`, à refaire à chaque nouvelle session de terminal) :

```bash
streamlit run main_app.py
```

Ça ouvre automatiquement un onglet dans le navigateur par défaut (sinon l'URL, en général `http://localhost:8501`, s'affiche directement dans le terminal). Pour arrêter l'application, `Ctrl+C` dans le terminal.

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

1. **Exploration et nettoyage** (`01_exploration_nettoyage.ipynb`) : traitement des valeurs manquantes, correction des catégories, **visualisations comparant les deux populations et leurs features** (taux de dépression, pression/satisfaction, habitudes, corrélations), séparation du dataset en deux populations (étudiants / employés), qui n'ont pas les mêmes variables pertinentes.
2. **Comparaison de modèles** (`02_comparaison_modeles.ipynb`) : test de 6 modèles supervisés (régression logistique, random forest, arbre de décision, k-NN, AdaBoost, gradient boosting) + un essai non supervisé (KMeans). La régression logistique s'est démarquée comme le modèle le plus performant.
3. **Modèle final** (`03_entrainement_modele_final.py`) : sélection des features les plus corrélées (calculée uniquement sur le jeu d'entraînement, pour ne pas biaiser l'évaluation), recherche des meilleurs hyperparamètres par GridSearchCV, diagnostic de sur-apprentissage (AUC train vs test, courbe d'apprentissage), évaluation (matrice de confusion, courbe ROC), sauvegarde du modèle.

## Résultats

| Population | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Étudiants | 0.93 | 0.93 | 0.93 | 0.93 |
| Employés | 0.99 | 0.99 | 0.99 | 0.99 |

*(moyennes pondérées du `classification_report` sur le jeu de test ; recall du modèle retenu par GridSearchCV : 0.94 pour les étudiants, 0.91 pour les employés)*

Le scoring a été optimisé sur le **recall** plutôt que l'accuracy : dans un contexte de santé, un faux positif (personne non déprimée signalée à tort) est préférable à un faux négatif (personne déprimée non détectée).

### À propos de l'AUC très élevé (~0.99)

Un tel score peut faire soupçonner du sur-apprentissage. Le script compare maintenant l'AUC sur le train, le test et en validation croisée : les trois sont très proches (écart < 0.01, voir la courbe d'apprentissage générée par `03_entrainement_modele_final.py`), et le score reste tout aussi élevé avec une régularisation forte ou avec un modèle différent (random forest peu profond). Ce n'est donc pas un sur-apprentissage classique (le modèle ne « mémorise » pas le train). L'explication la plus probable est que ce jeu de données (réponses à un sondage volontaire, pas un diagnostic clinique) sépare très nettement les deux classes à partir de quelques variables combinées (pression académique/professionnelle, pensées suicidaires, stress financier...). Les métriques ne doivent donc pas être lues comme une performance clinique réelle — voir aussi la section Limites ci-dessous.

## Données

Dataset : [Depression Survey/Dataset for Analysis](https://www.kaggle.com/datasets/sumansharmadataworld/depression-surveydataset-for-analysis) par Suman Sharma sur Kaggle, sous licence CC0 (domaine public).

Il s'agit de réponses à un sondage anonyme volontaire mené entre janvier et juin 2023 (facteurs liés au mode de vie et à la démographie), sans évaluation clinique.

Note : les colonnes `Name` et `City` du dataset original ont été retirées dès le nettoyage, n'étant pas pertinentes pour la prédiction.

## Limites et axes d'amélioration

- Les prédictions dépendent entièrement de la qualité et de la représentativité du dataset d'entraînement
- Le dataset est un sondage déclaratif (pas de diagnostic clinique), et sépare les deux classes de façon très nette sur quelques variables combinées : les métriques (AUC ~0.99) sont donc probablement optimistes par rapport à un cas réel, voir la section Résultats
- Le clustering non supervisé (KMeans) ne s'est pas avéré exploitable pour ce problème
- Piste future : tester d'autres modèles sur le pipeline final (pas seulement la régression logistique), ou du rééquilibrage de classes si le dataset est déséquilibré (le sous-groupe employés n'a que ~10% de cas positifs)

## Licence

Ce projet est sous licence MIT — voir le fichier [LICENSE](LICENSE).

## Auteur

Projet initialement réalisé en binôme, repris et retravaillé par **Capucine Gombert-Gillmann** — [GitHub](https://github.com/CapucineGG)