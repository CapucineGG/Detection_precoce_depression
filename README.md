# Application pour la détection précoce d’une dépression. 

La dépression est un trouble de santé mentale majeure qui affecte le fonctionnement cognitif, 
émotionnel et comportemental des individus. Elle peut être diagnostiquée lors d’un entretien 
clinique mené par un professionnel de santé, parfois complété par un questionnaire 
psychométrique. Certaines données comportementales sont mesurables et peuvent présenter des 
corrélations significatives avec un état dépressif. 

## Objectif   
L’objectif de ce projet est de mettre en place une application permettant de prédire précocement 
une dépression via des données comportementales. Cette prédiction est basée sur l’exploration, 
l’analyse, la modélisation et le nettoyage d’un gros volume de données open data, ensuite 
l’entraînement et l’évaluation de différents modèles de Machine Learning afin de choisir le plus 
performant. 

# Instructions

## Lancer l'app :

- Aller dans environnment conda ML (anaconda prompt) : conda activate ML

- Se placer dans le dossier projet_ml_depression: cd [chemin d'accès]

- Lancer : streamlit run main_app.py


## Etapes du code :

- Exploration du data set
- Traitement des données (données manquantes et séparation en deux cas employés et students)
- Modèles et métriques pour évaluer quel modèle est le plus interessant
- Pipeline du meilleur modèle + tri des feature inutiles/importante
- App Streamlit
