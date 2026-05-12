import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score,recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

#import file depression
df = pd.read_csv(r"C:\Users\capuc\Desktop\Inge_2\Machine_Learning\Projet\projet_ML_depression_Capu\Data\final_depression_dataset_1.csv")

#Exploration initiale :
#type de chaque colonne
print(df.dtypes)          

#valeurs manquantes (nbr nan)
print(df.isnull().sum())   

#TRAITEMENT 
#D'abord les colonnes sans valeurs manquantes 

#drop deux colonnes inutiles : Name et City
#features inutiles
df= df.drop(['Name', 'City'], axis=1)

#On explore les données de toutes les colonnes
for col in df.columns:
    print(df[col].value_counts())

# Il a une faute de frappe dans la profession 
#quand on explore le dataset on remarque qu'il y a une coquille d'écriture qui crée deux categories au lioeiu d'une 
#Finanancial Analyst        38
#Financial Analyst          36
#on met les deux cats ensembles 
df['Profession'] = df['Profession'].replace('Finanancial Analyst', 'Financial Analyst')

print(df['Profession'].value_counts())
#Financial Analyst          74

#soucis à résoudre pour plus tard (amélioration)
#on regroupe les professions en secteurs
#car sinon trop de différents métiers
#regroupement par "grp"

profession = {
    'Teacher': 'Education', 'Educational Consultant': 'Education',
    'Researcher': 'Education', 'Research Analyst': 'Education',

    'Doctor': 'Sante', 'Pharmacist': 'Sante', 'Chemist': 'Sante',

    'Software Engineer': 'Tech', 'Data Scientist': 'Tech', 'UX/UI Designer': 'Tech',
    'Architect': 'Tech', 'Civil Engineer': 'Tech', 'Mechanical Engineer': 'Tech', 'Electrician': 'Tech',

    'HR Manager': 'Business', 'Business Analyst': 'Business', 'Manager': 'Business',
    'Consultant': 'Business', 'Entrepreneur': 'Business', 'Accountant': 'Business',
    'Investment Banker': 'Business', 'Financial Analyst': 'Business',

    'Lawyer': 'Droit', 'Judge': 'Droit',

    'Sales Executive': 'Service', 'Marketing Manager': 'Service', 'Digital Marketer': 'Service',
    'Travel Consultant': 'Service', 'Customer Support': 'Service', 'Content Writer': 'Service',
    'Graphic Designer': 'Service', 'Chef': 'Service', 'Plumber': 'Service', 'Pilot': 'Service',
}
df['Profession'] = df['Profession'].replace(profession)

print(df['Profession'].value_counts())

"""Profession
Service      464
Business     434
Education    426
Tech         269
Sante        200
Droit         90"""

#meme chose pour degree d'educ:
#recherche sur inter pour bien regrouper 

degree= {
    'Class 12': 'Bac',

    'B.Com': 'Licence', 'B.Ed': 'Licence', 'BCA': 'Licence', 'BBA': 'Licence',
    'BHM': 'Licence', 'BA': 'Licence', 'B.Arch': 'Licence', 'B.Pharm': 'Licence',
    'BSc': 'Licence', 'BE': 'Licence', 'LLB': 'Licence', 'B.Tech': 'Licence',

    'MBA': 'Master', 'MSc': 'Master', 'M.Tech': 'Master', 'M.Pharm': 'Master',
    'ME': 'Master', 'LLM': 'Master', 'MHM': 'Master', 'M.Ed': 'Master',
    'MA': 'Master', 'MBBS': 'Master', 'M.Com': 'Master', 'MCA': 'Master',

    'PhD': 'Doctorat', 'MD': 'Doctorat',
}
df['Degree'] = df['Degree'].replace(degree)

print(df['Degree'].value_counts())

"""Degree
Licence     1109
Master      1017
Bac          275
Doctorat     155"""

#on a 4 sortie possible on va faire un replace, pas besoin de retirer toout la caractères spéciaux et str
#replace par des valeurs approximatives et moyennes
#pas de valeurs manquantes donc rien d'autre a faire 
sleep = {
    "Less than 5 hours": 4.5,
    "5-6 hours": 5.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 8.5
}
df['Sleep Duration'] = df['Sleep Duration'].replace(sleep)

print(df['Sleep Duration'].value_counts())
#vérifier si on est en int/floats
print(df['Sleep Duration'].dtype)

#on obtient une colonne en type : object, on veut mettre en float
df['Sleep Duration'] = df['Sleep Duration'].astype(float)
#float64

#on encode les variables catégorielles en variables numériques pour pouvoir les utiliser dans les modèles
#pas de valeurs manquantes

#colonne "Depression" : target (yes no) 
le = LabelEncoder()
cols_binaires = ['Gender', 'Depression', 'Have you ever had suicidal thoughts ?',
                 'Family History of Mental Illness', 'Working Professional or Student',
                 'Degree', 'Profession', 'Dietary Habits']
for col in cols_binaires:
    df[col] = le.fit_transform(df[col])

print(df.dtypes)

"""
Gender                                     int64
Age                                        int64
Working Professional or Student            int64
Profession                                 int64
Academic Pressure                        float64
Work Pressure                            float64
CGPA                                     float64
Study Satisfaction                       float64
Job Satisfaction                         float64
Sleep Duration                           float64
Dietary Habits                             int64
Degree                                     int64
Have you ever had suicidal thoughts ?      int64
Work/Study Hours                           int64
Financial Stress                           int64
Family History of Mental Illness           int64
Depression                                 int64"""

#Mtn on a que des valeurs en int ou float

#on s'attaque aux valeurs manquantes: Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction 
"""Profession                                673
Academic Pressure                        2054
Work Pressure                             502
CGPA                                     2054
Study Satisfaction                       2054
Job Satisfaction                          502"""

#seul truc c'est que ya des emplyes et des eleves donc normal qu'il y est pleins de valeurs manquantes pour ces colonnes la 

#on remarque qu'il y a deux groupes : Employés et Student
# verifie si ya des veleurs manquantes dans les colonnes pour les students
# on cherche pour les valeurs 0 dans 'Working Professional or student' si ya des Nan
#dans les colonnes academic pressure, capa, study satisfaction
print("Students:")
print(df[df['Working Professional or Student'] == 0][[
    'Academic Pressure', 'CGPA', 'Study Satisfaction']].isnull().sum())

#pareil pours les emplyés (1) avec les colonnes work pressure et job satisfaction
print("Employees:")
print(df[df['Working Professional or Student'] == 1][[
    'Work Pressure', 'Job Satisfaction', 'Profession']].isnull().sum())

#yen a pas 
#donc on remarque que les deux grp respectifs on répondu à toutes les questions spécifiques qui leur on été posé

#On separe la dataset en deux 
#Pour les étudiants, on drop les features liées au travail et pour les employés, on drop les features liées aux études
#Car remplacer les valeurs manquantes par la moyenne ou la medianne n'aurait pas de sens dans ce cas
df_employee = df[df['Working Professional or Student'] == 1]
df_employee = df_employee.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction', 'Working Professional or Student'])

print(df_employee.dtypes)
print(df_employee.isnull().sum())

#rien qui manque, on peut commencer l'étude!

#On va maintenant tester des modèles sur ces deux datatsets
#Modèle de classification car target "OUI/NON"
#Le Voting classifier nous permet de tester plusieur moèdle en plus de lui meme
    #Graphoique moustache
print("Moyenne :\n", df_employee.mean())
print("Ecart type :\n", df_employee.std())
df_employee.boxplot()
plt.show()

X = df_employee.drop('Depression', axis=1)
y = df_employee['Depression']

#pas de raison particuliere pour le 80%, c'est ce qui utilise usuellement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Centre les données autour de 0 avec écart-type 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2700)
rf= RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

classifiers = [
    ('Logistic Regression', lr),
    ('Random Forest', rf),
    ('Decision Tree', dt),
    ('K-Nearest Neighbors', knn),
    ('AdaBoost', ada),
    ('Gradient Boosting', gb)
]

plt.figure()
for name, model in classifiers:
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
        
    print(name)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(f'Recall: {recall_score(y_test, y_pred):.2f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.2f}')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

# Voting Classifier
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred2  = vc.predict(X_test)

print('Voting Classifier')
print(f'Accuracy: {accuracy_score(y_test, y_pred2):.2f}')
print(f'Recall: {recall_score(y_test, y_pred2):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred2):.2f}\n')

#Plot de la coubre ROC de tous les modèles compatibles
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('Faux Positifs (FPR)')
plt.ylabel('Vrais Positifs Recall (TPR)')
plt.title(f'Courbe ROC Comparaison des modèles : employees')
plt.legend()
plt.show()


#meilleur modèle avec apprentissage supervisé 
#logiqtique regression!!

#On test maintenat modèle non-supervisé : KMeans:
    
X = df_employee.drop('Depression', axis=1)

#ramène toutes les features entre 0 et 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#Courbe d'inertie :méthode du coude
clustering_score = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    clustering_score.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), clustering_score)
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.xticks(range(1, 20))
plt.title('Méthode du coude : employees')
plt.show()

#KMeans
#n_cluster=2 car deprression OUI/NON (malgrès resultats courbe d'inertie)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
pred = kmeans.predict(X_scaled)

print(pred)
    
df_result = df_employee.copy()
df_result['Cluster'] = pd.DataFrame(pred, columns=['cluster'])

labels = kmeans.labels_
print(f'Silhouette Score : {silhouette_score(X_scaled, labels, metric="euclidean"):.2f}')

#On compare les resulatts avec els vraies etiquettes :
ct = pd.crosstab(df_result['Cluster'], df['Depression'])
print(ct) 

#Le silhouette score est faible (0.14 proche de 0) donc les clusters se chevauchent
# La courbe ne présente pas de coude net à k=2
#Les données ne forme pas deux cluster net pour depression Oui/Non 
#Grace au crosstab: deux cluster ont la meme proportion de deprime/pas deprimé
#donc groupe ne sont pas fait selon depression : decoupage aléatoire

#Modèle non-supervisé pas exploitable, on reste sur logistic regression supervisé

#On cree donc un nouveau model de lr  pour y intégrer :
# - Matrice de correlatio,
# - Feature importance
# - GridSearchCV ( meilleurs param)
# - Confusion matrice
# - Courbe ROC  

#matrice de corrélation, arrondie a la 2eme decimale 
corr_matrix = df_employee.corr().round(2)
plt.figure(figsize=(12,10)) 
sns.heatmap(data=corr_matrix, annot=True)
plt.show()
    
#On drop les feature avec une cofrrelation inferieure à 0.05
corr_with_label = corr_matrix['Depression']
selected_features = corr_with_label[abs(corr_with_label) > 0.05].drop('Depression')
print(f'Features supprimés : {corr_with_label[abs(corr_with_label) <= 0.05]}')
print(f'Features gardés : {corr_with_label[abs(corr_with_label) > 0.05]}')

X = df_employee[selected_features.index]
y = df_employee['Depression']

#pas de raison particuliere pour le 80%, c'est ce qui utiliser usuellement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Feature importance à l'aide d'un random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
    
featureimpor = pd.DataFrame(rf.feature_importances_, index=X_train.columns,
                            columns=["importance"]).sort_values("importance", ascending=False)
plt.barh(featureimpor.index, featureimpor["importance"])
plt.title('Feature Importance (Random Forest) : employees')
plt.show()
    
#pipeline
pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000))
])

#GridSearchCV pour récuperer les meilleures paramètres
#C : petit=réduit coeff /grand=faible minimisation coeff
#L1 : met certains coeff à 0 / L2 : reduit ts les coeff sans lkes annuler
#libinerar/saga : supporte l1 et l2
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga']
}

# scoring='recall' car contexte santé : mieux vaut un faux positif (déprimé detecté mais ne l'est pas) 
# qu'un faux négatif (déprimé non détecté)
grid = GridSearchCV(pipe, param_grid, scoring='recall', cv=5)
grid.fit(X_train, y_train)

print(f'Meilleurs paramètres : {grid.best_params_}')
print(f'Meilleur Recall : {grid.best_score_:.2f}')

best_pipeline = grid.best_estimator_

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

#Confusion matrice 
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.title('Matrice de confusion : employees')
plt.show()

#Courbe ROC 
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel("Proportion mal classée")
plt.ylabel("Proportion bien classée")
plt.title('Courbe ROC : employees')
plt.legend(loc="lower right")
plt.show()

print(classification_report(y_test, y_pred))

#joblib.dump(best_pipeline, f'lr_employees.joblib')