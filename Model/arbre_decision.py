from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_clean = pd.read_csv('Data/depression_dataset_clean.csv')
df_students = pd.read_csv('Data/depression_dataset_students.csv')
df_employee = pd.read_csv('Data/depression_dataset_employee.csv')


def arbre_decision(df, label):

    X = df.drop('Depression', axis=1)
    y = df['Depression']

    features = X.columns.tolist()

    parameters = {"min_samples_leaf": np.arange(1,4), "max_depth": np.arange(1,10)}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtc=DecisionTreeClassifier(random_state=42)
    cv=GridSearchCV(dtc, param_grid=parameters, cv=5)
    cv.fit(X_train, y_train)
    best_tree = cv.best_estimator_
    y_pred = best_tree.predict(X_test)

    print(f"\nBest parameters : {cv.best_params_}")
    print(f"Best score : {cv.best_score_}")

    print(f"Accuracy score : {accuracy_score(y_test, y_pred)}")
    report= classification_report(y_test, y_pred)
    print(f"Classification report : \n {report}")  

    # Visualiser l'arbre graphiquement
    plt.figure(figsize=(30, 10))
    plt.title(f'Arbre de Décision {label}')
    plot_tree(best_tree, feature_names=features,
            class_names=['Non dépressif', 'Dépressif'],
            filled=True, rounded=True)
    plt.show()

    # Importance des features
    importances = pd.Series(best_tree.feature_importances_, index=features)
    print(importances.sort_values(ascending=False))

arbre_decision(df_clean, "clean")
arbre_decision(df_students, "students")
arbre_decision(df_employee, "employee")

