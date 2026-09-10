import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

df_students = pd.read_csv("data/students_clean.csv")
df_employee = pd.read_csv("data/employee_clean.csv")


def logistic_regression(df, label):
    X_full = df.drop('Depression', axis=1)
    y = df['Depression']

    X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)

    # matrice de corrélation calculée uniquement sur le train
    train_with_label = X_train_full.copy()
    train_with_label['Depression'] = y_train
    corr_matrix = train_with_label.corr().round(2)
    plt.figure(figsize=(12, 10))
    sns.heatmap(data=corr_matrix, annot=True)
    plt.title(f"Corrélations (train uniquement) : {label}")
    plt.show()

    # on garde les features corrélées à Depression (> 0.05), décidé sur le train
    corr_with_label = corr_matrix['Depression']
    selected_features = corr_with_label[abs(corr_with_label) > 0.05].drop('Depression')
    print(f'Features supprimés : {corr_with_label[abs(corr_with_label) <= 0.05]}')
    print(f'Features gardés : {corr_with_label[abs(corr_with_label) > 0.05]}')

    X_train = X_train_full[selected_features.index]
    X_test = X_test_full[selected_features.index]

    # feature importance avec un random forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    featureimpor = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=["importance"]).sort_values("importance", ascending=False)
    plt.barh(featureimpor.index, featureimpor["importance"])
    plt.title(f'Feature Importance (Random Forest) : {label}')
    plt.show()

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000))
    ])

    # Recherche des meilleurs hyperparamètres.
    param_grid = {
        'model__C': np.logspace(-2, 2, 20),
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }

    # scoring='recall' car en santé un faux positif vaut mieux qu'un faux négatif
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, scoring='recall', cv=cv)
    grid.fit(X_train, y_train)

    print(f'Meilleurs paramètres : {grid.best_params_}')
    print(f'Meilleur Recall : {grid.best_score_:.2f}')

    best_pipeline = grid.best_estimator_

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # Diagnostic overfitting : on compare l'AUC sur le train et sur le test
    train_auc = roc_auc_score(y_train, best_pipeline.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_proba)
    cv_auc_scores = [
        roc_auc_score(y_train.iloc[val_idx], best_pipeline.predict_proba(X_train.iloc[val_idx])[:, 1])
        for _, val_idx in cv.split(X_train, y_train)
    ]
    print(f"AUC train : {train_auc:.3f} | AUC test : {test_auc:.3f} | écart : {train_auc - test_auc:+.3f}")
    print(f"AUC en CV (5 folds, sur le train) : {np.mean(cv_auc_scores):.3f} ± {np.std(cv_auc_scores):.3f}")
    if train_auc - test_auc > 0.05:
        print("=> Écart train/test notable : signe classique de sur-apprentissage.")
    else:
        print("=> Écart train/test faible : l'AUC très élevé ne vient donc pas d'un sur-apprentissage "
              "classique (le modèle ne mémorise pas le train), mais plutôt du fait que ce jeu de "
              "données (sondage volontaire, pas de diagnostic clinique) sépare les deux classes de "
              "façon très nette. À interpréter avec prudence : voir la section Limites du README.")

    # courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline, X_train, y_train, cv=cv, scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 6), random_state=42)
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='AUC train')
    plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='AUC validation (CV)')
    plt.xlabel("Taille du jeu d'entraînement")
    plt.ylabel("AUC")
    plt.title(f"Courbe d'apprentissage : {label}")
    plt.legend(loc="lower right")
    plt.ylim(0, 1.05)
    plt.show()

    # matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title(f'Matrice de confusion : {label}')
    plt.show()

    # courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel("Proportion mal classée")
    plt.ylabel("Proportion bien classée")
    plt.title(f'Courbe ROC : {label}')
    plt.legend(loc="lower right")
    plt.show()

    print(classification_report(y_test, y_pred))

    # sauvegarde du modèle
    joblib.dump(best_pipeline, f'models/lr_{label}.joblib')
    print(f"Sauvegarde sous : models/lr_{label}.joblib")

    return {
        "label": label, "train_auc": train_auc, "test_auc": test_auc,
        "cv_auc_mean": np.mean(cv_auc_scores), "cv_auc_std": np.std(cv_auc_scores),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


results_students = logistic_regression(df_students, "students")
results_employee = logistic_regression(df_employee, "employee")