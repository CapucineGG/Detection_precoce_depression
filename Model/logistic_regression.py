import joblib

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

df_students = pd.read_csv('Data/depression_dataset_students.csv')
df_employee = pd.read_csv('Data/depression_dataset_employee.csv')

def logistic_regression(df, label):  

    #matrice de corrélation, arrondie a la 2eme decimale 
    corr_matrix = df.corr().round(2)
    plt.figure(figsize=(12,10)) 
    sns.heatmap(data=corr_matrix, annot=True)
    plt.show()

    corr_with_label = corr_matrix['Depression']
    selected_features = corr_with_label[abs(corr_with_label) > 0.05].drop('Depression')

    X = df[selected_features.index]
    y = df['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    featureimpor = pd.DataFrame(rf.feature_importances_, index=X_train.columns,
                                columns=["importance"]).sort_values("importance", ascending=False)
    plt.barh(featureimpor.index, featureimpor["importance"])
    plt.title(f'Feature Importance (Random Forest) : {label}')
    plt.show()

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000))
    ])

    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }

    grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    print(f'Meilleurs paramètres : {grid.best_params_}')
    print(f'Meilleur F1 (CV) : {grid.best_score_:.2f}')

    best_pipeline = grid.best_estimator_

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title(f'Matrice de confusion : {label}')
    plt.show()

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

    joblib.dump(best_pipeline, f'lr_{label}.joblib')


logistic_regression(df_students, "students")    
logistic_regression(df_employee, "employee")