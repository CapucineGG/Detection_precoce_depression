from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_clean = pd.read_csv('Data/depression_dataset_clean.csv')
df_students = pd.read_csv('Data/depression_dataset_students.csv')
df_employee = pd.read_csv('Data/depression_dataset_employee.csv')

def random_forest(df, label):
    X = df.drop(columns=['Depression'])
    y = df['Depression']

    features = X.columns.tolist() 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt', random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    print("Random Forest accuracy:", accuracy_score(y_test, rf.predict(X_test)))
    print(classification_report(y_test, rf.predict(X_test)))

    # Validation croisée (plus fiable qu'un seul split)
    scores = cross_val_score(rf, X, y, cv=5)
    print(f"CV scores : {scores.round(2)} — moyenne : {scores.mean():.2f}")

    featureimpor = pd.DataFrame(rf.feature_importances_,
                                index=features,
                                columns=["importance"]).sort_values("importance", ascending=False)
    

    plt.barh(featureimpor.index, rf.feature_importances_)
    plt.title(f'Feature Importance — {label}')
    plt.show()

random_forest(df_clean, 'All')
random_forest(df_students, 'Students')
random_forest(df_employee, 'Employees')