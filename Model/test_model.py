import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

df_students = pd.read_csv('Data/depression_dataset_students.csv')
df_employee = pd.read_csv('Data/depression_dataset_employee.csv')

def test_models(df, label):
    print("Moyenne :\n", df.mean())
    print("Ecart type :\n", df.std())
    df.boxplot()
    plt.show()

    X = df.drop('Depression', axis=1)
    y = df['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2700)
    rd= RandomForestClassifier(n_estimators=100, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    classifiers = [
        ('Logistic Regression', lr),
        ('Random Forest', rd),
        ('Classification Tree', dt),
        ('K-Nearest Neighbors', knn),
        ('AdaBoost', ada),
        ('Gradient Boosting', gb)
    ]

    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f'{label} Accuracy {clf_name} : {accuracy_score(y_test, y_pred):.2f}')
        print(f'{label} F1 Score {clf_name} : {f1_score(y_test, y_pred):.2f}')

    vc = VotingClassifier(estimators=classifiers)
    vc.fit(X_train, y_train)
    y_pred2 = vc.predict(X_test)
    print(f'{label} Accuracy Voting Classifier : {accuracy_score(y_test, y_pred2):.2f}')
    print(f'{label} F1 Score Voting Classifier : {f1_score(y_test, y_pred2):.2f}\n')
    
test_models(df_students, "students")
test_models(df_employee, "employee")