from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df_students = pd.read_csv('Data/depression_dataset_students.csv')
df_employee = pd.read_csv('Data/depression_dataset_employee.csv')

def knn_model(df, label):
    X = df.drop(columns=['Depression'])
    y = df['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Courbe de complexité:
    # train_scores, test_scores = [], []
    # for k in range(1, 11):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    #     test_scores.append(accuracy_score(y_test, knn.predict(X_test)))

    # plt.plot(range(1, 11), train_scores, label='training accuracy')
    # plt.plot(range(1, 11), test_scores, label='test accuracy')
    # plt.xlabel('n_neighbors')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title(f'Complexité du modèle KNN — {label}')
    # plt.show()

    param_grid = {'n_neighbors': range(1, 11)}
    knn= KNeighborsClassifier()
    knn_cv=GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(X_train, y_train)
    print(f"Meilleur k pour {label} : {knn_cv.best_params_['n_neighbors']}")
    print(f"Meilleure accuracy pour {label} : {knn_cv.best_score_:.2f}")

    knn_best = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
    knn_best.fit(X_train, y_train)
    y_pred = knn_best.predict(X_test)

    print(f"\n{label} — KNN Classifier (k={knn_best.n_neighbors}) :")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

knn_model(df_employee, "Working Professional")
knn_model(df_students, "Student")