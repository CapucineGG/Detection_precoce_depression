import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Data/final_depression_dataset_1.csv')
df_clean = df.copy()

#Features inutiles
df_clean = df_clean.drop(columns=['Name','City'])

#Correction fautes
df_clean['Profession'] = df_clean['Profession'].str.replace('Finanancial Analyst','Financial Analyst', regex=False)


# Regroupement de Profession : 35 métiers → 10 secteurs
profession = {
    'Teacher'               : 'Education',
    'Educational Consultant': 'Education',
    'Researcher'            : 'Education',
    'Research Analyst'      : 'Education',

    'Doctor'                : 'Sante',
    'Pharmacist'            : 'Sante',
    'Chemist'               : 'Sante',

    'Software Engineer'     : 'Tech',
    'Data Scientist'        : 'Tech',
    'UX/UI Designer'        : 'Tech',
    'Architect'             : 'Tech',
    'Civil Engineer'        : 'Tech',
    'Mechanical Engineer'   : 'Tech',
    'Electrician'           : 'Tech',

    'HR Manager'            : 'Business',
    'Business Analyst'      : 'Business',
    'Manager'               : 'Business',
    'Consultant'            : 'Business',
    'Entrepreneur'          : 'Business',
    'Accountant'            : 'Business',
    'Investment Banker'     : 'Business',
    'Financial Analyst'     : 'Business',

    'Lawyer'                : 'Droit',
    'Judge'                 : 'Droit',

    'Sales Executive'       : 'Service',
    'Marketing Manager'     : 'Service',
    'Digital Marketer'      : 'Service',
    'Travel Consultant'     : 'Service',
    'Customer Support'      : 'Service',
    'Content Writer'        : 'Service',
    'Graphic Designer'      : 'Service',
    'Chef'                  : 'Service',
    'Plumber'               : 'Service',
    'Pilot'                 : 'Service',
}

df_clean['Profession'] = df_clean['Profession'].replace(profession)

# Regroupement de Degree : 27 modalités → 4 niveaux
degree_map = {
    # Bac
    'Class 12'  : 'Bac',
    # Licence
    'B.Com'     : 'Licence', 'B.Ed'    : 'Licence', 'BCA'     : 'Licence',
    'BBA'       : 'Licence', 'BHM'     : 'Licence', 'BA'      : 'Licence',
    'B.Arch'    : 'Licence', 'B.Pharm' : 'Licence', 'BSc'     : 'Licence',
    'BE'        : 'Licence', 'LLB'     : 'Licence', 'B.Tech'  : 'Licence',
    # Master
    'MBA'       : 'Master',  'MSc'     : 'Master',  'M.Tech'  : 'Master',
    'M.Pharm'   : 'Master',  'ME'      : 'Master',  'LLM'     : 'Master',
    'MHM'       : 'Master',  'M.Ed'    : 'Master',  'MA'      : 'Master',
    'MBBS'      : 'Master',  'M.Com'   : 'Master',  'MCA'     : 'Master',
    # Doctorat
    'PhD'       : 'Doctorat', 'MD'     : 'Doctorat',
}

df_clean['Degree'] = df_clean['Degree'].replace(degree_map)

sleep = {
    "Less than 5 hours": 4.5,
    "5-6 hours": 5.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 8.5
}

df_clean['Sleep Duration'] = df_clean['Sleep Duration'].replace(sleep).astype(float)

#Encodage des donées :
le = LabelEncoder()

cols_binaires = ['Gender', 'Depression','Have you ever had suicidal thoughts ?','Family History of Mental Illness', 'Working Professional or Student','Degree', 'Profession', 'Dietary Habits']
for col in cols_binaires:
    df_clean[col] = le.fit_transform(df_clean[col]) 


#Séparation étudiant et employee:
#Student:
df_students = df_clean[df_clean['Working Professional or Student'] == 0]
df_students = df_students.drop(columns=['Work Pressure','Job Satisfaction','Profession', 'Working Professional or Student'])

print(df_students.dtypes)   
print(df_students.isnull().sum())

#Employee :
df_employee = df_clean[df_clean['Working Professional or Student'] == 1]
df_employee = df_employee.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction', 'Working Professional or Student'])

print(df_employee.dtypes)   
print(df_employee.isnull().sum())

#Sauvegarde Fichiers nettoyés :
import os
os.makedirs('Data', exist_ok=True)
df_clean.to_csv('Data/depression_dataset_clean.csv', index=False)
df_students.to_csv('Data/depression_dataset_students.csv', index=False)
df_employee.to_csv('Data/depression_dataset_employee.csv', index=False)