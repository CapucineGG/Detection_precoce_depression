import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/final_depression_dataset_1.csv')
df_clean = pd.read_csv('Data/depression_dataset_clean.csv')
df_students = pd.read_csv('Data/depression_dataset_students.csv')
df_employee = pd.read_csv('Data/depression_dataset_employee.csv')

#Exploration df :

print(df.isnull().sum())
print(df.info())
print(df.duplicated().sum())
print(df.select_dtypes(include=['str', 'float', 'int']).isnull().sum())
print(df["Depression"].value_counts())
print(df.columns)
print(df.dtypes)

cols_a_explorer=['Name', 'Gender', 'Age', 'City', 'Working Professional or Student',
       'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA',
       'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
       'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
       'Work/Study Hours', 'Financial Stress',
       'Family History of Mental Illness', 'Depression']
 
for col in cols_a_explorer:
    print(df[col].value_counts())

#Exploration df_clean :
print(df_clean.isnull().sum())
print(df_clean.info())
print(df_clean.duplicated().sum())
print(df_clean.dtypes)

#Exploration df_students :
print(df_students.isnull().sum())  
print(df_students.info())
print(df_students.duplicated().sum())
print(df_students.dtypes)

#Exploration df_employee :
print(df_employee.isnull().sum())
print(df_employee.info())
print(df_employee.duplicated().sum())
print(df_employee.dtypes)
