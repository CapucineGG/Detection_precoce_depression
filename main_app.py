import streamlit as st
import joblib
import pandas as pd

st.title("Détection précoce de la dépression")

profil = st.selectbox("Tu es :", ["Étudiant", "Employé"])

if profil == "Étudiant":
    #pipeline = lr student + scaler
    pipeline = joblib.load('App_train_model/lr_students.joblib')
    
    age = st.slider("Quel est ton âge ?", 15, 35)
    academic_pressure = st.slider("Pression académique (1=faible, 5=élevée)", 1, 5)
    study_satisfaction = st.slider("Satisfaction de tes études (1=faible, 5=élevée)", 1, 5)
    dietary = st.selectbox("Habitudes alimentaires", ["Mauvaises", "Moderées", "Bonnes"])
    dietary = {"Mauvaises": 0, "Moderées": 1, "Bonnes": 2}[dietary]
    degree = st.selectbox("Diplôme", ["Bac", "Licence", "Master", "Doctorat"])
    degree = {"Bac": 0, "Licence": 1, "Master": 2, "Doctorat": 3}[degree]
    suicidal_thoughts = st.selectbox("As-tu déjà eu des pensées suicidaires ?", ["Non", "Oui"])
    suicidal_thoughts = 1 if suicidal_thoughts == "Oui" else 0
    work_hours = st.slider("Combien d'heures travailles-tu par jour ?", 0, 12)
    financial_stress = st.slider("Stress financier (1=faible, 5=élevé)", 1, 5)
    family_history = st.selectbox("Antécédents familiaux de maladie mentale ?", ["Non", "Oui"])
    family_history = 1 if family_history == "Oui" else 0
    
    
    data = pd.DataFrame([[age, academic_pressure, study_satisfaction, dietary, degree, suicidal_thoughts,
                     work_hours, financial_stress, family_history]],
                    columns=['Age', 'Academic Pressure', 'Study Satisfaction',
                             'Dietary Habits', 'Degree',
                             'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
                             'Financial Stress', 'Family History of Mental Illness'])
    

else:
    #pipeline = lr employee + scaler
    pipeline = joblib.load('App_train_model/lr_employee.joblib')

    age = st.slider("Quel est ton âge ?", 18, 60)
    profession = st.selectbox("Profession", ["Education", "Santé", "Tech", "Business", "Droit", "Service"])
    profession = {"Education": 0, "Santé": 1, "Tech": 2, "Business": 3, "Droit": 4, "Service": 5}[profession]
    work_pressure = st.slider("Pression au travail (1=faible, 5=élevée)", 1, 5)
    job_satisfaction = st.slider("Satisfaction au travail (1=faible, 5=élevée)", 1, 5)
    sleep_duration = st.selectbox("Durée de sommeil", ["Moins de 5h", "5-6h", "7-8h", "Plus de 8h"])
    sleep_duration = {"Moins de 5h": 4.5, "5-6h": 5.5, "7-8h": 7.5, "Plus de 8h": 8.5}[sleep_duration]
    dietary = st.selectbox("Habitudes alimentaires", ["Mauvaises", "Moderées", "Bonnes"])
    dietary = {"Mauvaises": 0, "Moderées": 1, "Bonnes": 2}[dietary]
    degree = st.selectbox("Diplôme", ["Bac", "Licence", "Master", "Doctorat"])
    degree = {"Bac": 0, "Licence": 1, "Master": 2, "Doctorat": 3}[degree]
    suicidal = st.selectbox("As-tu déjà eu des pensées suicidaires ?", ["Non", "Oui"])
    suicidal = 1 if suicidal == "Oui" else 0
    work_hours = st.slider("Combien d'heures travailles-tu par jour ?", 0, 12)
    financial_stress = st.slider("Stress financier (1=faible, 5=élevé)", 1, 5)


    data = pd.DataFrame([[age, profession, work_pressure, job_satisfaction,
                          sleep_duration, dietary, degree, suicidal,
                          work_hours, financial_stress]],
                        columns=['Age', 'Profession', 'Work Pressure', 'Job Satisfaction',
                                 'Sleep Duration', 'Dietary Habits', 'Degree',
                                 'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
                                 'Financial Stress'])

if st.button("Prédire"):
    prediction = pipeline.predict(data)
    proba = pipeline.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(f"Risque de dépression détecté : {proba*100:.1f}%")
    else:
        st.success(f"Pas de risque détecté : {proba*100:.1f}%")