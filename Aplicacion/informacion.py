import streamlit as st

def informacion():
    st.header('Contexto')
    st.write("""
    Este conjunto de datos fue creado para predecir las admisiones a programas de posgrado desde una perspectiva india. 
    En el proceso de solicitud para programas de maestría, las instituciones educativas suelen considerar varios factores 
    para determinar la idoneidad de un candidato.
    """)

    st.header('Contenido')
    st.write("""
    El conjunto de datos contiene varios parámetros que se consideran importantes durante la solicitud para programas de maestría:
    """)
    st.markdown("""
    - **GRE (Graduate Record Examination)**: El GRE es un examen estandarizado que mide la aptitud verbal, cuantitativa y analítica de los estudiantes. Los puntajes de GRE están en una escala de 0 a 340.

    - **TOEFL (Test of English as a Foreign Language)**: El TOEFL es un examen estandarizado que evalúa la capacidad de comprensión auditiva, expresión oral, comprensión de lectura y expresión escrita en inglés de personas cuya lengua materna no es el inglés. Los puntajes de TOEFL van de 0 a 120.

    - **University Rating**: Esta calificación es una medida subjetiva de la reputación y calidad de la institución donde el solicitante obtuvo su título de pregrado. Se clasifica en una escala del 1 al 5, donde 1 indica una universidad de menor prestigio y 5 indica una universidad altamente prestigiosa.

    - **Statement of Purpose (SOP) and Letter of Recommendation Strength (LOR)**: Estos factores evalúan la calidad y persuasión de la declaración de propósito del solicitante, así como la fuerza y credibilidad de las cartas de recomendación proporcionadas por sus referencias académicas o profesionales. Ambos se califican en una escala del 1 al 5.

    - **Undergraduate GPA (CGPA)**: El GPA (Promedio de Calificaciones) de pregrado es un indicador de rendimiento académico durante los estudios universitarios de pregrado. Se clasifica en una escala del 0 al 10.

    - **Experiencia en Investigación (Research)**: Este parámetro indica si el solicitante tiene experiencia previa en investigación, con un valor de 1 si la tiene y 0 si no la tiene.

    - **Chance of Admit**: Esta es la variable objetivo que se busca predecir. Representa la probabilidad de que un solicitante sea admitido en el programa de posgrado y está en un rango de 0 a 1.
    """)

    st.header('Aclaraciones')
    st.write("""
    Para aclarar, la "Probabilidad de Admisión" es un parámetro que se les preguntó a los individuos (algunos valores fueron ingresados manualmente) antes de conocer los resultados de su solicitud. En palabras del autor:
    """)
    st.info("""
    "Para algunos perfiles, pregunté a los solicitantes qué tan seguros estaban de ser admitidos en términos de porcentaje. Agregué un decimal adicional para aumentar la precisión. Para el resto de los datos, dado que la regresión es un aprendizaje supervisado, proporcioné valores que realmente fueran comprensibles y tuvieran suficiente sentido"
    """)
    st.write("Para más información, visita la página del conjunto de datos en [Kaggle](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions/data?select=Admission_Predict.csv)")


    return None 