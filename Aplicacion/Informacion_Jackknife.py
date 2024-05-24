import streamlit as st


def info_jackknife():
    st.subheader('Jackknife')
    st.markdown("""
    Los parámetros/pesos de las submuestras se usarán de la siguiente manera:

    - **$\\theta_{biased}$**: Estimación Sesgada del Jackknife (media de los pesos)
    - **$SE(\\theta)$**: Error Estándar del Jackknife (desviación estándar de los pesos)
    - **$SE(\\theta) / |\\theta_{biased}|$**: Indicador rápido de la estabilidad de los parámetros/pesos

    Y finalmente,

    - **$\\theta_{jack}$**: Estimación del Jackknife Corregida por Sesgo (que vamos a utilizar como parámetros/pesos del modelo)
    - Test de Hipótesis de Jackknife (rechaza todo el modelo si algún parámetro/peso es inestable)
    """)

    st.subheader('Estimaciones de Jackknife') 

    st.markdown("""

    La **Estimación Sesgada del Jackknife** es la media de una estimación de parámetro específica (por ejemplo, la intersección) para todas las submuestras.

    $$
    \\theta_{biased} = \\frac{1}{n} \\sum_{i=1}^n \\theta_{(i)}
    $$

    En nuestro ejemplo, la Estimación Sesgada del Jackknife es
                """)
    
    
    st.markdown("""
    El núcleo del procedimiento del Jackknife es la pregunta sobre qué tan estables son realmente los parámetros del modelo.
    El **Error Estándar (SE) de las Estimaciones del Jackknife** es

    $$
    SE(\\theta) = \\sqrt{ \\frac{n - 1}{n} \\sum_{i=1}^n (\\theta_{(i)} - \\theta_{\\text{biased}})^2 }
    $$
    """)

    st.markdown("""
    La **estimación del Jackknife sin sesgo** o **estimación del Jackknife corregida por sesgo** es la estimación del parámetro de la muestra completa menos una corrección de sesgo específica del Jackknife.

    $$
    \\theta_{jack} = \\theta_{all} - bias_{jack}
    $$

    Primero, vamos a estimar el modelo en la muestra completa.
                """)

    st.markdown("""
    Luego $bias_{jack}$ es

    $$
    bias_{jack} = (n - 1) (\\theta_{biased}-\\theta_{all} )
    $$
                """)

    st.markdown("""
    Asi obtenemos $\\theta_{jack}$ o equivalentemente reemplazando en las formulas

    $$
    \\theta_{jack} = n \\, \\theta_{all} - (n - 1) \\, \\theta_{biased}
    $$

                """)
    st.subheader('Test de Hipotesis')

    st.markdown("""
    ¿Qué sucede si intentamos llevar esta idea más allá y usar los pseudovalores para construir un intervalo de confianza? Un enfoque razonable sería formar un intervalo
    $$
    \\tilde{\\theta} \\pm t_{n-1}^{(1-\\alpha)} \\widehat{\\mathrm{se}}_{\\mathrm{jack}},
    $$
    donde $t_{n-1}^{(1-\\alpha)}$ es el percentil $(1-\\alpha)$ de la distribución $t$ con $n-1$ grados de libertad. Resulta que este intervalo no funciona muy bien: en particular, no es significativamente mejor que intervalos más rudimentarios basados en la teoría normal. Se necesitan enfoques más refinados para la construcción de intervalos de confianza, como se describe en los Capítulos 12-14 del libro de Bradley Efron, R.J. Tibshirani (1993), "An Introduction to the Bootstrap".
                """)


    st.subheader('Bibliografia')
    st.markdown("""
    * Bradley Efron, R.J. Tibshirani (1993), "An Introduction to the Bootstrap", Chapter 11, [Google Books](https://books.google.de/books?id=gLlpIUxRntoC&lpg=PR14&ots=A9xuU4O5H5&lr&pg=PA141#v=onepage&q&f=false)
    * Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019
                """)
    return None 