import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

from Funciones_regresion_lineal import predictor_fn
from Funciones_regresion_lineal import loss_fn

def jackknife_regresion_lineal(X,y, test_s_lineal):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_s_lineal, random_state=42)

    N = len(y_train)
    ### Usaremos el modelo de Regresion Logistica ##
    if X_train.empty:
        st.warning("Seleccione parametros")
        st.stop()

    params = []  
    idx = np.arange(0, N)

    for i in idx:
        mask = idx != i

        clf = LinearRegression(
        fit_intercept=True,    # Incluir el término de sesgo (intercept)
        )
        clf.fit( X_train.iloc[mask, :],y_train[mask])
        beta = np.append(clf.intercept_,clf.coef_)
        params.append(beta)  
    params = np.vstack(params)  

    st.subheader("Plot de resultados")
    ###### Grafico de caja de los parametros #####
    st.subheader('Boxplot de los parametros')
    num_features = len(params[0])  
    features = ['intercept'] + [f'beta_{i}' for i in range(1, num_features)]  
    df_params = pd.DataFrame(params, columns=features)
    with st.expander("Eleccion de los parametros"):
        st.markdown("Para los graficos de Boxplot de los parametros del modelo, seleccione las columnas de interes")
        if st.checkbox("Todas ellos"):
            features_ =['intercept'] + [f'beta_{i}' for i in range(1, num_features)]  

        else:
            columnas=st.multiselect('Seleccione las columnas que le interesan para el boxplot de los parametros ', ['intercept'] + [f'beta_{i}' for i in range(1, num_features)]  
        )
            features_=columnas
    if len(features_)== 0:
        st.warning("Seleccione parametros para el boxplot")

    for feature in features_:
        fig = px.box(df_params, y=feature, color_discrete_sequence=['lightseagreen'])
        fig.update_layout(title=feature, yaxis_title="Valor")
        st.plotly_chart(fig, use_container_width=True)
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

    theta_biased = np.mean(params, axis=0)
    st.table(pd.DataFrame([theta_biased], columns=features))


    st.markdown("""
    El núcleo del procedimiento del Jackknife es la pregunta sobre qué tan estables son realmente los parámetros del modelo.
    El **Error Estándar (SE) de las Estimaciones del Jackknife** es

    $$
    SE(\\theta) = \\sqrt{ \\frac{n - 1}{n} \\sum_{i=1}^n (\\theta_{(i)} - \\theta_{\\text{biased}})^2 }
    $$
    """)

    def SE_jack(theta_i, theta_biased):
        return np.sqrt( (N-1)/N * np.sum((theta_i - theta_biased)**2, axis=0) )

    se_jack= SE_jack(params , theta_biased)

    st.table(pd.DataFrame([se_jack], columns=features))

    st.markdown("""
    La **estimación del Jackknife sin sesgo** o **estimación del Jackknife corregida por sesgo** es la estimación del parámetro de la muestra completa menos una corrección de sesgo específica del Jackknife.

    $$
    \\theta_{jack} = \\theta_{all} - bias_{jack}
    $$

    Primero, vamos a estimar el modelo en la muestra completa.
                """)

    clf = LinearRegression(
        fit_intercept=True,    # Incluir el término de sesgo (intercept)
        )

    clf.fit( X_train,y_train)
    theta_all= np.append(clf.intercept_,clf.coef_)

    st.table(pd.DataFrame([theta_all], columns=features))


    st.markdown("""
    Luego $bias_{jack}$ es

    $$
    bias_{jack} = (n - 1) (\\theta_{biased}-\\theta_{all} )
    $$
                """)

    
    bias_jack = (N - 1) * (theta_biased - theta_all)

    st.table(pd.DataFrame([bias_jack], columns=features))

    st.markdown("""
    Asi obtenemos $\\theta_{jack}$ o equivalentemente reemplazando en las formulas

    $$
    \\theta_{jack} = n \\, \\theta_{all} - (n - 1) \\, \\theta_{biased}
    $$

                """)


    theta_jack = N * theta_all - (N - 1) * theta_biased

    st.table(pd.DataFrame([theta_jack], columns=features))

    st.subheader('Test de Hipotesis')

    st.markdown("""
    ¿Qué sucede si intentamos llevar esta idea más allá y usar los pseudovalores para construir un intervalo de confianza? Un enfoque razonable sería formar un intervalo
    $$
    \\tilde{\\theta} \\pm t_{n-1}^{(1-\\alpha)} \\widehat{\\mathrm{se}}_{\\mathrm{jack}},
    $$
    donde $t_{n-1}^{(1-\\alpha)}$ es el percentil $(1-\\alpha)$ de la distribución $t$ con $n-1$ grados de libertad. Resulta que este intervalo no funciona muy bien: en particular, no es significativamente mejor que intervalos más rudimentarios basados en la teoría normal. Se necesitan enfoques más refinados para la construcción de intervalos de confianza, como se describe en los Capítulos 12-14 del libro de Bradley Efron, R.J. Tibshirani (1993), "An Introduction to the Bootstrap".
                """)

    st.markdown("""
    Basados en lo anterior, usaremos un intervalo basado en la teoria normal que es usalmente mas vista implementadas en paquetes como el paquete para python [Astropy](https://docs.astropy.org/en/stable/api/astropy.stats.jackknife_stats.html) en su seccion de Jackknife
                """)

    ########### Intervalo de confianza al  0.95 #### 
    from scipy.special import erfinv

    z_score = np.sqrt(2.0) * erfinv(0.95)
    conf_interval = theta_jack + z_score * np.array((-se_jack, se_jack))


    test_resumen=pd.DataFrame()

    test_resumen["Coeficientes"]= theta_all
    test_resumen['Coeficientes Jack']= theta_jack
    test_resumen['STD Jack']= se_jack
    test_resumen['Intervalo izq']= [x[0] for x in conf_interval.T]
    test_resumen['Intervalo der']=[x[1] for x in conf_interval.T]
    test_resumen['En Intervalo'] = test_resumen.apply(lambda row: row['Intervalo izq'] <= row['Coeficientes'] <= row['Intervalo der'], axis=1)

    st.table(test_resumen)


   
    st.subheader("Prediccion")

    #### Prediccion normal #####


    y_pred = predictor_fn(X_test, theta_all)   #### Prediccion con la totalidad de datos
    y_pred_jack = predictor_fn(X_test, theta_jack) #### Prediccion con el estimador de Jack



    predicciones=pd.DataFrame()
    predicciones['Real']= y_test
    predicciones['Prediccion']=y_pred
    predicciones['Prediccion Jack']= y_pred_jack
    st.dataframe(predicciones)
    
    ###############################################

    fig = px.scatter(predicciones, x='Real', y='Prediccion', labels={'x':'Valor Real', 'y':'Predicción'},
                    title='Comparación de Valores Reales y Predicciones')

    fig.add_shape(type='line', line=dict(dash='dash'),
                x0=predicciones['Real'].min(), y0=predicciones['Real'].min(),
                x1=predicciones['Real'].max(), y1=predicciones['Real'].max())

    st.plotly_chart(fig)


    fig = px.scatter(predicciones, x='Real', y='Prediccion Jack', labels={'x':'Valor Real', 'y':'Predicción Jack'},
                    title='Comparación de Valores Reales y Predicciones Jack')

    fig.add_shape(type='line', line=dict(dash='dash'),
                x0=predicciones['Real'].min(), y0=predicciones['Real'].min(),
                x1=predicciones['Real'].max(), y1=predicciones['Real'].max())

    st.plotly_chart(fig)


    fig = px.scatter(predicciones, x='Prediccion', y='Prediccion Jack', labels={'x':'Prediccion', 'y':'Predicción Jack'},
                    title='Comparación de Predicion y Predicciones Jack')

    fig.add_shape(type='line', line=dict(dash='dash'),
                x0=predicciones['Real'].min(), y0=predicciones['Real'].min(),
                x1=predicciones['Real'].max(), y1=predicciones['Real'].max())

    st.plotly_chart(fig)

    ###################################################


    loss_toda_muestra = loss_fn(y_test, y_pred)
    loss_jack = loss_fn(y_test, y_pred_jack)

    df_resultados = pd.DataFrame({
        'Parametros usando toda la muestra': [loss_toda_muestra],
        'Parametros Jack': [loss_jack]
    })

    st.table(df_resultados)


    st.subheader('Bibliografia')
    st.markdown("""
    * Bradley Efron, R.J. Tibshirani (1993), "An Introduction to the Bootstrap", Chapter 11, [Google Books](https://books.google.de/books?id=gLlpIUxRntoC&lpg=PR14&ots=A9xuU4O5H5&lr&pg=PA141#v=onepage&q&f=false)
    * Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019
                """)


    return None