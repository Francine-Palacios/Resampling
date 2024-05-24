import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from plotly.subplots import make_subplots  
import plotly.graph_objects as go
 


from Funciones_regresion_logistica import estimator_fn
from Funciones_regresion_logistica import predictor_fn

@st.cache_data
def regresion_logistica_jack(X_train, y_train):

    N = len(y_train)
    params = []  
    idx = np.arange(0, N)

    for i in idx:
        mask = idx != i

        clf = LogisticRegression(
        max_iter= 60000,
        fit_intercept=True,    # Incluir el término de sesgo (intercept)
        )
        clf.fit( X_train.iloc[mask, :],y_train[mask])
        beta = np.append(clf.intercept_,clf.coef_)
        params.append(beta)  
    params = np.vstack(params)  

    return params

@st.cache_data
def regresion_logistica_(X_train, y_train):
    
    clf = LogisticRegression(
        max_iter=60000,
        fit_intercept=True,    # Incluir el término de sesgo (intercept)
        )

    clf.fit( X_train,y_train)
    theta_all= np.append(clf.intercept_,clf.coef_)

    return theta_all, clf.classes_

def jackknife_regresion_logistica(X,y,test_s):
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_s, random_state=42)

    N = len(y_train)
    
    params= regresion_logistica_jack(X_train, y_train)

    #######################################################################################
    ######################## Histograma de los parametros #################################
    #######################################################################################

    st.subheader("Histograma de los parametros")
    num_features = len(params[0])  
    features = ['intercept'] + [f'beta_{i}' for i in range(1, num_features)]  
    df_params = pd.DataFrame(params, columns=features)
    total_distribuciones = len(df_params.columns)

    num_filas = (total_distribuciones + 1) // 2

    fig = make_subplots(rows=num_filas, cols=2, subplot_titles=[f'$\\theta_{{{i+1}}}$' for i in range(total_distribuciones)])

    for i, distribution in enumerate(df_params.columns):
        fig.add_trace(
            go.Histogram(
                x=df_params[distribution],
                nbinsx=30,  
                marker=dict(
                    line=dict(width=1, color='black') 
                ),
                name=rf'$\\theta_{i+1}$'  
            ),
            row=(i // 2) + 1,  
            col=(i % 2) + 1    
        )

    fig.update_layout(
        title_text='Histogramas de Distribuciones Bootstrap de las estimacion de los parametros de la regresion logistica',
        showlegend=False,
        title_x=0,  
        title_font_size=24  
    )

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=18)  
    fig.update_layout(height=800)  
    st.plotly_chart(fig, use_container_width=True)

    #######################################################################################
    #######################################################################################
    #######################################################################################


    #####################################################################################################
    ############## Calculos de estadisticas de la tecnica de Jackknife ##################################
    #####################################################################################################

    theta_biased = np.mean(params, axis=0)
    def SE_jack(theta_i, theta_biased):
        return np.sqrt( (N-1)/N * np.sum((theta_i - theta_biased)**2, axis=0) )

    se_jack= SE_jack(params , theta_biased)
    theta_all, classes = regresion_logistica_(X_train, y_train)
    bias_jack = (N - 1) * (theta_biased - theta_all)
    theta_jack = N * theta_all - (N - 1) * theta_biased

    #####################################################################################################
    #####################################################################################################
    #####################################################################################################


    #####################################################################################################
    ############################ Resumen de estadisticas ################################################
    #####################################################################################################

    st.subheader("Resumen de las estadisticas y del metodo Jackknife")
    st.markdown("""
    Basados en la seccion de Test de Hipotesis, usaremos un intervalo basado en la teoria normal que es usalmente mas vista implementadas en paquetes como el paquete para python [Astropy](https://docs.astropy.org/en/stable/api/astropy.stats.jackknife_stats.html) en su seccion de Jackknife. El intervalo de confianza se hara respecto al 95%
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

    st.dataframe(test_resumen)

    st.markdown("""
    Y como ya habíamos sospechado, las estimaciones de parámetros de la intersección son simplemente aleatorias, es decir, las estimaciones de parámetros de la intersección en las submuestras del Jackknife simplemente ocurrieron por *casualidad*.

    **Como consecuencia, se rechaza todo el modelo.**
                """)
    
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################


    #####################################################################################################
    ################################### Prediccion ######################################################
    #####################################################################################################

    st.subheader("Prediccion")

    #### Prediccion normal #####

    y_pred = predictor_fn(X_test, theta_all)
    y_pred_etiqueta = np.where(y_pred >= 0.5, 1, 0)

    cm = confusion_matrix(y_test, y_pred_etiqueta, labels=classes)
    fig = ff.create_annotated_heatmap(z=cm, x=classes.tolist(), y=classes.tolist())
    fig.update_layout(title='Matriz de Confusión', xaxis_title='Etiqueta Predicha', yaxis_title='Etiqueta Verdadera')

    st.plotly_chart(fig, use_container_width=True)


    ############## Prediccion con Jack ###################

    y_pred = predictor_fn(X_test, theta_jack)
    y_pred_etiqueta = np.where(y_pred >= 0.5, 1, 0)

    cm = confusion_matrix(y_test, y_pred_etiqueta, labels=classes)
    fig = ff.create_annotated_heatmap(z=cm, x=classes.tolist(), y=classes.tolist())
    fig.update_layout(title='Matriz de Confusión', xaxis_title='Etiqueta Predicha', yaxis_title='Etiqueta Verdadera')

    st.plotly_chart(fig, use_container_width=True)

    #####################################################################################################
    #####################################################################################################
    #####################################################################################################

    return None