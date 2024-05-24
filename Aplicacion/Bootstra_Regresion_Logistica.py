import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  
from sklearn.linear_model import LogisticRegression
from scipy.stats import bootstrap
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
########################################################################################################
################################### Funcion auxiliar ###################################################
########################################################################################################



def esta_en_intervalo(row):
    return row['Lower'] <= row['theta total datos'] <= row['Upper']


########################################################################################################
########################################################################################################
########################################################################################################


########################################################################################################
############################## Estadistica #############################################################
########################################################################################################

def estadistica_interes(*args):
    # tupla_arrays = (args)
    df = pd.DataFrame(args).transpose()
    X=df[df.columns[:-1]]
    y=df[df.columns[-1]]
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    # clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf = LogisticRegression(
         max_iter= 60000, 
     fit_intercept=True,# Incluir el término de sesgo (intercept)
     )
    clf.fit( X_train,y_train)
    # clf= clf.named_steps['logisticregression']
    return np.append(clf.intercept_,clf.coef_)

########################################################################################################
########################################################################################################
########################################################################################################


##########################################################################################################################################
############################### Funcion auxiliar para mostrar los intervalos de confianza #################################################
##########################################################################################################################################



def plot_intervalos_de_confianza(resultado_boots, tupla_arrays):
    
    intervalos_confianza = [(resultado_boots.confidence_interval[0][i], resultado_boots.confidence_interval[1][i]) for i in range(len(resultado_boots.confidence_interval[0]))]

    df_intervalos = pd.DataFrame(intervalos_confianza, columns=["Lower", "Upper"])
    df_intervalos['theta'] = [f'theta_{i}' for i in range(len(intervalos_confianza))]

    df_intervalos['theta total datos'] = estadistica_interes(*tupla_arrays)

    df_intervalos['En Intervalo'] = df_intervalos.apply(esta_en_intervalo, axis=1)

    df_intervalos = df_intervalos[['theta', 'theta total datos', 'Lower', 'Upper', 'En Intervalo']]


    return df_intervalos

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


##########################################################################################################################################
########################################## Funcion Que realiza Bootstrap ########################################################################
##########################################################################################################################################
@st.cache_data
def realizacion_bootstrap(df_data, _columnas, numero, Chance):
    
    tupla_arrays = tuple(df_data[col].to_numpy() for col in _columnas)  #Formato
    resultado_boots= bootstrap(tupla_arrays, estadistica_interes, vectorized=False, paired=True,
                n_resamples=numero, method='percentile', confidence_level=0.9)
    
    return resultado_boots

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


##########################################################################################################################################
############################################### Funcion que muestra el streamlit ##########################################################################
##########################################################################################################################################

def bootstrap_regresion_logistica(df_data,Chance,columnas, numero):

    ########################################################################################################
    ############################ Lectura de archivo y parametros ###########################################
    ########################################################################################################

    df_data[df_data.columns[-1]]= np.where(df_data[df_data.columns[-1]] >= Chance, 1, 0)   
  
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    
    ########################################################################################################
    ####################################### Regresion logistica ##########################################
    ########################################################################################################

    df = df_data[columnas]
    X=df[df.columns[:-1]]
    y=df[df.columns[-1]]
    y = np.where(y >= Chance, 1, 0)    #Realizaremos un cambio en la variable de interes, debido a restricciones del modelo

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    # clf = make_pipeline(StandardScaler(), LogisticRegression(fit_intercept=True))
    clf = LogisticRegression(
         max_iter= 60000,
     fit_intercept=True,# Incluir el término de sesgo (intercept)
     )
    clf.fit( X_train,y_train)
    precision = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    st.subheader("Precisión del Modelo")

    precision_md = f"""
    
    La precisión del modelo, que representa la proporción de predicciones correctas sobre el total de predicciones, es del **{precision:.3f}**.
    """

    st.markdown(precision_md)




    ########################################################################################################
    ########################################################################################################
    ########################################################################################################


    ########################################################################################################
    ########################################## Bootstrap ###################################################
    ########################################################################################################


    
    tupla_arrays = tuple(df_data[col].to_numpy() for col in columnas)  #Formato
    resultado_boots= realizacion_bootstrap(df_data, columnas, numero, Chance)
    
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    ########################################################################################################
    ################################# Graficar distribucion ################################################
    ########################################################################################################

    total_distribuciones = len(resultado_boots.bootstrap_distribution)

    num_filas = (total_distribuciones + 1) // 2

    fig = make_subplots(rows=num_filas, cols=2, subplot_titles=[rf'$\\theta$_{i+1}' for i in range(total_distribuciones)])

    for i, distribution in enumerate(resultado_boots.bootstrap_distribution):
        fig.add_trace(
            go.Histogram(
                x=distribution,
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
        title_text='Histogramas de Distribuciones Bootstrap de las estimacion de los parametros de la regresion lineal',
        showlegend=False,
        title_x=0,  
        title_font_size=24  
    )

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=18)  
    fig.update_layout(height=800)  
    st.plotly_chart(fig, use_container_width=True)



    ########################################################################################################
    ########################################################################################################
    ########################################################################################################


    ########################################################################################################
    ################################# Intervalo de Confianza ###############################################
    ########################################################################################################
    
    

    #################### Percentil Directo #################
    st.subheader("Intervalo de confianza Percentil Directo")
    st.dataframe(plot_intervalos_de_confianza(resultado_boots, tupla_arrays))

     ################# Percentil Basico #################################
    st.subheader("Intervalo de confianza basico")
    resultado_boots_basico= bootstrap(tupla_arrays, estadistica_interes, vectorized=False, paired=True,
                bootstrap_result=resultado_boots, n_resamples=0, method='basic', confidence_level=0.9)
    st.dataframe(plot_intervalos_de_confianza(resultado_boots_basico, tupla_arrays))

    ################# BCa #################################
    st.subheader("Intervalo de confianza BCa")
    resultado_boots_bca= bootstrap(tupla_arrays, estadistica_interes, vectorized=False, paired=True,
                bootstrap_result=resultado_boots, n_resamples=0, method='BCa', confidence_level=0.9)
    st.dataframe(plot_intervalos_de_confianza(resultado_boots_bca, tupla_arrays))
    
