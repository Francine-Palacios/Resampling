import streamlit as st
from informacion import informacion
from Analisis_descriptivo import analisis_descriptivo
from Jackknife_Regresion_logistica import jackknife_regresion_logistica
from Jackknife_Regresion_lineal import jackknife_regresion_lineal
from Bootstra_Regresion_Lineal import bootstrap_regresion_lineal
from Bootstra_Regresion_Logistica import bootstrap_regresion_logistica
from Informacion_Jackknife import info_jackknife

from Boostrap_informacion import booststrap_info
import pandas as pd

import numpy as np


########################################################################
############### Configuracion e informacion ############################
########################################################################


st.set_page_config(page_title='Tecnicas de Remuestreo', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)


_, _, col3 = st.columns([3,6,3])

with col3:
    url_imagen = "https://matematica.usm.cl/wp-content/themes/dmatUSM/assets/img/logoDMAT2.png"

    st.image(url_imagen, width=250)



_, col2, _ = st.columns([1,3,1])

with col2:
    st.title('Tecnicas de Remuestreo')


texto_descripcion = """
Ests aplicacion está dedicado al análisis exhaustivo de los parámetros obtenidos de un modelo de **regresión logística** y de **regresion lineal**. Se emplea los métodos de remuestreo Jackknife y Bootstrap para estimar la precisión y la confiabilidad de los parámetros estimados. El conjunto de datos utilizado proviene de registros de admisión a programas de posgrado 'Graduate Admission', el cual está disponible para el público por la plataforma [kaggle](https://www.kaggle.com/) 
"""


texto_info = """
- Autor: Francine Palacios
- Ramo: Topicos Avanzados
- Profesor: Ronny Vallejos
"""




_, exp_col, _ = st.columns([1,3,1])
with exp_col:
    with st.expander("** Información y usos **"):
        st.markdown(texto_descripcion)
        st.info(texto_info)
        


########################################################################
########################################################################
########################################################################


################### Subida de archivo de datos###################################
# uploaded_file = st.sidebar.file_uploader("Cargar el archivo de datos")

# if uploaded_file is None:
      # st.warning('Favor subir el archivo de datos')

InfoTab,Analisis,jackknif, boots = st.tabs(["Información","Analisis Descriptivo", 'Jackknife', 'Bootstrap'])


#########################################################################
#########################################################################
#########################################################################


#########################################################################
#########################################################################
#########################################################################
    
#########################################################################
###################### Cuerpo de la pagina ##############################
#########################################################################

##################################################################################################
############## Tabla con los datos ############################################
##################################################################################################
path='https://raw.githubusercontent.com/Francine-Palacios/Resampling/417e51132ad4f4c3a3af1fd46e6439d79bfdea81/Data/Admission_Predict_Ver1.1.csv'
df_data= pd.read_csv(path)
df_data = df_data.drop(columns=['Serial No.'])
   

with InfoTab:
    informacion()

with Analisis:
    analisis_descriptivo(df_data)

with jackknif:
    # jackknife_regresion_logistica(df_data)
    st.subheader("Entrenamiento")
    with st.form(key='my_form'):
        with st.expander("Parametros para el entrenamiento"):
            test_s= st.slider("¿cuanto de dato de testeo?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)
            Chance= st.slider("¿Para el entrenamiento (Regresion Logistica) cuanto considera que se acepta en 'Chance of Admit'?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)
            XX=df_data[df_data.columns[:-1]]
            y=df_data[df_data.columns[-1]]
            st.markdown("Para el entrenamiento, que columnas quiere utilizar")
            if st.checkbox("Todas aquellas"):
                X=XX.copy()
                pass
            else:
                col=st.multiselect('Seleccione las columnas que le interesan para el entrenamiento ', XX.columns)
                X= XX[col]
            submit_button = st.form_submit_button('Entrenar')

    Informacion_jack,Regresion_lineal_Jack, Regresgion_logistica_Jack = st.tabs(["Contexto","Regresion Lineal","Regresion Logistica"])
    
    with Informacion_jack:
        info_jackknife()
    with Regresion_lineal_Jack:
        if submit_button:
            jackknife_regresion_lineal(X,y, test_s)
        else: 
            st.warning("Seleccione parametros de entrenamiento")
    with Regresgion_logistica_Jack:
        if submit_button:
            y = np.where(y >= Chance, 1, 0)    #Realizaremos un cambio en la variable de interes, debido a restricciones del modelo
            jackknife_regresion_logistica(X,y, test_s)
        else:
            st.warning("Seleccione parametros de entrenamiento")


with boots:
    Info_boots,Regresion_lineal_boots, Regresion_logistica_boots = st.tabs(["Informacion", 'Regresion Lineal', "Regresion Logistica"])
    with Info_boots:
        st.header("Bootstrap")
        booststrap_info()
    with Regresion_lineal_boots:


        # path=r'C:\Users\Francine Palacios\Desktop\Topicos Avanzados\Resampling\Data\Admission_Predict_Ver1.1.csv'
        df_data= pd.read_csv(path)
        df_data = df_data.drop(columns=['Serial No.'])
        st.subheader("Entrenamiento")

        with st.form(key='Boots_R_Lineal'):
            with st.expander("Parametros para el entrenamiento"):
                # test_s= st.slider("¿cuanto de dato de testeo?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)
                # Chance= st.slider("¿Para el entrenamiento (Regresion Logistica) cuanto considera que se acepta en 'Chance of Admit'?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)
                # XX=df_data[df_data.columns[:-1]]
                # y=df_data[df_data.columns[-1]]
                st.markdown("Para el entrenamiento, que columnas quiere utilizar")
                if st.checkbox("Todas aquellas columnas para bootstrap"):
                    columnas=df_data.columns
                    pass
                else:
                    columnas=st.multiselect('Seleccione las columnas que le interesan para el entrenamiento de Boostrap, importante siempre tiene que estar la variable de interes al final ', df_data.columns)

                numero = st.number_input('Ingresa el numero de Resamples', min_value=100, max_value=10000, step=100)
                submit_button_boots_R_lineal = st.form_submit_button('Entrenar')
            
        if submit_button_boots_R_lineal:
            bootstrap_regresion_lineal(df_data,columnas, numero)
        else:
            st.warning("Seleccione parametros de entrenamiento")
    with Regresion_logistica_boots:
            
        # path=r'C:\Users\Francine Palacios\Desktop\Topicos Avanzados\Resampling\Data\Admission_Predict_Ver1.1.csv'
        df_data= pd.read_csv(path)
        df_data = df_data.drop(columns=['Serial No.'])
        st.subheader("Entrenamiento")
        with st.form(key='Boots_R_Logistica'):
            with st.expander("Parametros para el entrenamiento"):
                # test_s= st.slider("¿cuanto de dato de testeo?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)
                Chance= st.slider("¿Para el entrenamiento (Regresion Logistica) cuanto considera que se aceptaa en 'Chance of Admit'?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)
                # XX=df_data[df_data.columns[:-1]]
                # y=df_data[df_data.columns[-1]]
                st.markdown("Para el entrenamiento, que columnas quiere utilizar")
                if st.checkbox("Todas aquellas columnas para bootstrapp"):
                    columnas=df_data.columns
                    pass
                else:
                    columnas=st.multiselect('Seleccione las columnas que le interesan para el entrenamientoo de Boostrap, importante siempre tiene que estar la variable de interes al final ', df_data.columns)

                numero = st.number_input('Ingresa el numero de Resampless', min_value=100, max_value=10000, step=100)
                submit_button_boots_R_logistica = st.form_submit_button('Entrenar')
            
        if submit_button_boots_R_logistica:
            bootstrap_regresion_logistica(df_data, Chance, columnas, numero)
        else:
            st.warning("Seleccione parametros para el entrenamiento")



