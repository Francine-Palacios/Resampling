import streamlit as st
from informacion import informacion
from Analisis_descriptivo import analisis_descriptivo
from Jackknife_Regresion_logistica import jackknife_regresion_logistica
from Jackknife_Regresion_lineal import jackknife_regresion_lineal
import pandas as pd

import numpy as np


########################################################################
############### Configuracion e informacion ############################
########################################################################


st.set_page_config(page_title='Jackknife', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)


_, _, col3 = st.columns([3,6,3])

with col3:
    url_imagen = "https://matematica.usm.cl/wp-content/themes/dmatUSM/assets/img/logoDMAT2.png"

    st.image(url_imagen, width=250)



_, col2, _ = st.columns([1,3,1])

with col2:
    st.title('Graduate Admission')


texto_descripcion = """
Ests aplicacion está dedicado al análisis exhaustivo de los parámetros obtenidos de un modelo de **regresión logística** y de **regresion lineal**. Se emplea el método de remuestreo Jackknife para estimar la precisión y la confiabilidad de los parámetros estimados. El conjunto de datos utilizado proviene de registros de admisión a programas de posgrado, el cual está disponible para el público por la plataforma [kaggle](https://www.kaggle.com/) 
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


#########################################################################
################## Barra lateral#########################################
#########################################################################


##################### Informacion en la side bar ########################
        
st.sidebar.title("Jackknife y Bootstrap")
st.sidebar.caption("Universidad Tecnica Federico Santa María")


#########################################################################

################### Subida de archivo de datos###################################
# uploaded_file = st.sidebar.file_uploader("Cargar el archivo de datos")

# if uploaded_file is None:
      # st.warning('Favor subir el archivo de datos')

InfoTab,Analisis,jackknif = st.tabs(["Información","Analisis Descriptivo", 'Jackknife'])


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
    
    if X.empty:
        st.warning("Seleccione parametros")
        st.stop()

    Regresion_lineal_Jack, Regresgion_logistica_Jack = st.tabs(["Regresion Lineal","Regresion Logistica"])
    with Regresion_lineal_Jack:
        jackknife_regresion_lineal(X,y, test_s)
    with Regresgion_logistica_Jack:
        y = np.where(y >= Chance, 1, 0)    #Realizaremos un cambio en la variable de interes, debido a restricciones del modelo
        jackknife_regresion_logistica(X,y, test_s)






