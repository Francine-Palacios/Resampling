import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  
from sklearn.linear_model import LinearRegression
from scipy.stats import bootstrap
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 

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

    clf = LinearRegression(
    fit_intercept=True,# Incluir el término de sesgo (intercept)
    )
    clf.fit( X_train,y_train)
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

def texto_introduccion_IC():
    st.subheader("Intervalos de confianza")
    st.markdown(
        """
    En la biblioteca scipy.stats.bootstrap, hay varios métodos para calcular estos intervalos:

    - Percentil: Este método utiliza los percentiles de la distribución de remuestreo para definir el intervalo de confianza. Por ejemplo, para un intervalo de confianza del 95%, se usarían los percentiles 2.5 y 97.5.
    - BCa (Bias-Corrected and Accelerated): Este método ajusta el intervalo de confianza para corregir el sesgo y la aceleración en la distribución de remuestreo. Es más complejo que el método del percentil y generalmente proporciona mejores resultados cuando la distribución del estimador no es simétrica.
    - Básico (Reverse Percentile): Similar al método del percentil, pero invierte los roles de los percentiles. Por ejemplo, si normalmente tomaríamos los percentiles 2.5 y 97.5, aquí tomaríamos los percentiles 97.5 y 2.5 de la distribución de remuestreo.

    En general, se tiene las siguientes indicaciones: 

    - Percentil: Este método es simple y directo, y es útil cuando tienes una muestra grande y la distribución del estimador es aproximadamente simétrica. No ajusta por sesgo ni por la forma de la distribución de remuestreo.
    - BCa (Bias-Corrected and Accelerated): Es preferible cuando la distribución del estimador no es simétrica. También es útil cuando se sospecha que hay sesgo en los estimadores o cuando la distribución tiene colas pesadas. Es un método más robusto que el del percentil.
    - Básico (Reverse Percentil): Este método puede ser útil cuando el método del percentil no es adecuado debido a la asimetría en la distribución de remuestreo. Es una alternativa al método del percentil que puede ofrecer una mejor cobertura del intervalo de confianza en ciertas situaciones.
    En general, el método BCa suele ser la opción más segura y robusta, especialmente en muestras pequeñas o cuando la distribución del estimador es desconocida o complicada. Sin embargo, requiere más cálculos que el método del percentil. El método básico es menos común pero puede ser útil en casos específicos donde los otros métodos no funcionan bien.

    En adelante se construye un intervalo de confianza bilateral, con nivel de confianza $1-\\alpha$, para un parámetro $\\theta$ de la distribución $F$.
    Una vez elegido el método bootstrap adecuado, teniendo en cuenta la información disponible en el contexto del que se trate, otro aspecto importante es el método para la construcción del intervalo de confianza bootstrap de forma que la probabilidad de cobertura sea lo más parecida posible al nivel nominal $1-\\alpha$.

    Para un tratamiento más detallado, incluyendo los órdenes de los errores de cobertura, ver por ejemplo el [Capítulo 4](https://rubenfcasal.github.io/book_remuestreo/icboot.html) de Cao y Fernández-Casal (2021) o el Capítulo 5 de Davison y Hinkley (1997).
    """
    )


def texto_IC_normal():

    st.markdown(""" 
                Este método emplea las aproximaciones bootstrap del sesgo $$Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)$$ y de la varianza $$Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)$$, y asume que la distribución del correspondiente estadístico studentizado es una normal estándar
                $$\\frac{\\hat{\\theta} - Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right) - \\theta}{\\sqrt{Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)}} \\underset{aprox}{\\sim }\\mathcal{N}\\left( 0, 1 \\right).$$
                De esta forma se obtiene la estimación por intervalo de confianza:
                $$\\hat{I}_{norm}=\\left( \\hat{\\theta} - Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right) - z_{1-\\alpha /2}\\sqrt{Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)},\\hat{\\theta} - Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right) + z_{1 - \\alpha /2}\\sqrt{Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)} \\right).$$


                """)

    
def texto_IC_percentil_directo():

    st.markdown("""

        Este método se basa en la construcción del intervalo de confianza, mediante bootstrap, empleando como estadístico el estimador $$R = \\hat{\\theta}.$$

        Una vez elegido el método de remuestreo, empleando un estimador, $$\\hat{F}$$, de la
        distribución poblacional, $$F$$ , la distribución en el muestreo de $$R = \\hat{\\theta}$$ se aproxima directamente mediante la distribución bootstrap de $$R^{\\ast}= \\hat{\\theta}^{\\ast}$$.
        A partir de las réplicas bootstrap del estimador aproximamos los cuantiles $$x_{\\alpha /2}$$ y $$x_{1-\\alpha /2}$$ (denotando por $$x_{\\beta }$$ el valor verificando $$P^{\\ast}\\left( R^{\\ast }\\leq x_{\\beta } \\right) =\\beta$$), de forma que 
        
      $$
        1-\\alpha = 1-\\frac{\\alpha}{2}-\\frac{\\alpha}{2} = P^{\\ast}(\\hat{\\theta}^{\\ast}<x_{1-\\alpha/2}) - P^{\\ast}(\\hat{\\theta}^{\\ast}\\leq x_{\\alpha/2})
        $$
        $$
        = P^{\\ast}(x_{\\alpha/2}<\\hat{\\theta}^{\\ast}<x_{1-\\alpha/2})
        $$

                
        y asumimos que esto aproxima lo que ocurre con la distribución poblacional
        $$P\\left( x_{\\alpha /2} < \\hat{\\theta} < x_{1-\\alpha /2} \\right) \\approx 1-\\alpha.$$
        De donde se obtiene el intervalo de confianza bootstrap calculado 
        por el método percentil directo
        $$\\hat{I}_{perc}=\\left( x_{\\alpha /2}, x_{1-\\alpha /2}  \\right).$$

        Una ventaja de los intervalos construidos con este método es que son invariantes frente a transformaciones del estimador (en el caso de que fuese más adecuado trabajar en otra escala, no sería necesario conocer la transformación).
        Sin embargo, la precisión puede verse seriamente afectada en el caso de estimadores sesgados.

                """)

    
def texto_IC_percentil_basico():

    st.markdown("""

        En este método se emplea como estadístico el estimador centrado (no estandarizado)
        $$R = \\hat{\\theta}-\\theta.$$
        De forma análoga, la distribución en el muestreo de $$R$$ se aproxima mediante la distribución bootstrap de
        $$R^{\\ast}= \\hat{\\theta}^{\\ast}-\\theta \\left( \\hat{F} \\right) = \\hat{\\theta}^{\\ast}-\\hat{\\theta}.$$
        A partir de las réplicas bootstrap del estadístico se aproximan los cuantiles $$x_{\\alpha /2}$$ y $$x_{1-\\alpha /2}$$ tales que
        $$1-\\alpha = P^{\\ast}\\left( x_{\\alpha /2}<R^{\\ast}<x_{1-\\alpha /2} \\right),$$
        tomándolo como aproximación de lo que ocurre con la distribución poblacional
                
      $$
        1-\\alpha \\approx P(x_{\\alpha /2}<R<x_{1-\\alpha /2})
        $$
        $$
        = P(x_{\\alpha /2} < \\hat{\\theta}-\\theta < x_{1-\\alpha /2})
        $$
        $$
        = P(\\hat{\\theta} - x_{1-\\alpha /2} < \\theta <\\hat{\\theta} -x_{\\alpha /2})
        $$

                
        De donde se obtiene el intervalo de confianza bootstrap calculado 
        por el método percentil básico
        $$\\hat{I}_{basic}=\\left( \\hat{\\theta} - x_{1-\\alpha /2},\\hat{\\theta} - x_{\\alpha /2} \\right).$$


                """)


def texto_IC_BCa():

    st.markdown("""

        El método $$BCa$$ (bias-corrected and accelerated) propuesto por Efron (1987) considera una transformación de forma que la distribución se aproxime a la normalidad, construye el intervalo en esa escala asumiendo normalidad y transforma el resultado a la escala original empleando la distribución bootstrap.
        El intervalo obtenido es de la forma:
        $$\\hat{I}_{perc}=\\left( x_{\\alpha /2}, x_{1-\\alpha /2}  \\right),$$
        donde $$x_u = \\hat G^{-1}\\left(\\Phi\\left(z + \\frac{z + z_u}{1-a(z+z_u)}\\right)  \\right),$$
        siendo $$\\hat G$$ la distribución empírica de $$\\hat{\\theta}^{\\ast}$$, $$\\Phi(z)$$ la función de distribución de la normal estándar, $$z_u = \\Phi^{-1}(u)$$ el correspondiente cuantil y:

        - $$z = \\Phi^{-1}(\\hat G(\\hat\\theta))$$ un factor de corrección de sesgo.
        - $$a$$ la denominada constante aceleradora (o corrección de asimetría), que suele ser aproximada mediante jackknife.

        Para más detalles ver Sección 5.3.2 de Davison y Hinkley (1997).

                """)

def Bibliografia():
    bibliografia_md = """
    ### Bibliografía

    - Rubén Fernández-Casal, "Intervalos de confianza bootstrap", disponible en [Simbook](https://rubenfcasal.github.io/simbook/boot-ic.html#boot-ic-perc).
    - Ethan Wicker, "Bootstrap Resampling", publicado el 23 de febrero de 2021, disponible en [Ethan Wicker's Blog](https://ethanwicker.com/2021-02-23-bootstrap-resampling-001/).
    - Documentación de SciPy, "scipy.stats.bootstrap", disponible en [SciPy Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
    - B. Efron y R. J. Tibshirani, "An Introduction to the Bootstrap", Chapman & Hall/CRC, Boca Raton, FL, USA, 1993.
    - Nathaniel E. Helwig, "Bootstrap Confidence Intervals", disponible en [University of Minnesota](http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf).
    """

    st.markdown(bibliografia_md, unsafe_allow_html=True)


    
def bootstrap_regresion_lineal():
    st.subheader("Bootstrap")

    st.markdown("""
Bootstrap es una técnica estadística de remuestreo que permite estimar la distribución de una muestra. Se utiliza para calcular intervalos de confianza y la variabilidad de un estimador, como la media o la mediana, sin necesidad de suposiciones sobre la distribución de la población. Funciona generando múltiples muestras simuladas (remuestreos) de los datos observados, con reemplazo, para crear una “distribución bootstrap” de la estadística de interés. Esto permite evaluar la precisión y estabilidad de las estimaciones realizadas a partir de la muestra original
           
                """)
    


    ########################################################################################################
    ############################ Lectura de archivo y parametros ###########################################
    ########################################################################################################

    path=r'C:\Users\Francine Palacios\Desktop\Topicos Avanzados\Resampling\Data\Admission_Predict_Ver1.1.csv'
    df_data= pd.read_csv(path)
    df_data = df_data.drop(columns=['Serial No.'])
    st.subheader("Entrenamiento")
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
    if len(columnas)==0:
        st.warning("Seleccione parametros")
        st.stop()

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    
    ########################################################################################################
    ####################################### Regresion lineal ##########################################
    ########################################################################################################



    df = df_data[columnas]
    X=df[df.columns[:-1]]
    y=df[df.columns[-1]]
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    clf = LinearRegression(
    fit_intercept=True,# Incluir el término de sesgo (intercept)
    )
    clf.fit( X_train,y_train)
    r_cuadrado = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    st.subheader("R cuadrado")
    r_cuadrado_md = """
    $$ R^2 = 1 - \\frac{{SS_{{res}}}}{{SS_{{tot}}}} $$
    donde:
    - \( SS_{{res}} \) es la suma de cuadrados de los residuos.
    - \( SS_{{tot}} \) es la suma total de cuadrados.

    El valor de R^2 para el modelo es: **{r_cuadrado:.3f}**
    """

    # En Streamlit, puedes mostrar la fórmula y el valor de R^2 usando el siguiente comando:
    st.markdown(r_cuadrado_md.format(r_cuadrado=r_cuadrado), unsafe_allow_html=True)




    residuos = y_test - y_pred

    qq = sm.ProbPlot(residuos, fit= True)
    qq_theoretical = qq.theoretical_quantiles
    qq_sample = qq.sample_quantiles

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq_theoretical, y=qq_sample, mode='markers', name='Residuos'))
    fig.add_trace(go.Scatter(x=qq_theoretical, y=qq_theoretical, mode='lines', name='Línea de referencia'))

    fig.update_layout(title='QQ Plot de los Residuos',
                    xaxis_title='Cuantiles Teóricos',
                    yaxis_title='Cuantiles de la Muestra')

    st.plotly_chart(fig)

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################




    ########################################################################################################
    ########################################## Bootstrap ###################################################
    ########################################################################################################


    
    tupla_arrays = tuple(df_data[col].to_numpy() for col in columnas)  #Formato
    resultado_boots= bootstrap(tupla_arrays, estadistica_interes, vectorized=False, paired=True,
                n_resamples=numero, method='percentile', confidence_level=0.9)
    
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
    
    
    texto_introduccion_IC()

    #################### Percentil Directo #################
    st.subheader("Intervalo de confianza Percentil Directo")
    texto_IC_percentil_directo()
    st.dataframe(plot_intervalos_de_confianza(resultado_boots, tupla_arrays))

     ################# Percentil Basico #################################
    st.subheader("Intervalo de confianza basico")
    texto_IC_percentil_basico()
    resultado_boots_basico= bootstrap(tupla_arrays, estadistica_interes, vectorized=False, paired=True,
                bootstrap_result=resultado_boots, n_resamples=0, method='basic', confidence_level=0.9)
    st.dataframe(plot_intervalos_de_confianza(resultado_boots_basico, tupla_arrays))

    ################# BCa #################################
    st.subheader("Intervalo de confianza BCa")
    texto_IC_BCa()
    resultado_boots_bca= bootstrap(tupla_arrays, estadistica_interes, vectorized=False, paired=True,
                bootstrap_result=resultado_boots, n_resamples=0, method='BCa', confidence_level=0.9)
    st.dataframe(plot_intervalos_de_confianza(resultado_boots_bca, tupla_arrays))
    

    ##
    Bibliografia()