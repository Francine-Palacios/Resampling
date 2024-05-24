import streamlit as st



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
    return None

def texto_IC_normal():

    st.markdown(""" 
                Este método emplea las aproximaciones bootstrap del sesgo $$Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)$$ y de la varianza $$Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)$$, y asume que la distribución del correspondiente estadístico studentizado es una normal estándar
                $$\\frac{\\hat{\\theta} - Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right) - \\theta}{\\sqrt{Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)}} \\underset{aprox}{\\sim }\\mathcal{N}\\left( 0, 1 \\right).$$
                De esta forma se obtiene la estimación por intervalo de confianza:
                $$\\hat{I}_{norm}=\\left( \\hat{\\theta} - Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right) - z_{1-\\alpha /2}\\sqrt{Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)},\\hat{\\theta} - Sesgo^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right) + z_{1 - \\alpha /2}\\sqrt{Var^{\\ast}\\left( \\hat{\\theta}^{\\ast} \\right)} \\right).$$


                """)

    return None 
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

    return None
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
    return None 

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
    return None 
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

    return None 


def booststrap_info():
    st.markdown("""
Bootstrap es una técnica estadística de remuestreo que permite estimar la distribución de una muestra. Se utiliza para calcular intervalos de confianza y la variabilidad de un estimador, como la media o la mediana, sin necesidad de suposiciones sobre la distribución de la población. Funciona generando múltiples muestras simuladas (remuestreos) de los datos observados, con reemplazo, para crear una “distribución bootstrap” de la estadística de interés. Esto permite evaluar la precisión y estabilidad de las estimaciones realizadas a partir de la muestra original
           
                """)
    
    
    texto_introduccion_IC()
    st.subheader("Intervalo de Confianza: Percentil Directo")

    texto_IC_percentil_directo()
    st.subheader("Intervalo de Confianza: Percentil Basico")

    texto_IC_percentil_basico()
    st.subheader("Intervalo de Confianza: BCa")

    texto_IC_BCa()

    Bibliografia()

    return None