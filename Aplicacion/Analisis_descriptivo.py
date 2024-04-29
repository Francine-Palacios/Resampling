import streamlit as st
import plotly.express as px

def analisis_descriptivo(df_data):
        
    st.header('Datos')
    st.dataframe(df_data)
    ##################################################################################################
    ############## Tabla con alguans estadisticas de utilidad ################################################
    ##################################################################################################

    st.header('Estadisticas de utilidad')
    st.table(df_data.describe())

    #########################################################################
    #########################################################################
    #########################################################################

    #########################################################################
    ############## Graficos #################################################
    #########################################################################
    st.sidebar.header("Parametros para plot de datos")

    with st.sidebar.expander("Parametros para el histograma "):
        st.markdown("Para los graficos de histograma, seleccione las columnas de interes")
        if st.checkbox("Todas las columnas"):
            feature_names = df_data.columns
            pass
        else:
            columnas=st.multiselect('Seleccione las columnas que le interesan para los graficos ', df_data.columns)
            feature_names=columnas

        Chance= st.slider("¿Cuanto considera que se acepta en 'Chance of Admit'?", min_value=0.1,max_value=1.0,value=0.5, step=0.1)

    ############ Grafico de Histograma ############


    st.subheader("Histogramas")

    if len(feature_names)== 0:
        st.warning("Seleccione parametros al lado izquierdo de su pantalla")

    for name in feature_names:
        fig = px.histogram(df_data, x=name, marginal="rug", color_discrete_sequence=['indianred'])
        st.plotly_chart(fig, use_container_width=True)


    ########## Grafico de Histograma, diferenciando entre la variable objetivo #####
    st.subheader("Histograma diferenciado la variable objetivo")
    if len(feature_names)== 0:
        st.warning("Seleccione parametros al lado izquierdo de su pantalla")

    df_aux = df_data.copy()

    df_aux['Admit Category'] = df_aux['Chance of Admit '].apply(lambda x: f'Chance > {Chance}' if x > Chance else f'Chance <= {Chance}')

    # feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit ']

    for name in feature_names:
        fig = px.histogram(df_aux, x=name, color='Admit Category', barmode='overlay',
                        color_discrete_map={f'Chance > {Chance}': 'blue', f'Chance <= {Chance}': 'red'})
        st.plotly_chart(fig, use_container_width=True)




    ######## Boxplot ###########

    with st.sidebar.expander("Parametros para los graficos de Boxplot"):
        st.markdown("Para los graficos de Boxplot, seleccione las columnas de interes")
        if st.checkbox("Todas"):
            column_names = df_data.columns
            pass
        else:
            columnas=st.multiselect('Seleccione las columnas que le interesan para el boxplot ', df_data.columns)
            column_names=columnas
    st.subheader("Boxplot de los datos")
    if len(column_names)== 0:
        st.warning("Seleccione parametros al lado izquierdo de su pantalla")


    for name in column_names:
        fig = px.box(df_data, y=name, color_discrete_sequence=['goldenrod'])
        fig.update_layout(title=name, yaxis_title="Valor")
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("Correlacion")


    fig = px.scatter_matrix(df_data,
                            dimensions=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'],
                            color='Chance of Admit ',
                            title="Matriz de Dispersión: Correlaciones entre Variables")

    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=600, width=900, title_font_size=16)

    st.plotly_chart(fig)


    st.subheader("Matriz de correlacion")

    correlation_matrix = df_data.corr()


    fig = px.imshow(correlation_matrix,
                    labels=dict(x="Variables", y="Variables", color="Correlación"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale="Viridis",
                    text_auto=True,
                    width=3000)

    fig.update_layout(title="Matriz de Correlación",
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    xaxis_title="",
                    yaxis_title="",
                    height=600,
                    width=900)

    st.plotly_chart(fig)

    return None