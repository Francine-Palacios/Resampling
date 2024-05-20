import streamlit as st
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analisis_descriptivo(df_data):

    #############################################################################
    ############################### Datos #######################################
    #############################################################################

    st.header('Datos')
    st.dataframe(df_data,use_container_width=True)

    #############################################################################
    #############################################################################
    #############################################################################


    ##################################################################################################
    ############## Tabla con alguans estadisticas de utilidad ################################################
    ##################################################################################################

    st.header('Estadisticas de utilidad')
    st.table(df_data.describe())

    #########################################################################
    #########################################################################
    #########################################################################

    #########################################################################
    ######################### Grafico de Histograma #########################
    #########################################################################

    st.subheader("Histograma de los datos")

    feature_names = df_data.select_dtypes(include=['number']).columns

    fig = make_subplots(rows=2, cols=4, subplot_titles=feature_names)

    for index, name in enumerate(feature_names):
        row = index // 4 + 1
        col = index % 4 + 1
        fig.add_trace(
            go.Histogram(
                x=df_data[name], 
                name=name,
                marker=dict(
                    line=dict(
                        color='black',  # Color del borde
                        width=1         # Ancho del borde
                    )
                ),
            ),
            row=row, col=col
        )

    fig.update_layout(height=600, width=800, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    #########################################################################
    #########################################################################
    #########################################################################


    #########################################################################
    ##################### Histograma diferenciado ###########################
    #########################################################################
    st.subheader("Histograma diferenciado por 'Chance of Admit'")
    Chance= st.slider("Chance of Admit", min_value=0.1,max_value=1.0,value=0.5, step=0.1)

    df_aux=df_data.copy()
    df_aux['Admit Category'] = df_aux['Chance of Admit '].apply(lambda x: f'Chance > {Chance}' if x > Chance else f'Chance <= {Chance}')

    feature_names = df_aux.select_dtypes(include=['number']).columns

    fig = make_subplots(rows=2, cols=4, subplot_titles=feature_names)

    for index, name in enumerate(feature_names):
        row = index // 4 + 1
        col = index % 4 + 1
        fig.add_trace(
            go.Histogram(x=df_aux[df_aux['Admit Category'] == f'Chance > {Chance}'][name], 
                        name=f'Chance > {Chance}', 
                        marker=dict(color='blue', line=dict(color='black', width=1)),
                        opacity=0.6),  
            row=row, col=col
        )
        fig.add_trace(
            go.Histogram(x=df_aux[df_aux['Admit Category'] == f'Chance <= {Chance}'][name], 
                        name=f'Chance <= {Chance}', 
                        marker=dict(color='red', line=dict(color='black', width=1)),
                        opacity=0.6),  
            row=row, col=col
        )

    fig.update_layout(barmode='overlay', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


    #########################################################################
    #########################################################################
    #########################################################################


    #############################################################################
    ####################### Boxplot #############################################
    #############################################################################
    st.subheader("Boxplot")

    columnas_numericas = df_data.select_dtypes(include=['number']).columns

    fig = make_subplots(rows=2, cols=4, subplot_titles=columnas_numericas)

    for index, columna in enumerate(columnas_numericas):
        row = index // 4 + 1
        col = index % 4 + 1
        fig.add_trace(
            go.Box(y=df_data[columna], name=columna),
            row=row, col=col
        )

    fig.update_layout(height=600, width=800, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    #############################################################################
    #############################################################################
    #############################################################################

    #############################################################################
    ############################# Correlacion ###################################
    #############################################################################

    st.subheader("Dispersión")


    fig = px.scatter_matrix(df_data,
                            dimensions=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'],
                            color='Chance of Admit ',
                            title="Matriz de Dispersión: Correlaciones entre Variables")

    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=600, width=1200, title_font_size=16)

    st.plotly_chart(fig)

    #############################################################################
    #############################################################################
    #############################################################################


    #############################################################################
    ########################## Correlacion ################################
    #############################################################################


    st.subheader("Matriz de correlacion")

    correlation_matrix = df_data.corr()


    fig = px.imshow(correlation_matrix,
                    labels=dict(x="Variables", y="Variables", color="Correlación"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale="Viridis",
                    text_auto=False,
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