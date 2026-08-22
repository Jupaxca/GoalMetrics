import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análisis de Jugadores", layout="wide")

st.title("⚽ Dashboard de Jugadores - Análisis de Rendimiento")

# Carga automática desde tu Google Sheet
@st.cache_data
def cargar_datos():
    url = "https://docs.google.com/spreadsheets/d/1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4/export?format=csv"
    return pd.read_csv(url) 

df = cargar_datos()

# Filtros laterales
st.sidebar.header("Filtros")
jugador_seleccionado = st.sidebar.selectbox("Selecciona un jugador", df['Equipo'].unique())
nivel_filtro = st.sidebar.multiselect("Nivel de rival", df['Nivel Rival'].unique(), default=df['Nivel Rival'].unique())

# Filtrar datos
df_filtrado = df[(df['Equipo'] == jugador_seleccionado) & (df['Nivel Rival'].isin(nivel_filtro))]

# KPIs (Los cuadritos de arriba)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Goles Totales", int(df_filtrado['Goles'].sum()))
col2.metric("Asistencias", int(df_filtrado['Asistencias'].sum()))
col3.metric("Tiros A Puerta", int(df_filtrado['A Puerta'].sum()))
col4.metric("Partidos Analizados", len(df_filtrado))

# Gráfica de rendimiento
st.subheader(f"Evolución y Desempeño de {jugador_seleccionado}")
fig = px.bar(df_filtrado, x='Rival', y=['Goles', 'Asistencias'], barmode='group', 
             title="Goles y Asistencias por Rival", color_discrete_sequence=['#FF4B4B', '#00CC96'])
st.plotly_chart(fig, use_container_width=True)

# Lógica de la racha y probabilidad
st.subheader("Análisis de Racha y Probabilidad")
total_partidos = len(df_filtrado)
contribuciones = df_filtrado['Goles'].sum() + df_filtrado['Asistencias'].sum()
racha = 0
for gol in reversed(df_filtrado['Goles'].values):
    if gol == 0: racha += 1
    else: break

st.write(f"**Racha actual:** {racha} partidos sin marcar ni asistir")
if racha >= 3:
    st.success("⚠️ ALERTA: Alta probabilidad de acción (viene de sequía y supera la media)")
else:
    st.info("Estado actual: Probabilidad dentro de la media")
