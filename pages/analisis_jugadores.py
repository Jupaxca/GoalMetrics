import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="GoalMetrics - Jugadores", layout="wide")

st.title("🎯 Centro de Análisis de Jugadores")
st.markdown("Filtra el rendimiento individual por umbrales de tiros, goles y nivel del rival.")
st.markdown("---")

@st.cache_data
def cargar_datos():
    url = "https://docs.google.com/spreadsheets/d/1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4/export?format=csv"
    return pd.read_csv(url)

df = cargar_datos()

# Limpiar espacios en nombres de columnas por si acaso
df.columns = df.columns.str.strip()

# Sidebar con controles interactivos estilo motor de simulacion
st.sidebar.header("⚙️ Filtros y Umbrales")

# Selector de Jugador
jugador_seleccionado = st.sidebar.selectbox("Selecciona al Jugador", df['Equipo'].unique())

# Selector de Nivel de Rival
niveles_disponibles = df['Nivel Rival'].unique()
nivel_seleccionado = st.sidebar.multiselect("Nivel del Rival", niveles_disponibles, default=niveles_disponibles)

st.sidebar.markdown("---")
st.sidebar.subheader("Sliders de Evaluación")

# Sliders interactivos para definir qué quieres evaluar (umbrales)
min_goles = st.sidebar.slider("Mínimo de Goles en el partido", 0, int(df['Goles'].max()) if df['Goles'].max() > 0 else 3, 0)
min_tiros_puerta = st.sidebar.slider("Mínimo de Tiros a Puerta", 0, int(df['A Puerta'].max()) if df['A Puerta'].max() > 0 else 5, 0)

# Filtrar el DataFrame según los controles de la barra lateral
df_filtrado = df[
    (df['Equipo'] == jugador_seleccionado) & 
    (df['Nivel Rival'].isin(nivel_seleccionado)) &
    (df['Goles'] >= min_goles) &
    (df['A Puerta'] >= min_tiros_puerta)
]

# Panel Principal - Métricas (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Goles Registrados", int(df_filtrado['Goles'].sum()))
col2.metric("Asistencias", int(df_filtrado['Asistencias'].sum()))
col3.metric("Tiros a Puerta (Promedio)", f"{df_filtrado['A Puerta'].mean():.1f}" if len(df_filtrado) > 0 else "0")
col4.metric("Partidos Cumpliendo Filtro", len(df_filtrado))

st.markdown("---")

if len(df_filtrado) > 0:
    st.subheader(f"📊 Desempeño Detallado: {jugador_seleccionado}")
    
    # Gráfica interactiva de rendimiento por partido/rival
    fig = px.bar(
        df_filtrado, 
        x='Rival', 
        y=['Goles', 'Asistencias', 'A Puerta'], 
        barmode='group',
        title="Comparativa de Goles, Asistencias y Tiros a Puerta por Encuentro",
        color_discrete_sequence=['#FF4B4B', '#00CC96', '#636EFA']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Motor de análisis de racha integrado
    st.subheader("⚡ Estado de Racha y Probabilidad para el Siguiente Partido")
- total_p = len(df_filtrado)
    goles_tot = df_filtrado['Goles'].sum()
    asist_tot = df_filtrado['Asistencias'].sum()
    contribuciones = goles_tot + asist_tot
    
    if contribuciones > 0:
        ratio = total_p / contribuciones
        st.info(f"📈 En promedio, este jugador genera una contribución de gol/asistencia cada **{ratio:.1f} partidos** bajo estos filtros.")
    else:
        st.warning("⚠️ No hay suficientes contribuciones registradas con los filtros actuales.")
else:
    st.warning("⚠️ No se encontraron partidos que cumplan estrictamente con los umbrales seleccionados en los sliders. Intenta relajar los filtros en la barra lateral.")
