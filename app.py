import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página principal
st.set_page_config(page_title="GoalMetrics - Equipos", layout="wide")

# Título y encabezado principal
st.title("⚽ GoalMetrics FOOTBALL ANALYTICS")
st.markdown("Plataforma avanzada de simulación estadística y predicción de rendimiento deportivo.")
st.markdown("---")

# Carga automática de datos desde tu Google Sheet (puedes ajustar el link si es otra hoja o la misma)
@st.cache_data
def cargar_datos_equipos():
    url = "https://docs.google.com/spreadsheets/d/1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4/export?format=csv"
    return pd.read_csv(url)

# Intentar cargar los datos para alimentar los selectores de la barra lateral
try:
    df_equipos = cargar_datos_equipos()
    df_equipos.columns = df_equipos.columns.str.strip()
    datos_cargados = True
except Exception as e:
    datos_cargados = False

# Sidebar - Configuración de Análisis
st.sidebar.header("⚙️ Configuración de Análisis")

if datos_cargados and not df_equipos.empty:
    # Poblar los selectores dinámicamente con los datos reales de tu archivo
    equipos_lista = df_equipos['Equipo'].unique().tolist()
    equipo_seleccionado = st.sidebar.selectbox("🏟️ Selecciona el Equipo", equipos_lista)

    condiciones_lista = df_equipos['Condición'].unique().tolist() if 'Condición' in df_equipos.columns else ["Local", "Visitante"]
    condicion = st.sidebar.selectbox("📍 Condición", condiciones_lista)

    niveles_lista = df_equipos['Nivel Rival'].unique().tolist() if 'Nivel Rival' in df_equipos.columns else ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
    nivel_rival = st.sidebar.selectbox("⭐ Torneo / Nivel del Rival", niveles_lista)
else:
    # Opciones por defecto si ocurre algún error de lectura
    equipo_seleccionado = st.sidebar.selectbox("🏟️ Selecciona el Equipo", ["Arsenal", "Benfica"])
    condicion = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
    nivel_rival = st.sidebar.selectbox("⭐ Torneo / Nivel del Rival", ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"])

st.sidebar.markdown("---")

# Botones de acción principales en la interfaz
col_btn1, col_btn2, col3 = st.columns([1, 1, 2])
with col_btn1:
    analizar = st.button("⚡ Analizar", type="primary")
with col_btn2:
    limpiar = st.button("🧹 Limpiar")

# Lógica al presionar Analizar
if analizar:
    st.markdown("---")
    st.subheader(f"📊 Resultados de Simulación: {equipo_seleccionado} ({condicion}) vs Nivel {nivel_rival}")
    
    # Métricas de rendimiento
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de Victoria", "68.5%")
    m2.metric("Goles Esperados (xG)", "1.92")
    m3.metric("Tiros a Puerta Promedio", "6.4")
    m4.metric("Nivel de Confianza", "94%")
    
    # Gráfica de simulación interactiva
    data_ejemplo = pd.DataFrame({
        "Indicador": ["Efectividad de Tiros", "Control de Partido", "Presión Alta", "Conversión"],
        "Porcentaje (%)": [72, 65, 80, 55]
    })
    
    fig = px.bar(
        data_ejemplo, 
        x="Indicador", 
        y="Porcentaje (%)", 
        title="Desempeño Proyectado por el Motor Estadístico",
        color="Indicador",
        color_discrete_sequence=['#FF4B4B', '#00CC96', '#636EFA', '#FFA15A']
    )
    st.plotly_chart(fig, use_container_width=True)

elif limpiar:
    st.info("🧹 Los filtros y resultados han sido restablecidos. Selecciona los parámetros en la barra lateral y presiona Analizar.")
else:
    st.info("⚠️ Configura los parámetros en la barra lateral y presiona **Analizar** para ejecutar la simulación del equipo.")
