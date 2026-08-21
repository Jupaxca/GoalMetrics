import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# CONFIGURACIÓN DE LA PÁGINA (Modo ancho para mejor distribución)
st.set_page_config(page_title="Simulador Pro - Fútbol", page_icon="⚽", layout="wide")

# ESTILOS CSS PERSONALIZADOS PARA UN LOOK MÁS PROFESIONAL
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #10b981;
        color: white;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Simulador Táctico & Predictivo de Fútbol")
st.markdown("---")

# 1. CARGAR DATOS DESDE TU GOOGLE SHEETS
@st.cache_data
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=['Equipo', 'Fecha', 'Condición', 'Nivel Rival'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Equipo'] = df['Equipo'].astype(str).str.strip()
    df['Condición'] = df['Condición'].astype(str).str.strip()
    df['Nivel Rival'] = df['Nivel Rival'].astype(str).str.strip()
    return df

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"⚠️ Error al cargar los datos del Google Sheets. Detalle: {e}")
    st.stop()

# 2. PANEL DE CONTROL LATERAL
st.sidebar.header("⚙️ Configuración de Análisis")

lista_equipos = sorted([str(x) for x in df['Equipo'].unique() if pd.notna(x)])
lista_niveles = sorted([str(x) for x in df['Nivel Rival'].unique() if pd.notna(x)])

equipo_seleccionado = st.sidebar.selectbox("🏟️ Selecciona el Equipo", lista_equipos)
condicion_seleccionada = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
nivel_seleccionado = st.sidebar.selectbox("⭐ Nivel del Rival", lista_niveles)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Líneas de Apuesta / Estudio")
linea_goles = st.sidebar.slider("⚽ Línea de Goles", 0.5, 3.5, 1.5, 0.5)
linea_tiros = st.sidebar.slider("👟 Línea de Tiros Totales", 5.0, 25.0, 12.5, 0.5)
linea_tiros_puerta = st.sidebar.slider("🎯 Línea Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
linea_corners = st.sidebar.slider("🚩 Línea de Córners", 1.0, 15.0, 5.5, 0.5)
linea_faltas = st.sidebar.slider("🛑 Línea de Faltas", 5.0, 25.0, 10.5, 0.5)

# 3. BOTÓN DE EJECUCIÓN PRINCIPAL
st.markdown("### 🕹️ Panel de Simulación")
if st.button("🚀 Ejecutar Análisis Estadístico (10,000 Simulaciones)", type="primary"):
    
    historial = df[(df['Equipo'] == equipo_seleccionado) & 
                   (df['Condición'] == condicion_seleccionada) & 
                   (df['Nivel Rival'] == nivel_seleccionado)]
    
    # Encabezado visual atractivo del análisis actual
    st.markdown(f"""
        ### 📊 Reporte para: <span style='color:#10b981;'>{equipo_seleccionado.upper()}</span> 
        *(Jugando como **{condicion_seleccionada.upper()}** ante un rival nivel **{nivel_seleccionado.upper()}**)*
    """, unsafe_allow_html=True)
    
    if len(historial) < 2:
        st.warning(f"⚠️ Tienes solo {len(historial)} partido(s) registrado(s) para este escenario. Se necesitan al menos 2 para tener precisión.")
    else:
        # Cálculo Ponderado por Recencia
        historial['Dias_Pasados'] = (pd.Timestamp.now() - historial['Fecha']).dt.days.replace(0, 0.1)
        historial['Peso'] = 1 / (1 + (historial['Dias_Pasados'] / 30))
        
        def weighted_avg(col):
            return np.average(historial[col], weights=historial['Peso'])
        
        lambda_favor = weighted_avg('Goles')
        lambda_contra = weighted_avg('Goles Rival')
        
        col_tiros = 'Tiros Prom' if 'Tiros Prom' in historial.columns else 'Tiros'
        col_puerta = 'A Puerta Prom' if 'A Puerta Prom' in historial.columns else 'A Puerta'
        
        num_sim = 10000
        sim_goles_favor = np.random.poisson(lam=lambda_favor, size=num_sim)
        sim_goles_contra = np.random.poisson(lam=lambda_contra, size=num_sim)
        sim_tiros = np.random.poisson(lam=weighted_avg(col_tiros), size=num_sim)
        sim_tiros_puerta = np.random.poisson(lam=weighted_avg(col_puerta), size=num_sim)
        sim_corners = np.random.poisson(lam=weighted_avg('Corners'), size=num_sim)
        sim_faltas = np.random.poisson(lam=weighted_avg('Faltas'), size=num_sim)
        
        marcadores = [f"{f}-{c}" for f, c in zip(sim_goles_favor, sim_goles_contra)]
        conteo = Counter(marcadores)
        
        st.markdown("---")
        
        # DISTRIBUCIÓN EN DOS COLUMNAS LIMPIAS
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 🏆 Top 5 Marcadores Más Probables")
            # Creamos una mini tabla visual bonita
            tabla_data = []
            for res, freq in conteo.most_common(5):
                prob = (freq / num_sim) * 100
                tabla_data.append({"Marcador (Favor - Contra)": res, "Probabilidad": f"{prob:.1f}%"})
            st.dataframe(pd.DataFrame(tabla_data), hide_index=True, use_container_width=True)
                
        with col_right:
            st.markdown("#### 📈 Probabilidades de Líneas")
            st.metric(label=f"⚽ Más de {linea_goles} Goles", value=f"{(sim_goles_favor > linea_goles).mean() * 100:.1f}%")
            st.metric(label=f"👟 Más de {linea_tiros} Tiros Totales", value=f"{(sim_tiros > linea_tiros).mean() * 100:.1f}%")
            st.metric(label=f"🎯 Más de {linea_tiros_puerta} Tiros a Puerta", value=f"{(sim_tiros_puerta > linea_tiros_puerta).mean() * 100:.1f}%")
            st.metric(label=f"🚩 Más de {linea_corners} Córners", value=f"{(sim_corners > linea_corners).mean() * 100:.1f}%")
            st.metric(label=f"🛑 Más de {linea_faltas} Faltas", value=f"{(sim_faltas > linea_faltas).mean() * 100:.1f}%")

        st.markdown("---")
        st.info(f"💡 **Partidos analizados en el historial:** {len(historial)} encuentros ponderados por fecha (dando mayor peso a los más recientes).")
