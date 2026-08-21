import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(page_title="Simulador de Fútbol", page_icon="⚽", layout="centered")

st.title("⚽ Simulador de Predicciones de Fútbol")
st.markdown("---")

# 1. CARGAR DATOS DESDE TU GOOGLE SHEETS
@st.cache_data
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    # Limpieza de columnas y filas vacías
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

# 2. PANEL DE CONTROL LATERAL (MENÚS VISUALES)
st.sidebar.header("⚙️ Panel de Control")

# Limpiar lista de equipos y niveles para evitar nulos
lista_equipos = sorted([str(x) for x in df['Equipo'].unique() if pd.notna(x)])
lista_niveles = sorted([str(x) for x in df['Nivel Rival'].unique() if pd.notna(x)])

equipo_seleccionado = st.sidebar.selectbox("Selecciona el Equipo", lista_equipos)
condicion_seleccionada = st.sidebar.selectbox("Condición", ["Local", "Visitante"])
nivel_seleccionado = st.sidebar.selectbox("Nivel del Rival", lista_niveles)

st.sidebar.subheader("Líneas de Análisis")
linea_goles = st.sidebar.slider("Línea de Goles", 0.5, 3.5, 1.5, 0.5)
linea_tiros = st.sidebar.slider("Línea de Tiros Totales", 5.0, 25.0, 12.5, 0.5)
linea_tiros_puerta = st.sidebar.slider("Línea Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
linea_corners = st.sidebar.slider("Línea de Córners", 1.0, 15.0, 5.5, 0.5)
linea_faltas = st.sidebar.slider("Línea de Faltas", 5.0, 25.0, 10.5, 0.5)

# 3. BOTÓN DE EJECUCIÓN
if st.button("🚀 Ejecutar Simulación", type="primary"):
    
    historial = df[(df['Equipo'] == equipo_seleccionado) & 
                   (df['Condición'] == condicion_seleccionada) & 
                   (df['Nivel Rival'] == nivel_seleccionado)]
    
    st.subheader(f"📊 Analizando: {equipo_seleccionado.upper()} ({condicion_seleccionada.upper()} vs {nivel_seleccionado.upper()})")
    
    if len(historial) < 2:
        st.warning(f"⚠️ Tienes solo {len(historial)} partido(s) registrado(s) para este escenario. Se necesitan al menos 2.")
    else:
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top 5 Marcadores")
            for res, freq in conteo.most_common(5):
                prob = (freq / num_sim) * 100
                st.write(f"**{res}** ➔ `{prob:.1f}%`")
                
        with col2:
            st.markdown("### 📈 Probabilidades")
            st.metric(f"Más de {linea_goles} Goles", f"{(sim_goles_favor > linea_goles).mean() * 100:.1f}%")
            st.metric(f"Más de {linea_tiros} Tiros Totales", f"{(sim_tiros > linea_tiros).mean() * 100:.1f}%")
            st.metric(f"Más de {linea_tiros_puerta} Tiros a Puerta", f"{(sim_tiros_puerta > linea_tiros_puerta).mean() * 100:.1f}%")
            st.metric(f"Más de {linea_corners} Córners", f"{(sim_corners > linea_corners).mean() * 100:.1f}%")
            st.metric(f"Más de {linea_faltas} Faltas", f"{(sim_faltas > linea_faltas).mean() * 100:.1f}%")
