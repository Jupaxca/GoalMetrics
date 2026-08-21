import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="GoalMetrics | Football Analytics", 
    page_icon="📊", 
    layout="wide"
)

# 1. CARGAR DATOS DESDE GOOGLE SHEETS
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

# DICCIONARIO DE COLORES OFICIALES Y COMPLETOS
colores_equipos = {
    "Palmeiras": "#006400",
    "Flamengo": "#C8102E",
    "Paranaense": "#CC0000",
    "Fluminense": "#8B0000",
    "Vasco": "#333333",
    "Arsenal": "#EF0107",
    "Aston villa": "#670E36",
    "Barcelona": "#A50044",
    "Bayern Munich": "#DC052D",
    "Benfica": "#E30613",
    "Como": "#002D62",
    "Freiburg": "#222222",
    "Inter": "#010E80",
    "Liverpool": "#C8102E",
    "Lyon": "#1D428A",
    "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C",
    "Newcastle": "#241F20",
    "Porto": "#003399",
    "PSG": "#004170",
    "Real Madrid": "#00529F"
}

# 2. PANEL LATERAL
st.sidebar.header("⚙️ Configuración de Análisis")

lista_equipos = sorted([str(x) for x in df['Equipo'].unique() if pd.notna(x)])
equipo_seleccionado = st.sidebar.selectbox("🏟️ Selecciona el Equipo", lista_equipos)

df_equipo = df[df['Equipo'] == equipo_seleccionado]
lista_niveles_equipo = sorted([str(x) for x in df_equipo['Nivel Rival'].unique() if pd.notna(x)])

condicion_seleccionada = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
nivel_seleccionado = st.sidebar.selectbox("⭐ Torneo / Nivel del Rival", lista_niveles_equipo)

historial_exacto = df[(df['Equipo'] == equipo_seleccionado) & 
                       (df['Condición'] == condicion_seleccionada) & 
                       (df['Nivel Rival'] == nivel_seleccionado)]

st.sidebar.markdown("---")
if len(historial_exacto) >= 2:
    st.sidebar.success(f"✅ Partidos exactos encontrados: **{len(historial_exacto)}**")
else:
    st.sidebar.warning(f"⚠️ Pocos partidos exactos ({len(historial_exacto)}). Se activará respaldo automático.")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Líneas de Estudio / Apuesta")
linea_goles = st.sidebar.slider("⚽ Línea de Goles", 0.5, 3.5, 1.5, 0.5)
linea_tiros = st.sidebar.slider("👟 Línea de Tiros Totales", 5.0, 25.0, 12.5, 0.5)
linea_tiros_puerta = st.sidebar.slider("🎯 Línea Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
linea_corners = st.sidebar.slider("🚩 Línea de Córners", 1.0, 15.0, 5.5, 0.5)
linea_faltas = st.sidebar.slider("🛑 Línea de Faltas", 5.0, 25.0, 10.5, 0.5)

color_equipo = colores_equipos.get(equipo_seleccionado, "#3B82F6")

# ESTILOS CSS DINÁMICOS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #090D16;
        color: #F3F4F6;
    }}
    .stSidebar {{
        background-color: #111827;
    }}
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: {color_equipo};
        color: white;
        font-size: 16px;
        padding: 12px;
        border: none;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        opacity: 0.85;
    }}
    .team-header {{
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
        background-color: {color_equipo};
    }}
    .brand-title {{
        font-size: 32px;
        font-weight: 800;
        color: #F3F4F6;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .brand-subtitle {{
        color: #9CA3AF;
        font-size: 15px;
        margin-bottom: 5px;
    }}
    .brand-author {{
        color: {color_equipo};
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 20px;
    }}
    .insight-box {{
        padding: 15px;
        border-radius: 10px;
        background-color: #1F2937;
        border-left: 5px solid {color_equipo};
        margin-bottom: 20px;
        font-size: 16px;
        line-height: 1.5;
    }}
    .explanation-text {{
        color: #9CA3AF;
        font-size: 13px;
        margin-top: 5px;
        margin-bottom: 15px;
    }}
    </style>
""", unsafe_allow_html=True)

# ENCABEZADO DE MARCA CON TU FIRMA
st.markdown('<div class="brand-title">📊 GoalMetrics <span style="color: #3B82F6; font-size: 20px;">FOOTBALL ANALYTICS</span></div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Plataforma avanzada de simulación estadística y predicción de rendimiento deportivo.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="brand-author">By: Juan Camilo Barreto</div>', unsafe_allow_html=True)
st.markdown("---")

# 3. PANEL PRINCIPAL
st.markdown("### 🕹️ Centro de Simulación")

if st.button("⚡ Ejecutar Motor de Predicción", type="primary"):
    
    num_exactos = len(historial_exacto)
    historial = historial_exacto.copy()
    fuente_datos = "Exacto (Condición y Nivel)"
    
    if len(historial) < 2:
        condicion_contraria = "Visitante" if condicion_seleccionada == "Local" else "Local"
        historial_respaldo = df[(df['Equipo'] == equipo_seleccionado) & 
                                (df['Condición'] == condicion_contraria) & 
                                (df['Nivel Rival'] == nivel_seleccionado)]
        
        if len(historial_respaldo) + len(historial) >= 2:
            factor_ajuste = 0.85 if condicion_seleccionada == "Local" else 1.15
            historial = historial_respaldo.copy()
            for col in ['Goles', 'Tiros Prom', 'A Puerta Prom', 'Corners']:
                if col in historial.columns:
                    historial[col] = historial[col] * factor_ajuste
            fuente_datos = f"Mixto con respaldo ({condicion_contraria} ajustado)"
    
    st.markdown(f"""
        <div class="team-header">
            🛡️ {equipo_seleccionado.upper()} ({condicion_seleccionada.upper()} vs {nivel_seleccionado.upper()})
        </div>
    """, unsafe_allow_html=True)
    
    if len(historial) < 2:
        st.error(f"❌ No hay suficiente información en el registro para analizar a {equipo_seleccionado}. Se requieren al menos 2 partidos.")
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
        
        triunfos = (sim_goles_favor > sim_goles_contra).mean() * 100
        empates = (sim_goles_favor == sim_goles_contra).mean() * 100
        derrotas = (sim_goles_favor < sim_goles_contra).mean() * 100
        ambos_anotan = ((sim_goles_favor > 0) & (sim_goles_contra > 0)).mean() * 100
        
        doble_oportunidad_1x = triunfos + empates
        doble_oportunidad_x2 = derrotas + empates
        total_sin_empate = triunfos + derrotas
        dnb_favor = (triunfos / total_sin_empate * 100) if total_sin_empate > 0 else 50.0
        
        marcadores = [f"{f}-{c}" for f, c in zip(sim_goles_favor, sim_goles_contra)]
        conteo = Counter(marcadores)
        
        # EXPLICACIÓN DETALLADA DE LA FIABILIDAD
        if num_exactos >= 4:
            explicacion_fiabilidad = f"Alta. Se encontraron {num_exactos} partidos exactos cumpliendo con la condición y el nivel de rival seleccionados, lo que otorga una base estadística muy sólida."
        elif num_exactos >= 2:
            explicacion_fiabilidad = f"Media. Se cuenta con {num_exactos} partidos exactos; es una muestra válida, aunque limitada, ponderada por la fecha de los encuentros."
        else:
            explicacion_fiabilidad = f"Moderada / Respaldo Activado. Había pocos partidos exactos ({num_exactos}), por lo que el sistema aplicó el respaldo inteligente usando la condición contraria con su respectivo ajuste matemático."

        marcador_mas_comun = conteo.most_common(1)[0][0]
        if triunfos > 50:
            veredicto = f"Tendencia Fuerte: {equipo_seleccionado} muestra un dominio estadístico claro en esta condición. El marcador más repetido en las 10,000 simulaciones es {marcador_mas_comun}."
        elif derrotas > 50:
            veredicto = f"Alerta de Complicación: Las métricas favorecen al rival en este escenario. El marcador más probable es {marcador_mas_comun}, sugiriendo cautela."
        else:
            veredicto = f"Partido Muy Parejo: Escenario sumamente equilibrado con alta probabilidad de empate o diferencia mínima. El resultado más común simulado fue {marcador_mas_comun}."

        st.markdown(f'<div class="insight-box"><b>Veredicto GoalMetrics:</b> {veredicto}<br><br><b>Fiabilidad de los Datos:</b> {explicacion_fiabilidad}</div>', unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("🟢 Victoria (1)", f"{triunfos:.1f}%")
        col_m2.metric("🟡 Empate (X)", f"{empates:.1f}%")
        col_m3.metric("🔴 Derrota (2)", f"{derrotas:.1f}%")
        col_m4.metric("⚽ Ambos Anotan (BTTS)", f"{ambos_anotan:.1f}%")
        
        col_n1, col_n2, col_n3 = st.columns(3)
        col_n1.metric("🛡️ Doble Oportunidad (1X)", f"{doble_oportunidad_1x:.1f}%")
        col_n2.metric("🛡️ Doble Oportunidad (X2)", f"{doble_oportunidad_x2:.1f}%")
        col_n3.metric("⚖️ Apuesta sin Empate (DNB)", f"{dnb_favor:.1f}%")
        
        st.markdown("---")

        # GRÁFICO DE GOLES CON EXPLICACIÓN
        st.markdown("#### ⚽ Distribución de Probabilidad de Goles a Favor")
        st.markdown('<div class="explanation-text">Este gráfico muestra qué tan probable es que el equipo anote 0, 1, 2, 3 o más goles en el partido. Las barras más altas indican la cantidad de anotaciones que ocurrieron con mayor frecuencia en la simulación.</div>', unsafe_allow_html=True)
        conteo_goles = pd.Series(sim_goles_favor).value_counts().sort_index()
        df_goles_chart = pd.DataFrame({
            'Goles': conteo_goles.index,
            'Probabilidad (%)': (conteo_goles / num_sim) * 100
        }).set_index('Goles')
        st.bar_chart(df_goles_chart, color=color_equipo)

        # GRÁFICO DE CÓRNERS CON EXPLICACIÓN
        st.markdown("#### 🚩 Distribución de Probabilidad de Córners")
        st.markdown('<div class="explanation-text">Representa las probabilidades de saques de esquina que cobrará el equipo. Te ayuda a visualizar el volumen ofensivo a través de tiros de esquina esperados.</div>', unsafe_allow_html=True)
        conteo_corners = pd.Series(sim_corners).astype(int).value_counts().sort_index()
        df_corners_chart = pd.DataFrame({
            'Córners': conteo_corners.index,
            'Probabilidad (%)': (conteo_corners / num_sim) * 100
        }).set_index('Córners')
        st.bar_chart(df_corners_chart, color=color_equipo)
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 🏆 Top 5 Marcadores Más Probables")
            st.markdown('<div class="explanation-text">Estos son los 5 resultados finales (Goles de tu equipo - Goles del rival) que más veces ocurrieron tras simular el partido 10,000 veces.</div>', unsafe_allow_html=True)
            tabla_data = []
            for res, freq in conteo.most_common(5):
                prob = (freq / num_sim) * 100
                tabla_data.append({"Marcador (Favor - Contra)": res, "Probabilidad": f"{prob:.1f}%"})
            st.dataframe(pd.DataFrame(tabla_data), hide_index=True, use_container_width=True)
                
        with col_right:
            st.markdown("#### 📈 Probabilidades de Líneas")
            st.markdown('<div class="explanation-text">Porcentaje de probabilidad de superar las líneas estadísticas elegidas en la barra lateral (goles, tiros, córners y faltas).</div>', unsafe_allow_html=True)
            st.metric(label=f"⚽ Más de {linea_goles} Goles", value=f"{(sim_goles_favor > linea_goles).mean() * 100:.1f}%")
            st.metric(label=f"👟 Más de {linea_tiros} Tiros Totales", value=f"{(sim_tiros > linea_tiros).mean() * 100:.1f}%")
            st.metric(label=f"🎯 Más de {linea_tiros_puerta} Tiros a Puerta", value=f"{(sim_tiros_puerta > linea_tiros_puerta).mean() * 100:.1f}%")
            st.metric(label=f"🚩 Más de {linea_corners} Córners", value=f"{(sim_corners > linea_corners).mean() * 100:.1f}%")
            st.metric(label=f"🛑 Más de {linea_faltas} Faltas", value=f"{(sim_faltas > linea_faltas).mean() * 100:.1f}%")

        st.markdown("---")
        st.info(f"💡 **Base del análisis:** {len(historial)} partidos analizados bajo el modo: **{fuente_datos}** (ponderados por fecha).")
        
        with st.expander("📋 Ver detalle completo de los partidos utilizados (Rivales y Estadísticas)"):
            historial_display = historial.copy()
            historial_display['Fecha'] = pd.to_datetime(historial_display['Fecha']).dt.strftime('%Y-%m-%d')
            columnas_disponibles = [col for col in ['Fecha', 'Equipo', 'Condición', 'Rival', 'Nivel Rival', 'Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas'] if col in historial_display.columns]
            st.dataframe(historial_display[columnas_disponibles], hide_index=True, use_container_width=True)
