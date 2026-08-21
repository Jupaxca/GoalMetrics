import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="GoalMetrics | Football Analytics", 
    page_icon="📊", 
    layout="wide"
)

# 1. CARGA DE DATOS
@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=['Equipo', 'Fecha', 'Condición', 'Nivel Rival'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Equipo'] = df['Equipo'].astype(str).str.strip()
    df['Condición'] = df['Condición'].astype(str).str.strip().str.lower()
    df['Nivel Rival'] = df['Nivel Rival'].astype(str).str.strip()
    
    for col in ['Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# 2. MOTOR MATEMÁTICO SELLADO
@st.cache_data
def simular_montecarlo(lam_fav, lam_con, lam_tir, lam_tpuerta, lam_corn, lam_faltas):
    rng = np.random.default_rng(42)
    num_sim = 10000
    return (
        rng.poisson(lam=lam_fav, size=num_sim),
        rng.poisson(lam=lam_con, size=num_sim),
        rng.poisson(lam=lam_tir, size=num_sim),
        rng.poisson(lam=lam_tpuerta, size=num_sim),
        rng.poisson(lam=lam_corn, size=num_sim),
        rng.poisson(lam=lam_faltas, size=num_sim)
    )

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"⚠️ Error al cargar los datos: {e}")
    st.stop()

# DICCIONARIO DE COLORES
colores_equipos = {
    "Palmeiras": "#006400", "Flamengo": "#C8102E", "Paranaense": "#CC0000",
    "Fluminense": "#8B0000", "Vasco": "#333333", "Arsenal": "#EF0107",
    "Aston villa": "#670E36", "Barcelona": "#A50044", "Bayern Munich": "#DC052D",
    "Benfica": "#E30613", "Como": "#002D62", "Freiburg": "#222222",
    "Inter": "#010E80", "Liverpool": "#C8102E", "Lyon": "#1D428A",
    "Manchester City": "#6CABDD", "Manchester United": "#DA291C",
    "Newcastle": "#241F20", "Porto": "#003399", "PSG": "#004170",
    "Real Madrid": "#00529F"
}

# --- PANEL LATERAL ---
st.sidebar.header("⚙️ Configuración de Análisis")

lista_equipos = sorted([str(x) for x in df['Equipo'].unique() if pd.notna(x)])
equipo_sel = st.sidebar.selectbox("🏟️ Selecciona el Equipo", lista_equipos)

df_equipo = df[df['Equipo'] == equipo_sel]
lista_niveles = sorted([str(x) for x in df_equipo['Nivel Rival'].unique() if pd.notna(x)])

condicion_label = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
condicion_sel = condicion_label.lower()

nivel_sel = st.sidebar.selectbox("⭐ Torneo / Nivel del Rival", lista_niveles)

df_diagnostico = df_equipo.sort_values(by='Fecha', ascending=False)
exactos_check = df_diagnostico[(df_diagnostico['Condición'] == condicion_sel) & 
                               (df_diagnostico['Nivel Rival'] == nivel_sel)]
num_exactos = len(exactos_check)

st.sidebar.markdown("---")
if num_exactos >= 2:
    st.sidebar.success(f"✅ Partidos exactos encontrados: {num_exactos}")
elif num_exactos == 1:
    st.sidebar.warning(f"⚠️ Pocos partidos exactos (1). Se activará respaldo.")
else:
    st.sidebar.error("❌ 0 partidos exactos encontrados.")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Líneas de Estudio / Apuesta")
linea_goles = st.sidebar.slider("⚽ Línea de Goles", 0.5, 3.5, 1.5, 0.5)
linea_tiros = st.sidebar.slider("👟 Línea de Tiros Totales", 5.0, 25.0, 12.5, 0.5)
linea_tiros_puerta = st.sidebar.slider("🎯 Línea Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
linea_corners = st.sidebar.slider("🚩 Línea de Córners", 1.0, 15.0, 5.5, 0.5)
linea_faltas = st.sidebar.slider("🛑 Línea de Faltas", 5.0, 25.0, 10.5, 0.5)

color_equipo = colores_equipos.get(equipo_sel, "#3B82F6")

# --- ESTILOS CSS ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: #090D16; color: #F3F4F6; }}
    .stSidebar {{ background-color: #111827; }}
    .insight-box {{ padding: 15px; border-radius: 10px; background-color: #1F2937; border-left: 5px solid {color_equipo}; margin-bottom: 20px; font-size: 16px; line-height: 1.5; }}
    </style>
""", unsafe_allow_html=True)

# --- FUNCIÓN DE ADN CON ALTAIR ---
def renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa):
    val_ataque = min(round(lam_f * 3.33, 1), 10.0)
    val_tiros = min(round(lam_t / 2.5, 1), 10.0)
    val_precis = min(round(lam_tp * 1.66, 1), 10.0)
    val_corners = min(round(lam_co / 1.5, 1), 10.0)
    val_discip = min(round((25 - lam_fa) / 2.5, 1), 10.0)

    df_adn = pd.DataFrame({
        'Métrica': ['Ataque', 'Volumen Tiros', 'Precisión', 'Córners', 'Disciplina'],
        'Puntuación': [val_ataque, val_tiros, val_precis, val_corners, val_discip]
    })

    chart = alt.Chart(df_adn).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X('Puntuación:Q', scale=alt.Scale(domain=[0, 10]), title='Escala de Rendimiento (0 - 10)'),
        y=alt.Y('Métrica:N', sort='-x', title=None),
        color=alt.value(color_equipo),
        tooltip=['Métrica', 'Puntuación']
    ).properties(height=240)

    st.altair_chart(chart, width='stretch')

# --- PANEL PRINCIPAL ---
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    btn_analizar = st.button("⚡ Analizar", type="primary", width='stretch')
with col_btn2:
    if st.button("🧹 Limpiar", width='content'):
        st.rerun()

if btn_analizar:
    df_base = df[df['Equipo'] == equipo_sel].sort_values(by='Fecha', ascending=False)
    df_exactos = df_base[(df_base['Condición'] == condicion_sel) & (df_base['Nivel Rival'] == nivel_sel)]
    
    historial = pd.DataFrame()
    fuente_datos = ""
    
    if len(df_exactos) >= 2:
        historial = df_exactos.head(2).copy()
        fuente_datos = f"Exacto ({condicion_label} vs {nivel_sel})"
    elif len(df_exactos) == 1:
        partido_1 = df_exactos.head(1).copy()
        cond_opuesta = "visitante" if condicion_sel == "local" else "local"
        df_opuestos = df_base[df_base['Condición'] == cond_opuesta]
        
        if len(df_opuestos) >= 1:
            partido_2 = df_opuestos.head(1).copy()
            factor = 0.88 if condicion_sel == "visitante" else 1.12
            for col in ['Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas']:
                if col in partido_2.columns:
                    partido_2[col] = (partido_2[col] * factor).round(2)
            historial = pd.concat([partido_1, partido_2], ignore_index=True)
            fuente_datos = f"Mixto con respaldo ({cond_opuesta.capitalize()} ajustado)"
        else:
            historial = partido_1.copy()
            fuente_datos = "Solo 1 partido exacto disponible"
    else:
        st.error(f"❌ No hay datos suficientes para analizar a {equipo_sel}.")
        st.stop()

    st.markdown(f"""
        <div style="background-color: {color_equipo}; padding: 18px; border-radius: 12px; color: white; text-align: center; font-weight: bold; font-size: 22px; margin-bottom: 25px;">
            🛡️ {equipo_sel.upper()} ({condicion_label.upper()} vs {nivel_sel.upper()})
        </div>
    """, unsafe_allow_html=True)
    
    hoy = pd.Timestamp.today().normalize()
    historial['Dias_Pasados'] = (hoy - pd.to_datetime(historial['Fecha'])).dt.days.replace(0, 0.1)
    historial['Peso'] = 1 / (1 + (historial['Dias_Pasados'] / 30))
    
    def prom(col):
        return round(float(np.average(historial[col], weights=historial['Peso'])), 4) if col in historial.columns else 0.0

    lam_f = prom('Goles')
    lam_c = prom('Goles Rival')
    lam_t = prom('Tiros' if 'Tiros' in historial.columns else 'Tiros Prom')
    lam_tp = prom('A Puerta' if 'A Puerta' in historial.columns else 'A Puerta Prom')
    lam_co = prom('Corners')
    lam_fa = prom('Faltas')
    
    sg_fav, sg_con, s_tir, s_tpuerta, s_corn, s_faltas = simular_montecarlo(lam_f, lam_c, lam_t, lam_tp, lam_co, lam_fa)
    
    num_sim = 10000
    triunfos = (sg_fav > sg_con).mean() * 100
    empates = (sg_fav == sg_con).mean() * 100
    derrotas = (sg_fav < sg_con).mean() * 100
    ambos_anotan = ((sg_fav > 0) & (sg_con > 0)).mean() * 100
    
    doble_1x = triunfos + empates
    doble_x2 = derrotas + empates
    tot_sin_emp = triunfos + derrotas
    dnb = (triunfos / tot_sin_emp * 100) if tot_sin_emp > 0 else 50.0
    
    marcadores = [f"{f}-{c}" for f, c in zip(sg_fav, sg_con)]
    conteo = Counter(marcadores)
    marcador_mas_comun = conteo.most_common(1)[0][0]
    
    if triunfos > 50: veredicto = f"Tendencia Fuerte: Marcador proyectado {marcador_mas_comun}."
    elif derrotas > 50: veredicto = f"Alerta de Complicación: Marcador proyectado {marcador_mas_comun}."
    else: veredicto = f"Partido Muy Parejo: Marcador proyectado {marcador_mas_comun}."

    st.markdown(f'<div class="insight-box"><b>Veredicto GoalMetrics:</b> {veredicto}</div>', unsafe_allow_html=True)

    # ADN DEL EQUIPO
    st.subheader("🧬 ADN del Equipo")
    renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Victoria (1)", f"{triunfos:.1f}%")
    c2.metric("🟡 Empate (X)", f"{empates:.1f}%")
    c3.metric("🔴 Derrota (2)", f"{derrotas:.1f}%")
    c4.metric("⚽ Ambos Anotan (BTTS)", f"{ambos_anotan:.1f}%")
    
    c5, c6, c7 = st.columns(3)
    c5.metric("🛡️ Doble Oportunidad (1X)", f"{doble_1x:.1f}%")
    c6.metric("🛡️ Doble Oportunidad (X2)", f"{doble_x2:.1f}%")
    c7.metric("⚖️ Apuesta sin Empate (DNB)", f"{dnb:.1f}%")
    
    # --- SECCIÓN DE CUOTAS DE APUESTAS (FAIR ODDS) ---
    st.markdown("---")
    st.subheader("🎯 Cuotas Justas de Mercado (Fair Odds)")
    
    cuota_1 = round(100 / triunfos, 2) if triunfos > 0 else 0.0
    cuota_x = round(100 / empates, 2) if empates > 0 else 0.0
    cuota_2 = round(100 / derrotas, 2) if derrotas > 0 else 0.0
    
    prob_btts_no = 100 - ambos_anotan
    cuota_btts_si = round(100 / ambos_anotan, 2) if ambos_anotan > 0 else 0.0
    cuota_btts_no = round(100 / prob_btts_no, 2) if prob_btts_no > 0 else 0.0

    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Cuota Local (1)", f"{cuota_1}", delta=f"{triunfos:.1f}% prob")
    qc2.metric("Cuota Empate (X)", f"{cuota_x}", delta=f"{empates:.1f}% prob")
    qc3.metric("Cuota Visitante (2)", f"{cuota_2}", delta=f"{derrotas:.1f}% prob")

    qc4, qc5 = st.columns(2)
    qc4.metric("Cuota BTTS Sí", f"{cuota_btts_si}", delta=f"{ambos_anotan:.1f}% prob")
    qc5.metric("Cuota BTTS No", f"{cuota_btts_no}", delta=f"{prob_btts_no:.1f}% prob")

    st.markdown("---")
    
    def crear_grafico(serie, titulo):
        df_c = pd.DataFrame({titulo: serie.value_counts().sort_index().index.astype(str), 'Prob (%)': (serie.value_counts().sort_index() / num_sim) * 100})
        return alt.Chart(df_c).mark_bar(color=color_equipo).encode(x=alt.X(f"{titulo}:N", sort=None, labelAngle=0), y=alt.Y('Prob (%):Q', format='.1f')).properties(height=300)

    st.markdown("#### ⚽ Probabilidad de Goles a Favor")
    st.altair_chart(crear_grafico(pd.Series(sg_fav), 'Goles'), width='stretch')

    st.markdown("#### 🚩 Probabilidad de Córners")
    st.altair_chart(crear_grafico(pd.Series(s_corn).astype(int), 'Córners'), width='stretch')
    
    st.markdown("---")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🏆 Top 5 Marcadores")
        st.dataframe(pd.DataFrame([{"Marcador": r, "Probabilidad": f"{(f/num_sim)*100:.1f}%"} for r, f in conteo.most_common(5)]), hide_index=True, width='stretch')
            
    with col_r:
        st.markdown("#### 📈 Probabilidades de Líneas")
        st.metric(label=f"⚽ Más de {linea_goles} Goles", value=f"{(sg_fav > linea_goles).mean() * 100:.1f}%")
        st.metric(label=f"👟 Más de {linea_tiros} Tiros", value=f"{(s_tir > linea_tiros).mean() * 100:.1f}%")
        st.metric(label=f"🎯 Más de {linea_tiros_puerta} a Puerta", value=f"{(s_tpuerta > linea_tiros_puerta).mean() * 100:.1f}%")
        st.metric(label=f"🚩 Más de {linea_corners} Córners", value=f"{(s_corn > linea_corners).mean() * 100:.1f}%")
        st.metric(label=f"🛑 Más de {linea_faltas} Faltas", value=f"{(s_faltas > linea_faltas).mean() * 100:.1f}%")

    st.markdown("---")
    st.info(f"💡 Base del análisis: {len(historial)} partidos analizados bajo el modo: {fuente_datos}")
    
    with st.expander("📋 Ver detalle completo de los partidos utilizados (Rivales y Estadísticas)"):
        h_disp = historial.copy()
        h_disp['Fecha'] = pd.to_datetime(h_disp['Fecha']).dt.strftime('%Y-%m-%d')
        cols = [c for c in ['Fecha', 'Equipo', 'Condición', 'Rival', 'Nivel Rival', 'Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas'] if c in h_disp.columns]
        st.dataframe(h_disp[cols], hide_index=True, width='stretch')
    
