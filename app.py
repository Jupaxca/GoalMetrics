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

st.write("✅ La app está corriendo... cargando datos")

# 1. CARGA DE DATOS (con diagnóstico)
@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise Exception(f"No se pudo leer el Google Sheet. Error: {e}")
    
    df.columns = df.columns.astype(str).str.strip()
    
    columnas_necesarias = ['Equipo', 'Fecha', 'Condición', 'Nivel Rival']
    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        raise Exception(f"Faltan estas columnas en el Sheet: {faltantes}. Columnas encontradas: {list(df.columns)}")
    
    df = df.dropna(subset=columnas_necesarias)
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha'])
    
    df['Equipo'] = df['Equipo'].astype(str).str.strip()
    df['Condición'] = df['Condición'].astype(str).str.strip().str.lower()
    df['Nivel Rival'] = df['Nivel Rival'].astype(str).str.strip()
    
    for col in ['Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

# 2. MOTOR MATEMÁTICO
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

# ===== CARGA CON MENSAJE VISIBLE =====
with st.spinner("Cargando datos del Google Sheet..."):
    try:
        df = cargar_datos()
        st.success(f"✅ Datos cargados correctamente — {len(df)} filas")
    except Exception as e:
        st.error(f"⚠️ Error al cargar los datos:\n\n{e}")
        st.info("""
        **Posibles causas:**
        1. El Google Sheet no es público (debe estar en "Cualquier persona con el enlace")
        2. Problema de internet
        3. Nombres de columnas diferentes
        """)
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
exactos_check = df_diagnostico[
    (df_diagnostico['Condición'] == condicion_sel) & 
    (df_diagnostico['Nivel Rival'] == nivel_sel)
]
num_exactos = len(exactos_check)

st.sidebar.markdown("---")
if num_exactos >= 2:
    st.sidebar.success(f"✅ Partidos exactos encontrados: {num_exactos}")
elif num_exactos == 1:
    st.sidebar.warning("⚠️ Solo 1 partido exacto. Se activará respaldo.")
else:
    st.sidebar.error("❌ 0 partidos exactos encontrados.")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Líneas de Estudio / Apuesta")
linea_goles = st.sidebar.slider("⚽ Línea de Goles", 0.5, 3.5, 1.5, 0.5)
linea_tiros = st.sidebar.slider("👟 Línea de Tiros Totales", 5.0, 25.0, 12.5, 0.5)
linea_tiros_puerta = st.sidebar.slider("🎯 Línea Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
linea_corners = st.sidebar.slider("🚩 Línea de Córners", 1.0, 15.0, 5.5, 0.5)
linea_faltas = st.sidebar.slider("🛑 Línea de Faltas", 5.0, 25.0, 10.5, 0.5)

# ===== CUOTAS REALES DE LA CASA (Value Bet) =====
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Cuotas de la Casa (Value Bet)")
st.sidebar.caption("Escribe las cuotas que ves en BetPlay, Wplay, etc.")

cuota_casa_1 = st.sidebar.number_input("Cuota Real Victoria (1)", min_value=1.01, max_value=50.0, value=1.80, step=0.01, format="%.2f")
cuota_casa_x = st.sidebar.number_input("Cuota Real Empate (X)", min_value=1.01, max_value=50.0, value=3.40, step=0.01, format="%.2f")
cuota_casa_2 = st.sidebar.number_input("Cuota Real Derrota (2)", min_value=1.01, max_value=50.0, value=4.20, step=0.01, format="%.2f")
cuota_casa_btts_si = st.sidebar.number_input("Cuota Real BTTS Sí", min_value=1.01, max_value=50.0, value=1.75, step=0.01, format="%.2f")
cuota_casa_btts_no = st.sidebar.number_input("Cuota Real BTTS No", min_value=1.01, max_value=50.0, value=2.05, step=0.01, format="%.2f")

color_equipo = colores_equipos.get(equipo_sel, "#3B82F6")

# --- ESTILOS CSS ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: #090D16; color: #F3F4F6; }}
    .stSidebar {{ background-color: #111827; }}
    .insight-box {{ padding: 15px; border-radius: 10px; background-color: #1F2937; border-left: 5px solid {color_equipo}; margin-bottom: 20px; font-size: 16px; line-height: 1.5; }}
    .value-box {{ padding: 12px; border-radius: 8px; margin-bottom: 10px; font-size: 15px; }}
    .value-yes {{ background-color: #064e3b; border-left: 5px solid #10b981; }}
    .value-no {{ background-color: #1f2937; border-left: 5px solid #6b7280; }}
    </style>
""", unsafe_allow_html=True)

# --- FUNCIÓN DE ADN ---
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

    st.altair_chart(chart, use_container_width=True)

# --- PANEL PRINCIPAL ---
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    btn_analizar = st.button("⚡ Analizar", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🧹 Limpiar"):
        st.rerun()

if btn_analizar:
    df_base = df[df['Equipo'] == equipo_sel].sort_values(by='Fecha', ascending=False)
    df_exactos = df_base[
        (df_base['Condición'] == condicion_sel) & 
        (df_base['Nivel Rival'] == nivel_sel)
    ]
    
    historial = pd.DataFrame()
    fuente_datos = ""
    
    # ===== LÓGICA PRINCIPAL =====
    # Mínimo 2 → usa TODOS los que cumplan el filtro
    if len(df_exactos) >= 2:
        historial = df_exactos.copy()
        fuente_datos = f"Exacto ({condicion_label} vs {nivel_sel}) — {len(historial)} partidos"
    
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

    # ===== CABECERA =====
    st.markdown(f"""
        <div style="background-color: {color_equipo}; padding: 18px; border-radius: 12px; color: white; text-align: center; font-weight: bold; font-size: 22px; margin-bottom: 25px;">
            🛡️ {equipo_sel.upper()} ({condicion_label.upper()} vs {nivel_sel.upper()})
        </div>
    """, unsafe_allow_html=True)
    
    # ===== PESOS TEMPORALES =====
    hoy = pd.Timestamp.today().normalize()
    historial['Dias_Pasados'] = (hoy - pd.to_datetime(historial['Fecha'])).dt.days.replace(0, 0.1)
    historial['Peso'] = 1 / (1 + (historial['Dias_Pasados'] / 30))
    
    def prom(col):
        if col not in historial.columns:
            return 0.0
        return round(float(np.average(historial[col], weights=historial['Peso'])), 4)

    lam_f = prom('Goles')
    lam_c = prom('Goles Rival')
    lam_t = prom('Tiros' if 'Tiros' in historial.columns else 'Tiros Prom')
    lam_tp = prom('A Puerta' if 'A Puerta' in historial.columns else 'A Puerta Prom')
    lam_co = prom('Corners')
    lam_fa = prom('Faltas')
    
    sg_fav, sg_con, s_tir, s_tpuerta, s_corn, s_faltas = simular_montecarlo(
        lam_f, lam_c, lam_t, lam_tp, lam_co, lam_fa
    )
    
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
