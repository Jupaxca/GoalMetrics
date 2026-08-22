import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="📊",
    layout="wide"
)

# ====================== CARGA DE DATOS ======================
@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=['Equipo', 'Fecha', 'Condición', 'Nivel Rival'])
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha'])
    df['Equipo'] = df['Equipo'].astype(str).str.strip()
    df['Condición'] = df['Condición'].astype(str).str.strip().str.lower()
    df['Nivel Rival'] = df['Nivel Rival'].astype(str).str.strip()
    
    for col in ['Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

@st.cache_data
def simular_montecarlo(lam_fav, lam_con, lam_tir, lam_tpuerta, lam_corn, lam_faltas):
    rng = np.random.default_rng(42)
    num_sim = 10000
    return (
        rng.poisson(lam=max(lam_fav, 0.01), size=num_sim),
        rng.poisson(lam=max(lam_con, 0.01), size=num_sim),
        rng.poisson(lam=max(lam_tir, 0.01), size=num_sim),
        rng.poisson(lam=max(lam_tpuerta, 0.01), size=num_sim),
        rng.poisson(lam=max(lam_corn, 0.01), size=num_sim),
        rng.poisson(lam=max(lam_faltas, 0.01), size=num_sim)
    )

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"⚠️ Error al cargar los datos: {e}")
    st.stop()

# ====================== COLORES ======================
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

# ====================== SIDEBAR ======================
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

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Cuotas de la Casa (Value Bet)")
st.sidebar.caption("Escribe las cuotas que ves en la casa de apuestas")

cuota_casa_1 = st.sidebar.number_input("Cuota Real Victoria (1)", min_value=1.01, max_value=50.0, value=1.80, step=0.01, format="%.2f")
cuota_casa_x = st.sidebar.number_input("Cuota Real Empate (X)", min_value=1.01, max_value=50.0, value=3.40, step=0.01, format="%.2f")
cuota_casa_2 = st.sidebar.number_input("Cuota Real Derrota (2)", min_value=1.01, max_value=50.0, value=4.20, step=0.01, format="%.2f")
cuota_casa_btts_si = st.sidebar.number_input("Cuota Real BTTS Sí", min_value=1.01, max_value=50.0, value=1.75, step=0.01, format="%.2f")
cuota_casa_btts_no = st.sidebar.number_input("Cuota Real BTTS No", min_value=1.01, max_value=50.0, value=2.05, step=0.01, format="%.2f")

color_equipo = colores_equipos.get(equipo_sel, "#3B82F6")

# ====================== CSS ======================
st.markdown(f"""
    <style>
    .stApp {{ background-color: #090D16; color: #F3F4F6; }}
    .stSidebar {{ background-color: #111827; }}
    .insight-box {{
        padding: 15px; border-radius: 10px; background-color: #1F2937;
        border-left: 5px solid {color_equipo}; margin-bottom: 20px;
        font-size: 16px; line-height: 1.5;
    }}
    .value-box {{ padding: 12px; border-radius: 8px; margin-bottom: 10px; font-size: 15px; }}
    .value-yes {{ background-color: #064e3b; border-left: 5px solid #10b981; }}
    .value-no {{ background-color: #1f2937; border-left: 5px solid #6b7280; }}
    </style>
""", unsafe_allow_html=True)

# ====================== FUNCIÓN ADN ======================
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

# ====================== BOTONES ======================
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    btn_analizar = st.button("⚡ Analizar", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🧹 Limpiar"):
        st.rerun()

# ====================== ANÁLISIS ======================
if btn_analizar:
    df_base = df[df['Equipo'] == equipo_sel].sort_values(by='Fecha', ascending=False)
    df_exactos = df_base[
        (df_base['Condición'] == condicion_sel) & 
        (df_base['Nivel Rival'] == nivel_sel)
    ]
    
    historial = pd.DataFrame()
    fuente_datos = ""
    
    # Mínimo 2 → usa TODOS. Si hay 1 → respaldo. Si hay 0 → error.
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
        st.error(f"❌ No hay datos suficientes para analizar a {equipo_sel} con esos filtros.")
        st.stop()

    # Cabecera
    st.markdown(f"""
        <div style="background-color: {color_equipo}; padding: 18px; border-radius: 12px; color: white; text-align: center; font-weight: bold; font-size: 22px; margin-bottom: 25px;">
            🛡️ {equipo_sel.upper()} ({condicion_label.upper()} vs {nivel_sel.upper()})
        </div>
    """, unsafe_allow_html=True)
    
    # Pesos temporales
    hoy = pd.Timestamp.today().normalize()
    historial['Dias_Pasados'] = (hoy - pd.to_datetime(historial['Fecha'])).dt.days.replace(0, 0.1)
    historial['Peso'] = 1 / (1 + (historial['Dias_Pasados'] / 30))
    
    def prom(col):
        if col not in historial.columns:
            return 0.05
        vals = historial[col].fillna(0)
        return round(float(np.average(vals, weights=historial['Peso'])), 4)

    lam_f = prom('Goles')
    lam_c = prom('Goles Rival')
    lam_t = prom('Tiros')
    lam_tp = prom('A Puerta')
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
    marcador_mas_comun = conteo.most_common(1)[0][0]
    
    if triunfos > 50:
        veredicto = f"Tendencia Fuerte: Marcador proyectado {marcador_mas_comun}."
    elif derrotas > 50:
        veredicto = f"Alerta de Complicación: Marcador proyectado {marcador_mas_comun}."
    else:
        veredicto = f"Partido Muy Parejo: Marcador proyectado {marcador_mas_comun}."

    st.markdown(f'<div class="insight-box"><b>Veredicto GoalMetrics:</b> {veredicto}</div>', unsafe_allow_html=True)

    # ADN
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
    
    # Cuotas Justas + Value Bet
    st.markdown("---")
    st.subheader("🎯 Cuotas Justas + Value Bet")
    
    cuota_justa_1 = round(100 / triunfos, 2) if triunfos > 0 else 99.0
    cuota_justa_x = round(100 / empates, 2) if empates > 0 else 99.0
    cuota_justa_2 = round(100 / derrotas, 2) if derrotas > 0 else 99.0
    prob_btts_no = 100 - ambos_anotan
    cuota_justa_btts_si = round(100 / ambos_anotan, 2) if ambos_anotan > 0 else 99.0
    cuota_justa_btts_no = round(100 / prob_btts_no, 2) if prob_btts_no > 0 else 99.0

    def calcular_ev(prob_porcentaje, cuota_casa):
        if cuota_casa <= 1.0 or prob_porcentaje <= 0:
            return 0.0
        return round((prob_porcentaje / 100 * cuota_casa) - 1, 4)

    ev_1 = calcular_ev(triunfos, cuota_casa_1)
    ev_x = calcular_ev(empates, cuota_casa_x)
    ev_2 = calcular_ev(derrotas, cuota_casa_2)
    ev_btts_si = calcular_ev(ambos_anotan, cuota_casa_btts_si)
    ev_btts_no = calcular_ev(prob_btts_no, cuota_casa_btts_no)

    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Cuota Justa (1)", f"{cuota_justa_1}", delta=f"{triunfos:.1f}%")
    qc2.metric("Cuota Justa (X)", f"{cuota_justa_x}", delta=f"{empates:.1f}%")
    qc3.metric("Cuota Justa (2)", f"{cuota_justa_2}", delta=f"{derrotas:.1f}%")

    qc4, qc5 = st.columns(2)
    qc4.metric("Cuota Justa BTTS Sí", f"{cuota_justa_btts_si}", delta=f"{ambos_anotan:.1f}%")
    qc5.metric("Cuota Justa BTTS No", f"{cuota_justa_btts_no}", delta=f"{prob_btts_no:.1f}%")

    st.markdown("#### 💎 Análisis de Value Bet")
    st.caption("EV positivo = la casa paga más de lo que proyecta el modelo → hay valor.")

    def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob):
        es_value = ev > 0
        clase = "value-yes" if es_value else "value-no"
        icono = "✅ VALUE BET" if es_value else "❌ Sin valor"
        color_ev = "#10b981" if es_value else "#9ca3af"
        st.markdown(f"""
            <div class="value-box {clase}">
                <b>{nombre}</b><br>
                Probabilidad: <b>{prob:.1f}%</b> | Cuota justa: <b>{cuota_justa}</b> | Cuota casa: <b>{cuota_casa}</b><br>
                <span style="color:{color_ev}; font-weight:bold; font-size:17px;">
                    EV: {ev:+.2%} → {icono}
                </span>
            </div>
        """, unsafe_allow_html=True)

    mostrar_value("Victoria (1)", cuota_justa_1, cuota_casa_1, ev_1, triunfos)
    mostrar_value("Empate (X)", cuota_justa_x, cuota_casa_x, ev_x, empates)
    mostrar_value("Derrota (2)", cuota_justa_2, cuota_casa_2, ev_2, derrotas)
    mostrar_value("BTTS Sí", cuota_justa_btts_si, cuota_casa_btts_si, ev_btts_si, ambos_anotan)
    mostrar_value("BTTS No", cuota_justa_btts_no, cuota_casa_btts_no, ev_btts_no, prob_btts_no)

    st.markdown("---")
    
    def crear_grafico(serie, titulo):
        df_c = pd.DataFrame({
            titulo: serie.value_counts().sort_index().index.astype(str),
            'Prob (%)': (serie.value_counts().sort_index() / num_sim) * 100
        })
        return alt.Chart(df_c).mark_bar(color=color_equipo).encode(
            x=alt.X(f"{titulo}:N", sort=None, labelAngle=0),
            y=alt.Y('Prob (%):Q', format='.1f')
        ).properties(height=300)

    st.markdown("#### ⚽ Probabilidad de Goles a Favor")
    st.altair_chart(crear_grafico(pd.Series(sg_fav), 'Goles'), use_container_width=True)

    st.markdown("#### 🚩 Probabilidad de Córners")
    st.altair_chart(crear_grafico(pd.Series(s_corn).astype(int), 'Córners'), use_container_width=True)
    
    st.markdown("---")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🏆 Top 5 Marcadores")
        st.dataframe(
            pd.DataFrame([
                {"Marcador": r, "Probabilidad": f"{(f/num_sim)*100:.1f}%"}
                for r, f in conteo.most_common(5)
            ]),
            hide_index=True,
            use_container_width=True
        )
    with col_r:
        st.markdown("#### 📈 Probabilidades de Líneas")
        st.metric(f"⚽ Más de {linea_goles} Goles", f"{(sg_fav > linea_goles).mean()*100:.1f}%")
        st.metric(f"👟 Más de {linea_tiros} Tiros", f"{(s_tir > linea_tiros).mean()*100:.1f}%")
        st.metric(f"🎯 Más de {linea_tiros_puerta} a Puerta", f"{(s_tpuerta > linea_tiros_puerta).mean()*100:.1f}%")
        st.metric(f"🚩 Más de {linea_corners} Córners", f"{(s_corn > linea_corners).mean()*100:.1f}%")
        st.metric(f"🛑 Más de {linea_faltas} Faltas", f"{(s_faltas > linea_faltas).mean()*100:.1f}%")

    st.markdown("---")
    st.info(f"💡 Base del análisis: **{len(historial)} partidos** | Modo: {fuente_datos}")
    
    with st.expander("📋 Ver partidos utilizados (últimos 5)"):
        h_disp = historial.copy().sort_values(by='Fecha', ascending=False)
        h_mostrar = h_disp.head(5).copy()
        h_mostrar['Fecha'] = pd.to_datetime(h_mostrar['Fecha']).dt.strftime('%Y-%m-%d')
        cols = [c for c in ['Fecha', 'Equipo', 'Condición', 'Rival', 'Nivel Rival',
                           'Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas', 'Peso']
                if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)
        if len(historial) > 5:
            st.caption(f"Se usaron {len(historial)} partidos en total (los más antiguos tienen menos peso).")
