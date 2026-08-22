import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
import hashlib
import colorsys

# ====================== CARGA DE DATOS ======================
@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=['Equipo', 'Fecha', 'Condición', 'Nivel Rival'])
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
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

# ====================== COLORES + GENERADOR DINÁMICO ======================
colores_base_equipos = {
    "Palmeiras": "#006400", "Flamengo": "#C8102E", "Paranaense": "#CC0000",
    "Fluminense": "#8B0000", "Vasco": "#333333", "Arsenal": "#EF0107",
    "Aston villa": "#670E36", "Barcelona": "#A50044", "Bayern Munich": "#DC052D",
    "Benfica": "#E30613", "Como": "#002D62", "Freiburg": "#222222",
    "Inter": "#010E80", "Liverpool": "#C8102E", "Lyon": "#1D428A",
    "Manchester City": "#6CABDD", "Manchester United": "#DA291C",
    "Newcastle": "#241F20", "Porto": "#003399", "PSG": "#004170",
    "Real Madrid": "#00529F"
}

def generar_color_equipo(nombre):
    if nombre in colores_base_equipos:
        return colores_base_equipos[nombre]
    # Genera un color único y vibrante basado en el nombre del equipo
    hash_val = int(hashlib.md5(nombre.encode('utf-8')).hexdigest(), 16)
    hue = (hash_val % 360) / 360.0
    rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Configuración")

with st.sidebar.expander("🏟️ Partido", expanded=True):
    lista_equipos = sorted([str(x) for x in df['Equipo'].unique() if pd.notna(x)])
    equipo_sel = st.selectbox("Equipo", lista_equipos)
    
    df_equipo = df[df['Equipo'] == equipo_sel]
    lista_niveles = sorted([str(x) for x in df_equipo['Nivel Rival'].unique() if pd.notna(x)])
    
    condicion_label = st.selectbox("Condición", ["Local", "Visitante"])
    condicion_sel = condicion_label.lower()
    nivel_sel = st.selectbox("Nivel del Rival", lista_niveles)

df_diagnostico = df_equipo.sort_values(by='Fecha', ascending=False)
exactos_check = df_diagnostico[
    (df_diagnostico['Condición'] == condicion_sel) & 
    (df_diagnostico['Nivel Rival'] == nivel_sel)
]
num_exactos = len(exactos_check)

if num_exactos >= 2:
    st.sidebar.success(f"✅ {num_exactos} partidos exactos")
elif num_exactos == 1:
    st.sidebar.warning("⚠️ 1 partido → se usará respaldo")
else:
    st.sidebar.error("❌ 0 partidos exactos")

with st.sidebar.expander("🎯 Líneas de Estudio"):
    linea_goles = st.slider("Goles", 0.5, 3.5, 1.5, 0.5)
    linea_tiros = st.slider("Tiros Totales", 5.0, 25.0, 12.5, 0.5)
    linea_tiros_puerta = st.slider("Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
    linea_corners = st.slider("Córners", 1.0, 15.0, 5.5, 0.5)
    linea_faltas = st.slider("Faltas", 5.0, 25.0, 10.5, 0.5)

with st.sidebar.expander("💰 Cuotas 1X2 / BTTS / DNB"):
    cuota_casa_1 = st.number_input("Victoria (1)", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_casa_x = st.number_input("Empate (X)", min_value=1.01, value=3.40, step=0.01, format="%.2f")
    cuota_casa_2 = st.number_input("Derrota (2)", min_value=1.01, value=4.20, step=0.01, format="%.2f")
    cuota_casa_btts_si = st.number_input("BTTS Sí", min_value=1.01, value=1.75, step=0.01, format="%.2f")
    cuota_casa_btts_no = st.number_input("BTTS No", min_value=1.01, value=2.05, step=0.01, format="%.2f")
    cuota_casa_dnb = st.number_input("DNB", min_value=1.01, value=1.35, step=0.01, format="%.2f")

with st.sidebar.expander("📈 Cuotas de Líneas (Over)"):
    cuota_over_goles = st.number_input(f"Over {linea_goles} Goles", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_tiros = st.number_input(f"Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_over_puerta = st.number_input(f"Over {linea_tiros_puerta} a Puerta", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_over_corners = st.number_input(f"Over {linea_corners} Córners", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_faltas = st.number_input(f"Over {linea_faltas} Faltas", min_value=1.01, value=1.85, step=0.01, format="%.2f")

color_equipo = generar_color_equipo(equipo_sel)

# ====================== CSS ======================
st.markdown(f"""
    <style>
    .stApp {{ background-color: #0B0F19; color: #F3F4F6; }}
    .stSidebar {{ background-color: #111827; }}
    .header-box {{
        background: linear-gradient(90deg, {color_equipo} 0%, #1F2937 100%);
        padding: 22px 28px; border-radius: 14px; color: white;
        font-weight: 700; font-size: 24px; margin-bottom: 20px;
        text-align: center; letter-spacing: 0.5px;
    }}
    .veredicto-box {{
        padding: 16px 20px; border-radius: 12px; background-color: #1F2937;
        border-left: 5px solid {color_equipo}; margin-bottom: 24px;
        font-size: 16px; line-height: 1.5;
    }}
    .value-box {{
        padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 14px;
    }}
    .value-yes {{ background-color: #064e3b; border-left: 4px solid #10b981; }}
    .value-no {{ background-color: #1f2937; border-left: 4px solid #4b5563; }}
    div[data-testid="stMetric"] {{
        background-color: #1F2937; padding: 12px 16px; border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# ====================== FUNCIONES AUXILIARES ======================
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
        x=alt.X('Puntuación:Q', scale=alt.Scale(domain=[0, 10]), title=None),
        y=alt.Y('Métrica:N', sort='-x', title=None),
        color=alt.value(color_equipo),
        tooltip=['Métrica', 'Puntuación']
    ).properties(height=220)

    st.altair_chart(chart, use_container_width=True)

def crear_grafico(serie, titulo):
    serie = pd.Series(serie).dropna().astype(int)
    if len(serie) == 0:
        return None
    conteo = serie.value_counts().sort_index()
    df_c = pd.DataFrame({
        titulo: conteo.index.astype(str),
        'Prob (%)': (conteo / len(serie) * 100).round(1)
    })
    chart = alt.Chart(df_c).mark_bar(color=color_equipo).encode(
        x=alt.X(f'{titulo}:N', sort=None, title=titulo),
        y=alt.Y('Prob (%):Q', title='Probabilidad (%)'),
        tooltip=[f'{titulo}:N', 'Prob (%):Q']
    ).properties(height=280)
    return chart

def calcular_ev(prob, cuota):
    if cuota <= 1.0 or prob <= 0:
        return 0.0
    return round((prob / 100 * cuota) - 1, 4)

def calcular_kelly(prob, cuota):
    if cuota <= 1.0 or prob <= 0:
        return 0.0
    p = prob / 100.0
    b = cuota - 1.0
    if b <= 0:
        return 0.0
    kelly = (p * cuota - 1.0) / b
    # Half-Kelly (50% de la fórmula de Kelly) para mayor seguridad en el bankroll
    stake = max(0.0, kelly * 0.5 * 100)
    return round(stake, 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    icono = "✅ VALUE" if es_value else "❌ Sin valor"
    color_ev = "#10b981" if es_value else "#9ca3af"
    
    stake_kelly = calcular_kelly(prob, cuota_casa) if es_value else 0.0
    kelly_text = f" &nbsp;|&nbsp; Stake (Half-Kelly): <b>{stake_kelly}% del Bank</b>" if es_value else ""

    st.markdown(f"""
        <div class="value-box {clase}">
            <b>{nombre}</b><br>
            Prob: <b>{prob:.1f}%</b> &nbsp;|&nbsp; Justa: <b>{cuota_justa}</b> &nbsp;|&nbsp; Casa: <b>{cuota_casa}</b>{kelly_text}<br>
            <span style="color:{color_ev}; font-weight:bold; font-size:15px;">
                EV: {ev:+.2%} → {icono}
            </span>
        </div>
    """, unsafe_allow_html=True)

# ====================== TÍTULO + BOTÓN ======================
st.markdown("### 📊 GoalMetrics")
st.caption("Simulación Monte Carlo · Cuotas justas · Value Bets · Criterio de Kelly")

col_btn1, col_btn2, _ = st.columns([1.2, 1, 4])
with col_btn1:
    btn_analizar = st.button("⚡ Analizar", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🧹 Limpiar", use_container_width=True):
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

    # Cálculos
    hoy = pd.Timestamp.today().normalize()
    historial['Dias_Pasados'] = (hoy - pd.to_datetime(historial['Fecha'])).dt.days.replace(0, 0.1)
    historial['Peso'] = 1 / (1 + (historial['Dias_Pasados'] / 30))
    
    def prom(col):
        if col not in historial.columns:
            return 0.05
        return round(float(np.average(historial[col].fillna(0), weights=historial['Peso'])), 4)

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

    prob_over_goles = (sg_fav > linea_goles).mean() * 100
    prob_over_tiros = (s_tir > linea_tiros).mean() * 100
    prob_over_puerta = (s_tpuerta > linea_tiros_puerta).mean() * 100
    prob_over_corners = (s_corn > linea_corners).mean() * 100
    prob_over_faltas = (s_faltas > linea_faltas).mean() * 100
    
    marcadores = [f"{f}-{c}" for f, c in zip(sg_fav, sg_con)]
    conteo = Counter(marcadores)
    marcador_mas_comun = conteo.most_common(1)[0][0]
    
    if triunfos > 50:
        veredicto = f"Tendencia Fuerte · Marcador proyectado {marcador_mas_comun}"
    elif derrotas > 50:
        veredicto = f"Alerta de Complicación · Marcador proyectado {marcador_mas_comun}"
    else:
        veredicto = f"Partido Muy Parejo · Marcador proyectado {marcador_mas_comun}"

    # Cabecera
    st.markdown(f'<div class="header-box">🛡️ {equipo_sel.upper()} &nbsp;·&nbsp; {condicion_label.upper()} vs {nivel_sel.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="veredicto-box"><b>Veredicto:</b> {veredicto}</div>', unsafe_allow_html=True)
    st.caption(f"Base: {len(historial)} partidos · {fuente_datos}")

    # ====================== PESTAÑAS ======================
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumen", "💎 Value Bet", "📈 Líneas & Gráficos", "🔍 Detalle"])

    # ----- TAB 1: RESUMEN -----
    with tab1:
        st.subheader("🧬 ADN del Equipo")
        renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa)

        st.markdown("##### Probabilidades principales")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Victoria", f"{triunfos:.1f}%")
        c2.metric("🟡 Empate", f"{empates:.1f}%")
        c3.metric("🔴 Derrota", f"{derrotas:.1f}%")
        c4.metric("⚽ BTTS", f"{ambos_anotan:.1f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("1X", f"{doble_1x:.1f}%")
        c6.metric("X2", f"{doble_x2:.1f}%")
        c7.metric("DNB", f"{dnb:.1f}%")

        st.markdown("##### Expectativa del modelo")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Goles", f"{lam_f:.2f}")
        m2.metric("Goles Rival", f"{lam_c:.2f}")
        m3.metric("Tiros", f"{lam_t:.1f}")
        m4.metric("Córners", f"{lam_co:.1f}")

    # ----- TAB 2: VALUE BET -----
    with tab2:
        st.subheader("💎 Value Bet & Criterio de Kelly (Half-Kelly)")
        st.caption("El Stake te indica qué porcentaje de tu capital arriesgar de forma óptima según la ventaja matemática.")
        
        cuota_justa_1 = round(100 / triunfos, 2) if triunfos > 0 else 99.0
        cuota_justa_x = round(100 / empates, 2) if empates > 0 else 99.0
        cuota_justa_2 = round(100 / derrotas, 2) if derrotas > 0 else 99.0
        prob_btts_no = 100 - ambos_anotan
        cuota_justa_btts_si = round(100 / ambos_anotan, 2) if ambos_anotan > 0 else 99.0
        cuota_justa_btts_no = round(100 / prob_btts_no, 2) if prob_btts_no > 0 else 99.0
        cuota_justa_dnb = round(100 / dnb, 2) if dnb > 0 else 99.0

        ev_1 = calcular_ev(triunfos, cuota_casa_1)
        ev_x = calcular_ev(empates, cuota_casa_x)
        ev_2 = calcular_ev(derrotas, cuota_casa_2)
        ev_btts_si = calcular_ev(ambos_anotan, cuota_casa_btts_si)
        ev_btts_no = calcular_ev(prob_btts_no, cuota_casa_btts_no)
        ev_dnb = calcular_ev(dnb, cuota_casa_dnb)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            mostrar_value("Victoria (1)", cuota_justa_1, cuota_casa_1, ev_1, triunfos)
            mostrar_value("Empate (X)", cuota_justa_x, cuota_casa_x, ev_x, empates)
            mostrar_value("Derrota (2)", cuota_justa_2, cuota_casa_2, ev_2, derrotas)
        with col_v2:
            mostrar_value("BTTS Sí", cuota_justa_btts_si, cuota_casa_btts_si, ev_btts_si, ambos_anotan)
            mostrar_value("BTTS No", cuota_justa_btts_no, cuota_casa_btts_no, ev_btts_no, prob_btts_no)
            mostrar_value("DNB", cuota_justa_dnb, cuota_casa_dnb, ev_dnb, dnb)

        st.markdown("---")
        st.subheader("📈 Value Bet · Líneas (Over)")

        cuota_justa_og = round(100 / prob_over_goles, 2) if prob_over_goles > 0 else 99.0
        cuota_justa_ot = round(100 / prob_over_tiros, 2) if prob_over_tiros > 0 else 99.0
        cuota_justa_op = round(100 / prob_over_puerta, 2) if prob_over_puerta > 0 else 99.0
        cuota_justa_oc = round(100 / prob_over_corners, 2) if prob_over_corners > 0 else 99.0
        cuota_justa_of = round(100 / prob_over_faltas, 2) if prob_over_faltas > 0 else 99.0

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            mostrar_value(f"Over {linea_goles} Goles", cuota_justa_og, cuota_over_goles, calcular_ev(prob_over_goles, cuota_over_goles), prob_over_goles)
            mostrar_value(f"Over {linea_tiros} Tiros", cuota_justa_ot, cuota_over_tiros, calcular_ev(prob_over_tiros, cuota_over_tiros), prob_over_tiros)
            mostrar_value(f"Over {linea_tiros_puerta} a Puerta", cuota_justa_op, cuota_over_puerta, calcular_ev(prob_over_puerta, cuota_over_puerta), prob_over_puerta)
        with col_l2:
            mostrar_value(f"Over {linea_corners} Córners", cuota_justa_oc, cuota_over_corners, calcular_ev(prob_over_corners, cuota_over_corners), prob_over_corners)
            mostrar_value(f"Over {linea_faltas} Faltas", cuota_justa_of, cuota_over_faltas, calcular_ev(prob_over_faltas, cuota_over_faltas), prob_over_faltas)

    # ----- TAB 3: LÍNEAS & GRÁFICOS -----
    with tab3:
        st.subheader("📈 Probabilidades de Líneas")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.metric(f"Goles > {linea_goles}", f"{prob_over_goles:.1f}%")
        lc2.metric(f"Tiros > {linea_tiros}", f"{prob_over_tiros:.1f}%")
        lc3.metric(f"a Puerta > {linea_tiros_puerta}", f"{prob_over_puerta:.1f}%")
        lc4.metric(f"Córners > {linea_corners}", f"{prob_over_corners:.1f}%")
        lc5.metric(f"Faltas > {linea_faltas}", f"{prob_over_faltas:.1f}%")

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### ⚽ Distribución de Goles")
            chart_g = crear_grafico(pd.Series(sg_fav), 'Goles')
            if chart_g:
                st.altair_chart(chart_g, use_container_width=True)
        with col_g2:
            st.markdown("##### 🚩 Distribución de Córners")
            chart_c = crear_grafico(pd.Series(s_corn), 'Córners')
            if chart_c:
                st.altair_chart(chart_c, use_container_width=True)

        st.markdown("##### 🏆 Top 5 Marcadores más probables")
        st.dataframe(
            pd.DataFrame([
                {"Marcador": r, "Probabilidad": f"{(f/num_sim)*100:.1f}%"}
                for r, f in conteo.most_common(5)
            ]),
            hide_index=True,
            use_container_width=True
        )

    # ----- TAB 4: DETALLE -----
    with tab4:
        st.subheader("🔍 Partidos utilizados en el modelo")
        st.caption(f"Se analizaron {len(historial)} partidos · {fuente_datos}")
        
        h_disp = historial.copy().sort_values(by='Fecha', ascending=False)
        h_mostrar = h_disp.copy()
        h_mostrar['Fecha'] = pd.to_datetime(h_mostrar['Fecha']).dt.strftime('%Y-%m-%d')
        h_mostrar['Peso'] = h_mostrar['Peso'].round(3)
        
        cols = [c for c in ['Fecha', 'Condición', 'Rival', 'Nivel Rival',
                           'Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas', 'Peso']
                if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Parámetros λ del modelo Poisson")
        p1, p2, p3 = st.columns(3)
        p1.write(f"**Goles a favor:** {lam_f:.3f}")
        p1.write(f"**Goles en contra:** {lam_c:.3f}")
        p2.write(f"**Tiros:** {lam_t:.3f}")
        p2.write(f"**A puerta:** {lam_tp:.3f}")
        p3.write(f"**Córners:** {lam_co:.3f}")
        p3.write(f"**Faltas:** {lam_fa:.3f}")
