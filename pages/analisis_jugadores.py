import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ====================== CARGA DE DATOS ======================
@st.cache_data(ttl=600)
def cargar_datos_jugadores():
    url = "https://docs.google.com/spreadsheets/d/1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    
    if 'Jugador' not in df.columns and 'Equipo' in df.columns:
        df = df.rename(columns={'Equipo': 'Jugador'})
    
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    
    for col in ['Goles', 'Asistencias', 'Tiros', 'A Puerta', 'Faltas', 'Amarillas', 'Rojas']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'Condición' in df.columns:
        df['Condición'] = df['Condición'].astype(str).str.strip().str.lower()
    
    return df

try:
    df = cargar_datos_jugadores()
    datos_ok = True
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    datos_ok = False
    df = pd.DataFrame()

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Configuración del Jugador")

if datos_ok and not df.empty and 'Jugador' in df.columns:
    jugadores = sorted(df['Jugador'].dropna().unique().tolist())
    jugador_sel = st.sidebar.selectbox("👤 Selecciona al Jugador", jugadores)
    
    df_jugador = df[df['Jugador'] == jugador_sel].copy()
    if 'Fecha' in df_jugador.columns:
        df_jugador = df_jugador.sort_values('Fecha')
    
    condiciones = sorted(df_jugador['Condición'].dropna().unique().tolist()) if 'Condición' in df_jugador.columns else ["local", "visitante"]
    condicion_sel = st.sidebar.selectbox("📍 Condición", [c.capitalize() for c in condiciones])
    condicion_sel_lower = condicion_sel.lower()
    
    niveles = sorted(df_jugador['Nivel Rival'].dropna().unique().tolist()) if 'Nivel Rival' in df_jugador.columns else ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
    nivel_sel = st.sidebar.selectbox("⭐ Nivel del Rival", niveles)
else:
    jugador_sel = st.sidebar.selectbox("👤 Selecciona al Jugador", ["Sin datos"])
    condicion_sel = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
    condicion_sel_lower = condicion_sel.lower()
    nivel_sel = st.sidebar.selectbox("⭐ Nivel del Rival", ["DESCENSO"])
    df_jugador = pd.DataFrame()

st.sidebar.markdown("---")
with st.sidebar.expander("🎯 Líneas de Estudio (Player Props)", expanded=True):
    linea_goles = st.slider("⚽ Línea de Goles", 0.0, 3.0, 0.5, 0.5)
    linea_tiros = st.slider("🎯 Línea de Tiros Totales", 0.0, 10.0, 2.5, 0.5)
    linea_puerta = st.slider("🥅 Línea de Tiros a Puerta", 0.0, 5.0, 1.5, 0.5)
    linea_asist = st.slider("👟 Línea de Asistencias", 0.0, 2.0, 0.5, 0.5)
    linea_faltas = st.slider("⚠️ Línea de Faltas", 0.0, 5.0, 1.0, 0.5)

with st.sidebar.expander("💰 Cuotas de la Casa (Over / Props)"):
    cuota_casa_goles = st.number_input(f"Cuota Over {linea_goles} Goles", min_value=1.01, value=2.10, step=0.01, format="%.2f")
    cuota_casa_tiros = st.number_input(f"Cuota Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_casa_puerta = st.number_input(f"Cuota Over {linea_puerta} a Puerta", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_casa_asist = st.number_input(f"Cuota Over {linea_asist} Asistencias", min_value=1.01, value=2.50, step=0.01, format="%.2f")
    cuota_casa_faltas = st.number_input(f"Cuota Over {linea_faltas} Faltas", min_value=1.01, value=1.80, step=0.01, format="%.2f")

# Diagnóstico de partidos exactos
df_diagnostico = df_jugador.sort_values(by='Fecha', ascending=False) if 'Fecha' in df_jugador.columns and not df_jugador.empty else df_jugador
if not df_diagnostico.empty:
    exactos_check = df_diagnostico[
        (df_diagnostico['Condición'] == condicion_sel_lower) & 
        (df_diagnostico['Nivel Rival'] == nivel_sel)
    ]
    num_exactos = len(exactos_check)
else:
    num_exactos = 0

if num_exactos >= 2:
    st.sidebar.success(f"✅ {num_exactos} partidos exactos")
elif num_exactos == 1:
    st.sidebar.warning("⚠️ 1 partido exacto → se usará respaldo")
else:
    st.sidebar.error("❌ 0 partidos exactos")

# ====================== CSS ======================
st.markdown("""
    <style>
    .stApp { background-color: #0B0F19; color: #F3F4F6; }
    .stSidebar { background-color: #111827; }
    .header-box {
        background: linear-gradient(90deg, #3B82F6 0%, #1F2937 100%);
        padding: 22px 28px; border-radius: 14px; color: white;
        font-weight: 700; font-size: 24px; margin-bottom: 20px;
        text-align: center; letter-spacing: 0.5px;
    }
    .veredicto-box {
        padding: 16px 20px; border-radius: 12px; background-color: #1F2937;
        border-left: 5px solid #3B82F6; margin-bottom: 24px;
        font-size: 16px; line-height: 1.5;
    }
    .value-box {
        padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 14px;
    }
    .value-yes { background-color: #064e3b; border-left: 4px solid #10b981; }
    .value-no { background-color: #1f2937; border-left: 4px solid #4b5563; }
    div[data-testid="stMetric"] {
        background-color: #1F2937; padding: 12px 16px; border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== FUNCIONES AUXILIARES ======================
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
    stake = max(0.0, kelly * 0.5 * 100)  # Half-Kelly
    return round(stake, 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    
    stake_kelly = calcular_kelly(prob, cuota_casa) if es_value else 0.0
    kelly_text = f" &nbsp;|&nbsp; Stake (Half-Kelly): <b>{stake_kelly}% del Bank</b>" if es_value else ""

    st.markdown(f"""
        <div class="value-box {clase}">
            <b>{nombre}</b><br>
            Prob. Modelo: <b>{prob:.1f}%</b> &nbsp;|&nbsp; Justa: <b>{cuota_justa}</b> &nbsp;|&nbsp; Casa: <b>{cuota_casa}</b>{kelly_text}<br>
            <span style="color:{color_ev}; font-weight:bold; font-size:15px;">
                EV: {ev:+.2%} → {'✅ VALUE BET' if es_value else '❌ Sin valor'}
            </span>
        </div>
    """, unsafe_allow_html=True)

# ====================== TÍTULO + BOTÓN ======================
st.markdown("### 🎯 Centro de Análisis Individual de Jugadores")
st.caption("Evaluación estadística · Simulación con pesos temporales · Value Bets · Criterio de Kelly")

col_b1, col_b2, _ = st.columns([1.2, 1, 4])
with col_b1:
    analizar = st.button("⚡ Analizar", type="primary", use_container_width=True)
with col_b2:
    limpiar = st.button("🧹 Limpiar", use_container_width=True)

# ====================== FILTRADO Y LÓGICA ======================
if analizar and datos_ok and not df.empty and 'Jugador' in df.columns:
    df_jugador = df[df['Jugador'] == jugador_sel].copy()
    
    # Orden por fecha (más antiguo → más reciente)
    if 'Fecha' in df_jugador.columns:
        df_jugador = df_jugador.sort_values('Fecha')
    
    df_exactos = df_jugador[
        (df_jugador['Condición'] == condicion_sel_lower) & 
        (df_jugador['Nivel Rival'] == nivel_sel)
    ].copy()
    
    historial = pd.DataFrame()
    fuente = ""

    if len(df_exactos) >= 2:
        historial = df_exactos.copy()
        fuente = f"Exacto ({condicion_sel} vs {nivel_sel}) — {len(historial)} partidos"
    elif len(df_exactos) == 1:
        historial = df_exactos.copy()
        if len(df_jugador) > 1:
            # --- CORRECCIÓN APLICADA AQUÍ (Se eliminó la barra invertida \) ---
            extra = df_jugador[~df_jugador.index.isin(historial.index)].tail(1)
            historial = pd.concat([historial, extra])
            fuente = "1 partido exacto + 1 de respaldo reciente"
        else:
            fuente = "Solo 1 partido exacto disponible"
    else:
        if len(df_jugador) >= 2:
            historial = df_jugador.tail(5).copy()
            fuente = f"Sin partidos exactos → últimos {len(historial)} partidos del jugador"
        else:
            st.warning("⚠️ No hay suficientes registros para este jugador.")
            st.stop()

    # Cabecera
    st.markdown(f'<div class="header-box">👤 {jugador_sel.upper()} &nbsp;·&nbsp; {condicion_sel} vs {nivel_sel}</div>', unsafe_allow_html=True)
    st.caption(f"Base del análisis: {len(historial)} partidos · Modo: {fuente}")

    # Pesos temporales
    hoy = pd.Timestamp.today().normalize()
    if 'Fecha' in historial.columns:
        historial['Dias_Pasados'] = (hoy - pd.to_datetime(historial['Fecha'])).dt.days.replace(0, 0.1)
        historial['Peso'] = 1 / (1 + (historial['Dias_Pasados'] / 30))
    else:
        historial['Peso'] = 1.0
    
    pesos = historial['Peso'] / historial['Peso'].sum()
    n_simulaciones = 10000

    # Motor de simulación
    if len(historial) <= 3:
        lam_goles = np.average(historial['Goles'], weights=pesos)
        lam_tiros = np.average(historial['Tiros'], weights=pesos)
        lam_puerta = np.average(historial['A Puerta'], weights=pesos)
        lam_asist = np.average(historial['Asistencias'], weights=pesos)
        lam_faltas = np.average(historial['Faltas'], weights=pesos)

        sim_goles = np.random.poisson(lam=max(lam_goles, 0.01), size=n_simulaciones)
        sim_tiros = np.random.poisson(lam=max(lam_tiros, 0.01), size=n_simulaciones)
        sim_puerta = np.random.poisson(lam=max(lam_puerta, 0.01), size=n_simulaciones)
        sim_asist = np.random.poisson(lam=max(lam_asist, 0.01), size=n_simulaciones)
        sim_faltas = np.random.poisson(lam=max(lam_faltas, 0.01), size=n_simulaciones)
    else:
        sim_goles = np.random.choice(historial['Goles'].values, size=n_simulaciones, replace=True, p=pesos)
        sim_tiros = np.random.choice(historial['Tiros'].values, size=n_simulaciones, replace=True, p=pesos)
        sim_puerta = np.random.choice(historial['A Puerta'].values, size=n_simulaciones, replace=True, p=pesos)
        sim_asist = np.random.choice(historial['Asistencias'].values, size=n_simulaciones, replace=True, p=pesos)
        sim_faltas = np.random.choice(historial['Faltas'].values, size=n_simulaciones, replace=True, p=pesos)

    prob_goles = (sim_goles > linea_goles).mean() * 100
    prob_tiros = (sim_tiros > linea_tiros).mean() * 100
    prob_puerta = (sim_puerta > linea_puerta).mean() * 100
    prob_asist = (sim_asist > linea_asist).mean() * 100
    prob_faltas = (sim_faltas > linea_faltas).mean() * 100

    # Estadísticas globales
    total_partidos = len(df_jugador)
    goles_tot = df_jugador['Goles'].sum()
    asist_tot = df_jugador['Asistencias'].sum()
    tiros_prom = df_jugador['Tiros'].mean()
    puerta_prom = df_jugador['A Puerta'].mean()
    faltas_prom = df_jugador['Faltas'].mean()

    # ===== RACHA CORREGIDA (de más reciente a más antiguo) =====
    contribuciones = goles_tot + asist_tot
    ratio_contribucion = total_partidos / contribuciones if contribuciones > 0 else 99
    
    racha_seca = 0
    if 'Fecha' in df_jugador.columns:
        df_ordenado = df_jugador.sort_values(by='Fecha', ascending=False)
    else:
        df_ordenado = df_jugador.iloc[::-1]
    
    for _, row in df_ordenado.iterrows():
        if row['Goles'] == 0 and row['Asistencias'] == 0:
            racha_seca += 1
        else:
            break  # para al primer partido con aporte

    # ====================== PESTAÑAS ======================
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumen & Racha", "💎 Value Bet Props", "📈 Probabilidades", "🔍 Detalle Histórico"])

    with tab1:
        st.subheader("📊 Métricas Globales del Jugador")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Goles Totales", int(goles_tot))
        k2.metric("Asistencias", int(asist_tot))
        k3.metric("Promedio Tiros", f"{tiros_prom:.1f}")
        k4.metric("Tiros a Puerta Prom.", f"{puerta_prom:.1f}")

        k5, k6, k7 = st.columns(3)
        k5.metric("Promedio Faltas", f"{faltas_prom:.1f}")
        k6.metric("Partidos Registrados", total_partidos)
        k7.metric("Ratio Contribución", f"1 cada {ratio_contribucion:.1f} part.")

        st.markdown("---")
        st.subheader("⚡ Estado de Racha")
        st.caption("Calculada sobre todos los partidos del jugador (de más reciente a más antiguo).")
        if racha_seca >= ratio_contribucion:
            st.success(f"🔥 **ALTA PROBABILIDAD DE APORTE:** Lleva **{racha_seca} partidos** consecutivos sin gol ni asistencia, superando su media histórica.")
        else:
            st.info(f"ℹ️ **ESTADO NORMAL:** Lleva **{racha_seca} partidos** sin sumar aportes directos.")

    with tab2:
        st.subheader("💎 Value Bet & Criterio de Kelly (Player Props)")
        st.caption("Cruce de probabilidades de simulación contra las cuotas configuradas en el panel lateral.")

        cuota_justa_g = round(100 / prob_goles, 2) if prob_goles > 0 else 99.0
        cuota_justa_t = round(100 / prob_tiros, 2) if prob_tiros > 0 else 99.0
        cuota_justa_p = round(100 / prob_puerta, 2) if prob_puerta > 0 else 99.0
        cuota_justa_a = round(100 / prob_asist, 2) if prob_asist > 0 else 99.0
        cuota_justa_f = round(100 / prob_faltas, 2) if prob_faltas > 0 else 99.0

        ev_g = calcular_ev(prob_goles, cuota_casa_goles)
        ev_t = calcular_ev(prob_tiros, cuota_casa_tiros)
        ev_p = calcular_ev(prob_puerta, cuota_casa_puerta)
        ev_a = calcular_ev(prob_asist, cuota_casa_asist)
        ev_f = calcular_ev(prob_faltas, cuota_casa_faltas)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            mostrar_value(f"Over {linea_goles} Goles", cuota_justa_g, cuota_casa_goles, ev_g, prob_goles)
            mostrar_value(f"Over {linea_tiros} Tiros Totales", cuota_justa_t, cuota_casa_tiros, ev_t, prob_tiros)
            mostrar_value(f"Over {linea_puerta} Tiros a Puerta", cuota_justa_p, cuota_casa_puerta, ev_p, prob_puerta)
        with col_v2:
            mostrar_value(f"Over {linea_asist} Asistencias", cuota_justa_a, cuota_casa_asist, ev_a, prob_asist)
            mostrar_value(f"Over {linea_faltas} Faltas", cuota_justa_f, cuota_casa_faltas, ev_f, prob_faltas)

    with tab3:
        st.subheader("📈 Tasas de Acierto Proyectadas")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.metric(f"Goles > {linea_goles}", f"{prob_goles:.1f}%")
        lc2.metric(f"Tiros > {linea_tiros}", f"{prob_tiros:.1f}%")
        lc3.metric(f"a Puerta > {linea_puerta}", f"{prob_puerta:.1f}%")
        lc4.metric(f"Asist > {linea_asist}", f"{prob_asist:.1f}%")
        lc5.metric(f"Faltas > {linea_faltas}", f"{prob_faltas:.1f}%")

        st.markdown("---")
        cols_graf = [c for c in ['Goles', 'Tiros', 'A Puerta', 'Asistencias'] if c in df_jugador.columns]
        if cols_graf and 'Nivel Rival' in df_jugador.columns:
            fig = px.bar(
                df_jugador,
                x='Fecha' if 'Fecha' in df_jugador.columns else df_jugador.index,
                y=cols_graf,
                barmode='group',
                title=f"Historial de Rendimiento - {jugador_sel}",
                color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔍 Partidos de Referencia Utilizados")
        h_mostrar = historial.copy()
        if 'Fecha' in h_mostrar.columns:
            h_mostrar['Fecha'] = pd.to_datetime(h_mostrar['Fecha']).dt.strftime('%Y-%m-%d')
        h_mostrar['Peso'] = h_mostrar['Peso'].round(3) if 'Peso' in h_mostrar.columns else 1.0
        
        cols = [c for c in ['Fecha', 'Condición', 'Rival', 'Nivel Rival',
                           'Goles', 'Asistencias', 'Tiros', 'A Puerta', 'Faltas', 'Peso']
                if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)

elif limpiar:
    st.info("🧹 Filtros restablecidos.")
else:
    st.info("👈 Selecciona un jugador en la barra lateral, configura tus líneas y cuotas, luego haz clic en **Analizar**.")
