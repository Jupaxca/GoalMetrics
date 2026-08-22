import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GoalMetrics - Jugadores", layout="wide")

st.title("🎯 Centro de Análisis Individual de Jugadores")
st.markdown("Evaluación estadística con ponderación de forma reciente y tasas de acierto.")
st.markdown("---")

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
st.sidebar.header("🎯 Líneas de Estudio / Jugador")

if datos_ok and not df.empty and 'Jugador' in df.columns:
    jugadores = sorted(df['Jugador'].dropna().unique().tolist())
    jugador_sel = st.sidebar.selectbox("👤 Selecciona al Jugador", jugadores)
    
    condiciones = sorted(df['Condición'].dropna().unique().tolist()) if 'Condición' in df.columns else ["local", "visitante"]
    condicion_sel = st.sidebar.selectbox("📍 Condición", [c.capitalize() for c in condiciones])
    condicion_sel_lower = condicion_sel.lower()
    
    niveles = sorted(df['Nivel Rival'].dropna().unique().tolist()) if 'Nivel Rival' in df.columns else ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
    nivel_sel = st.sidebar.selectbox("⭐ Nivel del Rival", niveles)
else:
    jugador_sel = st.sidebar.selectbox("👤 Selecciona al Jugador", ["Sin datos"])
    condicion_sel = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
    condicion_sel_lower = condicion_sel.lower()
    nivel_sel = st.sidebar.selectbox("⭐ Nivel del Rival", ["DESCENSO"])

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Líneas a Evaluar")

linea_goles = st.sidebar.slider("⚽ Línea de Goles", 0.0, 3.0, 0.5, 0.5)
linea_tiros = st.sidebar.slider("🎯 Línea de Tiros Totales", 0.0, 10.0, 2.5, 0.5)
linea_puerta = st.sidebar.slider("🥅 Línea de Tiros a Puerta", 0.0, 5.0, 1.5, 0.5)
linea_asist = st.sidebar.slider("👟 Línea de Asistencias", 0.0, 2.0, 0.5, 0.5)
linea_faltas = st.sidebar.slider("⚠️ Línea de Faltas", 0.0, 5.0, 1.0, 0.5)

st.sidebar.markdown("---")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    analizar = st.button("⚡ Analizar", type="primary", use_container_width=True)
with col_b2:
    limpiar = st.button("🧹 Limpiar", use_container_width=True)

# ====================== FILTRADO ======================
if datos_ok and not df.empty and 'Jugador' in df.columns:
    df_jugador = df[df['Jugador'] == jugador_sel].copy()
    
    if 'Fecha' in df_jugador.columns:
        df_jugador = df_jugador.sort_values('Fecha')
    
    df_exactos = df_jugador[
        (df_jugador['Condición'] == condicion_sel_lower) & 
        (df_jugador['Nivel Rival'] == nivel_sel)
    ].copy()
else:
    df_jugador = pd.DataFrame()
    df_exactos = pd.DataFrame()

# ====================== LÓGICA PRINCIPAL ======================
if analizar:
    st.markdown("---")
    st.subheader(f"📈 Análisis Individual: **{jugador_sel}** ({condicion_sel} vs {nivel_sel})")
    
    total_partidos = len(df_jugador)
    goles_tot = df_jugador['Goles'].sum() if 'Goles' in df_jugador.columns else 0
    asist_tot = df_jugador['Asistencias'].sum() if 'Asistencias' in df_jugador.columns else 0
    tiros_prom = df_jugador['Tiros'].mean() if 'Tiros' in df_jugador.columns else 0
    puerta_prom = df_jugador['A Puerta'].mean() if 'A Puerta' in df_jugador.columns else 0
    faltas_prom = df_jugador['Faltas'].mean() if 'Faltas' in df_jugador.columns else 0
    amarillas_tot = df_jugador['Amarillas'].sum() if 'Amarillas' in df_jugador.columns else 0
    rojas_tot = df_jugador['Rojas'].sum() if 'Rojas' in df_jugador.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Goles Totales", int(goles_tot))
    k2.metric("Asistencias", int(asist_tot))
    k3.metric("Promedio Tiros", f"{tiros_prom:.1f}")
    k4.metric("Tiros a Puerta Prom.", f"{puerta_prom:.1f}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Promedio Faltas", f"{faltas_prom:.1f}")
    k6.metric("Amarillas", int(amarillas_tot))
    k7.metric("Rojas", int(rojas_tot))
    k8.metric("Partidos Registrados", total_partidos)

    st.markdown("---")
    st.subheader("📊 Simulación Monte Carlo (Ponderación de Forma Reciente)")

    historial = pd.DataFrame()
    fuente = ""

    if len(df_exactos) >= 2:
        historial = df_exactos.copy()
        fuente = f"Exacto ({condicion_sel} vs {nivel_sel}) — {len(historial)} partidos"
    elif len(df_exactos) == 1:
        historial = df_exactos.copy()
        if len(df_jugador) > 1:
            extra = df_jugador[df_jugador.index != df_exactos.index[0]].tail(1)
            historial = pd.concat([historial, extra])
            fuente = "1 partido exacto + 1 de respaldo (más reciente)"
        else:
            fuente = "Solo 1 partido exacto disponible"
    else:
        if len(df_jugador) >= 2:
            historial = df_jugador.tail(5).copy()
            fuente = f"Sin partidos exactos → usando últimos {len(historial)} partidos del jugador"
        else:
            st.warning("⚠️ No hay suficientes registros para este jugador.")
            st.stop()

    st.info(f"💡 Base del análisis: **{len(historial)} partidos** | Modo: {fuente}")

    if len(historial) >= 1:
        n_rows = len(historial)
        
        if 'Fecha' in historial.columns:
            hoy = pd.Timestamp.today()
            dias = (hoy - historial['Fecha']).dt.days.clip(lower=0.1)
            pesos = 1 / (1 + dias / 30)
        else:
            pesos = np.linspace(1.0, 3.0, n_rows)
        pesos = pesos / pesos.sum()

        n_simulaciones = 5000

        # Poisson cuando hay pocos partidos (evita solo 0% o 100%)
        if n_rows <= 3:
            lam_goles = np.average(historial['Goles'], weights=pesos)
            lam_tiros = np.average(historial['Tiros'], weights=pesos)
            lam_puerta = np.average(historial['A Puerta'], weights=pesos)
            lam_asist = np.average(historial['Asistencias'], weights=pesos)
            lam_faltas = np.average(historial['Faltas'], weights=pesos)

            sim_goles = np.random.poisson(lam=max(lam_goles, 0.05), size=n_simulaciones)
            sim_tiros = np.random.poisson(lam=max(lam_tiros, 0.05), size=n_simulaciones)
            sim_puerta = np.random.poisson(lam=max(lam_puerta, 0.05), size=n_simulaciones)
            sim_asist = np.random.poisson(lam=max(lam_asist, 0.05), size=n_simulaciones)
            sim_faltas = np.random.poisson(lam=max(lam_faltas, 0.05), size=n_simulaciones)
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

        real_goles = (historial['Goles'] > linea_goles).mean() * 100
        real_tiros = (historial['Tiros'] > linea_tiros).mean() * 100
        real_puerta = (historial['A Puerta'] > linea_puerta).mean() * 100
        real_asist = (historial['Asistencias'] > linea_asist).mean() * 100
        real_faltas = (historial['Faltas'] > linea_faltas).mean() * 100

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"⚽ **Más de {linea_goles} Goles**")
            st.markdown(f"### `{prob_goles:.1f}%`  \n*(Acierto real: {real_goles:.0f}%)*")
            
            st.markdown(f"🎯 **Más de {linea_tiros} Tiros Totales**")
            st.markdown(f"### `{prob_tiros:.1f}%`  \n*(Acierto real: {real_tiros:.0f}%)*")
            
            st.markdown(f"🥅 **Más de {linea_puerta} Tiros a Puerta**")
            st.markdown(f"### `{prob_puerta:.1f}%`  \n*(Acierto real: {real_puerta:.0f}%)*")
            
        with col_p2:
            st.markdown(f"👟 **Más de {linea_asist} Asistencias**")
            st.markdown(f"### `{prob_asist:.1f}%`  \n*(Acierto real: {real_asist:.0f}%)*")
            
            st.markdown(f"⚠️ **Más de {linea_faltas} Faltas**")
            st.markdown(f"### `{prob_faltas:.1f}%`  \n*(Acierto real: {real_faltas:.0f}%)*")

    # ===== RACHA =====
    st.markdown("---")
    st.subheader("⚡ Estado de Racha para el Siguiente Partido")
    
    contribuciones = goles_tot + asist_tot
    ratio_contribucion = total_partidos / contribuciones if contribuciones > 0 else 99
    
    racha_seca = 0
    if not df_jugador.empty and 'Goles' in df_jugador.columns and 'Asistencias' in df_jugador.columns:
        for _, row in df_jugador.iloc[::-1].iterrows():
            if row['Goles'] == 0 and row['Asistencias'] == 0:
                racha_seca += 1
            else:
                break

    if racha_seca >= ratio_contribucion:
        st.success(f"🔥 **ALTA PROBABILIDAD DE APORTE:** Lleva **{racha_seca} partidos** sin gol ni asistencia, superando su media de 1 contribución cada **{ratio_contribucion:.1f} partidos**.")
    else:
        st.info(f"ℹ️ **ESTADO NORMAL:** Lleva **{racha_seca} partidos** sin sumar. Su promedio indica una contribución cada **{ratio_contribucion:.1f} partidos**.")

    # ===== GRÁFICO =====
    if not df_jugador.empty:
        st.markdown("---")
        cols_graf = [c for c in ['Goles', 'Tiros', 'A Puerta', 'Asistencias'] if c in df_jugador.columns]
        if cols_graf and 'Nivel Rival' in df_jugador.columns:
            fig = px.bar(
                df_jugador,
                x='Nivel Rival',
                y=cols_graf,
                barmode='group',
                title=f"Historial de Rendimiento - {jugador_sel}",
                color_discrete_sequence=['#FF4B4B', '#00CC96', '#636EFA', '#FFA15A']
            )
            st.plotly_chart(fig, use_container_width=True)

elif limpiar:
    st.info("🧹 Filtros restablecidos. Vuelve a configurar y pulsa Analizar.")
else:
    st.info("👈 Configura las líneas en la barra lateral y pulsa **Analizar**.")
