import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="GoalMetrics - Jugadores", layout="wide")

st.title("🎯 Centro de Análisis Individual de Jugadores")
st.markdown("Evaluación estadística de rendimiento personal, rachas y líneas de estudio.")
st.markdown("---")

@st.cache_data
def cargar_datos_jugadores():
    url = "https://docs.google.com/spreadsheets/d/1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

try:
    df = cargar_datos_jugadores()
    datos_ok = True
except:
    datos_ok = False
    df = pd.DataFrame()

# Sidebar - Líneas de Estudio / Apuesta y Filtros Individuales
st.sidebar.header("🎯 Líneas de Estudio / Jugador")

if datos_ok and not df.empty:
    jugadores = df['Equipo'].unique().tolist()
    jugador_sel = st.sidebar.selectbox("👤 Selecciona al Jugador", jugadores)
    
    condiciones = df['Condición'].unique().tolist() if 'Condición' in df.columns else ["Local", "Visitante"]
    condicion_sel = st.sidebar.selectbox("📍 Condición", condiciones)
    
    niveles = df['Nivel Rival'].unique().tolist() if 'Nivel Rival' in df.columns else ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
    nivel_sel = st.sidebar.selectbox("⭐ Nivel del Rival", niveles)
else:
    jugador_sel = st.sidebar.selectbox("👤 Selecciona al Jugador", ["Tzolis", "Odegard"])
    condicion_sel = st.sidebar.selectbox("📍 Condición", ["Local", "Visitante"])
    nivel_sel = st.sidebar.selectbox("⭐ Nivel del Rival", ["DESCENSO"])

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Sliders de Líneas a Evaluar")

# Sliders idénticos al estilo de líneas de apuesta/estudio
linea_goles = st.sidebar.slider("⚽ Línea de Goles", 0.0, 3.0, 0.5, 0.5)
linea_tiros = st.sidebar.slider("🎯 Línea de Tiros Totales", 0.0, 10.0, 2.5, 0.5)
linea_puerta = st.sidebar.slider("🥅 Línea de Tiros a Puerta", 0.0, 5.0, 1.5, 0.5)
linea_asist = st.sidebar.slider("👟 Línea de Asistencias", 0.0, 2.0, 0.5, 0.5)
linea_faltas = st.sidebar.slider("⚠️ Línea de Faltas", 0.0, 5.0, 1.0, 0.5)

st.sidebar.markdown("---")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    analizar = st.button("⚡ Analizar", type="primary")
with col_b2:
    limpiar = st.button("🧹 Limpiar")

# Filtrar datos del jugador seleccionado en el Google Sheet
if datos_ok and not df.empty:
    df_jugador = df[df['Equipo'] == jugador_sel]
    df_filtrado = df_jugador[
        (df_jugador['Condición'] == condicion_sel) & 
        (df_jugador['Nivel Rival'] == nivel_sel)
    ]
else:
    df_jugador = pd.DataFrame()
    df_filtrado = pd.DataFrame()

# Lógica al presionar Analizar
if analizar:
    st.markdown("---")
    st.subheader(f"📈 Análisis Estadístico Individual: {jugador_sel} ({condicion_sel}) vs {nivel_sel}")
    
    # Métricas puramente personales
    total_partidos = len(df_jugador)
    goles_tot = df_jugador['Goles'].sum() if not df_jugador.empty and 'Goles' in df_jugador.columns else 0
    asist_tot = df_jugador['Asistencias'].sum() if not df_jugador.empty and 'Asistencias' in df_jugador.columns else 0
    tiros_prom = df_jugador['Tiros'].mean() if not df_jugador.empty and 'Tiros' in df_jugador.columns else 0
    puerta_prom = df_jugador['A Puerta'].mean() if not df_jugador.empty and 'A Puerta' in df_jugador.columns else 0
    faltas_prom = df_jugador['Faltas'].mean() if not df_jugador.empty and 'Faltas' in df_jugador.columns else 0
    amarillas_tot = df_jugador['Amarillas'].sum() if not df_jugador.empty and 'Amarillas' in df_jugador.columns else 0
    rojas_tot = df_jugador['Rojas'].sum() if not df_jugador.empty and 'Rojas' in df_jugador.columns else 0

    # Tarjetas de Estadísticas Individuales (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Goles Totales", int(goles_tot))
    k2.metric("Asistencias", int(asist_tot))
    k3.metric("Promedio de Tiros", f"{tiros_prom:.1f}")
    k4.metric("Tiros a Puerta Prom.", f"{puerta_prom:.1f}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Promedio de Faltas", f"{faltas_prom:.1f}")
    k6.metric("Amarillas", int(amarillas_tot))
    k7.metric("Rojas", int(rojas_tot))
    k8.metric("Partidos Registrados", total_partidos)

    # Motor de Racha y Probabilidad para el siguiente partido
    st.markdown("---")
    st.subheader("⚡ Probabilidad y Análisis de Racha para el Siguiente Partido")
    
    contribuciones = goles_tot + asist_tot
    ratio_contribucion = total_partidos / contribuciones if contribuciones > 0 else 99
    
    # Conteo de partidos consecutivos sin marcar ni asistir
    racha_seca = 0
    if not df_jugador.empty:
        for _, row in reversed(list(df_jugador.iterrows())):
            if row['Goles'] == 0 and row['Asistencias'] == 0:
                racha_seca += 1
            else:
                break

    # Definir si la probabilidad es alta por acumulación de partidos sin sumar
    if racha_seca >= ratio_contribucion:
        st.success(f"🔥 **ALTA PROBABILIDAD DE APORTE:** El jugador acumula **{racha_seca} partidos** sin sumar gol ni asistencia, superando su media histórica de una contribución cada **{ratio_contribucion:.1f} partidos**.")
    else:
        st.info(f"ℹ️ **ESTADO NORMAL:** Lleva **{racha_seca} partidos** sin sumar. Su promedio indica una contribución cada **{ratio_contribucion:.1f} partidos**.")

    # Gráfica de rendimiento individual por nivel de rival
    if not df_jugador.empty:
        st.markdown("---")
        fig = px.bar(
            df_jugador,
            x='Nivel Rival',
            y=['Goles', 'Tiros', 'A Puerta', 'Asistencias'],
            barmode='group',
            title=f"Historial de Rendimiento Personal - {jugador_sel}",
            color_discrete_sequence=['#FF4B4B', '#00CC96', '#636EFA', '#FFA15A']
        )
        st.plotly_chart(fig, use_container_width=True)

elif limpiar:
    st.info("🧹 Los filtros y líneas han sido restablecidos.")
else:
    st.info("👈 Configura las líneas de estudio en la barra lateral y presiona **Analizar** para evaluar las estadísticas individuales.")
