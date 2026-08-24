import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data(ttl=600)
def cargar_datos_jugadores():
    url = "https://docs.google.com/spreadsheets/d/1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    if "Jugador" not in df.columns and "Equipo" in df.columns:
        df = df.rename(columns={"Equipo": "Jugador"})
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    for col in ["Goles", "Asistencias", "Tiros", "A Puerta", "Faltas", "Amarillas", "Rojas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "Condicion" in df.columns and "Condición" not in df.columns:
        df = df.rename(columns={"Condicion": "Condición"})
    if "Condición" in df.columns:
        df["Condición"] = df["Condición"].astype(str).str.strip().str.lower()
    return df

def shrinkage_lambda(lam_obs, lam_prior, n_obs, k=5.0):
    n = max(float(n_obs), 0.0)
    return (n * lam_obs + k * lam_prior) / (n + k)

try:
    df = cargar_datos_jugadores()
    datos_ok = True
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    datos_ok = False
    df = pd.DataFrame()

st.sidebar.header("Configuracion del Jugador")

if datos_ok and not df.empty and "Jugador" in df.columns:
    jugadores = sorted(df["Jugador"].dropna().unique().tolist())
    jugador_sel = st.sidebar.selectbox("Selecciona al Jugador", jugadores)
    df_jugador = df[df["Jugador"] == jugador_sel].copy()
    if "Fecha" in df_jugador.columns:
        df_jugador = df_jugador.sort_values("Fecha")
    if "Condición" in df_jugador.columns:
        condiciones = sorted(df_jugador["Condición"].dropna().unique().tolist())
    else:
        condiciones = ["local", "visitante"]
    condicion_sel = st.sidebar.selectbox("Condicion", [c.capitalize() for c in condiciones])
    condicion_sel_lower = condicion_sel.lower()
    if "Nivel Rival" in df_jugador.columns:
        niveles = sorted(df_jugador["Nivel Rival"].dropna().unique().tolist())
    else:
        niveles = ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
    nivel_sel = st.sidebar.selectbox("Nivel del Rival", niveles)
else:
    jugador_sel = st.sidebar.selectbox("Selecciona al Jugador", ["Sin datos"])
    condicion_sel = st.sidebar.selectbox("Condicion", ["Local", "Visitante"])
    condicion_sel_lower = condicion_sel.lower()
    nivel_sel = st.sidebar.selectbox("Nivel del Rival", ["DESCENSO"])
    df_jugador = pd.DataFrame()

st.sidebar.markdown("---")
with st.sidebar.expander("Lineas de Estudio (Player Props)", expanded=True):
    linea_goles = st.slider("Linea de Goles", 0.0, 3.0, 0.5, 0.5)
    linea_tiros = st.slider("Linea de Tiros Totales", 0.0, 10.0, 2.5, 0.5)
    linea_puerta = st.slider("Linea de Tiros a Puerta", 0.0, 5.0, 1.5, 0.5)
    linea_asist = st.slider("Linea de Asistencias", 0.0, 2.0, 0.5, 0.5)
    linea_faltas = st.slider("Linea de Faltas", 0.0, 5.0, 1.0, 0.5)
    linea_contrib = st.slider("Linea Gol o Asistencia", 0.0, 3.0, 0.5, 0.5)

with st.sidebar.expander("Cuotas de la Casa (Over / Props)"):
    cuota_casa_goles = st.number_input(
        f"Cuota Over {linea_goles} Goles", min_value=1.01, value=2.10, step=0.01, format="%.2f"
    )
    cuota_casa_tiros = st.number_input(
        f"Cuota Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f"
    )
    cuota_casa_puerta = st.number_input(
        f"Cuota Over {linea_puerta} a Puerta", min_value=1.01, value=1.90, step=0.01, format="%.2f"
    )
    cuota_casa_asist = st.number_input(
        f"Cuota Over {linea_asist} Asistencias", min_value=1.01, value=2.50, step=0.01, format="%.2f"
    )
    cuota_casa_faltas = st.number_input(
        f"Cuota Over {linea_faltas} Faltas", min_value=1.01, value=1.80, step=0.01, format="%.2f"
    )
    cuota_casa_contrib = st.number_input(
        f"Cuota Over {linea_contrib} Gol/Asist", min_value=1.01, value=1.70, step=0.01, format="%.2f"
    )

# ===== SHRINKAGE: se puede ON/OFF =====
if "chk_shrinkage_jug" not in st.session_state:
    st.session_state.chk_shrinkage_jug = True

with st.sidebar.expander("Modelo"):
    usar_shrinkage = st.checkbox(
        "Usar Shrinkage (estabilizar con media del jugador)",
        key="chk_shrinkage_jug",
    )
    if usar_shrinkage:
        k_shrink = st.slider(
            "Fuerza prior (k)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=1.0,
            key="slider_k_jug",
        )
        st.caption("ON: mezcla filtro + media historica del jugador.")
    else:
        k_shrink = 5.0
        st.caption("OFF: solo promedios del filtro actual (sin acercar a la media).")

    if st.button("Reset opciones modelo"):
        for k in ["chk_shrinkage_jug", "slider_k_jug"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

if not df_jugador.empty and "Fecha" in df_jugador.columns:
    df_diagnostico = df_jugador.sort_values(by="Fecha", ascending=False)
else:
    df_diagnostico = df_jugador

if (
    not df_diagnostico.empty
    and "Condición" in df_diagnostico.columns
    and "Nivel Rival" in df_diagnostico.columns
):
    exactos_check = df_diagnostico[
        (df_diagnostico["Condición"] == condicion_sel_lower)
        & (df_diagnostico["Nivel Rival"] == nivel_sel)
    ]
    num_exactos = len(exactos_check)
else:
    num_exactos = 0

if num_exactos >= 2:
    st.sidebar.success(f"{num_exactos} partidos exactos")
elif num_exactos == 1:
    st.sidebar.warning("1 partido exacto -> respaldo")
else:
    st.sidebar.error("0 partidos exactos")

st.markdown(
    """
<style>
.stApp { background-color: #0B0F19; color: #F3F4F6; }
.stSidebar { background-color: #111827; }
.header-box {
    background: linear-gradient(90deg, #3B82F6 0%, #1F2937 100%);
    padding: 22px 28px; border-radius: 14px; color: white;
    font-weight: 700; font-size: 24px; margin-bottom: 20px; text-align: center;
}
.value-box { padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 14px; }
.value-yes { background-color: #064e3b; border-left: 4px solid #10b981; }
.value-no { background-color: #1f2937; border-left: 4px solid #4b5563; }
div[data-testid="stMetric"] { background-color: #1F2937; padding: 12px 16px; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

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
    return round(max(0.0, ((p * cuota - 1.0) / b) * 0.5 * 100), 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, real=None):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake_kelly = calcular_kelly(prob, cuota_casa) if es_value else 0.0
    kelly_text = f" | Half-Kelly: <b>{stake_kelly}% bank</b>" if es_value else ""
    real_txt = f" | Acierto real: <b>{real:.0f}%</b>" if real is not None else ""
    st.markdown(
        f'<div class="value-box {clase}">'
        f"<b>{nombre}</b><br>"
        f"Modelo: <b>{prob:.1f}%</b>{real_txt} | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_text}<br>"
        f'<span style="color:{color_ev}; font-weight:bold; font-size:15px;">'
        f"EV: {ev:+.2%} -> {'VALUE BET' if es_value else 'Sin valor'}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("### Centro de Analisis Individual de Jugadores")
st.caption("Props orientativas. Contrasta modelo vs acierto real. Gestiona el bank.")

with st.expander("Como interpretar este analisis", expanded=False):
    st.markdown(
        """
### Configuracion y Modelo
En la barra lateral puedes ajustar como el modelo calcula las proyecciones:
- **Shrinkage:** combina partidos del filtro con la media historica del jugador. Puedes activarlo o desactivarlo.
- **Fuerza (k):** solo aplica si Shrinkage esta ON. Mas k = mas peso a la media historica.

### Metricas clave
- **Lambda:** promedio esperado por prop en este filtro.
- **Ratio de contribucion:** cada cuantos partidos aporta gol o asistencia.
- **Racha:** partidos seguidos recientes sin gol ni asistencia vs su media.

### Value Bet Props
- **Modelo %:** probabilidad de superar la linea en la simulacion.
- **Acierto real:** % historico en la muestra usada.
- **EV > 0:** la cuota de la casa es mejor que la justa del modelo.
- **Half-Kelly:** % orientativo del bank si hay value.
"""
    )

if "analizado_jugadores" not in st.session_state:
    st.session_state.analizado_jugadores = False

col_b1, col_b2, _ = st.columns([1.2, 1, 4])
with col_b1:
    if st.button("Analizar", type="primary", use_container_width=True):
        st.session_state.analizado_jugadores = True
with col_b2:
    if st.button("Limpiar", use_container_width=True):
        st.session_state.analizado_jugadores = False
        st.rerun()

if st.session_state.analizado_jugadores and datos_ok and not df.empty and "Jugador" in df.columns:
    df_jugador = df[df["Jugador"] == jugador_sel].copy()
    if "Fecha" in df_jugador.columns:
        df_jugador = df_jugador.sort_values("Fecha")

    if "Condición" in df_jugador.columns and "Nivel Rival" in df_jugador.columns:
        df_exactos = df_jugador[
            (df_jugador["Condición"] == condicion_sel_lower)
            & (df_jugador["Nivel Rival"] == nivel_sel)
        ].copy()
    else:
        df_exactos = pd.DataFrame()

    historial = pd.DataFrame()
    fuente = ""

    if len(df_exactos) >= 2:
        historial = df_exactos.copy()
        fuente = f"Exacto ({condicion_sel} vs {nivel_sel}) - {len(historial)} partidos"
    elif len(df_exactos) == 1:
        historial = df_exactos.copy()
        if len(df_jugador) > 1:
            extra = df_jugador[\~df_jugador.index.isin(historial.index)].tail(1)
            historial = pd.concat([historial, extra])
            fuente = "1 partido exacto + 1 de respaldo reciente"
        else:
            fuente = "Solo 1 partido exacto disponible"
    else:
        if len(df_jugador) >= 2:
            historial = df_jugador.tail(5).copy()
            fuente = f"Sin exactos -> ultimos {len(historial)} partidos"
        else:
            st.warning("No hay suficientes registros para este jugador.")
            st.stop()

    n_obs = len(historial)
    muestra_pequena = n_obs <= 3

    st.markdown(
        f'<div class="header-box">{jugador_sel.upper()} | {condicion_sel} vs {nivel_sel}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Base: {n_obs} partidos | {fuente}")

    if usar_shrinkage:
        st.caption(f"Modelo: Poisson + Shrinkage (k={k_shrink:.0f})")
    else:
        st.caption("Modelo: Poisson puro (Shrinkage desactivado)")

    if muestra_pequena:
        st.warning("Muestra pequena. Interpreta Kelly con cautela.")

    hoy = pd.Timestamp.today().normalize()
    if "Fecha" in historial.columns:
        historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.replace(0, 0.1)
        historial["Peso"] = 1 / (1 + (historial["Dias_Pasados"] / 30))
    else:
        historial["Peso"] = 1.0
    pesos = historial["Peso"] / historial["Peso"].sum()

    def prom_w(col):
        return float(np.average(historial[col].fillna(0), weights=pesos))

    lam_g_raw = prom_w("Goles")
    lam_t_raw = prom_w("Tiros")
    lam_p_raw = prom_w("A Puerta")
    lam_a_raw = prom_w("Asistencias")
    lam_f_raw = prom_w("Faltas")

    prior_g = float(df_jugador["Goles"].mean()) if len(df_jugador) else lam_g_raw
    prior_t = float(df_jugador["Tiros"].mean()) if len(df_jugador) else lam_t_raw
    prior_p = float(df_jugador["A Puerta"].mean()) if len(df_jugador) else lam_p_raw
    prior_a = float(df_jugador["Asistencias"].mean()) if len(df_jugador) else lam_a_raw
    prior_f = float(df_jugador["Faltas"].mean()) if len(df_jugador) else lam_f_raw

    if usar_shrinkage:
        lam_g = shrinkage_lambda(lam_g_raw, prior_g, n_obs, k_shrink)
        lam_t = shrinkage_lambda(lam_t_raw, prior_t, n_obs, k_shrink)
        lam_p = shrinkage_lambda(lam_p_raw, prior_p, n_obs, k_shrink)
        lam_a = shrinkage_lambda(lam_a_raw, prior_a, n_obs, k_shrink)
        lam_f = shrinkage_lambda(lam_f_raw, prior_f, n_obs, k_shrink)
    else:
        lam_g, lam_t, lam_p, lam_a, lam_f = (
            lam_g_raw,
            lam_t_raw,
            lam_p_raw,
            lam_a_raw,
            lam_f_raw,
        )

    n_sim = 10000
    rng = np.random.default_rng(42)
    sim_goles = rng.poisson(max(lam_g, 0.01), n_sim)
    sim_tiros = rng.poisson(max(lam_t, 0.01), n_sim)
    sim_puerta = rng.poisson(max(lam_p, 0.01), n_sim)
    sim_asist = rng.poisson(max(lam_a, 0.01), n_sim)
    sim_faltas = rng.poisson(max(lam_f, 0.01), n_sim)
    sim_contrib = sim_goles + sim_asist

    prob_goles = (sim_goles > linea_goles).mean() * 100
    prob_tiros = (sim_tiros > linea_tiros).mean() * 100
    prob_puerta = (sim_puerta > linea_puerta).mean() * 100
    prob_asist = (sim_asist > linea_asist).mean() * 100
    prob_faltas = (sim_faltas > linea_faltas).mean() * 100
    prob_contrib = (sim_contrib > linea_contrib).mean() * 100

    real_goles = (historial["Goles"] > linea_goles).mean() * 100
    real_tiros = (historial["Tiros"] > linea_tiros).mean() * 100
    real_puerta = (historial["A Puerta"] > linea_puerta).mean() * 100
    real_asist = (historial["Asistencias"] > linea_asist).mean() * 100
    real_faltas = (historial["Faltas"] > linea_faltas).mean() * 100
    real_contrib = ((historial["Goles"] + historial["Asistencias"]) > linea_contrib).mean() * 100

    total_partidos = len(df_jugador)
    goles_tot = df_jugador["Goles"].sum()
    asist_tot = df_jugador["Asistencias"].sum()
    tiros_prom = df_jugador["Tiros"].mean()
    puerta_prom = df_jugador["A Puerta"].mean()
    faltas_prom = df_jugador["Faltas"].mean()

    contribuciones = goles_tot + asist_tot
    ratio_contribucion = total_partidos / contribuciones if contribuciones > 0 else 99.0

    racha_seca = 0
    if "Fecha" in df_jugador.columns:
        df_ordenado = df_jugador.sort_values(by="Fecha", ascending=False)
    else:
        df_ordenado = df_jugador.iloc[::-1]
    for _, row in df_ordenado.iterrows():
        if row["Goles"] == 0 and row["Asistencias"] == 0:
            racha_seca += 1
        else:
            break

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Resumen y Racha", "Value Bet Props", "Probabilidades", "Detalle"]
    )

    with tab1:
        st.subheader("Metricas globales")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Goles totales", int(goles_tot))
        k2.metric("Asistencias", int(asist_tot))
        k3.metric("Prom. tiros", f"{tiros_prom:.1f}")
        k4.metric("Prom. a puerta", f"{puerta_prom:.1f}")
        k5, k6, k7 = st.columns(3)
        k5.metric("Prom. faltas", f"{faltas_prom:.1f}")
        k6.metric("Partidos", total_partidos)
        k7.metric("Ratio contribucion", f"1 / {ratio_contribucion:.1f}")

        st.markdown("##### Lambda del modelo")
        l1, l2, l3, l4, l5 = st.columns(5)
        if usar_shrinkage:
            l1.metric("Goles", f"{lam_g:.2f}", delta=f"raw {lam_g_raw:.2f}")
        else:
            l1.metric("Goles", f"{lam_g:.2f}")
        l2.metric("Tiros", f"{lam_t:.2f}")
        l3.metric("Puerta", f"{lam_p:.2f}")
        l4.metric("Asist", f"{lam_a:.2f}")
        l5.metric("Faltas", f"{lam_f:.2f}")

        st.markdown("---")
        st.subheader("Racha")
        if racha_seca >= ratio_contribucion:
            st.success(
                f"Alta probabilidad de aporte: {racha_seca} partidos seguidos sin gol ni asistencia "
                f"(media: 1 cada {ratio_contribucion:.1f})."
            )
        else:
            st.info(f"Estado normal: {racha_seca} partidos sin aporte directo.")

    with tab2:
        st.subheader("Value Bet y Half-Kelly")

        def cj(p):
            return round(100 / p, 2) if p > 0 else 99.0

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            mostrar_value(
                f"Over {linea_goles} Goles",
                cj(prob_goles),
                cuota_casa_goles,
                calcular_ev(prob_goles, cuota_casa_goles),
                prob_goles,
                real_goles,
            )
            mostrar_value(
                f"Over {linea_tiros} Tiros",
                cj(prob_tiros),
                cuota_casa_tiros,
                calcular_ev(prob_tiros, cuota_casa_tiros),
                prob_tiros,
                real_tiros,
            )
            mostrar_value(
                f"Over {linea_puerta} a Puerta",
                cj(prob_puerta),
                cuota_casa_puerta,
                calcular_ev(prob_puerta, cuota_casa_puerta),
                prob_puerta,
                real_puerta,
            )
        with col_v2:
            mostrar_value(
                f"Over {linea_asist} Asistencias",
                cj(prob_asist),
                cuota_casa_asist,
                calcular_ev(prob_asist, cuota_casa_asist),
                prob_asist,
                real_asist,
            )
            mostrar_value(
                f"Over {linea_faltas} Faltas",
                cj(prob_faltas),
                cuota_casa_faltas,
                calcular_ev(prob_faltas, cuota_casa_faltas),
                prob_faltas,
                real_faltas,
            )
            mostrar_value(
                f"Over {linea_contrib} Gol o Asist",
                cj(prob_contrib),
                cuota_casa_contrib,
                calcular_ev(prob_contrib, cuota_casa_contrib),
                prob_contrib,
                real_contrib,
            )

    with tab3:
        st.subheader("Probabilidades del modelo")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric(f"Goles > {linea_goles}", f"{prob_goles:.1f}%", delta=f"Real {real_goles:.0f}%")
        lc2.metric(f"Tiros > {linea_tiros}", f"{prob_tiros:.1f}%", delta=f"Real {real_tiros:.0f}%")
        lc3.metric(f"a Puerta > {linea_puerta}", f"{prob_puerta:.1f}%", delta=f"Real {real_puerta:.0f}%")
        lc4, lc5, lc6 = st.columns(3)
        lc4.metric(f"Asist > {linea_asist}", f"{prob_asist:.1f}%", delta=f"Real {real_asist:.0f}%")
        lc5.metric(f"Faltas > {linea_faltas}", f"{prob_faltas:.1f}%", delta=f"Real {real_faltas:.0f}%")
        lc6.metric(
            f"Gol/Asist > {linea_contrib}", f"{prob_contrib:.1f}%", delta=f"Real {real_contrib:.0f}%"
        )

        cols_graf = [c for c in ["Goles", "Tiros", "A Puerta", "Asistencias"] if c in df_jugador.columns]
        if cols_graf:
            fig = px.bar(
                df_jugador,
                x="Fecha" if "Fecha" in df_jugador.columns else df_jugador.index,
                y=cols_graf,
                barmode="group",
                title=f"Historial - {jugador_sel}",
                color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"],
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Partidos usados")
        h_mostrar = historial.copy()
        if "Fecha" in h_mostrar.columns:
            h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
        if "Peso" in h_mostrar.columns:
            h_mostrar["Peso"] = h_mostrar["Peso"].round(3)
        cols = [
            c
            for c in [
                "Fecha",
                "Condición",
                "Rival",
                "Nivel Rival",
                "Goles",
                "Asistencias",
                "Tiros",
                "A Puerta",
                "Faltas",
                "Peso",
            ]
            if c in h_mostrar.columns
        ]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)
        if usar_shrinkage:
            st.caption(f"Prior = media del jugador | k={k_shrink:.0f} | n_filtro={n_obs}")
        else:
            st.caption("Shrinkage OFF: lambdas = promedio ponderado solo del filtro")

else:
    st.info("Elige jugador, lineas y cuotas, luego Analizar.")
