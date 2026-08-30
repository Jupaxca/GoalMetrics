import streamlit as st
import pandas as pd
import numpy as np

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
    if "Nivel Rival" in df.columns:
        df["Nivel Rival"] = df["Nivel Rival"].astype(str).str.strip()
    return df

def shrinkage_lambda(lam_obs, lam_prior, n_obs, k=5.0):
    n = max(float(n_obs), 0.0)
    return (n * lam_obs + k * lam_prior) / (n + k)

def obtener_peso_tier(tier):
    t = str(tier).upper().strip()
    if "TOP" in t or "CHAMPIONS" in t:
        return 3
    elif "MEDIA" in t:
        return 2
    elif "DESCENSO" in t or "BAJO" in t:
        return 1
    else:
        return 2

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
with st.sidebar.expander("Lineas de Estudio (Player Props)", expanded=False):
    linea_goles = st.slider("Linea de Goles", 0.0, 3.0, 0.5, 0.5)
    linea_tiros = st.slider("Linea de Tiros Totales", 0.0, 10.0, 2.5, 0.5)
    linea_puerta = st.slider("Linea de Tiros a Puerta", 0.0, 5.0, 1.5, 0.5)
    linea_asist = st.slider("Linea de Asistencias", 0.0, 2.0, 0.5, 0.5)
    linea_faltas = st.slider("Linea de Faltas", 0.0, 5.0, 1.0, 0.5)
    linea_contrib = st.slider("Linea Gol o Asistencia", 0.0, 3.0, 0.5, 0.5)

with st.sidebar.expander("Cuotas de la Casa (Over / Props)", expanded=False):
    cuota_casa_goles = st.number_input(f"Cuota Over {linea_goles} Goles", min_value=1.01, value=2.10, step=0.01, format="%.2f")
    cuota_casa_tiros = st.number_input(f"Cuota Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_casa_puerta = st.number_input(f"Cuota Over {linea_puerta} a Puerta", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_casa_asist = st.number_input(f"Cuota Over {linea_asist} Asistencias", min_value=1.01, value=2.50, step=0.01, format="%.2f")
    cuota_casa_faltas = st.number_input(f"Cuota Over {linea_faltas} Faltas", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_casa_contrib = st.number_input(f"Cuota Over {linea_contrib} Gol/Asist", min_value=1.01, value=1.70, step=0.01, format="%.2f")

with st.sidebar.expander("Modelo", expanded=False):
    shrink_opt = st.radio("Shrinkage", options=["ON", "OFF"], index=0, horizontal=True, key="radio_shrink_jug")
    usar_shrinkage = (shrink_opt == "ON")
    k_shrink = st.slider("Fuerza prior (k)", 1.0, 15.0, 5.0, 1.0, key="slider_k_jug", disabled=not usar_shrinkage)

# Comprobación de partidos exactos
if not df_jugador.empty and "Condición" in df_jugador.columns and "Nivel Rival" in df_jugador.columns:
    exactos_check = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] == nivel_sel)]
    num_exactos = len(exactos_check)
else:
    num_exactos = 0

if num_exactos >= 3:
    st.sidebar.success(f"{num_exactos} partidos exactos (Suficientes)")
elif num_exactos > 0:
    st.sidebar.warning(f"{num_exactos} partido(s) exacto(s) -> Respaldo inteligente activo")
else:
    st.sidebar.error("0 partidos exactos -> Respaldo cruzado activo")

st.markdown("""
<style>
.stApp { background-color: #0B0F19; color: #F3F4F6; }
.stSidebar { background-color: #111827; }
.header-box {
    background: linear-gradient(90deg, #3B82F6 0%, #1F2937 100%);
    padding: 22px 28px; border-radius: 14px; color: white;
    font-weight: 700; font-size: 24px; margin-bottom: 20px; text-align: center;
}
.veredicto-box {
    padding: 16px 20px; border-radius: 12px; background-color: #1F2937;
    border-left: 5px solid #3B82F6; margin-bottom: 20px; font-size: 16px;
}
.value-box { padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 14px; }
.value-yes { background-color: #064e3b; border-left: 4px solid #10b981; }
.value-no { background-color: #1f2937; border-left: 4px solid #4b5563; }
.top-pick-box { background: linear-gradient(135deg, #065f46 0%, #111827 100%); padding: 20px; border-radius: 12px; border: 2px solid #10b981; margin-bottom: 20px; }
div[data-testid="stMetric"] { background-color: #1F2937; padding: 12px 16px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

def calcular_ev(prob, cuota):
    if cuota <= 1.0 or prob <= 0:
        return 0.0
    return round((prob / 100 * cuota) - 1, 4)

def calcular_kelly(prob, cuota):
    if cuota <= 1.0 or prob <= 0:
        return 0.0
    p, b = prob / 100.0, cuota - 1.0
    if b <= 0:
        return 0.0
    return round(max(0.0, ((p * cuota - 1.0) / b) * 0.5 * 100), 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, real=None):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake = calcular_kelly(prob, cuota_casa) if es_value else 0.0
    kelly_txt = f" | Half-Kelly: <b>{stake}% bank</b>" if es_value else ""
    real_txt = f" | Acierto real: <b>{real:.0f}%</b>" if real is not None else ""
    st.markdown(
        f'<div class="value-box {clase}"><b>{nombre}</b><br>'
        f"Modelo: <b>{prob:.1f}%</b>{real_txt} | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_txt}<br>"
        f'<span style="color:{color_ev}; font-weight:bold; font-size:15px;">'
        f"EV: {ev:+.2%} -> {'VALUE BET' if es_value else 'Sin valor'}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("### Centro de Analisis Individual de Jugadores")
st.caption("Asistente inteligente de apuestas con control de dificultad de rival y análisis de valor matemático.")

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

    umbral_minimo = 3
    t_target = obtener_peso_tier(nivel_sel)

    if "Condición" in df_jugador.columns and "Nivel Rival" in df_jugador.columns:
        df_exactos = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] == nivel_sel)].copy()
    else:
        df_exactos = pd.DataFrame()

    historial_list = []

    if len(df_exactos) >= umbral_minimo:
        for _, row in df_exactos.tail(5).iterrows():
            r = row.to_dict()
            r["Factor_Ajuste"] = 1.0
            r["Tipo_Uso"] = "Exacto (Condición + Rival)"
            r["Peso_Contexto"] = 1.0
            historial_list.append(r)
        fuente = f"Exactos ({condicion_sel} vs {nivel_sel}) - {len(df_exactos)} partidos"
    else:
        for _, row in df_exactos.iterrows():
            r = row.to_dict()
            r["Factor_Ajuste"] = 1.0
            r["Tipo_Uso"] = "Exacto (Insuficiente, retenido)"
            r["Peso_Contexto"] = 1.0
            historial_list.append(r)

        def calcular_factores_respaldo(row_data, condicion_buscada, tier_objetivo):
            cond_partido = str(row_data.get("Condición", "")).lower()
            tier_partido = str(row_data.get("Nivel Rival", ""))
            t_match = obtener_peso_tier(tier_partido)

            if cond_partido == condicion_buscada:
                f_cond = 1.0
                tipo_cond = "Misma condición"
            else:
                if condicion_buscada == "visitante" and cond_partido == "local":
                    f_cond = 0.90
                    tipo_cond = "Cruzado (Casa -> Fuera)"
                elif condicion_buscada == "local" and cond_partido == "visitante":
                    f_cond = 1.05
                    tipo_cond = "Cruzado (Fuera -> Casa)"
                else:
                    f_cond = 1.0
                    tipo_cond = "Cruzado Estándar"

            diff = tier_objetivo - t_match
            if diff == 0:
                f_tier = 1.0
                tipo_tier = "Tier equivalente"
            elif diff > 0:
                f_tier = max(0.65, 1.0 - (diff * 0.12))
                tipo_tier = "Ajuste a la baja"
            else:
                f_tier = min(1.35, 1.0 + (abs(diff) * 0.10))
                tipo_tier = "Ajuste al alza"

            return f_cond * f_tier, f"Respaldo | {tipo_cond} | {tipo_tier} ({tier_partido})"

        if "Condición" in df_jugador.columns:
            df_misma_cond = df_jugador[df_jugador["Condición"] == condicion_sel_lower].copy()
            if not df_exactos.empty:
                df_misma_cond = df_misma_cond[~df_misma_cond.index.isin(df_exactos.index)]
        else:
            df_misma_cond = pd.DataFrame()

        faltantes = umbral_minimo - len(historial_list)
        comodines_tier = df_misma_cond.tail(faltantes)

        for _, row in comodines_tier.iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel_lower, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.85
            historial_list.append(r)

        if len(historial_list) < umbral_minimo:
            opuesto_lower = "local" if condicion_sel_lower == "visitante" else "visitante"
            if "Condición" in df_jugador.columns:
                df_contrarios = df_jugador[df_jugador["Condición"] == opuesto_lower].copy()
            else:
                df_contrarios = pd.DataFrame()

            faltantes_cruzados = umbral_minimo - len(historial_list)
            comodines_cruzados = df_contrarios.tail(faltantes_cruzados)

            for _, row in comodines_cruzados.iterrows():
                r = row.to_dict()
                f_tot, desc = calcular_factores_respaldo(r, condicion_sel_lower, t_target)
                r["Factor_Ajuste"] = f_tot
                r["Tipo_Uso"] = desc
                r["Peso_Contexto"] = 0.75
                historial_list.append(r)

        fuente = f"Muestra adaptada con Tiers ({len(historial_list)} partidos)"

    historial = pd.DataFrame(historial_list)

    for col in ["Goles", "Asistencias", "Tiros", "A Puerta", "Faltas"]:
        if col in historial.columns:
            historial[col] = historial[col] * historial["Factor_Ajuste"]

    n_obs = len(historial)
    muestra_pequena = n_obs <= 3

    st.markdown(f'<div class="header-box">{jugador_sel.upper()} | {condicion_sel} vs {nivel_sel}</div>', unsafe_allow_html=True)
    st.caption(f"Base analizada: {n_obs} partidos | Fuente: {fuente}")

    if muestra_pequena:
        st.warning("Muestra pequeña. Interpreta con cautela.")

    hoy = pd.Timestamp.today().normalize()
    if "Fecha" in historial.columns:
        historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.replace(0, 0.1)
        historial["Peso_Temporal"] = 1 / (1 + (historial["Dias_Pasados"] / 30))
    else:
        historial["Peso_Temporal"] = 1.0

    historial["Peso_Total"] = historial["Peso_Temporal"] * historial["Peso_Contexto"]
    pesos = historial["Peso_Total"] / historial["Peso_Total"].sum()

    def prom_w(col):
        return float(np.average(historial[col].fillna(0), weights=pesos)) if col in historial.columns else 0.0

    lam_g = shrinkage_lambda(prom_w("Goles"), float(df_jugador["Goles"].mean()) if len(df_jugador) else 0, n_obs, k_shrink) if usar_shrinkage else prom_w("Goles")
    lam_t = shrinkage_lambda(prom_w("Tiros"), float(df_jugador["Tiros"].mean()) if len(df_jugador) else 0, n_obs, k_shrink) if usar_shrinkage else prom_w("Tiros")
    lam_p = shrinkage_lambda(prom_w("A Puerta"), float(df_jugador["A Puerta"].mean()) if len(df_jugador) else 0, n_obs, k_shrink) if usar_shrinkage else prom_w("A Puerta")
    lam_a = shrinkage_lambda(prom_w("Asistencias"), float(df_jugador["Asistencias"].mean()) if len(df_jugador) else 0, n_obs, k_shrink) if usar_shrinkage else prom_w("Asistencias")
    lam_f = shrinkage_lambda(prom_w("Faltas"), float(df_jugador["Faltas"].mean()) if len(df_jugador) else 0, n_obs, k_shrink) if usar_shrinkage else prom_w("Faltas")

    rng = np.random.default_rng(42)
    sim_goles = rng.poisson(max(lam_g, 0.01), 10000)
    sim_tiros = rng.poisson(max(lam_t, 0.01), 10000)
    sim_puerta = rng.poisson(max(lam_p, 0.01), 10000)
    sim_asist = rng.poisson(max(lam_a, 0.01), 10000)
    sim_faltas = rng.poisson(max(lam_f, 0.01), 10000)
    sim_contrib = sim_goles + sim_asist

    prob_goles = (sim_goles > linea_goles).mean() * 100
    prob_tiros = (sim_tiros > linea_tiros).mean() * 100
    prob_puerta = (sim_puerta > linea_puerta).mean() * 100
    prob_asist = (sim_asist > linea_asist).mean() * 100
    prob_faltas = (sim_faltas > linea_faltas).mean() * 100
    prob_contrib = (sim_contrib > linea_contrib).mean() * 100

    def cj(p): return round(100 / p, 2) if p > 0 else 99.0

    lista_mercados = [
        {"nombre": f"Over {linea_goles} Goles", "prob": prob_goles, "cuota": cuota_casa_goles, "ev": calcular_ev(prob_goles, cuota_casa_goles)},
        {"nombre": f"Over {linea_tiros} Tiros", "prob": prob_tiros, "cuota": cuota_casa_tiros, "ev": calcular_ev(prob_tiros, cuota_casa_tiros)},
        {"nombre": f"Over {linea_puerta} a Puerta", "prob": prob_puerta, "cuota": cuota_casa_puerta, "ev": calcular_ev(prob_puerta, cuota_casa_puerta)},
        {"nombre": f"Over {linea_asist} Asistencias", "prob": prob_asist, "cuota": cuota_casa_asist, "ev": calcular_ev(prob_asist, cuota_casa_asist)},
        {"nombre": f"Over {linea_faltas} Faltas", "prob": prob_faltas, "cuota": cuota_casa_faltas, "ev": calcular_ev(prob_faltas, cuota_casa_faltas)},
        {"nombre": f"Over {linea_contrib} Gol/Asist", "prob": prob_contrib, "cuota": cuota_casa_contrib, "ev": calcular_ev(prob_contrib, cuota_casa_contrib)}
    ]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen Dinámico", "Value Bet Props", "🤖 Panel Inteligente & Parlay", "Probabilidades", "Detalle"])

    with tab1:
        st.subheader(f"Métricas promedio y eficiencia en el escenario: {condicion_sel} vs {nivel_sel}")
        st.caption("Valores calculados estrictamente sobre la muestra adaptada para este análisis.")

        # Análisis de Frecuencia y Veredicto para el próximo partido
        lam_contrib = lam_g + lam_a
        partidos_por_contrib = 1.0 / lam_contrib if lam_contrib > 0 else 0

        if prob_contrib >= 55:
            analisis_tendencia = f"Tendencia alta: El jugador muestra un ritmo sólido de aportación, participando en promedio **cada {partidos_por_contrib:.1f} partidos** en este escenario. Las simulaciones de Poisson proyectan una probabilidad sólida ({prob_contrib:.1f}%) de influir directamente en el marcador (Gol o Asistencia) en este próximo encuentro."
        elif prob_contrib >= 35:
            analisis_tendencia = f"Tendencia moderada: Registra una contribución en promedio **cada {partidos_por_contrib:.1f} partidos**. El encuentro presenta paridad, con una probabilidad estimada del {prob_contrib:.1f}% de sumar al menos una contribución."
        else:
            analisis_tendencia = f"Alerta de baja producción: Su frecuencia en este escenario se diluye a una contribución **cada {partidos_por_contrib:.1f} partidos**, arrojando una probabilidad reducida ({prob_contrib:.1f}%) de ver portería o asistir."

        st.markdown(f'<div class="veredicto-box"><b>📊 Análisis de Frecuencia y Próximo Partido:</b><br>{analisis_tendencia}</div>', unsafe_allow_html=True)

        metrics_data = {
            "Goles": {"prom": historial["Goles"].mean() if "Goles" in historial else 0, "lam": lam_g},
            "Asistencias": {"prom": historial["Asistencias"].mean() if "Asistencias" in historial else 0, "lam": lam_a},
            "Tiros": {"prom": historial["Tiros"].mean() if "Tiros" in historial else 0, "lam": lam_t},
            "A Puerta": {"prom": historial["A Puerta"].mean() if "A Puerta" in historial else 0, "lam": lam_p},
            "Faltas": {"prom": historial["Faltas"].mean() if "Faltas" in historial else 0, "lam": lam_f},
            "Gol o Asistencia": {"prom": (historial["Goles"] + historial["Asistencias"]).mean() if "Goles" in historial else 0, "lam": lam_g + lam_a}
        }

        cols = st.columns(3)
        for i, (var, datos) in enumerate(metrics_data.items()):
            col_target = cols[i % 3]
            col_target.metric(f"Prom. {var}", f"{datos['prom']:.2f}", f"λ: {datos['lam']:.2f}")

        st.markdown("---")
        st.subheader("🎯 Eficiencia y Conversión en el Escenario")

        total_goles_escenario = historial["Goles"].sum() if "Goles" in historial else 0
        total_puerta_escenario = historial["A Puerta"].sum() if "A Puerta" in historial else 0
        total_tiros_escenario = historial["Tiros"].sum() if "Tiros" in historial else 0

        ratio_puerta_gol = total_puerta_escenario / total_goles_escenario if total_goles_escenario > 0 else 0.0
        ratio_tiros_gol = total_tiros_escenario / total_goles_escenario if total_goles_escenario > 0 else 0.0

        e1, e2, e3 = st.columns(3)
        if total_goles_escenario > 0:
            e1.metric("Tiros a Puerta por Gol", f"{ratio_puerta_gol:.1f} tiros", "Ratio de conversión a puerta")
            e2.metric("Tiros Totales por Gol", f"{ratio_tiros_gol:.1f} tiros", "Eficiencia global de disparo")
        else:
            e1.metric("Tiros a Puerta por Gol", "N/A", "Sin goles en esta muestra")
            e2.metric("Tiros Totales por Gol", "N/A", "Sin goles en esta muestra")

        e3.metric("Partidos en Muestra", n_obs, "Filtro aplicado OK")

    with tab2:
        st.subheader("Value Bet Props (Gestión Half-Kelly)")
        st.markdown("💡 *Todas las apuestas con EV positivo están optimizadas para operar bajo el criterio de Half-Kelly (% de bank).*")

        mostrar_value(f"Over {linea_goles} Goles", cj(prob_goles), cuota_casa_goles, calcular_ev(prob_goles, cuota_casa_goles), prob_goles, (historial["Goles"] > linea_goles).mean() * 100)
        mostrar_value(f"Over {linea_tiros} Tiros", cj(prob_tiros), cuota_casa_tiros, calcular_ev(prob_tiros, cuota_casa_tiros), prob_tiros, (historial["Tiros"] > linea_tiros).mean() * 100)
        mostrar_value(f"Over {linea_puerta} a Puerta", cj(prob_puerta), cuota_casa_puerta, calcular_ev(prob_puerta, cuota_casa_puerta), prob_puerta, (historial["A Puerta"] > linea_puerta).mean() * 100)
        mostrar_value(f"Over {linea_asist} Asistencias", cj(prob_asist), cuota_casa_asist, calcular_ev(prob_asist, cuota_casa_asist), prob_asist, (historial["Asistencias"] > linea_asist).mean() * 100)
        mostrar_value(f"Over {linea_faltas} Faltas", cj(prob_faltas), cuota_casa_faltas, calcular_ev(prob_faltas, cuota_casa_faltas), prob_faltas, (historial["Faltas"] > linea_faltas).mean() * 100)
        mostrar_value(f"Over {linea_contrib} Gol/Asist", cj(prob_contrib), cuota_casa_contrib, calcular_ev(prob_contrib, cuota_casa_contrib), prob_contrib, ((historial["Goles"] + historial["Asistencias"]) > linea_contrib).mean() * 100)

    with tab3:
        st.subheader("🤖 Panel de Inteligencia & Value Bets")
        st.caption("Selección automática de la mejor oportunidad y constructor de combinadas (Parlay) con criterio matemático y gestión de bank.")

        value_bets_disponibles = [m for m in lista_mercados if m["ev"] > 0]

        if value_bets_disponibles:
            top_pick = max(value_bets_disponibles, key=lambda x: x["ev"])
            cuota_justa_top = cj(top_pick["prob"])
            stake_top = calcular_kelly(top_pick["prob"], top_pick["cuota"])

            st.markdown(
                f'<div class="top-pick-box">'
                f'<h3>🏆 La Joya del Partido (Top Value Bet)</h3>'
                f'<p style="font-size: 16px; margin-bottom: 8px;">Mercado recomendado: <b>{top_pick["nombre"]}</b></p>'
                f'<ul>'
                f'<li>Probabilidad del Modelo: <b>{top_pick["prob"]:.1f}%</b></li>'
                f'<li>Cuota Justa Calculada: <b>{cuota_justa_top}</b> | Cuota Casa: <b>{top_pick["cuota"]}</b></li>'
                f'<li><b>EV Matemático: {top_pick["ev"]:+.2%}</b> (Rentabilidad positiva a largo plazo)</li>'
                f'</ul>'
                f'<p style="color: #10b981; font-weight: bold; margin-top: 10px;">👉 Sugerencia de Stake: <b>{stake_top}% del Bank (Half-Kelly)</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ En este momento no hay mercados con EV positivo estricto (> 0) para las líneas configuradas.")

        st.markdown("---")
        st.subheader("🔗 Constructor de Combinada Inteligente (Parlay)")
        st.markdown("Selecciona mercados para calcular la probabilidad conjunta, su cuota justa y verificar si la combinada tiene **EV positivo** real frente al margen de la casa.")

        nombres_mercados = [m["nombre"] for m in lista_mercados]
        parlay_elegidos = st.multiselect("Elige los mercados para tu combinada:", options=nombres_mercados, key="parlay_jugador_input")

        if parlay_elegidos:
            prob_conjunta = 1.0
            for nombre in parlay_elegidos:
                m_info = next(m for m in lista_mercados if m["nombre"] == nombre)
                prob_conjunta *= (m_info["prob"] / 100.0)

            prob_conjunta_pct = prob_conjunta * 100.0
            cuota_justa_combinada = round(100 / prob_conjunta_pct, 2) if prob_conjunta_pct > 0 else 99.0

            st.markdown(f"**Probabilidad Conjunta del Modelo:** `{prob_conjunta_pct:.2f}%`")
            st.markdown(f"**Cuota Justa Combinada:** `{cuota_justa_combinada}`")

            cuota_casa_parlay = st.number_input("Introduce la cuota total que te paga la casa por esta combinada:", min_value=1.01, value=cuota_justa_combinada * 0.95, step=0.05, format="%.2f", key="cuota_parlay_jug_input")

            ev_parlay = calcular_ev(prob_conjunta_pct, cuota_casa_parlay)
            stake_parlay = calcular_kelly(prob_conjunta_pct, cuota_casa_parlay)

            col_p1, col_p2 = st.columns(2)
            col_p1.metric("EV de la Combinada", f"{ev_parlay:+.2%}", "Matemática de valor")

            if ev_parlay > 0:
                col_p2.metric("Stake Sugerido", f"{stake_parlay}% del Bank", "Combinada con EV positivo ✅")
                st.success(f"🎉 ¡Esta combinada tiene EV positivo! Stake sugerido: {stake_parlay}% del Bankroll (Half-Kelly).")
            else:
                col_p2.metric("Stake Sugerido", "0%", "EV Negativo ❌")
                st.warning("⚠️ Cuidado: Esta combinada tiene EV negativo debido al acumulado de margen que cobra la casa de apuestas.")

    with tab4:
        st.subheader("Probabilidades del Modelo")
        st.metric(f"Prob. Goles > {linea_goles}", f"{prob_goles:.1f}%")
        st.metric(f"Prob. Tiros > {linea_tiros}", f"{prob_tiros:.1f}%")
        st.metric(f"Prob. a Puerta > {linea_puerta}", f"{prob_puerta:.1f}%")
        st.metric(f"Prob. Asistencias > {linea_asist}", f"{prob_asist:.1f}%")
        st.metric(f"Prob. Faltas > {linea_faltas}", f"{prob_faltas:.1f}%")
        st.metric(f"Prob. Gol/Asist > {linea_contrib}", f"{prob_contrib:.1f}%")

    with tab5:
        st.subheader("Detalle y Auditoría")
        h_mostrar = historial.copy()
        if "Fecha" in h_mostrar.columns:
            h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
        cols_mostrar = [c for c in ["Fecha", "Condición", "Rival", "Nivel Rival", "Goles", "Asistencias", "Tiros", "A Puerta", "Faltas", "Tipo_Uso", "Factor_Ajuste"] if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols_mostrar], hide_index=True, use_container_width=True)
else:
    st.info("Configura las opciones en la barra lateral, elige jugador y haz clic en Analizar.")
