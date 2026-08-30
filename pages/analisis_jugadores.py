import html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter

try:
    import xgboost as xgb
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False

st.set_page_config(
    page_title="GoalMetrics | Análisis de Jugadores (Híbrido)",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data(ttl=600)
def cargar_datos_jugadores():
    sheet_id = st.secrets.get("JUGADORES_SHEET_ID", "1q98g-IxYaO8g3ksDb0vyZ9V7IrhPjDcVUtChZI8SNT4")
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.astype(str).str.strip()
    
    col_ligaencontrada = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["liga", "competición", "competicion", "torneo"]:
            col_ligaencontrada = col
            break
            
    if col_ligaencontrada and col_ligaencontrada != "Liga":
        df = df.rename(columns={col_ligaencontrada: "Liga"})
        
    if "Liga" in df.columns:
        df["Liga"] = df["Liga"].astype(str).str.strip()
        df["Liga"] = df["Liga"].replace(["nan", "None", ""], "Sin Liga")
    else:
        df["Liga"] = "General"

    if "Jugador" not in df.columns and "Equipo" in df.columns:
        df = df.rename(columns={"Equipo": "Jugador"})
    if "Jugador" in df.columns:
        df["Jugador"] = df["Jugador"].astype(str).str.strip()
        
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
        
    for col in ["Goles", "Asistencias", "Tiros", "A Puerta", "Faltas", "Amarillas", "Rojas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
            
    if "Condicion" in df.columns and "Condición" not in df.columns:
        df = df.rename(columns={"Condicion": "Condición"})
    if "Condición" in df.columns:
        df["Condición"] = df["Condición"].astype(str).str.strip().str.lower()
    else:
        df["Condición"] = "local"
        
    if "Nivel Rival" in df.columns:
        df["Nivel Rival"] = df["Nivel Rival"].astype(str).str.strip()
    else:
        df["Nivel Rival"] = "MEDIA TABLA"
        
    return df

def calcular_feature_engineering_jugadores(df):
    df = df.copy()
    if "Fecha" in df.columns and "Jugador" in df.columns:
        df = df.sort_values(by=["Jugador", "Fecha"])
    
    df["Conversion_Tiros"] = np.where(df["Tiros"] > 0, df["Goles"] / df["Tiros"], 0.0)
    df["Conversion_Puerta"] = np.where(df["A Puerta"] > 0, df["Goles"] / df["A Puerta"], 0.0)
    df["Contribucion_Total"] = df["Goles"] + df["Asistencias"]

    if "Jugador" in df.columns:
        rolling_goles = df.groupby("Jugador")["Goles"].rolling(window=5, min_periods=1)
        df["Goles_Media_Movil_5"] = rolling_goles.mean().reset_index(level=0, drop=True)
        df["Goles_Volatilidad_5"] = rolling_goles.std().fillna(0).reset_index(level=0, drop=True)
        
        rolling_tiros = df.groupby("Jugador")["Tiros"].rolling(window=5, min_periods=1)
        df["Tiros_Media_Movil_5"] = rolling_tiros.mean().reset_index(level=0, drop=True)
        
        media_global_goles = df["Goles"].mean() if len(df) > 0 else 0.2
        df["Momentum_Goles"] = df["Goles_Media_Movil_5"] - media_global_goles
        
    return df

def entrenar_predictor_xgboost_jugadores(df_historico, features_modelo):
    if not XGB_DISPONIBLE or len(df_historico) < 25:
        return None
        
    df_clean = df_historico.dropna(subset=features_modelo + ["Goles"]).copy()
    if len(df_clean) < 25:
        return None
        
    df_clean["Target_Gol"] = (df_clean["Goles"] > 0).astype(int)
    X = df_clean[features_modelo]
    y = df_clean["Target_Gol"]
    
    model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X, y)
    return model

def predecir_probabilidad_hibrida_jugador(prob_poisson, jugador_actual_df, features_modelo, modelo_xgb):
    if modelo_xgb is None or jugador_actual_df.empty:
        return prob_poisson
    ultima_fila = jugador_actual_df.tail(1)
    try:
        X_pred = ultima_fila[features_modelo]
        prob_xgb = float(modelo_xgb.predict_proba(X_pred)[0][1]) * 100.0
    except Exception:
        return prob_poisson

    prob_hibrida = (0.70 * prob_poisson) + (0.30 * prob_xgb)
    return round(prob_hibrida, 2)

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

try:
    df_raw = cargar_datos_jugadores()
    df = calcular_feature_engineering_jugadores(df_raw)
    datos_ok = True
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    datos_ok = False
    df = pd.DataFrame()

st.sidebar.header("Configuracion del Jugador")

if datos_ok and not df.empty and "Liga" in df.columns:
    ligas_disponibles = sorted([str(x) for x in df["Liga"].dropna().unique() if pd.notna(x)])
    liga_sel = st.sidebar.selectbox("Selecciona la Liga", ligas_disponibles)
    
    df_liga = df[df["Liga"] == liga_sel]
    jugadores = sorted([str(x) for x in df_liga["Jugador"].dropna().unique() if pd.notna(x)])
    if jugadores:
        jugador_sel = st.sidebar.selectbox("Selecciona al Jugador", jugadores)
        df_jugador = df_liga[df_liga["Jugador"] == jugador_sel].copy()
    else:
        jugador_sel = st.sidebar.selectbox("Selecciona al Jugador", ["Sin jugadores"])
        df_jugador = pd.DataFrame()
        
    if "Fecha" in df_jugador.columns:
        df_jugador = df_jugador.sort_values("Fecha")
        
    if "Condición" in df_jugador.columns:
        condiciones = sorted(df_jugador["Condición"].dropna().unique().tolist())
        if not condiciones:
            condiciones = ["local", "visitante"]
    else:
        condiciones = ["local", "visitante"]
        
    condicion_sel = st.sidebar.selectbox("Condicion", [c.capitalize() for c in condiciones])
    condicion_sel_lower = condicion_sel.lower()
    
    if "Nivel Rival" in df_jugador.columns:
        niveles = sorted(df_jugador["Nivel Rival"].dropna().unique().tolist())
        if not niveles:
            niveles = ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
    else:
        niveles = ["TOP", "CHAMPIONS", "MEDIA TABLA", "DESCENSO"]
        
    nivel_sel = st.sidebar.selectbox("Nivel del Rival", niveles)
else:
    liga_sel = st.sidebar.selectbox("Selecciona la Liga", ["General"])
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

total_partidos_jugador = len(df_jugador) if not df_jugador.empty else 0
if total_partidos_jugador == 0:
    st.sidebar.error("0 partidos registrados para este jugador en la liga seleccionada")
else:
    exactos_check = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] == nivel_sel)]
    num_exactos = len(exactos_check)
    if num_exactos >= 2:
        st.sidebar.success(f"{num_exactos} partidos exactos (Suficientes)")
    elif num_exactos == 1:
        st.sidebar.warning("1 partido exacto -> Respaldo inteligente activo (Mínimo 2)")
    else:
        st.sidebar.error("0 partidos exactos en este filtro")

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

st.markdown("### Centro de Analisis Individual de Jugadores (Híbrido)")
st.caption("Asistente inteligente de apuestas con control de dificultad de rival, ensemble XGBoost y análisis de valor matemático.")

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

if st.session_state.analizado_jugadores:
    if total_partidos_jugador == 0:
        st.error(f"❌ No se puede realizar el análisis porque el jugador **{jugador_sel}** tiene **0 partidos** registrados en esta liga.")
        st.stop()
        
    df_jugador = df_liga[df_liga["Jugador"] == jugador_sel].copy()
    if "Fecha" in df_jugador.columns:
        df_jugador = df_jugador.sort_values("Fecha")

    if "Condición" in df_jugador.columns and "Nivel Rival" in df_jugador.columns:
        df_exactos = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] == nivel_sel)].copy()
    else:
        df_exactos = pd.DataFrame()

    if len(df_exactos) == 0:
        st.error(f"❌ No se puede realizar el análisis: Hay 0 partidos exactos registrados para **{jugador_sel}** como **{condicion_sel}** contra rivales nivel **{nivel_sel}** en la liga **{liga_sel}**.")
        st.stop()

    UMBRAL_MINIMO = 2
    t_target = obtener_peso_tier(nivel_sel)
    historial_list = []

    for _, row in df_exactos.iterrows():
        r = row.to_dict()
        r["Factor_Ajuste"] = 1.0
        r["Tipo_Uso"] = f"Exacto ({condicion_sel} vs {nivel_sel})"
        r["Peso_Contexto"] = 1.0
        historial_list.append(r)
        
    fuente = f"Exactos ({len(historial_list)} partidos)"

    if len(historial_list) < UMBRAL_MINIMO:
        if "Condición" in df_jugador.columns:
            df_misma_cond = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] != nivel_sel)].copy()
        else:
            df_misma_cond = pd.DataFrame()

        faltantes = UMBRAL_MINIMO - len(historial_list)
        comodines_tier = df_misma_cond.tail(faltantes)

        for _, row in comodines_tier.iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel_lower, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.85
            historial_list.append(r)
        if len(historial_list) > len(df_exactos):
            fuente = "Muestra mixta (1 Exacto + Respaldo ajustado por Tier)"

    if len(historial_list) < UMBRAL_MINIMO:
        opuesto_lower = "local" if condicion_sel_lower == "visitante" else "visitante"
        if "Condición" in df_jugador.columns:
            df_contrarios = df_jugador[df_jugador["Condición"] == opuesto_lower].copy()
        else:
            df_contrarios = pd.DataFrame()

        faltantes_cruzados = UMBRAL_MINIMO - len(historial_list)
        comodines_cruzados = df_contrarios.tail(faltantes_cruzados)

        for _, row in comodines_cruzados.iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel_lower, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.75
            historial_list.append(r)
        fuente = "Muestra adaptada con respaldo cruzado y ajuste de tier"

    historial = pd.DataFrame(historial_list)

    for col in ["Goles", "Asistencias", "Tiros", "A Puerta", "Faltas"]:
        if col in historial.columns:
            historial[col] = historial[col] * historial["Factor_Ajuste"]

    n_obs = len(historial)
    muestra_pequena = n_obs <= 2

    st.markdown(f'<div class="header-box">{liga_sel.upper()} | {jugador_sel.upper()} | {condicion_sel} vs {nivel_sel}</div>', unsafe_allow_html=True)
    st.caption(f"Base analizada: {n_obs} partidos | Fuente: {fuente} | Ensemble Híbrido Activo")

    if muestra_pequena:
        st.warning("Muestra pequeña (Respaldo activo con 1 partido). Interpreta con cautela.")

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

    lam_g_raw = prom_w("Goles")
    lam_t_raw = prom_w("Tiros")
    lam_p_raw = prom_w("A Puerta")
    lam_a_raw = prom_w("Asistencias")
    lam_f_raw = prom_w("Faltas")

    df_tier_liga = df_liga[df_liga["Nivel Rival"] == nivel_sel]
    if len(df_tier_liga) == 0:
        df_tier_liga = df_liga

    prior_g = float(df_tier_liga["Goles"].mean()) if len(df_tier_liga) and "Goles" in df_tier_liga.columns else lam_g_raw
    prior_t = float(df_tier_liga["Tiros"].mean()) if len(df_tier_liga) and "Tiros" in df_tier_liga.columns else lam_t_raw
    prior_p = float(df_tier_liga["A Puerta"].mean()) if len(df_tier_liga) and "A Puerta" in df_tier_liga.columns else lam_p_raw
    prior_a = float(df_tier_liga["Asistencias"].mean()) if len(df_tier_liga) and "Asistencias" in df_tier_liga.columns else lam_a_raw
    prior_f = float(df_tier_liga["Faltas"].mean()) if len(df_tier_liga) and "Faltas" in df_tier_liga.columns else lam_f_raw

    if usar_shrinkage:
        lam_g = shrinkage_lambda(lam_g_raw, prior_g, n_obs, k_shrink)
        lam_t = shrinkage_lambda(lam_t_raw, prior_t, n_obs, k_shrink)
        lam_p = shrinkage_lambda(lam_p_raw, prior_p, n_obs, k_shrink)
        lam_a = shrinkage_lambda(lam_a_raw, prior_a, n_obs, k_shrink)
        lam_f = shrinkage_lambda(lam_f_raw, prior_f, n_obs, k_shrink)
    else:
        lam_g, lam_t, lam_p, lam_a, lam_f = lam_g_raw, lam_t_raw, lam_p_raw, lam_a_raw, lam_f_raw

    rng = np.random.default_rng(42)
    num_sim = 10000
    sim_goles = rng.poisson(max(lam_g, 0.01), num_sim)
    sim_tiros = rng.poisson(max(lam_t, 0.01), num_sim)
    sim_puerta = rng.poisson(max(lam_p, 0.01), num_sim)
    sim_asist = rng.poisson(max(lam_a, 0.01), num_sim)
    sim_faltas = rng.poisson(max(lam_f, 0.01), num_sim)
    sim_contrib = sim_goles + sim_asist

    prob_goles_base = (sim_goles > linea_goles).mean() * 100
    prob_tiros_base = (sim_tiros > linea_tiros).mean() * 100
    prob_puerta_base = (sim_puerta > linea_puerta).mean() * 100
    prob_asist_base = (sim_asist > linea_asist).mean() * 100
    prob_faltas_base = (sim_faltas > linea_faltas).mean() * 100
    prob_contrib_base = (sim_contrib > linea_contrib).mean() * 100

    features_modelo = ["Goles_Media_Movil_5", "Goles_Volatilidad_5", "Tiros_Media_Movil_5", "Conversion_Tiros", "Momentum_Goles"]
    modelo_xgb_global = entrenar_predictor_xgboost_jugadores(df, features_modelo)
    
    prob_goles = predecir_probabilidad_hibrida_jugador(prob_goles_base, historial, features_modelo, modelo_xgb_global)
    prob_contrib = predecir_probabilidad_hibrida_jugador(prob_contrib_base, historial, features_modelo, modelo_xgb_global)
    prob_tiros = prob_tiros_base
    prob_puerta = prob_puerta_base
    prob_asist = prob_asist_base
    prob_faltas = prob_faltas_base

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
        
        # --- NUEVA LÓGICA PROFESIONAL DE FRECUENCIA Y MOMENTUM ---
        partidos_por_gol = 1.0 / lam_g if lam_g > 0 else 0.0
        partidos_por_asist = 1.0 / lam_a if lam_a > 0 else 0.0
        
        # Porcentaje real de partidos con contribución en la muestra
        if len(historial) > 0:
            partidos_con_contrib = ((historial["Goles"] + historial["Asistencias"]) > 0).sum()
            pct_contribucion_real = (partidos_con_contrib / len(historial)) * 100.0
        else:
            pct_contribucion_real = 0.0

        # Análisis de Momentum Reciente (últimos 3 partidos)
        ultimos_partidos = historial.tail(min(3, len(historial)))
        goles_recientes = ultimos_partidos["Goles"].mean() if "Goles" in ultimos_partidos else 0.0
        asist_recientes = ultimos_partidos["Asistencias"].mean() if "Asistencias" in ultimos_partidos else 0.0
        
        if goles_recientes > lam_g or asist_recientes > lam_a:
            estado_momentum = "🔥 <b>Momentum al alza:</b> Su producción en los últimos encuentros supera su media histórica para este contexto."
        elif goles_recientes < lam_g * 0.5 and asist_recientes < lam_a * 0.5:
            estado_momentum = "❄️ <b>Momentum a la baja:</b> Su rendimiento reciente se encuentra por debajo de su estándar habitual."
        else:
            estado_momentum = "⚖️ <b>Momentum estable:</b> Su dinámica actual se alinea con su promedio histórico."

        freq_gol_txt = f"anota un gol cada <b>{partidos_por_gol:.1f} partidos</b>" if partidos_por_gol > 0 else "baja incidencia goleadora"
        freq_asist_txt = f"reparte una asistencia cada <b>{partidos_por_asist:.1f} partidos</b>" if partidos_por_asist > 0 else "baja incidencia en pases de gol"

        analisis_tendencia = (
            f"<b>Desglose de Frecuencia y Producción Pura:</b><br>"
            f"• En este escenario, el jugador {freq_gol_txt} y {freq_asist_txt}.<br>"
            f"• <b>Porcentaje de Contribución Real:</b> Aporta al menos un gol o asistencia en el <b>{pct_contribucion_real:.1f}%</b> de sus encuentros bajo este contexto.<br>"
            f"• <b>Evaluación frente a línea elegida (Over {linea_contrib}):</b> La probabilidad híbrida proyectada es del <b>{prob_contrib:.1f}%</b>.<br>"
            f"• {estado_momentum}"
        )

        st.markdown(f'<div class="veredicto-box"><b>📊 Análisis Profesional de Frecuencia:</b><br>{analisis_tendencia}</div>', unsafe_allow_html=True)

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
                f'<li>Probabilidad del Modelo Híbrido: <b>{top_pick["prob"]:.1f}%</b></li>'
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
        st.subheader("🔗 Constructor de Combinada Inteligente (Parlay con Muestras Pareadas)")
        nombres_mercados = [m["nombre"] for m in lista_mercados]
        parlay_elegidos = st.multiselect("Elige los mercados para tu combinada:", options=nombres_mercados, key="parlay_jugador_input")

        if parlay_elegidos:
            condiciones_sim = {
                f"Over {linea_goles} Goles": sim_goles > linea_goles,
                f"Over {linea_tiros} Tiros": sim_tiros > linea_tiros,
                f"Over {linea_puerta} a Puerta": sim_puerta > linea_puerta,
                f"Over {linea_asist} Asistencias": sim_asist > linea_asist,
                f"Over {linea_faltas} Faltas": sim_faltas > linea_faltas,
                f"Over {linea_contrib} Gol/Asist": sim_contrib > linea_contrib,
            }

            match_mask = np.ones(num_sim, dtype=bool)
            for nombre in parlay_elegidos:
                if nombre in condiciones_sim:
                    match_mask = match_mask & condiciones_sim[nombre]

            prob_conjunta_pct = float(match_mask.mean()) * 100.0
            cuota_justa_combinada = round(100 / prob_conjunta_pct, 2) if prob_conjunta_pct > 0 else 99.0

            st.markdown(f"**Probabilidad Conjunta Real (Simulada):** `{prob_conjunta_pct:.2f}%`")
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
