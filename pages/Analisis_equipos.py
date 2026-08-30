import html
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from collections import Counter
import hashlib
import colorsys

try:
    import xgboost as xgb
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False

st.set_page_config(
    page_title="GoalMetrics | Análisis de Equipos (Híbrido)",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = st.secrets.get("EQUIPOS_SHEET_ID", "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg")
    
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    df.columns = df.columns.astype(str).str.strip()
    
    if "Liga" in df.columns:
        df["Liga"] = df["Liga"].astype(str).str.strip()
        df["Liga"] = df["Liga"].replace(["nan", "None", ""], "Sin Liga")
    else:
        df["Liga"] = "General"

    df = df.dropna(subset=["Equipo", "Fecha", "Condición", "Nivel Rival"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df["Equipo"] = df["Equipo"].astype(str).str.strip()
    df["Condición"] = df["Condición"].astype(str).str.strip().str.lower()
    df["Nivel Rival"] = df["Nivel Rival"].astype(str).str.strip()
    
    cols_numericas = ["Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Atajadas", "Amarillas", "Rojas", "Corners Rival"]
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    if "Tiros a Puerta Rival" not in df.columns:
        df["Tiros a Puerta Rival"] = df["Goles Rival"] + df["Atajadas"]
    if "Corners Rival" not in df.columns:
        df["Corners Rival"] = 0.0
    
    return df

def calcular_feature_engineering(df):
    df = df.copy()
    if "Fecha" in df.columns and "Equipo" in df.columns:
        df = df.sort_values(by=["Equipo", "Fecha"])
    
    df["Conversion_Tiros"] = np.where(df["Tiros"] > 0, df["Goles"] / df["Tiros"], 0.0)
    df["Conversion_Puerta"] = np.where(df["A Puerta"] > 0, df["Goles"] / df["A Puerta"], 0.0)
    if "Goles Rival" in df.columns:
        df["Diff_Goles"] = df["Goles"] - df["Goles Rival"]
    if "Corners" in df.columns and "Tiros" in df.columns:
        df["Ratio_Corners_Tiros"] = np.where(df["Tiros"] > 0, df["Corners"] / df["Tiros"], 0.0)

    if "Equipo" in df.columns:
        rolling_goles = df.groupby("Equipo")["Goles"].rolling(window=5, min_periods=1)
        df["Goles_Media_Movil_5"] = rolling_goles.mean().reset_index(level=0, drop=True)
        df["Goles_Volatilidad_5"] = rolling_goles.std().fillna(0).reset_index(level=0, drop=True)
        
        rolling_tiros = df.groupby("Equipo")["Tiros"].rolling(window=5, min_periods=1)
        df["Tiros_Media_Movil_5"] = rolling_tiros.mean().reset_index(level=0, drop=True)
        
        media_global_goles = df["Goles"].mean() if len(df) > 0 else 1.0
        df["Momentum_Goles"] = df["Goles_Media_Movil_5"] - media_global_goles
        
    return df

def entrenar_predictor_xgboost(df_historico, features_modelo):
    if not XGB_DISPONIBLE or len(df_historico) < 20:
        return None
        
    df_clean = df_historico.dropna(subset=features_modelo + ["Goles", "Goles Rival"]).copy()
    if len(df_clean) < 20:
        return None
        
    df_clean["Target_Victoria"] = (df_clean["Goles"] > df_clean["Goles Rival"]).astype(int)
    X = df_clean[features_modelo]
    y = df_clean["Target_Victoria"]
    
    model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X, y)
    return model

def predecir_probabilidad_hibrida(prob_poisson, equipo_actual_df, features_modelo, modelo_xgb):
    if modelo_xgb is None or equipo_actual_df.empty:
        return prob_poisson
    ultima_fila = equipo_actual_df.tail(1)
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

def dixon_coles_tau(x, y, lam_x, lam_y, rho):
    if x == 0 and y == 0:
        return 1.0 - lam_x * lam_y * rho
    if x == 0 and y == 1:
        return 1.0 + lam_x * rho
    if x == 1 and y == 0:
        return 1.0 + lam_y * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

def poisson_pmf(k, lam):
    lam = max(float(lam), 1e-9)
    k = int(k)
    if k < 0:
        return 0.0
    if k == 0:
        return float(np.exp(-lam))
    log_p = -lam + k * np.log(lam) - np.sum(np.log(np.arange(1, k + 1)))
    return float(np.exp(log_p))

@st.cache_data
def simular_goles_dixon_coles(lam_fav, lam_con, rho=-0.10, num_sim=10000, max_goles=8, seed=42):
    rng = np.random.default_rng(seed)
    lam_fav = max(lam_fav, 0.05)
    lam_con = max(lam_con, 0.05)
    xs = np.arange(0, max_goles + 1)
    ys = np.arange(0, max_goles + 1)
    joint = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        px = poisson_pmf(x, lam_fav)
        for j, y in enumerate(ys):
            py = poisson_pmf(y, lam_con)
            tau = dixon_coles_tau(x, y, lam_fav, lam_con, rho)
            joint[i, j] = max(px * py * tau, 0.0)
    total = joint.sum()
    if total <= 0:
        return rng.poisson(lam_fav, num_sim), rng.poisson(lam_con, num_sim)
    joint = joint / total
    flat = joint.ravel()
    idx = rng.choice(len(flat), size=num_sim, p=flat)
    return xs[idx // joint.shape[1]], ys[idx % joint.shape[1]]

@st.cache_data
def simular_stats_poisson(lam_tir, lam_tpuerta, lam_corn, lam_faltas, num_sim=10000, seed=42):
    rng = np.random.default_rng(seed)
    return (
        rng.poisson(max(lam_tir, 0.01), num_sim),
        rng.poisson(max(lam_tpuerta, 0.01), num_sim),
        rng.poisson(max(lam_corn, 0.01), num_sim),
        rng.poisson(max(lam_faltas, 0.01), num_sim),
    )

try:
    df_raw = cargar_datos()
    df = calcular_feature_engineering(df_raw)
except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.stop()

colores_base_equipos = {
    "Palmeiras": "#006400", "Flamengo": "#C8102E", "Paranaense": "#CC0000",
    "Fluminense": "#8B0000", "Vasco": "#333333", "Arsenal": "#EF0107",
    "Aston villa": "#670E36", "Barcelona": "#A50044", "Bayern Munich": "#DC052D",
    "Benfica": "#E30613", "Como": "#002D62", "Freiburg": "#222222",
    "Inter": "#010E80", "Liverpool": "#C8102E", "Lyon": "#1D428A",
    "Manchester City": "#6CABDD", "Manchester United": "#DA291C",
    "Newcastle": "#241F20", "Porto": "#003399", "PSG": "#004170",
    "Real Madrid": "#00529F",
}

def generar_color_equipo(nombre):
    if nombre in colores_base_equipos:
        return colores_base_equipos[nombre]
    hash_val = int(hashlib.md5(nombre.encode("utf-8")).hexdigest(), 16)
    hue = (hash_val % 360) / 360.0
    rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

st.sidebar.header("Configuracion")

with st.sidebar.expander("Partido", expanded=True):
    ligas_disponibles = sorted([str(x) for x in df["Liga"].dropna().unique() if pd.notna(x)])
    liga_sel = st.selectbox("Liga", ligas_disponibles)
    
    df_liga = df[df["Liga"] == liga_sel]
    lista_equipos = sorted([str(x) for x in df_liga["Equipo"].unique() if pd.notna(x)])
    equipo_sel = st.selectbox("Equipo", lista_equipos)
    
    df_equipo = df_liga[df_liga["Equipo"] == equipo_sel]
    lista_niveles = sorted([str(x) for x in df_equipo["Nivel Rival"].unique() if pd.notna(x)])
    if not lista_niveles:
        lista_niveles = ["TOP", "MEDIA TABLA", "DESCENSO"]
    condicion_label = st.selectbox("Condicion", ["Local", "Visitante"])
    condicion_sel = condicion_label.lower()
    nivel_sel = st.selectbox("Nivel del Rival", lista_niveles)

df_diagnostico = df_equipo.sort_values(by="Fecha", ascending=False)
exactos_check = df_diagnostico[
    (df_diagnostico["Condición"] == condicion_sel) & (df_diagnostico["Nivel Rival"] == nivel_sel)
]
num_exactos = len(exactos_check)

if num_exactos >= 2:
    st.sidebar.success(f"{num_exactos} partidos exactos (Suficientes)")
elif num_exactos == 1:
    st.sidebar.warning("1 partido exacto -> Respaldo inteligente activo (Mínimo 2)")
else:
    st.sidebar.error("0 partidos exactos en este filtro")

with st.sidebar.expander("Lineas de Estudio"):
    linea_goles = st.slider("Goles (equipo)", 0.5, 3.5, 1.5, 0.5)
    linea_tiros = st.slider("Tiros Totales", 5.0, 25.0, 12.5, 0.5)
    linea_tiros_puerta = st.slider("Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
    linea_corners = st.slider("Corners", 1.0, 15.0, 5.5, 0.5)
    linea_faltas = st.slider("Faltas", 5.0, 25.0, 10.5, 0.5)
    linea_total_partido = st.slider("Total goles partido (Over)", 0.5, 4.5, 2.5, 0.5)

with st.sidebar.expander("Cuotas 1X2 / BTTS / DNB"):
    cuota_casa_1 = st.number_input("Victoria (1)", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_casa_x = st.number_input("Empate (X)", min_value=1.01, value=3.40, step=0.01, format="%.2f")
    cuota_casa_2 = st.number_input("Derrota (2)", min_value=1.01, value=4.20, step=0.01, format="%.2f")
    cuota_casa_1x = st.number_input("Doble Oportunidad (1X)", min_value=1.01, value=1.22, step=0.01, format="%.2f")
    cuota_casa_x2 = st.number_input("Doble Oportunidad (X2)", min_value=1.01, value=1.95, step=0.01, format="%.2f")
    cuota_casa_btts_si = st.number_input("BTTS Si", min_value=1.01, value=1.75, step=0.01, format="%.2f")
    cuota_casa_btts_no = st.number_input("BTTS No", min_value=1.01, value=2.05, step=0.01, format="%.2f")
    cuota_casa_dnb = st.number_input("DNB", min_value=1.01, value=1.35, step=0.01, format="%.2f")

with st.sidebar.expander("Cuotas de Lineas (Over)"):
    cuota_over_goles = st.number_input(f"Over {linea_goles} Goles", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_tiros = st.number_input(f"Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_over_puerta = st.number_input(f"Over {linea_tiros_puerta} a Puerta", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_over_corners = st.number_input(f"Over {linea_corners} Corners", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_faltas = st.number_input(f"Over {linea_faltas} Faltas", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_over_total = st.number_input(f"Over {linea_total_partido} Goles partido", min_value=1.01, value=1.90, step=0.01, format="%.2f")

with st.sidebar.expander("Modelo estadistico", expanded=True):
    st.markdown("**Shrinkage**")
    shrink_opt = st.radio(
        "Shrinkage hacia media del nivel",
        options=["ON", "OFF"],
        index=0,
        horizontal=True,
        key="radio_shrink_eq",
        label_visibility="collapsed"
    )
    usar_shrinkage = (shrink_opt == "ON")
    k_shrink = st.slider("Fuerza prior (k)", 1.0, 15.0, 5.0, 1.0, key="slider_k_eq", disabled=not usar_shrinkage)
    st.caption("Recomendado ON en 5.0")

    st.markdown("**Dixon-Coles**")
    dc_opt = st.radio(
        "Dixon-Coles",
        options=["ON", "OFF"],
        index=0,
        horizontal=True,
        key="radio_dc_eq",
        label_visibility="collapsed"
    )
    usar_dc = (dc_opt == "ON")
    rho_dc = st.slider("rho Dixon-Coles", -0.20, 0.05, -0.10, 0.01, key="slider_rho_eq", disabled=not usar_dc)
    st.caption("Recomendado ON en -0.10")

color_equipo = generar_color_equipo(equipo_sel)
equipo_sel_html = html.escape(equipo_sel)
nivel_sel_html = html.escape(nivel_sel)
liga_sel_html = html.escape(liga_sel)

st.markdown(f"""
<style>
.stApp {{ background-color: #0B0F19; color: #F3F4F6; }}
.stSidebar {{ background-color: #111827; }}
.header-box {{
    background: linear-gradient(90deg, {color_equipo} 0%, #1F2937 100%);
    padding: 22px 28px; border-radius: 14px; color: white;
    font-weight: 700; font-size: 24px; margin-bottom: 20px; text-align: center;
}}
.veredicto-box {{
    padding: 16px 20px; border-radius: 12px; background-color: #1F2937;
    border-left: 5px solid {color_equipo}; margin-bottom: 16px; font-size: 16px;
}}
.value-box {{ padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 14px; }}
.value-yes {{ background-color: #064e3b; border-left: 4px solid #10b981; }}
.value-no {{ background-color: #1f2937; border-left: 4px solid #4b5563; }}
.top-pick-box {{ background: linear-gradient(135deg, #065f46 0%, #111827 100%); padding: 20px; border-radius: 12px; border: 2px solid #10b981; margin-bottom: 20px; }}
div[data-testid="stMetric"] {{ background-color: #1F2937; padding: 12px 16px; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

def renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa):
    df_adn = pd.DataFrame({
        "Metrica": ["Ataque", "Volumen Tiros", "Precision", "Corners", "Disciplina"],
        "Puntuacion": [
            min(round(lam_f * 3.33, 1), 10.0),
            min(round(lam_t / 2.5, 1), 10.0),
            min(round(lam_tp * 1.66, 1), 10.0),
            min(round(lam_co / 1.5, 1), 10.0),
            min(round((25 - lam_fa) / 2.5, 1), 10.0),
        ],
    })
    chart = alt.Chart(df_adn).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X("Puntuacion:Q", scale=alt.Scale(domain=[0, 10]), title=None),
        y=alt.Y("Metrica:N", sort="-x", title=None),
        color=alt.value(color_equipo),
        tooltip=["Metrica", "Puntuacion"],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

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

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, muestra_pequena=False):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake = calcular_kelly(prob, cuota_casa) if es_value else 0.0
    kelly_txt = f" | Half-Kelly: <b>{stake}% bank</b>" if es_value else ""
    caution = " (muestra pequeña)" if muestra_pequena and es_value else ""
    st.markdown(
        f'<div class="value-box {clase}"><b>{html.escape(nombre)}</b>{caution}<br>'
        f"Prob: <b>{prob:.1f}%</b> | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_txt}<br>"
        f'<span style="color:{color_ev}; font-weight:bold; font-size:15px;">'
        f"EV: {ev:+.2%} -> {'VALUE' if es_value else 'Sin valor'}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("### GoalMetrics - Análisis de Equipos (Híbrido)")
st.caption("Simulación con Poisson, Dixon-Coles y Ensemble XGBoost.")

if "analizado_equipos" not in st.session_state:
    st.session_state.analizado_equipos = False

c1, c2, _ = st.columns([1.2, 1, 4])
with c1:
    if st.button("Analizar", type="primary", use_container_width=True):
        st.session_state.analizado_equipos = True
with c2:
    if st.button("Limpiar", use_container_width=True):
        st.session_state.analizado_equipos = False
        st.rerun()

if st.session_state.analizado_equipos:
    df_equipo = df_liga[df_liga["Equipo"] == equipo_sel].copy()
    if "Fecha" in df_equipo.columns:
        df_equipo = df_equipo.sort_values("Fecha")

    if "Condición" in df_equipo.columns and "Nivel Rival" in df_equipo.columns:
        df_exactos = df_equipo[(df_equipo["Condición"] == condicion_sel) & (df_equipo["Nivel Rival"] == nivel_sel)].copy()
    else:
        df_exactos = pd.DataFrame()

    if len(df_exactos) == 0:
        st.error(f"❌ No se puede realizar el análisis: Hay 0 partidos exactos registrados para **{equipo_sel}** como **{condicion_label}** contra rivales nivel **{nivel_sel}** en la liga **{liga_sel}**.")
        st.stop()

    UMBRAL_MINIMO = 2
    t_target = obtener_peso_tier(nivel_sel)
    historial_list = []

    for _, row in df_exactos.iterrows():
        r = row.to_dict()
        r["Factor_Ajuste"] = 1.0
        r["Tipo_Uso"] = f"Exacto ({condicion_label} vs {nivel_sel})"
        r["Peso_Contexto"] = 1.0
        historial_list.append(r)
        
    fuente_datos = f"Exactos ({len(historial_list)} partidos)"

    if len(historial_list) < UMBRAL_MINIMO:
        if "Condición" in df_equipo.columns:
            df_misma_cond = df_equipo[(df_equipo["Condición"] == condicion_sel) & (df_equipo["Nivel Rival"] != nivel_sel)].copy()
        else:
            df_misma_cond = pd.DataFrame()

        faltantes = UMBRAL_MINIMO - len(historial_list)
        comodines_tier = df_misma_cond.tail(faltantes)

        for _, row in comodines_tier.iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.85
            historial_list.append(r)
        if len(historial_list) > len(df_exactos):
            fuente_datos = "Muestra mixta (1 Exacto + Respaldo ajustado por Tier)"

    if len(historial_list) < UMBRAL_MINIMO:
        opuesto_lower = "local" if condicion_sel == "visitante" else "visitante"
        if "Condición" in df_equipo.columns:
            df_contrarios = df_equipo[df_equipo["Condición"] == opuesto_lower].copy()
        else:
            df_contrarios = pd.DataFrame()

        faltantes_cruzados = UMBRAL_MINIMO - len(historial_list)
        comodines_cruzados = df_contrarios.tail(faltantes_cruzados)

        for _, row in comodines_cruzados.iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.75
            historial_list.append(r)
        fuente_datos = "Muestra adaptada con respaldo cruzado y ajuste de tier"

    historial = pd.DataFrame(historial_list)

    cols_numericas_ajustar = ["Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Atajadas", "Amarillas", "Rojas", "Corners Rival", "Tiros a Puerta Rival"]
    for col in cols_numericas_ajustar:
        if col in historial.columns:
            historial[col] = historial[col] * historial["Factor_Ajuste"]

    if "Goles" in historial.columns and "Goles Rival" in historial.columns:
        historial["Diff_Goles"] = historial["Goles"] - historial["Goles Rival"]
    if "Goles Rival" in historial.columns and "Atajadas" in historial.columns:
        historial["Tiros a Puerta Rival"] = historial["Goles Rival"] + historial["Atajadas"]

    n_obs = len(historial)
    muestra_pequena = n_obs <= 2
    hoy = pd.Timestamp.today().normalize()
    if "Fecha" in historial.columns:
        historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.replace(0, 0.1)
        historial["Peso_Temporal"] = 1 / (1 + (historial["Dias_Pasados"] / 30))
    else:
        historial["Peso_Temporal"] = 1.0

    if "Peso_Contexto" not in historial.columns:
        historial["Peso_Contexto"] = 1.0

    historial["Peso_Total"] = historial["Peso_Temporal"] * historial["Peso_Contexto"]
    pesos = historial["Peso_Total"] / historial["Peso_Total"].sum()

    def prom(col):
        if col not in historial.columns or len(historial) == 0:
            return 0.05
        return round(float(np.average(historial[col].fillna(0), weights=pesos)), 4)

    lam_f_raw, lam_c_raw = prom("Goles"), prom("Goles Rival")
    lam_t_raw, lam_tp_raw = prom("Tiros"), prom("A Puerta")
    lam_co_raw, lam_fa_raw = prom("Corners"), prom("Faltas")
    lam_co_rival_raw = prom("Corners Rival") if "Corners Rival" in historial.columns else prom("Corners")

    df_nivel = df[(df["Liga"] == liga_sel) & (df["Nivel Rival"] == nivel_sel)]
    if len(df_nivel) == 0:
        df_nivel = df[df["Nivel Rival"] == nivel_sel]

    prior_f = float(df_nivel["Goles"].mean()) if len(df_nivel) else lam_f_raw
    prior_c = float(df_nivel["Goles Rival"].mean()) if len(df_nivel) and "Goles Rival" in df_nivel.columns else lam_c_raw
    prior_t = float(df_nivel["Tiros"].mean()) if len(df_nivel) and "Tiros" in df_nivel.columns else lam_t_raw
    prior_tp = float(df_nivel["A Puerta"].mean()) if len(df_nivel) and "A Puerta" in df_nivel.columns else lam_tp_raw
    prior_co = float(df_nivel["Corners"].mean()) if len(df_nivel) and "Corners" in df_nivel.columns else lam_co_raw
    prior_co_rival = float(df_nivel["Corners Rival"].mean()) if len(df_nivel) and "Corners Rival" in df_nivel.columns else lam_co_rival_raw
    prior_fa = float(df_nivel["Faltas"].mean()) if len(df_nivel) and "Faltas" in df_nivel.columns else lam_fa_raw

    if usar_shrinkage:
        lam_f = shrinkage_lambda(lam_f_raw, prior_f, n_obs, k_shrink)
        lam_c = shrinkage_lambda(lam_c_raw, prior_c, n_obs, k_shrink)
        lam_t = shrinkage_lambda(lam_t_raw, prior_t, n_obs, k_shrink)
        lam_tp = shrinkage_lambda(lam_tp_raw, prior_tp, n_obs, k_shrink)
        lam_co = shrinkage_lambda(lam_co_raw, prior_co, n_obs, k_shrink)
        lam_co_rival = shrinkage_lambda(lam_co_rival_raw, prior_co_rival, n_obs, k_shrink)
        lam_fa = shrinkage_lambda(lam_fa_raw, prior_fa, n_obs, k_shrink)
    else:
        lam_f, lam_c, lam_t, lam_tp, lam_co, lam_co_rival, lam_fa = lam_f_raw, lam_c_raw, lam_t_raw, lam_tp_raw, lam_co_raw, lam_co_rival_raw, lam_fa_raw

    num_sim = 10000
    if usar_dc:
        sg_fav, sg_con = simular_goles_dixon_coles(lam_f, lam_c, rho=rho_dc, num_sim=num_sim)
    else:
        rng = np.random.default_rng(42)
        sg_fav = rng.poisson(max(lam_f, 0.01), num_sim)
        sg_con = rng.poisson(max(lam_c, 0.01), num_sim)

    s_tir, s_tpuerta, s_corn, s_faltas = simular_stats_poisson(lam_t, lam_tp, lam_co, lam_fa, num_sim=num_sim)
    s_corn_rival = np.random.default_rng(42).poisson(max(lam_co_rival, 0.01), num_sim)

    triunfos_base = (sg_fav > sg_con).mean() * 100
    empates = (sg_fav == sg_con).mean() * 100
    derrotas_base = (sg_fav < sg_con).mean() * 100

    features_modelo = ["Goles_Media_Movil_5", "Goles_Volatilidad_5", "Tiros_Media_Movil_5", "Conversion_Tiros", "Momentum_Goles", "Diff_Goles"]
    modelo_xgb_global = entrenar_predictor_xgboost(df, features_modelo)
    
    triunfos = predecir_probabilidad_hibrida(triunfos_base, historial, features_modelo, modelo_xgb_global)
    derrotas = predecir_probabilidad_hibrida(derrotas_base, historial, features_modelo, modelo_xgb_global)
    empates = max(0.0, 100.0 - triunfos - derrotas)

    ambos_anotan = ((sg_fav > 0) & (sg_con > 0)).mean() * 100
    doble_1x, doble_x2 = triunfos + empates, derrotas + empates
    tot_sin_emp = triunfos + derrotas
    dnb = (triunfos / tot_sin_emp * 100) if tot_sin_emp > 0 else 50.0

    prob_over_goles = (sg_fav > linea_goles).mean() * 100
    prob_over_tiros = (s_tir > linea_tiros).mean() * 100
    prob_over_puerta = (s_tpuerta > linea_tiros_puerta).mean() * 100
    prob_over_corners = (s_corn > linea_corners).mean() * 100
    prob_over_faltas = (s_faltas > linea_faltas).mean() * 100
    prob_over_total = ((sg_fav + sg_con) > linea_total_partido).mean() * 100

    marcadores = [f"{f}-{c}" for f, c in zip(sg_fav, sg_con)]
    conteo = Counter(marcadores)
    marcador_mas_comun = conteo.most_common(1)[0][0]

    if triunfos > 50:
        veredicto = f"Tendencia Fuerte - Marcador proyectado {marcador_mas_comun}"
    elif derrotas > 50:
        veredicto = f"Alerta de Complicacion - Marcador proyectado {marcador_mas_comun}"
    else:
        veredicto = f"Partido Muy Parejo - Marcador proyectado {marcador_mas_comun}"

    st.markdown(f'<div class="header-box">{liga_sel_html.upper()} | {equipo_sel_html.upper()} - {condicion_label.upper()} vs {nivel_sel_html.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="veredicto-box"><b>Veredicto:</b> {html.escape(veredicto)}</div>', unsafe_allow_html=True)
    st.caption(f"Base: {n_obs} partidos - {fuente_datos} | Ensemble Híbrido Activo")

    if muestra_pequena:
        st.warning("Muestra pequeña (Respaldo activo con 1 partido exacto). Interpreta con cautela.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Resumen", "Value Bet", "🤖 Panel Inteligente", "Solidez Defensiva", "Fase Ofensiva", "Líneas y Gráficos", "Detalle"])

    with tab1:
        st.subheader("ADN del Equipo")
        renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa)
        a, b, c, d = st.columns(4)
        a.metric("Victoria", f"{triunfos:.1f}%")
        b.metric("Empate", f"{empates:.1f}%")
        c.metric("Derrota", f"{derrotas:.1f}%")
        d.metric("BTTS", f"{ambos_anotan:.1f}%")
        e, f, g = st.columns(3)
        e.metric("1X", f"{doble_1x:.1f}%")
        f.metric("X2", f"{doble_x2:.1f}%")
        g.metric("DNB", f"{dnb:.1f}%")
        m1, m2, m3, m4 = st.columns(4)
        if usar_shrinkage:
            m1.metric("Goles", f"{lam_f:.2f}", delta=f"raw {lam_f_raw:.2f}")
            m2.metric("Goles Rival", f"{lam_c:.2f}", delta=f"raw {lam_c_raw:.2f}")
        else:
            m1.metric("Goles", f"{lam_f:.2f}")
            m2.metric("Goles Rival", f"{lam_c:.2f}")
        m3.metric("Tiros", f"{lam_t:.1f}")
        m4.metric("Corners", f"{lam_co:.1f}")

    with tab2:
        st.subheader("Value Bet y Half-Kelly")
        items_1x2 = [
            ("Victoria (1)", round(100/triunfos,2) if triunfos>0 else 99, cuota_casa_1, calcular_ev(triunfos, cuota_casa_1), triunfos),
            ("Empate (X)", round(100/empates,2) if empates>0 else 99, cuota_casa_x, calcular_ev(empates, cuota_casa_x), empates),
            ("Derrota (2)", round(100/derrotas,2) if derrotas>0 else 99, cuota_casa_2, calcular_ev(derrotas, cuota_casa_2), derrotas),
            ("1X", round(100/doble_1x,2) if doble_1x>0 else 99, cuota_casa_1x, calcular_ev(doble_1x, cuota_casa_1x), doble_1x),
            ("X2", round(100/doble_x2,2) if doble_x2>0 else 99, cuota_casa_x2, calcular_ev(doble_x2, cuota_casa_x2), doble_x2),
            ("BTTS Si", round(100/ambos_anotan,2) if ambos_anotan>0 else 99, cuota_casa_btts_si, calcular_ev(ambos_anotan, cuota_casa_btts_si), ambos_anotan),
            ("BTTS No", round(100/(100-ambos_anotan),2) if ambos_anotan<100 else 99, cuota_casa_btts_no, calcular_ev(100-ambos_anotan, cuota_casa_btts_no), 100-ambos_anotan),
            ("DNB", round(100/dnb,2) if dnb>0 else 99, cuota_casa_dnb, calcular_ev(dnb, cuota_casa_dnb), dnb),
        ]
        items_1x2.sort(key=lambda x: x[3], reverse=True)
        ca, cb = st.columns(2)
        mid = (len(items_1x2)+1)//2
        with ca:
            for it in items_1x2[:mid]:
                mostrar_value(*it, muestra_pequena=muestra_pequena)
        with cb:
            for it in items_1x2[mid:]:
                mostrar_value(*it, muestra_pequena=muestra_pequena)
        st.markdown("---")
        st.subheader("Lineas (Over)")
        items_lineas = [
            (f"Over {linea_goles} Goles", round(100/prob_over_goles,2) if prob_over_goles>0 else 99, cuota_over_goles, calcular_ev(prob_over_goles, cuota_over_goles), prob_over_goles),
            (f"Over {linea_tiros} Tiros", round(100/prob_over_tiros,2) if prob_over_tiros>0 else 99, cuota_over_tiros, calcular_ev(prob_over_tiros, cuota_over_tiros), prob_over_tiros),
            (f"Over {linea_tiros_puerta} a Puerta", round(100/prob_over_puerta,2) if prob_over_puerta>0 else 99, cuota_over_puerta, calcular_ev(prob_over_puerta, cuota_over_puerta), prob_over_puerta),
            (f"Over {linea_corners} Corners", round(100/prob_over_corners,2) if prob_over_corners>0 else 99, cuota_over_corners, calcular_ev(prob_over_corners, cuota_over_corners), prob_over_corners),
            (f"Over {linea_faltas} Faltas", round(100/prob_over_faltas,2) if prob_over_faltas>0 else 99, cuota_over_faltas, calcular_ev(prob_over_faltas, cuota_over_faltas), prob_over_faltas),
            (f"Over {linea_total_partido} Goles partido", round(100/prob_over_total,2) if prob_over_total>0 else 99, cuota_over_total, calcular_ev(prob_over_total, cuota_over_total), prob_over_total),
        ]
        items_lineas.sort(key=lambda x: x[3], reverse=True)
        cc, cd = st.columns(2)
        mid2 = (len(items_lineas)+1)//2
        with cc:
            for it in items_lineas[:mid2]:
                mostrar_value(*it, muestra_pequena=muestra_pequena)
        with cd:
            for it in items_lineas[mid2:]:
                mostrar_value(*it, muestra_pequena=muestra_pequena)

    with tab3:
        st.subheader("🤖 Panel Inteligente & Parlay Híbrido")
        todos_mercados_eq = items_1x2 + items_lineas
        lista_mercados_eq_dict = [{"nombre": m[0], "prob": m[4], "cuota": m[2], "ev": m[3]} for m in todos_mercados_eq]
        value_bets_eq = [m for m in lista_mercados_eq_dict if m["ev"] > 0]

        if value_bets_eq:
            top_eq = max(value_bets_eq, key=lambda x: x["ev"])
            stake_top_eq = calcular_kelly(top_eq["prob"], top_eq["cuota"])
            st.markdown(
                f'<div class="top-pick-box">'
                f'<h3>🏆 La Joya del Partido (Top Value Bet)</h3>'
                f'<p style="font-size: 16px; margin-bottom: 8px;">Mercado recomendado: <b>{html.escape(top_eq["nombre"])}</b></p>'
                f'<ul>'
                f'<li>Probabilidad Híbrida del Modelo: <b>{top_eq["prob"]:.1f}%</b></li>'
                f'<li>Cuota Casa: <b>{top_eq["cuota"]}</b></li>'
                f'<li><b>EV Matemático: {top_eq["ev"]:+.2%}</b></li>'
                f'</ul>'
                f'<p style="color: #10b981; font-weight: bold; margin-top: 10px;">👉 Sugerencia de Stake: <b>{stake_top_eq}% del Bank (Half-Kelly)</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ No hay mercados con EV positivo estricto en este momento.")

        st.markdown("---")
        st.subheader("🔗 Constructor de Combinada Inteligente (Parlay)")
        nombres_mercados_eq = [m["nombre"] for m in lista_mercados_eq_dict]
        parlay_eq = st.multiselect("Elige los mercados para tu combinada:", options=nombres_mercados_eq, key="parlay_eq_input")

        mercados_goles_bool = {
            "Victoria (1)": sg_fav > sg_con,
            "Empate (X)": sg_fav == sg_con,
            "Derrota (2)": sg_fav < sg_con,
            "1X": sg_fav >= sg_con,
            "X2": sg_fav <= sg_con,
            "BTTS Si": (sg_fav > 0) & (sg_con > 0),
            "BTTS No": ~((sg_fav > 0) & (sg_con > 0)),
            f"Over {linea_goles} Goles": sg_fav > linea_goles,
            f"Over {linea_total_partido} Goles partido": (sg_fav + sg_con) > linea_total_partido,
        }

        if parlay_eq:
            legs_goles = [n for n in parlay_eq if n in mercados_goles_bool]
            legs_independientes = [n for n in parlay_eq if n not in mercados_goles_bool]

            if legs_goles:
                mask_conjunta = np.ones(num_sim, dtype=bool)
                for n in legs_goles:
                    mask_conjunta &= mercados_goles_bool[n]
                p_goles_conjunta = float(mask_conjunta.mean())
            else:
                p_goles_conjunta = 1.0

            p_indep = 1.0
            for n in legs_independientes:
                m_info = next(m for m in lista_mercados_eq_dict if m["nombre"] == n)
                p_indep *= (m_info["prob"] / 100.0)

            p_conj = p_goles_conjunta * p_indep
            p_conj_pct = p_conj * 100.0
            c_justa_parlay = round(100 / p_conj_pct, 2) if p_conj_pct > 0 else 99.0

            st.markdown(f"**Probabilidad Conjunta:** `{p_conj_pct:.2f}%`")
            st.markdown(f"**Cuota Justa Combinada:** `{c_justa_parlay}`")

            cuota_casa_parlay_eq = st.number_input("Cuota total que paga la casa:", min_value=1.01, value=c_justa_parlay * 0.95, step=0.05, format="%.2f", key="cuota_parlay_eq_input")
            ev_parlay_eq = calcular_ev(p_conj_pct, cuota_casa_parlay_eq)
            stake_parlay_eq = calcular_kelly(p_conj_pct, cuota_casa_parlay_eq)
            
            col_ep1, col_ep2 = st.columns(2)
            col_ep1.metric("EV de la Combinada", f"{ev_parlay_eq:+.2%}")
            if ev_parlay_eq > 0:
                col_ep2.metric("Stake Sugerido", f"{stake_parlay_eq}% del Bank")
                st.success(f"🎉 ¡Combinada con EV positivo! Stake recomendado: {stake_parlay_eq}%.")
            else:
                col_ep2.metric("Stake Sugerido", "0%")
                st.warning("⚠️ EV Negativo.")

    with tab4:
        st.subheader("🛡️ Solidez Defensiva (Comportamiento sin balón / Rival & Disciplina)")
        prom_tp_rival = prom("Tiros a Puerta Rival")
        prom_g_rival = prom("Goles Rival")
        prom_corners_rival = prom("Corners Rival") if "Corners Rival" in historial.columns else 0.0
        prom_faltas = prom("Faltas") if "Faltas" in historial.columns else 0.0
        prom_amarillas = prom("Amarillas") if "Amarillas" in historial.columns else 0.0
        prom_rojas = prom("Rojas") if "Rojas" in historial.columns else 0.0
        
        ratio_tp_gol_contra = prom_tp_rival / prom_g_rival if prom_g_rival > 0 else 0.0

        d1, d2, d3 = st.columns(3)
        d1.metric("Tiros a Puerta Permitidos", f"{prom_tp_rival:.1f} xG")
        d2.metric("Goles en Contra (Prom.)", f"{prom_g_rival:.2f} xG")
        d3.metric("Tiros a Puerta por Gol en Contra", f"{ratio_tp_gol_contra:.1f} tiros", "Eficiencia defensiva rival")

        d4, d5, d6 = st.columns(3)
        d4.metric("Corners Concedidos (Prom.)", f"{prom_corners_rival:.1f} xG")
        d5.metric("Faltas Propias (Prom.)", f"{prom_faltas:.1f} xG")
        d6.metric("Tarjetas (Amarillas / Rojas)", f"{prom_amarillas:.1f} A / {prom_rojas:.1f} R")

        if prom_g_rival > 0:
            feedback_def = f"En la faceta defensiva dentro de este contexto analítico, el equipo permite un promedio de {prom_tp_rival:.1f} tiros a puerta por partido y recibe {prom_g_rival:.2f} goles, lo que equivale a recibir un gol cada {ratio_tp_gol_contra:.1f} tiros a puerta en contra. Asimismo, concede un promedio de {prom_corners_rival:.1f} corners al rival, comete {prom_faltas:.1f} faltas y recibe {prom_amarillas:.1f} amarillas y {prom_rojas:.1f} rojas."
        else:
            feedback_def = f"El equipo mantiene una defensa sólida en esta muestra, permitiendo {prom_tp_rival:.1f} tiros a puerta en promedio sin encajar goles en contra, cometiendo {prom_faltas:.1f} faltas y recibiendo {prom_amarillas:.1f} amarillas y {prom_rojas:.1f} rojas."
        st.markdown(f'<div class="veredicto-box"><b>🔍 Retroalimentación Defensiva:</b><br>{feedback_def}</div>', unsafe_allow_html=True)

    with tab5:
        st.subheader("⚡ Fase Ofensiva (Producción propia con balón)")
        prom_tiros = prom("Tiros")
        prom_tp = prom("A Puerta")
        prom_goles = prom("Goles")
        prom_corners = prom("Corners") if "Corners" in historial.columns else 0.0
        
        ratio_tiros_gol = prom_tiros / prom_goles if prom_goles > 0 else 0.0
        ratio_tp_gol = prom_tp / prom_goles if prom_goles > 0 else 0.0

        o1, o2, o3 = st.columns(3)
        o1.metric("Tiros Totales (Prom.)", f"{prom_tiros:.1f} xG")
        o2.metric("Tiros a Puerta (Prom.)", f"{prom_tp:.1f} xG")
        o3.metric("Goles a Favor (Prom.)", f"{prom_goles:.2f} xG")

        o4, o5, o6 = st.columns(3)
        o4.metric("Tiros Totales por Gol", f"{ratio_tiros_gol:.1f} tiros", "Conversión global")
        o5.metric("Tiros a Puerta por Gol", f"{ratio_tp_gol:.1f} tiros", "Conversión a puerta")
        o6.metric("Corners a Favor (Prom.)", f"{prom_corners:.1f} xG")

        if prom_goles > 0:
            feedback_of = f"En la faceta ofensiva dentro de este contexto analítico, el equipo genera {prom_tiros:.1f} tiros totales y {prom_tp:.1f} tiros a puerta por encuentro, anotando {prom_goles:.2f} goles. Necesita en promedio {ratio_tiros_gol:.1f} tiros totales (o {ratio_tp_gol:.1f} a puerta) para convertir un gol. Además, cobra un promedio de {prom_corners:.1f} corners."
        else:
            feedback_of = f"El equipo registra una producción ofensiva de {prom_tiros:.1f} tiros y {prom_tp:.1f} a puerta, sin goles anotados en esta muestra específica."
        st.markdown(f'<div class="veredicto-box"><b>🔍 Retroalimentación Ofensiva:</b><br>{feedback_of}</div>', unsafe_allow_html=True)

    with tab6:
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric(f"Goles > {linea_goles}", f"{prob_over_goles:.1f}%")
        lc2.metric(f"Total > {linea_total_partido}", f"{prob_over_total:.1f}%")
        lc3.metric(f"Tiros > {linea_tiros}", f"{prob_over_tiros:.1f}%")
        
        st.markdown("---")
        st.subheader("🎯 Matriz de Probabilidad del Resultado Exacto")
        st.caption("Distribución de probabilidad cruzada entre goles del equipo y del rival según la simulación estocástica.")

        max_g = 5
        matriz_probs = np.zeros((max_g + 1, max_g + 1))
        text_data = []
        
        for f_g in range(max_g + 1):
            fila_texto = []
            for c_g in range(max_g + 1):
                prob = float(((sg_fav == f_g) & (sg_con == c_g)).mean() * 100.0)
                matriz_probs[f_g, c_g] = prob
                if prob >= 0.1:
                    fila_texto.append(f"{prob:.1f}%")
                else:
                    fila_texto.append("<0.1%")
            text_data.append(fila_texto)

        fig_matrix = go.Figure(data=go.Heatmap(
            z=matriz_probs,
            x=[str(i) for i in range(max_g + 1)],
            y=[str(i) for i in range(max_g + 1)],
            text=text_data,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorscale=[[0, "#111827"], [0.5, "#1d4ed8"], [1, "#10b981"]],
            showscale=False
        ))
        
        fig_matrix.update_layout(
            title=f"Goles Rival (Eje X) vs Goles {equipo_sel} (Eje Y)",
            xaxis_title="Goles Rival",
            yaxis_title=f"Goles {equipo_sel}",
            paper_bgcolor="#0B0F19",
            plot_bgcolor="#0B0F19",
            font=dict(color="#F3F4F6"),
            height=380,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Probabilidad Acumulada de Goles (Over X)")
        st.caption("Probabilidad de que el equipo anote más de X goles (Over acumulado).")

        max_g_sim = int(max(sg_fav)) + 2
        goal_vals = [float(i + 0.5) for i in range(max_g_sim)]
        cum_probs_goles = [float((sg_fav > g).mean() * 100) for g in goal_vals]

        fig_cum_goles = go.Figure(data=go.Scatter(
            x=goal_vals,
            y=cum_probs_goles,
            mode='lines+markers+text',
            text=[f"{p:.1f}%" for p in cum_probs_goles],
            textposition="top center",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8)
        ))
        fig_cum_goles.update_layout(
            title=f"Probabilidad Acumulada de Goles (Over X) - {equipo_sel}",
            xaxis=dict(
                title="Línea de Goles (Over)",
                tickmode='array',
                tickvals=goal_vals,
                ticktext=[str(g) for g in goal_vals],
                color="#F3F4F6"
            ),
            yaxis_title="Probabilidad (%)",
            paper_bgcolor="#0B0F19",
            plot_bgcolor="#0B0F19",
            font=dict(color="#F3F4F6"),
            height=380,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_cum_goles, use_container_width=True)

        st.markdown("---")
        st.subheader("⛳ Probabilidad Acumulada de Córners (Over X)")
        st.caption("Probabilidad acumulada de superar cada línea de córners simulada para el equipo.")

        max_c = int(max(s_corn)) + 2
        corner_vals = list(range(0, max_c))
        cum_probs_corners = [float((s_corn > c).mean() * 100) for c in corner_vals]

        fig_cum_corners = go.Figure(data=go.Scatter(
            x=corner_vals,
            y=cum_probs_corners,
            mode='lines+markers+text',
            text=[f"{p:.1f}%" for p in cum_probs_corners],
            textposition="top center",
            line=dict(color=color_equipo, width=3),
            marker=dict(size=8)
        ))
        fig_cum_corners.update_layout(
            title=f"Probabilidad Acumulada de Córners (Over X) - {equipo_sel}",
            xaxis=dict(
                title="Línea de Córners (> X)",
                tickmode='array',
                tickvals=corner_vals,
                ticktext=[str(c) for c in corner_vals],
                color="#F3F4F6"
            ),
            yaxis_title="Probabilidad (%)",
            paper_bgcolor="#0B0F19",
            plot_bgcolor="#0B0F19",
            font=dict(color="#F3F4F6"),
            height=380,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_cum_corners, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Gráfico de Dispersión: Tiros a Puerta vs Goles")
        st.caption("Relación partido a partido entre los Tiros a Puerta (Eje X) y los Goles Anotados (Eje Y) en la muestra.")

        if not historial.empty and "A Puerta" in historial.columns and "Goles" in historial.columns:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=historial["A Puerta"],
                y=historial["Goles"],
                mode="markers+text",
                text=historial.get("Rival", ""),
                textposition="top center",
                marker=dict(
                    size=12,
                    color=color_equipo,
                    line=dict(width=2, color="white"),
                    opacity=0.85
                ),
                hovertemplate="<b>Rival:</b> %{text}<br><b>Tiros a Puerta:</b> %{x}<br><b>Goles Anotados:</b> %{y}<extra></extra>"
            ))
            fig_scatter.update_layout(
                title="Tiros a Puerta vs Goles por Partido",
                xaxis_title="Tiros a Puerta",
                yaxis_title="Goles Anotados",
                paper_bgcolor="#0B0F19",
                plot_bgcolor="#0B0F19",
                font=dict(color="#F3F4F6"),
                height=380,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab7:
        h_mostrar = historial.copy().sort_values(by="Fecha", ascending=False)
        h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
        cols = [c for c in ["Fecha", "Liga", "Condición", "Rival", "Nivel Rival", "Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Tipo_Uso", "Factor_Ajuste"] if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)
else:
    st.info("Configura el partido en la barra lateral y pulsa Analizar.")
