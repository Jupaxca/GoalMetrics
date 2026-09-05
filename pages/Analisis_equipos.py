import html
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from collections import Counter
import hashlib
import colorsys
import unicodedata
import requests

try:
    import xgboost as xgb
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False

st.set_page_config(
    page_title="GoalMetrics | Análisis de Equipos (Híbrido Pro)",
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

def _entrenar_xgboost_real(df_historico, features_modelo):
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

def bootstrap_lambda_intervalo(valores, pesos=None, n_bootstrap=500, alpha=0.05):
    if len(valores) == 0:
        return 0.0, 0.0, 0.0
    vals = np.array(valores)
    if pesos is not None:
        p = np.array(pesos)
        p = p / p.sum()
    else:
        p = None
        
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(vals), size=len(vals), replace=True, p=p)
        muestra = vals[indices]
        boot_means.append(np.mean(muestra))
        
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(vals)), float(lower), float(upper)

def calcular_backtesting_retrospectivo(historial_filtrado):
    """Calcula el backtesting utilizando estrictamente la muestra filtrada y procesada del escenario."""
    if len(historial_filtrado) < 5:
        return None, None
    y_true, y_prob = [], []
    sub_df = historial_filtrado.tail(30)
    for i in range(2, len(sub_df)):
        train_window = sub_df.iloc[:i]
        test_row = sub_df.iloc[i:i+1]
        g_fav = test_row["Goles"].values[0]
        g_con = test_row["Goles Rival"].values[0]
        actual_win = 1 if g_fav > g_con else 0
        mean_diff = train_window["Diff_Goles"].mean() if "Diff_Goles" in train_window.columns else 0.0
        prob_est = 1.0 / (1.0 + np.exp(-mean_diff))
        y_true.append(actual_win)
        y_prob.append(np.clip(prob_est, 0.01, 0.99))
    if not y_true:
        return None, None
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    brier_score = np.mean((y_prob - y_true) ** 2)
    return round(float(log_loss), 4), round(float(brier_score), 4)

def generar_analisis_dinamico(equipo, condicion, nivel, n_obs, lam_f, lam_c, lam_t, lam_tp, lam_co, triunfos, ambos_anotan, prob_over_goles, prob_over_corners, prob_over_puerta):
    """Genera un reporte analítico dinámico adaptado y personalizado según las probabilidades reales del equipo."""
    
    if triunfos >= 55:
        perfil = "altamente proactivo, dominante y con clara tendencia a volcar el juego en campo rival"
    elif triunfos >= 40:
        perfil = "competitivo, equilibrado y con alta capacidad de alternar el ritmo del juego"
    else:
        perfil = "de alta exigencia táctica, con tramos de repliegue y partidos cerrados"

    mercados_destacados = []
    if prob_over_goles >= 55.0:
        mercados_destacados.append(f"<b>Over de Goles del equipo</b> (alta probabilidad del <b>{prob_over_goles:.1f}%</b>)")
    if prob_over_puerta >= 55.0:
        mercados_destacados.append(f"<b>Líneas de Tiros a Puerta</b> (respaldado por un promedio de <b>{lam_tp:.1f}</b> remates al arco)")
    if prob_over_corners >= 55.0:
        mercados_destacados.append(f"<b>Mercado de Córners / Saques de Esquina</b> (proyectando una media de <b>{lam_co:.1f}</b>)")
    if ambos_anotan >= 55.0:
        mercados_destacados.append(f"<b>Ambos Anotan (BTTS Sí)</b> con un soporte del <b>{ambos_anotan:.1f}%</b>")
    
    if not mercados_destacados:
        recomendacion_mercado = "escenarios conservadores, enfocándose en dobles oportunidades o líneas bajas debido a la paridad estadística del emparejamiento."
    else:
        recomendacion_mercado = "apostar con mayor probabilidad de éxito por: " + ", ".join(mercados_destacados) + "."

    texto = (
        f"<b>Reporte Analítico Táctico:</b> Para el planteamiento de <b>{equipo}</b> como <b>{condicion}</b> "
        f"frente a bloques de nivel <b>{nivel}</b> (muestra analizada de <b>{n_obs} partidos</b>), el modelo estructural "
        f"proyecta un comportamiento <b>{perfil}</b>. <br><br>"
        f"• <b>Producción Ofensiva y Remates:</b> El equipo registra una expectativa de <b>{lam_f:.2f} goles</b>, respaldada por un volumen "
        f"de <b>{lam_t:.1f} tiros totales</b> y <b>{lam_tp:.1f} remates dirigidos a puerta</b> por encuentro.<br>"
        f"• <b>Solidez y Exposición Defensiva:</b> Permite una media de <b>{lam_c:.2f} goles en contra</b>, situando la probabilidad "
        f"de que ambos marquen en un <b>{ambos_anotan:.1f}%</b>.<br>"
        f"• <b>Dinámica de Córners:</b> Se anticipa una media de <b>{lam_co:.1f} saques de esquina</b> favorables.<br>"
        f"• <b>Implicaciones de Mercado (Basado en Probabilidades):</b> De acuerdo con las simulaciones, lo más probable a cumplirse y donde se concentra el mayor valor analítico es en {recomendacion_mercado}"
    )
    return texto

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
        f_cond, tipo_cond = 1.0, "Misma condición"
    else:
        if condicion_buscada == "visitante" and cond_partido == "local":
            f_cond, tipo_cond = 0.90, "Cruzado (Casa -> Fuera)"
        elif condicion_buscada == "local" and cond_partido == "visitante":
            f_cond, tipo_cond = 1.05, "Cruzado (Fuera -> Casa)"
        else:
            f_cond, tipo_cond = 1.0, "Cruzado Estándar"

    diff = tier_objetivo - t_match
    if diff == 0:
        f_tier, tipo_tier = 1.0, "Tier equivalente"
    elif diff > 0:
        f_tier, tipo_tier = max(0.65, 1.0 - (diff * 0.12)), "Ajuste a la baja"
    else:
        f_tier, tipo_tier = min(1.35, 1.0 + (abs(diff) * 0.10)), "Ajuste al alza"

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
    "Arsenal": "#EF0107", "Aston villa": "#670E36", "Atletico de Madrid": "#CB352C",
    "Barcelona": "#A50044", "Bayern Munchen": "#DC052D", "Benfica": "#E30613",
    "Betis": "#009B48", "Chelsea": "#034694", "Como": "#002D62",
    "Dortmund": "#FDE100", "Flamengo": "#C8102E", "Fluminense": "#8B0000",
    "Freiburg": "#222222", "Inter": "#010E80", "Juventus": "#000000",
    "Liverpool": "#C8102E", "Lyon": "#1D428A", "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C", "Monaco": "#ED1C24", "Newcastle": "#241F20",
    "Palmeiras": "#006400", "Paranaense": "#CC0000", "Porto": "#003399",
    "PSG": "#004170", "Racing club": "#00529F", "Real Madrid": "#00529F",
    "Real sociedad": "#006699", "Vasco": "#333333"
}

def generar_color_equipo(nombre):
    for k, v in colores_base_equipos.items():
        if k.lower() in nombre.lower() or nombre.lower() in k.lower():
            return v
    hash_val = int(hashlib.md5(nombre.encode("utf-8")).hexdigest(), 16)
    hue = (hash_val % 360) / 360.0
    rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

def normalizar_texto(texto):
    if not texto:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower().strip()

def obtener_iniciales(nombre):
    partes = str(nombre).strip().split()
    if len(partes) >= 2:
        return (partes[0][0] + partes[1][0]).upper()
    return str(nombre)[:2].upper()

@st.cache_data(ttl=86400)
def obtener_logo_equipo(nombre):
    nombre_limpio = str(nombre).strip()
    if not nombre_limpio:
        return None

    aliases = {
        "bayern munchen": "FC Bayern Munich",
        "bayern munich": "FC Bayern Munich",
        "bayern": "FC Bayern Munich",
        "real madrid": "Real Madrid CF",
        "barcelona": "FC Barcelona",
        "manchester city": "Manchester City F.C.",
        "manchester united": "Manchester United F.C.",
        "psg": "Paris Saint-Germain F.C.",
        "inter": "Inter Milan",
        "atletico": "Atletico Madrid",
        "atletico de madrid": "Atletico Madrid",
        "dortmund": "Borussia Dortmund",
        "arsenal": "Arsenal F.C.",
        "liverpool": "Liverpool F.C.",
        "chelsea": "Chelsea F.C.",
        "juventus": "Juventus F.C.",
        "benfica": "S.L. Benfica",
        "porto": "FC Porto",
        "flamengo": "Clube de Regatas do Flamengo",
        "palmeiras": "Sociedade Esportiva Palmeiras",
        "fluminense": "Fluminense FC",
        "vasco": "CR Vasco da Gama",
        "paranaense": "Club Athletico Paranaense",
        "betis": "Real Betis",
        "real sociedad": "Real Sociedad",
        "monaco": "AS Monaco FC",
        "lyon": "Olympique Lyonnais",
        "freiburg": "SC Freiburg",
        "newcastle": "Newcastle United F.C.",
        "aston villa": "Aston Villa F.C.",
        "como": "Como 1907",
        "bahia": "Esporte Clube Bahia",
        "racing club": "Racing Club de Avellaneda",
        "racing": "Racing Club de Avellaneda",
    }

    key = normalizar_texto(nombre_limpio)
    busqueda = nombre_limpio
    for a, v in aliases.items():
        if a == key or a in key:
            busqueda = v
            break

    headers = {"User-Agent": "GoalMetricsApp/1.0 (contact@goalmetrics.com)"}
    queries = [
        f"{busqueda} football club",
        f"{busqueda} football",
        busqueda,
    ]

    for q in queries:
        try:
            url_search = (
                "https://en.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={requests.utils.quote(q)}&format=json"
            )
            res = requests.get(url_search, headers=headers, timeout=3).json()
            results = res.get("query", {}).get("search", [])
            if not results:
                continue
            page_title = results[0]["title"]
            url_image = (
                "https://en.wikipedia.org/w/api.php"
                f"?action=query&titles={requests.utils.quote(page_title)}"
                "&prop=pageimages&pithumbsize=200&format=json"
            )
            res_img = requests.get(url_image, headers=headers, timeout=3).json()
            pages = res_img.get("query", {}).get("pages", {})
            for _, page_info in pages.items():
                thumb = page_info.get("thumbnail", {}).get("source")
                if thumb:
                    return thumb
        except Exception:
            continue
    return None

def render_header_equipo(liga, equipo, condicion, nivel):
    liga_h = html.escape(str(liga).upper())
    equipo_h = html.escape(str(equipo).upper())
    cond_h = html.escape(str(condicion).upper())
    nivel_h = html.escape(str(nivel).upper())
    iniciales = html.escape(obtener_iniciales(equipo))
    logo_url = obtener_logo_equipo(equipo)

    if logo_url:
        badge = (
            f'<img src="{html.escape(logo_url)}" '
            f'style="height:48px;width:48px;object-fit:contain;border-radius:10px;'
            f'background:rgba(255,255,255,0.12);padding:4px;flex-shrink:0;" />'
        )
    else:
        badge = (
            f'<div style="height:48px;width:48px;border-radius:10px;flex-shrink:0;'
            f'background:rgba(255,255,255,0.15);display:flex;align-items:center;'
            f'justify-content:center;font-weight:800;font-size:15px;color:#fff;'
            f'border:1px solid rgba(255,255,255,0.25);">{iniciales}</div>'
        )

    st.markdown(
        f'<div class="header-box">'
        f"{badge}"
        f"<span>{liga_h} | {equipo_h} - {cond_h} vs {nivel_h}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

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
header[data-testid="stHeader"] {{
    visibility: visible !important;
    background: transparent !important;
}}

.header-box {{
    background: linear-gradient(135deg, {color_equipo} 0%, #111827 100%);
    padding: 24px 30px; 
    border-radius: 16px; 
    color: white;
    font-weight: 700; 
    font-size: 26px; 
    margin-bottom: 20px; 
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
}}
.pill-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.pill-green {{ background-color: rgba(6, 78, 59, 0.7); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }}
.pill-yellow {{ background-color: rgba(120, 53, 15, 0.7); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }}
.pill-red {{ background-color: rgba(127, 29, 29, 0.7); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }}

.veredicto-box {{
    padding: 18px 22px; border-radius: 14px; background-color: #111827;
    border: 1px solid #1f2937; border-left: 5px solid {color_equipo}; margin-bottom: 20px; font-size: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}}
.analisis-dinamico-box {{
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid {color_equipo}66;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    font-size: 15px;
    line-height: 1.6;
    color: #e5e7eb;
}}
.value-box {{ padding: 14px 16px; border-radius: 12px; margin-bottom: 10px; font-size: 14px; border: 1px solid #1f2937; }}
.value-yes {{ background-color: rgba(6, 78, 59, 0.4); border-left: 4px solid #10b981; }}
.value-no {{ background-color: #111827; border-left: 4px solid #4b5563; }}
.top-pick-box {{ background: linear-gradient(135deg, rgba(6, 95, 70, 0.8) 0%, #111827 100%); padding: 22px; border-radius: 14px; border: 2px solid #10b981; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15); }}

.saas-card {{
    background-color: #111827;
    border: 1px solid {color_equipo}44;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease-in-out;
}}
.saas-card:hover {{
    border-color: {color_equipo};
    box-shadow: 0 6px 25px {color_equipo}22;
}}

[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f2937;
}}
[data-testid="stDataFrame"] th {{
    background-color: #1f2937 !important;
    color: #f3f4f6 !important;
    font-weight: 600 !important;
}}
[data-testid="stDataFrame"] td {{
    background-color: #111827 !important;
    color: #9ca3af !important;
}}
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

def calcular_kelly_seguro(prob, cuota, n_obs):
    if cuota <= 1.0 or prob <= 0 or n_obs < 3:
        return 0.0
    p, b = prob / 100.0, cuota - 1.0
    if b <= 0:
        return 0.0
    kelly_fraction = ((p * cuota - 1.0) / b) * 0.5
    if kelly_fraction <= 0:
        return 0.0
    cap_max = 0.01 if n_obs < 6 else 0.02
    stake_final = min(kelly_fraction, cap_max)
    return round(stake_final * 100, 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, n_obs, muestra_pequena=False):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake = calcular_kelly_seguro(prob, cuota_casa, n_obs) if es_value else 0.0
    kelly_txt = f" | Half-Kelly: <b>{stake}% bank</b>" if es_value else ""
    caution = " (muestra pequeña)" if muestra_pequena and es_value else ""
    st.markdown(
        f'<div class="value-box {clase}"><b>{html.escape(nombre)}</b>{caution}<br>'
        f"Prob: <b>{prob:.1f}%</b> | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_txt}<br>"
        f'<span style="color:{color_ev}; font-weight:bold; font-size:15px;">'
        f"EV: {ev:+.2%} -> {'VALUE' if es_value else 'Sin valor'}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("### GoalMetrics - Análisis de Equipos (Híbrido Pro)")
st.caption("Simulación con Poisson, Dixon-Coles, Ensemble XGBoost, Bootstrap, Half-Life Decay, Backtesting y gestión de bankroll.")

with st.expander("📖 Guía Detallada: ¿Cómo funciona el Análisis de Equipos?", expanded=False):
    st.markdown("""
    Bienvenido al **Centro de Análisis de Equipos de GoalMetrics**. Esta herramienta combina estadística avanzada y Machine Learning. Aquí te detallamos cómo opera cada módulo interno:
    
    * **1. Semáforo de Confiabilidad:** Evalúa al instante la robustez de la muestra de partidos exactos. 
      * 🟢 *Verde:* Suficientes partidos exactos en el escenario buscado (>= 2).
      * 🟡 *Amarillo:* Muestra mixta o con 1 solo partido exacto, activando el respaldo inteligente ajustado por *Tier*.
      * 🔴 *Rojo:* Muestra crítica o escasa, requiere máxima precaución.
    * **2. Shrinkage (Compensación Estadística):** Cuando un equipo cuenta con pocos partidos en un escenario específico, los promedios empíricos pueden estar sesgados. El **Shrinkage** corrige esto ponderando la tasa observada hacia una media previa (*prior*) de la liga para ese mismo nivel de rival.
    * **3. Modelo Dixon-Coles (Corrección de Empates y Bajas):** Introduce un factor de corrección (tau) controlado por el parámetro de correlación $\rho$ para ajustar la probabilidad en marcadores cerrados y de baja anotación.
    * **4. Ensemble Híbrido (Poisson/Dixon-Coles + XGBoost):** Integra la solidez estocástica de las distribuciones de goles con modelos de Machine Learning (XGBoost).
    * **5. Value Bets & Criterio de Half-Kelly con Cap:** Evalúa el Valor Esperado (EV) contrastando las probabilidades frente a las cuotas de las casas de apuestas, aplicando un límite estricto de stake en muestras pequeñas para blindar el capital real.
    * **6. Bootstrap e Intervalos de Confianza:** Remuestreo no paramétrico que repite el cálculo de la tasa de goles ($\lambda$) cientos de veces para entregarte un intervalo de confianza real (IC 95%) y medir la incertidumbre.
    * **7. Half-Life Decay (Decaimiento Exponencial Temporal):** Asigna mayor peso a los partidos recientes mediante una vida media de 30 días, haciendo que los encuentros más antiguos pierdan peso analítico de forma no lineal para reflejar mejor el momento actual del equipo.
    * **8. Validación Retrospectiva (Log Loss & Brier Score):** Auditoría interna en ventana rodante que mide el error logarítmico y la calibración real de las probabilidades del modelo frente a los resultados históricos.
    """)

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

    if len(df_exactos) >= 2:
        st.markdown(
            '<div class="pill-badge pill-green">'
            '🟢 <b>Semáforo de Confiabilidad: ALTA</b> — Muestra robusta con suficientes partidos exactos.'
            '</div>', unsafe_allow_html=True
        )
    elif len(df_exactos) == 1:
        st.markdown(
            '<div class="pill-badge pill-yellow">'
            '🟡 <b>Semáforo de Confiabilidad: MEDIA</b> — 1 partido exacto encontrado. Respaldo inteligente activo.'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="pill-badge pill-red">'
            '🔴 <b>Semáforo de Confiabilidad: BAJA</b> — Muestra escasa, interpretar con máxima precaución.'
            '</div>', unsafe_allow_html=True
        )

    # Decaimiento Exponencial Temporal Avanzado (Half-Life Decay de 30 días)
    hoy = pd.Timestamp.today().normalize()
    half_life_days = 30.0
    if "Fecha" in historial.columns:
        historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.clip(lower=0)
        historial["Peso_Temporal"] = np.power(0.5, historial["Dias_Pasados"] / half_life_days)
    else:
        historial["Peso_Temporal"] = 1.0

    if "Peso_Contexto" not in historial.columns:
        historial["Peso_Contexto"] = 1.0

    historial["Peso_Total"] = historial["Peso_Temporal"] * historial["Peso_Contexto"]
    suma_pesos = historial["Peso_Total"].sum()
    pesos = historial["Peso_Total"] / suma_pesos if suma_pesos > 0 else np.ones(len(historial)) / len(historial)

    def prom(col):
        if col not in historial.columns or len(historial) == 0:
            return 0.05
        return round(float(np.average(historial[col].fillna(0), weights=pesos)), 4)

    def std_w(col):
        return float(historial[col].std()) if col in historial.columns and len(historial) > 1 else 0.0

    lam_f_raw, lam_c_raw = prom("Goles"), prom("Goles Rival")
    lam_t_raw, lam_tp_raw = prom("Tiros"), prom("A Puerta")
    lam_co_raw, lam_fa_raw = prom("Corners"), prom("Faltas")
    lam_co_rival_raw = prom("Corners Rival") if "Corners Rival" in historial.columns else prom("Corners")

    goles_vals = historial["Goles"].fillna(0).values if "Goles" in historial.columns else np.array([0])
    _, lam_f_inf, lam_f_sup = bootstrap_lambda_intervalo(goles_vals, pesos.values if len(pesos)==len(goles_vals) else None)

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

    triunfos_base = (sg_fav > sg_con).mean() * 100
    empates = (sg_fav == sg_con).mean() * 100
    derrotas_base = (sg_fav < sg_con).mean() * 100

    features_modelo = ["Goles_Media_Movil_5", "Goles_Volatilidad_5", "Tiros_Media_Movil_5", "Conversion_Tiros", "Momentum_Goles", "Diff_Goles"]
    modelo_xgb_global = _entrenar_xgboost_real(df, features_modelo)
    
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

    render_header_equipo(liga_sel, equipo_sel, condicion_label, nivel_sel)

    st.markdown(f'<div class="veredicto-box"><b>Veredicto:</b> {html.escape(veredicto)}</div>', unsafe_allow_html=True)
    
    analisis_texto = generar_analisis_dinamico(
        equipo_sel, condicion_label, nivel_sel, n_obs, 
        lam_f, lam_c, lam_t, lam_tp, lam_co, 
        triunfos, ambos_anotan, 
        prob_over_goles, prob_over_corners, prob_over_puerta
    )
    st.markdown(f'<div class="analisis-dinamico-box">{analisis_texto}</div>', unsafe_allow_html=True)

    st.caption(f"Base: {n_obs} partidos - {fuente_datos} | Half-Life Decay (30d) | IC 95% Bootstrap λ: [{lam_f_inf:.2f} - {lam_f_sup:.2f}]")

    if muestra_pequena:
        st.warning("Muestra pequeña (Respaldo activo con 1 partido exacto). Stake limitado por seguridad.")

    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard Principal & Gráficos",
        "💰 Value Bets & Inteligencia",
        "📋 Análisis Táctico & Auditoría"
    ])

    with tab1:
        st.subheader("ADN del Equipo y Métricas Probabilísticas")
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
        
        metrics_data_eq = {
            "Goles": {"val": lam_f, "vol": std_w("Goles"), "format": ".2f"},
            "Goles Rival": {"val": lam_c, "vol": std_w("Goles Rival"), "format": ".2f"},
            "Tiros": {"val": lam_t, "vol": std_w("Tiros"), "format": ".1f"},
            "A Puerta": {"val": lam_tp, "vol": std_w("A Puerta"), "format": ".1f"},
            "Corners": {"val": lam_co, "vol": std_w("Corners"), "format": ".1f"},
        }

        cols_m = st.columns(3)
        for i, (var, data) in enumerate(metrics_data_eq.items()):
            col_target = cols_m[i % 3]
            fmt = data["format"]
            col_target.metric(var, f"{data['val']:{fmt}}", f"λ: {data['val']:{fmt}} | σ: {data['vol']:.2f}")

        st.markdown("---")
        st.subheader("🎯 Matriz de Probabilidad del Resultado Exacto")
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
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#F3F4F6"), height=380, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 Curvas de Probabilidad Acumulada (Over X)")
        
        max_g_sim = int(max(sg_fav)) + 2
        goal_vals = [float(i + 0.5) for i in range(max_g_sim)]
        cum_probs_goles = [float((sg_fav > g).mean() * 100) for g in goal_vals]

        fig_cum_goles = go.Figure(data=go.Scatter(
            x=goal_vals, y=cum_probs_goles, mode='lines+markers+text',
            text=[f"{p:.1f}%" if p > 1.0 else "" for p in cum_probs_goles], textposition="top center",
            line=dict(color="#10b981", width=3), marker=dict(size=8)
        ))
        fig_cum_goles.update_layout(
            title=dict(text=f"<b>Acumulada de Goles (Over X)</b>", font=dict(size=14, color="#F3F4F6")),
            xaxis=dict(title="Línea de Goles", tickmode='array', tickvals=goal_vals, ticktext=[str(g) for g in goal_vals], color="#9ca3af", gridcolor="#1f2937"),
            yaxis=dict(title="Probabilidad (%)", color="#9ca3af", gridcolor="#1f2937", range=[0, 115]),
            paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F3F4F6"),
            height=350, margin=dict(l=30, r=20, t=40, b=30)
        )
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_cum_goles, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        max_c = int(max(s_corn)) + 2
        step_tick = 2 if max_c > 12 else 1
        corner_vals = list(range(0, max_c, step_tick))
        cum_probs_corners = [float((s_corn > c).mean() * 100) for c in corner_vals]

        fig_cum_corners = go.Figure(data=go.Scatter(
            x=corner_vals, y=cum_probs_corners, mode='lines+markers+text',
            text=[f"{p:.1f}%" if p > 5.0 else "" for p in cum_probs_corners], textposition="top center",
            line=dict(color=color_equipo, width=3), marker=dict(size=8)
        ))
        fig_cum_corners.update_layout(
            title=dict(text=f"<b>Acumulada de Córners (Over X)</b>", font=dict(size=14, color="#F3F4F6")),
            xaxis=dict(title="Línea de Córners", tickmode='array', tickvals=corner_vals, ticktext=[str(c) for c in corner_vals], color="#9ca3af", gridcolor="#1f2937"),
            yaxis=dict(title="Probabilidad (%)", color="#9ca3af", gridcolor="#1f2937", range=[0, 115]),
            paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F3F4F6"),
            height=350, margin=dict(l=30, r=20, t=40, b=30)
        )
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_cum_corners, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if not historial.empty and "A Puerta" in historial.columns and "Goles" in historial.columns:
            st.markdown("---")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=historial["A Puerta"], y=historial["Goles"], mode="markers+text",
                text=historial.get("Rival", ""), textposition="top center",
                marker=dict(size=12, color=color_equipo, line=dict(width=2, color="white"), opacity=0.85),
                hovertemplate="<b>Rival:</b> %{text}<br><b>Tiros a Puerta:</b> %{x}<br><b>Goles Anotados:</b> %{y}<extra></extra>"
            ))
            fig_scatter.update_layout(
                title="Tiros a Puerta vs Goles por Partido", xaxis_title="Tiros a Puerta", yaxis_title="Goles Anotados",
                paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F3F4F6"), height=350, margin=dict(l=40, r=40, t=40, b=40)
            )
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("💰 Value Bet y Half-Kelly Seguro")
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
                mostrar_value(it[0], it[1], it[2], it[3], it[4], n_obs=n_obs, muestra_pequena=muestra_pequena)
        with cb:
            for it in items_1x2[mid:]:
                mostrar_value(it[0], it[1], it[2], it[3], it[4], n_obs=n_obs, muestra_pequena=muestra_pequena)
                
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
                mostrar_value(it[0], it[1], it[2], it[3], it[4], n_obs=n_obs, muestra_pequena=muestra_pequena)
        with cd:
            for it in items_lineas[mid2:]:
                mostrar_value(it[0], it[1], it[2], it[3], it[4], n_obs=n_obs, muestra_pequena=muestra_pequena)

        st.markdown("---")
        st.subheader("🤖 Panel Inteligente & Parlay Híbrido")
        todos_mercados_eq = items_1x2 + items_lineas
        lista_mercados_eq_dict = [{"nombre": m[0], "prob": m[4], "cuota": m[2], "ev": m[3]} for m in todos_mercados_eq]
        value_bets_eq = [m for m in lista_mercados_eq_dict if m["ev"] > 0]

        if value_bets_eq:
            top_eq = max(value_bets_eq, key=lambda x: x["ev"])
            stake_top_eq = calcular_kelly_seguro(top_eq["prob"], top_eq["cuota"], n_obs)
            st.markdown(
                f'<div class="top-pick-box">'
                f'<h3>🏆 La Joya del Partido (Top Value Bet)</h3>'
                f'<p style="font-size: 16px; margin-bottom: 8px;">Mercado recomendado: <b>{html.escape(top_eq["nombre"])}</b></p>'
                f'<ul>'
                f'<li>Probabilidad Híbrida del Modelo: <b>{top_eq["prob"]:.1f}%</b></li>'
                f'<li>Cuota Casa: <b>{top_eq["cuota"]}</b></li>'
                f'<li><b>EV Matemático: {top_eq["ev"]:+.2%}</b></li>'
                f'</ul>'
                f'<p style="color: #10b981; font-weight: bold; margin-top: 10px;">👉 Sugerencia de Stake: <b>{stake_top_eq}% del Bank (Half-Kelly con Cap)</b></p>'
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
            stake_parlay_eq = calcular_kelly_seguro(p_conj_pct, cuota_casa_parlay_eq, n_obs)
            
            col_ep1, col_ep2 = st.columns(2)
            col_ep1.metric("EV de la Combinada", f"{ev_parlay_eq:+.2%}")
            if ev_parlay_eq > 0:
                col_ep2.metric("Stake Sugerido", f"{stake_parlay_eq}% del Bank")
                st.success(f"🎉 ¡Combinada con EV positivo! Stake recomendado: {stake_parlay_eq}%.")
            else:
                col_ep2.metric("Stake Sugerido", "0%")
                st.warning("⚠️ EV Negativo.")

    with tab3:
        st.subheader("📈 Validación Retrospectiva (Backtesting & Métricas de Error)")
        log_loss_val, brier_val = calcular_backtesting_retrospectivo(historial)
        
        bc1, bc2 = st.columns(2)
        if log_loss_val is not None:
            bc1.metric("Log Loss (Pérdida Logarítmica)", f"{log_loss_val:.4f}", "Menor es mejor calibración")
            bc2.metric("Brier Score", f"{brier_val:.4f}", "Precisión global 0 a 1 (0 es perfecto)")
            st.caption("ℹ️ Estas métricas evalúan retrospectivamente el error de las probabilidades del modelo frente a los resultados reales en este escenario filtrado.")
        else:
            st.info("ℹ️ Se requieren al menos 5 partidos en este filtro exacto para calcular las métricas de backtesting retrospectivo.")

        st.markdown("---")
        st.markdown("#### 📘 Guía de Interpretación: Log Loss & Brier Score")
        
        df_guia_limpia = pd.DataFrame([
            {"Métrica": "Log Loss", "Excelente": "< 0.50", "Aceptable": "0.50 - 0.69", "Deficiente": "> 0.69"},
            {"Métrica": "Brier Score", "Excelente": "< 0.15", "Aceptable": "0.16 - 0.25", "Deficiente": "> 0.25"}
        ])
        st.dataframe(df_guia_limpia, hide_index=True, use_container_width=True)

        st.markdown("""
        **💡 Interpretación Práctica (¿Qué hacer en cada caso?):**
        * **🟢 Rango Excelente:** El modelo está muy bien calibrado. Puedes operar con total confianza siguiendo el stake sugerido por Half-Kelly.
        * **🟡 Rango Aceptable:** Margen moderado de error. El modelo es funcional, pero debido a la varianza histórica es recomendable **reducir ligeramente el stake**.
        * **🔴 Rango Deficiente:** El modelo presenta alta desviación o descalibración para este filtro. La recomendación operativa es **evitar apostar o reducir drásticamente la exposición al riesgo**.
        """)

        st.markdown("---")
        st.subheader("📋 Auditoría de Partidos Filtrados")
        h_mostrar = historial.copy().sort_values(by="Fecha", ascending=False)
        h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
        cols = [c for c in ["Fecha", "Liga", "Condición", "Rival", "Nivel Rival", "Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Tipo_Uso", "Factor_Ajuste"] if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)
else:
    st.info("Configura el partido en la barra lateral y pulsa Analizar.")
