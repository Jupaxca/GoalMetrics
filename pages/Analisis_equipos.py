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

def predecir_probabilidad_hibrida_dinamica(prob_poisson, equipo_actual_df, features_modelo, modelo_xgb, n_obs):
    """Ponderación dinámica del XGBoost basada en el tamaño de la muestra (n)."""
    if modelo_xgb is None or equipo_actual_df.empty or n_obs < 5:
        return prob_poisson, "100% Poisson / Dixon-Coles (Muestra pequeña)"

    ultima_fila = equipo_actual_df.tail(1)
    try:
        X_pred = ultima_fila[features_modelo]
        prob_xgb = float(modelo_xgb.predict_proba(X_pred)[0][1]) * 100.0
    except Exception:
        return prob_poisson, "100% Poisson / Dixon-Coles (Fallback)"

    if n_obs >= 10:
        peso_xgb = 0.30
        modo_str = "Híbrido 70/30 (Poisson / XGBoost)"
    else:
        peso_xgb = 0.15
        modo_str = "Híbrido 85/15 (Muestra moderada)"

    peso_poisson = 1.0 - peso_xgb
    prob_hibrida = (peso_poisson * prob_poisson) + (peso_xgb * prob_xgb)
    return round(prob_hibrida, 2), modo_str

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

def calcular_backtesting_motor_real(historial_filtrado, usar_dc, rho_dc):
    """Backtesting estricto simulando el motor real (Poisson/Dixon-Coles) en ventana rodante."""
    if len(historial_filtrado) < 5:
        return None, None
        
    y_true, y_prob = [], []
    sub_df = historial_filtrado.tail(20)
    
    for i in range(2, len(sub_df)):
        train_window = sub_df.iloc[:i]
        test_row = sub_df.iloc[i:i+1]
        
        g_fav = test_row["Goles"].values[0]
        g_con = test_row["Goles Rival"].values[0]
        actual_win = 1 if g_fav > g_con else 0
        
        lam_f_past = max(float(train_window["Goles"].mean()), 0.05)
        lam_c_past = max(float(train_window["Goles Rival"].mean()), 0.05)
        
        if usar_dc:
            sg_f, sg_c = simular_goles_dixon_coles(lam_f_past, lam_c_past, rho=rho_dc, num_sim=2000)
        else:
            rng = np.random.default_rng(42)
            sg_f = rng.poisson(lam_f_past, 2000)
            sg_c = rng.poisson(lam_c_past, 2000)
            
        prob_win_est = float((sg_f > sg_c).mean())
        y_true.append(actual_win)
        y_prob.append(np.clip(prob_win_est, 0.01, 0.99))
        
    if not y_true:
        return None, None
        
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    brier_score = np.mean((y_prob - y_true) ** 2)
    return round(float(log_loss), 4), round(float(brier_score), 4)

def generar_analisis_dinamico(equipo, condicion, nivel, n_obs, lam_f, lam_c, lam_t, lam_tp, lam_co, triunfos, ambos_anotan, prob_over_goles, prob_over_corners, prob_over_puerta):
    if triunfos >= 55:
        perfil = "altamente proactivo, dominante y con clara tendencia a volcar el juego en campo rival"
    elif triunfos >= 40:
        perfil = "competitivo, equilibrado y con alta capacidad de alternar el ritmo del juego"
    else:
        perfil = "de alta exigencia táctica, con tramos de repliegue y partidos cerrados"

    mercados_destacados = []
    if prob_over_goles >= 55.0:
        mercados_destacados.append(f"<b>Over de Goles del equipo</b> (probabilidad del <b>{prob_over_goles:.1f}%</b>)")
    if prob_over_puerta >= 55.0:
        mercados_destacados.append(f"<b>Líneas de Tiros a Puerta</b> (media de <b>{lam_tp:.1f}</b> remates)")
    if prob_over_corners >= 55.0:
        mercados_destacados.append(f"<b>Córners / Saques de Esquina</b> (proyectando <b>{lam_co:.1f}</b>)")
    if ambos_anotan >= 55.0:
        mercados_destacados.append(f"<b>BTTS Sí</b> (soporte del <b>{ambos_anotan:.1f}%</b>)")
    
    recomendacion_mercado = "escenarios conservadores u opciones de Under si la cuota acompaña." if not mercados_destacados else "apostar con mayor respaldo analítico por: " + ", ".join(mercados_destacados) + "."

    return (
        f"<b>Reporte Analítico:</b> Para <b>{equipo}</b> como <b>{condicion}</b> "
        f"vs <b>{nivel}</b> ({n_obs} partidos analizados), el comportamiento proyectado es <b>{perfil}</b>.<br><br>"
        f"• <b>Ofensiva:</b> Expectativa de <b>{lam_f:.2f} goles</b>, con <b>{lam_t:.1f} tiros totales</b> y <b>{lam_tp:.1f} a puerta</b>.<br>"
        f"• <b>Defensa:</b> Concede <b>{lam_c:.2f} goles</b> por encuentro (BTTS: <b>{ambos_anotan:.1f}%</b>).<br>"
        f"• <b>Conclusión de Mercado:</b> {recomendacion_mercado}"
    )

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
def obtener_logo_equipo(nombre, liga):
    nombre_limpio = str(nombre).strip()
    if not nombre_limpio:
        return None

    # Corrección de Aliases específicos (ej. diferenciar Racing de Avellaneda vs España)
    key = normalizar_texto(nombre_limpio)
    busqueda = nombre_limpio
    
    if "racing" in key:
        liga_n = normalizar_texto(liga)
        if "espana" in liga_n or "laliga" in liga_n or "segunda" in liga_n:
            busqueda = "Real Racing Club de Santander"
        else:
            busqueda = "Racing Club"

    aliases = {
        "bayern munchen": "FC Bayern Munich",
        "real madrid": "Real Madrid CF",
        "barcelona": "FC Barcelona",
        "manchester city": "Manchester City F.C.",
        "psg": "Paris Saint-Germain F.C.",
        "inter": "Inter Milan",
        "palmeiras": "Sociedade Esportiva Palmeiras",
        "porto": "FC Porto"
    }

    for a, v in aliases.items():
        if a == key or a in key:
            busqueda = v
            break

    headers = {"User-Agent": "GoalMetricsApp/1.0"}
    try:
        url_search = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(busqueda)}&format=json"
        res = requests.get(url_search, headers=headers, timeout=2).json()
        results = res.get("query", {}).get("search", [])
        if results:
            page_title = results[0]["title"]
            url_img = f"https://en.wikipedia.org/w/api.php?action=query&titles={requests.utils.quote(page_title)}&prop=pageimages&pithumbsize=150&format=json"
            res_img = requests.get(url_img, headers=headers, timeout=2).json()
            pages = res_img.get("query", {}).get("pages", {})
            for _, p in pages.items():
                thumb = p.get("thumbnail", {}).get("source")
                if thumb:
                    return thumb
    except Exception:
        pass
    return None

def render_header_equipo(liga, equipo, condicion, nivel):
    liga_h = html.escape(str(liga).upper())
    equipo_h = html.escape(str(equipo).upper())
    cond_h = html.escape(str(condicion).upper())
    nivel_h = html.escape(str(nivel).upper())
    iniciales = html.escape(obtener_iniciales(equipo))
    logo_url = obtener_logo_equipo(equipo, liga)

    badge = f'<img src="{html.escape(logo_url)}" style="height:48px;width:48px;object-fit:contain;border-radius:10px;background:rgba(255,255,255,0.12);padding:4px;flex-shrink:0;" />' if logo_url else f'<div style="height:48px;width:48px;border-radius:10px;background:rgba(255,255,255,0.15);display:flex;align-items:center;justify-content:center;font-weight:800;color:#fff;">{iniciales}</div>'

    st.markdown(
        f'<div class="header-box">{badge}<span>{liga_h} | {equipo_h} - {cond_h} vs {nivel_h}</span></div>',
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
exactos_check = df_diagnostico[(df_diagnostico["Condición"] == condicion_sel) & (df_diagnostico["Nivel Rival"] == nivel_sel)]
num_exactos = len(exactos_check)

if num_exactos >= 2:
    st.sidebar.success(f"{num_exactos} partidos exactos (Suficientes)")
elif num_exactos == 1:
    st.sidebar.warning("1 partido exacto -> Respaldo inteligente activo")
else:
    st.sidebar.error("0 partidos exactos en este filtro")

with st.sidebar.expander("Lineas de Estudio"):
    linea_goles = st.slider("Goles (equipo)", 0.5, 3.5, 1.5, 0.5)
    linea_tiros = st.slider("Tiros Totales", 5.0, 25.0, 12.5, 0.5)
    linea_tiros_puerta = st.slider("Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
    linea_corners = st.slider("Corners", 1.0, 15.0, 5.5, 0.5)
    linea_faltas = st.slider("Faltas", 5.0, 25.0, 10.5, 0.5)
    linea_total_partido = st.slider("Total goles partido (Over/Under)", 0.5, 4.5, 2.5, 0.5)

with st.sidebar.expander("Cuotas 1X2 / BTTS / DNB"):
    cuota_casa_1 = st.number_input("Victoria (1)", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_casa_x = st.number_input("Empate (X)", min_value=1.01, value=3.40, step=0.01, format="%.2f")
    cuota_casa_2 = st.number_input("Derrota (2)", min_value=1.01, value=4.20, step=0.01, format="%.2f")
    cuota_casa_1x = st.number_input("Doble Oportunidad (1X)", min_value=1.01, value=1.22, step=0.01, format="%.2f")
    cuota_casa_x2 = st.number_input("Doble Oportunidad (X2)", min_value=1.01, value=1.95, step=0.01, format="%.2f")
    cuota_casa_btts_si = st.number_input("BTTS Si", min_value=1.01, value=1.75, step=0.01, format="%.2f")
    cuota_casa_btts_no = st.number_input("BTTS No", min_value=1.01, value=2.05, step=0.01, format="%.2f")
    cuota_casa_dnb = st.number_input("DNB", min_value=1.01, value=1.35, step=0.01, format="%.2f")

with st.sidebar.expander("Cuotas de Lineas (Over / Under)"):
    cuota_over_goles = st.number_input(f"Over {linea_goles} Goles", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_under_goles = st.number_input(f"Under {linea_goles} Goles", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_corners = st.number_input(f"Over {linea_corners} Corners", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_under_corners = st.number_input(f"Under {linea_corners} Corners", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_total = st.number_input(f"Over {linea_total_partido} Partido", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_under_total = st.number_input(f"Under {linea_total_partido} Partido", min_value=1.01, value=1.90, step=0.01, format="%.2f")

with st.sidebar.expander("Modelo estadistico", expanded=True):
    shrink_opt = st.radio("Shrinkage", options=["ON", "OFF"], index=0, horizontal=True, label_visibility="collapsed")
    usar_shrinkage = (shrink_opt == "ON")
    k_shrink = st.slider("Fuerza prior (k)", 1.0, 15.0, 5.0, 1.0, disabled=not usar_shrinkage)
    
    dc_opt = st.radio("Dixon-Coles", options=["ON", "OFF"], index=0, horizontal=True, label_visibility="collapsed")
    usar_dc = (dc_opt == "ON")
    rho_dc = st.slider("rho Dixon-Coles", -0.20, 0.05, -0.10, 0.01, disabled=not usar_dc)

color_equipo = generar_color_equipo(equipo_sel)

st.markdown(f"""
<style>
header[data-testid="stHeader"] {{ visibility: visible !important; background: transparent !important; }}
.header-box {{ background: linear-gradient(135deg, {color_equipo} 0%, #111827 100%); padding: 24px 30px; border-radius: 16px; color: white; font-weight: 700; font-size: 26px; margin-bottom: 20px; text-align: center; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); display: flex; align-items: center; justify-content: center; gap: 16px; }}
.pill-badge {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 9999px; font-weight: 600; font-size: 0.9rem; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }}
.pill-green {{ background-color: rgba(6, 78, 59, 0.7); color: #34d399; }}
.pill-yellow {{ background-color: rgba(120, 53, 15, 0.7); color: #fbbf24; }}
.pill-red {{ background-color: rgba(127, 29, 29, 0.7); color: #f87171; }}
.veredicto-box {{ padding: 18px 22px; border-radius: 14px; background-color: #111827; border: 1px solid #1f2937; border-left: 5px solid {color_equipo}; margin-bottom: 20px; font-size: 16px; }}
.analisis-dinamico-box {{ background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 20px; border-radius: 14px; border: 1px solid {color_equipo}66; margin-bottom: 20px; font-size: 15px; line-height: 1.6; color: #e5e7eb; }}
.value-box {{ padding: 14px 16px; border-radius: 12px; margin-bottom: 10px; font-size: 14px; border: 1px solid #1f2937; }}
.value-yes {{ background-color: rgba(6, 78, 59, 0.4); border-left: 4px solid #10b981; }}
.value-no {{ background-color: #111827; border-left: 4px solid #4b5563; }}
.top-pick-box {{ background: linear-gradient(135deg, rgba(6, 95, 70, 0.8) 0%, #111827 100%); padding: 22px; border-radius: 14px; border: 2px solid #10b981; margin-bottom: 20px; }}
.saas-card {{ background-color: #111827; border: 1px solid {color_equipo}44; border-radius: 14px; padding: 20px; margin-bottom: 20px; }}
</style>
""", unsafe_allow_html=True)

def renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa):
    df_adn = pd.DataFrame({
        "Metrica": ["Ataque", "Volumen Tiros", "Precision", "Corners", "Disciplina"],
        "Puntuacion": [
            min(round(lam_f * 3.33, 1), 10.0), min(round(lam_t / 2.5, 1), 10.0),
            min(round(lam_tp * 1.66, 1), 10.0), min(round(lam_co / 1.5, 1), 10.0),
            min(round((25 - lam_fa) / 2.5, 1), 10.0),
        ],
    })
    chart = alt.Chart(df_adn).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X("Puntuacion:Q", scale=alt.Scale(domain=[0, 10]), title=None),
        y=alt.Y("Metrica:N", sort="-x", title=None), color=alt.value(color_equipo), tooltip=["Metrica", "Puntuacion"],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

def calcular_ev(prob, cuota):
    if cuota <= 1.0 or prob <= 0:
        return 0.0
    return round((prob / 100 * cuota) - 1, 4)

def calcular_kelly_seguro(prob, cuota, n_obs, nivel_semaforo="verde"):
    if cuota <= 1.0 or prob <= 0 or n_obs < 3:
        return 0.0
    p, b = prob / 100.0, cuota - 1.0
    if b <= 0:
        return 0.0
    kelly_fraction = ((p * cuota - 1.0) / b) * 0.5
    if kelly_fraction <= 0:
        return 0.0
    
    # Atenuación de riesgo por semáforo / tamaño de muestra
    if nivel_semaforo == "rojo":
        cap_max = 0.005 # 0.5% max si la muestra es muy forzada
    elif nivel_semaforo == "amarillo":
        cap_max = 0.01  # 1% max si hay respaldo mixto
    else:
        cap_max = 0.02 if n_obs >= 6 else 0.015

    return round(min(kelly_fraction, cap_max) * 100, 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, n_obs, semaforo):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake = calcular_kelly_seguro(prob, cuota_casa, n_obs, semaforo) if es_value else 0.0
    kelly_txt = f" | Stake: <b>{stake}% bank</b>" if es_value else ""
    st.markdown(
        f'<div class="value-box {clase}"><b>{html.escape(nombre)}</b><br>'
        f"Prob: <b>{prob:.1f}%</b> | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_txt}<br>"
        f'<span style="color:{color_ev}; font-weight:bold;">EV: {ev:+.2%} -> {"VALUE" if es_value else "Sin valor"}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("### GoalMetrics - Análisis de Equipos (Híbrido Pro c/ Mejoras)")
st.caption("Simulación estocástica avanzada, intervalos Bootstrap, ponderación por muestra y gestión de riesgo integrada.")

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

    df_exactos = df_equipo[(df_equipo["Condición"] == condicion_sel) & (df_equipo["Nivel Rival"] == nivel_sel)].copy() if "Condición" in df_equipo.columns else pd.DataFrame()

    if len(df_exactos) == 0:
        st.error(f"❌ No hay partidos exactos para **{equipo_sel}** como **{condicion_label}** vs **{nivel_sel}**.")
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
        df_misma_cond = df_equipo[(df_equipo["Condición"] == condicion_sel) & (df_equipo["Nivel Rival"] != nivel_sel)].copy()
        for _, row in df_misma_cond.tail(UMBRAL_MINIMO - len(historial_list)).iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.85
            historial_list.append(r)
        fuente_datos = "Muestra mixta (Exacto + Respaldo Tier)"

    if len(historial_list) < UMBRAL_MINIMO:
        opuesto_lower = "local" if condicion_sel == "visitante" else "visitante"
        df_contrarios = df_equipo[df_equipo["Condición"] == opuesto_lower].copy()
        for _, row in df_contrarios.tail(UMBRAL_MINIMO - len(historial_list)).iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.75
            historial_list.append(r)
        fuente_datos = "Muestra adaptada con respaldo cruzado"

    historial = pd.DataFrame(historial_list)
    for col in ["Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Atajadas", "Amarillas", "Rojas", "Corners Rival"]:
        if col in historial.columns:
            historial[col] = pd.to_numeric(historial[col], errors="coerce").fillna(0) * historial["Factor_Ajuste"]

    if "Goles" in historial.columns and "Goles Rival" in historial.columns:
        historial["Diff_Goles"] = historial["Goles"] - historial["Goles Rival"]

    n_obs = len(historial)
    
    # Semáforo de riesgo para Kelly
    if len(df_exactos) >= 2:
        semaforo_val = "verde"
        st.markdown('<div class="pill-badge pill-green">🟢 <b>Semáforo: ALTA CONFIANZA</b> (Muestra robusta)</div>', unsafe_allow_html=True)
    elif len(df_exactos) == 1:
        semaforo_val = "amarillo"
        st.markdown('<div class="pill-badge pill-yellow">🟡 <b>Semáforo: CONFIABILIDAD MEDIA</b> (Respaldo activo - Stake atenuado)</div>', unsafe_allow_html=True)
    else:
        semaforo_val = "rojo"
        st.markdown('<div class="pill-badge pill-red">🔴 <b>Semáforo: BAJA CONFIANZA</b> (Muestra forzada - Operar con precaución)</div>', unsafe_allow_html=True)

    # Decaimiento Temporal (Half-Life 30d)
    hoy = pd.Timestamp.today().normalize()
    historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.clip(lower=0)
    historial["Peso_Temporal"] = np.power(0.5, historial["Dias_Pasados"] / 30.0)
    historial["Peso_Total"] = historial["Peso_Temporal"] * historial["Peso_Contexto"]
    pesos = historial["Peso_Total"] / historial["Peso_Total"].sum() if historial["Peso_Total"].sum() > 0 else np.ones(n_obs)/n_obs

    def prom(col):
        return round(float(np.average(historial[col].fillna(0), weights=pesos)), 4) if col in historial.columns else 0.05

    lam_f_raw, lam_c_raw = prom("Goles"), prom("Goles Rival")
    lam_t, lam_tp = prom("Tiros"), prom("A Puerta")
    lam_co, lam_fa = prom("Corners"), prom("Faltas")

    # Intervalos Bootstrap de Goles
    goles_vals = historial["Goles"].values if "Goles" in historial.columns else np.array([0])
    _, lam_f_inf, lam_f_sup = bootstrap_lambda_intervalo(goles_vals, pesos.values)

    df_nivel = df[(df["Liga"] == liga_sel) & (df["Nivel Rival"] == nivel_sel)]
    prior_f = float(df_nivel["Goles"].mean()) if len(df_nivel) else lam_f_raw
    prior_c = float(df_nivel["Goles Rival"].mean()) if len(df_nivel) else lam_c_raw

    if usar_shrinkage:
        lam_f = shrinkage_lambda(lam_f_raw, prior_f, n_obs, k_shrink)
        lam_c = shrinkage_lambda(lam_c_raw, prior_c, n_obs, k_shrink)
    else:
        lam_f, lam_c = lam_f_raw, lam_c_raw

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

    # XGBoost ponderado por tamaño de muestra
    features_modelo = ["Goles_Media_Movil_5", "Goles_Volatilidad_5", "Tiros_Media_Movil_5", "Conversion_Tiros", "Momentum_Goles", "Diff_Goles"]
    modelo_xgb_global = _entrenar_xgboost_real(df, features_modelo)
    
    triunfos, modo_hibrido = predecir_probabilidad_hibrida_dinamica(triunfos_base, historial, features_modelo, modelo_xgb_global, n_obs)
    derrotas, _ = predecir_probabilidad_hibrida_dinamica(derrotas_base, historial, features_modelo, modelo_xgb_global, n_obs)
    empates = max(0.0, 100.0 - triunfos - derrotas)

    ambos_anotan = ((sg_fav > 0) & (sg_con > 0)).mean() * 100
    doble_1x, doble_x2 = triunfos + empates, derrotas + empates
    tot_sin_emp = triunfos + derrotas
    dnb = (triunfos / tot_sin_emp * 100) if tot_sin_emp > 0 else 50.0

    prob_over_goles = (sg_fav > linea_goles).mean() * 100
    prob_under_goles = 100.0 - prob_over_goles
    prob_over_corners = (s_corn > linea_corners).mean() * 100
    prob_under_corners = 100.0 - prob_over_corners
    prob_over_total = ((sg_fav + sg_con) > linea_total_partido).mean() * 100
    prob_under_total = 100.0 - prob_over_total

    marcadores = [f"{f}-{c}" for f, c in zip(sg_fav, sg_con)]
    marcador_mas_comun = Counter(marcadores).most_common(1)[0][0]

    render_header_equipo(liga_sel, equipo_sel, condicion_label, nivel_sel)
    st.markdown(f'<div class="veredicto-box"><b>Veredicto:</b> Marcador más probable: <b>{marcador_mas_comun}</b> | Motor activo: <i>{modo_hibrido}</i></div>', unsafe_allow_html=True)
    
    analisis_texto = generar_analisis_dinamico(equipo_sel, condicion_label, nivel_sel, n_obs, lam_f, lam_c, lam_t, lam_tp, lam_co, triunfos, ambos_anotan, prob_over_goles, prob_over_corners, prob_over_tp=0)
    st.markdown(f'<div class="analisis-dinamico-box">{analisis_texto}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard Principal & Gráficos",
        "💰 Value Bets & Inteligencia",
        "📋 Análisis Táctico & Auditoría"
    ])

    with tab1:
        st.subheader("ADN del Equipo y Métricas con Intervalos Bootstrap")
        renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa)
        
        a, b, c, d = st.columns(4)
        a.metric("Victoria", f"{triunfos:.1f}%")
        b.metric("Empate", f"{empates:.1f}%")
        c.metric("Derrota", f"{derrotas:.1f}%")
        d.metric("BTTS", f"{ambos_anotan:.1f}%")

        st.markdown("---")
        e, f, g = st.columns(3)
        e.metric("Goles Esperados (λ)", f"{lam_f:.2f}", f"IC 95%: [{lam_f_inf:.2f} - {lam_f_sup:.2f}]")
        f.metric("Goles Rival (λ)", f"{lam_c:.2f}")
        g.metric("Tiros a Puerta", f"{lam_tp:.1f}")

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
                fila_texto.append(f"{prob:.1f}%" if prob >= 0.1 else "<0.1%")
            text_data.append(fila_texto)

        fig_matrix = go.Figure(data=go.Heatmap(
            z=matriz_probs, x=[str(i) for i in range(max_g + 1)], y=[str(i) for i in range(max_g + 1)],
            text=text_data, texttemplate="%{text}", textfont={"size": 13, "color": "white"},
            colorscale=[[0, "#111827"], [0.5, "#1d4ed8"], [1, "#10b981"]], showscale=False
        ))
        fig_matrix.update_layout(xaxis_title="Goles Rival", yaxis_title=f"Goles {equipo_sel}", paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F3F4F6"), height=380, margin=dict(l=40, r=40, t=40, b=40))
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("💰 Value Bets (Overs & Unders con Half-Kelly)")
        items_value = [
            ("Victoria (1)", triunfos, cuota_casa_1),
            ("Empate (X)", empates, cuota_casa_x),
            ("Derrota (2)", derrotas, cuota_casa_2),
            ("BTTS Si", ambos_anotan, cuota_casa_btts_si),
            ("BTTS No", 100 - ambos_anotan, cuota_casa_btts_no),
            (f"Over {linea_goles} Goles", prob_over_goles, cuota_over_goles),
            (f"Under {linea_goles} Goles", prob_under_goles, cuota_under_goles),
            (f"Over {linea_total_partido} Partido", prob_over_total, cuota_over_total),
            (f"Under {linea_total_partido} Partido", prob_under_total, cuota_under_total),
        ]
        
        lista_eval = []
        for nombre, prob, cuota in items_value:
            justa = round(100 / prob, 2) if prob > 0 else 99.0
            ev = calcular_ev(prob, cuota)
            lista_eval.append((nombre, justa, cuota, ev, prob))

        lista_eval.sort(key=lambda x: x[3], reverse=True)
        
        ca, cb = st.columns(2)
        mid = (len(lista_eval) + 1) // 2
        with ca:
            for it in lista_eval[:mid]:
                mostrar_value(it[0], it[1], it[2], it[3], it[4], n_obs=n_obs, semaforo=semaforo_val)
        with cb:
            for it in lista_eval[mid:]:
                mostrar_value(it[0], it[1], it[2], it[3], it[4], n_obs=n_obs, semaforo=semaforo_val)

    with tab3:
        st.subheader("📈 Validación Retrospectiva (Backtesting del Motor Real)")
        log_loss_val, brier_val = calcular_backtesting_motor_real(historial, usar_dc, rho_dc)
        
        bc1, bc2 = st.columns(2)
        if log_loss_val is not None:
            bc1.metric("Log Loss (Poisson Engine)", f"{log_loss_val:.4f}", "Calibración del motor")
            bc2.metric("Brier Score (Poisson Engine)", f"{brier_val:.4f}", "Precisión de probabilidad")
        else:
            st.info("ℹ️ Se requieren al menos 5 partidos en este filtro para ejecutar el backtesting estricto del motor.")

        st.markdown("---")
        st.markdown("#### 📘 Guía de Calibración Operativa")
        df_guia_limpia = pd.DataFrame([
            {"Métrica": "Log Loss", "Excelente": "< 0.50", "Aceptable": "0.50 - 0.69", "Deficiente": "> 0.69"},
            {"Métrica": "Brier Score", "Excelente": "< 0.15", "Aceptable": "0.16 - 0.25", "Deficiente": "> 0.25"}
        ])
        st.dataframe(df_guia_limpia, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Auditoría de Partidos Filtrados")
        h_mostrar = historial.copy().sort_values(by="Fecha", ascending=False)
        cols = [c for c in ["Fecha", "Liga", "Condición", "Rival", "Nivel Rival", "Goles", "Goles Rival", "Tiros", "A Puerta", "Tipo_Uso", "Factor_Ajuste"] if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)
else:
    st.info("Configura los parámetros en la barra lateral y pulsa Analizar.")
