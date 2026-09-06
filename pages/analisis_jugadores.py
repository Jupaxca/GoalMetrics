import html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import requests
import unicodedata

try:
    import xgboost as xgb
    import sklearn  # requerido por XGBClassifier
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False
    xgb = None

st.set_page_config(
    page_title="GoalMetrics | Análisis de Jugadores (Híbrido Pro)",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data(ttl=600)
def cargar_datos_jugadores():
    sheet_id = st.secrets.get("JUGADORES_SHEET_ID") or st.secrets.get("EQUIPOS_SHEET_ID")
    if not sheet_id:
        st.error("Error crítico: No se ha configurado 'JUGADORES_SHEET_ID' ni 'EQUIPOS_SHEET_ID' en st.secrets.")
        st.stop()
        
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

def normalizar_texto(texto):
    if not texto:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower().strip()

@st.cache_data(ttl=86400)
def obtener_foto_jugador(nombre, liga):
    nombre_limpio = str(nombre).strip()
    liga_limpia = str(liga).strip()
    
    nombres_busqueda = {
        "vinicius": "Vinicius Junior",
        "mbappe": "Kylian Mbappe",
        "tzolis": "Christos Tzolis",
        "odegard": "Martin Odegaard",
        "semenyo": "Antoine Semenyo",
        "haaland": "Erling Haaland",
        "yamal": "Lamine Yamal",
        "raphina": "Raphinha",
        "kane": "Harry Kane",
        "olise": "Michael Olise",
        "bruno fernandez": "Bruno Fernandes",
        "bruno fernandes": "Bruno Fernandes",
        "joao pedro": "Joao Pedro footballer",
        "palmer": "Cole Palmer",
        "gakpo": "Cody Gakpo",
        "szoboszlai": "Dominik Szoboszlai",
        "antony": "Antony Matheus dos Santos",
        "lookman": "Ademola Lookman",
        "gordon": "Anthony Gordon footballer",
        "bellingham": "Jude Bellingham"
    }
    
    nombre_lower = normalizar_texto(nombre_limpio)
    busqueda_exacta = nombres_busqueda.get(nombre_lower, nombre_limpio)
    
    queries = [
        f"{busqueda_exacta} {liga_limpia} football player",
        f"{busqueda_exacta} football player",
        f"{busqueda_exacta}"
    ]
    
    headers = {'User-Agent': 'GoalMetricsApp/1.0 (contact@goalmetrics.com)'}
    
    for q in queries:
        try:
            url_search = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(q)}&format=json"
            res = requests.get(url_search, headers=headers, timeout=3).json()
            search_results = res.get("query", {}).get("search", [])
            if search_results:
                page_title = search_results[0]["title"]
                url_image = f"https://en.wikipedia.org/w/api.php?action=query&titles={requests.utils.quote(page_title)}&prop=pageimages&pithumbsize=300&format=json"
                res_img = requests.get(url_image, headers=headers, timeout=3).json()
                pages = res_img.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if "thumbnail" in page_info:
                        return page_info["thumbnail"]["source"]
        except Exception:
            continue
            
    return f"https://api.dicebear.com/7.x/initials/svg?seed={requests.utils.quote(nombre_limpio)}&backgroundColor=3B82F6&textColor=ffffff&fontWeight=700"

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

@st.cache_resource(show_spinner=False)
def _entrenar_xgboost_jugadores_cached(df_hash, features_tuple):
    return None

def entrenar_predictor_xgboost_jugadores(df_historico, features_modelo):
    if not XGB_DISPONIBLE or len(df_historico) < 25:
        return None
        
    df_clean = df_historico.dropna(subset=list(features_modelo) + ["Goles"]).copy()
    if len(df_clean) < 25:
        return None
        
    df_clean["Target_Gol"] = (df_clean["Goles"] > 0).astype(int)
    X = df_clean[list(features_modelo)]
    y = df_clean["Target_Gol"]
    
    try:
        model = xgb.XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X, y)
        return model
    except Exception:
        return None

def predecir_probabilidad_hibrida_jugador(prob_poisson, jugador_actual_df, features_modelo, modelo_xgb, n_obs):
    if modelo_xgb is None or jugador_actual_df.empty or n_obs < 5:
        return prob_poisson
    ultima_fila = jugador_actual_df.tail(1)
    try:
        X_pred = ultima_fila[list(features_modelo)]
        prob_xgb = float(modelo_xgb.predict_proba(X_pred)[0][1]) * 100.0
    except Exception:
        return prob_poisson

    peso_xgb = 0.30 if n_obs >= 10 else 0.15
    peso_poisson = 1.0 - peso_xgb
    prob_hibrida = (peso_poisson * prob_poisson) + (peso_xgb * prob_xgb)
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

def calcular_backtesting_retrospectivo_jugadores(historial_filtrado):
    if len(historial_filtrado) < 5:
        return None, None, None, None, None
    y_true, y_prob = [], []
    sub_df = historial_filtrado.tail(30).copy()
    
    for i in range(2, len(sub_df)):
        train_window = sub_df.iloc[:i]
        test_row = sub_df.iloc[i:i+1]
        
        g_jugador = test_row["Goles"].values[0]
        actual_gol = 1 if g_jugador > 0 else 0
        
        mean_goles = train_window["Goles"].mean()
        prob_est = 1.0 - np.exp(-mean_goles)
        
        y_true.append(actual_gol)
        y_prob.append(np.clip(prob_est, 0.01, 0.99))
        
    if not y_true:
        return None, None, None, None, None
        
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    log_loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    bins = np.linspace(0, 1, 11)
    binned_prob = []
    binned_true = []
    counts = []
    for i in range(10):
        lower = bins[i]
        upper = bins[i+1] if i < 9 else 1.01
        mask = (y_prob >= lower) & (y_prob < upper)
        if np.any(mask):
            binned_prob.append(float(y_prob[mask].mean()))
            binned_true.append(float(y_true[mask].mean()))
            counts.append(int(np.sum(mask)))
            
    return round(float(log_loss), 4), round(float(brier_score), 4), binned_prob, binned_true, counts

def generar_analisis_dinamico_jugador(jugador, condicion, nivel, n_obs, lam_g, lam_t, lam_p, prob_goles, prob_puerta, prob_contrib):
    if prob_goles >= 45:
        tendencia = "altamente influyente, punzante y con gran protagonismo ofensivo"
    elif prob_goles >= 25:
        tendencia = "constante, participativo y con aportes ofensivos recurrentes"
    else:
        tendencia = "de perfil táctico y contenido, con menor frecuencia rematadora reciente"
    
    mercados_destacados = []
    if prob_goles >= 50.0:
        mercados_destacados.append(f"<b>Línea de Goles / Anotar</b> (alta probabilidad estimada en <b>{prob_goles:.1f}%</b>)")
    if prob_puerta >= 55.0:
        mercados_destacados.append(f"<b>Tiros a Puerta</b> (respaldado por una media sólida de <b>{lam_p:.1f}</b> remates al arco)")
    if prob_contrib >= 55.0:
        mercados_destacados.append(f"<b>Gol o Asistencia (Contribución)</b> con una proyección de éxito del <b>{prob_contrib:.1f}%</b>")
        
    if not mercados_destacados:
        recomendacion_mercado = "líneas conservadoras o mercados de menor exposición debido a la varianza y paridad del emparejamiento."
    else:
        recomendacion_mercado = "apostar con mayor respaldo estadístico por: " + ", ".join(mercados_destacados) + "."

    texto = (
        f"<b>Análisis Dinámico Automatizado:</b> Para el desempeño de <b>{jugador}</b> actuando como <b>{condicion}</b> "
        f"frente a rivales de nivel <b>{nivel}</b> (analizando una muestra filtrada de <b>{n_obs} partidos</b>), "
        f"el modelo proyecta un perfil <b>{tendencia}</b>. <br><br>"
        f"• <b>Producción Ofensiva:</b> Mantiene una expectativa de <b>{lam_g:.2f} goles</b> por encuentro.<br>"
        f"• <b>Volumen de Remates:</b> Registra una media de <b>{lam_t:.1f} tiros totales</b> y <b>{lam_p:.1f} remates dirigidos a puerta</b>.<br>"
        f"• <b>Implicaciones de Mercado (Basado en Probabilidades):</b> De acuerdo con las simulaciones estocásticas, el mayor valor analítico y probabilidad de cumplimiento se concentra en {recomendacion_mercado}"
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
/* Estilo SaaS de Élite para Jugadores */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

header[data-testid="stHeader"] {
    visibility: visible !important;
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

.header-box {
    background: linear-gradient(135deg, #3B82F6 0%, #0f172a 100%);
    padding: 24px 30px; 
    border-radius: 16px; 
    color: white;
    font-weight: 700; 
    font-size: 24px; 
    margin-bottom: 25px; 
    text-align: center;
    box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.pill-badge {
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
}
.pill-green { background-color: rgba(6, 78, 59, 0.7); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
.pill-yellow { background-color: rgba(120, 53, 15, 0.7); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
.pill-red { background-color: rgba(127, 29, 29, 0.7); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }

.veredicto-box {
    padding: 18px 22px; border-radius: 14px; background-color: #111827;
    border: 1px solid #1f2937; border-left: 5px solid #3B82F6; margin-bottom: 20px; font-size: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.analisis-dinamico-box {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #3B82F666;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    font-size: 15px;
    line-height: 1.6;
    color: #e5e7eb;
}
.value-box { padding: 14px 16px; border-radius: 12px; margin-bottom: 12px; font-size: 14px; border: 1px solid #1f2937; transition: all 0.2s ease; }
.value-yes { background-color: rgba(6, 78, 59, 0.4); border-left: 4px solid #10b981; }
.value-no { background-color: #111827; border-left: 4px solid #4b5563; }
.top-pick-box { background: linear-gradient(135deg, rgba(6, 95, 70, 0.8) 0%, #111827 100%); padding: 24px; border-radius: 14px; border: 2px solid #10b981; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2); }

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 16px 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease-in-out;
    margin-bottom: 12px;
}
div[data-testid="stMetric"]:hover {
    border-color: #374151;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
}
div[data-testid="stMetric"] label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

.saas-card {
    background-color: #111827;
    border: 1px solid #3B82F644;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease-in-out;
}
.saas-card:hover {
    border-color: #3B82F6;
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f2937;
}
[data-testid="stDataFrame"] th {
    background-color: #1f2937 !important;
    color: #f3f4f6 !important;
    font-weight: 600 !important;
}
[data-testid="stDataFrame"] td {
    background-color: #111827 !important;
    color: #9ca3af !important;
}
</style>
""", unsafe_allow_html=True)

def calcular_ev(prob, cuota):
    if cuota <= 1.0 or prob <= 0:
        return 0.0
    return round((prob / 100 * cuota) - 1, 4)

def calcular_kelly_seguro(prob, cuota, n_obs):
    if cuota <= 1.0 or prob <= 0 or n_obs < 2:
        return 0.0
    p, b = prob / 100.0, cuota - 1.0
    if b <= 0:
        return 0.0
    kelly_fraction = ((p * cuota - 1.0) / b) * 0.5
    if kelly_fraction <= 0:
        return 0.0
    cap_max = 0.015 if n_obs == 2 else (0.01 if n_obs < 6 else 0.02)
    stake_final = min(kelly_fraction, cap_max)
    return round(stake_final * 100, 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, n_obs, real=None, muestra_pequena=False):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake = calcular_kelly_seguro(prob, cuota_casa, n_obs) if es_value else 0.0
    
    if es_value and stake == 0.0:
        kelly_txt = " | <i>Muestra insuficiente para stake</i>"
    elif es_value:
        kelly_txt = f" | Half-Kelly: <b>{stake}% bank</b>"
    else:
        kelly_txt = ""

    real_txt = f" | Acierto real: <b>{real:.0f}%</b>" if real is not None else ""
    caution = " (muestra pequeña)" if muestra_pequena and es_value else ""
    
    st.markdown(
        f'<div class="value-box {clase}"><b>{nombre}</b>{caution}<br>'
        f"Modelo: <b>{prob:.1f}%</b>{real_txt} | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_txt}<br>"
        f'<span style="color:{color_ev}; font-weight:bold; font-size:15px;">'
        f"EV: {ev:+.2%} -> {'VALUE BET' if es_value else 'Sin valor'}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("### Análisis de jugadores")
st.caption("Asistente inteligente con semáforo de confiabilidad, compensación estadística (Shrinkage), Bootstrap IC y gráficos integrados.")

with st.expander("📖 Guía Detallada: ¿Cómo funciona el Análisis?", expanded=False):
    st.markdown("""
    Bienvenido al centro analítico de jugadores. A continuación se detalla cómo operan los módulos principales:
    
    * **1. Semáforo de Confiabilidad:** Clasifica la robustez de la muestra de partidos exactos. 
      * 🟢 *Verde:* Suficientes partidos exactos en el escenario buscado (≥ 2).
      * 🟡 *Amarillo:* Muestra mixta o con 1 solo partido exacto, activando el respaldo inteligente ajustado por *Tier*.
      * 🔴 *Rojo:* Muestra crítica o escasa, requiere máxima precaución.
    * **2. Shrinkage (Compensación Estadística):** Cuando un jugador cuenta con pocos partidos en un escenario específico, sus promedios aparentes pueden estar sesgados. El **Shrinkage** corrige esto encogiendo o ajustando las tasas empíricas hacia una media previa (*prior*) de la liga para ese mismo nivel de rival.
    * **3. Modelo Híbrido (Poisson + XGBoost):** Modela las variables de conteo mediante distribuciones de Poisson y refina las probabilidades con Machine Learning (XGBoost), evaluando momentum y medias móviles recientes de rendimiento.
    * **4. Métricas, Volatilidad (σ) e Intervalos (IC 95%):** Cada tarjeta muestra la tasa esperada (λ) junto con su desviación estándar e intervalos de confianza generados por Bootstrap (remuestreo 500x) para cuantificar la incertidumbre.
    * **5. Value Bets & Criterio de Half-Kelly:** Evalúa el Valor Esperado (EV) comparando la probabilidad del modelo frente a las cuotas de la casa de apuestas y dimensiona el stake de forma conservadora usando el criterio fraccional de Kelly.
    * **6. Validación y Curva de Calibración:** El sistema evalúa retrospectivamente su precisión prediciendo si el jugador anotará (Log Loss / Brier Score) y lo mapea visualmente para detectar sesgos.
    """)

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

    if len(df_exactos) >= 2:
        semaforo_val = "verde"
    elif len(df_exactos) == 1:
        semaforo_val = "amarillo"
    else:
        semaforo_val = "rojo"

    foto_url = obtener_foto_jugador(jugador_sel, liga_sel)

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

    def std_w(col):
        return float(historial[col].std()) if col in historial.columns and len(historial) > 1 else 0.0

    def calc_ci(col_name):
        vals = historial[col_name].fillna(0).values if col_name in historial.columns else np.array([0])
        w = pesos.values if len(pesos) == len(vals) else None
        _, inf, sup = bootstrap_lambda_intervalo(vals, w)
        return inf, sup

    lam_g_raw = prom_w("Goles")
    lam_t_raw = prom_w("Tiros")
    lam_p_raw = prom_w("A Puerta")
    lam_a_raw = prom_w("Asistencias")
    lam_f_raw = prom_w("Faltas")

    ic_goles = calc_ci("Goles")
    ic_asist = calc_ci("Asistencias")
    ic_tiros = calc_ci("Tiros")
    ic_puerta = calc_ci("A Puerta")
    ic_faltas = calc_ci("Faltas")
    
    vals_contrib = (historial["Goles"].fillna(0) + historial["Asistencias"].fillna(0)).values if "Goles" in historial and "Asistencias" in historial else np.array([0])
    _, inf_c, sup_c = bootstrap_lambda_intervalo(vals_contrib, pesos.values if len(pesos) == len(vals_contrib) else None)
    ic_contrib = (inf_c, sup_c)
    std_contrib = float(pd.Series(vals_contrib).std()) if len(vals_contrib) > 1 else 0.0

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

    lam_g = min(lam_g, 2.0)
    lam_t = min(lam_t, 10.0)
    lam_p = min(lam_p, lam_t)  
    lam_g = min(lam_g, lam_p)  
    lam_a = min(lam_a, 1.5)
    lam_f = min(lam_f, 6.0)

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
    
    prob_goles = predecir_probabilidad_hibrida_jugador(prob_goles_base, historial, features_modelo, modelo_xgb_global, n_obs)
    prob_contrib = predecir_probabilidad_hibrida_jugador(prob_contrib_base, historial, features_modelo, modelo_xgb_global, n_obs)
    prob_tiros = prob_tiros_base
    prob_puerta = prob_puerta_base
    prob_asist = prob_asist_base
    prob_faltas = prob_faltas_base

    def cj(p):
        return round(100 / p, 2) if p > 0 else 99.0

    lista_mercados = [
        {"nombre": f"Over {linea_goles} Goles", "prob": prob_goles, "cuota": cuota_casa_goles, "ev": calcular_ev(prob_goles, cuota_casa_goles)},
        {"nombre": f"Over {linea_tiros} Tiros", "prob": prob_tiros, "cuota": cuota_casa_tiros, "ev": calcular_ev(prob_tiros, cuota_casa_tiros)},
        {"nombre": f"Over {linea_puerta} a Puerta", "prob": prob_puerta, "cuota": cuota_casa_puerta, "ev": calcular_ev(prob_puerta, cuota_casa_puerta)},
        {"nombre": f"Over {linea_asist} Asistencias", "prob": prob_asist, "cuota": cuota_casa_asist, "ev": calcular_ev(prob_asist, cuota_casa_asist)},
        {"nombre": f"Over {linea_faltas} Faltas", "prob": prob_faltas, "cuota": cuota_casa_faltas, "ev": calcular_ev(prob_faltas, cuota_casa_faltas)},
        {"nombre": f"Over {linea_contrib} Gol/Asist", "prob": prob_contrib, "cuota": cuota_casa_contrib, "ev": calcular_ev(prob_contrib, cuota_casa_contrib)},
    ]
    lista_mercados.sort(key=lambda x: x["ev"], reverse=True)
    value_bets_disponibles = [m for m in lista_mercados if m["ev"] > 0]
    if value_bets_disponibles:
        top_pick = max(value_bets_disponibles, key=lambda x: x["ev"])
        stake_top = calcular_kelly_seguro(top_pick["prob"], top_pick["cuota"], n_obs)
        top_ev_val, top_ev_nombre = top_pick["ev"], top_pick["nombre"]
    else:
        top_pick = None
        stake_top = 0.0
        top_ev_val, top_ev_nombre = 0.0, "Sin value"

    analisis_jugador_texto = generar_analisis_dinamico_jugador(
        jugador_sel, condicion_sel, nivel_sel, n_obs,
        lam_g, lam_t, lam_p,
        prob_goles, prob_puerta, prob_contrib,
    )

    # ---------- métricas de tendencia (para resumen) ----------
    partidos_por_gol = 1.0 / lam_g if lam_g > 0 else 0.0
    partidos_por_asist = 1.0 / lam_a if lam_a > 0 else 0.0
    if len(historial) > 0:
        partidos_con_contrib = ((historial["Goles"] + historial["Asistencias"]) > 0).sum()
        pct_contribucion_real = (partidos_con_contrib / len(historial)) * 100.0
    else:
        pct_contribucion_real = 0.0
    ultimos_partidos = historial.tail(min(3, len(historial)))
    goles_recientes = ultimos_partidos["Goles"].mean() if len(ultimos_partidos) and "Goles" in ultimos_partidos else 0.0
    asist_recientes = ultimos_partidos["Asistencias"].mean() if len(ultimos_partidos) and "Asistencias" in ultimos_partidos else 0.0
    if goles_recientes > lam_g or asist_recientes > lam_a:
        estado_momentum = "Momentum al alza"
    elif goles_recientes < lam_g * 0.5 and asist_recientes < lam_a * 0.5:
        estado_momentum = "Momentum a la baja"
    else:
        estado_momentum = "Momentum estable"

    # ===================== UI LIMPIA =====================
    liga_sel_html = html.escape(str(liga_sel))
    jugador_sel_html = html.escape(str(jugador_sel))
    condicion_sel_html = html.escape(str(condicion_sel))
    nivel_sel_html = html.escape(str(nivel_sel))

    st.markdown(
        f'<div class="header-box" style="display: flex; align-items: center; gap: 20px;">'
        f'<img src="{foto_url}" style="height: 64px; width: 64px; border-radius: 50%; object-fit: cover; border: 2px solid rgba(255,255,255,0.3); background-color: #1f2937;" />'
        f"<div>"
        f'<div style="font-size: 22px; font-weight: 700;">{liga_sel_html.upper()} | {jugador_sel_html.upper()}</div>'
        f'<div style="font-size: 14px; color: #93c5fd; font-weight: 500; margin-top: 4px;">{condicion_sel_html} vs {nivel_sel_html}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if semaforo_val == "verde":
        st.markdown('<div class="pill-badge pill-green">Alta confianza · muestra robusta</div>', unsafe_allow_html=True)
    elif semaforo_val == "amarillo":
        st.markdown('<div class="pill-badge pill-yellow">Confianza media · respaldo activo</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill-badge pill-red">Baja confianza · máxima precaución</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="veredicto-box">'
        f"<b>Prob. anotar (Over {linea_goles}):</b> {prob_goles:.1f}%"
        f' &nbsp;·&nbsp; <span style="color:#9ca3af;font-size:0.9rem;">{estado_momentum}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Over {linea_goles} Goles", f"{prob_goles:.1f}%")
    k2.metric("λ Goles", f"{lam_g:.2f}")
    k3.metric("Top EV", f"{top_ev_val:+.1%}", top_ev_nombre)
    k4.metric("Stake sugerido", f"{stake_top:.2f}%", "Half-Kelly" if top_ev_val > 0 else "Sin value")

    st.caption(f"{n_obs} partidos · {fuente}")
    if muestra_pequena:
        st.caption("Muestra pequeña: stake limitado por seguridad.")

    tab_resumen, tab_value, tab_datos = st.tabs(["Resumen", "Value", "Datos"])

    with tab_resumen:
        c_a, c_b, c_c, c_d = st.columns(4)
        c_a.metric("λ Asistencias", f"{lam_a:.2f}")
        c_b.metric("λ Tiros", f"{lam_t:.1f}")
        c_c.metric("λ A puerta", f"{lam_p:.1f}")
        c_d.metric("Gol o Asist", f"{prob_contrib:.1f}%")

        freq_gol = f"gol cada {partidos_por_gol:.1f} partidos" if partidos_por_gol > 0 else "baja incidencia"
        st.markdown(
            f'<div class="veredicto-box">'
            f"• Anota {freq_gol}. Contribuye en el <b>{pct_contribucion_real:.0f}%</b> de los partidos del filtro.<br>"
            f"• {estado_momentum}."
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Métricas e intervalos", expanded=False):
            metrics_data = {
                "Goles": {"prom": historial["Goles"].mean() if "Goles" in historial else 0, "lam": lam_g, "vol": std_w("Goles"), "ic": ic_goles},
                "Asistencias": {"prom": historial["Asistencias"].mean() if "Asistencias" in historial else 0, "lam": lam_a, "vol": std_w("Asistencias"), "ic": ic_asist},
                "Tiros": {"prom": historial["Tiros"].mean() if "Tiros" in historial else 0, "lam": lam_t, "vol": std_w("Tiros"), "ic": ic_tiros},
                "A Puerta": {"prom": historial["A Puerta"].mean() if "A Puerta" in historial else 0, "lam": lam_p, "vol": std_w("A Puerta"), "ic": ic_puerta},
                "Faltas": {"prom": historial["Faltas"].mean() if "Faltas" in historial else 0, "lam": lam_f, "vol": std_w("Faltas"), "ic": ic_faltas},
            }
            cols = st.columns(3)
            for i, (var, datos) in enumerate(metrics_data.items()):
                col_target = cols[i % 3]
                ic_text = f"IC [{datos['ic'][0]:.2f}–{datos['ic'][1]:.2f}]"
                col_target.metric(var, f"{datos['prom']:.2f}", f"λ {datos['lam']:.2f} · σ {datos['vol']:.2f} · {ic_text}")

        with st.expander("Perfil (radar)", expanded=False):
            avg_g = float(historial["Goles"].mean()) if "Goles" in historial else 0.0
            avg_t = float(historial["Tiros"].mean()) if "Tiros" in historial else 0.0
            avg_p = float(historial["A Puerta"].mean()) if "A Puerta" in historial else 0.0
            avg_a = float(historial["Asistencias"].mean()) if "Asistencias" in historial else 0.0
            avg_f = float(historial["Faltas"].mean()) if "Faltas" in historial else 0.0
            avg_p = min(avg_p, avg_t)
            avg_g = min(avg_g, avg_p) if avg_p > 0 else avg_g
            categories = ["Goles", "Tiros", "A Puerta", "Asistencias", "Faltas"]
            vals = [min(avg_g, 10.0), min(avg_t, 10.0), min(avg_p, 10.0), min(avg_a, 10.0), min(avg_f, 10.0)]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                marker=dict(color="#3B82F6"),
                line=dict(color="#60A5FA", width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10], color="#9ca3af"), bgcolor="#111827"),
                showlegend=False, paper_bgcolor="#111827", plot_bgcolor="#111827",
                font=dict(color="#F3F4F6", size=11), height=320, margin=dict(l=20, r=20, t=10, b=10),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with st.expander("Curvas Over", expanded=False):
            def crear_grafico_acumulado(sim_data, titulo_metrica, color_linea="#10b981"):
                if "Gol" in titulo_metrica:
                    lines = [0.5, 1.5, 2.5, 3.5]
                elif "Puerta" in titulo_metrica:
                    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
                else:
                    lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
                probs = [(sim_data > l).mean() * 100 for l in lines]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[str(l) for l in lines], y=probs, mode="lines+markers+text",
                    text=[f"{p:.1f}%" if p > 0.05 else "0.0%" for p in probs],
                    textposition="top center",
                    line=dict(color=color_linea, width=3), marker=dict(size=8, color=color_linea),
                ))
                fig.update_layout(
                    title=f"{titulo_metrica} (Over X)",
                    xaxis_title="Línea", yaxis_title="Probabilidad (%)",
                    paper_bgcolor="#111827", plot_bgcolor="#111827",
                    font=dict(color="#F3F4F6", size=11),
                    yaxis=dict(range=[0, 115], gridcolor="#1f2937"),
                    xaxis=dict(gridcolor="#1f2937"),
                    height=300, margin=dict(l=30, r=20, t=40, b=30),
                )
                return fig
            st.plotly_chart(crear_grafico_acumulado(sim_goles, "Goles", "#10b981"), use_container_width=True)
            st.plotly_chart(crear_grafico_acumulado(sim_puerta, "Tiros a Puerta", "#3B82F6"), use_container_width=True)
            st.plotly_chart(crear_grafico_acumulado(sim_tiros, "Tiros Totales", "#F59E0B"), use_container_width=True)

        with st.expander("Reporte táctico", expanded=False):
            st.markdown(f'<div class="analisis-dinamico-box">{analisis_jugador_texto}</div>', unsafe_allow_html=True)

    with tab_value:
        if top_pick is not None:
            st.markdown(
                f'<div class="top-pick-box">'
                f"<h3>Top Value Bet</h3>"
                f'<p>Mercado: <b>{html.escape(top_pick["nombre"])}</b></p>'
                f"<p>Prob: <b>{top_pick['prob']:.1f}%</b> · Cuota: <b>{top_pick['cuota']}</b> · EV: <b>{top_pick['ev']:+.2%}</b></p>"
                f'<p style="color:#10b981;font-weight:bold;">Stake: {stake_top}% bank</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No hay mercados con EV positivo.")

        st.markdown("##### Props")
        for m in lista_mercados:
            real = None
            nom = m["nombre"]
            if "Goles" in nom and "Gol/Asist" not in nom:
                real = (historial["Goles"] > linea_goles).mean() * 100
            elif "Tiros" in nom and "Puerta" not in nom:
                real = (historial["Tiros"] > linea_tiros).mean() * 100
            elif "Puerta" in nom:
                real = (historial["A Puerta"] > linea_puerta).mean() * 100
            elif "Asist" in nom:
                real = (historial["Asistencias"] > linea_asist).mean() * 100
            elif "Faltas" in nom:
                real = (historial["Faltas"] > linea_faltas).mean() * 100
            elif "Gol/Asist" in nom:
                real = ((historial["Goles"] + historial["Asistencias"]) > linea_contrib).mean() * 100
            mostrar_value(m["nombre"], cj(m["prob"]), m["cuota"], m["ev"], m["prob"], n_obs=n_obs, real=real, muestra_pequena=muestra_pequena)

        with st.expander("Constructor de combinada", expanded=False):
            nombres_mercados = [m["nombre"] for m in lista_mercados]
            parlay_elegidos = st.multiselect("Mercados para la combinada:", options=nombres_mercados, key="parlay_jugador_input")
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
                st.markdown(f"**Probabilidad conjunta:** `{prob_conjunta_pct:.2f}%` · **Cuota justa:** `{cuota_justa_combinada}`")
                cuota_casa_parlay = st.number_input(
                    "Cuota de la casa:",
                    min_value=1.01,
                    value=float(cuota_justa_combinada * 0.95),
                    step=0.05,
                    format="%.2f",
                    key="cuota_parlay_jug_input",
                )
                ev_parlay = calcular_ev(prob_conjunta_pct, cuota_casa_parlay)
                stake_parlay = calcular_kelly_seguro(prob_conjunta_pct, cuota_casa_parlay, n_obs)
                if ev_parlay > 0:
                    st.success(f"EV {ev_parlay:+.2%} · Stake {stake_parlay}%")
                else:
                    st.warning(f"EV {ev_parlay:+.2%}")

    with tab_datos:
        with st.expander("Backtesting (anotar gol)", expanded=True):
            log_loss_val, brier_val, bin_p, bin_t, bin_c = calcular_backtesting_retrospectivo_jugadores(historial)
            bc1, bc2 = st.columns(2)
            if log_loss_val is not None:
                bc1.metric("Log Loss", f"{log_loss_val:.4f}")
                bc2.metric("Brier Score", f"{brier_val:.4f}")
                if bin_p and len(bin_p) > 0:
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfecta", line=dict(dash="dash", color="#9ca3af")))
                    fig_cal.add_trace(go.Scatter(
                        x=bin_p, y=bin_t, mode="lines+markers", name="Modelo",
                        text=[f"n={c}" for c in bin_c], hoverinfo="text+x+y",
                        marker=dict(size=[max(8, c * 3) for c in bin_c], color="#3B82F6"),
                        line=dict(color="#3B82F6", width=2),
                    ))
                    fig_cal.update_layout(
                        xaxis_title="Prob. predicha", yaxis_title="Frecuencia real",
                        paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F3F4F6"),
                        xaxis=dict(range=[0, 1], gridcolor="#1f2937", tickformat=".0%"),
                        yaxis=dict(range=[0, 1], gridcolor="#1f2937", tickformat=".0%"),
                        height=340, margin=dict(l=40, r=40, t=30, b=40),
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.info("Se necesitan al menos 5 partidos para backtesting.")

        with st.expander("Partidos del filtro", expanded=True):
            h_mostrar = historial.copy()
            if "Fecha" in h_mostrar.columns:
                h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
            cols_mostrar = [c for c in ["Fecha", "Condición", "Rival", "Nivel Rival", "Goles", "Asistencias", "Tiros", "A Puerta", "Faltas", "Tipo_Uso", "Factor_Ajuste"] if c in h_mostrar.columns]
            st.dataframe(h_mostrar[cols_mostrar], hide_index=True, use_container_width=True)

else:
    st.info("Configura las opciones en la barra lateral, elige jugador y haz clic en Analizar.")
