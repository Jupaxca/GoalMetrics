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
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False

st.set_page_config(
    page_title="GoalMetrics | Análisis de Jugadores (Híbrido Pro)",
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
    
    headers = {'User-Agent': 'GoalMetricsApp/1.0'}
    for q in queries:
        try:
            url_search = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(q)}&format=json"
            res = requests.get(url_search, headers=headers, timeout=2).json()
            search_results = res.get("query", {}).get("search", [])
            if search_results:
                page_title = search_results[0]["title"]
                url_image = f"https://en.wikipedia.org/w/api.php?action=query&titles={requests.utils.quote(page_title)}&prop=pageimages&pithumbsize=200&format=json"
                res_img = requests.get(url_image, headers=headers, timeout=2).json()
                pages = res_img.get("query", {}).get("pages", {})
                for _, page_info in pages.items():
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
        n_estimators=60, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss"
    )
    model.fit(X, y)
    return model

def predecir_probabilidad_hibrida_dinamica(prob_poisson, jugador_actual_df, features_modelo, modelo_xgb, n_obs):
    if modelo_xgb is None or jugador_actual_df.empty or n_obs < 5:
        return prob_poisson, "100% Poisson (Muestra pequeña)"

    ultima_fila = jugador_actual_df.tail(1)
    try:
        X_pred = ultima_fila[features_modelo]
        prob_xgb = float(modelo_xgb.predict_proba(X_pred)[0][1]) * 100.0
    except Exception:
        return prob_poisson, "100% Poisson (Fallback)"

    peso_xgb = 0.30 if n_obs >= 10 else 0.15
    peso_poisson = 1.0 - peso_xgb
    prob_hibrida = (peso_poisson * prob_poisson) + (peso_xgb * prob_xgb)
    modo_str = f"Híbrido {int(peso_poisson*100)}/{int(peso_xgb*100)}"
    return round(prob_hibrida, 2), modo_str

def shrinkage_lambda(lam_obs, lam_prior, n_obs, k=5.0):
    n = max(float(n_obs), 0.0)
    return (n * lam_obs + k * lam_prior) / (n + k)

def bootstrap_lambda_intervalo(valores, pesos=None, n_bootstrap=500, alpha=0.05):
    if len(valores) == 0:
        return 0.0, 0.0, 0.0
    vals = np.array(valores)
    p = np.array(pesos) / np.array(pesos).sum() if pesos is not None else None
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(vals), size=len(vals), replace=True, p=p)
        boot_means.append(np.mean(vals[indices]))
    return float(np.mean(vals)), float(np.percentile(boot_means, 100 * (alpha / 2))), float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

def generar_analisis_dinamico_jugador(jugador, condicion, nivel, n_obs, lam_g, lam_t, lam_p, prob_goles, prob_puerta, prob_contrib):
    if prob_goles >= 45:
        tendencia = "altamente influyente, punzante y con gran protagonismo ofensivo"
    elif prob_goles >= 25:
        tendencia = "constante, participativo y con aportes ofensivos recurrentes"
    else:
        tendencia = "de perfil táctico y contenido, con menor frecuencia rematadora reciente"
    
    mercados_destacados = []
    if prob_goles >= 50.0:
        mercados_destacados.append(f"<b>Línea de Goles / Anotar</b> (probabilidad del <b>{prob_goles:.1f}%</b>)")
    if prob_puerta >= 55.0:
        mercados_destacados.append(f"<b>Tiros a Puerta</b> (media de <b>{lam_p:.1f}</b> remates)")
    if prob_contrib >= 55.0:
        mercados_destacados.append(f"<b>Gol o Asistencia</b> (probabilidad del <b>{prob_contrib:.1f}%</b>)")
        
    recomendacion_mercado = "líneas conservadoras o Unders debido a la paridad." if not mercados_destacados else "apostar por: " + ", ".join(mercados_destacados) + "."

    return (
        f"<b>Análisis Dinámico:</b> Para <b>{jugador}</b> como <b>{condicion}</b> "
        f"vs <b>{nivel}</b> ({n_obs} partidos), el perfil es <b>{tendencia}</b>.<br><br>"
        f"• <b>Ofensiva:</b> Expectativa de <b>{lam_g:.2f} goles</b>, con <b>{lam_t:.1f} tiros</b> y <b>{lam_p:.1f} a puerta</b>.<br>"
        f"• <b>Recomendación:</b> {recomendacion_mercado}"
    )

def obtener_peso_tier(tier):
    t = str(tier).upper().strip()
    if "TOP" in t or "CHAMPIONS" in t: return 3
    elif "MEDIA" in t: return 2
    elif "DESCENSO" in t or "BAJO" in t: return 1
    return 2

def calcular_factores_respaldo(row_data, condicion_buscada, tier_objetivo):
    cond_partido = str(row_data.get("Condición", "")).lower()
    tier_partido = str(row_data.get("Nivel Rival", ""))
    t_match = obtener_peso_tier(tier_partido)

    f_cond = 1.0 if cond_partido == condicion_buscada else (0.90 if condicion_buscada == "visitante" else 1.05)
    diff = tier_objetivo - t_match
    f_tier = 1.0 if diff == 0 else (max(0.65, 1.0 - (diff * 0.12)) if diff > 0 else min(1.35, 1.0 + (abs(diff) * 0.10)))
    return f_cond * f_tier, f"Respaldo | Tier ({tier_partido})"

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
    jugador_sel = st.sidebar.selectbox("Selecciona al Jugador", jugadores if jugadores else ["Sin jugadores"])
    df_jugador = df_liga[df_liga["Jugador"] == jugador_sel].copy() if jugadores else pd.DataFrame()
        
    if "Fecha" in df_jugador.columns: df_jugador = df_jugador.sort_values("Fecha")
    condiciones = sorted(df_jugador["Condición"].dropna().unique().tolist()) if "Condición" in df_jugador.columns else ["local", "visitante"]
    condicion_sel = st.sidebar.selectbox("Condicion", [c.capitalize() for c in (condiciones if condiciones else ["local"])])
    condicion_sel_lower = condicion_sel.lower()
    
    niveles = sorted(df_jugador["Nivel Rival"].dropna().unique().tolist()) if "Nivel Rival" in df_jugador.columns else ["TOP", "MEDIA TABLA", "DESCENSO"]
    nivel_sel = st.sidebar.selectbox("Nivel del Rival", niveles if niveles else ["MEDIA TABLA"])
else:
    liga_sel, jugador_sel, condicion_sel, condicion_sel_lower, nivel_sel, df_jugador = "General", "Sin datos", "Local", "local", "DESCENSO", pd.DataFrame()

st.sidebar.markdown("---")
with st.sidebar.expander("Lineas de Estudio (Overs / Unders)", expanded=False):
    linea_goles = st.slider("Linea de Goles", 0.0, 3.0, 0.5, 0.5)
    linea_tiros = st.slider("Linea de Tiros Totales", 0.0, 10.0, 2.5, 0.5)
    linea_puerta = st.slider("Linea de Tiros a Puerta", 0.0, 5.0, 1.5, 0.5)
    linea_asist = st.slider("Linea de Asistencias", 0.0, 2.0, 0.5, 0.5)
    linea_faltas = st.slider("Linea de Faltas", 0.0, 5.0, 1.0, 0.5)
    linea_contrib = st.slider("Linea Gol o Asistencia", 0.0, 3.0, 0.5, 0.5)

with st.sidebar.expander("Cuotas (Over / Under)", expanded=False):
    cuota_over_goles = st.number_input(f"Over {linea_goles} Goles", min_value=1.01, value=2.10, step=0.01, format="%.2f")
    cuota_under_goles = st.number_input(f"Under {linea_goles} Goles", min_value=1.01, value=1.75, step=0.01, format="%.2f")
    cuota_over_tiros = st.number_input(f"Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_over_puerta = st.number_input(f"Over {linea_puerta} a Puerta", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_asist = st.number_input(f"Over {linea_asist} Asistencias", min_value=1.01, value=2.50, step=0.01, format="%.2f")
    cuota_over_contrib = st.number_input(f"Over {linea_contrib} Gol/Asist", min_value=1.01, value=1.70, step=0.01, format="%.2f")

with st.sidebar.expander("Modelo", expanded=False):
    shrink_opt = st.radio("Shrinkage", options=["ON", "OFF"], index=0, horizontal=True, key="radio_shrink_jug")
    usar_shrinkage = (shrink_opt == "ON")
    k_shrink = st.slider("Fuerza prior (k)", 1.0, 15.0, 5.0, 1.0, key="slider_k_jug", disabled=not usar_shrinkage)

total_partidos_jugador = len(df_jugador) if not df_jugador.empty else 0

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0B0F19; color: #F3F4F6; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
header[data-testid="stHeader"] { visibility: visible !important; background: transparent !important; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
.header-box { background: linear-gradient(135deg, #3B82F6 0%, #111827 100%); padding: 24px 30px; border-radius: 16px; color: white; font-weight: 700; font-size: 26px; margin-bottom: 20px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
.pill-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 9999px; font-weight: 600; font-size: 0.9rem; margin-bottom: 20px; }
.pill-green { background-color: rgba(6, 78, 59, 0.7); color: #34d399; }
.pill-yellow { background-color: rgba(120, 53, 15, 0.7); color: #fbbf24; }
.pill-red { background-color: rgba(127, 29, 29, 0.7); color: #f87171; }
.veredicto-box { padding: 18px 22px; border-radius: 14px; background-color: #111827; border: 1px solid #1f2937; border-left: 5px solid #3B82F6; margin-bottom: 20px; font-size: 16px; }
.analisis-dinamico-box { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 20px; border-radius: 14px; border: 1px solid #3B82F666; margin-bottom: 20px; font-size: 15px; line-height: 1.6; color: #e5e7eb; }
.value-box { padding: 14px 16px; border-radius: 12px; margin-bottom: 10px; font-size: 14px; border: 1px solid #1f2937; }
.value-yes { background-color: rgba(6, 78, 59, 0.4); border-left: 4px solid #10b981; }
.value-no { background-color: #111827; border-left: 4px solid #4b5563; }
.top-pick-box { background: linear-gradient(135deg, rgba(6, 95, 70, 0.8) 0%, #111827 100%); padding: 22px; border-radius: 14px; border: 2px solid #10b981; margin-bottom: 20px; }
div[data-testid="stMetric"] { background-color: #111827; border: 1px solid #1f2937; padding: 16px 20px; border-radius: 12px; }
div[data-testid="stMetric"] label { color: #9ca3af !important; font-size: 0.85rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem !important; }
.saas-card { background-color: #111827; border: 1px solid #3B82F644; border-radius: 14px; padding: 20px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

def calcular_ev(prob, cuota):
    if cuota <= 1.0 or prob <= 0: return 0.0
    return round((prob / 100 * cuota) - 1, 4)

def calcular_kelly_seguro(prob, cuota, n_obs, nivel_semaforo="verde"):
    if cuota <= 1.0 or prob <= 0 or n_obs < 3: return 0.0
    p, b = prob / 100.0, cuota - 1.0
    if b <= 0: return 0.0
    kelly = ((p * cuota - 1.0) / b) * 0.5
    if kelly <= 0: return 0.0
    cap = 0.005 if nivel_semaforo == "rojo" else (0.01 if nivel_semaforo == "amarillo" else 0.02)
    return round(min(kelly, cap) * 100, 2)

def mostrar_value(nombre, cuota_justa, cuota_casa, ev, prob, n_obs, semaforo):
    es_value = ev > 0
    clase = "value-yes" if es_value else "value-no"
    color_ev = "#10b981" if es_value else "#9ca3af"
    stake = calcular_kelly_seguro(prob, cuota_casa, n_obs, semaforo) if es_value else 0.0
    kelly_txt = f" | Stake: <b>{stake}% bank</b>" if es_value else ""
    st.markdown(
        f'<div class="value-box {clase}"><b>{nombre}</b><br>'
        f"Modelo: <b>{prob:.1f}%</b> | Justa: <b>{cuota_justa}</b> | Casa: <b>{cuota_casa}</b>{kelly_txt}<br>"
        f'<span style="color:{color_ev}; font-weight:bold;">EV: {ev:+.2%} -> {"VALUE" if es_value else "Sin valor"}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("### Centro de Analisis Individual de Jugadores (Híbrido Pro)")
st.caption("Asistente inteligente con semáforo, intervalos Bootstrap y gestión de riesgo.")

if "analizado_jugadores" not in st.session_state: st.session_state.analizado_jugadores = False

col_b1, col_b2, _ = st.columns([1.2, 1, 4])
with col_b1:
    if st.button("Analizar", type="primary", use_container_width=True): st.session_state.analizado_jugadores = True
with col_b2:
    if st.button("Limpiar", use_container_width=True):
        st.session_state.analizado_jugadores = False
        st.rerun()

if st.session_state.analizado_jugadores:
    if total_partidos_jugador == 0:
        st.error(f"❌ El jugador **{jugador_sel}** tiene 0 partidos registrados.")
        st.stop()
        
    df_exactos = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] == nivel_sel)].copy()
    if len(df_exactos) == 0:
        st.error(f"❌ 0 partidos exactos para **{jugador_sel}** como **{condicion_sel}** vs **{nivel_sel}**.")
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
        df_misma_cond = df_jugador[(df_jugador["Condición"] == condicion_sel_lower) & (df_jugador["Nivel Rival"] != nivel_sel)].copy()
        for _, row in df_misma_cond.tail(UMBRAL_MINIMO - len(historial_list)).iterrows():
            r = row.to_dict()
            f_tot, desc = calcular_factores_respaldo(r, condicion_sel_lower, t_target)
            r["Factor_Ajuste"] = f_tot
            r["Tipo_Uso"] = desc
            r["Peso_Contexto"] = 0.85
            historial_list.append(r)
        fuente = "Muestra mixta con respaldo Tier"

    historial = pd.DataFrame(historial_list)
    for col in ["Goles", "Asistencias", "Tiros", "A Puerta", "Faltas"]:
        if col in historial.columns: historial[col] = historial[col] * historial["Factor_Ajuste"]

    n_obs = len(historial)

    if len(df_exactos) >= 2:
        semaforo_val = "verde"
        st.markdown('<div class="pill-badge pill-green">🟢 <b>Semáforo: ALTA CONFIANZA</b></div>', unsafe_allow_html=True)
    elif len(df_exactos) == 1:
        semaforo_val = "amarillo"
        st.markdown('<div class="pill-badge pill-yellow">🟡 <b>Semáforo: CONFIABILIDAD MEDIA</b> (Respaldo activo - Stake atenuado)</div>', unsafe_allow_html=True)
    else:
        semaforo_val = "rojo"
        st.markdown('<div class="pill-badge pill-red">🔴 <b>Semáforo: BAJA CONFIANZA</b> (Muestra forzada)</div>', unsafe_allow_html=True)

    foto_url = obtener_foto_jugador(jugador_sel, liga_sel)
    st.markdown(
        f'<div class="header-box" style="display: flex; align-items: center; gap: 20px;">'
        f'<img src="{foto_url}" style="height: 64px; width: 64px; border-radius: 50%; object-fit: cover; border: 2px solid rgba(255,255,255,0.3); background-color: #1f2937;" />'
        f'<div><div style="font-size: 22px; font-weight: 700;">{liga_sel.upper()} | {jugador_sel.upper()}</div>'
        f'<div style="font-size: 14px; color: #93c5fd; font-weight: 500; margin-top: 4px;">Condición: {condicion_sel} vs {nivel_sel}</div></div></div>', 
        unsafe_allow_html=True
    )

    hoy = pd.Timestamp.today().normalize()
    historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.replace(0, 0.1) if "Fecha" in historial.columns else 0.1
    historial["Peso_Temporal"] = 1 / (1 + (historial["Dias_Pasados"] / 30))
    historial["Peso_Total"] = historial["Peso_Temporal"] * historial["Peso_Contexto"]
    pesos = historial["Peso_Total"] / historial["Peso_Total"].sum()

    def prom_w(col): return float(np.average(historial[col].fillna(0), weights=pesos)) if col in historial.columns else 0.0

    lam_g_raw = prom_w("Goles")
    lam_t = prom_w("Tiros")
    lam_p = prom_w("A Puerta")
    lam_a = prom_w("Asistencias")
    lam_f = prom_w("Faltas")

    # Intervalos Bootstrap de Goles
    goles_vals = historial["Goles"].values if "Goles" in historial.columns else np.array([0])
    _, lam_g_inf, lam_g_sup = bootstrap_lambda_intervalo(goles_vals, pesos.values)

    df_tier_liga = df_liga[df_liga["Nivel Rival"] == nivel_sel] if len(df_liga[df_liga["Nivel Rival"] == nivel_sel]) > 0 else df_liga
    prior_g = float(df_tier_liga["Goles"].mean()) if len(df_tier_liga) and "Goles" in df_tier_liga.columns else lam_g_raw

    if usar_shrinkage:
        lam_g = shrinkage_lambda(lam_g_raw, prior_g, n_obs, k_shrink)
    else:
        lam_g = lam_g_raw

    rng = np.random.default_rng(42)
    sim_goles = rng.poisson(max(lam_g, 0.01), 10000)
    sim_tiros = rng.poisson(max(lam_t, 0.01), 10000)
    sim_puerta = rng.poisson(max(lam_p, 0.01), 10000)
    sim_asist = rng.poisson(max(lam_a, 0.01), 10000)
    sim_faltas = rng.poisson(max(lam_f, 0.01), 10000)
    sim_contrib = sim_goles + sim_asist

    prob_goles_base = (sim_goles > linea_goles).mean() * 100
    prob_tiros_base = (sim_tiros > linea_tiros).mean() * 100
    prob_puerta_base = (sim_puerta > linea_puerta).mean() * 100
    prob_asist_base = (sim_asist > linea_asist).mean() * 100
    prob_contrib_base = (sim_contrib > linea_contrib).mean() * 100

    features_modelo = ["Goles_Media_Movil_5", "Goles_Volatilidad_5", "Tiros_Media_Movil_5", "Conversion_Tiros", "Momentum_Goles"]
    modelo_xgb_global = entrenar_predictor_xgboost_jugadores(df, features_modelo)
    
    prob_goles, modo_hibrido = predecir_probabilidad_hibrida_dinamica(prob_goles_base, historial, features_modelo, modelo_xgb_global, n_obs)
    prob_contrib, _ = predecir_probabilidad_hibrida_dinamica(prob_contrib_base, historial, features_modelo, modelo_xgb_global, n_obs)
    prob_tiros, prob_puerta, prob_asist = prob_tiros_base, prob_puerta_base, prob_asist_base

    analisis_texto = generar_analisis_dinamico_jugador(jugador_sel, condicion_sel, nivel_sel, n_obs, lam_g, lam_t, lam_p, prob_goles, prob_puerta, prob_contrib)
    st.markdown(f'<div class="analisis-dinamico-box">{analisis_texto}<br><small>Motor activo: {modo_hibrido}</small></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard Principal y Gráficos",
        "💰 Value Bets & Inteligencia",
        "📋 Auditoría y Datos"
    ])

    with tab1:
        st.subheader("Métricas y Tasas con Intervalos Bootstrap")
        a, b, c = st.columns(3)
        a.metric("Goles Esperados (λ)", f"{lam_g:.2f}", f"IC 95%: [{lam_g_inf:.2f} - {lam_g_sup:.2f}]")
        b.metric("Tiros Totales (λ)", f"{lam_t:.2f}")
        c.metric("Tiros a Puerta (λ)", f"{lam_p:.2f}")

        st.markdown("---")
        st.subheader("🕸️ Perfil de Atributos (Radar)")
        categories = ['Goles', 'Tiros', 'A Puerta', 'Asistencias', 'Faltas']
        vals = [min(lam_g*3, 10), min(lam_t, 10), min(lam_p*2, 10), min(lam_a*5, 10), min(lam_f*2, 10)]
        fig_radar = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]], fill='toself', marker=dict(color='#3B82F6'), line=dict(color='#60A5FA', width=2)))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], color="#9ca3af"), bgcolor="#111827"), showlegend=False, paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F3F4F6"), height=300, margin=dict(l=20, r=20, t=10, b=10))
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("💰 Value Bet Props (Overs & Unders con Half-Kelly)")
        items_value = [
            (f"Over {linea_goles} Goles", prob_goles, cuota_over_goles),
            (f"Under {linea_goles} Goles", 100.0 - prob_goles, cuota_under_goles),
            (f"Over {linea_tiros} Tiros", prob_tiros, cuota_over_tiros),
            (f"Over {linea_puerta} a Puerta", prob_puerta, cuota_over_puerta),
            (f"Over {linea_asist} Asistencias", prob_asist, cuota_over_asist),
            (f"Over {linea_contrib} Gol/Asist", prob_contrib, cuota_over_contrib),
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
        st.subheader("📋 Auditoría de Partidos Filtrados")
        h_mostrar = historial.copy()
        if "Fecha" in h_mostrar.columns: h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
        cols_mostrar = [c for c in ["Fecha", "Condición", "Rival", "Nivel Rival", "Goles", "Asistencias", "Tiros", "A Puerta", "Tipo_Uso", "Factor_Ajuste"] if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols_mostrar], hide_index=True, use_container_width=True)
else:
    st.info("Configura las opciones en la barra lateral y haz clic en Analizar.")
