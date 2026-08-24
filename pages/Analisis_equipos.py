import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
import hashlib
import colorsys

@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=["Equipo", "Fecha", "Condición", "Nivel Rival"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df["Equipo"] = df["Equipo"].astype(str).str.strip()
    df["Condición"] = df["Condición"].astype(str).str.strip().str.lower()
    df["Nivel Rival"] = df["Nivel Rival"].astype(str).str.strip()
    for col in ["Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def shrinkage_lambda(lam_obs, lam_prior, n_obs, k=5.0):
    n = max(float(n_obs), 0.0)
    return (n * lam_obs + k * lam_prior) / (n + k)

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
    df = cargar_datos()
except Exception as e:
    st.error(f"⚠️ Error al cargar los datos: {e}")
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

st.sidebar.header("⚙️ Configuración")

with st.sidebar.expander("🏟️ Partido", expanded=True):
    lista_equipos = sorted([str(x) for x in df["Equipo"].unique() if pd.notna(x)])
    equipo_sel = st.selectbox("Equipo", lista_equipos)
    df_equipo = df[df["Equipo"] == equipo_sel]
    lista_niveles = sorted([str(x) for x in df_equipo["Nivel Rival"].unique() if pd.notna(x)])
    condicion_label = st.selectbox("Condición", ["Local", "Visitante"])
    condicion_sel = condicion_label.lower()
    nivel_sel = st.selectbox("Nivel del Rival", lista_niveles)

df_diagnostico = df_equipo.sort_values(by="Fecha", ascending=False)
exactos_check = df_diagnostico[
    (df_diagnostico["Condición"] == condicion_sel) & (df_diagnostico["Nivel Rival"] == nivel_sel)
]
num_exactos = len(exactos_check)
if num_exactos >= 2:
    st.sidebar.success(f"✅ {num_exactos} partidos exactos")
elif num_exactos == 1:
    st.sidebar.warning("⚠️ 1 partido → se usará respaldo")
else:
    st.sidebar.error("❌ 0 partidos exactos")

with st.sidebar.expander("🎯 Líneas de Estudio"):
    linea_goles = st.slider("Goles (equipo)", 0.5, 3.5, 1.5, 0.5)
    linea_tiros = st.slider("Tiros Totales", 5.0, 25.0, 12.5, 0.5)
    linea_tiros_puerta = st.slider("Tiros a Puerta", 1.0, 10.0, 4.5, 0.5)
    linea_corners = st.slider("Córners", 1.0, 15.0, 5.5, 0.5)
    linea_faltas = st.slider("Faltas", 5.0, 25.0, 10.5, 0.5)
    linea_total_partido = st.slider("Total goles partido (Over)", 0.5, 4.5, 2.5, 0.5)

with st.sidebar.expander("💰 Cuotas 1X2 / BTTS / DNB"):
    cuota_casa_1 = st.number_input("Victoria (1)", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_casa_x = st.number_input("Empate (X)", min_value=1.01, value=3.40, step=0.01, format="%.2f")
    cuota_casa_2 = st.number_input("Derrota (2)", min_value=1.01, value=4.20, step=0.01, format="%.2f")
    cuota_casa_btts_si = st.number_input("BTTS Sí", min_value=1.01, value=1.75, step=0.01, format="%.2f")
    cuota_casa_btts_no = st.number_input("BTTS No", min_value=1.01, value=2.05, step=0.01, format="%.2f")
    cuota_casa_dnb = st.number_input("DNB", min_value=1.01, value=1.35, step=0.01, format="%.2f")

with st.sidebar.expander("📈 Cuotas de Líneas (Over)"):
    cuota_over_goles = st.number_input(f"Over {linea_goles} Goles", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_tiros = st.number_input(f"Over {linea_tiros} Tiros", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_over_puerta = st.number_input(f"Over {linea_tiros_puerta} a Puerta", min_value=1.01, value=1.80, step=0.01, format="%.2f")
    cuota_over_corners = st.number_input(f"Over {linea_corners} Córners", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    cuota_over_faltas = st.number_input(f"Over {linea_faltas} Faltas", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cuota_over_total = st.number_input(f"Over {linea_total_partido} Goles partido", min_value=1.01, value=1.90, step=0.01, format="%.2f")

with st.sidebar.expander("🧪 Modelo estadístico"):
    usar_shrinkage = st.checkbox("Shrinkage hacia media del nivel", value=True)
    k_shrink = st.slider("Fuerza prior shrinkage (k)", 1.0, 15.0, 5.0, 1.0)
    usar_dc = st.checkbox("Dixon–Coles (corrige empates bajos)", value=True)
    rho_dc = st.slider("ρ Dixon–Coles", -0.20, 0.05, -0.10, 0.01)

color_equipo = generar_color_equipo(equipo_sel)

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
div[data-testid="stMetric"] {{ background-color: #1F2937; padding: 12px 16px; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

def renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa):
    df_adn = pd.DataFrame({
        "Métrica": ["Ataque", "Volumen Tiros", "Precisión", "Córners", "Disciplina"],
        "Puntuación": [
            min(round(lam_f * 3.33, 1), 10.0),
            min(round(lam_t / 2.5, 1), 10.0),
            min(round(lam_tp * 1.66, 1), 10.0),
            min(round(lam_co / 1.5, 1), 10.0),
            min(round((25 - lam_fa) / 2.5, 1), 10.0),
        ],
    })
    chart = alt.Chart(df_adn).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X("Puntuación:Q", scale=alt.Scale(domain=[0, 10]), title=None),
        y=alt.Y("Métrica:N", sort="-x", title=None),
        color=alt.value(color_equipo),
        tooltip=["Métrica", "Puntuación"],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

def crear_grafico(serie, titulo):
    serie = pd.Series(serie).dropna().astype(int)
    if len(serie) == 0:
        return None
    conteo = serie.value_counts().sort_index()
    df_c = pd.DataFrame({titulo: conteo.index.astype(str), "Prob (%)": (conteo / len(serie) * 100).round(1)})
    return alt.Chart(df_c).mark_bar(color=color_equipo).encode(
        x=alt.X(f"{titulo}:N", sort=None, title=titulo),
        y=alt.Y("Prob (%):Q", title="Probabilidad (%)"),
        tooltip=[f"{titulo}:N", "Prob (%):Q"],
    ).properties(height=280)

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
    kelly_txt = f" &nbsp;|&nbsp; Half-Kelly: <b>{stake}% bank</b>" if es_value else ""
    caution = " <span style='color:#f59e0b;'>(muestra pequeña)</span>" if muestra_pequena and es_value else ""
    st.markdown(f"""
    <div class="value-box {clase}">
        <b>{nombre}</b>{caution}<br>
        Prob: <b>{prob:.1f}%</b> &nbsp;|&nbsp; Justa: <b>{cuota_justa}</b> &nbsp;|&nbsp; Casa: <b>{cuota_casa}</b>{kelly_txt}<br>
        <span style="color:{color_ev}; font-weight:bold; font-size:15px;">EV: {ev:+.2%} → {'✅ VALUE' if es_value else '❌ Sin valor'}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("### 📊 GoalMetrics · Análisis de Equipos")
st.caption("Simulación orientativa · EV y Kelly no son tips garantizados")

with st.expander("📖 Cómo interpretar este análisis", expanded=False):
    st.markdown("""
### Qué estás viendo
El modelo estima **goles y estadísticas** con el historial filtrado (equipo + local/visitante + nivel de rival),
da más peso a partidos recientes, estabiliza con **shrinkage** si hay pocos datos y corrige empates bajos con **Dixon–Coles**.

### Resumen
| Concepto | Significado |
|----------|-------------|
| **Victoria / Empate / Derrota** | % de simulaciones con ese resultado |
| **BTTS** | % en los que ambos marcan |
| **1X / X2 / DNB** | Doble oportunidad y empate no válido |
| **λ** | Promedio esperado por partido |
| **vs promedio del nivel** | Por encima o debajo de la media de ese nivel |
| **ADN** | Perfil 0–10 orientativo |

### Value Bet
| Concepto | Significado |
|----------|-------------|
| **Prob** | Probabilidad del modelo |
| **Cuota justa** | `100 / Prob` |
| **Cuota casa** | Lo que pagas en el bookie |
| **EV > 0** | Posible value (la casa paga de más según el modelo) |
| **Half-Kelly** | % sugerido del bank (conservador) |
| **Muestra pequeña** | ≤3 partidos → interpreta con cautela |

### Líneas (Over)
Probabilidad de **superar** la línea elegida en el sidebar.

### Modelo (sidebar)
- **Shrinkage:** acerca medias a la del nivel con pocos datos
- **Dixon–Coles:** corrige subestimación de 0-0 y 1-1
- **k más alto:** más peso al nivel
- **ρ más negativo:** más corrección a empates bajos


col_btn1, col_btn2, _ = st.columns([1.2, 1, 4])
with col_btn1:
    btn_analizar = st.button("⚡ Analizar", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🧹 Limpiar", use_container_width=True):
        st.rerun()

if btn_analizar:
    df_base = df[df["Equipo"] == equipo_sel].sort_values(by="Fecha", ascending=False)
    df_exactos = df_base[(df_base["Condición"] == condicion_sel) & (df_base["Nivel Rival"] == nivel_sel)]
    historial = pd.DataFrame()
    fuente_datos = ""

    if len(df_exactos) >= 2:
        historial = df_exactos.copy()
        fuente_datos = f"Exacto ({condicion_label} vs {nivel_sel}) — {len(historial)} partidos"
    elif len(df_exactos) == 1:
        partido_1 = df_exactos.head(1).copy()
        cond_opuesta = "visitante" if condicion_sel == "local" else "local"
        df_opuestos = df_base[df_base["Condición"] == cond_opuesta]
        if len(df_opuestos) >= 1:
            partido_2 = df_opuestos.head(1).copy()
            factor = 0.88 if condicion_sel == "visitante" else 1.12
            for col in ["Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas"]:
                if col in partido_2.columns:
                    partido_2[col] = (partido_2[col] * factor).round(2)
            historial = pd.concat([partido_1, partido_2], ignore_index=True)
            fuente_datos = f"Mixto con respaldo ({cond_opuesta.capitalize()} ajustado)"
        else:
            historial = partido_1.copy()
            fuente_datos = "Solo 1 partido exacto disponible"
    else:
        st.error(f"❌ No hay datos suficientes para analizar a {equipo_sel}.")
        st.stop()

    muestra_pequena = len(historial) <= 3
    n_obs = len(historial)
    hoy = pd.Timestamp.today().normalize()
    historial["Dias_Pasados"] = (hoy - pd.to_datetime(historial["Fecha"])).dt.days.replace(0, 0.1)
    historial["Peso"] = 1 / (1 + (historial["Dias_Pasados"] / 30))

    def prom(col):
        if col not in historial.columns:
            return 0.05
        return round(float(np.average(historial[col].fillna(0), weights=historial["Peso"])), 4)

    lam_f_raw, lam_c_raw = prom("Goles"), prom("Goles Rival")
    lam_t_raw, lam_tp_raw = prom("Tiros"), prom("A Puerta")
    lam_co_raw, lam_fa_raw = prom("Corners"), prom("Faltas")

    df_nivel = df[df["Nivel Rival"] == nivel_sel]
    prior_f = float(df_nivel["Goles"].mean()) if len(df_nivel) else lam_f_raw
    prior_c = float(df_nivel["Goles Rival"].mean()) if len(df_nivel) and "Goles Rival" in df_nivel.columns else lam_c_raw
    prior_t = float(df_nivel["Tiros"].mean()) if len(df_nivel) and "Tiros" in df_nivel.columns else lam_t_raw
    prior_tp = float(df_nivel["A Puerta"].mean()) if len(df_nivel) and "A Puerta" in df_nivel.columns else lam_tp_raw
    prior_co = float(df_nivel["Corners"].mean()) if len(df_nivel) and "Corners" in df_nivel.columns else lam_co_raw
    prior_fa = float(df_nivel["Faltas"].mean()) if len(df_nivel) and "Faltas" in df_nivel.columns else lam_fa_raw

    if usar_shrinkage:
        lam_f = shrinkage_lambda(lam_f_raw, prior_f, n_obs, k_shrink)
        lam_c = shrinkage_lambda(lam_c_raw, prior_c, n_obs, k_shrink)
        lam_t = shrinkage_lambda(lam_t_raw, prior_t, n_obs, k_shrink)
        lam_tp = shrinkage_lambda(lam_tp_raw, prior_tp, n_obs, k_shrink)
        lam_co = shrinkage_lambda(lam_co_raw, prior_co, n_obs, k_shrink)
        lam_fa = shrinkage_lambda(lam_fa_raw, prior_fa, n_obs, k_shrink)
    else:
        lam_f, lam_c, lam_t, lam_tp, lam_co, lam_fa = lam_f_raw, lam_c_raw, lam_t_raw, lam_tp_raw, lam_co_raw, lam_fa_raw

    num_sim = 10000
    if usar_dc:
        sg_fav, sg_con = simular_goles_dixon_coles(lam_f, lam_c, rho=rho_dc, num_sim=num_sim)
    else:
        rng = np.random.default_rng(42)
        sg_fav = rng.poisson(max(lam_f, 0.01), num_sim)
        sg_con = rng.poisson(max(lam_c, 0.01), num_sim)

    s_tir, s_tpuerta, s_corn, s_faltas = simular_stats_poisson(lam_t, lam_tp, lam_co, lam_fa, num_sim=num_sim)

    triunfos = (sg_fav > sg_con).mean() * 100
    empates = (sg_fav == sg_con).mean() * 100
    derrotas = (sg_fav < sg_con).mean() * 100
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
        veredicto = f"Tendencia Fuerte · Marcador proyectado {marcador_mas_comun}"
    elif derrotas > 50:
        veredicto = f"Alerta de Complicación · Marcador proyectado {marcador_mas_comun}"
    else:
        veredicto = f"Partido Muy Parejo · Marcador proyectado {marcador_mas_comun}"

    st.markdown(f'<div class="header-box">🛡️ {equipo_sel.upper()} · {condicion_label.upper()} vs {nivel_sel.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="veredicto-box"><b>Veredicto:</b> {veredicto}</div>', unsafe_allow_html=True)
    st.caption(f"Base: {n_obs} partidos · {fuente_datos}")
    mods = []
    if usar_shrinkage:
        mods.append(f"Shrinkage k={k_shrink:.0f}")
    if usar_dc:
        mods.append(f"Dixon–Coles ρ={rho_dc:.2f}")
    st.caption("Modelo: " + (" + ".join(mods) if mods else "Poisson clásico"))
    if muestra_pequena:
        st.warning("⚠️ Muestra pequeña (≤3). Interpreta EV y Kelly con cautela.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumen", "💎 Value Bet", "📈 Líneas & Gráficos", "🔍 Detalle"])

    with tab1:
        st.subheader("🧬 ADN del Equipo")
        renderizar_adn_altair(lam_f, lam_t, lam_tp, lam_co, lam_fa)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Victoria", f"{triunfos:.1f}%")
        c2.metric("🟡 Empate", f"{empates:.1f}%")
        c3.metric("🔴 Derrota", f"{derrotas:.1f}%")
        c4.metric("⚽ BTTS", f"{ambos_anotan:.1f}%")
        c5, c6, c7 = st.columns(3)
        c5.metric("1X", f"{doble_1x:.1f}%")
        c6.metric("X2", f"{doble_x2:.1f}%")
        c7.metric("DNB", f"{dnb:.1f}%")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Goles", f"{lam_f:.2f}", delta=f"raw {lam_f_raw:.2f}" if usar_shrinkage else None)
        m2.metric("Goles Rival", f"{lam_c:.2f}", delta=f"raw {lam_c_raw:.2f}" if usar_shrinkage else None)
        m3.metric("Tiros", f"{lam_t:.1f}")
        m4.metric("Córners", f"{lam_co:.1f}")
        v1, v2 = st.columns(2)
        diff_g = ((lam_f / prior_f) - 1) * 100 if prior_f > 0 else 0
        diff_t = ((lam_t / prior_t) - 1) * 100 if prior_t > 0 else 0
        v1.metric("Goles vs nivel", f"{diff_g:+.0f}%")
        v2.metric("Tiros vs nivel", f"{diff_t:+.0f}%")

    with tab2:
        st.subheader("💎 Value Bet & Half-Kelly")
        items_1x2 = [
            ("Victoria (1)", round(100/triunfos,2) if triunfos>0 else 99, cuota_casa_1, calcular_ev(triunfos, cuota_casa_1), triunfos),
            ("Empate (X)", round(100/empates,2) if empates>0 else 99, cuota_casa_x, calcular_ev(empates, cuota_casa_x), empates),
            ("Derrota (2)", round(100/derrotas,2) if derrotas>0 else 99, cuota_casa_2, calcular_ev(derrotas, cuota_casa_2), derrotas),
            ("BTTS Sí", round(100/ambos_anotan,2) if ambos_anotan>0 else 99, cuota_casa_btts_si, calcular_ev(ambos_anotan, cuota_casa_btts_si), ambos_anotan),
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
        st.subheader("📈 Líneas (Over)")
        items_lineas = [
            (f"Over {linea_goles} Goles", round(100/prob_over_goles,2) if prob_over_goles>0 else 99, cuota_over_goles, calcular_ev(prob_over_goles, cuota_over_goles), prob_over_goles),
            (f"Over {linea_tiros} Tiros", round(100/prob_over_tiros,2) if prob_over_tiros>0 else 99, cuota_over_tiros, calcular_ev(prob_over_tiros, cuota_over_tiros), prob_over_tiros),
            (f"Over {linea_tiros_puerta} a Puerta", round(100/prob_over_puerta,2) if prob_over_puerta>0 else 99, cuota_over_puerta, calcular_ev(prob_over_puerta, cuota_over_puerta), prob_over_puerta),
            (f"Over {linea_corners} Córners", round(100/prob_over_corners,2) if prob_over_corners>0 else 99, cuota_over_corners, calcular_ev(prob_over_corners, cuota_over_corners), prob_over_corners),
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
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric(f"Goles > {linea_goles}", f"{prob_over_goles:.1f}%")
        lc2.metric(f"Total > {linea_total_partido}", f"{prob_over_total:.1f}%")
        lc3.metric(f"Tiros > {linea_tiros}", f"{prob_over_tiros:.1f}%")
        lc4, lc5, lc6 = st.columns(3)
        lc4.metric(f"a Puerta > {linea_tiros_puerta}", f"{prob_over_puerta:.1f}%")
        lc5.metric(f"Córners > {linea_corners}", f"{prob_over_corners:.1f}%")
        lc6.metric(f"Faltas > {linea_faltas}", f"{prob_over_faltas:.1f}%")
        g1, g2 = st.columns(2)
        with g1:
            ch = crear_grafico(pd.Series(sg_fav), "Goles")
            if ch: st.altair_chart(ch, use_container_width=True)
        with g2:
            ch2 = crear_grafico(pd.Series(s_corn), "Córners")
            if ch2: st.altair_chart(ch2, use_container_width=True)
        st.dataframe(pd.DataFrame([{"Marcador": r, "Probabilidad": f"{(f/num_sim)*100:.1f}%"} for r, f in conteo.most_common(5)]), hide_index=True, use_container_width=True)

    with tab4:
        h_mostrar = historial.copy().sort_values(by="Fecha", ascending=False)
        h_mostrar["Fecha"] = pd.to_datetime(h_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")
        h_mostrar["Peso"] = h_mostrar["Peso"].round(3)
        cols = [c for c in ["Fecha", "Condición", "Rival", "Nivel Rival", "Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Peso"] if c in h_mostrar.columns]
        st.dataframe(h_mostrar[cols], hide_index=True, use_container_width=True)
        p1, p2, p3 = st.columns(3)
        p1.write(f"**Goles a favor:** {lam_f:.3f} (raw {lam_f_raw:.3f})")
        p1.write(f"**Goles en contra:** {lam_c:.3f} (raw {lam_c_raw:.3f})")
        p2.write(f"**Tiros:** {lam_t:.3f}")
        p2.write(f"**A puerta:** {lam_tp:.3f}")
        p3.write(f"**Córners:** {lam_co:.3f}")
        p3.write(f"**Faltas:** {lam_fa:.3f}")
