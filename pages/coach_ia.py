import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Coach | GoalMetrics", page_icon="🤖", layout="wide")

# ----------------------------------------------------------------------
# Estilos CSS de Élite y Animaciones (SaaS Theme)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
}

/* ================== KEYFRAMES ANIMACIONES ================== */
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
    0% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); }
    50% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.5); border-color: rgba(16, 185, 129, 0.8); }
    100% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); }
}
/* =========================================================== */

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
    background: linear-gradient(135deg, #8B5CF6 0%, #0f172a 100%);
    padding: 24px 30px; 
    border-radius: 16px; 
    color: white;
    font-weight: 700; 
    font-size: 24px; 
    margin-bottom: 25px; 
    text-align: center;
    box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.15);
    animation: fadeInUp 0.5s ease-out forwards;
}

.veredicto-box {
    padding: 18px 22px; border-radius: 14px; background-color: #111827;
    border: 1px solid #1f2937; border-left: 5px solid #8B5CF6; margin-bottom: 20px; font-size: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    animation: fadeInUp 0.7s ease-out forwards;
    transition: transform 0.3s ease;
}
.veredicto-box:hover {
    transform: translateY(-2px);
}

/* Animación en cascada para los KPIs */
div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 16px 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease-in-out;
    margin-bottom: 12px;
    opacity: 0;
    animation: fadeInUp 0.5s ease-out forwards;
}
div[data-testid="stMetric"]:nth-child(1) { animation-delay: 0.1s; }
div[data-testid="stMetric"]:nth-child(2) { animation-delay: 0.2s; }
div[data-testid="stMetric"]:nth-child(3) { animation-delay: 0.3s; }
div[data-testid="stMetric"]:nth-child(4) { animation-delay: 0.4s; }

div[data-testid="stMetric"]:hover {
    border-color: #374151;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    transform: translateY(-2px) scale(1.02);
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

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f2937;
    animation: fadeInUp 0.8s ease-out forwards;
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

# ----------------------------------------------------------------------
# Cliente de Supabase
# ----------------------------------------------------------------------
user = st.session_state.get("user")
if not user:
    st.warning("Por favor inicia sesión en la página principal para ver tu Coach.")
    st.stop()

if "supabase_client" not in st.session_state:
    st.warning("No se encontró la conexión a la base de datos. Vuelve a la página principal.")
    st.stop()

supabase = st.session_state.supabase_client
user_id = user.id

st.markdown(
    '<div class="header-box">🤖 Coach de Rendimiento</div>',
    unsafe_allow_html=True
)
st.caption("Winrate · ROI · Break-even · Rachas · Últimos 30 días")

# ====================== CARGA ======================
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
    raw = response.data if response.data else []
except Exception as e:
    st.error(f"Error al cargar apuestas: {e}")
    st.stop()

if len(raw) == 0:
    st.warning("No hay apuestas registradas todavía. Ve al Tracker para añadir tu primera apuesta.")
    st.stop()

df = pd.DataFrame(raw)

# Renombres flexibles
cols_lower = {c.lower(): c for c in df.columns}
rename_map = {}
for target, options in {
    "estado": ["estado", "status", "result", "resultado"],
    "pnl": ["pnl", "profit", "beneficio", "ganancia"],
    "mercado": ["mercado", "market", "tipo"],
    "stake": ["stake", "importe", "monto"],
    "cuota": ["cuota", "odds", "odd"],
    "fecha": ["fecha", "date"],
}.items():
    if target not in df.columns:
        for opt in options:
            if opt in cols_lower:
                rename_map[cols_lower[opt]] = target
                break
if rename_map:
    df = df.rename(columns=rename_map)

if "estado" not in df.columns or "pnl" not in df.columns:
    st.error("Faltan columnas obligatorias en la base de datos: estado y/o pnl.")
    st.stop()

if "mercado" not in df.columns:
    df["mercado"] = "Sin mercado"
if "stake" not in df.columns:
    df["stake"] = 0.0
if "cuota" not in df.columns:
    df["cuota"] = np.nan

df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
df["cuota"] = pd.to_numeric(df["cuota"], errors="coerce")
df["estado"] = df["estado"].astype(str).str.strip()

def normalizar_estado(x: str) -> str:
    x = x.lower().strip()
    if x in ("ganada", "ganado", "win", "won", "g"):
        return "Ganada"
    if x in ("perdida", "perdido", "loss", "lost", "l"):
        return "Perdida"
    if x in ("nulo", "void", "push", "empate", "cancelada"):
        return "Nulo"
    if x in ("pendiente", "pending", "open", "abierta"):
        return "Pendiente"
    return x.capitalize()

df["estado"] = df["estado"].apply(normalizar_estado)

# Fecha
if "fecha" in df.columns:
    df["_fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
elif "created_at" in df.columns:
    df["_fecha"] = pd.to_datetime(df["created_at"], errors="coerce")
else:
    df["_fecha"] = pd.NaT

df_decididas = df[df["estado"].isin(["Ganada", "Perdida"])].copy()
df_bank = df[df["estado"].isin(["Ganada", "Perdida", "Nulo"])].copy()

if df_decididas.empty:
    st.warning("Ninguna apuesta cerrada (Ganada/Perdida) todavía. Cierra al menos una en el Tracker.")
    pendientes_count = len(df[df["estado"] == "Pendiente"])
    st.info(f"Apuestas pendientes: {pendientes_count}")
    st.stop()

# ====================== MÉTRICAS GLOBALES ======================
total = len(df_decididas)
ganadas = len(df_decididas[df_decididas["estado"] == "Ganada"])
perdidas = total - ganadas
winrate = ganadas / total * 100

beneficio = df_bank["pnl"].sum()
stake_total = df_bank["stake"].sum()
roi = (beneficio / stake_total * 100) if stake_total > 0 else 0.0
stake_medio = df_decididas["stake"].mean()

# Cuota media y break-even
cuotas_validas = df_decididas["cuota"].dropna()
cuota_media = float(cuotas_validas.mean()) if len(cuotas_validas) else np.nan
break_even = (100 / cuota_media) if cuota_media and cuota_media > 1 else np.nan
edge_vs_be = (winrate - break_even) if not np.isnan(break_even) else np.nan

# ====================== ÚLTIMOS 30 DÍAS ======================
hace_30 = pd.Timestamp.now().normalize() - timedelta(days=30)
if df_decididas["_fecha"].notna().any():
    mask_30 = df_decididas["_fecha"] >= hace_30
    df_30 = df_decididas[mask_30].copy()
else:
    df_30 = df_decididas.copy()

total_30 = len(df_30)
if total_30 > 0:
    ganadas_30 = len(df_30[df_30["estado"] == "Ganada"])
    winrate_30 = ganadas_30 / total_30 * 100
    stake_30 = df_30["stake"].sum()
    pnl_30 = df_30["pnl"].sum()
    roi_30 = (pnl_30 / stake_30 * 100) if stake_30 > 0 else 0.0
else:
    ganadas_30 = winrate_30 = pnl_30 = roi_30 = 0.0

# ====================== RACHA ======================
df_ord = df_decididas.sort_values("_fecha", ascending=False) if df_decididas["_fecha"].notna().any() else df_decididas.iloc[::-1]
racha, tipo_racha = 0, None
for _, row in df_ord.iterrows():
    if tipo_racha is None:
        tipo_racha = row["estado"]
        racha = 1
    elif row["estado"] == tipo_racha:
        racha += 1
    else:
        break

# ====================== MERCADOS ======================
MIN_AP = 5
agrupado = df_decididas.groupby("mercado").agg(
    apuestas=("pnl", "count"),
    ganadas=("estado", lambda x: (x == "Ganada").sum()),
    pnl=("pnl", "sum"),
    stake=("stake", "sum"),
).reset_index()
agrupado["winrate"] = (agrupado["ganadas"] / agrupado["apuestas"] * 100).round(1)
agrupado["roi"] = np.where(agrupado["stake"] > 0, (agrupado["pnl"] / agrupado["stake"] * 100).round(1), 0.0)

validos = agrupado[agrupado["apuestas"] >= MIN_AP]
if len(validos):
    mejor = validos.loc[validos["roi"].idxmax()]
    peor = validos.loc[validos["roi"].idxmin()]
    mejor_m, mejor_roi = mejor["mercado"], mejor["roi"]
    peor_m, peor_roi = peor["mercado"], peor["roi"]
else:
    mejor_m = agrupado.loc[agrupado["pnl"].idxmax()]["mercado"] if len(agrupado) else "N/A"
    peor_m = agrupado.loc[agrupado["pnl"].idxmin()]["mercado"] if len(agrupado) else "N/A"
    mejor_roi = peor_roi = None

# ====================== CONSEJO PRIORITARIO ======================
def consejo_prioritario():
    if total < 8:
        return "ℹ️ **Prioridad:** acumula al menos 10–15 apuestas cerradas antes de cambiar de estrategia. La muestra aún es pequeña."
    if tipo_racha == "Perdida" and racha >= 3:
        return f"🛑 **Prioridad:** llevas **{racha} pérdidas seguidas**. Baja el stake ~30% esta semana y solo apuesta value claro."
    if not np.isnan(edge_vs_be) and edge_vs_be < -5:
        return f"⚠️ **Prioridad:** tu winrate ({winrate:.0f}%) está por debajo del break-even ({break_even:.0f}% con cuota media {cuota_media:.2f}). Reduce volumen o sube la exigencia de value."
    if roi_30 <= -15 and total_30 >= 5:
        return f"📉 **Prioridad:** en los últimos 30 días el ROI va en **{roi_30:+.1f}%**. Pausa mercados flojos y enfoca solo donde tengas edge."
    if mejor_roi is not None and mejor_roi >= 5:
        return f"🏆 **Prioridad:** tu mejor mercado es **{mejor_m}** (ROI {mejor_roi:+.1f}%). Prioriza análisis y stake ahí."
    if roi >= 8 and winrate >= 55:
        return f"✅ **Prioridad:** vas bien (ROI {roi:+.1f}%, WR {winrate:.0f}%). Mantén stakes estables; no subas por euforia."
    if peor_roi is not None and peor_roi <= -10:
        return f"🚫 **Prioridad:** **{peor_m}** te resta (ROI {peor_roi:+.1f}%). Corta o reduce fuerte ese mercado."
    return "📌 **Prioridad:** cierra pendientes, registra bien cuota/stake y apuesta solo cuando el modelo marque value (EV+)."

# ====================== UI ======================
st.markdown("---")
st.markdown(f'<div class="veredicto-box">{consejo_prioritario()}</div>', unsafe_allow_html=True)

st.markdown("### 📊 Resumen global")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Winrate", f"{winrate:.1f}%", delta=f"{ganadas}G / {perdidas}P")
c2.metric("P&L", f"{beneficio:+,.2f}")
c3.metric("ROI", f"{roi:+.1f}%")
c4.metric("Stake medio", f"{stake_medio:.2f}")

c5, c6, c7 = st.columns(3)
if not np.isnan(cuota_media):
    c5.metric("Cuota media", f"{cuota_media:.2f}")
    c6.metric("Break-even WR", f"{break_even:.1f}%")
    delta_be = f"{edge_vs_be:+.1f} pp" if not np.isnan(edge_vs_be) else None
    c7.metric("WR vs break-even", f"{winrate:.1f}%", delta=delta_be)
else:
    c5.metric("Cuota media", "N/D")
    c6.metric("Break-even WR", "N/D")
    c7.metric("WR vs break-even", "N/D")

st.markdown("### 📅 Últimos 30 días")
if not df_decididas["_fecha"].notna().any():
    st.caption("No hay fechas válidas: se muestran métricas globales como referencia.")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Apuestas", total_30)
d2.metric("Winrate 30d", f"{winrate_30:.1f}%")
d3.metric("P&L 30d", f"{pnl_30:+,.2f}")
d4.metric("ROI 30d", f"{roi_30:+.1f}%")

st.markdown("---")
r1, r2 = st.columns(2)
with r1:
    if tipo_racha == "Ganada":
        st.success(f"🔥 Racha: **{racha}** ganada(s)")
    elif tipo_racha == "Perdida":
        st.error(f"❄️ Racha: **{racha}** pérdida(s)")
    else:
        st.info("Sin racha")
with r2:
    st.metric("Cerradas G/P", total)

# Consejos secundarios
st.markdown("### 🧠 Más diagnósticos")
extras = []
if total < 10:
    extras.append("ℹ️ Muestra pequeña: no cambies el sistema por 2–3 resultados.")
if tipo_racha == "Ganada" and racha >= 4:
    extras.append("✅ Buena racha: no aumentes stake por euforia.")
if mejor_roi is not None:
    extras.append(f"🏆 Mejor mercado (≥{MIN_AP}): **{mejor_m}** · ROI {mejor_roi:+.1f}%")
    extras.append(f"🚫 Peor mercado (≥{MIN_AP}): **{peor_m}** · ROI {peor_roi:+.1f}%")
else:
    extras.append(f"📌 Sin mercados con ≥{MIN_AP} apuestas. Por P&L: mejor **{mejor_m}**, peor **{peor_m}**.")
if stake_medio > 0 and df_decididas["stake"].std() > stake_medio * 1.5:
    extras.append("⚖️ Stakes muy variables: unifica tamaño o usa % fijo del bank.")

for e in extras:
    st.info(e)

st.markdown("---")
st.subheader("📊 Por mercado")
tabla = agrupado[["mercado", "apuestas", "ganadas", "winrate", "pnl", "roi"]].sort_values("roi", ascending=False)
tabla.columns = ["Mercado", "Apuestas", "Ganadas", "Winrate %", "P&L", "ROI %"]
st.dataframe(
    tabla.style.format({"Winrate %": "{:.1f}", "P&L": "{:+.2f}", "ROI %": "{:+.1f}"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Recomendación de mercado solo con ≥ {MIN_AP} apuestas cerradas.")

if st.button("🔄 Actualizar datos"):
    st.rerun()
