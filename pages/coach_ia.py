import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client

st.set_page_config(page_title="Coach | GoalMetrics", page_icon="🤖", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_supabase()
user = st.session_state.get("user")

if not user:
    st.warning("Por favor inicia sesión en la página principal para ver tu Coach.")
    st.stop()

user_id = user.id

st.markdown("## 🤖 Coach de Rendimiento")
st.caption("Análisis de tu historial de apuestas · Winrate · ROI · Rachas · Mercados")

# ====================== CARGA ======================
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
    df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
except Exception as e:
    st.error(f"Error al cargar apuestas: {e}")
    df = pd.DataFrame()

if df.empty:
    st.info("Aún no tienes datos. Registra apuestas en el Tracker para activar el Coach.")
    st.stop()

# Validar columnas mínimas
columnas_necesarias = ["estado", "pnl", "mercado"]
faltantes = [c for c in columnas_necesarias if c not in df.columns]
if faltantes:
    st.error(f"Faltan columnas en la tabla de apuestas: {faltantes}")
    st.stop()

df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
if "stake" in df.columns:
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0)
else:
    df["stake"] = 0.0

df_cerradas = df[df["estado"].isin(["Ganada", "Perdida"])].copy()

if df_cerradas.empty:
    st.warning("Tienes apuestas, pero ninguna cerrada (Ganada/Perdida). Cierra al menos una para activar el análisis.")
    st.stop()

# ====================== MÉTRICAS GLOBALES ======================
total = len(df_cerradas)
ganadas = len(df_cerradas[df_cerradas["estado"] == "Ganada"])
perdidas = total - ganadas
winrate = (ganadas / total) * 100 if total > 0 else 0

beneficio = df_cerradas["pnl"].sum()
stake_total = df_cerradas["stake"].sum()
roi = (beneficio / stake_total * 100) if stake_total > 0 else 0
stake_medio = df_cerradas["stake"].mean() if total > 0 else 0

# ====================== RACHA ACTUAL ======================
# Ordenamos por fecha si existe; si no, por el orden de la tabla
if "fecha" in df_cerradas.columns:
    df_ord = df_cerradas.sort_values("fecha", ascending=False)
elif "created_at" in df_cerradas.columns:
    df_ord = df_cerradas.sort_values("created_at", ascending=False)
else:
    df_ord = df_cerradas.iloc[::-1]

racha = 0
tipo_racha = None
for _, row in df_ord.iterrows():
    estado = row["estado"]
    if tipo_racha is None:
        tipo_racha = estado
        racha = 1
    elif estado == tipo_racha:
        racha += 1
    else:
        break

# ====================== POR MERCADO ======================
MIN_APUESTAS_MERCADO = 5  # mínimo para recomendar

agrupado = df_cerradas.groupby("mercado").agg(
    apuestas=("pnl", "count"),
    ganadas=("estado", lambda x: (x == "Ganada").sum()),
    pnl=("pnl", "sum"),
    stake=("stake", "sum"),
).reset_index()

agrupado["winrate"] = (agrupado["ganadas"] / agrupado["apuestas"] * 100).round(1)
agrupado["roi"] = np.where(
    agrupado["stake"] > 0,
    (agrupado["pnl"] / agrupado["stake"] * 100).round(1),
    0.0,
)

# Solo mercados con volumen suficiente para “mejor/peor”
mercados_validos = agrupado[agrupado["apuestas"] >= MIN_APUESTAS_MERCADO].copy()

if not mercados_validos.empty:
    mejor = mercados_validos.loc[mercados_validos["roi"].idxmax()]
    peor = mercados_validos.loc[mercados_validos["roi"].idxmin()]
    mejor_mercado = mejor["mercado"]
    peor_mercado = peor["mercado"]
    mejor_roi = mejor["roi"]
    peor_roi = peor["roi"]
else:
    # Fallback: usar P&L si no hay volumen
    mejor_mercado = agrupado.loc[agrupado["pnl"].idxmax()]["mercado"] if not agrupado.empty else "N/A"
    peor_mercado = agrupado.loc[agrupado["pnl"].idxmin()]["mercado"] if not agrupado.empty else "N/A"
    mejor_roi = None
    peor_roi = None

# ====================== UI MÉTRICAS ======================
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Winrate", f"{winrate:.1f}%", delta=f"{ganadas}G / {perdidas}P")
c2.metric("💰 P&L Total", f"{beneficio:+,.2f}")
c3.metric("📊 ROI", f"{roi:+.1f}%")
c4.metric("🎲 Stake medio", f"{stake_medio:.2f}")

st.markdown("---")

# Racha
col_r1, col_r2 = st.columns(2)
with col_r1:
    if tipo_racha == "Ganada":
        st.success(f"🔥 **Racha actual:** {racha} ganada(s) seguidas")
    elif tipo_racha == "Perdida":
        st.error(f"❄️ **Racha actual:** {racha} pérdida(s) seguidas")
    else:
        st.info("Sin racha definida")

with col_r2:
    st.metric("Apuestas cerradas", total)

st.markdown("---")
st.subheader("🧠 Diagnóstico del Coach")

consejos = []

# Winrate
if total < 10:
    consejos.append("ℹ️ **Poca muestra:** Tienes menos de 10 apuestas cerradas. Las conclusiones aún son orientativas.")
elif winrate >= 58:
    consejos.append("🔥 **Buen nivel de acierto.** Tu winrate está en zona sólida. Mantén la disciplina de stakes.")
elif winrate < 45:
    consejos.append("⚠️ **Winrate bajo (<45%).** Reduce el stake hasta estabilizar resultados y revisa si estás forzando apuestas.")

# ROI
if stake_total > 0:
    if roi >= 8:
        consejos.append(f"📈 **ROI positivo fuerte ({roi:+.1f}%).** El sistema te está dejando margen. No subas el riesgo de golpe.")
    elif roi <= -10:
        consejos.append(f"📉 **ROI negativo ({roi:+.1f}%).** Momento de bajar volumen y volver a apuestas con valor claro (EV+).")

# Racha
if tipo_racha == "Perdida" and racha >= 3:
    consejos.append(f"🛑 **{racha} pérdidas seguidas.** Baja temporalmente el stake (\~30%) hasta cortar la racha.")
elif tipo_racha == "Ganada" and racha >= 4:
    consejos.append(f"✅ **{racha} ganadas seguidas.** Bien, pero no aumentes el stake por euforia.")

# Mercados
if mejor_roi is not None:
    consejos.append(
        f"🏆 **Mejor mercado:** **{mejor_mercado}** (ROI {mejor_roi:+.1f}%, ≥{MIN_APUESTAS_MERCADO} apuestas). Prioriza análisis ahí."
    )
    consejos.append(
        f"🚫 **Peor mercado:** **{peor_mercado}** (ROI {peor_roi:+.1f}%). Reduce o evita hasta revisar tu criterio."
    )
else:
    consejos.append(
        f"📌 Aún no hay mercados con ≥{MIN_APUESTAS_MERCADO} apuestas. Por P&L, el más rentable es **{mejor_mercado}** y el más débil **{peor_mercado}**."
    )

# Stake inconsistente (opcional)
if stake_medio > 0 and df_cerradas["stake"].std() > stake_medio * 1.5:
    consejos.append("⚖️ **Stakes muy variables.** Intenta unificar el tamaño de apuesta (o usar % fijo del bank) para medir mejor el edge.")

for c in consejos:
    st.info(c)

# ====================== TABLA MERCADOS ======================
st.markdown("---")
st.subheader("📊 Rendimiento por Mercado")

tabla = agrupado[["mercado", "apuestas", "ganadas", "winrate", "pnl", "roi"]].copy()
tabla = tabla.sort_values("roi", ascending=False)
tabla.columns = ["Mercado", "Apuestas", "Ganadas", "Winrate %", "P&L", "ROI %"]

st.dataframe(
    tabla.style.format({
        "Winrate %": "{:.1f}",
        "P&L": "{:+.2f}",
        "ROI %": "{:+.1f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Se recomienda un mercado solo si tiene al menos **{MIN_APUESTAS_MERCADO} apuestas** cerradas.")
