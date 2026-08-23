import streamlit as st
import pandas as pd
import datetime
from supabase import create_client, Client

st.set_page_config(page_title="Tracker Pro | GoalMetrics", page_icon="📈", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ESCUDO DE SEGURIDAD
if "user" not in st.session_state or st.session_state.user is None:
    st.warning("Sesión no detectada. Por favor regresa a la página principal e inicia sesión.")
    st.stop()

user = st.session_state.user
user_id = user.id

st.markdown("## 📈 Tracker de Apuestas & Análisis Pro")
st.caption("Registro · Cierre · Bank · ROI · Historial")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Bank en session_state para que no se resetee al interactuar
    if "bank_inicial" not in st.session_state:
        st.session_state.bank_inicial = 100.0
    
    bank_inicial = st.number_input(
        "Bank Inicial ($)",
        min_value=0.0,
        value=float(st.session_state.bank_inicial),
        step=10.0,
        key="input_bank",
    )
    st.session_state.bank_inicial = bank_inicial
    
    st.markdown("---")
    st.header("➕ Nueva Apuesta")
    
    with st.form("nueva_apuesta", clear_on_submit=True):
        evento = st.text_input("Evento / Partido")
        seleccion = st.text_input("Selección (ej: Más de 2.5, BTTS Sí)")
        
        opciones_mercado = [
            "Ganador (1X2)",
            "Doble Oportunidad",
            "Ambos Marcan (BTTS)",
            "Over/Under Goles",
            "Over/Under Córners",
            "Hándicap Asiático",
            "Hándicap Europeo",
            "Tarjetas",
            "Resultado Exacto",
            "Player Props",
            "Otro",
        ]
        mercado = st.selectbox("Mercado", opciones_mercado)
        
        cuota = st.number_input("Cuota", min_value=1.01, value=1.90, step=0.01, format="%.2f")
        stake = st.number_input("Stake ($)", min_value=0.5, value=10.0, step=0.5)
        
        if st.form_submit_button("Guardar Apuesta", type="primary"):
            if not evento.strip() or not seleccion.strip():
                st.error("El evento y la selección son obligatorios.")
            else:
                try:
                    data = {
                        "user_id": user_id,
                        "evento": evento.strip(),
                        "seleccion": seleccion.strip(),
                        "mercado": mercado,
                        "cuota": float(cuota),
                        "stake": float(stake),
                        "estado": "Pendiente",
                        "pnl": 0.0,
                        "fecha": str(datetime.date.today()),
                        "created_at": datetime.datetime.now().isoformat(),
                    }
                    supabase.table("apuestas").insert(data).execute()
                    st.success("¡Apuesta registrada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

# ====================== CARGAR APUESTAS ======================
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
    todas = response.data if response.data else []
except Exception as e:
    todas = []
    st.error(f"Error al conectar con la base de datos: {e}")

df = pd.DataFrame(todas) if todas else pd.DataFrame()

# ====================== CERRAR APUESTAS ======================
st.subheader("🏁 Cerrar Apuesta")

if not df.empty and "estado" in df.columns:
    pendientes = df[df["estado"] == "Pendiente"].copy()
else:
    pendientes = pd.DataFrame()

if not pendientes.empty:
    def label_apuesta(i):
        row = pendientes[pendientes["id"] == i].iloc[0]
        return f"{row.get('evento', '?')} | {row.get('seleccion', '?')} @ {row.get('cuota', '?')}"

    cols = st.columns([3, 1.2, 1.2])
    with cols[0]:
        apuesta_id = st.selectbox(
            "Apuesta pendiente",
            pendientes["id"].tolist(),
            format_func=label_apuesta,
        )
    with cols[1]:
        resultado = st.selectbox("Resultado", ["Ganada", "Perdida", "Nulo"])
    with cols[2]:
        st.write("")
        st.write("")
        if st.button("Actualizar", use_container_width=True, type="primary"):
            try:
                apuesta = pendientes[pendientes["id"] == apuesta_id].iloc[0]
                stake_val = float(apuesta["stake"])
                cuota_val = float(apuesta["cuota"])
                
                if resultado == "Ganada":
                    pnl = stake_val * (cuota_val - 1)
                elif resultado == "Perdida":
                    pnl = -stake_val
                else:  # Nulo / Void
                    pnl = 0.0
                
                supabase.table("apuestas").update({
                    "estado": resultado,
                    "pnl": float(pnl),
                }).eq("id", int(apuesta_id)).eq("user_id", user_id).execute()
                
                st.success("Apuesta actualizada.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al actualizar: {e}")
else:
    st.info("No tienes apuestas pendientes por cerrar.")

# ====================== DASHBOARD ======================
st.markdown("---")

if df.empty:
    st.write("Aún no tienes historial de apuestas registrado.")
    st.stop()

# Normalizar
df["pnl"] = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0)
df["stake"] = pd.to_numeric(df.get("stake", 0), errors="coerce").fillna(0)
df["cuota"] = pd.to_numeric(df.get("cuota", 0), errors="coerce").fillna(0)
df["estado"] = df["estado"].astype(str).str.strip()

df_cerradas = df[df["estado"].isin(["Ganada", "Perdida", "Nulo"])].copy()

total_pnl = df_cerradas["pnl"].sum() if not df_cerradas.empty else 0.0
stake_cerrado = df_cerradas["stake"].sum() if not df_cerradas.empty else 0.0
roi = (total_pnl / stake_cerrado * 100) if stake_cerrado > 0 else 0.0
bank_actual = bank_inicial + total_pnl

solo_gp = df_cerradas[df_cerradas["estado"].isin(["Ganada", "Perdida"])]
total_decididas = len(solo_gp)
ganadas = len(solo_gp[solo_gp["estado"] == "Ganada"])
winrate = (ganadas / total_decididas * 100) if total_decididas > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏦 Bank Actual", f"{bank_actual:,.2f} $", delta=f"{total_pnl:+,.2f} $")
c2.metric("💰 P&L", f"{total_pnl:+,.2f} $")
c3.metric("📊 ROI", f"{roi:+.1f}%")
c4.metric("🎯 Winrate", f"{winrate:.1f}%", delta=f"{ganadas}G / {total_decididas - ganadas}P")

# Curva ordenada por fecha
if not df_cerradas.empty:
    st.subheader("📊 Curva de Rendimiento (Bank)")
    
    if "fecha" in df_cerradas.columns:
        df_cerradas["_ord"] = pd.to_datetime(df_cerradas["fecha"], errors="coerce")
    elif "created_at" in df_cerradas.columns:
        df_cerradas["_ord"] = pd.to_datetime(df_cerradas["created_at"], errors="coerce")
    else:
        df_cerradas["_ord"] = range(len(df_cerradas))
    
    df_curva = df_cerradas.sort_values("_ord").copy()
    df_curva["acumulado"] = bank_inicial + df_curva["pnl"].cumsum()
    st.line_chart(df_curva.set_index("_ord")["acumulado"] if "_ord" in df_curva.columns else df_curva["acumulado"])
else:
    st.info("Cierra tu primera apuesta para ver la curva de rendimiento.")

# ====================== HISTORIAL CON FILTROS ======================
st.markdown("---")
st.subheader("Historial")

f1, f2 = st.columns(2)
with f1:
    estados_filtro = st.multiselect(
        "Filtrar por estado",
        options=["Pendiente", "Ganada", "Perdida", "Nulo"],
        default=["Pendiente", "Ganada", "Perdida", "Nulo"],
    )
with f2:
    mercados_disp = sorted(df["mercado"].dropna().unique().tolist()) if "mercado" in df.columns else []
    mercados_filtro = st.multiselect("Filtrar por mercado", options=mercados_disp, default=mercados_disp)

df_hist = df.copy()
if estados_filtro:
    df_hist = df_hist[df_hist["estado"].isin(estados_filtro)]
if mercados_filtro and "mercado" in df_hist.columns:
    df_hist = df_hist[df_hist["mercado"].isin(mercados_filtro)]

cols_mostrar = [c for c in ["id", "fecha", "evento", "seleccion", "mercado", "cuota", "stake", "estado", "pnl"] if c in df_hist.columns]
df_hist = df_hist.sort_values(by="id", ascending=False)

st.dataframe(df_hist[cols_mostrar], use_container_width=True, hide_index=True)

# ====================== BORRAR CON CONFIRMACIÓN ======================
with st.expander("🗑️ Gestionar / Borrar apuestas"):
    if df.empty:
        st.caption("No hay apuestas.")
    else:
        def label_borrar(i):
            row = df[df["id"] == i].iloc[0]
            return f"ID {i} · {row.get('evento', '?')} · {row.get('seleccion', '?')} · {row.get('estado', '?')}"

        apuesta_borrar = st.selectbox(
            "Selecciona la apuesta a eliminar",
            df["id"].tolist(),
            format_func=label_borrar,
        )
        confirmar = st.checkbox("Confirmo que quiero eliminar esta apuesta")
        
        if st.button("🗑️ Eliminar apuesta", type="primary", disabled=not confirmar):
            try:
                supabase.table("apuestas").delete().eq("id", int(apuesta_borrar)).eq("user_id", user_id).execute()
                st.success("Apuesta eliminada.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al eliminar (revisa política DELETE en Supabase): {e}")
