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
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Sesión no detectada. Por favor regresa a la página principal e inicia sesión.")
    st.stop()

user = st.session_state.user
user_id = user.id

st.markdown("## 📈 Tracker de Apuestas & Análisis Pro")

# --- LÓGICA DE REGISTRO ---
with st.sidebar:
    st.header("➕ Nueva Apuesta")
    with st.form("nueva_apuesta", clear_on_submit=True):
        evento = st.text_input("Evento / Partido")
        
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
            "Otro"
        ]
        mercado = st.selectbox("Mercado", opciones_mercado)
        
        cuota = st.number_input("Cuota", min_value=1.00, step=0.01)
        stake = st.number_input("Stake ($)", min_value=1.0, step=0.5)
        
        if st.form_submit_button("Guardar Apuesta"):
            if not evento:
                st.error("El evento es obligatorio.")
            else:
                try:
                    data = {
                        "user_id": user_id,
                        "evento": evento,
                        "mercado": mercado,
                        "cuota": float(cuota),
                        "stake": float(stake),
                        "estado": "Pendiente",
                        "pnl": 0.0,
                        "fecha": str(datetime.date.today())  # <--- AQUÍ ENVIAMOS LA FECHA ACTUAL
                    }
                    supabase.table("apuestas").insert(data).execute()
                    st.success("¡Apuesta registrada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

# --- PANEL DE RESULTADOS (CERRAR APUESTAS) ---
st.subheader("🏁 Cerrar Apuesta")
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).eq("estado", "Pendiente").execute()
    pendientes = response.data if response.data else []
except Exception:
    pendientes = []

if pendientes:
    df_pendientes = pd.DataFrame(pendientes)
    cols = st.columns([2, 1, 1])
    with cols[0]:
        apuesta_id = st.selectbox("Selecciona la apuesta a cerrar:", df_pendientes['id'].tolist(), format_func=lambda x: f"{df_pendientes[df_pendientes['id']==x]['evento'].values[0]} ({df_pendientes[df_pendientes['id']==x]['mercado'].values[0]})")
    with cols[1]:
        resultado = st.selectbox("Resultado", ["Ganada", "Perdida"])
    
    with cols[2]:
        st.write("")
        st.write("")
        if st.button("Actualizar Resultado", use_container_width=True):
            apuesta = df_pendientes[df_pendientes['id'] == apuesta_id].iloc[0]
            pnl = (apuesta['stake'] * (apuesta['cuota'] - 1)) if resultado == "Ganada" else -apuesta['stake']
            supabase.table("apuestas").update({"estado": resultado, "pnl": pnl}).eq("id", apuesta_id).execute()
            st.success("¡Apuesta actualizada!")
            st.rerun()
else:
    st.info("No tienes apuestas pendientes por cerrar.")

# --- DASHBOARD DE MÉTRICAS Y GRÁFICO ---
st.markdown("---")
try:
    res = supabase.table("apuestas").select("*").eq("user_id", user_id).order("id", asc=True).execute()
    df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
except Exception:
    df = pd.DataFrame()

if not df.empty:
    df_cerradas = df[df['estado'].isin(['Ganada', 'Perdida'])].copy()
    
    if not df_cerradas.empty:
        df_cerradas['acumulado'] = df_cerradas['pnl'].astype(float).cumsum()
        
        col1, col2 = st.columns(2)
        col1.metric("💰 Balance Total (PnL)", f"{df_cerradas['pnl'].astype(float).sum():,.2f} $")
        
        total_cerradas = len(df_cerradas)
        ganadas = len(df_cerradas[df_cerradas['estado']=='Ganada'])
        winrate = (ganadas / total_cerradas) * 100 if total_cerradas > 0 else 0
        col2.metric("🎯 Winrate", f"{winrate:.1f}%")
        
        st.subheader("📊 Curva de Rendimiento")
        st.line_chart(df_cerradas['acumulado'])
    else:
        st.info("Cierra tu primera apuesta para ver la curva de rendimiento y las métricas.")

    st.subheader("Historial Completo")
    st.dataframe(df.sort_values(by="id", ascending=False), use_container_width=True, hide_index=True)
else:
    st.write("Aún no tienes historial de apuestas.")
