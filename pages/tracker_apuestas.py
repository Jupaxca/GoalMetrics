import streamlit as st
import pandas as pd
from supabase import create_client, Client

st.set_page_config(page_title="Tracker Pro | GoalMetrics", page_icon="📈", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()
user = st.session_state.user

st.markdown("## 📈 Tracker de Apuestas & Análisis Pro")

# --- LÓGICA DE REGISTRO ---
with st.sidebar:
    st.header("➕ Nueva Apuesta")
    with st.form("nueva_apuesta", clear_on_submit=True):
        evento = st.text_input("Evento")
        mercado = st.text_input("Mercado")
        cuota = st.number_input("Cuota", min_value=1.00, step=0.01)
        stake = st.number_input("Stake ($)", min_value=1.0, step=0.5)
        if st.form_submit_button("Guardar"):
            supabase.table("apuestas").insert({
                "user_id": user.id, "evento": evento, "mercado": mercado,
                "cuota": float(cuota), "stake": float(stake), "estado": "Pendiente", "pnl": 0.0
            }).execute()
            st.rerun()

# --- PANEL DE RESULTADOS (CERRAR APUESTAS) ---
st.subheader("🏁 Cerrar Apuesta")
response = supabase.table("apuestas").select("*").eq("user_id", user.id).eq("estado", "Pendiente").execute()
pendientes = response.data

if pendientes:
    df_pendientes = pd.DataFrame(pendientes)
    cols = st.columns([2, 1, 1])
    with cols[0]:
        apuesta_id = st.selectbox("Selecciona la apuesta a cerrar:", df_pendientes['id'].tolist(), format_func=lambda x: df_pendientes[df_pendientes['id']==x]['evento'].values[0])
    with cols[1]:
        resultado = st.selectbox("Resultado", ["Ganada", "Perdida"])
    
    if st.button("Actualizar Resultado"):
        apuesta = df_pendientes[df_pendientes['id'] == apuesta_id].iloc[0]
        pnl = (apuesta['stake'] * (apuesta['cuota'] - 1)) if resultado == "Ganada" else -apuesta['stake']
        supabase.table("apuestas").update({"estado": resultado, "pnl": pnl}).eq("id", apuesta_id).execute()
        st.rerun()
else:
    st.info("No tienes apuestas pendientes por cerrar.")

# --- DASHBOARD DE MÉTRICAS Y GRÁFICO ---
st.markdown("---")
res = supabase.table("apuestas").select("*").eq("user_id", user.id).order("id", asc=True).execute()
df = pd.DataFrame(res.data)

if not df.empty:
    # Filtramos solo las cerradas para el gráfico
    df_cerradas = df[df['estado'].isin(['Ganada', 'Perdida'])].copy()
    
    if not df_cerradas.empty:
        # Calcular el PnL acumulado
        df_cerradas['acumulado'] = df_cerradas['pnl'].cumsum()
        
        # Métricas principales
        col1, col2 = st.columns(2)
        col1.metric("💰 Balance Total (PnL)", f"{df_cerradas['pnl'].sum():,.2f} $")
        col2.metric("🎯 Winrate", f"{(len(df_cerradas[df_cerradas['estado']=='Ganada'])/len(df_cerradas))*100:.1f}%")
        
        # Gráfico de progresión
        st.subheader("📊 Curva de Rendimiento")
        st.line_chart(df_cerradas['acumulado'])
    else:
        st.info("Cierra tu primera apuesta para ver la curva de rendimiento.")

    # Tabla histórica
    st.subheader("Historial Completo")
    st.dataframe(df.sort_values(by="id", ascending=False), use_container_width=True, hide_index=True)
else:
    st.write("Aún no tienes historial.")
