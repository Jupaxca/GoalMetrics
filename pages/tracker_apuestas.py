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

# --- CONFIGURACIÓN Y REGISTRO EN BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    bank_inicial = st.number_input("Bank Inicial ($)", min_value=0.0, value=100.0, step=10.0)
    
    st.markdown("---")
    st.header("➕ Nueva Apuesta")
    with st.form("nueva_apuesta", clear_on_submit=True):
        evento = st.text_input("Evento / Partido")
        seleccion = st.text_input("Selección (ej: Real Madrid, Más de 2.5)")
        
        opciones_mercado = [
            "Ganador (1X2)", "Doble Oportunidad", "Ambos Marcan (BTTS)", 
            "Over/Under Goles", "Over/Under Córners", "Hándicap Asiático", 
            "Hándicap Europeo", "Tarjetas", "Resultado Exacto", "Otro"
        ]
        mercado = st.selectbox("Mercado", opciones_mercado)
        
        cuota = st.number_input("Cuota", min_value=1.00, step=0.01)
        stake = st.number_input("Stake ($)", min_value=1.0, step=0.5)
        
        if st.form_submit_button("Guardar Apuesta"):
            if not evento or not seleccion:
                st.error("El evento y la selección son obligatorios.")
            else:
                try:
                    data = {
                        "user_id": user_id,
                        "evento": evento,
                        "seleccion": seleccion,
                        "mercado": mercado,
                        "cuota": float(cuota),
                        "stake": float(stake),
                        "estado": "Pendiente",
                        "pnl": 0.0,
                        "fecha": str(datetime.date.today()),
                        "created_at": datetime.datetime.now().isoformat()
                    }
                    supabase.table("apuestas").insert(data).execute()
                    st.success("¡Apuesta registrada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

# --- PANEL DE RESULTADOS (CERRAR APUESTAS) ---
st.subheader("🏁 Cerrar Apuesta")
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
    todas_las_apuestas = response.data if response.data else []
except Exception as e:
    todas_las_apuestas = []
    st.error(f"Error al conectar con la base de datos: {e}")

# Filtramos solo las pendientes para el selector de arriba
pendientes = [a for a in todas_las_apuestas if a.get('estado') == "Pendiente"]

if pendientes:
    df_pendientes = pd.DataFrame(pendientes)
    cols = st.columns([2, 1, 1])
    with cols[0]:
        apuesta_id = st.selectbox("Selecciona la apuesta a cerrar:", df_pendientes['id'].tolist(), format_func=lambda x: f"{df_pendientes[df_pendientes['id']==x]['evento'].values[0]} ({df_pendientes[df_pendientes['id']==x]['seleccion'].values[0]})")
    with cols[1]:
        resultado = st.selectbox("Resultado", ["Ganada", "Perdida"])
    
    with cols[2]:
        st.write("")
        st.write("")
        if st.button("Actualizar Resultado", use_container_width=True):
            try:
                apuesta = df_pendientes[df_pendientes['id'] == apuesta_id].iloc[0]
                pnl = (float(apuesta['stake']) * (float(apuesta['cuota']) - 1)) if resultado == "Ganada" else -float(apuesta['stake'])
                
                # Ejecutamos el Update
                supabase.table("apuestas").update({
                    "estado": resultado, 
                    "pnl": float(pnl)
                }).eq("id", int(apuesta_id)).execute()
                
                st.success("¡Apuesta actualizada con éxito!")
                st.rerun()
            except Exception as e:
                st.error(f"Error al actualizar: {e}")
else:
    st.info("No tienes apuestas pendientes por cerrar.")

# --- DASHBOARD DE MÉTRICAS Y GRÁFICO ---
st.markdown("---")

if todas_las_apuestas:
    df = pd.DataFrame(todas_las_apuestas)
    
    # Diagnóstico visual para salir de dudas
    with st.expander("🔍 Ver datos en bruto (Diagnóstico)"):
        st.write(df)

    # Filtramos cerradas (ignorando mayúsculas/minúsculas por seguridad)
    df_cerradas = df[df['estado'].str.capitalize().isin(['Ganada', 'Perdida'])].copy()
    
    total_pnl = df_cerradas['pnl'].astype(float).sum() if not df_cerradas.empty else 0.0
    bank_actual = bank_inicial + total_pnl
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🏦 Bank Actual", f"{bank_actual:,.2f} $", delta=f"{total_pnl:,.2f} $")
    col2.metric("💰 Beneficio PnL", f"{total_pnl:,.2f} $")
    
    total_cerradas = len(df_cerradas)
    ganadas = len(df_cerradas[df_cerradas['estado'].str.capitalize() == 'Ganada'])
    winrate = (ganadas / total_cerradas) * 100 if total_cerradas > 0 else 0
    col3.metric("🎯 Winrate", f"{winrate:.1f}%")
    
    if not df_cerradas.empty:
        df_cerradas['acumulado'] = bank_inicial + df_cerradas['pnl'].astype(float).cumsum()
        st.subheader("📊 Curva de Rendimiento (Evolución del Bank)")
        st.line_chart(df_cerradas['acumulado'])
    else:
        st.info("Cierra tu primera apuesta para ver la curva de rendimiento y la evolución de tu bank.")

    st.subheader("Historial Completo")
    st.dataframe(df.sort_values(by="id", ascending=False), use_container_width=True, hide_index=True)
else:
    st.write("Aún no tienes historial de apuestas registrado.")
