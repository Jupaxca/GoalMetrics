import streamlit as st
import pandas as pd
from supabase import create_client, Client

st.set_page_config(page_title="Tracker de Apuestas | GoalMetrics", page_icon="📈", layout="wide")

# Inicializar Supabase
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Verificar autenticación
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Debes iniciar sesión para usar el Tracker.")
    st.stop()

user = st.session_state.user
user_id = user.id

st.markdown("## 📈 Tracker de Apuestas & Análisis Pro")

# Formulario para agregar apuesta
with st.sidebar:
    st.header("➕ Nueva Apuesta")
    with st.form("nueva_apuesta", clear_on_submit=True):
        evento = st.text_input("Evento / Partido")
        mercado = st.selectbox("Mercado", ["Victoria (1)", "Ambos marcan", "Over 2.5", "Handicap", "Otro"])
        cuota = st.number_input("Cuota", min_value=1.00, step=0.01)
        stake = st.number_input("Stake (Valor)", min_value=1.0, step=0.5)
        
        submitted = st.form_submit_button("Guardar")
        
        if submitted:
            if not evento:
                st.error("El evento es obligatorio.")
            else:
                try:
                    # Aquí es donde ocurre la magia para cumplir con tu política RLS
                    data = {
                        "user_id": user_id,  # <--- CRUCIAL: Esto debe coincidir con el auth.uid()
                        "evento": evento,
                        "mercado": mercado,
                        "cuota": float(cuota),
                        "stake": float(stake),
                        "estado": "Pendiente",
                        "pnl": 0.0
                    }
                    
                    supabase.table("apuestas").insert(data).execute()
                    st.success("¡Apuesta registrada!")
                    st.rerun() # Recargar para ver los cambios
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

# Mostrar tabla de apuestas
st.subheader("Tus Apuestas")

try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).order("id", desc=True).execute()
    df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
    
    if not df.empty:
        # Estilizar el dataframe
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Sin apuestas. ¡Agrega la primera en la barra lateral!")
except Exception as e:
    st.error(f"No se pudieron cargar tus apuestas: {e}")
