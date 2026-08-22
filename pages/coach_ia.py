import streamlit as st
import pandas as pd
from supabase import create_client, Client

st.set_page_config(page_title="Coach IA | GoalMetrics", page_icon="🤖", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_supabase()
user = st.session_state.get('user')

if not user:
    st.warning("Por favor inicia sesión en la página principal para ver tu Coach.")
    st.stop()

user_id = user.id

st.markdown("## 🤖 Coach Inteligente de Apuestas")
st.caption("Tu analista personal basado en los datos de tu propia base de datos.")

# Cargar apuestas del usuario
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
    df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
except Exception as e:
    df = pd.DataFrame()

if df.empty:
    st.info("Aún no tienes suficientes datos. Registra algunas apuestas en el Tracker para que el Coach pueda analizarte.")
else:
    df_cerradas = df[df['estado'].isin(['Ganada', 'Perdida'])].copy()
    
    if df_cerradas.empty:
        st.warning("Tienes apuestas pendientes, pero ninguna cerrada (Ganada/Perdida). Cierra al menos una para activar el análisis.")
    else:
        # Cálculos de análisis
        total_apuestas = len(df_cerradas)
        ganadas = len(df_cerradas[df_cerradas['estado'] == 'Ganada'])
        winrate = (ganadas / total_apuestas) * 100
        beneficio_total = df_cerradas['pnl'].astype(float).sum()
        
        # Análisis por mercado
        df_cerradas['pnl'] = df_cerradas['pnl'].astype(float)
        rendimiento_mercado = df_cerradas.groupby('mercado')['pnl'].sum().reset_index()
        mejor_mercado = rendimiento_mercado.loc[rendimiento_mercado['pnl'].idxmax()]['mercado'] if not rendimiento_mercado.empty else "N/A"

        # Métricas principales del Coach
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Winrate Global", f"{winrate:.1f}%")
        col2.metric("🏆 Mejor Mercado", f"{mejor_mercado}")
        col3.metric("💡 P&L Histórico", f"{beneficio_total:+,.2f}")

        st.markdown("---")
        st.subheader("🧠 Diagnóstico del Asistente")

        # Consejos automáticos basados en IA/Lógica de datos
        consejos = []
        
        if winrate >= 60:
            consejos.append("🔥 **¡Excelente racha!** Tu porcentaje de acierto está en niveles muy altos. Mantén la disciplina en tus stakes.")
        elif winrate < 40:
            consejos.append("⚠️ **Cuidado con el volumen:** Tu efectividad está por debajo del 40%. Te sugiero reducir el importe (stake) de tus apuestas hasta recuperar confianza.")
            
        if beneficio_total > 0:
            consejos.append(f"📈 **Vas en positivo:** El mercado de **{mejor_mercado}** es el que más dinero te está dejando. Considera enfocar tus análisis en esa dirección.")
        else:
            consejos.append("📉 **Momento de pausa:** Estás en negativo general. Revisa si estás apostando por corazonada o por estadística fría.")

        for consejo in consejos:
            st.info(consejo)

        st.markdown("---")
        st.subheader("📊 Rendimiento por Mercado")
        st.dataframe(rendimiento_mercado, use_container_width=True, hide_index=True)
