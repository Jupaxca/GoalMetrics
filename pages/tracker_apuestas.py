import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from supabase import create_client, Client

@st.cache_resource
def init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_supabase()
user = st.session_state.get('user')

if not user:
    st.warning("Por favor inicia sesión en la página principal.")
    st.stop()

user_id = user.id

# Configuración inicial
if 'moneda_sel' not in st.session_state: st.session_state.moneda_sel = "$"
if 'bankroll_inicial' not in st.session_state: st.session_state.bankroll_inicial = 1000.0

def cargar_apuestas():
    try:
        response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame(columns=['id', 'fecha', 'seleccion', 'mercado', 'cuota', 'stake', 'estado', 'pnl'])
    except:
        return pd.DataFrame(columns=['id', 'fecha', 'seleccion', 'mercado', 'cuota', 'stake', 'estado', 'pnl'])

# Carga de datos
if 'historial_apuestas' not in st.session_state or st.session_state.get('loaded_user') != user_id:
    st.session_state.historial_apuestas = cargar_apuestas()
    st.session_state.loaded_user = user_id

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Configuración")
st.session_state.moneda_sel = st.sidebar.selectbox("💱 Moneda", ["$", "€", "COP $", "MXN $"])
st.session_state.bankroll_inicial = st.sidebar.number_input("Capital Inicial", value=float(st.session_state.bankroll_inicial))

with st.sidebar.form("nueva_apuesta", clear_on_submit=True):
    evento = st.text_input("Evento")
    mercado = st.selectbox("Mercado", ["Victoria (1)", "Empate (X)", "BTTS", "Over Línea"])
    cuota = st.number_input("Cuota", value=1.85, step=0.01)
    stake = st.number_input("Stake", value=50.0, step=10.0)
    
    if st.form_submit_button("💾 Guardar"):
        with st.spinner("Conectando a la nube..."):
            try:
                supabase.table("apuestas").insert({
                    'user_id': user_id, 'fecha': datetime.today().strftime('%Y-%m-%d'),
                    'seleccion': evento, 'mercado': mercado, 'cuota': float(cuota),
                    'stake': float(stake), 'estado': 'Pendiente', 'pnl': 0.0
                }).execute()
                st.session_state.historial_apuestas = cargar_apuestas()
                st.toast("✅ ¡Apuesta registrada!")
            except Exception as e:
                st.error(f"Error: {e}")

# ====================== PANTALLA PRINCIPAL ======================
st.markdown("### 📈 Tracker de Apuestas & Análisis Pro")

df_bets = st.session_state.historial_apuestas

if not df_bets.empty:
    df = df_bets.astype({'cuota': 'float', 'stake': 'float', 'pnl': 'float'})
    
    edited_df = st.data_editor(
        df,
        column_config={
            "estado": st.column_config.SelectboxColumn("Estado", options=["Pendiente", "Ganada", "Perdida", "Anulada"]),
            "pnl": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "id": None, "user_id": None, "created_at": None
        },
        hide_index=True, use_container_width=True
    )

    if not edited_df.equals(df):
        with st.spinner("Actualizando datos..."):
            for i, row in edited_df.iterrows():
                if row['estado'] == 'Ganada': nuevo_pnl = round(float(row['stake']) * (float(row['cuota']) - 1), 2)
                elif row['estado'] == 'Perdida': nuevo_pnl = round(-float(row['stake']), 2)
                else: nuevo_pnl = 0.0
                
                if edited_df.at[i, 'pnl'] != nuevo_pnl:
                    edited_df.at[i, 'pnl'] = nuevo_pnl
                    supabase.table("apuestas").update({'estado': row['estado'], 'pnl': nuevo_pnl}).eq("id", row['id']).execute()
            
            st.session_state.historial_apuestas = edited_df
            st.toast("🔄 Datos sincronizados")
            st.rerun()

    # ====================== MÉTRICAS PRO (PASO 3) ======================
    df_cerradas = st.session_state.historial_apuestas[st.session_state.historial_apuestas['estado'].isin(['Ganada', 'Perdida'])]
    beneficio = df_cerradas['pnl'].sum()
    total_apostado = df_cerradas['stake'].sum()
    
    # Cálculos avanzados
    yield_pct = (beneficio / total_apostado * 100) if total_apostado > 0 else 0.0
    winrate = (len(df_cerradas[df_cerradas['estado']=='Ganada']) / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0.0
    
    st.markdown("---")
    st.subheader("📊 Rendimiento Financiero")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Bankroll", f"{st.session_state.moneda_sel} {st.session_state.bankroll_inicial + beneficio:,.2f}")
    col2.metric("📈 P&L Neto", f"{st.session_state.moneda_sel} {beneficio:,.2f}")
    col3.metric("🎯 Yield / ROI", f"{yield_pct:+.2f}%")
    col4.metric("🔥 Winrate", f"{winrate:.1f}%")

    st.markdown("---")
    st.subheader("📉 Evolución del Capital")

    if not df_cerradas.empty:
        df_chart = df_cerradas.sort_values('fecha').copy()
        df_chart['Bankroll Acumulado'] = st.session_state.bankroll_inicial + df_chart['pnl'].cumsum()
        
        chart = alt.Chart(df_chart).mark_line(color='#10B981', strokeWidth=3, point=True).encode(
            x=alt.X('fecha:T', title='Fecha'),
            y=alt.Y('Bankroll Acumulado:Q', title=f'Capital ({st.session_state.moneda_sel})', scale=alt.Scale(zero=False)),
            tooltip=['fecha:T', 'seleccion:N', 'Bankroll Acumulado:Q', 'pnl:Q']
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Cierra al menos una apuesta para ver el gráfico de evolución.")

    st.markdown("---")
    if st.button("🗑️ Limpiar Historial"):
        supabase.table("apuestas").delete().eq("user_id", user_id).execute()
        st.session_state.historial_apuestas = cargar_apuestas()
        st.rerun()
else:
    st.info("Sin apuestas. ¡Agrega la primera en la barra lateral!")
