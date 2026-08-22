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
    st.warning("Por favor inicia sesión en la página principal para ver tu tracker.")
    st.stop()

user_id = user.id

if 'moneda_sel' not in st.session_state: st.session_state.moneda_sel = "$"
if 'bankroll_inicial' not in st.session_state: st.session_state.bankroll_inicial = 1000.0

# Función para cargar apuestas exclusivas del usuario desde Supabase
def cargar_apuestas():
    try:
        response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
        data = response.data
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=['id', 'fecha', 'seleccion', 'mercado', 'cuota', 'stake', 'estado', 'pnl'])
    except Exception as e:
        return pd.DataFrame(columns=['id', 'fecha', 'seleccion', 'mercado', 'cuota', 'stake', 'estado', 'pnl'])

if 'historial_apuestas' not in st.session_state or st.session_state.get('loaded_user') != user_id:
    st.session_state.historial_apuestas = cargar_apuestas()
    st.session_state.loaded_user = user_id

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Configuración")
simbolos = {"USD ($)": "$", "EUR (€)": "€", "COP ($)": "COP $", "MXN ($)": "MXN $"}
st.session_state.moneda_sel = simbolos[st.sidebar.selectbox("💱 Moneda", list(simbolos.keys()))]
st.session_state.bankroll_inicial = st.sidebar.number_input("Capital Inicial (Bankroll)", value=float(st.session_state.bankroll_inicial))

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Registrar Apuesta")

with st.sidebar.form("nueva_apuesta", clear_on_submit=True):
    evento = st.text_input("Evento (Ej: Real Madrid vs Barcelona)")
    mercado = st.selectbox("Mercado", ["Victoria (1)", "Empate (X)", "BTTS", "Over Línea"])
    cuota = st.number_input("Cuota Decimal", value=1.85, step=0.01)
    stake = st.number_input("Dinero arriesgado (Stake)", value=50.0, step=10.0)
    
    if st.form_submit_button("💾 Guardar en la Nube"):
        nueva_apuesta = {
            'user_id': user_id,
            'fecha': datetime.today().strftime('%Y-%m-%d'),
            'seleccion': evento,
            'mercado': mercado,
            'cuota': float(cuota),
            'stake': float(stake),
            'estado': 'Pendiente',
            'pnl': 0.0
        }
        try:
            supabase.table("apuestas").insert(nueva_apuesta).execute()
            st.success("¡Apuesta guardada en la nube!")
            st.session_state.historial_apuestas = cargar_apuestas()
            st.rerun()
        except Exception as e:
            st.error(f"Error al guardar: {e}")

# ====================== PANTALLA PRINCIPAL ======================
st.markdown("### 📈 Tracker de Apuestas & P&L (Nube)")
st.caption("Tus datos están seguros y vinculados a tu cuenta personal.")

df_bets = st.session_state.historial_apuestas

if not df_bets.empty:
    df = df_bets.astype({
        'estado': 'str', 'seleccion': 'str', 'mercado': 'str',
        'cuota': 'float', 'stake': 'float', 'pnl': 'float'
    })

    edited_df = st.data_editor(
        df,
        column_config={
            "estado": st.column_config.SelectboxColumn(
                "Estado",
                options=["Pendiente", "Ganada", "Perdida", "Anulada"],
                required=True,
            ),
            "cuota": st.column_config.NumberColumn(format="%.2f"),
            "stake": st.column_config.NumberColumn(format="%.2f"),
            "pnl": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "id": None, "user_id": None, "created_at": None
        },
        hide_index=True,
        use_container_width=True,
        key="editor_apuestas"
    )

    if not edited_df.equals(df):
        for i, row in edited_df.iterrows():
            if row['estado'] == 'Ganada':
                nuevo_pnl = round(float(row['stake']) * (float(row['cuota']) - 1), 2)
            elif row['estado'] == 'Perdida':
                nuevo_pnl = round(-float(row['stake']), 2)
            else:
                nuevo_pnl = 0.0
            
            edited_df.at[i, 'pnl'] = nuevo_pnl
            
            row_id = row.get('id')
            if row_id:
                try:
                    supabase.table("apuestas").update({
                        'estado': row['estado'],
                        'pnl': nuevo_pnl
                    }).eq("id", row_id).execute()
                except Exception as e:
                    st.error(f"Error al actualizar: {e}")
        
        st.session_state.historial_apuestas = edited_df
        st.rerun()

    df_cerradas = st.session_state.historial_apuestas[st.session_state.historial_apuestas['estado'].isin(['Ganada', 'Perdida'])].copy()
    beneficio = df_cerradas['pnl'].sum()
    bankroll_actual = st.session_state.bankroll_inicial + beneficio
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Bankroll Actual", f"{st.session_state.moneda_sel} {bankroll_actual:,.2f}", delta=f"{st.session_state.moneda_sel} {beneficio:+,.2f}")
    col2.metric("📊 P&L Neto", f"{st.session_state.moneda_sel} {beneficio:,.2f}")
    winrate = (len(df_cerradas[df_cerradas['estado']=='Ganada']) / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0.0
    col3.metric("🎯 Winrate", f"{winrate:.1f}%")

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
    if st.button("🗑️ Borrar Todo mi Historial"):
        try:
            supabase.table("apuestas").delete().eq("user_id", user_id).execute()
            st.session_state.historial_apuestas = cargar_apuestas()
            st.rerun()
        except Exception as e:
            st.error(f"Error al borrar: {e}")
else:
    st.info("Aún no tienes apuestas registradas. Usa el formulario de la barra lateral para añadir la primera.")
