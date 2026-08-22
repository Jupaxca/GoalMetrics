import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# ====================== CONFIGURACIÓN ======================
if 'bankroll_inicial' not in st.session_state: st.session_state.bankroll_inicial = 1000.0
if 'moneda_sel' not in st.session_state: st.session_state.moneda_sel = "$"

if 'historial_apuestas' not in st.session_state:
    st.session_state.historial_apuestas = pd.DataFrame(columns=[
        'Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'
    ])

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
    
    if st.form_submit_button("💾 Guardar Apuesta"):
        nueva = pd.DataFrame([{
            'Fecha': datetime.today().strftime('%Y-%m-%d'),
            'Selección': evento, 'Mercado': mercado,
            'Cuota': float(cuota), 'Stake': float(stake),
            'Estado': 'Pendiente', 'Ganancia/Pérdida': 0.0
        }])
        st.session_state.historial_apuestas = pd.concat([st.session_state.historial_apuestas, nueva], ignore_index=True)
        st.rerun()

# ====================== PANTALLA PRINCIPAL ======================
st.markdown("### 📈 Tracker de Apuestas & P&L")
st.caption("Haz clic en la columna 'Estado' para cambiar de 'Pendiente' a 'Ganada' o 'Perdida'. El capital y el gráfico se actualizan solos.")

if not st.session_state.historial_apuestas.empty:
    df = st.session_state.historial_apuestas.astype({
        'Estado': 'str', 'Selección': 'str', 'Mercado': 'str',
        'Cuota': 'float', 'Stake': 'float', 'Ganancia/Pérdida': 'float'
    })

    edited_df = st.data_editor(
        df,
        column_config={
            "Estado": st.column_config.SelectboxColumn(
                "Estado",
                options=["Pendiente", "Ganada", "Perdida", "Anulada"],
                required=True,
            ),
            "Cuota": st.column_config.NumberColumn(format="%.2f"),
            "Stake": st.column_config.NumberColumn(format="%.2f"),
            "Ganancia/Pérdida": st.column_config.NumberColumn(format="%.2f", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="editor_apuestas"
    )

    if not edited_df.equals(df):
        for i, row in edited_df.iterrows():
            if row['Estado'] == 'Ganada':
                edited_df.at[i, 'Ganancia/Pérdida'] = round(float(row['Stake']) * (float(row['Cuota']) - 1), 2)
            elif row['Estado'] == 'Perdida':
                edited_df.at[i, 'Ganancia/Pérdida'] = round(-float(row['Stake']), 2)
            else:
                edited_df.at[i, 'Ganancia/Pérdida'] = 0.0
        
        st.session_state.historial_apuestas = edited_df
        st.rerun()

    df_cerradas = st.session_state.historial_apuestas[st.session_state.historial_apuestas['Estado'].isin(['Ganada', 'Perdida'])].copy()
    beneficio = df_cerradas['Ganancia/Pérdida'].sum()
    bankroll_actual = st.session_state.bankroll_inicial + beneficio
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Bankroll Actual", f"{st.session_state.moneda_sel} {bankroll_actual:,.2f}", delta=f"{st.session_state.moneda_sel} {beneficio:+,.2f}")
    col2.metric("📊 P&L Neto", f"{st.session_state.moneda_sel} {beneficio:,.2f}")
    winrate = (len(df_cerradas[df_cerradas['Estado']=='Ganada']) / len(df_cerradas) * 100) if len(df_cerradas) > 0 else 0.0
    col3.metric("🎯 Winrate", f"{winrate:.1f}%")

    st.markdown("---")
    st.subheader("📉 Evolución del Capital")

    # ====================== GRÁFICO DE EVOLUCIÓN (DEVUELTO) ======================
    if not df_cerradas.empty:
        df_chart = df_cerradas.sort_values('Fecha').copy()
        df_chart['Bankroll Acumulado'] = st.session_state.bankroll_inicial + df_chart['Ganancia/Pérdida'].cumsum()
        
        chart = alt.Chart(df_chart).mark_line(color='#10B981', strokeWidth=3, point=True).encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Bankroll Acumulado:Q', title=f'Capital ({st.session_state.moneda_sel})', scale=alt.Scale(zero=False)),
            tooltip=['Fecha:T', 'Selección:N', 'Bankroll Acumulado:Q', 'Ganancia/Pérdida:Q']
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Cierra al menos una apuesta (cambiando su estado a Ganada o Perdida) para ver el gráfico de evolución.")

    st.markdown("---")
    if st.button("🗑️ Borrar Todo el Historial"):
        st.session_state.historial_apuestas = pd.DataFrame(columns=['Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'])
        st.rerun()
else:
    st.info("No hay apuestas registradas. Usa el formulario de la barra lateral.")
