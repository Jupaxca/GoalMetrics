import streamlit as st
import pandas as pd
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
st.session_state.bankroll_inicial = st.sidebar.number_input("Capital Total (Bankroll)", value=float(st.session_state.bankroll_inicial))

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Registrar Apuesta")

with st.sidebar.form("nueva_apuesta", clear_on_submit=True):
    evento = st.text_input("Evento (Ej: Real Madrid vs Barcelona)")
    mercado = st.selectbox("Mercado", ["Victoria (1)", "Empate (X)", "BTTS", "Over Línea"])
    cuota = st.number_input("Cuota Decimal", value=1.33, step=0.01)
    
    # Texto aclaratorio para evitar confusiones
    stake = st.number_input("Dinero que arriesgas en esta apuesta (Stake)", value=100.0, step=10.0)
    
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
st.markdown("### 📈 Tracker de Apuestas")
st.caption("Haz clic en la columna 'Estado' para cambiar de 'Pendiente' a 'Ganada' o 'Perdida'.")

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

    df_cerradas = st.session_state.historial_apuestas[st.session_state.historial_apuestas['Estado'].isin(['Ganada', 'Perdida'])]
    beneficio = df_cerradas['Ganancia/Pérdida'].sum()
    
    col1, col2 = st.columns(2)
    col1.metric("💰 Bankroll Total Actual", f"{st.session_state.moneda_sel} {st.session_state.bankroll_inicial + beneficio:,.2f}")
    col2.metric("📊 P&L Neto", f"{st.session_state.moneda_sel} {beneficio:,.2f}")

    if st.button("🗑️ Borrar Todo el Historial"):
        st.session_state.historial_apuestas = pd.DataFrame(columns=['Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'])
        st.rerun()
else:
    st.info("No hay apuestas registradas. Usa el formulario de la barra lateral.")
