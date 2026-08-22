import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# ====================== CONFIGURACIÓN DE SESIÓN ======================
if 'bankroll_inicial' not in st.session_state: st.session_state.bankroll_inicial = 1000.0
if 'moneda_sel' not in st.session_state: st.session_state.moneda_sel = "$"
if 'historial_apuestas' not in st.session_state:
    st.session_state.historial_apuestas = pd.DataFrame(columns=[
        'ID', 'Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'
    ])

# Función para calcular P&L
def calcular_pnl(df):
    df = df.copy()
    # Si cambia el estado, recalculamos el P&L
    for i, row in df.iterrows():
        if row['Estado'] == 'Ganada':
            df.at[i, 'Ganancia/Pérdida'] = round(float(row['Stake']) * (float(row['Cuota']) - 1), 2)
        elif row['Estado'] == 'Perdida':
            df.at[i, 'Ganancia/Pérdida'] = round(-float(row['Stake']), 2)
        else:
            df.at[i, 'Ganancia/Pérdida'] = 0.0
    return df

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Configuración Personal")
simbolos_moneda = {"USD ($)": "$", "EUR (€)": "€", "COP ($)": "COP $", "MXN ($)": "MXN $"}
moneda_key = st.sidebar.selectbox("💱 Moneda", list(simbolos_moneda.keys()))
st.session_state.moneda_sel = simbolos_moneda[moneda_key]
st.session_state.bankroll_inicial = st.sidebar.number_input(f"Capital Inicial ({st.session_state.moneda_sel})", value=float(st.session_state.bankroll_inicial))

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Registrar Nueva Apuesta")
with st.sidebar.form("form_apuesta", clear_on_submit=True):
    evento_sel = st.text_input("Selección / Evento")
    mercado_sel = st.selectbox("Mercado", ["Victoria (1)", "Empate (X)", "Derrota (2)", "BTTS", "DNB", "Over Línea"])
    cuota_sel = st.number_input("Cuota", min_value=1.01, value=1.85, step=0.01)
    stake_sel = st.number_input("Stake", min_value=1.0, value=20.0, step=5.0)
    submitted = st.form_submit_button("💾 Guardar Apuesta")
    
    if submitted:
        nueva_fila = pd.DataFrame([{
            'ID': len(st.session_state.historial_apuestas) + 1,
            'Fecha': datetime.today().strftime('%Y-%m-%d'),
            'Selección': evento_sel, 'Mercado': mercado_sel,
            'Cuota': cuota_sel, 'Stake': stake_sel,
            'Estado': 'Pendiente', 'Ganancia/Pérdida': 0.0
        }])
        st.session_state.historial_apuestas = pd.concat([st.session_state.historial_apuestas, nueva_fila], ignore_index=True)
        st.rerun()

# ====================== PANTALLA PRINCIPAL ======================
st.markdown("### 📈 Tracker de Apuestas")
st.caption("Haz clic en la columna 'Estado' para cambiar de 'Pendiente' a 'Ganada' o 'Perdida'.")

# Editor interactivo
if not st.session_state.historial_apuestas.empty:
    edited_df = st.data_editor(
        st.session_state.historial_apuestas,
        column_config={
            "Estado": st.column_config.SelectboxColumn(
                "Estado",
                options=["Pendiente", "Ganada", "Perdida", "Anulada"],
                required=True,
            ),
            "ID": None # Ocultar ID
        },
        hide_index=True,
        use_container_width=True
    )

    # Si hubo cambios, actualizamos el estado y recalculamos
    if not edited_df.equals(st.session_state.historial_apuestas):
        st.session_state.historial_apuestas = calcular_pnl(edited_df)
        st.rerun()

    # Cálculos para métricas
    df_cerradas = st.session_state.historial_apuestas[st.session_state.historial_apuestas['Estado'].isin(['Ganada', 'Perdida'])]
    beneficio = df_cerradas['Ganancia/Pérdida'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Bankroll", f"{st.session_state.moneda_sel} {st.session_state.bankroll_inicial + beneficio:,.2f}")
    col2.metric("📊 P&L Neto", f"{st.session_state.moneda_sel} {beneficio:,.2f}")
    col3.metric("🎯 Winrate", f"{(len(df_cerradas[df_cerradas['Estado']=='Ganada'])/len(df_cerradas)*100 if len(df_cerradas)>0 else 0):.1f}%")

    if st.button("🗑️ Borrar Historial"):
        st.session_state.historial_apuestas = pd.DataFrame(columns=['ID', 'Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'])
        st.rerun()
else:
    st.info("No hay apuestas registradas.")
