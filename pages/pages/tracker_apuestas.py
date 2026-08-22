import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# ====================== CONFIGURACIÓN DE SESIÓN ======================
if 'bankroll_inicial' not in st.session_state:
    st.session_state.bankroll_inicial = 1000.0

if 'moneda_sel' not in st.session_state:
    st.session_state.moneda_sel = "$"

if 'historial_apuestas' not in st.session_state:
    st.session_state.historial_apuestas = pd.DataFrame(columns=[
        'ID', 'Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'
    ])

# ====================== SIDEBAR: CONFIGURACIÓN FINANCIERA ======================
st.sidebar.header("⚙️ Configuración Personal")

# Selector de Moneda
simbolos_moneda = {"Dólar (USD)": "$", "Euro (EUR)": "€", "Libra (GBP)": "£", "Peso Colombiano (COP)": "COP $", "Peso Mexicano (MXN)": "MXN $"}
moneda_key = st.sidebar.selectbox("💱 Selecciona tu Moneda", list(simbolos_moneda.keys()))
simbolo = simbolos_moneda[moneda_key]
st.session_state.moneda_sel = simbolo

# Capital Inicial Personalizado
bankroll_input = st.sidebar.number_input(
    f"Capital Inicial ({simbolo})", 
    min_value=1.0, 
    value=float(st.session_state.bankroll_inicial), 
    step=50.0
)
if bankroll_input != st.session_state.bankroll_inicial:
    st.session_state.bankroll_inicial = bankroll_input

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Registrar Nueva Apuesta")

with st.sidebar.form("form_apuesta"):
    fecha_apuesta = st.date_input("Fecha", datetime.today())
    evento_sel = st.text_input("Selección / Evento", placeholder="Ej: Real Madrid vs Barcelona (1)")
    mercado_sel = st.selectbox("Mercado", ["Victoria (1)", "Empate (X)", "Derrota (2)", "BTTS", "DNB", "Over Línea", "Player Prop"])
    cuota_sel = st.number_input("Cuota Decimal", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    stake_sel = st.number_input(f"Stake ({simbolo} arriesgados)", min_value=0.1, value=20.0, step=5.0)
    estado_sel = st.selectbox("Estado del Partido", ["Pendiente", "Ganada", "Perdida", "Anulada"])
    
    submitted = st.form_submit_button("💾 Guardar Apuesta", use_container_width=True)
    
    if submitted:
        if estado_sel == "Ganada":
            pnl = round(stake_sel * (cuota_sel - 1), 2)
        elif estado_sel == "Perdida":
            pnl = round(-stake_sel, 2)
        else:
            pnl = 0.0
            
        nueva_fila = pd.DataFrame([{
            'ID': len(st.session_state.historial_apuestas) + 1,
            'Fecha': pd.to_datetime(fecha_apuesta),
            'Selección': evento_sel,
            'Mercado': mercado_sel,
            'Cuota': cuota_sel,
            'Stake': stake_sel,
            'Estado': estado_sel,
            'Ganancia/Pérdida': pnl
        }])
        
        st.session_state.historial_apuestas = pd.concat([st.session_state.historial_apuestas, nueva_fila], ignore_index=True)
        st.success("¡Apuesta registrada con éxito!")

# ====================== CSS ======================
st.markdown("""
    <style>
    .stApp { background-color: #0B0F19; color: #F3F4F6; }
    .stSidebar { background-color: #111827; }
    .header-box {
        background: linear-gradient(90deg, #10B981 0%, #1F2937 100%);
        padding: 22px 28px; border-radius: 14px; color: white;
        font-weight: 700; font-size: 24px; margin-bottom: 20px;
        text-align: center; letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] {
        background-color: #1F2937; padding: 12px 16px; border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== PANTALLA PRINCIPAL ======================
st.markdown('<div class="header-box">📈 GoalMetrics · P&L y Tracker de Apuestas</div>', unsafe_allow_html=True)
st.caption("Control financiero personalizado en tiempo real de tu bankroll, beneficios netos y rendimiento.")

df_bets = st.session_state.historial_apuestas
df_cerradas = df_bets[df_bets['Estado'].isin(['Ganada', 'Perdida'])]

total_apostado = df_cerradas['Stake'].sum() if not df_cerradas.empty else 0.0
beneficio_neto = df_cerradas['Ganancia/Pérdida'].sum() if not df_cerradas.empty else 0.0
bankroll_actual = st.session_state.bankroll_inicial + beneficio_neto

yield_pct = (beneficio_neto / total_apostado * 100) if total_apostado > 0 else 0.0
total_cerradas = len(df_cerradas)
ganadas = len(df_cerradas[df_cerradas['Estado'] == 'Ganada'])
winrate = (ganadas / total_cerradas * 100) if total_cerradas > 0 else 0.0

sim = st.session_state.moneda_sel

# Métricas adaptadas a la moneda elegida
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💰 Bankroll Actual", f"{sim} {bankroll_actual:,.2f}", delta=f"{sim} {beneficio_neto:+,.2f}")
m2.metric("📊 Beneficio Neto (P&L)", f"{sim} {beneficio_neto:,.2f}")
m3.metric("📈 Yield", f"{yield_pct:+.2f}%")
m4.metric("🎯 Winrate", f"{winrate:.1f}% ({ganadas}/{total_cerradas})")
m5.metric("📋 Total Apostado", f"{sim} {total_apostado:,.2f}")

st.markdown("---")

# ====================== GRÁFICO DE EVOLUCIÓN ======================
st.subheader("📉 Evolución del Capital")

if not df_cerradas.empty:
    df_chart = df_cerradas.sort_values('Fecha').copy()
    df_chart['Bankroll Acumulado'] = st.session_state.bankroll_inicial + df_chart['Ganancia/Pérdida'].cumsum()
    
    chart = alt.Chart(df_chart).mark_line(color='#10B981', strokeWidth=3, point=True).encode(
        x=alt.X('Fecha:T', title='Fecha'),
        y=alt.Y('Bankroll Acumulado:Q', title=f'Capital ({sim})', scale=alt.Scale(zero=False)),
        tooltip=['Fecha:T', 'Selección:N', 'Bankroll Acumulado:Q', 'Ganancia/Pérdida:Q']
    ).properties(height=320)
    
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("ℹ️ Registra al menos una apuesta con estado 'Ganada' o 'Perdida' en la barra lateral para ver el gráfico de evolución.")

st.markdown("---")

# ====================== TABLA DE HISTORIAL ======================
st.subheader("📋 Historial Completo de Apuestas")

if not df_bets.empty:
    df_display = df_bets.copy()
    df_display['Fecha'] = pd.to_datetime(df_display['Fecha']).dt.strftime('%Y-%m-%d')
    st.dataframe(df_display, hide_index=True, use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        csv_data = df_bets.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Historial en CSV",
            data=csv_data,
            file_name=f"goalmetrics_tracker_{datetime.today().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_c2:
        if st.button("🗑️ Borrar Todo el Historial", use_container_width=True):
            st.session_state.historial_apuestas = pd.DataFrame(columns=[
                'ID', 'Fecha', 'Selección', 'Mercado', 'Cuota', 'Stake', 'Estado', 'Ganancia/Pérdida'
            ])
            st.rerun()
else:
    st.info("Aún no hay apuestas registradas. Utiliza el formulario de la barra lateral para añadir la primera.")
