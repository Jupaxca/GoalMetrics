import streamlit as st

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="📊",
    layout="wide"
)

# Definimos las páginas del menú lateral con nombres limpios e iconos
panel_principal = st.Page(
    "pages/panel_principal.py", 
    title="Panel Táctico", 
    icon="📊", 
    default=True
)

analisis_jugadores = st.Page(
    "pages/analisis_jugadores.py", 
    title="Análisis de Jugadores", 
    icon="👥"
)

# Creamos y ejecutamos la navegación oficial
pg = st.navigation([panel_principal, analisis_jugadores])
pg.run()
