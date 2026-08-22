import streamlit as st

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="📊",
    layout="wide"
)

# Definimos las páginas con los nombres exactos para la barra lateral
analisis_equipos = st.Page(
    "pages/panel_principal.py", 
    title="Analisis equipos", 
    icon="📊", 
    default=True
)

analisis_jugadores = st.Page(
    "pages/analisis_jugadores.py", 
    title="Analisis jugadores", 
    icon="👥"
)

# Creamos y ejecutamos la navegación oficial
pg = st.navigation([analisis_equipos, analisis_jugadores])
pg.run()
