import streamlit as st

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="📊",
    layout="wide"
)

# Definimos las tres páginas de la aplicación
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

tracker_apuestas = st.Page(
    "pages/tracker_apuestas.py", 
    title="Tracker de Apuestas", 
    icon="📈"
)

# Creamos y ejecutamos la navegación oficial con las 3 opciones
pg = st.navigation([analisis_equipos, analisis_jugadores, tracker_apuestas])
pg.run()
