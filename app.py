import streamlit as st
from supabase import create_client, Client

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="📊",
    layout="wide"
)

# Inicializar conexión a Supabase usando los Secrets de Streamlit
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Control de sesión de usuario
if 'user' not in st.session_state:
    st.session_state.user = None

# Si no ha iniciado sesión, mostrar pantalla de Login / Registro
if st.session_state.user is None:
    st.markdown("## 🔐 GoalMetrics · Acceso de Usuarios")
    st.caption("Inicia sesión o regístrate para gestionar tus apuestas de forma segura en la nube.")
    
    tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Correo electrónico")
            password = st.text_input("Contraseña", type="password")
            submit_login = st.form_submit_button("Entrar", use_container_width=True)
            if submit_login:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.success("¡Inicio de sesión exitoso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al iniciar sesión: {e}")
                    
    with tab2:
        with st.form("signup_form"):
            email_su = st.text_input("Correo electrónico")
            password_su = st.text_input("Contraseña (mínimo 6 caracteres)", type="password")
            submit_signup = st.form_submit_button("Crear cuenta", use_container_width=True)
            if submit_signup:
                try:
                    res = supabase.auth.sign_up({"email": email_su, "password": password_su})
                    st.success("¡Cuenta creada con éxito! Ya puedes iniciar sesión en la otra pestaña.")
                except Exception as e:
                    st.error(f"Error al registrarse: {e}")
    st.stop()

# Si ya inició sesión, mostrar opción para salir en la barra lateral
if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

# Definir las páginas oficiales
analisis_equipos = st.Page("pages/Analisis_equipos.py", title="Analisis equipos", icon="📊", default=True)
analisis_jugadores = st.Page("pages/analisis_jugadores.py", title="Analisis jugadores", icon="👥")
tracker_apuestas = st.Page("pages/tracker_apuestas.py", title="Tracker de Apuestas", icon="📈")

pg = st.navigation([analisis_equipos, analisis_jugadores, tracker_apuestas])
pg.run()
