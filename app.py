import streamlit as st
from supabase import create_client, Client

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="icon.png",
    layout="wide"
)

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

if 'user' not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown("## 🔐 GoalMetrics · Acceso de Usuarios")
    
    tab1, tab2, tab3 = st.tabs(["Iniciar Sesión", "Registrarse", "Recuperar Clave"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Correo electrónico")
            password = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Entrar", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.success("¡Bienvenido!")
                    st.rerun()
                except Exception as e:
                    st.error("Credenciales incorrectas.")
                    
    with tab2:
        with st.form("signup_form"):
            email_su = st.text_input("Correo electrónico")
            password_su = st.text_input("Contraseña (mínimo 6 caracteres)", type="password")
            if st.form_submit_button("Crear cuenta", use_container_width=True):
                try:
                    supabase.auth.sign_up({"email": email_su, "password": password_su})
                    st.success("¡Cuenta creada! Revisa tu correo para confirmar.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.write("Ingresa tu correo y te enviaremos un enlace de recuperación.")
        with st.form("reset_form"):
            email_reset = st.text_input("Correo de la cuenta")
            if st.form_submit_button("Enviar enlace de recuperación", use_container_width=True):
                try:
                    supabase.auth.reset_password_for_email(email_reset)
                    st.success("¡Correo enviado! Revisa tu bandeja de entrada.")
                except Exception as e:
                    st.error(f"Error: {e}")
    st.stop()

if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

# Definir las páginas oficiales
analisis_equipos = st.Page("pages/Analisis_equipos.py", title="Analisis equipos", icon="📊", default=True)
analisis_jugadores = st.Page("pages/analisis_jugadores.py", title="Analisis jugadores", icon="👥")
tracker_apuestas = st.Page("pages/tracker_apuestas.py", title="Tracker de Apuestas", icon="📈")
coach_ia = st.Page("pages/coach_ia.py", title="Coach IA", icon="🤖")

pg = st.navigation([analisis_equipos, analisis_jugadores, tracker_apuestas, coach_ia])
pg.run()
