import logging
import re
import streamlit as st
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="⚽",
    layout="wide"
)

# ---------------------------------------------------------------------
# CSS AVANZADO (Estilo SaaS Profesional + Banner Hero + Menú Móvil Activo)
# ---------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. Importar tipografía moderna de la industria (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* 2. Ocultar elementos nativos pero MANTENER el header activo para móviles */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} -> Activo para que aparezca el botón hamburguesa en celulares */
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 3. Estilo global para la barra lateral (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* 4. Banner Hero Principal (Estilo superior que te gusta) */
    .hero-box {
        background: linear-gradient(135deg, #3B82F6 0%, #111827 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
    }

    /* 5. Tarjetas y contenedores estilo SaaS */
    .saas-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* 6. Estilización de Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b0f19;
        padding: 4px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px;
        color: #9ca3af;
        padding: 10px 20px;
        font-weight: 500;
        border: 1px solid #1f2937;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%) !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
    }

    /* 7. Botones principales */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ----------------------------------------------------------------------
# Cliente de Supabase
# ----------------------------------------------------------------------
@st.cache_resource
def get_supabase_config() -> tuple[str, str]:
    return st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]


def get_supabase_client() -> Client:
    if "supabase_client" not in st.session_state:
        url, key = get_supabase_config()
        st.session_state.supabase_client = create_client(url, key)

        tokens = st.session_state.get("supabase_tokens")
        if tokens:
            try:
                st.session_state.supabase_client.auth.set_session(
                    tokens["access_token"], tokens["refresh_token"]
                )
            except Exception:
                logger.exception("No se pudo restaurar la sesión de Supabase")
                st.session_state.supabase_tokens = None

    return st.session_state.supabase_client


def guardar_sesion(res) -> None:
    st.session_state.user = res.user
    session_data = getattr(res, "session", None)
    if session_data:
        st.session_state.supabase_tokens = {
            "access_token": session_data.access_token,
            "refresh_token": session_data.refresh_token,
        }


def cerrar_sesion() -> None:
    try:
        st.session_state.supabase_client.auth.sign_out()
    except Exception:
        logger.exception("Error al cerrar sesión en Supabase")
    st.session_state.user = None
    st.session_state.supabase_tokens = None


def mensaje_error_supabase(e: Exception, generico: str) -> str:
    msg = getattr(e, "message", None)
    return msg if msg else generico


supabase = get_supabase_client()

if "user" not in st.session_state:
    st.session_state.user = None


# ----------------------------------------------------------------------
# Zona pública: login / registro / recuperación de clave con Banner Hero
# ----------------------------------------------------------------------
if st.session_state.user is None:
    # Banner Hero Superior que querías conservar
    st.markdown("""
    <div class="hero-box">
        <h1>⚽ GoalMetrics Pro</h1>
        <p style="font-size: 18px; color: #93c5fd; margin-top: 10px;">
            Inicia sesión para acceder a tu centro de análisis avanzado
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        tab1, tab2, tab3 = st.tabs(["Iniciar Sesión", "Registrarse", "Recuperar Clave"])

        # --- Iniciar sesión ---
        with tab1:
            with st.form("login_form"):
                st.subheader("🔑 Acceso al Sistema")
                email = st.text_input("Correo electrónico")
                password = st.text_input("Contraseña", type="password")
                submitted = st.form_submit_button("Entrar", use_container_width=True)

                if submitted:
                    if not email or not password:
                        st.error("Completa correo y contraseña.")
                    elif not EMAIL_REGEX.match(email):
                        st.error("Ingresa un correo electrónico válido.")
                    else:
                        with st.spinner("Verificando credenciales..."):
                            try:
                                res = st.session_state.supabase_client.auth.sign_in_with_password(
                                    {"email": email, "password": password}
                                )
                                if res.user is None:
                                    st.error("No se pudo iniciar sesión. Intenta de nuevo.")
                                else:
                                    guardar_sesion(res)
                                    st.success("¡Bienvenido!")
                                    st.rerun()
                            except Exception as e:
                                logger.warning("Fallo de login para %s: %s", email, e)
                                st.error(mensaje_error_supabase(e, "Credenciales incorrectas."))

        # --- Registro ---
        with tab2:
            with st.form("signup_form"):
                st.subheader("📝 Crear Cuenta Nueva")
                email_su = st.text_input("Correo electrónico")
                password_su = st.text_input("Contraseña (mínimo 6 caracteres)", type="password")
                password_su_confirm = st.text_input("Confirmar contraseña", type="password")
                submitted_su = st.form_submit_button("Crear cuenta", use_container_width=True)

                if submitted_su:
                    if not email_su or not password_su:
                        st.error("Completa todos los campos.")
                    elif not EMAIL_REGEX.match(email_su):
                        st.error("Ingresa un correo electrónico válido.")
                    elif len(password_su) < 6:
                        st.error("La contraseña debe tener al menos 6 caracteres.")
                    elif password_su != password_su_confirm:
                        st.error("Las contraseñas no coinciden. Por favor, revísalas.")
                    else:
                        with st.spinner("Creando cuenta..."):
                            try:
                                st.session_state.supabase_client.auth.sign_up(
                                    {"email": email_su, "password": password_su}
                                )
                                st.success("¡Cuenta creada! Revisa tu correo para confirmar.")
                            except Exception as e:
                                logger.warning("Fallo de signup para %s: %s", email_su, e)
                                st.error(mensaje_error_supabase(e, "No se pudo crear la cuenta."))

        # --- Recuperar clave ---
        with tab3:
            with st.form("reset_form"):
                st.subheader("🔄 Recuperar Contraseña")
                st.write("Ingresa tu correo y te enviaremos un enlace de recuperación.")
                email_reset = st.text_input("Correo de la cuenta")
                submitted_reset = st.form_submit_button(
                    "Enviar enlace de recuperación", use_container_width=True
                )

                if submitted_reset:
                    if not email_reset or not EMAIL_REGEX.match(email_reset):
                        st.error("Ingresa un correo electrónico válido.")
                    else:
                        with st.spinner("Enviando enlace..."):
                            try:
                                st.session_state.supabase_client.auth.reset_password_for_email(
                                    email_reset,
                                    {"redirect_to": st.secrets.get("APP_URL", "")},
                                )
                                st.success("¡Correo enviado! Revisa tu bandeja de entrada.")
                            except Exception as e:
                                logger.warning("Fallo de reset para %s: %s", email_reset, e)
                                st.error(mensaje_error_supabase(e, "No se pudo enviar el correo."))

    st.stop()


# ----------------------------------------------------------------------
# Zona privada (usuario logueado)
# ----------------------------------------------------------------------
user_metadata = getattr(st.session_state.user, "user_metadata", {}) or {}
nombre_mostrado = user_metadata.get(
    "display_name", getattr(st.session_state.user, "email", "Analista")
)

st.sidebar.markdown(f"👋 Hola, **{nombre_mostrado}**")

if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
    cerrar_sesion()
    st.rerun()

analisis_equipos = st.Page("pages/Analisis_equipos.py", title="Analisis equipos", icon="📊", default=True)
analisis_jugadores = st.Page("pages/analisis_jugadores.py", title="Analisis jugadores", icon="👥")
tracker_apuestas = st.Page("pages/tracker_apuestas.py", title="Tracker de Apuestas", icon="📈")
coach_ia = st.Page("pages/coach_ia.py", title="Coach IA", icon="🤖")
perfil_usuario = st.Page("pages/perfil.py", title="Mi Perfil", icon="👤")

pg = st.navigation([analisis_equipos, analisis_jugadores, tracker_apuestas, coach_ia, perfil_usuario])
pg.run()
