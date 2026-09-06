import logging
import re
import streamlit as st
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GoalMetrics | Football Analytics Pro",
    page_icon="⚽",
    layout="wide"
)

# ---------------------------------------------------------------------
# CSS AVANZADO (Estilo SaaS de Élite + Banner Hero Dinámico + UI/UX Pro)
# ---------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. Tipografía Global e Interfaz Moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* 2. Ocultar elementos nativos manteniendo la barra superior móvil activa */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* 3. Estilo global para la barra lateral (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* 4. Banner Hero Principal (Estilo SaaS de Alto Rendimiento) */
    .hero-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 45px 35px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    
    .hero-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #10b981, #6366f1);
    }

    /* 5. Tarjetas y contenedores con efectos Glassmorphism / Hover */
    .saas-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        transition: all 0.25s ease-in-out;
    }
    
    .saas-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.1);
    }

    /* 6. Estilización Avanzada de Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0b0f19;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1f2937;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px;
        color: #9ca3af;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid #1f2937;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%) !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    }

    /* 7. Botones principales estilo Trading Pro */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
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
# Zona pública: login / registro / recuperación de clave con Banner Hero Pro
# ----------------------------------------------------------------------
if st.session_state.user is None:
    st.markdown("""
    <div class="hero-box">
        <h1 style="font-weight: 800; font-size: 36px; margin-bottom: 8px; letter-spacing: -0.5px;">⚽ GoalMetrics <span style="color: #3b82f6;">Pro</span></h1>
        <p style="font-size: 16px; color: #93c5fd; max-width: 600px; margin: 0 auto; line-height: 1.5;">
            Plataforma híbrida avanzada de modelado estadístico, Machine Learning y análisis de valor en mercados deportivos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l1, col_l2, col_l3 = st.columns([1, 2.2, 1])
    with col_l2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🔑 Iniciar Sesión", "📝 Registrarse", "🔄 Recuperar"])

        # --- Iniciar sesión ---
        with tab1:
            with st.form("login_form"):
                st.subheader("Acceso a la Terminal")
                email = st.text_input("Correo electrónico")
                password = st.text_input("Contraseña", type="password")
                submitted = st.form_submit_button("Entrar al Sistema", use_container_width=True)

                if submitted:
                    if not email or not password:
                        st.error("Completa correo y contraseña.")
                    elif not EMAIL_REGEX.match(email):
                        st.error("Ingresa un correo electrónico válido.")
                    else:
                        with st.spinner("Autenticando credenciales seguras..."):
                            try:
                                res = st.session_state.supabase_client.auth.sign_in_with_password(
                                    {"email": email, "password": password}
                                )
                                if res.user is None:
                                    st.error("No se pudo iniciar sesión. Intenta de nuevo.")
                                else:
                                    guardar_sesion(res)
                                    st.success("¡Bienvenido al sistema!")
                                    st.rerun()
                            except Exception as e:
                                logger.warning("Fallo de login para %s: %s", email, e)
                                st.error(mensaje_error_supabase(e, "Credenciales incorrectas."))

        # --- Registro ---
        with tab2:
            with st.form("signup_form"):
                st.subheader("Nueva Cuenta Pro")
                email_su = st.text_input("Correo electrónico")
                password_su = st.text_input("Contraseña (mínimo 6 caracteres)", type="password")
                password_su_confirm = st.text_input("Confirmar contraseña", type="password")
                submitted_su = st.form_submit_button("Crear cuenta de analista", use_container_width=True)

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
                        with st.spinner("Registrando perfil..."):
                            try:
                                st.session_state.supabase_client.auth.sign_up(
                                    {"email": email_su, "password": password_su}
                                )
                                st.success("¡Cuenta creada! Revisa tu bandeja para confirmar.")
                            except Exception as e:
                                logger.warning("Fallo de signup para %s: %s", email_su, e)
                                st.error(mensaje_error_supabase(e, "No se pudo crear la cuenta."))

        # --- Recuperar clave ---
        with tab3:
            with st.form("reset_form"):
                st.subheader("Restablecer Acceso")
                st.write("Ingresa tu correo registrado para recibir el enlace de recuperación.")
                email_reset = st.text_input("Correo de la cuenta")
                submitted_reset = st.form_submit_button(
                    "Enviar instrucciones", use_container_width=True
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
                                st.success("¡Correo enviado con éxito!")
                            except Exception as e:
                                logger.warning("Fallo de reset para %s: %s", email_reset, e)
                                st.error(mensaje_error_supabase(e, "No se pudo enviar el correo."))
        st.markdown('</div>', unsafe_allow_html=True)

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
