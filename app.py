import logging
import re
import streamlit as st
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GoalMetrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# CSS GLOBAL — sistema visual único (login + todas las páginas)
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Hero login */
    .hero-box {
        background: linear-gradient(145deg, #111827 0%, #0b0f19 100%);
        padding: 40px 32px;
        border-radius: 16px;
        color: white;
        text-align: center;
        border: 1px solid #1f2937;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #10b981);
    }

    /* Cards */
    .saas-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    /* Header partido / jugador */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }

    /* Veredicto / insight corto */
    .veredicto-box {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-left: 3px solid #3b82f6;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0 16px 0;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #e5e7eb;
    }

    /* Análisis largo (dentro de expander) */
    .analisis-dinamico-box {
        background-color: #0b0f19;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #d1d5db;
    }

    /* Semáforo */
    .pill-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .pill-yellow {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .pill-red {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    /* Top value pick */
    .top-pick-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, #111827 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }
    .top-pick-box h3 {
        margin: 0 0 8px 0;
        font-size: 1.05rem;
        color: #f3f4f6;
    }

    /* Value bet row */
    .value-row {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .value-row-positive {
        border-color: rgba(16, 185, 129, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b0f19;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px;
        color: #9ca3af;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #1f2937;
    }
    .stTabs [aria-selected="true"] {
        background: #1f2937 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }

    /* Primary buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
    }

    /* Metrics más compactas */
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px !important;
    }

    /* Caption / secondary text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9ca3af !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ----------------------------------------------------------------------
# Supabase
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
# Login / registro / recuperar
# ----------------------------------------------------------------------
if st.session_state.user is None:
    st.markdown(
        """
    <div class="hero-box">
        <h1 style="font-weight: 800; font-size: 32px; margin-bottom: 10px; letter-spacing: -0.4px;">
            GoalMetrics
        </h1>
        <p style="font-size: 15px; color: #93c5fd; max-width: 520px; margin: 0 auto; line-height: 1.55;">
            Análisis de equipos y jugadores con modelo estadístico.
            Probabilidades, valor esperado y stake — sin ruido.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_l1, col_l2, col_l3 = st.columns([1, 2.1, 1])
    with col_l2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Iniciar sesión", "Registrarse", "Recuperar"])

        with tab1:
            with st.form("login_form"):
                st.subheader("Entrar")
                email = st.text_input("Correo electrónico")
                password = st.text_input("Contraseña", type="password")
                submitted = st.form_submit_button("Entrar", use_container_width=True, type="primary")

                if submitted:
                    if not email or not password:
                        st.error("Completa correo y contraseña.")
                    elif not EMAIL_REGEX.match(email):
                        st.error("Correo no válido.")
                    else:
                        with st.spinner("Entrando..."):
                            try:
                                res = st.session_state.supabase_client.auth.sign_in_with_password(
                                    {"email": email, "password": password}
                                )
                                if res.user is None:
                                    st.error("No se pudo iniciar sesión.")
                                else:
                                    guardar_sesion(res)
                                    st.success("Bienvenido.")
                                    st.rerun()
                            except Exception as e:
                                logger.warning("Fallo de login para %s: %s", email, e)
                                st.error(mensaje_error_supabase(e, "Credenciales incorrectas."))

        with tab2:
            with st.form("signup_form"):
                st.subheader("Crear cuenta")
                email_su = st.text_input("Correo electrónico")
                password_su = st.text_input("Contraseña (mín. 6 caracteres)", type="password")
                password_su_confirm = st.text_input("Confirmar contraseña", type="password")
                submitted_su = st.form_submit_button("Crear cuenta", use_container_width=True, type="primary")

                if submitted_su:
                    if not email_su or not password_su:
                        st.error("Completa todos los campos.")
                    elif not EMAIL_REGEX.match(email_su):
                        st.error("Correo no válido.")
                    elif len(password_su) < 6:
                        st.error("La contraseña debe tener al menos 6 caracteres.")
                    elif password_su != password_su_confirm:
                        st.error("Las contraseñas no coinciden.")
                    else:
                        with st.spinner("Creando cuenta..."):
                            try:
                                st.session_state.supabase_client.auth.sign_up(
                                    {"email": email_su, "password": password_su}
                                )
                                st.success("Cuenta creada. Revisa tu correo para confirmar.")
                            except Exception as e:
                                logger.warning("Fallo de signup para %s: %s", email_su, e)
                                st.error(mensaje_error_supabase(e, "No se pudo crear la cuenta."))

        with tab3:
            with st.form("reset_form"):
                st.subheader("Recuperar acceso")
                st.caption("Te enviaremos un enlace a tu correo.")
                email_reset = st.text_input("Correo de la cuenta")
                submitted_reset = st.form_submit_button(
                    "Enviar enlace", use_container_width=True, type="primary"
                )

                if submitted_reset:
                    if not email_reset or not EMAIL_REGEX.match(email_reset):
                        st.error("Correo no válido.")
                    else:
                        with st.spinner("Enviando..."):
                            try:
                                st.session_state.supabase_client.auth.reset_password_for_email(
                                    email_reset,
                                    {"redirect_to": st.secrets.get("APP_URL", "")},
                                )
                                st.success("Correo enviado.")
                            except Exception as e:
                                logger.warning("Fallo de reset para %s: %s", email_reset, e)
                                st.error(mensaje_error_supabase(e, "No se pudo enviar el correo."))
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ----------------------------------------------------------------------
# Zona privada
# ----------------------------------------------------------------------
user_metadata = getattr(st.session_state.user, "user_metadata", {}) or {}
nombre_mostrado = user_metadata.get(
    "display_name", getattr(st.session_state.user, "email", "Analista")
)

st.sidebar.markdown(
    f"""
<div style="padding: 4px 0 12px 0;">
  <div style="font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em;">GoalMetrics</div>
  <div style="font-size: 14px; font-weight: 600; margin-top: 4px;">{nombre_mostrado}</div>
</div>
""",
    unsafe_allow_html=True,
)

if st.sidebar.button("Cerrar sesión", use_container_width=True):
    cerrar_sesion()
    st.rerun()

# Definición correcta de las páginas apuntando a la carpeta pages/
analisis_equipos = st.Page(
    "pages/Analisis_equipos.py",
    title="Equipos",
    icon="📊",
    default=True,
)
analisis_jugadores = st.Page(
    "pages/analisis_jugadores.py",
    title="Jugadores",
    icon="👥",
)
tracker_apuestas = st.Page(
    "pages/tracker_apuestas.py",
    title="Tracker",
    icon="📈",
)
coach_ia = st.Page(
    "pages/coach_ia.py",
    title="Coach IA",
    icon="🤖",
)
perfil_usuario = st.Page(
    "pages/perfil.py",
    title="Perfil",
    icon="👤",
)

pg = st.navigation(
    [analisis_equipos, analisis_jugadores, tracker_apuestas, coach_ia, perfil_usuario]
)
pg.run()
