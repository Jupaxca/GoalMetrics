import logging
import re
import streamlit as st
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GoalMetrics | Football Analytics",
    page_icon="icon.png",
    layout="wide"
)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ----------------------------------------------------------------------
# Cliente de Supabase
# ----------------------------------------------------------------------
# IMPORTANTE: NO usamos @st.cache_resource aquí porque ese decorador crea
# un objeto único COMPARTIDO por todos los usuarios de la app (vive a
# nivel de servidor). Como el cliente de supabase-py guarda el token de
# sesión internamente, cachearlo así puede filtrar la sesión de un
# usuario a otro. En su lugar, cacheamos solo la URL/KEY (que sí son
# iguales para todos) y creamos un cliente nuevo por sesión de Streamlit,
# guardándolo en st.session_state.

@st.cache_resource
def get_supabase_config() -> tuple[str, str]:
    return st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]


def get_supabase_client() -> Client:
    if "supabase_client" not in st.session_state:
        url, key = get_supabase_config()
        st.session_state.supabase_client = create_client(url, key)

        # Si ya teníamos tokens guardados en esta sesión (por ejemplo tras
        # un rerun), restauramos la sesión de auth en el cliente nuevo.
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
    """Guarda el usuario y los tokens en session_state tras login/signup."""
    st.session_state.user = res.user
    
    # CORRECCIÓN: Los tokens de acceso y refresco viven dentro de res.session
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
    """Extrae un mensaje legible de una excepción de supabase-py."""
    msg = getattr(e, "message", None)
    return msg if msg else generico


supabase = get_supabase_client()

if "user" not in st.session_state:
    st.session_state.user = None


# ----------------------------------------------------------------------
# Zona pública: login / registro / recuperación de clave
# ----------------------------------------------------------------------
if st.session_state.user is None:
    st.markdown("## 🔐 GoalMetrics · Acceso de Usuarios")

    tab1, tab2, tab3 = st.tabs(["Iniciar Sesión", "Registrarse", "Recuperar Clave"])

    # --- Iniciar sesión ---
    with tab1:
        with st.form("login_form"):
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
        st.write("Ingresa tu correo y te enviaremos un enlace de recuperación.")
        with st.form("reset_form"):
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
                            # Ajusta redirect_to a la URL real de tu app en
                            # producción para que el enlace del correo
                            # regrese al lugar correcto.
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
