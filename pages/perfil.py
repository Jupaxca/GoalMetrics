import streamlit as st

st.markdown("## 👤 Mi Perfil de Usuario")
st.caption("Administra tu información personal, tu identidad y la seguridad de tu cuenta.")

if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Por favor inicia sesión para ver tu perfil.")
    st.stop()

if "supabase_client" not in st.session_state:
    st.warning("No se encontró la conexión a la base de datos. Vuelve a la página principal.")
    st.stop()

supabase = st.session_state.supabase_client

user = st.session_state.user
user_metadata = getattr(user, "user_metadata", {})
nombre_actual = user_metadata.get("display_name", "")
email_actual = getattr(user, "email", "No disponible")
user_id = getattr(user, "id", "No disponible")


def sincronizar_tokens(auth_response) -> None:
    """Si la respuesta de Supabase trae una sesión nueva (tokens), la
    guardamos en session_state para que get_supabase_client() (en app.py)
    pueda restaurarla correctamente si el cliente se recrea más adelante."""
    session = getattr(auth_response, "session", None)
    if session and getattr(session, "access_token", None):
        st.session_state.supabase_tokens = {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        }


# Tarjeta con información de la cuenta
st.markdown("---")
st.markdown("### 📋 Información de la Cuenta")
st.info(f"**Correo Electrónico:** {email_actual}")
st.text(f"ID de Usuario: {user_id}")

# --- SECCIÓN 1: NOMBRE DE USUARIO (SOLO UNA VEZ Y BLOQUEADO) ---
st.markdown("---")
st.markdown("### ✏️ Nombre de Usuario")

if nombre_actual and nombre_actual.strip() != "":
    st.success(f"Tu nombre de usuario actual es: **{nombre_actual}**")
    st.info("ℹ️ Por seguridad, el nombre de usuario solo se pudo configurar una vez y ya se encuentra bloqueado definitivamente.")
else:
    st.warning("⚠️ Aún no has configurado tu nombre de usuario. Solo podrás guardarlo **una única vez** y después no se podrá cambiar.")
    with st.form("form_fijar_nombre"):
        nuevo_nombre = st.text_input("Nombre o Apodo en la App", placeholder="Ej. Juan")

        if st.form_submit_button("Guardar Nombre (Definitivo)", use_container_width=True):
            if not nuevo_nombre or nuevo_nombre.strip() == "":
                st.error("El nombre no puede estar vacío.")
            else:
                try:
                    res = supabase.auth.update_user({"data": {"display_name": nuevo_nombre.strip()}})
                    sincronizar_tokens(res)

                    if res.user:
                        st.session_state.user = res.user

                    st.success("¡Nombre guardado correctamente! Cambia de sección en el menú para ver tu saludo reflejado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar el nombre: {e}")

# --- SECCIÓN 2: CAMBIO DE CONTRASEÑA (LAS VECES QUE QUIERAN) ---
st.markdown("---")
st.markdown("### 🔑 Cambiar Contraseña")
st.caption("Puedes actualizar tu contraseña cuantas veces lo necesites.")

with st.form("form_cambiar_password"):
    old_password = st.text_input("Contraseña actual", type="password", placeholder="Ingresa tu clave actual")
    new_password = st.text_input("Nueva contraseña (mínimo 6 caracteres)", type="password", placeholder="Nueva clave")
    confirm_password = st.text_input("Confirma la nueva contraseña", type="password", placeholder="Repite la nueva clave")

    if st.form_submit_button("Actualizar Contraseña", use_container_width=True):
        if not old_password or not new_password or not confirm_password:
            st.error("Por favor completa todos los campos de contraseña.")
        elif new_password != confirm_password:
            st.error("Las nuevas contraseñas no coinciden.")
        elif len(new_password) < 6:
            st.error("La nueva contraseña debe tener al menos 6 caracteres.")
        else:
            try:
                # Paso 1: Validar que la contraseña actual sea correcta.
                # OJO: sign_in_with_password crea una sesión NUEVA en el
                # cliente; sincronizamos sus tokens para no perder la
                # sesión activa cuando el cliente se reconstruya después.
                try:
                    verify_res = supabase.auth.sign_in_with_password(
                        {"email": email_actual, "password": old_password}
                    )
                    sincronizar_tokens(verify_res)
                except Exception:
                    st.error("La contraseña actual es incorrecta. Verifícala e intenta de nuevo.")
                    st.stop()

                # Paso 2: Cambiar la contraseña (permite hacerlo las veces que desees)
                res = supabase.auth.update_user({"password": new_password})
                sincronizar_tokens(res)
                if res.user:
                    st.session_state.user = res.user

                st.success("¡Contraseña actualizada con éxito! Ya puedes usar tu nueva clave.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al actualizar la contraseña: {e}")
