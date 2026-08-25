import streamlit as st
from supabase import create_client, Client

st.markdown("## 👤 Mi Perfil de Usuario")
st.caption("Administra tu información personal y los datos de tu cuenta en GoalMetrics.")

# Verificación de seguridad por si entran sin sesión
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Por favor inicia sesión para ver tu perfil.")
    st.stop()

user = st.session_state.user
user_metadata = getattr(user, "user_metadata", {})
nombre_actual = user_metadata.get("display_name", "")
email_actual = getattr(user, "email", "No disponible")
user_id = getattr(user, "id", "No disponible")

# Tarjeta con información de la cuenta
st.markdown("---")
st.markdown("### 📋 Información de la Cuenta")
st.info(f"**Correo Electrónico:** {email_actual}")
st.text(f"ID de Usuario: {user_id}")

# Formulario para cambiar el nombre de usuario
st.markdown("---")
st.markdown("### ✏️ Configuración de Identidad")

with st.form("form_cambiar_nombre"):
    nuevo_nombre = st.text_input("Nombre o Apodo en la App", value=nombre_actual, placeholder="Ej. Juan")
    
    if st.form_submit_button("Guardar Cambios", use_container_width=True):
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
            supabase = create_client(url, key)
            
            # Actualizar metadatos en Supabase
            supabase.auth.update_user({"data": {"display_name": nuevo_nombre}})
            
            st.success("¡Nombre actualizado con éxito! Vuelve a hacer clic en cualquier sección del menú para ver tu nuevo saludo reflejado.")
        except Exception as e:
            st.error(f"Error al actualizar el perfil: {e}")
