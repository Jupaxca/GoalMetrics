import streamlit as st

st.markdown("## 👤 Mi Perfil de Usuario")
st.caption("Administra tu información personal y los datos de tu cuenta en GoalMetrics.")

if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Por favor inicia sesión para ver tu perfil.")
    st.stop()

user = st.session_state.user
user_metadata = getattr(user, "user_metadata", {})
nombre_actual = user_metadata.get("display_name", "")
email_actual = getattr(user, "email", "No disponible")
user_id = getattr(user, "id", "No disponible")

st.markdown("---")
st.markdown("### 📋 Información de la Cuenta")
st.info(f"**Correo Electrónico:** {email_actual}")
st.text(f"ID de Usuario: {user_id}")

st.markdown("---")
st.markdown("### ✏️ Configuración de Identidad")

with st.form("form_cambiar_nombre"):
    nuevo_nombre = st.text_input("Nombre o Apodo en la App", value=nombre_actual, placeholder="Ej. Juan")
    
    if st.form_submit_button("Guardar Cambios", use_container_width=True):
        try:
            # Reutilizamos el cliente ya autenticado de la sesión
            supabase = st.session_state.supabase_client
            
            # Actualizar metadatos en Supabase
            res = supabase.auth.update_user({"data": {"display_name": nuevo_nombre}})
            
            if res.user:
                st.session_state.user = res.user
            
            st.success("¡Nombre actualizado con éxito! Cambia de sección en el menú para ver tu nuevo saludo.")
        except Exception as e:
            st.error(f"Error al actualizar el perfil: {e}")
