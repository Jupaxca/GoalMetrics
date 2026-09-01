import streamlit as st

st.set_page_config(
    page_title="GoalMetrics | Inicio",
    page_icon="⚽",
    layout="wide"
)

# --- ESTILOS MODERNOS (CON HEADER ACTIVO PARA MÓVILES) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0F19;
    color: #F3F4F6;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* NOTA: El header se mantiene activo para que aparezca el botón de menú hamburguesa en celulares */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

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

.card-home {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease-in-out;
    height: 100%;
}
.card-home:hover {
    border-color: #3B82F6;
    box-shadow: 0 6px 25px rgba(59, 130, 246, 0.2);
}
</style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE SESIÓN / INICIO DE SESIÓN ---
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
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
        with st.form("form_login"):
            st.subheader("🔑 Acceso al Sistema")
            usuario_input = st.text_input("Usuario")
            password_input = st.text_input("Contraseña", type="password")
            submit_login = st.form_submit_button("Iniciar Sesión", use_container_width=True)
            
            if submit_login:
                # Puedes ajustar o conectar esto con tus validaciones previas si las tenías parametrizadas
                if usuario_input.strip() != "":
                    st.session_state.autenticado = True
                    st.session_state.usuario = usuario_input
                    st.rerun()
                else:
                    st.error("Por favor ingresa un usuario válido.")
    st.stop()

# --- BARRA LATERAL CON CONTROL DE SESIÓN ---
st.sidebar.success(f"👤 Sesión activa: {st.session_state.get('usuario', 'Usuario')}")
if st.sidebar.button("Cerrar Sesión", use_container_width=True):
    st.session_state.autenticado = False
    st.rerun()

st.sidebar.markdown("---")

# --- CONTENIDO DE LA PÁGINA PRINCIPAL (POST-LOGIN) ---
st.markdown("""
<div class="hero-box">
    <h1>⚽ GoalMetrics Pro</h1>
    <p style="font-size: 18px; color: #93c5fd; margin-top: 10px;">
        Plataforma Avanzada de Análisis Estadístico, Modelos Híbridos y Value Bets
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Bienvenido al Centro de Mando")
st.markdown("Utiliza el menú lateral izquierdo (o el botón superior en tu celular) para navegar entre los diferentes módulos de análisis:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card-home">
        <h3>📊 Análisis de Equipos</h3>
        <p style="color: #9ca3af; font-size: 14px; margin-top: 10px;">
            Simulaciones estocásticas con Poisson y Dixon-Coles, Ensemble XGBoost, semáforos de confiabilidad, matrices de resultados exactos y Value Bets colectivas con escudos oficiales reales.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card-home">
        <h3>👤 Análisis de Jugadores</h3>
        <p style="color: #9ca3af; font-size: 14px; margin-top: 10px;">
            Evaluación individual de Player Props (goles, tiros a puerta, asistencias y faltas), perfiles de atributos en radar, fotos oficiales de los jugadores y gestión de bankroll con Half-Kelly.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("GoalMetrics Engine Pro | Panel de Control Principal")
