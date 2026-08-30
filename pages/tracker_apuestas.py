import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="Tracker Pro | GoalMetrics", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------
# Cliente de Supabase
# ----------------------------------------------------------------------
# IMPORTANTE: ya NO creamos un cliente propio con @st.cache_resource aquí.
# Eso generaba un cliente COMPARTIDO entre todos los usuarios de la app
# (riesgo de mezclar sesiones). En su lugar, reutilizamos el cliente que
# app.py ya creó y guardó en st.session_state para esta sesión.

if "user" not in st.session_state or st.session_state.user is None:
    st.warning("Sesión no detectada. Regresa a la página principal e inicia sesión.")
    st.stop()

if "supabase_client" not in st.session_state:
    st.warning("No se encontró la conexión a la base de datos. Vuelve a la página principal.")
    st.stop()

supabase = st.session_state.supabase_client

user = st.session_state.user
user_id = user.id

st.markdown("## 📈 Tracker de Apuestas & Análisis Pro")
st.caption("Simples · Combinadas · Bank · ROI · Historial")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Configuración")
    if "bank_inicial" not in st.session_state:
        st.session_state.bank_inicial = 100.0
    bank_inicial = st.number_input(
        "Bank Inicial ($)",
        min_value=0.0,
        value=float(st.session_state.bank_inicial),
        step=10.0,
        key="input_bank",
    )
    st.session_state.bank_inicial = bank_inicial

    st.markdown("---")
    st.header("➕ Nueva Apuesta")

    tipo_apuesta = st.radio("Tipo", ["Simple", "Combinada"], horizontal=True)

    with st.form("nueva_apuesta", clear_on_submit=True):
        if tipo_apuesta == "Simple":
            evento = st.text_input("Evento / Partido")
            seleccion = st.text_input("Selección (ej: Más de 2.5, BTTS Sí)")
            opciones_mercado = [
                "Ganador (1X2)", "Doble Oportunidad", "Ambos Marcan (BTTS)",
                "Over/Under Goles", "Over/Under Córners", "Hándicap Asiático",
                "Hándicap Europeo", "Tarjetas", "Resultado Exacto", "Player Props", "Otro",
            ]
            mercado = st.selectbox("Mercado", opciones_mercado)
            cuota = st.number_input("Cuota", min_value=1.01, value=1.90, step=0.01, format="%.2f")
            n_legs = 1
            detalle_legs = ""
        else:
            # COMBINADA
            evento = st.text_input("Nombre de la combinada", placeholder="Ej: Triple del sábado")
            st.caption("Escribe cada leg en una línea: Partido | Selección | Cuota")
            detalle_legs = st.text_area(
                "Legs de la combinada",
                height=120,
                placeholder="Real Madrid vs Sevilla | Más 2.5 | 1.85\nBarcelona vs Girona | BTTS Sí | 1.70\n...",
            )
            mercado = "Combinada"
            seleccion = "Ver legs"
            cuota = st.number_input(
                "Cuota total de la combinada",
                min_value=1.01,
                value=5.00,
                step=0.01,
                format="%.2f",
                help="La cuota final que te da la casa",
            )
            n_legs = st.number_input("Nº de legs", min_value=2, max_value=20, value=3, step=1)

        stake = st.number_input("Stake ($)", min_value=0.5, value=10.0, step=0.5)

        if st.form_submit_button("Guardar Apuesta", type="primary"):
            if tipo_apuesta == "Simple":
                if not evento.strip() or not seleccion.strip():
                    st.error("Evento y selección son obligatorios.")
                else:
                    try:
                        data = {
                            "user_id": user_id,
                            "evento": evento.strip(),
                            "seleccion": seleccion.strip(),
                            "mercado": mercado,
                            "cuota": float(cuota),
                            "stake": float(stake),
                            "estado": "Pendiente",
                            "pnl": 0.0,
                            "fecha": str(datetime.date.today()),
                            "created_at": datetime.datetime.now().isoformat(),
                            "tipo": "Simple",
                            "n_legs": 1,
                            "detalle": "",
                        }
                        supabase.table("apuestas").insert(data).execute()
                        st.success("Apuesta simple registrada.")
                        st.rerun()
                    except Exception as e:
                        # Fallback si la tabla no tiene tipo/n_legs/detalle
                        try:
                            data_min = {
                                "user_id": user_id,
                                "evento": evento.strip(),
                                "seleccion": seleccion.strip(),
                                "mercado": mercado,
                                "cuota": float(cuota),
                                "stake": float(stake),
                                "estado": "Pendiente",
                                "pnl": 0.0,
                                "fecha": str(datetime.date.today()),
                                "created_at": datetime.datetime.now().isoformat(),
                            }
                            supabase.table("apuestas").insert(data_min).execute()
                            st.success("Apuesta registrada (campos extra no disponibles en BD).")
                            st.rerun()
                        except Exception as e2:
                            st.error(f"Error al guardar: {e2}")
            else:
                # Combinada
                if not evento.strip():
                    st.error("Pon un nombre a la combinada.")
                elif not detalle_legs.strip():
                    st.error("Añade al menos los legs en el cuadro de texto.")
                else:
                    seleccion_txt = f"Combinada {int(n_legs)} legs"
                    detalle = detalle_legs.strip()
                    try:
                        data = {
                            "user_id": user_id,
                            "evento": evento.strip(),
                            "seleccion": seleccion_txt,
                            "mercado": "Combinada",
                            "cuota": float(cuota),
                            "stake": float(stake),
                            "estado": "Pendiente",
                            "pnl": 0.0,
                            "fecha": str(datetime.date.today()),
                            "created_at": datetime.datetime.now().isoformat(),
                            "tipo": "Combinada",
                            "n_legs": int(n_legs),
                            "detalle": detalle,
                        }
                        supabase.table("apuestas").insert(data).execute()
                        st.success("Combinada registrada.")
                        st.rerun()
                    except Exception as e:
                        try:
                            data_min = {
                                "user_id": user_id,
                                "evento": evento.strip(),
                                "seleccion": f"{seleccion_txt} | {detalle[:200]}",
                                "mercado": "Combinada",
                                "cuota": float(cuota),
                                "stake": float(stake),
                                "estado": "Pendiente",
                                "pnl": 0.0,
                                "fecha": str(datetime.date.today()),
                                "created_at": datetime.datetime.now().isoformat(),
                            }
                            supabase.table("apuestas").insert(data_min).execute()
                            st.success("Combinada registrada (detalle en selección).")
                            st.rerun()
                        except Exception as e2:
                            st.error(f"Error al guardar: {e2}")

# ====================== CARGAR ======================
try:
    response = supabase.table("apuestas").select("*").eq("user_id", user_id).execute()
    todas = response.data if response.data else []
except Exception as e:
    todas = []
    st.error(f"Error BD: {e}")

df = pd.DataFrame(todas) if todas else pd.DataFrame()

# ====================== CERRAR ======================
st.subheader("🏁 Cerrar Apuesta")

if not df.empty and "estado" in df.columns:
    pendientes = df[df["estado"] == "Pendiente"].copy()
else:
    pendientes = pd.DataFrame()

if not pendientes.empty:
    def label_apuesta(i):
        row = pendientes[pendientes["id"] == i].iloc[0]
        tipo = row.get("tipo") or row.get("mercado") or ""
        return f"{row.get('evento', '?')} | {row.get('seleccion', '?')} @ {row.get('cuota', '?')} ({tipo})"

    cols = st.columns([3, 1.2, 1.2])
    with cols[0]:
        apuesta_id = st.selectbox("Apuesta pendiente", pendientes["id"].tolist(), format_func=label_apuesta)
    with cols[1]:
        resultado = st.selectbox("Resultado", ["Ganada", "Perdida", "Nulo"])
    with cols[2]:
        st.write("")
        st.write("")
        if st.button("Actualizar", use_container_width=True, type="primary"):
            try:
                apuesta = pendientes[pendientes["id"] == apuesta_id].iloc[0]
                stake_val = float(apuesta["stake"])
                cuota_val = float(apuesta["cuota"])
                if resultado == "Ganada":
                    pnl = stake_val * (cuota_val - 1)
                elif resultado == "Perdida":
                    pnl = -stake_val
                else:
                    pnl = 0.0
                supabase.table("apuestas").update({
                    "estado": resultado,
                    "pnl": float(pnl),
                }).eq("id", int(apuesta_id)).eq("user_id", user_id).execute()
                st.success("Apuesta actualizada.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("No hay apuestas pendientes.")

# ====================== DASHBOARD ======================
st.markdown("---")
if df.empty:
    st.write("Sin historial todavía.")
    st.stop()

df["pnl"] = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0)
df["stake"] = pd.to_numeric(df.get("stake", 0), errors="coerce").fillna(0)
df["cuota"] = pd.to_numeric(df.get("cuota", 0), errors="coerce").fillna(0)
df["estado"] = df["estado"].astype(str).str.strip()

df_cerradas = df[df["estado"].isin(["Ganada", "Perdida", "Nulo"])].copy()
total_pnl = df_cerradas["pnl"].sum() if not df_cerradas.empty else 0.0
stake_cerrado = df_cerradas["stake"].sum() if not df_cerradas.empty else 0.0
roi = (total_pnl / stake_cerrado * 100) if stake_cerrado > 0 else 0.0
bank_actual = bank_inicial + total_pnl

solo_gp = df_cerradas[df_cerradas["estado"].isin(["Ganada", "Perdida"])]
total_decididas = len(solo_gp)
ganadas = len(solo_gp[solo_gp["estado"] == "Ganada"])
winrate = (ganadas / total_decididas * 100) if total_decididas > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏦 Bank Actual", f"{bank_actual:,.2f} $", delta=f"{total_pnl:+,.2f} $")
c2.metric("💰 P&L", f"{total_pnl:+,.2f} $")
c3.metric("📊 ROI", f"{roi:+.1f}%")
c4.metric("🎯 Winrate", f"{winrate:.1f}%", delta=f"{ganadas}G / {total_decididas - ganadas}P")

# % combinadas
if "mercado" in df.columns:
    n_comb = len(df[df["mercado"] == "Combinada"])
    st.caption(f"Combinadas en historial: {n_comb} / {len(df)} apuestas")

if not df_cerradas.empty:
    st.subheader("📊 Curva de Bank")
    if "fecha" in df_cerradas.columns:
        df_cerradas["_ord"] = pd.to_datetime(df_cerradas["fecha"], errors="coerce")
    elif "created_at" in df_cerradas.columns:
        df_cerradas["_ord"] = pd.to_datetime(df_cerradas["created_at"], errors="coerce")
    else:
        df_cerradas["_ord"] = range(len(df_cerradas))
    df_curva = df_cerradas.sort_values("_ord")
    df_curva["acumulado"] = bank_inicial + df_curva["pnl"].cumsum()
    st.line_chart(df_curva["acumulado"].values)

# ====================== HISTORIAL ======================
st.markdown("---")
st.subheader("Historial")
f1, f2, f3 = st.columns(3)
with f1:
    estados_filtro = st.multiselect(
        "Estado",
        ["Pendiente", "Ganada", "Perdida", "Nulo"],
        default=["Pendiente", "Ganada", "Perdida", "Nulo"],
    )
with f2:
    mercados_disp = sorted(df["mercado"].dropna().unique().tolist()) if "mercado" in df.columns else []
    mercados_filtro = st.multiselect("Mercado", mercados_disp, default=mercados_disp)
with f3:
    tipo_filtro = st.multiselect("Tipo", ["Simple", "Combinada"], default=["Simple", "Combinada"])

df_hist = df.copy()
if estados_filtro:
    df_hist = df_hist[df_hist["estado"].isin(estados_filtro)]
if mercados_filtro and "mercado" in df_hist.columns:
    df_hist = df_hist[df_hist["mercado"].isin(mercados_filtro)]
# Filtrar combinadas/simples por mercado si no existe columna tipo
if "tipo" in df_hist.columns and tipo_filtro:
    df_hist = df_hist[df_hist["tipo"].isin(tipo_filtro)]
elif "mercado" in df_hist.columns and tipo_filtro:
    if "Combinada" not in tipo_filtro:
        df_hist = df_hist[df_hist["mercado"] != "Combinada"]
    if "Simple" not in tipo_filtro:
        df_hist = df_hist[df_hist["mercado"] == "Combinada"]

cols_mostrar = [c for c in ["id", "fecha", "evento", "seleccion", "mercado", "cuota", "stake", "estado", "pnl", "tipo", "n_legs"] if c in df_hist.columns]
df_hist = df_hist.sort_values(by="id", ascending=False)
st.dataframe(df_hist[cols_mostrar], use_container_width=True, hide_index=True)

# Detalle de combinada
if "detalle" in df.columns or "seleccion" in df.columns:
    with st.expander("🔎 Ver detalle de una combinada"):
        combos = df[df.get("mercado", pd.Series()) == "Combinada"] if "mercado" in df.columns else pd.DataFrame()
        if combos.empty:
            st.caption("No hay combinadas.")
        else:
            cid = st.selectbox("Combinada", combos["id"].tolist(), format_func=lambda i: combos[combos["id"]==i].iloc[0].get("evento", str(i)))
            row = combos[combos["id"] == cid].iloc[0]
            st.write(f"**Cuota total:** {row.get('cuota')} · **Stake:** {row.get('stake')} · **Estado:** {row.get('estado')}")
            det = row.get("detalle") or row.get("seleccion") or ""
            st.text(det)

# ====================== BORRAR ======================
with st.expander("🗑️ Borrar apuesta"):
    if not df.empty:
        def label_b(i):
            r = df[df["id"] == i].iloc[0]
            return f"ID {i} · {r.get('evento','?')} · {r.get('estado','?')}"
        apuesta_borrar = st.selectbox("Apuesta", df["id"].tolist(), format_func=label_b)
        confirmar = st.checkbox("Confirmo eliminar")
        if st.button("🗑️ Eliminar", type="primary", disabled=not confirmar):
            try:
                supabase.table("apuestas").delete().eq("id", int(apuesta_borrar)).eq("user_id", user_id).execute()
                st.success("Eliminada.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
