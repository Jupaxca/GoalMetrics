import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="GoalMetrics | Football Analytics", 
    page_icon="📊", 
    layout="wide"
)

# 1. CARGA DE DATOS
@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=['Equipo', 'Fecha', 'Condición', 'Nivel Rival'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Equipo'] = df['Equipo'].astype(str).str.strip()
    df['Condición'] = df['Condición'].astype(str).str.strip().str.lower()
    df['Nivel Rival'] = df['Nivel Rival'].astype(str).str.strip()
    
    for col in ['Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# 2. MOTOR MATEMÁTICO SELLADO
@st.cache_data
def simular_montecarlo(lam_fav, lam_con, lam_tir, lam_tpuerta, lam_corn, lam_faltas):
    rng = np.random.default_rng(42)
    num_sim = 10000
    return (
        rng.poisson(lam=lam_fav, size=num_sim),
        rng.poisson(lam=lam_con, size=num_sim),
        rng.poisson(lam=lam_tir, size=num_sim),
        rng.poisson(lam=lam_tpuerta, size=num_sim),
        rng.poisson(lam=lam_corn, size=num_sim),
