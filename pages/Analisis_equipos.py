import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
import hashlib
import colorsys

@st.cache_data(ttl=600)
def cargar_datos():
    sheet_id = "16oKLxQtC59_tiPSKLEOECN0kO2WCXUPLZg7q73WPXyg"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=["Equipo", "Fecha", "Condición", "Nivel Rival"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df["Equipo"] = df["Equipo"].astype(str).str.strip()
    df["Condición"] = df["Condición"].astype(str).str.strip().str.lower()
    df["Nivel Rival"] = df["Nivel Rival"].astype(str).str.strip()
    
    cols_numericas = ["Goles", "Goles Rival", "Tiros", "A Puerta", "Corners", "Faltas", "Atajadas", "Amarillas", "Rojas", "Corners Rival"]
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    df["Tiros a Puerta Rival"] = df["Goles Rival"] + df["Atajadas"]
    
    return df

def shrinkage_lambda(lam_obs, lam_prior, n_obs, k=5.0):
    n = max(float(n_obs), 0.0)
    return (n * lam_obs + k * lam_prior) / (n + k)

def dixon_coles_tau(x, y, lam_x, lam_y, rho):
    if x == 0 and y == 0:
        return 1.0 - lam_x * lam_y * rho
    if x == 0 and y == 1:
        return 1.0 + lam_x * rho
    if x == 1 and y == 0:
        return 1.0 + lam_y * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

def poisson_pmf(k, lam):
    lam = max(float(lam), 1e-9)
    k = int(k)
    if k < 0:
        return 0.0
    if k == 0:
        return float(np.exp(-lam))
    log_p = -lam + k * np.log(lam) - np.sum(np.log(np.arange(1, k + 1)))
    return float(np.exp(log_p))

def simular_goles_dixon_coles(lam_fav, lam_con, rho=-0.10, num_sim=10000, max_goles=8, seed=42):
    rng = np.random.default_rng(seed)
    lam_fav = max(lam_fav, 0.05)
    lam_con = max(lam_con, 0.05)
    xs = np.arange(0, max_goles + 1)
    ys = np.arange(0, max_goles + 1)
    joint = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        px = poisson_pmf(x, lam_fav)
        for j, y in enumerate(ys):
            py = poisson_pmf(y, lam_con)
            tau = dixon_coles_tau(x, y, lam_fav, lam_con, rho)
            joint[i, j] = max(px * py * tau, 0.0)
    total = joint.sum()
    if total <= 0:
        return rng.poisson(lam_fav, num_sim), rng.poisson(lam_con, num_sim)
    joint = joint / total
    flat = joint.ravel()
    idx = rng.choice(len(flat), size=num_sim, p=flat)
    return xs[idx // joint.shape[1]], ys[idx % joint.shape[1]]

@st.cache_data
def simular_stats_poisson(lam_tir, lam_tpuerta, lam_corn, lam_faltas, num_sim=10000, seed=42):
    rng = np.random.default_rng(seed)
    return (
        rng.poisson(max(lam_tir, 0.01), num_sim),
        rng.poisson(max(lam_tpuerta, 0.01), num_sim),
        rng.poisson(max(lam_corn, 0.01), num_sim),
        rng.poisson(max(lam_faltas,
