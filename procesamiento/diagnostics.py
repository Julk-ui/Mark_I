# procesamiento/diagnostics.py
"""
Model diagnostics and visual aids for ARIMA/SARIMA + (optional) GARCH.
- BIC heatmaps from candidate scans
- Residual diagnostics for the selected model
- GARCH(1,1) volatility plot (optional, requires `arch`)
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats as sstats

def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def bic_heatmap_arima(candidates: pd.DataFrame, symbol: str, outdir: str) -> list[str]:
    paths = []
    if candidates is None or candidates.empty:
        return paths
    for d in sorted(set(candidates["d"].dropna().astype(int))):
        sub = candidates[candidates["d"] == d].copy()
        if sub.empty: 
            continue
        pivot = sub.pivot_table(index="p", columns="q", values="bic", aggfunc="min")
        plt.figure(figsize=(6,5))
        plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="BIC (menor es mejor)")
        plt.title(f"{symbol} · mapa BIC ARIMA por d={d}")
        plt.xlabel("q"); plt.ylabel("p"); plt.tight_layout()
        path = os.path.join(outdir, f"{symbol}_ARIMA_bic_d{d}.png")
        plt.savefig(path); plt.close()
        paths.append(path)
    return paths

def bic_heatmap_sarima(candidates: pd.DataFrame, symbol: str, outdir: str) -> list[str]:
    paths = []
    if candidates is None or candidates.empty:
        return paths
    groups = candidates.groupby(["P","Q","s"])
    for (P,Q,s), g in groups:
        pivot = g.pivot_table(index="p", columns="q", values="bic", aggfunc="min")
        plt.figure(figsize=(6,5))
        plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="BIC (menor es mejor)")
        plt.title(f"{symbol} · mapa BIC SARIMA (P={P},Q={Q},s={s})")
        plt.xlabel("q"); plt.ylabel("p"); plt.tight_layout()
        path = os.path.join(outdir, f"{symbol}_SARIMA_bic_P{P}Q{Q}s{s}.png")
        plt.savefig(path); plt.close()
        paths.append(path)
    return paths

def _parse_spec(spec: str):
    m = re.match(r"ARIMA\((\d+),(\d+),(\d+)\)$", spec or "")
    if m:
        p,d,q = map(int, m.groups())
        return ("arima", (p,d,q), None)
    m = re.match(r"SARIMA\((\d+),(\d+),(\d+)\)x\((\d+),(\d+),(\d+)\)\[(\d+)\]$", spec or "")
    if m:
        p,d,q,P,D,Q,s = map(int, m.groups())
        return ("sarima", (p,d,q), (P,D,Q,s))
    return (None, None, None)

def residual_diagnostics(y: pd.Series, spec: str, symbol: str, outdir: str) -> dict:
    model_type, nd, ns = _parse_spec(spec)
    if model_type is None:
        return {"error":"unknown spec"}
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return {"error":"empty series"}

    if model_type == "arima":
        p,d,q = nd
        mod = SARIMAX(y, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
    else:
        p,d,q = nd; P,D,Q,s = ns
        mod = SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_stationarity=False, enforce_invertibility=False)

    res = mod.fit(disp=False)
    resid = pd.Series(res.resid).dropna()

    paths = {}
    fig = plt.figure(figsize=(11,4)); plot_acf(resid, lags=40, ax=plt.gca()); plt.title(f"{symbol} · ACF residuales"); plt.tight_layout()
    paths["acf_resid"] = os.path.join(outdir, f"{symbol}_acf_resid.png"); plt.savefig(paths["acf_resid"]); plt.close()

    fig = plt.figure(figsize=(11,4)); plot_pacf(resid, lags=40, ax=plt.gca(), method="ywm"); plt.title(f"{symbol} · PACF residuales"); plt.tight_layout()
    paths["pacf_resid"] = os.path.join(outdir, f"{symbol}_pacf_resid.png"); plt.savefig(paths["pacf_resid"]); plt.close()

    fig = plt.figure(figsize=(6,6)); sstats.probplot(resid, dist="norm", plot=plt); plt.title(f"{symbol} · QQ residuales"); plt.tight_layout()
    paths["qq_resid"] = os.path.join(outdir, f"{symbol}_qq_resid.png"); plt.savefig(paths["qq_resid"]); plt.close()

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb10 = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        lb20 = acorr_ljungbox(resid, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
    except Exception:
        lb10 = np.nan; lb20 = np.nan

    return {"lb_p_10": float(lb10), "lb_p_20": float(lb20), **paths}

def garch_vol_plot(logret: pd.Series, symbol: str, outdir: str) -> str | None:
    try:
        from arch.univariate import ConstantMean, GARCH, Normal
    except Exception:
        return None
    s = pd.Series(logret).dropna()
    if s.empty: return None
    am = ConstantMean(s*100)
    am.volatility = GARCH(1,1)
    am.distribution = Normal()
    res = am.fit(disp="off")
    sigma = res.conditional_volatility / 100.0
    sigma_ann = sigma * np.sqrt(252.0)
    plt.figure(figsize=(11,4))
    plt.plot(sigma_ann.index, sigma_ann.values)
    plt.title(f"{symbol} · GARCH(1,1) σ anualizada")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_10_garch_sigma.png")
    plt.savefig(path); plt.close()
    return path
