# procesamiento/stationarity.py
"""
Stationarity tests (ADF, KPSS) for price and log-returns.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def _format_adf(res):
    stat, pval, lags, nobs, crit, *_ = res
    return {
        "test": "ADF",
        "stat": float(stat),
        "pvalue": float(pval),
        "lags_used": int(lags),
        "nobs": int(nobs),
        "crit_1%": float(crit.get("1%")),
        "crit_5%": float(crit.get("5%")),
        "crit_10%": float(crit.get("10%")),
        "conclusion": "stationary" if pval < 0.05 else "non-stationary"
    }

def _format_kpss(res):
    stat, pval, lags, crit = res
    return {
        "test": "KPSS",
        "stat": float(stat),
        "pvalue": float(pval),
        "lags_used": int(lags),
        "crit_10%": float(crit.get("10%")),
        "crit_5%": float(crit.get("5%")),
        "crit_2.5%": float(crit.get("2.5%")),
        "crit_1%": float(crit.get("1%")),
        "conclusion": "non-stationary" if pval < 0.05 else "stationary"
    }

def stationarity_tests(price: pd.Series, name: str = "series") -> pd.DataFrame:
    """
    Run ADF and KPSS on:
      - price (level)
      - log(price) differences (returns)
    Return a tidy DataFrame with results.
    """
    out = []
    s_price = pd.to_numeric(price, errors="coerce").dropna()
    if s_price.empty:
        return pd.DataFrame([{"series": name, "error": "empty series"}])

    # ADF / KPSS on price
    try:
        out.append({"series": f"{name}[level]", **_format_adf(adfuller(s_price, autolag="AIC"))})
    except Exception as e:
        out.append({"series": f"{name}[level]", "test":"ADF", "error": str(e)})
    try:
        out.append({"series": f"{name}[level]", **_format_kpss(kpss(s_price, regression="c", nlags="auto"))})
    except Exception as e:
        out.append({"series": f"{name}[level]", "test":"KPSS", "error": str(e)})

    # Returns
    s_lr = np.log(s_price).diff().dropna()
    try:
        out.append({"series": f"{name}[Δlog]", **_format_adf(adfuller(s_lr, autolag="AIC"))})
    except Exception as e:
        out.append({"series": f"{name}[Δlog]", "test":"ADF", "error": str(e)})
    try:
        out.append({"series": f"{name}[Δlog]", **_format_kpss(kpss(s_lr, regression="c", nlags="auto"))})
    except Exception as e:
        out.append({"series": f"{name}[Δlog]", "test":"KPSS", "error": str(e)})

    return pd.DataFrame(out)
