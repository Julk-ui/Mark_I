# evaluacion/backtest_rolling.py
"""
Backtesting rolling-origin (expanding window) para modelos 1D:
- Baseline Random Walk (naive last)
- SARIMAX (ARIMA/SARIMA) con statsmodels
- (Opcional) ETS si statsmodels.holtwinters disponible

Incluye modo AUTO: re-selecciona dinámicamente ARIMA o SARIMA por BIC (y opcional Ljung–Box) en cada reentreno,
usando únicamente datos pasados (sin leakage).

Métricas: RMSE, MAE, MAPE, sMAPE, R^2, HitRate direccional.
Exporta Excel con resumen y hojas de predicciones por modelo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- opcionales ---
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _SM_OK = True
except Exception as _e:
    SARIMAX = None  # type: ignore
    _SM_OK = False
    print(f"ℹ️ SARIMAX no disponible ({_e}).")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _ETS_OK = True
except Exception:
    ExponentialSmoothing = None  # type: ignore
    _ETS_OK = False

# Scanners ARIMA/SARIMA (opcional)
try:
    from modelos.arima_scan import escanear_arima
    _ARIMA_SCAN_OK = True
except Exception as _e:
    _ARIMA_SCAN_OK = False
    print(f"ℹ️ ARIMA scan no disponible ({_e}).")

try:
    from modelos.sarima_scan import escanear_sarima
    _SARIMA_SCAN_OK = True
except Exception as _e:
    _SARIMA_SCAN_OK = False
    print(f"ℹ️ SARIMA scan no disponible ({_e}).")


# =========================
# Métricas
# =========================
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, -0.0) else np.nan

def metrics_regression(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(float).dropna()
    y_pred = pd.Series(y_pred).astype(float).reindex(y_true.index).dropna()
    y_true = y_true.reindex(y_pred.index)

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / y_true.replace(0, np.nan)))) * 100.0
    smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred)).replace(0, np.nan))) * 100.0
    # R^2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - _safe_div(ss_res, ss_tot)

    # Dirección (aciertos de signo vs variación real basada en y_{t-1})
    base = y_true.shift(1)
    real_up = np.sign(y_true - base)
    pred_up = np.sign(y_pred - base)
    hit = float(np.mean((real_up == pred_up).dropna())) * 100.0

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "sMAPE": smape, "R2": r2, "HitRate": hit}


# =========================
# Backtesting core
# =========================
@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    preds: pd.DataFrame  # columns: ['y_true','y_hat']

def _ensure_series(y: pd.Series) -> pd.Series:
    s = pd.Series(y).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()
    return s

def _rw_forecast(y_hist: pd.Series, horizon: int) -> float:
    """Random Walk (naive last): pronostica el último valor conocido."""
    last = float(y_hist.iloc[-1])
    return last

def _fit_predict_sarimax(y_hist: pd.Series, steps: int,
                         order: Tuple[int,int,int],
                         seasonal_order: Optional[Tuple[int,int,int,int]] = None,
                         enforce_stationarity: bool = False,
                         enforce_invertibility: bool = False) -> np.ndarray:
    if not _SM_OK:
        raise RuntimeError("statsmodels SARIMAX no está disponible.")
    model = SARIMAX(y_hist, order=order, seasonal_order=seasonal_order or (0,0,0,0),
                    enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps).predicted_mean.values
    return fc


# --------- AUTO selector (ARIMA vs SARIMA) ---------
def select_best_sarimax(y_hist: pd.Series, scan_cfg: dict | None = None) -> dict:
    """
    Escanea ARIMA y/o SARIMA y devuelve el mejor por BIC usando SOLO historial.
    scan_cfg:
      try_arima: bool (default True)
      try_sarima: bool (default True)
      max_p,max_q (ARIMA); max_P,max_Q (SARIMA); s_candidates; require_lb; lb_alpha
    """
    scan_cfg = scan_cfg or {}
    try_arima = bool(scan_cfg.get("try_arima", True))
    try_sarima = bool(scan_cfg.get("try_sarima", True))
    cand = []

    if try_arima and _ARIMA_SCAN_OK:
        try:
            ar_df = escanear_arima(y_hist, max_p=scan_cfg.get("max_p", 3), max_q=scan_cfg.get("max_q", 3))
            if ar_df is not None and not ar_df.empty:
                r = ar_df.iloc[0]
                cand.append(("ARIMA", (int(r["p"]), int(r["d"]), int(r["q"])), None, float(r["bic"]), r.get("lb_p", np.nan)))
        except Exception as e:
            print(f"ℹ️ select_best: ARIMA scan falló: {e}")

    if try_sarima and _SARIMA_SCAN_OK:
        try:
            sa_df = escanear_sarima(
                y_hist,
                s_candidates=scan_cfg.get("s_candidates", [5, 7]),
                max_p=scan_cfg.get("max_p", 2), max_q=scan_cfg.get("max_q", 2),
                max_P=scan_cfg.get("max_P", 1), max_Q=scan_cfg.get("max_Q", 1),
            )
            if sa_df is not None and not sa_df.empty:
                r = sa_df.iloc[0]
                cand.append(("SARIMA", (int(r["p"]), int(r["d"]), int(r["q"])),
                             (int(r["P"]), int(r["D"]), int(r["Q"]), int(r["s"])),
                             float(r["bic"]), r.get("lb_p", np.nan)))
        except Exception as e:
            print(f"ℹ️ select_best: SARIMA scan falló: {e}")

    if not cand:
        return {"name":"ARIMA(1,1,1)","kind":"sarimax","order":(1,1,1),"seasonal_order":None,"bic":np.nan,"lb_p":np.nan}

    require_lb = bool(scan_cfg.get("require_lb", False))
    lb_alpha   = float(scan_cfg.get("lb_alpha", 0.05))
    pool = [c for c in cand if (not require_lb) or (c[4] is not None and not pd.isna(c[4]) and float(c[4]) >= lb_alpha)]
    if not pool: pool = cand

    best = min(pool, key=lambda t: t[3])
    name = f"ARIMA{best[1]}" if best[0]=="ARIMA" else f"SARIMA{best[1]}x{best[2][:3]}[{best[2][3]}]"
    return {"name": name, "kind":"sarimax", "order": best[1], "seasonal_order": best[2], "bic": best[3], "lb_p": best[4]}

def rolling_backtest_sarimax(y: pd.Series,
                             order: Tuple[int,int,int],
                             seasonal_order: Optional[Tuple[int,int,int,int]] = None,
                             initial_train: int = 1000,
                             step: int = 20,
                             horizon: int = 1) -> tuple[Dict[str,float], pd.DataFrame]:
    """Backtest expanding-window (rolling origin) con orden fijo."""
    y = _ensure_series(y)
    assert initial_train >= 10, "initial_train demasiado pequeño"
    assert horizon >= 1, "horizon debe ser >=1"

    y_true_list = []
    y_hat_list = []

    start = initial_train
    while start + horizon <= len(y):
        train = y.iloc[:start]
        test_idx = y.index[start:start+horizon]
        fc = _fit_predict_sarimax(train, steps=horizon, order=order, seasonal_order=seasonal_order, maxiter = 80)
        y_hat_list.append(pd.Series(fc, index=test_idx))
        y_true_list.append(y.loc[test_idx])
        start += step

    if not y_hat_list:
        raise RuntimeError("Ventanas de backtest vacías. Ajusta initial_train/step/horizon.")

    y_hat = pd.concat(y_hat_list).sort_index()
    y_true = pd.concat(y_true_list).sort_index()
    idx = y_true.index.intersection(y_hat.index)
    y_true = y_true.loc[idx]; y_hat = y_hat.loc[idx]

    m = metrics_regression(y_true, y_hat)
    preds = pd.DataFrame({"y_true": y_true, "y_hat": y_hat})
    return m, preds


import time

def rolling_backtest_auto(
    y: pd.Series,
    initial_train: int = 1000,
    step: int = 20,
    horizon: int = 1,
    scan_cfg: dict | None = None,
    rescan_each_refit: bool = False,     # ← por defecto NO re-escanea cada vez
    rescan_every_refits: int = 5         # ← si activas re-scan, hazlo cada N
) -> tuple[dict, pd.DataFrame]:
    y = _ensure_series(y)
    assert initial_train >= 10 and horizon >= 1
    scan_cfg = scan_cfg or {}

    y_true_list, y_hat_list, model_used = [], [], []
    start = initial_train
    refit_count = 0
    current_spec = None

    total_iters = max(0, (len(y) - initial_train) // step)
    print(f"[AUTO] ventanas={total_iters}, step={step}, horizon={horizon}")

    while start + horizon <= len(y):
        train = y.iloc[:start]
        test_idx = y.index[start:start+horizon]

        do_scan = (current_spec is None) or (rescan_each_refit and (refit_count % max(1, rescan_every_refits) == 0))
        if do_scan:
            current_spec = select_best_sarimax(train, scan_cfg)

        t0 = time.time()
        fc = _fit_predict_sarimax(train, steps=horizon,
                                  order=current_spec["order"],
                                  seasonal_order=current_spec["seasonal_order"])
        dt = time.time() - t0
        print(f"[AUTO] {refit_count+1:02d}/{total_iters} fin={test_idx[-1].date()} spec={current_spec['name']} t={dt:.1f}s")

        y_hat_list.append(pd.Series(fc, index=test_idx))
        y_true_list.append(y.loc[test_idx])
        model_used.extend([current_spec["name"]] * horizon)

        start += step
        refit_count += 1

    if not y_hat_list:
        raise RuntimeError("Ventanas de backtest vacías. Ajusta initial_train/step/horizon.")

    y_hat = pd.concat(y_hat_list).sort_index()
    y_true = pd.concat(y_true_list).sort_index()
    idx = y_true.index.intersection(y_hat.index)
    y_true = y_true.loc[idx]; y_hat = y_hat.loc[idx]

    m = metrics_regression(y_true, y_hat)
    preds = pd.DataFrame({"y_true": y_true, "y_hat": y_hat})
    preds["model_used"] = model_used[:len(preds)]
    return m, preds

def rolling_backtest_rw(y: pd.Series,
                        initial_train: int = 1000,
                        step: int = 20,
                        horizon: int = 1) -> tuple[Dict[str,float], pd.DataFrame]:
    """Baseline Random Walk (naive last)."""
    y = _ensure_series(y)
    y_true_list = []; y_hat_list = []
    start = initial_train
    while start + horizon <= len(y):
        train = y.iloc[:start]
        test_idx = y.index[start:start+horizon]
        fc = np.array([_rw_forecast(train, horizon)] * horizon)
        y_hat_list.append(pd.Series(fc, index=test_idx))
        y_true_list.append(y.loc[test_idx])
        start += step

    y_hat = pd.concat(y_hat_list).sort_index()
    y_true = pd.concat(y_true_list).sort_index()
    idx = y_true.index.intersection(y_hat.index)
    y_true = y_true.loc[idx]; y_hat = y_hat.loc[idx]
    m = metrics_regression(y_true, y_hat)
    preds = pd.DataFrame({"y_true": y_true, "y_hat": y_hat})
    return m, preds


def rolling_backtest_ets(y: pd.Series,
                         initial_train: int = 1000,
                         step: int = 20,
                         horizon: int = 1,
                         trend: Optional[str] = None,
                         seasonal: Optional[str] = None,
                         seasonal_periods: Optional[int] = None) -> tuple[Dict[str,float], pd.DataFrame]:
    """Exponential Smoothing (si disponible)."""
    if not _ETS_OK:
        raise RuntimeError("ExponentialSmoothing no está disponible en este entorno.")
    y = _ensure_series(y)
    y_true_list = []; y_hat_list = []
    start = initial_train
    while start + horizon <= len(y):
        train = y.iloc[:start]
        test_idx = y.index[start:start+horizon]
        model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        res = model.fit(optimized=True, use_brute=True)
        fc = res.forecast(horizon).values
        y_hat_list.append(pd.Series(fc, index=test_idx))
        y_true_list.append(y.loc[test_idx])
        start += step

    y_hat = pd.concat(y_hat_list).sort_index()
    y_true = pd.concat(y_true_list).sort_index()
    idx = y_true.index.intersection(y_hat.index)
    y_true = y_true.loc[idx]; y_hat = y_hat.loc[idx]
    m = metrics_regression(y_true, y_hat)
    preds = pd.DataFrame({"y_true": y_true, "y_hat": y_hat})
    return m, preds


# =========================
# Ejecutar varios modelos
# =========================
def evaluate_many(y: pd.Series,
                  specs: List[Dict],
                  initial_train: int = 1000,
                  step: int = 20,
                  horizon: int = 1):
    """
    Ejecuta backtest para múltiples 'specs'. Cada spec:
      {'name': 'AUTO', 'kind': 'auto', 'scan': {...}, 'rescan_each_refit': True, 'rescan_every_refits': 1}
      {'name': 'ARIMA(3,1,1)', 'kind': 'sarimax', 'order':(3,1,1), 'seasonal_order':None}
      {'name': 'RW', 'kind': 'rw'}
      {'name': 'ETS', 'kind': 'ets', 'trend':'add', 'seasonal':'add', 'seasonal_periods':5}
    Devuelve (summary_df, preds_map) donde summary_df tiene métricas por modelo.
    """
    y = _ensure_series(y)
    rows = []
    preds_map: Dict[str, pd.DataFrame] = {}

    for sp in specs:
        name = sp.get("name", sp.get("kind", "model"))
        kind = sp.get("kind", "sarimax")
        try:
            if kind == "rw":
                m, p = rolling_backtest_rw(y, initial_train=initial_train, step=step, horizon=horizon)
            elif kind == "ets":
                m, p = rolling_backtest_ets(
                    y, initial_train=initial_train, step=step, horizon=horizon,
                    trend=sp.get("trend"), seasonal=sp.get("seasonal"), seasonal_periods=sp.get("seasonal_periods")
                )
            elif kind == "sarimax":
                m, p = rolling_backtest_sarimax(
                    y, order=sp["order"], seasonal_order=sp.get("seasonal_order"),
                    initial_train=initial_train, step=step, horizon=horizon
                )
            elif kind == "auto":
                m, p = rolling_backtest_auto(
                    y,
                    initial_train=initial_train, step=step, horizon=horizon,
                    scan_cfg=sp.get("scan", {}),
                    rescan_each_refit=sp.get("rescan_each_refit", True),
                    rescan_every_refits=sp.get("rescan_every_refits", 1)
                )
            else:
                raise ValueError(f"kind desconocido: {kind}")

            row = {"Modelo": name}; row.update(m)
            rows.append(row); preds_map[name] = p
        except Exception as e:
            row = {"Modelo": name, "ERROR": str(e)}
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df, preds_map

# =========================
# Guardado a Excel (safe sheet names)
# =========================
import re

def _sanitize_sheet_name(name: str, taken: set[str]) -> str:
    """
    Hace válido un nombre de hoja de Excel:
    - Reemplaza caracteres inválidos []:*?/\
    - Recorta a 31 caracteres
    - Evita duplicados añadiendo sufijos (2), (3)...
    """
    if name is None or str(name).strip() == "":
        name = "Sheet"
    s = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))  # reemplaza caracteres prohibidos
    s = s.strip()
    if len(s) > 31:
        s = s[:31]
    base = s
    i = 2
    while s in taken:
        suf = f" ({i})"
        s = (base[: max(0, 31 - len(suf))] + suf) if len(base) + len(suf) > 31 else base + suf
        i += 1
    taken.add(s)
    return s

def save_backtest_excel(path: str,
                        summary_df,
                        preds_map: Dict[str, pd.DataFrame]) -> None:
    """
    Guarda resultados en un Excel. 'summary_df' puede ser DataFrame o dict name->metrics.
    Nombres de hoja saneados para cumplir restricciones de Excel.
    """
    if isinstance(summary_df, dict):
        rows = []
        for name, m in summary_df.items():
            row = {"Modelo": name}; row.update(m)
            rows.append(row)
        summary_df = pd.DataFrame(rows)

    # Motor preferido
    try:
        import xlsxwriter  # noqa
        writer_kwargs = {"engine": "xlsxwriter", "datetime_format": "yyyy-mm-dd hh:mm"}
    except Exception:
        writer_kwargs = {"engine": "openpyxl"}

    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    taken: set[str] = set()
    with pd.ExcelWriter(path, **writer_kwargs) as w:
        resumen_name = _sanitize_sheet_name("Resumen", taken)
        summary_df.to_excel(w, sheet_name=resumen_name, index=False)

        for name, dfp in preds_map.items():
            safe = _sanitize_sheet_name(name, taken)
            tmp = dfp.copy()
            if isinstance(tmp.index, pd.DatetimeIndex):
                try:
                    tmp.index = tmp.index.tz_localize(None)
                except Exception:
                    pass
            tmp.to_excel(w, sheet_name=safe)

    print(f"💾 Backtest guardado en: {path}")
