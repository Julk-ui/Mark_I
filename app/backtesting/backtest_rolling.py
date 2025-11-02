from __future__ import annotations
"""
Módulo de backtesting (clásico y por clase).

Incluye dos motores complementarios:

1) evaluate_many(...): Walk-forward 'clásico' para ARIMA/SARIMA con:
   - Búsqueda ligera por BIC (ARIMA y SARIMA).
   - Umbrales de decisión (fixed / ATR / GARCH).
   - Métricas de regresión y de trading (RMSE, MAE, MAPE, R2, Directional Accuracy,
     HitRate, Total pips, MaxDD, Sortino Ratio).
   - Exportación de reportes Excel **centralizada** en reportes/reportes_excel.py.

2) run_backtest(...) / run_backtest_many(...): Backtest genérico para
   modelos con interfaz .fit/.predict (p.ej., Prophet, LSTM), retornando
   una matriz 'wide' (filas = fecha de inicio, columnas = 1..H).

Ambos motores están diseñados para ser llamados desde `main.py`.
"""

import os
import re
import time
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# Exportador CENTRALIZADO (reportes/)
# ---------------------------------------------------------------------
# --- al inicio del archivo (o donde tienes los imports utilitarios) ---
import importlib
from typing import Callable, Optional, Tuple

def _import_exporters() -> Tuple[Callable, Callable]:
    """
    Localiza y retorna las funciones del módulo de reportes Excel centralizado.
     Busca en varias rutas válidas y, si no encuentra, entrega stubs seguros.
    Devuelve: (export_excel_simple, export_backtest_result)
    """
    def _stub_simple(*args, **kwargs):
        print("ℹ️ export_excel_simple no disponible; se omite.")

    def _stub_result(*args, **kwargs):
        print("ℹ️ export_backtest_result no disponible; se omite.")

    candidates = [
        # ✅ ubicación real en tu repo (paquete en la raíz)
        ("reportes.reportes_excel", ["export_excel_simple", "export_backtest_result"]),
        # Otras variantes por si cambias layout
        ("app.reportes.reportes_excel", ["export_excel_simple", "export_backtest_result"]),
        ("reportes_excel", ["export_excel_simple", "export_backtest_result"]),
    ]

    for mod, names in candidates:
        try:
            m = importlib.import_module(mod)
            f_simple = getattr(m, names[0], None)
            f_result = getattr(m, names[1], None)
            if callable(f_simple) and callable(f_result):
                return f_simple, f_result
        except Exception:
            continue

    return _stub_simple, _stub_result

# Obtén las funciones al cargar el módulo
export_excel_simple, export_backtest_result = _import_exporters()

_export_excel_simple, _export_backtest_result = _import_exporters()

# ---------------------------------------------------------------------
# Import robusto del factory `get_model` (tolera distintas rutas de proyecto)
# ---------------------------------------------------------------------
def _import_get_model():
    """Intenta cargar `get_model` desde varias rutas conocidas."""
    try:
        from app.utils.registry import get_model  # type: ignore
        return get_model
    except Exception:
        try:
            from utils.registry import get_model  # type: ignore
            return get_model
        except Exception:
            from registry import get_model  # type: ignore
            return get_model

get_model = _import_get_model()

# -------------------------
# Utilidades varias
# -------------------------
def _pred_ret_to_pips(y_pred_ret: float, price_t: float, pip_size: float, use_exp: bool = True) -> float:
    """
    Convierte un retorno predicho (Δlog) a pips tomando `price_t` como base.

    Parameters
    ----------
    y_pred_ret : float
        Retorno predicho (en Δlog).
    price_t : float
        Precio actual (al final de la ventana de entrenamiento).
    pip_size : float
        Tamaño de pip para convertir el delta a pips.
    use_exp : bool, default True
        Si True, transforma con `exp(Δlog)`; si False, aproxima lineal.

    Returns
    -------
    float
        Pips de la predicción (NaN si valores inválidos).
    """
    if price_t is None or pip_size is None or pip_size <= 0:
        return np.nan
    if np.isnan(y_pred_ret):
        return np.nan
    if use_exp:
        y_pred_price = price_t * float(np.exp(y_pred_ret))
        delta = y_pred_price - price_t
    else:
        delta = float(y_pred_ret) * price_t
    return float(delta / pip_size)

def _safe_mkdir(path: str) -> None:
    """Crea el directorio si no existe (silencioso si ya existe)."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _sheet_name_safe(name: str) -> str:
    """Sanitiza el nombre para hojas de Excel (sin caracteres prohibidos y <=31)."""
    name = re.sub(r"[\[\]\:\*\?\/\\']", "_", str(name))
    return name[:31] or "Sheet"

def _file_name_safe(name: str) -> str:
    """Sanitiza nombres de archivo evitando caracteres problemáticos."""
    return re.sub(r"[\[\]\:\*\?\/\\']", "_", str(name))

# -------------------------
# Scans ARIMA / SARIMA
# -------------------------
def _scan_arima_returns(y: pd.Series, max_p=3, max_q=3, d=0, exog=None) -> Tuple[Tuple[int,int,int], float]:
    """
    Busca ARIMA(p,d,q) por BIC de forma ligera, evitando ARIMA(0,0,0).

    Returns
    -------
    (order, bic_min) : Tuple[Tuple[int,int,int], float]
    """
    best = None
    best_bic = float("inf")
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            p_ = p or 1 if q == 0 else p
            try:
                model = SARIMAX(y, order=(p_, d, q), exog=exog, trend="n",
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                bic = float(res.bic)
                if bic < best_bic:
                    best_bic = bic
                    best = (p_, d, q)
            except Exception:
                continue
    if best is None:
        best = (1, d, 0)
        best_bic = float("inf")
    return best, best_bic

def _scan_sarima_returns(y: pd.Series,
                         s_candidates=(5, 7),
                         max_p=2, max_q=2, max_P=1, max_Q=1,
                         d=0, D=0, exog=None) -> Tuple[Tuple[int,int,int,int,int,int,int], float]:
    """
    Busca SARIMA(p,d,q)×(P,D,Q)[s] por BIC de forma ligera, evitando todo-cero.
    """
    best = None
    best_bic = float("inf")
    for s in s_candidates:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        p_, q_, P_, Q_ = p, q, P, Q
                        if (p_ + q_ + P_ + Q_) == 0:
                            p_ = 1
                        try:
                            model = SARIMAX(y, order=(p_, d, q), seasonal_order=(P_, D, Q_, s),
                                            exog=exog, trend="n",
                                            enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False)
                            bic = float(res.bic)
                            if bic < best_bic:
                                best_bic = bic
                                best = (p_, d, q, P_, D, Q_, s)
                        except Exception:
                            continue
    if best is None:
        best = (1, d, 0, 0, D, 0, s_candidates[0])
    return best, best_bic

# -------------------------
# Umbral dinámico
# -------------------------
def _resolve_threshold(date_idx: pd.Timestamp,
                       mode: str,
                       fixed_pips: float,
                       min_threshold_pips: float,
                       atr_pips: Optional[pd.Series] = None,
                       atr_k: float = 0.6,
                       garch_sigma_pips: Optional[pd.Series] = None,
                       garch_k: float = 0.6) -> float:
    """
    Devuelve el umbral en pips para la fecha de corte según el modo.

    - 'fixed' : usa `fixed_pips`.
    - 'atr'   : usa `atr_k * ATR(date)` con piso `min_threshold_pips`.
    - 'garch' : usa `garch_k * sigma_t(date)` con piso `min_threshold_pips`.
    """
    mode = (mode or "fixed").lower()
    thr = fixed_pips
    if mode == "atr" and atr_pips is not None:
        val = float(atr_pips.reindex([date_idx]).ffill().iloc[-1])
        thr = atr_k * val
    elif mode == "garch" and garch_sigma_pips is not None:
        val = float(garch_sigma_pips.reindex([date_idx]).ffill().iloc[-1])
        thr = garch_k * val
    return float(max(thr, float(min_threshold_pips)))

# -------------------------
# Rolling windows
# -------------------------
def _rolling_windows_index(n: int, initial_train: int, step: int, horizon: int) -> List[Tuple[int,int,int]]:
    """
    Genera índices (inicio_train, fin_train, fin_test) para walk-forward.
    """
    out = []
    i = initial_train - 1
    while i + horizon < n:
        end_train = i
        end_test = i + horizon
        start_train = 0
        out.append((start_train, end_train, end_test))
        i += step
    return out

def _fit_and_forecast(y: pd.Series, y_next_idx: pd.Timestamp, spec: dict,
                      exog_train: Optional[pd.DataFrame] = None,
                      exog_next: Optional[pd.DataFrame] = None) -> Tuple[float, str]:
    """
    Entrena y predice 1 paso según el tipo de `spec`:
    - kind='rw'   : Random Walk (retorno 0).
    - kind='auto' : ARIMA/SARIMA con búsqueda ligera por BIC.
    """
    kind = str(spec.get("kind", "rw")).lower()
    if kind == "rw":
        return 0.0, "RW"

    scan = (spec.get("scan") or {})
    try_sarima = bool(scan.get("try_sarima", True))
    max_p = int(scan.get("max_p", 3))
    max_q = int(scan.get("max_q", 3))
    max_P = int(scan.get("max_P", 1))
    max_Q = int(scan.get("max_Q", 1))
    s_candidates = scan.get("s_candidates", [5, 7])

    best_txt = None
    res = None
    trend = "n"

    if try_sarima:
        order_seas, _ = _scan_sarima_returns(y, s_candidates=s_candidates,
                                             max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                                             d=0, D=0, exog=exog_train)
        p, d, q, P, D, Q, s = order_seas
        try:
            model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s),
                            exog=exog_train, trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            best_txt = f"SARIMA({p},{d},{q})x({P},{D},{Q})[{s}]"
        except Exception:
            res = None

    if res is None:
        order, _ = _scan_arima_returns(y, max_p=max_p, max_q=max_q, d=0, exog=exog_train)
        p, d, q = order
        try:
            model = SARIMAX(y, order=(p, d, q), exog=exog_train, trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            best_txt = f"ARIMA({p},{d},{q})"
        except Exception:
            return 0.0, "RW"

    try:
        y_pred = res.get_forecast(steps=1, exog=exog_next).predicted_mean.iloc[0]
        y_pred = float(y_pred)
    except Exception:
        y_pred = 0.0
        best_txt = (best_txt or "") + " [fallback]"

    return y_pred, best_txt

# -------------------------
# Métricas genéricas y reutilizables
# -------------------------
def _safe_mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-12) -> float:
    """
    MAPE robusto: mean(|e / max(|y|, eps)|)*100
    (Para retornos, evita división por ~0).
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float).reindex_like(y_true)
    denom = np.maximum(np.abs(y_true.values), eps)
    mape = np.mean(np.abs((y_true.values - y_pred.values) / denom)) * 100.0
    return float(mape)

def _directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    % de acierto direccional: sign(y_pred) == sign(y_true) sobre observaciones válidas.
    """
    yt = y_true.astype(float)
    yp = y_pred.reindex_like(yt).astype(float)
    mask = yt.notna() & yp.notna()
    if mask.sum() == 0:
        return 0.0
    acc = (np.sign(yp[mask]) == np.sign(yt[mask])).mean() * 100.0
    return float(acc)

def _sortino_ratio(pnl_series: pd.Series) -> float:
    """
    Sortino Ratio sobre la serie de resultados por decisión (pips).
    SR = mean(r) / std(r_negativos)
    Si no hay negativos o std=0, devuelve NaN.
    """
    r = pnl_series.dropna().astype(float)
    if len(r) == 0:
        return float("nan")
    downside = r[r < 0.0]
    if len(downside) == 0:
        return float("nan")
    dd = downside.std(ddof=0)
    if dd == 0:
        return float("nan")
    return float(r.mean() / dd)

def _compute_metrics(df_pred: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas genéricas (regresión + trading) a partir de df_pred estándar
    con columnas: y_true_ret, y_pred_ret, pnl_pips, cum_pips, etc.
    """
    out = {
        "RMSE": float("nan"),
        "MAE": float("nan"),
        "MAPE_%": float("nan"),
        "R2": float("nan"),
        "Directional_Accuracy_%": 0.0,
        "HitRate_%": 0.0,
        "Total_pips": 0.0,
        "MaxDD_pips": 0.0,
        "Sortino": float("nan"),
    }
    if df_pred is None or df_pred.empty:
        return out

    yt = df_pred.get("y_true_ret")
    yp = df_pred.get("y_pred_ret")

    # Métricas regresión
    if yt is not None and yp is not None and len(df_pred) > 0:
        out["RMSE"] = float(np.sqrt(mean_squared_error(yt, yp)))
        out["MAE"] = float(mean_absolute_error(yt, yp))
        try:
            out["R2"] = float(r2_score(yt, yp))
        except Exception:
            out["R2"] = float("nan")
        try:
            out["MAPE_%"] = _safe_mape(yt, yp)
        except Exception:
            out["MAPE_%"] = float("nan")
        out["Directional_Accuracy_%"] = _directional_accuracy(yt, yp)

    # Métricas trading
    trades = df_pred[df_pred.get("signal", 0) != 0]
    if len(trades):
        out["HitRate_%"] = float((trades["pnl_pips"] > 0).mean() * 100.0)
        out["Total_pips"] = float(trades["pnl_pips"].sum())
        out["Sortino"] = _sortino_ratio(trades["pnl_pips"])

    if "cum_pips" in df_pred.columns:
        dd = df_pred["cum_pips"].cummax() - df_pred["cum_pips"]
        out["MaxDD_pips"] = float(-dd.max()) if len(dd) else 0.0

    return out

# -------------------------
# Motor clásico (ARIMA/SARIMA)
# -------------------------
def evaluate_many(price: pd.Series,
                  specs: List[dict],
                  initial_train: int = 1000,
                  step: int = 10,
                  horizon: int = 1,
                  target: str = "returns",
                  pip_size: float = 0.0001,
                  threshold_pips: float = 15.0,
                  exog_ret: Optional[pd.Series] = None,
                  exog_lags: Optional[List[int]] = None,
                  threshold_mode: str = "fixed",
                  atr_pips: Optional[pd.Series] = None,
                  atr_k: float = 0.6,
                  garch_k: float = 0.6,
                  min_threshold_pips: float = 10.0,
                  garch_sigma_pips: Optional[pd.Series] = None,
                  log_threshold_used: bool = True,
                  decision_cfg: Optional[dict] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Ejecuta walk-forward usando especificaciones `specs` y calcula métricas.

    Returns
    -------
    summary_df : pd.DataFrame
        Tabla resumen por spec (RMSE, MAE, MAPE, R2, Directional Accuracy,
        HitRate, Total pips, MaxDD, Sortino).
    preds_map : Dict[str, pd.DataFrame]
        Diccionario {nombre_spec -> DataFrame con predicciones y PnL}.
    """
    price = price.astype(float).dropna()
    y_ret = np.log(price).diff().dropna()
    idx = price.index

    def _make_exog_lags(exog: pd.Series, lags: List[int]) -> Optional[pd.DataFrame]:
        if exog is None or not lags:
            return None
        s = exog.copy()
        out = {f"exog_lag{L}": s.shift(L) for L in lags}
        return pd.DataFrame(out)

    X = _make_exog_lags(exog_ret, exog_lags) if exog_lags else None

    def _windows(n, init, step_, h):
        out = []; i = init - 1
        while i + h < n:
            out.append((0, i, i + h))
            i += step_
        return out

    windows = _windows(len(price), initial_train, step, horizon)
    if not windows:
        raise ValueError("No hay ventanas válidas para backtesting.")

    preds_map: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    for spec in specs:
        name = spec.get("name", "MODEL")
        rows = []

        for i, (s0, e_tr, e_te) in enumerate(windows, start=1):
            date_end_train = idx[e_tr]
            date_next = idx[e_te]

            y_train = y_ret.loc[:date_end_train].dropna()
            if X is not None:
                exog_train = X.loc[y_train.index].dropna()
                y_train, exog_train = y_train.align(exog_train, join="inner")
            else:
                exog_train = None
            exog_next = X.reindex([date_next]) if X is not None else None

            thr = _resolve_threshold(
                date_end_train, threshold_mode,
                fixed_pips=threshold_pips, min_threshold_pips=min_threshold_pips,
                atr_pips=atr_pips, atr_k=atr_k, garch_sigma_pips=garch_sigma_pips, garch_k=garch_k
            )

            y_pred_ret, spec_txt = _fit_and_forecast(y_train, date_next, spec, exog_train, exog_next)

            y_true_ret = y_ret.reindex([date_next]).iloc[0]
            price_t = price.loc[date_end_train]
            price_tp1 = price.loc[date_next]

            y_pred_pips = _pred_ret_to_pips(y_pred_ret, price_t, pip_size, use_exp=True)
            signal = int(np.sign(y_pred_pips)) if abs(y_pred_pips) >= thr else 0
            pnl_pips = float(signal * ((price_tp1 - price_t) / pip_size)) if signal != 0 else 0.0
            y_pred_price = float(price_t * math.exp(y_pred_ret))

            rows.append({
                "date": date_next, "spec": spec_txt,
                "threshold_used_pips": float(thr) if log_threshold_used else np.nan,
                "y_true_ret": float(y_true_ret), "y_pred_ret": float(y_pred_ret),
                "y_pred_price": y_pred_price, "y_pred_pips": float(y_pred_pips),
                "signal": int(signal), "pnl_pips": float(pnl_pips),
                "price_t": float(price_t), "price_t1": float(price_tp1),
            })

            if i == 1 or i % 10 == 0 or i == len(windows):
                print(f"[AUTO] {i:02d}/{len(windows)} fin={date_end_train.date()} spec={spec_txt} thr={thr:.1f}p")

        df_pred = pd.DataFrame(rows).set_index("date").sort_index()
        df_pred["cum_pips"] = df_pred["pnl_pips"].cumsum()

        # --- Métricas genéricas y reutilizables ---
        met = _compute_metrics(df_pred)
        summary_rows.append({
            "Modelo": name,
            **met
        })

        preds_map[name] = df_pred

    summary_df = pd.DataFrame(summary_rows)

    # Exportación Excel CENTRALIZADA (si hay ruta configurada en decision_cfg)
    try:
        outxlsx = (decision_cfg or {}).get("outxlsx")
        if outxlsx:
            config_min = {
                "target": target,
                "pip_size": pip_size,
                "threshold_mode": threshold_mode,
                "threshold_pips": threshold_pips,
                "horizon": horizon,
            }
            # Usa el exportador central que además consolida 'Predicciones'
            _export_backtest_result(outxlsx, summary_df, preds_map, config_min)
    except Exception as e:
        # No interrumpir el flujo por problemas de exportación
        print(f"[WARN] Falló exportación centralizada: {e}")

    return summary_df, preds_map

# -------------------------
# Backtest por clase (.fit/.predict)
# -------------------------
def run_backtest(series: pd.Series, modelo, cfg: dict) -> pd.DataFrame:
    """
    Backtest rolling para modelos con interfaz:
        modelo.fit(serie_entrenamiento)
        modelo.predict(horizon, **kwargs) -> DataFrame con 'yhat'

    Returns
    -------
    pd.DataFrame
        Matriz 'wide' donde cada fila corresponde a la fecha de inicio de
        una ventana y las columnas 1..H son los yhat de cada horizonte.
    """
    win   = cfg["backtest"]["ventanas"]
    step  = cfg["backtest"]["step"]
    horiz = cfg["backtest"]["horizon"]

    preds, stamps = [], []
    for t0 in range(win, len(series) - horiz, step):
        train = series.iloc[t0 - win : t0]
        test_idx = series.index[t0 : t0 + horiz]

        modelo.fit(train)

        if "lstm" in modelo.__class__.__name__.lower():
            win_len = int(getattr(modelo, "model_cfg", {}).get("window", 64))
            last_window = train.iloc[-win_len:]
            fc = modelo.predict(horiz, last_window=last_window, last_timestamp=train.index[-1])
            fc.index = test_idx
            yhat = fc["yhat"].values
        else:
            fc = modelo.predict(horiz, index=test_idx)
            yhat = fc["yhat"].reindex(test_idx).values

        preds.append(yhat)
        stamps.append(test_idx[0])

    out_rows = [pd.Series(y, index=range(1, len(y) + 1), name=s) for s, y in zip(stamps, preds)]
    pred_df = pd.DataFrame(out_rows).sort_index()
    return pred_df

def run_backtest_many(series: pd.Series, modelos_cfg: List[dict], cfg_global: Dict[str, Any]):
    """
    Ejecuta backtest para múltiples modelos (interfaz .fit/.predict).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapa nombre_modelo -> matriz wide (como la de `run_backtest`).
    """
    results = {}
    for mdef in modelos_cfg:
        if not mdef.get("enabled", True):
            continue
        name = mdef["name"]
        objetivo = (mdef.get("objetivo") or cfg_global.get("target","returns")).lower()
        horizon = int(mdef.get("horizonte", cfg_global.get("backtest", {}).get("horizon", 1)))

        model_key = name.strip().lower()
        local = {
            "target": "returns" if objetivo == "retornos" else "close",
            "freq": cfg_global.get("freq", "H"),
            "backtest": {"ventanas": cfg_global["backtest"]["ventanas"],
                         "step": cfg_global["backtest"]["step"],
                         "horizon": horizon},
            model_key: mdef.get("params", {})
        }

        model = get_model(name, local)
        pred_df = run_backtest(series, model, local)
        results[name.upper()] = pred_df
    return results

# ---------------------------------------------------------------------
# [DEPRECADO] Wrapper de compatibilidad para exportar Excel localmente
# ---------------------------------------------------------------------
def save_backtest_excel(outxlsx: str, summary: pd.DataFrame, preds_map: Dict[str, pd.DataFrame]) -> None:
    """
    [DEPRECADO] Mantiene compatibilidad con código legado.
    Redirige al exportador central en reportes/reportes_excel.py.
    """
    import warnings
    warnings.warn(
        "save_backtest_excel está deprecado: se delega en reportes.reportes_excel.export_backtest_result",
        DeprecationWarning,
        stacklevel=2,
    )
    config_min = {"outxlsx": outxlsx}
    _export_backtest_result(outxlsx, summary, preds_map, config_min)

# ====== BACKTEST OUTPUT STANDARDIZATION & METRICS (GENÉRICAS) ======

import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _ensure_series(x) -> pd.Series:
    """Convierte ndarray/DataFrame/Series a Series con índice original si aplica."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        # si viene con 1 columna, úsala
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        # si trae columnas conocidas, prioriza 'yhat'
        for c in ("yhat", "pred", "forecast"):
            if c in x.columns:
                return x[c]
        # fallback: primera
        return x.iloc[:, 0]
    # ndarray
    return pd.Series(x)

def build_backtest_frame(
    price: pd.Series,
    pred: pd.Series | pd.DataFrame | np.ndarray,
    horizon: int,
    model_name: str,
    freq_hint: str | None = None,
) -> pd.DataFrame:
    """
    Estándar único de salida para backtests.
    Columnas resultantes:
      - ds      : timestamp (índice)
      - y       : precio real en t+h (target futuro)
      - yhat    : pronóstico hecho en t para t+h
      - base    : precio observado en t (referencia)
      - model   : nombre del modelo
      - horizon : pasos adelante usados
      - freq    : 'H'/'D' si se dispone
    """
    s_price = price.astype(float).copy()
    s_pred = _ensure_series(pred).astype(float)
    s_pred = s_pred.reindex(s_price.index)  # alinea a índice del precio si aplica

    # y_true = precio futuro
    y_true = s_price.shift(-horizon)

    df = pd.DataFrame({
        "ds": s_price.index,
        "y": y_true.values,
        "yhat": s_pred.values,
        "base": s_price.values,
    })
    df["model"] = str(model_name)
    df["horizon"] = int(horizon)
    df["freq"] = None if freq_hint is None else str(freq_hint)
    return df.dropna(subset=["yhat"]).reset_index(drop=True)

def compute_generic_metrics(
    df_bt: pd.DataFrame,
    pip_size: float = 0.0001
) -> Dict[str, float]:
    """
    Métricas genéricas sobre el backtest estandarizado (horizon=1 recomendado):
      - MAE, RMSE, MAPE
      - Accuracy direccional (signo de (y - base) vs (yhat - base))
      - Sortino (con una lógica simple de señales direccionales)
    """
    df = df_bt.dropna(subset=["y", "yhat", "base"]).copy()
    if df.empty:
        return {}

    # errores absolutos
    err = df["y"] - df["yhat"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    # MAPE (evitar división por cero)
    denom = np.where(df["y"] == 0, np.nan, np.abs(df["y"]))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)

    # accuracy direccional: compara signos de movimientos
    real_dir = np.sign(df["y"] - df["base"])
    pred_dir = np.sign(df["yhat"] - df["base"])
    acc = float(np.mean((real_dir == pred_dir).astype(float)))

    # sortino: estrategia “long si pred_dir>0, short si pred_dir<0”
    # Componemos un retorno ‘siguiente’ aprox. con y/base
    ret_real = (df["y"] - df["base"]) / df["base"]
    ret_strat = pred_dir * ret_real  # señal * retorno real
    # downside deviation
    downside = ret_strat.copy()
    downside[downside > 0] = 0.0
    dd = float(np.sqrt(np.mean(np.square(downside))))
    mean_ret = float(np.mean(ret_strat))
    sortino = float(mean_ret / dd) if dd > 0 else np.inf

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "Directional_Accuracy": acc,
        "Sortino": sortino,
    }