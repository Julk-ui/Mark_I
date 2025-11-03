# reportes/reportes_excel.py
# -*- coding: utf-8 -*-
"""
Módulo centralizado de exportación de reportes (CSV y Excel) para resultados de backtesting.

Funciones claves expuestas:
- export_backtest_csv_per_model(...)
- export_backtest_excel_consolidado(...)

Puntos importantes:
- NO borra lógica existente en otros módulos: este archivo solo centraliza exportación.
- Calcula y adjunta columnas estándar (y_pred, y_true, error, abs_error, sq_error, direction_*,
  threshold_pips, signal, pnl, pnl_cum, horizon) y fechas de ventana por fila (inicio/fin).
- Permite consolidar varios modelos en un único XLSX con hoja "metrics" + hoja por cada modelo + "config_info" opcional.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utilidades y helpers
# ---------------------------------------------------------------------

def _price_to_ds_ytrue(price: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza la serie de precio a DataFrame con columnas ['ds','y_true'].
    Acepta:
      - Series con índice datetime → ds = index, y_true = values
      - DataFrame con columnas ['ds', 'y'] o ['ds','close'] o ['ds','price'] o ['ds','y_true'].
    """
    if isinstance(price, pd.Series):
        df = pd.DataFrame({"ds": price.index, "y_true": price.values})
    else:
        cols = {c.lower(): c for c in price.columns}
        if "ds" in cols and ("y_true" in cols or "y" in cols or "close" in cols or "price" in cols):
            ycol = cols.get("y_true") or cols.get("y") or cols.get("close") or cols.get("price")
            df = price[[cols["ds"], ycol]].rename(columns={cols["ds"]: "ds", ycol: "y_true"}).copy()
        else:
            # fallback: si el índice es datetime y hay una sola columna
            if isinstance(price.index, pd.DatetimeIndex) and price.shape[1] == 1:
                df = price.copy()
                df = df.reset_index().rename(columns={price.index.name or "index": "ds", df.columns[1]: "y_true"})
            else:
                raise ValueError("No se pudo inferir 'ds' y 'y_true' desde el precio.")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    try:
        df["ds"] = df["ds"].dt.tz_localize(None)
    except Exception:
        pass
    return df.dropna(subset=["ds"]).sort_values("ds")


def _coerce_datetime_series(s: pd.Series) -> pd.Series:
    """Convierte a datetime naive (sin tz) de forma segura."""
    out = pd.to_datetime(s, errors="coerce")
    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def _guess_pred_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intenta adivinar la columna de predicción:
      - y_pred
      - yhat / yhat1 / forecast / pred
      - primera del set {1,2,3,4,5} si existen columnas numéricas de horizonte
    """
    if "y_pred" in df.columns:
        return "y_pred"
    for c in ["yhat", "yhat1", "forecast", "pred"]:
        if c in df.columns:
            return c
    # horizontes numerados
    for k in ["1", 1, "h1", "H1"]:
        if k in df.columns:
            return k
    for k in ["1", "2", "3", "4", "5"]:
        if k in df.columns:
            return k
    return None


def _ensure_cols(
    df: pd.DataFrame,
    *,
    price: pd.Series | pd.DataFrame | None,
    target: str = "returns",
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Normaliza un DataFrame de predicciones para que tenga columnas estándar:
      - ds (datetime), y_pred, y_true (opcional si 'price' no se provee), y luego calcula:
        error, abs_error, sq_error, direction_true, direction_pred.
    """
    out = df.copy()

    # Asegurar ds
    if "ds" not in out.columns:
        # Si vino por índice, lo promovemos a columna ds
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "ds"})
        else:
            raise KeyError("El DataFrame de predicciones debe contener la columna 'ds'.")
    out["ds"] = _coerce_datetime_series(out["ds"])

    # Asegurar y_pred
    pred_col = _guess_pred_column(out)
    if pred_col is None:
        raise KeyError("No se encontró columna de predicción (y_pred/yhat/1/... ) en el DataFrame.")
    if pred_col != "y_pred":
        out = out.rename(columns={pred_col: "y_pred"})

    # Asegurar y_true si hay 'price'
    if price is not None:
        p = _price_to_ds_ytrue(price)
        out = out.merge(p, on="ds", how="left")
    else:
        if "y_true" not in out.columns:
            out["y_true"] = np.nan

    # Errores base
    out["error"] = out["y_true"] - out["y_pred"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2

    # Direcciones
    if target.lower().startswith("level"):
        # dirección = cambio del nivel
        out["direction_true"] = np.sign(out["y_true"].diff())
    else:
        # para returns, el signo del retorno
        out["direction_true"] = np.sign(out["y_true"])
    out["direction_pred"] = np.sign(out["y_pred"])

    # Mantener orden flexible (no forzamos demasiado para no romper hojas previas)
    return out


def compute_directional_accuracy(df: pd.DataFrame) -> Optional[float]:
    """Accuracy de dirección (signo) entre y_true y y_pred. Requiere direction_true y direction_pred."""
    if not {"direction_true", "direction_pred"} <= set(df.columns):
        return None
    mask = df[["direction_true", "direction_pred"]].dropna()
    if mask.empty:
        return None
    return float((np.sign(mask["direction_true"]) == np.sign(mask["direction_pred"])).mean())


def compute_sortino(df: pd.DataFrame, annualization: Optional[float] = None) -> Optional[float]:
    """
    Sortino ratio usando 'pnl' si existe, o 'y_pred'/'y_true' en returns.
    sortino = mean(returns)/std(returns[returns<0])
    """
    series = None
    if "pnl" in df.columns:
        series = df["pnl"].dropna()
    elif {"y_pred", "y_true"} <= set(df.columns):
        # como proxy: señal*retorno si existiera; de lo contrario, diferencia
        series = (df["y_pred"] - df["y_true"]).dropna()
    if series is None or series.empty:
        return None

    downside = series[series < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return None

    mean_r = series.mean()
    std_down = downside.std(ddof=1)

    sortino = mean_r / std_down
    if annualization and annualization > 1:
        sortino *= np.sqrt(annualization)
    return float(sortino)


def compute_sharpe(df: pd.DataFrame, annualization: Optional[float] = None) -> Optional[float]:
    """Sharpe ratio básico usando 'pnl' si existe; de lo contrario, diferencia (y_pred - y_true)."""
    series = None
    if "pnl" in df.columns:
        series = df["pnl"].dropna()
    elif {"y_pred", "y_true"} <= set(df.columns):
        series = (df["y_pred"] - df["y_true"]).dropna()
    if series is None or series.empty:
        return None

    mu = series.mean()
    sd = series.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return None

    sharpe = mu / sd
    if annualization and annualization > 1:
        sharpe *= np.sqrt(annualization)
    return float(sharpe)


def compute_metrics_generic(
    df: pd.DataFrame,
    *,
    target: str = "returns",
    annualization: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas genéricas (MAE, RMSE, MAPE, Direction_Accuracy, Sortino, Sharpe)
    si existen 'y_true' y 'y_pred' (y, opcionalmente, pnl para ratios).
    """
    out: Dict[str, Any] = {}
    if not {"y_true", "y_pred"} <= set(df.columns):
        return out

    # MAE / RMSE / MAPE
    err = (df["y_true"] - df["y_pred"]).dropna()
    if not err.empty:
        out["MAE"] = float(err.abs().mean())
        out["RMSE"] = float(np.sqrt((err ** 2).mean()))
        y_true_nonzero = df["y_true"].replace(0, np.nan)
        mape = (err.abs() / y_true_nonzero).dropna()
        out["MAPE"] = float(mape.mean()) if not mape.empty else None
    else:
        out["MAE"] = out["RMSE"] = out["MAPE"] = None

    # Directional accuracy
    out["Direction_Accuracy"] = compute_directional_accuracy(df)

    # Sortino / Sharpe (basado en pnl si está, si no usa diferencia)
    out["Sortino"] = compute_sortino(df, annualization=annualization)
    out["Sharpe"] = compute_sharpe(df, annualization=annualization)

    return out


def _inject_window_bounds_per_row(
    df: pd.DataFrame,
    *,
    initial_train: Optional[int],
    price_index: Optional[pd.DatetimeIndex],
) -> pd.DataFrame:
    """
    Inserta por fila las columnas 'fecha_inicio_ventana' y 'fecha_fin_ventana' basadas en:
      - Posición de ds en el índice real de precio (price_index) si se proporciona.
      - initial_train: tamaño de la ventana de entrenamiento usada en el rolling walk-forward.
    Si no hay price_index o initial_train, pone el rango global como fallback.
    """
    out = df.copy()
    out["fecha_inicio_ventana"] = pd.NaT
    out["fecha_fin_ventana"] = pd.NaT

    if price_index is None or initial_train is None or initial_train <= 0:
        # Fallback: rango global
        if "ds" in out.columns and not out["ds"].isna().all():
            out["fecha_inicio_ventana"] = out["ds"].min()
            out["fecha_fin_ventana"] = out["ds"].max()
        return out

    # Creamos un lookup de posición en el índice real
    pos = pd.Series(index=price_index, data=np.arange(len(price_index)))
    ds_clean = _coerce_datetime_series(out["ds"])
    for i in range(len(out)):
        dsi = ds_clean.iloc[i]
        if pd.isna(dsi) or dsi not in pos.index:
            continue
        j = int(pos.loc[dsi])
        j0 = max(0, j - initial_train + 1)
        out.at[i, "fecha_inicio_ventana"] = price_index[j0]
        out.at[i, "fecha_fin_ventana"] = price_index[j]
    return out


def _add_eval_columns(
    df: pd.DataFrame,
    pip_size: float | None,
    threshold_mode: Optional[str] = None,
    threshold_pips: Optional[float] = None,
    target: str = "returns",
    horizon: int = 1,
    *,
    initial_train: Optional[int] = None,
    price_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Agrega columnas de evaluación + horizonte + ventana temporal (por fila si hay initial_train).
    - Si 'price_index' viene del precio original, las ventanas por fila usan ese índice real
      y evitan que 'fecha_inicio_ventana' quede constante.
    """
    out = df.copy()

    # Señales por umbral (opcional)
    out["threshold_pips"] = np.nan
    out["signal"] = 0.0
    if threshold_mode and threshold_pips and "y_pred" in out.columns:
        pred_pips = None
        if pip_size and target.lower().startswith("return"):
            pred_pips = out["y_pred"] / float(pip_size)
        else:
            if "y_true" in out.columns and pip_size:
                pred_pips = (out["y_pred"] - out["y_true"]) / float(pip_size)

        if pred_pips is not None:
            thr = float(threshold_pips)
            out["threshold_pips"] = thr
            out["signal"] = np.where(pred_pips > thr, 1.0, np.where(pred_pips < -thr, -1.0, 0.0))

    # P&L (simple) si tenemos retornos reales
    out["pnl"] = np.nan
    out["pnl_cum"] = np.nan
    if "signal" in out.columns and "y_true" in out.columns and target.lower().startswith("return"):
        pnl = out["signal"].shift(0) * out["y_true"].fillna(0.0)
        out["pnl"] = pnl
        out["pnl_cum"] = pnl.cumsum()

    # Horizonte
    out["horizon"] = horizon

    # Ventanas por fila (rolling) o global usando el índice REAL del precio si está disponible
    out = _inject_window_bounds_per_row(
        out,
        initial_train=initial_train,
        price_index=price_index,
    )

    return out


# ---------------------------------------------------------------------
# Exportadores
# ---------------------------------------------------------------------

@dataclass
class ModelExportMeta:
    model_name: str
    start_ds: Optional[pd.Timestamp]
    end_ds: Optional[pd.Timestamp]
    params: Dict[str, Any]
    horizon: int
    target: str


def _meta_from_df(model_name: str, df: pd.DataFrame, *, horizon: int, target: str, params: Dict[str, Any]) -> ModelExportMeta:
    """Crea metadatos simples para una hoja de modelo."""
    start_ds = pd.to_datetime(df["ds"]).min() if "ds" in df.columns and not df.empty else None
    end_ds = pd.to_datetime(df["ds"]).max() if "ds" in df.columns and not df.empty else None
    return ModelExportMeta(
        model_name=model_name,
        start_ds=start_ds,
        end_ds=end_ds,
        params=params or {},
        horizon=horizon,
        target=target,
    )


def export_backtest_csv_per_model(
    symbol: str,
    pred_map: Dict[str, pd.DataFrame],
    price: pd.Series | pd.DataFrame | None,
    outdir: Path,
    target: str = "returns",
    pip_size: Optional[float] = None,
    threshold_mode: Optional[str] = None,
    threshold_pips: Optional[float] = None,
    horizon: int = 1,
    *,
    initial_train: Optional[int] = None,
) -> None:
    """
    Exporta CSV simples por modelo (una línea por 'ds') con columnas estándar y
    ventanas por fila (si se indica initial_train).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Índice real del precio para el cálculo correcto de ventanas por fila
    price_idx = None
    if isinstance(price, (pd.Series, pd.DataFrame)):
        price_idx = _price_to_ds_ytrue(price)["ds"]
        price_idx = pd.to_datetime(price_idx, errors="coerce")
        try:
            price_idx = price_idx.dt.tz_localize(None)
        except Exception:
            pass
        price_idx = pd.DatetimeIndex(price_idx.dropna().unique()).sort_values()

    for model_name, df in pred_map.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cur = _ensure_cols(df, price=price, target=target, horizon=horizon)
        cur = _add_eval_columns(
            cur,
            pip_size=pip_size,
            threshold_mode=threshold_mode,
            threshold_pips=threshold_pips,
            target=target,
            horizon=horizon,
            initial_train=initial_train,
            price_index=price_idx,
        )
        fname = f"{symbol}_{model_name}_backtest.csv"
        cur.to_csv(outdir / fname, index=False)


def export_backtest_excel_consolidado(
    symbol: str,
    pred_map: Dict[str, pd.DataFrame],
    price: pd.Series | pd.DataFrame | None,
    excel_path: Path,
    *,
    target: str = "returns",
    pip_size: Optional[float] = None,
    threshold_mode: Optional[str] = None,
    threshold_pips: Optional[float] = None,
    horizon: int = 1,
    annualization: Optional[float] = None,
    per_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    config_info: Optional[Dict[str, Any]] = None,
    seasonality_m: int = 1,
    initial_train: Optional[int] = None,
) -> None:
    """
    Exporta un Excel con:
      - 'metrics': resumen por modelo (MAE, RMSE, MAPE, Direction_Accuracy, Sortino, Sharpe, fechas).
      - Una hoja por modelo con datos + metadatos arriba.
      - 'config_info' (opcional) para trazabilidad de la configuración usada.
    """
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Índice real del precio (para ventanas por fila)
    price_idx = None
    if isinstance(price, (pd.Series, pd.DataFrame)):
        price_idx = _price_to_ds_ytrue(price)["ds"]
        price_idx = pd.to_datetime(price_idx, errors="coerce")
        try:
            price_idx = price_idx.dt.tz_localize(None)
        except Exception:
            pass
        price_idx = pd.DatetimeIndex(price_idx.dropna().unique()).sort_values()

    rows_metrics: List[Dict[str, Any]] = []
    per_model_sheets: Dict[str, pd.DataFrame] = {}
    per_model_meta: Dict[str, ModelExportMeta] = {}

    for model_name, df in pred_map.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        cur = _ensure_cols(df, price=price, target=target, horizon=horizon)
        cur = _add_eval_columns(
            cur,
            pip_size=pip_size,
            threshold_mode=threshold_mode,
            threshold_pips=threshold_pips,
            target=target,
            horizon=horizon,
            initial_train=initial_train,
            price_index=price_idx,
        )

        # Métricas
        m = compute_metrics_generic(cur, target=target, annualization=annualization) or {}

        # Rango evaluado informado (usamos la primera fila si _inject_window_bounds_per_row generó por-fila)
        if not cur.empty and {"fecha_inicio_ventana", "fecha_fin_ventana"} <= set(cur.columns):
            start_ds = pd.to_datetime(cur["fecha_inicio_ventana"].iloc[0])
            end_ds = pd.to_datetime(cur["fecha_fin_ventana"].iloc[0])
        else:
            start_ds = pd.to_datetime(cur["ds"]).min() if "ds" in cur.columns else pd.NaT
            end_ds = pd.to_datetime(cur["ds"]).max() if "ds" in cur.columns else pd.NaT

        m_row = {
            "model": model_name,
            **m,
            "fecha_inicio_ventana": start_ds,
            "fecha_fin_ventana": end_ds,
        }
        rows_metrics.append(m_row)

        per_model_sheets[model_name] = cur
        params = (per_model_params or {}).get(model_name, {})
        per_model_meta[model_name] = _meta_from_df(model_name, cur, horizon=horizon, target=target, params=params)

    metrics_df = pd.DataFrame(rows_metrics) if rows_metrics else pd.DataFrame(
        columns=[
            "model", "MAE", "RMSE", "MAPE", "Direction_Accuracy", "Sortino", "Sharpe",
            "fecha_inicio_ventana", "fecha_fin_ventana"
        ]
    )

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as xw:
        metrics_df.to_excel(xw, sheet_name="metrics", index=False)

        for model_name, data in per_model_sheets.items():
            sheet = model_name[:31] if len(model_name) > 31 else model_name
            meta = per_model_meta[model_name]
            meta_rows = [
                ["symbol", symbol],
                ["model", meta.model_name],
                ["target", meta.target],
                ["horizon", meta.horizon],
                ["start_ds", meta.start_ds.isoformat() if meta.start_ds is not None and pd.notna(meta.start_ds) else ""],
                ["end_ds", meta.end_ds.isoformat() if meta.end_ds is not None and pd.notna(meta.end_ds) else ""],
                ["threshold_mode", threshold_mode or ""],
                ["threshold_pips", threshold_pips if threshold_pips is not None else ""],
                ["pip_size", pip_size if pip_size is not None else ""],
            ]
            params = meta.params or {}
            for k, v in params.items():
                meta_rows.append([f"param.{k}", str(v)])

            pd.DataFrame(meta_rows, columns=["key", "value"]).to_excel(
                xw, sheet_name=sheet, index=False, startrow=0
            )
            start_row = len(meta_rows) + 2
            data.to_excel(xw, sheet_name=sheet, index=False, startrow=start_row)

        if config_info:
            flat_rows = []
            def _flatten(prefix: str, obj: Any):
                if isinstance(obj, dict):
                    for kk, vv in obj.items():
                        _flatten(f"{prefix}.{kk}" if prefix else kk, vv)
                else:
                    flat_rows.append([prefix, str(obj)])
            _flatten("", config_info)
            pd.DataFrame(flat_rows, columns=["config_key", "value"]).to_excel(
                xw, sheet_name="config_info", index=False
            )
