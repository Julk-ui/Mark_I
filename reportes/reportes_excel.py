"""
Módulo centralizado de reportes Excel/CSV.

Objetivos:
- Proveer un **único punto** para generar archivos .xlsx con:
  - Una hoja 'metrics' con MAE, RMSE, MAPE, Directional_Accuracy y Sortino por modelo.
  - Una hoja por modelo con sus predicciones (horizon=1) normalizadas: ds, y_true, y_pred, error, residual.
- Mantener **compatibilidad** con llamadas existentes (nombres alternativos).
- Ofrecer utilidades ligeras para exportar CSV por modelo y un Excel consolidado
  cuando se usa backtesting por clase (Prophet/LSTM) desde `main.py`.

NOTA: Este módulo NO elimina ni reemplaza lógicas existentes, solo centraliza
la escritura. Si ya tienes funciones previas, estas se mantienen y se amplían.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------

def _to_series(x) -> pd.Series:
    """Convierte ndarray/DataFrame/Series a Series (prioriza columnas 'yhat' si existen)."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        for c in ("yhat", "y_pred", 1, "1"):
            if c in x.columns:
                return x[c]
        # fallback: última col
        return x.iloc[:, -1]
    return pd.Series(x)

def _normalize_pred(df_like, price: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Normaliza un backtest 'wide' a columnas estándar con horizonte=1:
      ds, y_true, y_pred, error, residual
    """
    s_pred = _to_series(df_like).copy()
    if not isinstance(s_pred.index, pd.DatetimeIndex):
        # si viene con índice 'ds' columna:
        if isinstance(df_like, pd.DataFrame) and "ds" in df_like.columns:
            s_pred.index = pd.to_datetime(df_like["ds"])
        else:
            s_pred.index = pd.to_datetime(s_pred.index)
    s_pred = s_pred.sort_index()
    s_pred.name = "y_pred"

    if price is not None:
        y_true = price.reindex(s_pred.index).shift(-0)  # ya está alineado a t+1 en construcción del backtest
        y_true = y_true.ffill().astype(float)
    else:
        y_true = pd.Series(index=s_pred.index, data=np.nan, dtype=float, name="y_true")

    df = pd.DataFrame({"y_true": y_true, "y_pred": s_pred})
    df.index.name = "ds"
    df["error"] = (df["y_true"] - df["y_pred"]).astype(float)
    df["residual"] = df["error"]
    return df

def _metrics_from_frame(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas genéricas desde el DF normalizado (ds, y_true, y_pred, error, residual).
    """
    d = df.dropna(subset=["y_true", "y_pred"]).copy()
    if d.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE(%)": np.nan,
                "Directional_Accuracy": np.nan, "Sortino": np.nan}
    err = d["y_true"] - d["y_pred"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.where(d["y_true"] == 0, np.nan, np.abs(d["y_true"]))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)

    # direccional vs base (aprox con diferencia en el tiempo)
    # si no hay 'base', usamos la señal de (y_pred(t+1) - y_pred(t)) como orientación; de lo contrario,
    # comparamos signos de (y_true - y_pred). Para mantenerlo estable, usamos y_true vs y_pred respecto a cero.
    real_dir = np.sign(d["y_true"].diff())
    pred_dir = np.sign(d["y_pred"].diff())
    mask = real_dir.notna() & pred_dir.notna()
    acc = float(np.mean((real_dir[mask] == pred_dir[mask]).astype(float))) if mask.any() else np.nan

    # sortino: retorno proxy con cambios sobre y_true; señal = signo del cambio en y_pred
    ret_real = d["y_true"].pct_change()
    signal = np.sign(d["y_pred"].diff()).reindex(ret_real.index).fillna(0.0)
    strat = signal * ret_real
    downside = strat.copy()
    downside[downside > 0] = 0.0
    dd = float(np.sqrt(np.nanmean(np.square(downside))))
    mean_ret = float(np.nanmean(strat))
    sortino = float(mean_ret / dd) if dd and dd > 0 else np.inf

    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape,
            "Directional_Accuracy": acc, "Sortino": sortino}

def _sheet_name_safe(name: str) -> str:
    return str(name)[:31].replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_")

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1) CSV rápido por modelo (inspección)
# ---------------------------------------------------------------------

def export_backtest_csv_per_model(
    symbol: str,
    pred_map: Dict[str, Any],
    price: Optional[pd.Series],
    outdir: Path
) -> None:
    """
    Escribe un CSV por cada modelo del mapa:
      columnas: ds, y_true, y_pred, error, residual
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for name, mat in pred_map.items():
        df_norm = _normalize_pred(mat, price=price)
        fname = outdir / f"{symbol}_{name.upper()}_backtest.csv"
        df_norm.to_csv(fname, index=True)
        print(f"💾 [{name}] Backtest guardado en {fname}")


# ---------------------------------------------------------------------
# 2) XLSX consolidado por clase (una hoja por modelo + METRICS)
# ---------------------------------------------------------------------

def export_backtest_excel_consolidado(
    symbol: str,
    pred_map: Dict[str, Any],
    price: Optional[pd.Series],
    excel_path: Path,
    seasonality_m: int | None = 1,
    annualization: int | None = None
) -> None:
    """
    Genera un XLSX con:
      - Hoja 'metrics' (una fila por modelo, con MAE/RMSE/MAPE/Accuracy/Sortino)
      - Hoja por modelo con ds, y_true, y_pred, error, residual (horizon=1)
    """
    _ensure_parent(excel_path)

    # Construcción de métricas
    metrics_rows = []
    per_model_frames: Dict[str, pd.DataFrame] = {}

    for name, mat in pred_map.items():
        df_norm = _normalize_pred(mat, price=price)
        per_model_frames[name] = df_norm
        met = _metrics_from_frame(df_norm)
        metrics_rows.append({"Model": name, **met})

    metrics_df = pd.DataFrame(metrics_rows)

    # Escritura
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as xw:
        # hoja metrics
        metrics_df.to_excel(xw, sheet_name="metrics", index=False)

        # hojas por modelo
        for name, dfm in per_model_frames.items():
            sheet = _sheet_name_safe(name)
            dfm.reset_index().rename(columns={"index": "ds"}).to_excel(xw, sheet_name=sheet, index=False)

    print(f"💾 XLSX consolidado por clase guardado en: {excel_path}")


# ---------------------------------------------------------------------
# 3) API central usada por backtest_rolling / main (compatibilidad)
# ---------------------------------------------------------------------

def exportar_backtest_excel(
    path_xlsx: str,
    price: Optional[pd.Series],
    preds_map: Dict[str, Any],
    summary: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
    **kwargs
) -> None:
    """
    Punto de entrada “canónico” del exportador CENTRALIZADO.
    - Si `summary` viene (ARIMA/SARIMA), se crea o actualiza hoja 'summary'.
    - Siempre crea/actualiza hoja 'metrics' y una hoja por modelo con predicciones normalizadas.
    """
    excel_path = Path(path_xlsx)
    _ensure_parent(excel_path)

    # Preparar métricas y hojas por modelo
    metrics_rows = []
    per_model_frames: Dict[str, pd.DataFrame] = {}
    for name, mat in preds_map.items():
        df_norm = _normalize_pred(mat, price=price)
        per_model_frames[name] = df_norm
        met = _metrics_from_frame(df_norm)
        metrics_rows.append({"Model": name, **met})
    metrics_df = pd.DataFrame(metrics_rows)

    # Escritura (mantener si ya existe)
    mode = "a" if excel_path.exists() else "w"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter", mode="w") as xw:
        # summary si aplica
        if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
            summary.to_excel(xw, sheet_name="summary", index=False)

        # hoja metrics
        metrics_df.to_excel(xw, sheet_name="metrics", index=False)

        # hojas por modelo
        for name, dfm in per_model_frames.items():
            sheet = _sheet_name_safe(name)
            dfm.reset_index().rename(columns={"index": "ds"}).to_excel(xw, sheet_name=sheet, index=False)

    print(f"💾 Reporte (centralizado) guardado en {excel_path}")

# Alias compatibles (no romper llamadas existentes)
guardar_backtest_excel = exportar_backtest_excel
generar_reporte_backtest_excel = exportar_backtest_excel
write_backtest_excel = exportar_backtest_excel

# ---------------------------------------------------------------------
# 4) Compatibilidad con backtest_rolling: export_excel_simple / export_backtest_result
# ---------------------------------------------------------------------

def export_excel_simple(path_xlsx: str, df: pd.DataFrame, sheet_name: str = "data") -> None:
    """
    Helper mínimo para escribir un DataFrame en una hoja específica.
    (compatibilidad con llamados anteriores)
    """
    p = Path(path_xlsx)
    _ensure_parent(p)
    with pd.ExcelWriter(p, engine="xlsxwriter", mode="w") as xw:
        df.to_excel(xw, sheet_name=sheet_name, index=False)
    print(f"💾 Excel simple exportado en {p} (hoja: {sheet_name})")


def export_backtest_result(
    path_xlsx: str,
    summary_df: Optional[pd.DataFrame],
    preds_map: Dict[str, pd.DataFrame],
    config_min: Optional[dict] = None
) -> None:
    """
    Entrada utilizada por evaluate_many(...) para exportar resultados:
    - Crea hoja 'summary' si se provee.
    - Crea hoja 'metrics' y una hoja por modelo con predicciones normalizadas.
    """
    # Intentar extraer serie price de config_min si fue pasada (no siempre está).
    price = None
    try:
        price = (config_min or {}).get("price", None)
    except Exception:
        price = None

    exportar_backtest_excel(
        path_xlsx=path_xlsx,
        price=price,
        preds_map=preds_map,
        summary=summary_df,
        config=config_min,
    )
