# reportes/reportes_excel.py
import os
import pandas as pd
from datetime import datetime

# Requiere: pandas>=1.4, openpyxl instalado

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    raise TypeError("Objeto no convertible a DataFrame")

def _build_dm_nivel_table(dm_dict: dict) -> pd.DataFrame:
    """dm_dict = {model: {'DM':..., 'p_value':..., 'T':...}, ...}"""
    rows = []
    for m, d in dm_dict.items():
        rows.append({
            "model": m,
            "DM_stat": d.get("DM"),
            "p_value": d.get("p_value"),
            "T": d.get("T")
        })
    if not rows:
        return pd.DataFrame(columns=["model","DM_stat","p_value","T"]).set_index("model")
    return pd.DataFrame(rows).set_index("model").sort_index()

def _build_dm_vol_table(vol_preds: pd.DataFrame, loss: str = "mse") -> pd.DataFrame:
    """Devuelve tabla (1 fila) con DM entre garch_rv y har_rv si se puede; sino DataFrame vacío."""
    from scipy import stats
    def diebold_mariano(y_true: pd.Series, y1: pd.Series, y2: pd.Series, h: int = 1, loss: str = "mse") -> dict:
        y_true = y_true.astype(float)
        y1 = y1.reindex_like(y_true).astype(float)
        y2 = y2.reindex_like(y_true).astype(float)
        if loss == "mse":
            d = (y_true - y1) ** 2 - (y_true - y2) ** 2
        else:
            d = (y_true - y1).abs() - (y_true - y2).abs()
        d = d.dropna(); T = len(d); 
        if T < 5:
            return {"DM": float("nan"), "p_value": float("nan"), "T": int(T)}
        dbar = d.mean()
        def acov(x, k): 
            import numpy as np
            return np.cov(x[k:], x[:-k], bias=True)[0, 1] if k > 0 else np.var(x, ddof=0)
        s = acov(d.values, 0)
        for k in range(1, h):
            s += 2 * (1 - k / h) * acov(d.values, k)
        var_dbar = s / T
        DM = dbar / (var_dbar ** 0.5 + 1e-12)
        p = 2 * (1 - stats.t.cdf(abs(DM), df=T - 1))
        return {"DM": float(DM), "p_value": float(p), "T": int(T)}

    df = vol_preds.copy()
    if "rv_true" not in df.columns:
        return pd.DataFrame()
    # Si garch viene como sigma, convertir a var
    if "garch_rv" not in df.columns:
        if "garch_sigma" in df.columns:
            df["garch_rv"] = df["garch_sigma"] ** 2

    if not {"garch_rv", "har_rv"}.issubset(df.columns):
        return pd.DataFrame()

    y = df["rv_true"].astype(float)
    g = df["garch_rv"].astype(float)
    h = df["har_rv"].astype(float)
    common = y.dropna().index.intersection(g.dropna().index).intersection(h.dropna().index)
    if len(common) < 5:
        return pd.DataFrame()

    dm = diebold_mariano(y.loc[common], g.loc[common], h.loc[common], h=1, loss=loss)
    out = pd.DataFrame([{
        "compare": "garch_rv vs har_rv",
        "DM_stat": dm["DM"],
        "p_value": dm["p_value"],
        "T": dm["T"]
    }]).set_index("compare")
    return out

def export_resultados_modelos(
    ruta_excel: str,
    res_nivel_metrics: pd.DataFrame,
    res_nivel_preds: pd.DataFrame,
    res_nivel_dm: dict,
    res_vol_metrics: pd.DataFrame,
    res_vol_preds: pd.DataFrame,
    har_summary_text: str,
    cfg_dict: dict | None = None
):
    """
    Escribe/actualiza un Excel con todas las hojas de resultados del modo 'modelos'.
    - Nivel_Metricas:    métricas de nivel por modelo
    - Nivel_DM_vs_RW:    tabla DM vs RW
    - Vol_Metricas:      métricas de volatilidad
    - Vol_DM:            DM(garch_rv vs har_rv) si es posible
    - Nivel_Predicciones / Vol_Predicciones: series reales y pronósticos
    - HAR_Resumen:       summary() del OLS HAR (como líneas de texto)
    - Config_Resumen:    parámetros clave de ejecución
    """
    _ensure_dir(ruta_excel)

    # Construir DM nivel y DM vol
    dm_nivel_df = _build_dm_nivel_table(res_nivel_dm)
    try:
        dm_vol_df = _build_dm_vol_table(res_vol_preds)
    except Exception:
        dm_vol_df = pd.DataFrame()

    # Config mínima
    if cfg_dict is None:
        cfg_dict = {}
    cfg_rows = []
    for k, v in {
        **cfg_dict,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }.items():
        cfg_rows.append([k, v])
    cfg_df = pd.DataFrame(cfg_rows, columns=["clave", "valor"])

    # HAR summary como tabla 1-col
    har_txt = pd.DataFrame({"HAR_summary": (har_summary_text or "").splitlines()})

    mode = "a" if os.path.exists(ruta_excel) else "w"
    with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
        # Nivel
        _to_df(res_nivel_metrics).to_excel(writer, sheet_name="Nivel_Metricas")
        if not dm_nivel_df.empty:
            dm_nivel_df.to_excel(writer, sheet_name="Nivel_DM_vs_RW")
        _to_df(res_nivel_preds).to_excel(writer, sheet_name="Nivel_Predicciones")

        # Volatilidad
        _to_df(res_vol_metrics).to_excel(writer, sheet_name="Vol_Metricas")
        _to_df(res_vol_preds).to_excel(writer, sheet_name="Vol_Predicciones")
        if not dm_vol_df.empty:
            dm_vol_df.to_excel(writer, sheet_name="Vol_DM")

        # Resúmenes
        cfg_df.to_excel(writer, sheet_name="Config_Resumen", index=False)
        har_txt.to_excel(writer, sheet_name="HAR_Resumen", index=False)
