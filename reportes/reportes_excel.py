# reportes/reportes_excel.py
# =============================================================================
# Exportadores de resultados a Excel:
#  - export_resultados_modelos: reporte "completo" (varias hojas)
#  - export_excel_simple:       reporte "simple"   (hojas mínimas)
#
# También se mantienen utilidades legacy:
#  - write_metrics_sheet
#  - append_history
#
# Requisitos:
#   pandas>=1.4, openpyxl instalado
# =============================================================================

import os
from datetime import datetime
import pandas as pd


# --------------------------- Utilidades internas -----------------------------

def _ensure_dir(path: str):
    """Crea la carpeta contenedora (si no existe) para la ruta de archivo dada."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _to_df(obj):
    """Convierte dict/DataFrame a DataFrame; lanza TypeError si no es posible."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    raise TypeError("Objeto no convertible a DataFrame (esperado dict o DataFrame).")

def _build_dm_nivel_table(dm_dict: dict) -> pd.DataFrame:
    """
    Convierte el dict de DM (nivel) a DataFrame indexado por 'model'.
    dm_dict = { model: {'DM':..., 'p_value':..., 'T':...}, ... }
    """
    rows = []
    for m, d in (dm_dict or {}).items():
        rows.append({
            "model": m,
            "DM_stat": d.get("DM"),
            "p_value": d.get("p_value"),
            "T": d.get("T")
        })
    if not rows:
        return pd.DataFrame(columns=["model","DM_stat","p_value","T"]).set_index("model")
    return pd.DataFrame(rows).set_index("model").sort_index()

def _build_dm_vol_table(vol_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla con la prueba DM entre GARCH y HAR en VOLATILIDAD,
    si las columnas necesarias están disponibles (rv_true, har_rv y garch_rv o garch_sigma).
    Devuelve DataFrame con 1 fila (compare='garch_rv vs har_rv') o vacío.
    """
    from scipy import stats

    def diebold_mariano(y_true: pd.Series, y1: pd.Series, y2: pd.Series,
                        h: int = 1, loss: str = "mse") -> dict:
        y_true = y_true.astype(float)
        y1 = y1.reindex_like(y_true).astype(float)
        y2 = y2.reindex_like(y_true).astype(float)
        if loss == "mse":
            d = (y_true - y1)**2 - (y_true - y2)**2
        else:
            d = (y_true - y1).abs() - (y_true - y2).abs()
        d = d.dropna(); T = len(d)
        if T < 5:
            return {"DM": float("nan"), "p_value": float("nan"), "T": int(T)}
        dbar = d.mean()
        def acov(x, k):
            import numpy as np
            return np.cov(x[k:], x[:-k], bias=True)[0, 1] if k > 0 else np.var(x, ddof=0)
        s = acov(d.values, 0)
        for k in range(1, h):
            s += 2 * (1 - k/h) * acov(d.values, k)
        var_dbar = s / T
        DM = dbar / ((var_dbar ** 0.5) + 1e-12)
        p = 2 * (1 - stats.t.cdf(abs(DM), df=T-1))
        return {"DM": float(DM), "p_value": float(p), "T": int(T)}

    if not isinstance(vol_preds, pd.DataFrame) or vol_preds.empty:
        return pd.DataFrame()

    df = vol_preds.copy()
    if "rv_true" not in df.columns:
        return pd.DataFrame()

    # Convertir sigma -> var si hace falta
    if "garch_rv" not in df.columns and "garch_sigma" in df.columns:
        df["garch_rv"] = df["garch_sigma"]**2

    if not {"garch_rv", "har_rv"}.issubset(df.columns):
        return pd.DataFrame()

    y = df["rv_true"].astype(float)
    g = df["garch_rv"].astype(float)
    h = df["har_rv"].astype(float)
    common = y.dropna().index.intersection(g.dropna().index).intersection(h.dropna().index)
    if len(common) < 5:
        return pd.DataFrame()

    dm = diebold_mariano(y.loc[common], g.loc[common], h.loc[common], h=1, loss="mse")
    out = pd.DataFrame([{
        "compare": "garch_rv vs har_rv",
        "DM_stat": dm["DM"],
        "p_value": dm["p_value"],
        "T": dm["T"]
    }]).set_index("compare")
    return out


# ----------------------- Exportador COMPLETO (modo modelos) -------------------

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
    Escribe/actualiza un Excel con TODAS las hojas del modo 'modelos'.

    Hojas que se crean/reemplazan:
      - Nivel_Metricas:    métricas de nivel por modelo (RMSE/MAE/MAPE/Theil’sU/HitRate)
      - Nivel_DM_vs_RW:    tabla DM vs RW (DM_stat, p_value, T)   [si hay datos]
      - Nivel_Predicciones:serie real y predicciones por modelo (tramo de evaluación)

      - Vol_Metricas:      métricas de volatilidad (RMSE/MAE) para har_rv, garch_rv/σ
      - Vol_Predicciones:  columnas: rv_true, har_rv, garch_rv (o garch_sigma)
      - Vol_DM:            DM(garch_rv vs har_rv)                 [si aplicable]

      - Config_Resumen:    parámetros clave + timestamp
      - HAR_Resumen:       texto del summary() del OLS HAR (una columna de líneas)
    """
    _ensure_dir(ruta_excel)

    # Construir tablas DM
    dm_nivel_df = _build_dm_nivel_table(res_nivel_dm)
    try:
        dm_vol_df = _build_dm_vol_table(res_vol_preds)
    except Exception:
        dm_vol_df = pd.DataFrame()

    # Config mínima
    cfg_dict = cfg_dict or {}
    cfg_rows = []
    for k, v in {**cfg_dict, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.items():
        cfg_rows.append([k, v])
    cfg_df = pd.DataFrame(cfg_rows, columns=["clave", "valor"])

    # HAR summary como tabla de 1 columna
    har_txt = pd.DataFrame({"HAR_summary": (har_summary_text or "").splitlines()})

    file_exists = os.path.exists(ruta_excel)
    if file_exists:
        with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
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
    else:
        with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode="w") as writer:
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


# ------------------------ Exportador SIMPLE (modo simple) ---------------------

def export_excel_simple(
    ruta_excel: str,
    metricas_df: pd.DataFrame,
    dm_dict: dict,
    preds_df: pd.DataFrame,
    config: dict
):
    """
    Crea/actualiza un Excel mínimo con 4 hojas:
      - Metricas_Nivel: RMSE/MAE/MAPE/Theil’sU/HitRate por modelo
      - DM_vs_RW:       estadístico y p-valor del DM (modelo vs RW)
      - Predicciones:   y real + columnas de modelos (tramo test)
      - Config:         parámetros básicos + timestamp

    Este reporte NO sustituye al completo; es una variante ligera.
    """
    _ensure_dir(ruta_excel)

    # DM dict -> DataFrame
    dm_rows = []
    for m, d in (dm_dict or {}).items():
        dm_rows.append({
            "model": m,
            "DM_stat": d.get("DM"),
            "p_value": d.get("p_value"),
            "T": d.get("T")
        })
    dm_df = pd.DataFrame(dm_rows).set_index("model") if dm_rows else pd.DataFrame(columns=["model","DM_stat","p_value","T"])

    # Config -> DataFrame
    cfg = {**(config or {}), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    cfg_df = pd.DataFrame([{"clave": k, "valor": v} for k, v in cfg.items()])

    file_exists = os.path.exists(ruta_excel)
    if file_exists:
        with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            metricas_df.to_excel(writer, sheet_name="Metricas_Nivel")
            if not dm_df.empty:
                dm_df.to_excel(writer, sheet_name="DM_vs_RW")
            preds_df.to_excel(writer, sheet_name="Predicciones")
            cfg_df.to_excel(writer, sheet_name="Config", index=False)
    else:
        with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode="w") as writer:
            metricas_df.to_excel(writer, sheet_name="Metricas_Nivel")
            if not dm_df.empty:
                dm_df.to_excel(writer, sheet_name="DM_vs_RW")
            preds_df.to_excel(writer, sheet_name="Predicciones")
            cfg_df.to_excel(writer, sheet_name="Config", index=False)


# ------------------------ Funciones legacy (compat) --------------------------

def write_metrics_sheet(ruta_excel: str, metrics: dict, sheet_name: str = 'Métricas Modelo'):
    """
    Reemplaza la hoja con las métricas de la última corrida (formato fila única).
    Se mantiene por compatibilidad con versiones previas.
    """
    _ensure_dir(ruta_excel)
    df = pd.DataFrame([metrics])
    with pd.ExcelWriter(ruta_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def append_history(ruta_excel: str, metrics: dict, hist_sheet: str = 'Historico Métricas'):
    """
    Agrega una fila al histórico (crea la hoja si no existe).
    Se mantiene por compatibilidad con versiones previas.
    """
    _ensure_dir(ruta_excel)
    row = pd.DataFrame([metrics])
    try:
        hist_exist = pd.read_excel(ruta_excel, sheet_name=hist_sheet)
        hist_concat = pd.concat([hist_exist, row], ignore_index=True)
    except Exception:
        hist_concat = row

    with pd.ExcelWriter(ruta_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        hist_concat.to_excel(writer, sheet_name=hist_sheet, index=False)
