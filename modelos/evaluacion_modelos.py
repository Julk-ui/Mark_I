# modelos/evaluacion_modelos.py
# Evaluación simple y clara: backtest de 1 split + walk-forward opcional,
# métricas comprensibles, DM vs baseline y gráficos básicos.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ===== Métricas básicas =====
def rmse(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def theils_u(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Relativo a un random walk: <1 mejora, 1 igual, >1 peor.
    Implementación simple: compara el error vs diferencias de y_true.
    """
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float).reindex_like(y_true)
    e = (y_pred - y_true).values
    dy = np.diff(y_true.values, prepend=y_true.values[0])
    denom = np.sum(dy[1:]**2)
    return float(np.sqrt(np.sum(e[1:]**2) / (denom + 1e-12)))

def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    % de aciertos en la DIRECCIÓN (sube/baja).
    """
    yt = pd.Series(y_true).astype(float)
    yp = pd.Series(y_pred).astype(float).reindex_like(yt)
    dir_t = np.sign(yt.diff()).dropna()
    dir_p = np.sign(yp.diff()).reindex(dir_t.index).dropna()
    return float((dir_t == dir_p).mean())

def diebold_mariano(y_true: pd.Series, y1: pd.Series, y2: pd.Series, h: int = 1, loss: str = "mse") -> dict:
    """
    DM test: compara y1 contra y2 (p.ej., modelo vs baseline).
    - loss='mse' o 'mae'
    """
    y_true = pd.Series(y_true).astype(float)
    y1 = pd.Series(y1).reindex_like(y_true).astype(float)
    y2 = pd.Series(y2).reindex_like(y_true).astype(float)

    if loss=="mse":
        d = (y_true - y1)**2 - (y_true - y2)**2
    else:
        d = (y_true - y1).abs() - (y_true - y2).abs()
    d = d.dropna(); T = len(d)
    if T < 10:
        return {"DM": np.nan, "p_value": np.nan, "T": int(T)}

    dbar = d.mean()
    def acov(x,k): return np.cov(x[k:], x[:-k], bias=True)[0,1] if k>0 else np.var(x, ddof=0)
    s = acov(d.values,0)
    for k in range(1,h): s += 2*(1-k/h)*acov(d.values,k)
    var_dbar = s / T
    DM = dbar / np.sqrt(var_dbar + 1e-12)
    p = 2*(1 - stats.t.cdf(np.abs(DM), df=T-1))
    return {"DM": float(DM), "p_value": float(p), "T": int(T)}

# ===== Gráficos simples =====
def plot_level_preds(df: pd.DataFrame, out_path: str, title: str = "Precio real vs predicciones"):
    """
    df con columnas: 'y' (real) y modelos ('rw','arima','ets','lstm'…).
    """
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["y"], label="Real", linewidth=2)
    for c in [c for c in df.columns if c!="y"]:
        plt.plot(df.index, df[c], label=c.upper(), alpha=0.9)
    plt.title(title); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()

def plot_trend(price: pd.Series, out_path: str, windows=(20,60)):
    """
    Muestra la serie y medias móviles (tendencia) para contexto visual.
    """
    plt.figure(figsize=(10,4))
    plt.plot(price.index, price.values, label="Precio", linewidth=1.5)
    for w in windows:
        plt.plot(price.index, price.rolling(w).mean(), label=f"MA{w}")
    plt.title("Tendencia (medias móviles)")
    plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()

# ===== Backtests =====
def backtest_un_split(price: pd.Series, pasos_test: int,
                      fit_funcs: dict, forecast_funcs: dict,
                      outdir: str) -> dict:
    """
    Un solo split (train|test) para simplicidad.
    - fit_funcs: {'arima': callable, 'ets': callable, 'lstm': callable?}
    - forecast_funcs: {'arima': callable, ...} que reciban (fit, steps, index)
    Devuelve df con predicciones y tabla de métricas + DM vs RW.
    """
    price = price.dropna()
    assert pasos_test < len(price), "pasos_test debe ser menor que el total"
    train = price.iloc[:-pasos_test]
    test  = price.iloc[-pasos_test:]
    test_idx = test.index

    # Baseline
    from modelos.baseline import predict_random_walk
    preds = {"rw": predict_random_walk(price, 1).reindex(test_idx)}

    # Modelos
    if "arima" in fit_funcs:
        ar_fit = fit_funcs["arima"](train)
        preds["arima"] = forecast_funcs["arima"](ar_fit, steps=len(test_idx), index=test_idx)
    if "ets" in fit_funcs:
        ets_fit = fit_funcs["ets"](train)
        preds["ets"] = forecast_funcs["ets"](ets_fit, steps=len(test_idx), index=test_idx)
    if "lstm" in fit_funcs:
        try:
            lstm_state = fit_funcs["lstm"](train)
            preds["lstm"] = forecast_funcs["lstm"](lstm_state, price).reindex(test_idx)
        except Exception as e:
            print(f"⚠️ LSTM omitido ({e}).")

    # Tabla consolidada
    df = pd.concat([test.rename("y")] + [p.rename(k) for k,p in preds.items()], axis=1).dropna()

    # Métricas por modelo
    rows = []
    for m in [c for c in df.columns if c!="y"]:
        rows.append({
            "model": m,
            "RMSE": rmse(df["y"], df[m]),
            "MAE": mae(df["y"], df[m]),
            "MAPE(%)": mape(df["y"], df[m]),
            "TheilsU": theils_u(df["y"], df[m]),
            "HitRate": hit_rate(df["y"], df[m])
        })
    met = pd.DataFrame(rows).set_index("model").sort_values(["RMSE","TheilsU"])

    # DM vs RW
    dm = {}
    for m in met.index:
        if m != "rw":
            dm[m] = diebold_mariano(df["y"], df[m], df["rw"], h=1, loss="mse")

    # Gráficos (predicciones + tendencia)
    os.makedirs(outdir, exist_ok=True)
    plot_level_preds(df, os.path.join(outdir, "nivel_pred_vs_real.png"))
    plot_trend(price, os.path.join(outdir, "tendencia_ma.png"))

    # Persistir CSVs simples
    df.to_csv(os.path.join(outdir, "predicciones_nivel.csv"))
    met.to_csv(os.path.join(outdir, "metricas_nivel.csv"))

    return {"preds": df, "metrics": met, "dm": dm}

def walk_forward_simple(price: pd.Series, horizon: int, n_splits: int = 5,
                        fit_funcs: dict = None, forecast_funcs: dict = None) -> pd.DataFrame:
    """
    Walk-forward muy simple: repite n_splits con una ventana creciente y
    test de longitud 'horizon'. Devuelve un DataFrame con métricas promedio.
    (Úsalo cuando quieras reportar medias/intervalos).
    """
    if fit_funcs is None or forecast_funcs is None:
        return pd.DataFrame()

    chunks = []
    N = len(price)
    min_train = max(200, int(0.6*N))
    step = max(horizon, int(0.1*N))
    i = min_train
    for s in range(n_splits):
        tr = price.iloc[:i]
        te = price.iloc[i:i+horizon]
        if len(te) < horizon: break
        res = backtest_un_split(price.iloc[:i+horizon], pasos_test=horizon,
                                fit_funcs=fit_funcs, forecast_funcs=forecast_funcs,
                                outdir=os.path.join("outputs","eda","WF_tmp"))
        metr = res["metrics"].assign(split=s)
        chunks.append(metr.reset_index())
        i += step
    if not chunks:
        return pd.DataFrame()
    all_m = pd.concat(chunks, ignore_index=True)
    return all_m.groupby("model").agg(["mean","std"])
