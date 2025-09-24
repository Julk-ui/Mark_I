# ==========================================
# app/main.py  (versión integrada MT5 + modelos · FIX steps)
# ==========================================
import os, sys, argparse, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# ----- Stats/ML -----
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from reportes.reportes_excel import export_resultados_modelos

# TensorFlow (opcional para LSTM)
_TF_OK = True
try:
    from tensorflow import keras
except Exception as _e:
    _TF_OK = False
    print(f"⚠️ TensorFlow no disponible ({_e}). LSTM se omitirá.")

# ARCH para GARCH
from arch import arch_model

# ----- Tu conexión MT5 -----
from conexion.easy_Trading import Basic_funcs

# (Opcional) mantén Prophet si existe en tu repo
try:
    from modelos.prophet_model import entrenar_modelo_prophet, predecir_precio
    _PROPHET_OK = True
except Exception as _e:
    _PROPHET_OK = False
    print(f"⚠️ Prophet no disponible ({_e}).")

# (Opcional) tu EDA si existe
try:
    from procesamiento.eda_crispdm import ejecutar_eda
    _EDA_OK = True
except Exception:
    _EDA_OK = False

# ==================== MÉTRICAS ====================
def rmse(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

def theils_u(y_true: pd.Series, y_pred: pd.Series) -> float:
    e = y_pred.values - y_true.values
    dy = np.diff(y_true.values, prepend=y_true.values[0])
    denom = np.sum(dy[1:]**2)
    return float(np.sqrt(np.sum(e[1:]**2) / (denom + 1e-12)))

def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    dy_t = y_true.diff().dropna()
    dy_p = y_pred.diff().reindex(dy_t.index).dropna()
    return float((np.sign(dy_t) == np.sign(dy_p)).mean())

def diebold_mariano(y_true: pd.Series, y1: pd.Series, y2: pd.Series, h: int = 1, loss: str = "mse") -> dict:
    y_true = y_true.astype(float); y1 = y1.reindex_like(y_true).astype(float); y2 = y2.reindex_like(y_true).astype(float)
    if loss=="mse":
        d = (y_true - y1)**2 - (y_true - y2)**2
    else:
        d = np.abs(y_true - y1) - np.abs(y_true - y2)
    d = d.dropna(); T = len(d); dbar = d.mean()
    def acov(x,k): return np.cov(x[k:], x[:-k], bias=True)[0,1] if k>0 else np.var(x, ddof=0)
    s = acov(d.values,0)
    for k in range(1,h): s += 2*(1-k/h)*acov(d.values,k)
    var_dbar = s / T
    DM = dbar / np.sqrt(var_dbar + 1e-12)
    p = 2*(1 - stats.t.cdf(np.abs(DM), df=T-1))
    return {"DM": float(DM), "p_value": float(p), "T": int(T)}

# ==================== FEATURES (RV/HAR) ====================
LOG2 = np.log(2.0)

def log_returns(price: pd.Series) -> pd.Series:
    r = np.log(price).diff().dropna(); r.name = "ret"; return r

def parkinson_rv(high: pd.Series, low: pd.Series) -> pd.Series:
    rv = (1.0 / (4.0 * LOG2)) * (np.log(high/low) ** 2)
    rv = rv.replace([np.inf, -np.inf], np.nan).dropna(); rv.name = "rv"; return rv

def har_lags(rv: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"rv": rv})
    df["rv_d"] = df["rv"].shift(1)
    df["rv_w"] = df["rv"].rolling(5).mean().shift(1)
    df["rv_m"] = df["rv"].rolling(22).mean().shift(1)
    Xy = pd.concat([np.log(df["rv"]).rename("log_rv"), df[["rv_d","rv_w","rv_m"]]], axis=1).dropna()
    return Xy

def add_iv(Xy: pd.DataFrame, iv: pd.Series | None) -> pd.DataFrame:
    if iv is None: return Xy
    ivl = np.log(iv.rename("log_iv")).reindex(Xy.index)
    return Xy.join(ivl, how="left").dropna()

# ==================== MODELOS (NIVEL) ====================
def predict_random_walk(price: pd.Series, horizon: int = 1) -> pd.Series:
    pred = price.shift(horizon); pred.name = "rw"; return pred

def fit_arima_series(price: pd.Series, order=(1,1,1)):
    m = ARIMA(price.astype(float), order=order, enforce_stationarity=False, enforce_invertibility=False)
    return m.fit()

# ---- NUEVO: pronóstico por "steps" (no por fechas) ----
def arima_forecast_steps(fit, steps: int, index) -> pd.Series:
    fc = fit.get_forecast(steps=steps).predicted_mean
    return pd.Series(np.asarray(fc), index=index, name="arima")

def fit_ets_series(price: pd.Series, trend="add", seasonal=None, seasonal_periods=None):
    m = ExponentialSmoothing(price.astype(float), trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    return m.fit(optimized=True)

def ets_forecast_steps(fit, steps: int, index) -> pd.Series:
    fc = fit.forecast(steps)
    return pd.Series(np.asarray(fc), index=index, name="ets")

def _supervised(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []; s = series.values.astype("float32")
    for i in range(window, len(s)):
        x.append(s[i-window:i]); y.append(s[i])
    X = np.array(x)[:, :, None]; y = np.array(y)[:, None]; return X, y

def train_lstm_returns(price: pd.Series, window: int = 22, epochs: int = 30, batch: int = 32):
    if not _TF_OK:
        raise RuntimeError("TensorFlow no disponible")
    r = np.log(price).diff().dropna()
    scaler = StandardScaler()
    r_sc = pd.Series(scaler.fit_transform(r.values.reshape(-1,1)).ravel(), index=r.index, name="ret_sc")
    X, y = _supervised(r_sc, window)
    model = keras.Sequential([keras.layers.Input(shape=(window,1)),
                              keras.layers.LSTM(32),
                              keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)
    return {"model": model, "scaler": scaler, "window": window}

def lstm_predict_price(state, price: pd.Series) -> pd.Series:
    scaler = state["scaler"]; W = state["window"]; model = state["model"]
    r = np.log(price).diff().dropna()
    r_sc = pd.Series(scaler.transform(r.values.reshape(-1,1)).ravel(), index=r.index)
    X, _ = _supervised(r_sc, W)
    pred_sc = model.predict(X, verbose=0).ravel()
    pred_r = scaler.inverse_transform(pred_sc.reshape(-1,1)).ravel()
    last_price = price.shift(1).dropna().values[-len(pred_r):]
    price_pred = last_price * np.exp(pred_r)
    out = pd.Series(price_pred, index=price.index[-len(price_pred):], name="lstm"); return out

# ==================== MODELOS (VOLATILIDAD) ====================
def fit_garch(ret: pd.Series, dist="StudentsT"):
    am = arch_model(ret.dropna().values*100, p=1, q=1, mean="Constant", vol="GARCH", dist=dist)
    return am.fit(disp="off")

def garch_sigma_series(res, index: pd.DatetimeIndex, horizon: int = 1) -> pd.Series:
    f = res.forecast(horizon=horizon, reindex=True)
    sigma2 = f.variance.iloc[:, -1] / (100**2)
    return sigma2.pow(0.5).rename("garch_sigma").reindex(index)

def garch_oos_sigma(ret: pd.Series, test_idx: pd.DatetimeIndex) -> pd.Series:
    """
    Pronóstico OOS de sigma_t con GARCH(1,1) rolling:
    para cada t en test_idx, ajusta con retornos hasta t-1 y predice h=1.
    Devuelve sigma en unidades de retorno (NO %), alineada con test_idx.
    """
    sigmas = []
    for t in test_idx:
        sub = ret.loc[:t].iloc[:-1].dropna()
        # seguridad: mínimo de datos para GARCH
        if len(sub) < 200:
            sigmas.append(np.nan)
            continue
        try:
            res = fit_garch(sub)  # usa tu fit_garch existente
            f = res.forecast(horizon=1, reindex=False)
            # 'f.variance' suele ser array-like -> última fila, col 0 (h=1)
            sigma2_pct2 = float(np.asarray(f.variance)[-1, 0])     # en %^2
            sigma2 = sigma2_pct2 / (100.0**2)                      # a unidades
            sigmas.append(np.sqrt(sigma2))
        except Exception:
            sigmas.append(np.nan)
    return pd.Series(sigmas, index=test_idx, name="garch_sigma")


def fit_har_model(logrv_df: pd.DataFrame, use_iv: bool = False):
    X = np.log(logrv_df[["rv_d","rv_w","rv_m"]])
    if use_iv and "log_iv" in logrv_df.columns:
        X = pd.concat([X, logrv_df[["log_iv"]]], axis=1)
    X = sm.add_constant(X)
    y = logrv_df["log_rv"]
    return sm.OLS(y, X, missing="drop").fit()

def har_predict_one(model, last_row: pd.Series) -> float:
    Xf = np.log(last_row[["rv_d","rv_w","rv_m"]]).to_frame().T
    if "log_iv" in last_row.index:
        Xf["log_iv"] = last_row["log_iv"]
    Xf = sm.add_constant(Xf, has_constant="add")
    log_rv_hat = float(model.predict(Xf).iloc[0])
    return np.exp(log_rv_hat)

# ==================== BACKTESTS ====================
@dataclass
class BacktestResult:
    preds: pd.DataFrame
    metrics: pd.DataFrame
    dm: dict

def backtest_nivel_generic(price: pd.Series, test_size_points: int, window_lstm: int, outdir: str) -> BacktestResult:
    price = price.dropna()
    assert test_size_points < len(price), "test_size_points debe ser menor al total de observaciones"
    split_idx = -test_size_points
    train = price.iloc[:split_idx]; test = price.iloc[split_idx:]; test_idx = test.index

    rw = predict_random_walk(price, horizon=1).reindex(test_idx)

    ar_fit = fit_arima_series(train, order=(1,1,1))
    ar_pred = arima_forecast_steps(ar_fit, steps=len(test_idx), index=test_idx)

    ets_fit = fit_ets_series(train, trend="add", seasonal=None)
    ets_pred = ets_forecast_steps(ets_fit, steps=len(test_idx), index=test_idx)

    lstm_pred = None
    try:
        lstm_state = train_lstm_returns(train, window=window_lstm, epochs=30, batch=32)
        lstm_pred = lstm_predict_price(lstm_state, price).reindex(test_idx)
    except Exception as e:
        print(f"⚠️ LSTM omitido en backtest ({e}).")

    cols = {"y": test, "rw": rw, "arima": ar_pred, "ets": ets_pred}
    if lstm_pred is not None: cols["lstm"] = lstm_pred
    df = pd.DataFrame(cols).dropna()

    rows = []
    for m in [c for c in df.columns if c!="y"]:
        rows.append({"model": m,
                     "RMSE": rmse(df["y"], df[m]),
                     "MAE": mae(df["y"], df[m]),
                     "MAPE(%)": mape(df["y"], df[m]),
                     "TheilsU": theils_u(df["y"], df[m]),
                     "HitRate": hit_rate(df["y"], df[m])})
    metr = pd.DataFrame(rows).set_index("model").sort_values(["RMSE","TheilsU"], ascending=[True,True])

    dm = {}
    for m in metr.index:
        if m != "rw":
            dm[m] = diebold_mariano(df["y"], df[m], df["rw"], h=1, loss="mse")

    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "predicciones_nivel.csv"))
    metr.to_csv(os.path.join(outdir, "metricas_nivel.csv"))
    return BacktestResult(preds=df, metrics=metr, dm=dm)

def backtest_vol_generic(df_ohlc: pd.DataFrame, iv: pd.Series | None, test_size_points: int, outdir: str) -> dict:
    close = df_ohlc["Close"].dropna()
    high  = df_ohlc["High"].reindex(close.index).ffill()
    low   = df_ohlc["Low"].reindex(close.index).ffill()
    ret = log_returns(close)
    rv = parkinson_rv(high, low).reindex(close.index)

    split_idx = -test_size_points
    rv_test_idx = rv.iloc[split_idx:].index

    gsig = garch_oos_sigma(ret, rv_test_idx)
    # Convertir sigma -> varianza para comparar con RV
    garch_rv = gsig.pow(2).rename("garch_rv")


    Xy = har_lags(rv)
    Xy = add_iv(Xy, iv.reindex(Xy.index) if iv is not None else None)
    use_iv = "log_iv" in Xy.columns
    hfit = fit_har_model(Xy.loc[:rv_test_idx[0]].iloc[:-1], use_iv=use_iv)
    
    har_pred = []
    valid_idx = Xy.index.intersection(rv_test_idx)
    for t in valid_idx:
        last = Xy.loc[:t].iloc[-1]
        har_pred.append((t, har_predict_one(hfit, last)))
    har_pred = pd.Series(dict(har_pred), name="har_rv").reindex(rv_test_idx)

    eval_df = pd.concat([rv.reindex(rv_test_idx).rename("rv_true"),
                        garch_rv,                      # ahora en varianza
                        har_pred.rename("har_rv")], axis=1)

    # Limpieza/fallback mínimo: si todas NaN, intenta forward-fill de 1 paso
    if eval_df.isna().all(axis=1).all():
        eval_df["garch_sigma"] = eval_df["garch_sigma"].ffill()
        eval_df["har_rv"] = eval_df["har_rv"].ffill()

    eval_df = eval_df.dropna()
    if eval_df.empty:
        # como último recurso, salimos con métricas vacías pero sin crashear
        os.makedirs(outdir, exist_ok=True)
        eval_df.to_csv(os.path.join(outdir, "predicciones_vol.csv"))
        metr = pd.DataFrame([{"model":"garch_sigma","RMSE":np.nan,"MAE":np.nan},
                             {"model":"har_rv","RMSE":np.nan,"MAE":np.nan}]).set_index("model")
        metr.to_csv(os.path.join(outdir, "metricas_vol.csv"))
        return {"preds": eval_df, "metrics": metr, "har_summary": hfit.summary().as_text()}

    rows = []
    for col in ["garch_rv","har_rv"]:
        rows.append({"model": col, "RMSE": rmse(eval_df["rv_true"], eval_df[col]),
                               "MAE": mae(eval_df["rv_true"], eval_df[col])})
    metr = pd.DataFrame(rows).set_index("model").sort_values("RMSE")

    os.makedirs(outdir, exist_ok=True)
    eval_df.to_csv(os.path.join(outdir, "predicciones_vol.csv"))
    metr.to_csv(os.path.join(outdir, "metricas_vol.csv"))
    return {"preds": eval_df, "metrics": metr, "har_summary": hfit.summary().as_text()}

# ==================== SELECCIÓN + SEÑAL ====================
def series_to_prophet_df(price: pd.Series) -> pd.DataFrame:
    dfp = price.reset_index()
    dfp.columns = ["ds", "y"]
    return dfp

def train_and_forecast_all(price: pd.Series, window_lstm: int = 22, include_prophet: bool = False):
    last_idx = price.index[-1]
    preds = {}

    # RW
    preds["rw"] = pd.Series(price.iloc[-1], index=[last_idx], name="rw")

    # ARIMA / ETS (por steps=1 sobre todo el histórico)
    ar_fit = fit_arima_series(price, order=(1,1,1))
    preds["arima"] = arima_forecast_steps(ar_fit, steps=1, index=[last_idx])

    ets_fit = fit_ets_series(price, trend="add", seasonal=None)
    preds["ets"] = ets_forecast_steps(ets_fit, steps=1, index=[last_idx])

    # LSTM
    if _TF_OK:
        try:
            lstm_state = train_lstm_returns(price, window=window_lstm, epochs=30, batch=32)
            lstm_all = lstm_predict_price(lstm_state, price)
            preds["lstm"] = pd.Series(lstm_all.iloc[-1], index=[last_idx], name="lstm")
        except Exception as e:
            print(f"⚠️ LSTM no disponible ({e}); se omite.")
    # Prophet (opcional)
    if include_prophet and _PROPHET_OK:
        try:
            dfp = series_to_prophet_df(price)
            m = entrenar_modelo_prophet(dfp)
            fc = predecir_precio(m, pasos=1, frecuencia="1min")
            preds["prophet"] = pd.Series(float(fc["yhat"].iloc[0]), index=[last_idx], name="prophet")
        except Exception as e:
            print(f"⚠️ Prophet no disponible en live ({e}).")

    return preds

def evaluate_recent_window(price: pd.Series, window_valid: int = 300, window_lstm: int = 22, include_prophet: bool = False):
    assert len(price) > window_valid + 50, "Pocos datos para validar"
    train = price.iloc[:-window_valid]
    test  = price.iloc[-window_valid:]
    idx   = test.index

    preds = {}
    preds["rw"] = predict_random_walk(price).reindex(idx)

    ar_fit = fit_arima_series(train, order=(1,1,1))
    preds["arima"] = arima_forecast_steps(ar_fit, steps=len(idx), index=idx)

    ets_fit = fit_ets_series(train, trend="add", seasonal=None)
    preds["ets"] = ets_forecast_steps(ets_fit, steps=len(idx), index=idx)

    if _TF_OK:
        try:
            lstm_state = train_lstm_returns(train, window=window_lstm, epochs=30, batch=32)
            preds["lstm"] = lstm_predict_price(lstm_state, price).reindex(idx)
        except Exception as e:
            print(f"⚠️ LSTM no disponible en validación ({e}).")

    if include_prophet and _PROPHET_OK:
        try:
            dfp = series_to_prophet_df(train)
            m = entrenar_modelo_prophet(dfp)
            fc = predecir_precio(m, pasos=len(idx), frecuencia="1min")
            preds["prophet"] = pd.Series(fc["yhat"].values, index=idx, name="prophet")
        except Exception as e:
            print(f"⚠️ Prophet no disponible en validación ({e}).")

    rows = []
    for k, p in preds.items():
        rows.append({"model": k, "RMSE": rmse(test, p), "TheilsU": theils_u(test, p), "HitRate": hit_rate(test, p)})
    metr = pd.DataFrame(rows).set_index("model").sort_values(["RMSE","TheilsU"], ascending=[True,True])

    best = metr.index[0]
    return best, metr, preds, test

def build_signal_from_pred(price_now: float, pred_next: float, umbral: float) -> str:
    delta = pred_next - price_now
    if delta > umbral: return "comprar"
    if delta < -umbral: return "vender"
    return "mantener"

# ==================== PLOTEO ====================
def plot_level_preds(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["y"], label="Real")
    for m in [c for c in df.columns if c!="y"]:
        plt.plot(df.index, df[m], label=m.upper())
    plt.title("Pronósticos de precio (nivel)"); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()

def plot_rolling_error(df: pd.DataFrame, model: str, out_path: str, kind: str = "abs"):
    err = (df[model]-df["y"]).abs() if kind=="abs" else (df[model]-df["y"])**2
    plt.figure(figsize=(10,4)); plt.plot(err.index, err.values)
    plt.title(f"Evolución del error ({kind}) · {model.upper()}"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()
    
def plot_vol_preds(vol_df: pd.DataFrame, out_path: str):
    """
    vol_df con columnas: rv_true, har_rv, garch_rv (o garch_sigma^2 ya convertido).
    Dibuja en la misma escala (varianza).
    """
    plt.figure(figsize=(10,5))
    base = vol_df.dropna(subset=["rv_true"])
    plt.plot(base.index, base["rv_true"], label="RV (realizada)")
    if "har_rv" in base.columns:
        plt.plot(base.index, base["har_rv"], label="HAR-RV")
    if "garch_rv" in base.columns:
        plt.plot(base.index, base["garch_rv"], label="GARCH (var)")
    plt.title("Volatilidad realizada vs pronósticos (varianza)")
    plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()


# ==================== HELPERS MT5 ====================
def obtener_df_desde_mt5(bf: Basic_funcs, symbol: str, timeframe: str, n_barras: int) -> pd.DataFrame:
    """Usa Basic_funcs (easy_Trading.py) para traer OHLCV con índice datetime naive."""
    df = bf.get_data_for_bt(timeframe, symbol, n_barras)
    # Normaliza nombres esperados
    cols_map = {"open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"TickVolume","real_volume":"Volume","time":"Date"}
    for k,v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    if "Date" in df.columns:
        df = df.set_index("Date")
    return df.sort_index()

# ==================== MAIN ====================
def main():
    # --- Config / CLI ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["normal","eda","modelos"], default="normal")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simbolo   = config.get("simbolo","EURUSD")
    timeframe = config.get("timeframe","M5")
    cantidad  = int(config.get("cantidad_datos", 2000))
    modelo_str = str(config.get("modelo","prophet")).lower()
    pasos_pred = int(config.get("pasos_prediccion", 100))
    frecuencia_pred = config.get("frecuencia_prediccion","5min")
    umbral_senal = float(config.get("umbral_senal", 0.0005))
    eda_cfg = config.get("eda", {})
    outdir = os.path.join(eda_cfg.get("outdir","outputs/eda"), simbolo)

    # --- MT5 ---
    mt5c = config.get("mt5", {})
    bf = Basic_funcs(mt5c.get("login"), mt5c.get("password"), mt5c.get("server"), mt5c.get("path"))
    print("✅ Conexión establecida con MetaTrader 5")

    try:
        # Datos OHLCV desde MT5
        df = obtener_df_desde_mt5(bf, simbolo, timeframe, cantidad)
        if df is None or df.empty:
            raise RuntimeError("No se obtuvieron datos desde MT5.")
        price = df["Close"].astype(float)

        # 🔧 Normaliza índice: datetime, único, ordenado
        price.index = pd.to_datetime(price.index, errors="coerce")
        price = price[~price.index.duplicated()].sort_index()
        price = price[price.index.notna()]

        # ---------- MODO: MODELOS (backtests) ----------
        if args.modo == "modelos":
            test_pts = max(500, int(0.25 * len(price)))
            res_nivel = backtest_nivel_generic(price, test_size_points=test_pts, window_lstm=22, outdir=outdir)
            plot_level_preds(res_nivel.preds, os.path.join(outdir, "nivel_pred_vs_real.png"))
            for m in [c for c in res_nivel.preds.columns if c!="y"]:
                plot_rolling_error(res_nivel.preds, m, os.path.join(outdir, f"errores_{m}.png"))
            res_vol = backtest_vol_generic(df[["High","Low","Close"]], iv=None, test_size_points=test_pts, outdir=outdir)

            print("\n== MÉTRICAS NIVEL =="); print(res_nivel.metrics.round(6))
            print("\n== DM vs RW =="); [print(k, v) for k, v in res_nivel.dm.items()]
            print("\n== MÉTRICAS VOL =="); print(res_vol["metrics"].round(6))
            print("\n== HAR SUMMARY =="); print(res_vol["har_summary"])
            ruta_xlsx = config.get("ruta_reporte", os.path.join("outputs", "reporte_inversion.xlsx"))

            # Asegúrate de que en res_vol["preds"] la columna de GARCH esté en varianza
            # (si tu backtest ya la llama garch_rv, esto no toca nada; si viene en sigma, convertimos aquí):
            res_vol_preds = res_vol["preds"].copy()
            if "garch_rv" not in res_vol_preds.columns and "garch_sigma" in res_vol_preds.columns:
                res_vol_preds["garch_rv"] = res_vol_preds["garch_sigma"] ** 2

            cfg_export = {
                "simbolo": simbolo,
                "timeframe": timeframe,
                "cantidad_datos": cantidad,
                "modo": "modelos",
                "outdir": outdir
            }

            export_resultados_modelos(
                ruta_excel=ruta_xlsx,
                res_nivel_metrics=res_nivel.metrics,   # DataFrame
                res_nivel_preds=res_nivel.preds,       # DataFrame
                res_nivel_dm=res_nivel.dm,             # dict
                res_vol_metrics=res_vol["metrics"],    # DataFrame
                res_vol_preds=res_vol_preds,           # DataFrame
                har_summary_text=res_vol["har_summary"], # str
                cfg_dict=cfg_export
            )

            print(f"💾 Reporte exportado a: {ruta_xlsx}")
            
            # ===== Generar y empaquetar imágenes en PDF =====
            from reportes.reportes_pdf import generate_pdf_report

            # 1) Asegura gráfico de volatilidad
            vol_img = os.path.join(outdir, "vol_pred_vs_real.png")
            plot_vol_preds(res_vol_preds, vol_img)

            # 2) Reunir imágenes ya generadas + la de vol
            image_paths = [os.path.join(outdir, "nivel_pred_vs_real.png")]
            for m in [c for c in res_nivel.preds.columns if c != "y"]:
                image_paths.append(os.path.join(outdir, f"errores_{m}.png"))
            image_paths.append(vol_img)

            # 3) Ruta PDF (si no está en YAML, derivarlo del xlsx)
            ruta_pdf = config.get("ruta_pdf", ruta_xlsx.replace(".xlsx", ".pdf"))
            generate_pdf_report(
                output_pdf_path=ruta_pdf,
                image_paths=image_paths,
                title="Informe de Resultados — Modo Modelos",
                metadata=cfg_export
            )
            print(f"📄 PDF exportado: {ruta_pdf}")
            return

        # ---------- MODO: EDA (si existe tu módulo) ----------
        if args.modo == "eda":
            if not _EDA_OK:
                print("⚠️ ejecutar_eda no disponible en procesamiento/eda_crispdm.py")
            else:
                ejecutar_eda(df_eurusd=df, df_spy=None, cfg=config)
            return

        # ---------- MODO: NORMAL (trading) ----------
        if modelo_str == "prophet":
            if not _PROPHET_OK:
                raise RuntimeError("Prophet no disponible en el entorno.")
            print("🤖 Entrenando modelo Prophet…")
            dfp = series_to_prophet_df(price)
            m = entrenar_modelo_prophet(dfp)
            fc = predecir_precio(m, pasos=pasos_pred, frecuencia=frecuencia_pred)
            price_now = float(price.iloc[-1])
            yhat_next = float(fc["yhat"].iloc[0])
            senal = build_signal_from_pred(price_now, yhat_next, umbral_senal)
            print(f"📈 Señal (Prophet): {senal} | y_now={price_now:.6f} → y_hat={yhat_next:.6f}")

        elif modelo_str in ("auto","comparativo"):
            print("🧪 Selección automática RW/ARIMA/ETS/LSTM (+Prophet si disponible)…")
            window_valid = min(300, max(100, int(0.25*len(price))))
            best, metr, preds_val, y_true = evaluate_recent_window(price, window_valid=window_valid, window_lstm=22, include_prophet=_PROPHET_OK)
            print("🏆 Mejor modelo:", best)
            preds_live = train_and_forecast_all(price, window_lstm=22, include_prophet=_PROPHET_OK)
            yhat_next = float(preds_live[best].iloc[0])
            price_now = float(price.iloc[-1])
            senal = build_signal_from_pred(price_now, yhat_next, umbral_senal)
            print(metr.round(6))
            print(f"📈 Señal (mejor={best}): {senal} | y_now={price_now:.6f} → y_hat={yhat_next:.6f}")
            os.makedirs(outdir, exist_ok=True)
            metr.to_csv(os.path.join(outdir, "metricas_comparativo_nivel.csv"))

        else:
            raise ValueError(f"Modelo '{modelo_str}' no implementado. Usa 'prophet' o 'auto'.")

        # ----- Enganche con tu Agente de Análisis (si existe) -----
        try:
            from agentes.agente_analisis import generar_senal_operativa
            senal_dict = generar_senal_operativa(senal_predicha=senal, precio_actual=price_now, yhat=yhat_next, umbral=umbral_senal)
            print("🤝 Agente de análisis:", senal_dict)
        except Exception as e:
            print(f"ℹ️ Agente de análisis no fue invocado ({e}). Usando señal local: {senal}")

    finally:
        try:
            from MetaTrader5 import shutdown as _mt5_shutdown
            _mt5_shutdown()
        except Exception:
            pass
        print("🛑 Conexión cerrada")
        
if __name__ == "__main__":
    main()
