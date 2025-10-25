# app/main.py
import os, sys, argparse, yaml
import numpy as np
import pandas as pd
import warnings

from evaluacion.backtest_rolling import (
    evaluate_many, save_backtest_excel, save_backtest_plots
)

from procesamiento.eda_crispdm import _ensure_dt_index, _find_close, _resample_ohlc

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# TensorFlow opcional
try:
    from tensorflow import keras  # noqa
except Exception as _e:
    print(f"ℹ️ TensorFlow no disponible ({_e}).")

# ARCH opcional (para umbral GARCH)
_HAS_GARCH = True
try:
    from arch import arch_model  # noqa
except Exception as _e:
    _HAS_GARCH = False
    print(f"ℹ️ GARCH no disponible ({_e}). Instala con: pip install arch")

from conexion.easy_Trading import Basic_funcs

# EDA opcional
try:
    from procesamiento.eda_crispdm import ejecutar_eda
    _EDA_OK = True
except Exception as _e:
    _EDA_OK = False
    print(f"⚠️ ejecutar_eda no disponible ({_e}).")


# ======================
# Utilidades
# ======================
def compute_atr_pips(df: pd.DataFrame, window: int = 14, pip_size: float = 0.0001) -> pd.Series | None:
    """
    ATR(14) en pips, para umbral dinámico tipo ATR.
    Requiere columnas High/Low/Close.
    """
    for c in ("High", "Low", "Close"):
        if c not in df.columns:
            return None
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    atr_pips = atr / pip_size
    return atr_pips


def compute_garch_sigma_pips(price: pd.Series, pip_size: float = 0.0001) -> pd.Series | None:
    """
    Estima σ_t (volatilidad condicional) con GARCH(1,1) sobre los log-returns
    y lo convierte a pips aproximados: sigma_ret_t * Price_t / pip_size.

    Nota: es in-sample (rápido y estable para usar como “proxy” de umbral).
    """
    if not _HAS_GARCH:
        return None
    s = price.astype(float).dropna()
    if len(s) < 250:
        print("ℹ️ Serie demasiado corta para GARCH (se requieren ~250+ puntos).")
        return None

    # log-returns como % para mejor estabilidad
    logret = np.log(s).diff().dropna() * 100.0
    if logret.isna().all():
        return None
    try:
        from arch import arch_model
        am = arch_model(logret, p=1, o=0, q=1, mean="Zero", vol="GARCH", dist="normal")
        res = am.fit(disp="off")
        sigma_pct = res.conditional_volatility  # en %
        sigma_ret = (sigma_pct / 100.0).reindex(s.index).ffill()
        sigma_pips = (sigma_ret * s) / pip_size
        return sigma_pips
    except Exception as e:
        print(f"ℹ️ No se pudo calcular GARCH: {e}")
        return None


def obtener_df_desde_mt5(bf: Basic_funcs, symbol: str, timeframe: str, n_barras: int) -> pd.DataFrame:
    df = bf.get_data_for_bt(timeframe, symbol, n_barras)
    cols_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "TickVolume",
        "real_volume": "Volume",
        "time": "Date",
    }
    for k, v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if "Date" in df.columns:
        df = df.set_index("Date")
    return df.sort_index()


def _filtered_kwargs(fn, **kwargs):
    """Filtra kwargs según la firma de la función (compatibilidad hacia atrás)."""
    import inspect
    allowed = set(inspect.signature(fn).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["normal", "eda", "backtest"], default="normal")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simbolo = config.get("simbolo", "EURUSD")
    timeframe = config.get("timeframe", "D1")
    cantidad = int(config.get("cantidad_datos", 2000))
    eda_cfg = config.get("eda", {})
    bt_cfg = config.get("bt", {})

    # Conexión MT5
    mt5c = config.get("mt5", {})
    bf = Basic_funcs(mt5c.get("login"), mt5c.get("password"), mt5c.get("server"), mt5c.get("path"))
    print("✅ Conexión establecida con MetaTrader 5")

    try:
        # Carga EURUSD
        df = obtener_df_desde_mt5(bf, simbolo, timeframe, cantidad)
        price_col = _find_close(df)
        df = _ensure_dt_index(df)
        df = _resample_ohlc(df, freq=eda_cfg.get("frecuencia_resampleo", "D"), price_col=price_col)
        price = df[price_col].astype(float)

        if args.modo == "eda":
            if _EDA_OK:
                ejecutar_eda(df_eurusd=df, df_spy=None, cfg=config)
            else:
                print("⚠️ ejecutar_eda no disponible en procesamiento/eda_crispdm.py")
            return

        if args.modo == "backtest":
            # === Config Backtest ===
            initial_train = int(bt_cfg.get("initial_train", 1000))
            step = int(bt_cfg.get("step", 10))
            horizon = int(bt_cfg.get("horizon", 1))
            target = bt_cfg.get("target", "returns")
            pip_size = float(bt_cfg.get("pip_size", 0.0001))

            # Umbrales
            threshold_mode = bt_cfg.get("threshold_mode", "garch").lower()
            threshold_pips = float(bt_cfg.get("threshold_pips", 15.0))  # para "fixed"
            atr_k = float(bt_cfg.get("atr_k", 0.6))
            atr_window = int(bt_cfg.get("atr_window", 14))
            garch_k = float(bt_cfg.get("garch_k", 0.6))  # <-- AHORA se respeta
            min_threshold_pips = float(bt_cfg.get("min_threshold_pips", 10.0))
            log_threshold_used = bool(bt_cfg.get("log_threshold_used", True))

            # ATR en pips
            atr_pips_series = compute_atr_pips(df.rename(columns={price_col: "Close"}), window=atr_window, pip_size=pip_size)

            # GARCH σ en pips
            garch_sigma_pips = compute_garch_sigma_pips(price, pip_size=pip_size) if _HAS_GARCH else None
            if threshold_mode == "garch" and not _HAS_GARCH:
                print("⚠️ threshold_mode='garch' pero ARCH no está instalado. Se usará 'atr' si está disponible.")
                threshold_mode = "atr"

            # Exógenas US500 (opcional)
            exog_cfg = bt_cfg.get("exog", {})
            exog_ret = None
            exog_lags = None
            if exog_cfg.get("enable", False):
                try:
                    spy_symbol = config.get("simbolo_spy")
                    if spy_symbol:
                        df_spy = obtener_df_desde_mt5(bf, spy_symbol, "D1", cantidad)
                        if df_spy is not None and not df_spy.empty:
                            df_spy = _ensure_dt_index(df_spy)
                            spy_price_col = _find_close(df_spy)
                            df_spy = _resample_ohlc(
                                df_spy, freq=eda_cfg.get("frecuencia_resampleo", "D"), price_col=spy_price_col
                            )
                            exog_ret = np.log(df_spy[spy_price_col]).diff()
                            exog_lags = exog_cfg.get("lags", [1, 2, 3, 4, 5])
                except Exception as e:
                    print(f"ℹ️ US500 exógeno no cargado ({e}).")

            # Specs: RW + AUTO(ARIMA/SARIMA)
            specs = [
                {"name": "RW_RETURNS", "kind": "rw"},
                {
                    "name": "AUTO(ARIMA/SARIMA)_RET",
                    "kind": "auto",
                    "scan": bt_cfg.get("auto", {}).get("scan", {}),
                    "rescan_each_refit": bt_cfg.get("auto", {}).get("rescan_each_refit", True),
                    "rescan_every_refits": bt_cfg.get("auto", {}).get("rescan_every_refits", 5),
                },
            ]

            # kwargs para evaluate_many
            eval_kwargs = dict(
                initial_train=initial_train,
                step=step,
                horizon=horizon,
                target=target,
                pip_size=pip_size,
                threshold_pips=threshold_pips,
                exog_ret=exog_ret,
                exog_lags=exog_lags,
                threshold_mode=threshold_mode,     # "fixed" | "atr" | "garch"
                atr_pips=atr_pips_series,
                atr_k=atr_k,
                garch_k=garch_k,                   # <-- AHORA se pasa
                min_threshold_pips=min_threshold_pips,
                garch_sigma_pips=garch_sigma_pips,
                log_threshold_used=log_threshold_used
            )
            eval_kwargs = _filtered_kwargs(evaluate_many, **eval_kwargs)

            print(f"[AUTO] ventanas=?, step={step}, horizon={horizon}, target={target}, thr_mode={threshold_mode}")
            if threshold_mode == "atr" and atr_pips_series is None:
                print("⚠️ No hay ATR disponible (faltan OHLC). Se usará threshold fijo.")
            if threshold_mode == "garch" and garch_sigma_pips is None:
                print("⚠️ No hay σ_GARCH disponible. Se usará threshold fijo o ATR si existe.")

            # Ejecuta backtest
            summary, preds_map = evaluate_many(price, specs, **eval_kwargs)

            # Salidas
            outxlsx = bt_cfg.get("outxlsx", "outputs/evaluacion.xlsx")
            save_backtest_excel(outxlsx, summary, preds_map)

            outdir_plots = bt_cfg.get("outdir_plots", "outputs/backtest_plots")
            save_backtest_plots(outdir_plots, price, preds_map, pip_size, threshold_pips)

            try:
                print(summary.to_string(index=False))
            except Exception:
                print(summary)
            return

        print("ℹ️ Modo no implementado en este snippet. Usa --modo eda o --modo backtest.")

    finally:
        try:
            from MetaTrader5 import shutdown as _mt5_shutdown
            _mt5_shutdown()
        except Exception:
            pass
        print("🛑 Conexión cerrada")


if __name__ == "__main__":
    main()
