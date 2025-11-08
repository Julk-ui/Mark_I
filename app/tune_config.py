from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import yaml

# Helpers de tu main para datos MT5 y preprocesado
from app.main import (
    obtener_df_desde_mt5,
    _find_close,
    _ensure_dt_index,
    _resample_ohlc,
    _HAS_MT5,
    _BF,
)

# Adapters / modelos propios
from modelos.arima.adapter import ArimaModel
from modelos.prophet.adapter import (
    entrenar_modelo_prophet,
    predecir_precio_prophet,
)
from modelos.lstm_model import LSTMModel


# ==========================================================
# Helpers para cargar y guardar configuración
# ==========================================================

def cargar_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def guardar_config(config: Dict[str, Any], path_out: str) -> None:
    path = Path(path_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"✅ Config optimizado guardado en: {path}")


# ==========================================================
# Descarga de datos desde MetaTrader5
# ==========================================================

def obtener_df(config: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    if not _HAS_MT5:
        raise SystemExit("❌ No hay conexión MT5 (Basic_funcs).")

    simbolo = config.get("simbolo", "EURUSD")
    timeframe = config.get("timeframe", "H1")
    cantidad = int(config.get("cantidad_datos", 3000))
    eda_cfg = config.get("eda", {})

    mt5c = config.get("mt5", {})
    bf = _BF(
        mt5c.get("login"),
        mt5c.get("password"),
        mt5c.get("server"),
        mt5c.get("path"),
    )  # type: ignore
    print("✅ Conexión MT5 para tuning")

    df = obtener_df_desde_mt5(bf, simbolo, timeframe, cantidad)
    price_col = _find_close(df)
    df = _ensure_dt_index(df)
    df = _resample_ohlc(
        df,
        freq=eda_cfg.get("frecuencia_resampleo", "H"),
        price_col=price_col,
    )

    return df, price_col


# ==========================================================
# Backtest walk-forward de ARIMA usando tu ArimaModel
# ==========================================================

def backtest_arima_adapter(
    price: pd.Series,
    order: List[int],
    cfg_bt: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> float:
    """
    Backtest walk-forward para ARIMA(p,d,q) usando modelos/arima/adapter.ArimaModel.
    """

    initial_train = int(cfg_bt.get("initial_train", 1500))
    step = int(cfg_bt.get("step", 10))
    horizon = int(cfg_bt.get("horizon", 1))

    modelo_cfg_base = global_cfg.get("modelo", {})
    modo = str(modelo_cfg_base.get("objetivo", "retornos")).lower()
    if modo not in {"nivel", "retornos"}:
        modo = "nivel"

    freq_cfg = str(global_cfg.get("eda", {}).get("frecuencia_resampleo", "H"))

    n = len(price)
    if n <= initial_train + horizon:
        raise ValueError("Serie demasiado corta para el backtest definido.")

    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for end_train in range(initial_train, n - horizon, step):
        train = price.iloc[:end_train]
        test = price.iloc[end_train:end_train + horizon]

        model_cfg = {
            "modo": modo,
            "auto": False,
            "order": order,
            "seasonal": False,
            "enforce_stationarity": False,
            "enforce_invertibility": False,
        }

        cfg_local = {"freq": freq_cfg}

        try:
            model = ArimaModel(model_cfg=model_cfg, cfg=cfg_local)
            model.fit(train)

            pred_df = model.predict(
                horizon,
                last_timestamp=train.index[-1],
            )
            pred_last = float(pred_df["yhat"].iloc[-1])
            true_last = float(test.iloc[-1])

            y_true_all.append(true_last)
            y_pred_all.append(pred_last)

        except Exception as e:
            print(f"⚠️ Fold fallido para ARIMA{order}: {e}")
            continue

    if not y_true_all:
        raise ValueError(f"No se pudieron generar predicciones válidas para ARIMA{order}.")

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    rmse = np.sqrt(((y_true_arr - y_pred_arr) ** 2).mean())
    return rmse


def tune_arima(price: pd.Series, base_cfg: Dict[str, Any]) -> tuple[List[int], float]:
    cfg_bt = base_cfg.get("bt", {})

    # Rango de búsqueda ARIMA
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    best_order: List[int] | None = None
    best_rmse: float | None = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = [p, d, q]
                try:
                    rmse = backtest_arima_adapter(price, order, cfg_bt, base_cfg)
                    print(f"ARIMA{order} → RMSE={rmse:.6f}")
                except Exception as e:
                    print(f"⚠️ Falló ARIMA{order}: {e}")
                    continue

                if (best_rmse is None) or (rmse < best_rmse):
                    best_rmse = rmse
                    best_order = order

    if best_order is None or best_rmse is None:
        raise RuntimeError("No se encontró ningún ARIMA válido.")

    print(f"🎯 Mejor ARIMA {best_order} con RMSE={best_rmse:.6f}")
    return best_order, best_rmse


# ==========================================================
# Backtest Prophet usando tu adapter funcional
# ==========================================================

def backtest_prophet_adapter(
    price: pd.Series,
    params: Dict[str, Any],
    cfg_bt: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> float:
    """
    Backtest walk-forward para Prophet usando entrenar_modelo_prophet / predecir_precio_prophet.
    """

    initial_train = int(cfg_bt.get("initial_train", 1500))
    step = int(cfg_bt.get("step", 10))
    horizon = int(cfg_bt.get("horizon", 1))

    modelo_cfg_base = global_cfg.get("modelo", {})
    modo = str(modelo_cfg_base.get("objetivo", "retornos"))
    freq_cfg = str(global_cfg.get("eda", {}).get("frecuencia_resampleo", "H"))

    n = len(price)
    if n <= initial_train + horizon:
        raise ValueError("Serie demasiado corta para el backtest definido.")

    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for end_train in range(initial_train, n - horizon, step):
        train = price.iloc[:end_train]
        test = price.iloc[end_train:end_train + horizon]

        df_train = pd.DataFrame(
            {"ds": train.index.to_pydatetime(), "y": train.values.astype(float)}
        )

        try:
            state = entrenar_modelo_prophet(
                df_train,
                modo=modo,
                frecuencia_hint=freq_cfg,
                **params,
            )

            pred_df = predecir_precio_prophet(state, pasos=horizon, frecuencia=freq_cfg)
            pred_last = float(pred_df["yhat"].iloc[-1])
            true_last = float(test.iloc[-1])

            y_true_all.append(true_last)
            y_pred_all.append(pred_last)

        except Exception as e:
            print(f"⚠️ Fold fallido Prophet params={params}: {e}")
            continue

    if not y_true_all:
        raise ValueError("No se pudieron generar predicciones válidas para Prophet.")

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    rmse = np.sqrt(((y_true_arr - y_pred_arr) ** 2).mean())
    return rmse


def tune_prophet(price: pd.Series, base_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
    cfg_bt = base_cfg.get("bt", {})

    # Grid pequeña de ejemplo – puedes ampliarla luego
    grid: List[Dict[str, Any]] = [
        {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
        {"changepoint_prior_scale": 0.50, "seasonality_mode": "additive"},
        {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
        {"changepoint_prior_scale": 0.50, "seasonality_mode": "multiplicative"},
    ]

    best_params: Dict[str, Any] | None = None
    best_rmse: float | None = None

    for params in grid:
        try:
            rmse = backtest_prophet_adapter(price, params, cfg_bt, base_cfg)
            print(f"PROPHET {params} → RMSE={rmse:.6f}")
        except Exception as e:
            print(f"⚠️ Falló PROPHET {params}: {e}")
            continue

        if (best_rmse is None) or (rmse < best_rmse):
            best_rmse = rmse
            best_params = params

    if best_params is None or best_rmse is None:
        raise RuntimeError("No se encontró configuración válida para PROPHET.")

    print(f"🎯 Mejor PROPHET {best_params} con RMSE={best_rmse:.6f}")
    return best_params, best_rmse


# ==========================================================
# Backtest LSTM usando tu LSTMModel
# ==========================================================

def backtest_lstm_adapter(
    price: pd.Series,
    params: Dict[str, Any],
    cfg_bt: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> float:
    """
    Backtest walk-forward para LSTMModel.
    """

    initial_train = int(cfg_bt.get("initial_train", 1500))
    step = int(cfg_bt.get("step", 10))
    horizon = int(cfg_bt.get("horizon", 1))

    freq_cfg = str(global_cfg.get("eda", {}).get("frecuencia_resampleo", "H"))

    n = len(price)
    if n <= initial_train + horizon:
        raise ValueError("Serie demasiado corta para el backtest definido.")

    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for end_train in range(initial_train, n - horizon, step):
        train = price.iloc[:end_train]
        test = price.iloc[end_train:end_train + horizon]

        try:
            model = LSTMModel(model_cfg=params, cfg={"freq": freq_cfg})
            model.fit(train)

            pred_df = model.predict(
                horizon,
                last_timestamp=train.index[-1],
            )
            pred_last = float(pred_df["yhat"].iloc[-1])
            true_last = float(test.iloc[-1])

            y_true_all.append(true_last)
            y_pred_all.append(pred_last)

        except Exception as e:
            print(f"⚠️ Fold fallido LSTM params={params}: {e}")
            continue

    if not y_true_all:
        raise ValueError("No se pudieron generar predicciones válidas para LSTM.")

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    rmse = np.sqrt(((y_true_arr - y_pred_arr) ** 2).mean())
    return rmse


def tune_lstm(price: pd.Series, base_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
    cfg_bt = base_cfg.get("bt", {})

    # Grid más completa: window, units, epochs, dropout, optimizer, scaler
    # (puedes ajustar o ampliar estas combinaciones según rendimiento/tiempo)
    grid: List[Dict[str, Any]] = [
        {
            "window": 32,
            "units": 32,
            "epochs": 20,
            "batch_size": 32,
            "dropout": 0.10,
            "optimizer": "adam",
            "scaler": "standard",
        },
        {
            "window": 32,
            "units": 32,
            "epochs": 20,
            "batch_size": 32,
            "dropout": 0.30,
            "optimizer": "adam",
            "scaler": "standard",
        },
        {
            "window": 64,
            "units": 64,
            "epochs": 20,
            "batch_size": 32,
            "dropout": 0.20,
            "optimizer": "adam",
            "scaler": "minmax",
        },
        {
            "window": 64,
            "units": 64,
            "epochs": 40,
            "batch_size": 32,
            "dropout": 0.20,
            "optimizer": "adam",
            "scaler": "standard",
        },
        {
            "window": 64,
            "units": 64,
            "epochs": 40,
            "batch_size": 32,
            "dropout": 0.30,
            "optimizer": "adam",
            "scaler": "minmax",
        },
    ]

    best_params: Dict[str, Any] | None = None
    best_rmse: float | None = None

    for params in grid:
        try:
            rmse = backtest_lstm_adapter(price, params, cfg_bt, base_cfg)
            print(f"LSTM {params} → RMSE={rmse:.6f}")
        except Exception as e:
            print(f"⚠️ Falló LSTM {params}: {e}")
            continue

        if (best_rmse is None) or (rmse < best_rmse):
            best_rmse = rmse
            best_params = params

    if best_params is None or best_rmse is None:
        raise RuntimeError("No se encontró configuración válida para LSTM.")

    print(f"🎯 Mejor LSTM {best_params} con RMSE={best_rmse:.6f}")
    return best_params, best_rmse


# ==========================================================
# Orquestador principal (respeta el config base)
# ==========================================================

def main_tuning(
    config_in: str = "utils/config.yaml",
    config_out: str = "utils/config_optimizado.yaml",
):
    # 1️⃣ Cargar configuración base y serie de precios
    base_cfg = cargar_config(config_in)
    df, price_col = obtener_df(base_cfg)
    price = df[price_col].astype(float)

    # 2️⃣ Tuning de cada modelo
    best_arima_order, best_arima_rmse = tune_arima(price, base_cfg)
    best_prophet_params, best_prophet_rmse = tune_prophet(price, base_cfg)
    best_lstm_params, best_lstm_rmse = tune_lstm(price, base_cfg)

    # 3️⃣ Partimos de una copia PROFUNDA del config base
    new_cfg = copy.deepcopy(base_cfg)

    # ============================
    # Actualizar bloque `modelo:`
    # ============================
    modelo_cfg = new_cfg.get("modelo", {})
    if isinstance(modelo_cfg, dict):
        nombre_base = str(modelo_cfg.get("nombre", "ARIMA")).upper()
        # Solo tocamos si el modelo principal es ARIMA
        if nombre_base == "ARIMA":
            params = modelo_cfg.get("params", {}) or {}
            # Actualizamos solo el order; dejamos el resto igual
            params["order"] = list(best_arima_order)
            # Si tenía auto en params, lo desactivamos
            if "auto" in params:
                params["auto"] = False
            modelo_cfg["params"] = params
            new_cfg["modelo"] = modelo_cfg

    # ============================
    # Actualizar bloque `modelos:`
    # ============================
    modelos_list = new_cfg.get("modelos", [])
    if isinstance(modelos_list, list):
        for m in modelos_list:
            name = str(m.get("name", "")).upper()
            params = m.get("params", {}) or {}

            if name == "ARIMA":
                # Respetamos todo lo que ya había y solo cambiamos lo tuneado
                params["order"] = list(best_arima_order)
                if "auto" in params:
                    params["auto"] = False
                m["params"] = params

            elif name == "PROPHET":
                # Mezclamos los params nuevos encima de los viejos
                params.update(best_prophet_params)
                m["params"] = params

            elif name == "LSTM":
                # Igual: solo sobreescribimos lo que hemos tuneado
                params.update(best_lstm_params)
                m["params"] = params

        new_cfg["modelos"] = modelos_list

    # 4️⃣ Guardar archivo YAML optimizado
    guardar_config(new_cfg, config_out)


# ==========================================================
# Punto de entrada CLI
# ==========================================================

if __name__ == "__main__":
    main_tuning()
