# modelos/lstm_model.py
# LSTM minimalista sobre RETORNOS (log-diff), re-integrando a precio.
# Si TensorFlow no está disponible, captura el error en quien lo invoque.

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def _supervised(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye dataset supervisado con ventana 'window'.
    X: (N, window, 1), y: (N, 1)
    """
    x, y = [], []
    s = series.values.astype("float32")
    for i in range(window, len(s)):
        x.append(s[i-window:i])
        y.append(s[i])
    X = np.array(x)[:, :, None]
    y = np.array(y)[:, None]
    return X, y

def train_lstm_returns(price: pd.Series, window: int = 22, epochs: int = 20, batch: int = 32):
    """
    Entrena LSTM sobre RETORNOS (log-diff). Suele ser más estable que niveles.
    - window: tamaño de ventana (22~1 mes de trading en D1).
    - epochs: pocas (queremos simple/rápido).
    """
    r = np.log(price).diff().dropna()
    scaler = StandardScaler()
    r_sc = pd.Series(
        scaler.fit_transform(r.values.reshape(-1,1)).ravel(),
        index=r.index, name="ret_sc"
    )
    X, y = _supervised(r_sc, window)
    model = keras.Sequential([
        keras.layers.Input(shape=(window,1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)
    return {"model": model, "scaler": scaler, "window": window}

def lstm_predict_price(state, price: pd.Series) -> pd.Series:
    """
    Predice el siguiente retorno estandarizado y lo re-integra a nivel de precio.
    Devuelve una serie alineada con el final de 'price'.
    """
    scaler = state["scaler"]; W = state["window"]; model = state["model"]
    r = np.log(price).diff().dropna()
    r_sc = pd.Series(
        scaler.transform(r.values.reshape(-1,1)).ravel(),
        index=r.index
    )
    X, _ = _supervised(r_sc, W)
    pred_sc = model.predict(X, verbose=0).ravel()
    pred_r = scaler.inverse_transform(pred_sc.reshape(-1,1)).ravel()
    last_price = price.shift(1).dropna().values[-len(pred_r):]
    price_pred = last_price * np.exp(pred_r)
    out = pd.Series(price_pred, index=price.index[-len(price_pred):], name="lstm")
    return out
