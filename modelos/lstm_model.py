import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

# TensorFlow es opcional; si no está, puedes envolver el train/predict en try/except.
from tensorflow import keras

def _supervised(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte una serie en dataset supervisado con ventana 'window'.
    Retorna X:(N,window,1) y y:(N,1)
    """
    x, y = [], []
    s = series.values.astype("float32")
    for i in range(window, len(s)):
        x.append(s[i-window:i])
        y.append(s[i])
    X = np.array(x)[:, :, None]
    y = np.array(y)[:, None]
    return X, y

def train_lstm_returns(price: pd.Series, window: int = 22, epochs: int = 30, batch: int = 32):
    """
    Entrena LSTM sobre RETORNOS (log-diff) y luego reconstruye el nivel.
    Más estable que modelar el nivel directo.
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
    Usa el estado entrenado para predecir el siguiente retorno y reconstruir el precio.
    Devuelve una serie alineada con el índice de entrada (últimos puntos).
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
