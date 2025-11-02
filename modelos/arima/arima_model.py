# modelos/arima/adapter.py
# Adapter ARIMA/SARIMA compatible con tu main.py (modo normal)
# Requisitos: statsmodels>=0.13, pandas, numpy

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------------------
# Helpers internos
# -------------------------------
def _ensure_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Usa columna 'time' o índice datetime. Ordena ascendente si es necesario.
    """
    if 'time' in df.columns:
        idx = pd.to_datetime(df['time'])
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        raise ValueError("Se requiere columna 'time' o índice datetime en el DataFrame.")
    if not idx.is_monotonic_increasing:
        df2 = df.copy()
        df2.index = idx
        df2 = df2.sort_index()
        return df2.index
    return idx

def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida columnas mínimas y devuelve df ordenado por tiempo sin NaN en Close.
    """
    if 'Close' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Close'.")

    idx = _ensure_time_index(df)
    base = df.copy()
    base.index = idx
    base = base.sort_index()
    base = base[['Close']].astype(float).dropna()
    if len(base) < 30:
        raise ValueError("Muy pocos datos para ARIMA/SARIMA (mín ~30 observaciones).")
    return base

def _build_future_index(last_time: pd.Timestamp, pasos: int, frecuencia: Optional[str], fallback: str) -> pd.DatetimeIndex:
    """
    Construye el índice futuro usando la frecuencia proporcionada o inferida.
    """
    freq = frecuencia or fallback or 'D'
    # Genera 'pasos' timestamps futuros excluyendo el último ya presente
    future_index = pd.date_range(start=last_time, periods=pasos+1, freq=freq)[1:]
    return future_index

# -------------------------------
# API pública del adapter
# -------------------------------
def entrenar_modelo_arima(
    df: pd.DataFrame,
    modo: str = 'nivel',                 # 'nivel' | 'retornos'
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,  # (P,D,Q,m); si None -> no estacional
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> Dict[str, Any]:
    """
    Ajusta ARIMA (o SARIMA si se provee seasonal_order) de forma compatible con main.py.
    - df: DataFrame con ['time','Close'] o índice datetime.
    - modo:
        'nivel'    -> modela precio (Close)
        'retornos' -> modela retornos (pct_change) y reconstruye precio al predecir
    """
    base = _validate_and_prepare(df)
    idx = base.index

    if modo not in ('nivel', 'retornos'):
        raise ValueError("Parametro 'modo' debe ser 'nivel' o 'retornos'.")

    if modo == 'retornos':
        y = base['Close'].pct_change().dropna()
        ultimo_close = float(base['Close'].iloc[-1])
    else:
        y = base['Close']
        ultimo_close = None

    seasonal_order = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        trend=None,
    )
    fitted = model.fit(disp=False)

    freq_inferida = pd.infer_freq(idx)
    last_time = idx.max()

    return {
        'fitted': fitted,
        'last_time': last_time,
        'freq': freq_inferida or 'D',
        'modo': modo,
        'ultimo_close': ultimo_close,
    }

def predecir_precio_arima(
    modelo: Dict[str, Any],
    pasos: int = 3,
    frecuencia: Optional[str] = None,
    alpha: float = 0.10,  # 90% CI por defecto
) -> pd.DataFrame:
    """
    Predice 'pasos' periodos hacia adelante y devuelve:
      ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado']
    """
    if pasos < 1:
        raise ValueError("El numero de 'pasos' debe ser >= 1.")

    fitted = modelo['fitted']
    last_time = modelo['last_time']
    fallback_freq = modelo.get('freq', 'D')
    modo = modelo.get('modo', 'nivel')
    ultimo_close = modelo.get('ultimo_close', None)

    fc = fitted.get_forecast(steps=pasos)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=alpha)

    low = conf.iloc[:, 0].to_numpy(dtype=float)
    up  = conf.iloc[:, 1].to_numpy(dtype=float)
    mean_np = mean.to_numpy(dtype=float)

    future_index = _build_future_index(last_time, pasos, frecuencia, fallback_freq)

    if modo == 'retornos':
        if ultimo_close is None:
            raise ValueError("No se encontró 'ultimo_close' para reconstruir precio en modo 'retornos'.")

        price_est = ultimo_close * np.cumprod(1.0 + mean_np)
        price_lo  = ultimo_close * np.cumprod(1.0 + low)
        price_up  = ultimo_close * np.cumprod(1.0 + up)

        out = pd.DataFrame({
            'timestamp_prediccion': future_index,
            'precio_estimado': price_est.astype(float),
            'min_esperado': price_lo.astype(float),
            'max_esperado': price_up.astype(float)
        })
    else:
        out = pd.DataFrame({
            'timestamp_prediccion': future_index,
            'precio_estimado': mean_np.astype(float),
            'min_esperado': low.astype(float),
            'max_esperado': up.astype(float)
        })

    return out

# --- Alias opcionales de compatibilidad ---
# (si en algún punto tu registry o main busca estos nombres genéricos)
entrenar_modelo = entrenar_modelo_arima
predecir_precio = predecir_precio_arima

__all__ = ["entrenar_modelo_arima", "predecir_precio_arima", "entrenar_modelo", "predecir_precio"]
