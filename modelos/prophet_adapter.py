# modelos/prophet_adapter.py
# Adapter Prophet compatible con la misma interfaz que ARIMA/MLP/LSTM
# Requisitos: prophet (pip install prophet), pandas, numpy

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from prophet import Prophet


# -------------------------------
# Helpers
# -------------------------------
def _ensure_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Obtiene un índice datetime consistente desde:
      - Columna 'time', o
      - Índice datetime (cualquier nombre)
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
    if 'Close' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Close'.")
    idx = _ensure_time_index(df)
    base = df.copy()
    base.index = idx
    base = base.sort_index()
    base = base[['Close']].astype(float).dropna()
    if len(base) < 40:
        raise ValueError("Muy pocos datos para Prophet (mín ~40 observaciones).")
    return base


def _seasonality_defaults_from_freq(freq: Optional[str]) -> Dict[str, bool]:
    """
    Heurística sencilla:
      - Intradía (minutos/horas): daily=True, weekly=True, yearly=False
      - Diario: weekly=True, yearly=True, daily=False
    """
    if not freq:
        return dict(daily=False, weekly=True, yearly=True)
    f = freq.lower()
    if any(x in f for x in ['min', 'h']):   # '15min', '30min', '1h'
        return dict(daily=True, weekly=True, yearly=False)
    # 'd' o 'b'
    if 'd' in f or 'b' in f:
        return dict(daily=False, weekly=True, yearly=True)
    # por defecto:
    return dict(daily=False, weekly=True, yearly=True)


# -------------------------------
# API pública
# -------------------------------
def entrenar_modelo_prophet(
    df: pd.DataFrame,
    modo: str = 'nivel',                     # 'nivel' | 'retornos'
    frecuencia_hint: Optional[str] = None,   # '15min' | '1H' | 'D' ... (opcional)
    interval_width: float = 0.90,            # bandas de confianza 90%
    seasonality_mode: str = 'additive',      # 'additive' | 'multiplicative'
    changepoint_prior_scale: float = 0.05,
) -> Dict[str, Any]:
    """
    Entrena Prophet para precio (nivel) o retornos, manteniendo el contrato del resto de adapters.
    Devuelve un dict con el modelo y metadatos.
    """
    if modo not in ('nivel', 'retornos'):
        raise ValueError("Parametro 'modo' debe ser 'nivel' o 'retornos'.")

    base = _validate_and_prepare(df)
    idx = base.index
    last_time = idx.max()
    freq_inferida = pd.infer_freq(idx)

    # Serie objetivo
    if modo == 'retornos':
        serie = base['Close'].pct_change().dropna()
        ultimo_close = float(base['Close'].iloc[-1])
        data = pd.DataFrame({'ds': serie.index, 'y': serie.values})
    else:
        serie = base['Close']
        ultimo_close = None
        data = pd.DataFrame({'ds': serie.index, 'y': serie.values})

    # Estacionalidades recomendadas según frecuencia
    seas = _seasonality_defaults_from_freq(frecuencia_hint or freq_inferida)

    m = Prophet(
        daily_seasonality=seas['daily'],
        weekly_seasonality=seas['weekly'],
        yearly_seasonality=seas['yearly'],
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width
    )

    m.fit(data)

    return {
        'modelo': m,
        'modo': modo,
        'ultimo_close': ultimo_close,   # para reconstrucción en 'retornos'
        'freq': (frecuencia_hint or freq_inferida or 'D'),
        'last_time': last_time,
    }


def predecir_precio_prophet(
    modelo_dict: Dict[str, Any],
    pasos: int = 3,
    frecuencia: Optional[str] = None
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
      ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado']
    - En 'retornos': trata yhat como retornos y reconstruye precio por acumulación desde ultimo_close.
    - En 'nivel'   : yhat se interpreta directamente como precio.
    """
    if pasos < 1:
        raise ValueError("El numero de 'pasos' debe ser >= 1.")

    m: Prophet = modelo_dict['modelo']
    modo = modelo_dict.get('modo', 'nivel')
    ultimo_close = modelo_dict.get('ultimo_close', None)

    # Frecuencia para el futuro
    freq = frecuencia or modelo_dict.get('freq', 'D')

    # Construimos horizonte futuro
    futuro = m.make_future_dataframe(periods=pasos, freq=freq)
    forecast = m.predict(futuro)
    tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(pasos).copy()

    if modo == 'retornos':
        if ultimo_close is None:
            raise ValueError("No se encontró 'ultimo_close' para reconstruir precio en modo 'retornos'.")

        # Reconstrucción multiplicativa con retornos pronosticados
        r_est = tail['yhat'].to_numpy(dtype=float)
        r_lo  = tail['yhat_lower'].to_numpy(dtype=float)
        r_up  = tail['yhat_upper'].to_numpy(dtype=float)

        price_est = ultimo_close * np.cumprod(1.0 + r_est)
        price_lo  = ultimo_close * np.cumprod(1.0 + r_lo)
        price_up  = ultimo_close * np.cumprod(1.0 + r_up)

        out = pd.DataFrame({
            'timestamp_prediccion': pd.to_datetime(tail['ds'].to_numpy()),
            'precio_estimado': price_est.astype(float),
            'min_esperado': price_lo.astype(float),
            'max_esperado': price_up.astype(float)
        })
    else:
        out = tail.rename(columns={
            'ds': 'timestamp_prediccion',
            'yhat': 'precio_estimado',
            'yhat_lower': 'min_esperado',
            'yhat_upper': 'max_esperado'
        })[['timestamp_prediccion', 'precio_estimado', 'min_esperado', 'max_esperado']]

    out['precio_estimado'] = out['precio_estimado'].astype(float)
    out['min_esperado']    = out['min_esperado'].astype(float)
    out['max_esperado']    = out['max_esperado'].astype(float)
    return out
