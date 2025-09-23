# arima_adapter.py
# Adapter ARIMA/SARIMA compatible con evaluacion_modelos.compute_metrics_prophet
# Autor: (tu equipo)
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
    Obtiene un índice de tiempo consistente a partir de:
    - Columna 'time', o
    - Índice datetime llamado 'time', o
    - Índice datetime genérico
    """
    if 'time' in df.columns:
        idx = pd.to_datetime(df['time'])
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        raise ValueError("Se requiere columna 'time' o índice datetime en el DataFrame.")
    if idx.is_monotonic_increasing is False:
        # Aseguramos orden temporal
        df_sorted = df.copy()
        df_sorted.index = idx
        df_sorted = df_sorted.sort_index()
        return df_sorted.index
    return idx


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida columnas mínimas y devuelve df ordenado por tiempo sin NaN en Close.
    """
    if 'Close' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Close'.")

    # Usamos índice/columna de tiempo consistente
    idx = _ensure_time_index(df)
    base = df.copy()
    base.index = idx
    base = base.sort_index()
    base = base[['Close']].astype(float)
    base = base.dropna()
    if len(base) < 30:
        raise ValueError("Muy pocos datos para ajustar un ARIMA/SARIMA (min ~30 observaciones).")
    return base


def _build_future_index(last_time: pd.Timestamp, pasos: int, frecuencia: Optional[str], fallback: str) -> pd.DatetimeIndex:
    """
    Construye el índice futuro usando la frecuencia proporcionada o inferida.
    """
    freq = frecuencia or fallback or 'D'
    # Genera pasos timestamps futuros excluyendo el último time ya presente
    future_index = pd.date_range(start=last_time, periods=pasos+1, freq=freq)[1:]
    return future_index


# -------------------------------
# API pública del adapter
# -------------------------------
def entrenar_modelo_arima(
    df: pd.DataFrame,
    modo: str = 'nivel',                 # 'nivel' o 'retornos'
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,  # p,D,q,m -> usa SARIMAX si no es None
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> Dict[str, Any]:
    """
    Ajusta ARIMA (o SARIMA si se provee seasonal_order) de forma compatible con la herramienta.
    - df: DataFrame con columnas ['time','Close'] o índice datetime.
    - modo:
        'nivel'    -> modela el precio (Close) con (p,d,q) y opcionalmente (P,D,Q,m)
        'retornos' -> modela retornos porcentuales y reconstruye precio en la predicción
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

    # Si seasonal_order es None, usamos SARIMAX igualmente con (0,0,0,0) para unificar interfaz
    seasonal_order = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)

    # Ajuste del modelo
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        trend=None,
        # simple_differencing=False mantiene la integración dentro del modelo
    )
    fitted = model.fit(disp=False)

    # Frecuencia inferida del histórico (fallback a 'D' si no se puede)
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
    Predice 'pasos' periodos hacia adelante y devuelve el DataFrame estándar:
    ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado']
    - Si 'modo' fue 'retornos': interpreta la media pronosticada como retornos y reconstruye precio.
    - Si 'modo' fue 'nivel'  : interpreta la media como precio.
    """
    if pasos < 1:
        raise ValueError("El numero de 'pasos' debe ser >= 1.")

    fitted = modelo['fitted']
    last_time = modelo['last_time']
    fallback_freq = modelo.get('freq', 'D')
    modo = modelo.get('modo', 'nivel')
    ultimo_close = modelo.get('ultimo_close', None)

    # Forecast multi-step
    fc = fitted.get_forecast(steps=pasos)
    mean = fc.predicted_mean  # Serie (n,)
    conf = fc.conf_int(alpha=alpha)  # DataFrame 2 columnas

    # Algunas versiones nombran columnas como ['lower y', 'upper y'] o similar.
    low = conf.iloc[:, 0].to_numpy(dtype=float)
    up  = conf.iloc[:, 1].to_numpy(dtype=float)
    mean_np = mean.to_numpy(dtype=float)

    # Construir índice futuro
    future_index = _build_future_index(last_time, pasos, frecuencia, fallback_freq)

    if modo == 'retornos':
        if ultimo_close is None:
            raise ValueError("No se encontró 'ultimo_close' para reconstruir precio en modo 'retornos'.")

        # Reconstruimos trayectoria de precio a partir de retornos pronosticados
        # mean: retornos esperados por paso (aprox). Usamos acumulado multiplicativo.
        price_path = ultimo_close * np.cumprod(1.0 + mean_np)

        # Bandas aproximadas: aplicamos las bandas de retornos a la reconstrucción
        # (asunción simplificadora; documentar en tesis)
        price_low = ultimo_close * np.cumprod(1.0 + low)
        price_up  = ultimo_close * np.cumprod(1.0 + up)

        out = pd.DataFrame({
            'timestamp_prediccion': future_index,
            'precio_estimado': price_path,
            'min_esperado': price_low,
            'max_esperado': price_up
        })

    else:  # modo == 'nivel'
        out = pd.DataFrame({
            'timestamp_prediccion': future_index,
            'precio_estimado': mean_np,
            'min_esperado': low,
            'max_esperado': up
        })

    # Garantizamos tipos float
    out['precio_estimado'] = out['precio_estimado'].astype(float)
    out['min_esperado']    = out['min_esperado'].astype(float)
    out['max_esperado']    = out['max_esperado'].astype(float)
    return out
