# modelos/arima_ets.py
# ARIMA y ETS sencillos, con firmas coherentes y fáciles de leer.

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------- ARIMA ----------
def fit_arima_series(price: pd.Series, order=(1,1,1)):
    """
    Ajusta un ARIMA(p,d,q) sobre niveles (no retornos).
    - order=(1,1,1) es un punto de partida clásico.
    - Desactivamos restricciones para robustez con datos financieros.
    """
    m = ARIMA(price.astype(float), order=order,
              enforce_stationarity=False, enforce_invertibility=False)
    return m.fit()

def arima_forecast_steps(fit, steps: int, index) -> pd.Series:
    """
    Pronóstico multi-paso ARIMA por número de pasos.
    Reasignamos el índice del conjunto de test.
    """
    fc = fit.get_forecast(steps=steps).predicted_mean
    return pd.Series(fc.values, index=index, name="arima")

# ---------- ETS (Holt-Winters) ----------
def fit_ets_series(price: pd.Series, trend="add", seasonal=None, seasonal_periods=None):
    """
    ETS sencillo. En intradía suele usarse sin estacionalidad.
    En diario puedes probar seasonal_periods=5/22.
    """
    m = ExponentialSmoothing(price.astype(float),
                             trend=trend, seasonal=seasonal,
                             seasonal_periods=seasonal_periods)
    return m.fit(optimized=True)

def ets_forecast_steps(fit, steps: int, index) -> pd.Series:
    """Pronóstico ETS multi-paso."""
    fc = fit.forecast(steps)
    return pd.Series(np.asarray(fc), index=index, name="ets")
