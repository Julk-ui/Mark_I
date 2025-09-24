import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

def fit_arima_series(price: pd.Series, order=(1,1,1)):
    """
    Ajusta un ARIMA(p,d,q) sobre la serie de niveles (no retornos).
    Desactiva restricciones de estacionariedad/invertibilidad para robustez.
    """
    m = ARIMA(price.astype(float), order=order,
              enforce_stationarity=False, enforce_invertibility=False)
    return m.fit()

def arima_forecast_steps(fit, steps: int, index) -> pd.Series:
    """Pronóstico ARIMA por número de pasos; reasigna el índice del test."""
    fc = fit.get_forecast(steps=steps).predicted_mean
    return pd.Series(fc.values, index=index, name="arima")


def fit_ets_series(price: pd.Series, trend="add", seasonal=None, seasonal_periods=None):
    """
    Holt–Winters / ETS. En financieros suele usarse sin estacionalidad explícita.
    """
    m = ExponentialSmoothing(price.astype(float),
                             trend=trend, seasonal=seasonal,
                             seasonal_periods=seasonal_periods)
    return m.fit(optimized=True)

def ets_forecast_steps(fit, steps: int, index) -> pd.Series:
    """Pronóstico ETS por número de pasos; reasigna el índice del test."""
    fc = fit.forecast(steps)
    return pd.Series(np.asarray(fc), index=index, name="ets")