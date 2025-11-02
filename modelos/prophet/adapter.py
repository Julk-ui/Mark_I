# modelos/prophet/adapter.py
from __future__ import annotations

from typing import Optional, Dict, Any
import warnings
import pandas as pd

try:
    from prophet import Prophet
except Exception as e:
    raise ImportError("No se pudo importar 'prophet'. Instala con: pip install prophet") from e


class _State:
    """Contenedor ligero del modelo Prophet entrenado y metadata mínima."""
    def __init__(self, model: Prophet, frecuencia: Optional[str]) -> None:
        self.model = model
        self.frecuencia = frecuencia  # "H" / "D" o None


def _make_prophet(
    interval_width: float = 0.8,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
    yearly_seasonality: bool | int = True,
    weekly_seasonality: bool | int = False,
    daily_seasonality: bool | int = False,
    **kwargs: Any,  # tolera extras sin romper
) -> Prophet:
    """Crea un Prophet con parámetros clave; ignora kwargs desconocidos."""
    m = Prophet(
        interval_width=interval_width,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )
    # Si quieres extra seasonality, holidays, etc., podrías leerlas aquí desde kwargs
    return m


def entrenar_modelo_prophet(
    df: pd.DataFrame,
    modo: str = "retornos",
    frecuencia_hint: Optional[str] = None,
    interval_width: float = 0.8,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
    yearly_seasonality: bool | int = True,
    weekly_seasonality: bool | int = False,
    daily_seasonality: bool | int = False,
    **kwargs: Any,
) -> _State:
    """
    Entrena Prophet con una serie univariada en formato (ds, y).

    Parámetros
    ----------
    df : DataFrame con columnas ['ds','y'].
    modo : 'retornos' | 'nivel'. En este adapter se entrena directamente sobre 'y'.
    frecuencia_hint : 'H' | 'D' | None. Si se provee, guía la construcción del futuro.
    interval_width, seasonality_mode, changepoint_prior_scale, yearly_seasonality,
    weekly_seasonality, daily_seasonality : hiperparámetros estándar de Prophet.
    **kwargs : tolera parámetros adicionales sin romper (para compatibilidad con registry/YAML).

    Retorna
    -------
    _State : objeto con el modelo Prophet entrenado y la frecuencia sugerida.
    """
    # Asegura formato
    if not {"ds", "y"}.issubset(df.columns):
        raise ValueError("df debe contener columnas 'ds' y 'y'.")

    # Copia defensiva
    data = df[["ds", "y"]].copy()
    data["ds"] = pd.to_datetime(data["ds"], utc=False)

    # Construye y entrena
    model = _make_prophet(
        interval_width=interval_width,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        **kwargs,  # se ignoran extras no soportados
    )
    model.fit(data)

    return _State(model=model, frecuencia=frecuencia_hint)


def predecir_precio_prophet(
    state: _State,
    pasos: int,
    frecuencia: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Genera pronóstico a 'pasos' usando Prophet y devuelve DataFrame con 'yhat'.

    Parámetros
    ----------
    state : _State devuelto por entrenar_modelo_prophet.
    pasos : int, horizonte de predicción.
    frecuencia : 'H' | 'D' | None. Si None, usa la almacenada en state.frecuencia; si no hay, Prophet infiere.
    **kwargs : tolera extras para compatibilidad.

    Retorna
    -------
    DataFrame con columnas ['ds','yhat'] y ds como índice.
    """
    m = state.model
    freq = frecuencia or state.frecuencia  # prioriza argumento directo

    if pasos <= 0:
        raise ValueError("pasos debe ser > 0")

    if freq is None:
        # Prophet puede inferir frecuencia del historial; si falla, advertimos.
        warnings.warn("No se especificó 'frecuencia'; Prophet intentará inferirla.", RuntimeWarning)
        future = m.make_future_dataframe(periods=pasos, include_history=False)
    else:
        # Construye el futuro con frecuencia explícita (H, D, etc.)
        future = m.make_future_dataframe(periods=pasos, include_history=False, freq=freq)

    fcst = m.predict(future)
    out = fcst[["ds", "yhat"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], utc=False)
    out.set_index("ds", inplace=True)
    return out
