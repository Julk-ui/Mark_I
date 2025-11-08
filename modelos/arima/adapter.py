# modelos/arima/adapter.py
# ARIMA/SARIMA adapter con selección automática por ventana (pmdarima)
# - Interfaz homogénea con LSTM:
#     .fit(series: pd.Series)              -> entrena
#     .predict(horizon, last_window=None,
#               last_timestamp=None,
#               index=None) -> DataFrame   -> SIEMPRE columna 'yhat' e índice temporal
# - Modo 'nivel' (precio) o 'retornos' (reconstruye precio desde el último close)
# - Auto-selección por ventana con límites de búsqueda (max_p,max_q,max_P,max_Q <= 2)
# - Expone metadatos del modelo elegido con .info()

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

# pmdarima para selección automática
try:
    import pmdarima as pm
except Exception as e:
    pm = None


class ArimaModel:
    """
    Adapter ARIMA/SARIMA univar con salida 'yhat' (igual a LSTM).
    """

    def __init__(self,
                 model_cfg: Optional[Dict[str, Any]] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        model_cfg = model_cfg or {}
        cfg = cfg or {}

        # ------------------------------
        # Config general
        # ------------------------------
        self.modo: str = str(model_cfg.get("modo", cfg.get("modo", "nivel"))).lower()
        if self.modo not in {"nivel", "retornos"}:
            self.modo = "nivel"

        # Frecuencia para construir índice futuro si no se pasa 'index'
        self._freq_cfg: str = str(cfg.get("freq", "H"))

        # ------------------------------
        # Selección automática (pmdarima)
        # ------------------------------
        self.auto: bool = bool(model_cfg.get("auto", True))
        self.metric: str = str(model_cfg.get("metric", "aic")).lower()  # "aic" | "bic"
        self.seasonal: bool = bool(model_cfg.get("seasonal", False))
        self.m: int = int(model_cfg.get("m", 12 if self.seasonal else 1))

        # Límites de búsqueda (topes seguros para que no explote)
        self.max_p: int = int(model_cfg.get("max_p", 2))
        self.max_q: int = int(model_cfg.get("max_q", 2))
        self.max_P: int = int(model_cfg.get("max_P", 2))
        self.max_Q: int = int(model_cfg.get("max_Q", 2))
        self.max_d: int = int(model_cfg.get("max_d", 2))   # differencing no estacional
        self.max_D: int = int(model_cfg.get("max_D", 1))   # differencing estacional

        # Parámetros fijos si auto=False
        self.order: Tuple[int, int, int] = tuple(model_cfg.get("order", (1, 1, 1)))
        self.seasonal_order: Tuple[int, int, int, int] = tuple(model_cfg.get("seasonal_order", (0, 0, 0, 0)))

        # Flags de restricciones (más laxo para evitar fallos de convergencia)
        self.enforce_stationarity: bool = bool(model_cfg.get("enforce_stationarity", False))
        self.enforce_invertibility: bool = bool(model_cfg.get("enforce_invertibility", False))

        # ------------------------------
        # Estado tras el fit
        # ------------------------------
        self._results = None                        # SARIMAXResults
        self._train_index: Optional[pd.DatetimeIndex] = None
        self._ultimo_close: Optional[float] = None  # base para reconstrucción (modo 'retornos')
        self._chosen: Dict[str, Any] = {}           # metadatos del mejor modelo en la ventana

    # ======================================================================
    # API pública
    # ======================================================================
    def fit(self, series: pd.Series) -> None:
        """
        Entrena el modelo con una pd.Series univar (índice datetime, orden ascendente).
        En modo 'retornos' modela pct_change() y guarda el último close para reconstruir.
        """
        s = pd.Series(series).astype(float).dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError("La serie debe tener índice de tiempo (DatetimeIndex).")
        s = s.sort_index()

        self._train_index = s.index

        if self.modo == "retornos":
            y = s.pct_change().dropna()
            if len(y) < 30:
                raise ValueError("Muy pocos datos para ARIMA en 'retornos' (mínimo recomendado ~30).")
            self._ultimo_close = float(s.iloc[-1])
        else:
            y = s
            if len(y) < 30:
                raise ValueError("Muy pocos datos para ARIMA en 'nivel' (mínimo recomendado ~30).")
            self._ultimo_close = None

        # ------------------------------
        # Selección automática por ventana
        # ------------------------------
        if self.auto:
            if pm is None:
                raise RuntimeError(
                    "pmdarima no está disponible. Instala 'pmdarima' o desactiva auto=True "
                    "y provee 'order'/'seasonal_order'."
                )

            # clamp de seguridad por si el usuario configura >2
            max_p = max(0, min(int(self.max_p), 2))
            max_q = max(0, min(int(self.max_q), 2))
            max_P = max(0, min(int(self.max_P), 2))
            max_Q = max(0, min(int(self.max_Q), 2))
            max_d = max(0, int(self.max_d))
            max_D = max(0, int(self.max_D))

            am = pm.auto_arima(
                y,
                seasonal=self.seasonal,
                m=(self.m if self.seasonal else 1),
                start_p=0, start_q=0, start_P=0, start_Q=0,
                max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                max_d=max_d, max_D=max_D,
                stepwise=True,
                trace=False,
                information_criterion=self.metric,  # "aic" o "bic"
                error_action="ignore",
                suppress_warnings=True,
                with_intercept=False,
            )

            order = am.order
            sorder = am.seasonal_order if self.seasonal else (0, 0, 0, 0)

            # Guardamos la elección (ojo: AIC/BIC finales los tomamos del SARIMAX de abajo)
            self._chosen = {
                "selected_by": "pmdarima.auto_arima",
                "metric": self.metric,
                "order": str(order),
                "seasonal_order": str(sorder) if sorder else "",
                "m": int(self.m if self.seasonal else 1),
                "max_p": max_p, "max_q": max_q, "max_P": max_P, "max_Q": max_Q,
                "max_d": max_d, "max_D": max_D,
            }

        else:
            order = tuple(self.order)
            sorder = tuple(self.seasonal_order if self.seasonal else (0, 0, 0, 0))
            self._chosen = {
                "selected_by": "manual",
                "metric": self.metric,
                "order": str(order),
                "seasonal_order": str(sorder) if sorder else "",
                "m": int(self.m if self.seasonal else 1),
            }

        # ------------------------------
        # Fit final homogéneo con SARIMAX
        # ------------------------------
        self.order = order
        self.seasonal_order = sorder

        self._results = SARIMAX(
            y,
            order=order,
            seasonal_order=sorder if self.seasonal else (0, 0, 0, 0),
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            trend=None,
        ).fit(disp=False)

        # métrica final del fit
        self._chosen.update({
            "aic": float(getattr(self._results, "aic", np.nan)),
            "bic": float(getattr(self._results, "bic", np.nan)),
            "nobs": int(getattr(self._results, "nobs", len(y))),
        })

    def predict(self,
                horizon: int,
                last_window: Optional[pd.Series] = None,
                last_timestamp: Optional[pd.Timestamp] = None,
                index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Devuelve SIEMPRE DataFrame con columna 'yhat' (igual a LSTM) y el índice temporal.
        """
        if self._results is None:
            raise RuntimeError("ARIMA no entrenado. Llama fit() antes de predict().")
        if horizon < 1:
            raise ValueError("`horizon` debe ser >= 1.")

        fc = self._results.get_forecast(steps=int(horizon))
        mean = fc.predicted_mean.to_numpy(dtype=float)

        if self.modo == "retornos":
            base = self._ultimo_close
            if base is None:
                # fallback: tomar del last_window si no guardamos el último close
                if last_window is None or len(last_window) == 0:
                    raise ValueError("No se pudo reconstruir precio: falta 'ultimo_close' o 'last_window'.")
                base = float(pd.Series(last_window).astype(float).iloc[-1])
            yhat_vals = base * np.cumprod(1.0 + mean)
        else:
            yhat_vals = mean

        # Índice de salida (mismo criterio que LSTM)
        out_idx = self._make_index(horizon, last_timestamp, index)

        return pd.DataFrame({"yhat": np.asarray(yhat_vals, dtype=float).reshape(-1)}, index=out_idx)

    def info(self) -> Dict[str, Any]:
        """
        Metadatos del modelo elegido en la ventana (para resumen/Excel).
        Incluye: order, seasonal_order, metric, aic, bic, nobs, etc.
        """
        base = dict(self._chosen) if self._chosen else {}
        # también recordamos los órdenes efectivos usados en SARIMAX
        base.update({
            "engine": "arima",
            "fitted_order": str(tuple(self.order)) if hasattr(self, "order") else "",
            "fitted_seasonal_order": str(tuple(self.seasonal_order)) if hasattr(self, "seasonal_order") else "",
            "seasonal": bool(self.seasonal),
        })
        return base

    # ======================================================================
    # Helpers
    # ======================================================================
    def _make_index(self,
                    horizon: int,
                    last_timestamp: Optional[pd.Timestamp],
                    index: Optional[pd.DatetimeIndex]) -> pd.Index:
        if index is not None:
            try:
                return pd.DatetimeIndex(index)
            except Exception:
                return pd.Index(index)

        freq = self._infer_freq_safe()
        if last_timestamp is None:
            # sin timestamp, usa un índice ordinal como fallback
            return pd.RangeIndex(start=1, stop=horizon + 1, step=1)

        try:
            return pd.date_range(start=pd.Timestamp(last_timestamp),
                                 periods=horizon + 1, freq=freq)[1:]
        except Exception:
            return pd.RangeIndex(start=1, stop=horizon + 1, step=1)

    def _infer_freq_safe(self) -> str:
        if isinstance(self._train_index, pd.DatetimeIndex):
            try:
                f = pd.infer_freq(self._train_index)
                if f:
                    return str(f)
            except Exception:
                pass
        # fallback razonable
        return "H" if str(self._freq_cfg).upper().startswith("H") else "D"
