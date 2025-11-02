# app/utils/registry.py
from __future__ import annotations

from typing import Protocol, Dict, Any, Optional
import importlib
import inspect
import pandas as pd


# ==============================
# Interfaz unificada de modelos
# ==============================
class ModelLike(Protocol):
    """Interfaz mínima que exponen todos los modelos para el pipeline."""
    def fit(self, series: pd.Series) -> None: ...
    def predict(
        self,
        horizon: int,
        last_window: Optional[pd.Series] = None,
        last_timestamp: Optional[pd.Timestamp] = None,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame: ...


# ==============================
# Utilidades internas
# ==============================
def _call_with_supported(fn, **kwargs):
    """
    Llama a `fn` pasando solo los kwargs soportados por su firma.
    Si `fn` acepta **kwargs, se pasan todos sin filtrar.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def _normalize_yhat_df(raw: pd.DataFrame, index: Optional[pd.Index]) -> pd.DataFrame:
    """
    Convierte un DataFrame de predicciones al formato estándar: una columna 'yhat'
    y un índice datetime. Busca nombres comunes si 'yhat' no existe.
    """
    if "yhat" in raw.columns:
        out = raw.copy()
    else:
        col = None
        for c in ["yhat", "y_pred", "yhat_pred", "precio_estimado", "forecast", "y"]:
            if c in raw.columns:
                col = c
                break
        if col is None:
            num_cols = raw.select_dtypes("number").columns
            if len(num_cols) == 0:
                raise ValueError("No se encontró columna numérica en salida del modelo.")
            col = num_cols[0]
        out = pd.DataFrame({"yhat": raw[col].values}, index=raw.index)
    if index is not None:
        out.index = index
    return out


def _get_ci(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    """Obtiene d[k] ignorando mayúsculas/minúsculas y variantes simples."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
        kl = k.lower()
        ku = k.upper()
        kc = k[:1].upper() + k[1:].lower()
        for cand in (kl, ku, kc):
            if cand in d and d[cand] is not None:
                return d[cand]
    return None


def _map_timeframe_to_freq(tf: Optional[str]) -> Optional[str]:
    """Mapea timeframes típicos MT5 a códigos Prophet/pandas: H1->H, D1->D, M15->'15min'."""
    if not tf:
        return None
    t = str(tf).strip().upper()
    if t == "D1":
        return "D"
    if t in {"H1", "H"}:
        return "H"
    if t in {"H4"}:
        return "H"
    if t in {"M30"}:
        return "30min"
    if t in {"M15"}:
        return "15min"
    if t in {"M5"}:
        return "5min"
    if t in {"M1"}:
        return "min"
    return None


def _ensure_dt_index(x) -> pd.DatetimeIndex:
    """
    Normaliza `x` (Series/Index/DatetimeIndex/array) a `DatetimeIndex` sin tz.
    Robusto ante distintos tipos de entrada.
    """
    # Ya es DatetimeIndex
    if isinstance(x, pd.DatetimeIndex):
        try:
            return x.tz_localize(None)
        except Exception:
            return x

    # Serie o Index -> convertir a datetime y luego a DatetimeIndex
    try:
        dt = pd.to_datetime(x)
    except Exception:
        dt = pd.to_datetime(x, errors="coerce")

    if isinstance(dt, pd.Series):
        # quitar tz si la hay y construir DatetimeIndex
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            # si no tiene tz, ignora
            pass
        return pd.DatetimeIndex(dt.values)

    if isinstance(dt, pd.DatetimeIndex):
        try:
            return dt.tz_localize(None)
        except Exception:
            return dt

    # Fallback genérico
    return pd.DatetimeIndex(dt)


def _infer_freq_from_index(idx: pd.DatetimeIndex) -> Optional[str]:
    """
    Intenta inferir una frecuencia a partir del index:
    - usa .inferred_freq si existe
    - si no, diferencia mediana y mapea a 'H', 'D', 'min', '15min', etc.
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return None
    if idx.inferred_freq:
        f = idx.inferred_freq
        if f.upper().startswith("H"):
            return "H"
        if f.upper().startswith("D"):
            return "D"
        if "T" in f.upper():  # 'T' ~ minute
            try:
                n = int(f.upper().replace("T", ""))
                return f"{n}min"
            except Exception:
                return "min"
        return f
    # fallback por diferencia mediana
    try:
        delta = pd.Series(idx[1:] - idx[:-1]).median()
        mins = int(delta.total_seconds() // 60)
        if mins == 60:
            return "H"
        if mins == 1440:
            return "D"
        if mins in (1, 5, 15, 30):
            return f"{mins}min" if mins != 1 else "min"
    except Exception:
        pass
    return None


def _extract_model_params(cfg_model: Dict[str, Any]) -> Dict[str, Any]:
    """Devuelve parámetros del modelo priorizando cfg_model['params']."""
    params = cfg_model.get("params")
    return params if isinstance(params, dict) else {}


def _extract_prophet_bt_params(cfg_model: Dict[str, Any]) -> Dict[str, Any]:
    """Devuelve cfg_model['backtest']['prophet'] si existe (parámetros prophet en el bloque del modelo)."""
    bt = cfg_model.get("backtest")
    if isinstance(bt, dict):
        pr = bt.get("prophet")
        return pr if isinstance(pr, dict) else {}
    return {}


def _get_freq_hint_from_any(
    cfg_model: Dict[str, Any],
    global_cfg: Optional[Dict[str, Any]] = None,
    train_index: Optional[pd.DatetimeIndex] = None,
) -> Optional[str]:
    """
    Obtiene una frecuencia consistente desde múltiples ubicaciones:
    (a nivel del MODELO)
      - cfg_model['params'].(frecuencia_hint|frequency_hint)
      - cfg_model['backtest']['prophet'].(frecuencia_hint|frequency_hint)
    (a nivel GLOBAL, si se provee)
      - global_cfg['eda']['frecuencia_resampleo'] in {"H","D"}
      - global_cfg['timeframe'] -> mapeo H1->"H", D1->"D", ...
    (por el índice de entrenamiento, como último recurso)
      - inferida del DatetimeIndex
    """
    # 1) Bloque del modelo
    params = _extract_model_params(cfg_model)
    freq = _get_ci(params, "frecuencia_hint", "frequency_hint")
    if freq:
        return str(freq)

    bt_prophet = _extract_prophet_bt_params(cfg_model)
    freq = _get_ci(bt_prophet, "frecuencia_hint", "frequency_hint")
    if freq:
        return str(freq)

    # 2) Desde config global (si el llamador la inyecta en cfg_model['__global__'])
    g = global_cfg or cfg_model.get("__global__")
    if isinstance(g, dict):
        eda = g.get("eda") or {}
        f = _get_ci(eda, "frecuencia_resampleo")
        if f and str(f).strip().upper() in {"H", "D"}:
            return str(f).strip().upper()
        tf = g.get("timeframe")
        mapped = _map_timeframe_to_freq(tf)
        if mapped:
            return mapped

    # 3) Inferir del índice de entrenamiento (robusto anti-warning)
    if train_index is not None:
        f = _infer_freq_from_index(train_index)
        if f:
            return f

    return None


# =========================================
# Wrappers para adapters "funcionales"
# =========================================
class ArimaFuncAdapter:
    """
    Envuelve ARIMA/SARIMA implementado como funciones en modelos/arima/adapter.py:
      - entrenar_modelo_arima(df, modo, order, seasonal_order, enforce_stationarity, enforce_invertibility)
      - predecir_precio_arima(state, pasos)
    """
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.state = None
        self._train_fn = None
        self._predict_fn = None

    def _ensure_imports(self) -> None:
        if self._train_fn is None or self._predict_fn is None:
            mod = importlib.import_module("modelos.arima.adapter")
            self._train_fn = getattr(mod, "entrenar_modelo_arima")
            self._predict_fn = getattr(mod, "predecir_precio_arima")

    def fit(self, series: pd.Series) -> None:
        self._ensure_imports()
        params = _extract_model_params(self.cfg)
        modo = self.cfg.get("objetivo", "retornos")
        df = pd.DataFrame({"ds": series.index, "y": series.values})

        self.state = _call_with_supported(
            self._train_fn,
            df=df,
            modo=modo,
            order=params.get("order", [1, 1, 1]),
            seasonal_order=params.get("seasonal_order", [0, 0, 0, 0]),
            enforce_stationarity=bool(params.get("enforce_stationarity", False)),
            enforce_invertibility=bool(params.get("enforce_invertibility", False)),
        )

    def predict(
        self,
        horizon: int,
        last_window: Optional[pd.Series] = None,
        last_timestamp: Optional[pd.Timestamp] = None,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        self._ensure_imports()
        raw = self._predict_fn(self.state, pasos=int(horizon))
        return _normalize_yhat_df(raw, index)


class ProphetFuncAdapter:
    """
    Envuelve Prophet implementado como funciones en modelos/prophet/adapter.py:
      - entrenar_modelo_prophet(df, modo, frecuencia_hint, interval_width, seasonality_mode, ...)
      - predecir_precio_prophet(state, pasos, frecuencia)
    Robusto: filtra kwargs por introspección y resuelve la frecuencia desde múltiples orígenes.
    """
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.state = None
        self._train_fn = None
        self._predict_fn = None
        self._train_freq_hint: Optional[str] = None  # inferida en fit si es necesario

    def _ensure_imports(self) -> None:
        if self._train_fn is None or self._predict_fn is None:
            mod = importlib.import_module("modelos.prophet.adapter")
            self._train_fn = getattr(mod, "entrenar_modelo_prophet")
            self._predict_fn = getattr(mod, "predecir_precio_prophet")

    def fit(self, series: pd.Series) -> None:
        self._ensure_imports()
        params = _extract_model_params(self.cfg)
        modo = self.cfg.get("objetivo", "retornos")
        df = pd.DataFrame({"ds": series.index, "y": series.values})

        # Asegurar un DatetimeIndex para inferencia robusta si hace falta
        train_idx = _ensure_dt_index(df["ds"])

        # Resolver frecuencia (modelo, backtest del modelo, global, o inferida del índice)
        freq_hint = _get_freq_hint_from_any(
            cfg_model=self.cfg,
            global_cfg=self.cfg.get("__global__"),
            train_index=train_idx,
        )
        self._train_freq_hint = freq_hint

        train_kwargs = dict(
            df=df,
            modo=modo,
            frecuencia_hint=freq_hint,
            interval_width=params.get("interval_width", 0.8),
            seasonality_mode=params.get("seasonality_mode", "additive"),
            changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
            yearly_seasonality=params.get("yearly_seasonality", True),
            weekly_seasonality=params.get("weekly_seasonality", False),
            daily_seasonality=params.get("daily_seasonality", False),
        )
        self.state = _call_with_supported(self._train_fn, **train_kwargs)

    def predict(
        self,
        horizon: int,
        last_window: Optional[pd.Series] = None,
        last_timestamp: Optional[pd.Timestamp] = None,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        self._ensure_imports()

        # Reintenta obtener la frecuencia (params/backtest del modelo, global, índice de la última ventana)
        last_idx = None
        if isinstance(last_window, pd.Series) and isinstance(last_window.index, pd.DatetimeIndex):
            last_idx = _ensure_dt_index(last_window.index)

        freq = _get_freq_hint_from_any(
            cfg_model=self.cfg,
            global_cfg=self.cfg.get("__global__"),
            train_index=last_idx,
        )
        if not freq:
            # fallback final: lo que se resolvió en fit()
            freq = self._train_freq_hint

        raw = _call_with_supported(
            self._predict_fn,
            state=self.state,
            pasos=int(horizon),
            frecuencia=freq,
        )
        return _normalize_yhat_df(raw, index)


# ==============================
# Fábrica principal
# ==============================
def get_model(nombre: str, cfg: Dict[str, Any]) -> ModelLike:
    """
    Devuelve un modelo con interfaz unificada .fit/.predict.
      - "ARIMA" / "SARIMA" -> adapter funcional (ArimaFuncAdapter)
      - "PROPHET"          -> adapter funcional (ProphetFuncAdapter)
      - "LSTM"             -> clase directa (modelos.lstm_model.LSTMModel)
    """
    name = (nombre or "").strip().upper()

    if name in {"ARIMA", "SARIMA"}:
        return ArimaFuncAdapter(cfg)

    if name == "PROPHET":
        return ProphetFuncAdapter(cfg)

    if name == "LSTM":
        mod = importlib.import_module("modelos.lstm_model")
        LSTMModel = getattr(mod, "LSTMModel")
        return LSTMModel(cfg.get("params") or {}, cfg)

    raise ValueError(f"Modelo no soportado: {nombre!r}")
