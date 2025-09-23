# procesamiento/features.py
# Indicadores técnicos y utilidades robustas (compatibles con el main original)
# - Mantiene la API: aplicar_todos_los_indicadores(df)
# - Añade API configurable: aplicar_indicadores(df, config=?, limpiar_nans=?)

from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


# =========================
# Utilidades base
# =========================
def ensure_time_index(
    df: pd.DataFrame,
    time_cols=("time", "timestamp", "date", "datetime", "Date", "Datetime")
) -> pd.DataFrame:
    """
    Asegura índice datetime creciente a partir de una columna temporal o del índice existente.
    No modifica otras columnas, solo reindexa/ordena.
    """
    d = df.copy()
    dtcol = next((c for c in time_cols if c in d.columns), None)

    if dtcol is not None:
        idx = pd.to_datetime(d[dtcol], errors="coerce", utc=False)
        d = d.loc[idx.notna()].copy()
        d.index = idx[idx.notna()]
    elif isinstance(d.index, pd.DatetimeIndex):
        d = d.copy()
    else:
        raise ValueError("Se requiere columna temporal (p. ej. 'time') o índice datetime.")

    d = d.sort_index()
    return d


def find_close(df: pd.DataFrame) -> str:
    """Busca la columna de cierre más probable."""
    for c in ["Close", "close", "Adj Close", "price", "Price"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontró columna de precio/cierre.")


# =========================
# Indicadores (robustos)
# =========================
def calcular_rsi(df: pd.DataFrame, periodo: int = 14, col: Optional[str] = None) -> pd.DataFrame:
    """
    RSI con suavizado tipo Wilder (aprox. ewm(alpha=1/periodo)) y manejo de divisiones por cero.
    """
    d = df.copy()
    col = col or find_close(d)
    delta = d[col].diff()
    up = np.clip(delta, 0, None)
    down = np.clip(-delta, 0, None)

    gain = pd.Series(up, index=d.index).ewm(alpha=1/periodo, adjust=False).mean()
    loss = pd.Series(down, index=d.index).ewm(alpha=1/periodo, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    return d


def calcular_macd(
    df: pd.DataFrame, rapida: int = 12, lenta: int = 26, signal: int = 9, col: Optional[str] = None
) -> pd.DataFrame:
    d = df.copy()
    col = col or find_close(d)
    ema_fast = d[col].ewm(span=rapida, adjust=False).mean()
    ema_slow = d[col].ewm(span=lenta, adjust=False).mean()
    d['MACD'] = ema_fast - ema_slow
    d['Signal_Line'] = d['MACD'].ewm(span=signal, adjust=False).mean()
    d['MACD_Hist'] = d['MACD'] - d['Signal_Line']
    return d


def calcular_retornos_log(df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    col = col or find_close(d)
    d['Log_Returns'] = np.log(d[col] / d[col].shift(1))
    return d


def calcular_atr(df: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
    """
    ATR(14) con TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|).
    Si faltan columnas OHLC, no modifica el DataFrame (no lanza error).
    """
    d = df.copy()
    cols = set([c.lower() for c in d.columns])
    if not {'high', 'low'}.issubset(cols):
        return d  # no hay OHLC suficiente

    # Detecta nombres reales respetando mayúsculas
    high = d[[c for c in d.columns if c.lower() == 'high'][0]].astype(float)
    low  = d[[c for c in d.columns if c.lower() == 'low'][0]].astype(float)
    close_col = next((c for c in d.columns if c.lower() in ('close', 'adj close', 'price')), None)
    if close_col is None:
        return d
    close = d[close_col].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(window=periodo).mean()
    return d


def calcular_bollinger(df: pd.DataFrame, periodo: int = 20, num_std: float = 2.0, col: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    col = col or find_close(d)
    ma = d[col].rolling(window=periodo, min_periods=max(2, periodo//3)).mean()
    std = d[col].rolling(window=periodo, min_periods=max(2, periodo//3)).std()
    d['BB_Media'] = ma
    d['BB_Upper'] = ma + num_std * std
    d['BB_Lower'] = ma - num_std * std
    return d


def calcular_momentum(df: pd.DataFrame, periodo: int = 10, col: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    col = col or find_close(d)
    d['Momentum'] = d[col] - d[col].shift(periodo)
    return d


def calcular_sma(df: pd.DataFrame, periodo: int = 20, col: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    col = col or find_close(d)
    d[f'SMA_{periodo}'] = d[col].rolling(window=periodo, min_periods=max(2, periodo//3)).mean()
    return d


def calcular_ema(df: pd.DataFrame, periodo: int = 20, col: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    col = col or find_close(d)
    d[f'EMA_{periodo}'] = d[col].ewm(span=periodo, adjust=False).mean()
    return d


def agregar_volumen(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volumen normalizado = Volume / media_rolling(Volume, window).
    Si no hay 'Volume', no modifica el DataFrame.
    """
    d = df.copy()
    if 'Volume' in d.columns:
        base = d['Volume'].rolling(window=window, min_periods=max(2, window//3)).mean()
        d['Volumen_normalizado'] = d['Volume'] / base.replace(0, np.nan)
    return d


# =========================
# Orquestador de features
# =========================
def aplicar_indicadores(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    limpiar_nans: bool = False
) -> pd.DataFrame:
    """
    Aplica una batería de indicadores de forma robusta y opcionalmente limpia NaNs iniciales.
    `config` permite desactivar/ajustar parámetros:
      {
        "rsi": {"periodo": 14}, "macd": {"rapida":12,"lenta":26,"signal":9},
        "atr": {"periodo":14}, "bollinger":{"periodo":20,"num_std":2.0},
        "momentum":{"periodo":10}, "sma":{"periodo":20}, "ema":{"periodo":20},
        "volumen":{"window":20}
      }
    """
    cfg = config or {}
    d = ensure_time_index(df)

    # En orden clásico
    if cfg.get("rsi", True):
        params = cfg.get("rsi", {}) if isinstance(cfg.get("rsi"), dict) else {}
        d = calcular_rsi(d, periodo=params.get("periodo", 14))

    if cfg.get("macd", True):
        params = cfg.get("macd", {}) if isinstance(cfg.get("macd"), dict) else {}
        d = calcular_macd(d,
                          rapida=params.get("rapida", 12),
                          lenta=params.get("lenta", 26),
                          signal=params.get("signal", 9))

    if cfg.get("log_returns", True):
        d = calcular_retornos_log(d)

    if cfg.get("atr", True):
        params = cfg.get("atr", {}) if isinstance(cfg.get("atr"), dict) else {}
        d = calcular_atr(d, periodo=params.get("periodo", 14))

    if cfg.get("bollinger", True):
        params = cfg.get("bollinger", {}) if isinstance(cfg.get("bollinger"), dict) else {}
        d = calcular_bollinger(d,
                               periodo=params.get("periodo", 20),
                               num_std=params.get("num_std", 2.0))

    if cfg.get("momentum", True):
        params = cfg.get("momentum", {}) if isinstance(cfg.get("momentum"), dict) else {}
        d = calcular_momentum(d, periodo=params.get("periodo", 10))

    if cfg.get("sma", True):
        params = cfg.get("sma", {}) if isinstance(cfg.get("sma"), dict) else {}
        d = calcular_sma(d, periodo=params.get("periodo", 20))

    if cfg.get("ema", True):
        params = cfg.get("ema", {}) if isinstance(cfg.get("ema"), dict) else {}
        d = calcular_ema(d, periodo=params.get("periodo", 20))

    if cfg.get("volumen", True):
        params = cfg.get("volumen", {}) if isinstance(cfg.get("volumen"), dict) else {}
        d = agregar_volumen(d, window=params.get("window", 20))

    if limpiar_nans:
        d = d.dropna().copy()

    return d


# =========================
# Compatibilidad con tu main.py
# =========================
def aplicar_todos_los_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper para mantener compatibilidad con el código existente.
    Aplica la batería de indicadores con parámetros por defecto.
    """
    return aplicar_indicadores(df, config=None, limpiar_nans=False)


__all__ = [
    "aplicar_todos_los_indicadores",
    "aplicar_indicadores",
    "calcular_rsi", "calcular_macd", "calcular_retornos_log",
    "calcular_atr", "calcular_bollinger", "calcular_momentum",
    "calcular_sma", "calcular_ema", "agregar_volumen",
    "ensure_time_index", "find_close",
]
