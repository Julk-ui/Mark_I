# modelos/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class TrainConfig:
    """Configuración mínima común para modelos de series de tiempo."""
    symbol: str
    target_col: str = "Close"      # "Close" o "returns"
    horizon: int = 1
    freq: Optional[str] = None     # p.ej., "H", "D", etc.
    model_name: str = "BASE"
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainResult:
    """Salida estandarizada de entrenamiento/predicción."""
    metrics: Dict[str, float]        # RMSE, MAPE, R2, etc.
    pred_df: pd.DataFrame            # índice datetime; columnas: y_true/y_pred (o *_level)
    config: TrainConfig
    model_obj: Optional[Any] = None  # opcional: objeto del modelo (statsmodels, prophet, etc.)
