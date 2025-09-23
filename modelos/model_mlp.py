# modelos/model_mlp.py
# Wrapper del MLP para evaluacion_modelos.compute_metrics_prophet
# Delega en modelos.mlp_adapter (entrenar_modelo_mlp / predecir_precio_mlp)
# y asegura firmas compatibles.

from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Optional

# Importa el adapter
from modelos.mlp_adapter import entrenar_modelo_mlp, predecir_precio_mlp

# === Firmas requeridas por compute_metrics_prophet ===
def entrenar_modelo(df_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Entrena el MLP con parámetros por defecto (ajusta si tu serie lo requiere).
    Debe devolver un dict 'modelo' que luego use predecir_precio(modelo,...).
    """
    modelo = entrenar_modelo_mlp(
        df=df_train,
        modo='nivel',                 # 'nivel' o 'retornos'
        lookback=20,                  # sube/baja según datos
        scaler_type='minmax',         # 'standard' si ya está centrada/esc.
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha_reg=1e-4,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )
    return modelo

def predecir_precio(modelo: Dict[str, Any], pasos: int, frecuencia: Optional[str]) -> pd.DataFrame:
    """
    Devuelve DataFrame estándar con columnas:
    ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado'].
    """
    return predecir_precio_mlp(
        modelo=modelo,
        pasos=pasos,
        frecuencia=frecuencia,
        alpha=0.10,  # 90% aprox
    )

