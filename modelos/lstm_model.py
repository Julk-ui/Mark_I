# modelos/lstm_model.py
# Wrapper del LSTM para evaluacion_modelos.compute_metrics_prophet
# Delega en modelos.lstm_adapter (entrenar_modelo_lstm / predecir_precio_lstm)
# y asegura firmas compatibles.

from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Optional

# IMPORTA el adapter nuevo:
from modelos.lstm_adapter import entrenar_modelo_lstm, predecir_precio_lstm

# === Funciones que exige compute_metrics_prophet ===
def entrenar_modelo(df_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Entrena el LSTM con parámetros por defecto (ajústalos a tu caso).
    Debe devolver un dict 'modelo' que luego use predecir_precio(modelo,...).
    """
    # Puedes cambiar lookback/epochs/units según tu serie:
    modelo = entrenar_modelo_lstm(
        df=df_train,
        modo='nivel',           # 'nivel' o 'retornos'. Si usas retornos, ver nota abajo.
        lookback=40,
        epochs=40,
        batch_size=32,
        validation_split=0.1,   # si tienes pocos datos, baja a 0.05 o 0.0
        patience=6,
        units=64,
        dense_units=32,
        dropout=0.0,
        scaler_type='minmax',   # 'standard' si tu serie es ya estacionaria
    )
    return modelo

def predecir_precio(modelo: Dict[str, Any], pasos: int, frecuencia: Optional[str]) -> pd.DataFrame:
    """
    Predice y devuelve el DataFrame estándar con columnas:
    ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado'].
    """
    return predecir_precio_lstm(
        modelo=modelo,
        pasos=pasos,
        frecuencia=frecuencia,
        alpha=0.10,  # 90% aprox
    )
