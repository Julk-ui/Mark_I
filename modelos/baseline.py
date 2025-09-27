# modelos/baseline.py
# Baseline "random walk": ŷ_{t+h} = y_t
# Es la referencia mínima. Un buen modelo debería mejorar a este.

import pandas as pd

def predict_random_walk(price: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Desplaza la serie 'price' hacia adelante 'horizon' pasos.
    Compatible con intradía y diario.
    """
    pred = price.shift(horizon)
    pred.name = "rw"
    return pred
