import pandas as pd

def predict_random_walk(price: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Baseline: ŷ_{t+h} = y_t
    Compatible con intradía y diario.
    """
    pred = price.shift(horizon)
    pred.name = "rw"
    return pred
