import numpy as np
import pandas as pd
from arch import arch_model

def fit_garch(ret: pd.Series, dist: str = "StudentsT"):
    """
    Ajusta GARCH(1,1) a retornos (log-returns), usando % internos del paquete.
    """
    am = arch_model(ret.dropna().values * 100,
                    p=1, q=1, mean="Constant", vol="GARCH", dist=dist)
    res = am.fit(disp="off")
    return res

def garch_sigma_series(res, index: pd.DatetimeIndex, horizon: int = 1) -> pd.Series:
    """
    Devuelve sigma (desviación condicional) alineada al índice solicitado.
    """
    f = res.forecast(horizon=horizon, reindex=True)
    sigma2 = f.variance.iloc[:, -1] / (100**2)  # vuelve de %^2 a unidades de retorno
    sigma = sigma2.pow(0.5).rename("garch_sigma").reindex(index)
    return sigma
