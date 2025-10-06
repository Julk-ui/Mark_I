# modelos/arima_scan.py
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def sugerir_d(series_log: pd.Series) -> int:
    s = pd.Series(series_log).dropna()
    if s.empty:
        return 0
    try:
        p = adfuller(s, autolag='AIC')[1]
        return 0 if p < 0.05 else 1
    except Exception:
        return 1

def escanear_arima(y: pd.Series, max_p: int = 3, max_q: int = 3, d: int | None = None,
                   lb_lag: int = 10) -> pd.DataFrame:
    s = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.DataFrame(columns=['p','d','q','bic','aic','lb_p']).astype({'p':int,'d':int,'q':int,'bic':float,'aic':float,'lb_p':float})
    if d is None:
        d = sugerir_d(np.log(s))
    y_log = np.log(s).replace([np.inf, -np.inf], np.nan).dropna()
    target = y_log.diff(d).dropna() if d > 0 else y_log
    resultados = []
    for p in range(max_p+1):
        for q in range(max_q+1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(target, order=(p, d, q)).fit(method_kwargs={'warn_convergence': False})
                resid = pd.Series(getattr(model, 'resid', pd.Series(index=target.index))).dropna()
                lb = acorr_ljungbox(resid, lags=[lb_lag], return_df=True)
                lb_p = float(lb['lb_pvalue'].iloc[0]) if not lb.empty else np.nan
                resultados.append({'p':int(p),'d':int(d),'q':int(q),'bic':float(model.bic),'aic':float(model.aic),'lb_p':lb_p})
            except Exception:
                continue
    if not resultados:
        return pd.DataFrame(columns=['p','d','q','bic','aic','lb_p']).astype({'p':int,'d':int,'q':int,'bic':float,'aic':float,'lb_p':float})
    res = pd.DataFrame(resultados).sort_values('bic').reset_index(drop=True)
    return res.head(10)
