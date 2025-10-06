# modelos/sarima_scan.py
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _adf_p(s: pd.Series) -> float:
    s = pd.Series(s).dropna()
    if s.empty:
        return 1.0
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return 1.0

def infer_seasonal_period(index: pd.Index) -> list[int]:
    # Diario: probar semana hábil (5) y semana calendario (7)
    try:
        idx = pd.to_datetime(index)
        if len(idx) >= 3:
            return [5, 7]
    except Exception:
        pass
    return [5, 7]

def sugerir_D(y_log: pd.Series, s: int) -> int:
    p0 = _adf_p(y_log)
    p1 = _adf_p(y_log.diff(s))
    return 1 if (p1 < 0.1 and p1 < p0) else 0

def escanear_sarima(y: pd.Series, s_candidates: list[int] | None = None,
                    max_p: int = 2, max_q: int = 2,
                    max_P: int = 1, max_Q: int = 1) -> pd.DataFrame:
    y = pd.Series(y).astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if y.empty:
        return pd.DataFrame(columns=["p","d","q","P","D","Q","s","bic","aic","lb_p"])
    y_log = np.log(y).replace([np.inf,-np.inf], np.nan).dropna()
    if s_candidates is None or not len(s_candidates):
        s_candidates = infer_seasonal_period(y.index)

    resultados = []
    for s in s_candidates:
        D = sugerir_D(y_log, s)
        d = 0  # tendencia se maneja en la parte no estacional
        target = y_log.diff(s).dropna() if D > 0 else y_log.copy()
        for p in range(max_p+1):
            for q in range(max_q+1):
                for P in range(max_P+1):
                    for Q in range(max_Q+1):
                        if p==q==P==Q==0 and D==0:
                            continue
                        try:
                            model = SARIMAX(target, order=(p,d,q),
                                            seasonal_order=(P,D,Q,s),
                                            trend=None,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False).fit(disp=False)
                            resid = pd.Series(getattr(model, "resid", pd.Series(index=target.index))).dropna()
                            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
                            lb_p = float(lb["lb_pvalue"].iloc[0]) if not lb.empty else np.nan
                            resultados.append({
                                "p":p,"d":d,"q":q,"P":P,"D":D,"Q":Q,"s":s,
                                "bic": float(model.bic), "aic": float(model.aic), "lb_p": lb_p
                            })
                        except Exception:
                            continue
    if not resultados:
        return pd.DataFrame(columns=["p","d","q","P","D","Q","s","bic","aic","lb_p"])
    res = pd.DataFrame(resultados).sort_values("bic").reset_index(drop=True)
    return res.head(10)
