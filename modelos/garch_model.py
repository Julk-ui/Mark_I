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
    Retorna sigma_t (desviación condicional) alineada al índice 'index'.
    Estrategia robusta:
      1) Intentamos usar el forecast out-of-sample h=1 y reindexarlo.
      2) Si sale vacío/NaN (por desalineación), usamos la sigma condicional in-sample
         desplazada 1 (info disponible t-1) y la reindexamos al 'index'.
    Todas en unidades de retorno (no %).
    """
    try:
        f = res.forecast(horizon=1, reindex=True)
        # 'variance' suele traer columnas tipo 'h.1', o última columna es h=1
        var_df = f.variance
        if isinstance(var_df, pd.DataFrame):
            if "h.1" in var_df.columns:
                sigma2 = var_df["h.1"]
            else:
                sigma2 = var_df.iloc[:, -1]
        else:
            sigma2 = var_df
        sigma = (sigma2 / (100.0**2)).pow(0.5)  # vuelve de %^2 a unidades
        gsig = sigma.reindex(index)
        # Si quedó todo NaN, hacemos fallback
        if gsig.isna().all():
            raise RuntimeError("Forecast GARCH desalineado; usando conditional_volatility shift(1).")
        return gsig.rename("garch_sigma")
    except Exception:
        # Fallback robusto: conditional_volatility ya alineada al índice del ajuste
        cs = getattr(res, "conditional_volatility", None)
        if cs is None or len(cs) == 0:
            # último recurso: serie vacía
            return pd.Series(index=index, dtype=float, name="garch_sigma")
        # conditional_volatility está en %, pasamos a unidades y shift(1)
        cs = pd.Series(cs, index=getattr(cs, "index", None)).astype(float)
        gsig = (cs.shift(1) / 100.0).reindex(index)
        return gsig.rename("garch_sigma")
