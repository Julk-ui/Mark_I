# modelos/evaluacion_modelos.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics_prophet(
    df_indicadores: pd.DataFrame,
    predicciones_live: pd.DataFrame,
    pasos_pred: int,
    frecuencia_pred: str,
    simbolo: str,
    timeframe_str: str,
    modelo_str: str,
    entrenar_fn,      # callable(df_train) -> modelo
    predecir_fn       # callable(modelo, pasos, frecuencia) -> df_pred
) -> dict:
    """
    Backtest simple: entrena con todo menos los últimos 'pasos_pred' y predice esos pasos
    para comparar vs valores reales. Calcula métricas + horizonte (desde predicciones live).
    """

    # >>> Aseguramos columna Close y orden temporal (por si viene desordenado)
    assert 'Close' in df_indicadores.columns, "df_indicadores debe contener la columna 'Close'"
    if not df_indicadores.index.is_monotonic_increasing:
        df_indicadores = df_indicadores.sort_index()

    # >>> Validaciones de tamaño
    pasos_pred = int(pasos_pred)
    assert pasos_pred > 0, "pasos_pred debe ser > 0"
    assert len(df_indicadores) > pasos_pred, "df_indicadores debe tener más filas que pasos_pred"

    # Split
    df_train = df_indicadores.iloc[:-pasos_pred].copy()
    df_test  = df_indicadores.iloc[-pasos_pred:].copy()

    # Entrenar & predecir backtest
    modelo_bt = entrenar_fn(df_train)
    preds_bt  = predecir_fn(modelo_bt, pasos=pasos_pred, frecuencia=frecuencia_pred)

    # >>> Validación de salida de predicción
    assert isinstance(preds_bt, pd.DataFrame), "predecir_fn debe devolver un DataFrame"
    assert 'precio_estimado' in preds_bt.columns, "preds_bt debe tener la columna 'precio_estimado'"
    assert len(preds_bt) >= len(df_test), "preds_bt debe tener al menos pasos_pred filas"

    # Alinear tamaños
    y_true = df_test['Close'].astype(float).values
    y_pred = preds_bt['precio_estimado'].astype(float).values[:len(y_true)]

    # Errores (métricas)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # >>> MAPE robusto a ceros en y_true
    eps = 1e-12
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    r2   = r2_score(y_true, y_pred)

    # Sortino (retornos de predicción vs real)
    returns = (y_pred - y_true) / np.where(np.abs(y_true) < eps, eps, y_true)
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else np.nan
    risk_free = 0.0
    sortino = float((np.mean(returns) - risk_free) / downside_std) if (downside_std not in [0.0, np.nan]) else np.nan

    # Accuracy direccional (robusto a series cortas)
    if len(y_true) >= 2 and len(y_pred) >= 2:
        dir_real = np.sign(np.diff(y_true))
        dir_pred = np.sign(np.diff(y_pred))
        # >>> Igualar longitudes por seguridad
        n = min(len(dir_real), len(dir_pred))
        aciertos = int(np.sum(dir_real[:n] == dir_pred[:n]))
        total    = int(n)
        accuracy_dir = float(aciertos / total) if total > 0 else np.nan
    else:
        accuracy_dir = np.nan

    # Horizonte (desde las predicciones live del bot)
    # >>> Robusto a DataFrame vacío o sin columna esperada
    if isinstance(predicciones_live, pd.DataFrame) and 'timestamp_prediccion' in predicciones_live.columns and len(predicciones_live) > 0:
        tmax = predicciones_live['timestamp_prediccion'].max()
        tmin = predicciones_live['timestamp_prediccion'].min()
        if pd.isna(tmax) or pd.isna(tmin):
            horizonte_dias = 0
            horizonte_horas_totales = 0.0
        else:
            horizonte = tmax - tmin
            horizonte_dias = int(getattr(horizonte, 'days', 0))
            horizonte_horas_totales = float(getattr(horizonte, 'total_seconds', lambda: 0)() / 3600)
    else:
        horizonte_dias = 0
        horizonte_horas_totales = 0.0

    return {
        'Fecha': pd.Timestamp.now(),
        'Simbolo': simbolo,
        'Timeframe': timeframe_str,
        'Modelo': modelo_str,
        'Pasos_pred': int(pasos_pred),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE_%': float(mape),
        'R2': float(r2),
        'Sortino': float(sortino) if not np.isnan(sortino) else np.nan,
        'Accuracy_direccional': float(accuracy_dir) if not np.isnan(accuracy_dir) else np.nan,
        'Horizonte_dias': int(horizonte_dias),
        'Horizonte_horas_totales': float(horizonte_horas_totales)
    }
