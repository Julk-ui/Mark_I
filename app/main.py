import os
import sys
import argparse
import yaml
import pandas as pd
import MetaTrader5 as mt5

from procesamiento.eda_crispdm import ejecutar_eda
from conexion.easy_Trading import Basic_funcs
from procesamiento.features import aplicar_todos_los_indicadores
from modelos.prophet_model import entrenar_modelo_prophet, predecir_precio
from agentes.agente_analisis import generar_senal_operativa
from agentes.agente_portafolio import asignar_capital
from agentes.agente_ejecucion import ejecutar_operacion, generar_reporte_excel
from modelos.evaluacion_modelos import compute_metrics_prophet
from reportes.reportes_excel import write_metrics_sheet, append_history


def obtener_df_desde_mt5(symbol: str, timeframe, n_barras: int) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: timestamp, Open, High, Low, Close, Volume (UTC)."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(n_barras))
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No se obtuvieron datos de {symbol} desde MT5.")
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"
    })
    return df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]


# =========================
# 1) CARGA DE CONFIGURACIÓN
# =========================
with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parámetros base
simbolo = config["simbolo"]
timeframe_str = config["timeframe"]
cantidad = config["cantidad_datos"]
modelo_str = config["modelo"]
pasos_pred = config["pasos_prediccion"]
frecuencia_pred = config["frecuencia_prediccion"]
umbral_senal = config["umbral_senal"]

# Nuevos parámetros YAML
riesgo_por_trade = config.get("riesgo_por_trade", 0.02)        # 2% por defecto
volumen_minimo   = float(config.get("volumen_minimo", 0.01))   # 0.01 lotes por defecto
stop_loss_pips   = float(config.get("stop_loss_pips", 10))     # 10 pips
take_profit_pips = float(config.get("take_profit_pips", 20))   # 20 pips
ruta_reporte     = config.get("ruta_reporte", "outputs/reporte_inversion.xlsx")
pip_size_cfg     = config.get("pip_size", None)                # opcional, ej. 0.0001 en EURUSD

# Mapear timeframe
timeframes = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'D1': mt5.TIMEFRAME_D1,
}
if timeframe_str not in timeframes:
    raise ValueError(f"Timeframe '{timeframe_str}' no soportado. Usa uno de {list(timeframes.keys())}.")
timeframe = timeframes[timeframe_str]

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--modo", choices=["normal", "eda"], default="normal",
                    help="Ejecuta el flujo normal (trading) o solo el EDA CRISP-DM.")
parser.add_argument("--freq", default=None,
                    help="Frecuencia de resampleo para EDA (ej. D, H, 15T). Si se pasa, sobreescribe config.yaml.")
args, _ = parser.parse_known_args()
if args.freq:
    config.setdefault("eda", {})["frecuencia_resampleo"] = args.freq


# =========================
# 2) CONEXIÓN MT5
# =========================
# Recomendado: mover credenciales a variables de entorno o a config.yaml (sección 'mt5')
mt5_cfg = config.get("mt5", {})
login = int(os.getenv("MT5_LOGIN", mt5_cfg.get("login", 68238343)))
clave = os.getenv("MT5_PASSWORD", mt5_cfg.get("password", "Colombia123*"))
servidor = os.getenv("MT5_SERVER", mt5_cfg.get("server", "RoboForex-Pro"))
path = os.getenv("MT5_PATH", mt5_cfg.get("path", r"C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe"))

if not mt5.initialize(login=login, password=clave, server=servidor, path=path):
    print("❌ Error al conectar a MT5:", mt5.last_error())
    sys.exit(1)
print("✅ Conexión establecida con MetaTrader 5")

try:
    # === EDA de dos activos con --modo eda ===
    if args.modo == "eda":
        # EURUSD (usa tu 'simbolo' actual o 'simbolo_eurusd' si está en config)
        simbolo_eur = config.get("simbolo_eurusd", simbolo)
        df_eur = obtener_df_desde_mt5(simbolo_eur, timeframe, cantidad)

        # SPY: intenta MT5 si tienes el símbolo; si no, usa CSV (ruta en config["spy_csv"])
        df_spy = None
        simbolo_spy = config.get("simbolo_spy")        # ej. "SPY" si tu broker lo ofrece
        ruta_spy_csv = config.get("spy_csv")           # ej. "data/spy.csv" si no está en MT5

        if simbolo_spy:
            try:
                df_spy = obtener_df_desde_mt5(simbolo_spy, timeframe, cantidad)
            except Exception as e:
                print(f"⚠️ No se pudo obtener SPY desde MT5: {e}")
        if df_spy is None and ruta_spy_csv:
            df_spy = pd.read_csv(ruta_spy_csv)
            if "timestamp" not in df_spy.columns:
                for c in ["Date", "Datetime", "date", "datetime"]:
                    if c in df_spy.columns:
                        df_spy = df_spy.rename(columns={c: "timestamp"})
                        break
            if "Close" not in df_spy.columns and "close" in df_spy.columns:
                df_spy = df_spy.rename(columns={"close": "Close"})

        # Ejecuta EDA (CRISP-DM: Understanding & Preparation)
        ejecutar_eda(df_eurusd=df_eur, df_spy=df_spy, cfg=config)
        print("✅ EDA completado (ver outputs/eda).")
        sys.exit(0)

    # Instancia de utilidades MT5
    BF = Basic_funcs(login, clave, servidor, path)

    # Tamaño de pip robusto
    info = mt5.symbol_info(simbolo)
    if info is None:
        raise RuntimeError(f"No se pudo obtener symbol_info de {simbolo}")
    point = info.point
    if pip_size_cfg is not None:
        pip = float(pip_size_cfg)
    elif info.digits in (3, 5):
        pip = point * 10  # FX con 3 ó 5 dígitos → 1 pip = 10 * point
    else:
        pip = point       # Acciones/índices u otros
    print(f"ℹ️ Símbolo={simbolo}, digits={info.digits}, point={point}, pip={pip}")

    # =========================
    # 3) EXTRACCIÓN DE DATOS
    # =========================
    print("⏳ Extrayendo datos de MT5...")
    df = BF.get_data_for_bt(timeframe, simbolo, cantidad)
    print("Última fecha en datos extraídos:", df.index.max())

    print("📈 Calculando indicadores técnicos...")
    df_indicadores = aplicar_todos_los_indicadores(df)

    # =========================
    # 3.1) EDA CRISP-DM (opcional por YAML)
    # =========================
    try:
        if config.get("eda", {}).get("habilitar", False):
            # Pasamos el DF con un 'timestamp' explícito para que el EDA sea robusto
            df_eur_eda = (
                df_indicadores
                .reset_index()
                .rename(columns={(df_indicadores.index.name or "index"): "timestamp"})
            )
            # Si no tienes SPY en este flujo, pasa None
            ejecutar_eda(df_eurusd=df_eur_eda, df_spy=None, cfg=config)
    except Exception as e:
        print(f"⚠️ EDA opcional no se ejecutó: {e}")

    # =========================
    # 4) ENTRENAR / PREDECIR
    # =========================
    if modelo_str == "prophet":
        print("🤖 Entrenando modelo Prophet...")
        modelo = entrenar_modelo_prophet(df_indicadores)

        print("🔮 Generando predicción futura...")
        predicciones = predecir_precio(modelo, pasos=pasos_pred, frecuencia=frecuencia_pred)
        print(predicciones)
    else:
        raise ValueError(f"Modelo '{modelo_str}' no implementado aún.")

    # =========================
    # 5) SEÑAL + ASIGNACIÓN DE CAPITAL
    # =========================
    senal = generar_senal_operativa(predicciones, umbral=umbral_senal)
    print(f"📢 Señal generada: {senal}")

    balance, _, _, _ = BF.info_account()
    capital = asignar_capital(balance, senal)
    print(f"💰 Capital asignado según la señal: ${capital:.2f}")

    # Simulación (para registro en reporte)
    precio_actual = (df_indicadores.get('Close', df_indicadores.get('close'))).iloc[-1]
    operacion = ejecutar_operacion(simbolo, senal, capital, precio_actual)
    print(f"🧾 Operación simulada: {operacion}")

    # =========================
    # 6) REPORTE BASE (asegura archivo)
    # =========================
    ruta_excel = ruta_reporte
    # Asegura directorio de salida antes de escribir
    os.makedirs(os.path.dirname(ruta_reporte) or ".", exist_ok=True)
    generar_reporte_excel(predicciones, senal, capital, operacion, umbral=umbral_senal, ruta=ruta_reporte)

    # =========================
    # 7) EVALUACIÓN (métricas + horizonte) MODULAR
    # =========================
    try:
        metrics = compute_metrics_prophet(
            df_indicadores=df_indicadores,
            predicciones_live=predicciones,
            pasos_pred=pasos_pred,
            frecuencia_pred=frecuencia_pred,
            simbolo=simbolo,
            timeframe_str=timeframe_str,
            modelo_str=modelo_str,
            entrenar_fn=entrenar_modelo_prophet,
            predecir_fn=predecir_precio
        )

        print(
            f"📏 MAE={metrics['MAE']:.6f}  RMSE={metrics['RMSE']:.6f}  "
            f"MAPE={metrics['MAPE_%']:.2f}%  R²={metrics['R2']:.4f}  "
            f"Sortino={metrics['Sortino']:.4f}  AccDir={metrics['Accuracy_direccional']:.2%}  "
            f"Horiz≈{metrics['Horizonte_horas_totales']:.1f}h"
        )

        # Guarda hoja de métricas de la corrida actual y actualiza histórico
        write_metrics_sheet(ruta_excel, metrics, sheet_name='Métricas Modelo')
        append_history(ruta_excel, metrics, hist_sheet='Historico Métricas')

    except Exception as e:
        print(f"⚠️ No se pudieron calcular/guardar métricas: {e}")

    # =========================
    # 8) EJECUCIÓN REAL (si aplica)
    # =========================
    if senal in ['comprar', 'vender'] and capital > 0:
        tipo_mt5 = mt5.ORDER_TYPE_BUY if senal == 'comprar' else mt5.ORDER_TYPE_SELL

        # Tamaño de posición por riesgo fijo del YAML
        distancia_sl_precio = stop_loss_pips * pip  # SL en precio (de pips a precio)
        volumen = BF.calculate_position_size(
            simbolo,
            tradeinfo=distancia_sl_precio,
            per_to_risk=riesgo_por_trade
        )

        # Volumen mínimo desde YAML
        if volumen < volumen_minimo:
            print(f"⚠️ Volumen calculado ({volumen}) < mínimo ({volumen_minimo}); ajustado.")
            volumen = volumen_minimo

        # SL / TP por señal (usando pips del YAML)
        if senal == 'comprar':
            sl = round(precio_actual - stop_loss_pips * pip, info.digits)
            tp = round(precio_actual + take_profit_pips * pip, info.digits)
        else:  # vender
            sl = round(precio_actual + stop_loss_pips * pip, info.digits)
            tp = round(precio_actual - take_profit_pips * pip, info.digits)

        print(
            f"📌 SL: {sl} | TP: {tp} | Vol: {volumen}  "
            f"(riesgo {riesgo_por_trade*100:.1f}% ; SL {stop_loss_pips} pips / TP {take_profit_pips} pips)"
        )

        BF.open_operations(
            par=simbolo,
            volumen=volumen,
            tipo_operacion=tipo_mt5,
            nombre_bot=f'Sistema Prophet - {senal.upper()}',
            sl=sl,
            tp=tp
        )
        print(f"🚀 Orden enviada a MT5: {senal.upper()} {simbolo} con {volumen} lotes")
    else:
        print("❎ No se envió operación real (señal fue 'mantener' o capital = 0)")

finally:
    # =========================
    # 9) CIERRE
    # =========================
    mt5.shutdown()
    print("🛑 Conexión cerrada")
