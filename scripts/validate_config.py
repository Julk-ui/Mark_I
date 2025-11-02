
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_config.py
------------------
Valida y hace un "dry-run" del archivo config.yaml para el proyecto Mark_I.

NOVEDAD:
- --preview  : intenta conectarse a MT5 (usando conexion.easy_Trading.Basic_funcs),
               descarga datos según config (simbolo, timeframe, cantidad_datos)
               y muestra un PREVIEW de la serie (head/tail, rango de fechas, dtypes,
               nulos, y un resample a la frecuencia del EDA si aplica).
- --rows N   : número de filas a mostrar en head/tail (por defecto 5).

Uso:
    python validate_config.py [ruta_config] [--preview] [--rows 8]
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse


# Asegura que el root del repo (carpeta padre de 'scripts/') esté en sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _exit(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)

def _warn(msg: str):
    print(f"⚠️  {msg}")

def _ok(msg: str):
    print(f"✅ {msg}")

def load_yaml(path: Path):
    try:
        import yaml
    except Exception:
        _exit("PyYAML no está instalado. Instala con: pip install pyyaml")
    if not path.exists():
        _exit(f"No se encontró el archivo: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except Exception as e:
            _exit(f"No se pudo parsear YAML: {e}")

def norm(s: str | None, default: str = "") -> str:
    return str(s or default).strip()

def to_bool(x, default=False):
    if isinstance(x, bool): return x
    if x is None: return default
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","t","si","sí"}

def show_section(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def guess_engine(model_name: str) -> str:
    m = model_name.lower()
    return "classic_auto" if m in {"arima", "sarima"} else "model"

def obtener_df_desde_mt5(bf, symbol: str, timeframe: str, n_barras: int):
    """Descarga datos OHLC desde MT5 con Basic_funcs.get_data_for_bt y estandariza nombres."""
    df = bf.get_data_for_bt(timeframe, symbol, n_barras)
    # Normalizar nombres
    cols_map = {"open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"TickVolume","real_volume":"Volume","time":"Date"}
    for k,v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    if "Date" in df.columns:
        df = df.set_index("Date")
    return df.sort_index()

def _find_close(df):
    return "Close" if "Close" in df.columns else (df.columns[-1] if len(df.columns) else None)

def _resample_ohlc(df, freq="D", price_col="Close"):
    """Resample simple; si hay OHLC usa .ohlc(), si no, usa mean sobre la serie."""
    try:
        if all(c in df.columns for c in ["Open","High","Low","Close"]):
            ohlc = df[["Open","High","Low","Close"]].resample(freq).agg({"Open":"first","High":"max","Low":"min","Close":"last"})
            others = [c for c in df.columns if c not in ["Open","High","Low","Close"]]
            if others:
                agg = {c:"sum" for c in others}
                ohlc[others] = df[others].resample(freq).agg(agg)
            return ohlc.dropna(how="all")
        else:
            s = df[price_col].resample(freq).mean()
            return s.to_frame(price_col).dropna(how="all")
    except Exception:
        return df

def preview_mt5(cfg: dict, rows: int = 5):
    """Intenta conectar, descargar y mostrar un preview de datos MT5 según la config."""
    show_section("PREVIEW MT5 (opcional)")
    # << NUEVO: asegurar sys.path y probar imports alternos >>
    try:
        from conexion.easy_Trading import Basic_funcs
    except Exception:
        try:
            from conexion.easy_Trading import Basic_funcs  # por si está bajo app/
        except Exception as e:
            _warn(f"No se pudo importar Basic_funcs desde conexion.easy_Trading/app.conexion ({e}). "
                  "Asegura que la carpeta 'conexion/' está en la raíz del repo y contiene __init__.py. "
                  "También puedes ejecutar el script desde la raíz: 'python scripts/validate_config.py --preview'.")
            return
    try:
        from conexion.easy_Trading import Basic_funcs
    except Exception as e:
        _warn(f"No se pudo importar Basic_funcs desde conexion.easy_Trading ({e}). Salteando preview MT5.")
        return

    mt5 = cfg.get("mt5", {})
    need = ["login","password","server","path"]
    missing = [k for k in need if k not in mt5]
    if missing:
        _warn(f"Faltan credenciales MT5 para preview: {missing}")
        return

    sim = norm(cfg.get("simbolo","EURUSD"))
    tframe = norm(cfg.get("timeframe","H1"))
    qty = int(cfg.get("cantidad_datos", 3000))

    try:
        bf = Basic_funcs(mt5.get("login"), mt5.get("password"), mt5.get("server"), mt5.get("path"))
        print("🔌 Conectado a MT5 (Basic_funcs). Descargando datos...")
        df = obtener_df_desde_mt5(bf, sim, tframe, qty)
    except Exception as e:
        _warn(f"No se pudieron obtener datos desde MT5: {e}")
        return

    if df is None or df.empty:
        _warn("DF vacío recibido de MT5.")
        return

    # Info básica
    print(f"Shape: {df.shape}")
    print(f"Rango de fechas: {df.index.min()}  →  {df.index.max()}")
    print("Columnas:", list(df.columns))
    print("\nHEAD:")
    print(df.head(rows))
    print("\nTAIL:")
    print(df.tail(rows))

    # Dtypes y nulos
    print("\nDTYPES:")
    try:
        print(df.dtypes)
    except Exception:
        pass
    print("\nNULOS POR COLUMNA:")
    try:
        print(df.isna().sum())
    except Exception:
        pass

    # Resample según EDA
    freq = norm(cfg.get("eda", {}).get("frecuencia_resampleo","H")).upper()
    price_col = _find_close(df) or "Close"
    print(f"\nResample a frecuencia EDA ({freq}) para {price_col} / OHLC si disponible:")
    df_r = _resample_ohlc(df, freq="H" if freq.startswith("H") else ("D" if freq.startswith("D") else freq), price_col=price_col)
    print(df_r.head(rows))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml", help="Ruta a config.yaml")
    ap.add_argument("--preview", action="store_true", help="Intenta conexión y muestra preview MT5.")
    ap.add_argument("--rows", type=int, default=5, help="Filas a mostrar en head/tail.")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    # --- Reutilizamos la lógica del validador original ---
    # Para mantener una única base de verdad, importamos y ejecutamos el bloque original
    # desde una función interna (copiamos el cuerpo anterior para no depender de import relativo).
    # Nota: Hemos mantenido este script auto-contenido para que puedas moverlo donde prefieras.

    # -------- General --------
    sim = norm(cfg.get("simbolo","EURUSD"))
    tframe = norm(cfg.get("timeframe","H1"))
    qty = int(cfg.get("cantidad_datos", 3000))
    spy = cfg.get("simbolo_spy")
    show_section("GENERAL")
    print(f"Símbolo principal: {sim}")
    print(f"Timeframe MT5    : {tframe}")
    print(f"Barras a descargar: {qty}")
    if spy:
        print(f"Símbolo exógeno  : {norm(spy)}")
    mt5 = cfg.get("mt5", {})
    need = ["login","password","server","path"]
    missing = [k for k in need if k not in mt5]
    if missing:
        _warn(f"Faltan credenciales MT5: {missing}")
    else:
        _ok("Credenciales MT5 configuradas.")

    # -------- EDA --------
    eda = cfg.get("eda", {})
    freq = norm(eda.get("frecuencia_resampleo","H")).upper()
    show_section("EDA / FRECUENCIA")
    print(f"Frecuencia de resampleo: {freq}")
    if freq[0] not in {"H","D","M"}:
        _warn("frecuencia_resampleo recomendada: 'H', 'D' o 'M'.")
    if tframe.startswith("H") and freq[0] != "H":
        _warn("Trabajas intradía (timeframe=H*). Considera EDA.frecuencia_resampleo='H' para coherencia.")
    if tframe == "D1" and freq[0] != "D":
        _warn("Trabajas diario (timeframe=D1). Considera EDA.frecuencia_resampleo='D'.")

    # -------- Modelo (modo normal) --------
    modelo = cfg.get("modelo", {})
    nombre_raw = norm(modelo.get("nombre","ARIMA"))
    nombre = nombre_raw.lower()
    objetivo = norm(modelo.get("objetivo","retornos")).lower()
    horizonte = int(modelo.get("horizonte", 1))
    params = modelo.get("params", {})

    show_section("MODO NORMAL – MODELO")
    print(f"Modelo seleccionado: {nombre_raw}")
    print(f"Objetivo           : {objetivo} (-> {'returns' if objetivo=='retornos' else 'close'})")
    print(f"Horizonte (pasos)  : {horizonte}")

    if nombre not in {"arima","sarima","prophet","lstm"}:
        _warn(f"Modelo '{nombre_raw}' no reconocido por el registry recomendado.")
    if nombre in {"arima","sarima"}:
        arima = {k: params.get(k) for k in ["order","seasonal_order","enforce_stationarity","enforce_invertibility"]}
        print(f"Parámetros ARIMA/SARIMA: {arima}")
    elif nombre == "prophet":
        phints = {k: params.get(k) for k in ["frecuencia_hint","interval_width","seasonality_mode","changepoint_prior_scale"]}
        print(f"Parámetros Prophet: {phints}")
    elif nombre == "lstm":
        lstm_keys = ["window","units","dropout","epochs","batch_size","lr","loss","optimizer","scaler","patience","model_dir"]
        lstm = {k: params.get(k) for k in lstm_keys if k in params}
        print(f"Parámetros LSTM: {lstm or '(usará valores por defecto)'}")

    # -------- Backtest --------
    bt = cfg.get("bt", {})
    show_section("BACKTEST – AJUSTES GENERALES")
    initial_train = int(bt.get("initial_train", 1500))
    step = int(bt.get("step", 10))
    bh = int(bt.get("horizon", 1))
    btarget = norm(bt.get("target","returns")).lower()
    pip = float(bt.get("pip_size", 0.0001))
    engine = norm(bt.get("engine")) or guess_engine(nombre)
    print(f"Motor de backtest : {engine} (auto={guess_engine(nombre)})")
    print(f"initial_train     : {initial_train}")
    print(f"step              : {step}")
    print(f"horizon           : {bh}")
    print(f"target            : {btarget}")
    print(f"pip_size          : {pip}")

    # Umbral
    print("\n— UMBRAL DE SEÑAL —")
    thr_mode = norm(bt.get("threshold_mode","garch")).lower()
    thr_pips = float(bt.get("threshold_pips", 12.0))
    atr_win = int(bt.get("atr_window", 14))
    atr_k  = float(bt.get("atr_k", 0.60))
    garch_k = float(bt.get("garch_k", 0.60))
    min_thr = float(bt.get("min_threshold_pips", 10.0))
    log_thr = to_bool(bt.get("log_threshold_used", False))
    print(f"threshold_mode    : {thr_mode}")
    print(f"threshold_pips    : {thr_pips}")
    print(f"ATR (win, k)      : ({atr_win}, {atr_k})")
    print(f"GARCH k           : {garch_k}")
    print(f"min_threshold_pips: {min_thr}")
    print(f"log_threshold_used: {log_thr}")

    # Auto-scan ARIMA
    auto = bt.get("auto", {})
    print("\n— AUTO-MODELO (ARIMA/SARIMA) —")
    print(f"scan               : {auto.get('scan', {})}")
    print(f"rescan_each_refit  : {to_bool(auto.get('rescan_each_refit', False))}")
    print(f"rescan_every_refits: {int(auto.get('rescan_every_refits', 25))}")

    # Exógenas
    exog = bt.get("exog", {})
    print("\n— EXÓGENAS —")
    print(f"enable: {to_bool(exog.get('enable', False))}")
    if to_bool(exog.get("enable", False)) and not cfg.get("simbolo_spy"):
        _warn("bt.exog.enable=true pero no hay simbolo_spy configurado en la raíz.")

    # Salidas
    print("\n— SALIDAS —")
    print(f"outxlsx     : {bt.get('outxlsx','outputs/evaluacion.xlsx')}")
    print(f"outdir_plots: {bt.get('outdir_plots','outputs/backtest_plots')}")

    # Modelos para engine=model
    print("\n— ENGINE='model' (PROPHET/LSTM) —")
    bt_prophet = bt.get("prophet", {})
    bt_lstm = bt.get("lstm", {})
    if bt_prophet:
        print(f"bt.prophet : { {k: bt_prophet.get(k) for k in ['changepoint_prior_scale','seasonality_mode','yearly_seasonality','weekly_seasonality','daily_seasonality','interval_width','frequency_hint']} }")
    else:
        print("bt.prophet : (no definido)")
    if bt_lstm:
        print(f"bt.lstm    : { {k: bt_lstm.get(k) for k in ['window','units','dropout','horizon','epochs','batch_size','lr','loss','optimizer','scaler','patience','model_dir']} }")
    else:
        print("bt.lstm    : (no definido)")

    # -------- Consistencias & Recomendaciones --------
    show_section("CHEQUEOS RÁPIDOS")
    # 1) Objetivo coherente
    if objetivo not in {"retornos","nivel"}:
        _warn("modelo.objetivo debe ser 'retornos' o 'nivel'. Usando 'retornos' por defecto en el pipeline de clases.")
    # 2) Deducción de engine
    if engine == "classic_auto" and nombre in {"prophet","lstm"}:
        _warn("Seleccionaste engine clásico pero el modelo es PROPHET/LSTM. El sistema usará el motor 'model' si detecta incompatibilidad.")
    if engine == "model" and nombre in {"arima","sarima"}:
        _warn("Seleccionaste engine='model' pero el modelo es ARIMA/SARIMA. Considera engine='classic_auto' para métricas PnL/plots tradicionales.")
    # 3) Frecuencia
    if freq.startswith("H") and tframe == "D1":
        _warn("D1 con frecuencia H puede desalinear índices de fechas. Evalúa usar EDA.frecuencia_resampleo='D'.")
    # 4) LSTM horizon alignment
    if nombre == "lstm":
        lstm_h = (params.get("horizon") if isinstance(params, dict) else None)
        if lstm_h is not None and int(lstm_h) != horizonte:
            _warn(f"LSTM en modo normal: modelo.params.horizon={lstm_h} difiere de modelo.horizonte={horizonte}.")
    if engine == "model" and bt_lstm:
        if "horizon" in bt_lstm and int(bt_lstm["horizon"]) != bh:
            _warn(f"LSTM en backtest: bt.lstm.horizon={bt_lstm['horizon']} difiere de bt.horizon={bh}.")
        else:
            _ok("bt.lstm.horizon coincide (o no especificado) con bt.horizon.")
    # 5) Prophet: frecuencia hint (intradia)
    if (nombre == "prophet" or bt_prophet):
        # ¿hay hint en modelo o en bt?
        has_model_hint = isinstance(params, dict) and bool(params.get("frecuencia_hint"))
        has_bt_hint    = isinstance(bt_prophet, dict) and bool(bt_prophet.get("frequency_hint"))
        if freq.startswith("H") and not (has_model_hint or has_bt_hint):
            _warn("Prophet en intradía: considera establecer 'frequency_hint=\"H\"' "
                "(en modelo.params.frecuencia_hint o en bt.prophet.frequency_hint).")
    # 6) GARCH disponible
    if thr_mode == "garch":
        try:
            import arch  # noqa
        except Exception:
            _warn("threshold_mode='garch' pero el paquete 'arch' no está instalado. Usa 'atr' o 'fixed' o instala: pip install arch")

    _ok("Validación completada.")

    # Preview MT5 (opcional)
    if args.preview:
        preview_mt5(cfg, rows=args.rows)

    print("\nSiguientes pasos sugeridos:")
    print("  python -m app.main --modo normal   --config", args.config)
    print("  python -m app.main --modo backtest --config", args.config)

if __name__ == "__main__":
    main()
