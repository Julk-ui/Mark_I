# Mark_I — EDA + Forecast + Señales (EURUSD / SPY)

Pipeline completo:
1) **EDA (CRISP-DM)** con tablas y gráficos (PDF/Excel).
2) **Modelado** (Prophet) con predicciones multi-paso.
3) **Señal operativa**, **asignación de capital**, **reporte** y **(opcional) ejecución en MT5**.

La configuración se gestiona vía `utils/config.yaml`.

---

## 📦 Requisitos e instalación

### 1) Entorno
- **Python 3.10–3.11** recomendado (64-bit).
- **Windows** para integrar con **MetaTrader 5** (recomendado 64-bit).

Crear entorno virtual:

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
2) Dependencias
Actualizar pip y luego instalar:

bash
Copiar código
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Opcional (LSTM): si no usarás lstm_adapter.py, puedes quitar tensorflow-cpu del requirements.txt.

3) MetaTrader 5
Instala la terminal de MetaTrader 5 y ten a mano credenciales.
El script se conecta con los parámetros de utils/config.yaml o variables de entorno:
MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.

🗂️ Estructura (carpetas clave)
bash
Copiar código
Mark_I/
├─ app/
│  └─ main.py                 # punto de entrada
├─ procesamiento/
│  ├─ eda_crispdm.py          # EDA CRISP-DM (PDF/Excel + diag.)
│  └─ features.py             # indicadores robustos (compatibles c/ main)
├─ modelos/
│  ├─ prophet_model.py        # adapter Prophet
│  ├─ evaluacion_modelos.py   # métricas + backtest simple
│  ├─ lstm_adapter.py         # (opcional) LSTM univariante
│  └─ mlp_adapter.py          # (opcional) MLP univariante
├─ agentes/
│  ├─ agente_analisis.py
│  ├─ agente_portafolio.py
│  └─ agente_ejecucion.py
├─ conexion/
│  └─ easy_Trading.py
├─ reportes/
│  └─ reportes_excel.py
├─ utils/
│  └─ config.yaml
└─ outputs/                   # resultados
Compatibilidad mantenida: features.py conserva aplicar_todos_los_indicadores(...).

⚙️ Configuración (utils/config.yaml)
Campos principales:

simbolo: p.ej. EURUSD

timeframe: M1|M5|M15|H1|D1

cantidad_datos: velas a extraer

modelo: prophet (en main.py actual)

pasos_prediccion: horizonte (n pasos)

frecuencia_prediccion: p.ej. 15min, H, D

umbral_senal: (p.ej. 0.0003)

riesgo_por_trade: p.ej. 0.02

volumen_minimo: p.ej. 0.01

stop_loss_pips / take_profit_pips

pip_size (opcional) — fuerza tamaño de pip si el broker reporta algo inusual

ruta_reporte: p.ej. outputs/reporte_inversion.xlsx

eda:

habilitar: true|false (genera EDA en el flujo normal)

frecuencia_resampleo: D|H|15T|...

outdir: outputs/eda

export_pdf: true|false

pdf_filename: nombre del PDF

SPY (segundo activo del EDA):

simbolo_spy: si tu broker lo tiene (p.ej. SPY, US500, etc.)

spy_csv: ruta CSV alternativa si el símbolo no está en MT5
(mínimo: timestamp, Close; ideal: Open, High, Low, Close, Volume)

Credenciales MT5 (recomendado por variables de entorno):

text
Copiar código
MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
▶️ Ejecución
EDA (solo análisis)
bash
Copiar código
# Diario
python -m app.main --modo eda --freq D

# Horario
python -m app.main --modo eda --freq H

# 15 minutos
python -m app.main --modo eda --freq 15T
Genera:

PDF: outputs/eda/EDA_informe.pdf

Excel: outputs/eda/EDA_informe.xlsx

Gráficos en outputs/eda/*.png

Flujo normal (forecast + señal + reporte + métrica)
bash
Copiar código
python -m app.main
# (equivalente a --modo normal)
Entrega:

Predicciones Prophet

Señal (comprar|vender|mantener)

Asignación (según riesgo)

Reporte Excel: outputs/reporte_inversion.xlsx

Métricas del modelo (MAE, RMSE, MAPE, R², Sortino, Accuracy direccional, Horizonte)

📈 Indicadores y Features
procesamiento/features.py:

Robustez de índice temporal (ensure_time_index) y detección de columna de cierre (find_close).

RSI (suavizado tipo Wilder), MACD, LogReturns, ATR (si hay OHLC), Bollinger, Momentum, SMA/EMA, Volumen normalizado.

Orquestador:

aplicar_todos_los_indicadores(df) — compatible con tu código actual.

aplicar_indicadores(df, config=?, limpiar_nans=?) — configurable.

Ejemplo configurable:

python
Copiar código
from procesamiento.features import aplicar_indicadores
df_feat = aplicar_indicadores(df, config={
  "rsi": {"periodo": 14},
  "bollinger": {"periodo": 20, "num_std": 2.0},
  "atr": {"periodo": 14},
  "ema": False  # desactiva EMA si no la quieres
}, limpiar_nans=True)
🧪 Métricas y Backtest simple
modelos/evaluacion_modelos.py hace:

Split (train hasta -pasos, test últimos pasos)

MAE, RMSE, MAPE, R²

Sortino sobre retornos (pred vs real)

Accuracy direccional

Horizonte tomando timestamps del forecast live

🚀 MT5: ejecución (opcional)
app/main.py calcula SL/TP por pips del YAML, tamaño por riesgo fijo, y abre orden con easy_Trading.Basic_funcs.

Se corrige UPPER() → upper() en logs de orden.

🛟 Problemas comunes
1) Prophet / CmdStan (compilación)

prophet usa cmdstanpy y puede descargar/compilar CmdStan al primer uso.

Requisitos del sistema:

Windows: Microsoft C++ Build Tools (VS 2019+), make (p. ej. RTools o mingw64 con make).

macOS: Xcode Command Line Tools.

Linux: gcc, g++, make.

Si falla la compilación, prueba:

python
Copiar código
import cmdstanpy
cmdstanpy.install_cmdstan()
y revisa el log de compilación que imprime la ruta de CmdStan.

2) TensorFlow

Si no usarás LSTM, quita tensorflow-cpu del requirements.txt.

En algunas GPUs/CPUs (Windows), tensorflow estándar puede dar conflictos; usa tensorflow-cpu.

3) MT5

Asegúrate de:

Python y MT5 sean ambos 64-bit.

MT5_PATH apunte al terminal64.exe.

El broker/servidor sea correcto.

4) Columnas de entrada

Para EDA/Features: la serie debe tener columna temporal (time|timestamp|...) o índice datetime y al menos una columna de cierre (Close|close|price|...).

📚 Referencias (metodología y libretas)
CRISP-DM 1.0 — guía del ciclo de analítica.

Hyndman & Athanasopoulos. Forecasting: Principles and Practice (Prophet/estacionalidad/ETS).

Box, Jenkins & Reinsel. Time Series Analysis (ARIMA/diagnóstico).