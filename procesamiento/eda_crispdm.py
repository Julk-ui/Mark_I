# procesamiento/eda_crispdm.py — EDA claro (CRISP-DM) con ARIMA/SARIMA, narrativa y exportación PDF/Excel
# Reemplaza tu archivo por este completo si ves errores de indentación o variables no definidas.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sstats
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

# ---------- ARIMA / SARIMA opcionales ----------
try:
    from modelos.arima_scan import escanear_arima
    _ARIMA_SCAN_OK = True
except Exception as _e:
    _ARIMA_SCAN_OK = False
    print(f"ℹ️ ARIMA scan no disponible ({_e}).")

try:
    from modelos.sarima_scan import escanear_sarima
    _SARIMA_SCAN_OK = True
except Exception as _e:
    _SARIMA_SCAN_OK = False
    print(f"ℹ️ SARIMA scan no disponible ({_e}).")


# =======================
# Utilidades básicas
# =======================
def _safe_mkdir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _ensure_dt_index(df: pd.DataFrame, col_candidates=("timestamp","date","datetime","Date","Datetime")) -> pd.DataFrame:
    """Asegura índice datetime (UTC) a partir de una columna probable; o valida si ya es DatetimeIndex."""
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        return d.sort_index()
    dtcol = next((c for c in col_candidates if c in d.columns), None)
    if dtcol is None:
        raise ValueError(f"No se encontró columna de tiempo en {list(d.columns)} ni índice datetime.")
    d[dtcol] = pd.to_datetime(d[dtcol], errors="coerce", utc=True)
    d = d.dropna(subset=[dtcol]).sort_values(dtcol).set_index(dtcol)
    return d

def _find_close(df: pd.DataFrame) -> str:
    for c in ["Close", "close", "Adj Close", "price", "Price"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontró columna de precio/cierre.")

def _resample_ohlc(df: pd.DataFrame, freq: str, price_col: str) -> pd.DataFrame:
    """Resamplea a la frecuencia deseada, preservando OHLC si existen; si no, último cierre y suma de volumen."""
    cols_lower = [c.lower() for c in df.columns]
    has_ohlc = all(x in cols_lower for x in ["open","high","low",price_col.lower()])
    if has_ohlc:
        def _first(name):
            return [c for c in df.columns if c.lower()==name][0]
        agg = {
            _first("open"): "first",
            _first("high"): "max",
            _first("low"):  "min",
            _first(price_col.lower()): "last",
        }
        vol_candidates = [c for c in df.columns if c.lower() in ("volume","tick_volume","vol")]
        if vol_candidates:
            agg[vol_candidates[0]] = "sum"
        out = df.resample(freq).agg(agg)
    else:
        out = pd.DataFrame(index=df.index)
        out[price_col] = df[price_col].resample(freq).last()
        vol_candidates = [c for c in df.columns if c.lower() in ("volume","tick_volume","vol")]
        if vol_candidates:
            out[vol_candidates[0]] = df[vol_candidates[0]].resample(freq).sum()
    return out.dropna(how="any")

def _stl_period_by_freq(freq: str) -> int:
    f = str(freq).upper()
    if f in ("D","1D"): return 7
    if f in ("H","1H"): return 24
    if f.endswith("T"):
        try:
            minutes = int(f[:-1])
            return max(7, int((24*60)/minutes))
        except Exception:
            return 7
    return 7

def _to_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex) and getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    return d


# =======================
# Cálculos
# =======================
def _compute_returns_blocks(df: pd.DataFrame, price_col: str):
    r = df[price_col].pct_change()
    lr = np.log(df[price_col]).diff()
    return r, lr

def _compute_drawdown(price: pd.Series) -> pd.Series:
    cummax = price.cummax()
    return price / cummax - 1.0

def _compute_stats(logret: pd.Series) -> pd.DataFrame:
    s = pd.Series(logret).dropna()
    if s.empty:
        return pd.DataFrame([{}])
    jb_stat, jb_p = sstats.jarque_bera(s)
    out = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "JB_stat": float(jb_stat),
        "JB_pvalue": float(jb_p),
        "VaR_95": float(np.percentile(s, 5)),
        "ES_95": float(s[s <= np.percentile(s, 5)].mean()) if (s <= np.percentile(s, 5)).any() else np.nan,
    }
    return pd.DataFrame([out])

def _atr_if_available(df: pd.DataFrame) -> pd.Series | None:
    cols = [c.lower() for c in df.columns]
    has = all(x in cols for x in ["high","low"])
    close_name = next((c for c in df.columns if c.lower() in ("close","price","adj close")), None)
    if has and close_name is not None:
        high = df[[c for c in df.columns if c.lower()=="high"][0]].astype(float)
        low = df[[c for c in df.columns if c.lower()=="low"][0]].astype(float)
        close = df[close_name].astype(float)
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr = tr.rolling(14).mean()
        return atr
    return None

def _signals_from_series(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    s = df[price_col].astype(float)
    out = {}
    sma50 = s.rolling(50).mean()
    sma200 = s.rolling(200).mean()
    if len(sma200.dropna()):
        out["Tendencia_MA"] = "Alcista (SMA50>SMA200)" if float(sma50.iloc[-1]) > float(sma200.iloc[-1]) else "Bajista (SMA50<=SMA200)"
    # RSI(14) (EMA)
    delta = s.diff()
    up = delta.clip(lower=0.0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    if not rsi.dropna().empty:
        rv = float(rsi.iloc[-1])
        if rv >= 70: out["RSI_14"] = f"{rv:.1f} (sobrecompra)"
        elif rv <= 30: out["RSI_14"] = f"{rv:.1f} (sobreventa)"
        else: out["RSI_14"] = f"{rv:.1f} (neutral)"
    # MACD (12,26,9)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    if not macd.dropna().empty and not macd_sig.dropna().empty:
        out["MACD"] = "Alcista (MACD>signal)" if float(macd.iloc[-1]) > float(macd_sig.iloc[-1]) else "Bajista (MACD<=signal)"
    return pd.DataFrame([out]) if out else pd.DataFrame([{}])


# =======================
# Gráficos
# =======================
def _plot_precio_tendencia(df, price_col, symbol, outdir, win_ma):
    plt.figure(figsize=(11,5))
    plt.plot(df.index, df[price_col], label="Precio")
    if win_ma and win_ma > 1 and win_ma < len(df):
        ma = df[price_col].rolling(win_ma, min_periods=max(2, win_ma//3)).mean()
        plt.plot(df.index, ma, label=f"Media móvil ({win_ma})")
    plt.title(f"{symbol} · 01 Precio y Tendencia (MA)")
    plt.xlabel("Tiempo"); plt.ylabel("Precio")
    plt.legend(); plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_01_precio_tendencia.png")
    plt.savefig(path); plt.close()
    return path

def _plot_serie_precio(df, price_col, symbol, outdir):
    plt.figure(figsize=(11,4))
    plt.plot(df.index, df[price_col])
    plt.title(f"{symbol} · 02 Serie de tiempo (Precio)")
    plt.xlabel("Tiempo"); plt.ylabel("Precio")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_02_serie_precio.png")
    plt.savefig(path); plt.close()
    return path

def _plot_stl(df, price_col, symbol, outdir, seasonal):
    y = np.log(df[price_col].dropna())
    if len(y) < seasonal*3:
        return None
    stl = STL(y, period=seasonal, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(10,7)
    fig.suptitle(f"{symbol} · 03 Descomposición STL (log precio)")
    fig.tight_layout()
    path = os.path.join(outdir, f"{symbol}_03_stl.png")
    fig.savefig(path); plt.close(fig)
    return path

def _plot_hist_kde(logret, symbol, outdir):
    s = pd.Series(logret).dropna()
    if s.empty: return None
    plt.figure(figsize=(10,5))
    plt.hist(s, bins=60, density=True, alpha=0.6)
    s.plot(kind="kde")
    plt.title(f"{symbol} · 04 Distribución de log-returns (Hist + KDE)")
    plt.xlabel("log-return"); plt.ylabel("Densidad")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_04_hist_kde_logret.png")
    plt.savefig(path); plt.close()
    return path

def _plot_qq(logret, symbol, outdir):
    s = pd.Series(logret).dropna()
    if s.empty: return None
    plt.figure(figsize=(6,6))
    sstats.probplot(s, dist="norm", plot=plt)
    plt.title(f"{symbol} · 05 QQ-plot (log-returns vs Normal)")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_05_qqplot_logret.png")
    plt.savefig(path); plt.close()
    return path

def _plot_rolling_vol(logret, symbol, outdir, windows=(20,60,120), atr=None):
    s = pd.Series(logret)
    if s.dropna().empty: return None
    plt.figure(figsize=(11,4))
    for w in windows:
        s.rolling(w).std().plot(label=f"σ rolling {w}")
    if atr is not None:
        atr.plot(label="ATR(14)", alpha=0.7)
    plt.legend()
    plt.title(f"{symbol} · 06 Volatilidad rolling (σ) y ATR(14)")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_06_rolling_vol.png")
    plt.savefig(path); plt.close()
    return path

def _plot_acf_pacf(logret, symbol, outdir, lags=40):
    s = pd.Series(logret).dropna()
    if len(s) < 10: return (None, None)
    fig = plt.figure(figsize=(12,4)); plot_acf(s, lags=lags, ax=plt.gca())
    plt.title(f"{symbol} · 07 ACF (log-returns)"); plt.tight_layout()
    p1 = os.path.join(outdir, f"{symbol}_07_acf_logret.png"); plt.savefig(p1); plt.close()
    fig = plt.figure(figsize=(12,4)); plot_pacf(s, lags=lags, ax=plt.gca(), method="ywm")
    plt.title(f"{symbol} · 08 PACF (log-returns)"); plt.tight_layout()
    p2 = os.path.join(outdir, f"{symbol}_08_pacf_logret.png"); plt.savefig(p2); plt.close()
    return (p1, p2)

def _plot_drawdown(price, symbol, outdir):
    dd = _compute_drawdown(price)
    plt.figure(figsize=(11,3.8))
    plt.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    plt.title(f"{symbol} · 09 Curva de drawdown")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_09_drawdown.png")
    plt.savefig(path); plt.close()
    return path

def _plot_rolling_corr(df_eur_lr, df_spy_lr, outdir, window=60, title_suffix="EURUSD vs SPY/US500"):
    s = pd.Series(df_eur_lr).dropna().rename("lr_EURUSD").to_frame().join(
        pd.Series(df_spy_lr).dropna().rename("lr_SPY").to_frame(), how="inner"
    ).dropna()
    if s.empty: return None, None
    rolling_corr = s["lr_EURUSD"].rolling(window).corr(s["lr_SPY"]).dropna().to_frame("rolling_corr")
    plt.figure(figsize=(11,4))
    plt.plot(rolling_corr.index, rolling_corr["rolling_corr"])
    plt.axhline(0, linestyle="--")
    plt.title(f"10 Correlación móvil ({window}) log-returns · {title_suffix}")
    plt.tight_layout()
    path = os.path.join(outdir, "EURUSD_SPY_10_rolling_corr.png")
    plt.savefig(path); plt.close()
    return path, rolling_corr


# =======================
# Exportadores (Excel + PDF)
# =======================
def _export_excel(outpath: str, heads: dict, resumenes: dict, stats_map: dict,
                  corr_df: pd.DataFrame|None, roll_corr: pd.DataFrame|None) -> None:
    # Motor preferido
    try:
        import xlsxwriter  # noqa
        writer_kwargs = {"engine": "xlsxwriter", "datetime_format": "yyyy-mm-dd hh:mm"}
    except ImportError:
        writer_kwargs = {"engine": "openpyxl"}

    with pd.ExcelWriter(outpath, **writer_kwargs) as w:
        for sym, head_df in heads.items():
            h = head_df.copy()
            if isinstance(h.index, pd.DatetimeIndex) and getattr(h.index, "tz", None) is not None:
                h.index = h.index.tz_localize(None)
            h.to_excel(w, sheet_name=f"{sym}_HEAD")

            r = resumenes[sym].copy()
            for c in ("inicio","fin"):
                if c in r.columns:
                    r[c] = pd.to_datetime(r[c], errors="coerce", utc=True).dt.tz_localize(None)
            r.to_excel(w, sheet_name=f"{sym}_RESUMEN", index=False)

            st = stats_map.get(sym)
            if st is not None:
                st.to_excel(w, sheet_name=f"{sym}_STATS", index=False)

        if corr_df is not None:
            corr_df.to_excel(w, sheet_name="Correlation_matrix")
        if roll_corr is not None:
            rc = roll_corr.copy()
            if isinstance(rc.index, pd.DatetimeIndex) and getattr(rc.index, "tz", None) is not None:
                rc.index = rc.index.tz_localize(None)
            rc.to_excel(w, sheet_name="Rolling_corr")

def _add_image_page(pdf: PdfPages, img_path: str, title: str | None = None) -> None:
    if not img_path or not os.path.exists(img_path):
        return
    img = plt.imread(img_path)
    fig = plt.figure(figsize=(11, 7))
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    plt.imshow(img)
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

def _add_table_page(pdf: PdfPages, df: pd.DataFrame, title: str, index: bool = False, max_rows: int = 30) -> None:
    if df is None or df.empty:
        return
    df_show = df.copy()
    if not index:
        df_show = df_show.reset_index(drop=True)
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    df_show = df_show.apply(lambda c: c.round(6) if hasattr(c, "dtype") and getattr(c, "dtype", None) is not None and c.dtype.kind in "fc" else c)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=12)
    tbl = ax.table(cellText=df_show.values, colLabels=df_show.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    pdf.savefig(fig)
    plt.close(fig)

def _add_text_page(pdf: PdfPages, title: str, text: str) -> None:
    import textwrap
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, pad=12)
    wrapped = textwrap.fill(text or "", width=120)
    ax.text(0.02, 0.95, wrapped, va="top", ha="left", fontsize=11, wrap=True, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

def _export_pdf(outdir: str, artifacts_by_symbol: dict,
                corr_df: pd.DataFrame | None,
                roll_corr: pd.DataFrame | None,
                narrative_map: dict | None = None,
                filename: str = "EDA_informe.pdf") -> None:
    pdf_path = os.path.join(outdir, filename)
    with PdfPages(pdf_path) as pdf:
        # Portada
        fig = plt.figure(figsize=(11, 7))
        plt.axis("off")
        plt.text(0.5, 0.72, "Informe EDA", ha="center", va="center", fontsize=28, weight="bold")
        plt.text(0.5, 0.60, "EURUSD y segundo activo (SPY/US500)", ha="center", va="center", fontsize=14)
        plt.text(0.5, 0.48, f"Carpeta: {outdir}", ha="center", va="center", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Por activo
        for symbol, art in artifacts_by_symbol.items():
            # 00 — Narrativa ejecutiva (si existe)
            if narrative_map and symbol in narrative_map:
                _add_text_page(pdf, f"{symbol} — 00 Narrativa ejecutiva", narrative_map.get(symbol, ""))

            # Tablas y gráficos
            _add_table_page(pdf, art.get("HEAD"),   f"{symbol} — HEAD (primeras filas)", index=True)
            _add_table_page(pdf, art.get("RESUMEN"),f"{symbol} — RESUMEN",               index=False)
            _add_table_page(pdf, art.get("STATS"),  f"{symbol} — STATS (log-returns)",   index=False)

            _add_image_page(pdf, art.get("IMG_01"), f"{symbol} — 01 Precio y Tendencia")
            _add_image_page(pdf, art.get("IMG_02"), f"{symbol} — 02 Serie de Precio")
            _add_image_page(pdf, art.get("IMG_03"), f"{symbol} — 03 Descomposición STL")
            _add_image_page(pdf, art.get("IMG_04"), f"{symbol} — 04 Distribución log-returns (Hist+KDE)")
            _add_image_page(pdf, art.get("IMG_05"), f"{symbol} — 05 QQ-plot log-returns")
            _add_image_page(pdf, art.get("IMG_06"), f"{symbol} — 06 Volatilidad rolling y ATR")
            _add_image_page(pdf, art.get("IMG_07"), f"{symbol} — 07 ACF log-returns")
            _add_image_page(pdf, art.get("IMG_08"), f"{symbol} — 08 PACF log-returns")
            _add_image_page(pdf, art.get("IMG_09"), f"{symbol} — 09 Curva de drawdown")
            # Si tienes GARCH, se incluye si existe:
            _add_image_page(pdf, art.get("IMG_10_GARCH"), f"{symbol} — 10 Volatilidad GARCH (σ anualizada)")

        # Correlación entre activos
        if corr_df is not None:
            _add_table_page(pdf, corr_df, "Matriz de correlación (log-returns)", index=True)
        if roll_corr is not None and not roll_corr.empty:
            _add_table_page(pdf, roll_corr, "Correlación móvil (ventana)", index=True)
            _add_image_page(pdf, os.path.join(outdir, "EURUSD_SPY_10_rolling_corr.png"),
                            "Correlación móvil EURUSD vs SPY/US500 — gráfico")
    print(f"📄 Informe PDF generado: {pdf_path}")


# =======================
# Narrativa ejecutiva
# =======================
def _narrativa_basica(symbol: str, resumen_df: pd.DataFrame, stats_df: pd.DataFrame,
                      best_model_txt: str | None = None,
                      corr_mean: float | None = None, corr_last: float | None = None) -> str:
    import re as _re
    r = resumen_df.iloc[0].to_dict() if resumen_df is not None and not resumen_df.empty else {}
    s = stats_df.iloc[0].to_dict() if stats_df is not None and not stats_df.empty else {}
    jb_p = s.get("JB_pvalue", float("nan"))
    jb_text = "no normal" if (isinstance(jb_p, (int,float)) and jb_p < 0.05) else "compatible con normalidad"
    try:
        mdd_pct = float((r.get("precio_min") / (r.get("precio_max")+1e-12)) - 1.0)
    except Exception:
        mdd_pct = float("nan")

    corr_txt = ""
    if (corr_mean is not None) and (corr_last is not None):
        corr_txt = f"\n• Correlación EURUSD–SPY: media {corr_mean:.2f}, actual {corr_last:.2f}."

    model_lines = []
    if isinstance(best_model_txt, str) and best_model_txt:
        model_lines.append(f"• Modelo recomendado (por BIC): {best_model_txt}")
        m_arima = _re.match(r"ARIMA\((\d+),(\d+),(\d+)\)", best_model_txt or "")
        m_sarima = _re.match(r"SARIMA\((\d+),(\d+),(\d+)\)x\((\d+),(\d+),(\d+)\)\[(\d+)\]", best_model_txt or "")
        if m_arima:
            model_lines += [
                "  - d: nº de diferencias sobre log-precio (quita tendencia).",
                "  - p: memoria AR en retornos (lags de la serie diferenciada).",
                "  - q: media móvil (impacto de shocks previos en el error).",
                "  - Diagnóstico deseable: Ljung–Box p ≥ 0.05 en residuales (ruido blanco).",
            ]
        elif m_sarima:
            model_lines += [
                "  - Parte no estacional (p,d,q) y parte estacional (P,D,Q) con periodo s.",
                "  - d/D: diferencias (no estacional / estacional) para estabilizar media.",
                "  - p/P: memoria a lags cortos / a múltiplos de s.",
                "  - q/Q: shocks a corto plazo / a múltiplos de s.",
                "  - Diagnóstico deseable: Ljung–Box p ≥ 0.05 en residuales (ruido blanco).",
            ]
    else:
        model_lines.append("• Modelos escaneados: ARIMA y SARIMA (se elige por BIC más bajo).")

    lines = [
        f"{symbol} (diario)",
        f"• Periodo: {r.get('inicio')} → {r.get('fin')}",
        f"• Precio medio: {r.get('precio_promedio', float('nan')):.4f} | Último: {r.get('precio_ultimo', float('nan'))} | Rango [{r.get('precio_min', float('nan'))}, {r.get('precio_max', float('nan'))}]",
        f"• Riesgo (log-returns): σ={s.get('std', float('nan')):.4f} (~{s.get('std', float('nan'))*100:.2f}%), VaR95={s.get('VaR_95', float('nan'))*100:.2f}%, ES95={s.get('ES_95', float('nan'))*100:.2f}%, JB p={jb_p:.2g} ⇒ {jb_text}.",
        f"• MDD aprox.: {mdd_pct:.2%}."
    ] + model_lines

    if corr_txt:
        lines.append(corr_txt)

    return "\n".join(lines) + "\n"


# =======================
# EDA (principal)
# =======================
def ejecutar_eda(df_eurusd: pd.DataFrame | None = None,
                 df_spy: pd.DataFrame | None = None,
                 cfg: dict | None = None) -> None:
    """
    EDA ampliado (CRISP-DM):
      - HEAD / RESUMEN / STATS
      - Precio y Tendencia (MA), Serie, STL
      - Distribución/QQ de log-returns
      - Volatilidad rolling y ATR
      - ACF/PACF
      - Drawdown
      - Correlación (si hay 2 activos)
      - Recomendación ARIMA/SARIMA (Top-1 por BIC) + narrativa ejecutiva (PDF/Excel)
    """
    eda_cfg = (cfg or {}).get("eda", {})
    freq = str(eda_cfg.get("frecuencia_resampleo", "D"))
    outdir = eda_cfg.get("outdir", "outputs/eda")
    win_ma = int(eda_cfg.get("ventana_media_movil", 30))
    acf_lags = int(eda_cfg.get("acf_lags", 40))
    rv_windows = eda_cfg.get("rolling_vol_windows", [20,60,120])
    rc_window = int(eda_cfg.get("rolling_corr_window", 60))
    alias_eur = eda_cfg.get("alias_eur", "EURUSD")
    alias_spy = eda_cfg.get("alias_spy", "SPY")
    _safe_mkdir(outdir)

    activos = [(alias_eur, df_eurusd), (alias_spy, df_spy)]
    heads, resumenes, stats_map = {}, {}, {}
    artifacts: dict = {}
    arima_map, sarima_map = {}, {}
    narrative_map, signals_map = {}, {}

    # --- Por activo ---
    for symbol, df in activos:
        if df is None:
            continue

        df = _ensure_dt_index(df)
        price_col = _find_close(df)
        df = _resample_ohlc(df, freq=freq, price_col=price_col)

        # Derivadas
        ret, logret = _compute_returns_blocks(df, price_col)
        atr = _atr_if_available(df)

        # HEAD
        head_df = _to_naive_index(df.head(5))
        heads[symbol] = head_df

        # RESUMEN
        resumen = pd.DataFrame([{
            "activo": symbol,
            "filas": int(df[price_col].count()),
            "inicio": df.index.min(),
            "fin": df.index.max(),
            "precio_ultimo": float(df[price_col].iloc[-1]),
            "precio_promedio": float(df[price_col].mean()),
            "precio_min": float(df[price_col].min()),
            "precio_max": float(df[price_col].max()),
        }])
        resumen["inicio"] = pd.to_datetime(resumen["inicio"], utc=True).dt.tz_localize(None)
        resumen["fin"]   = pd.to_datetime(resumen["fin"],   utc=True).dt.tz_localize(None)
        resumenes[symbol] = resumen

        # STATS
        stats_lr = _compute_stats(logret)
        stats_map[symbol] = stats_lr

        # Modelos ARIMA/SARIMA
        arima_df = None
        if _ARIMA_SCAN_OK:
            try:
                arima_df = escanear_arima(df[price_col], max_p=3, max_q=3)
            except Exception as e:
                print(f"ℹ️ ARIMA scan falló para {symbol}: {e}")
        arima_map[symbol] = arima_df

        sarima_df = None
        if _SARIMA_SCAN_OK:
            try:
                sarima_df = escanear_sarima(df[price_col], s_candidates=[5,7], max_p=2, max_q=2, max_P=1, max_Q=1)
            except Exception as e:
                print(f"ℹ️ SARIMA scan falló para {symbol}: {e}")
        sarima_map[symbol] = sarima_df

        # Señales guía
        try:
            sig_df = _signals_from_series(df, price_col)
            if sig_df is not None and not sig_df.empty:
                signals_map[symbol] = sig_df
        except Exception as e:
            print(f"ℹ️ Señales guía no disponibles para {symbol}: {e}")

        # Gráficos
        p1 = _plot_precio_tendencia(df, price_col, symbol, outdir, win_ma=win_ma)
        p2 = _plot_serie_precio(df, price_col, symbol, outdir)
        p3 = _plot_stl(df, price_col, symbol, outdir, seasonal=_stl_period_by_freq(freq))
        p4 = _plot_hist_kde(logret, symbol, outdir)
        p5 = _plot_qq(logret, symbol, outdir)
        p6 = _plot_rolling_vol(logret, symbol, outdir, windows=rv_windows, atr=atr)
        p7, p8 = _plot_acf_pacf(logret, symbol, outdir, lags=acf_lags)
        p9 = _plot_drawdown(df[price_col], symbol, outdir)

        artifacts[symbol] = {
            "HEAD": head_df,
            "RESUMEN": resumen,
            "STATS": stats_lr,
            "IMG_01": p1, "IMG_02": p2, "IMG_03": p3, "IMG_04": p4,
            "IMG_05": p5, "IMG_06": p6, "IMG_07": p7, "IMG_08": p8, "IMG_09": p9
        }

        # Consola
        print(f"— {symbol} —")
        print("HEAD (5 filas):"); print(head_df)
        print("Resumen:"); print(resumen.to_string(index=False))
        print("Stats log-returns:"); print(stats_lr.to_string(index=False))
        print(f"Gráficos guardados en: {outdir}")
        print("-"*60)

    # Correlación si hay ambos
    corr_df = roll_corr = None
    if (df_eurusd is not None) and (df_spy is not None):
        dfe = _resample_ohlc(_ensure_dt_index(df_eurusd), freq=freq, price_col=_find_close(df_eurusd))
        _, lre = _compute_returns_blocks(dfe, _find_close(dfe))
        dfs = _resample_ohlc(_ensure_dt_index(df_spy), freq=freq, price_col=_find_close(df_spy))
        _, lrs = _compute_returns_blocks(dfs, _find_close(dfs))
        m = lre.dropna().rename(f"lr_{alias_eur}").to_frame().join(
            lrs.dropna().rename(f"lr_{alias_spy}").to_frame(), how="inner"
        ).dropna()
        if not m.empty:
            corr_df = m.corr()
            _, roll_corr = _plot_rolling_corr(lre, lrs, outdir, window=rc_window,
                                              title_suffix=f"{alias_eur} vs {alias_spy}")

    # Narrativa ejecutiva (elige mejor BIC entre ARIMA y SARIMA)
    corr_mean = float(roll_corr["rolling_corr"].mean()) if (roll_corr is not None and not roll_corr.empty) else None
    corr_last = float(roll_corr["rolling_corr"].iloc[-1]) if (roll_corr is not None and not roll_corr.empty) else None
    for sym in heads.keys():
        ar_df = arima_map.get(sym); sa_df = sarima_map.get(sym)
        best_txt, best_bic = None, None
        if ar_df is not None and not ar_df.empty:
            r0 = ar_df.iloc[0]; best_txt = f"ARIMA({int(r0['p'])},{int(r0['d'])},{int(r0['q'])})"; best_bic = float(r0['bic'])
        if sa_df is not None and not sa_df.empty:
            r1 = sa_df.iloc[0]; sar_txt = f"SARIMA({int(r1['p'])},{int(r1['d'])},{int(r1['q'])})x({int(r1['P'])},{int(r1['D'])},{int(r1['Q'])})[{int(r1['s'])}]"
            if (best_bic is None) or (float(r1['bic']) < best_bic):
                best_txt, best_bic = sar_txt, float(r1['bic'])
        narrative_map[sym] = _narrativa_basica(sym, resumenes[sym], stats_map.get(sym),
                                               best_model_txt=best_txt, corr_mean=corr_mean, corr_last=corr_last)

    # Exporta Excel base
    if heads:
        out_xlsx = os.path.join(outdir, "EDA_informe.xlsx")
        _export_excel(out_xlsx, heads, resumenes, stats_map, corr_df, roll_corr)
        print(f"📊 Excel generado: {out_xlsx}")
        # Añade pestañas extra
        try:
            import openpyxl  # noqa
            with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                for sym, ar_df in arima_map.items():
                    if ar_df is not None and not ar_df.empty:
                        ar_df.to_excel(w, sheet_name=f"{sym}_ARIMA_candidates", index=False)
                for sym, sa_df in sarima_map.items():
                    if sa_df is not None and not sa_df.empty:
                        sa_df.to_excel(w, sheet_name=f"{sym}_SARIMA_candidates", index=False)
                for sym, txt in narrative_map.items():
                    pd.DataFrame({"Narrativa":[txt]}).to_excel(w, sheet_name=f"{sym}_Narrative", index=False)
                for sym, sig_df in signals_map.items():
                    if sig_df is not None and not sig_df.empty:
                        sig_df.to_excel(w, sheet_name=f"{sym}_Signals_guide", index=False)

                # Guía de columnas (una sola hoja)
                try:
                    guide_rows = [
                        {"Sección":"ARIMA_candidates","Columna":"p,d,q","Significado":"Órdenes no estacionales: AR (p), diferencias (d), MA (q)."},
                        {"Sección":"ARIMA_candidates","Columna":"bic","Significado":"Criterio de información bayesiano (menor es mejor)."},
                        {"Sección":"ARIMA_candidates","Columna":"aic","Significado":"Criterio de Akaike (menor es mejor)."},
                        {"Sección":"ARIMA_candidates","Columna":"lb_p","Significado":"p-value de Ljung–Box en residuales (≥0.05 sugiere ruido blanco)."},

                        {"Sección":"SARIMA_candidates","Columna":"p,d,q","Significado":"Parte no estacional del modelo."},
                        {"Sección":"SARIMA_candidates","Columna":"P,D,Q","Significado":"Parte estacional del modelo (lags a múltiplos de s)."},
                        {"Sección":"SARIMA_candidates","Columna":"s","Significado":"Periodo estacional (diario: 5=semana hábil, 7=semana calendario)."},
                        {"Sección":"SARIMA_candidates","Columna":"bic,aic,lb_p","Significado":"Mismos criterios que en ARIMA."},

                        {"Sección":"STATS","Columna":"mean,std","Significado":"Media y desviación típica diaria de log-returns."},
                        {"Sección":"STATS","Columna":"skew,kurtosis","Significado":"Asimetría y exceso de curtosis (colas)."},
                        {"Sección":"STATS","Columna":"JB_stat,JB_pvalue","Significado":"Test de normalidad Jarque–Bera (p<0.05 ⇒ no normal)."},
                        {"Sección":"STATS","Columna":"VaR_95,ES_95","Significado":"Pérdida al 95% y pérdida media condicional en el 5% peor."},

                        {"Sección":"RESUMEN","Columna":"precio_ultimo/promedio/min/max","Significado":"Niveles de precio del periodo resampleado."},
                        {"Sección":"RESUMEN","Columna":"inicio,fin,filas","Significado":"Fechas de cobertura y nº de barras."},

                        {"Sección":"Signals_guide","Columna":"Tendencia_MA","Significado":"Cruce SMA50 vs SMA200 (lectura descriptiva)."},
                        {"Sección":"Signals_guide","Columna":"RSI_14","Significado":"Sobrecompra/sobreventa/neutral (14)."},
                        {"Sección":"Signals_guide","Columna":"MACD","Significado":"Relación MACD vs señal (12,26,9)."},
                    ]
                    pd.DataFrame(guide_rows).to_excel(w, sheet_name="Guide_columns", index=False)
                except Exception as _ge:
                    print(f"ℹ️ No se pudo escribir Guide_columns: {_ge}")
        except Exception as e:
            print(f"ℹ️ No se pudieron añadir hojas ARIMA/SARIMA/Narrative/Signals: {e}")

    # Exporta PDF
    if (cfg or {}).get("eda", {}).get("export_pdf", True):
        _export_pdf(outdir, artifacts, corr_df, roll_corr,
                    narrative_map=narrative_map,
                    filename=(cfg or {}).get("eda", {}).get("pdf_filename", "EDA_informe.pdf"))

    print("✅ EDA completado.")
