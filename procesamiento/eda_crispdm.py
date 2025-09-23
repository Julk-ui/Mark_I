# procesamiento/eda_crispdm.py  (versión extendida con PDF + Excel + insights)
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sstats  # Jarque–Bera, etc.

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =======================
# Utilidades
# =======================
def _safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def _ensure_dt_index(df: pd.DataFrame, col_candidates=("timestamp","date","datetime","Date","Datetime")) -> pd.DataFrame:
    """Asegura índice datetime (UTC) a partir de una columna de tiempo conocida."""
    d = df.copy()
    dtcol = next((c for c in col_candidates if c in d.columns), None)
    if dtcol is None:
        if isinstance(d.index, pd.DatetimeIndex):
            return d.sort_index()
        raise ValueError(f"No se encontró columna de tiempo en {list(d.columns)}")
    d[dtcol] = pd.to_datetime(d[dtcol], errors="coerce", utc=True)
    d = d.dropna(subset=[dtcol]).sort_values(dtcol).set_index(dtcol)
    return d

def _find_close(df: pd.DataFrame) -> str:
    """Encuentra la columna de precio de cierre."""
    for c in ["Close", "close", "Adj Close", "price", "Price"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontró columna de precio/cierre.")

def _resample_ohlc(df: pd.DataFrame, freq: str, price_col: str) -> pd.DataFrame:
    """Resamplea a la frecuencia deseada, preservando OHLC si existen; de lo contrario, último cierre."""
    freq = str(freq).lower()
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
    """Elige un período STL razonable según la frecuencia."""
    f = str(freq).upper()
    if f in ("D","1D"):      # diario: semana
        return 7
    if f in ("H","1H"):      # horario: 24 horas
        return 24
    if f.endswith("T"):      # minutos: 1 día / intervalo
        try:
            minutes = int(f[:-1])
            return max(7, int((24*60)/minutes))
        except:
            return 7
    return 7

def _to_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Quita tz del índice para Excel/PDF."""
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex) and getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    return d

# =======================
# Cálculos adicionales
# =======================
def _compute_returns_blocks(df: pd.DataFrame, price_col: str):
    r = df[price_col].pct_change()
    lr = np.log(df[price_col]).diff()
    return r, lr

def _compute_drawdown(price: pd.Series) -> pd.Series:
    cummax = price.cummax()
    dd = price / cummax - 1.0
    return dd

def _compute_stats(logret: pd.Series) -> pd.DataFrame:
    s = logret.dropna()
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

# =======================
# Gráficos (nombres claros)
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
    s = logret.dropna()
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
    s = logret.dropna()
    if s.empty: return None
    plt.figure(figsize=(6,6))
    sstats.probplot(s, dist="norm", plot=plt)
    plt.title(f"{symbol} · 05 QQ-plot (log-returns vs Normal)")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_05_qqplot_logret.png")
    plt.savefig(path); plt.close()
    return path

def _plot_rolling_vol(logret, symbol, outdir, windows=(20,60,120), atr=None):
    s = logret
    if s.dropna().empty: return None
    plt.figure(figsize=(11,4))
    for w in windows:
        s.rolling(w).std().plot(label=f"σ rolling {w}")
    if atr is not None:
        # Escala ATR para que sea comparable (opcional). Aquí la dibujamos cruda.
        atr.plot(label="ATR(14)", alpha=0.7)
    plt.legend()
    plt.title(f"{symbol} · 06 Volatilidad rolling (σ) y ATR(14)")
    plt.tight_layout()
    path = os.path.join(outdir, f"{symbol}_06_rolling_vol.png")
    plt.savefig(path); plt.close()
    return path

def _plot_acf_pacf(logret, symbol, outdir, lags=40):
    s = logret.dropna()
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
    s = df_eur_lr.dropna().rename("lr_EURUSD").to_frame().join(
        df_spy_lr.dropna().rename("lr_SPY").to_frame(), how="inner"
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
                  corr_df: pd.DataFrame|None, roll_corr: pd.DataFrame|None):
    # elegir motor
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

def _add_image_page(pdf: PdfPages, img_path: str, title: str | None = None):
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

def _add_table_page(pdf: PdfPages, df: pd.DataFrame, title: str, index: bool = False, max_rows: int = 30):
    if df is None or df.empty:
        return
    df_show = df.copy()
    if not index:
        df_show = df_show.reset_index(drop=True)
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    df_show = df_show.applymap(lambda x: round(x, 6) if isinstance(x, (float, np.floating)) else x)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=12)
    tbl = ax.table(cellText=df_show.values, colLabels=df_show.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    pdf.savefig(fig)
    plt.close(fig)

def _export_pdf(outdir, artifacts_by_symbol: dict,
                corr_df: pd.DataFrame|None, roll_corr: pd.DataFrame|None,
                filename="EDA_informe.pdf"):
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
            _add_table_page(pdf, art.get("HEAD"), f"{symbol} — HEAD (primeras filas)", index=True)
            _add_table_page(pdf, art.get("RESUMEN"), f"{symbol} — RESUMEN", index=False)
            _add_table_page(pdf, art.get("STATS"), f"{symbol} — STATS (log-returns)", index=False)

            _add_image_page(pdf, art.get("IMG_01"), f"{symbol} — 01 Precio y Tendencia")
            _add_image_page(pdf, art.get("IMG_02"), f"{symbol} — 02 Serie de Precio")
            _add_image_page(pdf, art.get("IMG_03"), f"{symbol} — 03 Descomposición STL")
            _add_image_page(pdf, art.get("IMG_04"), f"{symbol} — 04 Distribución log-returns (Hist+KDE)")
            _add_image_page(pdf, art.get("IMG_05"), f"{symbol} — 05 QQ-plot log-returns")
            _add_image_page(pdf, art.get("IMG_06"), f"{symbol} — 06 Volatilidad rolling y ATR")
            _add_image_page(pdf, art.get("IMG_07"), f"{symbol} — 07 ACF log-returns")
            _add_image_page(pdf, art.get("IMG_08"), f"{symbol} — 08 PACF log-returns")
            _add_image_page(pdf, art.get("IMG_09"), f"{symbol} — 09 Curva de drawdown")

        # Correlación entre activos
        if corr_df is not None:
            _add_table_page(pdf, corr_df, "Matriz de correlación (log-returns)", index=True)
        if roll_corr is not None:
            _add_table_page(pdf, roll_corr, "Correlación móvil (ventana)", index=True)
            _add_image_page(pdf, os.path.join(outdir, "EURUSD_SPY_10_rolling_corr.png"),
                            "Correlación móvil EURUSD vs SPY/US500 — gráfico")
    print(f"📄 Informe PDF generado: {pdf_path}")

# =======================
# EDA (principal)
# =======================
def ejecutar_eda(df_eurusd=None, df_spy=None, cfg: dict = None):
    """
    EDA ampliado y claro:
    - HEAD (primeras filas)
    - Precio + Tendencia (MA)
    - Serie de Precio
    - Descomposición STL
    - Distribución (Hist+KDE) y QQ-plot de log-returns
    - Volatilidad rolling (σ) y ATR(14) si hay OHLC
    - ACF/PACF de log-returns
    - Curva de drawdown
    - Correlación móvil (si hay ambos activos)

    Control por config (ejemplo):
      cfg['eda'] = {
        'frecuencia_resampleo': 'D'     # o 'H', '15T', etc.
        'outdir': 'outputs/eda',
        'ventana_media_movil': 30,
        'acf_lags': 40,
        'rolling_vol_windows': [20,60,120],
        'rolling_corr_window': 60,
        'export_pdf': True,
        'pdf_filename': 'EDA_informe.pdf',
        'alias_eur': 'EURUSD',
        'alias_spy': 'SPY'
      }
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
    artifacts = {}

    # --- Procesa cada activo ---
    for symbol, df in activos:
        if df is None:
            continue

        # Preparación
        df = _ensure_dt_index(df)
        price_col = _find_close(df)
        df = _resample_ohlc(df, freq=freq, price_col=price_col)

        # Derivadas
        ret, logret = _compute_returns_blocks(df, price_col)
        atr = _atr_if_available(df)

        # HEAD
        head_df = _to_naive_index(df.head(5))
        heads[symbol] = head_df

        # RESUMEN básico
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

        # STATS de log-returns
        stats_lr = _compute_stats(logret)
        stats_map[symbol] = stats_lr

        # Gráficos
        p1 = _plot_precio_tendencia(df, price_col, symbol, outdir, win_ma=win_ma)
        p2 = _plot_serie_precio(df, price_col, symbol, outdir)
        p3 = _plot_stl(df, price_col, symbol, outdir, seasonal=_stl_period_by_freq(freq))
        p4 = _plot_hist_kde(logret, symbol, outdir)
        p5 = _plot_qq(logret, symbol, outdir)
        p6 = _plot_rolling_vol(logret, symbol, outdir, windows=rv_windows, atr=atr)
        p7, p8 = _plot_acf_pacf(logret, symbol, outdir, lags=acf_lags)
        p9 = _plot_drawdown(df[price_col], symbol, outdir)

        # Consolida artifacts para el PDF
        artifacts[symbol] = {
            "HEAD": head_df,
            "RESUMEN": resumen,
            "STATS": stats_lr,
            "IMG_01": p1, "IMG_02": p2, "IMG_03": p3, "IMG_04": p4,
            "IMG_05": p5, "IMG_06": p6, "IMG_07": p7, "IMG_08": p8, "IMG_09": p9
        }

        # Mensajes en consola
        print(f"— {symbol} —")
        print("HEAD (5 filas):")
        print(head_df)
        print("Resumen:")
        print(resumen.to_string(index=False))
        print("Stats log-returns:")
        print(stats_lr.to_string(index=False))
        print(f"Gráficos guardados en: {outdir}")
        print("-"*60)

    # --- Correlación si hay ambos ---
    corr_df = roll_corr = None
    if (df_eurusd is not None) and (df_spy is not None):
        # recompute con alias "limpios"
        # EUR
        dfe = _resample_ohlc(_ensure_dt_index(df_eurusd), freq=freq, price_col=_find_close(df_eurusd))
        _, lre = _compute_returns_blocks(dfe, _find_close(dfe))
        # SPY/US500
        dfs = _resample_ohlc(_ensure_dt_index(df_spy), freq=freq, price_col=_find_close(df_spy))
        _, lrs = _compute_returns_blocks(dfs, _find_close(dfs))

        # Matriz correlación
        m = lre.dropna().rename(f"lr_{alias_eur}").to_frame().join(
            lrs.dropna().rename(f"lr_{alias_spy}").to_frame(), how="inner"
        ).dropna()
        if not m.empty:
            corr_df = m.corr()

            # Rolling correlation
            rc_path, roll_corr = _plot_rolling_corr(lre, lrs, outdir, window=rc_window,
                                                    title_suffix=f"{alias_eur} vs {alias_spy}")

    # --- Exporta Excel ---
    if heads:
        out_xlsx = os.path.join(outdir, "EDA_informe.xlsx")
        _export_excel(out_xlsx, heads, resumenes, stats_map, corr_df, roll_corr)
        print(f"📊 Excel generado: {out_xlsx}")

    # --- Exporta PDF ---
    if (cfg or {}).get("eda", {}).get("export_pdf", True):
        _export_pdf(outdir, artifacts, corr_df, roll_corr,
                    filename=(cfg or {}).get("eda", {}).get("pdf_filename", "EDA_informe.pdf"))

    print("✅ EDA completado.")