# procesamiento/data_quality.py
"""
Data quality report helpers for time series (designed for OHLCV daily bars).
This module is *read-only*; it does not mutate input dataframes.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def data_quality_report(df: pd.DataFrame, freq: str = "D", price_col: str | None = None) -> pd.DataFrame:
    """
    Build a compact data quality report.
    - Basic shape and nulls
    - Duplicates
    - Gaps in datetime index at the given frequency
    - Simple semantic checks (negative/zero prices, zero/NaN volumes)
    Returns a long-form DataFrame with ("check", "metric", "value").
    """
    d = df.copy()
    out_rows = []

    # Index checks
    idx = d.index
    out_rows.append(("shape", "rows", int(len(d))))
    out_rows.append(("shape", "cols", int(d.shape[1])))
    out_rows.append(("index", "is_datetimeindex", bool(isinstance(idx, pd.DatetimeIndex))))
    tz = getattr(idx, "tz", None)
    out_rows.append(("index", "tzinfo", str(tz) if tz is not None else "naive"))

    # Nulls by column
    for c in d.columns:
        nnull = int(d[c].isna().sum())
        if nnull > 0:
            out_rows.append(("nulls", c, nnull))

    # Duplicates
    dup = int(d.duplicated().sum())
    out_rows.append(("duplicates", "rows_duplicated", dup))

    # Gaps (only if DatetimeIndex)
    if isinstance(idx, pd.DatetimeIndex):
        try:
            expected = pd.date_range(idx.min(), idx.max(), freq=freq, tz=idx.tz)
            missing = expected.difference(idx)
            out_rows.append(("gaps", "n_missing_timestamps", int(len(missing))))
            # for readability, show first/last 3 missing
            if len(missing) > 0:
                miss_preview = list(missing[:3].astype(str)) + (["..."] if len(missing) > 6 else []) + list(missing[-3:].astype(str))
                out_rows.append(("gaps", "missing_preview", ", ".join(miss_preview)))
        except Exception as e:
            out_rows.append(("gaps", "error", str(e)))

    # Semantic: price/volume quick checks
    if price_col and price_col in d.columns:
        s = pd.to_numeric(d[price_col], errors="coerce")
        out_rows.append(("price_checks", "n_nan_price", int(s.isna().sum())))
        out_rows.append(("price_checks", "n_nonpositive", int((s <= 0).sum())))
    # volume candidates
    vol_col = next((c for c in d.columns if c.lower() in ("volume","tick_volume","vol")), None)
    if vol_col:
        v = pd.to_numeric(d[vol_col], errors="coerce")
        out_rows.append(("volume_checks", "n_nan_volume", int(v.isna().sum())))
        out_rows.append(("volume_checks", "n_zero_volume", int((v == 0).sum())))

    return pd.DataFrame(out_rows, columns=["check","metric","value"])
