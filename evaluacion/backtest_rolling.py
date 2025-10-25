# evaluacion/backtest_rolling.py
# -*- coding: utf-8 -*-
import os
import re
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# Helpers
# =========================
def _pred_ret_to_pips(y_pred_ret: float, price_t: float, pip_size: float, use_exp: bool = True) -> float:
    """
    Convierte retorno predicho (Δlog) a pips respecto a price_t.
    use_exp=True: y_pred_price = P_t * exp(ret) (más correcto para Δlog)
    use_exp=False: aproximación lineal ret * P_t.
    """
    if price_t is None or pip_size is None or pip_size <= 0:
        return np.nan
    if np.isnan(y_pred_ret):
        return np.nan
    if use_exp:
        y_pred_price = price_t * float(np.exp(y_pred_ret))
        delta = y_pred_price - price_t
    else:
        delta = float(y_pred_ret) * price_t
    return float(delta / pip_size)


def _decide_signal_by_rule(y_pred_ret: float,
                           price_t: float,
                           pip_size: float,
                           decision_cfg: dict,
                           thr_pips: float,
                           sigma_next_pips: Optional[float]) -> tuple[float, int, float]:
    """
    Devuelve: (y_pred_pips, signal, z_used)
    - decision_cfg['type']: "pips" (por defecto) o "zscore"
    - thr_pips: umbral en pips (ya calculado con fixed/atr/garch)
    - sigma_next_pips: sigma prevista en pips (si existe), usada en zscore
    """
    y_pred_pips = _pred_ret_to_pips(y_pred_ret, price_t, pip_size, use_exp=True)
    decision_type = str((decision_cfg or {}).get("type", "pips")).lower()
    z_used = np.nan

    if decision_type == "pips":
        fire = bool(abs(y_pred_pips) >= float(thr_pips))
    else:
        z_thr = float((decision_cfg or {}).get("z_thr", 0.30))
        if sigma_next_pips is None or not np.isfinite(sigma_next_pips) or sigma_next_pips <= 0:
            fire = bool(abs(y_pred_pips) >= float(thr_pips))
        else:
            z_used = float(abs(y_pred_pips) / sigma_next_pips)
            fire = bool(z_used >= z_thr)

    signal = int(np.sign(y_pred_pips)) if fire else 0
    return y_pred_pips, signal, z_used


def _safe_mkdir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _sheet_name_safe(name: str) -> str:
    """Excel no acepta '[]:*?/\\' ni >31 chars."""
    name = re.sub(r"[\[\]\:\*\?\/\\']", "_", str(name))
    if len(name) > 31:
        name = name[:31]
    if not name:
        name = "Sheet"
    return name


def _file_name_safe(name: str) -> str:
    return re.sub(r"[\[\]\:\*\?\/\\']", "_", str(name))


def _rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else float("nan")


def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred)) if len(y_true) else float("nan")


def _r2(y_true, y_pred) -> float:
    try:
        return float(r2_score(y_true, y_pred)) if len(y_true) else float("nan")
    except Exception:
        return float("nan")


def _drawdown(series: pd.Series) -> pd.Series:
    cummax = series.cummax()
    return series - cummax  # en pips


def _sign(x: float) -> int:
    if x > 0: return 1
    if x < 0: return -1
    return 0


def _make_exog_lags(exog_ret: pd.Series, lags: List[int]) -> Optional[pd.DataFrame]:
    """Construye lags de exógena (rendimientos US500 por ejemplo)."""
    if exog_ret is None or not isinstance(exog_ret, (pd.Series, pd.DataFrame)) or not lags:
        return None
    if isinstance(exog_ret, pd.DataFrame):
        s = exog_ret.iloc[:, 0].copy()
    else:
        s = exog_ret.copy()
    out = {}
    for L in lags:
        out[f"exog_lag{L}"] = s.shift(L)
    X = pd.DataFrame(out)
    return X


# =========================
# Scans ARIMA / SARIMA (simple y robusto)
# =========================
def _scan_arima_returns(y: pd.Series, max_p=3, max_q=3, d=0, exog=None) -> Tuple[Tuple[int,int,int], float]:
    """
    Busca ARIMA(p,d,q) ligero por BIC. Evita el caso 0,0,0 sin
    modificar variables del bucle (bug frecuente).
    """
    best = None
    best_bic = float("inf")
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            p_ = p
            q_ = q
            if (p_ + q_) == 0:
                p_ = 1  # evita ARIMA(0,0,0) pero sin tocar 'p' del for
            try:
                model = SARIMAX(y, order=(p_, d, q_), exog=exog, trend="n",
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                bic = float(res.bic)
                if bic < best_bic:
                    best_bic = bic
                    best = (p_, d, q_)
            except Exception:
                continue
    if best is None:
        best = (0, d, 0)
        best_bic = float("inf")
    return best, best_bic


def _scan_sarima_returns(y: pd.Series,
                         s_candidates=(5, 7),
                         max_p=2, max_q=2, max_P=1, max_Q=1,
                         d=0, D=0, exog=None) -> Tuple[Tuple[int,int,int,int,int,int,int], float]:
    """
    Busca SARIMA(p,d,q)x(P,D,Q)[s] por BIC, evitando el todo-cero sin
    tocar variables de los bucles.
    """
    best = None
    best_bic = float("inf")
    for s in s_candidates:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        p_, q_, P_, Q_ = p, q, P, Q
                        if (p_ + q_ + P_ + Q_) == 0:
                            p_ = 1  # evita SARIMA(0,0,0)x(0,0,0)[s]
                        try:
                            model = SARIMAX(y, order=(p_, d, q_), seasonal_order=(P_, D, Q_, s),
                                            exog=exog, trend="n",
                                            enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False)
                            bic = float(res.bic)
                            if bic < best_bic:
                                best_bic = bic
                                best = (p_, d, q_, P_, D, Q_, s)
                        except Exception:
                            continue
    if best is None:
        best = (0, d, 0, 0, D, 0, s_candidates[0])
        best_bic = float("inf")
    return best, best_bic


# =========================
# Umbrales
# =========================
def _resolve_threshold(date_idx: pd.Timestamp,
                       mode: str,
                       fixed_pips: float,
                       min_threshold_pips: float,
                       atr_pips: pd.Series = None,
                       atr_k: float = 0.6,
                       garch_sigma_pips: pd.Series = None,
                       garch_k: float = 0.6) -> float:
    """
    Devuelve el umbral en pips para la fecha de corte.
    - fixed: valor fijo.
    - atr: atr_k * ATR(date) [pips], con piso.
    - garch: garch_k * sigma_garch(date) [pips], con piso.
    """
    mode = (mode or "fixed").lower()
    thr = fixed_pips
    if mode == "atr" and atr_pips is not None:
        val = float(atr_pips.reindex([date_idx]).ffill().iloc[-1])
        thr = atr_k * val
    elif mode == "garch" and garch_sigma_pips is not None:
        val = float(garch_sigma_pips.reindex([date_idx]).ffill().iloc[-1])
        thr = garch_k * val
    thr = max(float(thr), float(min_threshold_pips))
    return float(thr)


# =========================
# Rolling backtest core
# =========================
def _rolling_windows_index(n: int, initial_train: int, step: int, horizon: int) -> List[Tuple[int,int,int]]:
    out = []
    i = initial_train - 1
    while i + horizon < n:
        end_train = i
        end_test = i + horizon
        start_train = 0
        out.append((start_train, end_train, end_test))
        i += step
    return out


def _fit_and_forecast(y_train: pd.Series,
                      y_test_next_idx: pd.Timestamp,
                      spec: dict,
                      exog_train: Optional[pd.DataFrame] = None,
                      exog_next: Optional[pd.DataFrame] = None) -> Tuple[float, str]:
    """
    Entrena y predice 1 paso.
    spec['kind'] = 'rw' | 'auto' (por ejemplo) — si 'rw' devuelve 0.
    """
    kind = str(spec.get("kind", "rw")).lower()
    if kind == "rw":
        return 0.0, "RW"

    scan_cfg = (spec.get("scan") or {})
    try_sarima = bool(scan_cfg.get("try_sarima", True))
    max_p = int(scan_cfg.get("max_p", 3))
    max_q = int(scan_cfg.get("max_q", 3))
    max_P = int(scan_cfg.get("max_P", 1))
    max_Q = int(scan_cfg.get("max_Q", 1))
    s_candidates = scan_cfg.get("s_candidates", [5, 7])

    best_txt = None
    res = None

    exog_tr = exog_train
    exog_te = exog_next
    trend = "n"  # en retornos, lo más sano es sin constante

    if try_sarima:
        order_seas, _ = _scan_sarima_returns(y_train, s_candidates=s_candidates,
                                             max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                                             d=0, D=0, exog=exog_tr)
        p, d, q, P, D, Q, s = order_seas
        try:
            model = SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, s),
                            exog=exog_tr, trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            best_txt = f"SARIMA({p}, {d}, {q})x({P}, {D}, {Q})[{s}]"
        except Exception:
            res = None

    if res is None:
        order, _ = _scan_arima_returns(y_train, max_p=max_p, max_q=max_q, d=0, exog=exog_tr)
        p, d, q = order
        try:
            model = SARIMAX(y_train, order=(p, d, q), exog=exog_tr, trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            best_txt = f"ARIMA({p}, {d}, {q})"
        except Exception:
            return 0.0, "RW"

    try:
        y_pred = res.get_forecast(steps=1, exog=exog_te).predicted_mean.iloc[0]
        y_pred = float(y_pred)
    except Exception:
        y_pred = 0.0
        best_txt = (best_txt or "") + " [fallback]"

    return y_pred, best_txt


def evaluate_many(price: pd.Series,
                  specs: List[dict],
                  initial_train: int = 1000,
                  step: int = 10,
                  horizon: int = 1,
                  target: str = "returns",
                  pip_size: float = 0.0001,
                  threshold_pips: float = 15.0,
                  exog_ret: pd.Series = None,
                  exog_lags: List[int] = None,
                  threshold_mode: str = "fixed",
                  atr_pips: pd.Series = None,
                  atr_k: float = 0.6,
                  garch_k: float = 0.6,
                  min_threshold_pips: float = 10.0,
                  garch_sigma_pips: pd.Series = None,
                  log_threshold_used: bool = True,
                  decision_cfg: Optional[dict] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Walk-forward sobre specs.
    target='returns' => y = log(price).diff()
    decision_cfg: {"type": "pips" | "zscore", "z_thr": float}
    """
    price = price.astype(float).dropna()
    y_ret = np.log(price).diff().dropna()
    idx = price.index

    X = _make_exog_lags(exog_ret, exog_lags) if exog_lags else None
    windows = _rolling_windows_index(len(price), initial_train, step, horizon)
    n_win = len(windows)
    if n_win == 0:
        raise ValueError("No hay ventanas válidas. Revisa initial_train/step/horizon y el tamaño de la serie.")

    print(f"[AUTO] ventanas={n_win}, step={step}, horizon={horizon}, target={target}, thr_mode={threshold_mode}")

    preds_map: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    for spec in specs:
        name = spec.get("name", "MODEL")
        t0 = time.perf_counter()
        rows = []

        for i, (s0, e_tr, e_te) in enumerate(windows, start=1):
            date_end_train = idx[e_tr]
            date_next = idx[e_te]

            y_train = y_ret.loc[:date_end_train].dropna()
            if X is not None:
                exog_train = X.loc[y_train.index].dropna()
                y_train, exog_train = y_train.align(exog_train, join="inner")
            else:
                exog_train = None

            exog_next = X.reindex([date_next]) if X is not None else None

            # Umbral en pips (fijo / ATR / GARCH) para ESTA ventana
            thr = _resolve_threshold(
                date_end_train,
                threshold_mode,
                fixed_pips=threshold_pips,
                min_threshold_pips=min_threshold_pips,
                atr_pips=atr_pips,
                atr_k=atr_k,
                garch_sigma_pips=garch_sigma_pips,
                garch_k=garch_k
            )

            # Predicción 1-paso de retorno
            t_fit = time.perf_counter()
            y_pred_ret, spec_txt = _fit_and_forecast(y_train, date_next, spec, exog_train, exog_next)
            t_elapsed = time.perf_counter() - t_fit

            # Retorno real del día siguiente
            y_true_ret = y_ret.reindex([date_next]).iloc[0]

            price_t = price.loc[date_end_train]
            price_tp1 = price.loc[date_next]

            # Señal (por pips por defecto). Puedes pasar decision_cfg desde main/config.
            y_pred_pips, signal, z_used = _decide_signal_by_rule(
                y_pred_ret, price_t, pip_size,
                decision_cfg=(decision_cfg or {"type": "pips"}),
                thr_pips=thr,
                sigma_next_pips=None  # hook para zscore con sigma en pips
            )

            # PnL en pips
            pnl_pips = 0.0
            if signal != 0:
                move_pips = (price_tp1 - price_t) / pip_size
                pnl_pips = float(signal * move_pips)

            # Precio predicho
            y_pred_price = float(price_t * math.exp(y_pred_ret))

            rows.append({
                "date": date_next,
                "spec": spec_txt,
                "threshold_used_pips": float(thr) if log_threshold_used else np.nan,
                "y_true_ret": float(y_true_ret),
                "y_pred_ret": float(y_pred_ret),
                "y_pred_price": y_pred_price,
                "y_pred_pips": float(y_pred_pips),
                "signal": int(signal),
                "pnl_pips": float(pnl_pips),
                "price_t": float(price_t),
                "price_t1": float(price_tp1),
                "fit_seconds": round(t_elapsed, 1),
            })

            if i == 1 or i % 10 == 0 or i == n_win:
                print(f"[AUTO] {i:02d}/{n_win} fin={date_end_train.date()} spec={spec_txt} thr={thr:.1f}p t={t_elapsed:.1f}s")

        df_pred = pd.DataFrame(rows).set_index("date").sort_index()
        df_pred["cum_pips"] = df_pred["pnl_pips"].cumsum()

        # Métricas
        rmse = _rmse(df_pred["y_true_ret"], df_pred["y_pred_ret"])
        mae = _mae(df_pred["y_true_ret"], df_pred["y_pred_ret"])
        r2 = _r2(df_pred["y_true_ret"], df_pred["y_pred_ret"])

        price_err = df_pred["y_pred_price"] - df_pred["price_t1"]
        rmse_price = float(math.sqrt((price_err ** 2).mean())) if len(price_err) else float("nan")
        mae_price = float(price_err.abs().mean()) if len(price_err) else float("nan")

        err_pips = ((df_pred["y_pred_ret"] - df_pred["y_true_ret"]) * df_pred["price_t"]) / pip_size
        rmse_pips = float(math.sqrt((err_pips ** 2).mean())) if len(err_pips) else float("nan")
        mae_pips = float(err_pips.abs().mean()) if len(err_pips) else float("nan")

        pct_err = (price_err.abs() / df_pred["price_t1"]).replace([np.inf, -np.inf], np.nan).dropna()
        rmse_pct = float(math.sqrt(((price_err / df_pred["price_t1"]) ** 2).mean())) * 100.0 if len(pct_err) else float("nan")
        mae_pct = float(pct_err.mean() * 100.0) if len(pct_err) else float("nan")

        trades = df_pred[df_pred["signal"] != 0]
        n_trades = int(len(trades))
        wins = trades[trades["pnl_pips"] > 0]
        losses = trades[trades["pnl_pips"] < 0]
        hitrate = float((wins.shape[0] / n_trades) * 100.0) if n_trades > 0 else 0.0
        avg_gain = float(wins["pnl_pips"].mean()) if len(wins) else float("nan")
        avg_loss = float(losses["pnl_pips"].mean()) if len(losses) else float("nan")
        total_pips = float(trades["pnl_pips"].sum()) if n_trades > 0 else 0.0
        dd = _drawdown(df_pred["cum_pips"].fillna(0.0))
        maxdd = float(dd.min()) if len(dd) else 0.0
        sharpe_like = float(df_pred["pnl_pips"].mean() / (df_pred["pnl_pips"].std() + 1e-12) * math.sqrt(252.0)) if df_pred["pnl_pips"].std() > 0 else float("nan")

        elapsed = time.perf_counter() - t0
        summary_rows.append({
            "Modelo": name,
            "RMSE": rmse, "MAE": mae, "R2": r2,
            "RMSE_price": rmse_price, "MAE_price": mae_price,
            "RMSE_pips": rmse_pips, "MAE_pips": mae_pips,
            "RMSE_%": rmse_pct, "MAE_%": mae_pct,
            "HitRate": hitrate,
            "Total_pips": total_pips,
            "Trades": n_trades,
            "WinRate_%": hitrate,
            "AvgGain_pips": avg_gain,
            "AvgLoss_pips": avg_loss,
            "MaxDD_pips": maxdd,
            "Sharpe_like": sharpe_like,
            "elapsed_s": round(elapsed, 1)
        })

        preds_map[name] = df_pred

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, preds_map


# =========================
# Export / Plots
# =========================
def save_backtest_excel(outxlsx: str,
                        summary: pd.DataFrame,
                        preds_map: Dict[str, pd.DataFrame]):
    _safe_mkdir(os.path.dirname(outxlsx) or ".")
    try:
        import xlsxwriter  # noqa
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(outxlsx, engine=engine) as w:
        summary.round(6).to_excel(w, sheet_name="Summary", index=False)
        for name, df in preds_map.items():
            sheet = _sheet_name_safe(name)
            tmp = df.copy()
            # redondeos amigables
            for c in ("y_true_ret", "y_pred_ret"):
                if c in tmp.columns:
                    tmp[c] = tmp[c].round(6)
            for c in ("y_pred_price", "price_t", "price_t1"):
                if c in tmp.columns:
                    tmp[c] = tmp[c].round(6)
            for c in ("y_pred_pips", "threshold_used_pips", "pnl_pips", "cum_pips"):
                if c in tmp.columns:
                    tmp[c] = tmp[c].round(3)
            tmp.to_excel(w, sheet_name=sheet)


def save_backtest_plots(outdir: str,
                        y_price: pd.Series,
                        preds_map: Dict[str, pd.DataFrame],
                        pip_size: float,
                        threshold_pips: float):
    """
    4 gráficos por modelo: (retornos, pips, precio, equity)
    """
    _safe_mkdir(outdir)

    for name, df in preds_map.items():
        nm = _file_name_safe(name)
        d = df.copy().sort_index()

        if {"threshold_used_pips", "price_t"}.issubset(d.columns):
            # pips -> retorno equivalente: thr_ret = thr_pips * (pip_size / price_t)
            d["thr_ret"] = (d["threshold_used_pips"] * (pip_size / d["price_t"]))
        else:
            d["thr_ret"] = np.nan

        # 1) Retornos
        fig, ax = plt.subplots(figsize=(11, 4))
        if "y_true_ret" in d and "y_pred_ret" in d:
            ax.plot(d.index, d["y_true_ret"], label="Ret. real", linewidth=1.2)
            ax.plot(d.index, d["y_pred_ret"], label="Ret. pred", linewidth=1.0, alpha=0.9)
        if d["thr_ret"].notna().any():
            ax.plot(d.index, d["thr_ret"], linestyle="--", linewidth=1.0, label="+Umbral (ret equiv.)")
            ax.plot(d.index, -d["thr_ret"], linestyle="--", linewidth=1.0, label="-Umbral (ret equiv.)")

        if "signal" in d and "y_true_ret" in d:
            buys = d[d["signal"] == 1]
            sells = d[d["signal"] == -1]
            if not buys.empty:
                ax.scatter(buys.index, buys["y_true_ret"], marker="^", s=40, label="Buy signal", zorder=3)
            if not sells.empty:
                ax.scatter(sells.index, sells["y_true_ret"], marker="v", s=40, label="Sell signal", zorder=3)

        ax.set_title(f"{name} · Retornos (true vs pred) + umbral")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{nm}_ret_line.png"))
        plt.close(fig)

        # 2) Predicción en pips vs umbral
        fig, ax = plt.subplots(figsize=(11, 4))
        if "y_pred_pips" in d:
            ax.plot(d.index, d["y_pred_pips"], label="Predicción (pips)", linewidth=1.2)
        if "threshold_used_pips" in d:
            ax.plot(d.index, d["threshold_used_pips"], linestyle="--", linewidth=1.0, label="+Umbral (pips)")
            ax.plot(d.index, -d["threshold_used_pips"], linestyle="--", linewidth=1.0, label="-Umbral (pips)")

        if "signal" in d and "y_pred_pips" in d:
            trades_idx = d.index[d["signal"] != 0]
            if len(trades_idx):
                colors = np.where(d.loc[trades_idx, "signal"] > 0, "g", "r")
                ax.scatter(trades_idx, d.loc[trades_idx, "y_pred_pips"], c=colors, s=25, label="Señales")

        ax.set_title(f"{name} · Predicción en pips vs umbral")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{nm}_pips_vs_threshold.png"))
        plt.close(fig)

        # 3) Precio real vs predicho
        if {"price_t1", "y_pred_price"}.issubset(d.columns):
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(d.index, d["price_t1"], label="Precio real (t+1)", linewidth=1.2)
            ax.plot(d.index, d["y_pred_price"], label="Precio predicho", linewidth=1.0, alpha=0.9)

            if "signal" in d:
                buys = d[d["signal"] == 1]
                sells = d[d["signal"] == -1]
                if not buys.empty:
                    ax.scatter(buys.index, buys["price_t1"], marker="^", s=40, label="Buy", zorder=3)
                if not sells.empty:
                    ax.scatter(sells.index, sells["price_t1"], marker="v", s=40, label="Sell", zorder=3)

            ax.set_title(f"{name} · Precio real vs precio predicho")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{nm}_price_line.png"))
            plt.close(fig)

        # 4) Equity curve
        if "cum_pips" in d.columns:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(d.index, d["cum_pips"], linewidth=1.2)
            ax.set_title(f"{name} · PnL acumulado (pips)")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{nm}_equity.png"))
            plt.close(fig)
