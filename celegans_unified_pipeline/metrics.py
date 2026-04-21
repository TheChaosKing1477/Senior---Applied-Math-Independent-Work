from __future__ import annotations
from typing import Callable, Dict, Sequence, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV

from .preprocess import prepare_trace
from .config import (MAX_LAG, INFO_BINS, OUTER_TRAIN_FRACTION, INNER_TRAIN_FRACTION,
                     N_SURROGATES, RANDOM_STATE)

# ---------- helpers ----------

def blocked_split(n: int, fraction: float) -> Tuple[slice, slice]:
    cut = int(round(n*fraction))
    cut = min(max(cut, 10), max(n-5, 10))
    return slice(0, cut), slice(cut, n)

def _apply_lag(source, target, lag):
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    lag = int(max(0, lag))
    xs, ys = (source, target) if lag==0 else (source[:-lag], target[lag:])
    valid = np.isfinite(xs) & np.isfinite(ys)
    return xs[valid], ys[valid]

# ---------- lag selection ----------

def scan_optimal_lag(source: np.ndarray, target: np.ndarray, max_lag: int=MAX_LAG) -> Dict[str, float]:
    """Maximize |corr(source(t), target(t+lag))| across lag.

    This mirrors your MATLAB scanning step and is fast, but optimistic if lag is
    selected on the same samples used for evaluation. Use `nested_select_lag`
    for rigor when reporting r_test, MI, TE, etc.
    """
    best_lag = 0
    best_r = np.nan
    best_abs = -np.inf
    for lag in range(0, max_lag+1):
        xs, ys = _apply_lag(source, target, lag)
        if len(xs) < 10:
            continue
        if np.nanstd(xs) < 1e-12 or np.nanstd(ys) < 1e-12:
            continue
        r = float(np.corrcoef(xs, ys)[0,1])
        if abs(r) > best_abs:
            best_abs = abs(r)
            best_r = r
            best_lag = lag
    return {"lag": int(best_lag), "r": float(best_r) if np.isfinite(best_r) else np.nan}

def nested_select_lag(source, target, max_lag: int=MAX_LAG, outer_fraction: float=OUTER_TRAIN_FRACTION, inner_fraction: float=INNER_TRAIN_FRACTION):
    """Nested blocked lag selection.

    Outer split: train/selection vs test.
    Inner split: select lag on validation only.

    Returns aligned arrays and test slice indices.
    """
    source = np.asarray(source, float)
    target = np.asarray(target, float)

    tr_outer, _ = blocked_split(len(source), outer_fraction)
    s_train, t_train = source[tr_outer], target[tr_outer]

    best = {"lag": 0, "val_r": -np.inf}
    for lag in range(0, max_lag+1):
        xs, ys = _apply_lag(s_train, t_train, lag)
        if len(xs) < 25:
            continue
        tr_in, va_in = blocked_split(len(xs), inner_fraction)
        if np.nanstd(xs[tr_in])<1e-8 or np.nanstd(ys[tr_in])<1e-8 or np.nanstd(xs[va_in])<1e-8 or np.nanstd(ys[va_in])<1e-8:
            continue
        val_r = float(np.corrcoef(xs[va_in], ys[va_in])[0,1])
        if np.isfinite(val_r) and abs(val_r) > abs(best["val_r"]):
            best = {"lag": int(lag), "val_r": val_r}

    lag = int(best["lag"])
    xs_all, ys_all = _apply_lag(source, target, lag)
    tr_eval, te_eval = blocked_split(len(xs_all), outer_fraction)
    train_r = float(np.corrcoef(xs_all[tr_eval], ys_all[tr_eval])[0,1]) if np.nanstd(xs_all[tr_eval])>1e-8 and np.nanstd(ys_all[tr_eval])>1e-8 else np.nan
    test_r  = float(np.corrcoef(xs_all[te_eval], ys_all[te_eval])[0,1]) if np.nanstd(xs_all[te_eval])>1e-8 and np.nanstd(ys_all[te_eval])>1e-8 else np.nan
    return {"lag": lag, "train_r": train_r, "test_r": test_r, "aligned_source": xs_all, "aligned_target": ys_all, "test_slice": te_eval}

def cross_correlation_curve(source: np.ndarray, target: np.ndarray, max_lag: int=MAX_LAG):
    x = np.asarray(source, float)
    y = np.asarray(target, float)
    lags = np.arange(-max_lag, max_lag+1)
    corrs = np.full_like(lags, np.nan, dtype=float)
    for i, lag in enumerate(lags):
        if lag == 0:
            xs, ys = x, y
        elif lag > 0:
            xs, ys = x[:-lag], y[lag:]
        else:
            lag2 = -lag
            xs, ys = x[lag2:], y[:-lag2]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 10:
            continue
        xv, yv = xs[valid], ys[valid]
        if np.nanstd(xv)<1e-12 or np.nanstd(yv)<1e-12:
            continue
        corrs[i] = float(np.corrcoef(xv, yv)[0,1])
    return lags, corrs

# ---------- discretization + information ----------

def _quantile_digitize(x: np.ndarray, n_bins: int):
    x = np.asarray(x, float)
    q = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(np.nanmin(x), np.nanmax(x)+1e-9, max(2, n_bins+1))
    bins = np.digitize(x, edges[1:-1], right=False)
    return bins.astype(int), edges

def entropy_discrete(bins: np.ndarray) -> float:
    vals, counts = np.unique(np.asarray(bins, int), return_counts=True)
    p = counts / max(counts.sum(), 1)
    return float(-(p * np.log2(np.clip(p, 1e-12, None))).sum())

def mutual_information_discrete(x: np.ndarray, y: np.ndarray, n_bins: int=INFO_BINS) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return np.nan
    xb, _ = _quantile_digitize(x, n_bins)
    yb, _ = _quantile_digitize(y, n_bins)
    nx, ny = int(xb.max())+1, int(yb.max())+1
    joint = np.zeros((nx, ny), float)
    for i,j in zip(xb, yb):
        joint[i,j] += 1
    joint /= max(joint.sum(), 1)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px @ py)
        logterm = np.log2(np.clip(ratio, 1e-12, None))
    return float(np.nansum(joint * logterm))

def joint_counts_discrete(x: np.ndarray, y: np.ndarray, n_bins: int=INFO_BINS) -> np.ndarray:
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    xb, _ = _quantile_digitize(x, n_bins)
    yb, _ = _quantile_digitize(y, n_bins)
    nx, ny = int(xb.max())+1, int(yb.max())+1
    J = np.zeros((nx, ny), float)
    for i,j in zip(xb, yb):
        J[i,j] += 1
    return J

def transfer_entropy_discrete(source: np.ndarray, target: np.ndarray, lag: int=1, n_bins: int=INFO_BINS) -> float:
    """Discrete plug-in TE(source→target) = I(S_{t-lag}; T_t | T_{t-1}).

    This estimator is simple and interpretable, but can be biased for small n.
    Use it with:
    - stationarity checks,
    - held-out evaluation,
    - and surrogate controls.
    """
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    lag = int(max(1, lag))
    valid = np.isfinite(source) & np.isfinite(target)
    source = source[valid]
    target = target[valid]
    if len(source) < lag + 5:
        return np.nan

    sb, _ = _quantile_digitize(source, n_bins)
    tb, _ = _quantile_digitize(target, n_bins)

    s_prev = sb[:-lag]
    t_curr = tb[lag:]
    t_prev = tb[lag-1:-1]

    n_s = int(max(s_prev.max(), 0)) + 1
    n_t = int(max(t_curr.max(), 0)) + 1

    counts = np.zeros((n_t, n_t, n_s), dtype=float)
    for tc, tp, sp in zip(t_curr, t_prev, s_prev):
        counts[int(tc), int(tp), int(sp)] += 1.0

    total = counts.sum()
    if total <= 0:
        return np.nan

    p_xyz = counts / total
    p_tp_sp = p_xyz.sum(axis=0)   # (t_prev, s_prev)
    p_tc_tp = p_xyz.sum(axis=2)   # (t_curr, t_prev)
    p_tp = p_tc_tp.sum(axis=0)    # (t_prev)

    te = 0.0
    for tc in range(n_t):
        for tp in range(n_t):
            for sp in range(n_s):
                p = p_xyz[tc,tp,sp]
                if p <= 0:
                    continue
                p_xz = p_tp_sp[tp,sp]
                p_xy = p_tc_tp[tc,tp]
                p_y  = p_tp[tp]
                if p_xz <= 0 or p_xy <= 0 or p_y <= 0:
                    continue
                num = p / p_xz        # p(tc|tp,sp)
                den = p_xy / p_y      # p(tc|tp)
                te += p * np.log2(num/den)
    return float(te)

# ---------- continuous MI (kNN proxy via sklearn) ----------

def mutual_information_continuous_bits(x: np.ndarray, y: np.ndarray, random_state: int=RANDOM_STATE) -> float:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 20:
        return np.nan
    mi_xy = mutual_info_regression(x.reshape(-1,1), y, random_state=random_state)[0]
    mi_yx = mutual_info_regression(y.reshape(-1,1), x, random_state=random_state)[0]
    return float(0.5*(mi_xy+mi_yx)/np.log(2.0))

# ---------- Gaussian predictive information (TE analogue) ----------

def gaussian_predictive_information_bits(source: np.ndarray, target: np.ndarray, source_delay: int, ar_order: int=5, outer_fraction: float=OUTER_TRAIN_FRACTION) -> float:
    s = np.asarray(source, float)
    t = np.asarray(target, float)
    delay = int(max(0, source_delay))
    max_back = max(ar_order, ar_order + delay + 1)

    Y=[]; Xb=[]; Xf=[]
    for tt in range(max_back, len(t)):
        y_t = t[tt]
        t_hist = [t[tt-k] for k in range(1, ar_order+1)]
        s_hist = [s[tt-delay-k] for k in range(1, ar_order+1)]
        Y.append(y_t); Xb.append(t_hist); Xf.append(t_hist+s_hist)
    if len(Y) < 30:
        return np.nan
    y = np.asarray(Y, float)
    Xb = np.asarray(Xb, float)
    Xf = np.asarray(Xf, float)
    valid = np.isfinite(y) & np.isfinite(Xb).all(axis=1) & np.isfinite(Xf).all(axis=1)
    y, Xb, Xf = y[valid], Xb[valid], Xf[valid]

    tr, te = blocked_split(len(y), outer_fraction)
    base = RidgeCV(alphas=np.logspace(-4,3,12)).fit(Xb[tr], y[tr])
    full = RidgeCV(alphas=np.logspace(-4,3,12)).fit(Xf[tr], y[tr])
    e_base = y[te] - base.predict(Xb[te])
    e_full = y[te] - full.predict(Xf[te])
    vb, vf = np.var(e_base), np.var(e_full)
    if not np.isfinite(vb) or not np.isfinite(vf) or vb <= 1e-12 or vf <= 1e-12:
        return np.nan
    return float(max(0.0, 0.5*np.log2(vb/vf)))

# ---------- surrogates + BH-FDR ----------

def surrogate_metric_distribution(source, target, metric_fn: Callable[[np.ndarray,np.ndarray], float], n_surrogates: int=N_SURROGATES, mode: str="circular_shift", random_state: int=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    n = len(target)
    out = np.zeros(n_surrogates, float)
    for i in range(n_surrogates):
        if mode == "shuffle":
            t_s = target[rng.permutation(n)]
        else:
            shift = int(rng.integers(1, max(2,n)))
            t_s = np.roll(target, shift)
        out[i] = metric_fn(source, t_s)
    return out

def empirical_pvalue(obs: float, sur: np.ndarray) -> float:
    s = np.asarray(sur, float)
    s = s[np.isfinite(s)]
    if len(s) == 0 or not np.isfinite(obs):
        return np.nan
    return float((1.0 + np.sum(np.abs(s) >= abs(obs))) / (len(s)+1.0))

def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(1,m+1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(adj, 0, 1)
    q[valid] = out
    return q

# ---------- high-level pair metrics + screening ----------

def compute_pair_metrics(source: np.ndarray, target: np.ndarray, sample_rate: float,
                         preprocess: bool=True, bleach_window: int=301,
                         selection_mode: str="nested",
                         max_lag: int=MAX_LAG, n_bins: int=INFO_BINS, n_surrogates: int=N_SURROGATES,
                         compute_mi_vs_bins: bool=True, max_bins_scan: int=25) -> Dict[str, object]:
    source = np.asarray(source, float)
    target = np.asarray(target, float)

    if preprocess:
        source = prepare_trace(source, bleach_window=bleach_window)
        target = prepare_trace(target, bleach_window=bleach_window)

    if selection_mode == "simple":
        lag_info = scan_optimal_lag(source, target, max_lag=max_lag)
        lag = int(lag_info["lag"])
        xs, ys = _apply_lag(source, target, lag)
        tr, te = blocked_split(len(xs), OUTER_TRAIN_FRACTION)
        r_train = float(np.corrcoef(xs[tr], ys[tr])[0,1]) if np.nanstd(xs[tr])>1e-8 and np.nanstd(ys[tr])>1e-8 else np.nan
        r_test  = float(np.corrcoef(xs[te], ys[te])[0,1]) if np.nanstd(xs[te])>1e-8 and np.nanstd(ys[te])>1e-8 else np.nan
        test_slice = te
    else:
        lag_info = nested_select_lag(source, target, max_lag=max_lag)
        lag = int(lag_info["lag"])
        xs = lag_info["aligned_source"]
        ys = lag_info["aligned_target"]
        test_slice = lag_info["test_slice"]
        r_train = lag_info["train_r"]
        r_test = lag_info["test_r"]

    lag_seconds = lag / sample_rate if np.isfinite(sample_rate) and sample_rate>0 else np.nan
    lags_cc, cc = cross_correlation_curve(source, target, max_lag=max_lag)

    xs_test = xs[test_slice]
    ys_test = ys[test_slice]

    mi_bits_cont = mutual_information_continuous_bits(xs_test, ys_test)
    mi_bits_disc = mutual_information_discrete(xs_test, ys_test, n_bins=n_bins)
    te_bits_disc = transfer_entropy_discrete(source, target, lag=max(1, lag), n_bins=n_bins)
    te_lags, te_curve = transfer_entropy_curve(source, target, max_lag=max_lag, n_bins=n_bins)

    # TE surrogate null (shift source relative to target). This is informative when nonstationarity is controlled.
    te_sur = surrogate_te_distribution(source, target, lag=max(1, lag), n_bins=n_bins, n_surrogates=n_surrogates, mode="circular_shift_source")
    p_te = empirical_pvalue(te_bits_disc, te_sur)
    gpi_bits = gaussian_predictive_information_bits(source, target, source_delay=lag, ar_order=5)

    J = joint_counts_discrete(xs_test, ys_test, n_bins=n_bins)
    xb,_ = _quantile_digitize(xs_test, n_bins); yb,_ = _quantile_digitize(ys_test, n_bins)
    hx = entropy_discrete(xb); hy = entropy_discrete(yb)

    mi_norm = mi_bits_cont / max(min(hx,hy), 1e-8) if np.isfinite(mi_bits_cont) else np.nan
    gpi_norm = gpi_bits / max(hy, 1e-8) if np.isfinite(gpi_bits) else np.nan

    mi_bins = None
    if compute_mi_vs_bins:
        bins = np.arange(2, max(3, max_bins_scan)+1)
        mi_vals = np.array([mutual_information_discrete(xs_test, ys_test, n_bins=int(b)) for b in bins], float)
        mi_bins = {"bins": bins, "mi_bits": mi_vals}

    def _r_test_full(a,b):
        if selection_mode == "nested":
            info = nested_select_lag(a,b,max_lag=max_lag)
            return info["test_r"]
        lag2 = scan_optimal_lag(a,b,max_lag=max_lag)["lag"]
        aa,bb = _apply_lag(a,b,int(lag2))
        tr2,te2 = blocked_split(len(aa), OUTER_TRAIN_FRACTION)
        if np.nanstd(aa[te2])<1e-8 or np.nanstd(bb[te2])<1e-8:
            return 0.0
        return float(np.corrcoef(aa[te2], bb[te2])[0,1])

    sur = surrogate_metric_distribution(source, target, metric_fn=_r_test_full, n_surrogates=n_surrogates, mode="circular_shift")
    p_r = empirical_pvalue(r_test, sur)

    return {
        "lag": lag, "lag_seconds": lag_seconds,
        "r_train": r_train, "r_test": r_test,
        "mi_bits_cont": mi_bits_cont,
        "mi_bits_disc": mi_bits_disc,
        "te_bits_disc": te_bits_disc,
        "te_lags": te_lags,
        "te_curve": te_curve,
        "surrogate_te": te_sur,
        "p_te": p_te,
        "gpi_bits": gpi_bits,
        "hx_bits": hx, "hy_bits": hy,
        "mi_norm": mi_norm, "gpi_norm": gpi_norm,
        "source_trace_aligned": xs, "target_trace_aligned": ys, "test_slice": test_slice,
        "crosscorr_lags": lags_cc, "crosscorr_values": cc,
        "joint_counts": J,
        "mi_vs_bins": mi_bins,
        "p_r": p_r, "surrogate_r": sur,
    }

def screen_sensory_motor_pairs(dataset, sensory_ids: Sequence[str], motor_ids: Sequence[str],
                               top_k: int=20, **pair_kwargs) -> pd.DataFrame:
    idx = dataset.label_to_index()
    rows=[]
    for s in sensory_ids:
        if s not in idx:
            continue
        for m in motor_ids:
            if m not in idx or s == m:
                continue
            met = compute_pair_metrics(dataset.signals[:, idx[s]], dataset.signals[:, idx[m]], sample_rate=dataset.sample_rate, **pair_kwargs)
            rows.append({
                "source_id": s, "target_id": m,
                "lag": met["lag"], "lag_seconds": met["lag_seconds"],
                "r_test": met["r_test"], "p_r": met["p_r"], 
                "mi_bits": met["mi_bits_cont"],
                "te_bits_disc": met["te_bits_disc"],
                "gpi_bits": met["gpi_bits"],
                "mi_norm": met["mi_norm"],
                "gpi_norm": met["gpi_norm"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_r"] = benjamini_hochberg(df["p_r"].to_numpy())
    df["screen_score"] = 0.6*df["r_test"].abs().fillna(0) + 0.25*df["mi_norm"].clip(lower=0).fillna(0) + 0.15*df["gpi_norm"].clip(lower=0).fillna(0)
    df = df.sort_values(["q_r","screen_score","mi_bits","gpi_bits"], ascending=[True,False,False,False]).reset_index(drop=True)
    return df.head(top_k).copy()

def transfer_entropy_curve(source: np.ndarray, target: np.ndarray, max_lag: int=MAX_LAG, n_bins: int=INFO_BINS):
    """Compute discrete TE(source->target) for lag=1..max_lag."""
    vals = np.full(max_lag, np.nan, dtype=float)
    for lag in range(1, max_lag+1):
        vals[lag-1] = transfer_entropy_discrete(source, target, lag=lag, n_bins=n_bins)
    return np.arange(1, max_lag+1), vals

def surrogate_te_distribution(source: np.ndarray, target: np.ndarray, lag: int, n_bins: int, n_surrogates: int=N_SURROGATES,
                              mode: str="circular_shift_source", random_state: int=RANDOM_STATE) -> np.ndarray:
    """Surrogate null for TE(source→target) at a fixed lag.

    We recommend shifting the *source* relative to the target so the target's own history
    remains intact (since TE conditions on target history). This breaks alignment while
    preserving marginal distribution and autocorrelation structure.

    mode:
      - 'circular_shift_source' (default): roll source by random offsets
      - 'shuffle_source': permute source indices (stronger null; breaks autocorr)
    """
    rng = np.random.default_rng(random_state)
    s = np.asarray(source, float)
    t = np.asarray(target, float)
    n = len(t)
    out = np.full(n_surrogates, np.nan, dtype=float)
    for i in range(n_surrogates):
        if mode == "shuffle_source":
            s_s = s[rng.permutation(n)]
        else:
            shift = int(rng.integers(1, max(2, n)))
            s_s = np.roll(s, shift)
        out[i] = transfer_entropy_discrete(s_s, t, lag=int(max(1, lag)), n_bins=int(n_bins))
    return out
