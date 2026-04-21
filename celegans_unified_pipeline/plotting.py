from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from .preprocess import prepare_trace

def plot_pair_report(metrics: dict, source_id: str, target_id: str, sample_rate: float, save_dir: Optional[str|Path]=None):
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    s = np.asarray(metrics["source_trace_aligned"], float)
    t = np.asarray(metrics["target_trace_aligned"], float)
    sr = sample_rate if np.isfinite(sample_rate) and sample_rate>0 else 1.0
    time = np.arange(len(s))/sr

    plt.figure(figsize=(11,4))
    plt.plot(time, s, label=source_id, lw=1.8)
    plt.plot(time, t, label=target_id, lw=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Prepared calcium trace")
    plt.title(f"Aligned traces: {source_id} vs {target_id} (lag={metrics.get('lag_seconds',np.nan):.3f}s)")
    plt.grid(alpha=0.3); plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir/f"{source_id}_vs_{target_id}_timeseries.png", dpi=160, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(5.0,4.6))
    plt.scatter(s, t, s=15, alpha=0.65)
    if len(s) > 5 and np.nanstd(s)>1e-10:
        p = np.polyfit(s, t, 1)
        xs = np.linspace(np.nanmin(s), np.nanmax(s), 200)
        plt.plot(xs, np.polyval(p, xs), "--", lw=2)
    plt.xlabel(source_id); plt.ylabel(target_id)
    plt.title(f"Scatter (r_test={metrics.get('r_test',np.nan):.3f}, MI={metrics.get('mi_bits_cont',np.nan):.3f} bits)")
    plt.grid(alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir/f"{source_id}_vs_{target_id}_scatter.png", dpi=160, bbox_inches="tight")
    plt.show()

    lags = np.asarray(metrics.get("crosscorr_lags", []))
    cc = np.asarray(metrics.get("crosscorr_values", []))
    if len(lags) > 0:
        plt.figure(figsize=(7.0,4.0))
        plt.plot(lags/sr, cc, lw=1.8)
        plt.axvline(0, color="k", lw=0.8)
        plt.xlabel("Lag (s) [positive => target delayed]")
        plt.ylabel("Correlation")
        plt.title("Cross-correlation curve")
        plt.grid(alpha=0.3)
        if save_dir is not None:
            plt.savefig(save_dir/f"{source_id}_vs_{target_id}_crosscorr.png", dpi=160, bbox_inches="tight")
        plt.show()

    J = np.asarray(metrics.get("joint_counts", []), float)
    if J.size > 0:
        plt.figure(figsize=(5.2,4.4))
        plt.imshow(J, origin="lower", aspect="auto")
        plt.colorbar(label="Count")
        plt.xlabel(f"{target_id} bin")
        plt.ylabel(f"{source_id} bin")
        plt.title("Joint distribution (held-out segment)")
        if save_dir is not None:
            plt.savefig(save_dir/f"{source_id}_vs_{target_id}_joint.png", dpi=160, bbox_inches="tight")
        plt.show()

    mivb = metrics.get("mi_vs_bins", None)
    if isinstance(mivb, dict):
        bins = np.asarray(mivb.get("bins", []))
        mi = np.asarray(mivb.get("mi_bits", []))
        if len(bins) > 0:
            plt.figure(figsize=(6.4,4.0))
            plt.plot(bins, mi, marker="o", lw=1.6)
            plt.xlabel("Number of bins")
            plt.ylabel("Discrete MI (bits)")
            plt.title("MI vs bin count (held-out segment)")
            plt.grid(alpha=0.3)
            if save_dir is not None:
                plt.savefig(save_dir/f"{source_id}_vs_{target_id}_mi_vs_bins.png", dpi=160, bbox_inches="tight")
            plt.show()


    # Marginal entropies and MI upper bound (MI <= min(Hs, Ht))
    hx = float(metrics.get("hx_bits", np.nan))
    hy = float(metrics.get("hy_bits", np.nan))
    mi_c = float(metrics.get("mi_bits_cont", np.nan))
    mi_d = float(metrics.get("mi_bits_disc", np.nan))
    te_d = float(metrics.get("te_bits_disc", np.nan))
    gpi  = float(metrics.get("gpi_bits", np.nan))
    mi_max = np.nanmin([hx, hy]) if np.isfinite(hx) and np.isfinite(hy) else np.nan

    # (A) Entropy bars: H(source), H(target) with the MI upper bound overlaid
    if np.isfinite(hx) or np.isfinite(hy):
        plt.figure(figsize=(6.8,4.0))
        labels = [f"H({source_id})", f"H({target_id})"]
        vals = [hx, hy]
        plt.bar(labels, vals)
        if np.isfinite(mi_max):
            plt.axhline(mi_max, linestyle="--", linewidth=2, label=f"MI upper bound min(H)= {mi_max:.3f} bits")
        plt.ylabel("bits")
        plt.title("Marginal entropies (held-out segment)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        if save_dir is not None:
            plt.savefig(save_dir/f"{source_id}_vs_{target_id}_entropies.png", dpi=160, bbox_inches="tight")
        plt.show()

    # (B) Information bars: MI_cont, MI_disc, TE_disc, GPI (all in bits)
    info_names = ["MI_cont", "MI_disc", "TE_disc", "GPI"]
    info_vals  = [mi_c, mi_d, te_d, gpi]
    if np.isfinite(np.asarray(info_vals)).sum() > 0:
        plt.figure(figsize=(7.2,4.0))
        plt.bar(info_names, info_vals)
        if np.isfinite(mi_max):
            plt.axhline(mi_max, linestyle="--", linewidth=2, label=f"MI upper bound min(H)= {mi_max:.3f} bits")
        plt.ylabel("bits")
        plt.title("Information metrics (held-out segment)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        if save_dir is not None:
            plt.savefig(save_dir/f"{source_id}_vs_{target_id}_info_metrics.png", dpi=160, bbox_inches="tight")
        plt.show()

    # (C) Coupling summary bars (unitless): |r_test| and (optional) lag in seconds as a separate axis
    r_test = float(metrics.get("r_test", np.nan))
    lag_s  = float(metrics.get("lag_seconds", np.nan))
    if np.isfinite(r_test) or np.isfinite(lag_s):
        plt.figure(figsize=(7.2,4.0))
        # correlation magnitude is unitless; lag is in seconds (different scale). Plot as two y-axes.
        ax = plt.gca()
        ax.bar(["|r_test|"], [abs(r_test)], label="|r_test| (unitless)")
        ax.set_ylabel("|r_test|")
        ax.set_ylim(0, 1.05)
        ax2 = ax.twinx()
        ax2.bar(["lag (s)"], [lag_s], alpha=0.6, label="lag (s)")
        ax2.set_ylabel("seconds")
        ax.set_title("Coupling summary (held-out)")
        ax.grid(axis="y", alpha=0.3)
        # combined legend
        h1,l1 = ax.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc="upper right")
        if save_dir is not None:
            plt.savefig(save_dir/f"{source_id}_vs_{target_id}_coupling_summary.png", dpi=160, bbox_inches="tight")
        plt.show()
    sur = np.asarray(metrics.get("surrogate_r", []), float)
    if len(sur) > 0 and np.isfinite(sur).sum() > 5:
        plt.figure(figsize=(6.0,4.0))
        plt.hist(sur[np.isfinite(sur)], bins=25, alpha=0.75)
        plt.axvline(metrics.get("r_test", np.nan), color="r", lw=2, label=f"obs r_test={metrics.get('r_test',np.nan):.3f}")
        plt.xlabel("r_test under circular-shift null")
        plt.ylabel("count")
        plt.title(f"Surrogate null (p={metrics.get('p_r',np.nan):.4f})")
        plt.legend(); plt.grid(alpha=0.3)
        if save_dir is not None:
            plt.savefig(save_dir/f"{source_id}_vs_{target_id}_surrogate_hist.png", dpi=160, bbox_inches="tight")
        plt.show()

def _overlay_two(y, yhat, title, label_hat, r2, save_path=None):
    t = np.arange(len(y))
    plt.figure(figsize=(12,4))
    plt.plot(t, y, label="ground truth", lw=2)
    plt.plot(t, yhat, label=f"{label_hat} (R²={r2:.3f})", lw=1.6)
    plt.title(title)
    plt.xlabel("Held-out time index")
    plt.ylabel("Prepared calcium trace")
    plt.legend(); plt.grid(alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()

def plot_prediction_report(fit: dict, top_n: int=12, save_dir: Optional[str|Path]=None):
    """Prediction overlay + contributor bar plots (linear + neural).

    Adds:
    - separate overlays: ground vs linear / atlas / neural
    - neural contributor bar chart (grouped permutation importance)
    """
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(fit["y_test"], float)
    t = np.arange(len(y))

    # Main overlay (all curves)
    plt.figure(figsize=(12,4))
    plt.plot(t, y, label="ground truth", lw=2)
    plt.plot(t, fit["linear_yhat_test"], label=f"linear (R²={fit['linear_r2_test']:.3f})", lw=1.7)
    if fit.get("atlas_only") is not None:
        plt.plot(t, fit["atlas_only"]["yhat_test"], label=f"atlas-only (R²={fit['atlas_only']['r2_test']:.3f})", lw=1.4)
    plt.plot(t, fit["neural_yhat_test"], label=f"neural (R²={fit['neural_r2_test']:.3f})", lw=1.2)
    plt.title(f"Predicted held-out trajectory: {fit['target_id']}")
    plt.xlabel("Held-out time index")
    plt.ylabel("Prepared calcium trace")
    plt.legend(); plt.grid(alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir/f"{fit['target_id']}_prediction_overlay_all.png", dpi=160, bbox_inches="tight")
    plt.show()

    # Separate overlays for clarity
    p_lin = (save_dir/f"{fit['target_id']}_overlay_ground_vs_linear.png") if save_dir is not None else None
    p_atl = (save_dir/f"{fit['target_id']}_overlay_ground_vs_atlas.png") if save_dir is not None else None
    p_nn  = (save_dir/f"{fit['target_id']}_overlay_ground_vs_neural.png") if save_dir is not None else None

    _overlay_two(y, fit["linear_yhat_test"],
                 title=f"Ground truth vs Linear prediction: {fit['target_id']}",
                 label_hat="linear",
                 r2=float(fit["linear_r2_test"]),
                 save_path=p_lin)

    if fit.get("atlas_only") is not None:
        _overlay_two(y, fit["atlas_only"]["yhat_test"],
                     title=f"Ground truth vs Atlas-only prediction: {fit['target_id']}",
                     label_hat="atlas-only",
                     r2=float(fit["atlas_only"]["r2_test"]),
                     save_path=p_atl)

    _overlay_two(y, fit["neural_yhat_test"],
                 title=f"Ground truth vs Neural prediction: {fit['target_id']}",
                 label_hat="neural",
                 r2=float(fit["neural_r2_test"]),
                 save_path=p_nn)

    # Contributors (linear)
    imp = fit["linear_contributor_table"].head(top_n).copy()
    if len(imp) > 0:
        plt.figure(figsize=(8, max(3, 0.35*len(imp)+1)))
        plt.barh(imp["source_id"][::-1], imp["percent_contribution"][::-1])
        plt.xlabel("Grouped permutation importance (%)")
        plt.title("Top contributors (linear model)")
        plt.grid(axis="x", alpha=0.3)
        if save_dir is not None:
            plt.savefig(save_dir/f"{fit['target_id']}_contributors_linear.png", dpi=160, bbox_inches="tight")
        plt.show()

    
    # Contributors (atlas-only)
    if fit.get("atlas_only") is not None and fit["atlas_only"].get("contributor_table") is not None:
        impa = fit["atlas_only"]["contributor_table"].head(top_n).copy()
        if len(impa) > 0:
            plt.figure(figsize=(8, max(3, 0.35*len(impa)+1)))
            plt.barh(impa["source_id"][::-1], impa["percent_contribution"][::-1])
            plt.xlabel("Grouped permutation importance (%)")
            plt.title("Top contributors (atlas-only model)")
            plt.grid(axis="x", alpha=0.3)
            if save_dir is not None:
                plt.savefig(save_dir/f"{fit['target_id']}_contributors_atlas_only.png", dpi=160, bbox_inches="tight")
            plt.show()

    # Contributors (neural)
    impn = fit.get("neural_contributor_table")
    if impn is not None:
        impn = impn.head(top_n).copy()
        if len(impn) > 0:
            plt.figure(figsize=(8, max(3, 0.35*len(impn)+1)))
            plt.barh(impn["source_id"][::-1], impn["percent_contribution"][::-1])
            plt.xlabel("Grouped permutation importance (%)")
            plt.title("Top contributors (neural model)")
            plt.grid(axis="x", alpha=0.3)
            if save_dir is not None:
                plt.savefig(save_dir/f"{fit['target_id']}_contributors_neural.png", dpi=160, bbox_inches="tight")
            plt.show()

def plot_population_evaluation(eval_res: dict, target_id: str, save_dir: Optional[str|Path]=None):
    save_dir = Path(save_dir) if save_dir is not None else None
    y = np.asarray(eval_res["y"], float)
    t = np.arange(len(y))
    plt.figure(figsize=(12,4))
    plt.plot(t, y, label="ground truth", lw=2)
    plt.plot(t, eval_res["linear_yhat"], label=f"linear population (R²={eval_res['linear_r2']:.3f})", lw=1.6)
    plt.plot(t, eval_res["neural_yhat"], label=f"neural population (R²={eval_res['neural_r2']:.3f})", lw=1.2)
    plt.title(f"Population-trained evaluation: {target_id}")
    plt.xlabel("Time index")
    plt.ylabel("Prepared calcium trace")
    plt.legend(); plt.grid(alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir/f"{target_id}_population_eval.png", dpi=160, bbox_inches="tight")
    plt.show()

def plot_filter_diagnostics(source_raw, target_raw,
                            baseline_source, baseline_target,
                            corrected_source, corrected_target,
                            metrics_raw: dict, metrics_filt: dict,
                            source_id: str, target_id: str,
                            sample_rate: float, save_dir: Optional[str|Path]=None):
    """Additional qualitative diagnostics for photobleach correction / overfiltering.

    Produces:
    1) Raw trace with estimated baseline overlay (source + target)
    2) Corrected (baseline-subtracted) traces vs prepared (z-scored) traces
    3) Metric delta bars: r_test / MI_cont / MI_disc / TE_disc / GPI (raw vs filtered)

    The goal is to show whether preprocessing is removing slow drift without destroying
    meaningful fast structure.
    """
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    sr = sample_rate if np.isfinite(sample_rate) and sample_rate>0 else 1.0
    t = np.arange(len(source_raw))/sr

    # 1) Raw + baseline
    plt.figure(figsize=(12,4))
    plt.plot(t, source_raw, lw=1.2, label=f"{source_id} raw")
    plt.plot(t, baseline_source, lw=2.0, label=f"{source_id} baseline (percentile)")
    plt.xlabel("Time (s)"); plt.ylabel("raw fluorescence")
    plt.title(f"Photobleach baseline estimate (source): {source_id}")
    plt.grid(alpha=0.3); plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir/f"{source_id}_baseline_overlay.png", dpi=160, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(t, target_raw, lw=1.2, label=f"{target_id} raw")
    plt.plot(t, baseline_target, lw=2.0, label=f"{target_id} baseline (percentile)")
    plt.xlabel("Time (s)"); plt.ylabel("raw fluorescence")
    plt.title(f"Photobleach baseline estimate (target): {target_id}")
    plt.grid(alpha=0.3); plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir/f"{target_id}_baseline_overlay.png", dpi=160, bbox_inches="tight")
    plt.show()

    # 2) Corrected vs prepared (z-scored)
    plt.figure(figsize=(12,4))
    plt.plot(t, corrected_source, lw=1.4, label=f"{source_id} corrected (raw-baseline)")
    plt.plot(t, prepare_trace(source_raw), lw=1.2, label=f"{source_id} prepared (z-scored)")
    plt.xlabel("Time (s)"); plt.ylabel("amplitude")
    plt.title(f"Correction vs prepared trace (source): {source_id}")
    plt.grid(alpha=0.3); plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir/f"{source_id}_corrected_vs_prepared.png", dpi=160, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(t, corrected_target, lw=1.4, label=f"{target_id} corrected (raw-baseline)")
    plt.plot(t, prepare_trace(target_raw), lw=1.2, label=f"{target_id} prepared (z-scored)")
    plt.xlabel("Time (s)"); plt.ylabel("amplitude")
    plt.title(f"Correction vs prepared trace (target): {target_id}")
    plt.grid(alpha=0.3); plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir/f"{target_id}_corrected_vs_prepared.png", dpi=160, bbox_inches="tight")
    plt.show()

    # 3) Metric deltas (raw vs filtered)
    keys = [("r_test","r_test"),
            ("MI_cont","mi_bits_cont"),
            ("MI_disc","mi_bits_disc"),
            ("TE_disc","te_bits_disc"),
            ("GPI","gpi_bits")]
    raw_vals = []
    filt_vals = []
    labels = []
    for nm,k in keys:
        labels.append(nm)
        raw_vals.append(float(metrics_raw.get(k, np.nan)))
        filt_vals.append(float(metrics_filt.get(k, np.nan)))

    x = np.arange(len(labels))
    plt.figure(figsize=(9.5,4.2))
    plt.bar(x-0.2, raw_vals, width=0.38, label="raw/no preprocess")
    plt.bar(x+0.2, filt_vals, width=0.38, label="filtered/preprocessed")
    plt.xticks(x, labels)
    plt.ylabel("value (units vary)")
    plt.title(f"Effect of artifact removal on metrics: {source_id} → {target_id}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir/f"{source_id}_to_{target_id}_metrics_raw_vs_filtered.png", dpi=160, bbox_inches="tight")
    plt.show()