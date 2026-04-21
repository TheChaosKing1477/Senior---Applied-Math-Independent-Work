from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List
import pandas as pd

from .third_party import prepare_pipeline_environment
from .discover import discover_resources, choose_best_exported_dir, choose_brainscanner_dir
from .loaders import load_brainscanner_dataset, load_exported_experiment, parse_neuron_class_file, list_exported_ids
from .roles import build_role_table, ids_from_role_table
from .metrics import compute_pair_metrics, screen_sensory_motor_pairs
from .atlas import load_atlas_table, try_load_functional_atlas
from .model import rank_contributors, fit_predictor, fit_population_model, evaluate_population_model
from .plotting import plot_pair_report, plot_prediction_report, plot_population_evaluation, plot_filter_diagnostics

def build_context(discovery, repo_statuses, dataset, atlas_path: Optional[str|Path]=None, atlas_name: str="wild-type") -> Dict[str, Any]:
    atlas_df = None
    atlas_msg = "none"
    if atlas_path is not None:
        try:
            atlas_df = load_atlas_table(atlas_path)
            atlas_msg = "loaded atlas table"
        except Exception as e:
            atlas_msg = f"failed to load atlas: {type(e).__name__}: {e}"
            atlas_df = None
    func_atlas, func_msg = try_load_functional_atlas(repo_statuses, atlas_name=atlas_name)
    return {"atlas_df": atlas_df, "atlas_df_msg": atlas_msg, "functional_atlas": func_atlas, "functional_atlas_msg": func_msg, "dataset": dataset}

def analyze_specific_pair(dataset, source_id: str, target_id: str, selection_mode: str="nested",
                          n_surrogates: int=100, n_bins: int=10, save_dir: Optional[str|Path]=None,
                          show_filter_diagnostics: bool=True) -> Dict[str, Any]:
    """Deep-dive on a single pair.

    Produces the standard v10-style pair report plots. Additionally, if
    show_filter_diagnostics=True, it generates extra plots showing:
    - the estimated photobleach baseline and what is removed,
    - how metrics change from raw→filtered (to detect overfiltering).
    """
    idx = dataset.label_to_index()
    if source_id not in idx or target_id not in idx:
        raise KeyError("source_id or target_id not in dataset labels")

    source_raw = dataset.signals[:, idx[source_id]]
    target_raw = dataset.signals[:, idx[target_id]]

    # Main (filtered) metrics used throughout the pipeline
    met = compute_pair_metrics(source_raw, target_raw, sample_rate=dataset.sample_rate,
                               selection_mode=selection_mode, n_surrogates=n_surrogates, n_bins=n_bins,
                               preprocess=True)
    plot_pair_report(met, source_id, target_id, dataset.sample_rate, save_dir=save_dir)

    if show_filter_diagnostics:
        # Raw metrics without preprocessing (artifact-inclusive baseline)
        met_raw = compute_pair_metrics(source_raw, target_raw, sample_rate=dataset.sample_rate,
                                       selection_mode=selection_mode, n_surrogates=n_surrogates, n_bins=n_bins,
                                       preprocess=False)
        from .preprocess import prepare_trace_with_baseline
        z_s, b_s, corr_s = prepare_trace_with_baseline(source_raw)
        z_t, b_t, corr_t = prepare_trace_with_baseline(target_raw)

        plot_filter_diagnostics(source_raw, target_raw,
                                b_s, b_t,
                                corr_s, corr_t,
                                met_raw, met,
                                source_id, target_id,
                                dataset.sample_rate,
                                save_dir=save_dir)

        # attach raw metrics for programmatic access
        met["raw_metrics"] = met_raw

    return met


def run_pair_of_choice(dataset,
                       neuron_classes,
                       atlas_df=None,
                       source_id: Optional[str]=None,
                       target_id: Optional[str]=None,
                       selection_mode: str="nested",
                       n_surrogates: int=100,
                       n_bins: int=10,
                       save_dir: Optional[str|Path]=None,
                       show_filter_diagnostics: bool=True) -> Dict[str, Any]:
    """Run a deep-dive pair analysis for a user-specified (source_id, target_id)."""
    if source_id is None or target_id is None:
        raise ValueError("source_id and target_id must be provided.")
    return analyze_specific_pair(dataset, str(source_id), str(target_id),
                                 selection_mode=selection_mode,
                                 n_surrogates=n_surrogates,
                                 n_bins=n_bins,
                                 save_dir=save_dir,
                                 show_filter_diagnostics=show_filter_diagnostics)

def pick_best_or_worst_pair(screen_df: pd.DataFrame, mode: str="best") -> Optional[Dict[str,str]]:
    """Pick a pair from a screening table (best or worst)."""
    if screen_df is None or len(screen_df) == 0:
        return None
    mode = str(mode).lower().strip()
    if mode == "best":
        r = screen_df.iloc[0]
        return {"source_id": str(r["source_id"]), "target_id": str(r["target_id"])}
    if mode == "worst":
        df = screen_df.copy()
        if "screen_score" in df.columns:
            df = df.sort_values(["screen_score"], ascending=True)
        else:
            df = df.sort_values(["r_test"], key=lambda s: s.abs(), ascending=True)
        r = df.iloc[0]
        return {"source_id": str(r["source_id"]), "target_id": str(r["target_id"])}
    raise ValueError("mode must be 'best' or 'worst'")

def run_single_dataset(dataset,
                       neuron_classes: Dict[str, Sequence[str]],
                       atlas_df=None,
                       motor_target: Optional[str]=None,
                       include_sensory: bool=True,
                       include_motor: bool=True,
                       include_interneurons: bool=True,
                       include_other: bool=True,
                       include_target_history: bool=True,
                       top_pair_k: int=20,
                       top_contributor_k: int=12,
                       selection_mode: str="nested",
                       pair_selection: str="best",
                       custom_source: Optional[str]=None,
                       custom_target: Optional[str]=None,
                       save_dir: Optional[str|Path]=None) -> Dict[str, Any]:
    """End-to-end single-dataset analysis.

    Adds:
    - pair_selection='best' or 'worst' to choose which screened pair gets a full deep-dive plot report.
    - custom_source/custom_target to deep-dive a user-selected pair instead of the screened pair.
    """
    idx = dataset.label_to_index()
    role_table = build_role_table(dataset.labels, neuron_classes)
    sensory_ids = role_table.loc[role_table["is_sensory"], "neuron_id"].tolist()
    motor_ids = role_table.loc[role_table["is_motor"], "neuron_id"].tolist()
    all_ids = role_table["neuron_id"].tolist()

    screen = screen_sensory_motor_pairs(dataset, sensory_ids, motor_ids, top_k=top_pair_k, selection_mode=selection_mode)
    if screen.empty:
        # fallback: broaden if parsing failed
        if len(sensory_ids) == 0:
            sensory_ids = all_ids
        if len(motor_ids) == 0:
            motor_ids = all_ids
        screen = screen_sensory_motor_pairs(dataset, sensory_ids, motor_ids, top_k=top_pair_k, selection_mode=selection_mode)

    if motor_target is None:
        if not screen.empty:
            motor_target = str(screen.iloc[0]["target_id"])
        elif len(motor_ids) > 0:
            motor_target = str(motor_ids[0])
        elif len(all_ids) > 0:
            motor_target = str(all_ids[0])
        else:
            raise RuntimeError("No labeled neurons found to analyze.")

    pool = ids_from_role_table(role_table,
                               include_sensory=include_sensory,
                               include_motor=include_motor,
                               include_interneurons=include_interneurons,
                               include_other=include_other)
    pool = [x for x in pool if x != motor_target and x in idx]

    contrib = rank_contributors(dataset, motor_target, pool, atlas_df=atlas_df, top_k=max(25, top_contributor_k*2),
                                pair_kwargs={"selection_mode": selection_mode})
    if contrib.empty:
        top_sources = [x for x in all_ids if x != motor_target and x in idx][:top_contributor_k]
    else:
        top_sources = contrib["source_id"].head(top_contributor_k).tolist()

    if (not include_target_history) and len(top_sources)==0:
        any_other = [x for x in all_ids if x != motor_target and x in idx]
        top_sources = any_other[:1]

    fit = fit_predictor(dataset, motor_target, top_sources, include_target_history=include_target_history, atlas_df=atlas_df)
    plot_prediction_report(fit, top_n=top_contributor_k, save_dir=save_dir)

    # Deep-dive pair selection (best/worst or custom)
    pair_metrics = None
    if custom_source is not None and custom_target is not None:
        pair_metrics = analyze_specific_pair(dataset, str(custom_source), str(custom_target),
                                             selection_mode=selection_mode, save_dir=save_dir,
                                             show_filter_diagnostics=True)
    else:
        if not screen.empty:
            choice = pick_best_or_worst_pair(screen, mode=pair_selection)
            if choice is not None:
                pair_metrics = analyze_specific_pair(dataset, choice["source_id"], choice["target_id"],
                                                     selection_mode=selection_mode, save_dir=save_dir,
                                                     show_filter_diagnostics=True)

    # Also deep-dive the top contributor pair (if different from the screened one)
    top_contributor_pair_metrics = None
    if len(top_sources) > 0:
        top_src = str(top_sources[0])
        # avoid duplicating if it is same as primary pair
        if pair_metrics is None or top_src != (pair_metrics.get("source_id", top_src) if isinstance(pair_metrics, dict) else top_src):
            if top_src in idx and motor_target in idx:
                top_contributor_pair_metrics = analyze_specific_pair(
                    dataset, top_src, motor_target,
                    selection_mode=selection_mode, save_dir=save_dir,
                    show_filter_diagnostics=True
                )

    return {
        "role_table": role_table,
        "screen_table": screen,
        "motor_target": motor_target,
        "contributor_table": contrib,
        "predictor_fit": fit,
        "pair_metrics": pair_metrics,
        "top_contributor_pair_metrics": top_contributor_pair_metrics,
    }

def run_toggle_suite(dataset, neuron_classes, atlas_df=None, motor_target: Optional[str]=None,
                     selection_mode: str="nested", top_pair_k: int=20, top_contributor_k: int=12,
                     toggles: Optional[List[dict]]=None) -> pd.DataFrame:
    if toggles is None:
        toggles = [
            {"name":"all_on", "sensory":True, "motor":True, "interneurons":True, "other":True, "target_history":True},
            {"name":"no_interneurons", "sensory":True, "motor":True, "interneurons":False, "other":False, "target_history":True},
            {"name":"network_only_no_AR", "sensory":True, "motor":True, "interneurons":True, "other":True, "target_history":False},
        ]
    rows=[]
    for t in toggles:
        res = run_single_dataset(dataset, neuron_classes, atlas_df=atlas_df, motor_target=motor_target,
                                 include_sensory=t["sensory"], include_motor=t["motor"],
                                 include_interneurons=t["interneurons"], include_other=t.get("other", True),
                                 include_target_history=t["target_history"],
                                 top_pair_k=top_pair_k, top_contributor_k=top_contributor_k,
                                 selection_mode=selection_mode, save_dir=None)
        fit = res["predictor_fit"]
        rows.append({"name": t["name"], "motor_target": res["motor_target"], "include_target_history": t["target_history"],
                     "include_interneurons": t["interneurons"], "linear_R2": fit["linear_r2_test"], "neural_R2": fit["neural_r2_test"],
                     "atlas_only_R2": None if fit.get("atlas_only") is None else fit["atlas_only"]["r2_test"]})
    return pd.DataFrame(rows)

def run_multi_dataset(exported_dir: str|Path,
                      neuron_classes: Dict[str, Sequence[str]],
                      train_ids: Sequence[int],
                      test_id: int,
                      atlas_df=None,
                      target_id: Optional[str]=None,
                      include_sensory: bool=True,
                      include_motor: bool=True,
                      include_interneurons: bool=True,
                      include_other: bool=True,
                      include_target_history: bool=True,
                      top_contributor_k: int=12,
                      selection_mode: str="nested",
                      verbose: bool=True) -> Dict[str, Any]:
    """Train across many exported datasets and evaluate on a hold-out dataset.

    Fixes the common failure:
      KeyError: 'target X not in dataset'

    Strategy:
    1) If the requested target_id is not present in the reference dataset, find a reference dataset
       within train_ids that *does* contain the target.
    2) Filter train_ids to those that contain the target (otherwise training blocks can't be built).
    3) Verify the test dataset contains the target; if not, raise a clear error.

    Prints progress if verbose=True.
    """
    exported_dir = Path(exported_dir)
    loader_fn = load_exported_experiment

    # load test dataset first and check target presence later
    test_ds = loader_fn(exported_dir, test_id)
    test_labels = set(test_ds.label_to_index().keys())

    # choose a provisional reference (first train id)
    ref_id = train_ids[0]
    ref = loader_fn(exported_dir, ref_id)

    role_table = build_role_table(ref.labels, neuron_classes)
    sensory_ids = role_table.loc[role_table["is_sensory"], "neuron_id"].tolist()
    motor_ids = role_table.loc[role_table["is_motor"], "neuron_id"].tolist()
    all_ids = role_table["neuron_id"].tolist()

    # If target_id not specified: pick from a screen on ref (with fallback)
    if target_id is None:
        screen_ref = screen_sensory_motor_pairs(ref, sensory_ids, motor_ids, top_k=25, selection_mode=selection_mode)
        if screen_ref.empty:
            if len(sensory_ids)==0: sensory_ids = all_ids
            if len(motor_ids)==0: motor_ids = all_ids
            screen_ref = screen_sensory_motor_pairs(ref, sensory_ids, motor_ids, top_k=25, selection_mode=selection_mode)
        target_id = str(screen_ref.iloc[0]["target_id"]) if not screen_ref.empty else (motor_ids[0] if len(motor_ids)>0 else all_ids[0])

    # Ensure we have a reference dataset that contains target_id
    def has_target(eid: int) -> bool:
        try:
            ds = loader_fn(exported_dir, eid)
            return target_id in ds.label_to_index()
        except Exception:
            return False

    if target_id not in ref.label_to_index():
        found = None
        for eid in train_ids:
            if has_target(eid):
                found = eid
                break
        if found is None:
            raise KeyError(f"target {target_id} not found in ANY training dataset ids={list(train_ids)[:10]}... (and possibly more).")
        ref_id = found
        ref = loader_fn(exported_dir, ref_id)
        role_table = build_role_table(ref.labels, neuron_classes)
        sensory_ids = role_table.loc[role_table["is_sensory"], "neuron_id"].tolist()
        motor_ids = role_table.loc[role_table["is_motor"], "neuron_id"].tolist()
        all_ids = role_table["neuron_id"].tolist()
        if verbose:
            print(f"[multi] switched reference dataset to {ref_id} because it contains target {target_id}")

    # Filter train_ids to those containing target
    train_ids_ok = [eid for eid in train_ids if has_target(eid)]
    if verbose:
        print(f"[multi] train_ids with target {target_id}: {len(train_ids_ok)}/{len(train_ids)}")

    if len(train_ids_ok) < max(3, len(train_ids)//10):
        print("[multi] warning: very few training datasets contain the target; population model may be unstable.")

    # Confirm test has target
    if target_id not in test_labels:
        raise KeyError(f"target {target_id} not in test dataset {test_id}. Choose a different TEST_ID or target_id.")

    # Build contributor pool on reference dataset
    pool = ids_from_role_table(role_table,
                               include_sensory=include_sensory,
                               include_motor=include_motor,
                               include_interneurons=include_interneurons,
                               include_other=include_other)
    pool = [x for x in pool if x != target_id and x in ref.label_to_index()]

    contrib = rank_contributors(ref, target_id, pool, atlas_df=atlas_df,
                                top_k=max(25, top_contributor_k*2),
                                pair_kwargs={"selection_mode": selection_mode})
    top_sources = contrib["source_id"].head(top_contributor_k).tolist() if not contrib.empty else [x for x in all_ids if x != target_id][:top_contributor_k]

    if verbose:
        print(f"[multi] top_sources (n={len(top_sources)}): {top_sources[:min(10,len(top_sources))]}")

    pop_model = fit_population_model(exported_dir, train_ids_ok, loader_fn, target_id, top_sources,
                                     include_target_history=include_target_history, atlas_df=atlas_df)
    eval_res = evaluate_population_model(pop_model, test_ds)
    plot_population_evaluation(eval_res, target_id)

    if verbose:
        print(f"[multi] Evaluation R²: linear={eval_res['linear_r2']:.3f}, neural={eval_res['neural_r2']:.3f}")

    return {"target_id": target_id,
            "ref_id": ref_id,
            "train_ids_used": train_ids_ok,
            "top_sources": top_sources,
            "contributor_table_ref": contrib,
            "population_model": pop_model,
            "evaluation": eval_res}