from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, List

from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

from .preprocess import prepare_dataset_matrix
from .metrics import compute_pair_metrics, blocked_split
from .atlas import atlas_prior_for_target, atlas_kernel_for_pair, kernel_trace
from .config import MODEL_MAX_LAG, TEST_FRACTION, RANDOM_STATE, USE_ATLAS_KERNEL_BRANCH, ATLAS_WEIGHT, MAX_STIM_FEATURES

def build_stimulus_feature_matrix(dataset, max_features: int=MAX_STIM_FEATURES):
    if dataset.source_type != "exported_data":
        return None, []
    stim_vol = dataset.metadata.get("stim_volume_i")
    stim_neurons = dataset.metadata.get("stim_neurons")
    if stim_vol is None or len(np.ravel(stim_vol)) == 0:
        return None, []
    T = len(dataset.time)
    stim_vol = np.ravel(stim_vol)

    if stim_neurons is None:
        X = np.zeros((T,1), float)
        for v in stim_vol:
            iv = int(v)
            if 0 <= iv < T:
                X[iv,0] = 1.0
        return X, ["stim_any"]

    stim_neurons = np.ravel(stim_neurons)
    idx2lab = {v:k for k,v in dataset.label_to_index().items()}
    pairs=[]
    for v,i in zip(stim_vol, stim_neurons):
        try:
            iv = int(v); ii = int(i)
        except Exception:
            continue
        lab = idx2lab.get(ii, "unknown")
        if 0 <= iv < T:
            pairs.append((iv, lab))
    if not pairs:
        X = np.zeros((T,1), float)
        for v in stim_vol:
            iv = int(v)
            if 0 <= iv < T:
                X[iv,0] = 1.0
        return X, ["stim_any"]

    counts = pd.Series([lab for _,lab in pairs]).value_counts()
    keep = counts.head(max_features).index.tolist()
    X = np.zeros((T, len(keep)+1), float)
    names = [f"stim_{lab}" for lab in keep] + ["stim_other"]
    keep_map = {lab:j for j,lab in enumerate(keep)}
    for iv, lab in pairs:
        j = keep_map.get(lab, len(keep))
        X[iv, j] = 1.0
    return X, names

def _lag_block(trace: np.ndarray, max_lag: int) -> np.ndarray:
    trace = np.asarray(trace, float)
    T = len(trace)
    cols=[]
    for lag in range(0, max_lag+1):
        cols.append(trace[max_lag-lag:T-lag])
    return np.column_stack(cols)

def _make_design(signals: np.ndarray,
                 labels: Sequence[str],
                 target_id: str,
                 source_ids: Sequence[str],
                 max_lag: int=MODEL_MAX_LAG,
                 include_target_history: bool=True,
                 stim_features: Optional[np.ndarray]=None,
                 stim_feature_names: Optional[Sequence[str]]=None,
                 atlas_df=None) -> Dict[str, Any]:
    lab2idx = {lab:i for i,lab in enumerate(labels) if lab}
    if target_id not in lab2idx:
        raise KeyError(f"target_id {target_id} not in labels")
    T = signals.shape[0]
    blocks=[]; feature_names=[]; feature_groups=[]; feature_branches=[]

    if include_target_history:
        tidx = lab2idx[target_id]
        cols=[]
        for lag in range(1, max_lag+1):
            cols.append(signals[max_lag-lag:T-lag, tidx])
            feature_names.append(f"{target_id}_AR_lag{lag}")
            feature_groups.append(target_id)
            feature_branches.append("autoregressive")
        blocks.append(np.column_stack(cols))

    for src in source_ids:
        if src not in lab2idx or src == target_id:
            continue
        sidx = lab2idx[src]
        raw = signals[:, sidx]
        raw_block = _lag_block(raw, max_lag)
        blocks.append(raw_block)
        for lag in range(0, max_lag+1):
            feature_names.append(f"{src}_raw_lag{lag}")
            feature_groups.append(src)
            feature_branches.append("raw")

        if USE_ATLAS_KERNEL_BRANCH and atlas_df is not None:
            kern = atlas_kernel_for_pair(atlas_df, src, target_id)
            if kern is not None:
                filt = kernel_trace(kern, raw, kernel_len=max_lag*4)[:len(raw)]
                ker_block = _lag_block(filt, max_lag)
                blocks.append(ker_block)
                for lag in range(0, max_lag+1):
                    feature_names.append(f"{src}_atlas_lag{lag}")
                    feature_groups.append(src)
                    feature_branches.append("atlas_kernel")

    if stim_features is not None:
        stim_features = np.asarray(stim_features, float)
        for j in range(stim_features.shape[1]):
            block = _lag_block(stim_features[:, j], max_lag)
            blocks.append(block)
            name = stim_feature_names[j] if stim_feature_names else f"stim_{j}"
            for lag in range(0, max_lag+1):
                feature_names.append(f"{name}_lag{lag}")
                feature_groups.append(name)
                feature_branches.append("stimulus")

    X = np.column_stack(blocks) if blocks else np.zeros((T-max_lag, 0))
    y = signals[max_lag:, lab2idx[target_id]]
    return {"X": X, "y": y, "feature_names": feature_names, "feature_groups": feature_groups, "feature_branches": feature_branches}

def _group_permutation_importance(model, X_test, y_test, groups, random_state: int=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_pred = model.predict(X_test)
    base_r2 = r2_score(y_test, base_pred) if len(y_test) > 2 else np.nan
    groups = np.asarray(groups)
    rows=[]
    for grp in pd.unique(groups):
        idx = np.where(groups == grp)[0]
        Xp = X_test.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, idx] = Xp[perm][:, idx]
        pr = model.predict(Xp)
        r2p = r2_score(y_test, pr) if len(y_test) > 2 else np.nan
        delta = (base_r2 - r2p) if np.isfinite(base_r2) and np.isfinite(r2p) else np.nan
        rows.append({"source_id": grp, "delta_r2": delta})
    imp = pd.DataFrame(rows).sort_values("delta_r2", ascending=False).reset_index(drop=True)
    total = imp["delta_r2"].clip(lower=0).sum()
    imp["percent_contribution"] = 100 * imp["delta_r2"].clip(lower=0) / (total if total > 0 else 1.0)
    imp["base_r2"] = base_r2
    return imp

def rank_contributors(dataset, target_id: str, candidate_source_ids: Sequence[str], atlas_df=None, top_k: int=25, pair_kwargs: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
    pair_kwargs = pair_kwargs or {}
    idx = dataset.label_to_index()
    if target_id not in idx:
        raise KeyError(f"target {target_id} not in dataset")
    tidx = idx[target_id]
    rows=[]
    for src in candidate_source_ids:
        if src == target_id or src not in idx:
            continue
        met = compute_pair_metrics(dataset.signals[:, idx[src]], dataset.signals[:, tidx], sample_rate=dataset.sample_rate, **pair_kwargs)
        rows.append({
            "source_id": src,
            "lag": met["lag"],
            "lag_seconds": met["lag_seconds"],
            "r_test": met["r_test"],
            "p_r": met["p_r"],
            "mi_bits": met["mi_bits_cont"],
            "te_bits_disc": met["te_bits_disc"],
            "gpi_bits": met["gpi_bits"],
            "mi_norm": met["mi_norm"],
            "gpi_norm": met["gpi_norm"],
            "screen_score": 0.6*abs(met["r_test"] if np.isfinite(met["r_test"]) else 0) +
                            0.25*max(met["mi_norm"] if np.isfinite(met["mi_norm"]) else 0, 0) +
                            0.15*max(met["gpi_norm"] if np.isfinite(met["gpi_norm"]) else 0, 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if atlas_df is not None:
        pri = atlas_prior_for_target(atlas_df, target_id, df["source_id"].tolist())
        df["atlas_prior"] = df["source_id"].map(pri).fillna(0.0)
        df["combined_score"] = (1-ATLAS_WEIGHT)*df["screen_score"] + ATLAS_WEIGHT*df["atlas_prior"]
    else:
        df["atlas_prior"] = 0.0
        df["combined_score"] = df["screen_score"]
    return df.sort_values(["combined_score","screen_score","mi_bits","gpi_bits"], ascending=False).reset_index(drop=True).head(top_k)

def fit_predictor(dataset, target_id: str, source_ids: Sequence[str], include_target_history: bool=True, preprocess: bool=True, bleach_window: int=301, atlas_df=None) -> Dict[str, Any]:
    signals = prepare_dataset_matrix(dataset.signals, bleach_window=bleach_window) if preprocess else np.asarray(dataset.signals, float)
    stim_X, stim_names = build_stimulus_feature_matrix(dataset)
    design = _make_design(signals, dataset.labels, target_id, list(source_ids), include_target_history=include_target_history,
                          stim_features=stim_X, stim_feature_names=stim_names, atlas_df=atlas_df)

    X = np.asarray(design["X"], float)
    y = np.asarray(design["y"], float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]

    tr, te = blocked_split(len(y), 1.0-TEST_FRACTION)
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]

    n_splits = 5 if len(y_train) >= 60 else max(2, len(y_train)//20)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    linear = ElasticNetCV(l1_ratio=[0.2,0.5,0.8,1.0], alphas=np.logspace(-4,1,20), cv=tscv, random_state=RANDOM_STATE, max_iter=20000)
    linear.fit(X_train, y_train)
    yhat_lin = linear.predict(X_test)
    r2_lin = r2_score(y_test, yhat_lin) if len(y_test) > 2 else np.nan
    imp_lin = _group_permutation_importance(linear, X_test, y_test, design["feature_groups"])

    neural = MLPRegressor(hidden_layer_sizes=(128,64), activation="relu", random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=800)
    neural.fit(X_train, y_train)
    yhat_nn = neural.predict(X_test)
    r2_nn = r2_score(y_test, yhat_nn) if len(y_test) > 2 else np.nan
    imp_nn = _group_permutation_importance(neural, X_test, y_test, design["feature_groups"])


    atlas_only = None
    atlas_cols = [i for i,b in enumerate(design["feature_branches"]) if b in {"autoregressive","stimulus","atlas_kernel"}]
    if len(atlas_cols) > 0 and len(atlas_cols) < X.shape[1]:
        ridge = RidgeCV(alphas=np.logspace(-4,3,12)).fit(X_train[:, atlas_cols], y_train)
        yhat_atlas = ridge.predict(X_test[:, atlas_cols])
        r2_atlas = r2_score(y_test, yhat_atlas) if len(y_test) > 2 else np.nan

        # Atlas contributor importances: grouped permutation importance, restricted to the atlas-only subset.
        atlas_groups = [design["feature_groups"][i] for i in atlas_cols]
        atlas_imp = _group_permutation_importance(ridge, X_test[:, atlas_cols], y_test, atlas_groups)

        atlas_only = {
            "model": ridge,
            "yhat_test": yhat_atlas,
            "r2_test": r2_atlas,
            "contributor_table": atlas_imp,
            "atlas_cols": atlas_cols,
        }


    return {
        "target_id": target_id,
        "source_ids": list(source_ids),
        "design": design,
        "y_test": y_test,
        "linear_model": linear, "linear_yhat_test": yhat_lin, "linear_r2_test": r2_lin, "linear_contributor_table": imp_lin,
        "neural_model": neural, "neural_yhat_test": yhat_nn, "neural_r2_test": r2_nn, "neural_contributor_table": imp_nn,
        "atlas_only": atlas_only,
    }

def fit_population_model(exported_dir, train_ids: Sequence[int], loader_fn, target_id: str, source_ids: Sequence[str],
                         include_target_history: bool=True, atlas_df=None) -> Dict[str, Any]:
    """Fit population models by stacking time points across multiple exported experiments.

    IMPORTANT: Different experiments can have different *available* stimulus regressors and even slightly different
    label sets. This means the design matrix can have different columns if we naively one-hot encode stimuli per
    experiment. To make population fitting well-defined, we align design matrices by feature *name*:

    - Build per-experiment X as a DataFrame with columns=feature_names.
    - Take the union of all feature names across usable experiments.
    - Reindex each X to the union, filling missing features with 0 and dropping extras.
    """
    X_dfs=[]; y_blocks=[]; used_ids=[]
    feature_union=[]  # preserve order of first appearance
    feature_set=set()

    for eid in train_ids:
        ds = loader_fn(exported_dir, eid)
        signals = prepare_dataset_matrix(ds.signals)
        stim_X, stim_names = build_stimulus_feature_matrix(ds)
        design = _make_design(signals, ds.labels, target_id, list(source_ids),
                              include_target_history=include_target_history,
                              stim_features=stim_X, stim_feature_names=stim_names,
                              atlas_df=atlas_df)
        X = np.asarray(design["X"], float); y = np.asarray(design["y"], float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < 30:
            continue

        feat_names = list(design.get("feature_names", []))
        X_df = pd.DataFrame(X[valid], columns=feat_names)
        # Update union of features
        for fn in feat_names:
            if fn not in feature_set:
                feature_set.add(fn)
                feature_union.append(fn)

        X_dfs.append(X_df)
        y_blocks.append(y[valid])
        used_ids.append(eid)

    if not X_dfs:
        raise RuntimeError("No valid training blocks constructed (check target presence / labels).")

    # Align each design matrix to the union of features
    X_blocks=[]
    for X_df in X_dfs:
        X_aligned = X_df.reindex(columns=feature_union, fill_value=0.0)
        X_blocks.append(X_aligned.to_numpy(dtype=np.float32))

    X_train = np.vstack(X_blocks).astype(np.float32)
    y_train = np.concatenate(y_blocks).astype(np.float32)

    n_splits = 5 if len(y_train) >= 200 else max(2, len(y_train)//50)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    linear = ElasticNetCV(l1_ratio=[0.2,0.5,0.8,1.0],
                          alphas=np.logspace(-4,1,20),
                          cv=tscv, random_state=RANDOM_STATE,
                          max_iter=20000).fit(X_train, y_train)

    neural = MLPRegressor(hidden_layer_sizes=(128,64),
                          activation="relu",
                          random_state=RANDOM_STATE,
                          early_stopping=True,
                          validation_fraction=0.15,
                          n_iter_no_change=20,
                          max_iter=800).fit(X_train, y_train)

    # Save a lightweight "design_ref" for downstream plotting/importance computations
    design_ref = {"feature_names": feature_union}

    return {"target_id": target_id,
            "source_ids": list(source_ids),
            "include_target_history": include_target_history,
            "atlas_df": atlas_df,
            "design_ref": design_ref,
            "train_ids_used": used_ids,
            "linear": linear,
            "neural": neural,
            "linear_model": linear,
            "neural_model": neural,
            "feature_names": feature_union}


def evaluate_population_model(pop_model: Dict[str, Any], dataset, preprocess: bool=True) -> Dict[str, Any]:
    signals = prepare_dataset_matrix(dataset.signals) if preprocess else np.asarray(dataset.signals, float)
    stim_X, stim_names = build_stimulus_feature_matrix(dataset)
    design = _make_design(signals, dataset.labels, pop_model["target_id"], pop_model["source_ids"], include_target_history=pop_model["include_target_history"],
                          stim_features=stim_X, stim_feature_names=stim_names, atlas_df=pop_model["atlas_df"])
    X = np.asarray(design["X"], float); y = np.asarray(design["y"], float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    # Align test design columns to the population feature union used during training.
    feat = list(design.get("feature_names", []))
    X_df = pd.DataFrame(X, columns=feat)
    feature_union = list(pop_model.get("feature_names", feat))
    X_aligned = X_df.reindex(columns=feature_union, fill_value=0.0).to_numpy(dtype=float)
    # Backward-compatible model key names.
    lin = pop_model.get("linear_model", pop_model.get("linear"))
    nn = pop_model.get("neural_model", pop_model.get("neural"))
    if lin is None or nn is None:
        raise KeyError("Population model missing linear/neural estimators. Expected keys linear_model/neural_model (or linear/neural).")
    yhat_lin = lin.predict(X_aligned)
    yhat_nn = nn.predict(X_aligned)
    return {"y": y, "linear_yhat": yhat_lin, "neural_yhat": yhat_nn,
            "linear_r2": r2_score(y, yhat_lin) if len(y)>2 else np.nan,
            "neural_r2": r2_score(y, yhat_nn) if len(y)>2 else np.nan,
            "n_samples": int(len(y)), "n_features": int(X_aligned.shape[1])}