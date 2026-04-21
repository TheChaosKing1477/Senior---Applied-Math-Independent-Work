from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

def load_atlas_table(path: str|Path) -> pd.DataFrame:
    p = Path(path)
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception:
        df = pd.read_csv(p)
    if df.shape[1] == 3 and set(df.columns) != {"source_id","target_id","atlas_weight"}:
        df.columns = ["source_id","target_id","atlas_weight"]
    return df

def atlas_prior_for_target(atlas_df: Optional[pd.DataFrame], target_id: str, candidate_ids) -> pd.Series:
    if atlas_df is None or "atlas_weight" not in atlas_df.columns:
        return pd.Series(0.0, index=list(candidate_ids), dtype=float)
    sub = atlas_df[(atlas_df["target_id"]==target_id) & (atlas_df["source_id"].isin(candidate_ids))]
    out = pd.Series(0.0, index=list(candidate_ids), dtype=float)
    for _, row in sub.iterrows():
        out.loc[row["source_id"]] = float(row["atlas_weight"])
    if out.max() > 0:
        out = out/out.max()
    return out

def atlas_kernel_for_pair(atlas_df: Optional[pd.DataFrame], source_id: str, target_id: str):
    if atlas_df is None:
        return None
    req = {"source_id","target_id","amplitude","tau_rise","tau_decay1","tau_decay2"}
    if not req.issubset(set(atlas_df.columns)):
        return None
    sub = atlas_df[(atlas_df["source_id"]==source_id) & (atlas_df["target_id"]==target_id)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {k: float(r[k]) for k in ["amplitude","tau_rise","tau_decay1","tau_decay2"]}

def tri_exp_kernel(t, amplitude, tau_rise, tau_decay1, tau_decay2):
    tr = max(float(tau_rise), 1e-8)
    td1 = max(float(tau_decay1), 1e-8)
    td2 = max(float(tau_decay2), 1e-8)
    return float(amplitude) * (1 - np.exp(-t/tr)) * (0.5*np.exp(-t/td1) + 0.5*np.exp(-t/td2))

def kernel_trace(kernel_dict, source_trace, kernel_len=60):
    t = np.arange(kernel_len+1, dtype=float)
    k = tri_exp_kernel(t, kernel_dict["amplitude"], kernel_dict["tau_rise"], kernel_dict["tau_decay1"], kernel_dict["tau_decay2"])
    full = np.convolve(np.asarray(source_trace, float), k, mode="full")
    return full[:len(source_trace)]

def try_load_functional_atlas(repo_statuses: Optional[dict], atlas_name: str="wild-type", atlas_pickle_path: Optional[str|Path]=None):
    status = (repo_statuses or {}).get("wormfunconn")
    if status is None or not getattr(status, "imported", False):
        return None, "wormfunconn not imported"
    try:
        FunctionalAtlas = getattr(status.module, "FunctionalAtlas")
    except Exception:
        return None, "wormfunconn imported but FunctionalAtlas missing"
    try:
        if atlas_pickle_path is not None:
            atlas = FunctionalAtlas.from_file(str(atlas_pickle_path))
            return atlas, "loaded from pickle"
        if hasattr(FunctionalAtlas, "load"):
            atlas = FunctionalAtlas.load(atlas_name)
            return atlas, f"loaded via FunctionalAtlas.load({atlas_name})"
        return None, "FunctionalAtlas available but no loader found"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
