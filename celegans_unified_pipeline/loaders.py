from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import scipy.io as sio
import re

@dataclass
class DatasetRecord:
    name: str
    source_type: str  # 'brainscanner' or 'exported_data'
    signals: np.ndarray  # (T,N)
    labels: List[str]    # len N
    time: np.ndarray     # len T
    sample_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def label_to_index(self) -> Dict[str, int]:
        out={}
        for i, lab in enumerate(self.labels):
            if lab and lab not in out:
                out[lab]=i
        return out

def _mat_to_str(x):
    if x is None: return ""
    if isinstance(x, bytes): return x.decode("utf-8", errors="ignore").strip()
    if isinstance(x, str): return x.strip()
    if isinstance(x, np.ndarray):
        if x.size == 0: return ""
        try: return str(x.squeeze().item()).strip()
        except Exception: return str(x).strip()
    return str(x).strip()

def _sample_rate(time_vec):
    t = np.asarray(time_vec, float)
    if t.size < 2:
        return np.nan
    dt = np.nanmedian(np.diff(t))
    return 1.0/dt if np.isfinite(dt) and dt>0 else np.nan

def parse_neuron_class_file(path: str|Path) -> Dict[str, List[str]]:
    """Robust parser for neuron_class.rtf (handles real RTF).

    Returns dict: {'motor': [...], 'sensory': [...]}
    """
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    # Some RTF writers encode quotes and punctuation as hex escapes like \\'91 before tokens.
    # These escapes can glue digits to neuron IDs (e.g., \\'91RMEL), breaking word-boundary regex.
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)


    def extract_list(txt: str, key: str) -> List[str]:
        m = re.search(rf"{key}\s*=\s*\[(.*?)\]", txt, flags=re.S|re.I)
        if m is None:
            return []
        s = m.group(1).replace("'", " ")
        toks = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", s)
        return sorted(set(toks))

    motor = extract_list(raw, "motor_neuron_classes")
    sensory = extract_list(raw, "sensory_neuron_classes")
    if motor or sensory:
        return {"motor": motor, "sensory": sensory}

    # Strip RTF control words and braces
    txt = re.sub(r"[{}]", " ", raw)
    txt = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", txt)
    txt = re.sub(r"\s+", " ", txt)

    motor = extract_list(txt, "motor_neuron_classes")
    sensory = extract_list(txt, "sensory_neuron_classes")
    if motor or sensory:
        return {"motor": motor, "sensory": sensory}

    # fallback: tokens near keys
    def extract_near(txt2: str, key: str, window: int=800) -> List[str]:
        idx = txt2.lower().find(key.lower())
        if idx < 0:
            return []
        snippet = txt2[max(0, idx-window): idx+window]
        toks = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", snippet)
        toks = [t for t in toks if t.upper() not in {"MOTOR_NEURON_CLASSES","SENSORY_NEURON_CLASSES"}]
        return sorted(set(toks))

    return {"motor": extract_near(txt, "motor_neuron_classes"), "sensory": extract_near(txt, "sensory_neuron_classes")}

def load_brainscanner_dataset(dataset_dir: str|Path, signal_key: str="a") -> DatasetRecord:
    d = Path(dataset_dir)
    tmac = sio.loadmat(d/"tmac_output.mat", squeeze_me=True, struct_as_record=False)
    assign = list(d.glob("calcium_to_multicolor_assignments*.mat"))
    if not assign:
        raise FileNotFoundError(f"No calcium_to_multicolor_assignments*.mat in {d}")
    lab = sio.loadmat(assign[0], squeeze_me=True, struct_as_record=False)

    if signal_key not in tmac:
        for k in ["a","gcamp","signals","F"]:
            if k in tmac:
                signal_key = k
                break
        if signal_key not in tmac:
            raise KeyError(f"No signal key {signal_key} in tmac_output.mat. Keys: {list(tmac.keys())}")

    signals = np.asarray(tmac[signal_key], float)
    labels_struct = lab.get("labels", None)
    human_labels = getattr(labels_struct, "human_labels", None) if labels_struct is not None else None
    labels = [_mat_to_str(x) for x in np.ravel(human_labels)] if human_labels is not None else [f"n{i}" for i in range(signals.shape[1])]

    sr = float(tmac.get("sample_rate", np.nan))
    time = np.arange(signals.shape[0], dtype=float)/(sr if np.isfinite(sr) and sr>0 else 1.0)

    return DatasetRecord(
        name=d.name,
        source_type="brainscanner",
        signals=signals,
        labels=labels,
        time=time,
        sample_rate=sr,
        metadata={"tmac_keys":[k for k in tmac.keys() if not k.startswith("__")], "path": str(d)},
    )

def load_exported_experiment(exported_dir: str|Path, exp_id: int) -> DatasetRecord:
    d = Path(exported_dir)
    gcamp = np.loadtxt(d/f"{exp_id}_gcamp.txt", dtype=float)
    labels = [x.strip() for x in Path(d/f"{exp_id}_labels.txt").read_text(encoding="utf-8", errors="ignore").splitlines()]
    time = np.loadtxt(d/f"{exp_id}_t.txt", dtype=float) if (d/f"{exp_id}_t.txt").exists() else np.arange(gcamp.shape[0], dtype=float)
    stim_neurons = np.loadtxt(d/f"{exp_id}_stim_neurons.txt", dtype=float) if (d/f"{exp_id}_stim_neurons.txt").exists() else None
    stim_volume_i = np.loadtxt(d/f"{exp_id}_stim_volume_i.txt", dtype=float) if (d/f"{exp_id}_stim_volume_i.txt").exists() else None
    ds_name = Path(d/f"{exp_id}_ds_name.txt").read_text(encoding="utf-8", errors="ignore").strip() if (d/f"{exp_id}_ds_name.txt").exists() else str(exp_id)

    return DatasetRecord(
        name=f"exported_{exp_id}",
        source_type="exported_data",
        signals=np.asarray(gcamp, float),
        labels=labels,
        time=np.asarray(time, float),
        sample_rate=_sample_rate(time),
        metadata={"stim_neurons": stim_neurons, "stim_volume_i": stim_volume_i, "dataset_name": ds_name, "path": str(d)},
    )

def list_exported_ids(exported_dir: str|Path) -> List[int]:
    d = Path(exported_dir)
    out=[]
    for p in d.glob("*_gcamp.txt"):
        try:
            out.append(int(p.name.split("_")[0]))
        except Exception:
            pass
    return sorted(set(out))