from __future__ import annotations
import pandas as pd
from typing import Dict, Sequence, List

def build_role_table(labels: Sequence[str], neuron_classes: Dict[str, Sequence[str]]) -> pd.DataFrame:
    sensory = sorted(set(neuron_classes.get("sensory", [])), key=len, reverse=True)
    motor = sorted(set(neuron_classes.get("motor", [])), key=len, reverse=True)

    def match(label: str, classes: List[str]) -> bool:
        u = label.upper()
        return u in classes or any(u.startswith(c) for c in classes)

    rows=[]; seen=set()
    for lab in labels:
        lab = str(lab).strip()
        if not lab or lab.lower() == "merge" or lab in seen:
            continue
        is_s = match(lab, sensory)
        is_m = match(lab, motor)
        rows.append({
            "neuron_id": lab,
            "is_sensory": bool(is_s),
            "is_motor": bool(is_m),
            "is_interneuron_like": bool(not (is_s or is_m)),
        })
        seen.add(lab)
    return pd.DataFrame(rows).sort_values("neuron_id").reset_index(drop=True)

def ids_from_role_table(role_table: pd.DataFrame,
                        include_sensory: bool=True,
                        include_motor: bool=True,
                        include_interneurons: bool=True,
                        include_other: bool=True) -> List[str]:
    mask = pd.Series(False, index=role_table.index)
    if include_sensory: mask |= role_table["is_sensory"]
    if include_motor: mask |= role_table["is_motor"]
    if include_interneurons or include_other:
        mask |= role_table["is_interneuron_like"]
    return role_table.loc[mask, "neuron_id"].tolist()
