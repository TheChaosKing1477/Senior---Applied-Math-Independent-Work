from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
import importlib, sys, subprocess

from .discover import discover_resources

@dataclass
class RepoModuleStatus:
    name: str
    imported: bool
    module: object | None
    message: str
    repo_dir: Path | None

def _find_repo(discovery, hint: str) -> Optional[Path]:
    for p in discovery.repo_dirs:
        if hint.lower() in p.name.lower():
            return p
    return None

def _try_import(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        return True, mod, "imported"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def _try_import_from_repo(module_name: str, repo_dir: Optional[Path]):
    if repo_dir is None:
        ok, mod, msg = _try_import(module_name)
        return RepoModuleStatus(module_name, ok, mod, msg if ok else "repo not discovered; " + msg, None)
    repo_dir = Path(repo_dir)
    for p in [repo_dir, repo_dir.parent]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    ok, mod, msg = _try_import(module_name)
    return RepoModuleStatus(module_name, ok, mod, msg, repo_dir)

def _pip_install(spec: str, editable: bool=False) -> str:
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(spec)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out[-2000:]
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def prepare_pipeline_environment(
    extra_roots: Optional[Sequence[str|Path]]=None,
    unpack_archives: bool=False,
    include_home: bool=True,
    max_depth: int=10,
    attempt_local_pip_install: bool=False,
    editable_install: bool=False,
):
    discovery = discover_resources(extra_roots=extra_roots, unpack_archives=unpack_archives, include_home=include_home, max_depth=max_depth)

    wf_repo = _find_repo(discovery, "worm-functional-connectivity")
    pp_repo = _find_repo(discovery, "pumpprobe")
    wb_repo = _find_repo(discovery, "wormbrain")
    wdm_repo = _find_repo(discovery, "wormdatamodel")

    if attempt_local_pip_install:
        for repo in [wf_repo, pp_repo, wb_repo, wdm_repo]:
            if repo is None:
                continue
            _pip_install(str(repo), editable=editable_install)

    repo_statuses: Dict[str, RepoModuleStatus] = {}
    repo_statuses["wormfunconn"] = _try_import_from_repo("wormfunconn", wf_repo)
    repo_statuses["pumpprobe"] = _try_import_from_repo("pumpprobe", pp_repo)
    repo_statuses["wormbrain"] = _try_import_from_repo("wormbrain", wb_repo)
    repo_statuses["wormdatamodel"] = _try_import_from_repo("wormdatamodel", wdm_repo)

    return discovery, repo_statuses
