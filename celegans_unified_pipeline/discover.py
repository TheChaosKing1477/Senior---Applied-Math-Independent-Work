from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import os, time, zipfile, tarfile

DEFAULT_SKIP_DIR_NAMES = {
    ".git",".svn",".hg","__pycache__", ".ipynb_checkpoints",
    "node_modules",".conda","miniconda3","anaconda3",
    "Library","AppData","Applications",
}

EXPECTED_ARCHIVE_HINTS = (
    "exported_data",
    "worm-functional-connectivity",
    "wormbrain",
    "wormdatamodel",
    "pumpprobe",
    "Initial_Junang_NewDataZipFile",
    "wbi_data_UPDATEDWITHSPECIFITY",
    "250330",
)

@dataclass
class ResourceDiscovery:
    roots: List[Path] = field(default_factory=list)
    extracted_dirs: List[Path] = field(default_factory=list)

    exported_data_dirs: List[Path] = field(default_factory=list)
    brainscanner_dirs: List[Path] = field(default_factory=list)

    neuron_class_files: List[Path] = field(default_factory=list)
    atlas_tsv_files: List[Path] = field(default_factory=list)
    atlas_pickle_files: List[Path] = field(default_factory=list)

    repo_dirs: List[Path] = field(default_factory=list)

    zip_archives: List[Path] = field(default_factory=list)
    tar_archives: List[Path] = field(default_factory=list)

    def summary(self):
        return {k: [str(x) for x in getattr(self,k)] for k in self.__dataclass_fields__.keys()}

def _normalize_roots(extra_roots: Optional[Sequence[str|Path]]=None, include_home: bool=True) -> List[Path]:
    here = Path.cwd().resolve()
    roots = [here, here.parent, Path("/mnt/data")]
    if include_home:
        try:
            roots.append(Path.home().resolve())
        except Exception:
            pass
    if extra_roots:
        roots.extend(Path(x).expanduser().resolve() for x in extra_roots)
    out=[]; seen=set()
    for r in roots:
        if r.exists() and r not in seen:
            out.append(r); seen.add(r)
    return out

def _walk(root: Path, max_depth: int, skip_dir_names: set[str], time_limit_s: Optional[float], file_limit: Optional[int]):
    start = time.time()
    yielded_files = 0
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        try:
            depth = len(dp.relative_to(root).parts)
        except Exception:
            depth = 0
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in skip_dir_names and not d.startswith(".")]
        yield dp, dirnames, filenames
        yielded_files += len(filenames)
        if time_limit_s is not None and (time.time() - start) > time_limit_s:
            return
        if file_limit is not None and yielded_files >= file_limit:
            return

def _extract_archive(archive: Path, cache_root: Path) -> Optional[Path]:
    dest = cache_root / archive.stem.replace(".tar","")
    if dest.exists():
        return dest
    try:
        dest.mkdir(parents=True, exist_ok=True)
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest)
        return dest
    except Exception:
        return None

def discover_resources(extra_roots: Optional[Sequence[str|Path]]=None,
                       unpack_archives: bool=False,
                       include_home: bool=True,
                       max_depth: int=10,
                       skip_dir_names: Optional[set[str]]=None,
                       time_limit_s: Optional[float]=None,
                       file_limit: Optional[int]=None) -> ResourceDiscovery:
    skip_dir_names = skip_dir_names or set(DEFAULT_SKIP_DIR_NAMES)
    out = ResourceDiscovery(roots=_normalize_roots(extra_roots, include_home=include_home))
    cache_root = out.roots[0] / ".celegans_pipeline_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    search_roots = list(out.roots)

    # archive pass
    for r in out.roots:
        for dp,_,fnames in _walk(r, max_depth=max_depth, skip_dir_names=skip_dir_names,
                                time_limit_s=time_limit_s, file_limit=file_limit):
            for fn in fnames:
                low = fn.lower()
                p = dp/fn
                if not (low.endswith(".zip") or low.endswith(".tar.gz") or low.endswith(".tgz")):
                    continue
                if any(h.lower() in low for h in EXPECTED_ARCHIVE_HINTS):
                    if low.endswith(".zip"): out.zip_archives.append(p)
                    else: out.tar_archives.append(p)
                    if unpack_archives:
                        ex = _extract_archive(p, cache_root)
                        if ex is not None:
                            out.extracted_dirs.append(ex)
                            search_roots.append(ex)

    repo_names = {"worm-functional-connectivity-main","wormbrain-master","wormdatamodel-master","pumpprobe-main",
                  "worm-functional-connectivity","wormbrain","wormdatamodel","pumpprobe"}

    # file pass
    for r in search_roots:
        r = Path(r)
        if not r.exists():
            continue
        for dp,_,fnames in _walk(r, max_depth=max_depth, skip_dir_names=skip_dir_names,
                                time_limit_s=time_limit_s, file_limit=file_limit):
            name = dp.name
            if name == "exported_data" and any(dp.glob("*_gcamp.txt")):
                out.exported_data_dirs.append(dp)
            if name.startswith("BrainScanner_") and (dp/"tmac_output.mat").exists():
                out.brainscanner_dirs.append(dp)
            if name in repo_names:
                out.repo_dirs.append(dp)

            for fn in fnames:
                low = fn.lower()
                p = dp/fn
                if low == "neuron_class.rtf" or (("neuron_class" in low) and (low.endswith(".rtf") or low.endswith(".txt"))):
                    out.neuron_class_files.append(p)
                if low.endswith(".tsv") or low.endswith(".csv"):
                    if "atlas" in low or "funconn" in low or "functional" in low:
                        out.atlas_tsv_files.append(p)
                if low.endswith(".pickle") or low.endswith(".pkl"):
                    out.atlas_pickle_files.append(p)

    def _dedupe(xs):
        out2=[]; seen=set()
        for x in xs:
            if x not in seen:
                out2.append(x); seen.add(x)
        return out2
    for field in ["extracted_dirs","exported_data_dirs","brainscanner_dirs","neuron_class_files","atlas_tsv_files","atlas_pickle_files","repo_dirs","zip_archives","tar_archives"]:
        setattr(out, field, _dedupe(getattr(out, field)))
    return out

def choose_best_exported_dir(discovery: ResourceDiscovery) -> Optional[Path]:
    if not discovery.exported_data_dirs:
        return None
    return sorted(discovery.exported_data_dirs, key=lambda p: len(list(Path(p).glob("*_gcamp.txt"))), reverse=True)[0]

def choose_brainscanner_dir(discovery: ResourceDiscovery, preferred: Optional[str]=None) -> Optional[Path]:
    if not discovery.brainscanner_dirs:
        return None
    if preferred:
        for p in discovery.brainscanner_dirs:
            if preferred in str(p):
                return p
    return discovery.brainscanner_dirs[0]
