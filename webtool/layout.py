"""Run directory layout and JSON artifact IO.

Everything a run produces lives under one directory so the web tool can serve
it, archive it, or delete it as a unit:

    exports/runs/<run-id>/
        manifest.json                  run id, instance, params, status, timings
        logs/                          raw solver logs (one file per stage/round)
        anylogic/                      exported CSVs for the AnyLogic DES
        frontend/                      the artifact contract the UI reads
            overview.json
            instance.json
            pipeline/progress.json
            rounds/<r>/rlrp.json
            rounds/<r>/rlrp.convergence.json
            rounds/<r>/patt/index.json
            rounds/<r>/patt/<scenario>-<depot>.json
            rounds/<r>/sim.json
            feedback/timeline.json
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_ROOT = REPO_ROOT / "exports"
RUNS_ROOT = EXPORT_ROOT / "runs"
INSTANCES_ROOT = EXPORT_ROOT / "instances"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%y%m%d-%H%M%S')}"


# ---------------------------------------------------------------------------
# JSON io
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Make solver output JSON-serializable.

    Handles the shapes that actually show up in our results: enum members,
    tuple dict keys, numpy scalars, non-finite floats, dataclasses.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {_json_key(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    # numpy scalars / anything with .item()
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", ()) == ():
        return _json_safe(item())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "name") and hasattr(value, "value"):   # Enum
        return value.name
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return _json_safe(asdict(value))
    return str(value)


def _json_key(key: Any) -> str:
    if isinstance(key, tuple):
        return ",".join(str(_json_key(k)) for k in key)
    if hasattr(key, "name") and hasattr(key, "value"):       # Enum
        return key.name
    return str(key)


def write_json(path: Path, payload: Any) -> Path:
    """Atomic write, so the UI never reads a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_json_safe(payload), indent=2, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
class RunLayout:
    def __init__(self, run_root: Path):
        self.root = Path(run_root).resolve()

    # --- roots ---
    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def anylogic(self) -> Path:
        return self.root / "anylogic"

    @property
    def frontend(self) -> Path:
        return self.root / "frontend"

    # --- frontend contract ---
    @property
    def overview(self) -> Path:
        return self.frontend / "overview.json"

    @property
    def instance(self) -> Path:
        return self.frontend / "instance.json"

    @property
    def progress(self) -> Path:
        return self.frontend / "pipeline" / "progress.json"

    @property
    def timeline(self) -> Path:
        return self.frontend / "feedback" / "timeline.json"

    def round_dir(self, rnd: int) -> Path:
        return self.frontend / "rounds" / str(rnd)

    def rlrp(self, rnd: int) -> Path:
        return self.round_dir(rnd) / "rlrp.json"

    def rlrp_convergence(self, rnd: int) -> Path:
        return self.round_dir(rnd) / "rlrp.convergence.json"

    def patt_index(self, rnd: int) -> Path:
        return self.round_dir(rnd) / "patt" / "index.json"

    def patt_unit(self, rnd: int, scenario: int, depot: int) -> Path:
        return self.round_dir(rnd) / "patt" / f"{scenario}-{depot}.json"

    def sim(self, rnd: int) -> Path:
        return self.round_dir(rnd) / "sim.json"

    def log(self, name: str) -> Path:
        self.logs.mkdir(parents=True, exist_ok=True)
        return self.logs / name

    # --- lifecycle ---
    def ensure(self) -> "RunLayout":
        for path in (self.root, self.logs, self.anylogic, self.frontend):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def stop_requested(self) -> bool:
        """Cooperative cancellation: the backend touches this file, the
        pipeline checks it at stage/iteration boundaries."""
        return (self.root / "STOP").exists()

    def request_stop(self) -> None:
        (self.root / "STOP").touch()

    # --- manifest helpers ---
    def read_manifest(self) -> Dict[str, Any]:
        return read_json(self.manifest, {}) or {}

    def update_manifest(self, **changes: Any) -> Dict[str, Any]:
        data = self.read_manifest()
        data.update(changes)
        data["updated_at"] = now_iso()
        write_json(self.manifest, data)
        return data


def run_layout(run_id_or_path: str | Path) -> RunLayout:
    path = Path(run_id_or_path)
    if not path.is_absolute() and not path.exists():
        path = RUNS_ROOT / str(run_id_or_path)
    return RunLayout(path)


def list_runs() -> list[Dict[str, Any]]:
    if not RUNS_ROOT.exists():
        return []
    rows = []
    for child in sorted(RUNS_ROOT.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest = read_json(child / "manifest.json", None)
        if manifest:
            rows.append(manifest)
    return rows


def resolve_export_path(relative: str) -> Path:
    """Resolve a path relative to exports/, refusing to escape it."""
    target = (EXPORT_ROOT / relative).resolve()
    if target != EXPORT_ROOT and EXPORT_ROOT not in target.parents:
        raise ValueError(f"path escapes the export root: {relative}")
    return target
