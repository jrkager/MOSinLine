"""Run queue: one active pipeline subprocess at a time.

The backend never solves anything itself -- it builds an argv for run.py and
supervises the process, exactly as the CSPP reference tool does. That keeps the
Gurobi/ALNS work out of the event loop and makes the CLI and the web tool two
views of the same thing.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..layout import (REPO_ROOT, RUNS_ROOT, RunLayout, new_run_id, now_iso,
                      read_json, run_layout, write_json)
from ..params import RunParams

TERMINAL = {"completed", "failed", "stopped", "infeasible"}

_lock = threading.RLock()
_processes: Dict[str, subprocess.Popen] = {}
_queue: List[str] = []


# ---------------------------------------------------------------------------
# creating runs
# ---------------------------------------------------------------------------
def create_run(params: RunParams, *, run_name: Optional[str] = None) -> Dict[str, Any]:
    problems = params.validate()
    if problems:
        raise ValueError("; ".join(problems))

    run_id = _sanitize(run_name) if run_name else new_run_id()
    layout = RunLayout(RUNS_ROOT / run_id)
    if layout.root.exists():
        raise FileExistsError(f"run already exists: {run_id}")
    layout.ensure()

    params_path = layout.root / "params.json"
    write_json(params_path, params.to_dict())
    manifest = layout.update_manifest(
        run_id=run_id,
        status="queued",
        created_at=now_iso(),
        instance_kind=params.instance.kind,
        mode=params.feedback.mode,
        params=params.to_dict(),
    )

    with _lock:
        _queue.append(run_id)
    _pump()
    return manifest


def _sanitize(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name.strip())
    if not safe or safe.startswith("."):
        raise ValueError(f"invalid run name: {name!r}")
    return safe


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------
def _pump() -> None:
    """Start the next queued run if nothing is currently active."""
    with _lock:
        _reap()
        if _processes or not _queue:
            return
        run_id = _queue.pop(0)
        layout = run_layout(run_id)
        log_path = layout.log("pipeline.log")
        cmd = [sys.executable, str(REPO_ROOT / "run.py"), "run",
               "--run-name", run_id,
               "--params", str(layout.root / "params.json")]
        handle = open(log_path, "w", buffering=1)
        try:
            process = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=handle,
                stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            handle.close()
            layout.update_manifest(status="failed", reason=str(exc))
            return
        _processes[run_id] = process
        layout.update_manifest(status="running", started_at=now_iso(),
                               pid=process.pid)


def _reap() -> None:
    """Reconcile finished processes with the manifest. run.py writes the real
    status itself; this only catches crashes that never got that far."""
    for run_id, process in list(_processes.items()):
        if process.poll() is None:
            continue
        del _processes[run_id]
        layout = run_layout(run_id)
        manifest = layout.read_manifest()
        if str(manifest.get("status")) not in TERMINAL:
            layout.update_manifest(
                status="failed" if process.returncode else "completed",
                reason=f"pipeline exited with code {process.returncode}",
                finished_at=now_iso())


def poll() -> None:
    """Called periodically by the app so the queue advances even when idle."""
    _pump()


def stop_run(run_id: str) -> Dict[str, Any]:
    layout = run_layout(run_id)
    if not layout.root.exists():
        raise FileNotFoundError(run_id)
    with _lock:
        if run_id in _queue:
            _queue.remove(run_id)
            return layout.update_manifest(status="stopped",
                                          reason="dequeued before starting",
                                          finished_at=now_iso())
        layout.request_stop()          # cooperative: checked between stages
        process = _processes.get(run_id)
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
    with _lock:
        _reap()
    return layout.update_manifest(status="stopped", reason="stopped on request",
                                  finished_at=now_iso())


def delete_run(run_id: str) -> None:
    import shutil
    layout = run_layout(run_id)
    if not layout.root.exists():
        raise FileNotFoundError(run_id)
    status = str(layout.read_manifest().get("status") or "")
    if status not in TERMINAL:
        raise RuntimeError(f"run is still active ({status}); stop it first")
    with _lock:
        if run_id in _queue:
            _queue.remove(run_id)
    shutil.rmtree(layout.root)


def queue_state() -> Dict[str, Any]:
    with _lock:
        _reap()
        return {"active": list(_processes), "queued": list(_queue)}


def shutdown() -> None:
    with _lock:
        for process in _processes.values():
            if process.poll() is None:
                process.terminate()
        _processes.clear()
        _queue.clear()
