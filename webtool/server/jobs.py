"""Run queue: one active pipeline subprocess at a time.

The backend never solves anything itself -- it builds an argv for run.py and
supervises the process, exactly as the CSPP reference tool does. That keeps the
Gurobi/ALNS work out of the event loop and makes the CLI and the web tool two
views of the same thing.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..layout import (REPO_ROOT, RUNS_ROOT, RunLayout, list_runs, new_run_id,
                      now_iso, read_json, run_layout, write_json)
from ..params import RunParams
from .. import logfilter

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
def _drain_log(process: subprocess.Popen, log_path: Path) -> None:
    """Copy the pipeline's output to its log file, dropping solver boilerplate.

    Runs on its own daemon thread for the life of the process; reading the pipe
    also keeps the child from blocking once the OS buffer fills.
    """
    previous_blank = False
    try:
        with open(log_path, "w", buffering=1) as handle:
            for line in process.stdout or ():
                if logfilter.is_noise(line):
                    continue
                blank = not line.strip()
                if blank and previous_blank:
                    continue
                previous_blank = blank
                handle.write(line)
    except (OSError, ValueError):
        # the process was killed mid-write, or the pipe closed under us
        pass
    finally:
        if process.stdout:
            with contextlib.suppress(Exception):
                process.stdout.close()


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
        try:
            process = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:
            layout.update_manifest(status="failed", reason=str(exc))
            return
        # Gurobi's banner and "Set parameter" echoes reach stdout from the C
        # library, so they can only be stripped out here, between the pipe and
        # the file.
        threading.Thread(target=_drain_log, args=(process, log_path),
                         name=f"log-{run_id}", daemon=True).start()
        _processes[run_id] = process
        layout.update_manifest(status="running", started_at=now_iso(),
                               pid=process.pid)


def _finalize_progress(layout: RunLayout, status: str, reason: str,
                       *, force: bool = False) -> None:
    """Stamp a terminal status into progress.json.

    run.py normally does this itself, but a terminated or crashed process never
    gets the chance, which would leave the UI showing a dead run as `running`
    forever (it trusts the live status over the manifest). `force` overrides an
    already-terminal status, for when the caller knows the intent better than
    the exit code does."""
    progress = read_json(layout.progress, None)
    if not isinstance(progress, dict):
        return
    if not force and str(progress.get("status")) in TERMINAL:
        return
    progress["status"] = status
    progress["finished_at"] = progress.get("finished_at") or now_iso()
    progress["current_stage"] = None
    progress["current_detail"] = None
    result = progress.get("result")
    progress["result"] = {**(result if isinstance(result, dict) else {}),
                          "status": status, "reason": reason}
    write_json(layout.progress, progress)


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
            if process.returncode == 0:
                status, reason = "completed", "pipeline exited cleanly"
            elif layout.stop_requested():
                # we asked it to stop; a non-zero code is the signal, not a fault
                status, reason = "stopped", "stopped on request"
            else:
                status = "failed"
                reason = f"pipeline exited with code {process.returncode}"
            layout.update_manifest(status=status, reason=reason,
                                   finished_at=now_iso())
            _finalize_progress(layout, status, reason)


def reconcile_orphans() -> None:
    """Catch runs left marked active by a backend restart or an outright kill:
    the manifest says running but nothing is tracking it and its pid is gone."""
    for manifest in list_runs():
        run_id = str(manifest.get("run_id") or "")
        if not run_id or str(manifest.get("status")) in TERMINAL:
            continue
        if run_id in _processes or run_id in _queue:
            continue
        pid = manifest.get("pid")
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                continue          # still alive, just not ours to supervise
            except OSError:
                pass
        elif str(manifest.get("status")) == "queued":
            continue              # never started; the queue will pick it up
        layout = run_layout(run_id)
        reason = "process is gone (backend restart or external kill)"
        layout.update_manifest(status="failed", reason=reason,
                               finished_at=now_iso())
        _finalize_progress(layout, "failed", reason)


def poll() -> None:
    """Called periodically by the app so the queue advances even when idle."""
    reconcile_orphans()
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
    manifest = layout.update_manifest(status="stopped", reason="stopped on request",
                                      finished_at=now_iso())
    _finalize_progress(layout, "stopped", "stopped on request", force=True)
    return manifest


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
