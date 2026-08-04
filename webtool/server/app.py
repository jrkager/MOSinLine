"""FastAPI backend for the MOSinLine web tool.

Deliberately thin (cf. WEBTOOL.md section 1.2): create/stop/delete runs, list
them, and serve the artifact tree. All rendering data comes from the JSON files
the pipeline writes, so adding a screen needs no backend change.

    uvicorn webtool.server.app:app --reload
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..layout import (EXPORT_ROOT, RUNS_ROOT, list_runs, now_iso, read_json,
                      resolve_export_path, run_layout)
from ..params import RunParams
from . import jobs

POLL_SECONDS = 2.0


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    stop = asyncio.Event()

    async def poller():
        while not stop.is_set():
            jobs.poll()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)

    task = asyncio.create_task(poller())
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        jobs.shutdown()


app = FastAPI(title="MOSinLine Web Tool", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "updated_at": now_iso(), "queue": jobs.queue_state()}


@app.get("/api/defaults")
def defaults() -> Dict[str, Any]:
    """Default parameters, so the run form can be built from the backend's
    truth rather than a duplicated copy in the frontend."""
    return RunParams().to_dict()


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
@app.get("/api/runs")
def api_list_runs() -> Dict[str, Any]:
    rows = list_runs()
    for row in rows:
        progress = read_json(run_layout(row["run_id"]).progress, {}) or {}
        row["current_round"] = progress.get("current_round")
        row["current_stage"] = progress.get("current_stage")
        row["elapsed_sec"] = progress.get("elapsed_sec")
        row["live_status"] = progress.get("status")
    return {"updated_at": now_iso(), "runs": rows, "queue": jobs.queue_state()}


@app.post("/api/runs")
def api_create_run(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    raw_params = payload.get("params") if isinstance(payload, dict) else None
    try:
        params = RunParams.from_dict(raw_params)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        manifest = jobs.create_run(params, run_name=payload.get("run_name"))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"updated_at": now_iso(), "run": manifest}


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str) -> Dict[str, Any]:
    layout = run_layout(run_id)
    if not layout.root.exists():
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {
        "updated_at": now_iso(),
        "manifest": layout.read_manifest(),
        "progress": read_json(layout.progress, None),
        "overview": read_json(layout.overview, None),
        "instance": read_json(layout.instance, None),
        "timeline": read_json(layout.timeline, None),
    }


@app.post("/api/runs/{run_id}/stop")
def api_stop_run(run_id: str) -> Dict[str, Any]:
    try:
        manifest = jobs.stop_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc
    return {"updated_at": now_iso(), "run": manifest}


@app.delete("/api/runs/{run_id}")
def api_delete_run(run_id: str) -> Dict[str, Any]:
    try:
        jobs.delete_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"updated_at": now_iso(), "deleted": run_id}


@app.get("/api/runs/{run_id}/log")
def api_run_log(run_id: str, name: str = "pipeline.log",
                tail: int = Query(400, ge=1, le=20000)) -> Dict[str, Any]:
    layout = run_layout(run_id)
    path = layout.logs / Path(name).name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"log not found: {name}")
    lines = path.read_text(errors="replace").splitlines()
    return {"name": name, "total_lines": len(lines), "lines": lines[-tail:]}


@app.get("/api/runs/{run_id}/logs")
def api_run_logs(run_id: str) -> Dict[str, Any]:
    layout = run_layout(run_id)
    if not layout.logs.exists():
        return {"logs": []}
    return {"logs": sorted(p.name for p in layout.logs.iterdir() if p.is_file())}


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
@app.get("/api/artifact")
def api_artifact(path: str = Query(..., description="path relative to exports/")):
    """Serve any file from the artifact tree.

    This is the whole data plane: the frontend addresses artifacts by their
    contract path (e.g. runs/<id>/frontend/rounds/2/sim.json) and gets JSON
    back, so new screens need no new endpoint.
    """
    try:
        target = resolve_export_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"artifact not found: {path}")
    if target.suffix == ".json":
        try:
            with open(target, "r", encoding="utf-8") as fh:
                return JSONResponse(json.load(fh))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500,
                                detail=f"artifact is not valid JSON: {path}") from exc
    return FileResponse(target)


@app.get("/api/artifacts")
def api_artifact_index(prefix: str = Query("", description="path relative to exports/")):
    """List artifact paths under a prefix, with mtimes so a client can poll
    cheaply and only re-fetch what changed."""
    try:
        root = resolve_export_path(prefix) if prefix else EXPORT_ROOT
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not root.exists():
        return {"updated_at": now_iso(), "files": []}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append({
                "path": str(path.relative_to(EXPORT_ROOT)),
                "size": stat.st_size,
                "mtime_ms": int(stat.st_mtime * 1000),
            })
    return {"updated_at": now_iso(), "files": files}
