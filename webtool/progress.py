"""Live progress model for the loop view.

The whole point of the web tool (see WEBTOOL.md section 0) is to show the three
stages working together in a loop and where the algorithm currently is, so the
progress artifact is written to be consumed directly by a cycle diagram:

  * `stages` and `edges` describe the *shape* of the loop (static)
  * `current_*` says where execution is right now
  * `rounds[]` is the history: one entry per traversal of the cycle, each with
    per-stage status and the outcome that decided which edge was taken next

Every mutation rewrites `frontend/pipeline/progress.json` atomically, so the UI
can poll a single file.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .layout import RunLayout, now_iso, write_json

SCHEMA_VERSION = 1

STAGE_DEFS: List[Dict[str, str]] = [
    {
        "key": "rlrp",
        "title": "RLRP",
        "subtitle": "Robust Location-Routing",
        "decides": "which depots open, at what size, and which stores each one serves",
        "engine": "Gurobi MIP + scenario decomposition",
    },
    {
        "key": "patt",
        "title": "PATT",
        "subtitle": "Delivery Pattern Planning",
        "decides": "a weekly delivery pattern per store, delivery quantities and routes",
        "engine": "ALNS (patterns) + LNS (routing)",
    },
    {
        "key": "sim",
        "title": "SIM",
        "subtitle": "Discrete-Event Simulation",
        "decides": "what actually happens when the plan is executed for a year",
        "engine": "Python DES port of the AnyLogic model",
    },
]

EDGE_DEFS: List[Dict[str, str]] = [
    {
        "id": "rlrp->patt",
        "source": "rlrp",
        "target": "patt",
        "kind": "handoff",
        "label": "store assignment + depot capacity",
        "detail": "one PATT instance per (open depot, scenario)",
    },
    {
        "id": "patt->sim",
        "source": "patt",
        "target": "sim",
        "kind": "handoff",
        "label": "patterns, order-up-to levels, routes",
        "detail": "the plan the simulation then executes for 52 weeks",
    },
    {
        "id": "patt->rlrp",
        "source": "patt",
        "target": "rlrp",
        "kind": "feedback",
        "label": "depot capacity shortfall -> scale demand up, re-solve RLRP",
        "detail": "PATT needs more daily throughput than the depot RLRP sized",
    },
    {
        "id": "sim->patt",
        "source": "sim",
        "target": "patt",
        "kind": "feedback",
        "label": "simulated KPIs miss the prediction -> adjust lambda, re-solve PATT",
        "detail": "stockout-driven misses lower lambda (more weight on cost)",
    },
]

# terminal run states
TERMINAL = {"completed", "failed", "stopped", "infeasible"}


class ProgressTracker:
    def __init__(self, layout: RunLayout, *, run_id: str, instance_name: str,
                 params: Dict[str, Any], log_limit: int = 200):
        self.layout = layout
        self._t0 = time.time()
        self._log_limit = log_limit
        self._state: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "instance_name": instance_name,
            "status": "pending",
            "mode": params.get("feedback", {}).get("mode", "full"),
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "finished_at": None,
            "elapsed_sec": 0.0,
            "current_round": None,
            "current_stage": None,
            "current_detail": None,
            "current_edge": None,
            "stages": STAGE_DEFS,
            "edges": EDGE_DEFS,
            "rounds": [],
            "result": None,
            "log": [],
        }
        self._flush()

    # ------------------------------------------------------------- io ----
    def _flush(self) -> None:
        self._state["updated_at"] = now_iso()
        self._state["elapsed_sec"] = round(time.time() - self._t0, 2)
        write_json(self.layout.progress, self._state)

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    # ---------------------------------------------------------- rounds ----
    def _round(self, rnd: int) -> Dict[str, Any]:
        for entry in self._state["rounds"]:
            if entry["round"] == rnd:
                return entry
        entry = {
            "round": rnd,
            "status": "running",
            "started_at": now_iso(),
            "finished_at": None,
            "stages": {
                s["key"]: {
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "headline": None,
                    "reused": False,
                    "units": [],
                }
                for s in STAGE_DEFS
            },
            "outcome": None,
        }
        self._state["rounds"].append(entry)
        return entry

    def start_round(self, rnd: int) -> None:
        self._round(rnd)
        self._state["current_round"] = rnd
        self._state["status"] = "running"
        self._state["current_edge"] = None
        self.log(f"--- Round {rnd} ---")
        self._flush()

    def finish_round(self, rnd: int, *, outcome_kind: str, reason: str,
                     detail: Optional[Dict[str, Any]] = None,
                     edge_id: Optional[str] = None) -> None:
        """outcome_kind: accepted | feedback_capacity | feedback_lambda |
        aborted | infeasible"""
        entry = self._round(rnd)
        entry["status"] = "completed"
        entry["finished_at"] = now_iso()
        entry["outcome"] = {
            "kind": outcome_kind,
            "reason": reason,
            "detail": detail or {},
            "edge_id": edge_id,
        }
        self._state["current_edge"] = edge_id
        self.log(f"Round {rnd}: {reason}")
        self._flush()

    # ---------------------------------------------------------- stages ----
    def start_stage(self, rnd: int, stage: str, *, detail: Optional[str] = None,
                    reused: bool = False) -> None:
        st = self._round(rnd)["stages"][stage]
        st["status"] = "reused" if reused else "running"
        st["reused"] = reused
        st["started_at"] = now_iso()
        self._state["current_round"] = rnd
        self._state["current_stage"] = stage
        self._state["current_detail"] = detail
        self._state["current_edge"] = None
        self._flush()

    def stage_detail(self, detail: Optional[str]) -> None:
        self._state["current_detail"] = detail
        self._flush()

    def finish_stage(self, rnd: int, stage: str, *, headline: str,
                     status: str = "completed") -> None:
        st = self._round(rnd)["stages"][stage]
        st["status"] = status
        st["finished_at"] = now_iso()
        st["headline"] = headline
        self._state["current_stage"] = None
        self._state["current_detail"] = None
        self.log(f"{stage.upper()}: {headline}")
        self._flush()

    def set_stage_units(self, rnd: int, stage: str, units: List[Dict[str, Any]]) -> None:
        """PATT is solved per (scenario, depot); the diagram shows these as
        sub-tiles inside the PATT node."""
        self._round(rnd)["stages"][stage]["units"] = units
        self._flush()

    def update_unit(self, rnd: int, stage: str, unit_id: str, **changes: Any) -> None:
        for unit in self._round(rnd)["stages"][stage]["units"]:
            if unit.get("id") == unit_id:
                unit.update(changes)
                break
        self._flush()

    # ------------------------------------------------------- terminal ----
    def finish(self, status: str, *, result: Optional[Dict[str, Any]] = None,
               reason: Optional[str] = None) -> None:
        self._state["status"] = status
        self._state["finished_at"] = now_iso()
        self._state["current_stage"] = None
        self._state["current_detail"] = None
        self._state["result"] = {"status": status, "reason": reason, **(result or {})}
        if reason:
            self.log(reason)
        self._flush()

    # ------------------------------------------------------------ log ----
    def log(self, message: str) -> None:
        self._state["log"].append({"t": now_iso(),
                                   "elapsed_sec": round(time.time() - self._t0, 1),
                                   "message": message})
        if len(self._state["log"]) > self._log_limit:
            del self._state["log"][:-self._log_limit]
