"""Saved instances and the visual instance builder.

The builder edits a deliberately small document rather than the full payload:
a store carries one *mean daily demand*, and the segment x weekday x scenario
expansion is derived from instance-level shares, weekday multipliers and
per-scenario factors. Editing 54 numbers per store by hand would be useless,
and this is exactly how r101_segment_instance.py composes demand anyway.

Two files are written per saved instance:

    exports/instances/<name>.builder.json   the editable document
    exports/instances/<name>.json           the expanded instance payload

The pipeline only ever reads the second one, through the existing
`instance.kind = "payload"` path -- so nothing downstream needed changing.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .layout import INSTANCES_ROOT, now_iso, read_json, write_json

SCHEMA_VERSION = 1
SEGMENTS = ("dry", "fresh", "frozen")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat")

# defaults mirroring r101_segment_instance.py
DEFAULT_SHARES = {"dry": 0.52, "fresh": 0.35, "frozen": 0.13}
DEFAULT_WEEKDAY_MULTIPLIERS = [0.85, 0.90, 1.00, 1.05, 1.10, 1.10]
DEFAULT_SCENARIOS = [
    {"scenario_id": 1, "name": "base", "factor": 1.0},
    {"scenario_id": 2, "name": "growth", "factor": 1.20},
    {"scenario_id": 3, "name": "regional shift", "factor": 1.10},
]
DEFAULT_DEPOT = {"fixed_cost": 5600.0, "marginal_cost": 35.0, "max_size": 30.0}


def new_instance_name() -> str:
    """Names are timestamps, as the feature request specifies."""
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _.-]", "-", str(name)).strip()
    if not cleaned or cleaned.startswith("."):
        raise ValueError(f"invalid instance name: {name!r}")
    return cleaned


# ---------------------------------------------------------------------------
# the builder document
# ---------------------------------------------------------------------------
def empty_builder() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "",
        "source": "blank",
        "depots": [],
        "stores": [],
        "segment_shares": dict(DEFAULT_SHARES),
        "weekday_multipliers": list(DEFAULT_WEEKDAY_MULTIPLIERS),
        "scenarios": [dict(s) for s in DEFAULT_SCENARIOS],
        "second_stage_penalty_factor": 1.5,
    }


def validate_builder(doc: Any) -> List[str]:
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["builder document must be an object"]

    stores = doc.get("stores")
    depots = doc.get("depots")
    if not isinstance(stores, list) or not stores:
        problems.append("at least one store is required")
    if not isinstance(depots, list) or not depots:
        problems.append("at least one depot candidate is required")
    scenarios = doc.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        problems.append("at least one scenario is required")
    if problems:
        return problems

    seen_store, seen_depot, seen_scenario = set(), set(), set()
    for s in stores:
        sid = s.get("store_id")
        if not isinstance(sid, int) or sid <= 0:
            problems.append(f"store id must be a positive integer, got {sid!r}")
        elif sid in seen_store:
            problems.append(f"duplicate store id {sid}")
        else:
            seen_store.add(sid)
        for key in ("x", "y"):
            if not isinstance(s.get(key), (int, float)):
                problems.append(f"store {sid}: {key} must be numeric")
        d = s.get("demand_t_per_day")
        if not isinstance(d, (int, float)) or d < 0:
            problems.append(f"store {sid}: demand_t_per_day must be >= 0, got {d!r}")

    for d in depots:
        did = d.get("depot_id")
        if not isinstance(did, int) or did >= 0:
            problems.append(f"depot id must be a negative integer, got {did!r}")
        elif did in seen_depot:
            problems.append(f"duplicate depot id {did}")
        else:
            seen_depot.add(did)
        for key in ("x", "y", "fixed_cost", "marginal_cost", "max_size"):
            if not isinstance(d.get(key), (int, float)):
                problems.append(f"depot {did}: {key} must be numeric")
        if isinstance(d.get("max_size"), (int, float)) and d["max_size"] <= 0:
            problems.append(f"depot {did}: max_size must be positive")

    for sc in scenarios:
        sid = sc.get("scenario_id")
        if not isinstance(sid, int):
            problems.append(f"scenario id must be an integer, got {sid!r}")
        elif sid in seen_scenario:
            problems.append(f"duplicate scenario id {sid}")
        else:
            seen_scenario.add(sid)
        f = sc.get("factor")
        if not isinstance(f, (int, float)) or f <= 0:
            problems.append(f"scenario {sid}: factor must be > 0, got {f!r}")

    shares = doc.get("segment_shares") or {}
    total = sum(float(shares.get(k, 0.0)) for k in SEGMENTS)
    if abs(total - 1.0) > 1e-6:
        problems.append(f"segment shares must sum to 1, got {total:.4f}")

    mult = doc.get("weekday_multipliers") or []
    if len(mult) != 6 or not all(isinstance(m, (int, float)) and m >= 0 for m in mult):
        problems.append("weekday_multipliers must be 6 non-negative numbers (Mon..Sat)")

    return problems


def builder_to_payload(doc: Dict[str, Any], *, instance_id: str) -> Dict[str, Any]:
    """Expand the editable document into the instance payload the pipeline reads."""
    shares = {k: float((doc.get("segment_shares") or {}).get(k, 0.0)) for k in SEGMENTS}
    mult = [float(m) for m in doc["weekday_multipliers"]]

    demand: List[Dict[str, Any]] = []
    for sc in doc["scenarios"]:
        factor = float(sc["factor"])
        for store in doc["stores"]:
            base = float(store["demand_t_per_day"])
            # a store may override the instance-wide segment split
            local = store.get("segment_shares") or shares
            for seg in SEGMENTS:
                share = float(local.get(seg, shares.get(seg, 0.0)))
                if share <= 0:
                    continue
                for w, m in enumerate(mult):
                    value = base * share * m * factor
                    if value <= 0:
                        continue
                    demand.append({
                        "scenario_id": int(sc["scenario_id"]),
                        "store_id": int(store["store_id"]),
                        "segment": seg,
                        "weekday": w,
                        "demand_t": round(value, 6),
                    })

    return {
        "schema_version": 1,
        "instance_id": instance_id,
        "second_stage_penalty_factor": float(doc.get("second_stage_penalty_factor", 1.5)),
        "depots": [
            {
                "depot_id": int(d["depot_id"]),
                "x": float(d["x"]), "y": float(d["y"]),
                "fixed_cost": float(d["fixed_cost"]),
                "marginal_cost": float(d["marginal_cost"]),
                "max_size": float(d["max_size"]),
            }
            for d in doc["depots"]
        ],
        "stores": [
            {"store_id": int(s["store_id"]), "x": float(s["x"]), "y": float(s["y"])}
            for s in doc["stores"]
        ],
        "scenarios": [
            {"scenario_id": int(s["scenario_id"]), "name": str(s.get("name") or f"scenario {s['scenario_id']}")}
            for s in doc["scenarios"]
        ],
        "demand": demand,
    }


def builder_summary(doc: Dict[str, Any], *, vehicle_capacity_t: float = 25.6) -> Dict[str, Any]:
    """Numbers the editor shows live, including the feasibility warning that
    matters most: a single store whose peak-day demand cannot fit one vehicle
    can never be served, whatever the pattern."""
    mult = [float(m) for m in doc.get("weekday_multipliers") or DEFAULT_WEEKDAY_MULTIPLIERS]
    peak_mult = max(mult) if mult else 1.0
    factors = [float(s.get("factor", 1.0)) for s in doc.get("scenarios") or []] or [1.0]
    peak_factor = max(factors)

    per_scenario = []
    for sc in doc.get("scenarios") or []:
        factor = float(sc.get("factor", 1.0))
        weekly = sum(
            float(s.get("demand_t_per_day", 0.0)) * factor * m
            for s in doc.get("stores") or [] for m in mult
        )
        per_scenario.append({
            "scenario_id": sc.get("scenario_id"),
            "name": sc.get("name"),
            "weekly_demand_t": weekly,
            "daily_mean_t": weekly / 6 if weekly else 0.0,
        })

    oversized = [
        {
            "store_id": s.get("store_id"),
            "peak_day_t": float(s.get("demand_t_per_day", 0.0)) * peak_mult * peak_factor,
        }
        for s in doc.get("stores") or []
        if float(s.get("demand_t_per_day", 0.0)) * peak_mult * peak_factor > vehicle_capacity_t
    ]

    total_capacity = sum(float(d.get("max_size", 0.0)) for d in doc.get("depots") or [])
    worst_daily = max((sc["daily_mean_t"] for sc in per_scenario), default=0.0)

    warnings: List[str] = []
    for o in oversized:
        warnings.append(
            f"store {o['store_id']} needs {o['peak_day_t']:.2f} t on its peak day, "
            f"more than one vehicle ({vehicle_capacity_t} t) can carry")
    if total_capacity and worst_daily > total_capacity:
        warnings.append(
            f"peak scenario needs {worst_daily:.2f} t/day but all depot candidates "
            f"together cap at {total_capacity:.2f} t/day")

    return {
        "n_stores": len(doc.get("stores") or []),
        "n_depots": len(doc.get("depots") or []),
        "n_scenarios": len(doc.get("scenarios") or []),
        "total_max_capacity_t_per_day": total_capacity,
        "per_scenario": per_scenario,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# presets -> builder document
# ---------------------------------------------------------------------------
PRESETS = [
    {"id": "r101", "label": "Solomon R101", "params": ["stores", "index"]},
    {"id": "synthetic", "label": "Synthetic 5-store", "params": []},
]


def builder_from_preset(kind: str, *, k: int = 10, i: int = 1) -> Dict[str, Any]:
    """Load a predefined instance into the editable document.

    The per-store mean daily demand, the segment shares, the weekday shape and
    the scenario factors are recovered from the generated instance, so the
    document reproduces it closely and can then be edited freely. The
    per-store noise the generators add is not representable in this model and
    is therefore averaged out.
    """
    import main as M
    from .params import TransportParams
    from .instance_io import apply_transport_params

    apply_transport_params(M, TransportParams())
    if kind == "synthetic":
        inst = M.construct_test_instance()
    elif kind == "r101":
        from r101_segment_instance import construct_r101_instance
        inst = construct_r101_instance(k=k, i=i, _main_module=M)
    else:
        raise ValueError(f"unknown preset: {kind!r}")

    PC, WD = M.ProductClass, M.Weekday
    scenarios = sorted(inst.S)
    base_sc = scenarios[0]

    def total(sc: int) -> float:
        return sum(inst.demands[sc].values())

    base_total = total(base_sc) or 1.0

    # segment shares over the base scenario
    seg_totals = {
        pc.name.lower(): sum(
            inst.demands[base_sc].get((st, pc, w), 0.0)
            for st in inst.stores for w in WD)
        for pc in PC
    }
    seg_sum = sum(seg_totals.values()) or 1.0
    # round first, then put the rounding drift back into the largest share, so
    # the shares still sum to exactly 1 and pass validation
    shares = {k2: round(v / seg_sum, 6) for k2, v in seg_totals.items()}
    if shares:
        biggest = max(shares, key=lambda k2: shares[k2])
        shares[biggest] = round(shares[biggest] + (1.0 - sum(shares.values())), 6)

    # weekday shape, normalised to mean 1
    day_totals = [
        sum(inst.demands[base_sc].get((st, pc, w), 0.0) for st in inst.stores for pc in PC)
        for w in WD
    ]
    day_mean = (sum(day_totals) / len(day_totals)) or 1.0
    multipliers = [round(d / day_mean, 4) for d in day_totals]

    stores = []
    for st in inst.stores:
        weekly = sum(inst.demands[base_sc].get((st, pc, w), 0.0) for pc in PC for w in WD)
        stores.append({
            "store_id": int(st),
            "x": float(inst.locations[st][0]),
            "y": float(inst.locations[st][1]),
            # mean daily demand: the weekday shape is applied on top
            "demand_t_per_day": round(weekly / 6.0, 4),
        })

    depots = [
        {
            "depot_id": int(d),
            "x": float(inst.locations[d][0]),
            "y": float(inst.locations[d][1]),
            "fixed_cost": round(float(inst.fixed_warehouse_costs.get(d, DEFAULT_DEPOT["fixed_cost"])), 4),
            "marginal_cost": round(float(inst.marginal_warehouse_costs.get(d, DEFAULT_DEPOT["marginal_cost"])), 4),
            "max_size": float(inst.max_warehouse_size.get(d, DEFAULT_DEPOT["max_size"])),
        }
        for d in inst.depots
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "name": "",
        "source": f"{kind} ({inst.instance_name})",
        "depots": depots,
        "stores": stores,
        "segment_shares": shares,
        "weekday_multipliers": multipliers,
        "scenarios": [
            {
                "scenario_id": int(sc),
                "name": f"scenario {sc}",
                "factor": round(total(sc) / base_total, 4),
            }
            for sc in scenarios
        ],
        "second_stage_penalty_factor": float(inst.second_stage_penalty_factor),
    }


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def _paths(name: str) -> tuple[Path, Path]:
    safe = safe_name(name)
    return (INSTANCES_ROOT / f"{safe}.builder.json", INSTANCES_ROOT / f"{safe}.json")


def save_instance(doc: Dict[str, Any], *, name: Optional[str] = None) -> Dict[str, Any]:
    problems = validate_builder(doc)
    if problems:
        raise ValueError("; ".join(problems))

    final_name = safe_name(name or doc.get("name") or new_instance_name())
    builder_path, payload_path = _paths(final_name)
    INSTANCES_ROOT.mkdir(parents=True, exist_ok=True)

    stored = dict(doc)
    stored["name"] = final_name
    stored["saved_at"] = now_iso()
    write_json(builder_path, stored)
    write_json(payload_path, builder_to_payload(stored, instance_id=final_name))

    return describe(final_name)


def describe(name: str) -> Dict[str, Any]:
    builder_path, payload_path = _paths(name)
    doc = read_json(builder_path, {}) or {}
    summary = builder_summary(doc) if doc else {}
    return {
        "name": safe_name(name),
        "saved_at": doc.get("saved_at"),
        "source": doc.get("source"),
        "n_stores": summary.get("n_stores"),
        "n_depots": summary.get("n_depots"),
        "n_scenarios": summary.get("n_scenarios"),
        "warnings": summary.get("warnings", []),
        # relative to the repo root, which is what params.instance.payload_path wants
        "payload_path": str(payload_path.relative_to(INSTANCES_ROOT.parent.parent)),
    }


def list_instances() -> List[Dict[str, Any]]:
    if not INSTANCES_ROOT.exists():
        return []
    rows = []
    for path in sorted(INSTANCES_ROOT.glob("*.builder.json"), reverse=True):
        rows.append(describe(path.name[: -len(".builder.json")]))
    return rows


def load_builder(name: str) -> Dict[str, Any]:
    builder_path, _ = _paths(name)
    doc = read_json(builder_path, None)
    if doc is None:
        raise FileNotFoundError(name)
    return doc


def delete_instance(name: str) -> None:
    builder_path, payload_path = _paths(name)
    if not builder_path.exists() and not payload_path.exists():
        raise FileNotFoundError(name)
    builder_path.unlink(missing_ok=True)
    payload_path.unlink(missing_ok=True)
