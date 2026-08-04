"""Build a main.Instance from a run's instance source, and export it for the UI.

Also handles the instance payload format (schemas/instance-payload.schema.json):
a single JSON file with depots, stores, scenarios and long-form demand.
"""
from __future__ import annotations

import json
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .params import InstanceSource, TransportParams

SEGMENT_NAMES = ("dry", "fresh", "frozen")


# ---------------------------------------------------------------------------
# transport parameters
# ---------------------------------------------------------------------------
def apply_transport_params(main_module, transport: TransportParams) -> Dict[str, float]:
    """Push the run's transport parameters into main.TRANSPORT_PARAMS.

    main.derived_transport_coefficients() defaults to that module-level dict, so
    mutating it in place is what keeps the RLRP arc coefficients and the PATT
    scalarization consistent -- see reference/mosinline_param_alignment.py.
    """
    main_module.TRANSPORT_PARAMS.update({
        "c_km": transport.c_km,
        "c_fuel": transport.c_fuel,
        "eta": transport.eta,
        "theta_TR": transport.theta_TR,
        "W0": transport.W0,
        "Q": transport.Q,
        "lam": transport.lam,
    })
    return main_module.derived_transport_coefficients()


def set_lambda(main_module, instance, lam: float) -> None:
    """Change lambda everywhere it matters, together.

    The PATT side reads instance.weighting_factor_patt; the RLRP side gets it
    folded into c/alpha/gamma via derived_transport_coefficients(). Setting one
    without the other silently de-aligns the two objectives.
    """
    main_module.TRANSPORT_PARAMS["lam"] = lam
    instance.weighting_factor_patt = lam


# ---------------------------------------------------------------------------
# instance construction
# ---------------------------------------------------------------------------
def build_instance(main_module, source: InstanceSource, transport: TransportParams):
    """Returns (instance, source_description)."""
    apply_transport_params(main_module, transport)

    if source.kind == "synthetic":
        inst = main_module.construct_test_instance()
        desc = "synthetic 5-store instance (main.construct_test_instance)"
    elif source.kind == "r101":
        from r101_segment_instance import construct_r101_instance
        inst = construct_r101_instance(k=source.k, i=source.i,
                                       _main_module=main_module)
        desc = f"Solomon R101, {source.k} stores, instance index {source.i}"
    elif source.kind == "payload":
        if not source.payload_path:
            raise ValueError("instance.payload_path is required for kind 'payload'")
        inst = instance_from_payload(main_module, Path(source.payload_path), transport)
        desc = f"uploaded payload {Path(source.payload_path).name}"
    else:
        raise ValueError(f"unknown instance kind: {source.kind!r}")

    # keep the instance's own lambda in step with the run parameters
    inst.weighting_factor_patt = transport.lam
    return inst, desc


def instance_from_payload(main_module, path: Path, transport: TransportParams):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    problems = validate_payload(payload)
    if problems:
        raise ValueError("invalid instance payload:\n  - " + "\n  - ".join(problems))

    PC, WD = main_module.ProductClass, main_module.Weekday
    seg_by_name = {pc.name.lower(): pc for pc in PC}

    depots = [int(d["depot_id"]) for d in payload["depots"]]
    stores = [int(s["store_id"]) for s in payload["stores"]]
    locations: Dict[int, Tuple[float, float]] = {}
    for d in payload["depots"]:
        locations[int(d["depot_id"])] = (float(d["x"]), float(d["y"]))
    for s in payload["stores"]:
        locations[int(s["store_id"])] = (float(s["x"]), float(s["y"]))

    scenarios = [int(s["scenario_id"]) for s in payload["scenarios"]]
    demands: Dict[int, Dict[Tuple[int, Any, Any], float]] = {
        s: {(st, pc, w): 0.0 for st in stores for pc in PC for w in WD}
        for s in scenarios
    }
    for row in payload["demand"]:
        key = (int(row["store_id"]), seg_by_name[str(row["segment"]).lower()],
               WD(int(row["weekday"])))
        demands[int(row["scenario_id"])][key] = float(row["demand_t"])

    nodes = depots + stores
    distances = {
        (a, b): hypot(locations[a][0] - locations[b][0],
                      locations[a][1] - locations[b][1])
        for a in nodes for b in nodes
    }

    return main_module.Instance(
        instance_name=str(payload.get("instance_id") or path.stem),
        depots=depots,
        stores=stores,
        locations=locations,
        demands=demands,
        S=scenarios,
        pattern_operational_costs=None,
        foodwaste_emissions_factor=None,
        distances=distances,
        cost_per_km=transport.c_km,
        vehicle_capacity=transport.Q,
        vehicle_empty_weight=transport.W0,
        marginal_co2_emissions=main_module.derived_transport_coefficients()["marginal_co2_emissions"],
        fixed_warehouse_costs={int(d["depot_id"]): float(d["fixed_cost"]) for d in payload["depots"]},
        marginal_warehouse_costs={int(d["depot_id"]): float(d["marginal_cost"]) for d in payload["depots"]},
        max_warehouse_size={int(d["depot_id"]): float(d["max_size"]) for d in payload["depots"]},
        second_stage_penalty_factor=float(payload.get("second_stage_penalty_factor", 1.5)),
        weighting_factor_patt=transport.lam,
        weighting_factor_rlrp=0.5,
        number_of_realizations=int(payload.get("number_of_realizations", 10)),
    )


def validate_payload(payload: Any) -> List[str]:
    """Human-readable validation, mirroring schemas/instance-payload.schema.json."""
    problems: List[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]
    if payload.get("schema_version") != 1:
        problems.append("schema_version must be 1")

    depots = payload.get("depots")
    stores = payload.get("stores")
    scenarios = payload.get("scenarios")
    demand = payload.get("demand")
    for name, value in (("depots", depots), ("stores", stores),
                        ("scenarios", scenarios), ("demand", demand)):
        if not isinstance(value, list) or not value:
            problems.append(f"{name} must be a non-empty array")
    if problems:
        return problems

    depot_ids, store_ids, scenario_ids = set(), set(), set()
    for d in depots:
        did = d.get("depot_id")
        if not isinstance(did, int) or did >= 0:
            problems.append(f"depot_id must be a negative integer, got {did!r}")
        elif did in depot_ids:
            problems.append(f"duplicate depot_id {did}")
        else:
            depot_ids.add(did)
        for key in ("x", "y", "fixed_cost", "marginal_cost", "max_size"):
            if not isinstance(d.get(key), (int, float)):
                problems.append(f"depot {did}: {key} must be numeric")

    for s in stores:
        sid = s.get("store_id")
        if not isinstance(sid, int) or sid <= 0:
            problems.append(f"store_id must be a positive integer, got {sid!r}")
        elif sid in store_ids:
            problems.append(f"duplicate store_id {sid}")
        else:
            store_ids.add(sid)
        for key in ("x", "y"):
            if not isinstance(s.get(key), (int, float)):
                problems.append(f"store {sid}: {key} must be numeric")

    for s in scenarios:
        sc = s.get("scenario_id")
        if not isinstance(sc, int):
            problems.append(f"scenario_id must be an integer, got {sc!r}")
        elif sc in scenario_ids:
            problems.append(f"duplicate scenario_id {sc}")
        else:
            scenario_ids.add(sc)

    seen = set()
    for row in demand:
        sc, st = row.get("scenario_id"), row.get("store_id")
        seg, wd, val = row.get("segment"), row.get("weekday"), row.get("demand_t")
        if sc not in scenario_ids:
            problems.append(f"demand row references unknown scenario_id {sc!r}")
        if st not in store_ids:
            problems.append(f"demand row references unknown store_id {st!r}")
        if str(seg).lower() not in SEGMENT_NAMES:
            problems.append(f"segment must be one of {SEGMENT_NAMES}, got {seg!r}")
        if not isinstance(wd, int) or not 0 <= wd <= 5:
            problems.append(f"weekday must be an integer 0..5 (Mon..Sat), got {wd!r}")
        if not isinstance(val, (int, float)) or val < 0:
            problems.append(f"demand_t must be a non-negative number, got {val!r}")
        key = (sc, st, str(seg).lower(), wd)
        if key in seen:
            problems.append(f"duplicate demand row for {key}")
        seen.add(key)

    # cap the noise -- a malformed file would otherwise produce thousands of lines
    if len(problems) > 25:
        extra = len(problems) - 25
        problems = problems[:25] + [f"... and {extra} more problems"]
    return problems


# ---------------------------------------------------------------------------
# UI export
# ---------------------------------------------------------------------------
def instance_artifact(main_module, inst, source_description: str) -> Dict[str, Any]:
    """The instance as the frontend wants it: nodes with coordinates plus demand
    summaries per scenario / segment / weekday."""
    PC, WD = main_module.ProductClass, main_module.Weekday
    agg = inst.aggregate_demands_patt()

    nodes = [
        {"id": d, "kind": "depot", "x": inst.locations[d][0], "y": inst.locations[d][1],
         "fixed_cost": inst.fixed_warehouse_costs.get(d),
         "marginal_cost": inst.marginal_warehouse_costs.get(d),
         "max_size": inst.max_warehouse_size.get(d)}
        for d in inst.depots
    ] + [
        {"id": s, "kind": "store", "x": inst.locations[s][0], "y": inst.locations[s][1]}
        for s in inst.stores
    ]

    scenarios = []
    for s in inst.S:
        per_store = {
            st: sum(agg[s][st, w] for w in WD) for st in inst.stores
        }
        per_segment = {
            pc.name.lower(): sum(inst.demands[s].get((st, pc, w), 0.0)
                                 for st in inst.stores for w in WD)
            for pc in PC
        }
        per_weekday = [
            sum(agg[s][st, w] for st in inst.stores) for w in WD
        ]
        scenarios.append({
            "scenario_id": s,
            "weekly_demand_t": sum(per_store.values()),
            "per_store_weekly_t": per_store,
            "per_segment_weekly_t": per_segment,
            "per_weekday_t": per_weekday,
        })

    base = scenarios[0]["weekly_demand_t"] if scenarios else 0.0
    for sc in scenarios:
        sc["ratio_to_first"] = (sc["weekly_demand_t"] / base) if base else None

    return {
        "schema_version": 1,
        "instance_name": inst.instance_name,
        "source": source_description,
        "n_stores": len(inst.stores),
        "n_depots": len(inst.depots),
        "n_scenarios": len(inst.S),
        "vehicle_capacity_t": inst.vehicle_capacity,
        "vehicle_empty_weight_t": inst.vehicle_empty_weight,
        "nodes": nodes,
        "scenarios": scenarios,
        "weekday_names": [w.name.capitalize()[:3] for w in WD],
        "segment_names": [pc.name.lower() for pc in PC],
    }
