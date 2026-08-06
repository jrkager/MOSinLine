"""The RLRP -> PATT -> SIM loop, instrumented for the web tool.

This is the orchestration that main.py sketches and run_pipeline_B.py partially
implements, assembled into one parameterised, artifact-emitting entry point.

Loop structure (one traversal of the cycle == one "round"):

    RLRP  ---- assignment + depot capacity ---->  PATT  ---- plan ---->  SIM
      ^                                            |                     |
      |         capacity shortfall                 |     KPI miss        |
      +--------- scale demand up -------------------+   lower lambda -----+

A round ends with exactly one outcome, which names the edge taken next:
  accepted           - the plan survived the simulation, stop
  feedback_capacity  - PATT could not fit in the depot RLRP sized -> back to RLRP
  feedback_lambda    - the simulation disagreed with PATT's prediction -> back to PATT
  infeasible/aborted - out of rounds, or stopped
"""
from __future__ import annotations

import time
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .layout import RunLayout, now_iso, write_json
from .params import RunParams
from .progress import ProgressTracker
from . import instance_io

SEG2PROD = {"fresh": "A", "dry": "B", "frozen": "C"}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


class StopRequested(Exception):
    pass


# ===========================================================================
# stage 1 - RLRP
# ===========================================================================
def solve_rlrp(M, inst, params: RunParams, layout: RunLayout, rnd: int):
    """Returns (RLRPResult, artifact dict)."""
    import rlrp.classes as rlrp_classes
    from rlrp.algorithm import ourAlgorithm as rlrp_main

    logfile = layout.log(f"round{rnd}_rlrp.log")
    # create_rlrp_instance_data() reads this off the instance
    inst.second_stage_penalty_factor = params.rlrp.second_stage_penalty_factor
    alg_params = M.create_rlrp_instance_data(
        inst,
        gap=params.rlrp.gap,
        timelimit=params.rlrp.timelimit,
        logfile_name=str(logfile),
    )
    alg_params.HEURTIMELIMIT = params.rlrp.heur_timelimit
    alg_params.n_threads = params.rlrp.n_threads
    # The RLRP sees one aggregate number per store, with no weekday structure:
    # create_rlrp_instance_data() hardcodes the average (option 1), which is the
    # intended semantics. Override here rather than editing main.py so the
    # aggregation stays selectable without touching the glue layer.
    if params.rlrp.demand_aggregation != 1:
        alg_params.app.inst.beta_k_j = inst.aggregate_demands_rlrp(
            option=params.rlrp.demand_aggregation)

    t0 = time.time()
    try:
        stats, ret = rlrp_main(params=alg_params)
    finally:
        if getattr(alg_params, "logfile", None):
            try:
                alg_params.logfile.close()
            except Exception:
                pass
    runtime = time.time() - t0
    result = M.RLRPResult(ret)

    scenarios = []
    for s in inst.S:
        sizes = result.depot_sizes.get(s, {})
        assign = result.customer_depot_assignment.get(s, {})
        depots = []
        for d in inst.depots:
            stores = list(assign.get(d, []))
            size = float(sizes.get(d, 0.0))
            depots.append({
                "depot_id": d,
                "open": bool(stores) or size > 0,
                "size_t_per_day": size,
                "stores": stores,
                "n_stores": len(stores),
            })
        scenarios.append({"scenario_id": s, "depots": depots})

    open_depots = {
        s: [d["depot_id"] for d in sc["depots"] if d["open"]]
        for s, sc in zip(inst.S, scenarios)
    }
    artifact = {
        "schema_version": 1,
        "round": rnd,
        "runtime_sec": round(runtime, 2),
        "cost": _finite(getattr(stats, "COST", None)),
        "reached_gap": _finite(getattr(stats, "reached_gap", None)),
        "iterations": _int(getattr(stats, "ITERATIONS", None)),
        "n_opened": _int(getattr(stats, "OPENED", None)),
        "time_master_sec": _finite(getattr(stats, "TIME_MASTER", None)),
        "time_second_stage_sec": _finite(getattr(stats, "TIME_SS", None)),
        "scenarios": scenarios,
        "open_depots_per_scenario": open_depots,
        "log_file": logfile.name,
    }
    return result, artifact


# ===========================================================================
# stage 2 - PATT
# ===========================================================================
def build_patt_units(M, alns, inst, rlrp_result, params: RunParams) -> List[Dict[str, Any]]:
    """One unit per (scenario, open depot). Constructs the ALNS object so the
    capacity check and the solve can share it."""
    units = []
    for s in inst.S:
        for depot_id in inst.depots:
            assigned = rlrp_result.customer_depot_assignment.get(s, {}).get(depot_id, [])
            if not assigned:
                continue
            fname = M.create_patt_instance_data(inst, rlrp_result,
                                                depot_id=depot_id, scenario=s)
            if fname is None:
                continue
            try:
                idata = alns.load_instance(fname)
                algorithm_params = alns.default_algorithm_params()
                algorithm_params.update(params.patt.algorithm_params or {})
                a = alns.ComprehensiveALNS(idata, algorithm_params)
                handoff = _read_handoff(fname)
            finally:
                M.delete_patt_instance_file(fname)
            units.append({
                "id": f"{s}-{depot_id}",
                "scenario_id": s,
                "depot_id": depot_id,
                "n_stores": len(assigned),
                "stores": list(assigned),
                "alns": a,
                "instance_data": idata,
                "handoff": handoff,
                "status": "pending",
            })
    return units


def capacity_check(unit: Dict[str, Any], rlrp_result, params: RunParams) -> Dict[str, Any]:
    """Can the depot RLRP sized actually carry the minimum PATT throughput?

    Straight from run_pipeline_B.depot_check. The RLRP sized the depot for each
    store's *average daily* demand. PATT delivers to a store on only a few days
    a week, so a delivery carries several days of demand at once -- plus a bit
    more, because the (R,S) policy refills to the order-up-to level and so
    re-orders whatever expired on the shelf since the last visit. That surplus
    grows as delivery frequency drops.

    So: take the pattern with the smallest total weekly delivery for each store,
    sum those minima over the depot's stores, and spread over the 6 delivery
    days. If even that lower bound does not fit the depot, no combination of
    patterns can, and there is no point running PATT at all.
    """
    a = unit["alns"]
    s, depot_id = unit["scenario_id"], unit["depot_id"]
    cap = float(rlrp_result.depot_sizes[s].get(depot_id, 0.0))
    avg_demand = float(sum(unit["instance_data"]["daily_demands"]))
    min_delivered = sum(
        min(sum(a.p_frt.get((f, r, t), 0.0) for t in range(6))
            for r in a.feasible_patterns_by_store[f])
        for f in a.stores
    )
    need = min_delivered / 6.0
    ok = need * params.feedback.check_margin <= cap + 1e-9
    return {
        "scenario_id": s,
        "depot_id": depot_id,
        "capacity_t_per_day": cap,
        "required_t_per_day": need,
        "avg_demand_t_per_day": avg_demand,
        "headroom": (cap / need) if need > 1e-9 else None,
        "feasible": ok,
    }


def solve_patt_unit(unit: Dict[str, Any], params: RunParams, layout: RunLayout,
                    rnd: int, tracker: ProgressTracker, lam: float):
    a = unit["alns"]
    trajectory: List[Dict[str, float]] = []

    def progress_cb(iteration, cost, elapsed):
        trajectory.append({"iteration": int(iteration),
                           "cost": float(cost),
                           "elapsed_sec": round(float(elapsed), 2)})
        tracker.update_unit(rnd, "patt", unit["id"],
                            status="running",
                            iterations=int(iteration),
                            best_cost=float(cost),
                            trajectory=trajectory[-60:])

    def stop_cb():
        return layout.stop_requested()

    logfile = layout.log(f"round{rnd}_patt_{unit['id']}.log")
    t0 = time.time()
    with open(logfile, "w") as fh, redirect_stdout(fh):
        solution = a.run_alns(max_iterations=params.patt.max_iterations,
                              time_limit=params.patt.time_limit,
                              progress_cb=progress_cb,
                              stop_cb=stop_cb)
    runtime = time.time() - t0
    if layout.stop_requested():
        raise StopRequested()

    artifact = patt_artifact(a, solution, unit, lam, runtime, trajectory, logfile.name)
    unit["solution"] = solution
    unit["artifact"] = artifact
    return solution, artifact


def patt_artifact(a, solution, unit, lam, runtime, trajectory, log_name) -> Dict[str, Any]:
    ev = solution.evaluator
    sm = unit["instance_data"]["store_id_mapping"]

    # --- pattern calendar: the clearest view of a PATT solution ---
    stores = []
    for f in sorted(solution.pattern_assignments):
        r = solution.pattern_assignments[f]
        bits = list(a.patterns[r])
        qty = [float(solution.p_frt.get((f, r, t), 0.0)) for t in range(6)]
        stores.append({
            "internal_id": f,
            "store_id": sm[f],
            "x": a.loc[f][0],
            "y": a.loc[f][1],
            "pattern_id": int(r),
            "pattern": bits,
            "frequency": int(sum(bits)),
            "delivery_t": qty,
            "weekly_t": float(a.D_f[f]),
            "waste_fraction": float(ev.waste_fractions[f, r]),
            "stockout_fraction": float(ev.stockout_fractions[f, r]),
            "order_up_to": {seg: float(a.S_fsr.get((f, seg, r), 0.0))
                            for seg in SEG2PROD},
        })

    # --- routes per weekday ---
    routes_by_day = []
    for day in range(6):
        day_routes = []
        for vehicle_id, route in sorted(solution.routes_by_day.get(day, {}).items()):
            if len(route) <= 2:
                continue
            loads = [float(solution.p_frt.get((n, solution.pattern_assignments[n], day), 0.0))
                     if n != 0 else 0.0 for n in route]
            arcs = [float(ev.delta[route[i], route[i + 1]]) for i in range(len(route) - 1)]
            day_routes.append({
                "vehicle_id": int(vehicle_id),
                "stops": [int(n) for n in route],
                "stop_store_ids": [None if n == 0 else sm[n] for n in route],
                "coords": [[a.loc[n][0], a.loc[n][1]] for n in route],
                "loads_t": loads,
                "departure_load_t": sum(loads),
                "arc_lengths_km": arcs,
                "distance_km": sum(arcs),
            })
        routes_by_day.append({
            "day": day,
            "day_name": DAY_NAMES[day],
            "routes": day_routes,
            "n_vehicles": len(day_routes),
            "distance_km": sum(r["distance_km"] for r in day_routes),
            "delivered_t": sum(r["departure_load_t"] for r in day_routes),
        })

    # --- model-predicted KPIs (what SIM is checked against) ---
    demand = sum(a.D_f[f] for f in a.stores)
    waste = sum(a.D_f[f] * ev.waste_fractions[f, solution.pattern_assignments[f]]
                for f in a.stores)
    stockout = sum(a.D_f[f] * ev.stockout_fractions[f, solution.pattern_assignments[f]]
                   for f in a.stores)
    fw_co2 = sum(ev.calculate_fw_emission(f, solution.pattern_assignments[f])
                 for f in a.stores)

    tr_cost = tr_co2 = km = 0.0
    for day in range(6):
        for _, route in solution.routes_by_day.get(day, {}).items():
            if len(route) <= 2:
                continue
            loads = {n: solution.p_frt.get((n, solution.pattern_assignments[n], day), 0.0)
                     for n in route if n != 0}
            for i in range(len(route) - 1):
                d = ev.delta[route[i], route[i + 1]]
                carried = sum(loads[j] for j in route[i + 1:] if j != 0)
                fuel = ev.eta * (ev.W0 + carried) * d
                km += d
                tr_cost += ev.c_km * d + ev.c_fuel * fuel
                tr_co2 += ev.theta_TR * fuel

    try:
        operator_table = a.build_operator_performance_table().to_dict(orient="records")
    except Exception:
        operator_table = []

    violations = solution.validate_constraints() or []
    return {
        "schema_version": 1,
        "id": unit["id"],
        "scenario_id": unit["scenario_id"],
        "depot_id": unit["depot_id"],
        "lambda": lam,
        "runtime_sec": round(runtime, 2),
        "iterations_run": len(trajectory),
        "objective": float(solution.cost),
        "pattern_cost": float(solution.pattern_cost),
        "routing_cost": float(solution.routing_cost),
        "depot": {"x": a.loc[0][0], "y": a.loc[0][1]},
        "stores": stores,
        "routes_by_day": routes_by_day,
        "predicted": {
            "demand_t": demand,
            "waste_t": waste,
            "stockout_t": stockout,
            # Conservation identity: what must have been delivered to satisfy
            # this much demand given the waste and stockout the model predicts.
            "delivered_t": demand - stockout + waste,
            # Sum of p_frt: the conditional-mean delivery quantities the routes
            # actually carry. Reported for cross-checking only.
            "delivered_cond_t": sum(
                float(solution.p_frt.get((f, solution.pattern_assignments[f], t), 0.0))
                for f in a.stores for t in range(6)
            ),
            "waste_pct": 100 * waste / demand if demand else 0.0,
            "stockout_pct": 100 * stockout / demand if demand else 0.0,
            "fw_co2_kg_per_week": fw_co2,
            "transport_co2_kg_per_week": tr_co2,
            "transport_cost_per_week": tr_cost,
            "km_per_week": km,
        },
        "frequency_histogram": _histogram([s["frequency"] for s in stores]),
        "operator_performance": operator_table,
        "convergence": trajectory,
        "violations": [str(v) for v in violations],
        "feasible": not violations,
        "log_file": log_name,
    }


# ===========================================================================
# stage 3 - SIM
# ===========================================================================
def run_sim(units: List[Dict[str, Any]], params: RunParams, rnd: int) -> Dict[str, Any]:
    from sim_des_port import DESPort, StoreCfg, kpis_to_row

    per_unit = []
    for unit in units:
        a, solution = unit["alns"], unit["solution"]
        stores_cfg = {}
        for f in sorted(solution.pattern_assignments):
            r = solution.pattern_assignments[f]
            mu = {SEG2PROD[seg]: [a.mu_fst[f].get(seg, {t: 0.0 for t in range(6)})[t]
                                  for t in range(6)] for seg in SEG2PROD}
            S = {SEG2PROD[seg]: a.S_fsr.get((f, seg, r), 0.0) for seg in SEG2PROD}
            flags = [solution.p_frt.get((f, r, t), 0.0) > 1e-9 for t in range(6)]
            stores_cfg[f - 1] = StoreCfg(xy=a.loc[f], mu=mu, S=S, plan_flag=flags)

        routes_by_day = {}
        for day in range(6):
            lst = []
            for _, rt in sorted(solution.routes_by_day.get(day, {}).items()):
                custs = [n - 1 for n in rt if n != 0]
                if custs:
                    lst.append(custs)
            routes_by_day[day] = lst

        rows = [_patt_row(unit["artifact"])]
        weekly = {}
        for variant in params.sim.variants:
            aggregated = None
            for run_i in range(params.sim.runs):
                port = DESPort(stores_cfg, routes_by_day, a.loc[0], variant)
                kpis = port.run(record_weeks=params.sim.weeks,
                                warmup_weeks=params.sim.warmup_weeks,
                                seed=params.sim.seed + 37 * run_i)
                row = kpis_to_row(kpis, f"Variant {variant}")
                if aggregated is None:
                    aggregated = dict(row)
                else:
                    for key, value in row.items():
                        if isinstance(value, (int, float)):
                            aggregated[key] = (aggregated[key] + value) / 2
                if run_i == 0:
                    weekly[str(variant)] = _weekly_series(kpis)
            aggregated["variant"] = variant
            # kpis_to_row reports demand_u as a total over the recorded weeks
            # while every other column is per week -- and the PATT row is weekly.
            # Normalise so the absolute columns are comparable across rows.
            dem = float(aggregated.get("demand_u") or 0.0) / max(params.sim.weeks, 1)
            aggregated["demand_u"] = dem
            # same conservation identity as the PATT row
            aggregated["delivered_u"] = dem * (
                1.0 - float(aggregated["stockout%"]) / 100.0
                + float(aggregated["waste%"]) / 100.0)
            rows.append(aggregated)

        per_unit.append({
            "id": unit["id"],
            "scenario_id": unit["scenario_id"],
            "depot_id": unit["depot_id"],
            "rows": rows,
            "weekly": weekly,
        })

    return {
        "schema_version": 1,
        "round": rnd,
        "weeks": params.sim.weeks,
        "runs_per_variant": params.sim.runs,
        "warmup_weeks": params.sim.warmup_weeks,
        "variants": params.sim.variants,
        "units": per_unit,
        "columns": ["run", "waste%", "stockout%", "FW_CO2_kg/wk", "TR_CO2_kg/wk",
                    "TR_cost/wk", "km/wk", "cancel/wk", "drop/wk", "piggy_u/wk"],
    }


def _patt_row(patt_art: Dict[str, Any]) -> Dict[str, Any]:
    p = patt_art["predicted"]
    return {
        "run": "PATT model", "variant": None,
        "demand_u": p["demand_t"],
        "delivered_u": p["delivered_t"],
        "delivered_cond_u": p["delivered_cond_t"],
        "waste%": p["waste_pct"], "wasteA%": None,
        "stockout%": p["stockout_pct"],
        "FW_CO2_kg/wk": p["fw_co2_kg_per_week"],
        "TR_CO2_kg/wk": p["transport_co2_kg_per_week"],
        "TR_cost/wk": p["transport_cost_per_week"],
        "km/wk": p["km_per_week"],
        "cancel/wk": 0.0, "drop/wk": 0.0, "piggy_u/wk": 0.0,
    }


def _weekly_series(kpis) -> Dict[str, Any]:
    """Weekly waste/stockout series if the KPI object exposes one."""
    out: Dict[str, Any] = {}
    for attr, key in (("weekly_waste", "waste"), ("weekly_lost", "stockout"),
                      ("weekly_demand", "demand")):
        series = getattr(kpis, attr, None)
        if series is not None:
            try:
                out[key] = [float(v) for v in series]
            except TypeError:
                pass
    return out


def sim_verdict(sim_artifact: Dict[str, Any], params: RunParams) -> Dict[str, Any]:
    """Did the simulation confirm the PATT plan?

    Placeholder criterion pending a modelling decision (WEBTOOL.md 9.4): accept
    when the reference variant's waste% and stockout% do not exceed the PATT
    model's own prediction by more than the configured tolerance.
    """
    ref = params.feedback.reference_variant
    checks: List[Dict[str, Any]] = []
    # -inf so that a plan the simulation likes reports its true (negative) margin
    # rather than a misleading +0.00
    worst_waste = worst_stockout = float("-inf")
    for unit in sim_artifact["units"]:
        predicted = next((r for r in unit["rows"] if r["run"] == "PATT model"), None)
        simulated = next((r for r in unit["rows"] if r.get("variant") == ref), None)
        if not predicted or not simulated:
            continue
        d_waste = simulated["waste%"] - predicted["waste%"]
        d_stockout = simulated["stockout%"] - predicted["stockout%"]
        worst_waste = max(worst_waste, d_waste)
        worst_stockout = max(worst_stockout, d_stockout)
        checks.append({
            "id": unit["id"],
            "predicted_waste_pct": predicted["waste%"],
            "simulated_waste_pct": simulated["waste%"],
            "delta_waste_pp": d_waste,
            "predicted_stockout_pct": predicted["stockout%"],
            "simulated_stockout_pct": simulated["stockout%"],
            "delta_stockout_pp": d_stockout,
            "waste_ok": d_waste <= params.feedback.waste_tolerance_pp,
            "stockout_ok": d_stockout <= params.feedback.stockout_tolerance_pp,
        })

    if not checks:
        return {
            "accepted": False, "reference_variant": ref,
            "worst_delta_waste_pp": None, "worst_delta_stockout_pp": None,
            "driver": "no-data",
            "reason": f"no simulated rows for the reference variant {ref}",
            "tolerances": {"waste_pp": params.feedback.waste_tolerance_pp,
                           "stockout_pp": params.feedback.stockout_tolerance_pp},
            "checks": [],
        }

    waste_ok = worst_waste <= params.feedback.waste_tolerance_pp
    stockout_ok = worst_stockout <= params.feedback.stockout_tolerance_pp
    accepted = waste_ok and stockout_ok
    if accepted:
        reason = (f"simulated KPIs within tolerance of the PATT prediction "
                  f"(worst waste {worst_waste:+.2f} pp, "
                  f"worst stockout {worst_stockout:+.2f} pp)")
    elif not stockout_ok:
        reason = (f"simulated stockout exceeds the prediction by "
                  f"{worst_stockout:.2f} pp (> {params.feedback.stockout_tolerance_pp} pp)")
    else:
        reason = (f"simulated waste exceeds the prediction by "
                  f"{worst_waste:.2f} pp (> {params.feedback.waste_tolerance_pp} pp)")

    return {
        "accepted": accepted,
        "reference_variant": ref,
        "worst_delta_waste_pp": worst_waste,
        "worst_delta_stockout_pp": worst_stockout,
        "driver": None if accepted else ("stockout" if not stockout_ok else "waste"),
        "reason": reason,
        "tolerances": {"waste_pp": params.feedback.waste_tolerance_pp,
                       "stockout_pp": params.feedback.stockout_tolerance_pp},
        "checks": checks,
    }


# ===========================================================================
# the loop
# ===========================================================================
def run_pipeline(params: RunParams, layout: RunLayout,
                 tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    import main as M
    import patt.alns as alns

    layout.ensure()
    problems = params.validate()
    if problems:
        raise ValueError("invalid run parameters:\n  - " + "\n  - ".join(problems))

    inst, source_desc = instance_io.build_instance(M, params.instance, params.transport)
    write_json(layout.instance, instance_io.instance_artifact(M, inst, source_desc))

    if tracker is None:
        tracker = ProgressTracker(layout, run_id=layout.run_id,
                                  instance_name=inst.instance_name,
                                  params=params.to_dict())
    tracker.log(f"Instance: {inst.instance_name} ({source_desc})")

    rlrp_inst = deepcopy(inst)     # RLRP sees demand that feedback may scale up
    patt_inst = deepcopy(inst)     # PATT sees the true demand, and lambda feedback
    lam = params.transport.lam

    timeline: List[Dict[str, Any]] = []
    rlrp_result = None
    rlrp_artifact = None
    rnd = 0
    outcome: Dict[str, Any] = {"status": "infeasible",
                               "reason": "no rounds executed"}

    try:
        while rnd < params.feedback.max_rounds:
            rnd += 1
            tracker.start_round(rnd)
            _check_stop(layout)

            # ---------------- stage 1: RLRP ----------------
            reuse_rlrp = rlrp_result is not None and timeline and \
                timeline[-1]["outcome"]["kind"] == "feedback_lambda"
            tracker.start_stage(rnd, "rlrp", reused=reuse_rlrp,
                                detail=None if reuse_rlrp else "solving MIP")
            if reuse_rlrp:
                tracker.finish_stage(rnd, "rlrp", status="reused",
                                     headline="reused from the previous round "
                                              "(only lambda changed)")
            else:
                rlrp_result, rlrp_artifact = solve_rlrp(M, rlrp_inst, params, layout, rnd)
                opened = {s: [d["depot_id"] for d in sc["depots"] if d["open"]]
                          for s, sc in zip(rlrp_inst.S,
                                           rlrp_artifact["scenarios"])}
                headline = ", ".join(
                    f"s{s}: depot {' '.join(str(d) for d in ds) or 'none'}"
                    for s, ds in opened.items())
                tracker.finish_stage(rnd, "rlrp", headline=headline)
            write_json(layout.rlrp(rnd), {**(rlrp_artifact or {}), "round": rnd,
                                          "reused": reuse_rlrp})
            _check_stop(layout)

            # ---------------- edge: RLRP -> PATT ----------------
            instance_io.set_lambda(M, patt_inst, lam)
            units = build_patt_units(M, alns, patt_inst, rlrp_result, params)
            if not units:
                outcome = {"status": "infeasible",
                           "reason": "RLRP assigned no stores to any depot"}
                tracker.finish_round(rnd, outcome_kind="infeasible",
                                     reason=outcome["reason"])
                break

            tracker.set_stage_units(rnd, "patt", [
                {"id": u["id"], "scenario_id": u["scenario_id"],
                 "depot_id": u["depot_id"], "n_stores": u["n_stores"],
                 "status": "pending"} for u in units])
            write_json(layout.patt_index(rnd), {
                "round": rnd, "lambda": lam,
                "units": [{"id": u["id"], "scenario_id": u["scenario_id"],
                           "depot_id": u["depot_id"], "n_stores": u["n_stores"],
                           "stores": u["stores"], "handoff": u["handoff"]}
                          for u in units],
            })

            # ---------------- feedback edge PATT -> RLRP: capacity ----------
            if params.feedback.mode in {"capacity", "full"}:
                checks = [capacity_check(u, rlrp_result, params) for u in units]
                infeasible = [c for c in checks if not c["feasible"]]
                if infeasible:
                    scaled = _apply_capacity_feedback(
                        rlrp_inst, rlrp_result, infeasible, params)
                    detail = {"checks": checks, "rescaled": scaled}
                    reason = (f"{len(infeasible)} depot/scenario combination(s) cannot "
                              f"carry the minimum PATT throughput; scaling RLRP demand up")
                    tracker.finish_stage(rnd, "patt", status="blocked",
                                         headline="capacity shortfall - not solved")
                    tracker.finish_round(rnd, outcome_kind="feedback_capacity",
                                         reason=reason, detail=detail,
                                         edge_id="patt->rlrp")
                    timeline.append(_timeline_entry(rnd, lam, "feedback_capacity",
                                                    reason, detail))
                    write_json(layout.timeline, {"rounds": timeline})
                    outcome = {"status": "infeasible",
                               "reason": "capacity feedback did not converge"}
                    continue

            # ---------------- stage 2: PATT ----------------
            tracker.start_stage(rnd, "patt")
            for unit in units:
                _check_stop(layout)
                tracker.stage_detail(f"scenario {unit['scenario_id']}, "
                                     f"depot {unit['depot_id']} "
                                     f"({unit['n_stores']} stores)")
                tracker.update_unit(rnd, "patt", unit["id"], status="running")
                _, artifact = solve_patt_unit(unit, params, layout, rnd, tracker, lam)
                write_json(layout.patt_unit(rnd, unit["scenario_id"], unit["depot_id"]),
                           artifact)
                tracker.update_unit(rnd, "patt", unit["id"], status="completed",
                                    best_cost=artifact["objective"],
                                    feasible=artifact["feasible"])
            total_patterns = sum(len(u["artifact"]["stores"]) for u in units)
            total_routes = sum(sum(d["n_vehicles"] for d in u["artifact"]["routes_by_day"])
                               for u in units)
            tracker.finish_stage(rnd, "patt",
                                 headline=f"{total_patterns} store patterns, "
                                          f"{total_routes} routes, lambda={lam:.2f}")

            # ---------------- stage 3: SIM ----------------
            _check_stop(layout)
            tracker.start_stage(rnd, "sim",
                                detail=f"{params.sim.weeks} weeks x "
                                       f"{params.sim.runs} runs, "
                                       f"variants {params.sim.variants}")
            sim_art = run_sim(units, params, rnd)
            verdict = sim_verdict(sim_art, params)
            sim_art["verdict"] = verdict
            write_json(layout.sim(rnd), sim_art)
            tracker.finish_stage(rnd, "sim", headline=_sim_headline(sim_art, params))

            # Only "full" closes the SIM -> PATT edge. In "single" and "capacity"
            # the simulation is reporting, not steering: the run stops here
            # whatever the verdict says.
            if params.feedback.mode != "full":
                reason = (f"{params.feedback.mode} mode: simulation is reported but not "
                          f"fed back ({verdict['reason']})")
                tracker.finish_round(rnd, outcome_kind="accepted", reason=reason,
                                     detail={"verdict": verdict})
                timeline.append(_timeline_entry(rnd, lam, "accepted", reason,
                                                {"verdict": verdict}))
                outcome = {"status": "completed", "reason": reason, "rounds": rnd}
                break

            # ---------------- feedback edge SIM -> PATT: lambda --------------
            if verdict["accepted"]:
                tracker.finish_round(rnd, outcome_kind="accepted",
                                     reason=verdict["reason"],
                                     detail={"verdict": verdict})
                timeline.append(_timeline_entry(rnd, lam, "accepted",
                                                verdict["reason"], {"verdict": verdict}))
                outcome = {"status": "completed", "reason": verdict["reason"],
                           "rounds": rnd}
                break

            new_lam = max(0.0, lam * params.feedback.lambda_factor)
            reason = (f"{verdict['reason']}; lowering lambda "
                      f"{lam:.3f} -> {new_lam:.3f} and re-solving PATT")
            detail = {"verdict": verdict, "lambda_before": lam, "lambda_after": new_lam}
            tracker.finish_round(rnd, outcome_kind="feedback_lambda",
                                 reason=reason, detail=detail, edge_id="sim->patt")
            timeline.append(_timeline_entry(rnd, lam, "feedback_lambda", reason, detail))
            lam = new_lam
            outcome = {"status": "infeasible",
                       "reason": "lambda feedback did not converge"}
            write_json(layout.timeline, {"rounds": timeline})

        else:
            outcome = {"status": "infeasible",
                       "reason": f"maximum of {params.feedback.max_rounds} rounds reached",
                       "rounds": rnd}

    except StopRequested:
        outcome = {"status": "stopped", "reason": "stop requested", "rounds": rnd}
        tracker.finish("stopped", reason="Run stopped on request")
        write_json(layout.timeline, {"rounds": timeline})
        return outcome
    except Exception as exc:                     # noqa: BLE001 - surfaced to the UI
        outcome = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "rounds": rnd}
        tracker.finish("failed", reason=outcome["reason"], result=outcome)
        write_json(layout.timeline, {"rounds": timeline})
        return outcome

    write_json(layout.timeline, {"rounds": timeline})
    tracker.finish(outcome["status"], reason=outcome["reason"], result=outcome)
    write_json(layout.overview, _overview(layout, inst, params, outcome, timeline, rnd))
    return outcome


# ===========================================================================
# helpers
# ===========================================================================
def _check_stop(layout: RunLayout) -> None:
    if layout.stop_requested():
        raise StopRequested()


def _read_handoff(fname: str) -> Dict[str, Any]:
    """The RLRP->PATT instance JSON, kept so the UI can show what crosses the
    edge. Trimmed: the full distance matrix is not interesting here."""
    import json
    try:
        with open(fname, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    data.pop("distances", None)
    return data


def _apply_capacity_feedback(rlrp_inst, rlrp_result, infeasible, params) -> List[Dict[str, Any]]:
    """Scale the RLRP-side demand of the affected stores up, capped per round so
    the RLRP second-stage model stays solvable (run_pipeline_B's STEP_CAP)."""
    applied = []
    for check in infeasible:
        s, depot_id = check["scenario_id"], check["depot_id"]
        stores = rlrp_result.customer_depot_assignment.get(s, {}).get(depot_id, [])
        need, avg = check["required_t_per_day"], check["avg_demand_t_per_day"]
        factor = min(need * params.feedback.safety / avg, params.feedback.step_cap) \
            if avg > 1e-9 else params.feedback.step_cap
        for key in list(rlrp_inst.demands[s].keys()):
            if key[0] in stores:
                rlrp_inst.demands[s][key] *= factor
        applied.append({"scenario_id": s, "depot_id": depot_id,
                        "stores": list(stores), "factor": factor})
    return applied


def _timeline_entry(rnd, lam, kind, reason, detail) -> Dict[str, Any]:
    return {"round": rnd, "lambda": lam, "at": now_iso(),
            "outcome": {"kind": kind, "reason": reason, "detail": detail}}


def _sim_headline(sim_art: Dict[str, Any], params: RunParams) -> str:
    ref = params.feedback.reference_variant
    waste, stockout, n = 0.0, 0.0, 0
    for unit in sim_art["units"]:
        row = next((r for r in unit["rows"] if r.get("variant") == ref), None)
        if row:
            waste += row["waste%"]
            stockout += row["stockout%"]
            n += 1
    if not n:
        return "no simulated variants"
    return (f"Variant {ref}: waste {waste / n:.2f}%, "
            f"stockout {stockout / n:.2f}% (mean over {n} unit(s))")


def _overview(layout, inst, params, outcome, timeline, rounds) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": layout.run_id,
        "instance_name": inst.instance_name,
        "status": outcome["status"],
        "reason": outcome.get("reason"),
        "rounds": rounds,
        "mode": params.feedback.mode,
        "lambda_start": params.transport.lam,
        "lambda_end": timeline[-1]["lambda"] if timeline else params.transport.lam,
        "n_stores": len(inst.stores),
        "n_depots": len(inst.depots),
        "n_scenarios": len(inst.S),
        "patt_iterations": params.patt.max_iterations,
        "sim_variants": params.sim.variants,
        "finished_at": now_iso(),
    }


def _histogram(values: List[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _int(value):
    try:
        f = float(value)
        return int(f) if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None
