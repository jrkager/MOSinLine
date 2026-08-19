import math
import random
from typing import Dict, List, Tuple, Any


DEFAULT_SEGMENT_SHARES = {"DRY": 0.35, "FRESH": 0.50, "FNV": 0.15}

# Mon..Sat demand profile (German grocery shape: strong Fri/Sat), mean = 1
DEFAULT_WEEKDAY_WEIGHTS = [0.90, 0.85, 0.90, 1.00, 1.20, 1.15]

DEFAULT_SCENARIO_SPECS = [
    # 1: today's demand structure
    {"name": "base"},
    # 2: uniform growth — tests warehouse sizing headroom
    {"name": "growth_20pct", "global_scale": 1.20},
    # 3: segment shift toward fresh — same tonnage, different composition;
    #    only visible to the pipeline because PATT is now segment-aware
    {"name": "fresh_shift", "share_shift": {"FRESH": +0.10, "DRY": -0.07, "FNV": -0.03}},
]


def generate_scenarios(
    stores: List[int],
    ProductClass,                    # pass the enum class from main.py
    Weekday,                         # pass the enum class from main.py
    scenario_specs: List[Dict[str, Any]] = None,
    weekly_demand_range: Tuple[float, float] = (5.0, 30.0),   # t/store/week
    segment_shares: Dict[str, float] = None,
    share_jitter: float = 0.04,      # per-store perturbation of shares
    weekday_weights: List[float] = None,
    seed: int = 42,
):
    """Returns (demands, scenario_names, base_profile).

    demands:        Dict[scenario_id(int, 1-based), Dict[(store, pc, w), float]]
    scenario_names: Dict[scenario_id, str]
    base_profile:   Dict[store, {"weekly": float, "shares": {pc_name: float}}]
                    (the scenario-independent store baseline, for reporting)

    Construction, per store f:
      weekly_f ~ Uniform(weekly_demand_range)             [structural size]
      shares_f = normalized(segment_shares + jitter)      [store assortment]
      demand[f, pc, w] = weekly_f * shares_f[pc] * weekday_weights[w] / sum(w)

    Scenario spec keys (all optional, applied to the base):
      global_scale: float           — multiply all demand
      share_shift:  {pc_name: d}    — additive shift of segment shares
                                      (re-normalized; tonnage preserved)
      store_scale:  {store_id: f}   — per-store multipliers (regional shock)
    """
    rng = random.Random(seed)
    specs = scenario_specs if scenario_specs is not None else DEFAULT_SCENARIO_SPECS
    shares0 = dict(segment_shares if segment_shares is not None else DEFAULT_SEGMENT_SHARES)
    ww = list(weekday_weights if weekday_weights is not None else DEFAULT_WEEKDAY_WEIGHTS)
    assert len(ww) == len(list(Weekday)), "weekday_weights length must match Weekday enum"
    ww_sum = sum(ww)
    pcs = list(ProductClass)
    for pc in pcs:
        assert pc.name in shares0, f"segment_shares missing {pc.name}"

    # ---- scenario-independent store baseline ----
    base_profile = {}
    for f in stores:
        weekly = rng.uniform(*weekly_demand_range)
        jittered = {pc.name: max(0.01, shares0[pc.name] + rng.uniform(-share_jitter, share_jitter))
                    for pc in pcs}
        s = sum(jittered.values())
        shares_f = {k: v / s for k, v in jittered.items()}
        base_profile[f] = {"weekly": weekly, "shares": shares_f}

    # ---- scenarios as transformations of the baseline ----
    demands: Dict[int, Dict[Tuple[int, Any, Any], float]] = {}
    scenario_names: Dict[int, str] = {}

    for k, spec in enumerate(specs, start=1):
        scenario_names[k] = spec.get("name", f"scenario_{k}")
        g = float(spec.get("global_scale", 1.0))
        shift = spec.get("share_shift", {})
        store_scale = spec.get("store_scale", {})

        d_k: Dict[Tuple[int, Any, Any], float] = {}
        for f in stores:
            weekly = base_profile[f]["weekly"] * g * float(store_scale.get(f, 1.0))
            shares_f = dict(base_profile[f]["shares"])
            if shift:
                for pc_name, dv in shift.items():
                    shares_f[pc_name] = max(0.0, shares_f.get(pc_name, 0.0) + dv)
                s = sum(shares_f.values())
                shares_f = {kk: vv / s for kk, vv in shares_f.items()}
            for pc in pcs:
                for w in Weekday:
                    d_k[(f, pc, w)] = weekly * shares_f[pc.name] * ww[w.value] / ww_sum
        demands[k] = d_k

    return demands, scenario_names, base_profile


def check_scenarios_against_vehicle(demands, scenario_names, stores, ProductClass, Weekday,
                                    Q: float = 25.6, min_freq: int = 2):
    """Feasibility sanity check: under the sparsest admissible pattern
    (min_freq deliveries/week), the worst single drop is ~ weekly/min_freq.
    Flags any (scenario, store) where that exceeds vehicle capacity Q.
    Run once after generation; a violation means the demand range and Q are
    mis-scaled relative to each other."""
    ok = True
    for k, d_k in demands.items():
        for f in stores:
            weekly = sum(d_k[(f, pc, w)] for pc in ProductClass for w in Weekday)
            worst_drop = weekly / min_freq
            if worst_drop > Q:
                print(f"  WARNING [{scenario_names[k]}] store {f}: weekly {weekly:.1f} t "
                      f"-> min-freq drop {worst_drop:.1f} t > Q={Q}")
                ok = False
    if ok:
        print(f"  All (scenario, store) pairs feasible per drop at min_freq={min_freq}, Q={Q}.")
    return ok


# ----------------------------------------------------------------------------
# WIRING into main.py's test-instance builder — replace the hardcoded block
#
#     demand = {12: 19, 77: 14, 49: 30, 9: 16, 1: 10}
#     demands_per_scenario = {...}
#     demands = {1: ..., 2: ..., 3: ...}   # three identical copies
#
# with:
#
#     demands, scenario_names, base_profile = generate_scenarios(
#         stores, ProductClass, Weekday, seed=42)
#     check_scenarios_against_vehicle(demands, scenario_names, stores,
#                                     ProductClass, Weekday,
#                                     Q=TRANSPORT_PARAMS["Q"])
#     S = list(demands.keys())
#
# Everything downstream (aggregate_demands_rlrp, RLRP scenario iteration,
# create_patt_instance_data + demand_by_segment export) consumes this as-is.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Self-test with stub enums (mirrors main.py's definitions)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from enum import Enum

    class Weekday(Enum):
        MONDAY = 0; TUESDAY = 1; WEDNESDAY = 2; THURSDAY = 3; FRIDAY = 4; SATURDAY = 5

    class ProductClass(Enum):
        DRY = 0; FRESH = 1; FNV = 2

    stores = [12, 77, 49, 9, 1]
    demands, names, base = generate_scenarios(stores, ProductClass, Weekday, seed=42)

    print("Scenarios:", names)
    print(f"\n{'store':>6} {'scenario':>14} {'weekly t':>9} {'dry%':>6} {'fresh%':>7} {'fnv%':>8} {'Sat/Mon':>8}")
    for k in demands:
        for f in stores:
            weekly = sum(demands[k][(f, pc, w)] for pc in ProductClass for w in Weekday)
            seg = {pc.name: sum(demands[k][(f, pc, w)] for w in Weekday) for pc in ProductClass}
            mon = sum(demands[k][(f, pc, Weekday.MONDAY)] for pc in ProductClass)
            sat = sum(demands[k][(f, pc, Weekday.SATURDAY)] for pc in ProductClass)
            print(f"{f:>6} {names[k]:>14} {weekly:>9.2f} "
                  f"{seg['DRY']/weekly*100:>5.1f}% {seg['FRESH']/weekly*100:>6.1f}% "
                  f"{seg['FNV']/weekly*100:>7.1f}% {sat/mon:>8.2f}")

    print("\nVehicle feasibility check (Q=25.6):")
    check_scenarios_against_vehicle(demands, names, stores, ProductClass, Weekday, Q=25.6)

    # invariants
    for f in stores:
        w_base = sum(demands[1][(f, pc, w)] for pc in ProductClass for w in Weekday)
        w_grow = sum(demands[2][(f, pc, w)] for pc in ProductClass for w in Weekday)
        w_shift = sum(demands[3][(f, pc, w)] for pc in ProductClass for w in Weekday)
        assert abs(w_grow / w_base - 1.20) < 1e-9          # growth scenario scales tonnage
        assert abs(w_shift - w_base) < 1e-9                # shift scenario preserves tonnage
    fresh_b = sum(demands[1][(f, ProductClass.FRESH, w)] for f in stores for w in Weekday)
    fresh_s = sum(demands[3][(f, ProductClass.FRESH, w)] for f in stores for w in Weekday)
    assert fresh_s > fresh_b                                # ...but shifts composition to fresh
    print("\nInvariants OK (growth = x1.20 tonnage; fresh_shift = same tonnage, more fresh).")