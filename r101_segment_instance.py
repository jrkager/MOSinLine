import dataclasses
import hashlib
import json
import random
from math import hypot
from statistics import median

import numpy as np

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
SOLOMON_PATH        = "solomon_data/R101.txt"   # classic 100-customer R101
SEL_SEED_BASE       = 123    # store-selection seed base (same role as RANDOM_SEED)
NOISE_SEED_BASE     = 42     # demand-noise seed base   (same role as DAYVAR_SEED)
DAYVAR_CV           = 0.15   # noise on the day-of-week means (instance-level)
DEMAND_DIVISOR      = 8.0    # Solomon demand -> daily mean units (DPPP convention)

DAY_OF_WEEK_MULTIPLIERS = [0.85, 0.90, 1.00, 1.05, 1.10, 1.10]  # Mon..Sat

# Segment split of every store's demand (matches the smoke test A/B/C = 52/35/13)
SEGMENT_SHARES = {"DRY": 0.52, "FRESH": 0.35, "FROZEN": 0.13}

# RLRP demand scenarios (systematic factors ON TOP of the same base draws,
# so scenarios differ only by the intended structural shift, not by noise):
#   scenario 1: base
#   scenario 2: global high demand  (x1.2 everywhere)
#   scenario 3: regional shift      (x1.3 east of the store-median x, x0.9 west)
SCEN_GLOBAL_FACTOR   = 1.20
SCEN_REGIONAL_HI     = 1.30
SCEN_REGIONAL_LO     = 0.90


# ---------------------------------------------------------------------------
# Solomon parser (same logic as instance_generator.py)
# ---------------------------------------------------------------------------
def parse_solomon_file(solomon_path):
    depot, customers = None, []
    with open(solomon_path, "r") as f:
        lines = f.readlines()
    for line in lines[9:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        idx, x, y, d = int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
        if depot is None:
            depot = (x, y)
        else:
            customers.append({"id": idx, "x": x, "y": y, "demand": d})
    return depot, customers


# ---------------------------------------------------------------------------
# Main constructor
# ---------------------------------------------------------------------------
def construct_r101_instance(k=10, i=1, solomon_path=SOLOMON_PATH,
                            _main_module=None):
    """
    k : number of stores sampled from R101
    i : instance index (1..20 for a 20-instance experiment set)
    Returns a fully populated main.Instance ready for the pipeline.
    """
    # Import here (not at module top): the repo's main.py may execute the
    # pipeline on import if it still ends with a bare `main()` call.
    if _main_module is None:
        import main as _main_module
    M = _main_module

    base = M.construct_test_instance()   # inherit ALL calibrated parameters

    depot_xy, all_customers = parse_solomon_file(solomon_path)
    all_customers = sorted(all_customers, key=lambda c: c["id"])

    # ---- store selection: per-instance deterministic seed (FIX-1 style) ----
    sel_seed = SEL_SEED_BASE * 100000 + k * 1000 + i
    chosen = random.Random(sel_seed).sample(range(len(all_customers)), k)
    picked = [all_customers[j] for j in chosen]

    stores = [c["id"] for c in picked]           # keep ORIGINAL R101 ids
    depots = [-1, -2, -3]

    # ---- candidate warehouse locations ----
    cx = sum(c["x"] for c in picked) / k
    cy = sum(c["y"] for c in picked) / k
    locations = {
        -1: depot_xy,                                        # original R101 depot
        -2: (round(cx, 1), round(cy, 1)),                    # centroid of sample
        -3: (round((depot_xy[0] + cx) / 2 + 8.0, 1),         # offset third option
             round((depot_xy[1] + cy) / 2 - 8.0, 1)),
    }
    for c in picked:
        locations[c["id"]] = (c["x"], c["y"])

    # ---- per-segment, day-varying nominal demand (FIX-2 style seeding) ----
    noise_seed = NOISE_SEED_BASE * 100000 + k * 1000 + i
    rng = np.random.RandomState(noise_seed)

    PC, WD = M.ProductClass, M.Weekday
    shares = {p: SEGMENT_SHARES.get(p.name, 1.0 / len(PC)) for p in PC}

    # one noise draw per (store, segment, weekday); shared across scenarios
    base_demand = {}
    for c in picked:
        d_daily = c["demand"] / DEMAND_DIVISOR
        for p in PC:
            eps = np.clip(rng.normal(0.0, DAYVAR_CV, size=len(WD)), -0.3, 0.3)
            for w in WD:
                v = d_daily * shares[p] * DAY_OF_WEEK_MULTIPLIERS[w.value] \
                    * (1.0 + eps[w.value])
                base_demand[(c["id"], p, w)] = round(max(v, 0.01), 4)

    x_med = median(c["x"] for c in picked)
    east = {c["id"] for c in picked if c["x"] >= x_med}

    def scen(factor_fn):
        return {key: round(val * factor_fn(key[0]), 4)
                for key, val in base_demand.items()}

    demands = {
        1: scen(lambda n: 1.0),
        2: scen(lambda n: SCEN_GLOBAL_FACTOR),
        3: scen(lambda n: SCEN_REGIONAL_HI if n in east else SCEN_REGIONAL_LO),
    }

    # ---- euclidean distances over all nodes ----
    nodes = depots + stores
    distances = {(a, b): hypot(locations[a][0] - locations[b][0],
                               locations[a][1] - locations[b][1])
                 for a in nodes for b in nodes}

    # ---- warehouse cost dicts: keep base values, re-key defensively ----
    fwc = {d: next(iter(base.fixed_warehouse_costs.values()))    for d in depots}
    mwc = {d: next(iter(base.marginal_warehouse_costs.values())) for d in depots}
    mws = {d: next(iter(base.max_warehouse_size.values()))       for d in depots}

    inst = dataclasses.replace(
        base,
        instance_name=f"R101_{k}stores_i{i}",
        depots=depots,
        stores=stores,
        locations=locations,
        S=sorted(demands.keys()),
        demands=demands,
        distances=distances,
        fixed_warehouse_costs=fwc,
        marginal_warehouse_costs=mwc,
        max_warehouse_size=mws,
    )

    print(f"[r101_segment_instance] {inst.instance_name}: "
          f"stores={stores} | sel_seed={sel_seed} noise_seed={noise_seed} | "
          f"weekly demand ~{sum(base_demand.values()):.1f} u (scenario 1)")
    return inst


# ---------------------------------------------------------------------------
# Standalone: build i=1..20, verify uniqueness, show summaries
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import importlib, sys
    M = importlib.import_module("main")

    K = 10
    hashes = {}
    for i in range(1, 21):
        inst = construct_r101_instance(k=K, i=i, _main_module=M)
        sig = {"stores": inst.stores,
               "loc": {str(n): inst.locations[n] for n in inst.stores},
               "d": {f"{n}|{p.name}|{w.name}": inst.demands[1][(n, p, w)]
                     for (n, p, w) in inst.demands[1]}}
        h = hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()
        hashes.setdefault(h, []).append(i)

    n_files, n_unique = 20, len(hashes)
    print(f"\n{K} stores: {n_files} instances, {n_unique} unique",
          "OK" if n_files == n_unique else "DUPLICATES!!")
    if n_files != n_unique:
        for h, idxs in hashes.items():
            if len(idxs) > 1:
                print("  duplicate set:", idxs)
        sys.exit(1)

    # sanity: RLRP aggregation works and scenario factors are visible
    inst = construct_r101_instance(k=K, i=1, _main_module=M)
    agg = inst.aggregate_demands_rlrp(option=1)
    s1 = sum(agg[1].values()); s2 = sum(agg[2].values()); s3 = sum(agg[3].values())
    print(f"RLRP daily totals: s1={s1:.2f}  s2={s2:.2f} (x{s2/s1:.2f})  "
          f"s3={s3:.2f} (x{s3/s1:.2f})")