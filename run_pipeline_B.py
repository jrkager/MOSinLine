"""Pipeline with feedback B (v2):
- feedback raises RLRP demand gently (max +20% per round) instead of one big jump,
  because a 33% one-shot jump crashed the RLRP second-stage model
- RLRP solve auto-retries once on internal crash
- otherwise identical to v1

Run:  python run_pipeline_B.py
"""
import os, sys
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import patt.alns as alns
from r101_segment_instance import construct_r101_instance
from export_anylogic_csv import export_anylogic_csv
from export_for_anylogic import export_dist_and_depot

SAFETY     = 1.35          # target headroom on fed-back throughput
STEP_CAP   = 1.15          # max scaling per round (gentle on RLRP)
MAX_ROUNDS = 6
OUT_DIR    = "./anylogic_csv"
ITERS      = 500
K, I       = 20, 1         # 从 10 改成 20


def depot_check(inst, rlrp, depot_id, s):
    fname = M.create_patt_instance_data(inst, rlrp, depot_id=depot_id, scenario=s)
    idata = alns.load_instance(fname)
    a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
    M.delete_patt_instance_file(fname)
    cap = rlrp.depot_sizes[s].get(depot_id, 0.0)
    avg_dem = sum(idata["daily_demands"])
    min_del = sum(min(sum(a.p_frt.get((f, r, t), 0.0) for t in range(6))
                      for r in a.feasible_patterns_by_store[f])
                  for f in a.stores)
    return cap, avg_dem, min_del


def solve_rlrp_safe(rlrp_inst, logname):
    """RLRP with one automatic retry on internal crash."""
    for attempt in (1, 2):
        try:
            return M.solve_rlrp(rlrp_inst, logname)
        except Exception as e:
            print(f"RLRP crashed (attempt {attempt}): {type(e).__name__}: {e}")
            if attempt == 2:
                raise
            print("Retrying RLRP once...")


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    inst = construct_r101_instance(k=K, i=I, _main_module=M)   # DPPP sees this
    rlrp_inst = deepcopy(inst)                                 # RLRP sees this

    rlrp = None
    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"\n########## ROUND {rnd}: solving RLRP ##########")
        rlrp = solve_rlrp_safe(rlrp_inst, f"logs/rlrp_round{rnd}.txt")
        fixes = []
        for s in inst.S:
            for depot_id in inst.depots:
                stores = rlrp.customer_depot_assignment.get(s, {}).get(depot_id, [])
                if not stores:
                    continue
                cap, avg_dem, min_del = depot_check(inst, rlrp, depot_id, s)
                need = min_del / 6.0
                ok = need * 1.15 <= cap + 1e-9
                print(f"depot {depot_id} s{s}: cap={cap:.2f}  need={need:.2f}  "
                      f"(avg demand {avg_dem:.2f})  ->  {'OK' if ok else 'INFEASIBLE'}")
                if not ok:
                    factor = min(need * SAFETY / avg_dem, STEP_CAP)  # gentle step
                    fixes.append((depot_id, s, stores, factor))
        if not fixes:
            print(f"\nROUND {rnd}: all (scenario, depot) combinations feasible."
                  f" Proceeding to ALNS.")
            break
        if rnd == MAX_ROUNDS:
            print("\nMax rounds reached, still infeasible. Stopping.")
            return
        for depot_id, s, stores, factor in fixes:
            print(f"FEEDBACK: depot {depot_id} s{s}: scaling RLRP demand of stores "
                  f"{stores} by x{factor:.3f}")
            for key in list(rlrp_inst.demands[s].keys()):
                if key[0] in stores:
                    rlrp_inst.demands[s][key] *= factor

    # ---- final pass: ALNS + AnyLogic export ----
    for s in inst.S:
        for depot_id in inst.depots:
            stores = rlrp.customer_depot_assignment.get(s, {}).get(depot_id, [])
            if not stores:
                continue
            print(f"\n===== ALNS: scenario {s} | depot {depot_id} "
                  f"| {len(stores)} stores =====")
            fname = M.create_patt_instance_data(inst, rlrp, depot_id=depot_id,
                                                scenario=s)
            idata = alns.load_instance(fname)
            a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
            sol = a.run_alns(max_iterations=ITERS, time_limit=3600)
            M.delete_patt_instance_file(fname)
            viol = sol.validate_constraints()
            if viol:
                print(f"WARNING depot {depot_id} s{s}: solution still violates: "
                      f"{viol}")
            name = f"s{s}d{-depot_id}"
            sp, rp = export_anylogic_csv(a, sol, idata, name, OUT_DIR)
            dp, wp = export_dist_and_depot(a, name, OUT_DIR)
            print(f"scenario {s} depot {depot_id} -> {sp} | {rp} | {dp} | {wp} "
                  f"(objective {sol.cost:.2f})")


if __name__ == "__main__":
    run()