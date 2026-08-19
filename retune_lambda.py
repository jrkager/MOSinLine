"""Inner-loop objective retuning (monetised objective): freeze the RLRP
assignment, re-run DPPP for ONE depot at a NEW carbon-cost weight beta (and/or
carbon price c_co2), export AnyLogic CSVs with a new suffix.
Replaces the old lambda knob: objective = alpha*economic + beta*c_co2*emissions.

Usage (from D:\\PhD\\MOSinLine):
    python retune_lambda.py --suffix s1d1 --beta 1.5 --cap 27.20 --iters 500

Reads  anylogic_csv/stores_<suffix>.csv, dist_<suffix>.csv, depot_<suffix>.csv
Writes anylogic_csv/stores_<suffix>_lam<xx>.csv (+routes/dist/depot) for AnyLogic.
Also prints the drop-rule gate check (per-visit qty vs threshold 2) BEFORE you
spend a simulation run on it.
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patt.alns as alns
from export_anylogic_csv import export_anylogic_csv
from export_for_anylogic import export_dist_and_depot

DROP_THRESHOLD = 2.0

def build_instance(csv_dir, suffix, beta, cap, c_co2_per_tonne=350.0, alpha=1.0):
    stores = list(csv.DictReader(open(f"{csv_dir}/stores_{suffix}.csv")))
    N = len(stores)
    depot = open(f"{csv_dir}/depot_{suffix}.csv").read().strip().split(",")
    dist = [[float(x) for x in ln.split(",")]
            for ln in open(f"{csv_dir}/dist_{suffix}.csv").read().strip().splitlines()]
    jd = {
        "instance_name": f"{suffix}_beta{int(round(beta*100)):03d}",
        "depot": {"x": float(depot[0]), "y": float(depot[1]), "demand": 0},
        "stores": list(range(1, N + 1)),
        "id_map": {str(i + 1): int(s["origID"]) for i, s in enumerate(stores)},
        "loc": {str(i + 1): {"x": float(s["x"]), "y": float(s["y"])}
                for i, s in enumerate(stores)},
        "daily_demands": [sum(float(s[f"mu{p}_{t}"]) for p in "ABC" for t in range(6))
                          for s in stores],
        "distances": {f"({a},{b})": dist[a][b]
                      for a in range(N + 1) for b in range(N + 1)},
        "vehicle_capacity": 25.6, "vehicle_empty_weight": 14.4,
        "cost_per_km": 1.12, "fuel_price": 1.80, "eta": 0.05,
        "marginal_co2_emissions": 0.135,
        "c_co2_per_tonne": c_co2_per_tonne,
        "obj_alpha": alpha,
        "obj_beta": beta,
        "demand_by_segment": {str(i + 1): {
            "fresh":  [float(s[f"muA_{t}"]) for t in range(6)],
            "dry":    [float(s[f"muB_{t}"]) for t in range(6)],
            "fnv":    [float(s[f"muC_{t}"]) for t in range(6)],
        } for i, s in enumerate(stores)},
    }
    if cap is not None:
        jd["Q_day_max"] = cap
    return jd

def gate_check(a, sol):
    print("\n--- GATE CHECK: per-visit quantity vs drop threshold "
          f"{DROP_THRESHOLD} ---")
    victims = 0
    for f in sorted(a.stores):
        r = sol.pattern_assignments[f]
        qs = [(t, a.p_frt.get((f, r, t), 0.0)) for t in range(6)
              if a.patterns[r][t] == 1]
        low = [(t, q) for t, q in qs if q < DROP_THRESHOLD]
        orig = a.instance_data["id_map"].get(str(f), f)
        tag = "  <-- VICTIM" if low else ""
        if low:
            victims += 1
        print(f"store {orig}: freq {len(qs)}  drops "
              f"{['%d:%.2f' % (t, q) for t, q in qs]}{tag}")
    print(f"=> {victims} store(s) with sub-threshold days."
          f" {'Plan will bleed under V5.' if victims else 'Clean for V5.'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--beta", type=float, required=True,
                    help="weight on the carbon cost term (old lambda knob)")
    ap.add_argument("--c_co2", type=float, default=350.0,
                    help="carbon price in EUR per tonne CO2 (default UBA MK4.0 350)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight on economic costs")
    ap.add_argument("--cap", type=float, default=None,
                    help="Q_day_max from the final RLRP round (printed by the pipeline)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--dir", default="./anylogic_csv")
    args = ap.parse_args()

    fname = "_retune_tmp.json"
    json.dump(build_instance(args.dir, args.suffix, args.beta, args.cap,
                             c_co2_per_tonne=args.c_co2, alpha=args.alpha),
              open(fname, "w"))
    idata = alns.load_instance(fname)
    a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
    sol = a.run_alns(max_iterations=args.iters, time_limit=3600)
    os.remove(fname)

    viol = sol.validate_constraints()
    if viol:
        print(f"WARNING: solution violates: {viol}")
    gate_check(a, sol)

    name = f"{args.suffix}_beta{int(round(args.beta*100)):03d}"
    sp, rp = export_anylogic_csv(a, sol, idata, name, args.dir)
    dp, wp = export_dist_and_depot(a, name, args.dir)
    print(f"\nexported: {sp} | {rp} | {dp} | {wp}  (objective {sol.cost:.2f})")
    print(f"AnyLogic: _name = \"{name}\", configId = 4 (V5)")

if __name__ == "__main__":
    main()