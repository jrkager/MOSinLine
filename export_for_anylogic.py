import argparse, os, sys
from math import hypot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import patt.alns as alns
from export_anylogic_csv import export_anylogic_csv
from r101_segment_instance import construct_r101_instance


def export_dist_and_depot(a, name, out_dir):
    """Distance lookup table + depot position from the PATT instance's own
    coordinates (a.loc: 0 = depot, 1..N = stores in DES order)."""
    nodes = [0] + sorted(f for f in a.loc if f != 0)
    n = len(nodes)
    dist_path = os.path.join(out_dir, f"dist_{name}.csv")
    with open(dist_path, "w") as f:
        for i in nodes:
            f.write(",".join(
                f"{hypot(a.loc[i][0]-a.loc[j][0], a.loc[i][1]-a.loc[j][1]):.6f}"
                for j in nodes) + "\n")
    depot_path = os.path.join(out_dir, f"depot_{name}.csv")
    with open(depot_path, "w") as f:
        f.write(f"{a.loc[0][0]},{a.loc[0][1]}\n")
    return dist_path, depot_path


def run(out_dir, iters, k, i):
    os.makedirs(out_dir, exist_ok=True)
    inst = construct_r101_instance(k=k, i=i, _main_module=M)
    rlrp = M.solve_rlrp(inst, "logs/export_rlrp.txt")
    for s in inst.S:
        if s != 1: continue
        for depot_id in inst.depots:
            assigned = rlrp.customer_depot_assignment.get(s, {}).get(depot_id, [])
            if not assigned:
                continue
            fname = M.create_patt_instance_data(inst, rlrp, depot_id=depot_id, scenario=s)
            idata = alns.load_instance(fname)
            a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
            cap = rlrp.depot_sizes[s].get(depot_id, 0)
            avg_dem = sum(idata["daily_demands"])
            W = sum(a.D_f[f] for f in a.stores)
            min_del = sum(min(sum(a.p_frt.get((f, r, t), 0) for t in range(6))
                              for r in a.feasible_patterns_by_store[f]) for f in a.stores)
            print(f"--- depot {depot_id} | scenario {s} ---")
            print(f"[RLRP] avg daily demand = {avg_dem:.2f} | capacity y = {cap:.2f} | slack = {(cap/avg_dem-1)*100:+.1f}%")
            print(f"[DPPP] weekly demand = {W:.2f} | min weekly delivery (best pattern) = {min_del:.2f}")
            print(f"[TEST] min_del/6 = {min_del/6:.2f}  vs  y = {cap:.2f}   6y = {6*cap:.2f}")
            if min_del > 6 * cap:
                print("       => min_del > 6y: INFEASIBLE for ANY pattern combination (pigeonhole)")
            else:
                print(f"       => arithmetically possible only if daily loads flat within {(6*cap/min_del-1)*100:.1f}% headroom")
            sol = a.run_alns(max_iterations=iters, time_limit=3600)
            M.delete_patt_instance_file(fname)
            name = f"s{s}d{-depot_id}"  # matches the .alp naming stores_s<k>.csv / routes_s<k>.csv
            sp, rp = export_anylogic_csv(a, sol, idata, name, out_dir)
            dp, wp = export_dist_and_depot(a, name, out_dir)
            print(f"scenario {s} depot {depot_id} -> {sp} | {rp} | {dp} | {wp} "
                  f"(objective {sol.cost:.2f}, {len(a.stores)} stores)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/mosinline_anylogic_r101")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--i", type=int, default=1)
    args = ap.parse_args()
    run(args.out, args.iters, args.k, args.i)