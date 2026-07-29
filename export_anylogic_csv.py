import csv
import math
import os
import sys
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [os.path.join(_here, "patt", "alns.py"),          # inside the MOSinLine repo
               os.path.join(_here, "alns_patt_segments.py")]    # standalone
_path = next(p for p in _candidates if os.path.exists(p))
_spec = importlib.util.spec_from_file_location("alns_seg", _path)
alns_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alns_mod)

SEG_ORDER = ["fresh", "dry", "frozen"]   # -> DES products A, B, C


def export_anylogic_csv(alns, solution, instance_data, name, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    sm = instance_data["store_id_mapping"]
    stores = sorted(solution.pattern_assignments.keys())   # internal ids 1..n

    # ---- stores csv ----
    stores_path = os.path.join(out_dir, f"stores_{name}.csv")
    with open(stores_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = (["storeID", "origID", "x", "y"]
                  + [f"muA_{t}" for t in range(6)] + ["S_A", "gamma", "R_A"]
                  + [f"pdq_{t}" for t in range(6)]
                  + [f"muB_{t}" for t in range(6)] + [f"muC_{t}" for t in range(6)]
                  + ["S_B", "S_C"])
        w.writerow(header)
        for f in stores:
            r = solution.pattern_assignments[f]
            x, y = alns.loc[f]
            mu = {s: [alns.mu_fst[f].get(s, {t: 0.0 for t in range(6)})[t] for t in range(6)]
                  for s in SEG_ORDER}
            S = {s: alns.S_fsr.get((f, s, r), 0.0) for s in SEG_ORDER}
            pdq = [round(solution.p_frt.get((f, r, t), 0.0), 3) for t in range(6)]
            row = ([f - 1, sm.get(f, f), round(x, 4), round(y, 4)]
                   + [round(v, 4) for v in mu["fresh"]]
                   + [round(S["fresh"], 4), round(alns.gamma_f[f], 4),
                      int(math.ceil(S["fresh"]))]
                   + pdq
                   + [round(v, 4) for v in mu["dry"]]
                   + [round(v, 4) for v in mu["frozen"]]
                   + [round(S["dry"], 4), round(S["frozen"], 4)])
            w.writerow(row)

    # ---- routes csv ----
    routes_path = os.path.join(out_dir, f"routes_{name}.csv")
    with open(routes_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "route_id", "seq", "storeIdx"])
        for day in range(6):
            rid = 0
            for _, route in sorted(solution.routes_by_day[day].items()):
                custs = [n for n in route if n != 0]
                if not custs:
                    continue
                for seq, n in enumerate(custs):
                    w.writerow([day, rid, seq, n - 1])
                rid += 1

    return stores_path, routes_path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    instance_file, name = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    max_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    instance_data = alns_mod.load_instance(instance_file)
    params = alns_mod.default_algorithm_params()
    a = alns_mod.ComprehensiveALNS(instance_data, params)
    solution = a.run_alns(max_iterations=max_iter, time_limit=1800)

    sp, rp = export_anylogic_csv(a, solution, instance_data, name, out_dir)
    print(f"\nExported:\n  {sp}\n  {rp}")

    # quick sanity summary
    n_seg_cols = 33
    with open(sp) as fh:
        rows = list(csv.reader(fh))
    assert all(len(r) >= n_seg_cols for r in rows[1:]), "segment columns missing"
    print(f"stores csv: {len(rows)-1} stores, {len(rows[0])} columns (>=33: segment-aware OK)")


if __name__ == "__main__":
    main()
