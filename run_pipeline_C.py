"""Pipeline C — DPPP <-> ANYLOGIC buffer feedback loop (AnyLogic is the judge).

The knob is the planning capacity buffer b: DPPP plans routes against
Q_plan = b * Q; AnyLogic executes with the true Q = 25.6 and reports the
realized overload share in its [RS] log line. YOU close the loop by hand:

  1. python run_pipeline_C.py
       -> solves the RLRP feedback rounds (cached after the first run),
          reads buffer_state.json (auto-created, every group starts at 1.00),
          exports the AnyLogic CSVs for every group at its current b
          (file suffix sXdY_b100, b095, ...).
  2. In AnyLogic: _name = "s1d1_b100", configId = 0 (Variant 2), Run.
       Read the log line:  [RS] cumulative overloaded routes: X / Y ...
       overload% = 100 * X / Y.
  3. overload% >  5  ->  open buffer_state.json, lower that group's b by 0.05.
     overload% <= 5  ->  that group is done; its b is b*.
  4. python run_pipeline_C.py   (regenerates only groups whose CSVs are missing
       for their current b), then back to step 2 for the changed groups.
  5. All six groups <= 5%  ->  finished; buffer_state.json holds the b* values.
"""
import os, sys, json, pickle, zlib, random
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import patt.alns as alns
from r101_segment_instance import construct_r101_instance
from export_anylogic_csv import export_anylogic_csv
from export_for_anylogic import export_dist_and_depot

SAFETY     = 1.35
STEP_CAP   = 1.15
MAX_ROUNDS = 6
# Anchor every path to THIS script's folder, so it does not matter which
# directory the terminal happens to be in when you run it.
_HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(_HERE, "anylogic_csv")
ITERS      = 150            # shakedown; set 500 for production runs
K, I       = 20, 1
STATE_FILE = os.path.join(_HERE, "buffer_state.json")
RLRP_CACHE = os.path.join(_HERE, "logs", "rlrpC_cache.pkl")


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


def solve_rlrp_with_feedback(inst):
    """Pipeline-B RLRP capacity feedback; result cached so repeated b-iterations
    of this script don't re-solve Gurobi every time. Delete logs/rlrpC_cache.pkl
    to force a fresh solve."""
    if os.path.exists(RLRP_CACHE):
        with open(RLRP_CACHE, "rb") as f:
            rlrp = pickle.load(f)
        print(f"RLRP loaded from cache {RLRP_CACHE} (delete the file to re-solve).")
        return rlrp
    rlrp_inst = deepcopy(inst)
    rlrp = None
    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"\n########## ROUND {rnd}: solving RLRP ##########")
        rlrp = M.solve_rlrp(rlrp_inst, f"logs/rlrpC_round{rnd}.txt")
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
                    factor = min(need * SAFETY / avg_dem, STEP_CAP)
                    fixes.append((depot_id, s, stores, factor))
        if not fixes:
            print(f"\nROUND {rnd}: all (scenario, depot) combinations feasible.")
            break
        if rnd == MAX_ROUNDS:
            print("\nMax rounds reached, still infeasible. Stopping.")
            sys.exit(1)
        for depot_id, s, stores, factor in fixes:
            print(f"FEEDBACK: depot {depot_id} s{s}: scaling RLRP demand of stores "
                  f"{stores} by x{factor:.3f}")
            for key in list(rlrp_inst.demands[s].keys()):
                if key[0] in stores:
                    rlrp_inst.demands[s][key] *= factor
    os.makedirs("logs", exist_ok=True)
    with open(RLRP_CACHE, "wb") as f:
        pickle.dump(rlrp, f)
    print(f"RLRP result cached to {RLRP_CACHE}.")
    return rlrp


def load_state(groups):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        for g in groups:
            state.setdefault(g, 1.00)
    else:
        state = {g: 1.00 for g in groups}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return state


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)
    print(f"CSV output folder: {OUT_DIR}")
    inst = construct_r101_instance(k=K, i=I, _main_module=M)
    rlrp = solve_rlrp_with_feedback(inst)

    groups = []
    for s in inst.S:
        for depot_id in inst.depots:
            if rlrp.customer_depot_assignment.get(s, {}).get(depot_id, []):
                groups.append(f"s{s}d{-depot_id}")
    state = load_state(groups)

    made, skipped = [], []
    for s in inst.S:
        for depot_id in inst.depots:
            g = f"s{s}d{-depot_id}"
            if g not in groups:
                continue
            b = float(state[g])
            name = f"{g}_b{int(round(b*100)):03d}"
            print(f"\n===== DPPP: {g} at b={b:.2f} (Q_plan = {b*25.6:.2f}) =====")
            # Fixed per-group seed: the SAME group at the SAME b always yields the
            # SAME plan, and across b levels only the capacity ceiling changes —
            # the b comparison is ceteris paribus instead of a fresh random draw.
            random.seed(zlib.crc32(g.encode()))
            fname = M.create_patt_instance_data(inst, rlrp, depot_id=depot_id, scenario=s)
            idata = alns.load_instance(fname)
            M.delete_patt_instance_file(fname)
            idata["truck_buffer"] = b
            a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
            sol = a.run_alns(max_iterations=ITERS, time_limit=3600)
            # ---- HARD capacity repair at Q_plan ----
            # Some operator paths can leave a route above the planning ceiling
            # (observed: s1d1 route at 22.79 vs Q_plan 20.48, uncaught by
            # validate). Enforce the ceiling here, unconditionally: any route
            # above Q_plan is split — smallest drops move to a new route until
            # it fits. This guarantees every exported plan respects b * Q.
            Qp = a.Q_plan
            for d in range(6):
                day_routes = sol.routes_by_day[d]
                overflow = []
                for v in list(day_routes.keys()):
                    seen, rt = set(), []
                    for n in day_routes[v]:
                        if n == 0 or n not in seen:
                            rt.append(n)
                            if n != 0:
                                seen.add(n)
                    def _load(nodes):
                        return sum(sol.p_frt.get((f, sol.pattern_assignments[f], d), 0.0)
                                   for f in nodes if f != 0)
                    members = [n for n in rt if n != 0]
                    while members and _load(members) > Qp + 1e-9:
                        members.sort(key=lambda f: sol.p_frt.get((f, sol.pattern_assignments[f], d), 0.0))
                        overflow.append(members.pop(0))
                    day_routes[v] = [0] + members + [0] if members else [0, 0]
                nv = 1000
                while overflow:
                    chunk = []
                    while overflow and (sum(sol.p_frt.get((f, sol.pattern_assignments[f], d), 0.0)
                                            for f in chunk + [overflow[-1]]) <= Qp + 1e-9):
                        chunk.append(overflow.pop())
                    if not chunk:      # single store larger than Qp: ship alone
                        chunk = [overflow.pop()]
                    day_routes[nv] = [0] + chunk + [0]
                    print(f"  capacity repair: day {d} split off route {chunk} "
                          f"(load {sum(sol.p_frt.get((f, sol.pattern_assignments[f], d), 0.0) for f in chunk):.2f})")
                    nv += 1
            sol._calculate_cost()
            # final assert: no exported route above Q_plan
            for d in range(6):
                for v, rt in sol.routes_by_day[d].items():
                    L = sum(sol.p_frt.get((f, sol.pattern_assignments[f], d), 0.0)
                            for f in set(rt) if f != 0)
                    if L > Qp + 1e-6:
                        raise RuntimeError(f"route still over Q_plan after repair: day {d} load {L:.2f}")
            viol = sol.validate_constraints()
            # does the buffer actually bite this plan?
            max_load = 0.0
            for d in range(6):
                for _, rt in sol.routes_by_day[d].items():
                    if len(rt) > 2:
                        # dedupe: a store listed twice in a raw route is one physical stop
                        L = sum(sol.p_frt.get((f, sol.pattern_assignments[f], d), 0.0)
                                for f in set(rt) if f != 0)
                        max_load = max(max_load, L)
            binding = "BINDING (buffer shapes this plan)" if max_load > b * a.Q - 0.8 \
                      else "slack (plan sits below the ceiling; lowering b further is what changes it)"
            # WRITE the four AnyLogic CSVs (stores/routes + dist/depot)
            sp, rp = export_anylogic_csv(a, sol, idata, name, OUT_DIR)
            dp, wp = export_dist_and_depot(a, name, OUT_DIR)
            for _fp in (sp, rp, dp, wp):
                if not os.path.exists(_fp):
                    raise RuntimeError(f"export failed, file missing: {_fp}")
            _sumSA = sum(a.S_fsr.get((f, "fresh", sol.pattern_assignments[f]), 0.0) for f in a.stores) \
                     if hasattr(a, "S_fsr") else 0.0
            _sumPdq = sum(sol.p_frt.get((f, sol.pattern_assignments[f], t), 0.0)
                          for f in a.stores for t in range(6))
            _nroutes = sum(1 for d in range(6) for _, rt in sol.routes_by_day[d].items() if len(rt) > 2)
            print(f"exported {os.path.join(OUT_DIR, 'stores_' + name + '.csv')}")
            print(f"[FINGERPRINT] {name} | routes/week={_nroutes} | sum S_A={_sumSA:.3f} | sum pdq={_sumPdq:.3f}")
            print(f"exported {name}: objective {sol.cost:.2f}, violations {len(viol)}, "
                  f"max planned route load {max_load:.2f} vs Q_plan {b*a.Q:.2f} -> {binding}")
            made.append(name)

    print("\n" + "=" * 74)
    print("NEXT: run each NEW config in AnyLogic (configId = 0, i.e. Variant 2):")
    for name in made:
        print(f'  _name = "{name}"   -> read the [RS] overloaded routes line')
    print("Then: overload% > 5  ->  lower that group's b by 0.05 in "
          f"{STATE_FILE} and run this script again.")
    print("=" * 74)


if __name__ == "__main__":
    run()