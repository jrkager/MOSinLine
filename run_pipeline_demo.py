import json
import time

import main as M          # patched main.py (aligned params, scenarios, segment export)
import patt.alns as alns  # segment-aware PATT

BAR = "=" * 78

# short PATT runs for the demo; raise for real experiments
PATT_MAX_ITER = 25
PATT_TIME_LIMIT = 300


def show_rlrp_output(inst, rlrp_result):
    print(f"\n{BAR}\n[STAGE 1 OUTPUT] RLRP -> what Johannes's model hands over\n{BAR}")
    print("RLRPResult has exactly two fields:\n")
    for s in inst.S:
        print(f"  Scenario {s}:")
        sizes = rlrp_result.depot_sizes.get(s, {})
        assign = rlrp_result.customer_depot_assignment.get(s, {})
        for d in inst.depots:
            size = sizes.get(d, 0.0)
            stores = assign.get(d, [])
            status = f"OPEN, size {size:.2f} t/day" if size > 0 or stores else "closed"
            print(f"    depot_sizes[{s}][{d:>3}] = {size:8.2f}   ({status})")
            if stores:
                print(f"    customer_depot_assignment[{s}][{d:>3}] = {stores}")
        print()
    print("  -> PATT consumes ONLY customer_depot_assignment (which stores each")
    print("     depot serves). depot_sizes could additionally cap Q_day_max (optional wire).")


def show_patt_input(fname):
    print(f"\n{BAR}\n[HANDOFF] PATT instance JSON: {fname}\n{BAR}")
    d = json.load(open(fname))
    print(f"  instance_name      : {d['instance_name']}")
    print(f"  stores (internal)  : {d['stores']}")
    print(f"  id_map             : {d['id_map']}   (internal -> original store id)")
    print(f"  depot              : ({d['depot']['x']}, {d['depot']['y']})  [renumbered to node 0]")
    print(f"  distances          : {len(d['distances'])} arcs (same matrix as RLRP)")
    print(f"  vehicle_capacity Q : {d['vehicle_capacity']}    empty weight W0: {d['vehicle_empty_weight']}")
    print(f"  cost_per_km        : {d['cost_per_km']}   fuel_price: {d['fuel_price']}   eta: {d['eta']}")
    print(f"  marginal_co2       : {d['marginal_co2_emissions']}  -> PATT theta_TR = "
          f"{d['marginal_co2_emissions']/d['eta']:.2f}")
    print(f"  c_CO2 [EUR/t]      : {d.get('c_co2_per_tonne')}  alpha={d.get('obj_alpha')} beta={d.get('obj_beta')}")
    print(f"  demand_by_segment  : per store x segment x weekday [t]:")
    for st, segs in d["demand_by_segment"].items():
        orig = d["id_map"][st]
        for seg, vals in segs.items():
            week = sum(vals)
            days = " ".join(f"{v:5.2f}" for v in vals)
            print(f"      store {st} (orig {orig})  {seg:>6}: [{days}]  = {week:6.2f} t/week")


def show_patt_output(sol, idata, alns_module):
    ev = sol.evaluator
    sm = idata["store_id_mapping"]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    print(f"\n{BAR}\n[STAGE 2 OUTPUT] PATT -> what Kailin's model returns\n{BAR}")
    print("  main() returns (ComprehensiveSolution, instance_data). Contents:\n")

    print("  (a) pattern_assignments — one delivery pattern per store:")
    for f in sorted(sol.pattern_assignments):
        r = sol.pattern_assignments[f]
        bits = "".join(map(str, ev.patterns[r]))
        days = ",".join(day_names[t] for t in range(6) if ev.patterns[r][t])
        print(f"      store {f} (orig {sm[f]}): pattern {r:>2} [{bits}] "
              f"freq {sum(ev.patterns[r])}  -> {days}")

    print("\n  (b) p_frt — simulated delivery quantity per (store, pattern, day) [t]:")
    for f in sorted(sol.pattern_assignments):
        r = sol.pattern_assignments[f]
        qs = " ".join(f"{sol.p_frt.get((f, r, t), 0.0):5.2f}" for t in range(6))
        print(f"      store {f}: [{qs}]")

    print("\n  (c) routes_by_day — vehicle routes (internal ids, 0 = this depot):")
    for day in range(6):
        active = [(v, rt) for v, rt in sol.routes_by_day[day].items() if len(rt) > 2]
        if active:
            for v, rt in active:
                load = sum(sol.p_frt.get((n, sol.pattern_assignments[n], day), 0.0)
                           for n in rt if n != 0)
                print(f"      {day_names[day]}: vehicle {v}: {rt}   load {load:5.2f} t / Q {ev.Q}")

    print("\n  (d) cost/emission breakdown (weighted objective components):")
    pat = sum(ev.c_fr[f, sol.pattern_assignments[f]] for f in sol.pattern_assignments)
    fw_p = sum(ev.c_purchase * ev.D_f[f] * ev.waste_fractions[f, sol.pattern_assignments[f]]
               for f in sol.pattern_assignments)
    fw_e = sum(ev.calculate_fw_emission(f, sol.pattern_assignments[f])
               for f in sol.pattern_assignments)
    so = sum(ev.c_stockout * ev.D_f[f] * ev.stockout_fractions[f, sol.pattern_assignments[f]]
             for f in sol.pattern_assignments)
    print(f"      pattern operational cost : {pat:10.2f} EUR")
    print(f"      food-waste purchase cost : {fw_p:10.2f} EUR")
    print(f"      stockout cost            : {so:10.2f} EUR")
    print(f"      food-waste emissions     : {fw_e:10.2f} kg CO2e  (segment-weighted theta_FW)")
    print(f"      routing cost (weighted)  : {sol.routing_cost:10.2f}")
    print(f"      TOTAL objective          : {sol.cost:10.2f}")


def show_pattresult(patt_result, scenario):
    print(f"\n{BAR}\n[COLLECT] PATTResult for scenario {scenario} -> what the SIM stage receives\n{BAR}")
    print("  patterns (ORIGINAL store ids):")
    for st, pat in sorted(patt_result.patterns.items(), key=lambda x: str(x[0])):
        print(f"      store {st}: {tuple(pat)}")
    print("  routes with delivery_amounts (depot total at position 0 — bug-fixed lookup):")
    for day, routes in sorted(patt_result.routes.items()):
        for rt in routes:
            amounts = " ".join(f"{a:5.2f}" for a in rt.delivery_amounts)
            print(f"      day {day}: stops {rt.stops}  amounts [{amounts}]")


def run():
    t0 = time.time()
    print(BAR)
    print("MOSinLine END-TO-END PIPELINE DEMO  (RLRP -> PATT -> collect)")
    print(BAR)

    inst = M.construct_test_instance()
    print(f"\nInstance: {inst.instance_name} | depots {inst.depots} | stores {inst.stores} "
          f"| scenarios {inst.S}")

    # ---- STAGE 1: RLRP ----
    rlrp_result = M.solve_rlrp(inst, "logs/demo_rlrp.txt")
    show_rlrp_output(inst, rlrp_result)

    # ---- STAGE 2: PATT per (scenario, open depot) ----
    patt_results = {}
    first_shown = False
    for s in inst.S:
        patt_results[s] = M.PATTResult()
        for depot_id in inst.depots:
            assigned = rlrp_result.customer_depot_assignment.get(s, {}).get(depot_id, [])
            if not assigned:
                continue
            fname = M.create_patt_instance_data(inst, rlrp_result, depot_id=depot_id, scenario=s)
            if fname is None:
                continue
            if not first_shown:
                show_patt_input(fname)   # show the handoff JSON once in full
            print(f"\n--- PATT run: scenario {s}, depot {depot_id}, "
                  f"{len(assigned)} stores, {PATT_MAX_ITER} iterations ---")
            sol, idata = alns.main(instance_file_name=fname,
                                   max_iterations=PATT_MAX_ITER,
                                   time_limit=PATT_TIME_LIMIT,
                                   save_results=False, verbose_report=False)
            if not first_shown:
                show_patt_output(sol, idata, alns)
                first_shown = True
            else:
                print(f"    done: objective {sol.cost:.2f}, "
                      f"patterns {[sol.pattern_assignments[f] for f in sorted(sol.pattern_assignments)]}")
            patt_results[s].append_solution(sol, idata)
            M.delete_patt_instance_file(fname)

    # ---- COLLECT ----
    show_pattresult(patt_results[inst.S[0]], inst.S[0])

    print(f"\n{BAR}\nPIPELINE COMPLETE in {time.time()-t0:.1f}s "
          f"({sum(len(pr.patterns) for pr in patt_results.values())} store-patterns across "
          f"{len(inst.S)} scenarios)\n{BAR}")


if __name__ == "__main__":
    run()