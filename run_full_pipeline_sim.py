import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as M
from r101_segment_instance import construct_r101_instance
import patt.alns as alns
from sim_des_port import DESPort, StoreCfg, kpis_to_row, PRODUCTS, THETA_FW

BAR = "=" * 100
PATT_ITER = 25
SIM_WEEKS = 52
SIM_RUNS = 2
SEG2PROD = {"fresh": "A", "dry": "B", "frozen": "C"}


def build_sim_inputs(a, solution, instance_data):
    """PATT outputs -> DES-port inputs (DES store ids = internal id - 1)."""
    stores_cfg = {}
    for f in sorted(solution.pattern_assignments):
        r = solution.pattern_assignments[f]
        mu = {SEG2PROD[s]: [a.mu_fst[f].get(s, {t: 0.0 for t in range(6)})[t] for t in range(6)]
              for s in SEG2PROD}
        S = {SEG2PROD[s]: a.S_fsr.get((f, s, r), 0.0) for s in SEG2PROD}
        flags = [solution.p_frt.get((f, r, t), 0.0) > 1e-9 for t in range(6)]
        stores_cfg[f - 1] = StoreCfg(xy=a.loc[f], mu=mu, S=S, plan_flag=flags)
    routes_by_day = {}
    for day in range(6):
        lst = []
        for _, rt in sorted(solution.routes_by_day[day].items()):
            custs = [n - 1 for n in rt if n != 0]
            if custs:
                lst.append(custs)
        routes_by_day[day] = lst
    return stores_cfg, routes_by_day, a.loc[0]


def patt_predicted_row(a, solution):
    """The PATT model's own predictions for the chosen plan (weekly)."""
    ev = solution.evaluator
    dem = sum(a.D_f[f] for f in a.stores)
    waste = sum(a.D_f[f] * ev.waste_fractions[f, solution.pattern_assignments[f]] for f in a.stores)
    so = sum(a.D_f[f] * ev.stockout_fractions[f, solution.pattern_assignments[f]] for f in a.stores)
    fw_co2 = sum(ev.calculate_fw_emission(f, solution.pattern_assignments[f]) for f in a.stores)
    # raw (unweighted) transport KPIs from the committed routes
    tr_cost = tr_co2 = km = 0.0
    for day in range(6):
        for _, rt in solution.routes_by_day[day].items():
            if len(rt) <= 2:
                continue
            loads = {n: solution.p_frt.get((n, solution.pattern_assignments[n], day), 0.0)
                     for n in rt if n != 0}
            for i in range(len(rt) - 1):
                d = ev.delta[rt[i], rt[i + 1]]
                cur = sum(loads[j] for j in rt[i + 1:] if j != 0)
                fuel = ev.eta * (ev.W0 + cur) * d
                km += d
                tr_cost += ev.c_km * d + ev.c_fuel * fuel
                tr_co2 += ev.theta_TR * fuel
    deliv_cond = sum(solution.p_frt.get((f, solution.pattern_assignments[f], t), 0.0)
                     for f in a.stores for t in range(6))
    return {
        "run": "PATT model",
        "demand_u": round(dem, 2),
        "delivered_u": dem - so + waste,        # conservation: demand - stockout + waste
        "delivered_cond_u": deliv_cond,         # sum of p_frt (conditional mean; footnote only)
        "waste%": 100 * waste / dem if dem else 0,
        "wasteA%": None,
        "stockout%": 100 * so / dem if dem else 0,
        "FW_CO2_kg/wk": fw_co2,
        "TR_CO2_kg/wk": tr_co2,
        "TR_cost/wk": tr_cost,
        "km/wk": km,
        "cancel/wk": 0.0, "drop/wk": 0.0, "piggy_u/wk": 0.0,
        "so_cost/wk": ev.c_stockout * so,
        "fw_pur/wk": ev.c_purchase * waste,
    }


def fmt_table(rows):
    cols = ["run", "waste%", "stockout%", "FW_CO2_kg/wk", "TR_CO2_kg/wk",
            "TR_cost/wk", "km/wk", "cancel/wk", "drop/wk", "piggy_u/wk"]
    head = f"{'run':<12}" + "".join(f"{c:>14}" for c in cols[1:])
    print(head); print("-" * len(head))
    for r in rows:
        line = f"{r['run']:<12}"
        for c in cols[1:]:
            v = r[c]
            line += f"{'—':>14}" if v is None else f"{v:>14.2f}"
        print(line)


def run():
    t0 = time.time()
    print(BAR)
    print("FULL PIPELINE: RLRP -> segment PATT -> DES-port SIM (Variants 1-4)")
    print(BAR)

    inst = construct_r101_instance(k=10, i=1, _main_module=M)
    rlrp_result = M.solve_rlrp(inst, "logs/full_rlrp.txt")
    print("\nRLRP: open depots per scenario:",
          {s: {d: round(sz, 2) for d, sz in rlrp_result.depot_sizes.get(s, {}).items() if sz > 0}
           for s in inst.S})

    for s in inst.S:
        for depot_id in inst.depots:
            assigned = rlrp_result.customer_depot_assignment.get(s, {}).get(depot_id, [])
            if not assigned:
                continue
            print(f"\n{BAR}\nSCENARIO {s} | depot {depot_id} | {len(assigned)} stores\n{BAR}")
            fname = M.create_patt_instance_data(inst, rlrp_result, depot_id=depot_id, scenario=s)

            # PATT (need the ALNS object for mu_fst / S_fsr -> bypass alns.main)
            idata = alns.load_instance(fname)
            a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
            sol = a.run_alns(max_iterations=PATT_ITER, time_limit=600)
            M.delete_patt_instance_file(fname)

            print(f"\nPATT plan: patterns "
                  f"{ {f: ''.join(map(str, a.patterns[sol.pattern_assignments[f]])) for f in sorted(a.stores)} }")

            stores_cfg, routes_by_day, depot_xy = build_sim_inputs(a, sol, idata)

            rows = [patt_predicted_row(a, sol)]
            for variant in (2, 1, 3, 4):
                agg = None
                for run_i in range(SIM_RUNS):
                    port = DESPort(stores_cfg, routes_by_day, depot_xy, variant)
                    k = port.run(record_weeks=SIM_WEEKS, warmup_weeks=2, seed=20000 + 37 * run_i)
                    row = kpis_to_row(k, f"Variant {variant}")
                    if agg is None:
                        agg = row
                    else:
                        for key, v in row.items():
                            if isinstance(v, (int, float)):
                                agg[key] = (agg[key] + v) / 2
                rows.append(agg)

            print(f"\nSIM ({SIM_WEEKS} weeks x {SIM_RUNS} runs, per-week KPIs) vs PATT prediction:")
            fmt_table(rows)

    print(f"\n{BAR}\nFULL PIPELINE COMPLETE in {time.time()-t0:.1f}s\n{BAR}")


if __name__ == "__main__":
    run()