import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import patt.alns as alns
from sim_des_port import DESPort, StoreCfg, PRODUCTS, C_STOCKOUT, C_PURCHASE, Q_UNITS

SEG2PROD = {"fresh": "A", "dry": "B", "frozen": "C"}
PATT_ITER, SIM_WEEKS, SIM_RUNS = 25, 52, 3
LAM = 0.3

def build_inputs(a, sol):
    cfg = {}
    for f in sorted(sol.pattern_assignments):
        r = sol.pattern_assignments[f]
        mu = {SEG2PROD[s]: [a.mu_fst[f].get(s, {t:0.0 for t in range(6)})[t] for t in range(6)] for s in SEG2PROD}
        S = {SEG2PROD[s]: a.S_fsr.get((f, s, r), 0.0) for s in SEG2PROD}
        flags = [sol.p_frt.get((f, r, t), 0.0) > 1e-9 for t in range(6)]
        cfg[f-1] = StoreCfg(xy=a.loc[f], mu=mu, S=S, plan_flag=flags)
    routes = {}
    for day in range(6):
        lst = []
        for _, rt in sorted(sol.routes_by_day[day].items()):
            c = [n-1 for n in rt if n != 0]
            if c: lst.append(c)
        routes[day] = lst
    return cfg, routes, a.loc[0]

def plan_stats(a, sol):
    freqs = {}
    drops = []
    for f in a.stores:
        r = sol.pattern_assignments[f]
        fr = sum(a.patterns[r])
        freqs[fr] = freqs.get(fr, 0) + 1
        drops += [sol.p_frt.get((f, r, t), 0.0) for t in range(6) if a.patterns[r][t] == 1]
    small = sum(1 for d in drops if d < 5)
    thin = 0
    for t in range(6):
        for _, rt in sol.routes_by_day[t].items():
            custs = [n for n in rt if n != 0]
            if not custs: continue
            load = sum(sol.p_frt.get((n, sol.pattern_assignments[n], t), 0.0) for n in custs)
            if load < 15: thin += 1
    return {"freq": dict(sorted(freqs.items())), "n_drops": len(drops),
            "avg_drop": sum(drops)/len(drops), "small_drops": small, "thin_routes": thin}

def sim_avg(cfg, routes, depot_xy, variant):
    acc = None
    for ri in range(SIM_RUNS):
        k = DESPort(cfg, routes, depot_xy, variant).run(record_weeks=SIM_WEEKS, warmup_weeks=2, seed=20000+37*ri)
        dem = sum(k.demand.values())
        r = {"waste%": 100*k.waste_rate(), "so%": 100*k.stockout_rate(),
             "fw_co2": k.fw_co2_kg()/SIM_WEEKS, "tr_co2": k.transport_co2_kg/SIM_WEEKS,
             "tr_cost": k.transport_cost/SIM_WEEKS, "km": k.distance_km/SIM_WEEKS,
             "so_cost": C_STOCKOUT*sum(k.lost.values())/SIM_WEEKS,
             "fw_pur": C_PURCHASE*sum(k.waste.values())/SIM_WEEKS,
             "cancel": k.routes_cancelled/SIM_WEEKS, "drop": k.stores_dropped/SIM_WEEKS,
             "piggy_u": k.piggyback_units/SIM_WEEKS, "routes": k.routes_run/SIM_WEEKS}
        acc = r if acc is None else {kk: acc[kk]+v for kk, v in r.items()}
    r = {kk: v/SIM_RUNS for kk, v in acc.items()}
    r["var_cost"] = r["tr_cost"] + r["so_cost"] + r["fw_pur"]
    r["co2"] = r["fw_co2"] + r["tr_co2"]
    return r

def row(label, r):
    return (f"{label:<10}{r['waste%']:>8.2f}{r['so%']:>8.2f}{r['fw_co2']:>9.0f}{r['tr_co2']:>8.0f}"
            f"{r['co2']:>8.0f}{r['tr_cost']:>9.0f}{r['so_cost']:>8.0f}{r['fw_pur']:>8.0f}{r['var_cost']:>9.0f}"
            f"{r['km']:>7.0f}{r['routes']:>7.1f}{r['cancel']:>7.2f}{r['drop']:>7.2f}{r['piggy_u']:>8.1f}")

HEAD = (f"{'run':<10}{'waste%':>8}{'so%':>8}{'FW_CO2':>9}{'TR_CO2':>8}{'ΣCO2':>8}"
        f"{'TRcost':>9}{'SOcost':>8}{'FWpur':>8}{'Σvar€':>9}{'km':>7}{'routes':>7}{'cancel':>7}{'drop':>7}{'piggyU':>8}")

def main():
    t0 = time.time()
    inst = M.construct_test_instance()
    rlrp = M.solve_rlrp(inst, "logs/exp_rlrp.txt")
    scenario, depot_id = 1, -1
    fname = M.create_patt_instance_data(inst, rlrp, depot_id=depot_id, scenario=scenario)
    base_idata = alns.load_instance(fname)
    M.delete_patt_instance_file(fname)

    # ---------- Part 1: V2..V8 on the lambda=0.3 plan ----------
    import copy, random
    random.seed(7); 
    idata = copy.deepcopy(base_idata)
    a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
    sol = a.run_alns(max_iterations=PATT_ITER, time_limit=600)
    cfg, routes, depot_xy = build_inputs(a, sol)
    ps = plan_stats(a, sol)
    print("\n" + "="*118)
    print(f"PART 1 — scenario {scenario}, lambda={LAM} plan: freq={ps['freq']}, "
          f"avg drop={ps['avg_drop']:.2f}, drops<5: {ps['small_drops']}/{ps['n_drops']}, thin routes(<15): {ps['thin_routes']}/wk planned")
    print("="*118)
    print(HEAD); print("-"*118)
    for variant, lbl in [(2,"V2 as-is"),(1,"V1 all"),(3,"V3 d+c"),(4,"V4 d+c+pS"),
                         (5,"V5 drop"),(6,"V6 cancel"),(7,"V7 piggyA"),(8,"V8 piggyS")]:
        print(row(lbl, sim_avg(cfg, routes, depot_xy, variant)))

    # ---------- Part 2: lambda sweep executed under V2 and V1 ----------
    print("\n" + "="*118)
    print("PART 2 — lambda re-tuning: plan at each lambda, execute under V2 (as-is) and V1 (all rules)")
    print("="*118)
    for lam in [0.0, 0.15, 0.3, 0.5, 0.7]:
        random.seed(7)
        idata = copy.deepcopy(base_idata)
        idata["weighting_factor_patt"] = lam
        a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
        sol = a.run_alns(max_iterations=PATT_ITER, time_limit=600)
        cfg, routes, depot_xy = build_inputs(a, sol)
        ps = plan_stats(a, sol)
        print(f"\nlambda={lam}: freq={ps['freq']}, avg drop={ps['avg_drop']:.2f} t, "
              f"drops<5: {ps['small_drops']}/{ps['n_drops']}, thin routes(<15)/wk: {ps['thin_routes']}")
        print(HEAD); print("-"*118)
        print(row("  V2", sim_avg(cfg, routes, depot_xy, 2)))
        print(row("  V1", sim_avg(cfg, routes, depot_xy, 1)))
    print(f"\nDONE in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
