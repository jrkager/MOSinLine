import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M
import patt.alns as alns
from sim_des_port import (DESPort, StoreCfg, PRODUCTS, THETA_FW,
                          C_STOCKOUT, C_PURCHASE, Q_UNITS)

SEG2PROD = {"fresh": "A", "dry": "B", "fnv": "C"}
PATT_ITER = 25
SIM_WEEKS, SIM_RUNS = 52, 3
DAYN = ["Mon","Tue","Wed","Thu","Fri","Sat"]

def build_sim_inputs(a, sol):
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

def plan_diag(a, sol):
    d = {"patterns": {}, "freq_dist": {}, "day_load": {t: 0.0 for t in range(6)},
         "routes": {t: [] for t in range(6)}, "store_week": {}}
    for f in sorted(a.stores):
        r = sol.pattern_assignments[f]
        bits = a.patterns[r]
        d["patterns"][f] = "".join(map(str, bits))
        fr = sum(bits)
        d["freq_dist"][fr] = d["freq_dist"].get(fr, 0) + 1
        d["store_week"][f] = {
            "D_week": a.D_f[f],
            "seg": {s: round(a.D_fs[f][s], 2) for s in a.segment_names},
            "freq": fr,
            "drops": [round(sol.p_frt.get((f, r, t), 0.0), 2) for t in range(6)],
        }
    for t in range(6):
        for _, rt in sorted(sol.routes_by_day[t].items()):
            custs = [n for n in rt if n != 0]
            if not custs: continue
            load = sum(sol.p_frt.get((n, sol.pattern_assignments[n], t), 0.0) for n in custs)
            d["routes"][t].append({"stops": custs, "load": round(load, 2),
                                   "fill": round(load / a.Q, 3)})
            d["day_load"][t] += load
    return d

def patt_pred(a, sol):
    ev = sol.evaluator
    dem = sum(a.D_f[f] for f in a.stores)
    waste = sum(a.D_f[f]*ev.waste_fractions[f, sol.pattern_assignments[f]] for f in a.stores)
    so = sum(a.D_f[f]*ev.stockout_fractions[f, sol.pattern_assignments[f]] for f in a.stores)
    fw = sum(ev.calculate_fw_emission(f, sol.pattern_assignments[f]) for f in a.stores)
    pat_op = sum(ev.c_fr[f, sol.pattern_assignments[f]] for f in a.stores)
    trc = trg = km = 0.0
    for day in range(6):
        for _, rt in sol.routes_by_day[day].items():
            if len(rt) <= 2: continue
            loads = {n: sol.p_frt.get((n, sol.pattern_assignments[n], day), 0.0) for n in rt if n}
            for i in range(len(rt)-1):
                dd = ev.delta[rt[i], rt[i+1]]
                cur = sum(loads[j] for j in rt[i+1:] if j)
                fuel = ev.eta*(ev.W0+cur)*dd
                km += dd; trc += ev.c_km*dd + ev.c_fuel*fuel; trg += ev.theta_TR*fuel
    wsA = {}
    for s in a.segment_names:
        dse = sum(a.D_fs[f][s] for f in a.stores)
        wse = sum(a.D_fs[f][s]*a.waste_frac_seg.get((f, s, sol.pattern_assignments[f]), 0.0) for f in a.stores)
        wsA[s] = (dse, wse)
    return {"demand": dem, "waste": waste, "stockout": so, "fw_co2": fw,
            "pat_op": pat_op, "tr_cost": trc, "tr_co2": trg, "km": km, "seg": wsA}

def run():
    inst = M.construct_test_instance()
    rlrp = M.solve_rlrp(inst, "logs/an_rlrp.txt")
    out = {}
    for s in inst.S:
        for depot_id in inst.depots:
            assigned = rlrp.customer_depot_assignment.get(s, {}).get(depot_id, [])
            if not assigned: continue
            fname = M.create_patt_instance_data(inst, rlrp, depot_id=depot_id, scenario=s)
            idata = alns.load_instance(fname)
            a = alns.ComprehensiveALNS(idata, alns.default_algorithm_params())
            sol = a.run_alns(max_iterations=PATT_ITER, time_limit=600)
            M.delete_patt_instance_file(fname)
            cfg, routes, depot_xy = build_sim_inputs(a, sol)
            entry = {"plan": plan_diag(a, sol), "pred": patt_pred(a, sol),
                     "id_map": {f: idata["store_id_mapping"][f] for f in a.stores}, "sim": {}}
            for variant in (2, 1, 3, 4):
                acc = None
                for ri in range(SIM_RUNS):
                    port = DESPort(cfg, routes, depot_xy, variant)
                    k = port.run(record_weeks=SIM_WEEKS, warmup_weeks=2, seed=20000+37*ri)
                    rec = {
                        "demand": {p: k.demand[p] for p in PRODUCTS},
                        "waste": {p: k.waste[p] for p in PRODUCTS},
                        "lost": {p: k.lost[p] for p in PRODUCTS},
                        "fw_co2": k.fw_co2_kg(), "tr_co2": k.transport_co2_kg,
                        "tr_cost": k.transport_cost, "km": k.distance_km,
                        "routes_run": k.routes_run, "cancel": k.routes_cancelled,
                        "drops": k.stores_dropped, "piggy_stops": k.piggyback_stops,
                        "piggy_units": k.piggyback_units,
                        "cancel_by_day": dict(k.cancel_by_day),
                        "drop_by_store": dict(k.drop_by_store),
                        "piggy_by_store": dict(k.piggy_by_store),
                        "avg_fill": (sum(k.fill_rates)/len(k.fill_rates)) if k.fill_rates else 0.0,
                        "so_cost": C_STOCKOUT*sum(k.lost.values()),
                        "fw_pur": C_PURCHASE*sum(k.waste.values()),
                    }
                    if acc is None: acc = rec
                    else:
                        def merge(x, y):
                            if isinstance(x, dict):
                                keys = set(x) | set(y)
                                return {kk: merge(x.get(kk, 0), y.get(kk, 0)) for kk in keys}
                            return x + y
                        acc = merge(acc, rec)
                # average
                def scale(x):
                    if isinstance(x, dict): return {kk: scale(v) for kk, v in x.items()}
                    return x / SIM_RUNS
                entry["sim"][variant] = scale(acc)
            out[s] = entry
    json.dump(out, open("/home/claude/analysis_results.json", "w"), indent=1, default=str)
    print("saved /home/claude/analysis_results.json")

if __name__ == "__main__":
    run()