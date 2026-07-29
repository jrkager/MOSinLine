import math
import numpy as np
from dataclasses import dataclass, field

PRODUCTS = ["A", "B", "C"]                  # A=fresh, B=dry, C=frozen
SEG_OF = {"A": "fresh", "B": "dry", "C": "frozen"}
THETA_FW = {"A": 4000.0, "B": 1500.0, "C": 2500.0}   # kg CO2e per unit (1 t)
SHELF_LIFE_A = 4
NO_EXPIRY = 10 ** 9

Q_UNITS = 25.6
EMPTY_KG = 14400.0
UNIT_KG = 1000.0
COST_PER_KG_KM = 0.00009      # = c_fuel * eta per kg*km
C_KM = 1.12
CO2_PER_KG_KM = 0.135e-3      # kg CO2 per kg*km (0.135 g)
P_FIFO = 0.70
DEMAND_CV = 0.25
Z_URGENCY = 1.645
MAX_SUPPLEMENT_DISTANCE = 140.0
DROP_THRESHOLD = 5
MIN_ROUTE_UNITS = 15
FILL_TRIGGER = 0.75
C_STOCKOUT = 100.0
C_PURCHASE = 800.0


@dataclass
class StoreCfg:
    xy: tuple                      # (x, y) km
    mu: dict                       # mu[p][t] t/day, p in A/B/C, t 0..5
    S: dict                        # S[p] order-up-to
    plan_flag: list                # [t]=True if delivery scheduled on weekday t


@dataclass
class SimKPIs:
    weeks: int = 0
    demand: dict = field(default_factory=lambda: {p: 0 for p in PRODUCTS})
    served: dict = field(default_factory=lambda: {p: 0 for p in PRODUCTS})
    lost: dict = field(default_factory=lambda: {p: 0 for p in PRODUCTS})
    waste: dict = field(default_factory=lambda: {p: 0 for p in PRODUCTS})
    transport_cost: float = 0.0
    transport_co2_kg: float = 0.0
    distance_km: float = 0.0
    routes_run: int = 0
    routes_cancelled: int = 0
    stores_dropped: int = 0
    piggyback_stops: int = 0
    piggyback_units: int = 0
    cancel_by_day: dict = field(default_factory=lambda: {t: 0 for t in range(6)})
    drop_by_store: dict = field(default_factory=dict)
    piggy_by_store: dict = field(default_factory=dict)
    fill_rates: list = field(default_factory=list)   # planned fill per dispatched route

    def fw_co2_kg(self):
        return sum(THETA_FW[p] * self.waste[p] for p in PRODUCTS)

    def waste_rate(self, p=None):
        if p:
            return self.waste[p] / self.demand[p] if self.demand[p] else 0.0
        d = sum(self.demand.values())
        return sum(self.waste.values()) / d if d else 0.0

    def stockout_rate(self, p=None):
        if p:
            return self.lost[p] / self.demand[p] if self.demand[p] else 0.0
        d = sum(self.demand.values())
        return sum(self.lost.values()) / d if d else 0.0


class DESPort:
    def __init__(self, stores_cfg, routes_by_day, depot_xy, variant):
        """stores_cfg: {sid(0..N-1): StoreCfg}; routes_by_day: {t: [[sid,...],...]}
        Variant -> active rules:
          V2: none (plan as-is)          V1: drop + cancel + piggyback(all)
          V3: drop + cancel              V4: drop + cancel + piggyback(skipped-only)
          V5: drop only                  V6: cancel only
          V7: piggyback(all) only        V8: drop + piggyback(skipped-only)
                                             (targeted repair of drop-skips; no cancel)
        """
        assert variant in (1, 2, 3, 4, 5, 6, 7, 8)
        self.cfg = stores_cfg
        self.routes = routes_by_day
        self.depot = depot_xy
        self.variant = variant
        self.drop_active = variant in (1, 3, 4, 5, 8)
        self.cancel_active = variant in (1, 3, 4, 6)
        self.piggy_mode = {1: "all", 4: "skipped", 7: "all", 8: "skipped"}.get(variant, None)
        self.N = len(stores_cfg)

    # ---- helpers matching the .alp ----
    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _mu_until_next(self, sid, p, today):
        """Sum of mu from today until the next scheduled delivery day (exclusive),
        matching expectedDemandUntilNext + daysUntilNextDelivery (Sunday lambda=0)."""
        flags = self.cfg[sid].plan_flag
        days = 7  # fallback: one week
        for k in range(1, 7):
            if flags[(today + k) % 6]:
                days = k
                break
        mu = 0.0
        for k in range(days):
            mu += self.cfg[sid].mu[p][(today + k) % 6]
        return mu

    def _urgency(self, shelves, sid, today):
        """min over products of (onHand - (mu_untilNext + z*sqrt(mu))); onOrder=0
        in the decoupled flow, matching calculateUrgencyScore."""
        best = float("inf")
        for p in PRODUCTS:
            on_hand = len(shelves[sid][p])
            mu = self._mu_until_next(sid, p, today)
            u = on_hand - (mu + Z_URGENCY * math.sqrt(max(mu, 0.0)))
            best = min(best, u)
        return best

    # ---- main run ----
    def run(self, record_weeks=52, warmup_weeks=2, seed=12345):
        rng = np.random.RandomState(seed)
        k = SimKPIs(weeks=record_weeks)

        shelves = {i: {p: [] for p in PRODUCTS} for i in range(self.N)}
        demand_carry = {i: {p: 0.0 for p in PRODUCTS} for i in range(self.N)}   # lists of expiry days
        pending = {i: {p: int(round(self.cfg[i].S[p])) for p in PRODUCTS} for i in range(self.N)}
        deliver_today = {i: {p: 0 for p in PRODUCTS} for i in range(self.N)}

        total_days = (warmup_weeks + record_weeks) * 6
        record_start = warmup_weeks * 6

        for sim_day in range(total_days):
            t = sim_day % 6
            rec = sim_day >= record_start
            skipped_today = [False] * self.N
            piggybacked_today = [False] * self.N

            # ---- 05:00 (1) expiry: product A only ----
            for i in range(self.N):
                sl = shelves[i]["A"]
                fresh_kept = [e for e in sl if e > sim_day]
                expired = len(sl) - len(fresh_kept)
                if expired and rec:
                    k.waste["A"] += expired
                shelves[i]["A"] = fresh_kept

            # ---- 05:00 (2) LEAD-1: today's planned arrival ----
            for i in range(self.N):
                is_del = self.cfg[i].plan_flag[t]
                for p in PRODUCTS:
                    deliver_today[i][p] = pending[i][p] if is_del else 0

            # ---- 05:00 (3) order-up-to commit for TOMORROW (before execution!) ----
            t_tom = (t + 1) % 6
            for i in range(self.N):
                for pi, p in enumerate(PRODUCTS):
                    if self.cfg[i].plan_flag[t_tom]:
                        ip = len(shelves[i][p]) + deliver_today[i][p]
                        mu_today = self.cfg[i].mu[p][t]
                        pending[i][p] = max(0, int(round(self.cfg[i].S[p] - ip + mu_today)))
                    else:
                        pending[i][p] = 0

            # ---- 05:00 (4) morning delivery onto shelves ----
            for i in range(self.N):
                for p in PRODUCTS:
                    q = deliver_today[i][p]
                    if q > 0:
                        exp = sim_day + (SHELF_LIFE_A if p == "A" else NO_EXPIRY)
                        shelves[i][p].extend([exp] * q)

            # ---- 06:00 route execution ----
            for route in self.routes.get(t, []):
                planned_load = sum(deliver_today[i][p] for i in route for p in PRODUCTS)
                if planned_load <= 0:
                    continue

                # CANCEL rule (Variants 1/3/4)
                if self.cancel_active and planned_load < MIN_ROUTE_UNITS:
                    if rec:
                        k.routes_cancelled += 1
                        k.cancel_by_day[t] += 1
                    for i in route:
                        tot = sum(deliver_today[i][p] for p in PRODUCTS)
                        if tot > 0:
                            skipped_today[i] = True
                            for p in PRODUCTS:            # un-deliver (newest units)
                                q = deliver_today[i][p]
                                if q:
                                    del shelves[i][p][-q:]
                    continue

                # DROP rule (Variants 1/3/4)
                kept_stops = []
                for i in route:
                    tot = sum(deliver_today[i][p] for p in PRODUCTS)
                    if tot <= 0:
                        continue
                    if self.drop_active and tot < DROP_THRESHOLD:
                        skipped_today[i] = True
                        if rec:
                            k.stores_dropped += 1
                            k.drop_by_store[i] = k.drop_by_store.get(i, 0) + 1
                        for p in PRODUCTS:                # un-deliver
                            q = deliver_today[i][p]
                            if q:
                                del shelves[i][p][-q:]
                    else:
                        kept_stops.append(i)
                if not kept_stops:
                    continue

                loads = {i: {p: deliver_today[i][p] for p in PRODUCTS} for i in kept_stops}

                # PIGGYBACK (Variants 1 & 4), fill-rate trigger < 75%
                if self.piggy_mode is not None:
                    load = sum(loads[i][p] for i in kept_stops for p in PRODUCTS)
                    if load < FILL_TRIGGER * Q_UNITS:
                        spare = Q_UNITS - load
                        last_xy = self.cfg[kept_stops[-1]].xy
                        cand = []
                        for j in range(self.N):
                            if j in kept_stops or piggybacked_today[j]:
                                continue
                            if self.piggy_mode == "skipped":
                                if not skipped_today[j]:
                                    continue
                            else:
                                if self.cfg[j].plan_flag[t]:
                                    continue      # scheduled today -> own route
                            topup = {p: max(0, int(round(self.cfg[j].S[p] - len(shelves[j][p]))))
                                     for p in PRODUCTS}
                            if sum(topup.values()) <= 0:
                                continue
                            if self._dist(self.cfg[j].xy, last_xy) > MAX_SUPPLEMENT_DISTANCE:
                                continue
                            cand.append((self._urgency(shelves, j, t), j, topup))
                        cand.sort(key=lambda x: x[0])
                        for _, j, topup in cand:
                            any_added = False
                            for p in PRODUCTS:
                                q = min(int(spare), topup[p])
                                if q <= 0:
                                    continue
                                exp = sim_day + (SHELF_LIFE_A if p == "A" else NO_EXPIRY)
                                shelves[j][p].extend([exp] * q)
                                loads.setdefault(j, {pp: 0 for pp in PRODUCTS})
                                loads[j][p] += q
                                spare -= q
                                any_added = True
                                if rec:
                                    k.piggyback_units += q
                            if any_added:
                                kept_stops.append(j)
                                piggybacked_today[j] = True
                                if rec:
                                    k.piggyback_stops += 1
                                    k.piggy_by_store[j] = k.piggy_by_store.get(j, 0) + 1
                            if spare <= 0:
                                break

                # truck KPI: depot -> stops -> depot, decreasing load
                if rec:
                    k.routes_run += 1
                    k.fill_rates.append(planned_load / Q_UNITS)
                    pos = self.depot
                    remaining = sum(loads[i][p] for i in kept_stops for p in PRODUCTS)
                    for i in kept_stops:
                        d = self._dist(pos, self.cfg[i].xy)
                        mass = EMPTY_KG + remaining * UNIT_KG
                        k.distance_km += d
                        k.transport_cost += d * mass * COST_PER_KG_KM + d * C_KM
                        k.transport_co2_kg += d * mass * CO2_PER_KG_KM
                        remaining -= sum(loads[i].values())
                        pos = self.cfg[i].xy
                    d = self._dist(pos, self.depot)          # empty return
                    k.distance_km += d
                    k.transport_cost += d * EMPTY_KG * COST_PER_KG_KM + d * C_KM
                    k.transport_co2_kg += d * EMPTY_KG * CO2_PER_KG_KM
                else:
                    pass  # warmup: execute physically (already did), skip KPI

            # ---- 08:00-17:00 demand ----
            for i in range(self.N):
                for p in PRODUCTS:
                    mu = self.cfg[i].mu[p][t]
                    if mu <= 0:
                        continue
                    draw = max(0.0, rng.normal(mu, DEMAND_CV * mu))
                    pool = demand_carry[i][p] + draw          # carry-over discretization
                    n = int(pool)
                    demand_carry[i][p] = pool - n
                    if rec:
                        k.demand[p] += n
                    sl = shelves[i][p]
                    for _ in range(n):
                        if sl:
                            idx = 0 if rng.random() < P_FIFO else len(sl) - 1
                            sl.pop(idx)
                            if rec:
                                k.served[p] += 1
                        else:
                            if rec:
                                k.lost[p] += 1
        return k


def kpis_to_row(k, label):
    dem = sum(k.demand.values())
    return {
        "run": label,
        "demand_u": dem,
        "waste%": 100 * k.waste_rate(),
        "wasteA%": 100 * k.waste_rate("A"),
        "stockout%": 100 * k.stockout_rate(),
        "FW_CO2_kg/wk": k.fw_co2_kg() / k.weeks,
        "TR_CO2_kg/wk": k.transport_co2_kg / k.weeks,
        "TR_cost/wk": k.transport_cost / k.weeks,
        "km/wk": k.distance_km / k.weeks,
        "cancel/wk": k.routes_cancelled / k.weeks,
        "drop/wk": k.stores_dropped / k.weeks,
        "piggy_u/wk": k.piggyback_units / k.weeks,
        "so_cost/wk": C_STOCKOUT * sum(k.lost.values()) / k.weeks,
        "fw_pur/wk": C_PURCHASE * sum(k.waste.values()) / k.weeks,
    }
