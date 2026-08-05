# COMPREHENSIVE ALNS IMPLEMENTATION — MOSinLine PATT module, product-segment version
# Combines pattern optimization (Stage 1) and routing optimization (Stage 2)
# Following Huebner and Oestermeier (2019) approach exactly
#
# === MODIFICATIONS vs. standalone version ===
# 1. Day-varying demand: mu_{f,t} = d_bar_f * w_t (day-of-week multipliers)
# 2. Day-varying sigma: sigma_{f,t} = CV * mu_{f,t}
# 3. Unified (R,S) FIFO/LIFO shelf simulation for delivery quantities + waste
# 4. Food waste purchasing cost (c_purchase * D_f * wf) added to economic side
# 5. No alpha/beta — theta_FW and theta_TR are direct emission factors (kg CO2)
# 6. Mixed consumer behavior: p_FIFO oldest-first + p_LIFO newest-first
#
# === MOSinLine INTEGRATION (this file is a drop-in for patt/alns.py) ===
# A. load_instance passes through all extra coefficients from the instance JSON
# B. delta is taken from instance["distances"] if provided (same matrix as RLRP),
#    otherwise computed as euclidean from loc
# C. scalar parameters (vehicle_capacity, cost_per_km, marginal_co2_emissions,
#    weighting_factor_patt, vehicle_empty_weight, ...) are read from the instance
#    file with the standalone defaults as fallback.
#    RLRP consistency: marginal_co2_emissions == eta * theta_TR
# D. main(instance_file_name=...) returns (best_solution, instance_data)
#
# === PRODUCT SEGMENTS (tier 1) ===
# Segments follow the temperature-specific segmentation of grocery distribution
# (Huebner & Ostermeier 2018, Transportation Science) and MOSinLine's ProductClass:
#   dry (ambient), fresh (chilled), frozen.
# Per segment: own demand mu_{f,s,t}, own shelf life SL_s, own embodied-emission
# factor theta_FW_s (kg CO2e per t wasted, production-to-shelf boundary;
# ifeu 2020 / Poore & Nemecek 2018).
# One delivery pattern per store (co-delivery): the (R,S) simulation runs per
# (store, segment, pattern); truck loads aggregate over segments
#   p_frt[f,r,t] = sum_s p_fsrt[f,s,r,t]
# and the pattern shelf-life filter uses SL_eff(f) = min over the store's
# segments with finite SL. Segments with shelf_life=None never expire within
# the weekly horizon (expiry waste = 0 by construction).
# Backward compatible: without "demand_by_segment" in the instance, all demand
# is treated as segment "fresh" and the model behaves like the single-segment
# version.

import numpy as np
import json
import random
import math
import time
import itertools
import os
import pandas as pd
from copy import deepcopy
import sys
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.stats import norm

print("Starting COMPREHENSIVE ALNS Implementation (segment-aware, MOSinLine-ready)")
print("Implements full ALNS approach: Pattern optimization (Stage 1) + Routing optimization (Stage 2)")
print("=== DAY-VARYING demand | (R,S) FIFO/LIFO shelf simulation | product segments ===")

# Default product-segment configuration.
# shelf_life: days (None = no expiry within the weekly planning horizon)
# theta_FW: kg CO2e per tonne of wasted food, embodied (production-to-shelf) boundary.
# Order-of-magnitude defaults from representative baskets
# (ifeu 2020 "an der Supermarktkasse"; Poore & Nemecek 2018, Science).
DEFAULT_SEGMENTS = {
    "dry":    {"shelf_life": None, "theta_FW": 1500.0},
    "fresh":  {"shelf_life": 4,    "theta_FW": 4000.0},
    "frozen": {"shelf_life": None, "theta_FW": 2500.0},
}

class ComprehensiveSolution:
    """Represents a complete solution with patterns and routes"""
    def __init__(self, pattern_assignments, routes_by_day, evaluator, p_frt, stores):
        self.pattern_assignments = pattern_assignments
        self.routes_by_day = routes_by_day
        self.evaluator = evaluator
        self.p_frt = p_frt
        self.stores = stores
        self.cost = None
        self.pattern_cost = None
        self.routing_cost = None
        self.num_vehicles = None
        self._calculate_cost()
    
    def _calculate_cost(self):
        self.pattern_cost = 0
        for store, pattern_id in self.pattern_assignments.items():
            self.pattern_cost += self.evaluator.calculate_pattern_cost(store, pattern_id)
        
        self.routing_cost = 0
        for day, vehicle_routes in self.routes_by_day.items():
            for vehicle, route in vehicle_routes.items():
                if len(route) > 2:
                    loads = self._get_route_loads(route, day)
                    self.routing_cost += self.evaluator.calculate_route_cost(route, loads)
        
        violations = self.validate_constraints()
        penalty_per_violation = 10000
        constraint_penalty = len(violations) * penalty_per_violation
        
        self.cost = self.pattern_cost + self.routing_cost + constraint_penalty
    
    def _get_route_loads(self, route, day):
        loads = {}
        for node in route:
            if node != 0:
                pattern_id = self.pattern_assignments[node]
                loads[node] = self.p_frt.get((node, pattern_id, day), 0)
        return loads
    
    def copy(self):
        new_patterns = self.pattern_assignments.copy()
        new_routes = {}
        for day, vehicle_routes in self.routes_by_day.items():
            new_routes[day] = {}
            for vehicle, route in vehicle_routes.items():
                new_routes[day][vehicle] = route.copy()
        new_sol = ComprehensiveSolution(
            new_patterns, new_routes, self.evaluator,
            self.p_frt, self.stores
        )
        new_sol.num_vehicles = self.num_vehicles
        return new_sol
    
    def get_all_served_stores(self, day):
        served = set()
        for vehicle, route in self.routes_by_day[day].items():
            for node in route:
                if node != 0:
                    served.add(node)
        return served
    
    def get_required_deliveries(self, day):
        required = set()
        for store in self.stores:
            pattern_id = self.pattern_assignments[store]
            if self.evaluator.patterns[pattern_id][day] == 1:
                required.add(store)
        return required
    
    def verify_all_deliveries(self):
        for day in range(6):
            required = self.get_required_deliveries(day)
            served = self.get_all_served_stores(day)
            if required != served:
                missing = required - served
                extra = served - required
                if missing:
                    print(f"Day {day}: Missing deliveries to stores {missing}")
                if extra:
                    print(f"Day {day}: Extra deliveries to stores {extra}")
                return False
        return True
    
    def validate_constraints(self):
        violations = []
        
        for day in range(6):
            total_day_delivery = 0
            for store in self.get_all_served_stores(day):
                pattern_id = self.pattern_assignments[store]
                total_day_delivery += self.p_frt.get((store, pattern_id, day), 0)
            
            if total_day_delivery < self.evaluator.Q_day_min:
                violations.append(f"Day {day}: Total delivery {total_day_delivery:.1f} < {self.evaluator.Q_day_min}")
            elif total_day_delivery > self.evaluator.Q_day_max:
                violations.append(f"Day {day}: Total delivery {total_day_delivery:.1f} > {self.evaluator.Q_day_max}")
        
        for store in self.stores:
            pattern_id = self.pattern_assignments[store]
            for day in range(6):
                delivery_amount = self.p_frt.get((store, pattern_id, day), 0)
                if delivery_amount > self.evaluator.gamma_f[store]:
                    violations.append(f"Store {store} day {day}: Delivery {delivery_amount:.1f} > capacity {self.evaluator.gamma_f[store]}")
        
        for day in range(6):
            required = self.get_required_deliveries(day)
            served = self.get_all_served_stores(day)
            missing = required - served
            extra = served - required
            if missing:
                violations.append(f"Day {day}: Missing deliveries {sorted(missing)}")
            if extra:
                violations.append(f"Day {day}: Extra deliveries {sorted(extra)}")

        Q = getattr(self.evaluator, "Q", None)
        if Q is not None:
            for day in range(6):
                for vehicle, route in self.routes_by_day[day].items():
                    if len(route) <= 2:
                        continue
                    loads = self._get_route_loads(route, day)
                    total_load = sum(loads.values())
                    if total_load > Q + 1e-9:
                        violations.append(
                            f"Day {day} vehicle {vehicle}: Route load {total_load:.1f} > Q {Q}"
                        )

        return violations

class CombinedEvaluator:
    """Evaluates both pattern costs and route costs.
    No alpha/beta — theta_TR is a direct emission factor (kg CO2 per L).
    Food-waste emissions are PER SEGMENT: fw_emission[f, r] (kg CO2e) is
    precomputed as sum_s theta_FW_s * D_{f,s} * w_{f,s,r}; the scalar
    theta_FW is kept only as a legacy fallback.
    """
    def __init__(self, loc, delta, c_fr, waste_fractions, stockout_fractions, fw_emission,
                 D_f, patterns,
                 c_km, c_fuel, theta_FW, theta_TR, eta, W0, lambda_param, Q,
                 gamma_f, Q_day_min, Q_day_max,
                 c_purchase=150.0, c_stockout=300.0):
        self.c_fr = c_fr
        self.waste_fractions = waste_fractions
        self.stockout_fractions = stockout_fractions
        self.fw_emission = fw_emission
        self.D_f = D_f
        self.patterns = patterns
        self.theta_FW = theta_FW
        self.c_purchase = c_purchase
        self.c_stockout = c_stockout
        
        self.loc = loc
        self.delta = delta
        self.c_km = c_km
        self.c_fuel = c_fuel
        self.theta_TR = theta_TR
        self.eta = eta
        self.W0 = W0
        self.Q = Q
        
        self.lambda_param = lambda_param
        
        self.gamma_f = gamma_f
        self.Q_day_min = Q_day_min
        self.Q_day_max = Q_day_max
    
    def calculate_fw_emission(self, store, pattern_id):
        """Segment-weighted food-waste emission (kg CO2e) for (store, pattern)."""
        val = self.fw_emission.get((store, pattern_id), None)
        if val is not None:
            return val
        # legacy fallback: single theta_FW on the aggregate waste fraction
        wf = self.waste_fractions[store, pattern_id]
        return self.theta_FW * self.D_f[store] * wf
    
    def calculate_pattern_cost(self, store, pattern_id):
        economic_cost = self.c_fr[store, pattern_id]
        wf = self.waste_fractions[store, pattern_id]
        sf = self.stockout_fractions[store, pattern_id]
        
        fw_purchase_cost = self.c_purchase * self.D_f[store] * wf
        fw_emission_cost = self.calculate_fw_emission(store, pattern_id)
        stockout_cost = self.c_stockout * self.D_f[store] * sf
        
        return ((1 - self.lambda_param) * (economic_cost + fw_purchase_cost + stockout_cost) + 
                self.lambda_param * fw_emission_cost)
    
    def calculate_route_cost(self, route, loads):
        if len(route) <= 2:
            return 0
        
        transport_cost = 0
        pollution_cost = 0
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = self.delta[from_node, to_node]
            
            current_load = sum(loads[j] for j in route[i+1:] if j != 0)
            fuel = self.eta * (self.W0 + current_load) * distance
            transport_cost += self.c_km * distance + self.c_fuel * fuel
            emissions = self.theta_TR * fuel
            pollution_cost += emissions
        
        total_cost = (1 - self.lambda_param) * transport_cost + self.lambda_param * pollution_cost
        return total_cost
    
    def is_route_feasible(self, route, loads):
        total_load = sum(loads.get(node, 0) for node in route if node != 0)
        return total_load <= self.Q

class ALNSPatternOperators:
    """Pattern optimization operators from the paper"""
    
    def __init__(self, evaluator, stores, patterns, R, distances, p_frt, Q, gamma_f, feasible_patterns_by_store, alpha=6):
        self.evaluator = evaluator
        self.stores = stores
        self.patterns = patterns
        self.R = R
        self.distances = distances
        self.max_distance = max(distances.values()) if distances else 1.0
        self.alpha = alpha
        self.p_frt = p_frt
        self.Q = Q 
        self.gamma_f = gamma_f
        self.feasible_patterns_by_store = feasible_patterns_by_store
    
    def _is_pattern_feasible_for_store(self, store, pattern_id):
        return pattern_id in self.feasible_patterns_by_store.get(store, [])
    
    def _pattern_similarity(self, pattern1, pattern2):
        if len(pattern1) != len(pattern2) or len(pattern1) == 0:
            return 0.0
        overlap = sum(1 for t in range(len(pattern1)) if pattern1[t] == 1 and pattern2[t] == 1)
        return overlap / len(pattern1)

    def _pick_ranked_by_zeta_alpha(self, ranked_list):
        if not ranked_list:
            return None
        zeta = random.random()
        idx = int((zeta ** self.alpha) * len(ranked_list))
        idx = min(idx, len(ranked_list) - 1)
        return ranked_list[idx]

    def _choose_new_pattern_higher_similarity(self, seed_pat_id, cand_store, cand_current_pat_id):
        feasible = list(self.feasible_patterns_by_store.get(cand_store, []))
        if not feasible:
            return None
        seed_bits = self.patterns[seed_pat_id]
        current_bits = self.patterns[cand_current_pat_id]
        omega_prev = self._pattern_similarity(seed_bits, current_bits)
        improving = []
        for r in feasible:
            if r == cand_current_pat_id:
                continue
            omega_r = self._pattern_similarity(seed_bits, self.patterns[r])
            if omega_r > omega_prev:
                improving.append(r)
        if not improving:
            return None
        return random.choice(improving)

    def proximity_operator(self, solution, c_stores, beta=0.8, max_tries=50):
        changed_stores = []
        O = list(self.stores)
        if not O or c_stores <= 0:
            return changed_stores
        L = []
        L.append(random.choice(O))
        tries = 0
        while len(L) < c_stores:
            tries += 1
            if tries > max_tries * c_stores:
                break
            selected_store = random.choice(L)
            seed_pat = solution.pattern_assignments[selected_store]
            seed_bits = self.patterns[seed_pat]
            ranked = []
            for f in O:
                if f in L:
                    continue
                d = self.distances.get((selected_store, f), None)
                if d is None or self.max_distance <= 0:
                    geo = 1.0
                else:
                    geo = d / self.max_distance
                f_pat = solution.pattern_assignments[f]
                f_bits = self.patterns[f_pat]
                omega = self._pattern_similarity(seed_bits, f_bits)
                score = beta * geo + (1.0 - beta) * (1.0 - omega)
                ranked.append((f, score))
            if not ranked:
                break
            ranked.sort(key=lambda x: x[1])
            picked = self._pick_ranked_by_zeta_alpha(ranked)
            if picked is None:
                break
            selected_f = picked[0]
            cand_current_pat = solution.pattern_assignments[selected_f]
            if seed_pat == cand_current_pat:
                continue
            new_pat = self._choose_new_pattern_higher_similarity(
                seed_pat_id=seed_pat,
                cand_store=selected_f,
                cand_current_pat_id=cand_current_pat
            )
            if new_pat is None:
                continue
            L.append(selected_f)
            changed_stores.append((selected_f, new_pat))
        return changed_stores

    def sales_volume_operator(self, solution, c_stores, max_tries=50):
        changed_stores = []
        O = list(self.stores)
        if not O or c_stores <= 0:
            return changed_stores
        L = []
        L.append(random.choice(O))
        tries = 0
        while len(L) < c_stores:
            tries += 1
            if tries > max_tries * c_stores:
                break
            selected_store = random.choice(L)
            seed_pat = solution.pattern_assignments[selected_store]
            ranked = []
            for f in O:
                if f in L:
                    continue
                R3 = abs(self.evaluator.D_f[selected_store] - self.evaluator.D_f[f])
                ranked.append((f, R3))
            if not ranked:
                break
            ranked.sort(key=lambda x: x[1])
            picked = self._pick_ranked_by_zeta_alpha(ranked)
            if picked is None:
                break
            selected_f = picked[0]
            cand_current_pat = solution.pattern_assignments[selected_f]
            if seed_pat == cand_current_pat:
                continue
            new_pat = self._choose_new_pattern_higher_similarity(
                seed_pat_id=seed_pat,
                cand_store=selected_f,
                cand_current_pat_id=cand_current_pat
            )
            if new_pat is None:
                continue
            L.append(selected_f)
            changed_stores.append((selected_f, new_pat))
        return changed_stores
   
    def cost_related_operator(self, solution, c_stores, max_tries=50):
        changed_stores = []
        O = list(self.stores)
        L = []
        if not O or c_stores <= 0:
            return changed_stores
        tries = 0
        while len(L) < c_stores:
            tries += 1
            if tries > max_tries * c_stores:
                break
            remaining_stores = [f for f in O if f not in L]
            if not remaining_stores:
                break
            cost_ranking = []
            for f in remaining_stores:
                r_cur = solution.pattern_assignments[f]
                c_cur = self.evaluator.c_fr[f, r_cur]
                cost_ranking.append((f, c_cur))
            cost_ranking.sort(key=lambda x: x[1], reverse=True)
            zeta = random.random()
            idx = int((zeta ** self.alpha) * len(cost_ranking))
            idx = min(idx, len(cost_ranking) - 1)
            f_sel = cost_ranking[idx][0]
            r_cur = solution.pattern_assignments[f_sel]
            c_cur = self.evaluator.c_fr[f_sel, r_cur]
            feas = list(self.feasible_patterns_by_store.get(f_sel, []))
            if not feas:
                continue
            c_min = min(self.evaluator.c_fr[f_sel, r] for r in feas)
            if c_cur <= c_min + 1e-9:
                continue
            cheaper = [r for r in feas if r != r_cur and self.evaluator.c_fr[f_sel, r] < c_cur - 1e-9]
            if not cheaper:
                continue
            r_new = random.choice(cheaper)
            L.append(f_sel)
            changed_stores.append((f_sel, r_new))
        return changed_stores
    
    def move_one_operator(self, solution, c_stores, max_tries=50):
        changed_stores = []
        O = list(self.stores)
        L = []
        if not O or c_stores <= 0:
            return changed_stores
        if not hasattr(self, "_pattern_tuple_to_id"):
            self._pattern_tuple_to_id = {tuple(self.patterns[r]): r for r in self.R}
        tries = 0
        while len(L) < min(c_stores, len(O)):
            tries += 1
            if tries > max_tries * c_stores:
                break
            remaining = [f for f in O if f not in L]
            if not remaining:
                break
            f_sel = random.choice(remaining)
            r_cur = solution.pattern_assignments[f_sel]
            bits = list(self.patterns[r_cur])
            T = len(bits)
            freq = sum(bits)
            if freq != T:
                ones = [t for t in range(T) if bits[t] == 1]
                zeros = [t for t in range(T) if bits[t] == 0]
                if ones and zeros:
                    t1 = random.choice(ones)
                    t2 = random.choice(zeros)
                    new_bits = bits.copy()
                    new_bits[t1] = 0
                    new_bits[t2] = 1
                    new_id = self._pattern_tuple_to_id.get(tuple(new_bits), None)
                    if new_id is not None and new_id in self.feasible_patterns_by_store.get(f_sel, []):
                        changed_stores.append((f_sel, new_id))
            L.append(f_sel)
        return changed_stores

    def move_two_operator(self, solution, c_stores, max_tries=50):
        changed_stores = []
        O = list(self.stores)
        L = []
        if not O or c_stores <= 0:
            return changed_stores
        if not hasattr(self, "_pattern_tuple_to_id"):
            self._pattern_tuple_to_id = {tuple(self.patterns[r]): r for r in self.R}
        tries = 0
        while len(L) < min(c_stores, len(O)):
            tries += 1
            if tries > max_tries * c_stores:
                break
            remaining = [f for f in O if f not in L]
            if not remaining:
                break
            f_sel = random.choice(remaining)
            r_cur = solution.pattern_assignments[f_sel]
            bits = list(self.patterns[r_cur])
            T = len(bits)
            freq = sum(bits)
            if freq == T:
                pass
            elif freq == T - 1 or freq == 1:
                ones = [t for t in range(T) if bits[t] == 1]
                zeros = [t for t in range(T) if bits[t] == 0]
                if ones and zeros:
                    t1 = random.choice(ones)
                    t2 = random.choice(zeros)
                    new_bits = bits.copy()
                    new_bits[t1] = 0
                    new_bits[t2] = 1
                    new_id = self._pattern_tuple_to_id.get(tuple(new_bits), None)
                    if new_id is not None and new_id in self.feasible_patterns_by_store.get(f_sel, []):
                        changed_stores.append((f_sel, new_id))
            else:
                ones = [t for t in range(T) if bits[t] == 1]
                zeros = [t for t in range(T) if bits[t] == 0]
                if len(ones) >= 2 and len(zeros) >= 2:
                    move_days = random.sample(ones, 2)
                    target_days = random.sample(zeros, 2)
                    new_bits = bits.copy()
                    for t in move_days:
                        new_bits[t] = 0
                    for t in target_days:
                        new_bits[t] = 1
                    new_id = self._pattern_tuple_to_id.get(tuple(new_bits), None)
                    if new_id is not None and new_id in self.feasible_patterns_by_store.get(f_sel, []):
                        changed_stores.append((f_sel, new_id))
            L.append(f_sel)
        return changed_stores

    def random_operator(self, solution, c_stores, max_tries=50):
        changed_stores = []
        O = list(self.stores)
        L = []
        tries = 0
        while len(L) < min(c_stores, len(O)):
            tries += 1
            if tries > max_tries * c_stores:
                break
            remaining = [f for f in O if f not in L]
            if not remaining:
                break
            f = random.choice(remaining)
            cur = solution.pattern_assignments[f]
            feas = [r for r in self.feasible_patterns_by_store.get(f, []) if r != cur]
            if feas:
                changed_stores.append((f, random.choice(feas)))
            L.append(f)
        return changed_stores
    
    def food_waste_cost_operator(self, solution, c_stores, alpha=6):
        O = list(self.stores)
        L = []
        changes = []
        max_trials = 10 * max(1, len(O))
        trials = 0

        def fw_cost(f, r):
            ev = self.evaluator
            wf = ev.waste_fractions[f, r]
            purchase = ev.c_purchase * ev.D_f[f] * wf
            emission = ev.calculate_fw_emission(f, r)  # segment-weighted
            lam = ev.lambda_param
            return (1 - lam) * purchase + lam * emission

        while len(L) < c_stores and O and trials < max_trials:
            trials += 1
            ranked = []
            for f in O:
                r_cur = solution.pattern_assignments[f]
                ranked.append((fw_cost(f, r_cur), f))
            ranked.sort(key=lambda x: x[0], reverse=True)
            zeta = random.random()
            idx = int((zeta ** alpha) * len(ranked))
            idx = min(idx, len(ranked) - 1)
            _, f = ranked[idx]
            r_cur = solution.pattern_assignments[f]
            cur_cost = fw_cost(f, r_cur)
            feas = self.feasible_patterns_by_store.get(f, [])
            improving = [r for r in feas if (r != r_cur and fw_cost(f, r) < cur_cost)]
            if not improving:
                continue
            r_new = random.choice(improving)
            changes.append((f, r_new))
            O.remove(f)
            L.append(f)
        return changes

    def transport_pollution_cost_operator(self, solution, c_stores, alpha=6):
        O = list(self.stores)
        L = []
        changes = []
        max_trials = 10 * max(1, len(O))
        trials = 0

        def tp_proxy_cost(f, r):
            ev = self.evaluator
            total = 0.0
            dist_round = 2.0 * ev.delta[0, f]
            for t in range(6):
                if self.patterns[r][t] == 1:
                    q = self.p_frt.get((f, r, t), 0.0)
                    fuel = ev.eta * (ev.W0 + q) * dist_round
                    emissions = ev.theta_TR * fuel
                    total += emissions  # This operator only ranks by pollution, unchanged
            return total

        while len(L) < c_stores and O and trials < max_trials:
            trials += 1
            ranked = []
            for f in O:
                r_cur = solution.pattern_assignments[f]
                ranked.append((tp_proxy_cost(f, r_cur), f))
            ranked.sort(key=lambda x: x[0], reverse=True)
            zeta = random.random()
            idx = int((zeta ** alpha) * len(ranked))
            idx = min(idx, len(ranked) - 1)
            _, f = ranked[idx]
            r_cur = solution.pattern_assignments[f]
            cur_cost = tp_proxy_cost(f, r_cur)
            feas = self.feasible_patterns_by_store.get(f, [])
            improving = [r for r in feas if (r != r_cur and tp_proxy_cost(f, r) < cur_cost)]
            if not improving:
                continue
            r_new = random.choice(improving)
            changes.append((f, r_new))
            O.remove(f)
            L.append(f)
        return changes

    def ml_cluster_spatial_operator(self, solution, c_stores, n_clusters=5):
            changed_stores = []
            store_ids = list(self.stores)
            n_samples = len(store_ids)
            if n_samples < 2:
                return []
            X = np.array([self.evaluator.loc[s] for s in store_ids])
            heuristic_max = max(4, n_samples // 5)
            lower_bound = 2
            upper_bound = min(heuristic_max, n_samples)
            if upper_bound < lower_bound:
                upper_bound = lower_bound
            if upper_bound > n_samples:
                upper_bound = n_samples
            actual_k = random.randint(lower_bound, upper_bound)
            kmeans = KMeans(n_clusters=actual_k, n_init=10, random_state=random.randint(0, 10000))
            labels = kmeans.fit_predict(X)
            target_label = random.choice(np.unique(labels))
            target_indices = np.where(labels == target_label)[0]
            cluster_stores = [store_ids[i] for i in target_indices]
            random.shuffle(cluster_stores)
            for f in cluster_stores:
                if len(changed_stores) >= c_stores:
                    break
                feas = self.feasible_patterns_by_store.get(f, [])
                cur = solution.pattern_assignments[f]
                candidates = [r for r in feas if r != cur]
                if candidates:
                    r_new = random.choice(candidates)
                    changed_stores.append((f, r_new))
            return changed_stores
    
    def smart_eco_pattern_operator(self, solution, c_stores, max_tries=50):
        # Pure pattern-cost operator: greedily give each candidate store the feasible
        # pattern with the lowest calculate_pattern_cost (operational + FW purchase +
        # stockout + lambda*FW emission). NO transport proxy -- the JOINT operator already
        # handles transport via exact route insertion, so a crude round-trip proxy here is
        # redundant and can mislead. Store selection: cost-ranked descending + zeta^alpha
        # biased pick (same as baseline's pattern-dependent cost operator). Pattern choice
        # stays GREEDY (single cheapest feasible pattern).
        changed_stores = []
        O = list(self.stores)
        L = []
        if not O or c_stores <= 0:
            return changed_stores
        ev = self.evaluator
        tries = 0
        while len(L) < c_stores:
            tries += 1
            if tries > max_tries * c_stores:
                break
            remaining = [f for f in O if f not in L]
            if not remaining:
                break
            ranking = [(ev.calculate_pattern_cost(f, solution.pattern_assignments[f]), f)
                    for f in remaining]
            ranking.sort(key=lambda x: x[0], reverse=True)
            zeta = random.random()
            idx = int((zeta ** self.alpha) * len(ranking))
            idx = min(idx, len(ranking) - 1)
            store = ranking[idx][1]

            feas = list(self.feasible_patterns_by_store.get(store, []))
            if not feas:
                L.append(store)
                continue
            current_pattern = solution.pattern_assignments[store]
            best_r = current_pattern
            min_score = float('inf')
            for r in feas:
                total_score = ev.calculate_pattern_cost(store, r)
                if total_score < min_score:
                    min_score = total_score
                    best_r = r
            if best_r != current_pattern:
                changed_stores.append((store, best_r))
            L.append(store)
        return changed_stores

class LNSRoutingOperators:
    """Routing optimization operators (Stage-2 LNS)"""

    def __init__(self, evaluator, p_frt, alpha=6):
        self.evaluator = evaluator
        self.p_frt = p_frt
        self.alpha = alpha

    def shaw_removal(self, solution, day, num_to_remove, mu=0.6, xi=0.4):
        routes = solution.routes_by_day[day]
        all_customers = []
        for route in routes.values():
            for node in route:
                if node != 0:
                    all_customers.append(node)
        all_customers = list(dict.fromkeys(all_customers))
        if len(all_customers) <= 1:
            return []
        c_max = 0.0
        for i in all_customers:
            for j in all_customers:
                if i == j:
                    continue
                c_ij = self.evaluator.c_km * self.evaluator.delta[i, j]
                if c_ij > c_max:
                    c_max = c_ij
        if c_max <= 0:
            c_max = 1.0
        o_max = 0.0
        for i in all_customers:
            pat_i = solution.pattern_assignments[i]
            o_i = self.p_frt.get((i, pat_i, day), 0.0)
            if o_i > o_max:
                o_max = o_i
        if o_max <= 0:
            o_max = 1.0
        seed = random.choice(all_customers)
        removed = [seed]
        remaining = [c for c in all_customers if c != seed]
        while len(removed) < num_to_remove and remaining:
            ref = random.choice(removed)
            pat_ref = solution.pattern_assignments[ref]
            o_ref = self.p_frt.get((ref, pat_ref, day), 0.0)
            relatedness_scores = []
            for j in remaining:
                pat_j = solution.pattern_assignments[j]
                o_j = self.p_frt.get((j, pat_j, day), 0.0)
                c_refj = self.evaluator.c_km * self.evaluator.delta[ref, j]
                term1 = c_refj / c_max
                term3 = abs(o_ref - o_j) / o_max
                R = mu * term1 + xi * term3
                relatedness_scores.append((R, j))
            relatedness_scores.sort(key=lambda x: x[0])
            y = random.random()
            idx = int((y ** self.alpha) * len(relatedness_scores))
            idx = min(idx, len(relatedness_scores) - 1)
            chosen = relatedness_scores[idx][1]
            removed.append(chosen)
            remaining.remove(chosen)
        return removed

    def _build_loads_for_route(self, solution, day, route):
        loads = {}
        for n in route:
            if n == 0:
                continue
            pat = solution.pattern_assignments[n]
            loads[n] = self.p_frt.get((n, pat, day), 0.0)
        return loads

    def _route_cost_econ_plus_pollution_precise(self, solution, day, route):
        if route is None or len(route) <= 2:
            return 0.0
        ev = self.evaluator
        q = {}
        for n in route:
            if n != 0:
                pat = solution.pattern_assignments[n]
                q[n] = self.p_frt.get((n, pat, day), 0.0)
        suffix_load = [0.0] * (len(route) + 1)
        for i in range(len(route) - 1, -1, -1):
            suffix_load[i] = suffix_load[i + 1]
            node = route[i]
            if node != 0:
                suffix_load[i] += q.get(node, 0.0)
        transport_cost = 0.0
        pollution_cost = 0.0
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            dist = ev.delta[a, b]
            current_load = suffix_load[i + 1]
            fuel = ev.eta * (ev.W0 + current_load) * dist
            transport_cost += ev.c_km * dist + ev.c_fuel * fuel
            emissions = ev.theta_TR * fuel
            pollution_cost += emissions
        lam = ev.lambda_param
        return (1.0 - lam) * transport_cost + lam * pollution_cost

    def regret_k_insertion(self, solution, day, removed_customers, k=2):
        routes = solution.routes_by_day[day]
        removed_list = list(removed_customers)
        removed_set = set(removed_list)
        for veh, rt in list(routes.items()):
            new_rt = [n for n in rt if (n == 0 or n not in removed_set)]
            if not new_rt:
                new_rt = [0, 0]
            if new_rt[0] != 0:
                new_rt = [0] + new_rt
            if new_rt[-1] != 0:
                new_rt = new_rt + [0]
            if len(new_rt) < 2:
                new_rt = [0, 0]
            if new_rt == [0, 0, 0]:
                new_rt = [0, 0]
            routes[veh] = new_rt
        num_vehicles = getattr(solution, "num_vehicles", None)
        if num_vehicles is not None:
            for v in range(num_vehicles):
                if v not in routes:
                    routes[v] = [0, 0]

        def delta_cost_for_insertion(customer, vehicle, pos):
            route = routes[vehicle]
            new_route = route[:pos] + [customer] + route[pos:]
            loads_new = self._build_loads_for_route(solution, day, new_route)
            if not self.evaluator.is_route_feasible(new_route, loads_new):
                return None
            old_cost = self._route_cost_econ_plus_pollution_precise(solution, day, route)
            new_cost = self._route_cost_econ_plus_pollution_precise(solution, day, new_route)
            return new_cost - old_cost

        unassigned = list(removed_list)
        while unassigned:
            best_regret = -float("inf")
            best_customer = None
            best_vehicle = None
            best_pos = None
            for customer in unassigned:
                insertion_costs = []
                for vehicle, route in routes.items():
                    for pos in range(1, len(route)):
                        dc = delta_cost_for_insertion(customer, vehicle, pos)
                        if dc is None:
                            continue
                        insertion_costs.append((dc, vehicle, pos))
                if not insertion_costs:
                    continue
                insertion_costs.sort(key=lambda x: x[0])
                best_dc = insertion_costs[0][0]
                upto = min(k, len(insertion_costs))
                regret = 0.0
                for idx in range(1, upto):
                    regret += insertion_costs[idx][0] - best_dc
                if (regret > best_regret) or (regret == best_regret and best_dc < float("inf")):
                    best_regret = regret
                    best_customer = customer
                    best_vehicle = insertion_costs[0][1]
                    best_pos = insertion_costs[0][2]
            if best_customer is None:
                print(f"Warning: Could not insert {len(unassigned)} customers on day {day}")
                break
            rt = routes[best_vehicle]
            routes[best_vehicle] = rt[:best_pos] + [best_customer] + rt[best_pos:]
            unassigned.remove(best_customer)
        return len(unassigned) == 0

class JointOperators:
    def __init__(self, alns):
        self.alns = alns

    def remove_node_all_days(self, routes_by_day, f):
        for day in range(6):
            for v in list(routes_by_day[day].keys()):
                rt = routes_by_day[day][v]
                if f in rt:
                    routes_by_day[day][v] = [n for n in rt if (n == 0 or n != f)]
                    routes_by_day[day][v] = self.alns._normalize_route(routes_by_day[day][v])
        return routes_by_day

    def build_insertion_evaluator(self, routes_day, f):
        alns = self.alns
        ev = alns.evaluator
        delta_mat = ev.delta
        c_km = ev.c_km
        c_fuel = ev.c_fuel
        lam = ev.lambda_param
        theta_TR = ev.theta_TR
        eta = ev.eta
        W0 = ev.W0
        Q = ev.Q

        vehicle_data = []
        empty_vehicles = []
        
        for v, route in routes_day.items():
            route = alns._normalize_route(route)
            routes_day[v] = route
            customers = [n for n in route if n != 0]
            if not customers:
                empty_vehicles.append(v)
                continue
            
            loads_old = {}
            total_load = 0.0
            for n in customers:
                pat = alns.current_solution_pattern_lookup.get(n)
                ld = alns.p_frt.get((n, pat, alns.current_day_for_lookup), 0.0)
                loads_old[n] = ld
                total_load += ld
            
            n_rt = len(route)
            suffix_load = [0.0] * (n_rt + 1)
            for i in range(n_rt - 1, -1, -1):
                suffix_load[i] = suffix_load[i + 1]
                node = route[i]
                if node != 0:
                    suffix_load[i] += loads_old.get(node, 0.0)
            
            prefix_dist = [0.0] * (n_rt + 1)
            for i in range(1, n_rt):
                prefix_dist[i] = prefix_dist[i - 1] + delta_mat[route[i - 1], route[i]]
            
            vehicle_data.append((v, route, loads_old, total_load, suffix_load, prefix_dist))
        
        dist_depot_f = delta_mat[0, f]

        def _empty_route_cost(q):
            d2 = 2.0 * dist_depot_f
            fuel_out = eta * (W0 + q) * dist_depot_f
            fuel_back = eta * W0 * dist_depot_f
            transport = c_km * d2 + c_fuel * (fuel_out + fuel_back)
            poll = theta_TR * (fuel_out + fuel_back)
            return (1.0 - lam) * transport + lam * poll

        def _arc_cost(a, b, load_after_b):
            dist = delta_mat[a, b]
            fuel = eta * (W0 + load_after_b) * dist
            transport = c_km * dist + c_fuel * fuel
            poll = theta_TR * fuel
            return (1.0 - lam) * transport + lam * poll

        cache = {}

        def best(q):
            if q <= 0:
                return (0.0, None, None)
            if q > Q:
                return (float("inf"), None, None)
            if q in cache:
                return cache[q]
            
            best_delta = float("inf")
            best_v, best_pos = None, None
            
            pollution_corr_per_dist = lam * theta_TR * eta * q
            fuel_corr_per_dist = (1.0 - lam) * c_fuel * eta * q
            
            for (v, route, loads_old, total_load, suffix_load, prefix_dist) in vehicle_data:
                if total_load + q > Q + 1e-9:
                    continue
                
                n_rt = len(route)
                for pos in range(1, n_rt):
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    
                    old_arc = _arc_cost(prev_node, next_node, suffix_load[pos])
                    new_arc1 = _arc_cost(prev_node, f, suffix_load[pos] + q)
                    new_arc2 = _arc_cost(f, next_node, suffix_load[pos])
                    
                    correction = (pollution_corr_per_dist + fuel_corr_per_dist) * prefix_dist[pos]
                    
                    delta = new_arc1 + new_arc2 - old_arc + correction
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_v, best_pos = v, pos
            
            if empty_vehicles:
                empty_delta = _empty_route_cost(q)
                if empty_delta < best_delta:
                    best_delta = empty_delta
                    best_v = empty_vehicles[0]
                    best_pos = 1
            
            cache[q] = (best_delta, best_v, best_pos)
            return cache[q]
        
        return best

    def pattern_side_cost_from_days(self, f, deliver_days):
        alns = self.alns
        bits = [0] * 6
        for d in deliver_days:
            bits[d] = 1
        r = alns.bits_to_r.get(tuple(bits), None)
        if r is None:
            return float('inf')
        return alns.evaluator.calculate_pattern_cost(f, r)

    def compute_cover_to_next_quantities(self, deliver_days, f):
        """Look up simulation-based p_frt for given delivery days. Kept for API
        compatibility — the new joint_ds_visitdays_cover_to_next does not use
        this method anymore.
        """
        alns = self.alns
        bits = [0] * 6
        for d in deliver_days:
            bits[d] = 1
        r = alns.bits_to_r.get(tuple(bits), None)
        if r is None:
            return {d: 0.0 for d in deliver_days}
        return {d: alns.p_frt.get((f, r, d), 0.0) for d in deliver_days}

    def joint_ds_visitdays_cover_to_next(self, solution, f):
        """Enumerate feasible patterns for store f. For each r, evaluate:
        pattern_cost(f, r) + sum over delivery days of best insertion cost
        using the SIMULATION-based p_frt[f, r, t] (NOT the cycle formula).
        Pick the r with lowest total. This makes the DP evaluation perfectly
        consistent with the actual cost incurred when routes are committed.
        """
        alns = self.alns
        routes_by_day = solution.routes_by_day
        routes_by_day = self.remove_node_all_days(routes_by_day, f)
        alns.current_solution_pattern_lookup = solution.pattern_assignments.copy()

        # Build per-day insertion evaluators ONCE (reused across all candidate r)
        best_insert = {}
        for day in range(6):
            alns.current_day_for_lookup = day
            best_insert[day] = self.build_insertion_evaluator(routes_by_day[day], f)

        INF = float("inf")
        feasible_rs = list(alns.feasible_patterns_by_store.get(f, []))
        if not feasible_rs:
            return solution, set(), INF

        best_total = INF
        best_r = None
        best_inserts = None  # dict: day -> (qty, vehicle, position)

        for r in feasible_rs:
            bits = alns.patterns[r]
            deliver_days = [t for t in range(6) if bits[t] == 1]
            if len(deliver_days) < 2:
                continue

            # Use simulation p_frt directly — same as what will actually be loaded
            side_cost = alns.evaluator.calculate_pattern_cost(f, r)
            routing_cost = 0.0
            inserts_for_r = {}
            feasible = True

            for d in deliver_days:
                q = alns.p_frt.get((f, r, d), 0.0)
                if q <= 0:
                    # Pattern says deliver but quantity is zero — skip but record empty
                    inserts_for_r[d] = (0.0, None, None)
                    continue
                ins_cost, v, pos = best_insert[d](q)
                if ins_cost >= INF or v is None:
                    feasible = False
                    break
                routing_cost += ins_cost
                inserts_for_r[d] = (q, v, pos)

            if not feasible:
                continue

            total = side_cost + routing_cost
            if total < best_total:
                best_total = total
                best_r = r
                best_inserts = inserts_for_r

        if best_r is None:
            return solution, set(), INF

        # Commit the best pattern + insertions
        solution.pattern_assignments[f] = best_r
        touched_days = set()
        for d, (q, v, pos) in best_inserts.items():
            if v is None or q <= 0:
                continue
            rt = routes_by_day[d][v]
            routes_by_day[d][v] = rt[:pos] + [f] + rt[pos:]
            routes_by_day[d][v] = alns._normalize_route(routes_by_day[d][v])
            touched_days.add(d)

        solution._calculate_cost()
        return solution, touched_days, best_total

def _get_cycle_days(from_day, to_day):
    """Get list of days in cycle [from_day, ..., to_day-1] (mod 6)."""
    days = []
    d = from_day
    while d != to_day:
        days.append(d)
        d = (d + 1) % 6
    return days


class ComprehensiveALNS:
    """Main ALNS algorithm combining pattern and routing optimization"""
    
    def __init__(self, instance_data, algorithm_params):
        self.instance_data = instance_data
        self.params = algorithm_params
        self.F_per_visit  = float(self.params.get('F_per_visit', 150.0))
        self.r_unit       = float(self.params.get('r_unit', 4.0))
        self.holding_rate = float(self.params.get('holding_rate', 0.25))
        self.unit_value   = float(self.params.get('unit_value', 2000.0))
        self.h_week = self.holding_rate * self.unit_value / 52.0

        self.stores = instance_data['stores']
        self.loc = instance_data['loc']
        self.store_id_mapping = instance_data['store_id_mapping']
        self.daily_demands = instance_data['daily_demands']
        
        self.daily_demands_by_day = instance_data.get('daily_demands_by_day', None)
        
        self.patterns, self.R = self._generate_patterns()
        self.bits_to_r = {tuple(p): i for i, p in enumerate(self.patterns)}
        self.r_to_bits = {i: tuple(p) for i, p in enumerate(self.patterns)}

        try:
            print("ALNS idx(111111) =", self.bits_to_r[(1,1,1,1,1,1)])
        except Exception:
            pass
                    
        # MOSinLine integration: use the distance matrix from the instance file
        # if provided (identical arcs as RLRP), else euclidean from coordinates.
        self.delta = {}
        if instance_data.get("distances"):
            for ij_string, c in instance_data["distances"].items():
                i, j = map(int, str(ij_string).strip("() ").split(","))
                self.delta[i, j] = float(c)
        else:
            for i in [0] + self.stores:
                for j in [0] + self.stores:
                    if i in self.loc and j in self.loc:
                        self.delta[i, j] = euclidean(self.loc[i], self.loc[j])
        
        self._calculate_parameters()
        
        self.evaluator = CombinedEvaluator(
            self.loc, self.delta, self.c_fr, self.waste_fractions, self.stockout_fractions,
            self.fw_emission, self.D_f,
            self.patterns, self.c_km, self.c_fuel, self.theta_FW, self.theta_TR, self.eta, 
            self.W0, self.lambda_param, self.Q,
            self.gamma_f, self.Q_day_min, self.Q_day_max,
            c_purchase=self.c_purchase, c_stockout=self.c_stockout)

        # Per-store feasible patterns:
        #  (a) shelf-life filter with SL_eff(f) = min over the store's segments
        #      with finite shelf life (co-delivery: the shortest-lived segment
        #      governs the maximum delivery gap of the joint pattern)
        #  (b) physical feasibility (vehicle capacity, store capacity)
        self.feasible_patterns_by_store = {f: [] for f in self.stores}
        for f in self.stores:
            sl_eff = self.SL_eff.get(f, None)
            for r in self.R:
                if sl_eff is not None and self._max_gap(self.patterns[r]) > sl_eff:
                    continue
                is_physically_feasible = True
                for t in range(6):
                    drop = self.p_frt.get((f, r, t), 0.0)
                    if drop > self.Q or drop > self.gamma_f[f]:
                        is_physically_feasible = False
                        break
                if is_physically_feasible:
                    self.feasible_patterns_by_store[f].append(r)
        for f in self.stores:
            if not self.feasible_patterns_by_store[f]:
                viol_list = []
                for r in self.R:
                    worst = max(self.p_frt.get((f, r, t), 0.0) - min(self.Q, self.gamma_f[f]) for t in range(6))
                    viol_list.append((worst, r))
                viol_list.sort()
                self.feasible_patterns_by_store[f] = [r for _, r in viol_list[:3]]
        print("Per-store feasible patterns built (shelf-life filter: min over segments).")

        self.pattern_operators = ALNSPatternOperators(
            self.evaluator, self.stores, self.patterns, self.R, 
            self.delta, self.p_frt, self.Q, self.gamma_f, 
            feasible_patterns_by_store=self.feasible_patterns_by_store,
            alpha=self.params['alpha'])
        
        self.routing_operators = LNSRoutingOperators(
            self.evaluator, self.p_frt, alpha=self.params['alpha'])
        self.joint_ops = JointOperators(self)
 
        self.operator_names = [
            "JOINT DS VisitDays (cover-to-next)",
            "Proximity",
            "Sales Volume",
            "Move-one",
            "Move-two",
            "Smart Eco Pattern Optimization"
        ]
        n_ops = len(self.operator_names)
        self.operator_weights = [1.0] * n_ops
        self.operator_scores  = [0.0] * n_ops
        self.operator_uses    = [0]   * n_ops
        self.operator_stats = [
            {"name": self.operator_names[i],
             "best": 0, "better": 0, "accept": 0, "reject": 0, "calls": 0}
            for i in range(n_ops)
        ]
        self.pattern_history = []
        self.ucb_c = algorithm_params.get('ucb_c', 1.41)
        self.operator_total_score = [0.0] * n_ops 
        self.operator_total_calls = [0] * n_ops  
        for i in range(n_ops):
            self.operator_total_calls[i] = 1
            self.operator_total_score[i] = 1.0
        self.lns_stats = {
            "calls": 0, "improved": 0, "total_reduction": 0.0, "max_reduction": 0.0
        }
        self.solution_cache = {}
        self.partial_solutions_log = []
        self.max_stage1_retries = 100
    
    @staticmethod
    def _max_gap(bits):
        """Maximum delivery gap (days, mod 6) of a pattern bit vector."""
        dd = [t for t in range(6) if bits[t] == 1]
        if not dd:
            return 6
        return max(((dd[(i+1) % len(dd)] - dd[i]) % 6) or 6 for i in range(len(dd)))
        
    def _generate_patterns(self):
        """Generate the pattern universe:
        1. Exclude freq=0 and freq=1 (too few deliveries)
        2. Exclude freq=6 (daily delivery — eliminates waste artificially)
        3. For freq=2: exclude if min gap between deliveries < 2 days
        Shelf-life filtering is NO LONGER global: with product segments it is
        applied per store in feasible_patterns_by_store, using
        SL_eff(f) = min over the store's segments with finite shelf life.
        """
        kept = []
        for p in itertools.product([0, 1], repeat=6):
            freq = sum(p)
            if freq < 2 or freq > 5:
                continue
            if freq == 2:
                dd = [t for t in range(6) if p[t] == 1]
                if min((dd[1]-dd[0])%6, (dd[0]-dd[1])%6) < 2:
                    continue
            kept.append(p)
        patterns = sorted(kept, key=lambda p: int(''.join(map(str, p)), 2))
        R = list(range(len(patterns)))
        return patterns, R
    
    def _calculate_parameters(self):
        """Calculate all model parameters. Segment-aware:
        demand mu_{f,s,t}, shelf life SL_s and theta_FW_s live per segment;
        the delivery quantities, handling costs and routing use the aggregate
        p_frt[f,r,t] = sum_s p_fsrt[f,s,r,t].
        MOSinLine integration: scalar parameters are read from the instance
        file where present, with the standalone defaults as fallback."""
        inst = self.instance_data

        # ---- product segments ----
        seg_cfg = inst.get("segments", None) or self.params.get("segments", None) or DEFAULT_SEGMENTS
        self.segments = deepcopy(seg_cfg)
        self.segment_names = list(self.segments.keys())

        self.demand_cv = float(self.params.get('demand_cv', 0.25))
        seg_demand = inst.get("demand_by_segment", None)

        # ---- per-segment day-varying demand mu_{f,s,t} ----
        self.mu_fst = {}
        if seg_demand is not None:
            print(f"Using PER-SEGMENT day-varying demand (CV={self.demand_cv}), "
                  f"segments: {self.segment_names}.")
            for f in self.stores:
                sd = seg_demand.get(str(f), seg_demand.get(f, {}))
                self.mu_fst[f] = {}
                for s in self.segment_names:
                    vals = sd.get(s, [0.0] * 6)
                    self.mu_fst[f][s] = {t: float(vals[t]) for t in range(6)}
        elif self.daily_demands_by_day is not None:
            print(f"Using DAY-VARYING demand, single segment 'fresh' "
                  f"(backward compatible mode, CV={self.demand_cv}).")
            for i, store_id in enumerate(self.stores):
                str_id = str(store_id)
                if str_id in self.daily_demands_by_day:
                    mu_list = self.daily_demands_by_day[str_id]
                else:
                    d_const = self.daily_demands[i] / 2.0
                    mu_list = [d_const] * 6
                self.mu_fst[store_id] = {s: {t: 0.0 for t in range(6)} for s in self.segment_names}
                self.mu_fst[store_id]["fresh"] = {t: float(mu_list[t]) for t in range(6)}
        else:
            print(f"Using CONSTANT demand, single segment 'fresh' "
                  f"(backward compatible mode, CV={self.demand_cv}).")
            for i, store_id in enumerate(self.stores):
                d_const = self.daily_demands[i] / 2.0
                self.mu_fst[store_id] = {s: {t: 0.0 for t in range(6)} for s in self.segment_names}
                self.mu_fst[store_id]["fresh"] = {t: d_const for t in range(6)}

        self.sigma_fst = {f: {s: {t: self.demand_cv * self.mu_fst[f][s][t] for t in range(6)}
                              for s in self.segment_names}
                          for f in self.stores}

        # ---- aggregates over segments (reporting, capacities, handling) ----
        self.D_fs = {f: {s: sum(self.mu_fst[f][s][t] for t in range(6))
                         for s in self.segment_names}
                     for f in self.stores}
        self.mu_ft = {f: {t: sum(self.mu_fst[f][s][t] for s in self.segment_names)
                          for t in range(6)}
                      for f in self.stores}
        self.sigma_ft = {f: {t: math.sqrt(sum(self.sigma_fst[f][s][t] ** 2
                                              for s in self.segment_names))
                             for t in range(6)}
                         for f in self.stores}
        self.D_f = {f: sum(self.D_fs[f][s] for s in self.segment_names) for f in self.stores}
        self.d_f = {f: self.D_f[f] / 6.0 for f in self.stores}

        # ---- effective shelf life per store: min over carried segments ----
        self.SL_eff = {}
        for f in self.stores:
            finite = [self.segments[s]["shelf_life"] for s in self.segment_names
                      if self.segments[s]["shelf_life"] is not None
                      and self.D_fs[f].get(s, 0.0) > 1e-9]
            self.SL_eff[f] = min(finite) if finite else None
        self.shelf_life = int(self.params.get('shelf_life', 4))  # legacy scalar (unused by the simulation)
        
        # Unified simulation: computes BOTH p_frt and waste_fractions per segment
        _sim_start = time.time()
        self._calculate_delivery_quantities_and_waste()
        n_active_seg = len(self.segment_names)
        print(f"52-week (R,S) FIFO/LIFO simulation: {time.time() - _sim_start:.2f}s "
              f"({len(self.stores)} stores x {len(self.R)} patterns x {n_active_seg} segments, "
              f"p_FIFO={self.params.get('p_fifo', 0.70)}, N_RUNS=2)")
        
        self.gamma_f = {}
        for f in self.stores:
            base_cap = max(self.D_f[f] / 2, 11)          # PDF floor 11 t
            max_delivery = max(self.p_frt.get((f, r, t), 0.0)
                               for r in range(len(self.patterns)) for t in range(6))
            self.gamma_f[f] = max(base_cap, math.ceil(max_delivery))
        
        self.shelf_capacity = {f: int(self.gamma_f[f] * 0.7) for f in self.stores}
        self.Q_day_min = float(inst.get("Q_day_min", 0))
        self.Q_day_max = float(inst.get("Q_day_max", 99999))
        
        # ---- scalar parameters: instance file overrides defaults (MOSinLine) ----
        self.num_vehicles = int(inst.get("num_vehicles", 50))
        self.Q = float(inst.get("vehicle_capacity", 25.6))
        
        self.c_km = float(inst.get("cost_per_km", 1.12))
        self.c_fuel = float(inst.get("fuel_price", 1.80))
        self.W0 = float(inst.get("vehicle_empty_weight", 14.4))
        self.eta = float(inst.get("eta", 0.05))
        if "marginal_co2_emissions" in inst:
            # RLRP consistency: marginal_co2_emissions == eta * theta_TR
            self.theta_TR = float(inst["marginal_co2_emissions"]) / self.eta
        else:
            self.theta_TR = 2.7
        # legacy scalar fallback only; segment values self.segments[s]["theta_FW"] govern
        self.theta_FW = float(inst.get("theta_FW", 4000.0))
        self.lambda_param = float(inst.get("weighting_factor_patt", 0.3))
        
        self.c_purchase = float(inst.get("c_purchase", 800.0))
        self.c_stockout = float(inst.get("c_stockout", 100.0))

        self.c_order = 15.0
        self.c_receipt = 25.0
        self.wage_store = 19.9          # w^st
        self.wage_restock = 39.8        # w^re ~= 2*w^st  (backroom restock)
        self.wage_dc = 22.8             # w^dc
        self.unit_value = self.c_purchase   # u^prc for capital tie-up (~= c^pur)
        self.capital_rate = 0.10 / 52.0
        self.t_fill_base = 2.0          # h/t
        self.t_fill_decay = 0.72
        self.t_restock_base = 2.0       # h/t
        self.t_restock_g = 0.30         # CONVEX exponent (1+g)
        self.t_pick_base = 0.8          # h/t
        self.t_pick_decay = 0.68
        
        self._calculate_pattern_costs()
    
    def _calculate_delivery_quantities(self):
        """Stub: p_frt computed by _calculate_delivery_quantities_and_waste."""
        if not self.p_frt:
            self._calculate_delivery_quantities_and_waste()
        return self.p_frt
    
    def _calculate_delivery_quantities_and_waste(self):
        """(R,S) FIFO/LIFO simulation PER (store, segment, pattern).
        
        Per segment s: own mu_{f,s,t}, sigma_{f,s,t}, shelf life SL_s and
        order-up-to level S_{f,s,r}. Segments with shelf_life=None never
        expire within the horizon (expiry batch date set beyond the horizon),
        so their waste fraction is 0 by construction and their pattern choice
        is driven purely by handling/transport costs.
        
        Aggregation over segments (co-delivery, one pattern per store):
          p_frt[f,r,t]            = sum_s p_fsrt[f,s,r,t]     (truck load)
          waste_fractions[f,r]    = sum_s D_fs*w_fsr / D_f    (purchase cost, reporting)
          stockout_fractions[f,r] = sum_s D_fs*so_fsr / D_f
          fw_emission[f,r]        = sum_s theta_FW_s * D_fs * w_fsr   (kg CO2e)
        
        (R,S) policy: S=max(q_cycle), order on day before delivery (L=1).
        Per-day sequence: Expire -> Receive -> Order (placed in the morning,
        before today's demand is realized) -> Demand -> End-of-day update.
        Mixed FIFO/LIFO consumer behavior. N_RUNS=2, 52-week recording.
        """
        self.p_frt = {}
        self.p_fsrt = {}
        self.waste_fractions = {}
        self.stockout_fractions = {}
        self.fw_emission = {}
        self.waste_frac_seg = {}
        self.stockout_frac_seg = {}
        self.S_fsr = {}   # order-up-to level per (store, segment, pattern) — exported to the DES
        
        z_sl = 1.645
        N_RUNS = 10
        WARMUP_WEEKS = 2
        RECORD_WEEKS = 52
        SIM_SEED = 12345
        P_FIFO = float(self.params.get('p_fifo', 0.70))
        P_LIFO = 1.0 - P_FIFO
        NEVER = 10 ** 9  # expiry date for segments without shelf-life constraint
        
        rng = np.random.RandomState(SIM_SEED)
        
        for f in self.stores:
            for s in self.segment_names:
                SL = self.segments[s]["shelf_life"]
                sl_add = SL if SL is not None else NEVER
                D_week = self.D_fs[f][s]
                mu_arr = [self.mu_fst[f][s][t] for t in range(6)]
                sigma_arr = [self.sigma_fst[f][s][t] for t in range(6)]
                
                if D_week <= 0:
                    for r in self.R:
                        self.waste_frac_seg[f, s, r] = 0.0
                        self.stockout_frac_seg[f, s, r] = 0.0
                        self.S_fsr[f, s, r] = 0.0
                        for t in range(6):
                            self.p_fsrt[f, s, r, t] = 0.0
                    continue
                
                for r in self.R:
                    bits = self.patterns[r]
                    delivery_days = [t for t in range(6) if bits[t] == 1]
                    m = len(delivery_days)
                    
                    if m == 0:
                        self.waste_frac_seg[f, s, r] = 0.0
                        self.stockout_frac_seg[f, s, r] = 0.0
                        for t in range(6):
                            self.p_fsrt[f, s, r, t] = 0.0
                        continue
                    
                    # S = max(q) across delivery cycles (per segment)
                    q_by_cycle = {}
                    for idx, day in enumerate(delivery_days):
                        nxt = delivery_days[(idx + 1) % m]
                        cycle_days = _get_cycle_days(day, nxt)
                        mu_sum = sum(mu_arr[d_] for d_ in cycle_days)
                        sigma_sq_sum = sum(sigma_arr[d_] ** 2 for d_ in cycle_days)
                        q_by_cycle[day] = mu_sum + z_sl * np.sqrt(sigma_sq_sum)
                    S_level = max(q_by_cycle.values())
                    self.S_fsr[f, s, r] = S_level
                    
                    # Order days: place order on day t if tomorrow is a delivery day (L=1)
                    order_days_set = set()
                    for t in range(6):
                        if bits[(t + 1) % 6] == 1:
                            order_days_set.add(t)
                    delivery_days_set = set(delivery_days)
                    
                    delivery_qty_sums = {t: 0.0 for t in range(6)}
                    delivery_qty_counts = {t: 0 for t in range(6)}
                    
                    total_weeks = WARMUP_WEEKS + RECORD_WEEKS
                    total_days_sim = total_weeks * 6
                    record_start = WARMUP_WEEKS * 6
                    
                    waste_sum = 0.0
                    stockout_sum = 0.0
                    
                    for run in range(N_RUNS):
                        shelf = []
                        pending_order = None
                        run_waste = 0.0
                        run_stockout = 0.0
                        demand_carry = 0.0   # sub-unit demand is carried over, not discarded
                        
                        for abs_day in range(total_days_sim):
                            weekday = abs_day % 6
                            is_delivery = weekday in delivery_days_set
                            is_order = weekday in order_days_set
                            recording = abs_day >= record_start
                            
                            # Expire (never triggers for shelf_life=None segments)
                            new_shelf = []
                            for qty, exp in shelf:
                                if exp <= abs_day:
                                    if recording:
                                        run_waste += qty
                                else:
                                    new_shelf.append([qty, exp])
                            shelf = new_shelf
                            
                            # Receive
                            delivered_qty = 0.0
                            if is_delivery and pending_order is not None and pending_order[1] == abs_day:
                                delivered_qty = pending_order[0]
                                shelf.append([delivered_qty, abs_day + sl_add])
                                pending_order = None
                            elif is_delivery and abs_day == 0:
                                delivered_qty = float(int(round(S_level)))
                                shelf.append([delivered_qty, abs_day + sl_add])
                            
                            if recording and is_delivery and delivered_qty > 0:
                                delivery_qty_sums[weekday] += delivered_qty
                                delivery_qty_counts[weekday] += 1
                            
                            ip_pre = sum(q_ for q_, _ in shelf)
                            
                            # Order (placed in the morning, before today's demand)
                            if is_order:
                                order_qty = float(max(0, int(round(S_level - ip_pre + mu_arr[weekday]))))
                                pending_order = (order_qty, abs_day + 1)
                            
                            # Demand (carry-over discretization: fractional remainders
                            # accumulate until they form a whole unit -> exact conservation
                            # of expected volume even for sub-unit daily demands)
                            demand = max(0.0, rng.normal(mu_arr[weekday], sigma_arr[weekday]))
                            _pool = demand_carry + demand
                            demand = float(int(_pool))       # whole units released today
                            demand_carry = _pool - demand    # remainder carried to tomorrow
                            
                            # ---- Per-customer mixed FIFO/LIFO consumption (Path B) ----
                            # Discretize the day's demand into whole customers (the DES
                            # rounds demand to integer units), then serve them ONE AT A
                            # TIME. Each customer draws Bernoulli(P_FIFO) independently:
                            # a FIFO customer takes 1 unit from the OLDEST batch (front),
                            # a LIFO customer from the NEWEST (back). The shelf is updated
                            # after every customer.
                            n_customers = int(round(demand))
                            for _ in range(n_customers):
                                use_fifo = (rng.random() < P_FIFO)
                                need = 1.0  # one unit per customer
                                while need > 1e-9 and shelf:
                                    idx = 0 if use_fifo else len(shelf) - 1
                                    take = min(need, shelf[idx][0])
                                    shelf[idx][0] -= take
                                    need -= take
                                    if shelf[idx][0] <= 1e-9:
                                        shelf.pop(idx)
                                if need > 1e-9 and recording:
                                    run_stockout += need
                        
                        waste_per_week = run_waste / RECORD_WEEKS
                        stockout_per_week = run_stockout / RECORD_WEEKS
                        waste_sum += waste_per_week
                        stockout_sum += stockout_per_week
                    
                    avg_waste_per_week = waste_sum / N_RUNS
                    avg_stockout_per_week = stockout_sum / N_RUNS
                    self.waste_frac_seg[f, s, r] = (
                        max(0.0, avg_waste_per_week / D_week) if D_week > 0 else 0.0
                    )
                    self.stockout_frac_seg[f, s, r] = (
                        max(0.0, avg_stockout_per_week / D_week) if D_week > 0 else 0.0
                    )
                    
                    for t in range(6):
                        if bits[t] == 1 and delivery_qty_counts[t] > 0:
                            self.p_fsrt[f, s, r, t] = delivery_qty_sums[t] / delivery_qty_counts[t]
                        else:
                            self.p_fsrt[f, s, r, t] = 0.0
        
        # ---- aggregate over segments ----
        for f in self.stores:
            D_week_total = self.D_f[f]
            for r in self.R:
                for t in range(6):
                    self.p_frt[f, r, t] = sum(self.p_fsrt.get((f, s, r, t), 0.0)
                                              for s in self.segment_names)
                waste_qty = sum(self.D_fs[f][s] * self.waste_frac_seg.get((f, s, r), 0.0)
                                for s in self.segment_names)
                so_qty = sum(self.D_fs[f][s] * self.stockout_frac_seg.get((f, s, r), 0.0)
                             for s in self.segment_names)
                self.waste_fractions[f, r] = (waste_qty / D_week_total) if D_week_total > 0 else 0.0
                self.stockout_fractions[f, r] = (so_qty / D_week_total) if D_week_total > 0 else 0.0
                self.fw_emission[f, r] = sum(
                    self.segments[s]["theta_FW"] * self.D_fs[f][s]
                    * self.waste_frac_seg.get((f, s, r), 0.0)
                    for s in self.segment_names)
    
    def _calculate_pattern_costs(self):
        """C_fr per PDF eq(5) -- per-day Sternbeck on the SIMULATED p_frt[f,r,t]
        (aggregate over segments; handling is driven by total tonnage moved),
        identical to the Gurobi pattern cost:
          c^ord*m + c^rec*m
          + w^st * sum_t t^fill_b * min(p,k)^(1-t^fill_d)          (filling, concave)
          + w^re * sum_t t^re_b  * ((p-k)^+)^(1+t^re_g)            (restock overflow, convex)
          + w^dc * sum_t t^pick_b * p^(1-t^pick_d)                 (DC picking, concave)
          + rho * u^prc * 0.5 * mean(p)                            (capital tie-up)
        """
        self.c_fr = {}
        for f in self.stores:
            D_week = self.D_f[f]
            sc = self.shelf_capacity[f]
            if D_week <= 0:
                for r in self.R: self.c_fr[f, r] = 0.0
                continue
            for r in self.R:
                dd = [t for t in range(6) if self.patterns[r][t] == 1]
                m = len(dd)
                if m == 0:
                    self.c_fr[f, r] = 0.01
                    continue
                co = self.c_order * m; cr = self.c_receipt * m
                cf = 0.0; crs = 0.0; cp = 0.0; psum = 0.0
                for t in dd:
                    p = self.p_frt.get((f, r, t), 0.0)
                    psum += p
                    fill_q = min(p, sc)
                    over_q = max(0.0, p - sc)
                    if fill_q > 0: cf  += self.t_fill_base    * (fill_q ** (1 - self.t_fill_decay))
                    if over_q > 0: crs += self.t_restock_base * (over_q ** (1 + self.t_restock_g))
                    if p > 0:      cp  += self.t_pick_base    * (p      ** (1 - self.t_pick_decay))
                cf  *= self.wage_store
                crs *= self.wage_restock
                cp  *= self.wage_dc
                ci = self.capital_rate * self.unit_value * 0.5 * (psum / m)
                self.c_fr[f, r] = max(co + cr + cf + crs + ci + cp, 0.01)
    
    def _calculate_waste_fractions(self):
        """Stub: waste_fractions computed by _calculate_delivery_quantities_and_waste."""
        if not self.waste_fractions:
            self._calculate_delivery_quantities_and_waste()
    
    def _is_pattern_feasible_for_store(self, store, pattern_id):
        for day in range(6):
            delivery_amount = self.p_frt.get((store, pattern_id, day), 0)
            if delivery_amount > self.gamma_f[store]:
                return False
        return True
    
    def construct_initial_solution(self):
        print("Constructing initial solution...")
        pattern_assignments = {}
        for store in self.stores:
            feas = list(self.feasible_patterns_by_store.get(store, []))
            if feas:
                pattern_assignments[store] = random.choice(feas)
            else:
                cap = min(self.Q, self.gamma_f[store])
                def worst_violation(r):
                    return max(self.p_frt.get((store, r, t), 0.0) - cap for t in range(6))
                pattern_assignments[store] = min(self.R, key=worst_violation)
        routes_by_day = {}
        for day in range(6):
            routes_by_day[day] = {}
            stores_to_serve = []
            for store in self.stores:
                r = pattern_assignments[store]
                if self.patterns[r][day] == 1:
                    stores_to_serve.append(store)
            if not stores_to_serve:
                for v in range(self.num_vehicles):
                    routes_by_day[day][v] = [0, 0]
                continue
            savings = []
            for i in stores_to_serve:
                for j in stores_to_serve:
                    if i < j:
                        s_ij = (self.delta[0, i] + self.delta[0, j] - self.delta[i, j])
                        savings.append((s_ij, i, j))
            savings.sort(reverse=True)
            routes = []
            for store in stores_to_serve:
                r = pattern_assignments[store]
                load = self.p_frt.get((store, r, day), 0.0)
                routes.append({'route': [0, store, 0], 'load': load})
            for saving, i, j in savings:
                route_i = None
                route_j = None
                for r in routes:
                    if i in r['route'][1:-1]:
                        route_i = r
                    if j in r['route'][1:-1]:
                        route_j = r
                if route_i is None or route_j is None or route_i is route_j:
                    continue
                can_merge = False
                if route_i['route'][-2] == i and route_j['route'][1] == j:
                    can_merge = True
                    new_route = route_i['route'][:-1] + route_j['route'][1:]
                elif route_j['route'][-2] == j and route_i['route'][1] == i:
                    can_merge = True
                    new_route = route_j['route'][:-1] + route_i['route'][1:]
                if not can_merge:
                    continue
                new_load = route_i['load'] + route_j['load']
                if new_load <= self.Q:
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append({'route': new_route, 'load': new_load})
            for v in range(self.num_vehicles):
                if v < len(routes):
                    routes_by_day[day][v] = routes[v]['route']
                else:
                    routes_by_day[day][v] = [0, 0]
        solution = ComprehensiveSolution(
            pattern_assignments=pattern_assignments,
            routes_by_day=routes_by_day,
            evaluator=self.evaluator,
            p_frt=self.p_frt,
            stores=self.stores
        )
        solution.num_vehicles = self.num_vehicles
        print(f"Initial solution cost: {solution.cost:.2f}")
        return solution
    
    def _get_solution_signature(self, solution):
        return tuple(solution.pattern_assignments[s] for s in sorted(self.stores))

    def _check_stage1_feasibility(self, solution):
        for store in self.stores:
            pat_id = solution.pattern_assignments[store]
            for day in range(6):
                qty = self.p_frt.get((store, pat_id, day), 0)
                if qty > self.gamma_f[store]:
                    return False, f"Store {store} capacity exceeded"
        for day in range(6):
            total_day_load = sum(self.p_frt.get((s, solution.pattern_assignments[s], day), 0) for s in self.stores)
            if total_day_load < self.Q_day_min or total_day_load > self.Q_day_max:
                return False, f"DC load out of bounds on day {day}"
        return True, "Feasible"

    def run_alns(self, max_iterations=1000, time_limit=None, target_cost=None):
        print("\nRunning Comprehensive ALNS optimization...")
        alns_start_time = time.time()
        best_found_time = 0.0
        time_to_match_target = None
        current_solution = self.construct_initial_solution()
        best_solution = current_solution.copy()
        if not current_solution.verify_all_deliveries():
            print("Warning: Initial solution does not satisfy all delivery requirements!")
        initial_violations = current_solution.validate_constraints()
        if initial_violations:
            print(f"Initial solution has {len(initial_violations)} constraint violations")
        T_start = self._calculate_start_temperature(current_solution.cost, self.params["g"])
        T = T_start
        cooling_rate = self.params["d"]
        search_leg_size = 50
        iterations_without_improvement = 0
        reset_border = self.params["lambda"]
        theta_1 = self.params["theta_1"]
        theta_2 = self.params["theta_2"]
        theta_3 = self.params["theta_3"]
        tau = self.params["r"]
        print(f"Start temperature: {T_start:.2f}")

        for iteration in range(max_iterations):
            if time_limit is not None:
                elapsed = time.time() - alns_start_time
                if elapsed > time_limit:
                    print(f"\n\u23f0 Time limit of {time_limit/3600:.2f} hours reached at iteration {iteration}.")
                    break
            old_current_cost = current_solution.cost
            best_cost = best_solution.cost
            new_solution = None
            op_idx = -1
            stage1_feasible = False
            best_cand = None
            best_cand_over = None
            best_cand_meta = None
            def _dc_overload(sol_):
                tot = 0.0
                for _day in range(6):
                    _load = sum(self.p_frt.get((s_, sol_.pattern_assignments[s_], _day), 0) for s_ in self.stores)
                    if _load > self.Q_day_max:
                        tot += _load - self.Q_day_max
                return tot
            for retry in range(self.max_stage1_retries):
                temp_solution, temp_op_idx, touched_days, is_joint_op = self._stage1_alns(current_solution, iteration)
                temp_sig = self._get_solution_signature(temp_solution)
                self.partial_solutions_log.append(temp_sig)
                is_feasible, reason = self._check_stage1_feasibility(temp_solution)
                if is_feasible:
                    new_solution = temp_solution
                    op_idx = temp_op_idx
                    stage1_feasible = True
                    break
                _over = _dc_overload(temp_solution)
                if best_cand is None or _over < best_cand_over:
                    best_cand = temp_solution
                    best_cand_over = _over
                    best_cand_meta = (temp_op_idx, touched_days, is_joint_op)
            cand_feasible = stage1_feasible
            if not stage1_feasible:
                cur_over = _dc_overload(current_solution)
                if best_cand is not None:
                    new_solution = best_cand
                    op_idx, touched_days, is_joint_op = best_cand_meta
                    stage1_feasible = True
                    if iteration % 50 == 0:
                        print(f"Iteration {iteration}: repair mode, DC overload {cur_over:.2f} -> {best_cand_over:.2f}")
                else:
                    if iteration % 50 == 0:
                        print(f"Iteration {iteration}: no feasible candidate ({reason})")
                    continue
            solution_sig = self._get_solution_signature(new_solution)
            if solution_sig in self.solution_cache:
                cached_cost, cached_routes = self.solution_cache[solution_sig]
                new_solution.routes_by_day = deepcopy(cached_routes)
                new_solution.cost = cached_cost
                new_solution._calculate_cost()
            else:
                new_solution = self._stage2_lns(new_solution, touched_days=touched_days, skip_init_routing=is_joint_op)
                self.solution_cache[solution_sig] = (new_solution.cost, deepcopy(new_solution.routes_by_day))
            accept_status = "rejected"
            new_cost = new_solution.cost
            if new_cost < best_cost - 1e-6 and cand_feasible:
                accept_status = "new_best"
                best_solution = new_solution.copy()
                current_solution = new_solution
                iterations_without_improvement = 0
                best_found_time = time.time() - alns_start_time
                print(f"Iteration {iteration}: New best = {best_solution.cost:.2f} (at {best_found_time:.1f}s)")
                if target_cost is not None and time_to_match_target is None and best_solution.cost <= target_cost + 1e-6:
                    time_to_match_target = time.time() - alns_start_time
                    print(f"  \u2705 Matched/beat target {target_cost:.2f} at {time_to_match_target:.1f}s")
            elif new_cost < old_current_cost:
                accept_status = "new_incumbent"
                current_solution = new_solution
            else:
                delta = new_cost - old_current_cost
                prob = math.exp(-delta / T) if T > 1e-12 else 0.0
                if random.random() < prob:
                    accept_status = "accepted_sa"
                    current_solution = new_solution
                else:
                    accept_status = "rejected"
            stats = self.operator_stats[op_idx]
            stats["calls"] += 1
            if accept_status == "new_best": stats["best"] += 1
            elif accept_status == "new_incumbent": stats["better"] += 1
            elif accept_status == "accepted_sa": stats["accept"] += 1
            else: stats["reject"] += 1
            pattern_vector = [current_solution.pattern_assignments[s] for s in sorted(self.stores)]
            self.pattern_history.append(pattern_vector)
            self.operator_uses[op_idx] += 1
            if accept_status == "new_best": self.operator_scores[op_idx] += theta_1
            elif accept_status == "new_incumbent": self.operator_scores[op_idx] += theta_2
            elif accept_status == "accepted_sa": self.operator_scores[op_idx] += theta_3
            reward = 0
            if accept_status == "new_best": reward = theta_1
            elif accept_status == "new_incumbent": reward = theta_2
            elif accept_status == "accepted_sa": reward = theta_3
            normalized_reward = reward / theta_1 
            self.operator_total_score[op_idx] += normalized_reward
            self.operator_total_calls[op_idx] += 1
            T *= cooling_rate
            if accept_status != "new_best":
                iterations_without_improvement += 1
            if iterations_without_improvement >= reset_border:
                print(f"Reset at iteration {iteration}")
                current_solution = self._apply_random_destruction(current_solution)
                iterations_without_improvement = 0
            if (iteration + 1) % search_leg_size == 0:
                self._update_operator_weights(tau)

        print("\nRunning post-optimization...")
        for day in range(6):
            if time_limit is not None:
                elapsed = time.time() - alns_start_time
                if elapsed > time_limit:
                    print(f"\n\u23f0 Time limit reached during post-optimization (day {day}).")
                    break
            print(f"Post-optimizing day {day}...")
            best_solution = self._run_lns_for_day(best_solution, day, unsuccessful_limit=2000)

        print("\n" + "="*60)
        print("STAGE 2 (LNS) PERFORMANCE STATISTICS")
        print("="*60)
        total_calls = max(1, self.lns_stats["calls"])
        improved_count = self.lns_stats["improved"]
        success_rate = (improved_count / total_calls) * 100
        avg_reduction = 0
        if improved_count > 0:
            avg_reduction = self.lns_stats["total_reduction"] / improved_count
        print(f"Total LNS Executions:    {total_calls}")
        print(f"Effective Improvements:  {improved_count} ({success_rate:.1f}%)")
        print(f"Total Cost Reduced:      {self.lns_stats['total_reduction']:.2f}")
        print(f"Avg Reduction (Success): {avg_reduction:.2f}")
        print(f"Max Single Reduction:    {self.lns_stats['max_reduction']:.2f}")
        print("="*60 + "\n")
        print(f"\nBest solution found at {best_found_time:.1f}s (total runtime: {time.time() - alns_start_time:.1f}s)")
        self.time_to_match_target = time_to_match_target
        self.best_found_time = best_found_time
        return best_solution

    def build_operator_performance_table(self):
        rows = []
        for s in self.operator_stats:
            c = max(s["calls"], 1)
            rows.append({
                "Operator": s["name"],
                "% Best":   100.0 * s["best"]   / c,
                "% Better": 100.0 * s["better"] / c,
                "% Accept": 100.0 * s["accept"] / c,
                "% Reject": 100.0 * s["reject"] / c,
                "Calls":    c
            })
        df = pd.DataFrame(rows)
        return df

    def _stage1_alns(self, solution, iteration):
        operator_idx = self._select_operator()
        num_stores = len(self.stores)
        if num_stores <= 25: c_min, c_max = 5, 15
        elif num_stores <= 50: c_min, c_max = 5, 20
        elif num_stores <= 100: c_min, c_max = 5, 30
        else: c_min, c_max = 5, 50
        c_max = min(c_max, num_stores)
        c_min = min(c_min, c_max)
        c_stores = random.randint(c_min, c_max)
        new_solution = solution.copy()
        touched_days = set()
        is_joint_op = False 
        if operator_idx == 0:
            is_joint_op = True
            pick_n = min(c_stores, len(self.stores))
            stores_to_change = random.sample(self.stores, pick_n)
            touched_days = set()
            sol_work = new_solution
            for f in stores_to_change:
                cand = sol_work.copy()
                cand2, td, est = self.joint_ops.joint_ds_visitdays_cover_to_next(cand, f)
                if est == float("inf") or cand2 is None:
                    continue
                r_new = cand2.pattern_assignments.get(f, None)
                feas = self.feasible_patterns_by_store.get(f, [])
                if feas and (r_new not in feas):
                    continue
                sol_work = cand2
                touched_days |= set(td)
            sol_work._calculate_cost()
            return sol_work, operator_idx, touched_days, is_joint_op
        if operator_idx == 1:
            changes = self.pattern_operators.proximity_operator(new_solution, c_stores)
        elif operator_idx == 2:
            changes = self.pattern_operators.sales_volume_operator(new_solution, c_stores)
        elif operator_idx == 3:
            changes = self.pattern_operators.move_one_operator(new_solution, c_stores)
        elif operator_idx == 4:
            changes = self.pattern_operators.move_two_operator(new_solution, c_stores)
        else:
            changes = self.pattern_operators.smart_eco_pattern_operator(new_solution, c_stores)
        for f, r_new in changes:
            cur = new_solution.pattern_assignments.get(f, None)
            if cur is None or r_new == cur:
                continue
            feas = self.feasible_patterns_by_store.get(f, [])
            if feas and (r_new not in feas):
                continue
            new_solution.pattern_assignments[f] = r_new
        return new_solution, operator_idx, touched_days, False

    def _build_savings_routes_for_day(self, sol, day):
        stores_to_serve = []
        for f in self.stores:
            r = sol.pattern_assignments[f]
            if self.patterns[r][day] == 1:
                stores_to_serve.append(f)
        routes_day = {}
        if not stores_to_serve:
            for v in range(self.num_vehicles):
                routes_day[v] = [0, 0]
            return routes_day
        savings = []
        for i in stores_to_serve:
            for j in stores_to_serve:
                if i < j:
                    s_ij = self.delta[0, i] + self.delta[0, j] - self.delta[i, j]
                    savings.append((s_ij, i, j))
        savings.sort(reverse=True)
        routes = []
        for f in stores_to_serve:
            r = sol.pattern_assignments[f]
            load = self.p_frt.get((f, r, day), 0.0)
            routes.append({'route': [0, f, 0], 'load': load})
        for _, i, j in savings:
            route_i = None
            route_j = None
            for rr in routes:
                if i in rr['route'][1:-1]: route_i = rr
                if j in rr['route'][1:-1]: route_j = rr
            if route_i is None or route_j is None or route_i is route_j:
                continue
            can_merge = False
            if route_i['route'][-2] == i and route_j['route'][1] == j:
                can_merge = True
                new_route = route_i['route'][:-1] + route_j['route'][1:]
            elif route_j['route'][-2] == j and route_i['route'][1] == i:
                can_merge = True
                new_route = route_j['route'][:-1] + route_i['route'][1:]
            if not can_merge:
                continue
            new_load = route_i['load'] + route_j['load']
            if new_load <= self.Q:
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append({'route': new_route, 'load': new_load})
        for v in range(self.num_vehicles):
            if v < len(routes): routes_day[v] = routes[v]['route']
            else: routes_day[v] = [0, 0]
        return routes_day
    
    def _normalize_route(self, route):
        if route is None or len(route) == 0: return [0, 0]
        rt = list(route)
        if rt[0] != 0: rt = [0] + rt
        if rt[-1] != 0: rt = rt + [0]
        if len(rt) < 2: rt = [0, 0]
        if rt == [0, 0, 0]: rt = [0, 0]
        return rt

    def _day_routing_objective(self, sol, day):
        total = 0.0
        routes = sol.routes_by_day[day]
        for _, route in routes.items():
            route = self._normalize_route(route)
            total += self.routing_operators._route_cost_econ_plus_pollution_precise(sol, day, route)
        return total

    def _day_routes_feasible(self, sol, day):
        for _, route in sol.routes_by_day[day].items():
            route = self._normalize_route(route)
            loads = self.routing_operators._build_loads_for_route(sol, day, route)
            if not self.evaluator.is_route_feasible(route, loads):
                return False
        return True

    def _stage2_lns(self, solution, touched_days=None, skip_init_routing=False):
        sol = solution.copy()
        routing_iters = self.params.get("routing_iterations", 100)
        no_improve_limit = self.params.get("routing_no_improve_limit", 50)
        D = self.params.get("D", 0.003)
        k = self.params.get("k", 2)
        mu = self.params.get("mu", 0.6)
        xi = self.params.get("xi", 0.4)
        total_improvement_this_call = 0.0
        for day in range(6):
            if touched_days is None: touched_days = set()
            if skip_init_routing:
                for v in range(self.num_vehicles):
                    if v not in sol.routes_by_day[day]: sol.routes_by_day[day][v] = [0, 0]
            else:
                if len(touched_days) == 0:
                    sol.routes_by_day[day] = self._build_savings_routes_for_day(sol, day)
                else:
                    if day in touched_days:
                        sol.routes_by_day[day] = self._build_savings_routes_for_day(sol, day)
                    for v in range(self.num_vehicles):
                        if v not in sol.routes_by_day[day]: sol.routes_by_day[day][v] = [0, 0]
            baseline_cost = self._day_routing_objective(sol, day)
            current_routes = {v: list(r) for v, r in sol.routes_by_day[day].items()}
            best_routes = {v: list(r) for v, r in sol.routes_by_day[day].items()}
            best_cost = baseline_cost
            current_cost = best_cost 
            it_wo_improve = 0
            if len(sol.get_all_served_stores(day)) <= 1: continue
            for _ in range(routing_iters):
                if it_wo_improve >= no_improve_limit: break
                cand = sol.copy()
                cand.routes_by_day[day] = {v: list(r) for v, r in current_routes.items()}
                num_customers = len(cand.get_all_served_stores(day))
                if num_customers <= 1: break
                num_to_remove = max(1, int(0.3 * num_customers))
                removed = self.routing_operators.shaw_removal(cand, day, num_to_remove, mu=mu, xi=xi)
                if not removed:
                    it_wo_improve += 1; continue
                rem_set = set(removed)
                for v, route in cand.routes_by_day[day].items():
                    rr = [n for n in route if (n == 0 or n not in rem_set)]
                    cand.routes_by_day[day][v] = self._normalize_route(rr)
                ok = self.routing_operators.regret_k_insertion(cand, day, removed, k=k)
                if not ok or not self._day_routes_feasible(cand, day):
                    it_wo_improve += 1; continue
                cand_cost = self._day_routing_objective(cand, day)
                if cand_cost <= best_cost * (1.0 + D):
                    current_routes = {v: list(r) for v, r in cand.routes_by_day[day].items()}
                    current_cost = cand_cost
                    if cand_cost < best_cost - 1e-4:
                        best_cost = cand_cost
                        best_routes = {v: list(r) for v, r in cand.routes_by_day[day].items()}
                        it_wo_improve = 0
                    else: it_wo_improve += 1
                else: it_wo_improve += 1
            sol.routes_by_day[day] = {v: list(r) for v, r in best_routes.items()}
            sol = self._intra_route_2opt_day(sol, day)
            day_improvement = max(0.0, baseline_cost - best_cost)
            total_improvement_this_call += day_improvement
        self.lns_stats["calls"] += 1
        if total_improvement_this_call > 1e-4:
            self.lns_stats["improved"] += 1
            self.lns_stats["total_reduction"] += total_improvement_this_call
            if total_improvement_this_call > self.lns_stats["max_reduction"]:
                self.lns_stats["max_reduction"] = total_improvement_this_call
        sol._calculate_cost()
        return sol

    def _update_routing_for_patterns(self, solution):
        for day in range(6):
            currently_served = solution.get_all_served_stores(day)
            required = solution.get_required_deliveries(day)
            to_remove = currently_served - required
            if to_remove:
                for vehicle, route in solution.routes_by_day[day].items():
                    solution.routes_by_day[day][vehicle] = [
                        node for node in route if node not in to_remove]
            to_add = required - currently_served
            if to_add:
                self.routing_operators.regret_k_insertion(
                    solution, day, list(to_add), k=self.params['k'])
            self._enforce_capacity_for_day(solution, day)

    def _run_lns_for_day(self, solution, day, unsuccessful_limit=2000):
        sol = solution.copy()
        D  = self.params.get("D", 0.003)
        k  = self.params.get("k", 2)
        mu = self.params.get("mu", 0.6)
        xi = self.params.get("xi", 0.4)
        def day_cost(s):
            total = 0.0
            for _, rt in s.routes_by_day[day].items():
                rt = self._normalize_route(rt)
                total += self.routing_operators._route_cost_econ_plus_pollution_precise(s, day, rt)
            return total
        def day_feasible(s):
            for _, rt in s.routes_by_day[day].items():
                rt = self._normalize_route(rt)
                loads = self.routing_operators._build_loads_for_route(s, day, rt)
                if not self.evaluator.is_route_feasible(rt, loads):
                    return False
            return True
        current = sol.copy()
        best = sol.copy()
        best_c = day_cost(best)
        curr_c = best_c
        unsuccessful = 0
        while unsuccessful < unsuccessful_limit:
            num_customers = len(current.get_all_served_stores(day))
            if num_customers <= 1: break
            num_to_remove = max(1, int(0.3 * num_customers))
            removed = self.routing_operators.shaw_removal(current, day, num_to_remove, mu=mu, xi=xi)
            if not removed:
                unsuccessful += 1; continue
            cand = current.copy()
            rem_set = set(removed)
            for v, rt in cand.routes_by_day[day].items():
                rr = [n for n in rt if (n == 0 or n not in rem_set)]
                cand.routes_by_day[day][v] = self._normalize_route(rr)
            ok = self.routing_operators.regret_k_insertion(cand, day, removed, k=k)
            if not ok:
                unsuccessful += 1; continue
            if not day_feasible(cand):
                unsuccessful += 1; continue
            cand_c = day_cost(cand)
            if cand_c <= best_c * (1.0 + D):
                current = cand
                curr_c = cand_c
                if cand_c < best_c:
                    best = cand.copy()
                    best_c = cand_c
                    unsuccessful = 0
                else: unsuccessful += 1
            else: unsuccessful += 1
        sol.routes_by_day[day] = {v: list(r) for v, r in best.routes_by_day[day].items()}
        sol = self._intra_route_2opt_day(sol, day)
        sol._calculate_cost()
        return sol
    
    def _intra_route_2opt_day(self, solution, day):
        for v, route in solution.routes_by_day[day].items():
            route = self._normalize_route(route)
            customers = [n for n in route if n != 0]
            if len(customers) <= 1:
                continue
            if len(customers) <= 7:
                best_route = route
                best_cost = self.routing_operators._route_cost_econ_plus_pollution_precise(solution, day, route)
                for perm in itertools.permutations(customers):
                    cand_route = [0] + list(perm) + [0]
                    loads = self.routing_operators._build_loads_for_route(solution, day, cand_route)
                    if not self.evaluator.is_route_feasible(cand_route, loads):
                        continue
                    cand_cost = self.routing_operators._route_cost_econ_plus_pollution_precise(solution, day, cand_route)
                    if cand_cost < best_cost - 1e-6:
                        best_cost = cand_cost
                        best_route = cand_route
                solution.routes_by_day[day][v] = best_route
            else:
                improved = True
                while improved:
                    improved = False
                    route = solution.routes_by_day[day][v]
                    n = len(route)
                    best_cost = self.routing_operators._route_cost_econ_plus_pollution_precise(solution, day, route)
                    for i in range(1, n - 2):
                        for j in range(i + 1, n - 1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            loads = self.routing_operators._build_loads_for_route(solution, day, new_route)
                            if not self.evaluator.is_route_feasible(new_route, loads):
                                continue
                            new_cost = self.routing_operators._route_cost_econ_plus_pollution_precise(solution, day, new_route)
                            if new_cost < best_cost - 1e-6:
                                solution.routes_by_day[day][v] = new_route
                                best_cost = new_cost
                                improved = True
                                break
                        if improved:
                            break
        return solution
    
    def _calculate_start_temperature(self, initial_cost, g):
        return -g * initial_cost / np.log(0.5)
    
    def _select_operator(self):
        best_op = -1
        max_ucb = -float('inf')
        total_calls = sum(self.operator_total_calls)
        for i in range(len(self.operator_names)):
            avg_reward = self.operator_total_score[i] / (self.operator_total_calls[i] + 1e-6)
            exploration = self.ucb_c * math.sqrt(2 * math.log(total_calls + 1) / (self.operator_total_calls[i] + 1e-6))
            ucb_value = avg_reward + exploration
            if ucb_value > max_ucb:
                max_ucb = ucb_value
                best_op = i
        return best_op
    
    def _update_operator_weights(self, tau):
        for i in range(len(self.operator_names)):
            self.operator_scores[i] = 0
            self.operator_uses[i] = 0
    
    def _apply_random_destruction(self, solution):
        new_solution = solution.copy()
        num_to_change = len(self.stores) // 2
        stores_to_change = random.sample(self.stores, num_to_change)
        for store in stores_to_change:
            feas = self.feasible_patterns_by_store.get(store, [])
            if feas:
                new_solution.pattern_assignments[store] = random.choice(feas)
            else:
                cap = min(self.Q, self.gamma_f[store])
                def worst_violation(r):
                    return max(self.p_frt.get((store, r, t), 0.0) - cap for t in range(6))
                new_solution.pattern_assignments[store] = min(self.R, key=worst_violation)
        return new_solution
    
    def _enforce_capacity_for_day(self, solution, day):
        routes = solution.routes_by_day[day]
        Q = self.Q
        idle_vehicles = [v for v, r in routes.items() if len(r) == 2 and r[0] == 0 and r[1] == 0]
        def route_total_load(route):
            total = 0.0
            for node in route:
                if node != 0:
                    pat = solution.pattern_assignments[node]
                    total += self.p_frt.get((node, pat, day), 0.0)
            return total
        vehicles = list(routes.keys())
        for v in vehicles:
            route = routes[v]
            if len(route) <= 2: continue
            total_load = route_total_load(route)
            if total_load <= Q: continue
            customers = [n for n in route if n != 0]
            new_chunks = []
            current_chunk = []
            current_load = 0.0
            for cust in customers:
                pat = solution.pattern_assignments[cust]
                need = self.p_frt.get((cust, pat, day), 0.0)
                if need > Q: need = min(need, Q)
                if current_load + need <= Q or not current_chunk:
                    current_chunk.append(cust)
                    current_load += need
                else:
                    new_chunks.append(current_chunk)
                    current_chunk = [cust]
                    current_load = need
            if current_chunk:
                new_chunks.append(current_chunk)
            if new_chunks:
                routes[v] = [0] + new_chunks[0] + [0]
            for chunk in new_chunks[1:]:
                if idle_vehicles:
                    v_new = idle_vehicles.pop(0)
                else:
                    candidates = [vv for vv, rr in routes.items() if len(rr) == 2 and rr[0] == 0 and rr[1] == 0]
                    if candidates: v_new = candidates[0]
                    else: v_new = min(routes.keys(), key=lambda vv: route_total_load(routes[vv]))
                routes[v_new] = [0] + chunk + [0]

    def _best_insertion_delta_for(self, sol, f, day, r_new=None):
        if r_new is None:
            r_new = sol.pattern_assignments[f]
        need = self.p_frt.get((f, r_new, day), 0.0)
        if need <= 0:
            return 0.0
        best = float('inf')
        routes = sol.routes_by_day[day]
        for v, route in routes.items():
            for pos in range(1, len(route)):
                new_route = route[:pos] + [f] + route[pos:]
                loads_new = {n: self.p_frt.get((n, sol.pattern_assignments.get(n, None), day), 0.0)
                            for n in new_route if n != 0}
                loads_new[f] = need
                if not self.evaluator.is_route_feasible(new_route, loads_new):
                    continue
                loads_old = {n: self.p_frt.get((n, sol.pattern_assignments[n], day), 0.0)
                            for n in route if n != 0}
                delta = (self.evaluator.calculate_route_cost(new_route, loads_new)
                        - self.evaluator.calculate_route_cost(route, loads_old))
                if delta < best:
                    best = delta
        return best

    def _deletion_gain_for(self, sol, f, day):
        routes = sol.routes_by_day[day]
        for v, route in routes.items():
            if f in route:
                idx = route.index(f)
                new_route = route[:idx] + route[idx+1:]
                loads_old = {n: self.p_frt.get((n, sol.pattern_assignments[n], day), 0.0)
                            for n in route if n != 0}
                loads_new = {n: self.p_frt.get((n, sol.pattern_assignments[n], day), 0.0)
                            for n in new_route if n != 0}
                return (self.evaluator.calculate_route_cost(new_route, loads_new)
                        - self.evaluator.calculate_route_cost(route, loads_old))
        return 0.0


# Helper functions
def euclidean(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def load_instance(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    depot_xy = (data["depot"]["x"], data["depot"]["y"])
    stores = data["stores"]
    store_id_mapping = {}
    if "id_map" in data:
        for store_idx, store_id in enumerate(stores):
            str_store_id = str(store_id)
            if str_store_id in data["id_map"]:
                store_id_mapping[store_id] = data["id_map"][str_store_id]
            else:
                store_id_mapping[store_id] = store_id
    else:
        store_id_mapping = {store_id: store_id for store_id in stores}
    loc = {0: depot_xy}
    for store_id in stores:
        str_store_id = str(store_id)
        if str_store_id in data["loc"]:
            loc[store_id] = (data["loc"][str_store_id]["x"], data["loc"][str_store_id]["y"])
        else:
            loc[store_id] = (0, 0)
    daily_demands = data.get("daily_demands", [0.0] * len(stores))
    daily_demands_by_day = data.get("daily_demands_by_day", None)
    
    ret = {
        'stores': stores,
        'store_id_mapping': store_id_mapping,
        'loc': loc,
        'daily_demands': daily_demands,
        'daily_demands_by_day': daily_demands_by_day,
        'instance_name': data["instance_name"]
    }
    # MOSinLine integration: pass through all extra coefficients from the
    # instance file (distances, vehicle_capacity, cost_per_km,
    # marginal_co2_emissions, weighting_factor_patt, vehicle_empty_weight,
    # segments, demand_by_segment, ...)
    for k, v in data.items():
        if k not in ret:
            ret[k] = v
    return ret


def print_cost_and_metrics(title, lam,
                           rpe, rfp, rso,          # pattern side (EUR)
                           rtd, rtf,               # transport side (EUR): dist / fuel
                           rfe, rtp,               # environmental (kg CO2)
                           total_runtime, best_time,
                           fw_rate, so_rate,
                           total_distance, total_load, trucks_used):
    """Structured cost report:
       [1] Economic costs  -> pattern-side subtotal + transport-side subtotal
       [2] Environmental   -> FW emission + transport pollution
       [3] Weighted objective
       [4] Operational metrics
    """
    pattern_side = rpe + rfp + rso
    transport_side = rtd + rtf
    econ = pattern_side + transport_side          # EUR, unweighted
    env = rfe + rtp                               # kg CO2, unweighted
    w_econ = (1 - lam) * econ
    w_env = lam * env
    obj = w_econ + w_env

    bar = "=" * 60
    sub = "-" * 60
    print(f"\n{bar}")
    print(f"{title}  (lambda = {lam})")
    print(bar)

    # ---------- [1] ECONOMIC COSTS ----------
    print("[1] ECONOMIC COSTS (EUR, unweighted)")
    print("    Pattern / inventory side:")
    print(f"      Pattern operational cost   C_pat  : {rpe:12.2f}")
    print(f"      Food waste purchasing cost C_pur  : {rfp:12.2f}")
    print(f"      Stockout penalty cost      C_so   : {rso:12.2f}")
    print(f"      {'Subtotal (pattern side)':<33}: {pattern_side:12.2f}")
    print("    Transportation side:")
    print(f"      Distance-based cost   c_km*d      : {rtd:12.2f}")
    print(f"      Energy cost (dist&load) c_fuel*F  : {rtf:12.2f}")
    print(f"      {'Subtotal (transport side)':<33}: {transport_side:12.2f}")
    print(sub)
    print(f"    TOTAL ECONOMIC COST                 : {econ:12.2f}  EUR")

    # ---------- [2] ENVIRONMENTAL IMPACT ----------
    print(f"\n[2] ENVIRONMENTAL IMPACT (kg CO2, unweighted)")
    print(f"      Food waste emission        E_FW   : {rfe:12.2f}")
    print(f"      Transport pollution        E_tr   : {rtp:12.2f}")
    print(sub)
    print(f"    TOTAL ENVIRONMENTAL IMPACT          : {env:12.2f}  kg CO2")

    # ---------- [3] WEIGHTED OBJECTIVE ----------
    print(f"\n[3] WEIGHTED OBJECTIVE")
    print(f"      (1-lambda) * economic  = {1-lam:.2f} * {econ:11.2f} = {w_econ:12.2f}")
    print(f"       lambda    * environ.  = {lam:.2f} * {env:11.2f} = {w_env:12.2f}")
    print(sub)
    print(f"    TOTAL OBJECTIVE (scalarised)        : {obj:12.2f}")

    # ---------- [4] OPERATIONAL METRICS ----------
    print(f"\n[4] OPERATIONAL METRICS")
    print(f"      Total travelled distance          : {total_distance:12.2f}  km")
    print(f"      Total load delivered              : {total_load:12.2f}  t")
    print(f"      Total trucks used                 : {trucks_used:12d}")
    print(f"      Food waste rate                   : {fw_rate*100:12.2f}  %")
    print(f"      Stockout rate                     : {so_rate*100:12.2f}  %")
    print(f"      Total runtime                     : {total_runtime:12.2f}  s")
    print(f"      Best solution found at            : {best_time:12.2f}  s")
    print(bar)
    return obj

def save_comprehensive_results(solution, instance_data, runtime):
    instance_name = instance_data['instance_name']
    store_id_mapping = instance_data['store_id_mapping']
    ev = solution.evaluator
    
    total_distance = 0
    total_load = 0
    vehicles_used = set()
    
    raw_pattern_econ_cost = 0
    raw_fw_purchase_cost = 0
    raw_fw_emission_cost = 0
    raw_stockout_cost = 0
    
    for store, pattern_id in solution.pattern_assignments.items():
        econ = ev.c_fr[store, pattern_id]
        wf = ev.waste_fractions[store, pattern_id]
        sf = ev.stockout_fractions[store, pattern_id]
        
        fw_purchase = ev.c_purchase * ev.D_f[store] * wf
        fw_emission = ev.calculate_fw_emission(store, pattern_id)  # segment-weighted
        stockout = ev.c_stockout * ev.D_f[store] * sf
        
        raw_pattern_econ_cost += econ
        raw_fw_purchase_cost += fw_purchase
        raw_fw_emission_cost += fw_emission
        raw_stockout_cost += stockout
    
    raw_transport_dist_cost = 0.0   # c_km * distance  (distance-based)
    raw_transport_fuel_cost = 0.0   # c_fuel * fuel    (distance-and-load-based)
    raw_pollution_cost = 0.0
    
    for day in range(6):
        for vehicle, route in solution.routes_by_day[day].items():
            if len(route) > 2:
                vehicles_used.add(vehicle)
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    distance = ev.delta[from_node, to_node]
                    total_distance += distance
                    current_load = 0
                    for j in range(i+1, len(route)):
                        if route[j] != 0:
                            pattern_id = solution.pattern_assignments[route[j]]
                            current_load += solution.p_frt.get((route[j], pattern_id, day), 0)
                    fuel = ev.eta * (ev.W0 + current_load) * distance
                    raw_transport_dist_cost += ev.c_km * distance
                    raw_transport_fuel_cost += ev.c_fuel * fuel
                    emissions = ev.theta_TR * fuel
                    raw_pollution_cost += emissions

    raw_transport_econ_cost = raw_transport_dist_cost + raw_transport_fuel_cost

    # total_load = total delivered (consistent with Gurobi sum of da)
    total_load = sum(solution.p_frt.get((s, solution.pattern_assignments[s], t), 0.0)
                     for s in solution.stores for t in range(6))

    lambda_val = ev.lambda_param
    
    final_obj = ((1 - lambda_val) * (raw_pattern_econ_cost + raw_fw_purchase_cost + raw_stockout_cost + raw_transport_econ_cost) + 
                 lambda_val * (raw_fw_emission_cost + raw_pollution_cost))
    
    final_violations = solution.validate_constraints()
    
    results_data = {
        "instance": instance_name,
        "solver": "comprehensive_alns_segments",
        "lambda": lambda_val,
        "objective_value": final_obj,
        "economic_cost": raw_pattern_econ_cost + raw_fw_purchase_cost + raw_stockout_cost + raw_transport_econ_cost,
        "environmental_cost": raw_fw_emission_cost + raw_pollution_cost,
        "pattern_cost": raw_pattern_econ_cost,
        "fw_purchase_cost": raw_fw_purchase_cost,
        "stockout_cost": raw_stockout_cost,
        "transport_cost": raw_transport_econ_cost,
        "transport_dist_cost": raw_transport_dist_cost,
        "transport_fuel_cost": raw_transport_fuel_cost,
        "food_waste_emission_cost": raw_fw_emission_cost,
        "transport_pollution_cost": raw_pollution_cost,
        "runtime": runtime,
        "total_distance": total_distance,
        "total_load": total_load,
        "vehicles_used": len(vehicles_used),
        "num_stores": len(instance_data['stores']),
        "num_vehicles": getattr(solution, "num_vehicles", None),
        "num_feasible_patterns": len(ev.patterns),
        "model_type": "comprehensive_alns_segments",
        "constraint_violations": len(final_violations)
    }
    
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame([results_data])
    result_file = os.path.join("results", f"comprehensive_alns_{instance_name}.csv")
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', index=False, header=False)
    else:
        df.to_csv(result_file, mode='w', index=False, header=True)
    print(f"\n\u2705 Results saved to {result_file}")
    
    solution_details = {
        "instance_name": instance_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solver": "comprehensive_alns_segments",
        "objective_value": final_obj,
        "lambda": lambda_val,
        "runtime": runtime,
        "pattern_assignments": {},
        "daily_routes": {},
        "cost_breakdown": {
            "pattern_operational_cost": raw_pattern_econ_cost,
            "fw_purchase_cost": raw_fw_purchase_cost,
            "stockout_cost": raw_stockout_cost,
            "transport_cost": raw_transport_econ_cost,
            "transport_dist_cost": raw_transport_dist_cost,
            "transport_fuel_cost": raw_transport_fuel_cost,
            "fw_emission_cost": raw_fw_emission_cost,
            "transport_pollution_cost": raw_pollution_cost,
            "total_economic": raw_pattern_econ_cost + raw_fw_purchase_cost + raw_stockout_cost + raw_transport_econ_cost,
            "total_environmental": raw_fw_emission_cost + raw_pollution_cost,
            "weighted_objective": final_obj
        },
        "constraint_violations": final_violations
    }
    
    for store, pattern_id in solution.pattern_assignments.items():
        actual_store_id = store_id_mapping.get(store, store)
        solution_details["pattern_assignments"][str(actual_store_id)] = {
            "pattern_id": pattern_id,
            "pattern_bits": list(solution.evaluator.patterns[pattern_id]),
            "frequency": sum(solution.evaluator.patterns[pattern_id])
        }
    for day in range(6):
        solution_details["daily_routes"][str(day)] = {}
        for vehicle, route in solution.routes_by_day[day].items():
            if len(route) > 2:
                readable_route = []
                for node in route:
                    if node == 0:
                        readable_route.append(0)
                    else:
                        readable_route.append(store_id_mapping.get(node, node))
                solution_details["daily_routes"][str(day)][str(vehicle)] = readable_route
    solution_file = os.path.join("results", f"comprehensive_alns_solution_{instance_name}.json")
    with open(solution_file, 'w') as f:
        json.dump(solution_details, f, indent=2)
    print(f"\u2705 Detailed solution saved to {solution_file}")
    
    return results_data

def compute_food_waste_rate(solution):
    total_demand = 0.0
    total_waste_qty = 0.0
    for f, r in solution.pattern_assignments.items():
        D_week = solution.evaluator.D_f[f]
        wf = solution.evaluator.waste_fractions[f, r]
        total_demand += D_week
        total_waste_qty += D_week * wf
    if total_demand > 0:
        fw_rate = total_waste_qty / total_demand
    else:
        fw_rate = 0.0
    return fw_rate, total_waste_qty, total_demand

def compute_stockout_rate(solution):
    total_demand = 0.0
    total_stockout_qty = 0.0
    for f, r in solution.pattern_assignments.items():
        D_week = solution.evaluator.D_f[f]
        sf = solution.evaluator.stockout_fractions[f, r]
        total_demand += D_week
        total_stockout_qty += D_week * sf
    if total_demand > 0:
        so_rate = total_stockout_qty / total_demand
    else:
        so_rate = 0.0
    return so_rate, total_stockout_qty, total_demand

def print_segment_summary(solution, alns):
    """Per-segment demand, waste and emission summary for the chosen patterns."""
    print("\n--- PER-SEGMENT SUMMARY (chosen patterns) ---")
    print(f"{'Segment':>10} {'SL':>6} {'theta_FW':>10} {'Demand':>12} {'Waste':>10} {'Waste %':>9} {'FW CO2e':>12}")
    print("-" * 75)
    for s in alns.segment_names:
        SL = alns.segments[s]["shelf_life"]
        th = alns.segments[s]["theta_FW"]
        dem = sum(alns.D_fs[f][s] for f in alns.stores)
        waste = sum(alns.D_fs[f][s] * alns.waste_frac_seg.get((f, s, solution.pattern_assignments[f]), 0.0)
                    for f in alns.stores)
        co2 = th * waste
        rate = (waste / dem * 100) if dem > 0 else 0.0
        sl_str = str(SL) if SL is not None else "inf"
        print(f"{s:>10} {sl_str:>6} {th:>10.0f} {dem:>12.2f} {waste:>10.2f} {rate:>8.2f}% {co2:>12.2f}")

def print_comprehensive_solution(solution, instance_data, alns):
    store_id_mapping = instance_data['store_id_mapping']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    ev = solution.evaluator
    
    print("\n" + "="*100)
    print("COMPREHENSIVE SOLUTION DETAILS")
    print("="*100)
    
    print("\n--- DELIVERY PATTERN ASSIGNMENTS ---")
    print(f"{'Store ID':>10} {'Pattern':>8} {'Frequency':>10} {'Pattern Bits':>15} {'Delivery Days':>30}")
    print("-" * 85)
    store_pattern_info = []
    for store in solution.pattern_assignments:
        actual_id = store_id_mapping.get(store, store)
        pattern_id = solution.pattern_assignments[store]
        pattern_bits = ev.patterns[pattern_id]
        frequency = sum(pattern_bits)
        delivery_days = [day_names[i] for i in range(6) if pattern_bits[i] == 1]
        store_pattern_info.append((actual_id, store, pattern_id, pattern_bits, frequency, delivery_days))
    store_pattern_info.sort(key=lambda x: str(x[0]))
    for actual_id, gurobi_id, pattern_id, pattern_bits, frequency, delivery_days in store_pattern_info:
        pattern_str = ''.join(str(b) for b in pattern_bits)
        days_str = ', '.join(delivery_days)
        print(f"{str(actual_id):>10} {pattern_id:>8} {frequency:>10} {pattern_str:>15} {days_str:>30}")
    
    print("\n--- PATTERN FREQUENCY SUMMARY ---")
    freq_count = {}
    for _, _, _, _, frequency, _ in store_pattern_info:
        freq_count[frequency] = freq_count.get(frequency, 0) + 1
    for freq in sorted(freq_count.keys()):
        print(f"{freq} deliveries/week: {freq_count[freq]} stores ({freq_count[freq]/len(store_pattern_info)*100:.1f}%)")
    
    print("\n--- DAILY DEMAND PROFILE (mu_ft, aggregate over segments) ---")
    print(f"{'Store':>8} " + " ".join(f"{d:>10}" for d in day_names))
    print("-" * 75)
    for store in sorted(alns.stores):
        actual_id = store_id_mapping.get(store, store)
        vals = [alns.mu_ft[store][t] for t in range(6)]
        print(f"{str(actual_id):>8} " + " ".join(f"{v:>10.2f}" for v in vals))
    
    print_segment_summary(solution, alns)
    
    fw_rate, waste_qty, total_demand = compute_food_waste_rate(solution)
    print(f"\nFood waste quantity: {waste_qty:.2f} units out of {total_demand:.2f} units demand")
    print(f"Overall food waste rate: {fw_rate*100:.2f}%")
    
    so_rate, stockout_qty, _ = compute_stockout_rate(solution)
    print(f"Stockout quantity: {stockout_qty:.2f} units")
    print(f"Overall stockout rate: {so_rate*100:.2f}%")
    
    print("\n\n--- CONSTRAINT VIOLATIONS CHECK ---")
    violations = solution.validate_constraints()
    if violations:
        print(f"\u26a0\ufe0f  Found {len(violations)} constraint violations:")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print("\u2705 All constraints satisfied!")
    
    print("\n" + "="*100)


def default_algorithm_params():
    return {
        "alpha": 6,
        "beta": 0.8, "gamma": 0.2,
        "delta": 0.3, "eps": 0.7,
        "mu": 0.7, "xi": 0.3,
        "k": 2, "D": 0.003,
        "d": 0.99975, "g": 0.03,
        "lambda": 200,
        "theta_1": 33, "theta_2": 9, "theta_3": 11,
        "r": 0.1,
        "routing_iterations": 50,
        "routing_no_improve_limit": 30,
        "post_unsuccessful_limit": 2000,
        "p_fifo": 0.70,
    }


def main(instance_file_name=None, algorithm_params=None, max_iterations=None,
         time_limit=3600, target_cost=None, save_results=False, verbose_report=True):
    """MOSinLine entry point. Returns (best_solution, instance_data).
    
    Called from MOSinLine's main.py as:
        alns_solution, alns_instance_data = alns.main(instance_file_name=patt_file)
    save_results=False by default so that per-(depot, scenario) runs inside the
    pipeline do not spam results/*.csv; the standalone __main__ sets it True.
    """
    instance_file = instance_file_name or "instances/R101_10stores_s20.json"
    print(f"Loading instance: {instance_file}")
    instance_data = load_instance(instance_file)
    instance_name = instance_data['instance_name']
    print(f"Instance: {instance_name}")
    print(f"Stores: {len(instance_data['stores'])} stores")
    if instance_data.get('demand_by_segment') is not None:
        print("Per-segment demand data detected \u2705")
    elif instance_data.get('daily_demands_by_day') is not None:
        print("Day-varying demand data detected (single segment 'fresh') \u2705")
    else:
        print("Constant demand data (backward compatible mode)")
    
    if algorithm_params is None:
        algorithm_params = default_algorithm_params()

    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE ALNS OPTIMIZATION")
    print("Stage 1: Pattern optimization (ALNS)")
    print("Stage 2: Routing optimization (LNS)")
    print("=== (R,S) FIFO/LIFO shelf simulation | product segments ===")
    print("="*80)
    
    start_time = time.time()
    alns = ComprehensiveALNS(instance_data, algorithm_params)
    
    if max_iterations is None:
        if len(instance_data['stores']) <= 30: max_iterations = 1000
        elif len(instance_data['stores']) <= 100: max_iterations = 2000
        else: max_iterations = 3000
    
    best_solution = alns.run_alns(
        max_iterations=max_iterations,
        time_limit=time_limit,
        target_cost=target_cost
    )
    runtime = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ALNS OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Final objective value: {best_solution.cost:.2f}")
    print(f"Pattern cost: {best_solution.pattern_cost:.2f}")
    print(f"Routing cost: {best_solution.routing_cost:.2f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    if not best_solution.verify_all_deliveries():
        print("\n\u26a0\ufe0f Warning: Solution does not satisfy all delivery requirements!")
    else:
        print("\n\u2705 All delivery requirements satisfied")
    
    final_violations = best_solution.validate_constraints()
    if final_violations:
        print(f"\n\u26a0\ufe0f Final solution has {len(final_violations)} constraint violations")
    else:
        print("\n\u2705 All constraints satisfied")
    
    if verbose_report:
        print_comprehensive_solution(best_solution, instance_data, alns)
    
    if save_results:
        results = save_comprehensive_results(best_solution, instance_data, runtime)
        fw_rate, _, _ = compute_food_waste_rate(best_solution)
        so_rate, _, _ = compute_stockout_rate(best_solution)
        print_cost_and_metrics(
            "COMPREHENSIVE ALNS COST BREAKDOWN", best_solution.evaluator.lambda_param,
            results["pattern_cost"], results["fw_purchase_cost"], results["stockout_cost"],
            results["transport_dist_cost"], results["transport_fuel_cost"],
            results["food_waste_emission_cost"], results["transport_pollution_cost"],
            runtime, alns.best_found_time, fw_rate, so_rate,
            results["total_distance"], results["total_load"], results["vehicles_used"])
        op_df = alns.build_operator_performance_table()
        print("\nOperator performance (Stage-1 pattern operators):")
        print(op_df.to_string(index=False))
    
    return best_solution, instance_data


# Main execution
if __name__ == "__main__":
    instance_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(instance_file_name=instance_file, save_results=True)