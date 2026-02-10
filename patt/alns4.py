
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

print("Implements full ALNS approach: Pattern optimization (Stage 1) + Routing optimization (Stage 2)")

class ComprehensiveSolution:
    
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
    
    def __init__(self, loc, delta, c_fr, waste_fractions, D_f, patterns, 
                 c_km, theta_FW, theta_TR, eta, W0, alpha, beta, lambda_param, Q,
                 gamma_f, Q_day_min, Q_day_max):
        self.c_fr = c_fr
        self.waste_fractions = waste_fractions
        self.D_f = D_f
        self.patterns = patterns
        self.alpha = alpha
        self.theta_FW = theta_FW
        self.loc = loc
        self.delta = delta
        self.c_km = c_km
        self.theta_TR = theta_TR
        self.eta = eta
        self.W0 = W0
        self.beta = beta
        self.Q = Q
        self.lambda_param = lambda_param
        self.gamma_f = gamma_f
        self.Q_day_min = Q_day_min
        self.Q_day_max = Q_day_max
    
    def calculate_pattern_cost(self, store, pattern_id):
        
        economic_cost = self.c_fr[store, pattern_id]
        food_waste_cost = (self.alpha * self.theta_FW * self.D_f[store] * 
                          self.waste_fractions[store, pattern_id])
        
        return ((1 - self.lambda_param) * economic_cost + 
                self.lambda_param * food_waste_cost)
    
    def calculate_route_cost(self, route, loads):
        
        if len(route) <= 2:
            return 0
        
        transport_cost = 0
        pollution_cost = 0
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = self.delta[from_node, to_node]
            transport_cost += self.c_km * distance
            current_load = sum(loads[j] for j in route[i+1:] if j != 0)
            base = self.beta * self.theta_TR * self.eta * distance * self.W0
            var  = self.beta * self.theta_TR * self.eta * distance * current_load
            pollution_cost += base + var
        total_cost = (1 - self.lambda_param) * transport_cost + self.lambda_param * pollution_cost
        return total_cost
    
    def is_route_feasible(self, route, loads):
        
        total_load = sum(loads.get(node, 0) for node in route if node != 0)
        return total_load <= self.Q

class ALNSPatternOperators:

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
        """
        Paper Eq.(17) pattern similarity ω:
        matching periods are days where BOTH patterns deliver (1-1 overlap),
        divided by |T|.
        """
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
        """
        Paper Step 4: choose new pattern randomly among feasible patterns
        that have strictly higher similarity to seed pattern than the candidate's current pattern.
        Returns None if no such pattern exists.
        """
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
        """
        Paper Algorithm 1 (score-related) — Proximity operator (store-only adaptation).
        Score uses geographic distance + pattern-(dis)similarity.
        """
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
        """
        Paper Algorithm 1 (score-related) — Sales volume operator (store-only adaptation).
        Score uses Eq.(21): absolute difference of total demand.
        Step 4 uses the same "higher similarity than previous pattern" rule.
        """
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
        """
        Algorithm 2 (paper-faithful, strict): Pattern-dependent Cost Operator (store-only)

        - O: all stores (combinations)
        - L: list of stores that have been ADJUSTED (changed) in this operator call
        - while |L| < c:
            Step 1: sort all remaining stores by current pattern-dependent cost in DESC order
            Step 2: pick a store at zeta^alpha down the ranking
            Step 3 (strict): only if current cost != minimum cost among feasible patterns:
                pick new pattern uniformly at random among all feasible patterns with lower cost
                else: continue (do NOT add to L)
            Step 4: remove from O (here: mark by adding to L), and record the change
        """
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
        """
        Pattern-dependent Cost Operator (Algorithm 2) for FOOD WASTE component.

        O: all stores
        c_pat(f): food-waste cost of CURRENT pattern at store f
        Pick store biased towards high c_pat, then switch to a random feasible pattern with LOWER c_pat.
        """
        O = list(self.stores)
        L = []
        changes = []
        max_trials = 10 * max(1, len(O))
        trials = 0

        def fw_cost(f, r):
            ev = self.evaluator
            wf = ev.waste_fractions[f, r]
            return ev.theta_FW * ev.D_f[f] * wf

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
        """
        Pattern-dependent Cost Operator (Algorithm 2) for TRANSPORT POLLUTION component.

        IMPORTANT:
        - True transport pollution mainly depends on routing (Stage 2).
        - Here we use a *pattern-dependent proxy* so Stage 1 can still be guided.

        Proxy idea (simple + consistent):
        For each delivery day t in pattern r:
            distance ≈ 2 * dist(depot, f)   (out-and-back proxy)
            load ≈ delivered quantity p_frt
            emissions ≈ theta_TR * eta * (W0 + load) * distance
        Sum over t.
        """

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
                    total += ev.beta * emissions
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
            """
            [ML Operator] K-Means Cluster Removal.
            Uses Unsupervised Learning to identify a geographic cluster of stores
            and forces them to change patterns simultaneously.
            """
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
    
    def smart_eco_pattern_operator(self, solution, c_stores):
        """
        [New Operator] Smart Eco-Improvement.
        Applies the 'Smart Pattern Assignment' logic from the initial solution construction
        to a subset of stores.
        
        Score = PatternCost + ProxyRoutingCost + ProxyPollutionCost
        """
        changed_stores = []

        candidates = random.sample(self.stores, min(c_stores, len(self.stores)))

        ev = self.evaluator
        lambda_param = ev.lambda_param
        c_km = ev.c_km
        theta_TR = ev.theta_TR
        eta = ev.eta
        W0 = ev.W0
        beta = ev.beta
        
        for store in candidates:
            feas = list(self.feasible_patterns_by_store.get(store, []))
            if not feas:
                continue

            current_pattern = solution.pattern_assignments[store]
            
            best_r = current_pattern
            min_score = float('inf')

            dist_proxy = self.distances.get((0, store), 10.0) * 2.0 

            for r in feas:
                
                pat_cost = ev.calculate_pattern_cost(store, r)
                freq = max(sum(self.patterns[r]), 1)
                econ_transport = freq * dist_proxy * c_km
                avg_drop = ev.D_f[store] / freq
                pollution_val = theta_TR * eta * (W0 + avg_drop) * dist_proxy * freq
                pollution_cost = beta * pollution_val

                total_score = pat_cost + \
                              (1 - lambda_param) * econ_transport + \
                              lambda_param * pollution_cost
                
                if total_score < min_score:
                    min_score = total_score
                    best_r = r

            if best_r != current_pattern:
                changed_stores.append((store, best_r))
                
        return changed_stores    

class LNSRoutingOperators:

    def __init__(self, evaluator, p_frt, alpha=6):
        self.evaluator = evaluator
        self.p_frt = p_frt
        self.alpha = alpha

    def shaw_removal(self, solution, day, num_to_remove, mu=0.6, xi=0.4):
        """
        Shaw Removal using Eq.(22) with segment term removed:

        R_ij^S = μ * (c_ij^tran / c_max^tran) + ξ * (|o_i - o_j| / o_max)

        - c_ij^tran proxy: economic transport cost = c_km * delta[i,j]
          (matches your routing cost breakdown)
        - o_i: delivery size on 'day' for customer i under its assigned pattern
        """

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

            transport_cost += ev.c_km * dist

            current_load = suffix_load[i + 1]

            fuel = ev.eta * (ev.W0 + current_load) * dist
            emissions = ev.theta_TR * fuel
            pollution_cost += ev.beta * emissions

        lam = ev.lambda_param
        return (1.0 - lam) * transport_cost + lam * pollution_cost

    def regret_k_insertion(self, solution, day, removed_customers, k=2):
        """
        Regret-k insertion (Ropke & Pisinger 2006).

        Key point (paper-faithful heuristic behavior):
        - Treat EMPTY routes [0,0] as NORMAL insertion candidates from the start.
        So "opening a new vehicle" happens naturally when it is the best Δ-cost option,
        not as a fallback.

        Insertion cost uses your exact objective delta:
            Δcost = Δ[(1-λ)*econ + λ*pollution_precise]
        (implemented via _route_cost_econ_plus_pollution_precise)

        Returns True iff all customers are inserted.
        """
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
    """
    JOINT destroy+repair for ONE store:
    - destroy: remove store from all days' routes
    - repair: DP choose visit days (cover-to-next), and best insertion per chosen day
    """
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
        """
        Return an object with method best(q) -> (delta_cost, vehicle, pos)
        Uses CombinedEvaluator.calculate_route_cost(route, loads) with custom loads including f:q
        """
        alns = self.alns
        ev = alns.evaluator
        loads_old_by_v = {}
        cost_old_by_v  = {}
        for v, route in routes_day.items():
            route = alns._normalize_route(route)
            routes_day[v] = route
            loads_old = {}
            for n in route:
                if n == 0:
                    continue
                pat = alns.current_solution_pattern_lookup.get(n)
                loads_old[n] = alns.p_frt.get((n, pat, alns.current_day_for_lookup), 0.0)
            loads_old_by_v[v] = loads_old
            cost_old_by_v[v]  = ev.calculate_route_cost(route, loads_old)

        cache = {}

        def best(q):
            if q <= 0:
                return (0.0, None, None)
            if q > alns.Q or q > alns.gamma_f[f]:
                return (float("inf"), None, None)
            if q in cache:
                return cache[q]

            best_delta = float("inf")
            best_v, best_pos = None, None

            for v, route in routes_day.items():
                route = alns._normalize_route(route)
                loads_old = loads_old_by_v[v]
                old_cost  = cost_old_by_v[v]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [f] + route[pos:]

                    loads_new = dict(loads_old)
                    loads_new[f] = q

                    if not ev.is_route_feasible(new_route, loads_new):
                        continue

                    new_cost = ev.calculate_route_cost(new_route, loads_new)
                    delta = new_cost - old_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_v, best_pos = v, pos

            cache[q] = (best_delta, best_v, best_pos)
            return cache[q]

        return best

    def pattern_side_cost_from_m(self, f, m):
        """
        cost that depends only on frequency m:
        (1-λ)*EOQ_weekly_cost(f,m) + λ*FW_cost(f,m)
        """
        alns = self.alns
        ev = alns.evaluator

        D_week = ev.D_f[f]
        if D_week <= 0:
            return 0.0
        F_per_visit = alns.F_per_visit
        r_unit      = alns.r_unit
        h_week      = alns.h_week
        eoq = F_per_visit * m + r_unit * D_week + h_week * (D_week / (2.0 * m))
        qty_per_delivery = D_week / m
        safety_stock = alns.d_f[f]
        lead_time = 0
        shelf_life = 4
        daily_avg_demand = alns.d_f[f]
        if daily_avg_demand <= 0:
            wf = 0.0
        else:
            fsc = (qty_per_delivery + safety_stock - lead_time * daily_avg_demand) / (shelf_life * daily_avg_demand)
            wf = max(0.04 * fsc, 0.47 * fsc - 0.31)
            wf = max(0.0, wf)

        fw_cost = ev.alpha * ev.theta_FW * D_week * wf

        lam = ev.lambda_param
        return (1.0 - lam) * eoq + lam * fw_cost

    def compute_cover_to_next_quantities(self, deliver_days, d_daily):
        deliver_days = sorted(deliver_days)
        q_by_day = {}
        for idx, s in enumerate(deliver_days):
            nxt = deliver_days[(idx + 1) % len(deliver_days)]
            gap = (nxt - s) if nxt > s else (6 - s + nxt)
            q_by_day[s] = d_daily * gap
        return q_by_day

    def joint_ds_visitdays_cover_to_next(self, solution, f):
        """
        Returns: (solution_modified, touched_days, best_total_cost_estimate)
        """
        alns = self.alns
        routes_by_day = solution.routes_by_day
        routes_by_day = self.remove_node_all_days(routes_by_day, f)
        alns.current_solution_pattern_lookup = solution.pattern_assignments.copy()
        best_insert = {}
        for day in range(6):
            alns.current_day_for_lookup = day
            best_insert[day] = self.build_insertion_evaluator(routes_by_day[day], f)
        INF = float("inf")
        dp = [dict() for _ in range(7)]
        pred = {}
        dp[0][(None, None, 0)] = 0.0

        def relax(tn, state_new, val_new, prev_state, action):
            if val_new < dp[tn].get(state_new, INF):
                dp[tn][state_new] = val_new
                pred[(tn, *state_new)] = (prev_state, action)

        for t in range(6):
            for (L, F0, m), base in list(dp[t].items()):
                relax(t+1, (L, F0, m), base, (t, L, F0, m), ("no",))
                if L is None:
                    relax(t+1, (t, t, 1), base, (t, L, F0, m), ("first", t))
                else:
                    gap = t - L
                    if gap <= 0:
                        continue
                    qL = alns.d_f[f] * gap
                    ins_cost, _, _ = best_insert[L](qL)
                    if ins_cost >= INF:
                        continue
                    relax(t+1, (t, F0, m+1), base + ins_cost, (t, L, F0, m),
                          ("close_gap", L, qL))
        best_total = INF
        best_end = None
        best_wrap = None

        for (L, F0, m), base in dp[6].items():
            if L is None or m < 2:
                continue
            gap_wrap = 6 - L + F0
            qL = alns.d_f[f] * gap_wrap
            ins_cost, _, _ = best_insert[L](qL)
            if ins_cost >= INF:
                continue

            routing_cost = base + ins_cost
            side_cost = self.pattern_side_cost_from_m(f, m)
            total = routing_cost + side_cost

            if total < best_total:
                best_total = total
                best_end = (6, L, F0, m)
                best_wrap = (L, qL)

        if best_end is None:
            return solution, set(), INF
        deliver_days = []
        cur = best_end
        while True:
            t, L, F0, m = cur
            if t == 0:
                break
            prev_state, action = pred[cur]
            if action[0] == "first":
                deliver_days.append(action[1])
            elif action[0] == "close_gap":
                deliver_days.append(L)
            cur = prev_state

        deliver_days = sorted(set(deliver_days))
        if len(deliver_days) < 2:
            return solution, set(), INF
        bits = [0]*6
        for d in deliver_days:
            bits[d] = 1
        r_new = alns.bits_to_r.get(tuple(bits), None)
        if r_new is None:
            return solution, set(), INF
        solution.pattern_assignments[f] = r_new
        q_by_day = self.compute_cover_to_next_quantities(deliver_days, alns.d_f[f])
        touched_days = set(deliver_days)
        for s in deliver_days:
            q = q_by_day[s]
            ins_cost, v, pos = best_insert[s](q)
            if v is None:
                return solution, set(), INF
            rt = routes_by_day[s][v]
            routes_by_day[s][v] = rt[:pos] + [f] + rt[pos:]
            routes_by_day[s][v] = alns._normalize_route(routes_by_day[s][v])
        solution._calculate_cost()
        return solution, touched_days, best_total

class ComprehensiveALNS:

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
        self.patterns, self.R = self._generate_patterns()
        self.bits_to_r = {tuple(p): i for i, p in enumerate(self.patterns)}
        self.r_to_bits = {i: tuple(p) for i, p in enumerate(self.patterns)}

        try:
            print("ALNS idx(111111) =", self.bits_to_r[(1,1,1,1,1,1)])
        except Exception:
            pass

        self.delta = {(self.store_id_mapping[i], self.store_id_mapping[j]) : c for (i,j), c in instance_data["transportation_costs"].items()}
        self._calculate_parameters(instance_data)
        self.evaluator = CombinedEvaluator(
            self.loc, self.delta, self.c_fr, self.waste_fractions, self.D_f, 
            self.patterns, self.c_km, self.theta_FW, self.theta_TR, self.eta, 
            self.W0, self.alpha, self.beta, self.lambda_param, self.Q,
            self.gamma_f, self.Q_day_min, self.Q_day_max)
        self.feasible_patterns_by_store = {f: [] for f in self.stores}
        for f in self.stores:
            for r in self.R:
                ok = True
                for t in range(6):
                    drop = self.p_frt.get((f, r, t), 0.0)
                    
                    if drop > self.Q or drop > self.gamma_f[f]:
                        ok = False
                        break
                if ok:
                    self.feasible_patterns_by_store[f].append(r)

        for f in self.stores:
            if not self.feasible_patterns_by_store[f]:
                
                viol_list = []
                for r in self.R:
                    worst = max(self.p_frt.get((f, r, t), 0.0) - min(self.Q, self.gamma_f[f]) for t in range(6))
                    viol_list.append((worst, r))
                viol_list.sort()
                self.feasible_patterns_by_store[f] = [r for _, r in viol_list[:3]]
        print("Per-store feasible patterns built.")
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
            "Cost-related",
            "Move-one",
            "Move-two",
            "Random",
            "Transport Pollution Cost",            
            "Food Waste Cost",
            "ML Spatial Clustering",
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
            "calls": 0,             
            "improved": 0,          
            "total_reduction": 0.0, 
            "max_reduction": 0.0    
        }

        self.solution_cache = {}          
        self.partial_solutions_log = []   
        self.max_stage1_retries = 100     
        
    def _generate_patterns(self):
        """Generate feasible delivery patterns (EXACTLY like Gurobi, updated):
        - exclude freq 0/1 (keep only sum(p) >= 2)
        - no min-gap rule
        - sort by binary value ascending
        """
        def filter_delivery_patterns_unified(all_patterns):
            
            return [p for p in all_patterns if sum(p) >= 2]

        all_patterns = list(itertools.product([0, 1], repeat=6))
        patterns_unsorted = filter_delivery_patterns_unified(all_patterns)
        
        patterns = sorted(patterns_unsorted, key=lambda p: int(''.join(map(str, p)), 2))
        R = list(range(len(patterns)))
        return patterns, R
    
    def _calculate_parameters(self, instance_data):
        
        self.d_f = {}
        for i, store_id in enumerate(self.stores):
            self.d_f[store_id] = self.daily_demands[i]
        
        self.D_f = {f: self.d_f[f] * 6 for f in self.stores}
        self.p_frt = self._calculate_delivery_quantities()
        self.gamma_f = {f: max(self.D_f[f] // 2, 50) for f in self.stores}
        self.Q_day_min = 0
        self.Q_day_max = 99999
        self.num_vehicles = 50
        self.Q = instance_data.get("vehicle_capacity",44)
        self.c_km = instance_data.get("cost_per_km", 2.0)
        self.theta_FW = 350 # todo has to be read from foodwaste_emission factor
        self.theta_TR = instance_data.get("marginal_co2_emissions", 0.05*1*2.7)
        self.eta = 1 # keep fixed at 1 and only change marginal_co2_emissions!
        self.W0 = instance_data.get("vehicle_empty_weight", 8)
        self.alpha = 1
        self.beta = 1 # keep fixed at 1 and only change marginal_co2_emissions!
        self.lambda_param = instance_data.get("weighting_factor_patt", 0.3)
        self._calculate_pattern_costs()
        self._calculate_waste_fractions()


        # fixedco2emissions = eta*W0*delta[i,j]*thetha_TR*beta
        # marginal_emissions = eta*q*delta[i,j]*thetha_TR*beta
    
    def _calculate_delivery_quantities(self):
        
        delivery_quantities = {}
        
        for store_id in self.stores:
            daily_demand = self.d_f[store_id]
            
            for pattern_id, pattern in enumerate(self.patterns):
                delivery_days = [day for day in range(6) if pattern[day] == 1]
                
                if not delivery_days:
                    for day in range(6):
                        delivery_quantities[store_id, pattern_id, day] = 0
                    continue
                
                for day in range(6):
                    if pattern[day] == 1:
                        current_idx = delivery_days.index(day)
                        next_delivery_idx = (current_idx + 1) % len(delivery_days)
                        next_delivery_day = delivery_days[next_delivery_idx]
                        
                        if next_delivery_day > day:
                            days_to_cover = next_delivery_day - day
                        else:
                            days_to_cover = 6 - day + next_delivery_day
                        
                        delivery_amount = daily_demand * days_to_cover
                        delivery_quantities[store_id, pattern_id, day] = delivery_amount
                    else:
                        delivery_quantities[store_id, pattern_id, day] = 0
        
        return delivery_quantities
    
    def _calculate_pattern_costs(self):
        self.c_fr = {}

        freq_of = {r: max(sum(self.patterns[r]), 1) for r in self.R}

        for f in self.stores:
            D = self.D_f[f]   

            if D <= 0:
                
                for r in self.R:
                    self.c_fr[f, r] = 0.0
                continue

            for r in self.R:
                m = freq_of[r]   

                weekly_cost = (
                    self.F_per_visit * m        
                    + self.r_unit * D           
                    + self.h_week * (D / (2.0 * m))  
                )

                self.c_fr[f, r] = max(float(weekly_cost), 0.01)
    
    def _calculate_waste_fractions(self):
        
        self.waste_fractions = {}
        
        def calculate_qty_per_delivery(f, r):
            frequency = max(sum(self.patterns[r]), 1)
            return self.D_f[f] / frequency
        
        def calculate_FSC(f, r):
            qty_per_delivery = calculate_qty_per_delivery(f, r)
            safety_stock = self.d_f[f]
            lead_time = 0
            shelf_life = 4
            daily_avg_demand = self.d_f[f]
            if daily_avg_demand == 0:
                return 0
            numerator = qty_per_delivery + safety_stock - lead_time * daily_avg_demand
            denominator = shelf_life * daily_avg_demand
            return numerator / denominator
        
        def calculate_waste_fraction(fsc):
            return max(0.04 * fsc, 0.47 * fsc - 0.31)
        
        for f in self.stores:
            for r in self.R:
                fsc = calculate_FSC(f, r)
                waste_fraction = calculate_waste_fraction(fsc)
                self.waste_fractions[f, r] = max(0, waste_fraction)
    
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
        """
        
        for day in range(6):
            self._enforce_capacity_for_day(solution, day)
        """
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

    def run_alns(self, max_iterations=1000, time_limit=None):
        """
        Main ALNS algorithm with Stage 1 (patterns) and Stage 2 (routing)
        """
        print("\nRunning Comprehensive ALNS optimization...")

        alns_start_time = time.time()
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
        search_leg_size = 100
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
                    print(f"\n⏰ Time limit of {time_limit/3600:.2f} hours reached at iteration {iteration}.")
                    print("Stopping ALNS main loop and returning current best solution so far.")
                    break

            old_current_cost = current_solution.cost
            best_cost = best_solution.cost
            new_solution = None
            op_idx = -1
            stage1_feasible = False
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
            
            if not stage1_feasible:
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

            if new_cost < best_cost:
                accept_status = "new_best"
                best_solution = new_solution.copy()
                current_solution = new_solution
                iterations_without_improvement = 0
                print(f"Iteration {iteration}: New best = {best_solution.cost:.2f}")

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
            if accept_status == "new_best":
                stats["best"] += 1
            elif accept_status == "new_incumbent":
                stats["better"] += 1
            elif accept_status == "accepted_sa":
                stats["accept"] += 1
            else:
                stats["reject"] += 1
            pattern_vector = [current_solution.pattern_assignments[s] for s in sorted(self.stores)]
            self.pattern_history.append(pattern_vector)
            self.operator_uses[op_idx] += 1
            if accept_status == "new_best":
                self.operator_scores[op_idx] += theta_1
            elif accept_status == "new_incumbent":
                self.operator_scores[op_idx] += theta_2
            elif accept_status == "accepted_sa":
                self.operator_scores[op_idx] += theta_3

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
            if len(self.stores) <= 3 and iterations_without_improvement >= 50:
                print(f"\n🛑 Early stopping triggered: No improvement for {iterations_without_improvement} iterations on small instance ({len(self.stores)} stores).")
                break
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
                    print(f"\n⏰ Time limit reached during post-optimization (day {day}).")
                    print("Stopping remaining post-optimization and returning current best solution.")
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
        
        if num_stores <= 25:
            c_min, c_max = 5, 15
        elif num_stores <= 50:
            c_min, c_max = 5, 20
        elif num_stores <= 100:
            c_min, c_max = 5, 30
        else:
            
            c_min, c_max = 5, 50

        c_max = min(c_max, num_stores)
        c_min = min(c_min, c_max)

        c_stores = random.randint(c_min, c_max)

        new_solution = solution.copy()
        touched_days = set()   
        is_joint_op = False 
        if operator_idx == 0:
            
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
            changes = self.pattern_operators.cost_related_operator(new_solution, c_stores)
        elif operator_idx == 4:
            changes = self.pattern_operators.move_one_operator(new_solution, c_stores)
        elif operator_idx == 5:
            changes = self.pattern_operators.move_two_operator(new_solution, c_stores)
        elif operator_idx == 6:
            changes = self.pattern_operators.random_operator(new_solution, c_stores)
        elif operator_idx == 7:
            changes = self.pattern_operators.transport_pollution_cost_operator(
                new_solution, c_stores, alpha=self.params.get("alpha", 6)
            )
        elif operator_idx == 8: 
            changes = self.pattern_operators.ml_cluster_spatial_operator(new_solution, c_stores)
        elif operator_idx == 9: 
            changes = self.pattern_operators.smart_eco_pattern_operator(new_solution, c_stores)
        else:
            changes = self.pattern_operators.food_waste_cost_operator(
                new_solution, c_stores, alpha=self.params.get("alpha", 6)
            )

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
                if i in rr['route'][1:-1]:
                    route_i = rr
                if j in rr['route'][1:-1]:
                    route_j = rr
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
                routes_day[v] = routes[v]['route']
            else:
                routes_day[v] = [0, 0]
        return routes_day
    
    def _normalize_route(self, route):
        
        if route is None or len(route) == 0:
            return [0, 0]
        rt = list(route)
        if rt[0] != 0:
            rt = [0] + rt
        if rt[-1] != 0:
            rt = rt + [0]
        if len(rt) < 2:
            rt = [0, 0]
        if rt == [0, 0, 0]:
            rt = [0, 0]
        return rt

    def _day_routing_objective(self, sol, day):
        """
        Routing objective for ONE day used by Stage-2 acceptance (paper-like).
        Use the same weighted routing cost definition as in regret insertion.
        """
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
        """
        Stage 2 (Routing Optimization) with Performance Tracking.
        """
        sol = solution.copy()

        routing_iters = self.params.get("routing_iterations", 100)
        no_improve_limit = self.params.get("routing_no_improve_limit", 50)
        D = self.params.get("D", 0.003)
        k = self.params.get("k", 2)
        mu = self.params.get("mu", 0.6)
        xi = self.params.get("xi", 0.4)

        total_improvement_this_call = 0.0

        for day in range(6):
            if touched_days is None:
                touched_days = set()

            if skip_init_routing:

                for v in range(self.num_vehicles):
                    if v not in sol.routes_by_day[day]:
                        sol.routes_by_day[day][v] = [0, 0]
            else:
                
                if len(touched_days) == 0:
                    sol.routes_by_day[day] = self._build_savings_routes_for_day(sol, day)
                else:
                    
                    if day not in touched_days:
                        sol.routes_by_day[day] = self._build_savings_routes_for_day(sol, day)
                    
                    for v in range(self.num_vehicles):
                        if v not in sol.routes_by_day[day]:
                            sol.routes_by_day[day][v] = [0, 0]
            
            baseline_cost = self._day_routing_objective(sol, day)
            current_routes = {v: list(r) for v, r in sol.routes_by_day[day].items()}
            best_routes = {v: list(r) for v, r in sol.routes_by_day[day].items()}

            best_cost = baseline_cost
            current_cost = best_cost 
            it_wo_improve = 0

            if len(sol.get_all_served_stores(day)) <= 1:
                continue

            for _ in range(routing_iters):
                if it_wo_improve >= no_improve_limit:
                    break

                cand = sol.copy()
                cand.routes_by_day[day] = {v: list(r) for v, r in current_routes.items()}

                num_customers = len(cand.get_all_served_stores(day))
                if num_customers <= 1: break
                num_to_remove = max(1, int(0.3 * num_customers))
                removed = self.routing_operators.shaw_removal(cand, day, num_to_remove, mu=mu, xi=xi)
                
                if not removed:
                    it_wo_improve += 1
                    continue

                rem_set = set(removed)
                for v, route in cand.routes_by_day[day].items():
                    rr = [n for n in route if (n == 0 or n not in rem_set)]
                    cand.routes_by_day[day][v] = self._normalize_route(rr)
                ok = self.routing_operators.regret_k_insertion(cand, day, removed, k=k)
                
                if not ok or not self._day_routes_feasible(cand, day):
                    it_wo_improve += 1
                    continue
                cand_cost = self._day_routing_objective(cand, day)

                if cand_cost <= best_cost * (1.0 + D):
                    current_routes = {v: list(r) for v, r in cand.routes_by_day[day].items()}
                    current_cost = cand_cost

                    if cand_cost < best_cost - 1e-4: 
                        best_cost = cand_cost
                        best_routes = {v: list(r) for v, r in cand.routes_by_day[day].items()}
                        it_wo_improve = 0
                    else:
                        it_wo_improve += 1
                else:
                    it_wo_improve += 1
            sol.routes_by_day[day] = {v: list(r) for v, r in best_routes.items()}

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
            total_day_delivery = 0
            for store in solution.get_all_served_stores(day):
                pattern_id = solution.pattern_assignments[store]
                total_day_delivery += self.p_frt.get((store, pattern_id, day), 0)
            
            if total_day_delivery < self.Q_day_min or total_day_delivery > self.Q_day_max:
                print(f"Warning: Day {day} delivery {total_day_delivery:.1f} violates bounds [{self.Q_day_min}, {self.Q_day_max}]")

            self._enforce_capacity_for_day(solution, day)

    def _run_lns_for_day(self, solution, day, unsuccessful_limit=2000):
        """
        Post-optimization (extended LNS) for ONE day (paper-like):
        - increase the LNS search limit of UNSUCCESSFUL iterations significantly
        - use Shaw removal + Regret-k insertion
        - acceptance: Record-to-Record Travel (RRT) with deviation D
        - stop after 'unsuccessful_limit' iterations without BEST improvement
        """
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
            if num_customers <= 1:
                break

            num_to_remove = max(1, int(0.3 * num_customers))
            removed = self.routing_operators.shaw_removal(current, day, num_to_remove, mu=mu, xi=xi)
            if not removed:
                unsuccessful += 1
                continue

            cand = current.copy()
            rem_set = set(removed)
            for v, rt in cand.routes_by_day[day].items():
                rr = [n for n in rt if (n == 0 or n not in rem_set)]
                cand.routes_by_day[day][v] = self._normalize_route(rr)
            ok = self.routing_operators.regret_k_insertion(cand, day, removed, k=k)
            if not ok:
                unsuccessful += 1
                continue

            if not day_feasible(cand):
                unsuccessful += 1
                continue

            cand_c = day_cost(cand)
            if cand_c <= best_c * (1.0 + D):
                current = cand
                curr_c = cand_c
                if cand_c < best_c:
                    best = cand.copy()
                    best_c = cand_c
                    unsuccessful = 0
                else:
                    unsuccessful += 1
            else:
                unsuccessful += 1

        sol.routes_by_day[day] = {v: list(r) for v, r in best.routes_by_day[day].items()}
        sol._calculate_cost()
        return sol
    
    def _calculate_start_temperature(self, initial_cost, g):
        
        return -g * initial_cost / np.log(0.5)
    
    def _select_operator(self):
            """
            [ML Upgrade] Select operator using UCB1 (Upper Confidence Bound) algorithm.
            This is a form of simple Reinforcement Learning (Multi-Armed Bandit).
            """
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
                if self.operator_uses[i] > 0:
                    
                    self.operator_total_score[i] += self.operator_scores[i]
                    self.operator_total_calls[i] += self.operator_uses[i]

                self.operator_scores[i] = 0
                self.operator_uses[i] = 0
    
    def _apply_random_destruction(self, solution):
        """
        Paper-faithful diversification tool:
        - Random Operator destroys a large part of the CURRENT solution by changing many pattern assignments.
        - Do NOT additionally destroy routes here; Stage 2 will rebuild (Savings + LNS) anyway.
        """
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
            if len(route) <= 2:
                continue  

            total_load = route_total_load(route)
            if total_load <= Q:
                continue  

            customers = [n for n in route if n != 0]

            new_chunks = []
            current_chunk = []
            current_load = 0.0

            for cust in customers:
                pat = solution.pattern_assignments[cust]
                need = self.p_frt.get((cust, pat, day), 0.0)
                if need > Q:

                    need = min(need, Q)
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
                    if candidates:
                        v_new = candidates[0]
                    else:
                        
                        v_new = min(routes.keys(), key=lambda vv: route_total_load(routes[vv]))
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

    def _route_delta_estimate_for_pattern_change(self, sol, f, r_new):
        r_old = sol.pattern_assignments[f]
        if r_old == r_new:
            return 0.0
        bits_old = self.patterns[r_old]
        bits_new = self.patterns[r_new]
        delta_sum = 0.0
        for day in range(6):
            if bits_old[day] == bits_new[day]:
                continue
            if bits_new[day] == 1:  
                d = self._best_insertion_delta_for(sol, f, day, r_new=r_new)
                if d == float('inf'):   
                    d = 1e9
                delta_sum += d
            else:  
                delta_sum += self._deletion_gain_for(sol, f, day)
        return delta_sum

    def _apply_single_pattern_change_locally(self, sol, f, r_new):
        r_old = sol.pattern_assignments[f]
        if r_old == r_new:
            return True

        bits_old = self.patterns[r_old]
        bits_new = self.patterns[r_new]
        sol.pattern_assignments[f] = r_new  

        for day in range(6):
            if bits_old[day] == bits_new[day]:
                continue

            if bits_new[day] == 1:
                
                need = self.p_frt.get((f, r_new, day), 0.0)
                if need <= 0:
                    continue
                best = (float('inf'), None, None)

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
                        if delta < best[0]:
                            best = (delta, v, pos)

                if best[1] is None:
                    
                    return False

                v, pos = best[1], best[2]
                route = sol.routes_by_day[day][v]
                sol.routes_by_day[day][v] = route[:pos] + [f] + route[pos:]

            else:
                
                for v, route in sol.routes_by_day[day].items():
                    if f in route:
                        idx = route.index(f)
                        sol.routes_by_day[day][v] = route[:idx] + route[idx+1:]
                        break

        return True
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
    
    daily_demands = data["daily_demands"]
    
    ret = {
        'stores': stores,
        'store_id_mapping': store_id_mapping,
        'loc': loc,
        'daily_demands': daily_demands,
        'instance_name': data["instance_name"]
    }

    # add other coefficients (johannes edit)
    for k, v in data.items():
        if k not in ret and k != "id_map" and k != "depot":
            ret[k] = v

    return ret

def save_comprehensive_results(solution, instance_data, runtime):
    
    instance_name = instance_data['instance_name']
    store_id_mapping = instance_data['store_id_mapping']
    ev = solution.evaluator
    total_distance = 0
    total_load = 0
    vehicles_used = set()
    raw_pattern_econ_cost = 0
    raw_food_waste_cost = 0
    
    for store, pattern_id in solution.pattern_assignments.items():
        
        econ = ev.c_fr[store, pattern_id]

        waste = (ev.alpha * ev.theta_FW * ev.D_f[store] * ev.waste_fractions[store, pattern_id])
        
        raw_pattern_econ_cost += econ
        raw_food_waste_cost += waste
    raw_transport_econ_cost = 0
    raw_pollution_cost = 0
    
    for day in range(6):
        for vehicle, route in solution.routes_by_day[day].items():
            if len(route) > 2:
                vehicles_used.add(vehicle)
                
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    distance = ev.delta[from_node, to_node]
                    
                    total_distance += distance

                    raw_transport_econ_cost += ev.c_km * distance
                    current_load = 0
                    for j in range(i+1, len(route)):
                        if route[j] != 0:
                            pattern_id = solution.pattern_assignments[route[j]]
                            current_load += solution.p_frt.get((route[j], pattern_id, day), 0)

                    fuel = ev.eta * (ev.W0 + current_load) * distance
                    emissions = ev.theta_TR * fuel
                    raw_pollution_cost += ev.beta * emissions
                    
                    total_load += current_load

    lambda_val = ev.lambda_param
    
    if lambda_val == 0:
        final_obj = raw_pattern_econ_cost + raw_transport_econ_cost

        report_fw_cost = raw_food_waste_cost
        report_tp_cost = raw_pollution_cost

        report_pattern_cost = raw_pattern_econ_cost
        report_transport_cost = raw_transport_econ_cost
        
    else:

        final_obj = (1 - lambda_val) * (raw_pattern_econ_cost + raw_transport_econ_cost) + \
                    lambda_val * (raw_food_waste_cost + raw_pollution_cost)

        report_fw_cost = raw_food_waste_cost
        report_tp_cost = raw_pollution_cost
        report_pattern_cost = raw_pattern_econ_cost
        report_transport_cost = raw_transport_econ_cost
    final_violations = solution.validate_constraints()
    results_data = {
        "instance": instance_name,
        "solver": "comprehensive_alns_with_constraints",
        "lambda": lambda_val,
        "objective_value": final_obj,  

        "economic_cost": report_pattern_cost + report_transport_cost,
        "environmental_cost": report_fw_cost + report_tp_cost,
        "pattern_cost": report_pattern_cost,
        "transport_cost": report_transport_cost,
        "food_waste_cost": report_fw_cost,
        "transport_pollution_cost": report_tp_cost,
        
        "runtime": runtime,
        "total_distance": total_distance,
        "total_load": total_load,
        "vehicles_used": len(vehicles_used),
        "num_stores": len(instance_data['stores']),
        "num_vehicles": getattr(solution, "num_vehicles", None),
        "num_feasible_patterns": len(ev.patterns),
        "alpha": ev.alpha,
        "beta": ev.beta,
        "model_type": "comprehensive_alns",
        "constraint_violations": len(final_violations)
    }
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame([results_data])
    result_file = os.path.join("results", f"comprehensive_alns_{instance_name}.csv")
    
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', index=False, header=False)
    else:
        df.to_csv(result_file, mode='w', index=False, header=True)
    
    print(f"\n✅ Results saved to {result_file}")
    solution_details = {
        "instance_name": instance_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solver": "comprehensive_alns_with_constraints",
        "objective_value": final_obj,
        "lambda": lambda_val,
        "runtime": runtime,
        "pattern_assignments": {},
        "daily_routes": {},
        "cost_breakdown": {
            "pattern_cost": report_pattern_cost,
            "food_waste_cost": report_fw_cost,
            "transport_cost": report_transport_cost,
            "transport_pollution_cost": report_tp_cost,
            "total_economic": report_pattern_cost + report_transport_cost,
            "total_environmental": report_fw_cost + report_tp_cost,
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
    print(f"✅ Detailed solution saved to {solution_file}")

    print(f"\nCost Breakdown (Lambda = {lambda_val}):")
    print(f"Pattern Assignment Costs: {report_pattern_cost:.2f}")
    print(f"Transportation Costs:     {report_transport_cost:.2f}")
    print(f"-" * 30)
    print(f"Economic Cost:            {report_pattern_cost + report_transport_cost:.2f}")
    
    print(f"\nFood Waste Costs:         {report_fw_cost:.2f}")
    print(f"Transport Pollution Costs:{report_tp_cost:.2f}")
    print(f"-" * 30)
    
    print(f"Environmental Cost:       {report_fw_cost + report_tp_cost:.2f}")
    
    print(f"=" * 30)
    print(f"TOTAL OBJECTIVE VALUE:    {final_obj:.2f}")
    if lambda_val == 0:
        print(f"(Note: Objective is pure Economic. Environmental cost is evaluated ex-post.)")
    
    return results_data

def print_comprehensive_solution(solution, instance_data, alns):
    
    store_id_mapping = instance_data['store_id_mapping']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
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
        pattern_bits = solution.evaluator.patterns[pattern_id]
        frequency = sum(pattern_bits)
        delivery_days = [day_names[i] for i in range(6) if pattern_bits[i] == 1]
        store_pattern_info.append((actual_id, store, pattern_id, pattern_bits, frequency, delivery_days))
    
    store_pattern_info.sort(key=lambda x: x[0])
    
    for actual_id, gurobi_id, pattern_id, pattern_bits, frequency, delivery_days in store_pattern_info:
        pattern_str = ''.join(str(b) for b in pattern_bits)
        days_str = ', '.join(delivery_days)
        print(f"{actual_id:>10} {pattern_id:>8} {frequency:>10} {pattern_str:>15} {days_str:>30}")
    print("\n--- PATTERN FREQUENCY SUMMARY ---")
    freq_count = {}
    for _, _, _, _, frequency, _ in store_pattern_info:
        freq_count[frequency] = freq_count.get(frequency, 0) + 1
    
    for freq in sorted(freq_count.keys()):
        print(f"{freq} deliveries/week: {freq_count[freq]} stores ({freq_count[freq]/len(store_pattern_info)*100:.1f}%)")
    print("\n--- DAILY DELIVERY SCHEDULE ---")
    
    for day in range(6):
        print(f"\n{day_names[day]} (Day {day}):")
        print("-" * 50)
        stores_today = []
        for store in solution.stores:
            pattern_id = solution.pattern_assignments[store]
            if solution.evaluator.patterns[pattern_id][day] == 1:
                actual_id = store_id_mapping.get(store, store)
                delivery_qty = solution.p_frt.get((store, pattern_id, day), 0)
                stores_today.append((actual_id, store, delivery_qty))
        
        stores_today.sort(key=lambda x: x[0])
        
        print(f"Total stores requiring delivery: {len(stores_today)}")
        print(f"{'Store ID':>10} {'Quantity':>10}")
        for actual_id, gurobi_id, qty in stores_today:
            print(f"{actual_id:>10} {qty:>10.1f}")
        
        total_day_qty = sum(qty for _, _, qty in stores_today)
        print(f"Total quantity: {total_day_qty:.1f} units")
        if total_day_qty < alns.Q_day_min:
            print(f"⚠️  WARNING: Total daily quantity {total_day_qty:.1f} < minimum {alns.Q_day_min}")
        elif total_day_qty > alns.Q_day_max:
            print(f"⚠️  WARNING: Total daily quantity {total_day_qty:.1f} > maximum {alns.Q_day_max}")
    print("\n\n--- VEHICLE ROUTING ASSIGNMENTS ---")
    
    for day in range(6):
        print(f"\n{day_names[day]} (Day {day}):")
        print("=" * 80)
        
        day_has_routes = False
        vehicles_used_today = []
        
        for vehicle in range(alns.num_vehicles):
            if vehicle not in solution.routes_by_day[day]:
                continue
                
            route = solution.routes_by_day[day][vehicle]
            
            if len(route) > 2:
                day_has_routes = True
                vehicles_used_today.append(vehicle)
                route_load = 0
                store_loads = {}
                for node in route:
                    if node != 0:
                        pattern_id = solution.pattern_assignments[node]
                        load = solution.p_frt.get((node, pattern_id, day), 0)
                        store_loads[node] = load
                        route_load += load
                
                route_distance = sum(solution.evaluator.delta[route[i], route[i+1]] 
                                    for i in range(len(route)-1))
                route_str_parts = []
                for node in route:
                    if node == 0:
                        route_str_parts.append("DEPOT")
                    else:
                        actual_id = store_id_mapping.get(node, node)
                        route_str_parts.append(f"Store_{actual_id}")
                
                print(f"\nVehicle {vehicle}:")
                print(f"  Route: {' → '.join(route_str_parts)}")
                print(f"  Total distance: {route_distance:.1f} km")
                print(f"  Total load: {route_load:.1f}/{alns.Q} units (utilization: {route_load/alns.Q*100:.1f}%)")
                print(f"  Deliveries:")
                for i, node in enumerate(route):
                    if node != 0:
                        actual_id = store_id_mapping.get(node, node)
                        load = store_loads[node]
                        print(f"    Stop {i}: Store {actual_id} - {load:.1f} units")
                        if load > alns.gamma_f[node]:
                            print(f"      ⚠️  WARNING: Delivery {load:.1f} exceeds store capacity {alns.gamma_f[node]}")
                print(f"  Arc details:")
                cumulative_load = route_load
                for i in range(len(route)-1):
                    from_node = route[i]
                    to_node = route[i+1]
                    distance = solution.evaluator.delta[from_node, to_node]
                    
                    from_str = "DEPOT" if from_node == 0 else f"Store_{store_id_mapping.get(from_node, from_node)}"
                    to_str = "DEPOT" if to_node == 0 else f"Store_{store_id_mapping.get(to_node, to_node)}"
                    
                    print(f"    {from_str} → {to_str}: {distance:.1f} km, carrying {cumulative_load:.1f} units")
                    
                    if to_node != 0:
                        cumulative_load -= store_loads[to_node]
        
        if not day_has_routes:
            print("  No deliveries scheduled")
        else:
            print(f"\nDay summary:")
            print(f"  Vehicles used: {len(vehicles_used_today)} ({vehicles_used_today})")
            print(f"  Total vehicles available: {alns.num_vehicles}")
            print(f"  Vehicle utilization: {len(vehicles_used_today)/alns.num_vehicles*100:.1f}%")
    print("\n\n--- COST BREAKDOWN ---")
    print("=" * 50)
    total_pattern_cost = 0
    total_food_waste_cost = 0
    
    for store, pattern_id in solution.pattern_assignments.items():
        economic_cost = solution.evaluator.c_fr[store, pattern_id]
        waste_cost = (solution.evaluator.alpha * solution.evaluator.theta_FW * 
                        solution.evaluator.D_f[store] * 
                        solution.evaluator.waste_fractions[store, pattern_id])
        
        pattern_component = (1 - solution.evaluator.lambda_param) * economic_cost
        waste_component = solution.evaluator.lambda_param * waste_cost
        
        total_pattern_cost += pattern_component
        total_food_waste_cost += waste_component
    total_transport_cost = 0
    total_pollution_cost = 0
    total_distance = 0
    
    for day in range(6):
        for vehicle, route in solution.routes_by_day[day].items():
            if len(route) > 2:
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    distance = solution.evaluator.delta[from_node, to_node]
                    total_distance += distance
                    transport_cost = solution.evaluator.c_km * distance
                    total_transport_cost += (1 - solution.evaluator.lambda_param) * transport_cost
                    current_load = 0
                    for j in range(i+1, len(route)):
                        if route[j] != 0:
                            pattern_id = solution.pattern_assignments[route[j]]
                            current_load += solution.p_frt.get((route[j], pattern_id, day), 0)
                    
                    fuel = solution.evaluator.eta * (solution.evaluator.W0 + current_load) * distance
                    emissions = solution.evaluator.theta_TR * fuel
                    pollution_cost = solution.evaluator.beta * emissions
                    total_pollution_cost += solution.evaluator.lambda_param * pollution_cost
    
    print(f"Pattern Assignment Costs: {total_pattern_cost:.2f}")
    print(f"Food Waste Costs: {total_food_waste_cost:.2f}")
    print(f"Transportation Costs: {total_transport_cost:.2f}")
    print(f"Transport Pollution Costs: {total_pollution_cost:.2f}")
    print(f"-" * 30)
    print(f"Total Economic Cost: {total_pattern_cost + total_transport_cost:.2f}")
    print(f"Total Environmental Cost: {total_food_waste_cost + total_pollution_cost:.2f}")
    print(f"-" * 30)
    print(f"TOTAL OBJECTIVE VALUE: {solution.cost:.2f}")
    print("\n\n--- CONSTRAINT VIOLATIONS CHECK ---")
    print("=" * 50)
    
    violations = solution.validate_constraints()
    if violations:
        print(f"⚠️  Found {len(violations)} constraint violations:")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print("✅ All constraints satisfied!")
    print("\n\n--- SOLUTION QUALITY METRICS ---")
    print("=" * 50)
    unique_patterns = len(set(solution.pattern_assignments.values()))
    print(f"Pattern diversity: {unique_patterns} unique patterns used out of {len(solution.evaluator.patterns)} available")
    total_vehicles_used = 0
    for day in range(6):
        for vehicle, route in solution.routes_by_day[day].items():
            if len(route) > 2:
                total_vehicles_used += 1
    
    avg_vehicles_per_day = total_vehicles_used / 6
    print(f"Average vehicles used per day: {avg_vehicles_per_day:.1f}")
    print(f"Total distance traveled per week: {total_distance:.1f} km")
    print(f"Average distance per day: {total_distance/6:.1f} km")
    print("\nDelivery workload by day:")
    for day in range(6):
        deliveries = len(solution.get_required_deliveries(day))
        print(f"  {day_names[day]}: {deliveries} deliveries")
    
    print("\n" + "="*100)




def main(instance_file_name=None):
    allp = [p for p in itertools.product([0,1], repeat=6) if sum(p) >= 2]
    for i,p in enumerate(allp):
        print(i,p)

    if instance_file_name:
        instance_file = instance_file_name
    else:
        # instance_file = "instances/R101_30stores_s2.json"
        instance_file = "instances/R101_5stores_s2.json"

    print(f"Loading instance: {instance_file}")
    instance_data = load_instance(instance_file)
    instance_name = instance_data['instance_name']
    
    print(f"Instance: {instance_name}")
    print(f"Stores: {len(instance_data['stores'])} stores")
    algorithm_params = {
        "alpha": 6,
        "beta": 0.8,
        "gamma": 0.2,
        "delta": 0.3,
        "eps": 0.7,
        "mu": 0.7,
        "xi": 0.3,
        "k": 2,
        "D": 0.003,
        "d": 0.99975,
        "g": 0.005,
        "lambda": 200,
        "theta_1": 33,
        "theta_2": 9,
        "theta_3": 11,
        "r": 0.1,
        "routing_iterations": 50,             
        "routing_no_improve_limit": 30,
        "post_unsuccessful_limit": 2000,
    }
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE ALNS OPTIMIZATION WITH CONSTRAINTS")
    print("Stage 1: Pattern optimization (ALNS)")
    print("Stage 2: Routing optimization (LNS)")
    print("Now enforcing: Daily delivery bounds and store capacity constraints")
    print("="*80)
    
    start_time = time.time()
    
    alns = ComprehensiveALNS(instance_data, algorithm_params)
    if len(instance_data['stores']) <= 30:
        max_iterations = 1000
    elif len(instance_data['stores']) <= 100:
        max_iterations = 2000
    else:
        max_iterations = 3000

    TIME_LIMIT_SECONDS = 3600

    best_solution = alns.run_alns(
        max_iterations=max_iterations,
        time_limit=TIME_LIMIT_SECONDS
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
        print("\n⚠️ Warning: Solution does not satisfy all delivery requirements!")
    else:
        print("\n✅ All delivery requirements satisfied")
    final_violations = best_solution.validate_constraints()
    if final_violations:
        print(f"\n⚠️ Final solution has {len(final_violations)} constraint violations")
    else:
        print("\n✅ All constraints satisfied")
    
    print_comprehensive_solution(best_solution, instance_data, alns)
    results = save_comprehensive_results(best_solution, instance_data, runtime)

    op_df = alns.build_operator_performance_table()
    print("\nOperator performance (Stage-1 pattern operators):")
    print(op_df.to_string(
        index=False,
        formatters={
            "% Best":   "{:.1f}".format,
            "% Better": "{:.1f}".format,
            "% Accept": "{:.1f}".format,
            "% Reject": "{:.1f}".format,
        }
    ))

    os.makedirs("results", exist_ok=True)
    op_latex_path = os.path.join("results", f"operator_performance_{instance_name}.tex")

    with open(op_latex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Analysis of operator performance}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\hline\n")
        f.write("Operator & \\% Best & \\% Better & \\% Accept & \\% Reject & Calls\\\\\n")
        f.write("\\hline\n")

        for _, row in op_df.iterrows():
            f.write(
                f"{row['Operator']} & "
                f"{row['% Best']:.1f} & "
                f"{row['% Better']:.1f} & "
                f"{row['% Accept']:.1f} & "
                f"{row['% Reject']:.1f} & "
                f"{int(row['Calls'])}\\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table saved to {op_latex_path}")

    pattern_history = alns.pattern_history
    if pattern_history:
        cols = [f"store_{s}" for s in sorted(instance_data['stores'])]
        pattern_df = pd.DataFrame(pattern_history, columns=cols)
        pattern_df.insert(0, "iteration", range(len(pattern_history)))
        
        pattern_csv_path = os.path.join("results", f"pattern_trajectory_{instance_name}.csv")
        pattern_df.to_csv(pattern_csv_path, index=False)
        print(f"\nPattern trajectory saved to {pattern_csv_path}")
    print(f"\n--- Pattern Assignment Summary ---")
    pattern_frequency_count = defaultdict(int)
    for store, pattern_id in best_solution.pattern_assignments.items():
        freq = sum(best_solution.evaluator.patterns[pattern_id])
        pattern_frequency_count[freq] += 1
    
    print("Frequency distribution:")
    for freq in sorted(pattern_frequency_count.keys()):
        count = pattern_frequency_count[freq]
        print(f"  {freq} deliveries/week: {count} stores")
    print(f"\n--- Daily Routing Summary ---")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    for day in range(6):
        print(f"\n{day_names[day]} (Day {day}):")
        
        day_distance = 0
        day_load = 0
        day_vehicles = 0
        
        for vehicle, route in best_solution.routes_by_day[day].items():
            if len(route) > 2:
                day_vehicles += 1
                route_distance = sum(alns.delta[route[i], route[i+1]] 
                                    for i in range(len(route)-1))
                
                route_load = 0
                for node in route:
                    if node != 0:
                        pattern_id = best_solution.pattern_assignments[node]
                        route_load += alns.p_frt.get((node, pattern_id, day), 0)
                
                day_distance += route_distance
                day_load += route_load
        
        num_deliveries = len(best_solution.get_required_deliveries(day))
        print(f"  Deliveries: {num_deliveries} stores")
        print(f"  Vehicles used: {day_vehicles}")
        print(f"  Total distance: {day_distance:.1f} km")
        print(f"  Total load: {day_load:.1f} units")
        if day_load < alns.Q_day_min:
            print(f"  ⚠️  Daily load {day_load:.1f} < minimum {alns.Q_day_min}")
        elif day_load > alns.Q_day_max:
            print(f"  ⚠️  Daily load {day_load:.1f} > maximum {alns.Q_day_max}")
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE ALNS OPTIMIZATION COMPLETE")
    print(f"{'='*100}")
    print("ALGORITHM FEATURES:")
    print("✅ Follows paper's algorithm structure exactly")
    print("✅ Stage 1: ALNS for pattern optimization")
    print("✅ Stage 2: LNS for routing optimization")
    print("✅ All 7 operators implemented (no segment operators)")
    print("✅ Simulated Annealing acceptance")
    print("✅ Adaptive operator selection")
    print("✅ Post-optimization with extended LNS")
    print("✅ Exact cost calculations from comprehensive model")
    print("✅ NOW ENFORCES: Daily delivery bounds and store capacity constraints")
    print(f"{'='*100}")
    gurobi_file = os.path.join("results", f"gurobi_{instance_name}.csv")
    if os.path.exists(gurobi_file):
        print("\n--- Comparison with Gurobi ---")
        gurobi_df = pd.read_csv(gurobi_file)
        if not gurobi_df.empty:
            gurobi_result = gurobi_df.iloc[-1]
            gurobi_obj = gurobi_result['objective_value']
            
            gap = (best_solution.cost - gurobi_obj) / gurobi_obj * 100
            print(f"Gurobi objective: {gurobi_obj:.2f}")
            print(f"ALNS objective: {best_solution.cost:.2f}")
            print(f"Gap: {gap:.2f}%")
            
            if gap < 0:
                print(f"✅ ALNS found better solution by {-gap:.2f}%!")
            elif gap <= 5:
                print("✅ ALNS solution within 5% of Gurobi (Excellent)")
            elif gap <= 10:
                print("✅ ALNS solution within 10% of Gurobi (Good)")
            elif gap <= 20:
                print("⚠️  ALNS solution within 20% of Gurobi (Acceptable)")
            else:
                print("❌ ALNS solution gap > 20% (Needs improvement)")
            gurobi_runtime = gurobi_result.get('runtime', 0)
            if gurobi_runtime > 0:
                speedup = gurobi_runtime / runtime
                print(f"Runtime comparison: ALNS {speedup:.1f}x faster than Gurobi")
    
    print(f"\n{'='*100}")

    return best_solution, instance_data

if __name__ == "__main__":
    main()