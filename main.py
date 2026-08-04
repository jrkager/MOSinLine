# ------------------
# Import codebase of RLRP and PATT
# ------------------
from patt.alns import ComprehensiveSolution
import patt.alns as alns4

import rlrp.classes as rlrp_classes
from rlrp.algorithm import ourAlgorithm as rlrp_main
import rlrp.applications.lrp.model as rlrp_model
import rlrp.applications.lrp.instance as rlrp_instance
from rlrp.applications.lrp.instance import PenalizedCost

from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Set, Union, Sequence, Mapping, TypeVar, TypedDict, TypeAlias
from dataclasses import dataclass, field

from statistics import mean
from math import hypot
import json
import argparse
from contextlib import redirect_stdout
import os

from copy import deepcopy

# ------------------
# Define data structures
# ------------------
Store: TypeAlias = int
Depot: TypeAlias = int
Node: TypeAlias = int


# ============================================================
# ALIGNED TRANSPORT PARAMETERS — single source of truth (PATT values)
# ============================================================
TRANSPORT_PARAMS = {
    "c_km": 1.12, "c_fuel": 1.80, "eta": 0.05, "theta_TR": 2.7,
    "W0": 14.4, "Q": 25.6, "lam": 0.3,
}

def derived_transport_coefficients(p=TRANSPORT_PARAMS):
    lam = p["lam"]
    return {
        "c_coef":     (1 - lam) * p["c_km"],
        "alpha_coef": (1 - lam) * p["c_fuel"] * p["eta"] * p["W0"] + lam * p["eta"] * p["theta_TR"] * p["W0"],
        "gamma_coef": (1 - lam) * p["c_fuel"] * p["eta"]           + lam * p["eta"] * p["theta_TR"],
        "marginal_co2_emissions": p["eta"] * p["theta_TR"],
    }

def _export_demand_by_segment(instance, scenario, depot_stores):
    """Per-store, per-segment weekday demand for the PATT instance JSON."""
    return {
        str(i + 1): {
            pc.name.lower(): [float(instance.demands[scenario].get((st, pc, w), 0.0)) for w in Weekday]
            for pc in ProductClass
        }
        for i, st in enumerate(depot_stores)
    }


@dataclass
class RLRPResult:
    # dict of scenario to dict of depot_id to size of warehouse opened at this depot_id. id a warehouse is not opened, it will not appear in the dict or its size will be 0.
    # for each scneario this is different and consists of f.s.-solution + second stage recovered size
    depot_sizes: Dict[int, Dict[Depot, float]]
    # association of customers to warehouses per scenario
    customer_depot_assignment: Dict[int, Dict[Depot, List[Store]]] # dict of scenario to dict of store_id to depot_id. if a store is not assigned to any warehouse in a scenario, it will not appear in the dict for this scenario or its assigned depot_id will be None or -1.

    def __init__(self, ret:rlrp_classes.LRPReturnObject, s:rlrp_classes.AlgorithmStats=None):
        self.depot_sizes = ret.depot_sizes
        self.customer_depot_assignment = ret.customer_depot_assignment

    def __hash__(self):
        return hash(str(self.depot_sizes)+"CONCAT"+str(self.customer_depot_assignment))

# delivery pattern for a store is a tuple of 6 ints (0 or 1) indicating delivery on each day of the week
class Pattern(tuple): pass

class ALNSInstanceData(Dict):
    # dict as passed to print_comprehensive_solution in alns.py, containing keys ['stores', 'store_id_mapping', 'loc', 'daily_demands', 'instance_name']
    pass

@dataclass
class Route:
    # id of the vehicle used for this route
    vehicle_id: int
    # list depot i - store id 1 - store id 2 - ... - depot i in the order they are visited (including depot at the start and end) depots have negative ids, stores have positive ids
    stops: List[Node]
    # list of arc lengths in the same order as stops, so arc_lengths[i] is the length of the arc from stops[i] to stops[i+1]
    arc_lengths: List[float]
    # list of delivery amounts in the same order as stops, so delivery_amounts[i] is the amount delivered to store stops[i].
    # delivery_amounts[0] describes the total weight carryed by the vehicle at the start of the route at the depot.
    # delivery_amounts[len(stops)-1] should be 0, as the vehicle should return empty to the depot. Otherwise, this describes the amount left undelivered at the end of the route.
    delivery_amounts: List[float]

class Weekday(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5

class ProductClass(Enum):
    DRY = 0
    FRESH = 1
    FROZEN = 2

@dataclass
class PATTResult:
    """
    one PATTResult object per scenario.
    todo: implement an option to check infeasibility. How does alns report that?
    """
    # delivery pttern per store
    patterns: Dict[Store, Pattern] = field(default_factory=dict)
    # routes per day
    routes: Dict[Weekday, List[Route]] = field(default_factory=dict)

    def append_solution(self, solution: ComprehensiveSolution, instance_data: ALNSInstanceData):
        # extract patterns and routes from the solution
        sm = instance_data['store_id_mapping']
        for i, pattern_id in solution.pattern_assignments.items():
            self.patterns[sm[i]] = Pattern(solution.evaluator.patterns[pattern_id])

        for day, vehicle_routes in solution.routes_by_day.items():
            if day not in self.routes:
                self.routes[day] = []
            for vehicle_id, route in vehicle_routes.items():
                if len(route) > 2:
                    delivery_amounts = [0] * len(route)
                    for i in range(1, len(route)-1):
                        internal_id = route[i]   # p_frt keys are INTERNAL renumbered ids
                        delivery_amounts[i] = solution.p_frt.get((internal_id, solution.pattern_assignments[internal_id], day), 0)
                    delivery_amounts[0] = sum(delivery_amounts) # total amount at the start of the route at the depot
                    delivery_amounts[-1] = 0 # amount left undelivered at the end of the route is 0, as the vehicle returns empty to the depot in the current implementation
                    self.routes[day].append(Route(
                        vehicle_id=vehicle_id,
                        stops=route,
                        arc_lengths=[solution.evaluator.delta[route[i], route[i+1]] for i in range(len(route)-1)],
                        delivery_amounts=delivery_amounts
                    ))

    def check_infeasible(self):
        """
        return true if this PATTResult is infeasible
        """
        pass

class SIMResult:
    """
    holds all data that is generated in the simulation and needed for the evaluation of the solution.
    based on this result, we decide if the solutuion was good enough.
    """
    def is_good_enough(self):
        pass

@dataclass
class Instance:
    # this class contains all the data and parameters for all algorithms. it is a mirror of a instance file which should include all these data and parameters as well.
    # for the run of a algorithm, we just create the according instance file or objects out of this instance temporary before we run the algorithm
    instance_name: str
    depots: List[Depot] # have negative index
    stores: List[Store] # have positive index
    locations: Dict[Node, Tuple[float, float]] # dict of location (node) id to coordinates. depots have negative ids, stores have positive ids

    S: List[int] # list of scenario ids
    demands: Dict[int, Dict[Tuple[Node, ProductClass, Weekday], float]] # dict of scenario to dict of location id, product class, weekday to demand. this is the nominal demand for each scenario. SIM generates its own N relaziations then

    # todo this value is not read in alns model yet
    pattern_operational_costs: Dict[Tuple[Store, Pattern], float] # dict of (store_id, pattern) to operational cost for using this pattern at this store. only needed for kailin, not available in johannes
    # todo this value is not read in alns model yet
    foodwaste_emissions_factor: Dict[Tuple[Store, Pattern], float] # dict of (store_id, pattern) to CO2 emissions for food waste for using this pattern at this store. only needed for kailin, not available in johannes

    distances: Dict[Tuple[Node, Node], float] # dict of (location id 1, location id 2) to distance of arc. contains both directions. c in johannes and c^TR/c_km in kailin
    cost_per_km: float # cost per km for transportation. c_km = 2.0 kailin
    vehicle_capacity: float # capacity of the delivery vehicles. Q in johannes and Q in kailin
    vehicle_empty_weight: float # weight of empty vehicle. W0 in kailin

    # fixed_co2_emissions = marginal * curb_weight
    marginal_co2_emissions: float # dict of (location id 1, location id 2) to marginal CO2 emissions per unit of demand and per distance. gamma/distance in johannes and e^P1?? in kailin. eta*beta*theta_TR in original kailin code

    fixed_warehouse_costs: Dict[Depot, float] # dict of depot_id to fixed cost for opening warehouse depot_id at the position of WH candidate depot_id. e in johannes (f in code) and not available in kailin
    marginal_warehouse_costs: Dict[Depot, float] # dict of depot_id to marginal cost per unit of size for opening warehouse depot_id at the position of WH candidate depot_id. d in johannes (d in code) and not available in kailin
    max_warehouse_size: Dict[Depot, float] # dict of depot_id to maximum warehouse size at the position of WH candidate depot_id. A in johannes and not available in kailin

    second_stage_penalty_factor: float # penalty factor for warehouse costs in the second stage. 1.5 in the old instances of johannes

    weighting_factor_patt: float # factor for weighting the two objectives in the combined objective function in kailins model (lambda).
    weighting_factor_rlrp: float # factor for weighting the two objectives in the combined objective function in johannes model (was 0.5).

    number_of_realizations: int # N in SIM


    def aggregate_demands_patt(self):
        """
        sum over product classes
        """
        ret = {}
        for s in self.demands: # scenario s
            nodes = set(n for (n,p,w) in self.demands[s])
            product_classes = set(p for (n,p,w) in self.demands[s])
            # weekdays = set(w for (n,p,w) in self.demands[s])
            ret[s] = {(n, w) : sum(self.demands[s][n,p,w] for p in product_classes) for n in nodes for w in Weekday}
        return ret

    def aggregate_demands_rlrp(self, option=1):
        """
        three methods:
        1. avg over days
        2. max of the days
        3. n-th value in sort.desc(beta for w in days)
        """
        ret = {}
        agg_patt = self.aggregate_demands_patt()
        for s in agg_patt:
            nodes = set(n for (n, w) in agg_patt[s])
            weekdays = set(w for (n, w) in agg_patt[s])
            if option == 1:
                ret[s] = {n: sum(agg_patt[s][n, w] for w in weekdays) / len(weekdays) for n in nodes}
            if option == 2:
                ret[s] = {n: max(agg_patt[s][n, w] for w in weekdays) for n in nodes}
            if option == 3:
                n_th = 1
                ret[s] = {n: sorted((agg_patt[s][n, w] for w in weekdays), reverse=True)[n_th] for n in nodes}
        return ret

def create_rlrp_instance_data(inst: Instance, gap, timelimit, logfile_name=None) -> Tuple[rlrp_classes.AlgorithmParams, rlrp_instance.Instance]:
    rlrpinstance = rlrp_instance.Instance(I=inst.depots,
                                          J=inst.stores,
                                          beta_k_j=inst.aggregate_demands_rlrp(option = 1),
                                          f_i={k : PenalizedCost(v, inst.second_stage_penalty_factor * v) for k,v in inst.fixed_warehouse_costs.items()},
                                          d_i={k : PenalizedCost(v, inst.second_stage_penalty_factor * v) for k,v in inst.marginal_warehouse_costs.items()},
                                          c_ij={k : derived_transport_coefficients()["c_coef"] * v for k,v in inst.distances.items()},
                                          alpha_ij={k : derived_transport_coefficients()["alpha_coef"] * v for k,v in inst.distances.items()},
                                          gamma_ij={k : derived_transport_coefficients()["gamma_coef"] * v for k,v in inst.distances.items()},
                                          F=0,
                                          C_i=inst.max_warehouse_size,
                                          Q=inst.vehicle_capacity,
                                          loc=inst.locations, # in the createInstance method loc is created as numpy array, not possible here because of nefgative indizes for the depots
                                          name=inst.instance_name,
                                          sample_number=None,
                                          scenarios=list(inst.demands.keys()),
                                          weighting_factor_rlrp=inst.weighting_factor_rlrp)
    appl = rlrp_classes.Application(inst = rlrpinstance, MasterModel=rlrp_model.MasterModel, SecondStageModel=rlrp_model.SecondStageModel)
    start_sc = rlrp_model.initSubsetEmpty(inst)
    params = rlrp_classes.AlgorithmParams(app=appl,
                                          start_sc=start_sc,
                                          desired_gap=gap,
                                          MASTER_P=gap * 0.5, # use mu = 0.5
                                          HEURTIMELIMIT=0.1,
                                          total_timelimit=timelimit,
                                          n_threads=0) # use all threads available
    if logfile_name:
        params.logfile = open(logfile_name, "w")
    return params

def create_patt_instance_data(instance: Instance, RLRP_result: RLRPResult, depot_id: Node, scenario: int) -> str:
    """
    depot_id is negative!
    """
    depot_stores = [st for st in instance.stores if st in RLRP_result.customer_depot_assignment[scenario][depot_id]]
    if len(depot_stores) == 0:
        return None
    filename=f"temp-{instance.instance_name}-{depot_id}-{scenario}-{hash(RLRP_result)}.json"
    json_dict = {"instance_name": f"{instance.instance_name}-{depot_id}-{scenario}",
                 "depot": {"x": instance.locations[depot_id][0], "y": instance.locations[depot_id][1], "demand": 0},
                 "stores": list(range(1,len(depot_stores)+1)),
                 "id_map": {str(i+1):st for i, st in enumerate(depot_stores)},
                 "loc": {str(i+1):{"x":instance.locations[st][0],"y":instance.locations[st][1]} for i, st in enumerate(depot_stores)},
                 "daily_demands": [mean(instance.aggregate_demands_patt()[scenario][n, w] for w in Weekday) for n in depot_stores]
                 }
    # add other coefficients from instance
    mapper = {store_id : i for i, store_id in json_dict["id_map"].items()}
    mapper[0] = 0
    def reduce_depots(costs_dict):
        ret = {}
        for (n1, n2), c in instance.distances.items():
            if n1 < 0 and n1 != depot_id or n2 < 0 and n2 != depot_id:
                continue
            if n1 == depot_id:
                n1 = 0
            if n2 == depot_id:
                n2 = 0
            if n1 not in mapper or n2 not in mapper: continue
            ret[f"({mapper[n1]},{mapper[n2]})"] = c
        return ret
    json_dict["distances"] = reduce_depots(instance.distances)
    json_dict["vehicle_capacity"] = instance.vehicle_capacity
    json_dict["vehicle_empty_weight"] = instance.vehicle_empty_weight
    json_dict["cost_per_km"] = TRANSPORT_PARAMS["c_km"]      # UNWEIGHTED — PATT applies lambda itself
    json_dict["fuel_price"] = TRANSPORT_PARAMS["c_fuel"]
    json_dict["eta"] = TRANSPORT_PARAMS["eta"]
    json_dict["marginal_co2_emissions"] = instance.marginal_co2_emissions
    json_dict["weighting_factor_patt"] = instance.weighting_factor_patt
    json_dict["Q_day_max"] = RLRP_result.depot_sizes[scenario].get(depot_id, 99999)
    json_dict["demand_by_segment"] = _export_demand_by_segment(instance, scenario, depot_stores)

    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=2)
    return filename

def delete_patt_instance_file(patt_instance_file_name):
    if patt_instance_file_name and os.path.exists(patt_instance_file_name):
        os.remove(patt_instance_file_name)

# if __name__ == "main":
def construct_test_instance() -> Instance:
    depots: List[Depot] = [-1, -2, -3]
    stores: List[Store] = [12, 77, 49, 9, 1]

    # locations: include 3 depots + 5 stores
    locations: Dict[Node, Tuple[float, float]] = {}

    locations[-1] = (35.0, 35.0)
    locations[-2] = (45.0, 15.0)
    locations[-3] = (2.0, 23.0)
    locations[12] = (50.0, 35.0)
    locations[77] = (53.0, 43.0)
    locations[49] = (6.0, 68.0)
    locations[9] = (55.0, 60.0)
    locations[1] = (41.0, 49.)

    # Structural demand scenarios (base / growth / fresh_shift), calibrated to Q=25.6
    from scenario_gen import generate_scenarios, check_scenarios_against_vehicle
    demands, scenario_names, base_profile = generate_scenarios(stores, ProductClass, Weekday, seed=42)
    check_scenarios_against_vehicle(demands, scenario_names, stores, ProductClass, Weekday,
                                    Q=TRANSPORT_PARAMS["Q"])

    S = list(demands.keys())

    # distances: euclidean between all nodes, both directions (includes (i,i)=0 too)
    nodes: List[Node] = depots + stores
    distances: Dict[Tuple[Node, Node], float] = {}
    for i in nodes:
        xi, yi = locations[i]
        for j in nodes:
            xj, yj = locations[j]
            distances[(i, j)] = hypot(xi - xj, yi - yj)

    pattern_operational_costs = None
    foodwaste_emissions_factor = None

    # warehouse cost params (choose simple, consistent values)
    _oml = 1.0 - TRANSPORT_PARAMS["lam"]   # warehouse costs are economic -> pre-scale by (1-lambda)
    fixed_warehouse_costs = {d: _oml * 8_000.0 for d in depots}     # weekly-equivalent fixed cost
    marginal_warehouse_costs = {d: _oml * 50.0 for d in depots}
    max_warehouse_size = {-1: 30.0, -2: 30.0, -3: 30.0}

    # fill remaining scalars (use your provided ones + sensible defaults)
    inst = Instance(
        instance_name="R101_5stores_s3",
        depots=depots,
        stores=stores,
        locations=locations,
        demands=demands,
        S=S,

        pattern_operational_costs=pattern_operational_costs,
        foodwaste_emissions_factor=foodwaste_emissions_factor,

        distances=distances,
        cost_per_km=TRANSPORT_PARAMS["c_km"],
        vehicle_capacity=TRANSPORT_PARAMS["Q"],
        vehicle_empty_weight=TRANSPORT_PARAMS["W0"],
        marginal_co2_emissions=derived_transport_coefficients()["marginal_co2_emissions"],

        fixed_warehouse_costs=fixed_warehouse_costs,
        marginal_warehouse_costs=marginal_warehouse_costs,
        max_warehouse_size=max_warehouse_size,
        second_stage_penalty_factor=1.5,

        weighting_factor_patt=TRANSPORT_PARAMS["lam"],
        weighting_factor_rlrp=0.5,

        number_of_realizations=10,
    )

    return inst

def solve_rlrp(inst: Instance, logfile_rlrp: str) -> RLRPResult:
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          "+    Solving RLRP Instance with Gap 5% and timelimit 1800s   +\n"
          "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"(Logging to {logfile_rlrp})")
    try:
        _, ret = rlrp_main(params=create_rlrp_instance_data(inst, gap=0.05, timelimit=1800, logfile_name=logfile_rlrp))
    except rlrp_classes.TimeoutException as e:
        raise TimeoutError(f"RLRP algorithm timed out. Reached gap: {e.reached_gap * 100:.2f}%")
    except Exception as e:
        print("Error in RLRP")
        raise e
    rlrp_result = RLRPResult(ret)
    return rlrp_result

def solve_patt(inst: Instance, logfile_patt: str, logfile_rlrp: str, rlrp_result: RLRPResult) -> Dict[int, PATTResult]:
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          "+       Solving PATT (ALNS) for each scenario and depot      +\n"
          "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"(Logging to {logfile_patt})")
    patt_result_per_scenario = {}
    for s in inst.S:
        print(f"Scenario {s}:")
        patt_result_per_scenario[s] = PATTResult()
        for depot_id in inst.depots:
            print(f"  Depot {depot_id}")
            patt_instance_file_name = create_patt_instance_data(inst, rlrp_result, depot_id=depot_id, scenario=s)
            if patt_instance_file_name is None:
                continue
            alns_solution: ComprehensiveSolution
            alns_instance_data: ALNSInstanceData
            with open(logfile_rlrp, 'w') as f:
                with redirect_stdout(f):
                    alns_solution, alns_instance_data = alns4.main(instance_file_name=patt_instance_file_name)
            patt_result_per_scenario[s].append_solution(alns_solution, alns_instance_data)
            delete_patt_instance_file(patt_instance_file_name)

    print("Patt Result:")
    for s in inst.S:
        if s in patt_result_per_scenario:
            print(f"Scenario {s}:")
            print(f"{len(patt_result_per_scenario[s].patterns)} patterns, "
                  f"{sum(len(r) for r in patt_result_per_scenario[s].routes.values())} routes")
    return patt_result_per_scenario

def solve_sim(inst: Instance, patt_result_per_scenario: Dict[int, PATTResult], logfile_sim: str) -> SIMResult:
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          "+                        Solving SIM                         +\n"
          "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"(Logging to {logfile_sim})")
    # using the PATT result to run simulation
    # ...
    # filling SIMResult object with data
    sim_result = SIMResult()
    # ...
    return sim_result

def modify_instance_after_patt_infeasible(inst:Instance, rlrp_result: RLRPResult, patt_result_per_scenario, patt_failures):
    # find out in which scneario and which depot PATT failed. (todo)
    # possibly also distinguished per product class and per weekday
    failed_depots_per_scenario = ...
    for sc, failed_depots in failed_depots_per_scenario.items():
        for key in inst.demands[sc]:
            inst.demands[sc][key] *= 1.1 # increase demand by 10%

def modify_weights_after_sim_infeasible(patt_instance, patt_result_per_scenario, sim_result, sim_failures):
    patt_instance.weighting_factor_patt *= 0.9 # modify the weighting. Todo: this has to be changed (increased or decreased) depending on the result of the SIM!


def main():
    """
    Depots have negative ids (when calling ALNS, we cycle through every depot and use 0 as the id)
    Stores have positive ids.
    """

    print("+++++++++++++++++++++++++++++++++\n"
          "+ Constructing test instance... +\n"
          "+++++++++++++++++++++++++++++++++")
    original_inst = construct_test_instance()

    if not os.path.exists("logs"):
        os.makedirs("logs")
    logfile_rlrp = "logs/rlrplog"+original_inst.instance_name+".txt"
    logfile_patt = "logs/pattlog"+original_inst.instance_name+".txt"
    logfile_sim = "logs/simlog" + original_inst.instance_name + ".txt"

    max_patt_failures = 3
    max_sim_failures = 3
    patt_failures = 0

    rlrp_instance = deepcopy(original_inst)
    patt_instance = deepcopy(original_inst)

    while patt_failures < max_patt_failures:
        rlrp_result = solve_rlrp(rlrp_instance, logfile_rlrp)

        sim_failures = 0

        while sim_failures < max_sim_failures:
            patt_result_per_scenario = solve_patt(patt_instance, logfile_patt, logfile_rlrp, rlrp_result)

            if any(pr.check_infeasible() for pr in patt_result_per_scenario.values()):
                patt_failures += 1
                if patt_failures >= max_patt_failures:
                    return {
                        "status": "infeasible",
                        "reason": f"PATT infeasible {patt_failures} times"
                    }

                modify_instance_after_patt_infeasible(rlrp_instance, rlrp_result, patt_result_per_scenario, patt_failures)
                break  # go back to outer loop: rerun RLRP

            sim_result = solve_sim(original_inst, patt_result_per_scenario, logfile_sim)

            if sim_result.is_good_enough():
                return {
                    "status": "ok",
                    "rlrp_result": rlrp_result,
                    "patt_result": patt_result_per_scenario,
                    "sim_result": sim_result,
                }

            sim_failures += 1
            if sim_failures >= max_sim_failures:
                return {
                    "status": "infeasible",
                    "reason": f"SIM not good {sim_failures} times"
                }

            modify_weights_after_sim_infeasible(patt_instance, patt_result_per_scenario, sim_result, sim_failures)
            # continue inner loop: rerun PATT only

    return {
        "status": "infeasible",
        "reason": "Maximum number of PATT failures reached"
    }


if __name__ == "__main__":
    main()
