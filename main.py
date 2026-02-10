# ------------------
# Import codebase of RLRP and PATT
# ------------------
from notebook.auth import passwd

from patt.alns4 import ComprehensiveSolution
import patt.alns4 as alns4

import rlrp.classes as rlrp_classes
from rlrp.algorithm import ourAlgorithm as rlrp_main
import rlrp.applications.lrp.model as rlrp_model
import rlrp.applications.lrp.instance as rlrp_instance

from statistics import mean

import json

# Other imports
import argparse
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Set, Union, Sequence, Mapping, TypeVar, TypedDict, TypeAlias
from dataclasses import dataclass, field

# ------------------
# Define data structures
# ------------------
Store: TypeAlias = int
Depot: TypeAlias = int
Node: TypeAlias = int


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
        return hash((frozenset((s, frozenset(d.items())) for s, d in self.depot_sizes.items()),
                     frozenset((s, frozenset(d.items())) for s, d in self.customer_depot_assignment.items())))

# delivery pattern for a store is a tuple of 6 ints (0 or 1) indicating delivery on each day of the week
class Pattern(tuple): pass

class ALNSInstanceData(Dict):
    # dict as passed to print_comprehensive_solution in alns4.py, containing keys ['stores', 'store_id_mapping', 'loc', 'daily_demands', 'instance_name']
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
    one PATTResult object per scenario?
    """
    # delivery pttern per store
    patterns: Dict[Store, Pattern] = field(default_factory=dict)
    # routes per day
    routes: Dict[Weekday, Route] = field(default_factory=dict)

    def __init__(self, solution: ComprehensiveSolution, instance_data: ALNSInstanceData):
        # extract patterns and routes from the solution
        sm = instance_data['store_id_mapping']
        for i, pattern_id in solution.pattern_assignments.items():
            self.patterns[sm[i]] = Pattern(solution.evaluator.patterns[pattern_id])

        for day, vehicle_routes in solution.routes_by_day.items():
            for vehicle_id, route in vehicle_routes.items():
                if len(route) > 2:
                    delivery_amounts = [0] * len(route)
                    for i in range(1, len(route)-1):
                        store_id = sm[route[i]]
                        delivery_amounts[i] = solution.p_frt.get((store_id, solution.pattern_assignment[store_id], day), 0)
                    delivery_amounts[0] = sum(delivery_amounts) # total amount at the start of the route at the depot
                    delivery_amounts[-1] = 0 # amount left undelivered at the end of the route is 0, as the vehicle returns empty to the depot in the current implementation
                    self.routes[day] = Route(
                        vehicle_id=vehicle_id,
                        stops=route,
                        arc_lengths=[solution.evaluator.delta[route[i], route[i+1]] for i in range(len(route)-1)],
                        delivery_amounts=delivery_amounts
                    )


@dataclass
class Instance:
    # this class contains all the data and parameters for all algorithms. it is a mirror of a instance file which should include all these data and parameters as well.
    # for the run of a algorithm, we just create the according instance file or objects out of this instance temporary before we run the algorithm
    instance_name: str
    depots: List[Depot] # have negative index
    stores: List[Store] # have positive index
    locations: Dict[Node, Tuple[float, float]] # dict of location (node) id to coordinates. depots have negative ids, stores have positive ids
    demands: Dict[int, Dict[Tuple[Node, ProductClass, Weekday], float]] # dict of scenario to dict of location id, product class, weekday to demand. this is the nominal demand for each scenario. SIM generates its own N relaziations then
    transportation_costs: Dict[Tuple[Node, Node], float] # dict of (location id 1, location id 2) to transportation cost per unit of demand. contains both directions. c in johannes and c^TR in kailin
    fixed_co2_emissions: Dict[Tuple[Node,Node], float] # dict of (location id 1, location id 2) to fixed CO2 emissions for empty vehicle visiting travelling this arc. alpha in johannes and e^P0 in kailin
    marginal_co2_emissions: Dict[Tuple[Node,Node], float] # dict of (location id 1, location id 2) to marginal CO2 emissions per unit of demand for travelling this arc. gamma in johannes and e^P1 in kailin
    fixed_warehouse_costs: Dict[Depot, float] # dict of depot_id to fixed cost for opening warehouse depot_id at the position of WH candidate depot_id. e in johannes (f in code) and not available in kailin
    marginal_warehouse_costs: Dict[Depot, float] # dict of depot_id to marginal cost per unit of size for opening warehouse depot_id at the position of WH candidate depot_id. d in johannes (d in code) and not available in kailin
    max_warehouse_size: Dict[Depot, float] # dict of depot_id to maximum warehouse size at the position of WH candidate depot_id. A in johannes and not available in kailin
    second_stage_penalty_factor: float # penalty factor for warehouse costs in the second stage. 1.5 in the old instances of johannes
    vehicle_capacity: float # capacity of the delivery vehicles. Q in johannes and not available in kailin
    pattern_operational_costs: Dict[Tuple[Store, Pattern], float] # dict of (store_id, pattern) to operational cost for using this pattern at this store. only needed for kailin, not available in johannes
    foodwaste_emissions: Dict[Tuple[Store, Pattern], float] # dict of (store_id, pattern) to CO2 emissions for food waste for using this pattern at this store. only needed for kailin, not available in johannes
    weighting_factor_patt: float # factor for weighting the two objectives in the combined objective function in kailins model (lambda).
    weighting_factor_rlrp: float # factor for weighting the two objectives in the combined objective function in johannes model (was 0.5).
    number_of_realizations: int # N in SIM

    # todo kailin code has to read certain costs and parameters from this instance class. the instance files will be created as it is originally in kailins code

    # todo johannes: the objects needeed for johannes algorithm will be created by a function of this class, and the can be passed to johannes algorithm. no files needed

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


def create_patt_instance_data(instance: Instance, RLRP_result: RLRPResult, depot_id: Node, scenario: int) -> str:
    """
    depot_id is negative!
    """
    filename=f"temp-{instance.name}-{depot_id}-{scenario}-{hash(RLRP_result)}.json"
    json_dict = {"instance_name": instance.instance_name,
                 "depot": {"x": instance.locations[depot_id][0], "y": instance.locations[depot_id][1], "demand": 0},
                 "stores": list(range(1,len(instance.stores)+1)),
                 "id_map": {str(i+1):st for i, st in enumerate(instance.stores)},
                 "loc": {str(i+1):{"x":instance.locations[st][0],"y":instance.locations[st][1]} for i, st in enumerate(instance.stores)},
                 "daily_demands": [mean(instance.aggregate_demands_patt()[scenario][n, w] for w in Weekday) for n in instance.stores]
                 }
    # add other coefficients from instance
    def reduce_depots(costs_dict):
        ret = {}
        for (n1, n2), c in instance.transportation_costs:
            if n1 < 0 and n1 != depot_id or n2 < 0 and n2 != depot_id:
                continue
            if n1 == depot_id:
                n1 = 0
            if n2 == depot_id:
                n2 = 0
            ret[(n1,n2)] = c
        return ret
    json_dict["transportation_costs"] = reduce_depots(instance.transportation_costs)
    json_dict["fixed_co2_emissions"] = reduce_depots(instance.fixed_co2_emissions)
    json_dict["marginal_co2_emissions"] = reduce_depots(instance.marginal_co2_emissions)
    json_dict["vehicle_capacity"] = instance.vehicle_capacity
    json_dict["pattern_operational_costs"] = instance.pattern_operational_costs
    json_dict["foodwaste_emissions"] = instance.foodwaste_emissions
    json_dict["weighting_factor_patt"] = instance.weighting_factor_patt

    with open(filename, 'w') as f:
        json.dump(json_dict, f)
    return filename

def create_rlrp_instance_data(inst: Instance, gap, timelimit) -> (rlrp_classes.AlgorithmParams, rlrp_instance.Instance):
    c_ij = inst.fixed_warehouse_costs

    rlrpinstance = rlrp_instance.Instance(I=inst.depots,
                                          J=inst.stores,
                                          beta_k_j=inst.aggregate_demands_rlrp(option = 1),
                                          f_i=inst.fixed_warehouse_costs,
                                          d_i=inst.marginal_warehouse_costs,
                                          c_ij=c_ij,
                                          alpha_ij=inst.fixed_co2_emissions,
                                          gamma_ij=inst.marginal_co2_emissions,
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
    return params


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    instance = Instance()

    try:
        _, ret = rlrp_main(params=create_rlrp_instance_data(instance, gap = 0.05, timelimit = 1800))
    except rlrp_classes.TimeoutException as e:
        raise TimeoutError(f"RLRP algorithm timed out. Reached gap: {e.reached_gap*100:.2f}%")

    rlrp_result = RLRPResult(ret)

    patt_instance_file_name = create_patt_instance_data(instance, rlrp_result, depot_id = -1)
    alns_solution: ComprehensiveSolution
    alns_instance_data: ALNSInstanceData
    alns_solution, alns_instance_data = alns4.main(instance_file_name=patt_instance_file_name)
    patt_result = PATTResult(alns_solution, alns_instance_data)