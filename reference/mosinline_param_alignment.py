# ============================================================================
# MOSinLine main.py — PARAMETER ALIGNMENT PATCH
# Aligns ALL shared physical parameters to the PATT (Kailin) values and makes
# the RLRP transport objective EXACTLY equal to the PATT arc-cost scalarization
#   (1-lambda) * economic  +  lambda * emissions
# without touching rlrp/applications/lrp/model.py.
#
# Trick: the RLRP objective sums  c_ij*x + alpha_ij*x + gamma_ij*t  raw, so we
# fold the (1-lambda)/lambda weights INTO the coefficients. Warehouse costs are
# economic, so they get a (1-lambda) pre-scale, and weighting_factor_rlrp is
# set to 0.5 (uniform 0.5 scaling of the whole objective — argmin unchanged).
# Result: RLRP minimizes 0.5 * [ (1-lam)*(EUR) + lam*(kg CO2) ], i.e. a
# positive scalar multiple of the same scalarization the PATT uses.
# ============================================================================

# ----------------------------------------------------------------------------
# 1. SINGLE SOURCE OF TRUTH — put at module level in main.py.
#    European 40t tractor-trailer configuration (PATT values).
# ----------------------------------------------------------------------------

TRANSPORT_PARAMS = {
    "c_km":     1.12,   # EUR/km        distance cost (driver, wear, tolls)
    "c_fuel":   1.80,   # EUR/L         German diesel price
    "eta":      0.05,   # L/(t*km)      fuel consumption coefficient
    "theta_TR": 2.7,    # kg CO2/L      diesel combustion emission factor
    "W0":       14.4,   # t             empty vehicle weight (tractor+trailer)
    "Q":        25.6,   # t             payload (40t GVW limit - W0)
    "lam":      0.3,    # -             lambda, SAME value in RLRP and PATT
}

def derived_transport_coefficients(p=TRANSPORT_PARAMS):
    """Arc-cost building blocks (per km / per t*km), and the lambda-folded
    RLRP coefficients. PATT arc cost on arc (i,j) with load q:
        (1-lam)*[c_km*d + c_fuel*eta*(W0+q)*d] + lam*[eta*theta_TR*(W0+q)*d]
      = [ (1-lam)*c_km                                    ]*d      -> c_ij
      + [ (1-lam)*c_fuel*eta*W0 + lam*eta*theta_TR*W0     ]*d      -> alpha_ij
      + [ (1-lam)*c_fuel*eta    + lam*eta*theta_TR        ]*d*q    -> gamma_ij
    so RLRP (c_ij*x + alpha_ij*x + gamma_ij*t) == PATT arc cost, bit for bit.
    """
    lam = p["lam"]
    econ_empty_per_km = p["c_fuel"] * p["eta"] * p["W0"]   # 1.296  EUR/km
    econ_per_tkm      = p["c_fuel"] * p["eta"]             # 0.09   EUR/(t*km)
    em_empty_per_km   = p["eta"] * p["theta_TR"] * p["W0"] # 1.944  kg/km
    em_per_tkm        = p["eta"] * p["theta_TR"]           # 0.135  kg/(t*km)
    return {
        "c_coef":     (1 - lam) * p["c_km"],                              # 0.784
        "alpha_coef": (1 - lam) * econ_empty_per_km + lam * em_empty_per_km,  # 1.4904
        "gamma_coef": (1 - lam) * econ_per_tkm      + lam * em_per_tkm,       # 0.1035
        "marginal_co2_emissions": em_per_tkm,   # 0.135 — exported for PATT (theta_TR = this / eta)
    }


# ----------------------------------------------------------------------------
# 2. RLRP COEFFICIENT CONSTRUCTION — replace the current lines
#    (approx. 207-209 in main.py):
#
#    BEFORE:
#      c_ij     = {k: v * inst.cost_per_km for k, v in inst.distances.items()},
#      alpha_ij = {k: inst.marginal_co2_emissions * v * inst.vehicle_empty_weight ...},
#      gamma_ij = {k: inst.marginal_co2_emissions * v ...},
#
#    AFTER:
# ----------------------------------------------------------------------------

#      _dc = derived_transport_coefficients()
#      c_ij     = {k: _dc["c_coef"]     * v for k, v in inst.distances.items()},
#      alpha_ij = {k: _dc["alpha_coef"] * v for k, v in inst.distances.items()},
#      gamma_ij = {k: _dc["gamma_coef"] * v for k, v in inst.distances.items()},

# Warehouse costs are economic -> pre-scale by (1-lam) where the Instance is
# built (fixed_warehouse_costs, marginal_warehouse_costs). The second-stage
# penalty (second_stage_penalty_factor * marginal costs) inherits the scaling
# automatically. Then set weighting_factor_rlrp = 0.5 (pure uniform scaling).


# ----------------------------------------------------------------------------
# 3. INSTANCE CONSTRUCTION — replace the hardcoded values (approx. lines
#    337-348 in main.py):
#
#    BEFORE:                          AFTER:
#      cost_per_km=2.0,                cost_per_km=TRANSPORT_PARAMS["c_km"],        # 1.12
#      vehicle_capacity=44,            vehicle_capacity=TRANSPORT_PARAMS["Q"],      # 25.6
#      vehicle_empty_weight=8,         vehicle_empty_weight=TRANSPORT_PARAMS["W0"], # 14.4
#      marginal_co2_emissions=         marginal_co2_emissions=
#        0.05 * 1 * 2.7,                 derived_transport_coefficients()["marginal_co2_emissions"],  # 0.135
#      ...
#      weighting_factor_patt=0.3,      weighting_factor_patt=TRANSPORT_PARAMS["lam"],
#      weighting_factor_rlrp=0.5,      weighting_factor_rlrp=0.5,   # uniform scaling, see above
#
#    plus (1-lam) pre-scale on fixed_warehouse_costs / marginal_warehouse_costs:
#      _oml = 1.0 - TRANSPORT_PARAMS["lam"]
#      fixed_warehouse_costs    = {k: _oml * v for k, v in fixed_warehouse_costs.items()}
#      marginal_warehouse_costs = {k: _oml * v for k, v in marginal_warehouse_costs.items()}
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# 4. PATT INSTANCE EXPORT — in create_patt_instance_data, the current export
#    writes marginal_co2_emissions / vehicle_capacity / weighting_factor_patt
#    but MISSES four parameters (PATT would silently use its own defaults) and
#    still writes two obsolete keys. Replace the coefficient block with:
# ----------------------------------------------------------------------------

#     p = TRANSPORT_PARAMS
#     json_dict["distances"] = reduce_depots(instance.distances)
#     json_dict["vehicle_capacity"]       = instance.vehicle_capacity      # 25.6
#     json_dict["vehicle_empty_weight"]   = instance.vehicle_empty_weight  # 14.4
#     json_dict["cost_per_km"]            = p["c_km"]                      # 1.12  (UNWEIGHTED!)
#     json_dict["fuel_price"]             = p["c_fuel"]                    # 1.80
#     json_dict["eta"]                    = p["eta"]                       # 0.05
#     json_dict["marginal_co2_emissions"] = instance.marginal_co2_emissions  # 0.135 -> PATT theta_TR=2.7
#     json_dict["weighting_factor_patt"]  = instance.weighting_factor_patt   # 0.3
#     json_dict["demand_by_segment"]      = _export_demand_by_segment(instance, scenario, depot_stores)
#     # REMOVED: pattern_operational_costs, foodwaste_emissions (obsolete —
#     # the new PATT computes pattern costs and waste endogenously)
#
# IMPORTANT: the PATT applies lambda itself, so the exported cost_per_km /
# fuel_price are the RAW (unweighted) values. Only the RLRP coefficients in
# section 2 carry the folded lambda.


# ----------------------------------------------------------------------------
# 5. CONSISTENCY CHECK — run once after wiring, catches unit errors early.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    p = TRANSPORT_PARAMS
    dc = derived_transport_coefficients(p)
    lam = p["lam"]

    # exact-match test on one arbitrary arc
    d, q = 17.3, 9.4   # km, t

    # PATT arc cost (as computed in CombinedEvaluator.calculate_route_cost)
    fuel = p["eta"] * (p["W0"] + q) * d
    patt = ((1 - lam) * (p["c_km"] * d + p["c_fuel"] * fuel)
            + lam * (p["theta_TR"] * fuel))

    # RLRP arc cost with folded coefficients
    rlrp = dc["c_coef"] * d + dc["alpha_coef"] * d + dc["gamma_coef"] * d * q

    assert abs(patt - rlrp) < 1e-9, (patt, rlrp)
    assert abs(dc["marginal_co2_emissions"] - p["eta"] * p["theta_TR"]) < 1e-12
    assert abs(dc["marginal_co2_emissions"] / p["eta"] - p["theta_TR"]) < 1e-12

    print("Alignment OK")
    print(f"  c_coef     = {dc['c_coef']:.4f}  EUR/km   (weighted)")
    print(f"  alpha_coef = {dc['alpha_coef']:.4f}  per km   (weighted, empty vehicle)")
    print(f"  gamma_coef = {dc['gamma_coef']:.4f}  per t*km (weighted, load)")
    print(f"  marginal_co2_emissions = {dc['marginal_co2_emissions']:.3f} kg/(t*km)")
    print(f"  PATT arc cost = RLRP arc cost = {patt:.6f} on test arc (d={d}, q={q})")
