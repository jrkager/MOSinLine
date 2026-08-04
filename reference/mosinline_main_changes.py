# ============================================================================
# MOSinLine main.py — companion changes for the segment-aware PATT module
# (alns_patt_segments.py, drop-in replacement for patt/alns.py)
#
# Two changes:
#   1. create_patt_instance_data: export PER-SEGMENT weekday demand instead of
#      collapsing ProductClass via aggregate_demands_patt. Segment keys must be
#      the lowercase enum names: "dry", "fresh", "frozen".
#   2. PATTResult.append_solution: BUG FIX — p_frt is keyed by INTERNAL store
#      ids (1..n after renumbering), but the current code looks it up with the
#      MAPPED original ids (sm[route[i]]). With an id_map present this makes
#      every delivery_amount 0. Look up with route[i], map only for output.
#
# Also note (no code): the JSON keys pattern_operational_costs and
# foodwaste_emissions written for the old alns.py are obsolete — the new PATT
# computes pattern costs and waste endogenously and simply ignores them.
# Requirements: add scikit-learn and scipy to kailin_pvrp_algorithm deps.
# ============================================================================


# ----------------------------------------------------------------------------
# CHANGE 1 — in create_patt_instance_data(...), where the per-depot instance
# JSON dict is assembled (after depot_stores / id_map are built), ADD:
# ----------------------------------------------------------------------------

def _export_demand_by_segment(instance, scenario, depot_stores):
    """Per-store, per-segment weekday demand for the PATT instance JSON.

    Keys must match the PATT segment config: lowercase ProductClass names
    ("dry", "fresh", "frozen"). Stores are keyed by their INTERNAL renumbered
    id (str(i+1)), consistent with 'stores' and 'loc' in the PATT JSON.
    Weekday order must match the PATT day index 0..5 (Mon..Sat).
    """
    return {
        str(i + 1): {
            pc.name.lower(): [
                float(instance.demands[scenario].get((st, pc, w), 0.0))
                for w in Weekday
            ]
            for pc in ProductClass
        }
        for i, st in enumerate(depot_stores)
    }

# ...inside create_patt_instance_data, next to the existing json_dict entries:
#
#     json_dict["demand_by_segment"] = _export_demand_by_segment(
#         instance, scenario, depot_stores)
#
# Keep "daily_demands" as-is (the PATT uses it only as a last-resort fallback).
# The old "daily_demands_by_day" export can stay or go; with
# "demand_by_segment" present it is ignored.
#
# Optional consistency check (cheap, catches unit errors early):
#
#     for i, st in enumerate(depot_stores):
#         seg_total = sum(sum(v) for v in json_dict["demand_by_segment"][str(i+1)].values())
#         agg_total = sum(aggregated_weekday_demand_for(st))  # however the old path summed it
#         assert abs(seg_total - agg_total) < 1e-6, f"segment/aggregate mismatch at store {st}"


# ----------------------------------------------------------------------------
# CHANGE 2 — in class PATTResult, method append_solution: p_frt lookup bug.
#
# BEFORE (buggy — delivery_amounts silently all 0 when id_map is present):
#
#     for i in range(1, len(route) - 1):
#         amount = p_frt.get((sm[route[i]], pattern_id, day), 0.0)
#         ...
#
# AFTER:
# ----------------------------------------------------------------------------

#     for i in range(1, len(route) - 1):
#         internal_id = route[i]                       # p_frt keys are INTERNAL ids
#         amount = p_frt.get((internal_id, pattern_id, day), 0.0)
#         original_id = sm.get(internal_id, internal_id)   # map ONLY for output
#         delivery_amounts[original_id] = amount
#         ...

# If append_solution also stores Route.stops for the SIM stage, keep mapping
# the stops through sm as before — only the p_frt LOOKUP must use internal ids.


# ----------------------------------------------------------------------------
# CHANGE 3 (optional) — wire depot sizes into the PATT daily DC capacity.
# RLRPResult.depot_sizes is currently passed but unused. If a depot's size
# should cap the total daily outflow, add to the per-depot instance JSON:
#
#     json_dict["Q_day_max"] = float(depot_size_in_tonnes)
#
# The new PATT reads Q_day_min / Q_day_max from the instance file directly.
# ----------------------------------------------------------------------------
