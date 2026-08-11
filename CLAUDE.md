# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

MOSinLine is academic OR research code that chains **three independently developed
modules** into one grocery-retail supply-chain workflow:

```
[1] RLRP   robust location–routing (Johannes, Gurobi MIP + Benders-style scenario alg)
           -> which depots open, at which size, which stores each depot serves
             │
             ▼  main.create_patt_instance_data()  — one temp JSON per (depot, scenario)
[2] PATT   delivery-pattern planning (Kailin, ALNS + LNS metaheuristic)
           -> weekly delivery pattern per store, per-day delivery quantities, routes
             │
             ▼  export_anylogic_csv()  /  build_sim_inputs()
[3] SIM    discrete-event simulation of the executed plan
           (a) sim_des_port.py — Python port, runs inside the pipeline
           (b) anylogic/DES_datadriven_segments.alp — the real DES, reads CSVs
```

The explicit design goal (see [README.md](README.md)) was to **keep each original
codebase as untouched as possible**. `main.py` is the glue: data structures,
parameter alignment, and the handoff format between stages. Prefer adding
translation code in `main.py` / the `run_*.py` drivers over editing `rlrp/` or
`patt/alns.py` internals.

## Repository layout

| Path | Role |
|---|---|
| `main.py` | Glue layer. `Instance`, `RLRPResult`, `PATTResult`, `SIMResult`, `TRANSPORT_PARAMS`, `solve_rlrp/solve_patt/solve_sim`, the RLRP↔PATT↔SIM feedback loop, and `construct_test_instance()` (5 synthetic stores). |
| `rlrp/` | Johannes's RLRP package (Gurobi). Entry: `rlrp.algorithm.ourAlgorithm(params)`. Models in `rlrp/applications/lrp/`. |
| `patt/alns.py` | **Live PATT module** — segment-aware ALNS. Entry: `main(instance_file_name=...) -> (ComprehensiveSolution, instance_data)`. |
| `kailin_pvrp_algorithm/` | Kailin's *standalone* upstream ALNS (`alns4.py`) + JSON test instances + own README. Reference/legacy — **not** what the pipeline imports. |
| `scenario_gen.py` | Structural demand scenarios for the synthetic instance (base / +20 % growth / fresh-shift). |
| `r101_segment_instance.py` | Instance builder from Solomon R101: `construct_r101_instance(k, i)`. This is what the real experiment drivers use. |
| `sim_des_port.py` | Python port of the AnyLogic DES (`DESPort`, `StoreCfg`), Variants 1–8. |
| `anylogic/DES_datadriven_segments.alp` | The actual AnyLogic model (data-driven from the exported CSVs). |
| `export_anylogic_csv.py`, `export_for_anylogic.py` | PATT solution -> `stores_*.csv` / `routes_*.csv` / `dist_*.csv` / `depot_*.csv` for AnyLogic. |
| `validate_anylogic.py` | Reads AnyLogic output CSVs back and recomputes KPIs for cross-checking. |
| `reference/` | Documentation-only Python files describing *what was patched and why* (parameter alignment derivation, segment-demand export, the `p_frt` internal-id bug fix). Not imported anywhere — read these before touching the parameter math. |
| `solomon_data/` | Solomon VRP benchmark files. |
| `tex/` | Integration report (`.tex` + PDF), GoodNotes sketch PDF, flowchart. |
| `run.py`, `webtool/` | Web-tool layer: parameterised pipeline CLI, run-scoped artifact tree, progress model, FastAPI backend (`webtool/server/`). See [README_WEBTOOL.md](README_WEBTOOL.md) and [WEBTOOL.md](WEBTOOL.md). |
| `web-app/` | SvelteKit frontend (the loop view). |
| `schemas/`, `sample_instance_payload.json` | Instance payload format for the web tool. |
| `webtool/instances.py` | Saved instances and the visual builder. Edits a small document (one mean daily demand per store + instance-wide shares/weekday shape/scenario factors) and expands it into the payload; presets round-trip through it, losing only the per-store demand noise. |
| `logs/`, `results*/`, `out/`, `exports/` | Run artifacts; gitignored. |

## Setup

Dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `gurobipy`.
Gurobi needs a working license — a **restricted/trial license is too small for
anything past ~5 stores**; the verified runs below used an academic license.

```bash
pip install numpy pandas scikit-learn scipy gurobipy
```

Sanity check before debugging anything else:

```bash
python -c "import gurobipy as gp; m=gp.Model(); x=m.addVar(); m.setObjective(x); m.addConstr(x>=1); m.optimize(); print('Gurobi OK')"
```

Always run scripts **from the repo root** — paths (`solomon_data/…`, `logs/`,
`patt/alns.py`) are relative to the working directory.

## Running things

Verified working on this machine (Python 3.10, Gurobi 13, macOS):

```bash
python run_pipeline_demo.py
```
Stage-by-stage demo on the 5-store synthetic instance with printed handoffs
(RLRP output → PATT instance JSON → PATT solution → `PATTResult`). ~31 s.
This is the fastest way to see the data flow; start here.

```bash
python run_full_pipeline_sim.py
```
RLRP → PATT → Python DES port on `R101_10stores_i1`, comparing PATT's predicted
KPIs against simulated Variants 2/1/3/4. ~2 min. **This is the integration
regression test**: the "Variant 2" row must track the "PATT model" row closely.
Last observed (continuous world, `a577673`), waste PATT vs V2 per scenario:
0.38 / 0.42, 1.72 / 1.84, 0.40 / 0.44 %; stockout 0.45 / 0.45, 0.50 / 0.43,
0.45 / 0.43 %. A gap of more than ~0.2 pp means a handoff broke.

The ALNS is **not seeded**, so the levels move between runs (a second observation
of scenario 1 gave 0.50 / 0.50 % waste). Check the PATT-vs-V2 *gap*, not the
absolute figures — those are indicative only.

**Compare agreement, not levels, against any older note.** The absolute KPI
levels have moved twice and by a lot: the integer-ordering change (`3cc93a6`)
put scenario-1 waste at ~9 %, and the continuous-world change (`a577673`) took
it to ~0.4 %. Waste is far lower in the continuous model because a fluid drains
the oldest batch smoothly instead of leaving whole units to expire. Any figure
quoted from before these commits — in the report, in old sweep outputs — is
inconsistent with the current code.

```bash
python run_pipeline_B.py
```
20-store / 2-depot pipeline with the RLRP capacity-feedback loop (gently scales
RLRP demand up until each depot's size covers the PATT minimum throughput),
then exports AnyLogic CSVs to `./anylogic_csv`. Long-running (`ITERS = 500`).

```bash
python export_for_anylogic.py --out /path/to/csvdir --iters 500
```

```bash
python run.py run --run-name demo --stores 10 --patt-iterations 25
```
The parameterised entry point used by the web tool: runs the full RLRP → PATT →
SIM loop with feedback and writes `exports/runs/<id>/`. ~2 min (10 stores, 25
iterations, 3 rounds); ~1 min at `--stores 5 --patt-iterations 15`. Prefer this
over the older drivers when you need to vary parameters.

Note on timings: the `N_RUNS = 10` change in `3cc93a6` made PATT's internal
shelf simulation ~5× more expensive, and it runs once per solve *before* the
first ALNS iteration. Everything got roughly 2× slower, and any timing recorded
before that commit under-reports. It also means raising the iteration count is
cheaper than it looks — the fixed setup cost dominates short runs.

```bash
python patt/alns.py kailin_pvrp_algorithm/instances/R101_5stores_s6.json
```
PATT standalone on a JSON instance. Writes into `results/` in the **current
working directory** (that is where the checked-in `results_patt/` came from).

```bash
python r101_segment_instance.py
```
Self-test of the instance generator: builds i=1..20 and asserts all are unique.

Other drivers: `analyze_pipeline_results.py` (full KPI + diagnostics dump to
JSON), `experiment_v5v8_lambda.py` (variant ablations + λ sweep),
`check_rlrp.py` (RLRP only, quick).

`README_SETUP.md` documents the AnyLogic side (`_dir`, `configId` sweep mode,
required stop time 9072 h) — follow it rather than re-deriving.

Iteration counts in the drivers (`PATT_ITER`/`PATT_MAX_ITER = 25`) are **smoke-test
values**. Paper-grade numbers need 500–2000.

## Conventions you must respect

- **Node ids**: depots are **negative** (`-1, -2, -3`), stores **positive**. Inside
  a PATT instance everything is renumbered: depot → `0`, stores → `1..n`, with
  `id_map` / `store_id_mapping` carrying the original ids back.
  `solution.p_frt`, `solution.pattern_assignments`, `alns.loc`, `alns.S_fsr` are all
  keyed by **internal** ids. Looking them up with mapped original ids silently
  yields zeros — this was a real bug, see `reference/mosinline_main_changes.py`.
  In the DES port, store ids shift again: `des_id = internal_id - 1`.
- **Days**: 6-day week, Mon..Sat, index `0..5` everywhere (`Weekday` enum, pattern
  tuples, `p_frt[...,t]`, `routes_by_day`).
- **Segments**: `ProductClass.DRY/FRESH/FROZEN` → lowercase `"dry"/"fresh"/"frozen"`
  in the PATT JSON → `"B"/"A"/"C"` in the DES port and the AnyLogic CSVs
  (`SEG_ORDER = ["fresh", "dry", "frozen"]` → A, B, C). Get this mapping wrong and
  results look plausible but are wrong.
- **Units**: demand and capacity in **tonnes**; `Q = 25.6 t`, `W0 = 14.4 t`.
  Emission factors `theta_FW` are kg CO2e per tonne wasted.
- **Parameters**: `main.TRANSPORT_PARAMS` is the single source of truth. The RLRP's
  `c_ij/alpha_ij/gamma_ij` are built by `derived_transport_coefficients()` so that
  the RLRP arc cost equals the PATT scalarization
  `(1-λ)·economic + λ·emissions` exactly. **Do not change one side only** — the
  derivation is in `reference/mosinline_param_alignment.py`. λ (`lam = 0.3`) is
  shared by both models; warehouse costs are pre-scaled by `(1-λ)` and
  `weighting_factor_rlrp = 0.5` is a uniform scaling that does not move the argmin.

## Known quirks and rough edges

These are real, currently-true properties of the code — don't "fix" them silently,
and don't be surprised by them.

- **`rlrp/` has one small additive patch** (branch `webapp-with-model-results`):
  `LRPReturnObject` gained `arcs` / `arc_loads`, populated in `algorithm.py` from
  the `x` and `t` variables the second stage already solves. Before that the
  routing the RLRP computes was discarded and only the assignment survived. This
  is the one place the "don't touch `rlrp/`" rule was bent — there is no glue hook,
  because the variables only exist inside `ourAlgorithm()`. `main.RLRPResult`
  carries them on with `getattr`, so an older `rlrp/` still works.
- **Use `run.py`, not `main.py`, to execute the pipeline.** `run.py` +
  `webtool/pipeline.py` is the parameterised implementation of the same loop,
  with the capacity feedback that actually works and artifacts on disk.
- **`main.py`'s feedback loop is a scaffold, and running it is a waste of time.**
  `PATTResult.check_infeasible()` and `SIMResult.is_good_enough()` are empty stubs
  returning `None`, and `solve_sim()` returns an empty `SIMResult` without
  simulating. Because `is_good_enough()` is falsy, the inner loop always re-runs
  PATT `max_sim_failures = 3` times and the function can only return
  `{"status": "infeasible", "reason": "SIM not good 3 times"}` — three identical
  PATT rounds, then a guaranteed failure verdict. It also runs PATT at the ALNS
  default of 1000 iterations (~5 min per scenario on the 5-store instance,
  measured), so a full `python main.py` is ~45 min of pointless work. The
  *working* feedback logic lives in `run_pipeline_B.py` (RLRP capacity feedback)
  instead. Use the `run_*.py` drivers, not `main.py`, to execute the pipeline;
  import `main` as the library it effectively is.
- **`solve_patt()` clobbers the RLRP log.** [main.py:415](main.py:415) redirects the
  ALNS stdout into `logfile_rlrp`, not `logfile_patt`; `logfile_patt` is only
  printed, never written. Consequence: the RLRP log is truncated to 0 bytes and
  PATT progress is invisible on the console during `main.py`.
- **Two ALNS versions coexist.** `patt/alns.py` (segment-aware, MOSinLine entry
  point, 2858 lines) vs `kailin_pvrp_algorithm/alns4.py` (upstream standalone, no
  `demand_by_segment` support, 3015 lines). Changes to one do **not** propagate.
  The pipeline only ever uses `patt/alns.py`.
- **`patt/` has no `__init__.py`** — `import patt.alns` works only via Python 3
  namespace packages, and only from the repo root.
- **Case-sensitive-filesystem hazard**: `r101_segment_instance.SOLOMON_PATH` is
  `"solomon_data/R101.txt"` but the file on disk is `solomon_data/r101.txt`. Works
  on macOS/Windows, breaks on Linux.
- **`rlrp/example_run.py` is legacy** — it uses absolute imports (`import helper`)
  and only runs from inside `rlrp/`. It is not part of the pipeline.
- **Segment shares differ between instance builders**: `scenario_gen.py` uses
  DRY/FRESH/FROZEN = 0.35/0.50/0.15, `r101_segment_instance.py` uses
  0.52/0.35/0.13. Intentional (different instance families), but easy to trip over
  when comparing results.
- **Temp PATT instance files** (`temp-<instance>-<depot>-<scenario>-<hash>.json`)
  are written to the repo root and deleted by `delete_patt_instance_file()`. A
  crash or interrupt mid-run leaves them behind (confirmed) — stopping a run from
  the web tool does it every time, since the pipeline is SIGTERMed. They are
  gitignored now, but still worth sweeping occasionally.
- **Old sweep outputs are not comparable** to current ones — parameters, θ_FW units,
  and variant definitions all changed (see README_SETUP "Known caveats").
- `Instance.pattern_operational_costs` / `foodwaste_emissions_factor` are `None`
  and unused; the current PATT computes both endogenously.
- Some comments/prints are in Chinese (`check_rlrp.py`, `run_pipeline_B.py`) — the
  project has multiple authors.
- **The ALNS builds its own (R,S) shelf simulation at construction time**
  (`ComprehensiveALNS.__init__`, `N_RUNS = 10` × 52 weeks × stores × patterns ×
  segments). That is where most of the PATT setup time goes, before a single
  ALNS iteration runs.
- **The (R,S) shelf model exists twice** — in `patt/alns.py` (to score patterns)
  and in `sim_des_port.py` (to execute the plan) — and the two must stay
  identical or the PATT-vs-Variant-2 comparison is meaningless. Since `a577673`
  both are **continuous**: no integer rounding of delivered or order quantities,
  no demand carry-over discretization, and consumption is a *fluid FIFO/LIFO
  split* (`d_fifo = p_FIFO · demand`, remainder LIFO) with **the FIFO stream
  served first** under scarcity. That tie-break is an arbitrary convention, so if
  you touch it on one side you must touch the other. DES shelves are batch lists
  `[[qty, expiry], …]`, not one entry per unit.
  Useful invariant both sides report: `delivered = demand − stockout + waste`.

## Working style for this repo

- This is research code backing a paper, not a product. Reproducibility matters
  more than refactoring: seeds (`SEL_SEED_BASE`, `NOISE_SEED_BASE`, the DES
  `seed=20000 + 37*run`) are deliberate.
- After changing anything in the handoff path, re-run `run_full_pipeline_sim.py`
  and check the PATT-vs-Variant-2 agreement before anything else.
- There is no test suite and no linter config. `svelte-check` covers the frontend
  (`cd web-app && pnpm check`).
- The web tool renders only from the JSON artifact tree under
  `exports/runs/<id>/frontend/`. To add a screen, emit a new artifact in
  `webtool/pipeline.py` — no backend endpoint is needed.
- Instances built in the UI land in `exports/instances/` as a pair: the editable
  `<name>.builder.json` and the expanded `<name>.json` the pipeline reads. Only
  the second is an input to `run.py`; regenerate it by saving again, never edit
  it by hand.
