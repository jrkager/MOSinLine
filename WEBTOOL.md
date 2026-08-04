# WEBTOOL.md — plan for a MOSinLine web tool

Planning notes for wrapping the RLRP → PATT → SIM pipeline in a self-hosted web
tool, using [max-elia/cspp](https://github.com/max-elia/cspp) as the template.

---

## 0. Purpose and audience — read this first

**This tool is a demo of the integrated process, built for the final project
report.** Its job is to make one thing obvious to a reader who has never seen
the code: that three separately developed optimization/simulation stages —
RLRP, PATT, SIM — actually work together, **in a loop**, feeding results back
into each other.

Concretely, that means:

- **The loop diagram is the centerpiece, not a decoration.** The main screen is
  a cycle diagram of RLRP → PATT → SIM → (feedback) → RLRP, animated to show
  **where the algorithm currently is**: which stage is active, which round it is
  on, which feedback edge was just taken and why. Every other screen is
  secondary to this.
- The feedback edges must be *visible and labelled* — "PATT infeasible → demand
  ×1.1 → re-run RLRP", "SIM stockout-driven → λ ×0.9 → re-run PATT" — because
  the loop is the contribution being demonstrated. In the current codebase this
  is the least visible part (it only exists as stdout across three scripts).
- Screens should read as **explanation**, not as an expert console. Show what
  each stage decided and hand it to the next stage explicitly; the
  `run_pipeline_demo.py` "[STAGE 1 OUTPUT] → [HANDOFF] → [STAGE 2 OUTPUT]"
  narration is exactly the right instinct, and should become the UI.
- Aesthetics matter more than they would for an internal tool — this ends up in
  a report, possibly as screenshots.

**Not the audience:** us, running paper experiments. λ sweeps, run comparison,
and raw artifact access are therefore *nice-to-have*, not v1.

**Later:** a retail planner may use it. They will need more features (real
store data, editable parameters, plan export, what-if comparison). Design so
those can be added — keep the parameter plumbing complete underneath even where
the demo UI hides it — but **do not build them now**.

Consequences for this plan: the ALNS tuning block and most of §4.3 stay hidden
behind an "advanced" disclosure or are omitted; §5's S5 (loop + progress) and S9
(feedback timeline) get merged and promoted to the primary screen; S10 (compare
runs / λ sweep) drops out of scope for v1.

---

## 1. What the reference tool (CSPP) actually is

"Charging Station Placement Planning" — a self-hosted planning tool for siting
electric-truck chargers in a retail delivery network. Same problem family as
ours, same institutional context, and it ships a thesis PDF alongside the code.

**The decisive finding:** `cspp/src/cspp/core/` is a *fork of our `rlrp/`
package.** `algorithm.py`, `classes.py`, `helper.py`, `heuristics.py`,
`algorithm_types.py`, `applications/…/model.py|instance.py` are the same files —
CSPP's copy is further evolved (1138 vs 864 lines in `algorithm.py`: added
`ThreadPoolExecutor` cluster parallelism, retry-on-no-incumbent, `logging_utils`),
and it uses flat imports (`from helper import …`) with `PYTHONPATH` injection
where we use relative imports. So this is not a loose analogy: **CSPP is a
worked example of putting a web tool around exactly our solver core**, and the
patterns transfer close to 1:1.

### 1.1 Stack (verified from the repo)

| Layer | Choice |
|---|---|
| Backend | **FastAPI** + uvicorn, Python 3.11+ (`fastapi`, `uvicorn[standard]`, `python-multipart`) |
| Frontend | **SvelteKit 2 / Svelte 5**, TypeScript, **Tailwind 4**, `@sveltejs/adapter-static`, Vite 7 |
| Charts | **hand-rolled d3** (`d3` ^7.9) — no chart library |
| Maps | **hand-rolled SVG** (`SimpleMap.svelte`, 913 lines) — no Leaflet/Mapbox |
| Extra | `ml-kmeans` for browser-side clustering *preview* |
| Persistence | **none — JSON files on disk.** No database. |
| Solver | Gurobi, local license required |
| Package mgr | pnpm 9+, Node 20+ |

### 1.2 Architecture (the part worth copying)

```
  SvelteKit SPA  ──HTTP──>  FastAPI (thin control plane)
        │                        │  subprocess.Popen
        │                        ▼
        │                   src/run.py  (CLI, the real pipeline)
        │                        │  writes JSON artifacts
        │                        ▼
        └──polls /api/sync──  exports/runs/<run-id>/frontend/**.json
```

Four properties that make this work well and that we should keep:

1. **The CLI is the product; the web app drives the CLI.** `src/run.py` is a
   full standalone entrypoint (`run.py list`, `run.py run all full --run-name x`,
   `run.py import-instance …`). The backend just builds an argv and
   `subprocess.Popen`s it. Nothing solver-related lives in the web layer.
2. **The API is tiny.** The whole backend surface is ~100 lines
   (`webserver_backend/api.py`): `POST/GET/DELETE /api/instances`,
   `POST /api/instances/{id}/runs`, `POST /api/runs/{id}/stop`,
   `DELETE /api/runs/{id}`, `GET /api/runtimes`, `POST /api/uploads/file`,
   plus `GET /api/sync/index` + `GET /api/sync/file`.
3. **The "frontend contract".** One module (`frontend_exports.py`, 1471 lines)
   converts raw solver output into a fixed tree of small JSON files:
   `frontend/manifest.json`, `overview.json`, `map/customers.geojson`,
   `stage_1/clusters.json`, `stage_1/clusters/<id>.json`,
   `stage_2/scenarios.json`, `stage_3/overview.json`,
   `frontend/pipeline/progress.json`. The UI reads *only* these. Solver
   internals can change without touching the frontend.
4. **Generic file sync instead of per-view endpoints.** A Web Worker polls
   `/api/sync/index` every 10 s, diffs by sha256, downloads changed files into
   **IndexedDB**, and the Svelte stores read from that mirror at 1 s. Adding a
   new screen = writing a new JSON file, zero backend work.

Progress/liveness: the solver writes `live_state.json` / `current_state.json` /
`*_progress_snapshot.json` as it goes; `pipeline_progress.py` (141 lines) folds
those into one `progress.json` with per-stage `status ∈ {missing, pending,
running, completed, failed}`. Objective trajectories come from a **Gurobi
callback** appending `(runtime, best_obj)` pairs.

Also present, and worth *skipping* for us initially: SSH "runtime delegation"
(`configs/cspp_runtimes.json`) that syncs a prepared run to a remote VM, starts
it there, polls, and syncs artifacts back, with a per-runtime queue.

### 1.3 UI shape (from the two screenshots in `docs/images/`)

- Persistent **left sidebar**: `Instances` / `New Instance`, a selected-instance
  section (`Overview + Map`, `Derive Instance`, `Runs`), a **RUNS** list with
  status pills (`running` / `stopped`), and a selected-run section
  (`Progress`, `Map`, `Stage 1: Cluster Solve`).
- **Run header bar**: run id, runtime dropdown, `Solve Running` / `Stop Solve`
  buttons, and an estimate chip (`Est. 1d 10:51:05 · 8,011 runs · 4 cores`).
- **"Solve Pipeline" panel**: a row of metric tiles — STAGE / ELAPSED /
  ESTIMATE / REMAINING / RUNTIME / CORES — plus a collapsible pipeline log and a
  status badge.
- **Per-stage panel**: live objective-vs-runtime chart (current entity bold,
  finished ones ghosted), then result panels below. `Open stage` drills down.
- **New-instance wizard**: numbered tabs `1 Source → 2 Filtering → 3 Clustering`,
  a live map preview with a right-hand column of read-only summary cards
  (SELECTION / PREVIEW / CLUSTER SUMMARY), `Back` + `Create Instance`.

Route nesting mirrors the domain:
`/instances/[instanceId]/runs/[runId]/stage-1/clusters/[clusterId]`.

---

## 2. Recommendation for MOSinLine

**Adopt the CSPP stack essentially unchanged.** FastAPI + SvelteKit/Svelte 5 +
Tailwind 4 + d3 + adapter-static, file-based state, CLI-driven jobs. Rationale:

- The solver core is literally the same code — no impedance mismatch, and their
  Gurobi/`PYTHONPATH`/venv setup is directly reusable.
- We can lift components with light edits: `StatusPill`, `MetricGrid`,
  `JsonPanel`, `TrendLineChart`, `ObjectiveTrajectoryChart`,
  `CostDistributionChart`, `ScenarioComparisonChart`, `SimpleMap`, plus
  `app-state.ts` polling and `api.ts`.
- No DB means the tool is trivially self-hostable next to a Gurobi license,
  which is the actual deployment constraint.

**Two deliberate deviations:**

- **Skip the IndexedDB mirror in v1.** Keep the same *contract*
  (`/api/sync/index` + `/api/sync/file` over an artifact tree) but have the
  client fetch artifacts directly and poll. Our runs are far smaller than
  CSPP's. The mirror can be dropped in later without touching page code.
- **Skip SSH runtime delegation in v1.** One local queue, one active run.
  Keep the `runtime_id` field in the run manifest so it can be added later.

---

## 3. Structural mapping CSPP → MOSinLine

| CSPP | MOSinLine |
|---|---|
| Stage 1 cluster solve | **RLRP** — depot open/size + store assignment per scenario |
| Stage 2 scenario evaluation | **PATT** — patterns + routes per (scenario, depot) |
| Stage 3 cluster reoptimization | **SIM** — DES evaluation of the plan, Variants 1–8 |
| cluster | **(scenario, depot) pair** — the unit PATT is solved on |
| historic demand scenario | **demand scenario** `s ∈ {1,2,3}` (base / growth / regional or fresh-shift) |
| charger placement | **delivery pattern per store** + routes |
| WGS84 lat/lon → real map | **Euclidean x/y** from Solomon → plain XY scatter |

**Two things CSPP has no equivalent for, and that drive our UI:**

1. **The pipeline is a loop, not a chain.** `main.py` wraps RLRP→PATT→SIM in a
   retry loop (`modify_instance_after_patt_infeasible` bumps demand and re-runs
   RLRP; `modify_weights_after_sim_infeasible` scales λ and re-runs PATT), and
   `run_pipeline_B.py` implements a working RLRP capacity-feedback loop over up
   to `MAX_ROUNDS = 6` rounds. So progress is **stage × round**, not flat
   stages, and the UI needs a round selector / timeline.
2. **We have a real simulation stage** with 52-week KPI time series, 8 variants,
   and an AnyLogic handoff. That's a whole screen family CSPP doesn't have.

Also note: our coordinates are synthetic Euclidean (Solomon R101), **not**
lat/lon. `SimpleMap.svelte` must be re-projected to a plain XY viewport — a
simplification, not extra work. If real store addresses are ever used, the CSPP
geo path comes back.

---

## 4. Inputs

### 4.1 Instance payload (upload / create)

Mirror CSPP's single-JSON-file approach with a JSON Schema in `schemas/`.
Our instance is richer than theirs (segments × weekdays × scenarios), so:

```jsonc
{
  "schema_version": 1,
  "instance_id": "R101_10stores_i1",
  "depots": [                      // candidate warehouse sites (negative ids)
    { "depot_id": -1, "x": 35.0, "y": 35.0,
      "fixed_cost": 5600.0, "marginal_cost": 35.0, "max_size": 30.0 }
  ],
  "stores": [
    { "store_id": 22, "x": 50.0, "y": 35.0, "name": "optional" }
  ],
  "scenarios": [
    { "scenario_id": 1, "name": "base", "probability": null }
  ],
  "demand": [                      // long form, one row per (s, store, segment, weekday)
    { "scenario_id": 1, "store_id": 22, "segment": "dry",
      "weekday": 0, "demand_t": 1.83 }
  ],
  "segments": {                    // optional; falls back to DEFAULT_SEGMENTS
    "dry":    { "shelf_life": null, "theta_FW": 1500.0 },
    "fresh":  { "shelf_life": 4,    "theta_FW": 4000.0 },
    "frozen": { "shelf_life": null, "theta_FW": 2500.0 }
  }
}
```

Validation rules to enforce on import: store ids positive and unique, depot ids
negative and unique, `weekday ∈ 0..5` (Mon–Sat), `segment ∈ {dry,fresh,frozen}`,
`demand_t ≥ 0`, every `demand.store_id` resolvable, and — importantly — a
**per-store feasibility check against Q** (the existing
`check_scenarios_against_vehicle` already does this; surface its verdict in the
UI before a run is allowed).

Distances are derived (Euclidean over all node pairs), not uploaded.

### 4.2 Instance builders exposed in the UI

Three "source" options in the new-instance wizard, matching what the code
already supports:

1. **Upload JSON** (the payload above).
2. **Solomon-derived** — `construct_r101_instance(k, i)`: pick `k` stores and
   instance index `i` (1–20), deterministic via `sel_seed`/`noise_seed`. Show
   the resulting store set and weekly tonnage in the preview.
3. **Synthetic** — `construct_test_instance()` + `scenario_gen.generate_scenarios`:
   5 stores, 3 structural scenarios, seed configurable.

### 4.3 Run parameters (the "configure run" form)

Grouped exactly as the code groups them. Defaults from
`TRANSPORT_PARAMS`, `default_algorithm_params()`, and `sim_des_port.py`.

**Shared / physical** (`main.TRANSPORT_PARAMS` — single source of truth):
`c_km` 1.12 · `c_fuel` 1.80 · `eta` 0.05 · `theta_TR` 2.7 · `W0` 14.4 ·
`Q` 25.6 · **`lam` 0.3**.

> λ is the headline knob — it weights economic vs environmental objective and is
> shared by RLRP and PATT. Changing it must re-derive the RLRP coefficients via
> `derived_transport_coefficients()`. The form must make it impossible to set λ
> on one side only.

**RLRP**: `gap` (0.05) · `timelimit` (1800 s) · `n_threads` (0 = all) ·
`HEURTIMELIMIT` (0.1) · `second_stage_penalty_factor` (1.5) ·
demand aggregation `option ∈ {1 avg, 2 max, 3 n-th largest}`.

**PATT**: `max_iterations` (**25 = smoke, 500–2000 = paper-grade** — expose this
prominently with that warning) · `time_limit` · the ALNS tuning block
(`alpha, beta, gamma, delta, eps, mu, xi, k, D, d, g, lambda, theta_1..3, r,
routing_iterations, routing_no_improve_limit, post_unsuccessful_limit, p_fifo`)
behind an "advanced" disclosure.

**SIM**: `variants` (multi-select from 1–8; default `[2,1,3,4]`) ·
`weeks` (52) · `runs` (2–3) · `warmup_weeks` (2) · `seed` ·
DES params (`DEMAND_CV` 0.25, `Z_URGENCY` 1.645, `MAX_SUPPLEMENT_DISTANCE` 140,
`DROP_THRESHOLD` 5, `MIN_ROUTE_UNITS` 15, `FILL_TRIGGER` 0.75,
`C_STOCKOUT` 100, `C_PURCHASE` 800, `P_FIFO` 0.70).

**Feedback loop**: `mode ∈ {single-pass, capacity-feedback (pipeline B), full
loop}` · `SAFETY` (1.35) · `STEP_CAP` (1.15) · `MAX_ROUNDS` (6) ·
`max_patt_failures` / `max_sim_failures` (3).

**Sweep mode** (optional, later): a list of λ values → N runs, feeding the
comparison screen. `experiment_v5v8_lambda.py` already does this offline.

A **runtime estimate** should be shown before starting, like CSPP's estimate
chip. We have real numbers to seed it from: PATT ≈ 10 s per (scenario, depot) at
25 iterations and ≈ 5 min at the 1000-iteration default, on a 5-store instance;
RLRP on 5–10 stores is ~1–2 s.

---

## 5. Output screens

### S1 · Instances (landing)
Table of instances: id, source (upload / R101 / synthetic), #stores, #depots,
#scenarios, weekly tonnage, #runs, last activity. Actions: new, derive, delete.

### S2 · New instance wizard
Tabs `1 Source → 2 Stores & Scenarios → 3 Depot candidates`, live XY preview +
right-hand summary cards (stores, weekly demand per scenario, per-segment split,
**Q-feasibility verdict**), `Create Instance`.

### S3 · Instance overview + map
XY scatter (depot candidates as squares, stores as dots sized by demand),
demand tables: per scenario × segment × weekday heatmap, per-store weekly
tonnage, scenario deltas (`s2 = ×1.20`, `s3 = ×1.09` etc. — the generator
already prints these).

### S4 · Configure & launch run
The §4.3 form, grouped, with defaults, an advanced disclosure, a validation
summary, the runtime estimate, and `Start Run`.

### S5 · Loop view ★★ — **the primary screen** (merges old S5 + S9)
This is what the report is about; build it first and best.

- **Cycle diagram**, large, centred: three stage nodes `RLRP → PATT → SIM`
  arranged as a cycle, with the two feedback edges drawn returning
  (`SIM ⇢ PATT`: λ ×0.9; `PATT ⇢ RLRP`: demand ×1.1 / capacity rescale).
  - The **active stage pulses**; completed stages are solid; not-yet-reached
    stages are ghosted. The edge currently being traversed animates.
  - Each node shows a one-line "what it decides" caption (RLRP: *depots + sizes
    + assignment*; PATT: *pattern per store + routes*; SIM: *executes the plan
    for 52 weeks*) and, once done, its headline number.
  - Each edge is labelled with **what data crosses it** (the handoff), and
    clicking it opens the actual payload — the RLRP→PATT instance JSON, the
    PATT→SIM store/route config.
  - Feedback edges show **why** they fired and what was changed.
- **Round timeline** beneath: one chip per round, `Round 1 · Round 2 · …`, with
  the per-round outcome (`PATT infeasible → rescale`, `SIM stockout → λ↓`,
  `accepted`). Selecting a round rewinds the whole page to that round's state.
- **Live detail panel** to the side: elapsed / remaining, current sub-unit
  (`scenario 2, depot −1`), RLRP bound convergence (UB/LB vs time, straight from
  the Gurobi callback) and PATT objective-vs-iteration, current bold and
  finished ghosted.
- Collapsible pipeline log tail.
- Must be **watchable end-to-end on a smoke run** (25 PATT iterations, ~30–60 s)
  so it demos live, and **replayable** from a finished run's artifacts so it can
  be screenshotted for the report without re-solving.

### S6 · Stage 1 — RLRP results
Per scenario: which depots opened and at what size (t/day), store→depot
assignment on the XY map (colour = depot), first/second-stage cost split, final
gap, and **which scenarios ended up in the final selection** — our algorithm
reports this (`Scenarios in final selection: {2}`) and it is genuinely
interesting output, not just diagnostics.

### S7 · Stage 2 — PATT results
Per (scenario, depot):
- **Pattern calendar**: stores × Mon–Sat grid of delivery flags, with delivery
  frequency and quantity per cell — the single most communicative view of a PATT
  solution.
- **Routes per weekday**: XY map with a day selector; arcs annotated with load;
  vehicle count and km per day.
- **Cost breakdown**: pattern cost, routing cost, food-waste purchase, stockout,
  transport distance/fuel, FW emissions, transport pollution — all already
  computed by `save_comprehensive_results()`.
- **ALNS convergence** + **operator performance table** (`build_operator_performance_table()`
  already returns a DataFrame; it is currently dumped as `.tex`).
- Waste % and stockout % predicted by the model.

### S8 · Stage 3 — SIM results
- **The validation table**: `PATT model` row vs `Variant 2/1/3/4/…` rows across
  waste %, stockout %, FW CO2, TR CO2, TR cost, km, cancel, drop, piggyback.
  This is the existing `fmt_table()` output and it is *the* headline result —
  V2 should match the PATT prediction closely.
- Variant comparison bars (per KPI, across selected variants).
- Weekly time series: waste and stockout per week per segment, with the warmup
  weeks visually marked as excluded.
- Per-store drill-down: waste/stockout contribution, pattern, frequency.

### S9 · Feedback-loop detail — *merged into S5*
Kept as a drill-down table behind the round timeline: one row per round with
what RLRP decided, whether PATT was feasible, the depot capacity check (`cap` vs
`need` per depot/scenario, as `run_pipeline_B.py` prints), what was rescaled and
by how much, whether SIM passed.

### S10 · Compare runs / λ sweep — **out of scope for v1**
(Scientist-facing; see §0.) Select N runs → KPI table + scatter of economic cost
vs CO2 across λ. Revisit only if the report needs it.

### S11 · Exports
Download buttons for the AnyLogic bundle (`stores_s*.csv`, `routes_s*.csv`,
`dist_*.csv`, `depot_*.csv`), the raw artifact JSON, the solver logs, and the
operator-performance `.tex`. Show the AnyLogic wiring checklist from
`README_SETUP.md` §6 inline (the `_dir`, `configId`, stop-time ≥ 9072 h items).

---

## 6. What has to change in *our* code first

This is the real work; the web layer is the easy half. None of it is throwaway —
it all improves the CLI too.

1. **A real CLI entrypoint** (`src/run.py` equivalent). Today the drivers are
   scripts with hardcoded module constants (`PATT_ITER`, `K`, `I`, `SAFETY`,
   `OUT_DIR`, `ITERS`). Needed: `run.py import-instance`, `run.py run
   {rlrp,patt,sim,all} --run-dir … --params params.json`. Every parameter in
   §4.3 must be settable from outside.
2. **A run-scoped export tree.** Today output scatters into `results/`,
   `results_patt/`, `logs/`, `./anylogic_csv`, plus `temp-*.json` in the repo
   root. Move everything under `exports/runs/<run-id>/…` with a stable layout.
3. **A `frontend_exports.py` equivalent** — the artifact contract of §1.2(3).
4. **Structured results instead of prints.** `RLRPResult` and `PATTResult` are
   fine as dataclasses but have no serializer; the interesting numbers
   (cost breakdown, KPI table, convergence) currently only exist as stdout or
   `.tex`. Add `to_json()` for each, and make `fmt_table()`'s data available as
   data.
5. **Progress emission.**
   - RLRP: `rlrp/algorithm.py` already has Gurobi callbacks
     (`filter_and_terminate_cb`) and a `params.logfile` handle — hook
     `(time, UB, LB, gap)` writes there.
   - PATT: `run_alns()` has no callback parameter. Add
     `progress_cb=None` and call it at each new best (it already computes
     `best_found_time` and prints `Iteration N: New best = X`).
   - Write `live_state.json` per (scenario, depot) and fold into
     `progress.json`, CSPP-style.
6. **Implement the SIM verdict — now blocking.** `SIMResult` is empty,
   `is_good_enough()` and `check_infeasible()` return `None`, so `main.py`'s loop
   can only ever end in `"infeasible"` (see [CLAUDE.md](CLAUDE.md)). Since §0
   makes the loop the thing being demonstrated, this can no longer be deferred:
   the SIM→PATT feedback edge needs a real acceptance criterion (§9.4) and
   `PATTResult.check_infeasible()` needs to report the depot-capacity violation
   that `run_pipeline_B.py` already detects externally.
7. **Cancellation.** `POST /api/runs/{id}/stop` needs the pipeline to die
   promptly. RLRP has `total_timelimit` and a `TimeoutException`; PATT has
   `time_limit` in `run_alns`. Process-level SIGTERM plus a
   check-for-stop-file at iteration boundaries is enough.
8. **Fix the log plumbing.** `solve_patt()` redirects ALNS stdout into
   `logfile_rlrp` and truncates it; `logfile_patt` is never written. The web UI
   needs one log file per stage.
9. **Import hygiene.** `patt/alns.py` prints three banner lines at import time
   and `save_comprehensive_results()` writes to `results/` relative to cwd. Both
   break under a server process. Also add `patt/__init__.py`.
10. **Determinism/reproducibility.** Persist every seed and the full resolved
    parameter set into the run manifest, so a run can be replayed from the UI.

---

## 7. Proposed repo layout

Keep the current modules where they are; add the wrapper around them.

```
MOSinLine-Code/
  main.py, rlrp/, patt/, sim_des_port.py, scenario_gen.py, …   # unchanged core
  src/
    run.py                     # CLI entrypoint
    instance_payload.py        # validate + import the JSON payload
    frontend_exports.py        # artifact contract
    pipeline_progress.py       # progress.json
    webserver.py               # app = create_app()
    webserver_backend/
      app.py  api.py  storage.py  run_service.py
      instance_service.py  pipeline_jobs.py
  schemas/instance-payload.schema.json
  docs/instance-format.md
  web-app/                     # SvelteKit
  exports/
    instances/<instance-id>/frontend/**.json
    runs/<run-id>/{state,logs,frontend,anylogic}/…
  sample_instance_payload.json
  requirements.txt   init_env.sh
```

Artifact contract sketch:

```
exports/runs/<run-id>/frontend/
  manifest.json                       # run id, instance id, params, status, timings
  overview.json                       # headline KPIs
  pipeline/progress.json              # stage × round status, estimates
  instance/map.json                   # nodes + coords
  rounds/<r>/rlrp/summary.json        # depots, sizes, assignment, costs, gap
  rounds/<r>/rlrp/convergence.json    # (t, UB, LB)
  rounds/<r>/patt/index.json          # list of (scenario, depot) units
  rounds/<r>/patt/<s>-<d>.json        # patterns, routes, costs, operator table
  rounds/<r>/patt/<s>-<d>.convergence.json
  rounds/<r>/sim/summary.json         # PATT-vs-variant KPI table
  rounds/<r>/sim/weekly.json          # weekly waste/stockout series
  feedback/timeline.json              # S9
```

---

## 8. Phased build plan

| Phase | Deliverable |
|---|---|
| **0** | Decide open questions in §9. Clone CSPP locally as a working reference. |
| **1** | `src/run.py` CLI + run-scoped `exports/` tree + params JSON. No web code. Verify parity: CLI reproduces `run_full_pipeline_sim.py`'s numbers. |
| **2** | `frontend_exports.py` artifact contract + `instance_payload.py` + JSON Schema + sample payload. Still no web code. |
| **3** | FastAPI backend: instances CRUD, create/stop/delete run, `subprocess.Popen` job runner with a single local queue, `/api/sync/*`. |
| **4** | SvelteKit skeleton lifted from CSPP: layout, sidebar, `api.ts`, `app-state.ts` polling, `StatusPill`, `MetricGrid`, `JsonPanel`. Screens S1, S3, S4. |
| **5** | **S5 loop view** — cycle diagram, round timeline, handoff inspection. The centerpiece; do it properly. |
| **6** | Progress emission (§6.5) → S5 animates live during a solve. |
| **7** | Result screens S6, S7, S8 + the XY map component. |
| **8** | S11 exports. Polish for report screenshots. |
| **9** | Optional / later (planner-facing): IndexedDB mirror, SSH runtime delegation, auth, S10 comparison. |

Phases 1–2 are worth doing **regardless** of whether the web tool ships — they
fix the reproducibility problems already noted in `CLAUDE.md`.

---

## 9. Open questions

1. ~~**Who uses this?**~~ **Answered — see §0.** It is a demo of the integrated
   loop for the final project report; a planner may use it later, we scientists
   are not the audience. The loop diagram becomes the primary screen, ALNS
   tuning is hidden, S10 drops out of v1.
2. **Geographic coordinates?** Everything today is Solomon XY. If real store
   locations are in scope, the CSPP geo/GeoJSON path is reusable verbatim;
   if not, the map component simplifies a lot.
3. **Which SIM backend is authoritative** — `sim_des_port.py` in-process, or
   AnyLogic? AnyLogic can't be driven from the web tool (it is a GUI app with a
   manual `_dir`/`configId` setup), so v1 realistically means: Python DES inside
   the tool, AnyLogic as a downloadable export + checklist.
4. **The feedback loop is now the point (§0), so it has to actually run.** The
   RLRP↔PATT edge already works for real in `run_pipeline_B.py` (capacity
   feedback, up to `MAX_ROUNDS = 6`) — that edge is demo-ready. The SIM→PATT
   λ edge is **not**: `is_good_enough()` is an empty stub. Minimum viable
   criterion for the demo: accept when simulated stockout % and waste % are
   within a configured tolerance of the PATT prediction, else scale λ per the
   existing note (*stockout-driven → λ down*). Needs sign-off on the thresholds
   — this is a modelling decision, not a UI one.
5. **Deployment target** — localhost only, or a shared VM? Determines whether we
   need auth and the SSH runtime machinery at all. CSPP has no auth.
6. **Gurobi licensing** on the host. Restricted/trial licenses cap instance size
   at roughly our 5-store test case; anything larger needs the academic license
   on the machine running the backend.
7. **Do we vendor CSPP's improved solver core?** Their `core/` has cluster
   parallelism and retry logic ours lacks. Merging is attractive but is a
   separate decision from the web tool, and would touch the one part of the
   codebase we have been keeping untouched.

---

## 10. Reference material

- Repo: https://github.com/max-elia/cspp — README, `docs/instance-format.md`,
  `schemas/instance-payload.schema.json`, `sample_instance_payload.json`,
  `thesis.pdf`, and the two UI screenshots in `docs/images/`.
- Files most worth reading before writing anything:
  `src/run.py` (CLI shape), `src/webserver_backend/api.py` (whole API surface),
  `src/frontend_exports.py` (artifact contract), `src/pipeline_progress.py`
  (progress model), `web-app/src/lib/app-state.ts` (client state),
  `web-app/src/lib/components/SimpleMap.svelte` (map).
