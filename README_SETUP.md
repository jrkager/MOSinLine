# MOSinLine full-chain setup: RLRP → segment PATT → AnyLogic SIM

This guide takes you from an empty VS Code project to running the complete
pipeline on your own machine, with each handoff verified.

Chain and interfaces:

    [1] RLRP (Johannes, Gurobi)      decides: open depots, sizes, store→depot assignment
            │  RLRPResult.customer_depot_assignment[scenario][depot] = [stores]
            ▼
        create_patt_instance_data()  writes one JSON per (open depot, scenario):
            distances, Q/W0/c_km/fuel/eta, marginal_co2 (→ theta_TR), lambda,
            demand_by_segment (store × {dry,fresh,frozen} × weekday)
            ▼
    [2] PATT (your segment ALNS)     decides: pattern per store, p_frt, routes
            │  main() returns (ComprehensiveSolution, instance_data)
            │  PATTResult.append_solution() collects patterns + routes + delivery_amounts
            ▼
        export_anylogic_csv()        writes stores_s<k>.csv (33 cols) + routes_s<k>.csv
            ▼
    [3] SIM
        (a) fast: sim_des_port.py   Python port of the DES, runs inside the pipeline
        (b) full: DES_datadriven_segments.alp in AnyLogic, reads the CSVs,
            executes the plan for 54 weeks under Variants 1–8

---

## 0. Prerequisites

- Python 3.10+ (`python --version`)
- VS Code + Python extension
- Git
- Gurobi with a license. TUM academic license: register at gurobi.com with your
  @tum.de mail, get a key, run `grbgetkey <key>` once. (The pip-installed
  `gurobipy` also ships a size-limited trial license that is enough for the
  5-store test instance, so you can start before sorting the license.)
- AnyLogic (PLE or University) — only for step 7.

## 1. Project setup in VS Code

Open a terminal (VS Code: Terminal → New Terminal) and run:

    git clone https://github.com/jrkager/MOSinLine.git
    cd MOSinLine
    code .            # reopen VS Code in the repo folder if not already

Create and select the virtual environment:

    python -m venv .venv
    .venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
    pip install numpy pandas scikit-learn scipy gurobipy

In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`.

## 2. Overlay the patched files

Unpack `mosinline_bundle.zip` (from our session) INTO the repo root, letting it
overwrite. It contains, at the right paths:

    main.py                      ← replaces Johannes's main.py (aligned params,
                                   scenario generator, segment export, p_frt fix,
                                   no auto-run on import)
    patt/alns.py                 ← replaces the old PATT with your segment ALNS
    scenario_gen.py              ← structural demand scenarios (base/growth/fresh_shift)
    sim_des_port.py              ← Python port of the DES (Variants 1–8)
    run_pipeline_demo.py         ← stage-by-stage demo with printed handoffs
    run_full_pipeline_sim.py     ← RLRP → PATT → Python-SIM, all variants
    analyze_pipeline_results.py  ← full-KPI + diagnostics dump (JSON)
    experiment_v5v8_lambda.py    ← single-rule ablations + lambda sweeps
    export_anylogic_csv.py       ← solved run → AnyLogic CSVs
    export_for_anylogic.py       ← glue: whole pipeline → CSVs per scenario
    anylogic/DES_datadriven_segments.alp   ← the segment-aware DES
    reference/                   ← patch documentation (what changed and why)

Tip: `git diff main.py` shows you exactly what changed vs Johannes's version.

## 3. Verify stage 1: RLRP alone

    python -c "import gurobipy as gp; m=gp.Model(); x=m.addVar(); m.setObjective(x); m.addConstr(x>=1); m.optimize(); print('Gurobi OK')"

Then:

    python run_pipeline_demo.py

Expected: the RLRP solves (3 depots, 5 stores, 3 scenarios), prints which depot
opens per scenario (depot −1, size ≈15.22 t/day, all 5 stores), then prints the
handoff JSON (check `demand_by_segment` values), runs your PATT per scenario,
and prints the PATTResult with NON-ZERO delivery_amounts. Runtime ≈ 30–60 s.

If Gurobi complains about the license → sort step 0; the trial license bundled
with pip gurobipy handles this instance size.

## 4. Verify stage 2: PATT + fast SIM (Python)

    python run_full_pipeline_sim.py

Expected: per scenario, a table comparing the PATT model's predicted KPIs with
the simulated KPIs under Variants 2/1/3/4. The validation check is the V2 row:
waste/stockout/transport should sit within a few tenths of a percentage point /
~5% of the PATT prediction (residual = integer units + RNG).

Optional deeper runs:

    python analyze_pipeline_results.py      # full KPIs + diagnostics → analysis_results.json
    python experiment_v5v8_lambda.py        # V5–V8 ablations + lambda sweep

Increase `PATT_ITER` in these scripts (25 → 500–2000) for paper-grade numbers;
the 25-iteration default is for fast smoke runs.

## 5. Export the plan for AnyLogic

    python export_for_anylogic.py --out C:/mosinline_anylogic --iters 500

Writes `stores_s1.csv … stores_s3.csv` + `routes_s1.csv … routes_s3.csv`
(33-column segment format; also backward compatible with the old single-segment
.alp). Use forward slashes in the path — the .alp startup code expects them.

## 6. Wire up AnyLogic

Open `anylogic/DES_datadriven_segments.alp` in AnyLogic. Before running:

1. In Main → "On startup" code, set `_dir` to your CSV folder, e.g.
   `String _dir = "C:/mosinline_anylogic/";`  (forward slashes, trailing slash).
2. Main parameters: `KM_PER_PX = 0.1` (should already be set).
3. The `stores` agent population size must equal the store count in the CSV
   (5 for the test instance). Adjust the population if needed.
4. The startup code still reads the capacity Excel (`excelCapFile`) first and
   throws if it has fewer rows than stores — keep your existing Excel with ≥N
   store rows (its values are then overwritten from the CSV, so content is
   irrelevant, only row count matters).
5. `configId` parameter selects the run:
   - `configId = -1` → SWEEP mode: each Run advances one config
     (160 total = s1..s20 × Variants [2,1,3,4,5,6,7,8]); progress is stored in
     `_sweepCounter.txt` in `_dir` — delete it to restart at config 0.
   - `configId = k` → single config; `k % 8` picks the variant
     (0→V2, 1→V1, 2→V3, 3→V4, 4→V5, 5→V6, 6→V7, 7→V8), `k / 8 + 1` picks the
     instance (s1…). For the pipeline export use configs 0–7 with s1, etc.
6. Model stop time ≥ 54 weeks = 9072 hours (2 warmup + 52 recording).
7. Run. Outputs land in `_dir`:
   `WeeklyWasteReport.csv` (per store/week waste per segment + CO2e in kg),
   `WeeklyStockoutReport.csv`, `trucks.csv` — each tagged with
   `outSuffix = _s<k>_variant<v>`.

## 7. Verification checklist (do once, in this order)

- [ ] `run_pipeline_demo.py` prints non-zero delivery_amounts (p_frt bug fix active)
- [ ] `run_full_pipeline_sim.py`: V2 ≈ PATT prediction per scenario
- [ ] AnyLogic Variant 2 (config 0) KPIs ≈ Python-port V2 ≈ PATT prediction
      (drop the first 2 weeks of the weekly reports before aggregating; expect
      the same integer-unit residual as the Python port)
- [ ] AnyLogic Variant 3 shows "DROP rule"/"Route cancelled" traceln lines and
      Variant 2 shows none
- [ ] Then start the variant sweep / experiments

## 8. Where each connection lives (for debugging)

| Handoff | Code location |
|---|---|
| RLRP → PATT | `main.py: create_patt_instance_data()` (JSON incl. `demand_by_segment` via `_export_demand_by_segment`) |
| PATT entry | `patt/alns.py: main(instance_file_name=...)` → `(solution, instance_data)` |
| PATT → pipeline | `main.py: PATTResult.append_solution()` (patterns + routes + delivery_amounts, internal-ID lookup fixed) |
| PATT → AnyLogic | `export_anylogic_csv.py` (33-column stores CSV + routes CSV; S per (store, segment) from `alns.S_fsr`) |
| AnyLogic ingest | `.alp` Main "On startup" data-driven loader (`_dir`, column layout documented in `export_anylogic_csv.py` header) |
| Fast SIM stand-in | `sim_des_port.py` (event-by-event port of the .alp; same variants) |
| Feedback knobs | `main.py: modify_instance_after_patt_infeasible` (demand ×1.1 → re-RLRP), `modify_weights_after_sim_infeasible` (λ ×0.9 → re-PATT; empirically: stockout-driven → λ down) |

## Known caveats

- Restricted/trial Gurobi caps model size; the 5-store instance fits. Larger
  instances need the academic license.
- AnyLogic Java compilation is the one thing not machine-verified in this
  session — if the compiler complains, the first place to look is the three
  retyped Main variables (`targetLevelS: double[][]`, `pendingOrderQty`,
  `deliverTodayQty: int[][]`).
- Old sweep outputs (mirror/v1/v2 era) are not comparable: parameters, θ_FW
  units (now kg CO2e per unit), variant definitions and a drop/cancel
  causality fix all changed.
