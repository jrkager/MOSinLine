# MOSinLine web tool

A demo of the integrated pipeline for the project report: three stages —
**RLRP → PATT → SIM** — working together **in a loop**, with a live view of
which stage the algorithm is in and which feedback edge it took.

Design notes and the full screen plan are in [WEBTOOL.md](WEBTOOL.md).

```
  SvelteKit SPA  ──HTTP──>  FastAPI (control plane only)
        │                        │  subprocess
        │                        ▼
        │                     run.py   (the pipeline CLI)
        │                        │  writes JSON artifacts
        └──polls──────────  exports/runs/<run-id>/frontend/**.json
```

The backend never solves anything itself: it builds an argv for `run.py` and
supervises the process. The CLI and the web tool are two views of the same
pipeline, and every screen is rendered from files on disk — there is no
database.

## Requirements

- Python 3.10+, a working **Gurobi** license (a restricted/trial license only
  covers instances up to roughly 5 stores)
- `pip install fastapi "uvicorn[standard]" numpy pandas scikit-learn scipy gurobipy`
- Node 20+ and pnpm 9+ (frontend only)

## Run it

Backend, from the repo root:

```bash
python3 -m uvicorn webtool.server.app:app --reload --port 8000
```

Frontend, in a second terminal:

```bash
cd web-app && pnpm install && pnpm dev
```

Then open <http://localhost:5173>. Vite proxies `/api` to `127.0.0.1:8000`, so
no CORS or base-URL configuration is needed for local use. To point the
frontend at a different backend, set `PUBLIC_API_BASE_URL`.

Start a run from **New run**. The *Demo preset* (5 stores, 15 ALNS iterations)
finishes in about a minute and still exercises the full loop, including real
feedback rounds — that is the one to use when demoing live. Most of that minute
is PATT's own (R,S) shelf simulation, which runs once per solve before the first
ALNS iteration; raising the iteration count is cheaper than it looks.

## Or drive it from the CLI

The web tool is optional; the same pipeline runs headless.

```bash
python3 run.py params > params.json
```

```bash
python3 run.py run --run-name demo --stores 10 --patt-iterations 25
```

```bash
python3 run.py list
```

```bash
python3 run.py stop demo
```

Other flags: `--payload`, `--instance-kind`, `--lam`, `--mode`, `--max-rounds`,
`--sim-weeks`, `--sim-runs`, `--variants`, `--instance-index`. `run.py validate`
checks a parameter file and an instance payload without solving.

## Instance input

Three sources, all selectable in the new-run form:

| Source | What it is |
|---|---|
| `r101` | `k` stores sampled from Solomon R101, instance index `i` (1–20), deterministic |
| `synthetic` | the 5-store instance from `main.construct_test_instance()` |
| `payload` | an uploaded JSON file |

The payload format is documented by
[`schemas/instance-payload.schema.json`](schemas/instance-payload.schema.json),
with a working example in
[`sample_instance_payload.json`](sample_instance_payload.json):

```bash
python3 run.py run --payload sample_instance_payload.json --patt-iterations 15
```

Watch the conventions — they are the easiest thing to get wrong: depot ids are
**negative**, store ids **positive**, `weekday` is `0..5` for Mon–Sat, segments
are `dry|fresh|frozen`, and demand is in **tonnes**. Distances are derived
(Euclidean over all node pairs), not uploaded. Coordinates are plain XY, not
lat/lon.

## The screens

| Screen | Shows |
|---|---|
| **Loop** | The cycle diagram. Active stage pulses, completed stages carry their headline result, unreached stages are ghosted, and the feedback edge that fired animates. Click an edge to see what data crosses it. The round timeline rewinds everything to an earlier round. |
| **RLRP** | The network map: depot squares scaled by capacity built, filled when opened and dashed when closed; store dots scaled by demand and coloured by the depot serving them; store→depot assignment spokes; and the second stage's own aggregate-demand tours. Plus capacity-built-vs-demand-assigned bars, a per-depot table, and all scenarios side by side as small multiples. |
| **PATT** | The delivery-pattern calendar (stores × Mon–Sat, with a 6-bit pattern ribbon, shaded by tonnage or stacked by segment), the delivery-frequency distribution, weekly tonnage per segment, load per weekday stacked by segment against vehicle capacity Q, the routes map per weekday with vehicle fill, all six days as small multiples, predicted KPIs, ALNS convergence and operator performance. |
| **SIM** | The predicted-vs-simulated KPI table (the headline validation result, including the `delivered = demand - stockout + waste` conservation check), the acceptance verdict, and the variant comparison. |
| **How it works** | Standalone documentation page: an annotated protocol diagram (what each module consumes and produces, and the named transform between them) plus prose on why the RLRP and PATT disagree about demand, the capacity pre-check, the conventions, and the objective alignment. |

## Output on disk

```
exports/runs/<run-id>/
  manifest.json          status, parameters, timings
  params.json            the exact resolved parameters (replayable)
  logs/                  pipeline.log + one log per stage and round
  frontend/              the artifact tree the UI reads
    overview.json  instance.json  pipeline/progress.json
    rounds/<r>/rlrp.json
    rounds/<r>/patt/index.json   rounds/<r>/patt/<scenario>-<depot>.json
    rounds/<r>/sim.json
    feedback/timeline.json
```

`exports/` is gitignored. Deleting a run directory removes it from the UI.

## Feedback modes

| Mode | Behaviour |
|---|---|
| `single` | one pass, no feedback — fastest |
| `capacity` | PATT → RLRP only: when a depot cannot carry PATT's minimum throughput, RLRP demand is scaled up (capped per round) and RLRP re-solves. The simulation still runs and reports, but does not steer. |
| `full` | the above plus SIM → PATT: when simulated KPIs miss PATT's prediction, λ is lowered and PATT re-solves |

The capacity edge is the one that was already implemented and validated (it
comes from `run_pipeline_B.py`). Its check is: take the pattern with the
smallest total weekly delivery for each store, sum those minima over the depot's
stores, divide by the six delivery days, and require `check_margin` headroom
against the depot size. If even that lower bound does not fit, no combination of
patterns can, so PATT is skipped entirely and the requirement goes back to the
RLRP. **The SIM → PATT acceptance criterion is
provisional**: it accepts when the reference variant's waste % and stockout %
stay within a tolerance of the PATT prediction. The thresholds are a modelling
decision that still needs sign-off — see WEBTOOL.md §9.4. They are exposed under
*advanced parameters*.

## Not built yet

Deliberately out of scope for this demo (WEBTOOL.md §0): run comparison and λ
sweeps, AnyLogic CSV export from the UI, uploading a payload through the browser
(the path is entered server-side), authentication, and remote/SSH execution.
