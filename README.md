# MOSinLine

Integrated optimization–simulation workflow for sustainable grocery-retail
logistics. The repo brings together three independently developed pieces of
research code and makes them work as one pipeline — and, crucially, as a
**loop**, where each stage can hand its problem back to the previous one.

| Stage | Question it answers | Author / method |
|---|---|---|
| **RLRP** | Which depots open, at what size, and which stores does each serve? | Johannes — Gurobi MIP with robust scenario decomposition (ASBP) |
| **PATT** (DPPP) | On which weekdays is each store delivered, how much arrives, and on which routes? | Kailin — ALNS for patterns + LNS for routing |
| **SIM** | What actually happens when that plan is executed for a year? | Discrete-event simulation (Python port + AnyLogic model) |

```
        ┌──────────────── capacity shortfall: scale demand up ───────────────┐
        │                                                                    │
        ▼                                                                    │
     ┌──────┐   assignment + depot size   ┌──────┐   patterns + routes   ┌──────┐
     │ RLRP │ ─────────────────────────►  │ PATT │ ────────────────────► │ SIM  │
     └──────┘                             └──────┘                       └──────┘
                                              ▲                              │
                                              └──── KPI miss: adjust λ ───────┘
```

The design goal was to **keep each original codebase as untouched as possible**.
`main.py` is the glue: shared data structures, the parameter alignment that makes
both models optimise the same objective, and the handoff format between stages.
PATT is invoked by writing a temporary instance file, which is deleted again
afterwards.

## Quick start

Needs Python 3.10+ and a working **Gurobi** license (a restricted/trial license
only covers instances up to roughly 5 stores).

```bash
pip install numpy pandas scikit-learn scipy gurobipy fastapi "uvicorn[standard]"
```

Run the whole pipeline from the command line — a 5-store smoke run takes about
a minute and still exercises the full loop, feedback rounds included:

```bash
python3 run.py run --run-name demo --stores 5 --patt-iterations 15
```

Or use the web tool, which shows the loop live and is the easiest way to
understand what the pipeline does. Backend and frontend in two terminals:

```bash
python3 -m uvicorn webtool.server.app:app --reload --port 8000
```

```bash
cd web-app && pnpm install && pnpm dev
```

Then open <http://localhost:5173>; its **How it works** tab documents the whole
protocol. Setup details are in [README_WEBTOOL.md](README_WEBTOOL.md).

Always run scripts **from the repo root** — paths are relative to the working
directory.

## Documentation

| File | Contents |
|---|---|
| [README_WEBTOOL.md](README_WEBTOOL.md) | Running the web tool and the pipeline CLI, the instance payload format, the artifact tree |
| [WEBTOOL.md](WEBTOOL.md) | Design notes and screen plan for the web tool, and the open modelling questions |
| [README_SETUP.md](README_SETUP.md) | The AnyLogic side: CSV export, `_dir`, `configId` sweep mode, verification checklist |
| [CLAUDE.md](CLAUDE.md) | Repo orientation: conventions, verified run commands, known rough edges |
| [`reference/`](reference) | Documentation-only files explaining *what was patched and why* — the parameter-alignment derivation, the segment-demand export, the `p_frt` internal-id bug fix. Read these before touching the parameter math. |
| [`tex/`](tex) | Integration report (`.tex` + PDF) and the GoodNotes sketches the pipeline was designed from. The `.tex` was drafted from those sketches and has not been fully checked; the plots are unpolished. |

## Repository map

| Path | Role |
|---|---|
| `main.py` | Glue layer: `Instance`, the `*Result` classes, `TRANSPORT_PARAMS`, the RLRP↔PATT handoff |
| `rlrp/` | RLRP solver package (Gurobi) |
| `patt/alns.py` | The live PATT module — segment-aware ALNS |
| `sim_des_port.py` | Python port of the AnyLogic DES, execution variants 1–8 |
| `anylogic/` | The AnyLogic model, driven by exported CSVs |
| `run.py`, `webtool/` | Parameterised pipeline CLI, artifact contract, FastAPI backend |
| `web-app/` | SvelteKit frontend (the loop view) |
| `r101_segment_instance.py`, `scenario_gen.py` | Instance and demand-scenario builders |
| `run_pipeline_*.py`, `experiment_*.py` | The original experiment drivers |
| `kailin_pvrp_algorithm/` | Kailin's standalone upstream ALNS — reference, not what the pipeline imports |

## Things worth knowing before you change anything

- **Conventions are load-bearing.** Depot ids are negative and store ids
  positive, but PATT renumbers internally (depot→0, stores→1..n) and the DES
  shifts again. The week is six days, Mon–Sat, indexed 0..5. Segments are
  `dry`/`fresh`/`frozen` in the models but `B`/`A`/`C` in the DES and the
  AnyLogic CSVs. Getting one of these wrong produces plausible-looking wrong
  numbers rather than an error.
- **λ is shared.** Both models optimise `(1-λ)·economic + λ·emissions`. The
  weighting is folded into the RLRP's arc coefficients when the instance is
  built, so changing λ on one side only silently de-aligns the two objectives.
- **The regression check** after touching anything in the handoff path is
  `python3 run_full_pipeline_sim.py`: the Variant 2 row must track the PATT
  prediction closely.
- **Iteration counts in the drivers are smoke-test values.** Paper-grade PATT
  results need 500–2000 ALNS iterations, not 25.

`main.py` still contains an older sketch of the feedback loop whose infeasibility
checks are unimplemented stubs; use `run.py` (or `run_pipeline_B.py`) to execute
the pipeline and import `main` as the library it effectively is.
