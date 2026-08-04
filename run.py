#!/usr/bin/env python3
"""MOSinLine pipeline CLI.

The web tool drives this script; it is also the supported way to run the
pipeline by hand with explicit parameters (the older run_*.py drivers keep their
settings as module constants).

    python run.py params > params.json      # dump the resolved defaults
    python run.py validate params.json      # check parameters + instance payload
    python run.py run --params params.json --run-name demo
    python run.py list                      # list runs under exports/runs
    python run.py stop <run-id>

Everything a run produces lands in exports/runs/<run-id>/ -- see webtool/layout.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webtool.layout import (RUNS_ROOT, RunLayout, list_runs, new_run_id,  # noqa: E402
                            now_iso, run_layout, write_json)
from webtool.params import RunParams                                      # noqa: E402


def _load_params(path: str | None) -> RunParams:
    if not path:
        return RunParams()
    with open(path, "r", encoding="utf-8") as fh:
        return RunParams.from_dict(json.load(fh))


def _apply_overrides(params: RunParams, args: argparse.Namespace) -> RunParams:
    if args.instance_kind:
        params.instance.kind = args.instance_kind
    if args.payload:
        params.instance.kind = "payload"
        params.instance.payload_path = args.payload
    if args.stores is not None:
        params.instance.k = args.stores
    if args.instance_index is not None:
        params.instance.i = args.instance_index
    if args.patt_iterations is not None:
        params.patt.max_iterations = args.patt_iterations
    if args.lam is not None:
        params.transport.lam = args.lam
    if args.mode:
        params.feedback.mode = args.mode
    if args.max_rounds is not None:
        params.feedback.max_rounds = args.max_rounds
    if args.sim_weeks is not None:
        params.sim.weeks = args.sim_weeks
    if args.sim_runs is not None:
        params.sim.runs = args.sim_runs
    if args.variants:
        params.sim.variants = [int(v) for v in args.variants.split(",") if v.strip()]
    return params


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_params(args: argparse.Namespace) -> int:
    params = _apply_overrides(_load_params(args.params), args)
    print(json.dumps(params.to_dict(), indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    params = _apply_overrides(_load_params(args.params), args)
    problems = params.validate()
    if params.instance.kind == "payload" and params.instance.payload_path:
        from webtool.instance_io import validate_payload
        try:
            with open(params.instance.payload_path, "r", encoding="utf-8") as fh:
                problems += validate_payload(json.load(fh))
        except (OSError, ValueError) as exc:
            problems.append(f"could not read payload: {exc}")
    if problems:
        print("INVALID:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("OK: parameters are valid")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from webtool.pipeline import run_pipeline
    from webtool.progress import ProgressTracker

    params = _apply_overrides(_load_params(args.params), args)
    problems = params.validate()
    if problems:
        print("INVALID PARAMETERS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    run_id = args.run_name or new_run_id()
    layout = RunLayout(RUNS_ROOT / run_id).ensure()
    layout.update_manifest(
        run_id=run_id,
        status="running",
        created_at=now_iso(),
        instance_kind=params.instance.kind,
        params=params.to_dict(),
    )
    tracker = ProgressTracker(layout, run_id=run_id,
                              instance_name=params.instance.kind,
                              params=params.to_dict())
    print(f"Run {run_id} -> {layout.root}")

    outcome = run_pipeline(params, layout, tracker)

    layout.update_manifest(status=outcome["status"],
                           reason=outcome.get("reason"),
                           rounds=outcome.get("rounds"),
                           finished_at=now_iso())
    print(json.dumps({k: v for k, v in outcome.items() if k != "traceback"}, indent=2))
    if outcome.get("traceback"):
        print(outcome["traceback"], file=sys.stderr)
    return 0 if outcome["status"] == "completed" else 1


def cmd_list(args: argparse.Namespace) -> int:
    rows = list_runs()
    if not rows:
        print("no runs under exports/runs")
        return 0
    print(f"{'run id':<28} {'status':<12} {'rounds':>6}  instance")
    for row in rows:
        print(f"{row.get('run_id', '?'):<28} {str(row.get('status')):<12} "
              f"{str(row.get('rounds') or ''):>6}  {row.get('instance_kind', '')}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    layout = run_layout(args.run_id)
    if not layout.root.exists():
        print(f"run not found: {args.run_id}", file=sys.stderr)
        return 1
    layout.request_stop()
    print(f"stop requested for {layout.run_id}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--params", help="path to a params JSON file")
        p.add_argument("--instance-kind", choices=["payload", "r101", "synthetic"])
        p.add_argument("--payload", help="instance payload JSON (implies --instance-kind payload)")
        p.add_argument("--stores", type=int, help="r101: number of stores (k)")
        p.add_argument("--instance-index", type=int, help="r101: instance index 1..20 (i)")
        p.add_argument("--patt-iterations", type=int,
                       help="ALNS iterations; 25 = smoke run, 500-2000 = paper grade")
        p.add_argument("--lam", type=float, help="lambda: economic vs environmental weight")
        p.add_argument("--mode", choices=["single", "capacity", "full"],
                       help="feedback mode")
        p.add_argument("--max-rounds", type=int)
        p.add_argument("--sim-weeks", type=int)
        p.add_argument("--sim-runs", type=int)
        p.add_argument("--variants", help="comma-separated DES variants, e.g. 2,1,3,4")

    p_params = sub.add_parser("params", help="print the resolved parameter set")
    add_common(p_params)
    p_params.set_defaults(func=cmd_params)

    p_validate = sub.add_parser("validate", help="validate parameters and payload")
    add_common(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run the pipeline")
    add_common(p_run)
    p_run.add_argument("--run-name", help="run id (default: timestamped)")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list runs")
    p_list.set_defaults(func=cmd_list)

    p_stop = sub.add_parser("stop", help="request a run to stop")
    p_stop.add_argument("run_id")
    p_stop.set_defaults(func=cmd_stop)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
