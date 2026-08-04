"""Resolved parameter set for one pipeline run.

Single place where every knob the web tool can set is declared, with the
defaults taken from the existing code so that a default RunParams reproduces
what the CLI drivers already do:

  transport    -> main.TRANSPORT_PARAMS
  patt         -> patt.alns.default_algorithm_params()
  sim          -> sim_des_port module constants
  feedback     -> run_pipeline_B.py / main.py loop constants
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# instance source
# ---------------------------------------------------------------------------
@dataclass
class InstanceSource:
    """Where the instance comes from.

    kind == "payload"   -> path to an instance payload JSON (see schemas/)
    kind == "r101"      -> r101_segment_instance.construct_r101_instance(k, i)
    kind == "synthetic" -> main.construct_test_instance()
    """
    kind: str = "r101"
    payload_path: Optional[str] = None
    k: int = 10          # r101 only: number of stores
    i: int = 1           # r101 only: instance index 1..20


# ---------------------------------------------------------------------------
# stage parameters
# ---------------------------------------------------------------------------
@dataclass
class TransportParams:
    """Shared physical parameters. lam is the economic/environmental weight and
    MUST stay the same on the RLRP and PATT side -- main.derived_transport_
    coefficients() folds it into the RLRP arc coefficients."""
    c_km: float = 1.12
    c_fuel: float = 1.80
    eta: float = 0.05
    theta_TR: float = 2.7
    W0: float = 14.4
    Q: float = 25.6
    lam: float = 0.3


@dataclass
class RLRPParams:
    gap: float = 0.05
    timelimit: float = 1800.0
    heur_timelimit: float = 0.1
    n_threads: int = 0                  # 0 = all available
    second_stage_penalty_factor: float = 1.5
    demand_aggregation: int = 1         # 1 avg / 2 max / 3 n-th largest


@dataclass
class PATTParams:
    max_iterations: int = 25            # 25 = smoke run; 500-2000 = paper grade
    time_limit: float = 600.0
    algorithm_params: Dict[str, Any] = field(default_factory=dict)  # {} -> ALNS defaults


@dataclass
class SIMParams:
    variants: List[int] = field(default_factory=lambda: [2, 1, 3, 4])
    weeks: int = 52
    runs: int = 2
    warmup_weeks: int = 2
    seed: int = 20000


@dataclass
class FeedbackParams:
    """Controls the RLRP <-> PATT <-> SIM loop.

    mode:
      "single"   -> one pass, no feedback (fastest; still renders the loop
                    diagram, just without return edges)
      "capacity" -> RLRP<-PATT capacity feedback only (the edge that is
                    genuinely implemented, from run_pipeline_B.py)
      "full"     -> capacity feedback + the SIM->PATT lambda edge
    """
    mode: str = "full"

    # --- PATT -> RLRP edge (capacity feedback, from run_pipeline_B.py) ---
    safety: float = 1.35        # target headroom on fed-back throughput
    step_cap: float = 1.15      # max demand scaling per round (gentle on RLRP)
    max_rounds: int = 6

    # --- SIM -> PATT edge (lambda feedback) ---
    lambda_factor: float = 0.9  # applied when the SIM verdict fails
    max_sim_failures: int = 3

    # SIM acceptance criterion. NOTE: these thresholds are a placeholder pending
    # a modelling decision (see WEBTOOL.md section 9.4). The rule is: accept the
    # plan when the simulated KPIs of the reference variant do not exceed the
    # PATT model's own prediction by more than the tolerance.
    reference_variant: int = 2
    waste_tolerance_pp: float = 2.0      # percentage points
    stockout_tolerance_pp: float = 2.0   # percentage points


@dataclass
class RunParams:
    instance: InstanceSource = field(default_factory=InstanceSource)
    transport: TransportParams = field(default_factory=TransportParams)
    rlrp: RLRPParams = field(default_factory=RLRPParams)
    patt: PATTParams = field(default_factory=PATTParams)
    sim: SIMParams = field(default_factory=SIMParams)
    feedback: FeedbackParams = field(default_factory=FeedbackParams)

    # ---------------------------------------------------------------- io ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RunParams":
        raw = raw or {}
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            section = raw.get(f.name)
            if section is None:
                kwargs[f.name] = f.default_factory()          # type: ignore[misc]
                continue
            if not isinstance(section, dict):
                raise ValueError(f"params.{f.name} must be an object")
            sub_cls = f.default_factory                        # type: ignore[misc]
            defaults = sub_cls()
            known = {sf.name for sf in fields(defaults)}
            unknown = set(section) - known
            if unknown:
                raise ValueError(
                    f"unknown keys in params.{f.name}: {sorted(unknown)}")
            for key, value in section.items():
                setattr(defaults, key, value)
            kwargs[f.name] = defaults
        return cls(**kwargs)

    # ------------------------------------------------------------ checks ----
    def validate(self) -> List[str]:
        """Returns a list of human-readable problems; empty means OK."""
        problems: List[str] = []
        if self.instance.kind not in {"payload", "r101", "synthetic"}:
            problems.append(f"instance.kind must be payload|r101|synthetic, got {self.instance.kind!r}")
        if self.instance.kind == "payload" and not self.instance.payload_path:
            problems.append("instance.payload_path is required when instance.kind == 'payload'")
        if self.instance.kind == "r101" and not (1 <= self.instance.i <= 20):
            problems.append(f"instance.i must be within 1..20, got {self.instance.i}")
        if not 0.0 <= self.transport.lam <= 1.0:
            problems.append(f"transport.lam must be within [0, 1], got {self.transport.lam}")
        if self.transport.Q <= 0:
            problems.append("transport.Q must be positive")
        if self.patt.max_iterations < 1:
            problems.append("patt.max_iterations must be >= 1")
        if self.sim.runs < 1 or self.sim.weeks < 1:
            problems.append("sim.runs and sim.weeks must be >= 1")
        bad_variants = [v for v in self.sim.variants if v not in range(1, 9)]
        if bad_variants:
            problems.append(f"sim.variants must be within 1..8, got {bad_variants}")
        if self.feedback.mode not in {"single", "capacity", "full"}:
            problems.append(f"feedback.mode must be single|capacity|full, got {self.feedback.mode!r}")
        if self.feedback.mode == "full" and self.feedback.reference_variant not in self.sim.variants:
            problems.append(
                f"feedback.reference_variant {self.feedback.reference_variant} "
                f"must be one of sim.variants {self.sim.variants}")
        if self.feedback.max_rounds < 1:
            problems.append("feedback.max_rounds must be >= 1")
        return problems
