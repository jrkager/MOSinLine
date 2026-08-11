"""Drop solver boilerplate from the pipeline log.

Gurobi writes its licence banner and a `Set parameter ...` line for every
parameter it touches straight to the process's stdout, from the C library — so
`contextlib.redirect_stdout` inside the pipeline never sees it and it lands in
`pipeline.log` between the lines we actually care about. On a multi-round run
that is dozens of lines of noise around each solve.

This filters those lines out. It is deliberately narrow: only banner and
parameter-echo lines are dropped, never anything the pipeline itself printed.
Running `run.py` directly still shows everything, which is what you want when
diagnosing a licence problem.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, List

# Lines Gurobi emits that carry no information for us.
_NOISE_PATTERNS: List[re.Pattern] = [
    # "Set parameter Username", "Set parameter LogToConsole to value 0", ...
    re.compile(r"^Set parameter\b"),
    # "Academic license - for non-commercial use only - expires 2026-11-14"
    # also covers "Restricted license", "WLS license", "Gurobi license" wording
    re.compile(r"^\s*\w[\w ]* license\b.*(?:expires|non-commercial|non-production)", re.I),
    # startup banner, when it appears
    re.compile(r"^Gurobi Optimizer version\b"),
    re.compile(r"^Copyright \(c\) \d{4}, Gurobi Optimization\b"),
    re.compile(r"^Thread count: .*logical processors", re.I),
]


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _NOISE_PATTERNS)


def filter_lines(lines: Iterable[str]) -> Iterator[str]:
    """Yield only the lines worth keeping, collapsing the blank runs that the
    removed banners leave behind."""
    previous_blank = False
    for line in lines:
        if is_noise(line):
            continue
        blank = not line.strip()
        if blank and previous_blank:
            continue
        previous_blank = blank
        yield line


def filter_text(text: str) -> str:
    return "\n".join(filter_lines(text.splitlines()))
