"""Web-tool layer around the MOSinLine RLRP -> PATT -> SIM pipeline.

The solver modules (main.py, rlrp/, patt/, sim_des_port.py) stay untouched; this
package adds the parameterised entry point, the run-scoped artifact tree and the
progress model that the FastAPI backend and the SvelteKit frontend consume.
"""

from .layout import RunLayout, run_layout, list_runs, new_run_id  # noqa: F401
from .params import RunParams                                     # noqa: F401
