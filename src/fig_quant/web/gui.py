"""Taipy GUI frontend for the FIG Quant Pod pipeline.

Implements the "Web Backend: FastAPI + Pydantic Async Tasks -> Frontend UI:
Taipy Webapp" leg of the architecture diagram. Runs as its own native
Flask-SocketIO server (see `fig_quant.web.server`), not mounted into the
FastAPI ASGI app, because Taipy's frontend depends on a live WebSocket
connection for state sync that a WSGI mount cannot proxy correctly.

IMPORTANT: `Gui(page=...)` is instantiated at true module level (below),
not inside a factory/builder function. Taipy resolves which module owns a
page's bound variables and callbacks (`log_severities_csv`,
`on_compute_lcb`, etc.) by inspecting the call stack at construction time.
Wrapping the construction in a function such as `def build_gui(): return
Gui(page=_page)` adds an extra stack frame between the module's top level
and the `Gui()` call, which broke that resolution in an earlier revision:
every bound variable fell back to an empty `SimpleNamespace` state (blank
inputs, garbage widget defaults, "on_action: ... is not a valid function"
warnings for both buttons). Constructing `gui` here, at import time of
this module, keeps the frame that Taipy inspects pointed at
`fig_quant.web.gui` itself, where all of the state and callbacks below
actually live.
"""

from __future__ import annotations

import numpy as np
from taipy.gui import Gui

from fig_quant.capital.bayesian_sizing import expected_severity_lcb, update_nig_posterior
from fig_quant.data.signal_validation import validate_features

# --- Bound state -------------------------------------------------------
# Every name referenced in `_page` below via `{name}` must exist as a
# module-level variable here; Taipy binds the page to this module's
# globals.
log_severities_csv = "6.2, 6.5, 6.1, 6.4, 6.3"
confidence = 0.95
posterior_mu = 0.0
severity_lcb = 0.0

n_permutations = 500
fdr_q = 0.10
validation_summary = "Run validation to see results."

_page = """
# FIG Quant Pod — Research Console

## Bayesian Small-Sample Severity Sizing

Log-severities (comma-separated): <|{log_severities_csv}|input|>

Confidence: <|{confidence}|slider|min=0.5|max=0.999|step=0.001|>  (<|{confidence}|text|format=%.3f|>)

<|Compute LCB|button|on_action=on_compute_lcb|>

Posterior mean (log-severity): **<|{posterior_mu}|text|format=%.4f|>**

Conservative expected-severity LCB: **<|{severity_lcb}|text|format=%.2f|>**

## Incremental Signal Validation (demo dataset)

Permutations: <|{n_permutations}|number|>

BHY target FDR (q): <|{fdr_q}|number|step=0.01|>

<|Run validation|button|on_action=on_run_validation|>

<|{validation_summary}|text|>
"""


def on_compute_lcb(state) -> None:
    """Recomputes the Bayesian posterior LCB from the bound input.

    Args:
      state: Taipy GUI state proxy carrying the current values of every
        bound page variable (`log_severities_csv`, `confidence`, etc.).
    """
    try:
        values = np.array(
            [float(v) for v in state.log_severities_csv.split(",") if v.strip()]
        )
        if values.size == 0:
            state.validation_summary = "Enter at least one comma-separated log-severity value."
            return
        posterior = update_nig_posterior(values)
        state.posterior_mu = posterior.mu
        state.severity_lcb = expected_severity_lcb(posterior, state.confidence)
    except ValueError:
        state.validation_summary = (
            "Could not parse log-severities; use comma-separated numbers, e.g. 6.2, 6.5, 6.1"
        )


def on_run_validation(state) -> None:
    """Runs the demo incremental-IC/permutation/BHY validation harness.

    Args:
      state: Taipy GUI state proxy carrying `n_permutations` and `fdr_q`.
    """
    rng = np.random.default_rng(0)
    n = 1500
    groups = rng.integers(0, 40, size=n)
    controls = np.column_stack([np.ones(n), rng.normal(size=n)])
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    target = 0.4 * signal + 0.3 * controls[:, 1] + rng.normal(scale=0.5, size=n)
    results = validate_features(
        {"alt_hazard_score": signal, "random_noise_feature": noise},
        target,
        controls,
        groups,
        q=state.fdr_q,
        n_permutations=int(state.n_permutations),
    )
    lines = [
        f"{r.feature_name}: IC={r.incremental_ic:+.4f}  p={r.p_value:.4f}  rejected={r.rejected}"
        for r in results
    ]
    state.validation_summary = "\n\n".join(lines)


# Constructed at module level -- see the module docstring for why this
# must not be wrapped in a factory function.
gui = Gui(page=_page)


def build_gui() -> Gui:
    """Returns the module-level `Gui` instance (kept for import-site clarity).

    Returns:
      The already-constructed :class:`taipy.gui.Gui` singleton bound to
      this module's state and callbacks.
    """
    return gui
