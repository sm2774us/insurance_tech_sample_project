# fig-quant

Production-grade quant research pipeline for a FIG-style ("AI-native, full-stack
specialty insurance carrier operated like a quantitative hedge fund") pod: bi-temporal
point-in-time data, incremental signal validation (IC/permutation/BHY), CANN severity
pricing with Combinatorial Purged Cross-Validation, GARCH(1,1)+EVT conditional tail
risk, Bayesian small-sample capital sizing, and a falsification-gated quadratic capital
allocator.

## Architecture

```
[ STANDALONE MODE ]                          [ CLUSTER MODE ]
Local SSD (Parquet)                          S3-compatible object storage
        │                                              │
        └───────────── Data Access Layer (DuckDB / Arrow / URI-swapped) ───┘
        │                                              │
Execution: DuckDB + Polars                   Execution: PyIceberg + Ray (extension point)
        │                                              │
        └────────── Arrow Table / RecordBatch (unified compute boundary) ──┘
        │                                              │
AI Engines: PyTorch / XGBoost (local)        AI Engines: Ray Train / Distributed PyTorch
        │
Web Backend: FastAPI + Pydantic  ──►  Frontend: Taipy
```

Swappable LLM logic: `fig_quant.llm_modules.base.BaseLLM` is the single interface;
`get_llm(use_local=...)` routes to `local_hf.LocalHfLLM` (in-process HF Transformers)
or `external_lite.ExternalLiteLLM` (LiteLLM-normalized hosted providers) with zero
call-site changes.

## Layout

```
src/fig_quant/
  pit/            bi-temporal point-in-time data store (DuckDB/Arrow, local or S3 URI)
  data/           signal validation harness + rate-limited alt-data ETL client
  models/         CANN severity pricing model + Combinatorial Purged CV
  risk/           GARCH(1,1) + EVT conditional tail risk / risk-loaded premium
  capital/        Bayesian small-sample sizing + falsification-gated capital allocator
  llm_modules/    swappable local/external LLM router
  web/            FastAPI + Pydantic async backend
  cli.py          Typer CLI (fig-quant)
notebooks/research.ipynb   plotly research notebook (HTML+PNG export, GitHub-safe)
infra/{aws,gcp,azure}      Terraform per cloud, each with its own README
tests/                     pytest suite (see coverage.xml / Codecov)
.github/workflows/ci.yml   lint, type-check, test+coverage, Codecov upload, release
```

## Quickstart

```bash
pip install uv

#If python 3.13 not installed then
# Natively download and cache isolated CPython variants
#uv python install 3.13

# Establish a target virtual environment pinned to the 3.13 executable
uv venv --python 3.13
# For linux use
source .venv/bin/activate

# For windows-11 use
# .\.venv\Scripts\activate.bat

uv sync --all-extras
uv pip install -e ".[notebook,dev,plot]"
uv run pytest
uv run fig-quant validate-signal
uv run fig-quant serve
```

`fig-quant serve` starts **two independent OS processes**, each speaking its
own required protocol:

- `http://localhost:8000/` — the Taipy GUI research console, run under
  Taipy's own Flask-SocketIO dev server (`web/server.py` → `Gui.run(...)`,
  foreground process). Taipy's frontend depends on a live **WebSocket**
  connection for state sync, so it must run natively rather than be
  mounted as WSGI-under-ASGI — a WSGI mount cannot proxy the WebSocket
  upgrade handshake (it comes back `403`), leaving the page blank.
- `http://localhost:8001/` — the FastAPI JSON API (`web/app.py`), under
  `uvicorn`, with interactive docs at `/docs`. Launched via
  `subprocess.Popen` as a genuinely independent process — deliberately
  **not** `multiprocessing.Process`, whose `spawn` start method re-imports
  the packaged console-script entry point in the child. Since that entry
  point calls the CLI unconditionally at import time, a
  `multiprocessing`-spawned child doesn't just run uvicorn — it
  re-executes the whole `fig-quant serve` command, recursively launching
  a second nested GUI+API pair that collides on the same ports as the
  first (this was diagnosed from an earlier revision: the browser ended
  up talking to the API on what was meant to be the GUI's port).

Run only the API with `uv run fig-quant serve-api`.

**Note on `web/gui.py`:** the Taipy `Gui(page=...)` object is constructed at
true module level, not inside a factory function. Taipy resolves which
module owns a page's bound variables/callbacks by inspecting the call stack
at construction time; wrapping `Gui(...)` in a `def build_gui(): return
Gui(...)` adds an extra stack frame that broke this resolution in an
earlier revision — every input fell back to an empty state (blank fields,
garbage widget defaults) and both buttons logged
`on_action(): '<name>' is not a valid function`. Keep `gui = Gui(page=_page)`
at module scope if you extend this file.

## Design principles mapped to requirements

- **Point-in-time architecture** — `pit.bitemporal.BitemporalStore.as_of(T)` enforces
  `t_event <= T AND t_ingest_start <= T < t_ingest_end`, blocking event- and
  ingestion-time lookahead by construction.
- **Capital deployment pipeline** — `capital.deployment.deploy_capital` runs the
  out-of-sample falsification gate before the quadratic mean-variance allocator;
  rejected proposals receive `K_m = 0` by construction.
- **Small-sample regime sizing** — `capital.bayesian_sizing` performs closed-form
  Normal-Inverse-Gamma posterior updates and sizes off the posterior LCB, not the
  posterior mean.
- **Conditional tail risk** — `risk.garch_evt` chains a GARCH(1,1) volatility filter
  into a GPD tail fit for `Pi_t = E[L|X] + lambda*sigma_t + kappa*VaR_alpha*sigma_t`.
- **Tooling independence** — `data.storm_events.StormEventClient` is a fully
  self-contained, dependency-injected, rate-limited, retrying ETL client requiring no
  platform-team scaffolding.
