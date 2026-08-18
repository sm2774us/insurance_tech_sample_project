"""fig-quant CLI entry point."""

from __future__ import annotations

import typer

app = typer.Typer(help="FIG Pod quant research pipeline CLI.")


@app.command()
def validate_signal(n_permutations: int = 2000, q: float = 0.10) -> None:
    """Runs the incremental-IC/permutation/BHY signal validation demo."""
    import numpy as np

    from fig_quant.data.signal_validation import validate_features

    rng = np.random.default_rng(0)
    n = 2000
    groups = rng.integers(0, 50, size=n)
    control = np.column_stack([np.ones(n), rng.normal(size=n)])
    signal = rng.normal(size=n)
    target = 0.4 * signal + 0.3 * control[:, 1] + rng.normal(scale=0.5, size=n)
    noise = rng.normal(size=n)
    results = validate_features(
        {"alt_hazard_score": signal, "random_noise_feature": noise},
        target,
        control,
        groups,
        q=q,
        n_permutations=n_permutations,
    )
    for r in results:
        typer.echo(f"{r.feature_name}: IC={r.incremental_ic:.4f} p={r.p_value:.4f} rejected={r.rejected}")


@app.command()
def serve(
    api_host: str = "0.0.0.0",
    api_port: int = 8001,
    gui_host: str = "0.0.0.0",
    gui_port: int = 8000,
) -> None:
    """Starts the full stack: FastAPI JSON API + Taipy GUI console.

    Point a browser at http://<gui_host>:<gui_port>/ for the human-facing
    Taipy research console. The FastAPI JSON API (with interactive docs
    at /docs) runs alongside it on a separate port, since Taipy's
    WebSocket-based frontend cannot be correctly proxied through a WSGI
    mount inside the FastAPI ASGI app.
    """
    from fig_quant.web.server import run_full_stack

    run_full_stack(api_host=api_host, api_port=api_port, gui_host=gui_host, gui_port=gui_port)


@app.command()
def serve_api(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Starts only the FastAPI JSON API (no Taipy GUI)."""
    import uvicorn

    uvicorn.run("fig_quant.web.app:api", host=host, port=port)


if __name__ == "__main__":
    app()
