"""`python -m nutrilens_ml` / `nutrilens-ml` CLI entry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler

from nutrilens_ml.config import ConfigError, load_settings

app = typer.Typer(
    help="NutriLens ML pipeline — train, eval, export, serve, ingest.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@app.callback()
def _root(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config overlay")] = None,
) -> None:
    try:
        settings = load_settings(yaml_override=config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    _configure_logging(settings.log_level)


@app.command()
def train(
    task: Annotated[str, typer.Argument(help="Task to train: plate | pour")],
    run_name: Annotated[str, typer.Option("--run-name", "-n")] = "default",
) -> None:
    """Train a model for the given task."""
    # Phase 3/4 will dispatch to `training.plate` / `training.pour`.
    console.print(f"[green]train[/green] task={task} run={run_name}")
    raise typer.Exit(code=0)


@app.command()
def evaluate(
    task: Annotated[str, typer.Argument(help="Task to evaluate: plate | pour")],
    bench: Annotated[str, typer.Option("--bench", "-b")] = "v0",
) -> None:
    """Run the held-out benchmark for a task."""
    console.print(f"[green]eval[/green] task={task} bench={bench}")
    raise typer.Exit(code=0)


@app.command()
def export(
    task: Annotated[str, typer.Argument(help="Task to export: plate | pour")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    out_dir: Annotated[Path, typer.Option("--out")] = Path("runs/exports"),
) -> None:
    """PyTorch checkpoint -> ONNX -> CoreML."""
    console.print(f"[green]export[/green] task={task} ckpt={checkpoint} out={out_dir}")
    raise typer.Exit(code=0)


@app.command()
def ingest(
    dataset: Annotated[str, typer.Argument(help="Dataset to ingest: plate | pour")],
    source: Annotated[str, typer.Option("--source", help="s3://bucket/prefix or local path")],
) -> None:
    """Pull a dataset from object storage into the local cache."""
    console.print(f"[green]ingest[/green] dataset={dataset} source={source}")
    raise typer.Exit(code=0)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
) -> None:
    """Run the FastAPI inference server (Phase 6)."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print("[red]install the `serve` extra: pip install '.[serve]'[/red]")
        raise typer.Exit(code=1) from exc

    settings = load_settings()
    uvicorn.run(
        "nutrilens_ml.serve.app:app",
        host=host or settings.serve_host,
        port=port or settings.serve_port,
        reload=False,
    )
