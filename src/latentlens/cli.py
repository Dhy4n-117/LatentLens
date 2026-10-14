"""``latentlens`` command-line interface."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from latentlens.config import Settings, get_settings
from latentlens.types import format_timestamp

if TYPE_CHECKING:
    from latentlens.encoders import VJEPAEncoder
    from latentlens.index import VectorStore

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

app = typer.Typer(
    name="latentlens",
    help="Video Q&A over the real world: V-JEPA 2 visual memory + Gemini.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# -- commands ---------------------------------------------------------------


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(exists=True, help="Video files or folders.")],
    force: Annotated[bool, typer.Option(help="Re-index videos already in the index.")] = False,
) -> None:
    """Embed videos with V-JEPA 2 and add them to the index."""
    settings = get_settings()
    videos = sorted(set(_discover(paths)))
    if not videos:
        console.print("[yellow]No video files found.[/]")
        raise typer.Exit(1)

    from latentlens.pipeline import IngestionPipeline

    encoder, store = _load_components(settings)
    pipeline = IngestionPipeline(encoder, store, settings)

    table = Table(title="Ingestion summary", header_style="bold cyan")
    for column in ("Video", "Duration", "Clips", "Time", "Status"):
        table.add_column(column)

    with _progress() as progress:
        for video in videos:
            task = progress.add_task(video.name, total=None)

            def advance(done: int, total: int, task: TaskID = task) -> None:
                progress.update(task, completed=done, total=total)

            report = pipeline.ingest(video, force=force, on_progress=advance)
            progress.remove_task(task)
            store.save(settings.index_dir)  # checkpoint after each video

            status = "[dim]skipped[/]" if report.skipped else "[green]indexed[/]"
            table.add_row(
                video.name,
                format_timestamp(report.duration_sec),
                str(report.num_clips),
                f"{report.elapsed_sec:.1f}s",
                status,
            )

    console.print(table)
    console.print(f"Index now holds [bold]{len(store)}[/] clips → {settings.index_dir}")


@app.command()
def search(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Query video.")],
    at: Annotated[float, typer.Option("--at", "-t", help="Query window start (seconds).")] = 0.0,
    k: Annotated[int, typer.Option("--top-k", "-k", min=1, help="Number of results.")] = 5,
    other_videos_only: Annotated[
        bool, typer.Option("--other-videos-only", help="Exclude the query video entirely.")
    ] = False,
) -> None:
    """Find moments that look and move like a given moment."""
    settings = get_settings()
    from latentlens.pipeline import SearchService

    encoder, store = _load_components(settings)
    if len(store) == 0:
        console.print("[yellow]Index is empty — run `latentlens ingest` first.[/]")
        raise typer.Exit(1)

    with console.status("Encoding query clip…"):
        hits = SearchService(encoder, store, settings).search_by_moment(
            video, at, k=k, include_same_video=not other_videos_only
        )

    table = Table(
        title=f"Moments similar to {video.name} @ {format_timestamp(at)}",
        header_style="bold magenta",
    )
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Video")
    table.add_column("Span")
    for hit in hits:
        table.add_row(
            str(hit.rank),
            f"{hit.score:.3f}",
            Path(hit.span.video_path).name,
            hit.span.label(),
        )
    console.print(table if hits else "[yellow]No matches.[/]")


@app.command()
def stats() -> None:
    """Show what's in the index."""
    settings = get_settings()
    from latentlens.index import VectorStore

    if not (settings.index_dir / "metadata.json").exists():
        console.print("[yellow]No index yet.[/]")
        raise typer.Exit()

    store = VectorStore.load(settings.index_dir)
    table = Table(title=f"{store.encoder_name}  ·  dim={store.dim}", header_style="bold cyan")
    table.add_column("Video ID")
    table.add_column("Path")
    table.add_column("Clips", justify="right")
    for video_id, (path, count) in sorted(store.videos().items(), key=lambda kv: kv[1][0]):
        table.add_row(video_id, path, str(count))
    console.print(table)
    console.print(f"Total clips: [bold]{len(store)}[/]")


@app.command()
def remove(
    video_id: Annotated[str, typer.Argument(help="ID shown by `latentlens stats`.")],
) -> None:
    """Remove a video's clips from the index."""
    settings = get_settings()
    from latentlens.index import VectorStore

    store = VectorStore.load(settings.index_dir)
    removed = store.remove_video(video_id)
    if not removed:
        console.print(f"[yellow]No clips found for {video_id}.[/]")
        raise typer.Exit(1)
    store.save(settings.index_dir)
    console.print(f"Removed [bold]{removed}[/] clips.")


# -- helpers ----------------------------------------------------------------


def _load_components(settings: Settings) -> tuple[VJEPAEncoder, VectorStore]:
    from latentlens.encoders import VJEPAEncoder
    from latentlens.index import IncompatibleIndexError, VectorStore

    encoder = VJEPAEncoder(settings.encoder_model_id, device=settings.device)
    try:
        store = VectorStore.open(settings.index_dir, dim=encoder.dim, encoder_name=encoder.name)
    except IncompatibleIndexError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
    return encoder, store


def _discover(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from (
                p.resolve() for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
            )
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path.resolve()


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


if __name__ == "__main__":
    app()
