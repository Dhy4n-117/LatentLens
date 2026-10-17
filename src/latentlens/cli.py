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
    video: Annotated[
        Path | None,
        typer.Argument(help="Query video (optional if --text is provided)."),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", "-q", help="Search video clips using natural language text query."),
    ] = None,
    at: Annotated[float, typer.Option("--at", "-t", help="Query window start (seconds).")] = 0.0,
    k: Annotated[int, typer.Option("--top-k", "-k", min=1, help="Number of results.")] = 5,
    other_videos_only: Annotated[
        bool, typer.Option("--other-videos-only", help="Exclude the query video entirely.")
    ] = False,
) -> None:
    """Find similar moments from a video clip, or search clips with a text query."""
    settings = get_settings()
    from latentlens.encoders import SentenceTransformerTextEncoder
    from latentlens.index import CaptionStore
    from latentlens.pipeline import SearchService

    if video is None and not text:
        console.print(
            "[red]Error: You must provide either a query video or a text query (--text).[/]"
        )
        raise typer.Exit(1)

    captions = (
        CaptionStore.load(settings.captions_path) if settings.captions_path.exists() else None
    )

    if text:
        if captions is None or len(captions) == 0:
            console.print(
                "[yellow]No captions available to search. Run `latentlens caption` first.[/]"
            )
            raise typer.Exit(1)

        text_encoder = SentenceTransformerTextEncoder(
            settings.text_model_id, device=settings.device
        )
        encoder, store = _load_components(settings)
        service = SearchService(
            encoder=encoder,
            store=store,
            settings=settings,
            text_encoder=text_encoder,
            captions=captions,
        )

        with console.status("Searching clips by text..."):
            hits = service.search_by_text(text, k=k)

        title = f'Clips matching "{text}"'
    else:
        assert video is not None
        if not video.is_file():
            console.print(f"[red]Error: Query video file not found: {video}[/]")
            raise typer.Exit(1)

        encoder, store = _load_components(settings)
        if len(store) == 0:
            console.print("[yellow]Index is empty — run `latentlens ingest` first.[/]")
            raise typer.Exit(1)

        service = SearchService(
            encoder=encoder,
            store=store,
            settings=settings,
            captions=captions,
        )

        with console.status("Encoding query clip…"):
            hits = service.search_by_moment(
                video, at, k=k, include_same_video=not other_videos_only
            )

        title = f"Moments similar to {video.name} @ {format_timestamp(at)}"

    table = Table(title=title, header_style="bold magenta")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Video")
    table.add_column("Span")
    if any(h.caption for h in hits):
        table.add_column("Caption")

    for hit in hits:
        row = [
            str(hit.rank),
            f"{hit.score:.3f}",
            Path(hit.span.video_path).name,
            hit.span.label(),
        ]
        if any(h.caption for h in hits):
            row.append(hit.caption or "")
        table.add_row(*row)

    console.print(table if hits else "[yellow]No matches.[/]")


@app.command()
def caption(
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", min=1, help="Max clips to caption (free quota budget)."),
    ] = None,
) -> None:
    """Generate Gemini captions for clips in the index."""
    settings = get_settings()
    from latentlens.index import CaptionStore, VectorStore
    from latentlens.llm import GeminiCaptioner, RateLimiter, ResponseCache
    from latentlens.pipeline import CaptioningPipeline

    if not (settings.index_dir / "metadata.json").exists():
        console.print("[yellow]Index is empty — run `latentlens ingest` first.[/]")
        raise typer.Exit(1)

    api_key = settings.require_gemini_key()
    store = VectorStore.load(settings.index_dir)
    spans = store.spans()
    if not spans:
        console.print("[yellow]No clips found in index.[/]")
        raise typer.Exit()

    captions = CaptionStore.load(settings.captions_path)
    limiter = RateLimiter.per_minute(settings.gemini_rpm)
    cache = ResponseCache(settings.cache_dir)
    captioner = GeminiCaptioner(
        api_key=api_key,
        model=settings.gemini_model,
        limiter=limiter,
        cache=cache,
    )
    pipeline = CaptioningPipeline(captioner, captions, settings)

    with _progress() as progress:
        task = progress.add_task("Captioning clips...", total=None)

        def advance(done: int, total: int, t: TaskID = task) -> None:
            progress.update(t, completed=done, total=total)

        report = pipeline.run(spans, limit=limit, on_progress=advance)
        progress.remove_task(task)

    console.print(
        f"[green]Done![/] Captioned: [bold]{report.captioned}[/], "
        f"Already cached/done: [bold]{report.already_done}[/], "
        f"Failed: [bold]{report.failed}[/], "
        f"Missing video files: [bold]{report.missing_video}[/]"
    )
    console.print(f"Total captions in store: [bold]{len(captions)}[/] → {settings.captions_path}")


@app.command()
def stats() -> None:
    """Show what's in the index and caption store."""
    settings = get_settings()
    from latentlens.index import CaptionStore, VectorStore

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

    if settings.captions_path.exists():
        captions = CaptionStore.load(settings.captions_path)
        console.print(f"Total captions: [bold]{len(captions)}[/] → {settings.captions_path}")


@app.command()
def remove(
    video_id: Annotated[str, typer.Argument(help="ID shown by `latentlens stats`.")],
) -> None:
    """Remove a video's clips and captions from the index."""
    settings = get_settings()
    from latentlens.index import CaptionStore, VectorStore

    store = VectorStore.load(settings.index_dir)
    removed = store.remove_video(video_id)
    if not removed:
        console.print(f"[yellow]No clips found for {video_id}.[/]")
        raise typer.Exit(1)
    store.save(settings.index_dir)

    removed_caps = 0
    if settings.captions_path.exists():
        captions = CaptionStore.load(settings.captions_path)
        removed_caps = captions.remove_video(video_id)
        if removed_caps:
            captions.save()

    console.print(f"Removed [bold]{removed}[/] clips and [bold]{removed_caps}[/] captions.")


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
