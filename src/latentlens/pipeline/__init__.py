"""High-level workflows composed from video, encoder and index primitives."""

from latentlens.pipeline.ingest import IngestionPipeline, IngestReport
from latentlens.pipeline.search import SearchService

__all__ = ["IngestReport", "IngestionPipeline", "SearchService"]
