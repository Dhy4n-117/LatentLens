"""High-level workflows composed from video, encoder and index primitives."""

from latentlens.pipeline.caption import CaptioningPipeline, CaptionReport
from latentlens.pipeline.fusion import reciprocal_rank_fusion
from latentlens.pipeline.ingest import IngestionPipeline, IngestReport
from latentlens.pipeline.qa import WEAK_EVIDENCE_REFUSAL, GroundedQAPipeline
from latentlens.pipeline.search import SearchService

__all__ = [
    "WEAK_EVIDENCE_REFUSAL",
    "CaptionReport",
    "CaptioningPipeline",
    "GroundedQAPipeline",
    "IngestReport",
    "IngestionPipeline",
    "SearchService",
    "reciprocal_rank_fusion",
]
