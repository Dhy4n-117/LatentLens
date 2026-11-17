"""Web API and interactive UI package for LatentLens."""

from latentlens.web.controller import WebController
from latentlens.web.schemas import (
    AskRequest,
    AskResponse,
    ClipPlaybackInfo,
    IndexStatsResponse,
    SearchHitSchema,
)
from latentlens.web.server import LatentLensWebApp
from latentlens.web.ui import render_dashboard_html

__all__ = [
    "AskRequest",
    "AskResponse",
    "ClipPlaybackInfo",
    "IndexStatsResponse",
    "LatentLensWebApp",
    "SearchHitSchema",
    "WebController",
    "render_dashboard_html",
]
