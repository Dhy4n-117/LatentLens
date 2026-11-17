"""Tests for HTML dashboard rendering."""

from __future__ import annotations

from latentlens.web.schemas import IndexStatsResponse
from latentlens.web.ui import render_dashboard_html


def test_render_dashboard_html_embeds_stats_and_player() -> None:
    stats = IndexStatsResponse(
        encoder_name="vjepa2-vitl",
        dim=1024,
        total_clips=42,
        total_captions=40,
        videos=[{"video_id": "v1", "path": "v1.mp4", "clips": 42}],
    )
    html = render_dashboard_html(stats)
    assert "<!DOCTYPE html>" in html
    assert "vjepa2-vitl" in html
    assert "42" in html
    assert 'id="clip-player"' in html
    assert "/api/ask" in html
