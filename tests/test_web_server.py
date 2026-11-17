"""Offline tests for LatentLensWebApp HTTP/WSGI routing and video streaming."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from latentlens.agent.harness import VideoAgentHarness
from latentlens.agent.model import ModelTurn
from latentlens.agent.tools import build_default_registry
from latentlens.config import Settings
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan
from latentlens.web.controller import WebController
from latentlens.web.server import LatentLensWebApp
from tests.conftest import ColorEncoder
from tests.test_agent_harness import ScriptedAgentModel
from tests.test_text_search import FakeTextEncoder


def _build_app(tmp_path: Path, settings: Settings) -> LatentLensWebApp:
    vid = tmp_path / "demo.mp4"
    vid.write_bytes(b"mp4-binary-stream")

    store = VectorStore(dim=3, encoder_name="fake")
    span = ClipSpan("demo", str(vid), 0.0, 4.0)
    store.add([span], np.ones((1, 3), dtype=np.float32))

    captions = CaptionStore(tmp_path / "captions.json")
    captions.add(span, "A dog catches a frisbee in the yard", "m")

    search = SearchService(
        encoder=ColorEncoder(),
        store=store,
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    harness = VideoAgentHarness(
        model=ScriptedAgentModel([ModelTurn(text="Dog catches frisbee [demo @ 00:00.0–00:04.0].")]),
        registry=build_default_registry(search, store, captions),
    )
    ctrl = WebController(
        store=store,
        captions=captions,
        search_service=search,
        agent_harness=harness,
    )
    return LatentLensWebApp(ctrl)


def test_web_app_routes_dashboard_stats_search_ask_and_video(
    tmp_path: Path, settings: Settings
) -> None:
    app = _build_app(tmp_path, settings)

    # 1. Dashboard HTML
    status, headers, body = app.handle_request("GET", "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert b"LatentLens" in body

    # 2. Stats API
    status, _, body = app.handle_request("GET", "/api/stats")
    assert status == 200
    assert json.loads(body)["total_clips"] == 1

    # 3. Search API
    status, _, body = app.handle_request("GET", "/api/search", "q=frisbee&k=2")
    assert status == 200
    assert len(json.loads(body)["hits"]) == 1

    status, _, _ = app.handle_request("GET", "/api/search", "")
    assert status == 400

    # 4. Ask API via WSGI __call__
    ask_payload = json.dumps({"question": "What did the dog catch?", "mode": "agent"}).encode()
    captured: list[str] = []
    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/api/ask",
        "CONTENT_LENGTH": str(len(ask_payload)),
        "wsgi.input": io.BytesIO(ask_payload),
    }
    chunks = app(environ, lambda s, h: captured.append(s))
    assert captured[0].startswith("200")
    parsed_ask = json.loads(chunks[0])
    assert "Dog catches frisbee" in parsed_ask["answer"]

    # Invalid ask payload
    status, _, _ = app.handle_request("POST", "/api/ask", body=b"{}")
    assert status == 400

    # 5. Video stream + 404s
    status, headers, body = app.handle_request("GET", "/api/video/demo")
    assert status == 200
    assert body == b"mp4-binary-stream"

    status, _, _ = app.handle_request("GET", "/api/video/unknown")
    assert status == 404
    status, _, _ = app.handle_request("GET", "/unknown/route")
    assert status == 404
