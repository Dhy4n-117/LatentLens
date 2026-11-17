"""Zero-dependency HTTP/WSGI web application for REST API and clip streaming."""

from __future__ import annotations

import json
import mimetypes
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from latentlens.web.controller import WebController
from latentlens.web.schemas import AskRequest
from latentlens.web.ui import render_dashboard_html


class LatentLensWebApp:
    """WSGI application serving the LatentLens UI, REST endpoints, and video files."""

    def __init__(self, controller: WebController) -> None:
        self._controller = controller

    def handle_request(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: bytes = b"",
    ) -> tuple[int, dict[str, str], bytes]:
        """Dispatch an HTTP request in memory and return ``(status, headers, body)``."""
        method_up = method.upper()
        clean_path = path.rstrip("/") or "/"

        if method_up == "GET" and clean_path == "/":
            html = render_dashboard_html(self._controller.get_stats())
            return 200, {"Content-Type": "text/html; charset=utf-8"}, html.encode("utf-8")

        if method_up == "GET" and clean_path == "/api/stats":
            stats = self._controller.get_stats()
            return _json_response(200, stats.model_dump())

        if method_up == "GET" and clean_path == "/api/search":
            params = parse_qs(query_string)
            q = (params.get("q") or [""])[0].strip()
            if not q:
                return _json_response(400, {"error": "Query parameter 'q' is required"})
            top_k = int((params.get("k") or ["5"])[0])
            hits = self._controller.search_text(q, top_k=top_k)
            return _json_response(200, {"hits": [h.model_dump() for h in hits]})

        if method_up == "POST" and clean_path == "/api/ask":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
                req = AskRequest.model_validate(payload)
                resp = self._controller.ask(req)
                return _json_response(200, resp.model_dump())
            except Exception as exc:
                return _json_response(400, {"error": str(exc)})

        if method_up == "GET" and clean_path.startswith("/api/video/"):
            video_id = clean_path.removeprefix("/api/video/")
            video_path = self._controller.resolve_video_path(video_id)
            if video_path is None:
                return _json_response(404, {"error": f"Video '{video_id}' not found"})
            mime, _ = mimetypes.guess_type(str(video_path))
            data = video_path.read_bytes()
            return (
                200,
                {
                    "Content-Type": mime or "video/mp4",
                    "Content-Length": str(len(data)),
                    "Accept-Ranges": "bytes",
                },
                data,
            )

        return _json_response(404, {"error": "Not found"})

    def __call__(self, environ: dict[str, Any], start_response: Any) -> list[bytes]:
        """Standard PEP 3333 WSGI entrypoint."""
        method = str(environ.get("REQUEST_METHOD", "GET"))
        path = str(environ.get("PATH_INFO", "/"))
        query_string = str(environ.get("QUERY_STRING", ""))
        content_len = int(environ.get("CONTENT_LENGTH") or 0)
        stream = environ.get("wsgi.input")
        body = stream.read(content_len) if stream and content_len > 0 else b""

        status_code, headers, payload = self.handle_request(method, path, query_string, body)
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status_code, "OK")
        start_response(f"{status_code} {reason}", list(headers.items()))
        return [payload]

    def serve_forever(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the blocking development HTTP server."""
        with make_server(host, port, self) as httpd:
            httpd.serve_forever()


def _json_response(status: int, data: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return status, {"Content-Type": "application/json; charset=utf-8"}, raw


__all__ = [
    "LatentLensWebApp",
]
