"""Self-contained HTML5 dashboard with timestamped clip playback for LatentLens."""

from __future__ import annotations

from latentlens.web.schemas import IndexStatsResponse


def render_dashboard_html(stats: IndexStatsResponse) -> str:
    """Render the single-page web UI with video player, search, and grounded QA."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LatentLens — Grounded Video Q&amp;A</title>
  <style>
    :root {{ --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --text: #f8fafc; }}
    body {{
      font-family: system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1.5rem;
    }}
    .container {{ max-width: 1080px; margin: 0 auto; display: grid; gap: 1.25rem; }}
    .card {{ background: var(--card); border-radius: 10px; padding: 1.25rem; }}
    .row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }}
    input, select, button {{
      padding: 0.6rem 0.9rem;
      border-radius: 6px;
      border: 1px solid #334155;
      background: #0f172a;
      color: var(--text);
    }}
    input {{ flex: 1; min-width: 240px; }}
    button {{
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      cursor: pointer;
      border: none;
    }}
    video {{ width: 100%; max-height: 420px; background: #000; border-radius: 8px; }}
    .cite-pill {{
      display: inline-block;
      margin: 0.25rem;
      padding: 0.3rem 0.65rem;
      border-radius: 999px;
      background: #0284c7;
      color: #fff;
      cursor: pointer;
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>LatentLens — V-JEPA 2 + Gemini Video Q&amp;A</h1>
      <p>Encoder: <strong>{stats.encoder_name}</strong> (dim={stats.dim}) &middot;
         Clips: <strong>{stats.total_clips}</strong> &middot;
         Captions: <strong>{stats.total_captions}</strong></p>
    </div>
    <div class="card">
      <h2>Ask or Search Videos</h2>
      <div class="row">
        <input id="query-input" type="text" placeholder="Ask a question about your videos..." />
        <select id="mode-select">
          <option value="agent">Agent Harness (Multi-Step)</option>
          <option value="rag">Single-Pass Grounded QA</option>
        </select>
        <button onclick="askQuestion()">Ask</button>
      </div>
      <div id="answer-box" style="margin-top: 1rem; white-space: pre-wrap;"></div>
      <div id="citations-box" style="margin-top: 0.75rem;"></div>
    </div>
    <div class="card">
      <h2>Clip Player</h2>
      <video id="clip-player" controls preload="metadata"></video>
    </div>
  </div>
  <script>
    function playClip(mediaUrl, startSec) {{
      const player = document.getElementById('clip-player');
      player.src = mediaUrl;
      player.currentTime = startSec || 0;
      player.play().catch(() => {{}});
    }}
    async function askQuestion() {{
      const q = document.getElementById('query-input').value.trim();
      const mode = document.getElementById('mode-select').value;
      if (!q) return;
      document.getElementById('answer-box').textContent = 'Reasoning over video memory...';
      const res = await fetch('/api/ask', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ question: q, mode: mode }})
      }});
      const data = await res.json();
      document.getElementById('answer-box').textContent = data.answer || data.error || '';
      const box = document.getElementById('citations-box');
      box.innerHTML = '';
      (data.citations || []).forEach(c => {{
        const btn = document.createElement('span');
        btn.className = 'cite-pill';
        btn.textContent = c.video_name + ' @ ' + c.label;
        btn.onclick = () => playClip(c.media_url, c.start_sec);
        box.appendChild(btn);
      }});
    }}
  </script>
</body>
</html>"""


__all__ = [
    "render_dashboard_html",
]
