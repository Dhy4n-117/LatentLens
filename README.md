# LatentLens

**Video Q&A over the real world**: [V-JEPA 2](https://ai.meta.com/vjepa/) provides the visual memory and Gemini handles the language reasoning.
Every part of the stack is free: open-weight models, open-source libraries, and the Gemini free tier.

> **Status:** Phase 1 (Visual memory core) is complete. Phase 2 (Language bridge) is in progress: the free-tier foundations (rate limiter, response cache, keyframe encoding) are done. See [ROADMAP.md](ROADMAP.md).

## Why V-JEPA?
V-JEPA learns by predicting *masked regions in latent space* instead of reconstructing pixels.
The result is embeddings that capture **motion and physical dynamics**, not just appearance.
That suits "what is happening" retrieval, and sending V-JEPA vectors to the LLM is far cheaper than sending raw video.

## Architecture (Phase 1)

```mermaid
flowchart LR
    F[Video file] --> R["VideoReader<br/>single-pass streaming decode"]
    P["plan_windows<br/>sliding windows"] --> R
    R -->|"(T,H,W,3) clips"| E["VJEPAEncoder<br/>mean-pooled patch tokens"]
    E -->|"unit vectors"| S[("VectorStore<br/>FAISS + metadata")]
    Q["Query moment"] --> E2[VJEPAEncoder] --> S --> H["Ranked SearchHits"]
```

```
src/latentlens/
├── config.py          # pydantic-settings, LATENTLENS_* env vars
├── types.py           # ClipSpan, SearchHit
├── video/
│   ├── segmenter.py   # pure window planning (no I/O)
│   └── reader.py      # streaming OpenCV decoder + content fingerprint
├── encoders/
│   ├── base.py        # VideoEncoder protocol
│   └── vjepa.py       # V-JEPA 2 (Hugging Face transformers)
├── index/store.py     # FAISS store, atomic persistence, compat checks
├── pipeline/
│   ├── ingest.py      # idempotent ingestion
│   └── search.py      # moment → similar moments
└── cli.py             # Typer + Rich
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev]"

latentlens ingest path/to/videos/                    # first run downloads V-JEPA 2 (~1.3 GB)
latentlens stats
latentlens search path/to/videos/cooking.mp4 --at 42 -k 5
latentlens search clip.mp4 --at 10 --other-videos-only
```

Configure the tool through environment variables or `.env` (see [.env.example](.env.example)).
On CPU, ViT-L runs at roughly 1–3 s per clip. A free Colab or Kaggle GPU is about 20× faster.

## Development

```bash
pytest          # runs fully offline: fake encoder + synthetic videos
ruff check src tests
```

## Design notes
- **Single-pass decoding.** Overlapping windows share frames. Each frame is decoded once and routed to every window that needs it, and a window is released as soon as it is complete.
- **Idempotent ingestion.** Videos are identified by a content fingerprint, not by path. Renamed files are not re-indexed, and `--force` rebuilds them.
- **Embedding-space safety.** The index records the encoder name and dimension, so it refuses to mix vectors from different models.
- **Crash-safe persistence.** Writes are atomic (temp file + rename), with a checkpoint after every video.
- **Swappable backbones.** Anything that implements the `VideoEncoder` protocol plugs in. Phase 4 adds a CLIP baseline this way.

## License
MIT
