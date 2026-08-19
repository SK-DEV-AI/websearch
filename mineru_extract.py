"""C3: opt-in high-quality HTML extraction via MinerU-HTML (opendatalab).

SLM (Qwen3-0.6B / hunyuan0.5B-compact) main-content extractor for hard HTML.
CPU transformers backend only — vLLM needs 8 GB VRAM, this laptop has 4 GB.
PDF/DOCX stay on pdf_extract.py (docling) — mineru_html is HTML-only.

Lazy import: the model (~1.2 GB resident) only loads when opted in.
"""

from __future__ import annotations

from config import MINERU_DEVICE


async def extract_with_mineru(html: str, *, output_format: str = "markdown",
                              backend: str = "transformers",
                              device: str = "") -> dict:
    """Extract main content from *html* via MinerU-HTML on CPU."""
    try:
        from mineru_html import MinerUHTML  # heavy import — opt-in only
    except Exception as e:
        return {"success": False, "error": f"mineru_html unavailable: {e}",
                "method": "mineru_html"}
    try:
        ex = MinerUHTML(device=device or MINERU_DEVICE)
        out = ex.extract(html, output_format=output_format)
        if not out:
            return {"success": False, "error": "mineru returned empty",
                    "method": "mineru_html"}
        return {"success": True, "content": out, "method": "mineru_html"}
    except Exception as e:
        return {"success": False, "error": f"mineru extraction failed: {e}",
                "method": "mineru_html"}
