"""PDF extraction: PyMuPDF fast path (text layer) -> Docling OCR fallback (scanned)."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import get_http_client
from security import validate_url

__all__ = ["extract_pdf"]

_OCR_PAGE_BUDGET = 300  # seconds for a whole docling run


async def _download_pdf(url: str, dst: Path) -> Path:
    url = await validate_url(url)
    c = get_http_client()
    r = await c.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return dst


def _pymupdf_open(path: str, password: str = ""):
    import pymupdf

    return pymupdf.open(path, password=password) if password else pymupdf.open(path)


def _pymupdf_extract(path: str, pages: str = "", password: str = "") -> tuple[str, int]:
    """Fast text-layer extraction. Returns (text, page_count); text empty => scanned."""
    doc = _pymupdf_open(path, password)
    try:
        total = doc.page_count
        parts: list[str] = []
        if pages:
            sep = pages.replace(" ", "")
            for rng in sep.split(","):
                if not rng:
                    continue
                if "-" in rng:
                    a, _, b = rng.partition("-")
                    lo = max(1, int(a))
                    hi = min(total, int(b)) if b else total
                    for p in range(lo, hi + 1):
                        parts.append(doc[p - 1].get_text("text"))
                else:
                    p = int(rng)
                    if 1 <= p <= total:
                        parts.append(doc[p - 1].get_text("text"))
        else:
            for i in range(total):
                parts.append(doc[i].get_text("text"))
        text = "\n\n".join(parts).strip()
        return text, total
    finally:
        doc.close()


def _docling_extract(path: str) -> str:
    """Full OCR pipeline (CPU, rapidocr backend). Called only when no text layer exists."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
    )

    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=2)
    opts.do_table_structure = False
    opts.do_formula_enrichment = False
    opts.do_code_enrichment = False
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.ocr_options = RapidOcrOptions()

    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    res = conv.convert(path)
    md = res.document.export_to_markdown()
    return (md or "").strip()


async def extract_pdf(
    input_path: str | list[str],
    output_dir: str = "",
    format: str = "markdown",
    password: str = "",
    quiet: bool = True,
    sanitize: bool = False,
    keep_line_breaks: bool = False,
    pages: str = "",
    hybrid: str = "",
    hybrid_mode: str = "",
    hybrid_url: str = "",
    hybrid_timeout: str = "",
    table_method: str = "",
    reading_order: str = "",
    image_output: str = "",
    image_format: str = "",
    include_header_footer: bool = False,
    detect_strikethrough: bool = False,
    markdown_with_html: bool = False,
    use_struct_tree: bool = False,
    content_safety_off: str = "",
    threads: str = "",
    replace_invalid_chars: str = "",
) -> dict[str, Any]:
    sources = [input_path] if isinstance(input_path, str) else input_path
    local_files: list[str] = []
    tmpdir = ""
    out_tmpdir = ""
    force_ocr = bool(hybrid) or bool(hybrid_mode)
    results: dict[str, Any] = {}
    try:
        for src in sources:
            src = src.strip()
            if not src:
                continue
            if os.path.isfile(src):
                local_files.append(src)
                continue
            if src.startswith(("http://", "https://", "file://")):
                if not tmpdir:
                    tmpdir = tempfile.mkdtemp(prefix="pdfx_")
                fname = os.path.basename(urlsplit(src).path) or f"download_{len(local_files)}.pdf"
                if not fname.lower().endswith(".pdf"):
                    fname += ".pdf"
                dl = Path(tmpdir) / fname
                await _download_pdf(src, dl)
                local_files.append(str(dl))
                continue
            local_files.append(src)

        if not local_files:
            return {"success": False, "error": "No valid PDF files provided"}

        for fp in local_files:
            path = Path(fp)
            t0 = time.monotonic()
            method = "pymupdf"
            try:
                text, page_count = _pymupdf_extract(str(path), pages, password)
                # Below a threshold the PDF is scanned/image-only: no text layer.
                if force_ocr or len(text) < 30:
                    text = await asyncio.wait_for(
                        asyncio.to_thread(_docling_extract, str(path)),
                        timeout=_OCR_PAGE_BUDGET)
                    method = "docling-ocr"
            except asyncio.TimeoutError:
                return {"success": False, "error": f"OCR timed out after {_OCR_PAGE_BUDGET}s: {path.name}"}
            except Exception as e:
                return {"success": False, "error": f"PDF extraction failed: {path.name}: {e}"}
            if len(text) > 50000:
                text = text[:50000] + "\n\n[... truncated at 50000 chars ...]"
            results[path.name] = {
                "content": text,
                "method": method,
                "pages": page_count,
                "elapsed_s": round(time.monotonic() - t0, 1),
            }

        if output_dir.strip():
            out = Path(output_dir.strip())
            out.mkdir(parents=True, exist_ok=True)
            for name, r in results.items():
                (out / f"{Path(name).stem}.md").write_text(r["content"], encoding="utf-8")
        else:
            out_tmpdir = tempfile.mkdtemp(prefix="pdfx_out_")
            out = Path(out_tmpdir)
            for name, r in results.items():
                (out / f"{Path(name).stem}.md").write_text(r["content"], encoding="utf-8")

        return {
            "success": True,
            "files": str(out),
            "results": results,
            "input_count": len(local_files),
        }
    except Exception as e:
        return {"success": False, "error": f"PDF extraction failed: {e}"}
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        if out_tmpdir and os.path.isdir(out_tmpdir):
            shutil.rmtree(out_tmpdir, ignore_errors=True)