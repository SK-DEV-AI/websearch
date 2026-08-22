"""PDF extraction: PyMuPDF fast path (text layer) -> Docling OCR fallback (scanned).

Formats: markdown (default), json (docling export_to_dict / per-page text dict),
html (docling export_to_html / pymupdf per-page html). Comma-separated combos
(e.g. "markdown,json") return every requested format.
Note: docling 2.x removed export_to_tagged_pdf, so "tagged-pdf" is not offered.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import get_http_client
from security import safe_fetch, validate_url

__all__ = ["extract_pdf"]

_OCR_PAGE_BUDGET = 300  # seconds for a whole docling run
_OCR_INIT_BUDGET = 30  # seconds for docling init (import + converter build);
#                       # ONNX Runtime's C++ constructors can hang — donsetch ocr.rs guard
_OCR_MAX_PAGES = 25  # per-document OCR page cap; giant scans must not eat the budget
_PUA_FLOOR = 0.3  # PUA ratio above which a glyph stream is garbage (broken ToUnicode)
_FORMATS = ("markdown", "json", "html")
_EXT = {"markdown": "md", "json": "json", "html": "html"}


async def _download_pdf(url: str, dst: Path) -> Path:
    c = get_http_client()
    r = await safe_fetch(c, url, timeout=60)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return dst


def _pymupdf_open(path: str, password: str = ""):
    import pymupdf

    return pymupdf.open(path, password=password) if password else pymupdf.open(path)


def _pymupdf_extract(path: str, pages: str = "", password: str = "",
                     fmt: str = "markdown") -> tuple[str, int]:
    """Fast text-layer extraction. Returns (content, page_count); text empty => scanned."""
    doc = _pymupdf_open(path, password)
    try:
        total = doc.page_count
        wanted: list[int] = []
        if pages:
            sep = pages.replace(" ", "")
            for rng in sep.split(","):
                if not rng:
                    continue
                if "-" in rng:
                    a, _, b = rng.partition("-")
                    lo = max(1, int(a))
                    hi = min(total, int(b)) if b else total
                    wanted.extend(range(lo, hi + 1))
                else:
                    p = int(rng)
                    if 1 <= p <= total:
                        wanted.append(p)
        else:
            wanted = list(range(1, total + 1))
        if fmt == "markdown":
            parts = [doc[p - 1].get_text("text") for p in wanted]
            return "\n\n".join(parts).strip(), total
        if fmt == "json":
            obj = [{"page": p, "text": doc[p - 1].get_text("text")} for p in wanted]
            return json.dumps(obj, ensure_ascii=False), total
        if fmt == "html":
            parts = [doc[p - 1].get_text("html") for p in wanted]
            return "\n".join(parts).strip(), total
        raise ValueError(f"unsupported format: {fmt}")
    finally:
        doc.close()


def _pua_ratio(text: str) -> float:
    """Fraction of chars in the Private Use Area (broken ToUnicode maps).

    donsetch ocr.rs fusion-trust audit: a glyph stream full of PUA chars is
    garbage (broken encoding), not text — treat it as scanned and OCR it.
    """
    if not text:
        return 0.0
    bad = sum(1 for ch in text if 0xE000 <= ord(ch) <= 0xF8FF or 0xF0000 <= ord(ch) <= 0xFFFFD)
    return bad / len(text)


def _docling_build() -> Any:
    """Import docling + build the converter (slow; 30s init guard)."""
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

    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _docling_extract(conv: Any, path: str, fmts: list[str], max_pages: int = _OCR_MAX_PAGES) -> str | dict[str, str]:
    """Full OCR pipeline (CPU, rapidocr backend). Called only when no usable
    text layer exists (empty, PUA-garbage, or force_ocr).

    `conv` is the pre-built DocumentConverter (init guarded separately).
    Converts ONCE; every requested format is exported from the same in-memory
    document (a multi-format request previously re-ran the full OCR per format).
    The donsetch page cap is applied here: docs beyond max_pages get a
    pymupdf subset saved to a temp file so OCR cost stays bounded.
    """
    src = path
    tmp_subset = ""
    if max_pages:
        doc = _pymupdf_open(path)
        try:
            if doc.page_count > max_pages:
                doc.select(list(range(max_pages)))
                tmp_subset = path + ".subset.pdf"
                doc.save(tmp_subset, garbage=3, deflate=True)
                src = tmp_subset
        finally:
            doc.close()
    try:
        res = conv.convert(src)
        doc = res.document
        out: dict[str, str] = {}
        if "markdown" in fmts:
            out["markdown"] = (doc.export_to_markdown() or "").strip()
        if "json" in fmts:
            out["json"] = json.dumps(doc.export_to_dict(), ensure_ascii=False)
        if "html" in fmts:
            out["html"] = (doc.export_to_html() or "").strip()
        if len(fmts) == 1:
            return out[fmts[0]]
        return out
    finally:
        if tmp_subset and os.path.isfile(tmp_subset):
            try:
                os.remove(tmp_subset)
            except OSError:
                pass


def _formats_arg(raw: str) -> list[str]:
    fmts = [f.strip() for f in raw.split(",") if f.strip()]
    bad = [f for f in fmts if f not in _FORMATS]
    if bad:
        raise ValueError(f"unsupported format(s): {', '.join(bad)}; supported: {', '.join(_FORMATS)}")
    return fmts or ["markdown"]


async def extract_pdf(
    input_path: str | list[str],
    output_dir: str = "",
    format: str = "markdown",
    password: str = "",
    pages: str = "",
    hybrid: str = "",
    hybrid_mode: str = "",
) -> dict[str, Any]:
    sources = [input_path] if isinstance(input_path, str) else input_path
    local_files: list[str] = []
    tmpdir = ""
    out_tmpdir = ""
    force_ocr = bool(hybrid) or bool(hybrid_mode)
    results: dict[str, Any] = {}
    fmts = _formats_arg(format)
    try:
        for src in sources:
            src = src.strip()
            if not src:
                continue
            if os.path.isfile(src):
                local_files.append(src)
                continue
            if src.startswith("file://"):
                continue  # file:// not supported (SSRF risk)
            if src.startswith(("http://", "https://")):
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
                text, page_count = _pymupdf_extract(str(path), pages, password, "markdown")
                # No usable text layer (empty, PUA-garbage) or forced OCR:
                # donsetch fusion-trust audit — a PUA-heavy glyph stream is a
                # broken encoding (garbage), not text, so it must be OCR'd.
                if force_ocr or len(text) < 30 or _pua_ratio(text) > _PUA_FLOOR:
                    method = "docling-ocr"
                    # Init (import + converter build) has its own guard: ONNX
                    # Runtime C++ constructors can hang, and init failure should
                    # not burn the whole OCR budget (donsetch ocr.rs).
                    try:
                        conv = await asyncio.wait_for(
                            asyncio.to_thread(_docling_build), timeout=_OCR_INIT_BUDGET)
                    except asyncio.TimeoutError:
                        return {"success": False,
                                "error": f"OCR engine init timed out after {_OCR_INIT_BUDGET}s: {path.name}"}
                    content = await asyncio.wait_for(
                        asyncio.to_thread(
                            _docling_extract, conv, str(path), fmts, _OCR_MAX_PAGES),
                        timeout=_OCR_PAGE_BUDGET - _OCR_INIT_BUDGET)
                elif len(fmts) == 1:
                    if fmts[0] != "markdown":
                        text = (await asyncio.to_thread(
                            _pymupdf_extract, str(path), pages, password, fmts[0]))[0]
                    content = text
                else:
                    got: dict[str, str] = {}
                    for f in fmts:
                        got[f] = (await asyncio.to_thread(
                            _pymupdf_extract, str(path), pages, password, f))[0]
                    content = got
            except asyncio.TimeoutError:
                return {"success": False, "error": f"OCR timed out after {_OCR_PAGE_BUDGET}s: {path.name}"}
            except Exception as e:
                return {"success": False, "error": f"PDF extraction failed: {path.name}: {e}"}
            if isinstance(content, str):
                if len(content) > 50000:
                    content = content[:50000] + "\n\n[... truncated at 50000 chars ...]"
            else:
                for f, c in content.items():
                    if len(c) > 50000:
                        content[f] = c[:50000] + "\n\n[... truncated at 50000 chars ...]"
            results[path.name] = {
                "content": content,
                "method": method,
                "pages": page_count,
                "elapsed_s": round(time.monotonic() - t0, 1),
            }

        if output_dir.strip():
            out = Path(output_dir.strip())
            out.mkdir(parents=True, exist_ok=True)
        else:
            out_tmpdir = tempfile.mkdtemp(prefix="pdfx_out_")
            out = Path(out_tmpdir)
        for name, r in results.items():
            if isinstance(r["content"], str):
                (out / f"{Path(name).stem}.{_EXT[fmts[0]]}").write_text(r["content"], encoding="utf-8")
            else:
                for f, c in r["content"].items():
                    (out / f"{Path(name).stem}.{_EXT[f]}").write_text(c, encoding="utf-8")

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
