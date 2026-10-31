"""PDF extraction: PyMuPDF fast path (text layer) -> Docling OCR fallback (scanned).

Formats: markdown (default), json (docling export_to_dict / per-page text dict),
html (docling export_to_html / pymupdf per-page html). Comma-separated combos
(e.g. "markdown,json") return every requested format.
Note: docling 2.x removed export_to_tagged_pdf, so "tagged-pdf" is not offered.

OCR engine (env WEBSEARCH_OCR_ENGINE): "pymupdf" (default) uses PyMuPDF's own
tesseract bridge — ~8-11 s/page and ~400 MB RSS, no model download, tesseract
already installed — while "docling" uses the RT-DETR layout + RapidOCR stack
(~100-250 s/page cold, ~1 GB RSS) which is the only path that gives reading
order and table structure on scans. Residual scans still fall back to docling
when the fast path yields no text.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import get_http_client
from security import safe_stream, validate_url

__all__ = ["extract_pdf"]

_PDF_MAX_BYTES = 50 * 1024 * 1024  # 50MB cap: unbounded r.content was a zip-bomb vector

# Text cap per document. 50K was a context-bomb guard, but it also made long
# reports unreachable: `fetch` paginates INSIDE this string, so anything past
# the cap could never be paged to, no matter what offset the caller passed.
# 100K still bounds memory (a 100K string is nothing) and keeps whole papers.
_PDF_TEXT_MAX = int(os.environ.get("WEBSEARCH_PDF_MAX_CHARS", "100000") or 0) or 50000

_OCR_PAGE_BUDGET = 300  # seconds for a whole docling run
_OCR_INIT_BUDGET = 30  # seconds for docling init (import + converter build);
#                       # ONNX Runtime's C++ constructors can hang — donsetch ocr.rs guard
_OCR_MAX_PAGES = 25  # per-document OCR page cap; giant scans must not eat the budget
_PUA_FLOOR = 0.3  # PUA ratio above which a glyph stream is garbage (broken ToUnicode)
_FORMATS = ("markdown", "json", "html")
_EXT = {"markdown": "md", "json": "json", "html": "html"}

# OCR engine selection (2026-10-27 research). PyMuPDF ships a tesseract bridge
# (page.get_textpage_ocr) that needs no Python ML stack: measured 8-11 s/page
# and ~400 MB RSS on this box, versus docling's 100-250 s/page cold at ~1 GB.
# tesseract 5.5.3 + /usr/share/tessdata are already installed, so the fast
# engine adds zero dependencies. docling stays available (and is still the
# fallback for scans the fast path cannot read) because it is the only engine
# that recovers reading order and tables.
_OCR_ENGINE = (os.environ.get("WEBSEARCH_OCR_ENGINE", "pymupdf") or "pymupdf").strip().lower()
_OCR_DPI = int(os.environ.get("WEBSEARCH_OCR_DPI", "200") or 200)
_OCR_LANG = (os.environ.get("WEBSEARCH_OCR_LANG", "eng") or "eng").strip()
# RapidOCR (docling) takes full language names; our fast engine takes ISO codes.
_OCR_LANG_DOCLING = (os.environ.get("WEBSEARCH_OCR_LANG_DOCLING", "english") or "english").strip()
_OCR_MIN_FREE_MB_FAST = int(os.environ.get("WEBSEARCH_OCR_MIN_FREE_MB_PYMUPDF", "400") or 0)
# Docling fallback when the fast engine returns nothing readable (a scan whose
# text tesseract cannot segment). On by default; disable with
# WEBSEARCH_OCR_FALLBACK=0 to keep OCR to one engine.
_OCR_FALLBACK = (os.environ.get("WEBSEARCH_OCR_FALLBACK", "1") or "1").strip() not in ("", "0", "off", "false")

# OCR resources (2026-10-27 incident). Three separate problems, one cause: the
# docling stack (RT-DETR layout + RapidOCR) used to be built per call on the
# event loop's shared default executor, so (a) every OCR reloaded the models,
# (b) a long OCR occupied a thread every other tool queues behind, and (c) the
# ~1 GB of models stayed resident with their threads. Now: one cached
# converter, a dedicated single-slot executor, and an idle unload.
_OCR_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
_OCR_CONVERTER: Any = None
_OCR_LAST_USE = 0.0
_OCR_IDLE_UNLOAD = float(os.environ.get("WEBSEARCH_OCR_IDLE_UNLOAD", "90") or 0)
_OCR_MIN_FREE_MB = int(os.environ.get("WEBSEARCH_OCR_MIN_FREE_MB", "900") or 0)
_CONVERTER_LOCK = asyncio.Lock()


def _mem_available_mb() -> int:
    """MemAvailable from /proc/meminfo, or 0 when it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _drop_converter() -> None:
    """Release the docling models so their RAM returns to the OS."""
    global _OCR_CONVERTER
    if _OCR_CONVERTER is None:
        return
    _OCR_CONVERTER = None
    gc.collect()


async def _ocr_converter() -> Any:
    """Build the converter once, on the dedicated OCR thread."""
    global _OCR_CONVERTER, _OCR_LAST_USE
    async with _CONVERTER_LOCK:
        if _OCR_CONVERTER is None:
            loop = asyncio.get_running_loop()
            _OCR_CONVERTER = await loop.run_in_executor(_OCR_EXEC, _docling_build)
        _OCR_LAST_USE = time.monotonic()
        return _OCR_CONVERTER


def _schedule_ocr_unload() -> None:
    """Drop the models once OCR has been idle for a while."""
    if _OCR_IDLE_UNLOAD <= 0:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _later() -> None:
        await asyncio.sleep(_OCR_IDLE_UNLOAD)
        if time.monotonic() - _OCR_LAST_USE >= _OCR_IDLE_UNLOAD:
            _drop_converter()

    try:
        loop.create_task(_later())
    except RuntimeError:
        pass


async def _download_pdf(url: str, dst: Path) -> Path:
    c = get_http_client()
    body = await safe_stream(c, url, max_bytes=_PDF_MAX_BYTES, timeout=60)
    dst.write_bytes(body)
    return dst


def _pymupdf_open(path: str, password: str = ""):
    import pymupdf

    doc = pymupdf.open(path, password=password) if password else pymupdf.open(path)
    # An encrypted PDF that opens but yields no text used to fall through to the
    # OCR path, which cannot use the password at all, so the caller got a
    # confusing OCR failure instead of "this file needs a password".
    if doc.needs_pass and not password:
        doc.close()
        raise ValueError("PDF is encrypted; pass password=... to extract it")
    return doc


@contextlib.contextmanager
def _quiet():
    """Keep PyMuPDF's advisory prints off stdout.

    This module runs inside an MCP stdio server, where stdout IS the JSON-RPC
    channel: one stray ``print`` from the C layer corrupts the frame stream.
    PyMuPDF does write things like "Consider using the pymupdf_layout package"
    to sys.stdout, so every PyMuPDF call site runs inside this guard.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _pymupdf_extract(path: str, pages: str = "", password: str = "",
                     fmt: str = "markdown") -> tuple[str, int]:
    """Fast text-layer extraction. Returns (content, page_count); text empty => scanned.

    Wrapped in _quiet(): PyMuPDF's C layer prints advisories to sys.stdout, and
    sys.stdout is the MCP wire in this process.
    """
    with _quiet():
        return _pymupdf_extract_inner(path, pages, password, fmt)


def _pymupdf_extract_inner(path: str, pages: str = "", password: str = "",
                           fmt: str = "markdown") -> tuple[str, int]:
    doc = _pymupdf_open(path, password)
    try:
        total = doc.page_count
        wanted = _page_numbers(total, pages)
        if fmt == "markdown":
            # sort=True restores reading order on multi-column pages; without it
            # PyMuPDF emits blocks in stream order and two-column text interleaves.
            parts = [doc[p - 1].get_text("text", sort=True) for p in wanted]
            return "\n\n".join(parts).strip(), total
        if fmt == "json":
            obj = [{"page": p, "text": doc[p - 1].get_text("text", sort=True)} for p in wanted]
            return json.dumps(obj, ensure_ascii=False), total
        if fmt == "html":
            parts = [doc[p - 1].get_text("html") for p in wanted]
            return "\n".join(parts).strip(), total
        raise ValueError(f"unsupported format: {fmt}")
    finally:
        doc.close()


def _page_numbers(total: int, pages: str = "") -> list[int]:
    """Parse a ``pages`` spec ("1-3,7") into 1-based page numbers."""
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
        return wanted
    return list(range(1, total + 1))


def _pymupdf_ocr(path: str, fmts: list[str], pages: str = "", password: str = "",
                 max_pages: int = _OCR_MAX_PAGES, dpi: int = _OCR_DPI,
                 lang: str = _OCR_LANG,
                 budget_s: float = _OCR_PAGE_BUDGET) -> tuple[str | dict[str, str], dict]:
    """OCR with PyMuPDF's built-in tesseract bridge (fast, no ML stack).

    ``full=True`` is deliberate: the partial mode only treats ``U+FFFD`` as
    illegible, so the Private-Use-Area glyph streams that send us here would
    not be re-OCR'd at all (2026-10-27 research, PyMuPDF 1.28.2).

    Pages are OCR'd one at a time so a giant scan degrades into "we read the
    first N pages and said so" instead of one 300 s timeout that returns
    nothing. Returns (content, meta) where meta records how much was read.
    """
    with _quiet():
        import pymupdf

        t0 = time.monotonic()
        doc = _pymupdf_open(path, password)
        per_page: list[str] = []
        read = 0
        stop = ""
        try:
            total = doc.page_count
            wanted = _page_numbers(total, pages)
            if max_pages and len(wanted) > max_pages:
                wanted = wanted[:max_pages]
                stop = f"page cap ({max_pages} pages)"
            for p in wanted:
                if time.monotonic() - t0 > budget_s:
                    stop = f"time budget ({int(budget_s)}s)"
                    break
                tp = doc[p - 1].get_textpage_ocr(full=True, dpi=dpi, language=lang)
                if "html" in fmts and len(fmts) == 1:
                    per_page.append(tp.extractHTML())
                else:
                    per_page.append(tp.extractText(sort=True))
                read += 1
            meta = {"pages_ocr": read, "pages_total": total,
                    "ocr_engine": "pymupdf-tesseract", "dpi": dpi,
                    "lang": lang}
            if stop:
                meta["ocr_stopped_at"] = stop
            text = "\n\n".join(per_page).strip()
            if len(fmts) == 1:
                return text, meta
            out = {}
            # One OCR pass serves every requested format: markdown/json carry the
            # plain text, html carries the tesseract HTML rendering.
            for f in fmts:
                if f == "html":
                    out[f] = "\n".join(per_page).strip()
                elif f == "json":
                    out[f] = json.dumps(
                        [{"page": i + 1, "text": t} for i, t in enumerate(per_page)],
                        ensure_ascii=False)
                else:
                    out[f] = text
            return out, meta
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
    # Table structure is OPT-IN: the TableFormer weights add real RSS on top of
    # the ~1 GB RapidOCR/layout set, and this box is memory-tight (the OCR guard
    # already refuses under 900 MB free). Digital PDFs never reach this path.
    opts.do_table_structure = (os.environ.get("WEBSEARCH_OCR_TABLE_STRUCTURE", "0") or "0").strip() not in ("", "0", "off", "false")
    opts.do_formula_enrichment = False
    opts.do_code_enrichment = False
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.ocr_options = RapidOcrOptions(
        lang=[_OCR_LANG_DOCLING],
        # Orientation classification is a whole extra model pass per page for no
        # gain on upright documents.
        use_cls=False,
    )

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
    # Docling logs through rich/print on stdout; same wire-corruption risk.
    with _quiet():
        return _docling_extract_inner(conv, path, fmts, max_pages)


def _docling_extract_inner(conv: Any, path: str, fmts: list[str],
                           max_pages: int = _OCR_MAX_PAGES) -> str | dict[str, str]:
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


def _ocr_chars(content: str | dict[str, str] | None) -> int:
    """Readable characters produced by an OCR pass (0 means it found nothing)."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    return sum(len(v) for v in content.values() if isinstance(v, str))


async def _ocr_fast(path: Path, fmts: list[str], pages: str,
                    password: str) -> dict:
    """Run the PyMuPDF/tesseract engine on the single-slot OCR thread."""
    free_mb = _mem_available_mb()
    if _OCR_MIN_FREE_MB_FAST > 0 and 0 < free_mb < _OCR_MIN_FREE_MB_FAST:
        return {"ok": False, "payload": {
            "success": False, "ocr_skipped": "low_memory",
            "free_mb": free_mb, "need_mb": _OCR_MIN_FREE_MB_FAST,
            "error": (f"OCR skipped: {free_mb} MB RAM available, below "
                      f"WEBSEARCH_OCR_MIN_FREE_MB_PYMUPDF="
                      f"{_OCR_MIN_FREE_MB_FAST}. Free memory and retry, or "
                      f"set it to 0 to force OCR.")}}
    loop = asyncio.get_running_loop()
    content, meta = await asyncio.wait_for(
        loop.run_in_executor(_OCR_EXEC, _pymupdf_ocr, str(path), fmts,
                             pages, password),
        timeout=_OCR_PAGE_BUDGET)
    return {"ok": True, "content": content, "meta": meta,
            "method": "pymupdf-ocr"}


async def _ocr_docling(path: Path, fmts: list[str]) -> dict:
    """Run the docling (RT-DETR + RapidOCR) engine. ~10-25x slower than fast."""
    # Memory guard: docling loads ~1 GB of models into THIS process. On a box
    # with ~1 GB available and swap already heavy, that turns every other
    # request into a swap-thrash victim (2026-10-27 incident).
    free_mb = _mem_available_mb()
    if _OCR_MIN_FREE_MB > 0 and 0 < free_mb < _OCR_MIN_FREE_MB:
        return {"ok": False, "payload": {
            "success": False, "ocr_skipped": "low_memory",
            "free_mb": free_mb, "need_mb": _OCR_MIN_FREE_MB,
            "error": (f"OCR skipped: {free_mb} MB RAM available, below "
                      f"WEBSEARCH_OCR_MIN_FREE_MB={_OCR_MIN_FREE_MB}. The "
                      f"docling OCR engine loads ~1 GB of models and would "
                      f"stall every other request. Free memory and retry, or "
                      f"set WEBSEARCH_OCR_MIN_FREE_MB=0 to force it.")}}
    try:
        conv = await asyncio.wait_for(_ocr_converter(), timeout=_OCR_INIT_BUDGET)
    except asyncio.TimeoutError:
        return {"ok": False, "payload": {
            "success": False,
            "error": f"OCR engine init timed out after {_OCR_INIT_BUDGET}s: {path.name}"}}
    loop = asyncio.get_running_loop()
    content = await asyncio.wait_for(
        loop.run_in_executor(_OCR_EXEC, _docling_extract, conv, str(path),
                             fmts, _OCR_MAX_PAGES),
        timeout=_OCR_PAGE_BUDGET - _OCR_INIT_BUDGET)
    _schedule_ocr_unload()
    return {"ok": True, "content": content, "meta": {},
            "method": "docling-ocr"}


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
    skipped: list[str] = []
    tmpdir = ""
    out_tmpdir = ""
    force_ocr = bool(hybrid) or bool(hybrid_mode)
    results: list[dict[str, Any]] = []
    failed: dict[str, Any] = {}
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
                skipped.append(f"{src} (file:// URLs not supported — pass a local path instead)")
                continue
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
            if skipped:
                return {"success": False,
                        "error": "No valid PDF files provided. Skipped: " + "; ".join(skipped)}
            return {"success": False, "error": "No valid PDF files provided"}

        for fp in local_files:
            path = Path(fp)
            t0 = time.monotonic()
            method = "pymupdf"
            ocr_ms = 0
            try:
                text, page_count = _pymupdf_extract(str(path), pages, password, "markdown")
                # No usable text layer (empty, PUA-garbage) or forced OCR:
                # donsetch fusion-trust audit — a PUA-heavy glyph stream is a
                # broken encoding (garbage), not text, so it must be OCR'd.
                if force_ocr or len(text) < 30 or _pua_ratio(text) > _PUA_FLOOR:
                    t_ocr = time.monotonic()
                    ocr_meta: dict = {}
                    run: dict = {"ok": True, "content": None, "meta": {}, "method": ""}
                    # Fast engine first unless the caller pinned docling. A scan
                    # tesseract cannot segment (<30 readable chars) falls through
                    # to docling, which recovers layout for exactly those pages.
                    if _OCR_ENGINE != "docling":
                        run = await _ocr_fast(path, fmts, pages, password)
                        if not run["ok"]:
                            failed[path.name] = run["payload"]
                            continue
                        method = run["method"]
                        ocr_meta = run["meta"]
                    if ((not run["ok"] or _ocr_chars(run["content"]) < 30)
                            and (_OCR_ENGINE == "docling" or _OCR_FALLBACK)):
                        d = await _ocr_docling(path, fmts)
                        if not d["ok"]:
                            if run["ok"] and _ocr_chars(run["content"]) > 0:
                                pass  # keep the partial fast result below
                            else:
                                failed[path.name] = d["payload"]
                                continue
                        elif run["ok"] and _ocr_chars(d["content"]) >= _ocr_chars(run["content"]):
                            # Fast engine saw almost nothing and docling found
                            # more: docling wins, but say we fell back.
                            ocr_meta = dict(run["meta"] or {})
                            ocr_meta["ocr_fallback_from"] = "pymupdf-tesseract"
                            run, method = d, d["method"]
                        elif not run["ok"]:
                            run, method = d, d["method"]
                    content = run["content"]
                    ocr_meta.update(run["meta"] or {})
                    ocr_ms = int((time.monotonic() - t_ocr) * 1000)
                    if _ocr_chars(content) < 30:
                        # Honest failure: an empty OCR result must not be
                        # presented as a successful extraction.
                        failed[path.name] = {"success": False,
                                "ocr_engine": ocr_meta.get("ocr_engine", method),
                                "error": (f"OCR produced no readable text after "
                                          f"{ocr_ms} ms: {path.name}. The page is "
                                          f"probably a blank/photo scan.")}
                        continue
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
                failed[path.name] = {"success": False, "error": f"OCR timed out after {_OCR_PAGE_BUDGET}s: {path.name}"}
                continue
            except Exception as e:
                failed[path.name] = {"success": False, "error": f"PDF extraction failed: {path.name}: {e}"}
                continue
            truncated = False
            _cap = _PDF_TEXT_MAX
            if isinstance(content, str):
                if len(content) > _cap:
                    truncated = True
                    content = content[:_cap] + f"\n\n[... truncated at {_cap} chars ...]"
            else:
                for f, c in (content.items() if isinstance(content, dict) else []):
                    if isinstance(c, str) and len(c) > _cap:
                        truncated = True
                        content[f] = c[:_cap] + f"\n\n[... truncated at {_cap} chars ...]"
            # Normalize page-break artifact blank lines: collapse 3+ newlines
            # to a single blank line (paragraph structure preserved).
            if isinstance(content, str):
                content = re.sub(r"\n{3,}", "\n\n", content)
            elif isinstance(content, dict):
                for _f, _c in list(content.items()):
                    if isinstance(_c, str):
                        content[_f] = re.sub(r"\n{3,}", "\n\n", _c)
            entry = {
                "filename": path.name,
                "name": path.name,
                "content": content,
                "method": method,
                "pages": page_count,
                "elapsed_s": round(time.monotonic() - t0, 1),
                # Same rule as fetch: a partial extraction must say so in a field.
                "truncated": truncated,
                "truncated_at_chars": _cap if truncated else 0,
            }
            if method in ("docling-ocr", "pymupdf-ocr") and ocr_ms:
                # How long the OCR actually took, so a slow scan is visible in
                # the payload instead of only in the client's timeout.
                entry["ocr_ms"] = ocr_ms
                entry["ocr_engine"] = ocr_meta.get(
                    "ocr_engine", "docling" if method == "docling-ocr" else "pymupdf-tesseract")
                for _k in ("pages_ocr", "pages_total", "dpi", "lang",
                           "ocr_stopped_at", "ocr_fallback_from"):
                    if ocr_meta.get(_k) not in (None, ""):
                        entry[_k] = ocr_meta[_k]
            results.append(entry)

        if output_dir.strip():
            out = Path(output_dir.strip())
            out.mkdir(parents=True, exist_ok=True)
        else:
            out_tmpdir = tempfile.mkdtemp(prefix="pdfx_out_")
            out = Path(out_tmpdir)
        for r in results:
            name = r.get("filename", "")
            # H1-para: contain writes inside `out` — `name` derives from a remote
            # URL path basename which can carry `..` and escape output_dir.
            safe_stem = Path(name).stem
            if ".." in safe_stem or "" == safe_stem or safe_stem in (".", ".."):
                safe_stem = "extract"
            if isinstance(r["content"], str):
                (out / f"{safe_stem}.{_EXT[fmts[0]]}").write_text(r["content"], encoding="utf-8")
            else:
                for f, c in r["content"].items():
                    (out / f"{safe_stem}.{_EXT[f]}").write_text(c, encoding="utf-8")

        return {
            "success": len(failed) == 0 and len(results) > 0,
            "files": str(out),
            "results": results,
            "failed": failed,
            "failed_files": list(failed.keys()),
            "succeeded_files": [r.get("filename", "") for r in results],
            "input_count": len(local_files),
            "error": ("; ".join(f"{k}: {v.get('error')}" for k, v in failed.items())
                      if failed else None),
        }
    except Exception as e:
        return {"success": False, "error": f"PDF extraction failed: {e}"}
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        if out_tmpdir and os.path.isdir(out_tmpdir):
            shutil.rmtree(out_tmpdir, ignore_errors=True)
