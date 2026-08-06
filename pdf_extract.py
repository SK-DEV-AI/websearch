"""PDF extraction via opendataloader-pdf."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import get_http_client
from security import validate_url
import opendataloader_pdf

__all__ = ["extract_pdf"]


async def _download_pdf(url: str, dst: Path) -> Path:
    url = await validate_url(url)
    c = get_http_client()
    r = await c.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return dst


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
                    tmpdir = tempfile.mkdtemp(prefix="odl_")
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

        out = output_dir.strip() or tempfile.mkdtemp(prefix="odl_out_")
        if not output_dir.strip():
            out_tmpdir = out

        kwargs: dict[str, Any] = {
            "input_path": local_files if len(local_files) > 1 else local_files[0],
            "output_dir": out,
            "format": format,
            "quiet": quiet,
        }
        if password:
            kwargs["password"] = password
        if sanitize:
            kwargs["sanitize"] = True
        if keep_line_breaks:
            kwargs["keep_line_breaks"] = True
        if pages:
            kwargs["pages"] = pages
        if hybrid:
            kwargs["hybrid"] = hybrid
        if hybrid_mode:
            kwargs["hybrid_mode"] = hybrid_mode
        if hybrid_url:
            kwargs["hybrid_url"] = hybrid_url
        if hybrid_timeout:
            kwargs["hybrid_timeout"] = hybrid_timeout
        if table_method:
            kwargs["table_method"] = table_method
        if reading_order:
            kwargs["reading_order"] = reading_order
        if image_output:
            kwargs["image_output"] = image_output
        if image_format:
            kwargs["image_format"] = image_format
        if include_header_footer:
            kwargs["include_header_footer"] = True
        if detect_strikethrough:
            kwargs["detect_strikethrough"] = True
        if markdown_with_html:
            kwargs["markdown_with_html"] = True
        if use_struct_tree:
            kwargs["use_struct_tree"] = True
        if content_safety_off:
            kwargs["content_safety_off"] = content_safety_off
        if threads:
            kwargs["threads"] = threads
        if replace_invalid_chars:
            kwargs["replace_invalid_chars"] = replace_invalid_chars

        try:
            await asyncio.wait_for(
                asyncio.to_thread(opendataloader_pdf.convert, **kwargs),
                timeout=300)
        except asyncio.TimeoutError:
            return {"success": False, "error": "PDF conversion timed out after 300s"}

        result_files: dict[str, dict[str, str]] = {}
        out_dir = Path(out)
        if out_dir.is_dir():
            for fmt_dir in sorted(out_dir.iterdir()):
                if not fmt_dir.is_dir():
                    continue
                fmt_name = fmt_dir.name
                result_files[fmt_name] = {}
                for f in sorted(fmt_dir.iterdir()):
                    if f.is_file() and f.stat().st_size > 0:
                        content = f.read_text(encoding="utf-8", errors="replace")
                        if len(content) > 50000:
                            content = content[:50000] + "\n\n[... truncated at 50000 chars ...]"
                        content_key = result_files[fmt_name]
                        if isinstance(content_key, dict):
                            content_key[f.name] = content

        return {
            "success": True,
            "files": str(out_dir),
            "results": result_files,
            "input_count": len(local_files),
        }
    except Exception as e:
        return {"success": False, "error": f"PDF extraction failed: {e}"}
    finally:
        import shutil
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        if out_tmpdir and os.path.isdir(out_tmpdir):
            shutil.rmtree(out_tmpdir, ignore_errors=True)
