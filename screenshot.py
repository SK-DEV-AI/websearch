from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time

from search_gai import _get_optimized_page, _cleanup_orphan_tabs


async def cdpa11y_snapshot(url: str, verbose: bool = False, max_chars: int = 10000,
                           depth: int = 5, boxes: bool = False) -> dict:
    try:
        page = await _get_optimized_page(block_resources=False)
        try:
            await page.goto(url, wait_until="commit", timeout=30)
            await page.wait_for_load_state("domcontentloaded", timeout=10)
            await asyncio.sleep(0.3)

            # Tries getFullAXTree first (real AX tree), falls back to Page.captureSnapshot
            nodes = await page.get_ax_tree(depth=depth)
            if nodes:
                lines = _format_ax_nodes(nodes)
            else:
                snap_kwargs: dict = {"mode": "ai"}
                if boxes:
                    snap_kwargs["boxes"] = True
                raw = await page.aria_snapshot(**snap_kwargs)
                lines = raw.split("\n") if raw else []

            if not lines or all(not l.strip() for l in lines):
                html_c = await page.document_html()
                return {"success": True, "url": url, "snapshot": "",
                        "note": "no aria snapshot",
                        "fallback_html_len": len(html_c)}

            if not verbose:
                ax_roles = {"- link", "- button", "- textbox", "- searchbox",
                            "- combobox", "- checkbox", "- radio", "- switch",
                            "- slider", "- tab", "- menuitem", "- option",
                            "- heading", "- listitem", "- treeitem",
                            "- spinbutton", "- img", "- navigation",
                            "- banner", "- main", "- complementary", "- form",
                            "- search", "- region"}
                lines = [
                    l for l in lines
                    if any(l.strip().startswith(r) for r in ax_roles)
                    or (not l.strip().startswith("-") and l.strip())
                ]

            snap = "\n".join(lines)
            if len(snap) > max_chars:
                snap = snap[:max_chars] + f"\n... [truncated at {max_chars} chars]"
            return {"success": True, "url": url, "snapshot": snap}
        finally:
            try:
                await page.close()
            except Exception:
                pass
            asyncio.ensure_future(_cleanup_orphan_tabs())
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


def _format_ax_nodes(nodes: list[dict], depth: int = 0) -> list[str]:
    """Format Accessibility.getFullAXTree nodes into indented role lines."""
    lines: list[str] = []
    for node in nodes:
        role = node.get("role", {}).get("value", "unknown")
        name = node.get("name", {}).get("value", "")
        value = node.get("value", {}).get("value", "")
        indent = "  " * depth
        line = f"{indent}- {role}"
        if name:
            line += f' "{name}"'
        if value and value != name:
            line += f': {value}'
        lines.append(line)
        children = node.get("childIds", [])
        if children and depth < 10:
            child_nodes = [n for n in nodes if n.get("nodeId") in children]
            lines.extend(_format_ax_nodes(child_nodes, depth + 1))
    return lines


async def screenshot_cdp(url: str, full_page: bool = True,
                         clip_x: float = 0, clip_y: float = 0,
                         clip_width: float = 0, clip_height: float = 0,
                         scale: str = "css", animations: str = "allow",
                         quality: int | None = None,
                         image_type: str = "png",
                         omit_background: bool = False,
                         caret: str = "initial") -> dict:
    try:
        page = await _get_optimized_page(block_resources=False)
        try:
            await page.goto(url, wait_until="commit", timeout=30)
            await page.wait_for_load_state("domcontentloaded", timeout=10)
            await asyncio.sleep(0.3)
            ss_kwargs: dict = {"captureBeyondViewport": full_page, "format": image_type}
            if omit_background:
                ss_kwargs["omitBackground"] = True
            if caret in ("hide", "initial"):
                ss_kwargs["caret"] = caret
            if clip_width > 0 and clip_height > 0:
                ss_kwargs["clip"] = {"x": clip_x, "y": clip_y,
                                     "width": clip_width, "height": clip_height}
            if scale in ("css", "device"):
                ss_kwargs["scale"] = scale
            if animations in ("allow", "disabled"):
                ss_kwargs["animations"] = animations
            if quality is not None and image_type == "jpeg":
                ss_kwargs["quality"] = quality
            b64_bytes = await page.screenshot(**ss_kwargs)
            if not b64_bytes:
                return {"success": False, "url": url, "error": "no screenshot captured"}
            encoded = base64.b64encode(b64_bytes).decode("utf-8")
            if len(encoded) > 2_000_000:
                fp = os.path.join(tempfile.gettempdir(), f"ss_{int(time.monotonic())}.{image_type}")
                with open(fp, "wb") as f:
                    f.write(b64_bytes)
                return {"success": True, "url": url,
                        "screenshot_base64": f"[saved to {fp} ({len(b64_bytes)//1024}KB)]"}
            return {"success": True, "url": url, "screenshot_base64": encoded}
        finally:
            try:
                await page.close()
            except Exception:
                pass
            asyncio.ensure_future(_cleanup_orphan_tabs())
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}
