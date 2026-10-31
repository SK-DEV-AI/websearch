from __future__ import annotations

import asyncio

from search_gai import _get_optimized_page, _cleanup_orphan_tabs
from security import validate_url, SecurityError


async def cdpa11y_snapshot(url: str, verbose: bool = False, max_chars: int = 10000,
                           depth: int = 5) -> dict:
    try:
        await validate_url(url)
        page = await _get_optimized_page(block_resources=False)
        try:
            await page.goto(url, wait_until="commit", timeout=30)
            await page.wait_for_load_state("domcontentloaded", timeout=10)
            await asyncio.sleep(0.3)

            # Real AX tree (protocol param is "depth" — verified). Retry at
            # full depth before giving up; Page.captureSnapshot is NOT a
            # fallback (it only emits MHTML, not an accessibility tree).
            nodes = await page.get_ax_tree(depth=depth)
            if not nodes:
                nodes = await page.get_ax_tree(depth=100)
            lines = _format_ax_nodes(nodes, verbose=verbose) if nodes else []

            if not lines or all(not l.strip() for l in lines):
                html_c = await page.document_html()
                return {"success": True, "url": url, "snapshot": "",
                        "note": "no aria snapshot",
                        "fallback_html_len": len(html_c)}

            snap = "\n".join(lines)
            truncated = len(snap) > max_chars
            if truncated:
                snap = snap[:max_chars] + f"\n... [truncated at {max_chars} chars]"
            return {"success": True, "url": url, "snapshot": snap,
                    "truncated": truncated,
                    "truncated_at_chars": max_chars if truncated else 0}
        finally:
            try:
                await page.close()
            except Exception:
                pass
            asyncio.ensure_future(_cleanup_orphan_tabs())
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


def _format_ax_nodes(nodes: list[dict], depth: int = 0, verbose: bool = False) -> list[str]:
    """Format Accessibility.getFullAXTree nodes into indented role lines.

    When a parent is filtered out (non-verbose role filter), its children
    are promoted so indentation reflects the nodes actually emitted rather
    than the original tree depth.
    """
    lines: list[str] = []
    by_id = {n.get("nodeId"): n for n in nodes}

    ax_roles = {"link", "button", "textbox", "searchbox", "combobox",
                "checkbox", "radio", "switch", "slider", "tab", "menuitem",
                "option", "heading", "listitem", "treeitem", "spinbutton",
                "img", "navigation", "banner", "main", "complementary",
                "form", "search", "region"}

    def keep(node: dict) -> bool:
        if verbose:
            return True
        role = node.get("role", {}).get("value", "unknown")
        return role in ax_roles

    def emit(node: dict, depth: int) -> None:
        role = node.get("role", {}).get("value", "unknown")
        name = node.get("name", {}).get("value", "")
        value = node.get("value", {}).get("value", "")
        if keep(node):
            indent = "  " * depth
            line = f"{indent}- {role}"
            if name:
                line += f' "{name}"'
            if value and value != name:
                line += f': {value}'
            lines.append(line)
            child_depth = depth + 1
        else:
            # Filtered out: promote children to this depth (collapse).
            child_depth = depth
        if depth < 10:
            for cid in node.get("childIds", []):
                child = by_id.get(cid)
                if child is not None:
                    emit(child, child_depth)

    # Emit true roots only: a flat getFullAXTree lists every node, so
    # emitting all of them as roots duplicates each child under its parent.
    child_ids: set = set()
    for node in nodes:
        child_ids.update(node.get("childIds", []))
    for node in nodes:
        if node.get("nodeId") not in child_ids:
            emit(node, depth)
    return lines

