from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from config import get_http_client


async def search_arxiv(query: str, count: int = 3, search_field: str = "all",
                       sort_by: str = "relevance", sort_order: str = "descending",
                       start: int = 0, id_list: str = "",
                       category: str = "", raw_query: str = "") -> dict:
    try:
        if raw_query:
            params = {"search_query": raw_query, "start": start,
                "max_results": min(count, 50), "sortBy": sort_by, "sortOrder": sort_order}
        elif id_list:
            params = {"id_list": id_list, "start": start,
                "max_results": min(count, 50), "sortBy": sort_by, "sortOrder": sort_order}
        else:
            search_q = f"{search_field}:{query}" if search_field != "all" else f"all:{query}"
            if category:
                search_q += f" AND cat:{category}"
            params = {"search_query": search_q, "start": start,
                "max_results": min(count, 50), "sortBy": sort_by, "sortOrder": sort_order}
        c = get_http_client()
        r = await c.get("https://export.arxiv.org/api/query", params=params,
                        headers={"User-Agent": "mcp-codesearch/1.0"}, timeout=15)
        if r.status_code != 200:
            return {"results": [], "total_available": 0}
        ns = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(r.text)
        total = root.findtext("opensearch:totalResults", "0",
                              {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"})
        results = []
        for entry in root.findall("atom:entry", ns)[:count]:
            title = entry.findtext("atom:title", "", ns).replace("\n", " ").strip()
            summary_text = entry.findtext("atom:summary", "", ns).replace("\n", " ").strip()[:500]
            url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    url = link.get("href", "")
                    break
                if link.get("rel") == "alternate":
                    url = link.get("href", "")
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            published = entry.findtext("atom:published", "", ns)[:10]
            updated = entry.findtext("atom:updated", "", ns)[:10]
            categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]
            comment = entry.findtext("arxiv:comment", "", ns)
            journal_ref = entry.findtext("arxiv:journal_ref", "", ns)
            doi = entry.findtext("arxiv:doi", "", ns)
            pc = entry.find("arxiv:primary_category", ns)
            primary_cat = pc.get("term", "") if pc is not None else ""
            id_text = entry.findtext("atom:id", "", ns)
            arxiv_id = id_text.split("/abs/")[-1] if "/abs/" in id_text else ""
            results.append({
                "title": title, "url": url, "snippet": summary_text,
                "authors": authors, "published": published, "updated": updated,
                "source": "arxiv", "arxiv_id": arxiv_id,
                "categories": categories, "primary_category": primary_cat,
                "comment": comment, "journal_ref": journal_ref, "doi": doi,
            })
        return {"results": results, "total_available": int(total)}
    except Exception:
        return {"results": [], "total_available": 0}
