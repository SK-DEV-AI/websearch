# websearch MCP Server

Multi-engine web search + fetch MCP server. Parallel fan-out across DuckDuckGo, Brave, Tavily, Wikipedia, arXiv, Google News RSS (plus opt-in Marginalia), Google AI Mode, and intent-gated vertical lanes (MDN, Stack Exchange, GitHub, HN, Semantic Scholar) — with consensus-merge dedup, a local cross-encoder reranker, and Groq synthesis with grounding checks.

## Tools (9)

- **search** — Open research questions. depth=1 snippets, depth≥2 full-page fetch + rerank. `synthesize` returns a Groq answer with [N] citations.
- **fetch** — URL to markdown/text. Direct fetch with auto-fallback to the Helium CDP browser for blocked/JS pages. PDF/EPUB/DOCX supported. Paginated (`offset`). For structured JSON use `extract`.
- **screenshot** — ARIA accessibility snapshot (LLM-readable page text and interactive structure).
- **wikipedia** — Full Wikipedia API client: articles, summaries, categories, links, pageviews, revisions, backlinks, geosearch, recent changes, random pages.
- **arxiv** — arXiv paper search with boolean/phrase/wildcard query syntax (single-source; multi-source paper search lives in codesearch `papers`).
- **extract** — Structured JSON from a page via CSS (fast, free), LLM, or regex strategies. Repeat calls cached.
- **pdf_extract** — PDF to markdown/json/html. PyMuPDF fast path for text-layer PDFs, Docling+OCR fallback for scanned PDFs.
- **map_site** — Survey a site's pages via sitemap XML (primary) + HTML link fallback, before fetching individually.
- **ping** — Connectivity check for the internet and key search endpoints. Use before expensive calls when uncertain.

## Setup

Requires Python 3.14+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp run.example run   # fill in your API keys (run is gitignored — never commit it)
chmod +x run
```

Point your MCP client at `run` (stdio):

```json
{ "websearch": { "command": "/path/to/websearch/run" } }
```

### API keys (`run`)

| Key | Used for | Required? |
|---|---|---|
| `GROQ_API_KEYS` | query rewrite, synthesis, grounding checks | Recommended (search works without, no synthesis) |
| `TAVILY_KEYS` | Tavily engine lane | Optional |
| `NV_KEY` | NVIDIA embedding endpoint | Optional |

Everything else (DDG, Brave scrape, Wikipedia, arXiv, RSS) is keyless.

## Notes

- **Reranker (optional, local):** ranking uses a local `gte-reranker-modernbert-base` cross-encoder over a Unix socket (`RERANKER_MODEL` overrides the model). Without it, results return unranked with `relevance_score 0.0` — everything still works.
- **Helium CDP (optional):** bot-walled pages escalate to a Chrome DevTools browser at `127.0.0.1:9222` when available; without it those pages return a terminal verdict instead of hanging.
- **Caching:** fetch results cache for 1h (`cache_ttl=0` forces fresh). No database, no state beyond the cache dir.
- **Safety:** all user-URL fetches are SSRF-validated (private-IP blocking + redirect-hop revalidation + robots.txt gate).
