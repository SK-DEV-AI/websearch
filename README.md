# websearch MCP Server

`mcp-name: io.github.SK-DEV-AI/websearch`

Companion agent skills: [`SK-DEV-AI/skills`](https://github.com/SK-DEV-AI/skills) · `npx skills add SK-DEV-AI/skills`

Multi-engine web search + fetch MCP server. Parallel fan-out across DuckDuckGo, Bing Web/News RSS, Tavily, Wikipedia, arXiv, Google News RSS, Brave (last resort — rate-limited from some networks), Google AI Mode, and intent-gated vertical lanes (MDN, Stack Exchange, GitHub, HN, Semantic Scholar, OSV vuln, grep.app/Sourcegraph code, GDELT news) — with consensus-merge dedup, a local cross-encoder reranker, and Groq synthesis with grounding checks.

## Tools (11)

- **doctor** — **Run this first.** Reports which lanes are usable right now and the exact command to enable the rest. Distinguishes "broken" from "needs optional setup". Never prints key values.
- **search** — Open research questions. depth=1 snippets, depth≥2 full-page fetch + rerank. `synthesize` returns a Groq answer with [N] citations.
- **fetch** — URL to markdown/text. Direct fetch with auto-fallback to a Chromium CDP browser for blocked/JS pages. PDF/EPUB/DOCX supported. Paginated (`offset`). For structured JSON use `extract`.
- **screenshot** — ARIA accessibility snapshot (LLM-readable page text and interactive structure).
- **browser_action** — Drive a real browser page (navigate, click, type, press a key, evaluate JS, read HTML/text, ARIA tree, screenshot). Use when a page needs interaction; for plain reading use `fetch`.
- **wikipedia** — Full Wikipedia API client: articles, summaries, categories, links, pageviews, revisions, backlinks, geosearch, recent changes, random pages.
- **arxiv** — arXiv paper search with boolean/phrase/wildcard query syntax (single-source; multi-source paper search lives in codesearch `papers`).
- **extract** — Structured JSON from a page via CSS (fast, free), LLM, or regex strategies. Repeat calls cached.
- **pdf_extract** — PDF to markdown/json/html. PyMuPDF fast path for text-layer PDFs, Docling+OCR fallback for scanned PDFs.
- **map_site** — Survey a site's pages via sitemap XML (primary) + HTML link fallback, before fetching individually.
- **ping** — Connectivity check for the internet and key search endpoints. Use before expensive calls when uncertain.

## What works out of the box, and what is optional

**Everything keyless works immediately** after `pip install`. No API keys, no browser, no GPU: search (DuckDuckGo, Bing RSS, Wikipedia, arXiv, Stack Exchange, GitHub, HN, grep.app, Google News RSS), fetch, map_site, pdf_extract, screenshot.

Four lanes need optional setup. Run **`doctor()`** and it tells you which are live and prints the exact command for the rest:

| Lane | Needs | On a headless server |
|---|---|---|
| **Google AI Mode** (`search(google_ai_only=True)`) | a Chromium-based browser with `--remote-debugging-port` **+ a signed-in Google account** in that profile | **unavailable** — no signed-in browser exists there |
| **browser_action** | same browser, no Google login needed | **unavailable** |
| **CDP fallback for bot-walled pages** | same browser | unavailable; those pages return a terminal verdict instead of hanging |
| **Cross-encoder reranking** | an optional local worker (separate download, ~500MB model) | works if installed; otherwise scoring falls back to blended relevance |

Enabling the browser lanes:

```bash
# any Chromium-family browser: chromium, google-chrome, brave, Helium, edge, vivaldi
chromium --remote-debugging-port=9222 --user-data-dir="$HOME/.mcp-browser"
```

Then open google.com in that window and **sign in once** (needed only for Google AI Mode). Point elsewhere with `WEBSEARCH_CDP_URL=http://host:port`.

```bash
# confirm
doctor()                          # per-lane report
search(query="...", google_ai_only=True)   # explicit error + fix list if no browser
```

The browser lanes are **local-only by design**. On a server the HTTP lanes are unaffected, so the server is still fully useful — it just cannot query Google AI Mode. Without a browser, `google_ai_only=True` returns a structured error naming the cause and the fix, never an empty result set.

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

Everything else (DDG, Bing RSS, Wikipedia, arXiv, RSS) is keyless. Brave's HTML scrape is currently rate-limited from some networks, so Bing RSS is the primary general-web door and Brave is a last resort.

## Install from PyPI

```bash
pip install skdev-websearch-mcp          # core install, small
pip install skdev-websearch-mcp[all]     # + browser/OCR/extract extras (multi-GB)
```

The `websearch-mcp` console script speaks MCP over stdio:

```json
{ "websearch": { "command": "websearch-mcp" } }
```

## Docker

```bash
docker build -t websearch-mcp .
docker run --rm -i websearch-mcp
```

Keys are passed at run time, never baked in: `-e GROQ_API_KEYS=...`.
Heavy extras: `docker build --build-arg EXTRAS="[all]" -t websearch-mcp .`

## Notes

- **Reranker (optional, local):** ranking uses a local `gte-reranker-modernbert-base` cross-encoder over a Unix socket (`RERANKER_MODEL` overrides the model). Not bundled with the pip package — set `RERANKER_WORKER=/path/to/worker.py` to use one. Without it, ranking falls back to blended relevance scores; everything still works. `doctor()` reports whichever is in effect.
- **CDP browser (optional):** bot-walled pages escalate to a Chrome DevTools endpoint (`WEBSEARCH_CDP_URL`, default `127.0.0.1:9222`) when available; without it those pages return a terminal verdict instead of hanging.
- **Caching:** fetch results cache for 1h (`cache_ttl=0` forces fresh). No database, no state beyond the cache dir.
- **Safety:** all user-URL fetches are SSRF-validated (private-IP blocking + redirect-hop revalidation + robots.txt gate).
- **Groq model split:** synthesis + grounding + extract-llm use `openai/gpt-oss-120b`; rewrite/classify/expand/decompose use `openai/gpt-oss-20b` (the previous `llama-3.1-8b-instant` was retired by Groq and now 404s). Constants `GROQ_SMART_MODEL`/`GROQ_CHEAP_MODEL` in config.py.
- **Keyless lanes (all advisory-only):** OSV.dev vuln vertical + NVD CVE enrichment; grep.app MCP code search (Sourcegraph backup); Bing Web RSS + Bing News RSS (Brave's scrape is rate-limited, GDELT 429s from some networks); Wikipedia vertical; Jina Reader pre-CDP tier; Wayback CDX-history recovery in enrich(); Common Crawl domain-survey assist in map_site.
- **Local reranker:** a `gte-reranker-modernbert-base` cross-encoder runs as one shared worker over a Unix socket; either server spawns it on demand and it unloads when idle. Optional — without it results are engine-ranked only.
- **Optional keys (graceful keyless fallback):** `SE_KEY`/`STACKEXCHANGE_KEY` (10k/day SE), `S2_KEY`/`S2_API_KEY` (dedicated S2 1 RPS).
- **Marginalia:** best-effort lane only (shared public key pool, 503s under load); independent-index recall, no dedicated key wired (kontakt@marginalia.nu for one).

