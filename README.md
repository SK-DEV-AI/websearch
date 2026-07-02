# websearch MCP Server

Multi-engine web search MCP server for OpenCode. Searches DuckDuckGo, Google AI Mode (udm=50), Tavily, Wikipedia, arXiv, and AnySearch in parallel with embedding dedup + cross-encoder reranking.

## Tools

- **search** — Multi-engine search with dedup + reranking (depth param controls snippet-only vs full fetch)
- **fetch** — URL to markdown/text via CDP-first (Helium browser) with Cloudflare bypass
- **crawl** — BFS/DFS deep crawl via Crawl4ai (CDP-connected Helium)
- **screenshot** — CDP screenshot or ARIA accessibility snapshot
- **wikipedia** — Search, summaries, geosearch, categories, pageviews, revisions
- **arxiv** — Academic paper search with full metadata
- **map_site** — URL structure discovery via sitemap XML (supports RSS/Atom/plain text/gzipped), robots.txt, and HTML link fallback
- **pdf_extract** — Full PDF extraction (tables, OCR, formulas) via opendataloader-pdf

## Setup

1. Copy `run.example` to `run` and fill in API keys
2. Install dependencies in a Python 3.14+ venv
3. Run via OpenCode MCP config pointing to `run`
