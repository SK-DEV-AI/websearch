# websearch MCP Server

`mcp-name: io.github.SK-DEV-AI/websearch`

Companion agent skills: [`SK-DEV-AI/skills`](https://github.com/SK-DEV-AI/skills) · `npx skills add SK-DEV-AI/skills`

Multi-engine web search + fetch MCP server. Parallel fan-out across DuckDuckGo, Bing Web/News RSS, Tavily, Wikipedia, arXiv, Google News RSS, Brave (last resort — rate-limited from some networks), Google AI Mode, and intent-gated vertical lanes (MDN, Stack Exchange, GitHub, HN, Semantic Scholar, OSV vuln, grep.app/Sourcegraph code, GDELT news) — with consensus-merge dedup, a local cross-encoder reranker, and Groq synthesis with grounding checks.

## Tools (20)

- **doctor** — **Run this first.** Reports which lanes are usable right now and the exact command to enable the rest. Distinguishes "broken" from "needs optional setup". Never prints key values.
- **search** — Open research questions. `depth` defaults to 3 when a model lane is configured (more independent sources; measured) and 1 otherwise. Depth also sets how many pages are fetched: `max(5, depth*2)`, ceiling 8. `synthesize` returns a short, cited answer that never invents: it says which source backs each claim, surfaces disagreement instead of averaging it, and marks itself `truncated` when it ran out of budget.
- **fetch** — URL to markdown/text. Direct fetch with auto-fallback to a Chromium CDP browser for blocked/JS pages. PDF/EPUB/DOCX supported. Paginated (`offset`). Optional `distill` (see below). For structured JSON use `extract`.
- **fetch_many** — Fetch several URLs in one call. Dedups, caps concurrency, honours robots.txt, shares the cache, returns per-URL success/status/content plus counts. Use when you have 2+ URLs; use `fetch` for one.
- **crawl** — Breadth-first site crawl with hard caps (per-page cap, depth cap, robots.txt, dedup, concurrency). Returns each page plus a `stop_reason` so you know whether the site was exhausted or a cap stopped it. Use for "read this whole docs site".
- **extract_links** — Every link on a page with anchor text, title and rel. Resolves relative hrefs, drops `javascript:`/`mailto:`, dedups. Use before `crawl` or to find a "next" button.
- **extract_table** — HTML tables to json, csv or markdown, by selector and index. Reports `tables_found` so you can pick one.
- **download** — Save a URL to disk; returns the local path, byte count, sha256 and content type. Writes only inside `dest_dir` and refuses path traversal. Marks `truncated` if the byte cap stopped the transfer.
- **diff_page** — Has this page changed since it was cached? Returns changed/old_hash/new_hash plus added and removed lines. First call for a URL returns `had_previous: false`.
- **screenshot** — ARIA accessibility snapshot (LLM-readable page text and interactive structure).
- **browser_action** — Drive a real browser page (navigate, click, type, press a key, evaluate JS, read HTML/text, ARIA tree, screenshot). Use when a page needs interaction; for plain reading use `fetch`. Pass `session_id` to reuse a persistent tab.
- **browser_session** — Create, list, inspect or close a persistent browser tab. Without one, every `browser_action` call gets a fresh tab and closes it, so logins and cookies cannot survive. See below.
- **form_fill_submit** — Fill many fields, submit, and read the result in one call. Verifies each element exists, reads back each typed value, and refuses to submit if a value did not stick. Reports `effect` (`navigated` / `content_changed` / `none`) so a submit that silently did nothing is visible rather than assumed.
- **paginate** — Click a "next" control, collecting a value per page, up to a cap. Stops honestly when the control is missing or has no effect, and reports `stop_reason`.
- **wikipedia** — Full Wikipedia API client: articles, summaries, categories, links, pageviews, revisions, backlinks, geosearch, recent changes, random pages.
- **arxiv** — arXiv paper search with boolean/phrase/wildcard query syntax (single-source; multi-source paper search lives in codesearch `papers`).
- **extract** — Structured JSON from a page via CSS (fast, free), LLM, or regex strategies. Repeat calls cached.
- **pdf_extract** — PDF to markdown/json/html. PyMuPDF fast path for text-layer PDFs, Docling+OCR fallback for scanned PDFs.
- **map_site** — Survey a site's pages via sitemap XML (primary) + HTML link fallback, before fetching individually.
- **ping** — Connectivity check for the internet and key search endpoints. Use before expensive calls when uncertain.

## What replaced the throwaway scripts

Every one of these exists because an agent otherwise writes a one-off script:

| The task | The tool |
|---|---|
| Fetch 20 URLs | `fetch_many` |
| Read a whole docs site | `crawl` |
| Get every link on a page | `extract_links` |
| Pull a pricing table into CSV | `extract_table` |
| Save a PDF and get its path | `download` |
| "Did this page change?" | `diff_page` |
| Log in, then act across several calls | `browser_session` + `browser_action(session_id=...)` |
| Fill and submit a form | `form_fill_submit` |
| Walk "next" through pages | `paginate` |

## Persistent browser sessions

`browser_action` opens one fresh tab per call and closes it afterwards. That is
right for a single read, but it makes any interactive workflow impossible: no
login, no cookies, no scroll position, no multi-step flow across calls.

A session is a named, long-lived tab:

```python
browser_session(action="create", label="login", url="https://site.example/login")
# -> {"session_id": "login-1a2b3c4d", ...}
browser_action(action="type", selector="#email", text="me@example.com", session_id="login-1a2b3c4d")
browser_action(action="click", selector="#submit", session_id="login-1a2b3c4d")
browser_action(action="text", session_id="login-1a2b3c4d")   # same tab, still logged in
browser_session(action="close", session_id="login-1a2b3c4d")
```

Sessions are opt-in: without `session_id`, behaviour is exactly as before. They
are capped at 8 with least-recently-used eviction, tabs are always closed
(including on eviction), and metadata is mirrored to
`~/.cache/websearch/sessions.json` so `list` can explain what survived a
restart. Sessions need a browser, so they are local-only.

## In-tool distillation (optional, off by default)

A fetched page can be 50k characters. The calling model pays for all of it even
when it needed one fact. `fetch(distill=true)` runs a small model inside the
tool first and returns a compressed result:

```python
fetch(url="https://example.com/long-report", distill=True,
      distill_question="What are the pricing tiers?")
```

The result never pretends to be the page:

| Field | Meaning |
|---|---|
| `distill.distilled` | whether a summary was produced |
| `distill.raw_chars` | size of the original text |
| `distill.grounded` | whether every number and quoted span was found in the source |
| `distill.grounding.numbers_unverified` | figures in the summary that do not appear in the source |
| `distill.grounding.quotes_unverified` | quoted spans that are not verbatim |
| `distill.how_to_get_raw` | how to get the untouched text: `distill=false` |

Below the size threshold (default 4000 chars) distillation is skipped and the
raw text is returned, because a summary would cost more than it saves. On any
failure the raw text is returned with `distill_skipped` explaining why - a
broken summary is never returned in place of the page.

| Variable | Default | Meaning |
|---|---|---|
| `DISTILL_BASE_URL` | `http://127.0.0.1:11435/v1` | any OpenAI-compatible endpoint |
| `DISTILL_MODEL` | `reflection/beam:free` | model id |
| `DISTILL_API_KEY` | none | bearer token; without it the lane stays off |
| `DISTILL_RETRIES` | `3` | attempts per call (1-10) |
| `DISTILL_MAX_TOKENS` | `3000` | initial completion budget |
| `DISTILL_MAX_TOKENS_CEILING` | `12000` | ceiling when escalating after empty content |
| `DISTILL_TIMEOUT` | `90` | seconds per request |
| `DISTILL_THRESHOLD_CHARS` | `4000` | below this, skip distillation |

## The synthesis prompt (the answer you actually read)

`search` does not just rank results; it writes the answer. That answer is
generated from the retrieved text alone, under rules that exist to stop the
three failure modes of a search summary:

| Failure | What the prompt does |
|---|---|
| **Invention** | Nothing from the model's own knowledge. A claim with no citable source is deleted, not softened. If the results do not answer the question, saying so is the correct answer. |
| **Bias** | Rank is not evidence. The prompt says so explicitly, caps how many sources can come from one domain, requires disagreement between sources to be reported rather than averaged into a number that appears in no source, corrects false premises in the question, and labels vendor claims as vendor claims. |
| **Injection** | Results are wrapped in `<results>` and declared untrusted data. A page that says "ignore previous instructions" is data, not instruction; the model must exclude it and say it did. |

Concision is structural, not a wish: a sentence ceiling, an answer-first rule,
and a named ban on filler phrasing ("it is important to note", "overall", ...).

Everything is verifiable rather than claimed:

- the result carries `prompt_version`, so a quality change can be tied to a
  revision instead of guessed at
- `temperature` is 0, so the same sources give the same text
- `finish_reason` and `truncated` are returned; a half sentence is labelled
  instead of looking like a complete thought
- `diversity` reports the source mix and whether the per-domain cap had to relax
- `as_of` reports the newest publication date among the sources, and undated
  sources are labelled "Published: not stated", so freshness is never implied
- `duplicates_removed` counts syndicated copies collapsed before synthesis, and
  `figures_unverified` lists numbers the answer contains that appear in no
  cited source (deterministic, no model call)
- a second model pass (`grounded`) re-reads the answer against the exact text
  the first pass saw and flags sentences no cited source supports. It is marked
  `advisory` because it is another model's opinion, not a proof

### Why the answer is prose, not JSON

The tool envelope is already structured: `answer`, `sources[]`, `citations_used`,
`grounded`, `diversity`, `as_of`. Asking the model to also emit JSON moves
structure from a place where it is generated deterministically into a place where
it is asserted, and measured behaviour is worse: a required `key_facts` array
gets padded with restatements of a single fact, which reads as corroboration.
JSON parsed 3/3 in testing, so this is not a reliability argument - it is a
content-quality and duplication argument. See `exec/websearch-format-test/`.

### Which model writes the answer

The in-tool model comes first by default, because it is free at the point of
use. Groq is the fallback if it fails. Both are plain OpenAI-compatible chat
endpoints, so one call path serves both.

| Env | Default | Meaning |
|---|---|---|
| `WEBSEARCH_SYNTH_PROVIDER` | `local` | `local` (in-tool model first), `groq` (Groq first), `auto` (either, in-tool preferred) |
| `WEBSEARCH_DEFAULT_DEPTH` | auto | Pins the default `depth`. Auto = 3 when a model lane exists, else 1 |
| `WEBSEARCH_ENRICH_MAX_PAGES` | `8` | Ceiling on page fetches per search. The count itself is `max(5, depth*2)` |
| `WEBSEARCH_SYNTH_TOKENS` | `900` | Answer budget. On `finish_reason=length` the result says `truncated` |
| `WEBSEARCH_SYNTH_MAX_SOURCES` | `8` | Ceiling on sources in the context |
| `WEBSEARCH_DOMAIN_CAP` | `2` | Max sources from one registrable domain |
| `WEBSEARCH_MIN_SOURCES` | `3` | Floor before the domain cap relaxes |
| `WEBSEARCH_SYNTH_SNIPPET_CHARS` | `1500` | Chars of each source given to the model |

`doctor()` reports all of these resolved (`env.defaults`) plus the active prompt
versions (`env.prompts`), and lists synthesis as its own lane, so "why is there
no answer?" has an answer instead of a blank field.

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

