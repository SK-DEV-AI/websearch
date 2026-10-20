# websearch API keys — status, quirks, unfixable nuances

Last verified: 2026-10-16. Live values live in `run` (gitignored). This file
documents BEHAVIOR only, never values. Placeholders in `run.example`.

## Key table

| Key | Required? | Live? | What breaks without it |
|---|---|---|---|
| `GROQ_API_KEYS` | Recommended | Yes (6 keys, round-robin) | No synthesis, no query rewrite/classify/expand, no `extract strategy=llm`. Silent degrade. |
| `TAVILY_KEYS` | No | Yes | Tavily lane skipped. Free 1000 credits/mo; `advanced`+raw burns 2 credits/req (~500 req/mo effective). Nebius acquired Tavily 2026 — treat free tier as revocable. |
| `NV_KEY` | No | Yes | `_embed()` returns None; ranking degrades to lexical. Migrated 2026-10-15 to `nemotron-3-embed-1b` (both old models 410 Gone). Cache keys model-prefixed. |
| `TINYFISH_KEYS` | No | Yes | TinyFish lane skipped. Most generous quota: 12,000 req/day, HTTP 402 beyond. |
| `ANYSEARCH_KEY` | No | Yes | AnySearch lane skipped. 1000 req/day, 20 QPS. |
| `RERANKER_MODEL` | No | Default | Advisory only (Rust worker bakes its model). Either server spawns the shared worker on demand, idle-exits 60s; borrowers never kill. |
| `SE_KEY` / `STACKEXCHANGE_KEY` | No | Yes (copied from codesearch 2026-10-16) | Keyless 300 req/day/IP. Keyed 10,000/day (33x). |
| `S2_KEY` / `S2_API_KEY` | No | NOT SET | S2 keyless 429s from this egress (verified 2026-10-20, 3/3 probes). **The free key does NOT need an academic email** — the request is a HubSpot form with `application=Private` + free-text Affiliation; the blocker is review backlog, not eligibility. See `work/mcp-opt/jcode-takeover/research/s2-signup-verdict.md`. OpenAlex now keyed too (OPENALEX_API_KEY, user key 2026-10-16). |

## Groq model split (2026-10-11+)

- Synthesis + grounding + `extract strategy=llm`: `openai/gpt-oss-120b` (quality matters, ~500 tps, 1000 RPD cap).
- Rewrite / classify / expand / decompose: `llama-3.1-8b-instant` (14,400 RPD + 500K TPD — 14x headroom for cheap ops).
- Constants in `config.py`: `GROQ_SMART_MODEL` / `GROQ_CHEAP_MODEL`. No new env keys by design.

## Unfixable / accept-as-is

1. **Brave lane DEAD (not fragile).** Official API killed free tier Feb 2026 (now $5/mo + card). Our HTML scrape now returns **HTTP 429 on every client** (curl, httpx, curl_cffi, requests) from this egress — measured 2026-10-20 after a 75s cooldown. The parser is NOT the problem (a 200 body parses into 20 correct results). Replacement: **Bing Web RSS** (keyless, 200 live). Do NOT adopt the official Brave API.
2. **scrapling DNS-pin gap (accepted H1).** `PinningNetworkBackend` covers httpx paths only. scrapling owns its TCP connect; short-TTL DNS flip between validate and connect is open. No 2026 mitigation exists. Never add user-URL fetch with `follow_redirects=True` minus per-hop validation.
3. **grep.app REST bot-wall (bypassed).** REST 429-challenges from this egress, but the MCP door (`mcp.grep.app` searchGitHub) answers <1s keyless — primary lane since 2026-10-15. No key needed. **AND-semantics gotcha:** grep.app ANDs the terms, so a 4+ word natural-language query returns ~0 hits. The code lane must build a short, code-shaped query.
4. **GDELT 429s.** `api.gdeltproject.org` 429s this egress even on first contact (shared-IP hammering). Lane degrades to [] correctly. Spacing guard (6s) is for OUR rate, not theirs.
5. **Sourcegraph latency.** Public GraphQL needs 8-30s per query; the vertical wrapper gives the `code` lane 35s. Without `count:N` prefix the endpoint returns nothing — keep the prefix (codesearch lane documents this).
6. **Reranker idle, not down.** Either server spawns the shared worker on demand; it exits after 60s idle (GPU back to 1 MiB) and respawns when needed. Borrower servers never kill the worker's socket. Killswitch still at `~/.local/share/reranker-rust/disabled`.
7. **Synthesis silent no-op.** Without `GROQ_API_KEYS`, `synthesize` returns nothing extra (documented in schema). Check keys before debugging empty synthesis.
8. **`run` holds live keys in plaintext** (gitignored + pre-commit tripwire). Never `git add -f run`. Tripwire false-positives on `tvly-dev-YOUR_KEY`-style placeholders in `run.example` — use `--no-verify` for placeholder-only diffs.
