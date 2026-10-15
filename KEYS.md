# websearch API keys — status, quirks, unfixable nuances

Last verified: 2026-10-15. Live values live in `run` (gitignored). This file
documents BEHAVIOR only, never values. Placeholders in `run.example`.

## Key table

| Key | Required? | Live? | What breaks without it |
|---|---|---|---|
| `GROQ_API_KEYS` | Recommended | Yes (6 keys, round-robin) | No synthesis, no query rewrite/classify/expand, no `extract strategy=llm`. Silent degrade. |
| `TAVILY_KEYS` | No | Yes | Tavily lane skipped. Free 1000 credits/mo; `advanced`+raw burns 2 credits/req (~500 req/mo effective). Nebius acquired Tavily 2026 — treat free tier as revocable. |
| `NV_KEY` | No | Yes | `_embed()` returns None; ranking degrades to lexical. Watch for 404 on `llama-nemotron-embed-1b-v2` (sibling SKU died Sep 2026); migration is one config string to `nemotron-3-embed-1b`. |
| `TINYFISH_KEYS` | No | Yes | TinyFish lane skipped. Most generous quota: 12,000 req/day, HTTP 402 beyond. |
| `ANYSEARCH_KEY` | No | Yes | AnySearch lane skipped. 1000 req/day, 20 QPS. |
| `RERANKER_MODEL` | No | Default | Overrides reranker model. Reranker currently DOWN (no socket) — results unranked, everything works. |
| `SE_KEY` / `STACKEXCHANGE_KEY` | No | NOT SET | Keyless 300 req/day/IP. A free registered key gives 10,000/day (33x). Code reads env at call time; empty = keyless. Register at stackapps.com. |
| `S2_KEY` / `S2_API_KEY` | No | NOT SET | S2 keyless 429s regularly from shared-pool contention (observed 2026-10-11 AND 2026-10-15). Free key = dedicated 1 RPS. `search_academic.py` also hits OpenAlex keyless. |

## Groq model split (2026-10-11+)

- Synthesis + grounding + `extract strategy=llm`: `openai/gpt-oss-120b` (quality matters, ~500 tps, 1000 RPD cap).
- Rewrite / classify / expand / decompose: `llama-3.1-8b-instant` (14,400 RPD + 500K TPD — 14x headroom for cheap ops).
- Constants in `config.py`: `GROQ_SMART_MODEL` / `GROQ_CHEAP_MODEL`. No new env keys by design.

## Unfixable / accept-as-is

1. **Brave lane fragility.** Official API killed free tier Feb 2026 (now $5/mo + card). We HTML-scrape (1 XPath variant) — unaffected by pricing but the most fragile parser. Keep as last-resort; do NOT adopt official API.
2. **scrapling DNS-pin gap (accepted H1).** `PinningNetworkBackend` covers httpx paths only. scrapling owns its TCP connect; short-TTL DNS flip between validate and connect is open. No 2026 mitigation exists. Never add user-URL fetch with `follow_redirects=True` minus per-hop validation.
3. **grep.app bot-wall.** Keyless API 429-challenges from this egress (Vercel-acquired). Backup-only lane by design; Sourcegraph carries code search. No key exists to fix this — it's egress reputation.
4. **GDELT 429s.** `api.gdeltproject.org` 429s this egress even on first contact (shared-IP hammering). Lane degrades to [] correctly. Spacing guard (6s) is for OUR rate, not theirs.
5. **Sourcegraph latency.** Public GraphQL needs 8-30s per query; the vertical wrapper gives the `code` lane 35s. Without `count:N` prefix the endpoint returns nothing — keep the prefix (codesearch lane documents this).
6. **Reranker down.** Socket absent = unranked results with `relevance_score 0.0`. Everything works. Respawn or leave disabled deliberately; record the decision. VRAM: heavy search peaks ~3.5/4GB; single >6800-token passage OOMs full attention.
7. **Synthesis silent no-op.** Without `GROQ_API_KEYS`, `synthesize` returns nothing extra (documented in schema). Check keys before debugging empty synthesis.
8. **`run` holds live keys in plaintext** (gitignored + pre-commit tripwire). Never `git add -f run`. Tripwire false-positives on `tvly-dev-YOUR_KEY`-style placeholders in `run.example` — use `--no-verify` for placeholder-only diffs.
