"""Detailed end-to-end benchmark of websearch search_multi with all engine keys loaded.

Diagnostic only. Loads env from the server's `run` file via bash (so ${VAR:-default}
expands exactly as the server sees), instruments every engine + phase on the `research`
namespace, runs search_multi once, prints a per-phase bill.
"""
import asyncio, os, subprocess, sys, time, tempfile

WSDIR = "/home/sk/session-root/repos/websearch"

res = subprocess.run(["bash", "-c",
    f"source <(grep -E '^export ' {WSDIR}/run); env | grep -E '^(GROQ|TAVILY|TINYFISH|ANYSEARCH|NV|RERANKER)_'"],
    capture_output=True, text=True)
for line in res.stdout.splitlines():
    if "=" in line:
        k, _, v = line.partition("=")
        os.environ[k] = v

sys.path.insert(0, WSDIR)
import research

from collections import defaultdict
TIMES = defaultdict(float)
COUNTS = defaultdict(int)
RESULTS = defaultdict(int)
_t0 = {}

def _timeit(name):
    def deco(fn):
        async def wrap(*a, **k):
            key = name
            if key not in _t0:
                _t0[key] = time.monotonic()
            r = await fn(*a, **k)
            dt = time.monotonic() - _t0.pop(key, time.monotonic())
            TIMES[key] += dt
            COUNTS[key] += 1
            if isinstance(r, dict):
                RESULTS[key] += len(r.get("results", []))
            elif isinstance(r, list):
                RESULTS[key] += len(r)
            return r
        return wrap
    return deco

_existing = {}
def patch(mod, name):
    _existing[name] = getattr(mod, name)
    setattr(mod, name, _timeit(name)(_existing[name]))

# Engine phases in research namespace
for fn in ["search_google_rss", "search_tavily", "search_wikipedia", "search_reddit",
           "search_arxiv", "search_anysearch", "tinyfish_search", "search_ddg",
           "search_brave"]:
    if hasattr(research, fn):
        patch(research, fn)
# Groq + rerank phases
for fn in ["rewrite_query", "classify_need", "decompose_query", "expand_query", "_rerank"]:
    if hasattr(research, fn):
        patch(research, fn)

query = sys.argv[1] if len(sys.argv) > 1 else "best laptops 2026 under 1000 dollars"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 50

async def main():
    t0 = time.monotonic()
    from config import TAVILY_KEYS, TINYFISH_KEYS, GROQ_API_KEYS, NV_KEY
    print(f"keys: tavily={len(TAVILY_KEYS)} tinyfish={len(TINYFISH_KEYS)} groq={len(GROQ_API_KEYS)} nv={'y' if NV_KEY else 'n'}", flush=True)
    r = await research.search_multi(query, count=count, tavily_depth="advanced")
    twall = time.monotonic() - t0
    print(f"\n=== SEARCH_MULTI  q={query!r} count={count} ===")
    print(f"{'phase':<22}{'sec':>8}{'calls':>6}{'res':>6}")
    for k in sorted(TIMES, key=lambda x: -TIMES[x]):
        print(f"{k:<22}{TIMES[k]:>8.2f}{COUNTS[k]:>6}{RESULTS[k]:>6}")
    print(f"\nTOTAL WALL: {twall:.1f}s  results={len(r.get('results', []))}")

asyncio.run(main())