"""Ranking stack ported from donsetch (github.com/dondai44423/donsetch):

- weighted RRF (per-index-family mass) + cross-engine consensus
- BM25-lite relevance bonus (IDF estimated from the result set)
- query-aware domain priors (intent tables + DIY utility table)
- vertical-only penalty for un-corroborated vertical hits
- 60/40 min-max blend with the cross-encoder (rust reranker)
- entity coverage penalty (hyphenated compounds, version/year drift)
- authority layer (official domains, title decisiveness, news freshness)
- syndicated-title dedup + per-domain diversity cap

Staged so the calling server can interleave the external rust
reranker between merge_base() and the penalties (it does the blend).

Pure stdlib, no external deps.
"""

from __future__ import annotations

import re
import time
import urllib.parse

# ────────────────────────────────────────────────────────────────
# rank.rs
# ────────────────────────────────────────────────────────────────

RRF_K = 60.0
CONSENSUS_MULT = 0.5
BM25_WEIGHT = 0.3
PRIOR_WEIGHT = 0.15
MAX_PER_DOMAIN = 2
VERTICAL_WEIGHT = 0.6
BLEND_ALPHA = 0.6

_VERTICALS = {"github", "hn", "wikipedia", "scholar", "news", "arxiv", "stackexchange", "mdn", "reddit"}


def is_vertical(engine: str) -> bool:
    return engine in _VERTICALS or engine.startswith("reddit")


def engine_family(engine: str) -> str:
    """Index family: engines sharing an index count once for consensus."""
    if is_vertical(engine):
        return f"vertical:{engine}"
    e = engine.split("-", 1)[0]
    return {
        "duckduckgo": "bing",       # ddg shares the Bing tail index
        "google": "google",
        "tavily": "tavily",
        "anysearch": "anysearch",
        "tinyfish": "tinyfish",
    }.get(e, e)


def norm_key(raw: str) -> str:
    """Normalize a URL for consensus matching: scheme-less, www-less,
    trailing-slash-less, lowercase host."""
    try:
        u = urllib.parse.urlparse(raw)
    except ValueError:
        return raw.lower()
    host = (u.netloc or u.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = u.path.rstrip("/").removesuffix("/index.html").removesuffix("/index.htm").lower()
    query = f"?{u.query}" if u.query else ""
    return f"{host}{path}{query}"


def host_of(raw: str) -> str:
    try:
        h = urllib.parse.urlparse(raw).netloc or urllib.parse.urlparse(raw).hostname or ""
    except ValueError:
        return ""
    if h.startswith("www."):
        h = h[4:]
    return h.lower()


def _relevance(query: str, docs: list[tuple[str, str]]) -> list[float]:
    """BM25-lite relevance of (title, snippet) against query.
    IDF is estimated from the result set itself."""
    terms = [t for t in re.split(r"[^a-zA-Z0-9]", query.lower()) if len(t) > 2]
    if not terms:
        return [0.0] * len(docs)
    n = max(len(docs), 1)
    tokenized: list[list[str]] = []
    for t, s in docs:
        tokenized.append([w for w in re.split(r"[^a-zA-Z0-9]", f"{t} {s}".lower())])
    df: dict[str, int] = {}
    for toks in tokenized:
        for term in terms:
            if term in toks:
                df[term] = df.get(term, 0) + 1
    avg_len = sum(len(t) for t in tokenized) / n
    k1, b = 1.2, 0.75
    out = []
    for toks in tokenized:
        score = 0.0
        for term in terms:
            tf = toks.count(term)
            if tf == 0:
                continue
            dfv = df.get(term, 0)
            idf = ((n - dfv + 0.5) / (dfv + 0.5) + 1.0) if dfv > 0 else 0.0
            idf = __import__("math").log(idf) if idf else 0.0
            len_norm = 1.0 - b + b * (len(toks) / max(avg_len, 1.0))
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
        out.append(score)
    return out


def merge_base(per_engine: dict[str, list[dict]], query: str, intent: str) -> list[dict]:
    """Group by normalized URL, weight RRF mass per index family, add
    consensus multiplier, BM25 bonus, domain prior, vertical-only
    penalty. Returns Merged dicts: {title, url, snippet, sources,
    score, published}. score = pre-blend base."""
    groups: dict[str, dict] = {}
    family_best: dict[tuple[str, str], float] = {}
    conceptual = is_conceptual(query)
    for engine, hits in per_engine.items():
        tw = VERTICAL_WEIGHT if (is_vertical(engine) and not (conceptual and engine == "wikipedia")) else 1.0
        family = engine_family(engine)
        for rank, hit in enumerate(hits):
            if not hit.get("url"):
                continue
            key = norm_key(hit["url"])
            contribution = tw / (RRF_K + rank + 1.0)
            fb = family_best.get((key, family))
            family_best[(key, family)] = contribution if fb is None else max(fb, contribution)
            entry = groups.get(key)
            if entry is None:
                entry = {
                    "title": hit.get("title", ""),
                    "url": hit["url"],
                    "snippet": hit.get("snippet", ""),
                    "sources": [],
                    "score": 0.0,
                    "published": hit.get("published") or hit.get("date"),
                }
                groups[key] = entry
            snip = hit.get("snippet", "") or ""
            if len(snip) > len(entry["snippet"]) and not snip.startswith("Redirecting"):
                entry["snippet"] = snip
            bad = lambda t: " › " in t or t.startswith("http") or len(t) < 3
            t = hit.get("title", "") or ""
            if not bad(t) and (bad(entry["title"]) or len(t) < len(entry["title"])):
                entry["title"] = t
            if entry.get("published") is None and (hit.get("published") or hit.get("date")):
                entry["published"] = hit.get("published") or hit.get("date")
            entry["sources"].append((engine, rank))

    results = list(groups.values())
    for (key, _family), contribution in family_best.items():
        entry = groups.get(key)
        if entry is not None:
            entry["score"] += contribution

    # Consensus multiplier (independent indexes only)
    for r in results:
        independent = []
        for e, _ in r["sources"]:
            fam = engine_family(e)
            if not any(engine_family(x) == fam for x in independent):
                independent.append(e)
        consensus = len(independent)
        r["score"] *= 1.0 + CONSENSUS_MULT * max(consensus - 1.0, 0.0)

    # BM25 relevance bonus
    docs = [(r["title"], r["snippet"]) for r in results]
    rel = _relevance(query, docs)
    max_rel = max(rel) if rel else 0.0
    if max_rel > 1e-9:
        for r, rv in zip(results, rel):
            r["score"] += BM25_WEIGHT * (rv / max_rel)

    # Domain prior bonus
    for r in results:
        prior = domain_prior(intent, host_of(r["url"]), query)
        r["score"] += PRIOR_WEIGHT * prior

    # Vertical-only penalty: un-corroborated vertical hits are weak signals
    for r in results:
        if not any(not is_vertical(e) for e, _ in r["sources"]):
            r["score"] *= 0.4

    return results


def blend(results: list[dict], semantic_scores: list[float | None]) -> None:
    """60/40 min-max blend of base score with cross-encoder scores.
    semantic_scores is aligned with results (None = no semantic score)."""
    rrf = [r["score"] for r in results]
    xenc = [s if s is not None else 0.5 for s in semantic_scores]
    rrf_n = _min_max(rrf)
    xenc_n = _min_max(xenc)
    for r, (rn, xn) in zip(results, zip(rrf_n, xenc_n)):
        r["score"] = BLEND_ALPHA * rn + (1.0 - BLEND_ALPHA) * xn


def _min_max(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / span for v in values]


def is_weak(results: list[dict], merged_total: int) -> bool:
    """No cross-family consensus on the top result + shallow merge."""
    if not results:
        return True
    top = results[0]
    families = {engine_family(e) for e, _ in top["sources"]}
    return len(families) < 2 and merged_total < 8


def merged_total(per_engine: dict[str, list[dict]]) -> int:
    return len({norm_key(h["url"]) for hits in per_engine.values() for h in hits if h.get("url")})


def finalize(results: list[dict], max_results: int) -> list[dict]:
    """Sort desc, collapse syndicated (identical-title) content,
    per-domain diversity cap, truncate."""
    results.sort(key=lambda r: r["score"], reverse=True)
    deduped = []
    seen_titles: set[str] = set()
    for r in results:
        title_key = " ".join(
            "".join(c for c in r["title"].lower().strip() if c.isalnum() or c == " ").split())
        if title_key and title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        deduped.append(r)
    domain_count: dict[str, int] = {}
    diverse, overflow = [], []
    for r in deduped:
        host = host_of(r["url"])
        c = domain_count.get(host, 0)
        if c < MAX_PER_DOMAIN:
            domain_count[host] = c + 1
            diverse.append(r)
        else:
            overflow.append(r)
        if len(diverse) >= max_results:
            break
    if len(diverse) < max_results:
        diverse.extend(overflow[: max_results - len(diverse)])
    return diverse[:max_results]


# ────────────────────────────────────────────────────────────────
# intent.rs
# ────────────────────────────────────────────────────────────────

_CODE_STRONG = ["code", "programming", "syntax", "error", "exception", "compile",
                "compiler", "debug", "debugging", "runtime", "dependency", "api"]
_CODE_TECH_GATED = ["library", "framework", "function", "method", "class", "package",
                    "module", "install", "setup", "implementation", "api"]
_TECH_WORDS = ["python", "rust", "go", "java", "javascript", "typescript", "c++", "c#",
               "cpp", "kotlin", "swift", "ruby", "php", "dart", "zig", "elixir", "haskell",
               "scala", "julia", "lua", "react", "vue", "angular", "svelte", "node",
               "nodejs", "nextjs", "django", "flask", "fastapi", "rails", "flutter",
               "tailwind", "vite", "webpack", "docker", "kubernetes", "terraform",
               "ansible", "nginx", "postgres", "postgresql", "mysql", "sqlite",
               "mongodb", "redis", "kafka", "pytorch", "tensorflow", "huggingface",
               "numpy", "pandas", "opencv", "ollama", "langchain", "mcp", "git", "vim",
               "neovim", "emacs", "bash", "ffmpeg", "curl", "openssl", "ssh", "linux",
               "arch", "ubuntu", "debian", "fedora", "dotnet", "golang", "csharp", "htmx"]
_PAPER_SIGNALS = ["paper", "papers", "study", "studies", "research", "arxiv", "survey",
                  "thesis", "experiment", "dataset", "benchmark"]
_NEWS_SIGNALS = ["news", "headlines", "breaking", "latest", "today", "update", "coverage",
                 "election", "market", "stock", "war", "crash", "report"]
_ENTITY_SIGNALS = ["what is", "who is", "who was", "define", "meaning of"]
_CONCEPT_SIGNALS = ["what is", "who is", "who was", "explained", "explain", "how does",
                    "how do ", "meaning", "concept", "theory", "mechanism", "history of",
                    "difference between", " vs "]


def is_conceptual(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _CONCEPT_SIGNALS)


def detect_intent(query: str) -> str:
    q = query.lower()
    tech = any(w in re.split(r"[^a-zA-Z0-9+]+", q) for w in _TECH_WORDS)
    code = sum(1 for s in _CODE_STRONG if s in q) + (sum(1 for s in _CODE_TECH_GATED if s in q) + 1 if tech else 0)
    paper = sum(1 for s in _PAPER_SIGNALS if s in q)
    news = sum(1 for s in _NEWS_SIGNALS if s in q)
    entity = sum(1 for s in _ENTITY_SIGNALS if s in q)
    best = max(code, paper, news, entity)
    if best == 0:
        words = query.split()
        if len(words) <= 3 and sum(1 for w in words if w[:1].isupper()) >= 2:
            return "entity"
        return "web"
    if code == best:
        return "code"
    if news == best:
        return "news"
    if paper == best:
        return "paper"
    return "entity"


def verticals_for(intent: str, query: str) -> list[str]:
    if intent == "code":
        return ["stackexchange", "mdn", "github", "hn"]
    if intent == "paper":
        return ["scholar", "arxiv"]
    if intent == "news":
        return ["news", "hn"]
    if intent == "entity":
        return ["wikipedia"]
    if intent == "web" and is_conceptual(query):
        return ["wikipedia"]
    return []


_DOMAIN_PRIOR = {
    "code": ["stackoverflow.com", "github.com", "docs.rs", "developer.mozilla.org",
             "learn.microsoft.com", "doc.rust-lang.org", "pkg.go.dev", "pypi.org",
             "crates.io", "npmjs.com", "readthedocs.io", "superuser.com",
             "serverfault.com", "news.ycombinator.com", "git-scm.com"],
    "paper": ["arxiv.org", "semanticscholar.org", "scholar.google.com", "nature.com",
              "science.org", "acm.org", "ieee.org", "openreview.net",
              "pubmed.ncbi.nlm.nih.gov", "doi.org"],
    "news": ["reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
             "theguardian.com", "arstechnica.com", "techcrunch.com",
             "news.ycombinator.com", "bloomberg.com", "wsj.com"],
    "entity": ["wikipedia.org", "britannica.com", "wikidata.org", "imdb.com"],
    "web": ["cloudflare.com", "developer.mozilla.org", "wikipedia.org",
            "learn.microsoft.com", "aws.amazon.com", "kubernetes.io", "ietf.org",
            "rfc-editor.org"],
}

_UTILITY_WORDS = ["how to", "fix", "repair", "install", "replace", "clean", "build",
                  "make", "remove"]
_DIY_DOMAINS = ["ifixit.com", "wikihow.com", "thisoldhouse.com", "familyhandyman.com",
                "homedepot.com", "lowes.com", "thespruce.com", "hgtv.com",
                "instructables.com"]


def utility_prior(query: str, host: str) -> float:
    q = query.lower()
    if not any(w in q for w in _UTILITY_WORDS):
        return 0.0
    return 1.0 if _host_in(host, _DIY_DOMAINS) else 0.0


def domain_prior(intent: str, host: str, query: str) -> float:
    table = _DOMAIN_PRIOR.get(intent, [])
    prior = 1.0 if _host_in(host, table) else 0.0
    return max(prior, utility_prior(query, host))


def variant(query: str) -> str | None:
    """Strip question scaffolding for recall variants."""
    q = query
    for pre in ("how to ", "how do i ", "how do you ", "what is ", "who is ",
                "why does ", "why is "):
        if q.lower().startswith(pre):
            q = q[len(pre):]
            break
    q = q.rstrip("?.").strip()
    return q or None


def _host_in(host: str, domains: list[str]) -> bool:
    return any(host == d or host.endswith(f".{d}") for d in domains)


# ────────────────────────────────────────────────────────────────
# coverage.rs
# ────────────────────────────────────────────────────────────────

ANCHOR_MISS_PENALTY = 0.3
SPECIFIER_MISMATCH_PENALTY = 0.3


def _extract_entities(query: str) -> tuple[list[dict], list[dict]]:
    anchors, specifiers = [], []
    for token in query.split():
        clean = token.lower().strip("!\"#$%&'()*+,/:;<=>?@[\\]^_`{|}~")
        clean = clean.strip("-.")
        if not clean or len(clean) <= 1:
            continue
        if "-" in clean and any(c.isalpha() for c in clean):
            anchors.append({"variants": [clean, clean.replace("-", " "), clean.replace("-", "")]})
            continue
        v = _parse_version(clean)
        if v:
            specifiers.append({"variants": [v]})
            continue
        if _is_year(clean):
            specifiers.append({"variants": [clean]})
    return anchors, specifiers


def _parse_version(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    if len(parts[0]) > 3 or len(parts[1]) > 3:
        return None
    if not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return token


def _is_year(token: str) -> bool:
    return len(token) == 4 and token.startswith("20") and token.isdigit()


def _find_versions(text: str) -> list[str]:
    return re.findall(r"(?<!\d)\d{1,3}\.\d{1,3}(?!\d)", text)


def _find_years(text: str) -> list[str]:
    return re.findall(r"(?<!\d)20\d\d(?!\d)", text)


def apply_coverage(query: str, results: list[dict]) -> None:
    """Entity coverage penalties — post-rerank, before authority."""
    anchors, specifiers = _extract_entities(query)
    if not anchors and not specifiers:
        return
    query_versions = [e["variants"][0] for e in specifiers if _parse_version(e["variants"][0])]
    query_years = [e["variants"][0] for e in specifiers if _is_year(e["variants"][0])]
    for r in results:
        text = f"{r['title']} {r['snippet']} {r['url']}".lower()
        penalty = 1.0
        for a in anchors:
            if not any(v in text for v in a["variants"]):
                penalty *= ANCHOR_MISS_PENALTY
        if not any(v in text for v in query_versions):
            for qv in query_versions:
                major = qv.split(".")[0]
                found = _find_versions(text)
                if any(fv != qv and fv.split(".")[0] == major for fv in found):
                    penalty *= SPECIFIER_MISMATCH_PENALTY
                    break
        if not any(y in text for y in query_years) and query_years:
            if any(fy not in query_years for fy in _find_years(text)):
                penalty *= SPECIFIER_MISMATCH_PENALTY
        r["score"] *= penalty


# ────────────────────────────────────────────────────────────────
# authority.rs
# ────────────────────────────────────────────────────────────────

_OFFICIAL: list[tuple[list[str], list[str]]] = [
    # ── Languages & runtimes ──
    (["rust"], ["rust-lang.org", "doc.rust-lang.org", "docs.rs", "crates.io"]),
    (["python"], ["docs.python.org", "pypi.org"]),
    (["javascript"], ["developer.mozilla.org"]),
    (["typescript"], ["typescriptlang.org"]),
    (["nodejs"], ["nodejs.org"]),
    (["node", "js"], ["nodejs.org"]),
    (["node", "javascript"], ["nodejs.org"]),
    (["golang"], ["go.dev"]),
    (["go", "goroutine"], ["go.dev"]),
    (["go", "concurrency"], ["go.dev"]),
    (["go", "channel"], ["go.dev"]),
    (["java"], ["docs.oracle.com", "dev.java"]),
    (["kotlin"], ["kotlinlang.org"]),
    (["swift"], ["swift.org", "developer.apple.com"]),
    (["ruby"], ["ruby-lang.org", "ruby-doc.org"]),
    (["php"], ["php.net"]),
    (["cpp"], ["cppreference.com"]),
    (["dotnet"], ["learn.microsoft.com", "dotnet.microsoft.com"]),
    (["csharp"], ["learn.microsoft.com"]),
    (["dotnet", "c#"], ["learn.microsoft.com"]),
    (["dart"], ["dart.dev"]),
    (["zig"], ["ziglang.org"]),
    (["elixir"], ["elixir-lang.org"]),
    (["haskell"], ["haskell.org"]),
    (["scala"], ["scala-lang.org"]),
    (["julia"], ["docs.julialang.org"]),
    (["lua"], ["lua.org"]),
    # ── Frameworks & libraries ──
    (["react"], ["react.dev"]),
    (["react", "hooks"], ["react.dev"]),
    (["vue"], ["vuejs.org"]),
    (["angular"], ["angular.dev"]),
    (["svelte"], ["svelte.dev"]),
    (["nextjs"], ["nextjs.org"]),
    (["next", "js"], ["nextjs.org"]),
    (["nuxt"], ["nuxt.com"]),
    (["django"], ["docs.djangoproject.com"]),
    (["flask"], ["flask.palletsprojects.com"]),
    (["fastapi"], ["fastapi.tiangolo.com"]),
    (["rails"], ["guides.rubyonrails.org"]),
    (["ruby", "on", "rails"], ["guides.rubyonrails.org"]),
    (["spring", "boot"], ["spring.io"]),
    (["spring", "framework"], ["spring.io"]),
    (["spring", "java"], ["spring.io"]),
    (["laravel"], ["laravel.com"]),
    (["flutter"], ["flutter.dev", "api.flutter.dev"]),
    (["react", "native"], ["reactnative.dev"]),
    (["tailwind"], ["tailwindcss.com"]),
    (["bootstrap"], ["getbootstrap.com"]),
    (["vite"], ["vitejs.dev"]),
    (["webpack"], ["webpack.js.org"]),
    (["expressjs"], ["expressjs.com"]),
    (["express", "js"], ["expressjs.com"]),
    (["express", "node"], ["expressjs.com"]),
    (["tokio"], ["tokio.rs"]),
    (["serde"], ["serde.rs"]),
    (["jquery"], ["jquery.com"]),
    (["htmx"], ["htmx.org"]),
    # ── Data stores & messaging ──
    (["postgresql"], ["postgresql.org"]),
    (["postgres"], ["postgresql.org"]),
    (["mysql"], ["dev.mysql.com", "mysql.com"]),
    (["sqlite"], ["sqlite.org"]),
    (["mongodb"], ["mongodb.com"]),
    (["redis"], ["redis.io"]),
    (["kafka"], ["kafka.apache.org"]),
    (["elasticsearch"], ["elastic.co"]),
    (["rabbitmq"], ["rabbitmq.com"]),
    (["clickhouse"], ["clickhouse.com"]),
    # ── Infra & DevOps ──
    (["docker"], ["docs.docker.com"]),
    (["kubernetes"], ["kubernetes.io"]),
    (["k8s"], ["kubernetes.io"]),
    (["terraform"], ["developer.hashicorp.com", "terraform.io"]),
    (["ansible"], ["docs.ansible.com"]),
    (["nginx"], ["nginx.org"]),
    (["apache"], ["httpd.apache.org"]),
    (["caddy"], ["caddyserver.com"]),
    (["traefik"], ["doc.traefik.io"]),
    (["haproxy"], ["haproxy.org"]),
    (["prometheus"], ["prometheus.io"]),
    (["grafana"], ["grafana.com"]),
    (["jenkins"], ["jenkins.io"]),
    (["gitlab"], ["docs.gitlab.com"]),
    (["systemd"], ["systemd.io"]),
    (["nginx", "ingress"], ["kubernetes.io", "docs.docker.com"]),
    # ── Cloud & platforms ──
    (["aws"], ["aws.amazon.com", "docs.aws.amazon.com"]),
    (["amazon", "web", "services"], ["aws.amazon.com", "docs.aws.amazon.com"]),
    (["azure"], ["learn.microsoft.com", "azure.microsoft.com"]),
    (["gcp"], ["cloud.google.com"]),
    (["google", "cloud"], ["cloud.google.com"]),
    (["vercel"], ["vercel.com"]),
    (["netlify"], ["docs.netlify.com"]),
    (["cloudflare"], ["developers.cloudflare.com", "cloudflare.com"]),
    (["fastly"], ["developer.fastly.com"]),
    (["heroku"], ["devcenter.heroku.com"]),
    (["digitalocean"], ["docs.digitalocean.com"]),
    (["supabase"], ["supabase.com"]),
    (["firebase"], ["firebase.google.com"]),
    (["stripe"], ["docs.stripe.com", "stripe.com"]),
    (["twilio"], ["twilio.com"]),
    (["fly", "io"], ["fly.io"]),
    # ── Dev tools ──
    (["git"], ["git-scm.com"]),
    (["github", "actions"], ["docs.github.com"]),
    (["archlinux"], ["wiki.archlinux.org"]),
    (["arch", "linux"], ["wiki.archlinux.org"]),
    (["ubuntu"], ["help.ubuntu.com", "documentation.ubuntu.com"]),
    (["debian"], ["debian.org"]),
    (["fedora"], ["docs.fedoraproject.org"]),
    (["neovim"], ["neovim.io"]),
    (["vim"], ["vimhelp.org"]),
    (["emacs"], ["gnu.org"]),
    (["bash"], ["gnu.org"]),
    (["ffmpeg"], ["ffmpeg.org"]),
    (["curl"], ["curl.se", "everything.curl.dev"]),
    (["wget"], ["gnu.org"]),
    (["openssl"], ["openssl.org", "docs.openssl.org"]),
    (["openssh"], ["man.openbsd.org", "openssh.com"]),
    (["ssh"], ["man.openbsd.org", "openssh.com"]),
    (["pip"], ["pip.pypa.io"]),
    (["npm"], ["docs.npmjs.com"]),
    (["yarn"], ["yarnpkg.com"]),
    (["homebrew"], ["docs.brew.sh"]),
    (["cargo", "rust"], ["doc.rust-lang.org"]),
    # ── AI/ML ──
    (["pytorch"], ["pytorch.org"]),
    (["tensorflow"], ["tensorflow.org"]),
    (["huggingface"], ["huggingface.co"]),
    (["scikit"], ["scikit-learn.org"]),
    (["sklearn"], ["scikit-learn.org"]),
    (["numpy"], ["numpy.org"]),
    (["pandas"], ["pandas.pydata.org"]),
    (["opencv"], ["docs.opencv.org"]),
    (["ollama"], ["ollama.com"]),
    (["openai"], ["openai.com", "platform.openai.com"]),
    (["anthropic"], ["anthropic.com", "docs.anthropic.com"]),
    (["claude"], ["docs.anthropic.com", "anthropic.com"]),
    (["langchain"], ["python.langchain.com"]),
    (["mcp"], ["modelcontextprotocol.io"]),
    (["model", "context", "protocol"], ["modelcontextprotocol.io"]),
    (["transformers", "huggingface"], ["huggingface.co"]),
    # ── Protocols & specs ──
    (["oauth"], ["oauth.net"]),
    (["jwt"], ["jwt.io"]),
    (["json", "web", "token"], ["jwt.io"]),
    (["grpc"], ["grpc.io"]),
    (["graphql"], ["graphql.org"]),
    (["websocket"], ["developer.mozilla.org"]),
    (["websockets"], ["developer.mozilla.org"]),
    (["jsonrpc"], ["jsonrpc.org"]),
    (["json", "rpc"], ["jsonrpc.org"]),
    (["rfc"], ["rfc-editor.org", "datatracker.ietf.org"]),
    (["ietf"], ["datatracker.ietf.org", "ietf.org"]),
    # ── Research repositories ──
    (["arxiv"], ["arxiv.org"]),
    (["semanticscholar"], ["semanticscholar.org"]),
    (["openreview"], ["openreview.net"]),
]

_TITLE_STOPWORDS = {
    "the", "and", "for", "how", "what", "why", "who", "does", "with", "using", "from",
    "into", "about", "are", "was", "were", "has", "have", "you", "your", "can", "not",
    "but", "all", "any", "get", "set", "new", "latest", "news", "update", "updates",
    "official", "documentation", "docs", "doc", "guide", "tutorial", "explained",
    "example", "examples", "vs", "versus", "best", "good", "modern", "simple",
    "complete", "difference", "between", "when",
}
_DOCS_WORDS = ["docs", "documentation", "documented", "reference", "api", "specification",
               "spec", "official", "manual", "man page", "rfc", "changelog", "release notes"]
_PAPER_WORDS = ["paper", "papers", "study", "studies", "research", "arxiv"]
_PAPER_AUTHORITY = ["arxiv.org", "openreview.net", "semanticscholar.org",
                    "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "aclanthology.org",
                    "biorxiv.org", "papers.neurips.cc", "openaccess.thecvf.com"]

OFFICIAL_MULT = 1.6
OFFICIAL_CODE_MULT = 1.8
DOCS_SEEK_MULT = 1.15
PRIOR_MULT = 1.2
PAPER_SEEK_MULT = 1.35
TITLE_W = 0.35
PHRASE_MULT = 1.15


def _official_domains(query: str) -> list[str]:
    toks = {t for t in re.split(r"[^a-zA-Z0-9]", query.lower()) if len(t) >= 2}
    out = []
    for entry_tokens, domains in _OFFICIAL:
        if all(t in toks for t in entry_tokens):
            out.extend(domains)
    return out


def _title_terms_ratio(query: str, title: str) -> float:
    terms = [t for t in re.split(r"[^a-zA-Z0-9]", query.lower())
             if len(t) > 2 and t not in _TITLE_STOPWORDS]
    if not terms:
        return 0.0
    tl = title.lower()
    return sum(1 for t in terms if t in tl) / len(terms)


def _phrase_in_title(query: str, title: str) -> bool:
    norm = lambda s: " ".join(re.split(r"[^a-zA-Z0-9]", s.lower())).strip()
    q = norm(query)
    if len(q) < 8:
        return False
    return q in norm(title)


def _freshness_mult(published: str | None) -> float:
    if not published:
        return 1.0
    days = _iso_days_ago(published)
    if days is None:
        return 1.0
    if days <= 1:
        return 1.5
    if days <= 3:
        return 1.3
    if days <= 7:
        return 1.15
    if days <= 30:
        return 1.0
    return 0.85


def _iso_days_ago(iso: str) -> int | None:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12) or not (1 <= d <= 31):
        return None
    then = _days_from_civil(y, mo, d)
    now = int(time.time() // 86400)
    return now - then


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Howard Hinnant's days_from_civil — proleptic Gregorian."""
    y = y - 1 if m <= 2 else y
    era = y // 400 if y >= 0 else (y - 399) // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def apply_authority(query: str, intent: str, results: list[dict]) -> None:
    """Top-placement layer: official domains, title decisiveness,
    news freshness. Multiplicative, so near-zero misses stay down."""
    official = _official_domains(query)
    docs_seek = any(w in query.lower() for w in _DOCS_WORDS)
    paper_seek = any(w in query.lower() for w in _PAPER_WORDS)
    for r in results:
        host = host_of(r["url"])
        m = 1.0
        official_hit = bool(official) and _host_in(host, official)
        if official_hit:
            m *= OFFICIAL_CODE_MULT if intent == "code" else OFFICIAL_MULT
            if docs_seek:
                m *= DOCS_SEEK_MULT
        elif domain_prior(intent, host, query) > 0.0:
            m *= PRIOR_MULT
        if paper_seek and not official_hit and _host_in(host, _PAPER_AUTHORITY):
            m *= PAPER_SEEK_MULT
        ratio = _title_terms_ratio(query, r["title"])
        m *= 1.0 + TITLE_W * ratio
        if _phrase_in_title(query, r["title"]):
            m *= PHRASE_MULT
        if intent == "news":
            m *= _freshness_mult(r.get("published"))
        r["score"] *= m


# ────────────────────────────────────────────────────────────────
# Self-check: donsetch's own benchmark shapes must hold.
# ────────────────────────────────────────────────────────────────

def _selfcheck() -> None:
    def hit(url, title="t", snippet="s", rank=0):
        return {"url": url, "title": title, "snippet": snippet, "rank": rank}

    # 1. consensus beats vertical-only
    per = {
        "duckduckgo": [hit("https://a.com/x", rank=5)],
        "tavily": [hit("https://a.com/x", rank=5)],
        "wikipedia": [hit("https://b.com/y", rank=0)],
    }
    out = merge_base(per, "rust async runtime", "code")
    out.sort(key=lambda r: r["score"], reverse=True)
    assert out[0]["url"] == "https://a.com/x", f"consensus must win: {out}"

    # 2. general engine beats vertical-only at same consensus
    per = {
        "duckduckgo": [hit("https://doc.rust-lang.org/borrow",
                           "Borrowing - Rust By Example",
                           "Rust uses a borrowing mechanism to access data without taking ownership", 3)],
        "github": [hit("https://github.com/zigsafe",
                       "zigsafe: ownership checker for Zig, Rust-style borrow-check",
                       "Optional static ownership checker for Zig with Rust-style borrow-check diagnostics", 0)],
    }
    out = merge_base(per, "zigsafe ownership checker", "code")
    out.sort(key=lambda r: r["score"], reverse=True)
    assert out[0]["url"] == "https://doc.rust-lang.org/borrow", f"general must beat vertical: {out}"

    # 3. authority flips official doc over aggregator
    rs = [
        {"title": "Understanding Ownership in Rust - SomeBlog", "url": "https://someblog.dev/rust-ownership",
         "snippet": "", "sources": [("duckduckgo", 0)], "score": 0.80, "published": None},
        {"title": "Ownership - Rust By Example", "url": "https://doc.rust-lang.org/rust-by-example/scope/ownership.html",
         "snippet": "", "sources": [("duckduckgo", 1)], "score": 0.55, "published": None},
    ]
    apply_authority("rust ownership explained", "web", rs)
    rs.sort(key=lambda r: r["score"], reverse=True)
    assert "doc.rust-lang.org" in rs[0]["url"], f"official must place first: {rs}"

    # 4. coverage: B-tree vs binary tree
    rs = [
        {"title": "Binary Tree Traversal", "url": "https://x.com/tree", "snippet": "binary tree traversal explained",
         "sources": [("duckduckgo", 0)], "score": 0.8, "published": None},
        {"title": "B-Tree Implementation in C", "url": "https://y.com/btree", "snippet": "B-tree data structure",
         "sources": [("duckduckgo", 1)], "score": 0.5, "published": None},
    ]
    apply_coverage("b-tree vs binary tree", rs)
    assert rs[0]["score"] < rs[1]["score"], f"coverage must fix b-tree: {rs}"

    # 5. version drift: 5.2 query vs 5.5 result
    rs = [
        {"title": "Python 5.5 changes", "url": "https://z.com/55", "snippet": "python 5.5 release",
         "sources": [("duckduckgo", 0)], "score": 0.8, "published": None},
        {"title": "Python 5.2 changes", "url": "https://w.com/52", "snippet": "python 5.2 release",
         "sources": [("duckduckgo", 1)], "score": 0.6, "published": None},
    ]
    apply_coverage("python 5.2 changes", rs)
    assert rs[0]["score"] < rs[1]["score"], f"version drift must penalize: {rs}"

    # 6. intent detection
    assert detect_intent("rust async runtime comparison") == "code"
    assert detect_intent("latest news on the election") == "news"
    assert detect_intent("arxiv paper on transformers") == "paper"
    assert is_conceptual("what is a monad")

    # 7. syndicated dedup + diversity cap
    rs = [
        {"title": "Storm hits town", "url": "https://fox43.com/s", "snippet": "", "sources": [("duckduckgo", 0)], "score": 0.9, "published": None},
        {"title": "Storm hits town", "url": "https://localmemphis.com/s", "snippet": "", "sources": [("tavily", 1)], "score": 0.5, "published": None},
        {"title": "Storm hits town", "url": "https://third.local/s", "snippet": "", "sources": [("anysearch", 2)], "score": 0.4, "published": None},
    ]
    out = finalize(rs, 10)
    assert len(out) == 1, f"syndicated dedup failed: {out}"

    print("merge.py selfcheck: 7/7 OK")


if __name__ == "__main__":
    _selfcheck()
