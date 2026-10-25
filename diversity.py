"""Stop one website from becoming "the" answer.

A search returns results ranked by relevance, and relevance is not diversity: a
single prolific domain can own six of the top ten (docs sites, aggregators,
stackoverflow, one vendor's blog network). Feeding all six to a synthesis model
is a bias generator - the model sees one framing repeated and reports it as the
consensus, with citations that all point at the same author.

So the context handed to the model is capped per registrable domain. The cap
relaxes only if the pool is too small to fill the context otherwise, and when it
relaxes the result says so (`per_domain_effective`), because "we admitted
duplicates" is information the caller should get rather than a silent fallback.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit

# Best-effort public-suffix handling. A real PSL parser is a dependency and a
# data file for a cap that only needs to be approximately right: getting
# `bbc.co.uk` and `bbc.com` into the same bucket matters, getting every
# obscure second-level domain perfect does not. Unlisted suffixes fall back to
# the last two labels, which is correct for the vast majority of hosts.
_TWO_LABEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk", "sch.uk",
    "co.in", "net.in", "org.in", "ac.in", "gov.in", "firm.in", "gen.in", "ind.in",
    "co.jp", "or.jp", "ne.jp", "ac.jp", "go.jp",
    "co.kr", "or.kr", "ac.kr", "go.kr",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "id.au",
    "com.br", "net.br", "org.br", "gov.br",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.mx", "com.ar", "com.tr", "com.sg", "com.hk", "com.tw",
    "co.za", "co.nz", "co.il", "co.id", "co.th", "co.ke", "co.ve",
    "com.ua", "com.pl", "com.es", "com.my", "com.ph", "com.vn", "com.pk",
    "com.eg", "com.sa", "com.ng",
    "org.au", "net.nz", "org.nz", "govt.nz",
}

# Hosts whose subdomains are genuinely different publishers, not one site. Each
# subdomain carries its own community and its own framing.
_MULTI_TENANT = (
    "github.io", "gitlab.io", "substack.com", "wordpress.com", "blogspot.com",
    "medium.com", "reddit.com", "stackexchange.com", "stackoverflow.com",
    "readthedocs.io", "notion.site", "vercel.app", "netlify.app", "pages.dev",
    "workers.dev", "tumblr.com", "quora.com", "zhihu.com", "csdn.net",
)


def registrable(host: str) -> str:
    """Reduce a host to the unit that counts as 'one publisher'.

    Returns '' for an empty host so callers can group junk results together
    instead of crashing on them.
    """
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    # An IP address is its own publisher.
    if host.replace(".", "").isdigit():
        return host
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    # A multi-tenant host keeps one extra label: alice.github.io != bob.github.io
    for mt in _MULTI_TENANT:
        if host == mt or host.endswith("." + mt):
            base = mt.split(".")[0]
            # keep the tenant label in front of the platform domain
            idx = len(labels) - len(mt.split(".")) - 1
            return ".".join(labels[max(0, idx):]) if idx >= 0 else ".".join(labels[-3:])
    last_two = ".".join(labels[-2:])
    if last_two in _TWO_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return last_two


def domain_of(url: str) -> str:
    try:
        return registrable(urlsplit(url).hostname or "")
    except Exception:
        return ""


def drop_near_duplicates(items: list[dict], threshold: float = 0.88,
                         url_key: str = "url", title_key: str = "title",
                         snippet_key: str = "snippet") -> tuple[list[dict], dict]:
    """Remove syndicated copies of the same article.

    A wire story published on five sites is still one source, and a synthesis
    that cites all five looks corroborated while being a single origin. Titles
    are compared because the same story keeps its headline across syndication
    while the URL changes completely.

    Only the *later* copy is dropped: the first appearance in rank order wins, so
    ordering still means something. Different titles on the same topic are kept -
    this catches duplication, not similarity of subject.
    """
    if not items:
        return [], {"dropped": [], "removed": 0}

    kept: list[dict] = []
    dropped: list[dict] = []
    seen_keys: set[str] = set()
    for it in items:
        title = (it.get(title_key) or "").strip()
        url = it.get(url_key) or ""

        def norm(t: str) -> str:
            return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split())

        nt = norm(title)
        if not nt:
            kept.append(it)
            continue
        # Cheap exact check first, then fuzzy against the kept titles.
        if nt in seen_keys:
            dropped.append({"url": url, "title": title, "reason": "same title"})
            continue
        dup_of = ""
        for prev in kept:
            pn = norm(prev.get(title_key) or "")
            if not pn:
                continue
            ratio = SequenceMatcher(None, nt, pn).ratio()
            if ratio >= threshold:
                dup_of = prev.get(url_key, "")
                break
        if dup_of:
            dropped.append({"url": url, "title": title,
                            "reason": f"near-duplicate title of {dup_of}"})
            continue
        seen_keys.add(nt)
        kept.append(it)

    return kept, {"dropped": dropped[:10], "removed": len(dropped)}


def cap_per_domain(items: list[dict], limit: int, per_domain: int = 2,
                   min_sources: int = 3, url_key: str = "url") -> tuple[list[dict], dict]:
    """Keep at most `per_domain` items per domain, up to `limit` items.

    `limit` is a ceiling, not a target. Returning four diverse sources beats
    returning six where four share an author, because the cap exists precisely to
    stop one publisher's framing being reported as consensus.

    The cap only relaxes to guarantee a usable floor (`min_sources`), and never
    beyond it: a thin pool still yields a few duplicate domains rather than an
    empty context, and the effective cap is reported so the caller can see that
    duplicates were admitted.
    """
    if not items:
        return [], {"domains": [], "n_domains": 0, "per_domain_effective": per_domain,
                    "duplicate_domains": {}, "relaxed": False, "requested": limit}

    limit = max(1, int(limit))
    per_domain = max(1, int(per_domain))
    floor = min(limit, max(1, int(min_sources)))

    def take(cap: int, cap_n: int) -> list[dict]:
        counts: dict[str, int] = {}
        kept: list[dict] = []
        for it in items:
            d = domain_of(it.get(url_key, "") or "")
            if counts.get(d, 0) >= cap:
                continue
            counts[d] = counts.get(d, 0) + 1
            kept.append(it)
            if len(kept) >= cap_n:
                break
        return kept

    kept = take(per_domain, limit)
    effective = per_domain
    relaxed = False
    if len(kept) < floor:
        for cap in (per_domain * 2, 10 ** 6):
            kept = take(cap, floor)
            effective = cap if cap < 10 ** 6 else -1  # -1 = unbounded, stated
            relaxed = True
            if len(kept) >= floor:
                break

    domains = [domain_of(it.get(url_key, "") or "") for it in kept]
    dupes: dict[str, int] = {}
    for d in domains:
        dupes[d] = dupes.get(d, 0) + 1
    dupes = {d: n for d, n in dupes.items() if n > 1}

    return kept, {
        "n_domains": len({d for d in domains if d}),
        "domains": [d for d in dict.fromkeys(domains) if d],
        "per_domain_effective": effective,
        "relaxed": relaxed,
        "duplicate_domains": dupes,
        "requested": limit,
        "returned": len(kept),
    }


def _selfcheck() -> None:
    checks: list[tuple[str, bool]] = []

    checks.append(("www stripped", registrable("www.example.com") == "example.com"))
    checks.append(("subdomain collapses", registrable("docs.example.com") == "example.com"))
    checks.append(("co.uk handling", registrable("news.bbc.co.uk") == "bbc.co.uk"))
    checks.append(("co.in handling", registrable("www.flipkart.co.in") == "flipkart.co.in"))
    checks.append(("plain two-label host", registrable("example.com") == "example.com"))
    checks.append(("ip is its own publisher",
                   registrable("192.168.1.1") == "192.168.1.1"))
    checks.append(("github pages keep tenant",
                   registrable("alice.github.io") == "alice.github.io"
                   and registrable("bob.github.io") == "bob.github.io"))
    checks.append(("github pages tenants differ",
                   registrable("alice.github.io") != registrable("bob.github.io")))
    checks.append(("empty host is safe", registrable("") == ""))
    checks.append(("no port confusion", domain_of("https://example.com:8443/a") == "example.com"))
    checks.append(("junk url is safe", domain_of("not a url") == ""))

    items = [{"url": f"https://a.com/{i}"} for i in range(6)] + \
            [{"url": f"https://b.com/{i}"} for i in range(4)]
    kept, info = cap_per_domain(items, limit=6, per_domain=2)
    checks.append(("cap keeps 2 per domain", len(kept) == 4))
    checks.append(("cap reports the domains", info["n_domains"] == 2))
    checks.append(("cap reports no dupes when capped",
                   all(n <= 2 for n in info["duplicate_domains"].values())))

    # A pool with only one domain must still yield a usable floor, and must say
    # it relaxed rather than silently returning two items with no explanation.
    one_domain = [{"url": f"https://a.com/{i}"} for i in range(10)]
    kept2, info2 = cap_per_domain(one_domain, limit=6, per_domain=2)
    checks.append(("single-domain pool reaches the floor", len(kept2) == 3))
    checks.append(("relaxation is reported", info2["relaxed"] is True))
    checks.append(("relaxation reports the effective cap",
                   info2["per_domain_effective"] != 2))
    # The whole point: relaxation must NOT let one domain take the whole context.
    checks.append(("relaxation never fills the context from one domain",
                   len(kept2) < 6))

    # Order must be preserved: rank still means something inside the kept set.
    kept3, _ = cap_per_domain(items, limit=4, per_domain=2)
    checks.append(("input order preserved",
                   [it["url"] for it in kept3] ==
                   ["https://a.com/0", "https://a.com/1",
                    "https://b.com/0", "https://b.com/1"]))

    kept4, info4 = cap_per_domain([], limit=5)
    checks.append(("empty input is safe", kept4 == [] and info4["n_domains"] == 0))

    checks.append(("limit never exceeded",
                   len(cap_per_domain(items, limit=3, per_domain=2)[0]) == 3))
    checks.append(("per_domain of 1 relaxes only to the floor",
                   len(cap_per_domain(items, limit=10, per_domain=1)[0]) == 3))
    checks.append(("limit is a ceiling, not a target",
                   cap_per_domain(items, limit=9, per_domain=2)[1]["returned"] == 4))
    checks.append(("a pool at the floor does not relax",
                   cap_per_domain(items, limit=4, per_domain=2)[1]["relaxed"] is False))

    # Syndication: the same headline on five sites is one source.
    synd = [
        {"url": "https://a.com/x", "title": "Rust 1.90 released with new borrow checker"},
        {"url": "https://b.com/y", "title": "Rust 1.90 released with new borrow checker"},
        {"url": "https://c.com/z", "title": "Rust 1.90 Released With New Borrow Checker!"},
        {"url": "https://d.com/q", "title": "Something else entirely about Go"},
    ]
    kept5, info5 = drop_near_duplicates(synd)
    checks.append(("syndicated copies collapse", len(kept5) == 2))
    checks.append(("first in rank order wins", kept5[0]["url"] == "https://a.com/x"))
    checks.append(("duplicate removal is reported", info5["removed"] == 2))
    checks.append(("different topics kept", any("Go" in i["title"] for i in kept5)))
    kept6, _ = drop_near_duplicates([{"url": "u", "title": ""}, {"url": "v", "title": ""}])
    checks.append(("untitled items are all kept", len(kept6) == 2))
    checks.append(("empty input is safe", drop_near_duplicates([])[0] == []))

    for name, ok in checks:
        print(f"{'OK' if ok else 'FAIL'}  {name}")
    print(f"diversity selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


if __name__ == "__main__":
    _selfcheck()
