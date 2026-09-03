"""Port of microlinkhq/is-antibot — detect anti-bot challenge pages from
static HTTP response data (status code, headers, cookies, HTML, URL).

Returns vendor name when a known challenge page is detected, so the caller
can decide to escalate (e.g. CDP fallback) instead of returning fake content.

Sources (MIT license):
  https://github.com/microlinkhq/is-antibot
  src/providers.json  (31 providers, pattern definitions)
  src/index.js        (compilation + matching logic)
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Provider definitions  —  verbatim from is-antibot src/providers.json
# ---------------------------------------------------------------------------
# Each entry: { name, detections: [{ type, rules, domain?, statusCodes? }] }
# Rule types: headers (equals/startsWith/exists/oneOf/headerNamePattern),
#             cookies (cookie prefix), html (contains/regex),
#             url (contains/regex), status_code (status)

_PROVIDERS = [
    {
        "name": "cloudflare",
        "detections": [
            {"type": "headers", "rules": [{"header": "cf-mitigated", "equals": "challenge"}]},
            {"type": "cookies", "rules": [{"cookie": "cf_clearance="}]},
        ],
    },
    {
        "name": "vercel",
        "detections": [
            {"type": "headers", "rules": [{"header": "x-vercel-mitigated", "equals": "challenge"}]},
        ],
    },
    {
        "name": "akamai",
        "detections": [
            {
                "type": "headers",
                "rules": [
                    {"header": "akamai-cache-status", "startsWith": "Error"},
                    {"header": "x-akamai-session-info", "exists": True},
                ],
            },
            {"type": "cookies", "rules": [{"cookie": "_abck="}]},
            {"type": "html", "rules": [{"contains": "bmak."}]},
        ],
    },
    {
        "name": "datadome",
        "detections": [
            {
                "type": "headers",
                "rules": [
                    {"header": "x-dd-b", "oneOf": ["1", "2"]},
                    {"header": "x-datadome", "exists": True, "except": "protected"},
                    {"header": "x-datadome-cid", "exists": True},
                ],
            },
            {"type": "cookies", "rules": [{"cookie": "datadome="}]},
        ],
    },
    {
        "name": "perimeterx",
        "detections": [
            {"type": "headers", "rules": [{"header": "x-px-authorization", "exists": True}]},
            {
                "type": "html",
                "rules": [
                    {"contains": "window._pxAppId"},
                    {"contains": "pxInit"},
                    {"contains": "_pxAction"},
                ],
            },
            {"type": "cookies", "rules": [{"cookie": "_px3="}, {"cookie": "_pxhd="}]},
        ],
    },
    {
        "name": "shapesecurity",
        "detections": [
            {
                "type": "headers",
                "rules": [{"headerNamePattern": "^x-[a-z0-9]{8}-[abcdfz]$", "flags": "i"}],
            },
            {"type": "html", "rules": [{"regex": r"shapesecurity\.\w+\s*\(", "flags": "i"}]},
        ],
    },
    {
        "name": "kasada",
        "detections": [
            {
                "type": "headers",
                "rules": [
                    {"header": "x-kasada", "exists": True},
                    {"header": "x-kasada-challenge", "exists": True},
                ],
            },
            {"type": "html", "rules": [{"contains": "__kasada"}, {"contains": "kasada.js"}]},
        ],
    },
    {
        "name": "imperva",
        "detections": [
            {
                "type": "headers",
                "rules": [
                    {"header": "x-cdn", "equals": "Incapsula"},
                    {"header": "x-iinfo", "exists": True},
                ],
            },
            {
                "type": "html",
                "rules": [{"regex": r"(?:incapsula|imperva)\.\w+\s*\(", "flags": "i"}],
            },
            {
                "type": "cookies",
                "rules": [
                    {"cookie": "incap_ses_"},
                    {"cookie": "visid_incap_"},
                    {"cookie": "reese84="},
                ],
            },
        ],
    },
    {
        "name": "reblaze",
        "detections": [
            {"type": "cookies", "rules": [{"cookie": "rbzid="}, {"cookie": "rbzsessionid="}]},
            {"type": "html", "rules": [{"contains": "Protected by Reblaze"}]},
        ],
    },
    {
        "name": "cheq",
        "detections": [
            {"type": "html", "rules": [{"contains": "CheqSdk"}, {"contains": "cheqzone.com"}]},
            {"type": "url", "rules": [{"regex": r"cheqzone\.com", "flags": "i"}, {"regex": r"cheq\.ai", "flags": "i"}]},
        ],
    },
    {
        "name": "sucuri",
        "detections": [
            {"type": "html", "rules": [{"contains": "Sucuri Website Firewall"}]},
        ],
    },
    {
        "name": "threatmetrix",
        "detections": [
            {"type": "html", "rules": [{"regex": r"ThreatMetrix\.\w+\s*\(", "flags": ""}]},
            {"type": "url", "rules": [{"contains": "fp/check.js"}]},
        ],
    },
    {
        "name": "meetrics",
        "detections": [
            {"type": "html", "rules": [{"regex": r"meetricsGlobal\.\w+\s*\("}]},
            {"type": "url", "rules": [{"regex": r"meetrics\.com", "flags": "i"}]},
        ],
    },
    {
        "name": "ocule",
        "detections": [
            {"type": "html", "rules": [{"contains": "ocule.co.uk"}]},
            {"type": "url", "rules": [{"regex": r"ocule\.co\.uk", "flags": "i"}]},
        ],
    },
    {
        "name": "cloudflare-turnstile",
        "detections": [
            {
                "type": "url",
                "rules": [{"regex": r"challenges\.cloudflare\.com/turnstile", "flags": "i"}],
            },
            {
                "type": "html",
                "rules": [
                    {"contains": "cf-turnstile"},
                    {"contains": "challenges.cloudflare.com/turnstile"},
                ],
            },
        ],
    },
    {
        "name": "anubis",
        "detections": [
            {
                "type": "html",
                "rules": [
                    {"regex": r'<script id="anubis_challenge"', "flags": ""},
                    {"contains": "/.within.website/x/cmd/anubis/"},
                ],
            },
        ],
    },
    {
        "name": "recaptcha",
        "detections": [
            {
                "type": "url",
                "rules": [
                    {"contains": "recaptcha/api"},
                    {"contains": "gstatic.com/recaptcha"},
                    {"contains": "recaptcha.net"},
                    {"regex": r"google\.com/recaptcha", "flags": "i"},
                ],
            },
            {
                "type": "html",
                "rules": [
                    {"regex": r"\b(?:window\.)?grecaptcha\s*\.(?:execute|render|ready|getResponse|enterprise)\b", "flags": "i"},
                    {"regex": r"\b(?:window\.)?grecaptcha\s*\(", "flags": "i"},
                    {"regex": r"\b__grecaptcha_cfg\b", "flags": "i"},
                ],
            },
            {
                "type": "html",
                "statusCodes": [403, 429, 503],
                "rules": [{"contains": "g-recaptcha"}],
            },
        ],
    },
    {
        "name": "hcaptcha",
        "detections": [
            {"type": "url", "rules": [{"regex": r"hcaptcha\.com", "flags": "i"}]},
            {"type": "html", "rules": [{"contains": "hcaptcha.com"}, {"contains": "h-captcha"}]},
        ],
    },
    {
        "name": "funcaptcha",
        "detections": [
            {"type": "url", "rules": [{"regex": r"arkoselabs\.com", "flags": "i"}, {"contains": "funcaptcha"}]},
            {"type": "html", "rules": [{"contains": "arkoselabs.com"}, {"regex": r"funcaptcha\.\w+\s*\("}]},
        ],
    },
    {
        "name": "geetest",
        "detections": [
            {"type": "url", "rules": [{"regex": r"geetest\.com", "flags": "i"}]},
            {"type": "html", "rules": [{"regex": r"geetest\.\w+\s*\("}]},
        ],
    },
    {
        "name": "friendly-captcha",
        "detections": [
            {"type": "url", "rules": [{"regex": r"friendlycaptcha\.com", "flags": "i"}]},
            {"type": "html", "rules": [{"contains": "frc-captcha"}, {"contains": "friendlyChallenge"}]},
        ],
    },
    {
        "name": "captcha-eu",
        "detections": [
            {"type": "url", "rules": [{"regex": r"captcha\.eu", "flags": "i"}]},
            {"type": "html", "rules": [{"contains": "CaptchaEU"}, {"contains": "captchaeu"}]},
        ],
    },
    {
        "name": "qcloud-captcha",
        "detections": [
            {"type": "url", "rules": [{"regex": r"turing\.captcha\.qcloud\.com", "flags": "i"}]},
            {"type": "html", "rules": [{"contains": "TencentCaptcha"}, {"contains": "turing.captcha"}]},
        ],
    },
    {
        "name": "aliexpress-captcha",
        "detections": [
            {"type": "url", "rules": [{"regex": r"punish\?x5secdata", "flags": "i"}]},
            {"type": "html", "rules": [{"contains": "x5secdata"}]},
        ],
    },
    {
        "name": "houzz",
        "detections": [
            {"type": "status_code", "domain": "houzz.com", "rules": [{"status": 429}]},
        ],
    },
    {
        "name": "reddit",
        "detections": [
            {"type": "status_code", "domain": "reddit.com", "rules": [{"status": 403}, {"status": 429}]},
            {
                "type": "html",
                "domain": "reddit.com",
                "rules": [
                    {"regex": r"blocked by network security\.", "flags": "i"},
                    {"contains": "Please wait for verification"},
                ],
            },
        ],
    },
    {
        "name": "google",
        "detections": [
            {"type": "url", "rules": [{"contains": "consent.google.com"}]},
        ],
    },
    {
        "name": "linkedin",
        "detections": [
            {"type": "status_code", "domain": "linkedin.com", "rules": [{"status": 999}]},
        ],
    },
    {
        "name": "instagram",
        "detections": [
            {
                "type": "html",
                "domain": "instagram.com",
                "rules": [{"regex": r"<title>\s*Login\s*[•·]\s*Instagram\s*</title>", "flags": "i"}],
            },
        ],
    },
    {
        "name": "youtube",
        "detections": [
            {"type": "html", "rules": [{"regex": r"<title>\s*-\s*YouTube</title>", "flags": "i"}]},
        ],
    },
    {
        "name": "amazon",
        "detections": [
            {
                "type": "headers",
                "domainWithoutSuffix": "amazon",
                "statusCodes": [500],
                "rules": [{"header": "x-cache", "startsWith": "Error from cloudfront"}],
            },
            {
                "type": "html",
                "domainWithoutSuffix": "amazon",
                "rules": [{"contains": "csm-captcha-instrumentation"}],
            },
        ],
    },
    {
         "name": "fullstory-challenge",
         "detections": [
             {"type": "html", "rules": [{"contains": "_fs-ch-"}]},
             {"type": "cookies", "rules": [{"cookie": "_fs_ch_st_"}]},
         ],
     },
     {
         "name": "aws-waf",
         "detections": [
             {"type": "headers", "rules": [{"header": "x-amzn-waf-action", "exists": True}]},
             {"type": "html", "rules": [{"regex": r"aws-waf\.\w+\s*\("}, {"contains": "gokuProps"}]},
             {"type": "html", "statusCodes": [202, 403, 405, 429, 503], "rules": [{"regex": r"awswaf\.com|\/awswaf\/"}]},
             {"type": "cookies", "statusCodes": [202, 403, 405, 429, 503], "rules": [{"cookie": "aws-waf-token="}]},
         ],
     },
     {
         "name": "cloudfront",
         "detections": [
             {"type": "headers", "statusCodes": [403, 502, 503, 504], "rules": [{"header": "x-cache", "startsWith": "Error from cloudfront"}]},
             {"type": "html", "statusCodes": [403, 502, 503, 504], "rules": [{"contains": "The request could not be satisfied"}]},
         ],
     },
     {
         "name": "weibo",
         "detections": [
             {"type": "html", "domainWithoutSuffix": "weibo", "rules": [{"contains": "Sina Visitor System"}]},
         ],
     },
     {
         "name": "dribbble",
         "detections": [
             {"type": "status_code", "domain": "dribbble.com", "rules": [{"status": 403}]},
         ],
     },
     {
         "name": "douban",
         "detections": [
             {"type": "status_code", "domain": "doubanio.com", "rules": [{"status": 418}]},
         ],
     },
 ]

# ---------------------------------------------------------------------------
# Compiled rules
# ---------------------------------------------------------------------------

def _compile_text_rule(rule: dict) -> dict:
    """Compile a contains/regex rule into a check function + precomputed data."""
    if "contains" in rule:
        val = rule["contains"].lower()
        return {"type": "contains", "value": val, "check": lambda t, v=val: v in t}
    flags = rule.get("flags", "i")
    regex = re.compile(rule["regex"], re.IGNORECASE if "i" in flags else 0)
    return {"type": "regex", "check": lambda t, r=regex: bool(r.search(t))}


def _compile_detection(det: dict) -> dict:
    """Compile a detection entry into {'type', 'domain', 'domainWithoutSuffix', 'statusCodes', 'check'}."""
    det_type = det["type"]
    rules = det["rules"]
    compiled: dict[str, Any] = {
        "type": det_type,
        "domain": det.get("domain"),
        "domainWithoutSuffix": det.get("domainWithoutSuffix"),
        "statusCodes": det.get("statusCodes"),
    }

    if det_type == "cookies":
        prefixes = [r["cookie"] for r in rules]
        lower_prefixes = [p.lower() for p in prefixes]
        compiled["_prefixes"] = prefixes
        compiled["_lower_prefixes"] = lower_prefixes
    elif det_type in ("html", "url"):
        compiled["_patterns"] = [_compile_text_rule(r) for r in rules]
    elif det_type == "headers":
        compiled["_header_rules"] = rules
    elif det_type == "status_code":
        compiled["_codes"] = {r["status"] for r in rules}

    return compiled


def _compile_providers():
    for p in _PROVIDERS:
        p["_detections"] = [_compile_detection(d) for d in p["detections"]]


_compile_providers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_domain(url: str) -> str | None:
    """Extract the hostname (domain.tld) from a URL."""
    m = re.search(r"://([^/]+)", url)
    if not m:
        return None
    return m.group(1).split("@")[-1].split(":")[0].lower()


def _domain_without_suffix(domain: str | None) -> str | None:
    """Return domain without TLD suffix (e.g. 'www.amazon' from 'www.amazon.com')."""
    if not domain:
        return None
    parts = domain.rsplit(".", 2)
    # www.example.com -> www.example
    if len(parts) >= 2:
        return parts[0] + "." + parts[1] if len(parts) == 3 else parts[0]
    return domain


def _has_cookie(raw_set_cookie: str | list[str] | None, prefixes: list[str]) -> bool:
    """Check if any Set-Cookie header starts with one of the given prefixes."""
    if not raw_set_cookie:
        return False
    if isinstance(raw_set_cookie, str):
        # Quick candidate check via substring match
        if not any(p in raw_set_cookie for p in prefixes):
            return False
        cookies = raw_set_cookie.split(",")  # crude split, fine for prefix matching
    else:
        cookies = raw_set_cookie
    pref_lower = [p.lower() for p in prefixes]
    for cookie in cookies:
        c = cookie.lstrip().lower()
        for p in pref_lower:
            if c.startswith(p):
                return True
    return False


def _check_headers_rule(get_header, rule: dict, header_names: list[str]) -> bool:
    """Check a single header rule."""
    header = rule.get("header")
    if header:
        val = get_header(header)
        if "equals" in rule:
            return val is not None and val == rule["equals"]
        if "startsWith" in rule:
            return val is not None and val.startswith(rule["startsWith"])
        if "exists" in rule and "except" in rule:
            return val is not None and str(val).lower() != rule["except"].lower()
        if "exists" in rule:
            return val is not None
        if "oneOf" in rule:
            return val is not None and val in rule["oneOf"]
        return False
    if "headerNamePattern" in rule:
        flags = rule.get("flags", "i")
        regex = re.compile(rule["headerNamePattern"], re.IGNORECASE if "i" in flags else 0)
        for name in header_names:
            if regex.search(name):
                return True
        return False
    return False


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def detect_antibot(
    *,
    html: str = "",
    url: str = "",
    status_code: int | None = None,
    headers: dict | None = None,
    set_cookie: str | list[str] | None = None,
) -> tuple[bool, str | None, str | None]:
    """Check if an HTTP response is a bot challenge page.

    Args:
        html: Raw HTML response body.
        url: Request URL (for domain filtering + URL pattern checks).
        status_code: HTTP status code.
        headers: Response headers dict (keys are header names).
        set_cookie: 'Set-Cookie' header value, or a list of them.

    Returns:
        (detected, provider_name, detection_type)
        e.g. (True, "cloudflare", "headers") or (False, None, None)
    """
    domain = _parse_domain(url) if url else None
    domain_ws = _domain_without_suffix(domain) if domain else None

    html_lower = html.lower() if html else ""
    url_lower = url.lower() if url else ""

    # Normalize headers dict
    if headers:
        get_header = lambda n, h=headers: h.get(n.lower(), h.get(n, None))
        header_names = list(headers.keys())
    else:
        get_header = lambda n: None
        header_names = []

    for provider in _PROVIDERS:
        for det in provider["_detections"]:
            # Domain filter
            if det["domain"] and domain != det["domain"]:
                continue
            if det["domainWithoutSuffix"]:
                if domain_ws is None or det["domainWithoutSuffix"] not in domain_ws:
                    continue

            # Status code filter
            if det.get("statusCodes") and status_code not in det["statusCodes"]:
                continue

            matched = False

            if det["type"] == "cookies":
                if _has_cookie(set_cookie, det["_prefixes"]):
                    matched = True

            elif det["type"] == "html":
                for pat in det["_patterns"]:
                    if pat["check"](html_lower):
                        matched = True
                        break

            elif det["type"] == "url":
                for pat in det["_patterns"]:
                    if pat["check"](url_lower):
                        matched = True
                        break

            elif det["type"] == "headers":
                for rule in det["_header_rules"]:
                    if _check_headers_rule(get_header, rule, header_names):
                        matched = True
                        break

            elif det["type"] == "status_code":
                if det["_codes"] and status_code in det["_codes"]:
                    matched = True

            if matched:
                return True, provider["name"], det["type"]

    return False, None, None
