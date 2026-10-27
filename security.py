"""URL validation with SSRF protection — private IP blocking, DNS rebinding detection, scheme validation."""

import asyncio
import httpcore
import httpx
import ipaddress
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

_MAX_URL_LENGTH = 8192

_BLOCKED_SCHEMES = frozenset({
    "file", "ftp", "gopher", "data", "javascript", "vbscript",
    "about", "chrome", "chrome-extension", "jar",
})

_BLOCKED_HOSTNAMES = frozenset({
    "localhost", "metadata.google.internal", "169.254.169.254",
    "0.0.0.0", "0",
})

_DNS_REBINDING_SUFFIXES = (
    ".nip.io", ".sslip.io", ".xip.io", ".nip.name", ".1u.ms",
)

_PRIVATE_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::/128"),
]


class SecurityError(ValueError):
    """Raised when input fails security validation."""
    pass


# ── DNS-rebinding TOCTOU pinning ─────────────────────────────────
# validate_url resolves and checks a hostname, but the HTTP client
# re-resolves at connect time — a short-TTL DNS record can flip a public
# answer to a private one inside that window (classic rebinding TOCTOU).
# validate_url stores its validated IPs; PinningNetworkBackend is wired
# into the shared httpx clients so connect_tcp uses exactly those IPs,
# keeping the hostname for TLS SNI/cert validation. Unvalidated hosts
# pass through untouched; pins expire so behavior degrades to plain
# resolution rather than failing.

_PIN_TTL = 60.0  # s — covers the validate→connect window
_pinned: dict[str, tuple[float, list[str]]] = {}

# Per-attempt connect budget for a pinned IP. This hotspot's IPv6 resolves but
# never answers, so a pinned IPv6 used to burn the whole client timeout
# (25-30 s) with no error and no fallback to a working IPv4.
_PIN_ATTEMPT_TIMEOUT = 5.0


def _pin_host(hostname: str, ips: list[str]) -> None:
    _pinned[hostname] = (time.monotonic(), list(ips))


def _ordered_pins(ips: list[str]) -> list[str]:
    """IPv4 first. The pin set contains only VALIDATED public IPs, so
    reordering does not weaken the DNS-rebinding protection — we still never
    re-resolve the hostname."""
    v4 = [ip for ip in ips if ":" not in ip]
    v6 = [ip for ip in ips if ":" in ip]
    return v4 + v6


class PinningNetworkBackend(httpcore.AsyncNetworkBackend):
    """httpcore network backend that connects to validated IPs."""

    def __init__(self) -> None:
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None,
                          local_address=None, socket_options=None):
        entry = _pinned.get(host)
        if entry:
            stamp, ips = entry
            if time.monotonic() - stamp <= _PIN_TTL and ips:
                # Try the pinned IPs in order (IPv4 first) with a short
                # per-attempt timeout, falling through to the next pinned IP
                # on failure. ONLY pinned/validated IPs are used — the
                # hostname is never re-resolved, so the DNS-rebinding
                # TOCTOU hole stays closed.
                total = 30.0
                if timeout:
                    try:
                        total = float(timeout)
                    except (TypeError, ValueError):
                        total = 30.0
                deadline = time.monotonic() + total
                last_exc: Exception | None = None
                for ip in _ordered_pins(ips):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    attempt = min(_PIN_ATTEMPT_TIMEOUT, remaining)
                    try:
                        return await self._inner.connect_tcp(
                            ip, port, timeout=attempt,
                            local_address=local_address,
                            socket_options=socket_options)
                    except Exception as e:
                        last_exc = e
                        continue
                if last_exc is not None:
                    raise last_exc
        return await self._inner.connect_tcp(
            host, port, timeout=timeout,
            local_address=local_address,
            socket_options=socket_options)


def _normalize_ip_notation(host: str) -> str | None:
    """Resolve alternate IP notations that libcurl/httpx would resolve.
    Octal, hex, decimal integer, short-form dotted — normalizes to dotted-decimal."""
    cleaned = host.strip("[]")
    if cleaned.isdigit():
        try:
            if len(cleaned) > 1 and cleaned[0] == '0' and all(c in '01234567' for c in cleaned):
                val = int(cleaned, 8)
            else:
                val = int(cleaned)
            if val <= 0xFFFFFFFF:
                return f"{(val >> 24) & 0xFF}.{(val >> 16) & 0xFF}.{(val >> 8) & 0xFF}.{val & 0xFF}"
        except (ValueError, OverflowError):
            pass
        return None
    parts = cleaned.split('.')
    if not (1 <= len(parts) <= 4):
        return None
    try:
        resolved = []
        for p in parts:
            if not p:
                return None
            if len(p) > 1 and p[0] == '0' and all(c in '01234567' for c in p):
                resolved.append(int(p, 8))
            elif p.lower().startswith('0x'):
                resolved.append(int(p, 0))
            else:
                resolved.append(int(p))
        if len(resolved) == 1:
            # Single-int hex/octal (0x7f000001): expand to 4 octets so
            # alternate-notation loopback doesn't slip past as fail-open.
            val = resolved[0]
            if val < 0 or val > 0xFFFFFFFF:
                return None
            resolved = [(val >> s) & 0xFF for s in (24, 16, 8, 0)]
        if len(resolved) == 2:
            resolved = [resolved[0], 0, 0, resolved[1]]
        elif len(resolved) == 3:
            resolved = [resolved[0], resolved[1], 0, resolved[2]]
        if any(o < 0 or o > 255 for o in resolved):
            return None
        return '.'.join(str(o) for o in resolved)
    except (ValueError, OverflowError):
        return None


async def resolve_hostname(hostname: str, timeout: float = 5.0) -> list[str]:
    """Resolve hostname to IP addresses via DNS."""
    try:
        infos = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(hostname, None),
            timeout=timeout
        )
        return sorted({info[4][0] for info in infos})
    except Exception:
        return []


def _is_private_address(ip_str: str) -> bool:
    """Check if an IP string is in private/reserved ranges, including
    IPv4-mapped IPv6."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        normalized = _normalize_ip_notation(ip_str)
        if normalized is None:
            return False
        try:
            addr = ipaddress.ip_address(normalized)
        except ValueError:
            return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        for net in _PRIVATE_NETWORKS:
            if isinstance(net, ipaddress.IPv4Network) and addr.ipv4_mapped in net:
                return True
    for net in _PRIVATE_NETWORKS:
        try:
            if addr in net:
                return True
        except TypeError:
            pass
    return False


def _validate_syntax(url: str, allow_internal: bool = False) -> str:
    """Syntactic URL validation — scheme, length, backslash, brackets, hostname.
    Does NOT do DNS resolution (that's async). Returns cleaned URL."""
    url = url.strip()
    if not url:
        raise SecurityError("URL must be a non-empty string")
    if len(url) > _MAX_URL_LENGTH:
        raise SecurityError(f"URL exceeds maximum length of {_MAX_URL_LENGTH} characters")
    if "\\" in url:
        raise SecurityError("URL contains backslash character (potential SSRF bypass)")
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SecurityError(f"Malformed URL: {e}")
    scheme = (parsed.scheme or "").lower()
    if scheme in _BLOCKED_SCHEMES:
        raise SecurityError(f"URL scheme '{scheme}' is not allowed")
    if scheme not in ("http", "https"):
        raise SecurityError(f"Only http and https URLs are supported, got: {scheme}")
    netloc_lower = (parsed.netloc or "").lower()
    if "[" in netloc_lower or "]" in netloc_lower:
        if not (netloc_lower.startswith("[") and "]" in netloc_lower):
            raise SecurityError("URL contains malformed bracketed host (potential SSRF bypass)")
        bracket_content = netloc_lower[1:netloc_lower.index("]")]
        try:
            ipaddress.IPv6Address(bracket_content)
        except ipaddress.AddressValueError:
            if not re.match(r'^v[0-9]+\..+$', bracket_content, re.IGNORECASE):
                raise SecurityError("URL contains invalid bracketed host (potential SSRF bypass)")
    hostname = parsed.hostname
    if not hostname:
        raise SecurityError("URL has no valid hostname")
    return url


async def validate_url(url: str, allow_internal: bool = False) -> str:
    """Validate a URL: scheme, syntax, private IP (sync+async DNS), DNS rebinding.
    Returns the validated URL string or raises SecurityError."""
    url = _validate_syntax(url, allow_internal)
    parsed = urlparse(url)
    hostname = parsed.hostname.lower().rstrip(".")  # strip FQDN trailing dot

    if allow_internal:
        return url

    # Check if hostname is a bare IP
    is_ip = False
    try:
        ipaddress.ip_address(hostname)
        is_ip = True
    except ValueError:
        normalized = _normalize_ip_notation(hostname)
        if normalized is not None:
            try:
                ipaddress.ip_address(normalized)
                is_ip = True
            except ValueError:
                pass

    if is_ip:
        if _is_private_address(hostname):
            raise SecurityError(f"URL targets internal/private IP: {hostname}")
        return url

    # Hostname (not bare IP) — check blocked hostnames and DNS rebinding
    if hostname in _BLOCKED_HOSTNAMES:
        raise SecurityError(f"URL targets internal service: {hostname}")

    # Check after normalization too (e.g., "LOCALHOST" with different case)
    lowered_try = hostname.replace("_", "-")  # some DNS rebinding obfuscation
    if lowered_try in _BLOCKED_HOSTNAMES:
        raise SecurityError(f"URL targets internal service: {hostname}")

    if hostname.endswith(_DNS_REBINDING_SUFFIXES):
        raise SecurityError(f"URL uses DNS rebinding service: {hostname}")

    # Resolve DNS — check resolved IPs against private ranges
    ips = await resolve_hostname(hostname)
    if ips:
        for ip_str in ips:
            if _is_private_address(ip_str):
                raise SecurityError(
                    f"URL resolves to internal IP ({hostname} -> {ip_str})"
                )
    # All resolved IPs public — pin them so the HTTP layer connects to
    # exactly these addresses instead of re-resolving (DNS-rebinding TOCTOU)
    if ips:
        _pin_host(hostname, ips)

    return url


async def safe_fetch_capped(client: Any, url: str, *, max_bytes: int = 2_000_000,
                            max_hops: int = 5, allow_internal: bool = False,
                            meta_out: dict[str, Any] | None = None,
                            **kwargs) -> tuple[bytes, bool]:
    """safe_fetch variant: streams the body with a hard byte cap, keeping
    per-hop SSRF validation. Returns (body_bytes, truncated).

    The body is never buffered past max_bytes: the stream is aborted as soon
    as the cap is reached and the connection is closed on exit.

    Unlike safe_stream (which raises on non-200) the body is returned
    whatever the status was. Callers that branch on the status code (or need
    the final URL) pass ``meta_out`` — a dict that receives ``status``,
    ``url`` and ``headers`` of the final hop.
    """
    current = url
    for _ in range(max_hops + 1):
        current = await validate_url(current, allow_internal=allow_internal)
        async with client.stream("GET", current, follow_redirects=False, **kwargs) as resp:
            if resp.is_redirect and resp.headers.get("location"):
                current = urljoin(str(resp.url), resp.headers["location"])
                continue
            if meta_out is not None:
                meta_out["status"] = resp.status_code
                meta_out["url"] = str(resp.url)
                meta_out["headers"] = dict(resp.headers)
            body = b""
            truncated = False
            async for chunk in resp.aiter_bytes():
                room = max_bytes - len(body)
                if room <= 0:
                    truncated = True
                    break
                if len(chunk) > room:
                    body += chunk[:room]
                    truncated = True
                    break
                body += chunk
            return body, truncated
    raise SecurityError(f"Too many redirects (> {max_hops}): {url}")


async def safe_fetch(client: Any, url: str, *, max_hops: int = 5,
                    allow_internal: bool = False, **kwargs) -> Any:
    """GET with per-hop SSRF validation — follows redirects manually, validating
    each hop's URL before requesting it. Returns the final response."""
    current = url
    for _ in range(max_hops + 1):
        current = await validate_url(current, allow_internal=allow_internal)
        resp = await client.get(current, follow_redirects=False, **kwargs)
        if resp.is_redirect and resp.headers.get("location"):
            current = urljoin(str(resp.url), resp.headers["location"])
            continue
        return resp
    raise SecurityError(f"Too many redirects (> {max_hops}): {url}")


async def safe_stream(client: Any, url: str, *, max_bytes: int,
                     max_hops: int = 5, allow_internal: bool = False,
                     **kwargs) -> bytes:
    """safe_fetch variant that streams and caps the body at max_bytes, keeping
    per-hop SSRF validation. Returns the body bytes (truncated at max_bytes).
    Raises OSError-adjacent httpx.HTTPStatusError on non-200 so callers can
    distinguish an HTTP error from an empty body."""
    current = url
    for _ in range(max_hops + 1):
        current = await validate_url(current, allow_internal=allow_internal)
        async with client.stream("GET", current, follow_redirects=False, **kwargs) as resp:
            if resp.is_redirect and resp.headers.get("location"):
                current = urljoin(str(resp.url), resp.headers["location"])
                continue
            if resp.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp)
            body = b""
            async for chunk in resp.aiter_bytes():
                # Slice the overshooting chunk instead of keeping it whole: a
                # stream that yields 8 MB chunks previously let a "50 MB" cap
                # hold ~58 MB, and under concurrent searches that is real RAM.
                remaining = max_bytes - len(body)
                if remaining <= 0:
                    break
                body += chunk[:remaining]
                if len(body) >= max_bytes:
                    break
            return body
    raise SecurityError(f"Too many redirects (> {max_hops}): {url}")

