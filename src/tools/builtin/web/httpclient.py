from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Mapping
from dataclasses import dataclass

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0 "
    "EvanaAgent/1.0"
)

# Some ordinary sites return different content or a softer block depending on
# request headers. This is only a fetch-layer fallback; no tool contract changes.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/131.0 Safari/537.36"
)

DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5


def _safe_text(value: object) -> str:
    """Prevent invalid Unicode surrogate characters in transport strings."""
    return str(value or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")


_ALLOWED_REQUEST_HEADERS = {
    "user-agent": "User-Agent",
    "accept": "Accept",
    "accept-language": "Accept-Language",
    "referer": "Referer",
    "cache-control": "Cache-Control",
    "pragma": "Pragma",
}


def _normalize_request_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Validate optional, non-sensitive request headers.

    Credentials, cookies, host/routing headers and hop-by-hop headers are
    intentionally unsupported. Headers are user/model input, so reject
    unknown names and CR/LF rather than forwarding arbitrary HTTP metadata.
    """
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise FetchError("invalid_headers", "headers must be an object of string values.")
    if len(headers) > len(_ALLOWED_REQUEST_HEADERS):
        raise FetchError(
            "invalid_headers",
            "Too many headers. Only User-Agent, Accept, Accept-Language, Referer, Cache-Control and Pragma are supported.",
        )

    result: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        if not isinstance(raw_name, str):
            raise FetchError("invalid_headers", "Header names must be strings.")
        canonical = _ALLOWED_REQUEST_HEADERS.get(raw_name.strip().lower())
        if canonical is None:
            raise FetchError(
                "invalid_headers",
                f"Header '{raw_name}' is not allowed. Credentials, cookies and routing headers are not supported.",
            )
        if not isinstance(raw_value, str):
            raise FetchError(
                "invalid_headers",
                f"The value for '{canonical}' must be a string.",
            )
        value = _safe_text(raw_value).strip()
        if "\r" in value or "\n" in value:
            raise FetchError("invalid_headers", f"Invalid newline in '{canonical}'.")
        if len(value) > 512:
            raise FetchError("invalid_headers", f"The value for '{canonical}' is too long (max 512 characters).")
        result[canonical] = value

    return result


class FetchError(Exception):
    """Structured fetch failure. `error_type` is shown to the model."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = _safe_text(message)


@dataclass
class HttpResponse:
    url: str  # final URL after redirects
    status: int
    content_type: str  # mime type only, lower-case
    charset: str | None
    body: bytes
    truncated: bool  # True when the body was cut at max_bytes


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def default_timeout() -> float:
    try:
        value = float(os.environ.get("EVANA_WEB_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        value = DEFAULT_TIMEOUT

    return min(max(value, 1.0), 120.0)


def _allow_private() -> bool:
    return os.environ.get("EVANA_WEB_ALLOW_PRIVATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


# ----------------------------------------------------------------------
# URL safety (SSRF guard)
# ----------------------------------------------------------------------



def _prepare_http_url(url: str) -> str:
    """Convert Unicode URLs to an HTTP request-safe ASCII URL.

    Hostnames are IDNA/Punycode encoded and non-ASCII path/query characters
    are UTF-8 percent-encoded. Existing percent escapes are preserved.
    Fragments are removed because they are never sent to an HTTP server.
    """
    url = _safe_text(url).strip()

    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise FetchError("invalid_url", f"Invalid URL: {exc}") from exc

    if parts.scheme not in ("http", "https"):
        raise FetchError(
            "unsupported_scheme",
            f"Only http and https URLs are supported (got '{parts.scheme or 'none'}').",
        )

    host = parts.hostname
    if not host:
        raise FetchError("invalid_url", "URL has no host.")

    # IPv6 literals must stay in brackets; DNS hostnames use IDNA.
    if ":" in host and not host.startswith("["):
        encoded_host = host
    else:
        try:
            encoded_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise FetchError(
                "invalid_url",
                f"Could not encode hostname '{host}': {exc}",
            ) from exc

    netloc = encoded_host
    if ":" in encoded_host and not encoded_host.startswith("["):
        netloc = f"[{encoded_host}]"

    if port is not None:
        netloc = f"{netloc}:{port}"

    # Keep valid percent-escapes intact while UTF-8 encoding Unicode chars.
    path = urllib.parse.quote(
        parts.path,
        safe="/%:@!$&'()*+,;=-._~%",
    )
    query = urllib.parse.quote(
        parts.query,
        safe="/%?:@!$&'()*+,;=-._~%=&",
    )

    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), netloc, path, query, "")
    )

def assert_public_url(url: str) -> None:
    """
    Raise FetchError unless `url` is an http(s) URL that only resolves to
    public internet addresses.

    The agent reads untrusted pages, so it must not be steerable into
    localhost services (Ollama, databases), the LAN, or cloud metadata
    endpoints. Set EVANA_WEB_ALLOW_PRIVATE=1 to disable this check.

    Known limit: DNS is resolved here and again by urllib when connecting
    (DNS-rebinding window). Acceptable for a local agent; use an egress
    proxy if you need a hard guarantee.
    """

    url = _safe_text(url).strip()

    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise FetchError("invalid_url", f"Invalid URL: {exc}") from exc

    if parts.scheme not in ("http", "https"):
        raise FetchError(
            "unsupported_scheme",
            f"Only http and https URLs are supported (got '{parts.scheme or 'none'}').",
        )

    host = parts.hostname

    if not host:
        raise FetchError("invalid_url", "URL has no host.")

    if _allow_private():
        return

    if port is None:
        port = 443 if parts.scheme == "https" else 80

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(
            "dns_error",
            f"Could not resolve host '{host}': {exc}",
        ) from exc

    for info in infos:
        address = info[4][0].split("%")[0]

        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue

        if ip.version == 6 and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped

        if not ip.is_global:
            raise FetchError(
                "blocked_address",
                f"Blocked: '{host}' resolves to a non-public address ({ip}). "
                "Only public internet URLs can be fetched.",
            )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates every redirect target with the same SSRF guard."""

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        normalized = _prepare_http_url(newurl)
        assert_public_url(normalized)
        return super().redirect_request(req, fp, code, msg, headers, normalized)


# ----------------------------------------------------------------------
# Body decoding
# ----------------------------------------------------------------------


def _decompress(
    raw: bytes,
    encoding: str,
    limit: int,
) -> tuple[bytes, bool]:
    encoding = (encoding or "").strip().lower()

    if encoding in ("", "identity"):
        return raw, False

    if encoding in ("gzip", "x-gzip"):
        options = [zlib.MAX_WBITS | 16]
    elif encoding == "deflate":
        options = [zlib.MAX_WBITS, -zlib.MAX_WBITS]
    else:
        raise FetchError(
            "unsupported_encoding",
            f"Unsupported Content-Encoding: {encoding}",
        )

    last_error: Exception | None = None

    for wbits in options:
        try:
            decompressor = zlib.decompressobj(wbits)
            data = decompressor.decompress(raw, limit)
            return data, bool(decompressor.unconsumed_tail)
        except zlib.error as exc:
            last_error = exc

    raise FetchError(
        "decode_error",
        f"Could not decompress response ({encoding}): {last_error}",
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def http_get(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    """GET `url` with SSRF protection and a hard cap on the body size."""

    # Reject invalid caller headers before performing DNS or network work.
    normalized_headers = _normalize_request_headers(headers)
    url = _safe_text(url).strip()
    url = _prepare_http_url(url)
    assert_public_url(url)

    opener = urllib.request.build_opener(_SafeRedirectHandler())

    base_headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/json;q=0.9,"
            "text/plain;q=0.8,*/*;q=0.7"
        ),
        "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.7,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
    }
    # Validate caller-provided headers before opening a connection. They can
    # override only the safe allowlist above; transport/security headers stay
    # under this client's control.
    base_headers.update(normalized_headers)

    try:
        request = urllib.request.Request(url, headers=base_headers)
        with opener.open(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200) or 200)
            final_url = response.geturl()
            response_headers = response.headers
            raw = response.read(max_bytes + 1)

    except FetchError:
        raise

    except urllib.error.HTTPError as exc:
        hints = {
            400: " The URL or request may be invalid; verify the public page URL.",
            401: " The page requires authentication; use an authorized public page instead.",
            403: " Access was denied. Do not retry with spoofed credentials or attempt to bypass the site's controls.",
            404: " The URL was not found; verify it or discover the correct URL through web_search.",
            429: " The site is rate limiting requests; stop and retry later rather than increasing request frequency.",
        }
        hint = hints.get(exc.code, "")
        message = f"HTTP {exc.code} {exc.reason} for {url}.{hint}"
        exc.close()
        raise FetchError("http_error", message) from exc

    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise FetchError(
                "timeout",
                f"Timed out after {timeout:.0f}s: {url}",
            ) from exc

        raise FetchError(
            "network_error",
            f"Could not reach {url}: {exc.reason}",
        ) from exc

    except (TimeoutError, socket.timeout) as exc:
        raise FetchError(
            "timeout",
            f"Timed out after {timeout:.0f}s: {url}",
        ) from exc

    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise FetchError(
            "network_error",
            f"Request failed for {url}: {exc}",
        ) from exc

    truncated = len(raw) > max_bytes

    if truncated:
        raw = raw[:max_bytes]

    body, decompress_truncated = _decompress(
        raw,
        response_headers.get("Content-Encoding", ""),
        max_bytes,
    )

    return HttpResponse(
        url=_safe_text(final_url),
        status=status,
        content_type=(response_headers.get_content_type() or "").lower(),
        charset=response_headers.get_content_charset(),
        body=body,
        truncated=truncated or decompress_truncated,
    )
