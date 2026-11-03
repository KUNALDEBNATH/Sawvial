"""
Safe HTTP fetching for online research. Everything downloaded is UNTRUSTED input.

Controls
--------
* only http/https, no credentials in the URL, only ports 80/443
* the host is resolved and EVERY address must be public: loopback, private (RFC1918), link-local
  (incl. cloud metadata 169.254.169.254), multicast, reserved and carrier-grade-NAT ranges are refused
* the check is repeated at connect time and the socket connects to the validated IP
  (defence against DNS rebinding between check and use)
* redirects are followed manually, at most N hops, and each hop is validated again
* connect/read timeout plus an overall deadline
* download size limit (checked from Content-Length and while streaming; no compression accepted)
* Content-Type allow-list, plus a magic-bytes check for PDFs
* environment proxy settings are ignored (a proxy would bypass the address checks)
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
import urllib3.util.connection as _u3

from sawvial import settings

USER_AGENT = "SAWVIAL-research/1.0 (agricultural evidence collector)"
_BLOCKED_NAMES = {"localhost", "metadata.google.internal", "metadata"}
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp")


class UnsafeURL(ValueError):
    """The URL (or a redirect/DNS answer) points somewhere we refuse to connect to."""


class FetchError(RuntimeError):
    """The download failed or violated a limit. The message is safe to show to the user.
    `status` is the HTTP status code when the server answered with an error."""
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _ip_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return addr.is_global and not addr.is_multicast


def _domain_allowed(host: str, allowed: Iterable[str]) -> bool:
    allowed = [d.lower().lstrip(".") for d in allowed]
    return not allowed or any(host == d or host.endswith("." + d) for d in allowed)


def validate_url(url: str, allow_private: bool = False,
                 allowed_domains: Optional[Iterable[str]] = None) -> str:
    """Raise UnsafeURL unless `url` is safe to request. Returns the normalised URL."""
    try:
        p = urlparse(url.strip())
        port = p.port
    except ValueError as exc:
        raise UnsafeURL(f"malformed URL: {exc}") from exc
    if p.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme {p.scheme!r} is not allowed (only http/https)")
    host = (p.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeURL("URL has no host")
    if p.username or p.password:
        raise UnsafeURL("credentials in URLs are not allowed")
    if allow_private:                       # tests only: local fixture servers use random ports
        return p.geturl()
    if port not in (None, 80, 443):
        raise UnsafeURL(f"port {port} is not allowed")
    if host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        raise UnsafeURL(f"host {host!r} is an internal name")
    if allowed_domains is not None and not _domain_allowed(host, allowed_domains):
        raise UnsafeURL(f"host {host!r} is not in the allowed domain list")
    try:
        infos = socket.getaddrinfo(host, port or (443 if p.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"cannot resolve host {host!r}: {exc}") from exc
    bad = sorted({i[4][0] for i in infos if not _ip_is_public(i[4][0])})
    if bad:
        raise UnsafeURL(f"host {host!r} resolves to a non-public address ({', '.join(bad)})")
    return p.geturl()


# ---- connect-time guard (thread-local, active only inside fetch()) ------------
_local = threading.local()
_original_create_connection = _u3.create_connection


def _guarded_create_connection(address, *args, **kwargs):
    if not getattr(_local, "guard", False) or getattr(_local, "allow_private", False):
        return _original_create_connection(address, *args, **kwargs)
    host, port = address
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ips = [i[4][0] for i in infos]
    if not ips or any(not _ip_is_public(ip) for ip in ips):
        raise UnsafeURL(f"refusing to connect to non-public address for {host!r}")
    return _original_create_connection((ips[0], port), *args, **kwargs)


_u3.create_connection = _guarded_create_connection


@dataclass
class FetchResult:
    url: str
    final_url: str
    content_type: str
    content: bytes
    sha256: str
    redirects: List[str] = field(default_factory=list)
    charset: Optional[str] = None          # from the Content-Type header, if declared


def fetch(url: str, *, allowed_types: Iterable[str], max_bytes: Optional[int] = None,
          timeout: Optional[int] = None, max_redirects: Optional[int] = None,
          allowed_domains: Optional[Iterable[str]] = None, allow_private: bool = False,
          accept: str = "*/*") -> FetchResult:
    """Download `url` under all the limits above. `allow_private=True` exists for tests only."""
    max_bytes = max_bytes or settings.FETCH_MAX_BYTES
    timeout = timeout or settings.FETCH_TIMEOUT_S
    max_redirects = settings.FETCH_MAX_REDIRECTS if max_redirects is None else max_redirects
    allowed_types = {t.lower() for t in allowed_types}
    deadline = time.monotonic() + timeout * 3
    sess = requests.Session()
    sess.trust_env = False
    headers = {"User-Agent": USER_AGENT, "Accept": accept, "Accept-Encoding": "identity"}
    current, hops = url, []
    resp = None
    try:
        for _ in range(max_redirects + 1):
            validate_url(current, allow_private=allow_private, allowed_domains=allowed_domains)
            _local.guard, _local.allow_private = True, allow_private
            try:
                resp = sess.get(current, headers=headers, stream=True, timeout=(timeout, timeout),
                                allow_redirects=False)
            except requests.exceptions.RequestException as exc:
                if "UnsafeURL" in repr(exc) or "non-public" in str(exc):
                    raise UnsafeURL(str(exc)) from exc
                raise FetchError(f"request failed: {exc.__class__.__name__}") from exc
            finally:
                _local.guard = False
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                resp.close()
                if not loc:
                    raise FetchError("redirect without Location header")
                current = urljoin(current, loc)
                hops.append(current)
                continue
            break
        else:
            raise FetchError(f"more than {max_redirects} redirects")
        if resp.status_code != 200:
            raise FetchError(f"HTTP {resp.status_code} from {urlparse(current).hostname}", status=resp.status_code)
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype not in allowed_types:
            raise FetchError(f"content type {ctype or 'unknown'!r} is not allowed")
        if resp.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
            raise FetchError("compressed responses are not accepted")
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise FetchError(f"download too large ({int(declared)} bytes > {max_bytes})")
        buf, total = bytearray(), 0
        for chunk in resp.iter_content(65536):
            total += len(chunk)
            if total > max_bytes:
                raise FetchError(f"download exceeded the size limit of {max_bytes} bytes")
            if time.monotonic() > deadline:
                raise FetchError("download exceeded the time limit")
            buf.extend(chunk)
        data = bytes(buf)
        if ctype == "application/pdf" and not data.lstrip()[:5] == b"%PDF-":
            raise FetchError("response claims to be a PDF but is not")
        m = re.search(r"charset=([\w-]+)", resp.headers.get("Content-Type", ""), re.I)
        return FetchResult(url=url, final_url=current, content_type=ctype, content=data,
                           sha256=hashlib.sha256(data).hexdigest(), redirects=hops,
                           charset=m.group(1) if m else None)
    finally:
        if resp is not None:
            resp.close()
        sess.close()
