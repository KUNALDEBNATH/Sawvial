import pytest

from sawvial.research.safe_http import FetchError, UnsafeURL, fetch, validate_url
from tests.netfixture import web  # noqa: F401

PDF = b"%PDF-1.4\n" + b"x" * 200


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x", "http://localhost/x", "http://169.254.169.254/latest/meta-data", "http://10.0.0.5/",
    "http://192.168.1.1/", "http://[::1]/", "http://0.0.0.0/", "http://[::ffff:127.0.0.1]/", "http://100.64.0.1/",
    "file:///etc/passwd", "ftp://example.com/a", "gopher://x/", "http://user:pw@example.com/", "https://example.com:8443/",
    "http://printer.local/", "http:///nohost"])
def test_unsafe_urls_are_refused(url):
    with pytest.raises(UnsafeURL):
        validate_url(url)


def test_allowed_domain_list(web):
    with pytest.raises(UnsafeURL):
        validate_url("http://public.test/x", allowed_domains=["icar.org.in"])
    assert validate_url("http://public.test/x", allowed_domains=["public.test"])


def test_successful_pdf_download(web):
    web["/a.pdf"] = (200, {"Content-Type": "application/pdf"}, PDF)
    r = fetch("http://public.test/a.pdf", allowed_types={"application/pdf"})
    assert r.content == PDF and r.content_type == "application/pdf" and len(r.sha256) == 64


def test_redirect_to_internal_address_is_blocked(web):
    web["/r"] = (302, {"Location": "http://meta.test/latest/meta-data"}, b"")
    with pytest.raises(UnsafeURL):
        fetch("http://public.test/r", allowed_types={"application/pdf"})


def test_redirect_to_private_ip_literal_blocked(web):
    web["/r2"] = (301, {"Location": "http://127.0.0.1:9/x"}, b"")
    with pytest.raises(UnsafeURL):
        fetch("http://public.test/r2", allowed_types={"application/pdf"})


def test_redirect_loop_limited(web):
    web["/loop"] = (302, {"Location": "http://public.test/loop"}, b"")
    with pytest.raises(FetchError, match="redirects"):
        fetch("http://public.test/loop", allowed_types={"application/pdf"}, max_redirects=3)


def test_valid_redirect_followed(web):
    web["/start"] = (302, {"Location": "/a.pdf"}, b"")
    web["/a.pdf"] = (200, {"Content-Type": "application/pdf"}, PDF)
    r = fetch("http://public.test/start", allowed_types={"application/pdf"})
    assert r.final_url.endswith("/a.pdf") and r.redirects


def test_size_limit_from_header_and_streaming(web):
    web["/big"] = (200, {"Content-Type": "application/pdf"}, b"%PDF-" + b"0" * 3000)
    with pytest.raises(FetchError, match="too large"):
        fetch("http://public.test/big", allowed_types={"application/pdf"}, max_bytes=1000)


def test_content_type_allowlist_and_pdf_magic(web):
    web["/z"] = (200, {"Content-Type": "application/zip"}, b"PK")
    web["/fake"] = (200, {"Content-Type": "application/pdf"}, b"<html>not a pdf</html>")
    with pytest.raises(FetchError, match="not allowed"):
        fetch("http://public.test/z", allowed_types={"application/pdf"})
    with pytest.raises(FetchError, match="but is not"):
        fetch("http://public.test/fake", allowed_types={"application/pdf"})


def test_http_error_status(web):
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch("http://public.test/missing", allowed_types={"application/pdf"})
