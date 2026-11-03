"""Local HTTP server + DNS/connection patching so the REAL safe_http guard code runs without internet.
'public.test' resolves to a public-looking IP, 'meta.test' to the cloud-metadata IP; connections to the
public-looking IP are redirected to the local fixture server."""
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sawvial.research import safe_http

ROUTES = {}      # path -> (status, headers, body)
SEEN = {}        # path -> request headers (to check what was sent where)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        SEEN[self.path] = dict(self.headers)
        status, headers, body = ROUTES.get(self.path, (404, {}, b"nf"))
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        if "Content-Length" not in headers:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass


@pytest.fixture()
def web(monkeypatch):
    ROUTES.clear()
    SEEN.clear()
    srv = HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    real_gai, real_cc = socket.getaddrinfo, safe_http._original_create_connection
    PUB = "93.184.216.34"
    fake = {h: PUB for h in ("public.test", "agritech.tnau.ac.in", "www.icar.gov.in", "epubs.icar.org.in", "vikaspedia.in",
                             "doi.org", "shady.test", "blog.example.com")}
    fake.update({"meta.test": "169.254.169.254", "internal.test": "10.1.2.3"})

    def gai(host, port, *a, **k):
        if host in fake:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (fake[host], port or 80))]
        return real_gai(host, port, *a, **k)

    def cc(address, *a, **k):
        return real_cc(("127.0.0.1", port), *a, **k) if address[0] == "93.184.216.34" else real_cc(address, *a, **k)

    from sawvial import settings
    monkeypatch.setattr(settings, "RESEARCH_HOST_DELAY_S", 0.0)       # no politeness sleeps in tests
    monkeypatch.setattr(socket, "getaddrinfo", gai)
    monkeypatch.setattr(safe_http, "_original_create_connection", cc)
    yield ROUTES
    srv.shutdown()
