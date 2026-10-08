"""serve/test_security.py - the Host check (DNS rebinding) and the Origin check for requests without an API key,
against the mock engine (no GPU, no pack).

    python -m unittest serve.test_security -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import queue
import socket
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (API_KEY_SCOPES, TUNE_KEYS, ByteTokenizer, MockEngine, Service,  # noqa: E402
                          StrataEngine, allowed_hosts_of, api_key_allow_of, clean_tune, host_allowed,
                          host_name, host_names_for, key_needed_for, origin_allowed, parse_netblocks, serve,
                          tune_str)

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {"localhost", "127.0.0.1", "::1"}


class HostNames(unittest.TestCase):
    def test_host_name(self):
        self.assertEqual(host_name("Example.COM:8080"), "example.com")
        self.assertEqual(host_name("localhost"), "localhost")
        self.assertEqual(host_name("[::1]:8095"), "::1")
        self.assertEqual(host_name("[::1]"), "::1")
        self.assertEqual(host_name("strata.example.com."), "strata.example.com")
        for bad in ("", "a:b", "[::1", "[::1]x", "a b", "evil.com/x"):
            self.assertEqual(host_name(bad), "", bad)

    def test_allowed_hosts_of(self):
        self.assertEqual(allowed_hosts_of(None), [])
        self.assertEqual(allowed_hosts_of("Strata.Example.com"), ["strata.example.com"])
        self.assertEqual(allowed_hosts_of(["https://a.example.com:8443/x", ".example.org", "*"], "box, nas.lan "),
                         ["a.example.com", ".example.org", "*", "box", "nas.lan"])
        for bad in (["bad name"], 5, [3], "http://"):
            with self.assertRaises(ValueError):
                allowed_hosts_of(bad)

    def test_names_from_the_bind_address(self):
        with mock.patch("serve.server.lan_addresses", return_value=["192.168.1.20"]):
            self.assertEqual(host_names_for("127.0.0.1"), LOCAL)                 # this PC only: nothing else
            self.assertEqual(host_names_for("localhost"), LOCAL)
            names = host_names_for("0.0.0.0")                                    # other devices too
            self.assertIn(socket.gethostname().lower(), names)
            self.assertIn(socket.gethostname().lower() + ".local", names)
            self.assertIn("192.168.1.20", names)
            self.assertIn("10.0.0.7", host_names_for("10.0.0.7"))
            names = host_names_for("127.0.0.1", ["strata.example.com", "*"], ["https://chat.example.net"])
            self.assertTrue({"strata.example.com", "chat.example.net"} <= names)
            self.assertNotIn("*", names)


class HostCheck(unittest.TestCase):
    def test_accepted(self):
        for host in ("localhost:8095", "LOCALHOST", "127.0.0.1:1", "[::1]:8095", "192.168.1.20:8095", "[fe80::1]",
                     "app.localhost:8095", None, "", "box:8095", "a.example.org", "example.org:443"):
            self.assertTrue(host_allowed(host, LOCAL | {"box", ".example.org"}), host)

    def test_refused(self):
        # a rebinding page sends its own name; lookalikes of allowed names stay out
        for host in ("evil.example.com", "evil.example.com:8095", "localhost.evil.com", "127.0.0.1.nip.io",
                     "box.evil.com", "xexample.org", "bad host", "a:b:c:d:e:f:g:h:i"):
            self.assertFalse(host_allowed(host, LOCAL | {"box", ".example.org"}), host)

    def test_any_host(self):
        self.assertTrue(host_allowed("evil.example.com", LOCAL, any_host=True))


class OriginCheck(unittest.TestCase):
    def test_accepted(self):
        names = LOCAL | {"box"}
        self.assertTrue(origin_allowed("http://127.0.0.1:8095", "127.0.0.1:8095", names))      # Strata's own page
        self.assertTrue(origin_allowed("http://192.168.1.20:8095", "192.168.1.20:8095", names))  # ... by LAN IP
        self.assertTrue(origin_allowed("http://localhost:3000", "127.0.0.1:8095", names))      # a local app
        self.assertTrue(origin_allowed("http://[::1]:3000/", "127.0.0.1:8095", names))
        self.assertTrue(origin_allowed("https://box", "127.0.0.1:8095", names))
        self.assertTrue(origin_allowed("https://chat.example.com", "x", names, ["https://chat.example.com"]))
        self.assertTrue(origin_allowed("https://any.example.com", "x", names, ["*"]))           # cors_origins ["*"]
        # browser extensions and desktop apps: no web site can send another scheme than http(s) (or "null")
        for origin in ("chrome-extension://abcdef", "moz-extension://1234-5678", "app://."):
            self.assertTrue(origin_allowed(origin, "127.0.0.1:8095", names), origin)

    def test_refused(self):
        names = LOCAL | {"box"}
        for origin in ("http://evil.example.com", "http://1.2.3.4", "http://192.168.1.99:8095", "null", "",
                       "http://localhost.evil.com", "https://box.evil.com", "://x"):
            self.assertFalse(origin_allowed(origin, "127.0.0.1:8095", names, ["https://chat.example.com"]), origin)


class HttpHarness:
    """A running server against the mock engine, driven as a browser and as curl/the SDKs drive it."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = None

    def start(self, **attrs):
        for k, v in attrs.items():
            setattr(self.svc, k, v)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def req(self, method, path, body=None, headers=None, host=None):
        """(status, JSON, what the server printed); `host` replaces the Host header http.client sends."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = dict(headers or {})
        if host is not None:
            h["Host"] = host
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                c.request(method, path, body=data, headers=h)
                r = c.getresponse()
                raw = r.read()
            return r.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw, out.getvalue()
        finally:
            c.close()

    def chat_body(self):
        return {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}


class OverHttp(HttpHarness, unittest.TestCase):
    """The Host and Origin checks in the running server."""

    # --- Host
    def test_rebinding_name_is_refused_everywhere(self):
        self.start()
        for method, path in (("GET", "/status"), ("GET", "/metrics"), ("GET", "/"), ("GET", "/health"),
                             ("POST", "/settings"), ("POST", "/v1/chat/completions"), ("OPTIONS", "/v1/models")):
            code, body, log = self.req(method, path, {} if method == "POST" else None,
                                       {"Content-Type": "application/json"}, host=f"evil.example.com:{self.port}")
            self.assertEqual(code, 403, path)
            if method != "OPTIONS":
                self.assertIn("allowed_hosts", body["error"]["message"])
            self.assertEqual(log.count("\n"), 1, path)                          # one log line each
            self.assertIn("evil.example.com", log)

    def test_local_names_and_ips_pass(self):
        self.start()
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}",
                     f"192.168.1.20:{self.port}"):
            self.assertEqual(self.req("GET", "/status", host=host)[0], 200, host)

    def test_no_host_header_passes(self):
        self.start()
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as s:      # an HTTP/1.0 client
            s.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            raw = b""
            while chunk := s.recv(65536):
                raw += chunk
        self.assertTrue(raw.startswith(b"HTTP/1.0 200"), raw[:40])

    def test_allowed_hosts_and_wildcard(self):
        self.start(allowed_hosts=["strata.example.com", ".home.arpa"])
        self.assertEqual(self.req("GET", "/status", host="strata.example.com")[0], 200)
        self.assertEqual(self.req("GET", "/status", host="nas.home.arpa:8095")[0], 200)
        self.assertEqual(self.req("GET", "/status", host="evil.example.com")[0], 403)
        self.svc.allowed_hosts = ["*"]
        self.assertEqual(self.req("GET", "/status", host="evil.example.com")[0], 200)

    def test_refusal_names_both_ways_out(self):
        self.start()
        message = self.req("GET", "/status", host="evil.example.com")[1]["error"]["message"]
        self.assertIn("allowed_hosts", message)
        self.assertIn("api_key", message)

    def test_with_a_key_the_host_check_off_only_for_authenticated_requests(self):
        # A CHANGE from upstream 0.1.38, and the reason is the opt-in exemption: with a key set upstream skipped
        # the Host check for EVERYONE, but scope "lan"/"localhost" exempts 127.0.0.1 from the key - and a DNS
        # rebinding page arrives from 127.0.0.1 too.  So the check now runs for a request that does NOT present
        # the key, and is skipped for one that does: a tunnel or proxy that passes its own name on keeps working
        # as long as it sends the key (or its name is in allowed_hosts).  The exemption itself is opt-in now -
        # the default scope is upstream's "all", which exempts nobody - so it is asked for here.
        self.start(api_key="s3cret", api_key_scope="lan")
        tunnel = "random-words.trycloudflare.com"
        self.assertEqual(self.req("GET", "/status", host=tunnel)[0], 403)
        self.assertIn("ev.example.com", self.req("GET", "/status", host="ev.example.com")[2])
        self.assertEqual(self.req("GET", "/status", headers={"Authorization": "Bearer s3cret"}, host=tunnel)[0], 200)
        self.assertEqual(self.req("GET", "/health", headers={"Authorization": "Bearer s3cret"}, host=tunnel)[0], 200)
        code, _, log = self.req("POST", "/v1/chat/completions", self.chat_body(),
                                {"Content-Type": "application/json", "Authorization": "Bearer s3cret"}, host=tunnel)
        self.assertEqual(code, 200)
        self.assertNotIn("refused", log)
        # a wrong key is refused by /status (which needs it) and still leaves the Host check standing
        self.assertEqual(self.req("GET", "/status", headers={"Authorization": "Bearer wrong"}, host=tunnel)[0], 403)
        # the other way out, as before: name the tunnel in allowed_hosts (it feeds host_names at start)
        self.tearDown()
        self.start(api_key="s3cret", api_key_scope="lan", allowed_hosts=[tunnel])
        self.assertEqual(self.req("GET", "/status", host=tunnel)[0], 200)    # this PC is exempt in scope "lan"
        self.assertEqual(self.req("GET", "/status", host="other.example.com")[0], 403)

    def test_the_host_check_is_not_the_negation_of_the_key_check(self):
        """Pins the distinction a "DRY" pass broke once: replacing the Host/Origin predicate with `not _key_ok()`
        turns the rebinding check OFF on a KEYLESS server, which is exactly the server it exists for (the API check
        passes there because there is no key to present).  Three cases, in order: keyless, exempt-with-a-key, and
        authenticated-with-a-key."""
        self.start()                                                                    # no key configured at all
        self.assertEqual(self.req("GET", "/status", host="evil.example.com")[0], 403)
        self.tearDown()                                                                  # close that server first
        self.start(api_key="s3cret", api_key_scope="lan")                                # this PC is exempt
        self.assertEqual(self.req("GET", "/status", host="evil.example.com")[0], 403)
        self.assertEqual(self.req("GET", "/status", host="evil.example.com",
                                  headers={"Authorization": "Bearer s3cret"})[0], 200)   # upstream's tunnel rule

    def test_a_trusted_origin_s_host_is_allowed(self):
        self.start(trusted_origins=["https://strata.example.com"])
        self.assertEqual(self.req("GET", "/status", host="strata.example.com")[0], 200)

    # --- Origin on /v1 without an API key
    def test_curl_and_sdks_without_origin(self):
        self.start()
        for ctype in ("application/json", "text/plain", None):              # curl -d sends a form type
            code, body, _ = self.req("POST", "/v1/chat/completions", self.chat_body(),
                                     {"Content-Type": ctype} if ctype else {})
            self.assertEqual(code, 200, ctype)
            self.assertEqual(body["choices"][0]["message"]["content"], "ok")

    def test_cross_site_page_is_refused(self):
        self.start()
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"):
            for ctype in ("text/plain", "application/json"):
                code, body, log = self.req("POST", path, self.chat_body(),
                                           {"Content-Type": ctype, "Origin": "http://evil.example.com"})
                self.assertEqual(code, 403, (path, ctype))
                self.assertIn("api_key", body["error"]["message"])
                self.assertIn("evil.example.com", log)
        self.assertEqual(self.req("POST", "/v1/chat/completions", self.chat_body(),
                                  {"Content-Type": "text/plain", "Origin": "null"})[0], 403)

    def test_own_and_local_pages_pass(self):
        self.start()
        json_type = {"Content-Type": "application/json"}
        for origin in (f"http://127.0.0.1:{self.port}", "http://localhost:3000", "chrome-extension://abcdef"):
            self.assertEqual(self.req("POST", "/v1/chat/completions", self.chat_body(),
                                      {**json_type, "Origin": origin})[0], 200, origin)
        # a browser page sends JSON; text/plain from a page is the cross-site "simple request" shape
        self.assertEqual(self.req("POST", "/v1/chat/completions", self.chat_body(),
                                  {"Content-Type": "text/plain", "Origin": f"http://127.0.0.1:{self.port}"})[0], 415)

    def test_configured_origins_pass(self):
        self.start(cors_origins=["https://chat.example.com"], trusted_origins=["https://strata.example.com"],
                   allowed_hosts=["webui.lan"])
        for origin in ("https://chat.example.com", "https://strata.example.com", "http://webui.lan:3000"):
            self.assertEqual(self.req("POST", "/v1/chat/completions", self.chat_body(),
                                      {"Content-Type": "application/json", "Origin": origin})[0], 200, origin)

    def test_with_a_key_the_key_decides(self):
        # With a key set the ORIGIN check is skipped for a caller that presents it (upstream's rule).  An
        # exempt caller (this PC under scope "lan", opted into here - the default is upstream's "all") does not
        # need the key, so here the origin check still stands and refuses the foreign page: 403, not 401.
        self.start(api_key="s3cret", api_key_scope="lan")
        headers = {"Content-Type": "text/plain", "Origin": "http://evil.example.com"}
        code, body, _ = self.req("POST", "/v1/chat/completions", self.chat_body(), headers)
        self.assertEqual(code, 403)
        self.assertIn("api_key", body["error"]["message"])
        self.assertEqual(self.req("POST", "/v1/chat/completions", self.chat_body(),
                                  {**headers, "Authorization": "Bearer s3cret"})[0], 200)
        # a caller the key does not cover is answered by the key, whatever the page it came from: over a loopback
        # socket the peer is always this PC, so assert the predicate the server uses
        self.svc.api_key_scope = "localhost"
        self.assertTrue(self.svc.key_needed_for("192.168.1.5"))
        self.assertFalse(self.svc.key_needed_for("127.0.0.1"))

    # --- /unload and /load
    def test_control_body_arrives_before_the_operation_and_reply(self):
        self.start()
        for body, expected in ((b"{}", 200), (b"{", 400)):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            try:
                with mock.patch.object(self.svc, "unload", return_value="unloaded") as unload:
                    c.putrequest("POST", "/unload")
                    c.putheader("Content-Type", "application/json")
                    c.putheader("Content-Length", "2")
                    c.endheaders()
                    c.sock.settimeout(0.1)
                    with self.assertRaises(socket.timeout):
                        c.sock.recv(1)
                    unload.assert_not_called()
                    c.sock.settimeout(5)
                    c.send(body)
                    if len(body) < 2:
                        c.sock.shutdown(socket.SHUT_WR)
                    r = c.getresponse()
                    self.assertEqual(r.status, expected)
                    r.read()
                    self.assertEqual(unload.call_count, int(expected == 200))
            finally:
                c.close()

    def test_unload_and_load_need_json_from_the_own_page(self):
        self.start()
        form = {"Content-Type": "application/x-www-form-urlencoded"}
        for path in ("/unload", "/load"):
            self.assertEqual(self.req("POST", path, b"a=1", form)[0], 415, path)
            self.assertEqual(self.req("POST", path, b"", {})[0], 415, path)
            self.assertEqual(self.req("POST", path, {}, {"Content-Type": "application/json",
                                                         "Origin": "http://evil.example.com"})[0], 403, path)
        self.assertEqual(self.req("POST", "/unload", {}, {"Content-Type": "application/json"})[0], 200)
        self.assertEqual(self.req("POST", "/load", {}, {"Content-Type": "application/json",
                                                        "Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    # --- /slots/0?action=save|restore (session files)
    def test_slots_need_json_from_the_own_page(self):
        self.start()
        body = {"filename": "a.bin"}
        for action in ("save", "restore"):
            path = f"/slots/0?action={action}"
            self.assertEqual(self.req("POST", path, b"filename=a.bin",
                                      {"Content-Type": "application/x-www-form-urlencoded"})[0], 415, path)
            self.assertEqual(self.req("POST", path, body, {"Content-Type": "text/plain"})[0], 415, path)
            for origin in ("http://evil.example.com", "null"):
                self.assertEqual(self.req("POST", path, body, {"Content-Type": "application/json",
                                                               "Origin": origin})[0], 403, (path, origin))
            # JSON without an Origin (curl, scripts) or from the own page reaches the slot API: off here, so 501
            self.assertEqual(self.req("POST", path, body, {"Content-Type": "application/json"})[0], 501, path)
            self.assertEqual(self.req("POST", path, body, {"Content-Type": "application/json",
                                                           "Origin": f"http://127.0.0.1:{self.port}"})[0], 501, path)
        self.assertEqual(self.req("POST", "/slots/0?action=save", body, {"Content-Type": "application/json"},
                                  host="rebind.example.com")[0], 403)


class ApiKeyScope(unittest.TestCase):
    """Who has to present the key: the scope (all - the default - | lan | localhost | off) and api_key_allow."""

    def test_lan_scope_exempts_this_pc_and_the_local_network(self):
        for addr in ("127.0.0.1", "127.9.9.9", "::1", "10.0.0.7", "10.255.255.255", "172.16.0.1", "172.31.0.1",
                     "192.168.0.1", "192.168.40.33", "169.254.3.4", "fc00::1", "fd12:3456::1", "fe80::1%eth0"):
            self.assertFalse(key_needed_for(addr, "k", "lan"), addr)
        for addr in ("8.8.8.8", "1.1.1.1", "172.32.0.1", "172.15.0.1", "100.64.0.1", "192.169.0.1",
                     "203.0.113.7", "2606:4700::1111"):
            self.assertTrue(key_needed_for(addr, "k", "lan"), addr)

    def test_localhost_scope_exempts_this_pc_only(self):
        for addr in ("127.0.0.1", "127.1.2.3", "::1"):
            self.assertFalse(key_needed_for(addr, "k", "localhost"), addr)
        for addr in ("192.168.1.5", "10.0.0.1", "8.8.8.8", "fd00::1", "127.0.0.1.5"):
            self.assertTrue(key_needed_for(addr, "k", "localhost"), addr)

    def test_scope_off_asks_nobody(self):
        for addr in ("8.8.8.8", "2606:4700::1111", "192.168.1.5", "not-an-ip"):
            self.assertFalse(key_needed_for(addr, "k", "off"), addr)
        for scope in API_KEY_SCOPES:               # with no key configured, nothing is ever asked
            self.assertFalse(key_needed_for("8.8.8.8", "", scope), scope)

    def test_all_scope_is_upstreams_behaviour(self):
        """No exemption at all: every caller, this PC included, must present the key (what 0.1.38 did)."""
        for addr in ("8.8.8.8", "192.168.1.5", "127.0.0.1", "::1"):
            self.assertTrue(key_needed_for(addr, "k", "all"), addr)
            self.assertFalse(key_needed_for(addr, "", "all"), addr)     # ... still nothing with no key set
        nets = parse_netblocks(["10.1.0.0/16"])
        self.assertTrue(key_needed_for("10.1.2.3", "k", "all", nets), "the allow list does not exempt in \"all\"")

    def test_allow_list_adds_addresses_under_any_scope(self):
        nets = parse_netblocks(["192.168.4.7", "10.1.0.0/16", "2001:db8::/32"])
        for addr in ("192.168.4.7", "10.1.99.99", "2001:db8::5"):
            self.assertFalse(key_needed_for(addr, "k", "localhost", nets), addr)
        for addr in ("192.168.4.8", "10.2.0.1", "10.0.0.1"):                     # 10.2/10.0 are outside 10.1/16
            self.assertTrue(key_needed_for(addr, "k", "localhost", nets), addr)
        # the allow list is honoured in every scope, "lan" included: it can put a public address in
        public = parse_netblocks(["8.8.8.8"])
        self.assertFalse(key_needed_for("8.8.8.8", "k", "lan", public))
        self.assertTrue(key_needed_for("8.8.4.4", "k", "lan", public))
        # and it exempts nobody while no key is set: there is nothing to be exempt from
        for scope in API_KEY_SCOPES:
            self.assertFalse(key_needed_for("192.168.4.7", "", scope, nets), scope)

    def test_an_unreadable_peer_address_needs_the_key(self):
        for addr in ("", "not-an-ip", "192.168.1.5:1234", None, "::1/128"):
            self.assertTrue(key_needed_for(addr, "k", "lan"), addr)

    def test_parse_netblocks_and_allow_of(self):
        self.assertEqual([str(n) for n in parse_netblocks(["10.0.0.0/8", "192.168.4.7", "::1", "10.0.0.0/8"])],
                         ["10.0.0.0/8", "192.168.4.7/32", "::1/128"])
        self.assertEqual(api_key_allow_of(None), [])
        self.assertEqual(api_key_allow_of(""), [])
        self.assertEqual(api_key_allow_of("192.168.4.7"), ["192.168.4.7/32"])
        self.assertEqual(api_key_allow_of("192.168.4.7 , 172.16.0.0/12"),          # --api-key-allow / the env
                         ["192.168.4.7/32", "172.16.0.0/12"])
        self.assertEqual(api_key_allow_of(["10.0.0.0/8"]), ["10.0.0.0/8"])          # the config's list form
        for bad in (["evil.com"], ["10.0.0.0/33"], ["192.168.1.5:8080"], ["10.0.0.0/8", None], 5, [3],
                    ["192.168.4.7/24"]):        # a host inside a netblock is a typo, not a /24
            with self.assertRaises(ValueError, msg=bad):
                api_key_allow_of(bad)

    def test_service_layer_matches_the_pure_function(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(svc.api_key_scope, "all")  # the default: upstream 0.1.38, no exemption at all
        svc.api_key = "k"
        for addr in ("127.0.0.1", "::1", "192.168.5.5", "10.0.0.1", "8.8.8.8"):
            self.assertTrue(svc.key_needed_for(addr), addr)
        policy = svc.set_api_key_policy(scope="LAN", allow=["10.0.0.0/8"])
        self.assertEqual(policy["api_key_scope"], "lan")
        self.assertEqual(policy["api_key_allow"], ["10.0.0.0/8"])
        self.assertTrue(policy["api_key"])
        self.assertFalse(svc.key_needed_for("192.168.5.5"))     # now the local network is exempt
        self.assertFalse(svc.key_needed_for("127.0.0.1"))
        self.assertTrue(svc.key_needed_for("8.8.8.8"))
        with self.assertRaises(ValueError):
            svc.set_api_key_policy(scope="wan")
        with self.assertRaises(ValueError):
            svc.set_api_key_policy(allow=["nope"])
        self.assertEqual(svc.api_key_scope, "lan")             # a refused call changed nothing
        self.assertEqual(svc.api_key_allow, ["10.0.0.0/8"])

    def test_the_default_asks_everyone_exactly_as_upstream_did(self):
        """No scope given anywhere: the key is required from every caller, this PC included (0.1.38's behaviour)."""
        self.assertTrue(key_needed_for("127.0.0.1", "k"))       # the pure function's own default too
        self.assertTrue(key_needed_for("192.168.1.5", "k"))
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(svc.api_key_scope, "all")
        svc.api_key = "k"
        for addr in ("127.0.0.1", "::1", "192.168.1.5", "10.0.0.1", "fd00::1", "8.8.8.8", "not-an-ip"):
            self.assertTrue(svc.key_needed_for(addr), addr)
        svc.api_key = ""                                        # nothing to be exempt from, so nothing asked
        for addr in ("127.0.0.1", "8.8.8.8"):
            self.assertFalse(svc.key_needed_for(addr), addr)

    def test_the_scope_and_allow_list_are_invocation_flags(self):
        """The policy can be chosen on the command line, not only through POST /props."""
        import subprocess
        p = subprocess.run([sys.executable, str(ROOT / "serve/server.py"), "--help"],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, 0, p.stderr)
        for flag in ("--api-key-scope", "--api-key-allow"):
            self.assertIn(flag, p.stdout)
        self.assertIn("all", p.stdout)
        bad = subprocess.run([sys.executable, str(ROOT / "serve/server.py"), "--api-key-scope", "wan"],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(bad.returncode, 2)                     # argparse refuses a scope that is not one of them
        self.assertIn("invalid choice", bad.stderr)


class TuneKeys(unittest.TestCase):
    """The engine tune keys POST /props can set as every request's default."""

    def test_clean_tune_keeps_known_keys_and_checks_the_values(self):
        self.assertEqual(clean_tune({"pcie_frac": 0.4, "prefill": 4096}), {"pcie_frac": 0.4, "prefill": 4096})
        self.assertEqual(clean_tune({"prefill": 4096.0}), {"prefill": 4096})    # an int key is an int
        self.assertEqual(clean_tune({}), {})
        for bad in ({"nope": 1}, {"prefill": "4096"}, {"prefill": True}, {"prefill": -1}, {"pcie_frac": 1.5},
                    {"mtp_max_t": 99}, {"suffix_draft": 65}):
            with self.assertRaises(ValueError, msg=bad):
                clean_tune(bad)
            # the lenient form (a request's own keys) drops the offender instead of failing the request
            self.assertEqual(clean_tune({**bad, "prefill": 512}, strict=False), {"prefill": 512}, bad)

    def test_tune_str_is_the_engine_s_spelling(self):
        self.assertEqual(tune_str({"prefill": 4096, "spec_min_p": 0.4}), " prefill=4096 spec_min_p=0.4")
        self.assertEqual(tune_str({}), "")
        self.assertEqual(StrataEngine.sampling_keys({"strata_tune": {"prefill": 2048}}), " prefill=2048")

    def test_every_documented_scope_and_key_is_reachable_from_the_module(self):
        self.assertEqual(set(API_KEY_SCOPES), {"lan", "localhost", "all", "off"})
        self.assertTrue({"pcie_frac", "spec_min_p"} <= set(TUNE_KEYS))

    def test_the_engine_gets_a_tune_line_only_when_the_values_change(self):
        """The server->engine half, against a fake pipe: a `TUNE` line goes out when the request's tune differs
        from what the engine already has, and not when it does not."""
        def engine():
            eng = StrataEngine.__new__(StrataEngine)          # no process, no GPU: only the write path
            # upstream 0.1.41's generate() checks alive() first, which asks proc.poll(): the fake pipe answers
            eng.proc = types.SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
            eng.applied_tune, eng.last, eng.progress, eng.prefill_tok_s_mean = {}, {}, None, None
            eng.lines, eng.can_stop, eng.silence_s, eng.silent_note = queue.Queue(), True, 0.0, None
            return eng

        def run(eng, sampling):
            for line in ("T 7\n", "DONE 1 3 10 20 stop 0 0 3 0 0 0 0 0.0 3\n"):
                eng.lines.put(line)
            list(eng.generate([1, 2, 3], 1, sampling, threading.Event()))
            return eng.proc.stdin.getvalue()

        eng = engine()
        written = run(eng, {"strata_tune": {"prefill": 4096, "spec_min_p": 0.4}})
        self.assertIn("TUNE prefill=4096 spec_min_p=0.4\n", written)
        self.assertIn("GEN 1 prefill=4096 spec_min_p=0.4 1,2,3\n", written)
        eng.proc.stdin = io.StringIO()
        self.assertNotIn("TUNE", run(eng, {"strata_tune": {"prefill": 4096, "spec_min_p": 0.4}}))
        eng.proc.stdin = io.StringIO()
        self.assertIn("TUNE\n", run(eng, {}))                 # bare TUNE: back to the engine's own defaults
        eng.proc.stdin = io.StringIO()
        self.assertIn("TUNE prefill=2048\n", run(eng, {"strata_tune": {"prefill": 2048, "nope": 1}}))


class PropsOverHttp(HttpHarness, unittest.TestCase):
    """GET and POST /props in the running server: the key's policy, retunable while it runs."""

    HDR = {"Content-Type": "application/json", "Authorization": "Bearer s3cret"}

    def test_get_props_reports_the_policy(self):
        self.start(api_key="s3cret", api_key_scope="lan")
        code, body, _ = self.req("GET", "/props", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(code, 200)
        self.assertTrue(body["api_key"])
        self.assertEqual(body["api_key_scope"], "lan")
        self.assertEqual(body["api_key_allow"], [])
        self.assertEqual(body["strata_tune"], {})

    def test_this_pc_needs_no_key_and_the_api_answers(self):
        self.start(api_key="s3cret", api_key_scope="lan")
        self.assertEqual(self.req("GET", "/v1/models")[0], 200)
        self.assertEqual(self.req("POST", "/v1/chat/completions", self.chat_body(),
                                  {"Content-Type": "application/json"})[0], 200)
        self.assertEqual(self.req("GET", "/props")[0], 200)          # GET /props is not the write path

    def test_scope_off_asks_nobody(self):
        self.start(api_key="s3cret", api_key_scope="off")
        self.assertEqual(self.req("GET", "/v1/models")[0], 200)
        self.assertEqual(self.req("GET", "/v1/models", headers={"Authorization": "Bearer wrong"})[0], 200)

    def test_props_needs_the_key_when_one_is_set(self):
        # the exempt address must not be able to turn the exemption into everyone's
        self.start(api_key="s3cret", api_key_scope="lan")   # an exempting scope: this PC skips the key
        code, body, _ = self.req("POST", "/props", {"api_key_scope": "off"}, {"Content-Type": "application/json"})
        self.assertEqual(code, 401)
        self.assertIn("api_key_scope", body["error"]["message"])
        self.assertEqual(self.svc.api_key_scope, "lan")
        code, body, _ = self.req("POST", "/props", {"api_key_scope": "off"}, self.HDR)
        self.assertEqual(code, 200)
        self.assertEqual(body["changed"], ["api_key_scope"])
        self.assertEqual(self.svc.api_key_scope, "off")
        self.assertTrue(body["success"])

    def test_scope_and_allow_list_are_retunable(self):
        self.start(api_key="s3cret")
        code, body, _ = self.req("POST", "/props",
                                 {"api_key_scope": "localhost", "api_key_allow": ["10.1.0.0/16", "192.168.4.7"]},
                                 self.HDR)
        self.assertEqual(code, 200)
        self.assertEqual(body["api_key_allow"], ["10.1.0.0/16", "192.168.4.7/32"])
        self.assertTrue(self.svc.key_needed_for("192.168.9.9"))      # the LAN left the exemption
        self.assertFalse(self.svc.key_needed_for("10.1.2.3"))        # the allow list put it back, per address
        self.assertFalse(self.svc.key_needed_for("127.0.0.1"))       # this PC never needs it
        code, body, _ = self.req("POST", "/props", {"api_key_scope": "lan", "api_key_allow": []}, self.HDR)
        self.assertEqual(code, 200)
        self.assertEqual(body["api_key_allow"], [])
        self.assertFalse(self.svc.key_needed_for("192.168.9.9"))

    def test_the_key_itself_is_retunable_and_can_be_cleared(self):
        self.start(api_key="s3cret", api_key_scope="localhost")
        code, _, _ = self.req("POST", "/props", {"api_key": "another"}, self.HDR)
        self.assertEqual(code, 200)
        self.assertEqual(self.svc.api_key, "another")
        # A loopback peer can never be made to present a key - that is what the exemptions ARE - so assert the
        # predicate the server uses for the addresses the key does cover, and that the new key is the one that works
        self.assertTrue(self.svc.key_needed_for("192.168.9.9"))
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer another"}
        self.assertEqual(self.req("GET", "/props", headers={"Authorization": "Bearer another"})[0], 200)
        self.assertEqual(self.req("POST", "/props", {"api_key_scope": "lan"}, hdr)[0], 200)
        self.assertEqual(self.req("POST", "/props", {"api_key": ""}, hdr)[0], 200)
        self.assertEqual(self.svc.api_key, "")
        self.assertEqual(self.req("GET", "/v1/models")[0], 200)

    def test_bad_values_are_400_and_change_nothing(self):
        self.start(api_key="s3cret")
        before = (self.svc.api_key, self.svc.api_key_scope, list(self.svc.api_key_allow), dict(self.svc.tune_defaults))
        for body in ({"api_key_scope": "wan"}, {"api_key_scope": 5}, {"api_key_allow": ["evil.com"]},
                     {"api_key_allow": "10.0.0.0/8"}, {"api_key_allow": ["10.0.0.0/33"]}, {"api_key": 5},
                     {"nosuchkey": 1}, {"strata_tune": {"nosuch": 1}}, {"strata_tune": {"prefill": 999999}},
                     {"strata_tune": ["prefill"]}, {"api_key_scope": "off", "api_key_allow": ["nope"]}, []):
            # the last-but-one: a valid scope with a bad allow list must apply NEITHER ("nothing half-applied")
            code, payload, _ = self.req("POST", "/props", body, self.HDR)
            self.assertEqual(code, 400, body)
            self.assertEqual((self.svc.api_key, self.svc.api_key_scope, list(self.svc.api_key_allow),
                              dict(self.svc.tune_defaults)), before, body)
        code, body, _ = self.req("POST", "/props", {}, self.HDR)         # nothing to change: a no-op, not an error
        self.assertEqual((code, body["changed"]), (200, []))

    def test_props_from_a_foreign_page_or_a_form_is_refused(self):
        self.start(api_key="s3cret", api_key_scope="lan")   # an exempting scope: this PC skips the key
        self.assertEqual(self.req("POST", "/props", {"api_key_scope": "off"},
                                  {**self.HDR, "Origin": "http://evil.example.com"})[0], 403)
        self.assertEqual(self.req("POST", "/props", b"a=1",
                                 {"Content-Type": "application/x-www-form-urlencoded",
                                  "Authorization": "Bearer s3cret"})[0], 415)
        self.assertEqual(self.svc.api_key_scope, "lan")

    def test_strata_tune_becomes_every_request_s_default(self):
        self.start()
        code, body, _ = self.req("POST", "/props", {"strata_tune": {"prefill": 4096, "spec_min_p": 0.4,
                                                                   "adapt_every": 100000}},
                                 {"Content-Type": "application/json"})
        self.assertEqual(code, 200)
        self.assertEqual(body["strata_tune"], {"prefill": 4096, "spec_min_p": 0.4, "adapt_every": 100000})
        self.assertTrue({"prefill", "pcie_frac"} <= set(body["tune_keys"]))
        keys = StrataEngine.sampling_keys({"strata_tune": self.svc.tune_defaults})
        for want in (" prefill=4096", " spec_min_p=0.4", " adapt_every=100000"):
            self.assertIn(want, keys)
        # a request's own value wins over the default; a null drops the default again
        self.assertIn(" prefill=1024", StrataEngine.sampling_keys({"strata_tune": {"prefill": 1024}}))
        self.assertEqual(self.req("POST", "/props", {"strata_tune": {"prefill": None,
                                                                     "adapt_every": None}},
                                  {"Content-Type": "application/json"})[0], 200)
        self.assertEqual(self.svc.tune_defaults, {"spec_min_p": 0.4})

    def test_a_rebinding_page_is_still_refused_when_this_pc_is_exempt(self):
        # the regression that matters: with a key set, scope "lan" exempts 127.0.0.1, and a rebinding page
        # arrives from 127.0.0.1 - the Host check is the only thing left standing for it
        self.start(api_key="s3cret", api_key_scope="lan")
        code, body, log = self.req("GET", "/v1/models", host="evil.example.com")
        self.assertEqual(code, 403)
        self.assertIn("allowed_hosts", body["error"]["message"])
        self.assertIn("evil.example.com", log)
        self.assertEqual(self.req("GET", "/v1/models", headers={"Authorization": "Bearer s3cret"},
                                  host="random-words.trycloudflare.com")[0], 200)

    def test_a_cross_site_page_is_still_refused_when_this_pc_is_exempt(self):
        self.start(api_key="s3cret", api_key_scope="lan")
        code, body, _ = self.req("POST", "/v1/chat/completions", self.chat_body(),
                                 {"Content-Type": "text/plain", "Origin": "http://evil.example.com"})
        self.assertEqual(code, 403)
        self.assertIn("api_key", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
