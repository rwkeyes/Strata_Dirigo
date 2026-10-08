"""serve/test_minefield.py - the serving-path checks the Blackwellboy minefield doctor found on this server.

Each test is named after the trap it pins, so a regression reads as the trap reopening:

  * 78  a `tool_choice` this server cannot honour used to be accepted and ignored (it failed OPEN: a turn the
        caller believed was read-only called a tool),
  * 77  any invented top-level request field was accepted with a 200, so a typo was silent and the status code
        confirmed nothing; every response now carries the effective settings in a `strata` block,
  * 12  a reply that spends its whole budget thinking has no answer, and the response now says so,
  * 04/25 a prior assistant turn with no reasoning rendered an empty <think></think> wrapper.

    python -m unittest serve.test_minefield -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, offer_tools, tool_choice_strict, unknown_params  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, check_params,  # noqa: E402
                          effective_settings, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "the weather",
                                           "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "read_file", "description": "read a file",
                                           "parameters": {"type": "object", "properties": {}}}}]


class Recorder(MockEngine):
    """Keeps the prompt the engine was asked to read, so a test can see what the request rendered to."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        self.last_max_new = max_new
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def _svc(text="</think>\n\nan answer", engine_cls=Recorder, **attrs):
    tok = ByteTokenizer()
    eng = engine_cls(tok, text, max_context=CTX)
    svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    for k, v in attrs.items():
        setattr(svc, k, v)
    return svc, eng


class HttpCase(unittest.TestCase):
    """A server on a random port against the mock engine; `post` returns (status, body)."""

    engine_cls = Recorder
    text = "</think>\n\nan answer"

    @classmethod
    def setUpClass(cls):
        cls.svc, cls.engine = _svc(cls.text, **getattr(cls, "svc_attrs", {}))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        body = {"model": "x", "max_tokens": 40, "messages": [{"role": "user", "content": "hi"}], **extra}
        return self.post("/v1/chat/completions", body)

    def prompt_text(self, engine=None):
        eng = engine or self.engine
        return bytes(i for i in eng.last_ids if i < 256).decode("utf-8", "replace")


# -------------------------------------------------------------------------------------------------- 78 tool_choice
class ToolChoiceUnit(unittest.TestCase):
    """`tool_choice_of` on the values clients send, and `offer_tools` on what it does with them."""

    def test_absent_is_auto(self):
        self.assertEqual(tool_choice_strict({}), ("auto", None))

    def test_the_three_words(self):
        for word in ("auto", "none", "required"):
            self.assertEqual(tool_choice_strict({"tool_choice": word}), (word, None), word)

    def test_a_named_function_in_both_dialects(self):
        self.assertEqual(tool_choice_strict({"tool_choice": {"type": "function",
                                                         "function": {"name": "f"}}}), ("function", "f"))
        self.assertEqual(tool_choice_strict({"tool_choice": {"type": "tool", "name": "f"}}, "anthropic"),
                         ("function", "f"))

    def test_anthropics_any_is_required(self):
        self.assertEqual(tool_choice_strict({"tool_choice": {"type": "any"}}, "anthropic"), ("required", None))

    def test_a_value_that_cannot_be_honoured_is_refused(self):
        for bad in ("sometimes", {"type": "occasionally"}, {"type": "function"}, {"type": "tool"}, (1, 2), 7):
            with self.assertRaises(ValueError, msg=bad):
                tool_choice_strict({"tool_choice": bad})

    def test_none_omits_the_payload_entirely(self):
        tools, report = offer_tools(TOOLS, "none", None)
        self.assertIsNone(tools)                          # nothing is sent, so nothing can be called
        self.assertTrue(report["applied"])

    def test_auto_and_required_keep_the_payload(self):
        for mode in ("auto", "required"):
            tools, report = offer_tools(TOOLS, mode, None)
            assert tools is not None                       # the payload is kept for both
            self.assertEqual(len(tools), 2, mode)
            self.assertEqual(report["requested"], mode)

    def test_required_is_reported_applied(self):
        # 2026-10-08: upstream 0.1.41 enforces "required" by opening the call in the prompt itself (forced_call),
        # so this fork reports it applied.  Before, nothing could force a call and the report said so.
        _tools, report = offer_tools(TOOLS, "required", None)
        self.assertTrue(report["applied"])
        self.assertIn("forced_call", report["how"])

    def test_a_named_function_narrows_the_offer(self):
        tools, report = offer_tools(TOOLS, "function", "read_file")
        assert tools is not None
        self.assertEqual([t["function"]["name"] for t in tools], ["read_file"])
        self.assertTrue(report["applied"])

    def test_a_name_the_request_does_not_offer_is_refused(self):
        with self.assertRaises(ValueError):
            offer_tools(TOOLS, "function", "not_a_tool")

    def test_no_tools_at_all_is_reported(self):
        self.assertEqual(offer_tools(None, "none", None)[1]["applied"], False)


class ToolChoiceOverHttp(HttpCase):
    def test_none_keeps_the_tools_out_of_the_prompt(self):
        status, b = self.chat(tools=TOOLS, tool_choice="none")
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertNotIn("get_weather", text)             # the engine was never offered a tool
        self.assertNotIn("<tool_call>", text)
        self.assertEqual(b["strata"]["tool_choice"]["applied"], True)
        self.assertEqual(b["strata"]["tools_offered"], 0)

    def test_without_a_choice_the_tools_are_offered(self):
        status, b = self.chat(tools=TOOLS)
        self.assertEqual(status, 200, b)
        self.assertIn("get_weather", self.prompt_text())
        self.assertEqual(b["strata"]["tools_offered"], 2)

    def test_a_named_function_leaves_the_others_out(self):
        status, b = self.chat(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "read_file"}})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertIn("read_file", text)
        self.assertNotIn("get_weather", text)

    def test_the_anthropic_path_honours_none_too(self):
        anthropic_tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
                            "input_schema": t["function"]["parameters"]} for t in TOOLS]
        status, b = self.post("/v1/messages", {"model": "x", "max_tokens": 40, "tools": anthropic_tools,
                                               "tool_choice": {"type": "none"},
                                               "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, b)
        self.assertNotIn("get_weather", self.prompt_text())
        self.assertEqual(b["strata"]["tools_offered"], 0)

    def test_a_bad_choice_is_a_400_naming_it(self):
        for bad in ("sometimes", {"type": "function"}, {"type": "occasionally"}):
            status, b = self.chat(tools=TOOLS, tool_choice=bad)
            self.assertEqual(status, 400, (bad, b))
            self.assertIn("tool_choice", json.dumps(b))

    def test_an_unknown_function_is_a_400(self):
        status, b = self.chat(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}})
        self.assertEqual(status, 400, b)
        self.assertIn("nope", json.dumps(b))


# ----------------------------------------------------------------------------------------- 77 the request surface
class RequestSurfaceUnit(unittest.TestCase):
    def test_a_field_this_server_does_not_implement_is_named(self):
        self.assertEqual(unknown_params({"model": "x", "__probe__": 1}), ["__probe__"])

    def test_a_known_field_is_not(self):
        for field in ("model", "messages", "temperature", "max_tokens", "tools", "strata_tune",
                      "chat_template_kwargs", "reasoning_budget_tokens"):
            self.assertEqual(unknown_params({field: 1}), [], field)

    def test_check_params_400s_only_when_strict(self):
        svc = Service.__new__(Service)                    # no engine needed for this helper
        svc.strict_params, svc.params_warned = False, set()
        self.assertEqual(check_params({"__probe__": 1}, "openai", svc), ["__probe__"])
        svc.strict_params = True
        with self.assertRaises(ValueError):
            check_params({"__probe__": 1}, "openai", svc)

    def test_a_name_is_recorded_so_the_log_says_it_once(self):
        svc = Service.__new__(Service)
        svc.strict_params, svc.params_warned = False, set()
        self.assertEqual(check_params({"__probe__": 1}, "openai", svc), ["__probe__"])
        self.assertEqual(svc.params_warned, {"__probe__"})        # the log names it once, not per request
        self.assertEqual(check_params({"__probe__": 1, "__other__": 2}, "openai", svc),
                         ["__probe__", "__other__"])              # the caller still sees every unknown field


class StrictParamsOverHttp(HttpCase):
    svc_attrs = {"strict_params": True}

    def test_an_invented_field_is_refused(self):
        status, b = self.chat(__minefield_unvalidated_field_probe__="yes")
        self.assertEqual(status, 400, b)
        self.assertIn("__minefield_unvalidated_field_probe__", json.dumps(b))

    def test_a_normal_request_still_works(self):
        status, b = self.chat(temperature=0.5, top_p=0.9, tools=TOOLS, tool_choice="auto")
        self.assertEqual(status, 200, b)


class LenientParamsOverHttp(HttpCase):
    def test_an_invented_field_is_accepted_and_does_nothing(self):
        status, b = self.chat(__minefield_unvalidated_field_probe__="yes")
        self.assertEqual(status, 200, b)
        self.assertEqual(b["strata"]["max_tokens"], 40)    # what ran is reported, not what was sent


# --------------------------------------------------------------------------------------- 77 the effective settings
class EffectiveSettingsOverHttp(HttpCase):
    def test_the_response_says_what_ran(self):
        status, b = self.chat(temperature=0.5, reasoning_budget_tokens=64)
        self.assertEqual(status, 200, b)
        s = b["strata"]
        self.assertEqual(s["max_tokens"], 40)
        self.assertIsInstance(s["thinking"], bool)
        self.assertIn("temperature=0.5", s["sampling"])     # the engine's own spelling, not a second guess
        self.assertEqual(s["reasoning_budget_tokens"], 64)
        self.assertEqual(s["tool_choice"]["requested"], "auto")

    def test_a_stream_carries_it_on_the_first_chunk(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"model": "x", "max_tokens": 20, "stream": True,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            first = json.loads(r.read().split(b"\n\n")[0].removeprefix(b"data: "))
        self.assertEqual(first["choices"][0]["delta"].get("role"), "assistant")
        self.assertEqual(first["strata"]["max_tokens"], 20)

    def test_the_settings_helper_reports_the_budget_only_while_thinking(self):
        svc, _eng = _svc()
        s = effective_settings(svc, {"reasoning_budget_tokens": 32}, thinking=True, tools=None,
                               tool_choice={"requested": "auto"}, max_new=10)
        self.assertEqual(s["reasoning_budget_tokens"], 32)
        s = effective_settings(svc, {"reasoning_budget_tokens": 32}, thinking=False, tools=None,
                               tool_choice={"requested": "auto"}, max_new=10)
        self.assertNotIn("reasoning_budget_tokens", s)      # no thinking, no budget to report


# ------------------------------------------------------------------------------------------------ 12 the cap hit
class CapHitOverHttp(HttpCase):
    text = "I keep thinking and never stop"            # all reasoning, no </think>, so no answer

    def test_a_budget_spent_thinking_is_named_in_the_response(self):
        status, b = self.chat(max_tokens=12)
        self.assertEqual(status, 200, b)
        self.assertEqual(b["choices"][0]["finish_reason"], "length")
        self.assertIsNone(b["choices"][0]["message"]["content"])
        self.assertEqual(b["strata"]["cap_hit"], "reasoning")


class RetunableOverProps(HttpCase):
    """The two Minefield knobs are retunable at runtime, like the API-key policy (POST /props, no restart)."""

    def test_props_reports_them(self):
        status, b = self.post("/props", {})
        self.assertEqual(status, 200, b)
        self.assertIn("strict_params", b)
        self.assertIn("preserve_empty_think", b)

    def test_a_request_field_can_be_refused_without_a_restart(self):
        try:
            status, b = self.post("/props", {"strict_params": True})
            self.assertEqual(status, 200, b)
            self.assertEqual(b["strict_params"], True)
            status, b = self.chat(__probe__=1)
            self.assertEqual(status, 400, b)                   # now a field it cannot honour is a 400
        finally:
            self.post("/props", {"strict_params": False})
        status, b = self.chat(__probe__=1)
        self.assertEqual(status, 200, b)                       # ... and back to naming it and carrying on

    def test_a_non_boolean_is_a_400(self):
        status, b = self.post("/props", {"preserve_empty_think": "yes"})
        self.assertEqual(status, 400, b)
        self.assertIn("true or false", json.dumps(b))


class EmptyThinkGuardModule(unittest.TestCase):
    """The guard that goes into a PACK's template (tools/empty_think_guard.py).

    The server renders the pack's `tokenizer/chat_template.jinja` in preference to the repo's file
    (`server.py`: `ChatTemplate(pack_tpl if pack_tpl.exists() else ROOT / "serve/chat_template.jinja")`), and the
    pack's copy is extracted from the GGUF metadata by `tools/strata_tokenizer.py` - so the same guard has to be
    applied where a pack is built, and to packs built before the fix (`~/bin/strata-fix-pack-template.sh`).
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        import empty_think_guard as g
        cls.g = g

    def test_it_reproduces_the_repos_own_template_byte_for_byte(self):
        """The guard's output IS serve/chat_template.jinja: the two paths cannot drift apart unnoticed."""
        upstream = self.g.revert((ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))[0]
        self.assertEqual(self.g.apply(upstream)[0], (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))

    def test_the_repo_template_is_already_guarded(self):
        text = (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8")
        self.assertEqual(self.g.apply(text)[1], "already guarded")

    def test_an_unguarded_template_gains_the_guard_exactly_once(self):
        upstream = self.g.revert((ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))[0]
        once, what = self.g.apply(upstream)
        self.assertEqual(what, "applied")
        self.assertEqual(self.g.apply(once)[1], "already guarded")     # idempotent
        self.assertIn("preserve_empty_think", once)

    def test_revert_restores_the_checkpoints_template(self):
        upstream = self.g.revert((ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))[0]
        self.assertEqual(self.g.revert(self.g.apply(upstream)[0])[0], upstream)

    def test_another_architectures_template_is_left_alone(self):
        other = '{%- if message.role == "assistant" %}{{- content }}{%- endif %}'
        got, what = self.g.apply(other)
        self.assertEqual((got, what), (other, "pattern not found"))   # no guessing, no rewriting

    def test_the_guarded_pack_render_skips_the_empty_wrapper(self):
        t = ChatTemplate(ROOT / "serve/chat_template.jinja")
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "again"}]
        self.assertNotIn("<think>\n\n</think>", t.render(history, add_generation_prompt=False))
        self.assertIn("<think>\n\n</think>",
                      t.render(history, add_generation_prompt=False, preserve_empty_think=True))


# --------------------------------------------------------------------------------------- 04/25 the empty think block
class EmptyThinkTemplate(unittest.TestCase):
    """The vendored template's rendering of a prior assistant turn that has no reasoning."""

    @classmethod
    def setUpClass(cls):
        cls.t = ChatTemplate(ROOT / "serve/chat_template.jinja")

    messages = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "again"}]

    def test_the_packs_rendering_has_the_empty_wrapper(self):
        got = self.t.render(self.messages, add_generation_prompt=False, preserve_empty_think=True)
        self.assertIn("<think>\n\n</think>", got)          # what the checkpoint's own template does

    def test_this_server_skips_an_empty_wrapper(self):
        got = self.t.render(self.messages, add_generation_prompt=False)
        self.assertNotIn("<think>", got)
        self.assertIn("<|im_start|>assistant\nhi there<|im_end|>", got)

    def test_real_reasoning_is_still_preserved(self):
        with_reasoning = [dict(m, reasoning_content="2+2=4") if m["role"] == "assistant" else m
                          for m in self.messages]
        got = self.t.render(with_reasoning, add_generation_prompt=False)
        self.assertIn("<think>\n2+2=4\n</think>", got)

    def test_the_goldens_match_this_tree(self):
        cases = json.loads((ROOT / "serve/chat_golden.json").read_text())
        flags = ("add_generation_prompt", "enable_thinking", "preserve_thinking", "reasoning_effort")
        for c in cases:
            kw = {**(c.get("kwargs") or {}), **{k: c[k] for k in flags if k in c}}
            got = self.t.render(c["messages"], tools=c.get("tools"), **kw)
            self.assertEqual(got, c["rendered"], c["name"])


class EmptyThinkOverHttp(HttpCase):
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"},
               {"role": "user", "content": "again"}]

    def test_a_request_gets_the_pack_rendering_when_it_asks_for_it(self):
        status, b = self.chat(messages=self.history,
                              chat_template_kwargs={"preserve_empty_think": True})
        self.assertEqual(status, 200, b)
        self.assertIn("<think>\n\n</think>", self.prompt_text())

    def test_by_default_the_empty_block_is_not_rendered(self):
        status, b = self.chat(messages=self.history)
        self.assertEqual(status, 200, b)
        self.assertNotIn("<think>\n\n</think>", self.prompt_text())


class EmptyThinkFromConfig(HttpCase):
    svc_attrs = {"preserve_empty_think": True}

    def test_the_config_restores_the_pack_rendering(self):
        status, b = self.chat(messages=EmptyThinkOverHttp.history)
        self.assertEqual(status, 200, b)
        self.assertIn("<think>\n\n</think>", self.prompt_text())

    def test_a_request_can_still_opt_out(self):
        status, b = self.chat(messages=EmptyThinkOverHttp.history,
                              chat_template_kwargs={"preserve_empty_think": False})
        self.assertEqual(status, 200, b)
        self.assertNotIn("<think>\n\n</think>", self.prompt_text())


if __name__ == "__main__":
    unittest.main()
