# The minefield doctor on this fork's lane — findings, fixes, retest

[`Blackwellboy/model-serving-minefield`](https://github.com/Blackwellboy/model-serving-minefield) is a registry of
traps in how requests are served: defects that are **not** the weights. Run against Strata's OpenAI surface it
audits the layer between the client and the model — template rendering, the request field set, tool gating,
thinking/routing and the budget arithmetic — which is exactly the layer this fork changes.

Run (cloned fresh, from the source, not a local adaptation):

```sh
git clone --depth 1 https://github.com/Blackwellboy/model-serving-minefield.git
python3 -m minefield quick --base-url http://127.0.0.1:18110/v1 --model <model-id> \
        --api-key "$STRATA_API_KEY" --hf-repo ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
        --hf-revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --json doctor.json
```

13 requests, 20 checks, the pinned checkpoint passed so its configuration checks fire.

## What the doctor found

| trap | what it means on this lane | verdict |
|---|---|---|
| **78** | `tool_choice: "none"` was accepted and **ignored**: a control with tools and no `tool_choice` called a tool, and the identical request with `"none"` called one too. **Fails open** — an agent loop with a side-effecting tool can act on a turn the caller believed read-only. | **problem, fixed** |
| **77** | The request surface was unvalidated: an invented top-level field was accepted with `200`, so a misspelling was silent and a status code confirmed nothing. | **problem, fixed** |
| **12** | A hard task at `max_tokens=512`: `finish=length`, **empty content**, 1966 chars of reasoning. Honest truncation — but a harness that scores that zero is measuring its own budget, not the model. | **problem, fixed** |
| **21** | The GGUF repo ships **no `generation_config.json`**: there is no such thing as "model defaults" here, sampling must be set explicitly per request/mode. | **problem, fixed** |
| **04/25** | History rendered **2 empty think blocks** for prior turns: a prior assistant turn whose reasoning was not resent got `<think>\n\n</think>`. Nudges thinking collapse on later turns, and makes equivalent histories render differently (a conversation-cache miss). | **problem, fixed** |
| **01, 02, 03, 20, 23, 29, 19, 26, mm** | Reasoning arrives under `reasoning_content` (and `reasoning` is a dead write name); no orphaned `</think>`; streamed deltas and real structured `tool_calls` on the forced probe; a text-only lane rejects an image part with a 400 naming the modality. | **clean** |
| **07** | `reasoning_effort` deadness — one render cannot show it; needs a two-render diff (`checks/preflight_template.py`). | inconclusive |
| **10, 17, 141, 22** | Coverage/scoping: config.json 404 (a GGUF repo), no shipped generation config, an SGLang-only entry, and "one budget cannot characterise the floor". | inconclusive |

The clean rows matter as much as the problems: they say the answers themselves are sound (no dequant corruption,
no orphaned tags, real tool calls), so every finding above is a serving-path defect and **none of them needs a
weight change, a re-quant, or a different checkpoint**.

## What changed in the fork

All server-side or template-side; the engine and the weights are untouched.

| trap | fix | where | setting |
|---|---|---|---|
| 78 | `tool_choice` implemented: `"none"` **omits the tools payload** from what the engine is offered (a lane never offered a tool cannot call one, whatever the template does), a named function narrows the offer to that one, `"required"` is offered and **reported unenforced**, and any value that cannot be honoured is a **400**. Anthropic's `{"type": "none"│"any"│"tool"}` too. | `serve/frontend.py` (`tool_choice_of`, `offer_tools`), `serve/server.py` (both API paths), and MCP tools are not even collected under `"none"` | per request |
| 77 | Unknown fields are named **once each** in the log (the API and the field), and **every response carries a `strata` block** with the effective settings, so a client asserts on the response per request rather than on the status code. `"strict_params": true` turns an unknown field into a **400** listing it. | `serve/frontend.py` (`unknown_params`, the field sets), `serve/server.py` (`check_params`, `effective_settings`, `openai_collect`) | `strict_params` (config or `POST /props`) |
| 12 | A reply whose whole budget went on thinking is **named**: `strata.cap_hit = "reasoning"` when `finish=length` with empty content and non-empty reasoning. The budget itself (`reasoning_budget_tokens`) already existed — see below for what the retest sets. | `serve/server.py` (`openai_collect`) | request or config |
| 21 | Sampling is reported as the engine was actually given it (`strata.sampling`, the engine's own spelling), so "unset" is visible instead of implied; the model configs carry the card's sampling per mode. | `serve/server.py` (`effective_settings`) | config `sampling` |
| 04/25 | The template writes the `<think>` wrapper **only when there is reasoning to preserve**. `preserve_empty_think: true` — config, a request's `chat_template_kwargs`, or `POST /props` — restores the checkpoint template's rendering exactly. | `serve/chat_template.jinja`, **and the pack** (see below) | `preserve_empty_think` (config, request, or `POST /props`) |

### 04/25 is a pack fix, not only a repo fix

The doctor reads the template the server **publishes** (`/props` → `chat_template`), and the server renders the
**pack's** `tokenizer/chat_template.jinja` in preference to the repo's file
(`server.py`: `ChatTemplate(pack_tpl if pack_tpl.exists() else ROOT / "serve/chat_template.jinja")`).  That pack
copy is extracted from the GGUF's own metadata by `tools/strata_tokenizer.py`, so editing the repo's template
alone never reaches a running deployment — the first retest proved it: 77 and 78 came back clean, 04/25 did not,
because the live pack still held the checkpoint's unguarded template.

The fix is therefore in three places, and they cannot drift apart silently:

* `tools/empty_think_guard.py` — the guard as a targeted, idempotent, revertible string replacement with a
  `--dry-run`; a template without the pattern it fixes (another architecture) is left exactly as shipped.
* `tools/strata_tokenizer.py` — applies it as each pack is written, so new packs carry it and the tool's output
  is byte-identical to `serve/chat_template.jinja`.
* `~/bin/strata-fix-pack-template.sh [--undo|--check] [TREE]` — applies it to packs built before the fix, one
  backup per template under `~/.backup/files/pack-template/`, idempotent, with the restore command printed.

Applied to the four packs on this box (`coder-iq1_m`, `iq3_s`, `iq3_xxs`, `swift-iq3_xxs`): all four verified
guarded, the empty pair gone from their renders, and `preserve_empty_think=true` restoring it.

`strict_params` and `preserve_empty_think` are retunable while the server runs (`POST /props`), like the API-key
policy, so a deployment can be tightened or relaxed without a restart.

### The golden cases the template change moves

3 of the pack's 10 golden cases contain a prior turn with no reasoning, so the fork's rendering differs from the
checkpoint template's there: `multi-turn`, `no generation prompt`, `tool call and response`. `serve/chat_golden.json`
records the new rendering with a `note` on each, and `preserve_empty_think=true` reproduces the pack's rendering
**byte-for-byte on all 10** — which is what makes the change a deliberate fork decision rather than a template
regression. It is the same trade the API-key scope makes: the safer default now, upstream's behaviour one setting
away.

## Tests

`serve/test_minefield.py` — one class per trap, named after it, so a regression reads as the trap reopening:
`tool_choice` (11 tests incl. that `"none"` keeps the tool out of the prompt the engine was handed, that a bad
value is a 400, and that a name the request does not offer is refused), the request surface (unit + strict/lenient
over HTTP + the settings you can change while it runs), the effective-settings echo (non-stream, stream's first chunk, Anthropic), the
cap-hit, and the template (the pack's rendering with that setting on, this server's without it, real reasoning still
preserved, and all 10 goldens matching this tree).

    python -m unittest serve.test_minefield -v      # 39 tests
    python -m unittest discover -s serve -p 'test_*.py'

## Retest

Protocol: the **same checkpoint on the same storage** as the pre-fix run (the NVMe-staged IQ1_M Coder, so the only
variable is the code), the fork's `serve/` against the same engine binary, then the upstream doctor +
`checks/dequant_fidelity.py` + this fork's 5-probe battery, with `strict_params: true` so trap 77's own probe is
judged by the fixed code.  `~/bin/strata-minefield-retest.sh` runs the whole thing; the logs are
`logs/minefield-retest.*`.

| trap | before (the arm's own doctor run) | after (the fixed server) |
|---|---|---|
| **77** unvalidated request surface | problem: invented field accepted with 200 | **CLEAN** — "the request surface is validated: an invented top-level field was rejected (http 400) while the identical baseline request stayed 200" |
| **78** `tool_choice` ignored | problem: `none` called a tool exactly as the control did | **CLEAN** — "`tool_choice` none binds: a control with tools and no tool_choice called a tool, and the identical request with tool_choice none did not" |
| **04/25** empty think shells | problem: "history renders 2 empty think block(s) for prior turns" | **CLEAN** — no `<think></think>` pair in the turn-3 render.  This needed the **pack** fix above: the first retest still reported it, which is what exposed that the server renders the pack's template |
| **12** empty content at a 512 cap | problem: `finish=length`, empty content, 1536 chars of reasoning | still reported — and that is correct: the doctor's ceiling probe asks for `enable_thinking: true` at `max_tokens=512`, so a thinking model caps out there whatever the server does.  What changed is that the answer is now **named and fixable**: the same request reports `strata.cap_hit = "reasoning"`, and with a per-request budget the same prompt answers |
| **21** no `generation_config.json` | problem: the checkpoint ships none | still reported — a property of the checkpoint, not of the code.  What changed: the effective sampling is in every response's `strata.sampling` (the engine's own spelling), so "unset" is visible and assertable instead of implied |

The doctor's coverage line moved with it: "a clean run above is a statement about **12 trap ids**" (up from 8 on
the pre-fix run of the same lane), and the trap list it now clears includes 77, 78, 04/25 and 25.

### The trap-12 remedy, measured on the same lane

```
 no budget: finish=length content=0ch     reasoning=1736ch  strata.cap_hit='reasoning'  budget=0
budget 128: finish=length content=938ch   reasoning=505ch   strata.cap_hit=None         budget=128
```

Same prompt, same cap: without a budget the reply spends all 512 tokens thinking and has no answer; with
`reasoning_budget_tokens: 128` it wraps up the thinking and writes 938 characters of answer.  The server's own log
says which happened (`"the thinking reached reasoning_budget_tokens"` vs `"the reply reached max tokens while still
thinking, so it has no answer"`).

### What the fork's own battery said on the same lane

`strata-minefield.py`: **5/5 scored**, every probe `finish=stop`, answers `Paris` / `9.9` / `ok` / `3` / `olleh`.  The
upstream `dequant_fidelity.py` reported `capital ok, decimal BLOCKING (out=''), nonempty ok` on the same lane — its
generation mode sends no `chat_template_kwargs` and a small budget, so on a thinking-by-default lane it reads the
empty `content` of a reply that was still thinking — exactly what trap 12 describes.  The two are not in
conflict: the fork's battery asks for thinking off and strips the CoT, which is the difference between measuring
the model and measuring the budget.  No dequant corruption is indicated by either.

### Reproducing

```sh
bash ~/bin/strata-fix-pack-template.sh --check          # every pack guarded?
bash ~/bin/strata-minefield-retest.sh                   # ~2 min: load + doctor + dequant + probes + budget test
python3 ~/bin/strata-arm-diff.py coder-iq1_m-hdd-16k coder-iq1_m-nvme-16k    # the storage A/B
```
