# Fork features — what `Strata-warm-retunable` adds over upstream

This is the feature ledger for the fork: every capability that upstream [Niko1221/Strata](https://github.com/Niko1221/Strata)
**v0.1.41** does not have, how to use it, what it costs, and how far it has been verified.  It is written to be
read on its own — the measurements and the audit behind each item are in `WARM-RETUNABLE.md` and
`warm-retune/RETUNE-CANDIDATES.md`.

Base: upstream **v0.1.41**.  Branch: **`warm-retune-0.1.41`**, the fork's default branch (`main`).
Delta: see `git diff --stat v0.1.41` on that branch — it is docs, benchmark evidence, the fork's own test files
and out-of-tree scripts plus the four source files below, and NOTHING upstream already carries.

### Upstream has taken four of this fork's seven items — what is left here

| the fork's item | status upstream |
|---|---|
| **AVX1 floor** (`STRATA_ISA_FLOOR`, the AVX1 router dot, the CPU gate) | **v0.1.39 and later carry it**, credited "From the Strata_Dirigo fork (rwkeyes)" in `CMakeLists.txt`. Upstream's spelling is the string `avx`/`none`, not this fork's old boolean `1`/`0`; the fork's copied source is retired here and its evidence (`bench/results/2026-10-01-avx1-floor/`) is kept. |
| **`tool_choice` gating** | **v0.1.41 enforces it in the prompt** (`forced_call`: the server writes the call's opening, so the model can only go on with a call). This fork keeps its own stricter reading on top: a choice it cannot honour is a **400**, and every response's `strata` block reports what was applied. |
| **`strata_tune` per request** | **v0.1.40.x takes `pcie_frac` and `spec_min_p`** per request. The fork's full `TUNE` channel (nine keys, `POST /props` defaults, the engine's `TUNED` reply) is still only here. |
| **`api_key_scope` / `api_key_allow`**, **`--no-open`**, **the named request surface** (`strict_params`, the `strata` block), **the empty-`<think>` guard** | **not upstream.**  These are what the fork is for. |

## At a glance

| # | Feature | Where it lives | Retunable at runtime | Verified |
|---|---|---|---|---|
| 1 | llama-server changes its own sizes: slots, per-slot context, KV pool, cache types — weights stay loaded | `warm-retune/retune-llama-server-3cf0325.patch` (vendored llama.cpp `tools/server`) | yes — `POST /props` | upstream patch: measured (0–0.4 ms to change the shares, ~0.8 s to enlarge the context) |
| 2 | **Strata's own engine** re-tunes without a restart | `src/program/generate.cpp` (`TUNE`), `serve/server.py` (`POST /props`) | yes — the whole point | tests + live HTTP; engine side compile-verified, not yet run on a GPU |
| 3 | API-key **scope** + **ALLOW list** (upstream's behaviour by default, exemption opt-in at invocation) | `serve/server.py` | yes — `POST /props` and the CLI flags | 47 security tests + live smoke |
| 4 | A start that does **not** pop a browser tab | `serve/server.py` (`--no-open`, `$STRATA_NO_BROWSER`) | no — invocation-time by design | tests + `--help` |
| 5 | **AVX1 floor** (upstream's since v0.1.39, from this fork) | upstream's `CMakeLists.txt`, `src/kernels/cpu/kq_avx1.*`, `src/core/expert_source.cpp`, `setup.py` | no — build-time | measured on a Xeon E5-2687W and on gfx1100 (`WARM-RETUNABLE.md`) |
| 6 | setup.py carries the llama.cpp patch and keeps `tools/ui` off Windows | `setup.py` | no | the build path is exercised by every setup run |
| 7 | The retune **audit**: what else could be retuned, and in which class | `warm-retune/RETUNE-CANDIDATES.md` | — | source audit, file/line evidence |
| 8 | **`tool_choice` is honoured** — `none` never offers the engine a tool | `serve/frontend.py`, `serve/server.py` | per request | 11 tests + the doctor's own probe |
| 9 | A request field this server does not implement is **named**, and every response says **what ran** | `serve/frontend.py`, `serve/server.py` | `strict_params` (config / `POST /props`) | tests + the doctor's probe |
| 10 | History without an **empty thinking block** (repo template **and** every pack) | `serve/chat_template.jinja`, `tools/empty_think_guard.py`, `tools/strata_tokenizer.py` | `preserve_empty_think` | 10 goldens byte-for-byte + the doctor's own probe clean |
| 11 | The minefield findings, fixes and retest, in one place | `warm-retune/MINEFIELD-FINDINGS.md` | — | the doctor's own probes |

## 1 — The vendored llama.cpp runtime retune (llama-server)

Upstream's vendored llama.cpp (`3cf0325`) answers `POST /props` with a stub, `{"success": true}` and nothing
else.  `warm-retune/retune-llama-server-3cf0325.patch` (1107 lines, 6 files) changes the same sizes on a **running**
server without reloading the weights:

| request | effect |
|---|---|
| `{"parallel": N, "ctx_per_slot": T}` | divide the unified KV pool across N slots, with no restart — 0–0.4 ms |
| `{"ctx": N}`, `{"cache_type_k": "q8_0"}`, `{"flash_attn": "on"}`, `{"parallel_max": N}`, `{"kv_unified": true}` | rebuild the context without restarting (~0.8 s for 32k→262k); the weights never reload |
| `{"allow_oversubscribe": true}` | let slots share a pool that cannot back them all |

A request that would evict a busy slot is refused with `HTTP 400 slot N is busy, cannot take it out of
service` — never a hang.  `/props` reports `total_slots`, `slots_max`, `kv_unified`, `kv_pool_n_ctx`.
`--kv-unified` is what makes the slot count a server-side policy instead of a load-time property, and `--props`
is what enables the write path.

`setup.py` applies the patch automatically once the vendored tree exists (`apply_warm_retune()`; never fatal);
by hand it is `warm-retune/apply.sh apply|undo|check`.  Strata's own engine does **not** build `tools/server`,
so the patch is inert until a `llama-server` is built from that tree.
Worked example with the reasons, the measured costs and the VRAM-per-pool table:
[`warm-retune/SLOTS-AND-KV-POOL.md`](warm-retune/SLOTS-AND-KV-POOL.md).

## 2 — Strata's own engine re-tunes without a restart

The same idea, inside Strata.  The engine is a long-running process fed one request per line on stdin
(`GEN <max_new> key=value … <ids>`); those keys already carried per-request tuning (`pcie_frac`, `spec_min_p`),
which is what setup's calibration measured from.  The fork adds the missing half:

* **`TUNE key=value …`** — an engine line that changes settings of the running engine between requests, with no
  reload, no VRAM movement and no session state dropped.  It answers `TUNED <applied> refused:<…>`, so a
  rejected key is visible rather than silent.
* **`POST /props {"strata_tune": {…}}`** (server, JSON from Strata's own page) — stores those keys as the
  default every later request carries; a request's own `strata_tune` in its body wins, `null` drops a key.
  `GET /props` reports the current values.  The server sends a `TUNE` line only when the values change.
* Both ends validate: `TUNE_KEYS` answers **400** for an unknown key or an out-of-range value (a typo must not
  look applied).

| key | what it changes |
|---|---|
| `prefill` | the prompt-path chunk — and so how much of the expert cache a prompt borrows. Down any time; **up only to the startup chunk** (the prompt buffers were sized then) |
| `short_read` | how many fresh tokens are read through the decode windows instead of the batched prompt path |
| `prompt_cache`, `prompt_cache_every` | conversation checkpoints kept between requests, and their cadence |
| `adapt_every` | the adaptive expert-swap cadence in decode rounds (`100000` = static, the reproducibility setting) |
| `suffix_draft` | prompt-lookup draft depth, `0` = MTP only (**capped at the startup depth**) |
| `mtp_max_t` | the MTP's window cap, re-issued through `mtp.set_max_drafts()` |
| `pcie_frac`, `spec_min_p` | the PCIe share of a missed expert; the draft-probability floor |

**Not** in this class, deliberately: `--spec` (the verify geometry is built from it), `--max-context`/`--kv`
(session state), the expert tiers (allocated), and every `--native-*`/graph flag (baked into captured graphs).
Those are the rebuild class — `warm-retune/RETUNE-CANDIDATES.md` says which is which and why.

```sh
curl -s -X POST localhost:8080/props -H 'Content-Type: application/json' \
     -d '{"strata_tune": {"prefill": 4096, "adapt_every": 100000}}'
```

## 3 — The API key's scope, and who may skip it

Upstream 0.1.38 sets `api_key` and then requires it from **every** caller.  **That is this fork's default**, so
nothing changes unless you ask: the exemption is chosen when the server starts and can be changed while it runs.

| `api_key_scope` | who skips the key |
|---|---|
| **`all` (default)** | nobody — the key is required from every caller, this PC included (upstream's behaviour) |
| `lan` | this PC **and the local network** — `10/8`, `172.16/12`, `192.168/16`, link-local, IPv6 `fc00::/7` and `fe80::/10` |
| `localhost` | this PC only |
| `off` | nobody is asked at all (the check is off) |

`"api_key_allow": ["10.1.2.0/24", "192.168.4.7"]` exempts named addresses and netblocks on top of the scope —
under the default scope it is the only thing that exempts anyone.  An explicit netblock is parsed strictly (a
host inside a network is a typo, refused), a bare address is a /32 (/128 in IPv6), and an unreadable peer address
needs the key (fail closed).  Carrier-grade NAT (`100.64/10`) is deliberately *not* "your network".  Chosen at
invocation — `--api-key-scope` / `--api-key-allow`, `"api_key_scope"` / `"api_key_allow"` in the config, or
`$STRATA_API_KEY_SCOPE` / `$STRATA_API_KEY_ALLOW` — and **changeable while it runs**:

```sh
./serve/server.py --api-key-scope lan --api-key-allow 10.1.0.0/16   # the exemption, opted into at invocation
curl -s -X POST localhost:8080/props -H 'Content-Type: application/json' \
     -d '{"api_key_scope": "localhost", "api_key_allow": ["10.1.0.0/16"]}'
curl -s -X POST localhost:8080/props -H 'Content-Type: application/json' -d '{"api_key": ""}'
```

Two rules make it safe to hand out exemptions (they only bite once one is in play):

* **Changing the policy needs the key when one is set** — `POST /props` ignores the scope for this, so an
  exempt LAN client cannot turn its own exemption off for everybody.
* **The DNS-rebinding `Host` check and the cross-site `Origin` check follow the request's authentication**,
  not the existence of a key.  A request that carries the key skips them (upstream's tunnel/proxy behaviour,
  kept); one that does not keeps them **even from an address the scope exempts** — because a rebinding page
  arrives from `127.0.0.1`, which scope `lan` exempts.  Skipping the check for exempt callers would re-open the
  hole the check exists to close, so it does not.

## 4 — No browser tab on start

Setup writes `--open` into every launcher, so a start pops the web app in a browser.  On a kiosk, a headless box
or a remote session that window lands where it should not:

```sh
serve/server.py --no-open                  # invocation-time only, deliberately not retunable
STRATA_NO_BROWSER=1 ./run-coder-iq1_m.sh   # beats --open in either order, needs no edit of a launcher
```

The address is still printed, so nothing is hidden.

## 5 — The AVX1 floor

Upstream cannot start on a CPU without AVX2 + FMA/F16C: its CPU expert kernels are AVX2 at least, and the
released ggml-cpu is compiled for the *build host*, so an AVX-only machine gets an illegal instruction instead
of an error message.  `STRATA_ISA_FLOOR=1` compiles ggml-cpu once for SSE4.2+AVX and relaxes the startup gate to
"AVX2 with FMA/F16C **or** AVX1"; the router's lookahead gets an AVX1 kernel (`bf16_rows_dot_multi_avx1`,
264–489× the scalar fallback it replaced) and the `iq_avx2` sign table became `constexpr` (its runtime
constructor had been vectorised into AVX-2 and ran before `main`).

Because `STRATA_ISA_FLOOR` is read by `CMakeLists.txt` but is **not** declared as a CMake `option()`, a fresh
build directory silently builds build-host-native; `setup.py` here passes `-DSTRATA_ISA_FLOOR=ON` explicitly
when `STRATA_ISA_FLOOR=1` is set, and a hand-run cmake needs it too.  The floor is ggml-cpu-only — Strata's own
AVX2/AVX-512 kernel units keep their ISA and runtime dispatch.

## 6 — setup.py behaviours the fork adds

* `get_llama_cpp()` runs `warm-retune/apply.sh apply` once the vendored tree exists (a no-op when it is absent
  or already patched; a failure warns instead of aborting the setup).
* The zip extraction drops `tools/ui` **on Windows only** (upstream drops it everywhere for the 260-character
  path limit, #206) — `tools/CMakeLists.txt` adds it unconditionally when `LLAMA_BUILD_SERVER` is on, so
  without it no `llama-server` can even be configured from the vendored tree on Linux.

## 8 — Tool calls can be gated per request (`tool_choice`)

Upstream accepted `tool_choice` and **ignored** it — a *fails-open* defect: an agent loop with a side-effecting tool
acts on a turn the caller believed was read-only (minefield trap 78). The fork implements it, and the enforcement is
structural rather than a plea to the template:

| `tool_choice` | what the fork does |
|---|---|
| absent / `"auto"` | offered as sent (unchanged) |
| **`"none"`** | the tools payload is **not sent to the engine at all** — and the MCP tools are not even collected — so no template, parser or model state can call one |
| `{"type": "function", "function": {"name": "X"}}` (Anthropic `{"type": "tool", "name": "X"}`) | only `X` is offered, so nothing else can be called; a name the request does not offer is a **400** |
| `"required"` (Anthropic `{"type": "any"}`) | offered as sent, and reported **`applied: false`** with the reason: no chat template here can force a call |
| anything else | **400**, because a choice this server cannot honour must not read as one it did |

## 9 — The request surface is named, and every response says what ran

Two halves of minefield trap 77 — a `200` used to confirm nothing:

* **Unknown fields are named.** Anything outside the API's field set is printed once per server lifetime (with the
  API it arrived on), so a typo is visible instead of silently doing nothing. `"strict_params": true` makes it a
  **400** listing the fields, for deployments that would rather fail loudly.
* **Every response carries the effective settings.** A top-level `"strata"` block (and the first chunk of a stream)
  with `thinking`, the effective `max_tokens`, `reasoning_budget_tokens` when thinking, `tools_offered` after
  `tool_choice`, the applied `tool_choice` report, `sampling` in the engine's own spelling, and `cap_hit:
  "reasoning"` when a reply spent its whole budget thinking and so has no answer (trap 12 — bucket those before
  scoring, or you are measuring the budget). Additive and namespaced: existing clients ignore it.

## 10 — History without an empty thinking block

Trap 04/25: when a conversation came back with an assistant turn whose reasoning was not resent, the template wrote
`<think>\n\n</think>` into the prompt. That nudges the model to skip its reasoning on later turns, and makes two
histories that should render identically differ — a conversation-cache miss. The fork writes the wrapper only when
there is **reasoning to preserve**; `"preserve_empty_think": true` (config, a request's `chat_template_kwargs`, or
`POST /props`) restores the checkpoint template's rendering exactly — verified byte-for-byte on all 10 golden cases,
3 of which this changes. Real reasoning is preserved either way.

**Where the fix has to land matters here.** The server renders the **pack's** `tokenizer/chat_template.jinja` in
preference to the repo's `serve/chat_template.jinja`, and that pack copy is extracted from the GGUF's own metadata
by `tools/strata_tokenizer.py` — so editing the repo's template alone never reaches a running deployment. Hence:

* `tools/empty_think_guard.py` — the guard as an idempotent, revertible, `--dry-run`-able replacement that leaves a
  template it does not recognise exactly as the model shipped it;
* `tools/strata_tokenizer.py` applies it as each pack is written (new packs carry it, and the tool's output is
  byte-identical to `serve/chat_template.jinja` — a test asserts that, so the two cannot drift apart);
* `~/bin/strata-fix-pack-template.sh [--undo|--check]` applies it to packs built before the fix, one backup per
  template under `~/.backup/files/pack-template/`, with the restore command printed.

Both new settings can be changed while the server runs, like the API-key policy.

## 11 — The retune audit (documentation, not code)

`warm-retune/RETUNE-CANDIDATES.md` is the source-level audit behind feature 2: which parameters are read when
they are *used* (changeable while it runs — the nine keys above, plus the ones still unwired: `--turn-token`, the PLE
I/O settings, the server-side `idle_unload_s`/`min_free_vram_mib`/reasoning budget), which are consumed once
(session state, expert arena, CPU pool — the rebuild class), and which are baked into captured CUDA graphs and
can never move.  It exists so the next person does not re-derive it, and so an end user with a use case we did
not imagine can see what is cheap to add.

## What the fork does NOT change

Everything else is upstream's behaviour, including the defaults a user already relies on.  The API key is one of
them: **with no flags or config, this server behaves exactly as 0.1.38 did** — the key is required from every
caller — and upstream's own test files are byte-identical here, which is the check that keeps it that way.  These
are the changes an existing user can actually meet:

| change | why | how to get upstream's behaviour back |
|---|---|---|
| With `api_key` set, the **`Host`/`Origin` checks are skipped only for a request that presents the key**, not for every request | an address exemption would otherwise re-open the DNS-rebinding hole (a rebinding page arrives from an exempt `127.0.0.1`) | a tunnel that passes its own name on should send the key, or be listed in `allowed_hosts` |
| **A prior assistant turn with no reasoning renders without the empty `<think></think>` wrapper** | the empty block nudges thinking collapse and costs the conversation cache (minefield 04/25) | `"preserve_empty_think": true` |
| **`tool_choice: "none"` actually gates the turn** where upstream ignored it | ignoring it fails *open* (minefield 78) | nothing needed — an absent `tool_choice` behaves exactly as before |
| **A `tool_choice` this server cannot honour is a 400**, not a silent "auto" (upstream logs it and lets the model decide) | a typo must not read as a choice that was applied | send a value upstream accepts: `"auto"`, `"none"`, `"required"`, or a function name the request actually offers |

## File map

```
warm-retune/retune-llama-server-3cf0325.patch        feature 1 — the vendored llama.cpp patch (1107 lines)
warm-retune/apply.sh                                 feature 1 — apply | undo | check, idempotent
warm-retune/RETUNE-CANDIDATES.md                     feature 7 — the audit behind feature 2
src/program/generate.cpp                             feature 2 — the TUNE line and its handler
serve/server.py                                      features 2,3,4 — POST /props, the scope, --no-open
CMakeLists.txt                                       feature 5 — upstream's since v0.1.39 (this fork's origin)
src/kernels/cpu/kq_avx1.cpp + kq_avx1.hpp            feature 5 — upstream's since v0.1.39 (this fork's origin)
src/kernels/cpu/expert_layout.cpp + .hpp,
src/kernels/cpu/native_expert.cpp,
src/kernels/cpu/iq_avx2.cpp,
src/core/expert_source.cpp,
src/kernels/native_expert_parity.cpp                 feature 5 — the AVX1 expert-row path and dispatch
setup.py                                             features 5,6
tools/empty_think_guard.py                           feature 10 — the guard a PACK's template gets
tools/strata_tokenizer.py                            feature 10 — applies it as each pack is written
serve/test_security.py                               47 tests: scope, allow list, POST /props, the checks' rules
serve/test_minefield.py                              45 tests: one class per trap (78, 77, 12, 04/25), named after it
warm-retune/MINEFIELD-FINDINGS.md                    the doctor's findings, the fixes, and the retest
warm-retune/NVME-VS-HDD.md                           the storage A/B (cold start + the bench table)
WARM-RETUNABLE.md                                    per-feature measurements and caveats
```

Ops scripts (not in the repo, they drive a deployment):
`~/bin/strata-fix-pack-template.sh` (the pack migration), `~/bin/strata-minefield-retest.sh` (the retest),
`~/bin/strata-arm-run.sh` + `~/bin/strata-arm-diff.py` (the storage A/B).

## Verification status

* **Features 2–4, 8–10, on the v0.1.41 base**: **667 tests green, 11 skipped**, which is the WHOLE `serve/`
  suite (`python -m unittest discover -s serve -p 'test_*.py'`): test_security 47, test_minefield 45, test_server
  (upstream's own, ~460) and the rest.  The previously-noted `test_detok` errors are gone on this base.
  **Upstream's own test files are upstream's**: `serve/test_server.py` is upstream's file plus this fork's
  `NoBrowser` class and **three adjusted methods**, each naming the fork behaviour it disagrees with —
  `ForcedToolChoice.test_values_it_cannot_honour_are_a_400_here` (upstream logs and acts as "auto"),
  `ForcedToolChoice.test_anthropic_any_tool_none` (a name the request does not offer: 400 here, "auto" there) and
  `LiteralThinkTags.test_a_client_that_sends_the_reasoning_inline` (upstream counts the empty `<think></think>`
  wrapper's markers; this fork writes no wrapper unless asked, Minefield 04/25).  Every other file of upstream's
  test set is byte-identical to the release.  Features 8–10 also carry the upstream minefield doctor's own probes — 77,
  78 and 04/25 come back **clean** on the retest, and the two that remain (12, 21) are properties of the lane and
  the checkpoint, reported and remediable as `warm-retune/MINEFIELD-FINDINGS.md` sets out.  The engine half of
  feature 2 compiles (`g++ -fsyntax-only … 0 errors`) but has **not been run** on a pack yet.
* **Feature 1**: measured on an RX 7900 XTX (values quoted in `WARM-RETUNABLE.md`); the patch's content is
  identical to the original it was forward-ported from.
* **Feature 5**: measured on a Xeon E5-2687W and on gfx1100; the floor's cost is a *null result* on gfx1100 and
  ~12% decode on a tighter controlled test — treat "no cost" as unmeasured.
* **Feature 6**: exercised by every setup run; the vendored tree it patches is gitignored and re-extracted by
  setup, which is exactly why the patch ships as a patch.
* **The v0.1.41 sync**: the engine **builds and loads** on gfx1100 with `STRATA_ISA_FLOOR=avx`
  (`engine/BUILD.json`, version 0.1.41, source hash `ae92017307778aa2`).  It has **not been run on a GPU** from
  this tree — the machine that carried the sync has no AMD card — so the fork makes **no throughput claim** for
  this build; the floor's own measurements above are the only engine-level numbers it has.

## Licence

Upstream Strata is MIT (see `LICENSE`); llama.cpp is MIT, and feature 1 is a derivative of llama.cpp under the
same terms.
