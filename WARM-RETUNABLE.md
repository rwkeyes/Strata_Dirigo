# Strata v0.1.41 — warm-retunable + API-key scope + AVX1 floor

A fork of [Strata](https://github.com/Niko1221/Strata) **v0.1.41** carrying what upstream does not.
**The AVX1 floor below is upstream's own since v0.1.39** (contributed from this fork, credited in its
`CMakeLists.txt`; upstream's spelling is `STRATA_ISA_FLOOR=avx`) — the section is kept for the measurements.
The rest is still this fork's:

1. a vendored-llama.cpp **runtime retune** patch — change a running llama-server's slot count, per-slot
   context and KV pool **without reloading the weights**,
2. the same idea for **Strata's own engine**: a `TUNE` line on its serve protocol and a `POST /props` that
   stores defaults, so the settings that are read per request can be changed under a running server,
3. an **API-key scope** — by default the key is required from everyone, exactly as upstream; an exemption
   (this PC, or this PC and the local network) is opt-in when the server starts (`--api-key-scope`,
   `--api-key-allow`, the config keys or the environment) and retunable while it runs, turning the security
   feature off or listing specific addresses/netblocks on an ALLOW list,
4. **`--no-open` / `$STRATA_NO_BROWSER`** — a start that must not pop a browser tab (kiosk, headless, remote),
5. an **AVX1 floor** build — SSE4.2 + AVX, `STRATA_ISA_FLOOR=ON` — so the engine also runs on an AVX-only
   CPU (a Sandy Bridge Xeon), which upstream's AVX2 baseline refuses with an illegal instruction
   (see the AVX1 floor section below).

Upstream's vendored llama.cpp (`3cf0325`, the revision v0.1.38 still pins) answers `POST /props` with a stub —
`{"success": true}` and nothing else. With this patch it actually changes a running server's sizes:

| request | effect |
|---|---|
| `{"parallel": N, "ctx_per_slot": T}` | divide the unified KV pool across N slots, weights stay loaded |
| `{"ctx": N}`, `{"cache_type_k": "q8_0"}`, `{"flash_attn": "on"}`, `{"parallel_max": N}`, `{"kv_unified": true}` | rebuild the context without restarting (the weights stay loaded) |
| `{"allow_oversubscribe": true}` | let slots share a pool that cannot back them all |

Measured on an RX 7900 XTX serving a 35B-A3B MoE (Vulkan, KV q8_0):

* dividing the pool differently, with no restart: **0–0.4 ms**
* context rebuild (pool 32k → 262k): **~0.8 s**, weights never reloaded
* re-slicing *while other slots are decoding*: 0.35 ms
* a request that would evict a busy slot: **HTTP 400 `slot N is busy, cannot take it out of service`**
  (never a hang)
* `parallel_max` rebuild with 4 slots reserved: 814 ms

## The AVX1 floor (SSE4.2 + AVX)

Upstream cannot start on a CPU without AVX2 + FMA/F16C: the CPU expert kernels are AVX2 at least, and the
released ggml-cpu is compiled for the *build host*, so such a machine gets an illegal instruction instead of
an error message. With `STRATA_ISA_FLOOR=ON` this fork compiles ggml-cpu **once** for ggml's own
`sandybridge` feature set (SSE4.2 + AVX — no FMA, no F16C, no AVX2) and relaxes the startup gate to
"AVX2 with FMA/F16C, **or** AVX1". Measured on the machine this exists for: a Xeon E5-2687W (AVX only)
*started* fine and then died in `bf16_rows_dot_multi+0x1d9` (`vpmovzxwd`) on the **first request** — the
router lookahead calls an `-mavx2` translation unit before anything else runs, so that call now has a scalar
fallback, and the AVX2 sign table in `iq_avx2.cpp` is `constexpr` (its runtime constructor had been
vectorised into AVX-2 and ran before `main`).

```sh
STRATA_ISA_FLOOR=1 ./setup.sh --backend hip --family qwen --model IQ3_XXS --context 32768 \
    --kv int8 --gguf-dir /path/to/the/two/shards --build --no-start --yes
```

**`STRATA_ISA_FLOOR` is not a CMake `option()`** in this tree — it is only read by `if(STRATA_ISA_FLOOR)`,
so on a fresh build directory it is undefined (OFF) and you silently get a build-host-native engine. That is
why `setup.py` here passes `-DSTRATA_ISA_FLOOR=ON` explicitly when `STRATA_ISA_FLOOR=1` is set (see
`isa_floor_defs()`), and why a hand-run cmake needs the same flag. Leaving it unset breaks nothing — you
simply never get the floor.

**What it costs:** the floor is ggml-cpu-only. Strata's own AVX2/AVX-512 kernel units (`iq_avx2.cpp`,
`iq_avx512.cpp`) keep their ISA and their run-time dispatch; only the ggml-cpu tier moves down. Our own A/B
on gfx1100 — the same binary on the forced-AVX1 path versus its native AVX-512 path — was a **null result**
(trial-to-trial spread ~5×, because first-touch reads of a 55 GB pack from a rotational drive dominate), and
a tighter controlled test measured decode **21.8 → 19.1 tok/s (~12%) with byte-identical output**. So treat
"no cost" as unmeasured, not proven: the floor's reason for existing is that an AVX-only CPU otherwise
cannot run at all. Both raw arms and the caveats are in `bench/results/2026-10-01-avx1-floor/`; the curated
entry is `docs/benchmarks/2026-10-01-gfx1100-avx1-floor.json`.

**The router dot — the reason the floor is usable at all.** The lookahead that warms the file tier runs one
router dot per layer, and the floor's first version did it scalar with `std::fma` — which on a CPU with no
FMA instruction is a **libm call per element**, not an instruction. Measured on the Xeon E5-2665 this fork
exists for, one layer's router `[512 experts x 5120 embd]`:

| | ms/layer | per 48-layer lookahead pass | vs scalar `std::fma` |
|---|---|---|---|
| scalar `std::fma` (the first version) | 214.3 | 10.3 s | 1x |
| scalar mul+add | 3.33 | 160 ms | 64x |
| **`bf16_rows_dot_multi_avx1`** (this fork) | **0.81** | **39 ms** | **264x** |

For a 4- or 6-token window it is 1.91 ms / 2.63 ms per layer (450x / 489x, 92 ms / 126 ms per pass), and
the kernel agrees with the scalar reference to `max |diff| = 1.1e-4` on values ~56 (the mul+add vs FMA
rounding). Ten seconds to forty milliseconds per pass is the difference between a floor build that serves
and one that does not. The remaining AVX1 cost is the expert rows themselves: `native_gu_rows` falls
through to ggml-cpu's **single-token** `vec_dot`, so a 4-6 token verify window re-reads each weight row
4-6 times — a multi-token AVX1 kernel is the next win, and a real port (AVX1 has neither FMA nor 256-bit
integer ops).

**Pre-existing, not this fork:** the project's own GPU-free setup tests have two failures on this tree
(`tools/test_setup_amd.py` 1, `tools/test_setup_golden.py` 46) with identical counts on upstream v0.1.37
without the fork work.

## The engine's own retune (`TUNE` + `POST /props`)

The same idea inside Strata: the engine is a long-running process fed one request per line on stdin
(`GEN <max_new> key=value … <ids>`), and those keys already carried per-request tuning (`pcie_frac`,
`spec_min_p`, the sampling keys — the code calls them "tuning keys … without restarting the engine", which
is where the setup's calibration measures from). This fork adds:

* **`TUNE key=value …`** — a line the engine applies to its live option struct and answers
  `TUNED <applied> refused:<...>`; it is handled between requests, before the request's busy scope.
* **`POST /props`** (server, JSON from Strata's own page) with `{"strata_tune": {...}}`, stored as the default
  every later request carries; a request's own `strata_tune` wins, `null` drops a key. `GET /props` reports
  the current values. The server sends a `TUNE` line only when the values differ from what the engine has.
* Both ends validate: `TUNE_KEYS` in `serve/server.py` answers **400** for an unknown key or an out-of-range
  value (a typo must not look applied), and the engine refuses what it does not know.

| key | what it changes |
|---|---|
| `prefill` | the prompt-path chunk, and so how much of the expert cache a prompt borrows. Down any time; **up only to the startup chunk**, because the prompt buffers were sized from it at load |
| `short_read` | how many fresh tokens are read through the decode windows instead of the batched prompt path |
| `prompt_cache`, `prompt_cache_every` | conversation checkpoints kept between requests, and how often one is taken |
| `adapt_every` | the adaptive expert-swap cadence, in decode rounds (also the reproducibility setting: `100000` = static) |
| `suffix_draft` | prompt-lookup draft depth, `0` = MTP only (**capped at the startup depth**) |
| `mtp_max_t` | the MTP's window cap — re-issued through `mtp.set_max_drafts()`, a plain field write |
| `pcie_frac`, `spec_min_p` | the PCIe share of a missed expert, the draft-probability floor |

Nothing is reloaded, no VRAM moves and no session state is dropped: these are read out of the engine's option
struct when they are *used*, so writing the field is the whole change. Deliberately not in this class:
`--spec` (the verify geometry is built from it), `--max-context`/`--kv` (session state), the expert tiers
(allocated), and every `--native-*`/graph flag (baked into captured graphs — see `session.h`'s "a graph bakes
in its arguments"). `warm-retune/RETUNE-CANDIDATES.md` is the full audit, including that rebuild class and
what an end user might still want from it.

## The API key's scope

Upstream 0.1.38 sets `api_key` and then requires it from **every** caller. **That is this fork's default too:**
nothing changes unless you ask for an exemption, and you ask for it when the server starts — `--api-key-scope`
(`--api-key-allow` for named addresses), or `"api_key_scope"` / `"api_key_allow"` in the config, or
`$STRATA_API_KEY_SCOPE` / `$STRATA_API_KEY_ALLOW`.

| `api_key_scope` | who skips the key |
|---|---|
| `all` (**the default**) | nobody — the key is required from every caller, this PC included (0.1.38's behaviour) |
| `lan` | this PC **and the local network**: `10/8`, `172.16/12`, `192.168/16`, link-local, IPv6's `fc00::/7` and `fe80::/10` |
| `localhost` | this PC only |
| `off` | nobody is asked for a key at all (the feature is off) |

`"api_key_allow": ["10.1.2.0/24", "192.168.4.7"]` (also `--api-key-allow`, `$STRATA_API_KEY_ALLOW`) exempts
named addresses and netblocks on top of the scope; `lan` and `localhost` honour it, `all` does not — under the
default scope it is the only thing that exempts anyone. Carrier NAT (`100.64/10`) is deliberately *not* "your
network". An unreadable peer address needs the key (fail closed). An unknown scope is refused at invocation
(argparse) and a bad value in POST /props answers 400. **Change either while it runs:**

```sh
./serve/server.py --api-key-scope lan --api-key-allow 10.1.0.0/16   # chosen at invocation
curl -s localhost:8080/props | jq '{api_key, api_key_scope, api_key_allow}'
curl -s -X POST localhost:8080/props -d '{"api_key_scope": "localhost", "api_key_allow": ["10.1.0.0/16"]}'
curl -s -X POST localhost:8080/props -d '{"api_key_scope": "off"}'      # the check off, for the whole server
curl -s -X POST localhost:8080/props -d '{"api_key": ""}'               # no key at all
```

**Two behaviours change, on purpose, and only once an exemption is in play:**

* **Changing the policy needs the key when one is set** — `POST /props` ignores the scope for this, so an
  exempt LAN client cannot switch its own exemption off for everybody.
* **The DNS-rebinding `Host` check now follows the key, not the setting of one.** Upstream turned the check
  off for everyone as soon as `api_key` was set. If it stayed that way, scope `lan` would exempt `127.0.0.1`
  from the key *and* switch off the check that exists for pages arriving from `127.0.0.1` — the hole the check
  is for. So: a request that **carries** the key skips the check (upstream's behaviour for tunnels and
  proxies, kept), and a request that does not carry it keeps the check, exempt address or not. A tunnel that
  passes its own name on should therefore send the key, or have its name in `allowed_hosts`.
  `serve/test_security.py` has both cases as tests.

## No browser tab on start

Setup writes `--open` into every launcher, so a start pops the web app in a browser — on a kiosk, a headless
box or a remote session that window goes somewhere it should not. `--no-open` on the server's arguments (or
`STRATA_NO_BROWSER=1`, which beats `--open` whatever the order and needs no edit of a launcher setup wrote)
suppresses it; the address is still printed. Invocation-time only, deliberately not retunable.

## What this fork changes

1. `warm-retune/retune-llama-server-3cf0325.patch` and `warm-retune/apply.sh` — the llama.cpp patch, plus an
   idempotent `apply` / `undo` / `check` script (undo reverse-applies the patch; nothing else is needed
   because the vendored tree is regenerated by setup).
2. `warm-retune/RETUNE-CANDIDATES.md` — the audit behind the engine retune: what can change while it runs, what needs a
   rebuild, what can never move.
3. `setup.py` — three changes, all inert on a checkout with no vendored tree yet:
   * `get_llama_cpp()` calls `apply_warm_retune()` once the tree is present (freshly extracted or
     already on disk), so a setup run leaves the tree warm-retunable; a failure warns instead of
     aborting the setup.
   * the extraction no longer drops `tools/ui` **on non-Windows platforms**. Upstream excludes it for
     Windows' 260-character path limit (#206), but `tools/CMakeLists.txt` adds it unconditionally when
     `LLAMA_BUILD_SERVER` is on — without it, no llama-server can even be configured from this tree.
   * `isa_floor_defs()` adds `-DSTRATA_ISA_FLOOR=ON` to a local HIP/CUDA build when `STRATA_ISA_FLOOR=1`.
4. `src/program/generate.cpp` — the `TUNE` line and its handler (class 1 keys, with the two ceilings).
5. `serve/server.py` — `TUNE_KEYS` + `clean_tune`/`tune_str`, `POST /props`, the API-key scope and allow
   list (`key_needed_for`, `set_api_key_policy`), the `Host`/`Origin`-check rule above, `--no-open` /
   `browser_suppressed` / `open_browser`; `serve/test_security.py`, `serve/test_server.py` and the other
   suites carry the tests.
6. the AVX1 floor itself: the `STRATA_ISA_FLOOR` block in `CMakeLists.txt`, the relaxed startup gate plus
   the scalar router fallback (`src/program/generate.cpp`, `src/core/expert_source.cpp`), the `constexpr`
   sign table and the AVX1 expert-row path (`src/kernels/cpu/`). Measured on a Xeon E5-2687W and on
   gfx1100 — see the section above.
7. `src/kernels/cpu/kq_avx1.cpp` (+ `kq_avx1.hpp`), built with `-mavx -msse4.2`: the AVX1 router dot, 264x
   to 489x faster than the scalar fallback on the CPU this floor is for (see the table above). It is
   dispatched from `expert_source.cpp` only when the CPU has AVX1 but not AVX2, so AVX2/AVX-512 hosts keep
   the existing `-mavx2` kernel, and the scalar path that remains (no AVX at all) no longer uses
   `std::fma`.

## Verification status (be exact about this)

* `serve/test_security.py` (45 tests) — the scope matrix, the allow list, `POST /props` incl. the
  key-required-for-policy rule, the rebinding/cross-site regression with an exempt caller, the `TUNE` line's
  format and the server's "only when it changed" logic — all against the mock engine, no GPU.
* `serve/test_server.py` (121), `test_lifecycle` (8), `test_mcp` (25), `test_monitor` (7),
  `test_structured` (8) — green. The 3 errors in `serve/test_detok.py` are pre-existing on v0.1.38 (identical
  on the untouched tree).
* `src/program/generate.cpp` — **compiles** (`g++ -fsyntax-only -std=c++20 -D__HIP_PLATFORM_AMD__
  -Iinclude -Iinclude/strata/hip_compat -I/opt/rocm/include …`, 0 errors). Not linked, and **not yet run**: a
  live `TUNE` round trip needs the HIP engine built and a pack loaded (the 7900 XTX was serving another model
  while this was written, and the box was at 16/16 GB swap). Until that is done, treat the engine-side keys
  as compile-verified only.

## Use it

```sh
./setup.sh --family coder --model IQ1_M --context 65536 --kv int8 --backend hip \
           --gguf-dir /path/to/the/two/shards --build --no-start --yes
```

`setup.py` in this fork applies the patch automatically right after it extracts
`third_party/llama.cpp` (it runs `warm-retune/apply.sh`; the step is a no-op when the patch is already
present or the tree is absent, and it never fails the setup). To do it by hand:

```sh
warm-retune/apply.sh check     # is it present?
warm-retune/apply.sh apply     # idempotent
warm-retune/apply.sh undo      # reverse-applies the patch
```

### Build a server from the vendored tree

Strata's own build compiles ggml/gguf/mtmd — **not `tools/server`** — so the patch is inert until you
build a server yourself:

```sh
cmake -G Ninja -B build-llamacpp -S third_party/llama.cpp -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DGGML_VULKAN=ON          # or -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build-llamacpp -j8 --target llama-server
```

**Known wart:** the vendored tree is *pruned* — `tools/ui/` is missing, and `tools/CMakeLists.txt`
adds it unconditionally when `LLAMA_BUILD_SERVER` is on, so the configure step above fails with
`add_subdirectory given source "ui" which is not an existing directory` until you restore it from the
same upstream revision:

```sh
cp -r /path/to/a/full/llama.cpp-at-the-pinned-commit/tools/ui third_party/llama.cpp/tools/
```

Then serve it with the retune endpoint enabled:

```sh
llama-server -m model.gguf -ngl 99 -fa on --kv-unified --props -np 4 -c 131072 -a my-model
curl -s localhost:8080/props | jq '{total_slots, slots_max, kv_pool_n_ctx, kv_unified}'
curl -s -X POST localhost:8080/props -d '{"parallel":2,"ctx_per_slot":65536}'
```

`--props` is what enables the write path; `--kv-unified` makes the slot count a server-side policy
instead of a load-time property.

## The patch

`warm-retune/retune-llama-server-3cf0325.patch` (1107 lines, 6 files, 626 added):

* `tools/server/server-context.cpp` — the `SERVER_TASK_TYPE_RECONFIGURE` task, the slot policy
  changed without a restart, the context-rebuild path, the busy-slot refusal and the lazy KV release
* `tools/server/server-context.{h,server-task.h}` — the runtime-state struct and task/result types
* `common/common.{cpp,h}` — `reload_context()` plus a threadpool reset/init so a rebuild reuses the
  process instead of restarting it
* `tools/server/README.md` — `POST /props` documented

It targets llama.cpp `3cf0325` (the revision v0.1.37 pins — unchanged since v0.1.31) and was forward-ported from a tree built
against `7fe450e19` by `git apply --3way`; the added/removed content is **identical** to the original
(verified by diffing the two patches' content lines and by an independent audit).

## Licence

Upstream Strata is MIT (see `LICENSE`); llama.cpp is MIT. The patch is a derivative of llama.cpp and
carries the same terms.
