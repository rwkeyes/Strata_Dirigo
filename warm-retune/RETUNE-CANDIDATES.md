# Retune candidates — what a running Strata engine can have changed under it

An audit of the engine's parameters against the warm-retune idea, done for the AVX1/retunable fork (base:
v0.1.38 + `warm-retune/retune-llama-server-3cf0325.patch`).  It answers the question "slots and the KV cache
are retunable — what else could be?", and it is deliberately NOT filtered by use case: the point is that a
user may know a reason we do not.  Line numbers are from the v0.1.38 tree and may drift with a release; the
flag names and the read sites are what to grep.

## The two mechanisms a retune can ride

1. **The engine's stdin protocol** (`src/program/generate.cpp`, the `--serve` loop).  Requests arrive one per
   line, `GEN <max_new> key=value … <ids>`, and the loop answers `T …`, `PP …`, `DONE …`.  Two kinds of key
   already ride it:
   * per-request keys the server fills in: `temperature top_p top_k min_p penalty_* seed cvec pcie_frac
     spec_min_p` — the comment there calls them "tuning keys (setup's calibration measures settings without
     restarting the engine)", which is this fork's idea already in the tree.
   * a **`TUNE key=value …`** line (added here): the same keys, but persistent — the engine applies them to
     the live `Options` struct and answers `TUNED <applied> refused:<...>`.
2. **`POST /props`** (`serve/server.py`) — the server's side of the same thing: `{"strata_tune": {...}}` stores
   defaults that every later request carries (a request's own `strata_tune` wins), the API-key policy keys,
   and `GET /props` reports both.  `TUNE_KEYS` in `serve/server.py` is the validating table; the engine
   validates again, and refuses what it does not know.

## Class 1 — retunable in place (no restart, no state dropped, no VRAM moved)

These are read out of `Options` at the moment they are USED, and the per-request lambdas capture `o` by
reference, so writing the field is the whole change.  Implemented in this fork:

| key | what it changes | read at |
|---|---|---|
| `prefill` | the prompt-path chunk (and so how much of the expert cache a prompt borrows) | `request_chunk()` (v0.1.38: `:3993`) and the per-prompt lend plan |
| `short_read` | read at most N fresh tokens through the decode windows instead of the batched path | the request's prompt read |
| `prompt_cache` | conversation checkpoints kept between requests (its eviction loop trims to the new N) | `checkpoint_at()` lambda |
| `prompt_cache_every` | checkpoint cadence | the prompt-chunk callback |
| `adapt_every` | the adaptive expert-swap cadence, in decode rounds | the decode loop, `(rounds+1) % o.adapt_every` |
| `suffix_draft` | prompt-lookup draft depth (`0` = MTP only) | the per-round draft block |
| `mtp_max_t` | the MTP's window cap — re-issued through `mtp.set_max_drafts()` (a plain field write) | the draft policy |
| `pcie_frac` | the share of a missed expert's bytes moved host↔device | per-request default |
| `spec_min_p` | the draft-probability floor | per-request default |

Two of them have a ceiling and the engine says so in its reply (`capped at the startup chunk/depth`):

* `prefill` may shrink any time; it may grow only to the startup value, because the prompt-path buffers were
  sized from it at load (`160 + chunk*680/1024` MiB, plus the slots a chunk borrows).
* `suffix_draft` likewise — the `SuffixDrafter`'s slots were sized at startup.

**Still Class 1 in principle, not wired yet** (each is a read of `o` inside the loop or a per-request lambda;
the work is a key in `TUNE_KEYS` plus a branch in the engine's handler):

* `--turn-token ID` — the chat-turn boundary the checkpoint root is taken from (per request).
* `--expert-profile-save` / `-save-every` — checked between requests already (`--serve` re-saves on the timer).
* `--ple-row-cache N`, `--ple-inflight N` — the PLE row cache and read depth: built into the `PleIoOptions`
  at load, but it is a bounded ring, so a resize is contained rather than structural.
* `--resident-pin`, `--resident-headroom`, `--adapt-swaps` — the RAM-tier placement policy around the resident
  mode (the adaptive tier already swaps experts VRAM↔RAM at runtime; a retune only biases it).
* `--conversation-cache-min-free-mib` — read at *park* time.
* Server-side, no engine state at all: `idle_unload_s`, `min_free_vram_mib`, `before_load`, `fit_max_tokens`,
  `reasoning_budget_tokens`, the `sampling` defaults block.  These are ordinary Python attributes read per
  request, so they are the cheapest class-1 wins of all.

## Class 2 — the rebuild class (weights stay; state is recreated)

The analogue of the llama.cpp patch's `ctx` / `cache_type_*` / `kv_unified` group: each is consumed once, when
session state, the expert arena, or the CPU pool is created.  A rebuild in Strata today means an engine
restart (the server's unload/load path, seconds rather than the ~5-6 min cold start), because the engine has
no "recreate the session, keep the weights" path — that path is the missing piece the vendored patch adds for
`llama_context`.

| flag | consumed at | analogy |
|---|---|---|
| `--max-context` | session-state carve (`session_bytes()`) | `ctx` |
| `--kv fp16\|int8\|q4_0\|k8v4` | state layout | `cache_type_k` / `cache_type_v` |
| `--kv-resident N` | `qsa_set_kv_resident()` at load | `kv_unified` / KV offload |
| `--expert-cache N`, `-per-layer`, `--vram-reserve-mib`, `-device1..3`, `-remote-placement` | the VRAM arena | the "slots" of the GPU tier |
| `--resident-budget-gib`, `--resident-experts`, `--resident-cpu-experts`, `--mmap-experts`, `--shared-expert-arena` | the RAM/file tier | — |
| `--conversation-cache-mib` | arena sized at load (its *kept count* is class 1) | — |
| `--pool-workers N`, `--pool-affinity MODE` | `ExpertPool` construction | the vendored patch already resets/init a threadpool for its rebuild — the precedent exists |
| `--spec T` | `mtp.load(..., o.spec, ...)`; `S` and `DraftPolicy` are const/locals before the serve loop | draft geometry |
| `--rope-scaling`, `--rope-scale`, `--yarn-*` | the code says "no per-request form: K sits in the cache POST-RoPE" | exactly what a KV-dropping rebuild satisfies — so this belongs here, not in "never" |
| `--ple-io`, `--split-*`, `--layer-split`, `--no-pool`, `--no-prefill-borrow`, `--no-split-rows` | load-time layout | — |

## Class 3 — not retunable without a rebuild of the binary's graph set

Every `--native-*`, `--gr-*`, `--stream-token`, `--no-capture`, `--no-token-graph`, `--no-fused-gr`,
`--no-hit-poke`, `--no-ple-prefetch`, `--no-host-worker`, `--shared-late`, `--keep-canonical`,
`--cpu-oracle-q8-0`, `--vision`.  Reason, from `include/strata/core/session.hpp`: *"A graph bakes in its
arguments … everything per-token arrives through fixed-address device buffers."*  Per-token values are dynamic
by construction; everything else is fixed when the graph is captured.

## Notes for whoever extends this

* **Both ends validate.**  `TUNE_KEYS` in `serve/server.py` refuses a bad value with a 400 (a typo must not
  look applied); the engine refuses unknown keys and non-numeric values in its `TUNED` reply alongside the
  ones it did apply, so a mismatch is visible rather than silent.
* **`clean_tune(strict=False)`** is the lenient path used for a *request's* own keys: a bad one is dropped so
  a request is never failed by its tuning.  `strict=True` (POST /props) is the one that reports.
* **A TUNE line is only handled between requests** (before the request's busy scope), so nothing races the
  decode loop — that is also why `mtp.set_max_drafts()` can be re-issued safely.
* **`--spec` is deliberately not retunable** in place: `S` is a const local and the verify geometry is built
  from it.  Changing it needs the rebuild class.
* The measurement harness for anything here is `tools/hip/bench_prefill.py` plus a restarted engine per arm
  (see the `strata-llm-engine` skill); per-request rates come from the server's journal, not the engine log.
