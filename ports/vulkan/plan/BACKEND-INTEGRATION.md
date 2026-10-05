# Backend integration — the engine side of the Vulkan port (HANDOFF §4.2)

Written 2026-10-05 on `vega`, HEAD `f878e34`, branch `vulkan-arc-port`, tree clean.  Milestone **M-A is closed**:
`PORT-MAP.tsv`'s `todo` column is zero, so every GPU entry point the decode path reaches is ported and gated.
This document is the *engine-side* plan — milestone M-B's precondition — and it is deliberately **not** a build.
Increment 0 (the skeleton below) is the only thing that has landed with it: the macro, the tree, and one entry
point wired at the build-system level.  The estimate in §6 is what the next step is approved on.

---

## 1. Where the entry-point list comes from (not a guess)

`ports/vulkan/PORT-MAP.tsv` classifies **every** `kernels::` symbol `src/core/` calls, and
`ports/vulkan/tools/check_port_map.py` (run by `gates/run_gate.sh`) fails if the map drifts from the engine or
from the built shaders.  Parsed at HEAD:

| kind | rows | what it means here |
|---|---|---|
| `kernel` | **28** | GPU work with a ported shader — the backend must dispatch it |
| `host` | **49** | the engine's own host side — a backend swap mostly does not touch it |
| `todo` | **0** | M-A closed; nothing left unported on the decode path |
| **total** | **77** | every decode-path symbol |

The 28 `kernel` rows are the **backend surface**: each is an engine entry point (a thin wrapper in
`include/strata/kernels/*.hpp`) whose body launches a shader.  Several rows are *composite* — one entry point
that launches several shaders — so the 28 rows name **49 shader dispatches** (e.g. `native_mmvq` → six
`*_mmvq` shaders, `native_expert_grouped` → five, `ple_block` → four).

The 49 `host` rows split by where their body lives:

* **37 are pure host** — sizes (`*_scratch_bytes`, `iq_row_bytes`, `kv_*_bytes_per_*`, `qsa_step_bytes`),
  capability checks (`native_mmvq_supported`, `fused_gr_supported`, `native_gdn_enabled`,
  `native_qsa_indexer_enabled`, `embed_type_supported`), shapes (`qsa_real_shapes`, `qsa_selection_width`,
  `native_expert_layout`, `rope_scaling`), host bookkeeping (`ngram_rows`, `kv_stream_*`, `kv_ring_restore`,
  `penalty_rows`) and pure conversions (`f16_from_f32`, `f32_from_f16`).  A backend swap does not touch these.
* **12 live in a CUDA translation unit and cross the device runtime**, so they need a Vulkan home (verified by
  grepping their definitions — the file each one is defined in is given):

  | symbol | defined in | why it crosses the runtime |
  |---|---|---|
  | `copy_from_mapped` | `src/kernels/cuda/elementwise.cu` | mapped-pinned copy in the layer chain |
  | `copy_i32_from_mapped` | `src/kernels/cuda/elementwise.cu` | ditto (QSA step/positions) |
  | `copy_or_zero_from_mapped` | `src/kernels/cuda/verify_kernels.cu` | ditto (verify window) |
  | `doorbell_publish` | `src/kernels/cuda/elementwise.cu` | host/device handshake, **no Vulkan equivalent** |
  | `doorbell_ring` | `src/kernels/cuda/elementwise.cu` | ditto |
  | `doorbell_wait` | `src/kernels/cuda/elementwise.cu` | a kernel that SPINS on host memory |
  | `build_rope_table` | `src/kernels/cuda/rope.cu` | builds + uploads the RoPE table |
  | `rope_table_set` / `rope_table_release` | `src/kernels/cuda/native_rope.cu` | device-table upload/teardown |
  | `kv_ring_table` | `src/kernels/cuda/kv_stream.cu` | the streaming ring's table |
  | `kv_stream_reset` | `src/kernels/cuda/kv_stream.cu` | device-side reset |
  | `gr_workspace_init` | `src/kernels/cuda/gr.cu` | hands out a carve of the arena (pure carve; moves to host) |

The `doorbell_*` rows are the three the plan refuses to translate: `elementwise.cu`'s doorbell is a kernel that
spins until the host answers, ordered by `__threadfence_system()`, and **Vulkan has no equivalent** (PORT-PLAN
§2.3).  On the display card a hung compute kernel is a KMD timeout at best and a Battlemage wedge at worst, so
the Vulkan path must replace it with fences/timeline semaphores, never a waiting kernel.

**So the backend must implement 28 kernel entry points + 12 device-crossing host rows = 40 entry points,
dispatching 49 shaders.**  That is the number this plan prices.

Two entry points already exist and are easiest to over-count: `fwht256_inplace_cuda` is the wrapper for the
`fwht256` shader (its body calls `fwht256_cuda`), and `sample_tokens` chooses between the two samplers
(`sampler_greedy`, `sampler_kernel`), both already ported.

---

## 2. Backend selection — how it is wired (the file's own pattern)

Followed, not invented.  `CMakeLists.txt` already selects backends by a compile-time option with a
`message(WARNING "experimental")` and a mutual-exclusion `FATAL_ERROR` (`STRATA_ENABLE_SYCL`, the closest
precedent: a *separate* tree, `docs/INTEL_ARC.md`).  Increment 0 adds:

```cmake
option(STRATA_ENABLE_VULKAN "EXPERIMENTAL: build the Vulkan engine from vulkan/ ..." OFF)
if(STRATA_ENABLE_VULKAN)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR STRATA_ENABLE_SYCL)
    message(FATAL_ERROR "STRATA_ENABLE_VULKAN is a separate backend; leave CUDA, HIP and SYCL off")
  endif()
  message(WARNING "STRATA_ENABLE_VULKAN: the Vulkan backend is experimental ...")
  add_subdirectory(vulkan)
  return()
endif()
```

* **One macro, one name:** `STRATA_ENABLE_VULKAN`, defined on the backend target; a TU self-asserts it
  (`#if !defined(STRATA_ENABLE_VULKAN) #error`), so it cannot be built into the wrong backend.
* **Mutual exclusion** covers all three siblings (the SYCL block's check also gained `STRATA_ENABLE_VULKAN`,
  so either order of the two options errors).
* **Per-backend TU layout:** a separate tree `vulkan/`, mirroring `sycl/`, so nothing under `src/kernels/cuda/`
  is compiled.  Increment 0: `vulkan/CMakeLists.txt`, `vulkan/include/strata/vulkan/vk_backend.hpp` (the
  device-layer seam) and one TU, `vulkan/src/kernels/fwht_vk.cpp`.  Later increments add
  `vulkan/src/device/` (the device layer) and one TU per subsystem.
* **The engine's headers are not edited.**  Each entry point is the thin wrapper already in
  `include/strata/kernels/*.hpp` (`kv_q4.hpp`'s `fwht256_inplace_cuda(...) { fwht256_cuda(...); }`); the backend
  answers the symbol the wrapper calls.  Integration is "implement N symbols", not "edit N headers".

## 3. The increments (each with what proves it)

The port's own numeric gate (`gates/run_gate.sh`) proves a **shader**.  It says nothing about a **program**, and
that is where the surprises are.  So every increment's proof is split: the shader's existing case stays the
kernel proof, and a new integration case exercises the *engine entry point* through the backend.

### I1 — device layer + arena + the first entry point, end to end
* **Covers:** adopt `ports/vulkan/harness/vk_compute.*` as `vulkan/src/device/`; build the **arena** (the engine
  plans one buffer carved by byte offsets) and the **pointer→buffer resolution** the engine's raw device
  pointers require; the **pipeline cache keyed as the engine dispatches**; finish `strata::vulkan::fwht256`
  so the increment-0 entry point runs.
* **Files:** new `vulkan/src/device/*`, finished `vulkan/src/kernels/fwht_vk.cpp`; engine headers unchanged.
* **Proves:** a gate case that calls `strata::kernels::fwht256_cuda` (the engine wrapper, not the shader
  harness) and compares **bitwise** to the port's already-green `fwht256` case — plus a falsification that a
  wrong view offset changes the answer.
* **Risk (cannot predict):** the engine's pointer arithmetic beyond the known row slices (`X + t0*K`); whether
  `minStorageBufferOffsetAlignment` (measured 4 B on the Arc) holds for every slice the engine forms; aliasing
  between arena regions the gate never poses.

### I2 — the glue wave and the device-crossing host rows
* **Covers (11 kernel rows):** `add_inplace`, `embedding_gather`, `gather_rows`, `scatter_rows_f32`,
  `cvec_apply`, `scale_inplace`, `silu_inplace`, `f32_to_f16_bulk`, `f32_to_bf16_bulk`, `rms_norm_weighted`,
  `gdn_gate`.  **And the host rows:** the three mapped copies, `build_rope_table`, `rope_table_set/release`,
  `kv_ring_table`, `kv_stream_reset`, `gr_workspace_init` — and the **doorbell redesign**.
* **Files:** `vulkan/src/kernels/elementwise_vk.cpp`, `.../rope_vk.cpp`, `.../kv_stream_vk.cpp`; a new
  `vulkan/src/device/sync.*` for the fence/semaphore replacement.
* **Proves:** each kernel's port case re-run through the engine wrapper; the doorbell replacement by a
  host/device handoff case that would **deadlock or time out** on a translating spin.
* **Risk (cannot predict):** the doorbell.  Its host spin and `__threadfence_system()` ordering have no Vulkan
  form; replacing it changes the recorded step's structure and the host's per-token loop, and it is the single
  biggest unknown in the whole integration.

### I3 — the quantised matvec / GEMV / KV family
* **Covers (9 kernel rows):** `native_mmvq` (6 shaders), `native_q5_k_f32`, `native_quantize_q8_1`,
  `quantize_q8_` (4), `quantize_q8_0_scaled`, `quantize_q8_1_rows`, `s_gemv_split_async`,
  `kv_append_q4_step`, `kv_gather_q4_step`.
* **Files:** `vulkan/src/kernels/native_mmvq_vk.cpp`, `.../s_gemv_vk.cpp`, `.../quantize_vk.cpp`,
  `.../kv_q4_vk.cpp`.
* **Proves:** a projection + a KV append/gather recorded into ONE command buffer and replayed, bitwise against
  the port's cases and against the single-shot path (the stage-3 discipline).
* **Risk (cannot predict):** pipeline-cache keying across 49 dispatches; 16-bit/8-bit storage alignment for
  the `kv_q8` codes on the Arc; the `s_gemv_split` async split's ordering inside a recorded step.

### I4 — attention, QSA and the MoE/expert half
* **Covers (12 kernel rows):** `qsa_decode_attn_step`, `ple_block` (4), `ple_history_advance`,
  `moe_grouped_s2` (4), `moe_hit_add`, `moe_hit_grouped_s2` (4), `moe_hit_grouped_s2_cpu_order` (5),
  `moe_hit_grouped_s2_dev` (5), `moe_hit_select`, `native_expert_grouped` (5), `iq_dequant_f32`,
  `iq_embed_rows`.
* **Files:** `vulkan/src/kernels/attn_vk.cpp`, `.../qsa_vk.cpp`, `.../ple_vk.cpp`, `.../moe_vk.cpp`,
  `.../iq_vk.cpp`.
* **Proves:** a full layer's math (attention + router + experts) through the backend; per-kernel gate cases;
  the grouped-expert grid — **capacity, not live count** (HANDOFF §2.3).
* **Risk (cannot predict):** the y-dimension grouping (the group COUNT lives on the device); the PLE path's
  history window; whether the composite entry points' shared scratch fits the arena as planned.

### I5 — sampler + the recorded decode step in the engine (M-B)
* **Covers (2 kernel rows):** `sample_tokens` (`sampler_greedy` + `sampler_kernel`); wiring the recorded decode
  step into the engine so `src/program/generate.cpp` drives it.
* **Files:** `vulkan/src/kernels/sampler_vk.cpp`, the engine's step/session wiring.
* **Proves:** **M-B** — a Vulkan-backed single-layer forward pass with random weights (needs no model at all)
  as a gate case; then the decode step replays and equals the single-shot path, byte for byte.
* **Risk (cannot predict):** the *program* — the composed step, the pass ordering, the host/device handoff
  under a real layer.  This increment is where a "kernel is green but the layer is wrong" surprise lives.

## 4. What stays deferred (deliberately NOT in this plan)

* **The batched PREFILL path (~250 KB).**  `src/prefill/kernels.cu`, `moe_fused*.cu`, `moe_mmq.cu`, the seven
  cuBLASLt sites and `fused_gr.cu` are NOT ported.  §4.3's bypass stands: **feed the prompt through the DECODE
  path**, token by token — slow but real, and it keeps every M-B milestone on decode-only code.
* **The expert file-tier streaming (58 GB against 24 GiB).**  The streaming logic is **host-side and already in
  the engine** (the map's `kv_stream_*` rows); what the Vulkan path must do is satisfy its residency
  assumptions (the ring table, the device resets) — the `kv_ring_table` / `kv_stream_reset` rows of I2.  No new
  host streaming code is planned, and the model run itself is M-C, after M-B.

## 5. Increment 0 — what landed (and what it proves)

`STRATA_ENABLE_VULKAN` as the option above; the `vulkan/` tree; and **one** entry point,
`strata::kernels::fwht256_cuda` (wrapper `fwht256_inplace_cuda` in `include/strata/kernels/kv_q4.hpp`), in
`vulkan/src/kernels/fwht_vk.cpp`, compiled only behind the macro.  Proven by a `cmake` CONFIGURE (measured wall
time in the commit message / the port's `STATUS.md`) and a compile of the single TU; the port's own gate re-run
green afterwards.  It does **not** run: the body is the seam above `strata::vulkan::fwht256`, which I1
implements against the arena.  This target is intentionally not link-complete until I1.

## 6. Estimate for the remaining work (what the next step is approved on)

Measured rate for this port: **~40 minutes per kernel end to end including gating** (RUN-ON-B70.md, the first
eight kernels).  Per increment, entry points are counted from §1; "flags" are the shaders.

| Increment | Entry points | Shader dispatches | New files | Wall clock | Needs a REAL engine build? |
|---|---|---|---|---|---|
| I1 device layer + arena + fwht256 | 1 (+ device layer) | 1 | ~4 (`device/`) | **1 session** | **yes** — the engine loops through the wrapper; a small target first, then the engine |
| I2 glue + host rows + doorbell | 11 kernel + 12 host | ~13 | ~5 | **1–1.5 sessions** | **yes** — the doorbell redesign is proven in the host loop |
| I3 matvec / GEMV / KV | 9 | ~17 | ~4 | **1–1.5 sessions** | **yes** — the recorded step is the point |
| I4 attention / QSA / MoE | 12 | ~24 | ~5 | **1.5–2 sessions** | **yes** |
| I5 sampler + recorded step (M-B) | 2 (+ engine wiring) | 2 | ~2 + engine edits | **1 session** | **yes** — M-B is a program proof |

**Total: ~5.5–7 sessions.**  The kernel half is the predictable part (the port has moved 28 kernels at ~40
min each, largely in parallelisable waves); the integration is I1's arena and I2's doorbell — the two places a
day can vanish — plus I5's program.  Every increment needs a real engine build to *prove* its claim (the port's
gate proves kernels, not programs), so each is a checkpoint the user approves; I1 is the smallest and should go
first.  **Nothing in this document was built; every number above is either read from `PORT-MAP.tsv` /
`run_gate.sh` / `RUN-ON-B70.md` or is a wall-clock estimate labelled as one.**
