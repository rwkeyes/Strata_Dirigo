# vulkan/ - the engine's Vulkan backend (experimental; I1 and I2 landed, M-B not yet)

This is the **backend tree** the top-level `-DSTRATA_ENABLE_VULKAN=ON` option configures.  It is a *separate*
tree like `sycl/` (docs/INTEL_ARC.md), not a branch of `src/kernels/cuda/`: the port's kernels are GLSL
(`ports/vulkan/shaders/`), its device layer is `VkDeviceMemory` over a command buffer, and no `.cu` file is
compiled here.  The integration plan - every decode-path entry point, grouped into increments, with what proves
each one - is `ports/vulkan/plan/BACKEND-INTEGRATION.md`.  Read that first; this file is just the layout.

## Layout

    vulkan/
      CMakeLists.txt                     the backend target(s); configured from the top-level option
      README.md                          this file
      include/strata/vulkan/
        vk_backend.hpp                   the device-layer SEAM: the declarations the kernel TUs call
      src/device/                        the device layer (I1):
        vk_compute.hpp/.cpp              the port's harness device layer, ADOPTED (namespace strata::vulkan)
        vk_compat.hpp/.cpp               deps of vk_compute (host/driver/stack facts)
        vk_stack.hpp/.cpp                deps of vk_compute (loader/ICD/firmware detection)
        vk_arena.hpp/.cpp                the ARENA, the pointer -> buffer resolution, the stream registry
        sync.hpp/.cpp                    (I2) the DOORBELL REPLACEMENT: the fence + host-driven handoff
      src/kernels/
        fwht_vk.cpp                      the fwht256 entry point: strata::kernels::fwht256_cuda + fwht256
        elementwise_vk.cpp               (I2) silu_inplace / scale_inplace / f32_to_bf16_bulk + the I2c glue
        doorbell_vk.cpp                  (I2b) the five doorbell_* symbols (device<->host, from sync.*)
        gdn_vk.cpp                       (I2d/I2e) the GDN / DeltaNet mixer entry points (all fourteen; the layer body)
        native_caps_vk.cpp               the native_*_enabled() capability answers the backend owns
      tests/
        entry_point_smoke.cpp            strata_vk_entry_smoke: builds + RUNS the entry points via the wrappers

Later increments add the per-subsystem kernel TUs the plan names:

    vulkan/
      src/kernels/<subsystem>_vk.cpp     one TU per engine entry-point group (rope, kv, gemv, attn, moe)
      src/device/sync.*                  (LANDED, I2) the fence/semaphore replacement for the CUDA doorbell

## How the engine's sources join this configuration (I1's decision)

The top-level `STRATA_ENABLE_VULKAN` block still `return()`s before the CUDA engine is configured (src/'s source
list assumes CUDA and is not well-formed without it).  The engine's sources join through
`strata_vulkan_resolve()` in `CMakeLists.txt`, which **mirrors `../sycl/CMakeLists.txt`'s `strata_resolve`**: a
source is taken from `vulkan/` when a migrated copy exists there, otherwise from the engine tree
(`${STRATA_ROOT}`).  I1 needs no engine-root source - the fwht256 entry point is the header wrapper in
`include/strata/kernels/kv_q4.hpp` answered by `vulkan/src/kernels/fwht_vk.cpp` - so the first engine-root
sources are I2's glue and are named through the same function.  The engine's HEADERS are never edited.

## How one entry point is wired (the pattern, not an invention)

Every GPU entry point the engine calls is a thin wrapper in the engine's own header.  `kv_q4.hpp` is the shape:

```cpp
void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
inline void fwht256_inplace_cuda(float* data, int64_t n_rows, void* stream) {
    fwht256_cuda(data, data, n_rows, stream);   // the engine always calls the wrapper
}
```

The wrapper is backend-agnostic.  `src/kernels/cuda/kv_q4.cu` answers `fwht256_cuda` on a CUDA/HIP build;
`vulkan/src/kernels/fwht_vk.cpp` answers it on a Vulkan build.  The engine header is **not modified** by the
backend - which is why the plan can enumerate the integration as "implement N symbols", not "edit N headers".

## The device layer, and why it is a separate copy

`ports/vulkan/harness/vk_compute.*` is the port's device layer, used by the numeric gate.  I1 ADOPTS it as
`vulkan/src/device/` (with `vk_compat.*`/`vk_stack.*`, which it includes) so the engine build does not depend on
`ports/`.  The adopted copy's namespace is `portvk` -> `strata::vulkan` so one translation unit can hold BOTH
(the gate's case drives the engine wrapper and compares against its own shader path).  The port's harness copy is
untouched and remains the gate's oracle.

On top of the adopted layer, `vk_arena.*` adds:

* **THE ARENA** - one device-local buffer carved by byte offsets.  `arena_alloc` bump-allocates, aligned to the
  device's `minStorageBufferOffsetAlignment` (raised to 256), so every view it hands out is bindable.
* **THE SYNTHETIC ADDRESS SPACE** - a device pointer is `kArenaBase + byte_offset`.  The engine holds raw device
  pointers (`X + t0*K`) and does arithmetic on them as it did against `cudaMalloc`; the value only has to be
  RESOLVABLE, not readable (the arena is unmappable VRAM on the Arc, so there is no host address to hand out).
* **POINTER -> BUFFER** - `arena_resolve` subtracts the base, range-checks the LIVE region, and returns
  `view(arena, offset)`.  A pointer outside the arena is refused, not bound.

## What I1 proves, and what it does not

* **Proves:** the `STRATA_ENABLE_VULKAN` configuration builds the device layer + the kernel TU + an engine-side
  target (`strata_vk_entry_smoke`) that runs; the engine wrapper `strata::kernels::fwht256_cuda` runs end to end
  through the arena and is compared **bitwise** against the port's already-green shader case by
  `case_fwht256_entry` in the port's gate; and a wrong dispatch view offset changes the answer
  (`gates/inject-verify.sh fwht-entry-wrong-view-offset`).
* **Does NOT prove:** the engine program itself.  Nothing in `src/` is compiled under `STRATA_ENABLE_VULKAN` yet,
  and the other 39 entry points are the plan's later increments.  **Do not wire this into `setup.py` until M-B
  (one layer, end to end) is green.**

## What I2 adds (2026-10-05)

* `src/device/sync.*` - the CUDA doorbell's replacement.  The CUDA handshake is a kernel that spins on host
  memory ordered by `__threadfence_system()`; this backend splits the two directions: device->host is the
  submission FENCE (every submit is fenced and waited, so a publish that returns is a publish the host can
  read), host->device is a HOST-DRIVEN SPLIT SUBMISSION (the consumer is submitted only after the host has
  written the answer), so no kernel ever waits.  The header states what each CUDA call site is for and why
  this mechanism; the gate's `case_sync_handoff` pins the ordering.  `sync_copy_fenced` exposes the one fenced
  copy the whole handshake is built on, so the engine's doorbell symbols reuse it rather than duplicating it.
* `src/kernels/elementwise_vk.cpp` - the six glue entry points the LAYER BODY reaches, in order:
  `silu_inplace` (`src/core/layer.cpp:257`), `scale_inplace` (:276), `f32_to_bf16_bulk` (:290), `gdn_gate`
  (:300), `rms_norm_weighted` (:880) and `embedding_gather` (:1083).  Each is the thin wrapper the engine
  header declares; the gate's `case_*_entry` re-runs it through the ENGINE wrapper bitwise against the port's
  own shader path and against the explicit oracle.
* `src/kernels/doorbell_vk.cpp` - the five `doorbell_*` symbols `layer.cpp:380/389` and `session.cpp:873` call,
  answered from `sync.*`: device->host is the fenced publish (+ the ring), host->device `doorbell_wait` submits
  NOTHING (the host writes the answer first; an un-answered handoff is a loud refusal, never a spin).
* `strata_vk_entry_smoke` now RUNS fwht256 + the six glue wrappers + the five `doorbell_*` + the handoff on the
  device, on the CMake-built object code (the numeric proof stays in the port's gate).

## What the engine-program measurement found (2026-10-05)

I2's numbers are the port's own backend + smoke target.  The ENGINE'S OWN PROGRAM is a separate matter and it
was measured, not estimated: under `STRATA_ENABLE_VULKAN=ON` the top-level `CMakeLists.txt` `return()`s before
`src/` is configured, so (1) CONFIGURE builds the backend only; (2) COMPILE of `src/core/*.cpp` and
`src/program/generate.cpp` is clean except `STRATA_VERSION` (a top-level `add_compile_definitions` after the
`return()`); (3) LINK of one layer body fails with 80 distinct `strata::kernels::` symbols (4 answered) and 12
CUDA-runtime symbols — the engine calls the CUDA runtime directly in 18 host files; (4) the deepest
engine-linked target that runs is `strata_vk_entry_smoke`.  Whether that is I5's wiring or a re-scope is the
user's call; `ports/vulkan/NEXT.md`'s I2-continued section reports it and does not re-scope it.

## What I2 continued-further adds (2026-10-05)

* **Five more glue entry points** in `vulkan/src/kernels/elementwise_vk.cpp`, in the order the forward path
  reaches them: `add_inplace` (`expert_source.cpp:2353`, the MoE expert pool's hit combine), `scatter_rows_f32`
  (`peer_experts.cpp:241`), `cvec_apply` (`layer.cpp:1330`, the ONLY one `layer.cpp` calls DIRECTLY),
  `gather_rows` (`mtp.cpp:450`, the MTP drafter) and `f32_to_f16_bulk` (no `src/core/` call site - it is on the
  plan's I2 list and in `elementwise.hpp`'s contract).  Four are thin wrappers over the port's own already-gated
  shaders (`add.spv`, `scatter_rows_f32.spv`, `gather_rows.spv`, `f32_to_f16.spv`).
* **`cvec_apply` is NOT a thin wrapper**: it reads the engine's control-vector MODULE (`strata::kernels::cvec()`)
  that `cvec.cu` owns, so this TU also answers the cvec `host` row (`cvec`, `cvec_upload`, `cvec_replicate`,
  `cvec_set_enabled`, `cvec_enabled`).  The device tables are placed LAZILY into each `Stream`'s arena (the
  tables live in `Stream::cvec_tables`, `vk_arena.hpp`), because `cvec_upload` carries no stream and the CUDA's
  per-"current device" table has no Vulkan analogue.
* **The proof is the established one**: each is re-run through the ENGINE WRAPPER and required to agree BITWISE
  with the shader path AND with the case's explicit oracle (`case_*_entry` in `ports/vulkan/harness/vk_gate.cpp`,
  each pinned to the harness device with `EnginePin`).  `strata_vk_entry_smoke` now RUNS all five too.
* **The LINK PROGRESS (the port's progress bar toward a layer that LINKS):** the one-layer-body link
  (`src/core/layer.cpp` vs `libstrata_vulkan_kernels.a` + `libstrata_vulkan_device.a`) moved **214 -> 204**
  undefined references / **80 -> 73** distinct `strata::kernels::` symbols.  THIS batch accounts for two of those
  seven (`cvec_apply`, `cvec` - the only two of the five `layer.cpp` itself reaches); the other five are I2b's
  glue/`doorbell_*` that `layer.cpp` also calls and the "backend answers 4" baseline predates.  The remaining 73
  are grouped in `ports/vulkan/NEXT.md`'s I2-continued-further section, with the reproducing command.

## What I3 adds (2026-10-05)

* `vulkan/src/kernels/matvec_vk.cpp` - the first EIGHT matvec/GEMV/KV entry points, in the order the layer body
  reaches them (derived from `src/core/layer.cpp` and its siblings, NOT the plan's list): `quantize_q8_K`
  (`layer.cpp:236`), `quantize_q8_0` (`:237`), `native_quantize_q8_1` (`:150`), `native_mmvq` (`:151`, the
  COMPOSITE - six `*_mmvq` shaders chosen by ggml type), `kv_append_q4_step` (`:935`), `kv_gather_q4_step`
  (`:985`), `quantize_q8_0_scaled` (`session.cpp:868`) and `native_q5_k_f32` (`native_head.cpp:78`).  The other
  two of the increment's ten are DEFERRED because the single-GPU decode path does not reach them:
  `quantize_q8_1_rows` (the peer-expert pool, `peer_experts.cpp:230`) and `s_gemv_split_async` (only the
  standalone mains `overlap_main.cpp`/`concurrent_main.cpp`).
* **The `host` rows this TU answers** (they are not a bare bind): `iq_row_bytes` (the per-format row stride the
  six `*_mmvq` shaders take as `row_bytes` - a quantisation constant), `native_mmvq_supported` (the CAPABILITY
  CHECK that gates the composite: TRUE for exactly the six types with a shader, FALSE otherwise, so the engine's
  native loader cannot route to an unported kernel), `native_mmvq_weight_bytes` and `native_q8_1_bytes` (the
  weight/activation sizes the wrappers range-check).
* **The IQ grid tables** (`native_mmvq`'s four grid-taking arms) are placed per-stream in `Stream::iq_grids`
  (lazily, `vulkan/src/device/vk_arena.hpp`), from `vulkan/src/kernels/iq_grids_vk.hpp` - a verbatim copy of the
  port's generated harness header, adopted the way I1 adopted `vk_compute.*`.  A per-dispatch upload would
  exhaust the arena (`arena_alloc` never decreases), so it is a stream-lifetime cache.
* **The proof is the established one**: each is re-run through the ENGINE WRAPPER and required to agree BITWISE
  with the shader path AND with the case's explicit oracle (`case_*_entry` in `ports/vulkan/harness/vk_gate.cpp`,
  each pinned to the harness device with `EnginePin`).  `native_mmvq`'s case drives the IQ2_S arm (the grid
  path); `native_q5_k_f32`'s feeds the shader path the wrapper's OWN activation bytes (the quantiser's division
  form is chosen per pipeline instance - measured to differ on RADV - so comparing two quantiser runs would not
  be a dot-kernel test; the quantiser half is gated by `case_native_quantize_q8_1_entry`).
* **THE LINK PROGRESS (the port's progress bar toward a layer that LINKS):** the one-layer-body link moved
  **188 → 170** undefined references / **59 → 53** distinct full-signature `strata::kernels::` symbols /
  **57 → 51** under the parent's name-only pattern.  Six of the eight move THIS link (the two reached by sibling
  files - `quantize_q8_0_scaled`, `native_q5_k_f32` - do not, exactly as I2c's `add_inplace`/`gather_rows` did
  not).  The matvec/GEMV/KV group falls **21 → 15**; the other groups are unchanged.  The recipe and the group
  table are in `ports/vulkan/NEXT.md`'s I3 section.

## What I2 continued-further-still and I2e add (2026-10-05)

* `vulkan/src/kernels/gdn_vk.cpp` (new in I2d, completed in I2e) - **the GDN / DeltaNet MIXER entry points**, all
  fourteen the layer body reaches, in the order `gdn_layer` (`src/core/layer.cpp:223`, the mixer for 36 of the 48
  layers) reaches them.  I2d: `fused_gdn_conv_l2` (:250), `native_gdn_conv_silu` (:253), `gdn_conv_step` (:255),
  `native_gdn_l2_norm` (:266/267), `gdn_l2_norm` (:269/270), `fused_gdn_ab` (:287).  I2e: `native_gdn_beta_gate`
  (:296), `native_gdn_gate` (:297), `gdn_beta_gate` (:299), `native_gdn_step` (:308), `gdn_step` (:309),
  `fused_gdn_step_norm` (:322), `native_gdn_out_norm` (:324), `gdn_out_norm` (:325) - the beta/gate and step/norm
  stages, COMPLETING the mixer.  (`gdn_gate` :300 is answered in `elementwise_vk.cpp`.)  Engine headers unchanged.
  Each is proved by a new `case_*_entry` in the port's gate through the ENGINE WRAPPER, BITWISE against the port's
  own shader path AND against the case's explicit oracle (a double transcription of the engine's own CUDA body),
  pinned to the harness device (`EnginePin`); the conv and step cases compare the mutated STATE bitwise too.
  `strata_vk_entry_smoke` runs them.
* **No `host` row was needed, and for I2e that is MEASURED:** the only row the mixer reaches is
  `native_gdn_enabled()`, already answered by `native_caps_vk.cpp` (TRUE).  The GDN headers carry no
  `*_scratch_bytes` and no shape accessor - `GdnShapes` is a by-value POD, not a symbol - and (I2e) the
  one-layer-body link drops by exactly the eight symbols this file adds with NO new undefined reference.  None of
  them reads module state (unlike `cvec_apply`); they carry the CUDA wrappers' argument contracts (`d_conv == 4`,
  `channels % 128 == 0`, `n_embd % 8 == 0`, `S == 128`, `cols == 128`, `h_v % h_k == 0`) as loud refusals.
* **The LINK PROGRESS moved `204 -> 196` undefined references / `73 -> 67` distinct `strata::kernels::` symbols**
  (the six this batch answers; all six references `layer.cpp` makes).  The remaining 67, grouped by subsystem
  (glue 0 - matvec/GEMV/KV 21 - attention/QSA/MoE/GR/PLE/rope 36 - GDN mixer 8 - other 2), are at the top of
  `ports/vulkan/NEXT.md`'s I2-continued-further-still section, with the reproducing command.
* **Left in the mixer:** the remaining 8 GDN symbols (the beta/gate and step/norm stages, including
  `fused_gdn_step_norm`).



## The CUDA-runtime shim + the engine's own targets (2026-10-05)

The engine's host TUs call the CUDA runtime DIRECTLY, so a Vulkan engine build needs an answer for those names.
`vulkan/include/cuda_compat/cuda_runtime.h` (put FIRST on the include path - the `include/strata/platform/hip_compat/`
mechanism) + `vulkan/src/compat/cuda_runtime.cpp` implement them over this device layer: arena allocation
(`cudaMalloc`), the STAGING copies (`cudaMemcpy`/`cudaMemcpyAsync`/`cudaMemcpy2DAsync`), host-visible coherent
memory (`cudaHostAlloc`/`cudaHostGetDevicePointer`/`cudaFreeHost`), `cudaMemsetAsync` as a staged fill,
`cudaDeviceSynchronize` as the submission fence, and a real error enum + `cudaGetErrorString` +
`cudaPeekAtLastError`/`cudaGetLastError`.  Every place CUDA could NOT be honoured exactly is stated in the
header: `cudaEventElapsedTime` is a HOST wall-clock figure (this device layer enables no Vulkan timestamp query),
`cudaFree` releases nothing (the arena is a bump allocator), a mapped host pointer is NOT device-dereferenceable
(the doorbell replacement is `src/device/sync.*`), and the CUDA graph API / stream capture is NOT provided
(deferred to I5).  `strata_vk_cudart_smoke` (new) RUNS every entry point on the Arc and checks the bytes.

The one-layer-body link (`src/core/layer.cpp` vs the backend + the shim) drops **`170 -> 118` undefined references
with the 53 distinct `strata::kernels::` symbols UNCHANGED**: the shim resolves all 12 CUDA-runtime symbols.

`vulkan/CMakeLists.txt` also defines the ENGINE'S OWN targets now, mirroring `../sycl/CMakeLists.txt`:
`strata_vulkan_core`, `strata_vulkan_engine`, `strata_vulkan_spec`, `strata_vulkan_kernels_cpu` and the
`strata_vulkan` executable, so a `-DSTRATA_ENABLE_VULKAN=ON` configure has an engine binary and not just the
backend library.  Excluded (and why): the three `src/core/*.cu`, the prefill path, the parity/bench TUs, and the
native-expert/ggml half of the CPU kernels.  The `STRATA_VERSION` define moved ABOVE the backend-option blocks in
the top-level `CMakeLists.txt` (each `return()`s), which is what let `generate.cpp` compile.

## What I4 adds (2026-10-05)

* `vulkan/src/kernels/matvec_vk.cpp` - the NEXT EIGHT matvec/GEMV/KV entry points, in the order the layer body
  names them (source order in `src/core/layer.cpp`, NOT the plan's list): the BF16 GEMV family
  (`bf16_gemv_fp32_mmvf` `layer.cpp:97`, `bf16_gemv_split` `:98`, `bf16_gemv` `:99` - the first two branches of
  `project_bf16`; `bf16_gemv`/`bf16_gemv_split` share `bf16_gemv.spv`), the S2 GEMV (`s2_gemv_q8` `:166`), and
  the FP16/INT8 KV cache (`kv_append_q8_step` `:934`, `kv_append_step` `:943`, `kv_gather_q8_step` `:983`,
  `kv_gather_step` `:989`).  Engine headers unchanged.
* **THE `host` ROW: `kv_block_bytes(s, fmt)`** (the size of one KV page, transcribed from `kv_stream.cu`).  The
  other three KV-stream rows (`kv_stream_reset`, `kv_ring_table`, `kv_stream_resolve`) are the STREAMING RESIDENT
  TIER (`--kv-resident`, off by default): the engine calls the first two with a NULL stream (no default stream
  exists here) and the third needs resolve/copy shaders this tree does not build - so they are NOT answered.
* **A LATENT I3 DEFECT FIXED:** `kv_append_q4_step` bound the POOL with `host_layout = host ? 1 : 0`, but
  `host_layout` picks the ROW RULE (page-table row vs identity row), not whether to write - so under KV streaming
  it wrote only the host row and left the pool stale.  It now always writes the pool AND (when a host pool is
  present) the host buffers.  `kv_append_q8_step`/`kv_append_step` implement the same two-dispatch shape.
* **The proof is the established one**: each is re-run through the ENGINE WRAPPER and required to agree BITWISE
  with the shader path AND with the case's explicit oracle (`case_*_entry` in `ports/vulkan/harness/vk_gate.cpp`,
  each pinned to the harness device with `EnginePin`).  The q8 append case also proves the host-copy arm with a
  table whose pool row and identity row DIFFER.
* **No `*_enabled()` capability was turned on**, so `case_native_capabilities` needed no new arm.
* **THE LINK PROGRESS (the port's progress bar toward a layer that LINKS):** the one-layer-body link (the standard
  recipe in `ports/vulkan/NEXT.md`, now including the shim library) moved **118 -> 106** undefined references /
  **53 -> 44** distinct full-signature `strata::kernels::` symbols / **51 -> 42** under the parent's name-only
  pattern.  The matvec/GEMV/KV group falls **15 -> 6**.  The engine's five cross-TU symbols
  (`LayerView::name`, `WeightTable::find`, `native_embed`, `NativeEmbed::gather_one`, `main`) all have a home in
  the targets the shim batch added; their TUs compile clean under the shim, and only the sibling TUs that need the
  DEFERRED CUDA graph API (I5's) keep those targets from building.

## What I5 (engine) adds (2026-10-05)

* `vulkan/src/kernels/qsa_vk.cpp` (new) - **the ATTENTION / QSA / rope entry points**, the non-GDN half of the
  decode path: the first eight entry points the non-GDN layer body `qsa_layer` (`src/core/layer.cpp:876`) reaches,
  in call-site order - `native_qsa_rms_norm_weighted` (`:879`), `native_rope_apply` (`:881`), `rope_neox_apply`
  (`:882`), `qsa_block_scores` (`:970`), `qsa_block_topk` (`:971`), `native_qsa_gate_apply` (`:1010`),
  `qsa_gate_apply_f32` (`:1011`) and `native_router_top10` (`:370`, the expert routing).  Engine headers unchanged.
  Each proved by a new `case_*_entry` through the ENGINE WRAPPER, BITWISE against the port's shader path AND against
  the case's explicit oracle (`EnginePin`-pinned); `strata_vk_entry_smoke` links the new TU (the numeric proof is
  the gate cases, which run the wrapper end to end).
* **A CROSS-CUTTING DEFECT FOUND AND FIXED WHILE WIRING (the last batch's latent-defect class): the norm WEIGHT
  was indexed by the ELEMENT index instead of the COLUMN index.**  `rms_norm.comp` and
  `native_qsa_rms_norm_weighted.comp` read `w.v[i]`/`gamma.v[i]` (`i = row*cols+c`) where the engine's contract is
  `w[n_cols]` broadcast over rows (`native_qsa.cu` `gamma[col]`; `elementwise.cu` `r[c]*w[c]`).  Every row past the
  first read the wrong weight, and a cols-long engine weight was read OUT OF BOUNDS - a silently wrong token on
  every QSA layer.  Three cases' oracles reproduced the shader's indexing and masked it.  Fixed both shaders
  (`[c]`), the three oracles (column index), and `elementwise_vk.cpp`'s weight resolution (`cols` floats).
* **THE `host` ROW:** `rope_scaling()` (+ `rope_scaling_set`) - the rope constants, owned by `rope_scaling.cu` on a
  CUDA build.  No capability flipped.
* **REPORTED, NOT WIRED:** `qsa_decode_attn_step` (`layer.cpp:980`) has NO shader in this tree - PORT-MAP maps it
  to `attn_decode_short`, but that shader IS `native_flash_attn_short_step`'s (a gathered f16 window) while this
  symbol reads the KV POOLS through a page table.  A shader-port job, not a stub.
* **THE LINK PROGRESS:** the one-layer-body link moved **106 -> 94** undefined references / **44 -> 35**
  full-signature / **42 -> 33** name-only.  The attention/QSA/MoE/GR/PLE/rope group falls **36 -> 27**.  All 35
  remaining are referenced by `layer.cpp` itself; the only other structural blocker is the deferred CUDA graph API
  (I5's).

## What the NEXT batch adds: the DEFAULT QSA decode attention was a HOLE, and is now ported (2026-10-05)

* **`qsa_decode_attn_step` is REACHABLE, and it had no shader** - so it was a HOLE of the class the two batches
  before it found.  Its call site (`src/core/layer.cpp:980`) is inside
  `if (g_fast_attn && !native_flash_attn_short && dump == nullptr)`, whose three inputs are ALL the shipped
  configuration's defaults: `g_fast_attn` **true** (`layer.cpp:42`), `native_flash_attn_short` **false**
  (`layer.cpp:92`, set only by `--native-flash-attn-short`, NOT by `--native`), `dump` **nullptr** on the decode
  path.  It reads the KV POOLS through the PAGE TABLE - a DIFFERENT kernel from `attn_decode_short` (which is
  `native_flash_attn_short_step`, a gathered `[cap,2,256]` f16 window), so PORT-MAP's mapping and the
  reachability audit's row were both wrong.  Both are corrected.
* `ports/vulkan/shaders/qsa_decode_attn.comp` - the pools read directly through the page table (no gather), one
  workgroup per query head, online softmax, mode 0 = f16 pools (the shipped `--kv fp16` default), mode 1 = int8
  codes + fp16 scale / 64; q4_0 and the K8V4 hybrid are refused loudly.  The engine's chunk+merge CUDA is
  RE-DERIVED onto one workgroup with barrier-tree reductions (subgroup ops banned) and needs no scratch; the gate
  MEASURES the difference against a DOUBLE transcription of the engine's rule.
* Wired in `vulkan/src/kernels/qsa_vk.cpp` as `strata::kernels::qsa_decode_attn_step` (its 9th entry point) plus
  the `host` row `qsa_decode_attn_scratch_floats`; `strata_vk_entry_smoke` runs it.  Proved by
  `case_qsa_decode_attn` (3 arms, the shader path vs the ENGINE WRAPPER BITWISE and both vs the rule),
  FALSIFIED by `gates/inject-verify.sh qsa-decode-attn-drop-kv-head` -> `FAIL ... 0/6144 worst 2.26e+04`.
* **THE LINK PROGRESS:** `94 -> 91` undefined references / `35 -> 33` full-signature / `33 -> 31` name-only; the
  attention/QSA/MoE/GR/PLE/rope group falls **27 -> 25**.  Gate on `vega`: Arc 612/0/0 (exit 0), llvmpipe 600/0/3,
  radeon iGPU 603/0/2.  `PORT-MAP.tsv` regenerates byte-identically (one row: `qsa_decode_attn_step -> qsa_decode_attn`).
  Bench (Arc): `qsa_decode_attn` 0.38 / 1.29 / **2.52 ms** at n_ids 256 / 1024 / 2048 - linear in the selection
  width, the honest cost of the correctness form.

## What the PLE / GR batch adds (2026-10-05)

* `vulkan/src/kernels/ple_vk.cpp` (new) - **the PLE / GR SHARED STAGES and the MoE ROUTING ROWS**: `gr_write`
  (`layer.cpp:1261`, `:1195`/`:1329`), `ple_block` (`:1208`, layer 1), `ple_history_advance` (`:1222`),
  `gr_read` (`:1255`), `router_top10` (`:373`, `moe_route`'s generic router) and `native_moe_combine` (`:463`,
  the DEFAULT combine) - in the SOURCE order `block_layer_pre`/`block_layer_post` reach them.  Engine headers
  unchanged.  The GR pair runs for EVERY layer; `ple_block` carries its own key/value PROJECTIONS
  (`quantize_q8_0`+`s2_gemv_q8`, `f32_to_bf16`+`bf16_gemv`), which the port's four `ple_*` shaders alone do not.
* **The two `host` rows:** `gr_workspace_init` (the GR workspace table, `block_buffers_init`, layer.cpp:1140) and
  `ple_block_scratch_bytes` (layer.cpp:1203/1316), plus the PLE/GR branch policy (`ple_native_bf16_enabled` /
  `ple_native_postops_enabled` -> FALSE; the two `gr_set_*` -> no-ops), the `native_gdn_enabled()` pattern.
* **A LATENT DEFECT FOUND AND FIXED:** the port's GR shaders store the bf16 activations as **f32**
  (`gr_norm.comp` binding 3 is `float v[]`, "bf16(xn) as f32") while the engine's `GrWorkspace` sizes `xq`/`lq`
  as **uint16** - so `gr_read` with the engine's own workspace wrote 4 bytes into a 2-byte region.  `gr_workspace_init`
  now sizes them for f32 (legal: the struct is opaque pointers and `gr_workspace_bytes` is the allocation authority).
  The gate's first run caught it as `gr_read entry ... 0/66` on every device.  A second defect - `ple_block`'s
  wrapper writing `gnorm(gated)` back onto `gated` - was caught as `0/10240` bitwise and fixed at the source.
* **Each proved by a new `case_*_entry`** through the ENGINE WRAPPER, BITWISE against the port's shader path AND
  against the case's explicit oracle (a transcription of the engine's `.cu` rule, not the shader), `EnginePin`-pinned.
  Every rival reading has its OWN observable and a host-side margin proving it MOVES the reference.
  `strata_vk_entry_smoke` RUNS four of the six wrappers on the Arc.
* **THE LINK PROGRESS:** the one-layer-body link moved **91 -> 75** undefined references / **33 -> 25**
  full-signature / **31 -> 23** name-only.  The attention/QSA/MoE/GR/PLE/rope group falls **25 -> 17**.
  Gate on `vega`: Arc 639/0/0 (exit 0), llvmpipe 627/0/3, radeon iGPU 630/0/2.  **`z820b` PENDING.**

