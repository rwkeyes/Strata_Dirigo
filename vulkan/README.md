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


