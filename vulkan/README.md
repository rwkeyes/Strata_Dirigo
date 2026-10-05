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
        elementwise_vk.cpp               (I2) silu_inplace / scale_inplace / f32_to_bf16_bulk
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

