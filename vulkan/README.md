# vulkan/ - the engine's Vulkan backend (experimental; I1 landed, M-B not yet)

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
      src/kernels/
        fwht_vk.cpp                      the fwht256 entry point: strata::kernels::fwht256_cuda + fwht256
      tests/
        entry_point_smoke.cpp            strata_vk_entry_smoke: builds + RUNS the entry point via the wrapper

Later increments add the per-subsystem kernel TUs the plan names:

    vulkan/
      src/kernels/<subsystem>_vk.cpp     one TU per engine entry-point group (elementwise, rope, kv, gemv, attn, moe)
      src/device/sync.*                  the fence/semaphore replacement for the CUDA doorbell (I2)

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
