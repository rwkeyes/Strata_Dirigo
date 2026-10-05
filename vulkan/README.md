# vulkan/ - the engine's Vulkan backend (experimental, increment 0)

This is the **backend tree** the top-level `-DSTRATA_ENABLE_VULKAN=ON` option configures.  It is a *separate*
tree like `sycl/` (docs/INTEL_ARC.md), not a branch of `src/kernels/cuda/`: the port's kernels are GLSL
(`ports/vulkan/shaders/`), its device layer is `VkDeviceMemory` over a command buffer, and no `.cu` file is
compiled here.  The integration plan - every decode-path entry point, grouped into increments, with what proves
each one - is `ports/vulkan/plan/BACKEND-INTEGRATION.md`.  Read that first; this file is just the layout.

## Layout

    vulkan/
      CMakeLists.txt                     the backend target; configured from the top-level option
      README.md                          this file
      include/strata/vulkan/
        vk_backend.hpp                   the device-layer SEAM: the declarations the kernel TUs call
      src/kernels/
        fwht_vk.cpp                      increment 0's ONE entry point: strata::kernels::fwht256_cuda

Increment 1 grows this into the full per-subsystem layout the plan names:

    vulkan/
      src/device/                        the device layer (the port's harness/vk_compute.* is the seed):
                                         arena, staging, pipeline cache, recorded step
      src/kernels/<subsystem>_vk.cpp     one TU per engine entry-point group (elementwise, gemv, kv, attn, moe)

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
backend - which is the whole point of the wrapper convention, and why the plan can enumerate the integration as
"implement N symbols", not "edit the engine's headers".

## What increment 0 proves, and what it does not

* **Proves:** the `STRATA_ENABLE_VULKAN` macro is wired per the file's own option pattern (mutually exclusive
  with CUDA/HIP/SYCL); the tree exists; one entry point compiles behind the macro and refuses to compile without
  it; `cmake` configures the whole option.  See the plan for the measured configure and compile times.
* **Does NOT prove:** that the entry point runs.  Its body is the seam above `strata::vulkan::fwht256`, which
  increment 1 implements against the arena.  This target is therefore not link-complete, on purpose, until
  increment 1 lands.  **Do not wire this into `setup.py` until M-B (one layer, end to end) is green.**
