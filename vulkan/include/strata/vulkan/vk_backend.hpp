// vulkan/include/strata/vulkan/vk_backend.hpp - the Vulkan backend's device-layer seam (increment 0).
//
// WHY THIS HEADER EXISTS: the engine reaches every GPU kernel through a thin wrapper in its OWN header
// (include/strata/kernels/*.hpp).  kv_q4.hpp is the pattern:
//
//     void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
//     inline void fwht256_inplace_cuda(float* data, int64_t n_rows, void* stream) {
//         fwht256_cuda(data, data, n_rows, stream);
//     }
//
// The wrapper is backend-agnostic: a CUDA/HIP build answers `fwht256_cuda` from src/kernels/cuda/kv_q4.cu, and a
// Vulkan build answers it from vulkan/src/kernels/.  A kernel TU therefore knows the engine's argument list and
// nothing else; the device layer - the arena, the pipeline cache, the recorded decode step - sits behind the
// declarations below.  That is the same split `ports/vulkan/harness/vk_compute.hpp` describes between a case and
// the device; the backend is where the engine's arguments, not the gate's, are honoured.

#pragma once

#include <cstdint>

namespace strata::vulkan {

/// The backend's device and its decode step, as the engine's opaque `void* stream` names it.  On CUDA/HIP the
/// same `void*` is a `cudaStream_t`; here it is a `Stream*` (one device, one recorded step - PORT-PLAN's
/// "one command buffer per decode step shape", re-submitted per token).
struct Stream;

/// Resolve the engine's opaque stream handle to the backend's device, or null when `stream` is not a Vulkan
/// stream - so a wrong handle fails loudly instead of dereferencing garbage (the port's rule: refuse, never
/// degrade).
///
/// **A NULL HANDLE IS CUDA'S DEFAULT STREAM, NOT AN ERROR.**  The engine passes `nullptr` wherever it means "the
/// current stream" - `src/program/generate.cpp:3893` is `void* token_stream = o.stream_token ? main_cs : nullptr`
/// (so the DEFAULT single-token path reaches `embed_row` -> `embedding_gather` with a null stream), and the same
/// shape recurs at every op that does not take an explicit stream.  Under CUDA a null `cudaStream_t` is the
/// legacy default stream; this backend has ONE stream per device, so the shim publishes it as the DEFAULT STREAM
/// below and `stream_of(nullptr)` returns it.  Refusing a null handle here would refuse the engine's own decode
/// at its FIRST op - and then at each op after it, which is why the resolution is fixed here once rather than
/// per symbol.
Stream* stream_of(void* stream);

/// The backend's DEFAULT stream: the shim's own "current stream" (`cuda_compat_set_stream` / the stream its
/// `cudaStreamCreate` handed out), stored in the device layer so a null handle and the shim's current stream
/// cannot be two different things.  One storage; `cuda_runtime.cpp`'s `g_current` IS this object.
Stream*& default_stream_ref();
Stream* default_stream();

// ---- ENTRY POINTS IMPLEMENTED BY THE BACKEND ----------------------------------------------------------------
// One function per engine `kernels::` symbol whose body is GPU work.  The first is the skeleton's proof; the
// rest are the plan's increments in ports/vulkan/plan/BACKEND-INTEGRATION.md.

/// `fwht256_cuda`: the orthonormal 256-point Walsh-Hadamard rotation of the Q4_0 KV path.  Shader
/// `fwht256.comp` (SRC, DST; push {int n_rows}; one workgroup per row).  Implemented against the arena in
/// increment 1; declared here so the entry point compiles now.
void fwht256(Stream& s, const float* src, float* dst, int64_t n_rows);

// ---- I2: THE ELEMENTWISE GLUE (vulkan/src/kernels/elementwise_vk.cpp) ---------------------------------------
// The first three glue kernels the layer body reaches, in the order `src/core/layer.cpp`'s `gdn_layer`
// reaches them: silu_inplace (:257), scale_inplace (:276), f32_to_bf16_bulk (:290).  Each answers the
// engine wrapper in `include/strata/kernels/elementwise.hpp`; the shader each drives is named here so the
// gate can hold the two paths to the same .spv.
//
// `silu_inplace`   -> shader silu_f32.spv    (1 storage buffer, push {int n})
// `scale_inplace`  -> shader scale.spv       (1 storage buffer, push {int n; float s})
// `f32_to_bf16_bulk` -> shader f32_to_bf16.spv (X float[] read, Y uint16_t[] write, push {int n})
void silu_inplace(Stream& s, float* x, int64_t n);
void scale_inplace(Stream& s, float* x, int64_t n, float factor);
void f32_to_bf16_bulk(Stream& s, const float* x, uint16_t* y, int64_t n);

// ---- I2, CONTINUED: the next three the layer body reaches (vulkan/src/kernels/elementwise_vk.cpp) -----------
// `gdn_gate` (layer.cpp:300, the GDN gate) -> shader gdn_gate.spv (4 float[] buffers, push {int h_v; int n_tokens})
// `rms_norm_weighted` (layer.cpp:880, the QSA norm) -> shader rms_norm.spv (X rw, W read, push {int rows; int cols; float eps})
// `embedding_gather` (layer.cpp:1083, the token embedding) -> shader embedding_gather.spv (5 buffers, push PC)
void gdn_gate(Stream& s, const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v);
void rms_norm_weighted(Stream& s, float* x, const float* w, int64_t rows, int64_t cols, float eps);
void embedding_gather(Stream& s, const uint8_t* codes, const float* scales, const float* offsets, int64_t n,
                      int code_bits, int code_bias, int group_elems, float* out);

}  // namespace strata::vulkan
