// vulkan/include/cuda_compat/cuda_runtime.h - THE CUDA-RUNTIME COMPATIBILITY SHIM over the Vulkan device layer.
//
// ============================================================================================================
// WHY THIS FILE EXISTS
// ============================================================================================================
//
// The engine calls the CUDA runtime API DIRECTLY from its host translation units (`src/core/*.cpp`,
// `src/program/*.cpp`): `cudaMalloc`, `cudaMemcpy`, `cudaHostAlloc`, `cudaMemsetAsync`, streams, events.  On a
// CUDA/HIP build those come from the toolkit; on the Vulkan port there is no toolkit, so the engine's own
// sources would not link.  This header is put FIRST on the include path of the Vulkan engine build (the same
// mechanism `include/strata/platform/hip_compat/` uses) so `#include <cuda_runtime.h>` resolves HERE.  The
// engine's headers and sources are NOT edited - the plan's principle, kept.
//
// WHAT IT IS BUILT ON: the port's Vulkan device layer - `strata::vulkan::Stream` (one device, one arena of
// device-local memory carved by byte offsets, the pointer->buffer resolution) and its fenced submit path
// (`Ctx::dispatch` submits and WAITS a fence, so a call that returns is a call whose device work has
// completed).  `vulkan/src/compat/cuda_runtime.cpp` is the implementation.
//
// ============================================================================================================
// THE SEMANTICS THAT COULD NOT BE HONOURED EXACTLY - READ BEFORE TRUSTING A NUMBER
// ============================================================================================================
//
// The port's rule is "refuse, never degrade", so each place CUDA and this backend genuinely differ is stated
// here rather than approximated silently:
//
//   * `cudaHostAlloc(CUDA_HOSTALLOC_MAPPED)` -> a HOST_VISIBLE | HOST_COHERENT buffer.  CUDA gives the device
//     a ZERO-COPY device pointer into host memory.  This device layer has NO host-visible memory a shader can
//     bind (the arena is DEVICE_LOCAL and, on a discrete card, unmappable), so the device pointer returned by
//     `cudaHostGetDevicePointer` is a HOST pointer: the SHIM's own copy path (`cudaMemcpy*`) recognises and
//     handles it, but a SHADER cannot dereference it.  The engine's zero-copy handshake (the doorbell) is
//     replaced in this backend by `vulkan/src/device/sync.*` (fenced copies), NOT by this pointer.
//
//   * `cudaEventElapsedTime` -> a HOST-SIDE WALL-CLOCK approximation (`std::chrono::steady_clock` between
//     `cudaEventRecord` calls).  A real Vulkan TIMESTAMP query is NOT enabled in this device layer (the query
//     pool is not created), so this number is the time between the two host calls and CANNOT mean device
//     execution time.  Every engine timing that goes through it is therefore a host-time figure here.
//
//   * `cudaDeviceSynchronize` -> the submission fence.  Every `Ctx::dispatch` already submits with a fence and
//     WAITS it before returning, so at the moment this is called nothing is in flight: it is a real barrier
//     that happens to be vacuous.  It does not wait for any asynchronous work, because this backend has none.
//
//   * `cudaFree` / `cudaFreeHost` -> the arena is a BUMP allocator (`arena_alloc` never decreases; see
//     `vulkan/src/device/vk_arena.hpp`).  `cudaFree` therefore releases NOTHING back to the arena; the memory
//     is reclaimed when the stream closes.  This is honest for the engine's lifetime pattern (one arena per
//     session) but it means a workload that allocates/frees in a loop will exhaust the arena.  `cudaFreeHost`
//     DOES free (a host region is its own buffer).
//
//   * `cudaMemcpy`/`cudaMemcpyAsync`/`cudaMemcpy2DAsync` -> the transfers go through the device layer's
//     STAGING path (a host-visible buffer + `vkCmdCopyBuffer` with the right barrier), never a memcpy on
//     device memory.  A device->device copy is staged through the host (read then write): correct, but it is
//     NOT a device-side copy and pays two transfers.  A host<->host copy is a plain memcpy.
//
//   * `cudaMemsetAsync` -> implemented as a staging fill (a host buffer of the repeated byte, uploaded).  There
//     is no memset shader in this device layer.
//
//   * The CUDA GRAPH API and stream capture (`cudaGraph*`, `cudaStreamBeginCapture`/`EndCapture`) are NOT
//     provided - deferred to increment I5 by the user.  Any engine TU that references them will not link
//     against this shim.
//
// Nothing here silently fakes a figure: the timing is labelled host-side, the memory figure comes from the
// device's own `VK_EXT_memory_budget`, and an unsupported shape is an ERROR CODE, not a plausible value.
// ============================================================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(STRATA_ENABLE_VULKAN)
#error "the Vulkan CUDA-runtime compat header is for a -DSTRATA_ENABLE_VULKAN=1 build only"
#endif

// ---- host-compile decorations -----------------------------------------------------------------------------
// The engine's headers carry `__host__ __device__` / `__forceinline__` on host-visible helpers (f16_bits.hpp,
// rope.hpp, bf16_bits.hpp).  A host compiler has no such keywords; the toolkit's host_defines.h makes them
// empty, and so does this shim (guarded, so a TU that already saw another definition is not broken).
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __restrict__
#define __restrict__
#endif
#ifndef __shared__
#define __shared__ static
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __align__
#define __align__(n)
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

// ---- the error type: a REAL enum, and a REAL string, not a constant ----------------------------------------
// Values follow the toolkit's own (so an engine comparison against `cudaErrorNotReady` still means 34).
enum cudaError {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorInvalidDevice         = 10,
    cudaErrorUnknown               = 30,
    cudaErrorNotReady              = 34,
    cudaErrorNotSupported          = 801,
    cudaErrorStreamCaptureUnsupported = 900,
};
typedef enum cudaError cudaError_t;

// ---- the opaque handles -----------------------------------------------------------------------------------
// The engine holds a `void* stream` and casts it to `cudaStream_t` (layer.cpp:232, :717, :1301 ...); on this
// backend that void* IS a `strata::vulkan::Stream*`.  `cudaEvent_t` is a shim-owned host-timing object.
struct cudaStream_st;
struct cudaEvent_st;
typedef struct cudaStream_st* cudaStream_t;
typedef struct cudaEvent_st* cudaEvent_t;

// ---- memcpy kinds (toolkit values: 0..4) ------------------------------------------------------------------
enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
};

// ---- host-allocation flags --------------------------------------------------------------------------------
enum cudaHostAllocFlags {
    cudaHostAllocDefault       = 0x00,
    cudaHostAllocPortable      = 0x01,
    cudaHostAllocMapped        = 0x02,
    cudaHostAllocWriteCombined = 0x04,
};
// Event-create flags (the shim's events carry no device timing; the flag is accepted, not honoured differently
// - see `cudaEventElapsedTime` above).
enum cudaEventFlags {
    cudaEventDefault       = 0x00,
    cudaEventBlockingSync  = 0x01,
    cudaEventDisableTiming = 0x02,
    cudaEventInterprocess  = 0x04,
};

extern "C" {

// ---- memory ------------------------------------------------------------------------------------------------
// Carve `count` bytes from the current stream's arena and return it as a device pointer.  A stream-less call
// (CUDA's "current device") uses the shim's current stream (see cuda_compat_set_stream below), or - when
// exactly one stream is live - that stream.  Refuses when there is no stream to allocate from.
cudaError_t cudaMalloc(void** devPtr, size_t count);
cudaError_t cudaFree(void* devPtr);
// HOST-VISIBLE | HOST_COHERENT region; *hostPtr is the mapped host address.  See the header note on what the
// device pointer from cudaHostGetDevicePointer therefore can and cannot be.
cudaError_t cudaHostAlloc(void** hostPtr, size_t count, unsigned int flags);
cudaError_t cudaHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
cudaError_t cudaFreeHost(void* hostPtr);
// The device layer's OWN budget: free = the driver's figure minus the desktop reserve (the number the port
// sizes against), total = the DEVICE_LOCAL heap.  Not a guess, and not a per-process usage figure.
cudaError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes);

// ---- copies (staged through the device layer; see the header note) -----------------------------------------
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                              size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

// ---- the fence ---------------------------------------------------------------------------------------------
// The submission fence (vacuous here - every submit already fenced; see the header note).
cudaError_t cudaDeviceSynchronize(void);
// Held for ABI completeness: this backend has no asynchronous submission to drain, so it is the same fence.
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

// ---- events (HOST-side wall-clock; see the header note) -----------------------------------------------------
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);

// ---- errors (the shim carries its OWN last error; see cuda_runtime.cpp) ------------------------------------
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t error);

}  // extern "C"

// ---- the shim's own seam (NOT part of CUDA) ----------------------------------------------------------------
namespace strata::vulkan {
struct Stream;

// The stream the stream-less CUDA calls (cudaMalloc/cudaMemcpy/cudaDeviceSynchronize/cudaHostAlloc) use, the
// stand-in for CUDA's process "current device".  The engine's program sets it once the device is up; when it is
// not set, a call that arrives with an explicit stream also sets it (the "most recently used stream"), which
// is what makes the layer body's stream-less `cudaMemcpy` uploads resolve to the stream the layer is driving.
// Documented rather than silent: this is the shim's approximation of CUDA's current-device model.
void cuda_compat_set_stream(Stream* s);
Stream* cuda_compat_current_stream();

}  // namespace strata::vulkan
