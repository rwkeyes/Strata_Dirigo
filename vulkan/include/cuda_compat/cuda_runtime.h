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
//   * The CUDA GRAPH API and stream capture (`cudaGraph*`, `cudaStreamBeginCapture`/`EndCapture`) are PROVIDED
//     over the port's OWN recorded step, not a second mechanism: a graph IS a recorded step (one persistent
//     command buffer, re-submitted and never re-recorded), and a capture diverts `dispatch` to record instead
//     of submit.  The semantics that CANNOT be honoured exactly - the multi-stream model, and host<->device
//     copies / memsets inside a capture, which this layer records as nothing because it has no pure device
//     command for them - are stated in full at the graph declarations below rather than approximated here.
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

// ---- the shim's CUDA version -------------------------------------------------------------------------------
// `CUDART_VERSION` is a toolkit MACRO.  The engine uses it two ways: in `#if CUDART_VERSION >= 120xx` guards
// (`src/core/verify.cpp` 12.3, `src/core/expert_cache.cpp` 12.5) and as a RUNTIME value in `generate.cpp`'s
// libcudart-mismatch check (`CUDART_VERSION / 1000` vs `cudaRuntimeGetVersion`).  There is no CUDA toolkit on
// this build, so the value is the shim's OWN: 12000 (CUDA 12.0).  Chosen so both `#if` guards stay FALSE (the
// same branch an undefined macro took, since an undefined identifier is 0 in `#if`), and so
// `cudaRuntimeGetVersion` can return the SAME constant - the mismatch check then cannot disagree with itself
// or print its "GPU properties can read wrong" warning, which describes a CUDA-toolkit defect that cannot
// exist here.  Stated rather than hidden: this is a version of the SHIM, not of a CUDA runtime.
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
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
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled  = 705,
};
typedef enum cudaError cudaError_t;

// ---- the opaque handles -----------------------------------------------------------------------------------
// The engine holds a `void* stream` and casts it to `cudaStream_t` (layer.cpp:232, :717, :1301 ...); on this
// backend that void* IS a `strata::vulkan::Stream*`.  `cudaEvent_t` is a shim-owned host-timing object.
struct cudaStream_st;
struct cudaEvent_st;
typedef struct cudaStream_st* cudaStream_t;
typedef struct cudaEvent_st* cudaEvent_t;

// ---- streams, capture and graphs (deliverable A of the approved hybrid route) ------------------------------
// THE ENGINE'S RECORDER (`src/core/session.cpp`, `src/core/mtp.cpp`, `src/core/verify.cpp`) drives the CUDA
// STREAM-CAPTURE + GRAPH API directly.  On this backend that API is answered over the port's OWN recorded step
// (`strata::vulkan::Ctx::record_*`, `vulkan/src/device/vk_compute.{hpp,cpp}`), NOT a second mechanism:
//
//   * A STREAM here is the one device + arena (`Stream*`), as everywhere else in this shim.  `cudaStreamCreate`
//     returns that stream rather than a new queue: this backend is ONE queue with synchronous submits, so the
//     engine's multi-stream concurrency (cross-stream capture, events between streams) is not expressible and a
//     capture is confined to the one device.  `cudaStreamDestroy` therefore releases only the shim's own handle,
//     never the device.
//   * A GRAPH is a recorded step: one persistent command buffer holding the dispatches (and device->device
//     copies) a capture body issued between `cudaStreamBeginCapture` and `cudaStreamEndCapture`, re-submitted by
//     every `cudaGraphLaunch` and never re-recorded.  `cudaGraphInstantiate` makes that recording launchable;
//     `cudaGraphLaunch` submits and waits it (the port's own fenced submit - a call that returns is a call whose
//     work has completed); `cudaGraphDestroy`/`cudaGraphExecDestroy` free it.
//
// THE CAPTURE IS A RECORDING, NOT AN EXECUTION.  While a capture is active the port's `Ctx::dispatch` RECORDS
// instead of submitting (exactly what CUDA's capture does to a launch).  A command buffer can only hold pure
// device commands, so the semantics this backend CANNOT honour exactly are stated here rather than hidden:
//
//   * A HOST<->DEVICE COPY INSIDE A CAPTURE IS REFUSED.  `cudaMemcpyAsync(H2D/D2H)` STAGES through host memory
//     in this shim (there is no device address space for host memory - see the header note on cudaHostAlloc), so
//     it has no pure device command to record.  CUDA records it as a memcpy node whose host source is re-read at
//     replay; this backend would have to refresh a staging image the host never wrote.  Rather than replay a
//     memset/copy of stale bytes, the capture is INVALIDATED and `cudaStreamEndCapture` returns
//     `cudaErrorStreamCaptureUnsupported`.  (The engine's shipped default avoids this: `g_publish_kernel` is
//     true, so the per-token step state reaches the device through the `copy_i32_from_mapped` KERNEL and a
//     mapped host pointer, not an H2D copy - the exact reason `layer.cpp:901` calls that upload "what keeps
//     this layer capturable".)
//   * A `cudaMemsetAsync` INSIDE A CAPTURE IS REFUSED for the same reason (this layer has no memset shader; the
//     shim fills through staging).  It is not reached in the shipped capture body.
//   * A DEVICE->DEVICE COPY IS RECORDED (`vkCmdCopyBuffer`), because it IS a pure device command.
//   * THE RECORDING BAKES BUFFER BINDINGS, NOT DATA.  Like CUDA (graph.hpp NOTE 2), a replayed graph re-reads the
//     CONTENTS of the buffers it bound at capture, but a pointer that the host changes between replays is not
//     seen - the recording still names the buffer it captured.  The engine's contract is fixed addresses
//     (`session.hpp`: the per-token counts live in buffers allocated before capture and never reallocated); this
//     backend holds that contract and cannot see a violation of it.
//   * `cudaGraphUpload` is a no-op: the recording is host-side command buffers, so there is nothing to upload.
//     `cudaGraphGetNodes`/`cudaGraphNodeGetType` report the recording's node COUNT and kinds (kernel / memcpy);
//     `cudaGraphKernelNodeGetParams` is `cudaErrorNotSupported` (the shim keeps no CUDA kernel params - the
//     engine reads this only under `STRATA_VERIFY_NODES`).
struct cudaGraph_st;
struct cudaGraphExec_st;
struct cudaGraphNode_st;
typedef struct cudaGraph_st* cudaGraph_t;
typedef struct cudaGraphExec_st* cudaGraphExec_t;
typedef struct cudaGraphNode_st* cudaGraphNode_t;

// Stream creation flags (toolkit values).  `cudaStreamNonBlocking` is accepted; this backend has one queue, so
// the flag changes nothing - stated rather than silently dropped.
enum cudaStreamFlags {
    cudaStreamDefault     = 0x00,
    cudaStreamNonBlocking = 0x01,
};

// Capture mode.  All three are accepted and all three behave the same here: this backend cannot capture across
// streams (there is one), so the thread-local vs global distinction has nothing to distinguish.
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2,
};

// What a recorded node is.  The shim produces kernel (dispatch) and memcpy (device->device copy) nodes.
enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel      = 0,
    cudaGraphNodeTypeMemcpy      = 1,
    cudaGraphNodeTypeMemset      = 2,
    cudaGraphNodeTypeHost        = 3,
    cudaGraphNodeTypeGraph       = 4,
    cudaGraphNodeTypeEmpty       = 5,
    cudaGraphNodeTypeWaitEvent   = 6,
    cudaGraphNodeTypeEventRecord = 7,
};

// The node-params shape the engine's diagnostic names (`verify.cpp`).  This shim keeps no CUDA kernel params,
// so `cudaGraphKernelNodeGetParams` answers `cudaErrorNotSupported` rather than a plausible-looking zero.
struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
};
struct cudaKernelNodeParams {
    void* func = nullptr;
    struct dim3 gridDim;
    struct dim3 blockDim;
    unsigned int sharedMemBytes = 0;
    void** kernelParams = nullptr;
    void** extra = nullptr;
};

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

// ---- THE MULTI-GPU / PINNED-HOST / CUDA-DRIVER SURFACE ------------------------------------------------
// The engine's PEER path (`peer_experts.cpp`, `remote_experts.cpp`) and its expert pinning
// (`expert_source.cpp`) and its VMM probe (`expert_cache.cpp`) call a CUDA surface this backend does not
// implement, because it is ONE Vulkan device with no peer access and no CUDA driver API.  These are declared
// so the engine's TUs COMPILE (and so the link bar can be measured), and each is a LOUD REFUSAL rather than a
// plausible success - a silent "peer access enabled" on a single-GPU Vulkan device would pick the multi-GPU
// path and read the wrong device.
enum cudaHostRegisterFlags {
    cudaHostRegisterDefault   = 0x00,
    cudaHostRegisterPortable  = 0x01,
    cudaHostRegisterMapped    = 0x02,
    cudaHostRegisterIoMemory  = 0x04,
};
// cudaDeviceSetFlags / cudaSetDeviceFlags values (toolkit values), used by `cudaInitDevice`.
enum cudaDeviceFlags {
    cudaDeviceScheduleAuto  = 0x00,
    cudaDeviceScheduleSpin  = 0x01,
    cudaDeviceScheduleYield = 0x02,
    cudaDeviceScheduleBlockingSync = 0x04,
    cudaDeviceMapHost       = 0x08,
    cudaDeviceLmemResizeToMax = 0x10,
};
// The `cudaGetDriverEntryPoint` result set (toolkit values).
enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess                = 0,
    cudaDriverEntryPointSymbolNotFound         = 1,
    cudaDriverEntryPointVersionNotSufficent    = 2,
};
enum { cudaEnableDefault = 0x00, cudaEnableLazyLoad = 0x01 };
// Event-create flags (the shim's events carry no device timing; the flag is accepted, not honoured differently
// - see `cudaEventElapsedTime` above).
enum cudaEventFlags {
    cudaEventDefault       = 0x00,
    cudaEventBlockingSync  = 0x01,
    cudaEventDisableTiming = 0x02,
    cudaEventInterprocess  = 0x04,
};

// ============================================================================================================
// THE DEVICE-PROPERTY SURFACE - AND THE HONEST ANSWER FOR WHAT A VULKAN DEVICE DOES NOT HAVE
// ============================================================================================================
//
// `src/program/generate.cpp` asks CUDA for the device's properties and for two attributes (the SM count and the
// clock rate) to size its multi-GPU `--layer-split` placement heuristic.  A Vulkan device is not a CUDA device
// and exposes NEITHER an SM count NOR a clock rate - so the answer matters, and a lazy 1 or 0 that feeds a
// heuristic is exactly the silently-wrong-number defect class this port has already paid for (a fabricated
// figure reads as a real one forever).  What is reported, and why:
//
//   * `cudaGetDeviceProperties(&prop, 0)`:
//       - `prop.name`           = the real `VkPhysicalDeviceProperties.deviceName` (e.g. "Intel(R) Arc(TM) Pro
//                                 B70 Graphics").  The engine PRINTS it and uses it in the code-for-this-GPU
//                                 error message (`generate.cpp:3083/3096/3111`).
//       - `prop.totalGlobalMem` = the device's real `DEVICE_LOCAL` heap total (the same figure `cudaMemGetInfo`
//                                 reports, from `VK_EXT_memory_budget`).  The engine uses it for the layer-split
//                                 reserve check (`generate.cpp:1668`, `reserve_mib * 100 > 12 * total_mib`).
//       - `prop.major`/`minor`  = 0.  A Vulkan device has NO CUDA compute capability; 0 is the honest value and
//                                 is PRINTED (`generate.cpp:3096`, "compute capability 0.0").  It can also make
//                                 the engine refuse to run (`device_code_error()`): that is the engine deciding
//                                 it has no code for a card it cannot name, which is correct behaviour on a
//                                 device that is not a CUDA card, not a Vulkan-port failure.
//       - `prop.multiProcessorCount` / `prop.clockRate` = 0, with the same reasoning as the attributes below
//                                 (the fields exist for ABI shape; the engine does not read them - it calls
//                                 `cudaDeviceGetAttribute` for the two, which REFUSES).
//
//   * `cudaDeviceGetAttribute(v, attr, 0)`:
//       - `cudaDevAttrComputeCapabilityMajor` / `...Minor` -> 0 (success): no CUDA compute capability exists.
//       - `cudaDevAttrMultiProcessorCount` and `cudaDevAttrClockRate` -> **`cudaErrorInvalidValue`**: a Vulkan
//         device exposes no SM count and no clock rate, and there is no Vulkan query that answers either (the
//         VkPhysicalDeviceProperties limits describe maximum workgroup geometry, not a machine width).  Refusing
//         is the port's rule; inventing a number is the defect.
//         WHAT THE ENGINE DOES WITH IT: the ONLY consumer is `generate.cpp:2759-2766`, the multi-GPU
//         `--layer-split` AUTO placement heuristic (`speed = std::max(1.0, sms * khz / 1e6)`).  That block runs
//         only when `multi_gpu && split_auto`, and this backend reports ONE device
//         (`cudaGetDeviceCount() == 1`) with no peer access, so `multi_gpu` is false and the block is NOT
//         ENTERED.  Even if it were, the engine's own guards bound it: it ignores this call's return and uses
//         the `std::max(1.0, ...)` floor, and it already treats a REFUSED clock as 1.8 GHz
//         (`generate.cpp:2760`: `if (cudaDeviceGetAttribute(&khz, ...) != cudaSuccess || khz <= 0) khz = 1800000`).
//         So a refusal here degrades a heuristic to its documented floor, never a kernel's geometry.
//       - every other attribute (shared memory, warp size, ...) -> 0 (success): those describe a CUDA SIMT
//         machine this backend is not, and the engine's prefill TU that reads them is not compiled here.
//
// `cudaMallocHost` is CUDA's `cudaHostAlloc(..., cudaHostAllocDefault)`; see the cudaHostAlloc note above for
// what the device pointer it produces therefore is and is not.
struct cudaDeviceProp {
    char name[256];                       // the Vulkan deviceName
    size_t totalGlobalMem;                // the DEVICE_LOCAL heap total (real; not a model budget)
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;                        // 0: a Vulkan device has no clock rate (see the note above)
    size_t totalConstMem;
    int major, minor;                     // 0/0: no CUDA compute capability on a Vulkan device
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;              // 0: a Vulkan device has no SM count (see the note above)
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID, pciDeviceID, pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate, memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    char gcnArchName[256];                // HIP only; left empty here (the HIP branches are not compiled)
    int reservedSharedMemPerBlock;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
};

// The attribute ids the engine names, at their toolkit VALUES (so an engine comparison against a numeric literal
// would still mean what it meant).  See the note above for which two are refused.
enum cudaDeviceAttr {
    cudaDevAttrMaxSharedMemoryPerBlock          = 8,
    cudaDevAttrWarpSize                         = 10,
    cudaDevAttrClockRate                        = 13,
    cudaDevAttrMultiProcessorCount              = 16,
    cudaDevAttrIntegrated                       = 18,
    cudaDevAttrComputeCapabilityMajor           = 75,
    cudaDevAttrComputeCapabilityMinor           = 76,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrCooperativeLaunch                = 95,
    cudaDevAttrMaxSharedMemoryPerBlockOptin     = 97,
    cudaDevAttrReservedSharedMemoryPerBlock     = 111,
};

extern "C" {

// ---- memory ------------------------------------------------------------------------------------------------
// Carve `count` bytes from the current stream's arena and return it as a device pointer.  A stream-less call
// (CUDA's "current device") uses the shim's current stream (see cuda_compat_set_stream below), or - when
// exactly one stream is live - that stream.  Refuses when there is no stream to allocate from.
cudaError_t cudaMalloc(void** devPtr, size_t count);
cudaError_t cudaFree(void* devPtr);
// CUDA's `cudaMallocHost` == `cudaHostAlloc(..., cudaHostAllocDefault)`: a page-locked host region and (here) the
// HOST_VISIBLE | HOST_COHERENT buffer behind it.  See the cudaHostAlloc note for what its device pointer is.
cudaError_t cudaMallocHost(void** ptr, size_t count);
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
// The `stream` defaults to the shim's current stream, exactly as the toolkit's own `cudaStream_t stream = 0`
// default reads (the engine calls the 4-argument form with no stream at `generate.cpp:1170`).
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream = nullptr);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                              size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = nullptr);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = nullptr);

// ---- the fence ---------------------------------------------------------------------------------------------
// The submission fence (vacuous here - every submit already fenced; see the header note).
cudaError_t cudaDeviceSynchronize(void);
// Held for ABI completeness: this backend has no asynchronous submission to drain, so it is the same fence.
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

// ---- events (HOST-side wall-clock; see the header note) -----------------------------------------------------
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);
// A QUERY, never a blocking sync (the engine's host loop spins on it): since every submit in this backend
// already fenced and waited, a RECORDED event is complete the moment this is asked - cudaSuccess.  An event that
// was never recorded answers cudaErrorNotReady, as CUDA's does.
cudaError_t cudaEventQuery(cudaEvent_t event);
// Cross-stream ordering.  This backend has ONE queue and every submit is already complete when its call
// returns, so there is no pending work for a wait to order against: the wait is satisfied the moment it is
// asked.  Accepted and a no-op - stated here rather than silently dropped.
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

// ---- the current-device model (one device) -----------------------------------------------------------------
// The engine asks CUDA for the current device index and switches between devices (the multi-GPU peer path).
// This backend has ONE device, so it answers index 0 and REFUSES any other, rather than pretending to switch.
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
// The device's properties.  `device` must be 0 (the one device); any other is `cudaErrorInvalidDevice`.  See the
// device-property note above for exactly what each field holds and, crucially, which two are zero because a
// Vulkan device has no such quantity.
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
// One device attribute.  `cudaDevAttrMultiProcessorCount` and `cudaDevAttrClockRate` are REFUSED
// (`cudaErrorInvalidValue`) - see the note above for why, and for what the engine does with the answer.
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
// The shim's CUDART_VERSION (12000).  There is no CUDA runtime to ask, so the answer is the compile-time
// constant this shim defines; the engine's header-vs-runtime mismatch check therefore cannot fire.
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);

// ---- host callbacks enqueued on a stream -------------------------------------------------------------------
// CUDA runs `fn(userData)` on the stream after the work already queued.  Every submit here is complete when its
// call returns, so "after the queued work" IS the calling thread: the callback runs INLINE (ordered exactly as
// CUDA would order it).  Inside a capture it is refused like any other op this layer cannot record.
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData);

// ---- errors (the shim carries its OWN last error; see cuda_runtime.cpp) ------------------------------------
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t error);

// ---- streams (see the capture/graph note above: one queue, so a stream is the device) -----------------------
// Returns the shim's current stream (the one device).  When none is set it opens one from
// STRATA_VK_SPV_DIR / the default shaders directory, so a program that creates its streams before any other
// call is not left without a device.  Refuses (cudaErrorInitializationError) if that fails.
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
// Releases the shim's handle only; the device (and its arena) live until the process closes them.  A no-op on
// the current stream, stated rather than hidden.
cudaError_t cudaStreamDestroy(cudaStream_t stream);
// A QUERY, never a blocking sync: every submit in this backend already fenced and waited, so work is complete
// by the time any call returns and the answer is the completed state.
cudaError_t cudaStreamQuery(cudaStream_t stream);

// ---- stream capture (records into the port's recorded step; see the note above) ----------------------------
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
// On success `graph` owns the recording (the stream stops capturing).  A capture that recorded NOTHING, or that
// hit an op this backend cannot record (a host<->device copy, a memset), leaves `*graph` null and returns
// cudaErrorStreamCaptureUnsupported.
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);

// ---- graphs -------------------------------------------------------------------------------------------------
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec);
// No-op on this backend (the recording is host-side command buffers).
cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t stream);
// The recording's node count.  `nodes` may be null (CUDA's own two-call pattern); otherwise up to `*n` handles
// are written and `*n` is set to the total.
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* type);
// cudaErrorNotSupported: this shim keeps no CUDA kernel parameters (read only under STRATA_VERIFY_NODES).
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* params);

// ---- the multi-GPU / pinned-host / CUDA-driver surface (see the note above; each is a LOUD REFUSAL) ----------
// cudaHostRegister has no effect this backend can honour: a host pointer is NOT addressable by a shader here (see
// the cudaHostAlloc note), so pinning it for device access would be a lie.  cudaErrorNotSupported; the caller
// (`expert_source.cpp`) checks and falls back.
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
// ONE device.
cudaError_t cudaGetDeviceCount(int* count);
// NO peer access exists between one device and anything: answer 0, and refuse to enable it.
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
// `cudaInitDevice`'s flags (spin scheduling, mapped host) describe a CUDA runtime this backend does not have;
// device 0 is accepted (there is one device), any other refused.
cudaError_t cudaInitDevice(int device, unsigned int flags, unsigned int flags2);
// The CUDA DRIVER API entry points (`cuMemCreate`, ...) do not exist on a Vulkan device, so the symbol is
// reported NOT FOUND - the honest answer, and the one `expert_cache.cpp`'s VMM probe already handles by falling
// back.
cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                    enum cudaDriverEntryPointQueryResult* driverStatus);

}  // extern "C"

// ---- cudaGraphInstantiate: TWO signatures, as the toolkit has (the engine calls both) -----------------------
// `session.cpp`/`mtp.cpp`/`verify.cpp` call the 3-argument form `cudaGraphInstantiate(&exec, graph, 0)`; four
// parity TUs call the 5-argument form with a null error node and log.  C++ linkage (not extern "C") so both can
// exist; the engine's TUs are C++.
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, cudaGraphNode_t* errorNode,
                                 char* logBuffer, size_t bufferSize);

// ---- the TYPED-pointer cudaMalloc (C++, non-extern-"C") -----------------------------------------------------
// The toolkit declares `cudaMalloc` ONLY as `void**`.  The engine's own code is written as if it were typed -
// `float* d; cudaMalloc(&d, n)` with no `(void**)` cast (`generate.cpp:3039/3875/3884`, and 39 more) - because
// on the real toolkit `float**`->`void**` is accepted under the compiler's permissive rule that an engine build
// already relies on.  Rather than require a cast the engine does not write, this shim provides the typed
// overload the call sites imply.  The `void**` overload above still wins for an explicit `(void**)` cast (a
// non-template is preferred over a template on an exact match), so no existing call changes meaning.
template <typename T>
inline cudaError_t cudaMalloc(T** devPtr, size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(devPtr), count);
}

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
