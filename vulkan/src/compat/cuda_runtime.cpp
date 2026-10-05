// vulkan/src/compat/cuda_runtime.cpp - the CUDA-runtime compatibility shim's IMPLEMENTATION over the port's
// Vulkan device layer.  The semantics, and every place CUDA could not be honoured exactly, are stated in the
// companion header `vulkan/include/cuda_compat/cuda_runtime.h` - read that first; this file only realises it.
//
// It is deliberately thin: every operation is one of the device layer's own primitives (arena_alloc, the
// pointer->buffer resolution, the STAGING read/write, the fenced submit), so there is no second memory model
// and no second ordering model.  What CUDA called directly, this file expresses in the arena.
#include "cuda_runtime.h"                 // this shim's own declarations (found via -Ivulkan/include/cuda_compat)

#include "vk_arena.hpp"                   // Stream, arena_alloc, arena_resolve, stream_write/stream_read
#include "strata/vulkan/vk_backend.hpp"   // stream_of: the stream-handle registry check

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using strata::vulkan::Buf;
using strata::vulkan::Stream;

// THE CURRENT STREAM - the shim's stand-in for CUDA's process-wide "current device".  A stream-less call
// (cudaMalloc/cudaMemcpy/cudaDeviceSynchronize/cudaHostAlloc) uses it; a call carrying a stream also BINDS it
// (the "most recently used stream" rule, documented in the header).
Stream* g_current = nullptr;

// THE SHIM'S OWN LAST ERROR.  Real, not a constant: every refusal below sets it, and cudaPeekAtLastError /
// cudaGetLastError carry it (see the header).  thread_local, as CUDA's is per-thread.
thread_local cudaError_t g_last = cudaSuccess;
cudaError_t fail(cudaError_t e) { g_last = e; return e; }

// ---- host regions: what cudaHostAlloc handed out -----------------------------------------------------------
struct HostRegion {
    void* host = nullptr;          // the mapped host address returned to the caller
    uint64_t bytes = 0;
    Buf buf{};                     // the HOST_VISIBLE | HOST_COHERENT buffer behind it
    Stream* stream = nullptr;      // the stream whose ctx owns it (for cudaFreeHost)
};
std::vector<HostRegion>& host_regions() {
    static std::vector<HostRegion> v;
    return v;
}
HostRegion* host_region_of(const void* p) {
    if (p == nullptr) return nullptr;
    for (HostRegion& r : host_regions())
        if (r.host == p) return &r;
    return nullptr;
}

Stream* current_or(Stream* s) {
    if (s != nullptr) g_current = s;      // bind the most recently used stream
    return s != nullptr ? s : g_current;
}

// ---- one transfer, expressed in the device layer's primitives ----------------------------------------------
// Every device pointer is resolved to an arena view first; anything not in the arena is a HOST pointer (either
// one this shim's cudaHostAlloc handed out, or engine host memory the caller is uploading/downloading).
// `count == 0` is success (CUDA's contract).  A device->device pair is STAGED through the host: this device
// layer exposes no device-side copy primitive, so two transfers are the honest realisation, not one.
cudaError_t transfer(Stream* s, void* dst, const void* src, uint64_t count) {
    if (count == 0) { g_last = cudaSuccess; return cudaSuccess; }
    if (s == nullptr)
        return fail(cudaErrorInvalidValue);

    Buf dv{}, sv{};
    const bool d_arena = strata::vulkan::arena_resolve(*s, dst, count, dv);
    const bool s_arena = strata::vulkan::arena_resolve(*s, src, count, sv);

    // ---- CAPTURE: record what is recordable, refuse what is not (see the header note) ------------------------
    // A device->device copy IS a pure device command, so a capture RECORDS it (vkCmdCopyBuffer) instead of
    // executing it.  A host end makes it a host<->device transfer, which this shim stages through host memory
    // and therefore cannot record as a device command: rather than replay stale bytes, INVALIDATE the capture
    // so cudaStreamEndCapture refuses loudly.
    if (s->ctx != nullptr && s->ctx->capturing()) {
        if (d_arena && s_arena) {
            s->ctx->capture_copy(dv, sv, count);
            g_last = cudaSuccess;
            return cudaSuccess;
        }
        s->ctx->capture_invalidate();
        return fail(cudaErrorStreamCaptureUnsupported);
    }

    HostRegion* d_host = d_arena ? nullptr : host_region_of(dst);
    HostRegion* s_host = s_arena ? nullptr : host_region_of(src);

    const void* host_src = (s_host != nullptr) ? s_host->buf.mapped : src;
    void* host_dst = (d_host != nullptr) ? d_host->buf.mapped : dst;

    if (d_arena && s_arena) {                       // device -> device, staged through the host
        std::vector<uint8_t> tmp((size_t) count);
        strata::vulkan::stream_read(*s, src, tmp.data(), count);
        strata::vulkan::stream_write(*s, dst, tmp.data(), count);
    } else if (d_arena) {                           // host -> device
        strata::vulkan::stream_write(*s, dst, host_src, count);
    } else if (s_arena) {                           // device -> host
        strata::vulkan::stream_read(*s, src, host_dst, count);
    } else {                                        // host -> host (incl. mapped host -> mapped host)
        std::memcpy(host_dst, host_src, (size_t) count);
    }
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t transfer_2d(Stream* s, void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height) {
    if (s == nullptr) return fail(cudaErrorInvalidValue);
    for (size_t row = 0; row < height; ++row) {
        const cudaError_t e = transfer(s, static_cast<uint8_t*>(dst) + row * dpitch,
                                       static_cast<const uint8_t*>(src) + row * spitch, width);
        if (e != cudaSuccess) return e;
    }
    g_last = cudaSuccess;
    return cudaSuccess;
}

}  // namespace

// ---- the graph objects: a recorded step the shim owns (see the header's capture/graph note) ----------------
// A `cudaGraph_t` is a finished recording; `cudaGraphExec_t` is the same recording made launchable.  Both hold
// the port's own recording handles (one persistent command buffer + one fence, submitted and waited by the
// device layer's `submit_owned`) and the `Ctx` that owns them.
// A recorded node handle.  The opaque handles the header forward-declares must be at GLOBAL scope (their names
// are what `cudaGraph_t`/`cudaGraphNode_t` point at), so these three are not in a namespace.
struct cudaGraphNode_st {
    cudaGraphNodeType type = cudaGraphNodeTypeKernel;
};

struct cudaGraph_st {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    strata::vulkan::Ctx* owner = nullptr;
    uint32_t dispatches = 0;
    uint32_t copies = 0;
    bool taken = false;                     // instantiate moved the recording to the exec
    std::vector<cudaGraphNode_st> node_store;
};

struct cudaGraphExec_st {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    strata::vulkan::Ctx* owner = nullptr;
    uint32_t dispatches = 0;
    uint32_t copies = 0;
};

namespace {

// The portable `cudaGraphInstantiate` body, shared by the two signatures.
cudaError_t instantiate_impl(cudaGraphExec_t* exec, cudaGraph_t graph) {
    if (exec == nullptr) return fail(cudaErrorInvalidValue);
    *exec = nullptr;
    if (graph == nullptr || graph->cb == VK_NULL_HANDLE || graph->taken)
        return fail(cudaErrorInvalidValue);
    cudaGraphExec_st* e = new cudaGraphExec_st();
    e->cb = graph->cb;
    e->fence = graph->fence;
    e->owner = graph->owner;
    e->dispatches = graph->dispatches;
    e->copies = graph->copies;
    graph->cb = VK_NULL_HANDLE;
    graph->fence = VK_NULL_HANDLE;
    graph->taken = true;
    *exec = e;
    g_last = cudaSuccess;
    return cudaSuccess;
}

}  // namespace

// ---- the shim's own seam -----------------------------------------------------------------------------------
namespace strata::vulkan {
void cuda_compat_set_stream(Stream* s) { g_current = s; }
Stream* cuda_compat_current_stream() { return g_current; }
}  // namespace strata::vulkan

extern "C" {

// ---- memory ------------------------------------------------------------------------------------------------
cudaError_t cudaMalloc(void** devPtr, size_t count) {
    if (devPtr == nullptr) return fail(cudaErrorInvalidValue);
    Stream* s = current_or(nullptr);
    if (s == nullptr)
        return fail(cudaErrorInvalidValue);   // no current stream: nothing to carve from
    *devPtr = strata::vulkan::arena_alloc(*s, count == 0 ? 1 : count);
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    // The arena is a BUMP allocator: memory returns to the arena when the stream closes, never here.  A
    // pointer that is not in the arena is refused rather than ignored (the port's rule).
    if (devPtr == nullptr) { g_last = cudaSuccess; return cudaSuccess; }
    Stream* s = current_or(nullptr);
    if (s == nullptr) return fail(cudaErrorInvalidValue);
    Buf v{};
    if (!strata::vulkan::arena_resolve(*s, devPtr, 1, v)) return fail(cudaErrorInvalidValue);
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void** hostPtr, size_t count, unsigned int flags) {
    (void) flags;      // Mapped/Portable/WriteCombined do not change what this device layer can offer
    if (hostPtr == nullptr) return fail(cudaErrorInvalidValue);
    Stream* s = current_or(nullptr);
    if (s == nullptr || s->ctx == nullptr) return fail(cudaErrorInvalidValue);
    Buf b = s->ctx->alloc(count == 0 ? 1 : count);      // HOST_VISIBLE | HOST_COHERENT (Ctx::alloc's type)
    if (b.mapped == nullptr)
        return fail(cudaErrorMemoryAllocation);         // the device offered no mappable type: refuse
    HostRegion r;
    r.host = b.mapped;
    r.bytes = count;
    r.buf = b;
    r.stream = s;
    host_regions().push_back(r);
    *hostPtr = r.host;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags) {
    (void) flags;
    if (devPtr == nullptr) return fail(cudaErrorInvalidValue);
    // There is NO separate device address space for host memory in this layer (see the header): the mapped
    // host address IS the token the shim's copies accept.  A pointer this shim did not hand out is refused,
    // so a caller cannot get a token for bytes the shim cannot stage.
    if (host_region_of(hostPtr) == nullptr) return fail(cudaErrorInvalidValue);
    *devPtr = hostPtr;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* hostPtr) {
    HostRegion* r = host_region_of(hostPtr);
    if (r == nullptr) return fail(cudaErrorInvalidValue);
    Stream* s = r->stream;
    if (s != nullptr && s->ctx != nullptr) s->ctx->free(r->buf);
    for (size_t i = 0; i < host_regions().size(); ++i) {
        if (&host_regions()[i] == r) {
            host_regions()[i] = host_regions().back();
            host_regions().pop_back();
            break;
        }
    }
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void** ptr, size_t count) {
    // CUDA's cudaMallocHost IS cudaHostAlloc with the default flags; realised here as the same call so the two
    // cannot drift (the engine uses both - `peer_experts.cpp` cudaHostAlloc, `generate.cpp:1164` cudaMallocHost).
    return cudaHostAlloc(ptr, count, cudaHostAllocDefault);
}

cudaError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    Stream* s = current_or(nullptr);
    if (s == nullptr || s->ctx == nullptr) return fail(cudaErrorInvalidValue);
    // The device's OWN figures: `usable_bytes()` is the driver's free HEAP figure (VK_EXT_memory_budget, the
    // whole-heap usage) minus the desktop reserve - the number the port sizes against; the total is the
    // DEVICE_LOCAL heap.  Honest about what they are; NOT this process's allocation.
    if (freeBytes != nullptr) *freeBytes = (size_t) s->ctx->usable_bytes();
    if (totalBytes != nullptr) *totalBytes = (size_t) s->ctx->info().heap_device_local_bytes;
    g_last = cudaSuccess;
    return cudaSuccess;
}

// ---- copies ------------------------------------------------------------------------------------------------
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    (void) kind;   // every kind resolves through `transfer`'s arena/host classification
    return transfer(current_or(nullptr), dst, src, count);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
    (void) kind;
    return transfer(current_or(reinterpret_cast<Stream*>(stream)), dst, src, count);
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                              size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    (void) kind;
    return transfer_2d(current_or(reinterpret_cast<Stream*>(stream)), dst, dpitch, src, spitch, width, height);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    return cudaMemsetAsync(devPtr, value, count, reinterpret_cast<cudaStream_t>(g_current));
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    Stream* s = current_or(reinterpret_cast<Stream*>(stream));
    if (s == nullptr) return fail(cudaErrorInvalidValue);
    if (count == 0) { g_last = cudaSuccess; return cudaSuccess; }
    // Inside a capture there is no pure device memset command (this layer has no memset shader; the fill below
    // stages through the host), so the capture is invalidated and the caller refuses - see the header note.
    if (s->ctx != nullptr && s->ctx->capturing()) {
        s->ctx->capture_invalidate();
        return fail(cudaErrorStreamCaptureUnsupported);
    }
    // A STAGED FILL: this device layer has no memset shader, so the byte is replicated in a host buffer and
    // uploaded through the same staging path every other host->device transfer uses.
    std::vector<uint8_t> fill((size_t) count, (uint8_t) (value & 0xff));
    return transfer(s, devPtr, fill.data(), count);
}

// ---- the fence ---------------------------------------------------------------------------------------------
cudaError_t cudaDeviceSynchronize(void) {
    // Every Ctx::dispatch submits with a fence and WAITS it, so nothing is in flight here: a real barrier,
    // vacuously satisfied (see the header).  It is NOT a wait for asynchronous work, because there is none.
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (stream != nullptr) g_current = reinterpret_cast<Stream*>(stream);
    g_last = cudaSuccess;
    return cudaSuccess;
}

// ---- events (HOST-side wall-clock; see the header) ---------------------------------------------------------
struct cudaEvent_st {
    std::chrono::steady_clock::time_point t{};
    bool recorded = false;
};

cudaError_t cudaEventCreate(cudaEvent_t* event) { return cudaEventCreateWithFlags(event, 0); }

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    (void) flags;      // no device timing exists to disable
    if (event == nullptr) return fail(cudaErrorInvalidValue);
    *event = new cudaEvent_st();
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (stream != nullptr) g_current = reinterpret_cast<Stream*>(stream);
    if (event == nullptr) return fail(cudaErrorInvalidValue);
    event->t = std::chrono::steady_clock::now();
    event->recorded = true;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (event == nullptr) return fail(cudaErrorInvalidValue);
    g_last = cudaSuccess;
    return cudaSuccess;      // the timestamp is already host-side: there is nothing to wait for
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) return fail(cudaErrorInvalidValue);
    const std::chrono::duration<float, std::milli> d = end->t - start->t;
    *ms = d.count();
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (event == nullptr) return fail(cudaErrorInvalidValue);
    // Every submit in this backend already fenced and waited, so a recorded event is complete by the time this
    // returns.  A never-recorded event is not ready, as CUDA's is.
    if (!event->recorded) return fail(cudaErrorNotReady);
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void) flags;
    if (stream != nullptr) {
        Stream* s = strata::vulkan::stream_of(reinterpret_cast<void*>(stream));
        if (s == nullptr) return fail(cudaErrorInvalidValue);
        g_current = s;
    }
    if (event == nullptr) return fail(cudaErrorInvalidValue);
    g_last = cudaSuccess;      // one synchronous queue: there is no pending work for a wait to order against
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
    if (device == nullptr) return fail(cudaErrorInvalidValue);
    *device = 0;               // the one device
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    if (device != 0) return fail(cudaErrorInvalidDevice);   // one device: refuse to pretend to switch
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
    if (prop == nullptr) return fail(cudaErrorInvalidValue);
    if (device != 0) return fail(cudaErrorInvalidDevice);   // one device
    Stream* s = current_or(nullptr);
    // NO STREAM, NO DEVICE TO DESCRIBE: still answer with what is true of the machine (0/empty), because the
    // engine calls this before its first stream exists in one path (`generate.cpp:1666`, before the arena).
    std::memset(prop, 0, sizeof(*prop));
    if (s != nullptr && s->ctx != nullptr) {
        const strata::vulkan::DeviceInfo& di = s->ctx->info();
        std::snprintf(prop->name, sizeof(prop->name), "%s", di.name.c_str());
        prop->totalGlobalMem = (size_t) di.heap_device_local_bytes;   // the real DEVICE_LOCAL heap total
    } else {
        std::snprintf(prop->name, sizeof(prop->name), "(no Vulkan stream open)");
    }
    // major/minor, multiProcessorCount and clockRate stay 0 - a Vulkan device has none of the four; see the
    // header's device-property note for what the engine does with each.
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device) {
    if (value == nullptr) return fail(cudaErrorInvalidValue);
    if (device != 0) return fail(cudaErrorInvalidDevice);
    switch (attr) {
    case cudaDevAttrMultiProcessorCount:
    case cudaDevAttrClockRate:
        // A Vulkan device exposes neither an SM count nor a clock rate.  REFUSE rather than fabricate: the one
        // consumer is the multi-GPU layer-split heuristic (unreachable on this one-device backend).  See the
        // header's device-property note.
        return fail(cudaErrorInvalidValue);
    case cudaDevAttrComputeCapabilityMajor:
    case cudaDevAttrComputeCapabilityMinor:
        *value = 0;                 // no CUDA compute capability exists on a Vulkan device - the honest value
        break;
    default:
        *value = 0;                 // no CUDA SIMT geometry to report; the prefill TU that reads these is not built
        break;
    }
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    if (runtimeVersion == nullptr) return fail(cudaErrorInvalidValue);
    // There is no CUDA runtime here; the answer is this shim's own constant, so the engine's header-vs-runtime
    // mismatch check cannot fire (see the CUDART_VERSION note in the header).
    *runtimeVersion = CUDART_VERSION;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData) {
    Stream* s = (stream != nullptr) ? strata::vulkan::stream_of(reinterpret_cast<void*>(stream)) : g_current;
    if (s == nullptr) return fail(cudaErrorInvalidValue);
    g_current = s;
    if (s->ctx != nullptr && s->ctx->capturing()) {   // not recordable as a device command
        s->ctx->capture_invalidate();
        return fail(cudaErrorStreamCaptureUnsupported);
    }
    if (fn != nullptr) fn(userData);    // prior work is complete: inline IS "after the queued work"
    g_last = cudaSuccess;
    return cudaSuccess;
}

// ---- the multi-GPU / pinned-host / CUDA-driver surface (see the header note; each is a LOUD REFUSAL) -------
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    (void) ptr; (void) size; (void) flags;
    // A host pointer is not shader-addressable on this backend (the arena is DEVICE_LOCAL and, on a discrete
    // card, unmappable), so "pinning" it for device access would be a lie.  Refuse; the caller falls back.
    return fail(cudaErrorNotSupported);
}

cudaError_t cudaHostUnregister(void* ptr) {
    (void) ptr;                 // nothing was ever registered: succeed so cleanup paths do not report an error
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int* count) {
    if (count == nullptr) return fail(cudaErrorInvalidValue);
    *count = 1;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    if (canAccessPeer == nullptr) return fail(cudaErrorInvalidValue);
    if (device != 0 || peerDevice != 0) return fail(cudaErrorInvalidDevice);
    *canAccessPeer = 0;         // one device: nothing to have peer access to
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void) peerDevice; (void) flags;
    return fail(cudaErrorNotSupported);
}

cudaError_t cudaInitDevice(int device, unsigned int flags, unsigned int flags2) {
    (void) flags; (void) flags2;
    if (device != 0) return fail(cudaErrorInvalidDevice);
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                    enum cudaDriverEntryPointQueryResult* driverStatus) {
    (void) symbol; (void) flags;
    if (funcPtr != nullptr) *funcPtr = nullptr;
    if (driverStatus != nullptr) *driverStatus = cudaDriverEntryPointSymbolNotFound;
    return fail(cudaErrorNotSupported);   // there is no CUDA driver API on a Vulkan device
}

// ---- streams, capture and graphs (see the header's closure note) -------------------------------------------
cudaError_t cudaStreamCreate(cudaStream_t* stream) { return cudaStreamCreateWithFlags(stream, 0); }

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    (void) flags;      // one queue: cudaStreamNonBlocking has nothing to change
    if (stream == nullptr) return fail(cudaErrorInvalidValue);
    Stream* s = g_current;
    if (s == nullptr) {
        // No device is up yet: open one the way the port's own smokes do, from STRATA_VK_SPV_DIR (the engine's
        // program sets it) or the default shaders directory.  Refuse rather than hand back a stream with no
        // arena, which would fail later at a confusing call site.
        const char* dir = std::getenv("STRATA_VK_SPV_DIR");
        const std::string spv = (dir != nullptr && *dir != '\0') ? dir : "ports/vulkan/shaders";
        s = strata::vulkan::stream_open(64ull << 20, spv);
        if (s == nullptr) return fail(cudaErrorInitializationError);
        g_current = s;
    }
    *stream = reinterpret_cast<cudaStream_t>(s);
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    // The stream IS the one device + arena; it is not ours to tear down here (the program closes it), so this
    // releases only the handle the caller held.  Stated in the header rather than silent.
    (void) stream;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    if (stream != nullptr) {
        Stream* s = strata::vulkan::stream_of(reinterpret_cast<void*>(stream));
        if (s == nullptr) return fail(cudaErrorInvalidValue);
        g_current = s;
    }
    g_last = cudaSuccess;      // every submit already fenced and waited: work is complete
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
    (void) mode;      // one queue, so the thread-local/global distinction has nothing to distinguish
    Stream* s = strata::vulkan::stream_of(reinterpret_cast<void*>(stream));
    if (s == nullptr || s->ctx == nullptr) return fail(cudaErrorInvalidValue);
    if (s->ctx->capturing()) return fail(cudaErrorStreamCaptureUnsupported);   // nested capture
    g_current = s;
    s->ctx->capture_begin();
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    if (graph == nullptr) return fail(cudaErrorInvalidValue);
    *graph = nullptr;
    Stream* s = strata::vulkan::stream_of(reinterpret_cast<void*>(stream));
    if (s == nullptr || s->ctx == nullptr) return fail(cudaErrorInvalidValue);
    if (!s->ctx->capturing()) return fail(cudaErrorStreamCaptureUnsupported);
    // A capture that hit an unrecordable op, or recorded nothing, is refused: an empty graph replays nothing
    // and produces no error - the silent kind of failure this port keeps paying for.
    if (!s->ctx->capture_end()) {
        s->ctx->discard_recording();
        return fail(cudaErrorStreamCaptureUnsupported);
    }
    cudaGraph_st* g = new cudaGraph_st();
    s->ctx->take_recording(g->cb, g->fence, g->dispatches, g->copies);
    g->owner = s->ctx;
    *graph = g;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    if (exec == nullptr) return fail(cudaErrorInvalidValue);
    Stream* s = (stream != nullptr) ? strata::vulkan::stream_of(reinterpret_cast<void*>(stream)) : g_current;
    if (s == nullptr) s = g_current;
    if (s == nullptr || s->ctx == nullptr) return fail(cudaErrorInvalidValue);
    g_current = s;
    if (exec->owner == nullptr) return fail(cudaErrorInvalidValue);
    if (exec->owner->capturing()) return fail(cudaErrorStreamCaptureUnsupported);   // a launch inside a capture
    exec->owner->submit_owned(exec->cb, exec->fence);   // the port's own submit + fence, the same path a replay uses
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    if (graph == nullptr) return fail(cudaErrorInvalidValue);
    if (!graph->taken && graph->cb != VK_NULL_HANDLE && graph->owner != nullptr)
        graph->owner->destroy_owned(graph->cb, graph->fence);
    delete graph;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    if (exec == nullptr) return fail(cudaErrorInvalidValue);
    if (exec->cb != VK_NULL_HANDLE && exec->owner != nullptr)
        exec->owner->destroy_owned(exec->cb, exec->fence);
    delete exec;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t stream) {
    (void) exec;
    (void) stream;
    g_last = cudaSuccess;   // the recording is host-side command buffers: there is nothing to upload
    return cudaSuccess;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    if (graph == nullptr || numNodes == nullptr) return fail(cudaErrorInvalidValue);
    const size_t total = (size_t) graph->dispatches + (size_t) graph->copies;
    if (nodes == nullptr) { *numNodes = total; g_last = cudaSuccess; return cudaSuccess; }
    const size_t cap = *numNodes;
    graph->node_store.resize(total);
    for (size_t i = 0; i < total; ++i)
        graph->node_store[i].type = (i < (size_t) graph->dispatches) ? cudaGraphNodeTypeKernel : cudaGraphNodeTypeMemcpy;
    for (size_t i = 0; i < total && i < cap; ++i) nodes[i] = &graph->node_store[i];
    *numNodes = total;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* type) {
    if (node == nullptr || type == nullptr) return fail(cudaErrorInvalidValue);
    *type = node->type;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* params) {
    (void) node;
    (void) params;
    // This shim keeps no CUDA kernel parameters (the recording holds Vulkan bindings), so there is nothing to
    // report.  `cudaErrorNotSupported` rather than a plausible-looking zero - the engine reads this only under
    // STRATA_VERIFY_NODES, where it names node kinds, not kernel params.
    return fail(cudaErrorNotSupported);
}

// ---- errors ------------------------------------------------------------------------------------------------
cudaError_t cudaPeekAtLastError(void) { return g_last; }

cudaError_t cudaGetLastError(void) {
    const cudaError_t e = g_last;
    g_last = cudaSuccess;
    return e;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInitializationError: return "initialization error";
        case cudaErrorInvalidDevice: return "invalid device ordinal";
        case cudaErrorUnknown: return "unknown error";
        case cudaErrorNotReady: return "device not ready";
        case cudaErrorNotSupported: return "operation not supported";
        case cudaErrorStreamCaptureUnsupported: return "operation not permitted when stream is capturing";
        default: return "unrecognised CUDA error code";
    }
}

}  // extern "C"

// ---- cudaGraphInstantiate: the two toolkit signatures (C++ linkage; the engine calls both) ------------------
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags) {
    (void) flags;      // the port has no instantiate flags to honour; the recording is already launchable
    return instantiate_impl(exec, graph);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, cudaGraphNode_t* errorNode,
                                 char* logBuffer, size_t bufferSize) {
    if (errorNode != nullptr) *errorNode = nullptr;
    if (logBuffer != nullptr && bufferSize != 0) logBuffer[0] = '\0';
    return instantiate_impl(exec, graph);
}
