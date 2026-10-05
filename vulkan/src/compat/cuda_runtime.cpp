// vulkan/src/compat/cuda_runtime.cpp - the CUDA-runtime compatibility shim's IMPLEMENTATION over the port's
// Vulkan device layer.  The semantics, and every place CUDA could not be honoured exactly, are stated in the
// companion header `vulkan/include/cuda_compat/cuda_runtime.h` - read that first; this file only realises it.
//
// It is deliberately thin: every operation is one of the device layer's own primitives (arena_alloc, the
// pointer->buffer resolution, the STAGING read/write, the fenced submit), so there is no second memory model
// and no second ordering model.  What CUDA called directly, this file expresses in the arena.
#include "cuda_runtime.h"                 // this shim's own declarations (found via -Ivulkan/include/cuda_compat)

#include "vk_arena.hpp"                   // Stream, arena_alloc, arena_resolve, stream_write/stream_read

#include <chrono>
#include <cstdio>
#include <cstring>
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
