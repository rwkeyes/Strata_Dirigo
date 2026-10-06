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
#include <thread>
#include <vector>

namespace {

using strata::vulkan::Buf;
using strata::vulkan::Stream;

// THE CURRENT STREAM - the shim's stand-in for CUDA's process-wide "current device".  A stream-less call
// (cudaMalloc/cudaMemcpy/cudaDeviceSynchronize/cudaHostAlloc) uses it; a call carrying a stream also BINDS it
// (the "most recently used stream" rule, documented in the header).
//
// IT IS A REFERENCE TO THE DEVICE LAYER'S DEFAULT STREAM, not a second variable: CUDA treats a NULL
// `cudaStream_t` as the default stream, and the engine passes NULL wherever it means "the current stream"
// (generate.cpp:3893).  `stream_of(nullptr)` returns this same object (vk_arena.cpp), so the two can never
// disagree - one storage, one meaning.
Stream*& g_current = strata::vulkan::default_stream_ref();

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

// THE ENGINE'S DEVICE BRING-UP (a stream-less allocation path).
//
// The engine program (`src/program/generate.cpp`) allocates through cudaMalloc/cudaHostAlloc and creates its
// FIRST stream only AFTER the weight arena (generate.cpp:2243 vs :2339), and on this backend a Stream is ONE
// device and ONE FIXED arena (`vk_arena.hpp`): every cudaMalloc/cudaHostAlloc carves from that one buffer, so
// its size must be declared before the first allocation.  Without a stream, cudaMalloc refused
// (cudaErrorInvalidValue) and the program stopped at device bring-up - the first gap past MODEL LOAD.
//
// STRATA_VK_ARENA_GIB is that declaration: with it set, a stream-less allocation opens the device with a
// <G> GiB arena.  **UNSET (the smokes, and every gate case - they open their own streams), the behaviour is
// EXACTLY the old refuse, so this cannot move a gate verdict.**  This is host-side device bring-up, not a
// kernel wrapper: no dispatch, no capture node, no shader.
Stream* ensure_device() {
    if (g_current != nullptr) return g_current;
    // THE ARENA SIZE, AND WHY IT HAS A SUB-GiB FORM.  Whole-GiB steps cannot express the all-resident fit on
    // this box: the layout needs 29,656,718,848 bytes (27.62 GiB) and the reserve-bounded usable is 27.92 GiB
    // (28,593 MiB), so 27 GiB is 635 MiB short and 28 GiB is 79 MiB over.  `STRATA_VK_ARENA_MIB` (integer MiB)
    // and a fractional `STRATA_VK_ARENA_GIB` ("27.7") both exist for that gap; without them the only lever left
    // is the desktop reserve, which is already at its 512 MiB floor.
    uint64_t arena_bytes = 0;
    const char* mib = std::getenv("STRATA_VK_ARENA_MIB");
    if (mib != nullptr && *mib != '\0') {
        const double m = std::strtod(mib, nullptr);
        if (m > 0.0) arena_bytes = (uint64_t) (m * 1048576.0);
    }
    if (arena_bytes == 0) {
        const char* gib = std::getenv("STRATA_VK_ARENA_GIB");
        if (gib == nullptr || *gib == '\0') return nullptr;   // not asked to bring the device up
        const double g = std::strtod(gib, nullptr);           // fractional GiB accepted ("27.7")
        if (!(g > 0.0)) return nullptr;
        arena_bytes = (uint64_t) (g * 1073741824.0);
    }
    if (arena_bytes == 0) return nullptr;
    const char* dir = std::getenv("STRATA_VK_SPV_DIR");
    const std::string spv = (dir != nullptr && *dir != '\0') ? dir : "ports/vulkan/shaders";
    Stream* s = strata::vulkan::stream_open(arena_bytes, spv);
    if (s == nullptr) return nullptr;
    g_current = s;
    std::fprintf(stderr, "strata vulkan shim: device up, one arena of %.3f GiB (%llu bytes; "
                         "STRATA_VK_ARENA_MIB/STRATA_VK_ARENA_GIB - every cudaMalloc/cudaHostAlloc carves "
                         "from it)\n",
                 (double) arena_bytes / 1073741824.0, (unsigned long long) arena_bytes);
    return s;
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
    std::vector<strata::vulkan::CaptureSeg> segs;
    std::vector<strata::vulkan::CaptureBoundary> bounds;
    strata::vulkan::Ctx* owner = nullptr;
    uint32_t dispatches = 0;
    uint32_t copies = 0;
    bool taken = false;                     // instantiate moved the recording to the exec
    std::vector<cudaGraphNode_st> node_store;
};

struct cudaGraphExec_st {
    std::vector<strata::vulkan::CaptureSeg> segs;
    std::vector<strata::vulkan::CaptureBoundary> bounds;
    strata::vulkan::Ctx* owner = nullptr;
    uint32_t dispatches = 0;
    uint32_t copies = 0;
};

namespace {

// ---- THE SEGMENTED-LAUNCH DRIVER (the P6 verify handshake seam) --------------------------------------------
// One queue, so at most one segmented launch is in flight at a time.  `next` is the index of the next segment to
// submit; bounds[j] is the host boundary BETWEEN segs[j] and segs[j+1], so submitting segs[next] (next >= 1)
// requires `*(volatile uint32_t*)bounds[next-1].flag >= bounds[next-1].value`.  The poll is a plain HOST read of
// a mapped, coherent word - the engine writes it from its own host loop (Verifier::run) - and it happens on the
// host thread between the split submissions, driven by the engine's cudaStreamQuery/cudaStreamSynchronize calls.
// NO KERNEL WAITS: the device is never inside a queue item across a boundary.
struct Inflight {
    cudaGraphExec_st* exec = nullptr;
    size_t next = 0;
    bool active = false;
};
Inflight& inflight() {
    static Inflight f;
    return f;
}

void advance_inflight() {
    Inflight& f = inflight();
    if (!f.active || f.exec == nullptr) return;
    while (f.next < f.exec->segs.size()) {
        const strata::vulkan::CaptureBoundary& b = f.exec->bounds[f.next - 1];
        const uint32_t cur = (b.flag != nullptr) ? *(const volatile uint32_t*) b.flag : UINT32_MAX;
        if (cur < b.value) return;                       // the host has not served: leave the rest pending
        f.exec->owner->submit_segment(f.exec->segs[f.next]);
        ++f.next;
    }
}

// Drain an in-flight launch, bounded.  `where` names the caller for the refusal.  The engine raises every flag
// before it synchronizes, so a boundary that is STILL unsatisfied here means the host loop never served it: that
// is a LOUD REFUSAL (exit 2), never a hang - a hung compute kernel on the display card is the exact failure this
// design exists to remove (PORT-PLAN 2.3).
void drain_inflight(const char* where) {
    Inflight& f = inflight();
    if (!f.active) return;
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        advance_inflight();
        if (f.next >= f.exec->segs.size()) { f.active = false; f.exec = nullptr; f.next = 0; return; }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
                            .count();
        if (ms > 10000) {
            const strata::vulkan::CaptureBoundary& b = f.exec->bounds[f.next - 1];
            std::fprintf(stderr,
                         "%s: a verify-window handshake boundary was never satisfied (want flag at %p >= %u) - "
                         "REFUSING rather than leaving a submitted step waiting on the host\n",
                         where, b.flag, b.value);
            std::exit(2);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// The portable `cudaGraphInstantiate` body, shared by the two signatures.
cudaError_t instantiate_impl(cudaGraphExec_t* exec, cudaGraph_t graph) {
    if (exec == nullptr) return fail(cudaErrorInvalidValue);
    *exec = nullptr;
    if (graph == nullptr || graph->segs.empty() || graph->taken)
        return fail(cudaErrorInvalidValue);
    cudaGraphExec_st* e = new cudaGraphExec_st();
    e->segs = std::move(graph->segs);
    e->bounds = std::move(graph->bounds);
    e->owner = graph->owner;
    e->dispatches = graph->dispatches;
    e->copies = graph->copies;
    graph->segs.clear();
    graph->bounds.clear();
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
    if (s == nullptr) s = ensure_device();     // a stream-less engine allocation (STRATA_VK_ARENA_GIB)
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
    if (s == nullptr) s = ensure_device();     // a stream-less engine allocation (STRATA_VK_ARENA_GIB)
    if (s == nullptr || s->ctx == nullptr) return fail(cudaErrorInvalidValue);
    Buf b = s->ctx->alloc_host(count == 0 ? 1 : count);   // HOST_VISIBLE | HOST_COHERENT, SYSTEM RAM where offered
    if (b.mapped == nullptr)
        return fail(cudaErrorMemoryAllocation);         // the device offered no mappable type: refuse
    // THE HOST TIER'S FOOTPRINT, MEASURED.  `cudaHostAlloc` used to hand out a DEVICE_LOCAL (BAR VRAM) block and
    // spend device memory on it; with STRATA_VK_MEM_TRACE=1 each allocation prints its size and its heap, so the
    // freed device memory is a SUM rather than a claim, and the type trace alone cannot be mistaken for it.
    {
        const char* mt = std::getenv("STRATA_VK_MEM_TRACE");
        if (mt != nullptr && *mt != '\0')
            std::fprintf(stderr, "vk_host: cudaHostAlloc %.3f MiB -> type %u (heap %u, device_local=%d)\n",
                         (double) b.bytes / 1048576.0, b.mem_type, s->ctx->type_of(b).heap,
                         (int) b.device_local);
    }
    HostRegion r;
    r.host = b.mapped;
    r.bytes = count;
    r.buf = b;
    r.stream = s;
    // Register the region with the device layer so `copy_from_mapped` can bind its DEVICE-VISIBLE buffer (a
    // shader cannot dereference `host`).  A failure here means the same mapping was registered twice - a defect,
    // so refuse loudly rather than hand back a region binds cannot resolve.
    if (!strata::vulkan::mapped_register(r.host, r.buf, r.bytes)) {
        std::fprintf(stderr, "cudaHostAlloc: the mapping %p is already a live region - refusing\n", r.host);
        return fail(cudaErrorMemoryAllocation);
    }
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
    strata::vulkan::mapped_unregister(r->host);      // the device layer's view of the mapping goes with it
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
    if (s == nullptr) s = ensure_device();     // a stream-less engine allocation (STRATA_VK_ARENA_GIB)
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
    // vacuously satisfied (see the header).  It is NOT a wait for asynchronous work, because there is none -
    // EXCEPT a segmented verify-window launch, whose remaining segments must be driven out first.
    if (inflight().active) drain_inflight("cudaDeviceSynchronize");
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (stream != nullptr) g_current = reinterpret_cast<Stream*>(stream);
    // A segmented verify-window launch in flight is drained here: the engine calls this AFTER its host loop has
    // raised every handshake flag, so the remaining boundaries are satisfied and the segments submit back to back.
    // A boundary that is nevertheless unsatisfied is a LOUD REFUSAL inside `drain_inflight` - never a hang.
    if (inflight().active) drain_inflight("cudaStreamSynchronize");
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

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
                                cudaStream_t stream) {
    (void) dst; (void) dstDevice; (void) src; (void) srcDevice; (void) count; (void) stream;
    // No peer access between one device and anything, so a peer copy cannot be realised.  Refuse rather than
    // fake a copy (which would silently DROP the bytes on a path that is never selected here).
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
    // A segmented verify-window launch in flight is ADVANCED here, on the host thread: every segment whose leading
    // handshake boundary is satisfied is submitted (and its fence waited).  `cudaErrorNotReady` while boundaries
    // remain unsatisfied or segments remain pending; `cudaSuccess` once the whole recording has run - exactly what
    // the engine's own `while (*seq < want) { ...; cudaStreamQuery(cs_); ... }` loop needs.
    if (inflight().active) {
        advance_inflight();
        if (inflight().next >= inflight().exec->segs.size()) {
            inflight().active = false;
            inflight().exec = nullptr;
            inflight().next = 0;
            g_last = cudaSuccess;
            return cudaSuccess;
        }
        g_last = cudaErrorNotReady;
        return cudaErrorNotReady;
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
    s->ctx->take_recording(g->segs, g->bounds);
    if (g->segs.empty()) {
        delete g;
        return fail(cudaErrorStreamCaptureUnsupported);
    }
    g->owner = s->ctx;
    for (const strata::vulkan::CaptureSeg& sg : g->segs) {
        g->dispatches += sg.dispatches;
        g->copies += sg.copies;
    }
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
    // ONE segmented launch at a time (one queue): drain any earlier one before starting this.
    if (inflight().active) {
        if (inflight().exec == exec) return fail(cudaErrorInvalidValue);   // re-launch while in flight
        drain_inflight("cudaGraphLaunch");
    }
    if (exec->bounds.empty()) {
        // THE PLAIN STEP: submit every segment back to back (normally exactly one) and wait.  A boundary-less
        // capture is the engine's per-layer decode step, whose split the ENGINE already provides.
        for (const strata::vulkan::CaptureSeg& sg : exec->segs) exec->owner->submit_segment(sg);
        g_last = cudaSuccess;
        return cudaSuccess;
    }
    // THE VERIFY WINDOW: submit the FIRST segment (so the window starts and rings its sequence word) and leave the
    // rest to the host-driven poll.  A call that returns here has NOT run the whole window - the engine's own host
    // loop raises the flags, and its cudaStreamQuery/cudaStreamSynchronize calls advance the segments between them.
    exec->owner->submit_segment(exec->segs[0]);
    Inflight& f = inflight();
    f.exec = exec;
    f.next = 1;
    f.active = true;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    if (graph == nullptr) return fail(cudaErrorInvalidValue);
    if (!graph->taken && !graph->segs.empty() && graph->owner != nullptr) {
        for (const strata::vulkan::CaptureSeg& sg : graph->segs) graph->owner->destroy_owned(sg);
        graph->owner->release_recording();
    }
    delete graph;
    g_last = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    if (exec == nullptr) return fail(cudaErrorInvalidValue);
    if (exec->owner != nullptr) {
        // Never leave a destroyed launch in flight: this is the destructor path, and a still-advancing graph would
        // otherwise name freed command buffers.
        if (inflight().active && inflight().exec == exec) { inflight().active = false; inflight().exec = nullptr; inflight().next = 0; }
        for (const strata::vulkan::CaptureSeg& sg : exec->segs) exec->owner->destroy_owned(sg);
        exec->owner->release_recording();
    }
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
