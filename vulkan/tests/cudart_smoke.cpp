// vulkan/tests/cudart_smoke.cpp - THE CUDA-RUNTIME SHIM, RUN (not merely linked).
//
// The shim (`vulkan/include/cuda_compat/cuda_runtime.h` + `vulkan/src/compat/cuda_runtime.cpp`) answers the
// CUDA-runtime entry points the engine's host TUs call, over the port's Vulkan device layer.  This target
// exercises EVERY entry point the shim provides, end to end, on a real device, and checks the bytes round-trip
// - so "the shim compiles" is never mistaken for "the shim works".  It is the Vulkan analogue of the engine's
// own CUDA-runtime usage: allocate, upload, run, download, free.
//
// It does NOT run a layer: the one-layer-body link still has 53 unresolved `strata::kernels::` entry points
// (the plan's I4/I5), so a full single-layer forward cannot run yet.  What it proves is the part that IS
// ready: the runtime seam under the layer, on the device the port ships for.
#include "cuda_runtime.h"          // THE SHIM - the CUDA names, over the device layer
#include "vk_arena.hpp"            // Stream, stream_open (the device layer)
#include "vk_compute.hpp"          // Ctx: the live batch's own view (`live_pending`) for the visibility arm

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_bad = 0;

void check(const char* what, bool ok) {
    std::printf("  %-62s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_bad;
}

}  // namespace

int main(int argc, char** argv) {
    std::string spv = (argc > 1) ? argv[1] : "ports/vulkan/shaders";
    if (const char* e = std::getenv("STRATA_VK_SPV_DIR"); e && *e) spv = e;

    strata::vulkan::Stream* s = strata::vulkan::stream_open(/*arena_bytes=*/64ull << 20, spv);
    if (s == nullptr) { std::fprintf(stderr, "cudart_smoke: no stream\n"); return 1; }
    // The shim's stand-in for CUDA's current device: the layer program sets it once the device is up.
    strata::vulkan::cuda_compat_set_stream(s);
    const cudaStream_t cs = reinterpret_cast<cudaStream_t>(s);
    std::printf("strata_vk_cudart_smoke: device \"%s\"\n", s->ctx->info().name.c_str());

    const char* err_str = cudaGetErrorString(cudaSuccess);
    check("cudaGetErrorString(cudaSuccess) is \"no error\"",
          err_str != nullptr && std::strcmp(err_str, "no error") == 0);

    // ---- cudaMalloc + cudaMemcpy host->device->host, bit-exact -----------------------------------------------
    {
        const size_t n = 4096;
        std::vector<uint32_t> h(n);
        uint32_t st = 1234567u;
        for (uint32_t& v : h) { st = st * 1664525u + 1013904223u; v = st; }
        void* d = nullptr;
        cudaError_t e = cudaMalloc(&d, n * 4);
        check("cudaMalloc returns cudaSuccess and a non-null arena pointer",
              e == cudaSuccess && d != nullptr);
        e = cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice);
        std::vector<uint32_t> back(n, 0);
        e = (e == cudaSuccess) ? cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost) : e;
        check("cudaMemcpy host->device->host round-trips bit-exact",
              e == cudaSuccess && std::memcmp(h.data(), back.data(), n * 4) == 0);
        // The async form through an explicit stream, the way the layer body calls it.
        std::vector<uint32_t> back2(n, 0);
        e = cudaMemcpyAsync(back2.data(), d, n * 4, cudaMemcpyDeviceToHost, cs);
        check("cudaMemcpyAsync(device->host) == cudaMemcpy", 
              e == cudaSuccess && std::memcmp(back2.data(), h.data(), n * 4) == 0);
        cudaFree(d);
        check("cudaFree of an arena pointer is cudaSuccess", cudaPeekAtLastError() == cudaSuccess);
    }

    // ---- cudaMemsetAsync ------------------------------------------------------------------------------------
    {
        const size_t n = 1024;
        void* d = nullptr;
        cudaMalloc(&d, n);
        cudaMemsetAsync(d, 0xA5, n, cs);
        std::vector<uint8_t> back(n, 0);
        cudaError_t e = cudaMemcpy(back.data(), d, n, cudaMemcpyDeviceToHost);
        int bad = (e == cudaSuccess) ? 0 : 1;
        for (uint8_t v : back) if (v != 0xA5) ++bad;
        check("cudaMemsetAsync fills every byte", bad == 0);
        cudaFree(d);
    }

    // ---- cudaHostAlloc (mapped) + cudaHostGetDevicePointer ---------------------------------------------------
    {
        const size_t n = 512;
        void* h = nullptr;
        cudaError_t e = cudaHostAlloc(&h, n * 4, cudaHostAllocMapped | cudaHostAllocPortable);
        check("cudaHostAlloc(mapped) returns a mapped host pointer", e == cudaSuccess && h != nullptr);
        void* d = nullptr;
        e = cudaHostGetDevicePointer(&d, h, 0);
        check("cudaHostGetDevicePointer succeeds for a shim host region", e == cudaSuccess);
        auto* hp = static_cast<uint32_t*>(h);
        for (size_t i = 0; i < n; ++i) hp[i] = (uint32_t) (i * 3 + 1);
        // The host buffer uploaded into an arena buffer: the host writes it, the device copies it in.
        void* da = nullptr;
        cudaMalloc(&da, n * 4);
        e = cudaMemcpy(da, d, n * 4, cudaMemcpyHostToDevice);
        std::vector<uint32_t> back(n, 0);
        e = (e == cudaSuccess) ? cudaMemcpy(back.data(), da, n * 4, cudaMemcpyDeviceToHost) : e;
        int bad = (e == cudaSuccess) ? 0 : 1;
        for (size_t i = 0; i < n; ++i) if (back[i] != hp[i]) ++bad;
        check("mapped host memory -> device round-trips", bad == 0);
        cudaFree(da);
        cudaFreeHost(h);
    }

    // ---- cudaMemcpy2DAsync (device -> device) with pitched rows ----------------------------------------------
    {
        const size_t rows = 8, cols = 16, dpitch = cols * 4, spitch = (cols + 4) * 4;
        std::vector<uint32_t> src(rows * spitch / 4), want(rows * cols);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) {
                const uint32_t v = (uint32_t) (r * 100 + c);
                src[r * spitch / 4 + c] = v;
                want[r * cols + c] = v;
            }
        void* dsrc = nullptr;
        void* ddst = nullptr;
        cudaMalloc(&dsrc, rows * spitch);
        cudaMalloc(&ddst, rows * dpitch);
        cudaError_t e = cudaMemcpy(dsrc, src.data(), rows * spitch, cudaMemcpyHostToDevice);
        e = (e == cudaSuccess) ? cudaMemcpy2DAsync(ddst, dpitch, dsrc, spitch, cols * 4, rows,
                                                   cudaMemcpyDeviceToDevice, cs) : e;
        std::vector<uint32_t> got(rows * cols, 0);
        e = (e == cudaSuccess) ? cudaMemcpy(got.data(), ddst, rows * dpitch, cudaMemcpyDeviceToHost) : e;
        check("cudaMemcpy2DAsync device->device de-pitches every row",
              e == cudaSuccess && std::memcmp(got.data(), want.data(), want.size() * 4) == 0);
        cudaFree(dsrc);
        cudaFree(ddst);
    }

    // ---- events: HOST-side wall-clock, labelled as such ------------------------------------------------------
    {
        cudaEvent_t a = nullptr, b = nullptr;
        cudaError_t e = cudaEventCreate(&a);
        e = (e == cudaSuccess) ? cudaEventCreateWithFlags(&b, cudaEventDisableTiming) : e;
        cudaEventRecord(a, cs);
        void* d = nullptr;
        cudaMalloc(&d, 8u << 20);
        cudaMemsetAsync(d, 0x11, 8u << 20, cs);
        cudaDeviceSynchronize();
        cudaEventRecord(b, cs);
        float ms = -1.0f;
        e = (e == cudaSuccess) ? cudaEventElapsedTime(&ms, a, b) : e;
        check("cudaEventElapsedTime returns a finite host-wall figure", e == cudaSuccess && ms >= 0.0f);
        std::printf("  (host-wall cudaEventElapsedTime over an 8 MiB staged fill: %.3f ms)\n", (double) ms);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        cudaFree(d);
    }

    // ---- cudaMemGetInfo: the device layer's own budget, printed ----------------------------------------------
    {
        size_t freeb = 0, totalb = 0;
        const cudaError_t e = cudaMemGetInfo(&freeb, &totalb);
        std::printf("  cudaMemGetInfo: free %.2f GiB (driver figure - reserve) / total %.2f GiB (DEVICE_LOCAL heap)\n",
                    (double) freeb / 1073741824.0, (double) totalb / 1073741824.0);
        check("cudaMemGetInfo reports the device layer's own budget", e == cudaSuccess && totalb > 0);
    }

    // ---- the error path: a real enum, a real string, a real last-error ---------------------------------------
    {
        void* bogus = nullptr;
        const cudaError_t e = cudaHostGetDevicePointer(&bogus, (void*) (uintptr_t) 0x1234, 0);
        check("cudaHostGetDevicePointer refuses a foreign pointer", e == cudaErrorInvalidValue);
        check("cudaPeekAtLastError carries it (not a constant)", cudaPeekAtLastError() == cudaErrorInvalidValue);
        check("cudaGetErrorString names it",
              std::strcmp(cudaGetErrorString(cudaErrorInvalidValue), "invalid argument") == 0);
        const cudaError_t cleared = cudaGetLastError();
        check("cudaGetLastError returns it and clears it",
              cleared == cudaErrorInvalidValue && cudaPeekAtLastError() == cudaSuccess);
    }

    // ---- THE LIVE-BATCH VISIBILITY CONTRACT: does `cudaStreamQuery` report a PENDING batch as complete? ------
    // `Ctx::dispatch` ENCODES into the live batch and returns; the batch is submitted and fenced only at a flush.
    // CUDA's cudaStreamQuery contract is "cudaSuccess means all preceding work in the stream has completed", so
    // answering cudaSuccess while a batch is still queued is a WRONG ANSWER that a caller reading a device buffer
    // would act on.  This arm ISSUES a live dispatch, queries immediately, and checks the query's answer against
    // the device layer's own view of what is outstanding (`Ctx::live_pending`).  `STRATA_VK_QUERY_NOFIX=1` keeps
    // the pre-fix answer, so the SAME binary shows both sides.
    {
        void* dsrc = nullptr;
        void* ddst = nullptr;
        const size_t bytes = 64u * sizeof(float);
        cudaMalloc(&dsrc, bytes);
        cudaMalloc(&ddst, bytes);
        std::vector<float> h(64, 0.0f);
        for (size_t i = 0; i < h.size(); ++i) h[i] = 0.5f + (float) i;
        cudaMemset(ddst, 0, bytes);
        cudaMemcpy(dsrc, h.data(), bytes, cudaMemcpyHostToDevice);   // ends in a flush: nothing pending after it
        strata::vulkan::Buf sd{}, dd{};
        const bool resolved = strata::vulkan::arena_resolve(*s, dsrc, bytes, sd) &&
                              strata::vulkan::arena_resolve(*s, ddst, bytes, dd);
        check("live-batch probe: both probe buffers resolve into the arena", resolved);
        if (resolved) {
            // copy.spv: 2 storage buffers, push {int n} - the same primitive the handshake copies use.
            VkPipeline p = s->ctx->pipeline(spv + "/copy.spv", 2, (uint32_t) sizeof(int32_t));
            int32_t n = (int32_t) h.size();
            s->ctx->dispatch(p, {&sd, &dd}, &n, (uint32_t) sizeof(n), /*groups=*/1, /*groups_y=*/1);
            const bool pending_before = s->ctx->live_pending();
            check("live-batch probe: the dispatch left work QUEUED (nothing submitted yet)", pending_before);
            const cudaError_t q = cudaStreamQuery(cs);
            const bool pending_after = s->ctx->live_pending();
            std::printf("  live-batch probe: cudaStreamQuery -> %s; outstanding before/after = %d/%d\n",
                        cudaGetErrorString(q), (int) pending_before, (int) pending_after);
            // THE ASSERTION: a "complete" answer with work still outstanding is the defect.
            check("live-batch probe: cudaStreamQuery does not say 'complete' while work is queued",
                  !(pending_after && q == cudaSuccess));
            std::vector<float> got(64, 0.0f);
            cudaMemcpy(got.data(), ddst, bytes, cudaMemcpyDeviceToHost);
            bool same = true;
            for (size_t i = 0; i < got.size(); ++i) same = same && (got[i] == h[i]);
            check("live-batch probe: the queued copy's bytes are readable afterwards", same);
        }
        cudaFree(dsrc);
        cudaFree(ddst);
    }

    strata::vulkan::stream_close(s);
    std::printf("strata_vk_cudart_smoke: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    return g_bad == 0 ? 0 : 1;
}
