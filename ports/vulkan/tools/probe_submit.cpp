// ports/vulkan/tools/probe_submit.cpp - WHAT ONE UPLOAD SUBMIT COSTS, AS A FUNCTION OF ITS SIZE.
//
// The engine's cold start was measured (2026-10-06, `vega`, Arc Pro B70) to spend ~12-14 s of HOST time inside
// 16,114 small one-shot upload submits - ~0.75 ms per `vkQueueSubmit`, against ~5 us for a live-batch submit and
// ~24 us for a recorded-segment submit in the SAME process.  A per-call cost that large is only fixable by
// submitting FEWER, LARGER copies, and that is only sound if the cost is per CALL rather than per BYTE.
//
// This probe separates the two with the port's own device layer, the same way tools/probe_mem.cpp does:
// it copies a fixed 2 GiB into the arena in chunks of 1 / 4 / 16 / 64 MiB and reports, per chunk size, the
// submit count, the host time per submit, and the achieved host->device bandwidth.
//
//   per-call overhead  -> us/submit is roughly FLAT in chunk size, and GB/s rises with it  (batching wins)
//   bandwidth-bound    -> us/submit is roughly PROPORTIONAL to chunk size, GB/s flat        (batching cannot win)
//
// Build:
//   g++ -std=c++20 -O2 -DSTRATA_ENABLE_VULKAN=1 -Iinclude -Ivulkan/include -Ivulkan/include/cuda_compat \
//       -Ivulkan/src/device ports/vulkan/tools/probe_submit.cpp vulkan/src/device/*.cpp -o /tmp/probe_submit \
//       -lvulkan -lpthread
// Run:
//   STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 STRATA_VK_SPV_DIR=ports/vulkan/shaders \
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json /tmp/probe_submit [total_MiB]
#include "vk_arena.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::vulkan;

template <class F>
static double ms_of(F f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    const uint64_t total = (argc > 1 ? (uint64_t) std::atoi(argv[1]) : 2048ull) << 20;   // default 2 GiB
    const char* spv = std::getenv("STRATA_VK_SPV_DIR");
    Stream* s = stream_open(8ull << 30, spv ? spv : "ports/vulkan/shaders");
    if (s == nullptr) { std::fprintf(stderr, "probe_submit: no stream\n"); return 2; }

    Buf dev = s->ctx->alloc_device(total);              // the arena type: not host-visible -> STAGES
    Buf map = s->ctx->alloc(total);                     // the mapped type the bench uses
    std::vector<uint8_t> src((size_t) total, 0x5a);
    std::fprintf(stderr, "probe_submit: total %llu MiB | dev.mapped=%p (must be null) | mapped.mapped=%p\n",
                 (unsigned long long) (total >> 20), dev.mapped, map.mapped);

    const uint64_t sizes[] = {1ull << 20, 4ull << 20, 16ull << 20, 64ull << 20};
    for (uint64_t chunk : sizes) {
        if (chunk > total) continue;
        const uint64_t n = total / chunk;
        // warm once at this chunk size so the first submit's page-table work is not in the figure
        s->ctx->write(dev, src.data(), chunk);
        const double ms = ms_of([&] {
            for (uint64_t i = 0; i < n; ++i) s->ctx->write(dev, src.data() + i * chunk, chunk);
        });
        const double gb = (double) (n * chunk) / (ms * 1e-3) / 1e9;
        std::fprintf(stderr, "SUBMIT dev   chunk %4llu MiB | submits %5llu | total %8.1f ms | "
                             "%8.1f us/submit | %6.2f GB/s\n",
                     (unsigned long long) (chunk >> 20), (unsigned long long) n, ms, ms * 1000.0 / (double) n, gb);
    }
    {   // the MAPPED destination: no staging, no copy command - the host stores straight into the buffer
        const uint64_t chunk = 16ull << 20, n = total / chunk;
        const double ms = ms_of([&] {
            for (uint64_t i = 0; i < n; ++i)
                std::memcpy((uint8_t*) map.mapped + i * chunk, src.data() + i * chunk, (size_t) chunk);
        });
        std::fprintf(stderr, "SUBMIT mapped chunk %4llu MiB | host memcpy only, %8.1f ms | %6.2f GB/s\n",
                     (unsigned long long) (chunk >> 20), ms, (double) (n * chunk) / (ms * 1e-3) / 1e9);
    }
    return 0;
}
