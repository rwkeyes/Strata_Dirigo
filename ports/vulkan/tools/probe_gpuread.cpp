// ports/vulkan/tools/probe_gpuread.cpp - DOES THE GPU READ A MAPPABLE VRAM BUFFER AS FAST AS AN UNMAPPABLE ONE?
//
// WHY THIS EXISTS.  `probe_mem.cpp` measured the two halves of an upload on the Arc Pro B70:
//     cpu_write_mapped 5.64 GB/s   (host stores straight into a mapped VRAM buffer)
//     cpu_read_mapped  0.06 GB/s   (host READS that same buffer - write-combined, 94x slower)
//     h2d_from_alloc_host 1.92 GB/s | h2d_from_sysram 1.88 GB/s   (the staged path's device-side copy)
// So lever C - upload straight into a mappable buffer instead of staging through host RAM - is worth 2.9x on
// host->device traffic (53.7 GB of cold-start uploads: ~29 s -> ~10 s).  The port has never measured the OTHER
// side of that trade: what the GPU pays when it READS a mappable allocation instead of an unmappable one.  C is
// only safe if that read is not slower - the model's expert weights live in those buffers and the GPU reads them
// on every token.  This is the check the ledger says is owed before C becomes a default.
//
// METHOD.  Same kernel (`bw_stream.spv`: out[i] = f(in[i])), same WRITE destination (an unmappable buf), same
// byte volume - only the SOURCE buffer's memory type differs (ctx->alloc() = the mapped type the bench uses vs
// ctx->alloc_device() = not host-visible -> the staged/VRAM type).  A pass is dispatched `reps` times with one
// flush, so the figure is steady-state bandwidth, not a cold submit.
//
// Build: g++ -std=c++20 -O2 -DSTRATA_ENABLE_VULKAN=1 -Iinclude -Ivulkan/include -Ivulkan/include/cuda_compat \
//            -Ivulkan/src/device ports/vulkan/tools/probe_gpuread.cpp vulkan/src/device/*.cpp \
//            -o /tmp/probe_gpuread -lvulkan -lpthread
// Run:   STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 \
//        STRATA_VK_SPV_DIR=ports/vulkan/bench/build/spv VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
//        /tmp/probe_gpuread [MiB] [reps]
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

// one pass over n floats, reading `src` and writing `dst`; returns ms for `reps` passes after `warm` warmups
static double pass_ms(Ctx& ctx, VkPipeline p, Buf& src, Buf& dst, int n, int reps, int warm, uint32_t groups) {
    struct { int32_t n; } pc{n};
    for (int i = 0; i < warm; ++i) { ctx.dispatch(p, {&src, &dst}, &pc, sizeof(pc), groups, 1); ctx.flush(); }
    const double ms = ms_of([&] {
        for (int i = 0; i < reps; ++i) ctx.dispatch(p, {&src, &dst}, &pc, sizeof(pc), groups, 1);
        ctx.flush();
    });
    return ms;
}

int main(int argc, char** argv) {
    const char* spv = std::getenv("STRATA_VK_SPV_DIR");
    Stream* s = stream_open(4ull << 30, spv ? spv : "ports/vulkan/bench/build/spv");
    if (s == nullptr) { std::fprintf(stderr, "probe_gpuread: no stream\n"); return 2; }
    Ctx& ctx = *s->ctx;

    const int mib = argc > 1 ? std::atoi(argv[1]) : 256;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 8;
    const int n = mib << 20 >> 2;                      // floats
    const uint64_t bytes = (uint64_t) n * 4;
    const uint32_t groups = (uint32_t) (((uint64_t) n + 255) / 256);

    Buf mapped = ctx.alloc(bytes);                     // the mapped type the bench uses (host-writable)
    Buf device = ctx.alloc_device(bytes);              // not host-visible -> the staged/VRAM type
    Buf out    = ctx.alloc_device(bytes);              // the SAME write destination for both arms

    std::vector<float> src((size_t) n);
    for (int i = 0; i < n; ++i) src[(size_t) i] = (float) (i % 977) * 0.001f;

    std::fprintf(stderr, "probe_gpuread: %d MiB | mapped.mapped=%p | device.mapped=%p (expect null)\n",
                 mib, mapped.mapped, device.mapped);
    if (mapped.mapped == nullptr) { std::fprintf(stderr, "probe_gpuread: the mapped arm did not map - no comparison possible\n"); return 3; }
    std::memcpy(mapped.mapped, src.data(), (size_t) bytes);      // fill the mapped source by host store
    ctx.write(device, src.data(), bytes);                        // fill the unmappable source through the port's path

    // The kernel exists in the bench's tree; if the spv dir is wrong this throws loudly rather than measuring nothing.
    VkPipeline p = ctx.pipeline(std::string(spv ? spv : "ports/vulkan/bench/build/spv") + "/bw_stream.spv", 2, 4);

    const double ms_mapped = pass_ms(ctx, p, mapped, out, n, reps, 2, groups);
    const double ms_device = pass_ms(ctx, p, device, out, n, reps, 2, groups);
    // ...and once more, mapped first, to catch a warm-up order effect
    const double ms_mapped2 = pass_ms(ctx, p, mapped, out, n, reps, 0, groups);
    const double ms_device2 = pass_ms(ctx, p, device, out, n, reps, 0, groups);

    const double rw = 2.0 * (double) bytes * (double) reps;       // read + write bytes moved per arm
    std::fprintf(stderr, "GPUREAD  source=mapped (host-visible device-local) | %7.2f ms / %d passes | %6.2f GB/s\n",
                 ms_mapped, reps, rw / (ms_mapped * 1e-3) / 1e9);
    std::fprintf(stderr, "GPUREAD  source=device (unmappable)                 | %7.2f ms / %d passes | %6.2f GB/s\n",
                 ms_device, reps, rw / (ms_device * 1e-3) / 1e9);
    std::fprintf(stderr, "GPUREAD  repeat: mapped %7.2f ms | device %7.2f ms  ->  mapped/device %.3f\n",
                 ms_mapped2, ms_device2, ms_mapped2 / ms_device2);
    std::fprintf(stderr, "GPUREAD  VERDICT: %.3f (first) / %.3f (repeat) - <=1.05 means C's premise HOLDS; "
                         ">1.05 means the GPU read pays for the mapping and C must be scoped around it\n",
                 ms_mapped / ms_device, ms_mapped2 / ms_device2);
    ctx.free(mapped); ctx.free(device); ctx.free(out);
    return 0;
}
