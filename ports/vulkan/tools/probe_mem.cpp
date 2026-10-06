// ports/vulkan/tools/probe_mem.cpp - the INDEPENDENT oracle for the engine PCIe probe (see NEXT.md 2026-10-06).
// Build: g++ -std=c++20 -O2 -DSTRATA_ENABLE_VULKAN=1 -Iinclude -Ivulkan/include -Ivulkan/include/cuda_compat \
//        -Ivulkan/src/device ports/vulkan/tools/probe_mem.cpp vulkan/src/device/*.cpp -o /tmp/probe_mem -lvulkan -lpthread
// Run:   STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 STRATA_VK_SPV_DIR=ports/vulkan/shaders \
//        VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json /tmp/probe_mem

// probe_mem.cpp - an INDEPENDENT measurement of what the engine's PCIe probe times on this port.
// The engine's `probe_pcie_h2d_gbps` (generate.cpp:1159) does cudaMallocHost(256 MiB) -> cudaMalloc(256 MiB)
// -> 4 timed cudaMemcpyAsync(H2D) of 256 MiB, best-of.  This program reproduces that operation against the
// port's device layer DIRECTLY and also times the two host-side halves of it, so the attribution is measured
// rather than inferred.  Nothing here is part of the shipped port; it is an instrument.
#include "vk_arena.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::vulkan;

static double gbps(uint64_t bytes, double ms) { return ms > 0.01 ? (double) bytes / (ms * 1e-3) / 1e9 : -1.0; }
template <class F>
static double ms_of(F f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const char* spv = std::getenv("STRATA_VK_SPV_DIR");
    Stream* s = stream_open(3ull << 30, spv ? spv : "ports/vulkan/shaders");
    if (s == nullptr) { std::fprintf(stderr, "probe_mem: no stream\n"); return 2; }
    const size_t N = 256ull << 20;

    Buf hostb = s->ctx->alloc(N);          // the port's "host" type: host-visible, DEVICE_LOCAL preferred
    Buf dev = s->ctx->alloc_device(N);     // VRAM (the arena type)
    std::fprintf(stderr, "probe_mem: hostbuf.mapped=%p devbuf.mapped=%p\n", hostb.mapped, dev.mapped);
    if (hostb.mapped == nullptr) { std::fprintf(stderr, "probe_mem: host alloc has no mapping\n"); return 2; }

    std::memset(hostb.mapped, 0x5a, N);    // fault the pages in, as the engine's probe does
    s->ctx->write(dev, hostb.mapped, N);   // warmup: the engine's probe warms once too

    // (a) CPU WRITE into the mapped "host" block
    const double ms_w = ms_of([&] { std::memset(hostb.mapped, 0x3c, N); });
    // (b) CPU READ of the mapped "host" block (one byte per cache line: line-fetch bandwidth)
    volatile unsigned long long acc = 0;
    unsigned char* p = (unsigned char*) hostb.mapped;
    const double ms_r = ms_of([&] { unsigned long long a = 0; for (size_t i = 0; i < N; i += 64) a += p[i]; acc = a; });
    // (c) THE PROBE'S OWN OPERATION: host->device copy whose source is the mapped "host" block
    const double ms_c = ms_of([&] { s->ctx->write(dev, hostb.mapped, N); });
    // (d) the same copy with a plain SYSTEM-RAM source
    void* sys = std::malloc(N);
    std::memset(sys, 0x11, N);
    const double ms_s = ms_of([&] { s->ctx->write(dev, sys, N); });
    std::free(sys);

    std::fprintf(stderr,
                 "probe_mem: 256 MiB | cpu_write_mapped=%.1f ms (%.2f GB/s) | cpu_read_mapped=%.1f ms (%.2f GB/s) | "
                 "h2d_from_mapped=%.1f ms (%.2f GB/s) | h2d_from_sysram=%.1f ms (%.2f GB/s) | readacc=%llu\n",
                 ms_w, gbps(N, ms_w), ms_r, gbps(N, ms_r), ms_c, gbps(N, ms_c), ms_s, gbps(N, ms_s), acc);
    return 0;
}
