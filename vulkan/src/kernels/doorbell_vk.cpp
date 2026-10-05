// vulkan/src/kernels/doorbell_vk.cpp - THE ENGINE'S `doorbell_*` ENTRY POINTS, answered by the Vulkan backend.
//
// ============================================================================================================
// WHAT THESE ARE, AND WHY THEY ARE ANSWERED HERE RATHER THAN TRANSLATED
// ============================================================================================================
//
// `src/kernels/cuda/elementwise.cu` implements the engine's MoE host/device handshake with FOUR kernels that
// SPIN on mapped host memory, ordered by `__threadfence_system()`:
//
//   doorbell_publish       (elementwise.cu:353, called from layer.cpp:380, INSIDE the captured `pre[l]`) - the
//                          device copies the router's x / ids / weights into mapped host memory, fences, and
//                          increments the mapped ring.
//   doorbell_publish_res   (elementwise.cu:361) - the same publish with a residency check on the ids; a
//                          verifier-only variant (verify.cpp).
//   doorbell_publish_value (elementwise.cu:345) - the ring STORED (a capture-time constant) instead of
//                          read-modify-written over PCIe; the HIP `STRATA_DOORBELL_STORE=1` variant.
//   doorbell_ring          (elementwise.cu:379, layer.cpp:389) - the fallback: the ring alone.
//   doorbell_wait          (elementwise.cu:220, session.cpp:873, INSIDE the captured graph) - the one that
//                          SPINS: a one-thread kernel that loops until the host-written flag equals the ring.
//
// The first four are DEVICE -> HOST: the device writes the payload, then rings so the host learns it landed.
// The fifth is HOST -> DEVICE: the device waits for the host's answer.
//
// THE REPLACEMENT IS IN `vulkan/src/device/sync.hpp` AND IS NOT TRANSLATED HERE - it is the design this file
// implements, so restate only its consequence:
//
//   DEVICE -> HOST = the SUBMISSION FENCE.  `Ctx::dispatch` submits a one-shot command buffer with a fence and
//       waits it before returning, so a publish that returns is a publish whose bytes are HOST-VISIBLE.  That
//       is exactly the ordering `__threadfence_system()` provided, and (the handoff regions being coherent) it
//       needs no counter to poll and no flush.  THE FENCE IS THE RING.
//   HOST -> DEVICE = a HOST-DRIVEN SPLIT SUBMISSION.  The host owns the step: it produces the answer, writes
//       it, and only THEN submits the consumer.  **NO KERNEL EVER WAITS.**  A translating spin is forbidden
//       (PORT-PLAN 2.3): on the display card a hung compute kernel is a KMD timeout at best and a Battlemage
//       wedge at worst.
//
// THE RING IS THE HOST'S OWN COUNT, and that is the design's choice rather than a shortcut: `sync.hpp` states
// that in this synchronous backend "the fence is the ring" and the count "lives where it is read (the host)".
// The CUDA ring had to be incremented on the DEVICE only because the graph is asynchronous and the host had to
// POLL for the datum; here the wrapper's caller IS the host and the fenced submit has already completed when it
// returns, so the host raises the ring itself.  When the recorded step becomes a genuinely asynchronous
// re-submission (a later increment), the one place that changes is `sync_publish` - the seam was kept narrow
// for exactly that swap.
//
// THE PAYLOAD COPY IS `sync_copy_fenced` - the SAME fenced copy `sync_publish`/`sync_consume` use, exposed from
// `sync.*` so the two directions of the handshake have ONE definition and not two that can drift.
#if !defined(STRATA_ENABLE_VULKAN)
#error "doorbell_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/elementwise.hpp"   // the engine wrappers: the five doorbell_* the layer body calls
#include "strata/vulkan/vk_backend.hpp"     // stream_of
#include "vk_arena.hpp"                     // Stream, arena_resolve, stream_read/write
#include "sync.hpp"                         // sync_copy_fenced - the handshake's fenced copy

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::vulkan {
namespace {

Stream* require_stream(void* stream, const char* what) {
    Stream* s = stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", what);
        std::exit(2);
    }
    return s;
}

// Resolve one raw device pointer to an arena view.  A pointer outside the arena is a refusal, never a wrong
// bind (vk_arena.hpp).
Buf view_of(Stream& s, const void* p, uint64_t bytes, const char* what) {
    Buf b{};
    if (!arena_resolve(s, p, bytes, b)) {
        std::fprintf(stderr, "%s: a pointer is not inside this stream's arena - refusing rather than binding a "
                             "wrong view\n", what);
        std::exit(2);
    }
    return b;
}

void copy(Stream& s, const void* src, const void* dst, uint64_t bytes, const char* what) {
    if (bytes == 0) return;
    sync_copy_fenced(*s.ctx, s.spv_dir, view_of(s, src, bytes, what), view_of(s, dst, bytes, what), bytes, what);
}

// THE RING.  In the synchronous backend the fenced submit above already ordered the payload; this records the
// publication count where the host reads it (see the file header).  `store` selects the `_value` variant.
uint32_t ring_raise(Stream& s, uint32_t* d_seq, bool store, uint32_t value, const char* what) {
    if (d_seq == nullptr) return 0;
    (void) view_of(s, d_seq, sizeof(uint32_t), what);      // refuse a ring outside the arena, like every other bind
    uint32_t cur = 0;
    stream_read(s, d_seq, &cur, sizeof(uint32_t));
    const uint32_t next = store ? value : cur + 1u;
    stream_write(s, d_seq, &next, sizeof(uint32_t));
    return next;
}

}  // namespace
}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/elementwise.hpp declares ------------------
namespace strata::kernels {

// elementwise.hpp: the fused publish - x (n floats), ids and weights (k each) into host memory, then the ring.
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::require_stream(stream, "doorbell_publish");
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    // DEVICE -> HOST: the payload copy.  Each `copy` ends in a fenced submit, so the host may read x_out/ids_out/
    // weights_out the moment this returns.  THAT is the handshake, not a ring the device spins to raise.
    strata::vulkan::copy(*s, x, x_out, (uint64_t) n * sizeof(float), "doorbell_publish/x");
    strata::vulkan::copy(*s, ids, ids_out, (uint64_t) k * sizeof(int32_t), "doorbell_publish/ids");
    strata::vulkan::copy(*s, weights, weights_out, (uint64_t) k * sizeof(float), "doorbell_publish/weights");
    strata::vulkan::ring_raise(*s, d_seq, /*store=*/false, 0, "doorbell_publish/seq");
}

// elementwise.hpp: #649's store-the-ring variant (HIP STRATA_DOORBELL_STORE=1).  Same publish; the ring takes
// the caller's value instead of an increment.
void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                            float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value,
                            void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::require_stream(stream, "doorbell_publish_value");
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    strata::vulkan::copy(*s, x, x_out, (uint64_t) n * sizeof(float), "doorbell_publish_value/x");
    strata::vulkan::copy(*s, ids, ids_out, (uint64_t) k * sizeof(int32_t), "doorbell_publish_value/ids");
    strata::vulkan::copy(*s, weights, weights_out, (uint64_t) k * sizeof(float), "doorbell_publish_value/weights");
    strata::vulkan::ring_raise(*s, d_seq, /*store=*/true, value, "doorbell_publish_value/seq");
}

// elementwise.hpp: the residency-checked publish.  The CUDA kernel copies `ids` ALWAYS and `x` ONLY when a
// selected id is a miss (`d_res == nullptr`, out of range, or `d_res[id] < 0`); the decision is a pure function
// of `ids` and `d_res`, so this reads them on the HOST and keeps the copy itself a device->host transfer.  It is
// the P6 verifier's variant (verify.cpp), not the forward path.
void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n,
                          int64_t k, float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::require_stream(stream, "doorbell_publish_res");
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish_res: k too large\n"); std::exit(1); }
    strata::vulkan::copy(*s, ids, ids_out, (uint64_t) k * sizeof(int32_t), "doorbell_publish_res/ids");

    bool any_miss = (d_res == nullptr || n_expert <= 0);
    if (!any_miss) {
        std::vector<int32_t> h_ids((size_t) (k > 0 ? k : 1), 0), h_res((size_t) n_expert, 0);
        if (k > 0) strata::vulkan::stream_read(*s, ids, h_ids.data(), (uint64_t) k * sizeof(int32_t));
        strata::vulkan::stream_read(*s, d_res, h_res.data(), (uint64_t) n_expert * sizeof(int32_t));
        for (int64_t i = 0; i < k; ++i) {
            const int32_t id = h_ids[(size_t) i];
            if (id < 0 || id >= n_expert || h_res[(size_t) id] < 0) { any_miss = true; break; }
        }
    }
    if (any_miss) strata::vulkan::copy(*s, x, x_out, (uint64_t) n * sizeof(float), "doorbell_publish_res/x");
    strata::vulkan::ring_raise(*s, d_seq, /*store=*/false, 0, "doorbell_publish_res/seq");
}

// elementwise.hpp: the ring alone, the fallback when the fused publish is off (the payload is copied by other
// nodes).  In this backend the fence already ordered any preceding copy, so this only records the publication.
void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;                       // the CUDA form returns on null
    strata::vulkan::Stream* s = strata::vulkan::require_stream(stream, "doorbell_ring");
    strata::vulkan::ring_raise(*s, d_seq, /*store=*/false, 0, "doorbell_ring/seq");
}

// elementwise.hpp: THE ONE THAT SPINS IN CUDA.  Its replacement is the host->device half of the split
// submission: the host writes its answer BEFORE this is called, and the consumer that follows is a later,
// fenced submission - so this call submits NOTHING and no kernel ever waits.
//
// It is not an empty stub.  It is the handoff BOUNDARY, and it enforces the one ordering the replacement
// requires: the host must have served the ring (the answer flag has reached the ring value) before the consumer
// is submitted.  An un-answered handoff is a LOUD REFUSAL - never a hang, because a hang on the display card is
// the exact failure this design exists to remove.
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;  // the CUDA form returns on null
    strata::vulkan::Stream* s = strata::vulkan::require_stream(stream, "doorbell_wait");
    uint32_t flag = 0, ring = 0;
    strata::vulkan::stream_read(*s, d_flag, &flag, sizeof(uint32_t));
    strata::vulkan::stream_read(*s, d_seq, &ring, sizeof(uint32_t));
    if (flag < ring) {
        std::fprintf(stderr,
                     "doorbell_wait: the host has not answered (flag %u < ring %u).  This backend never asks the "
                     "device to wait: the host writes the answer, THEN submits the consumer.  Refusing rather "
                     "than submitting a waiting kernel\n", flag, ring);
        std::exit(2);
    }
    // Nothing is submitted.  The consumer's own submit is the visibility edge for whatever the host wrote.
}

}  // namespace strata::kernels
