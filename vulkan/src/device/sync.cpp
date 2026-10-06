// vulkan/src/device/sync.cpp - the handoff's implementation (see sync.hpp for the design and the reasoning;
// this file is only the code that realises it).
//
// THE ORDERING FACT THIS FILE RESTS ON, stated where the code is: the handshake's primitive (`copy_body`, below)
// ends in `ctx.flush()`, which submits the batch's one command buffer with a fence and
// `vkWaitForFences(..., UINT64_MAX)` before it returns.  So a function that ends in a handshake copy has, by the
// time it returns, (a) the device's writes COMPLETE and (b) them ordered against any host read of coherent
// memory.  That is the whole replacement for `__threadfence_system()` + the mapped ring, and it is why no counter
// is polled and no kernel spins here.  (`Ctx::dispatch` itself BATCHES and returns after ENCODING - the explicit
// flush in `copy_body` is what keeps THIS contract true.)
#include "sync.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::vulkan {

namespace {

// One element of the copy is a float (copy.spv's binding is `float[]`, as the gate's recorded-step case
// uses it); the handoff's payload and answer are float regions.
constexpr uint64_t kElem = sizeof(float);
constexpr uint32_t kLocalSize = 256;

uint32_t groups_for(uint64_t n_elems) { return (uint32_t) ((n_elems + kLocalSize - 1) / kLocalSize); }

// The publish/consume copy is the ported `copy.spv` (2 storage buffers, push {int n}).  It is a shader and
// not `vkCmdCopyBuffer` on purpose: the copy must run on the COMPUTE queue, ordered with the shaders that
// produced the payload, and it is the shader the port's own recorded-step case has already gated.
VkPipeline copy_pipeline(Ctx& ctx, const std::string& spv_dir) {
    return ctx.pipeline(spv_dir + "/copy.spv", 2, sizeof(int32_t));
}

// The shared body of every handshake copy: `src -> dst`, dispatched and (by `Ctx::dispatch`) fenced.  Shared by
// `sync_copy_fenced` (the engine's doorbell entry points) and the Handoff operations below, so there is ONE
// definition of the primitive the whole handshake rests on.
void copy_body(Ctx& ctx, const std::string& spv_dir, const Buf& src, const Buf& dst, uint64_t bytes,
               const char* what) {
    if (bytes == 0) return;
    if (bytes % kElem != 0) {
        std::fprintf(stderr, "strata::vulkan::sync: %s: %llu bytes is not a whole number of floats\n", what,
                     (unsigned long long) bytes);
        std::exit(2);
    }
    const uint64_t n = bytes / kElem;
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::sync: %s: %llu elements overflows the shader's int\n", what,
                     (unsigned long long) n);
        std::exit(2);
    }
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    ctx.dispatch(copy_pipeline(ctx, spv_dir), {&src, &dst}, &pc, sizeof(pc), groups_for(n));
    // THE HANDOFF REST ON THIS.  `Ctx::dispatch` now BATCHES (it returns after encoding, not after the work), so
    // the handshake's `sync.cpp`-level contract - "the copy is complete and visible to the host when this
    // returns" - is realised by this explicit flush, which submits the batch and waits on the fence.  Without it
    // the publish would be the one place the old per-dispatch submit+wait was load-bearing.
    ctx.flush();
}

void copy_regions(Handoff& h, const Buf& src, const Buf& dst, uint64_t bytes, const char* what) {
    copy_body(*h.ctx, h.spv_dir, src, dst, bytes, what);
}

}  // namespace

void sync_copy_fenced(Ctx& ctx, const std::string& spv_dir, const Buf& src, const Buf& dst, uint64_t bytes,
                      const char* what) {
    copy_body(ctx, spv_dir, src, dst, bytes, what);
}

Handoff* sync_open(Ctx& ctx, uint64_t payload_bytes, uint64_t answer_bytes, std::string spv_dir) {
    Handoff* h = new Handoff();
    h->ctx = &ctx;
    h->spv_dir = std::move(spv_dir);
    // The two regions the handshake used: `alloc` is HOST_VISIBLE|HOST_COHERENT (the type the device layer
    // requires for it), which is what makes a plain `read`/`write` correct without a flush.
    h->payload = ctx.alloc(payload_bytes ? payload_bytes : kElem);
    h->answer = ctx.alloc(answer_bytes ? answer_bytes : kElem);
    h->seq = 0;
    return h;
}

void sync_close(Handoff* h) {
    if (h == nullptr) return;
    if (h->ctx != nullptr) {
        if (h->payload.mem != VK_NULL_HANDLE) h->ctx->free(h->payload);
        if (h->answer.mem != VK_NULL_HANDLE) h->ctx->free(h->answer);
    }
    delete h;
}

void sync_publish(Handoff& h, const Buf& src, uint64_t bytes) {
    // The publish copy.  `copy_regions` ends in a FENCED submit, so on return the device's bytes are in the
    // host-visible payload AND visible to the host.  That return is the ring.
    copy_regions(h, src, h.payload, bytes, "sync_publish");
    ++h.seq;
}

uint32_t sync_ring(const Handoff& h) { return h.seq; }

void sync_read_payload(const Handoff& h, void* dst, uint64_t bytes) {
    // Valid only after sync_publish (the contract in the header); it is the host reading what the device
    // published.  A coherent mapping, so this is the read the CUDA `doorbell_ring` existed to make safe.
    h.ctx->read(h.payload, dst, bytes, 0);
}

void sync_write_answer(Handoff& h, const void* src, uint64_t bytes) {
    // The host's half.  Written to coherent memory, and NOTHING is submitted here: the answer becomes
    // visible to the device at the next submission, which is `sync_consume`.  That is the host-side handoff.
    h.ctx->write(h.answer, src, bytes, 0);
}

void sync_consume(Handoff& h, const Buf& dst, uint64_t bytes) {
    // The consumer.  Whatever the answer buffer holds WHEN THIS IS CALLED is what the device reads - the
    // ordering is the host's call order, not a device wait.  Ends in a fenced submit.
    copy_regions(h, h.answer, dst, bytes, "sync_consume");
}

}  // namespace strata::vulkan
