// vulkan/src/device/sync.hpp - THE DOORBELL REPLACEMENT (BACKEND-INTEGRATION.md I2).
//
// ============================================================================================================
// WHAT THE CUDA HANDSHAKE IS FOR, READ FROM EACH CALL SITE BEFORE ANYTHING WAS CHOSEN
// ============================================================================================================
//
// `src/kernels/cuda/elementwise.cu` implements the engine's host/device handshake with kernels that SPIN on
// host memory, ordered by `__threadfence_system()`.  The handshake is the MoE router's: the GPU routes a
// token to its k experts, and a CPU expert pool computes the misses WHILE the GPU keeps working.  Five
// symbols, and each has a different job:
//
//   * `doorbell_publish(x, ids, weights, ...)` (`elementwise.cu:293-306`, called from `layer.cpp:380` inside
//     `moe_route`, i.e. inside the CAPTURED `pre[l]` step).  The GPU COPIES the router's input `x`
//     (n_embd floats), the selected expert `ids` (k int32) and the routing `weights` (k floats) into MAPPED
//     PINNED HOST memory, then `__threadfence_system()`, then increments the mapped ring.  WHO PUBLISHES:
//     the device.  WHO READS: the host, which then runs the pool.  WHAT THE HOST DOES MEANWHILE: keeps
//     working - the ring is the only thing it watches.
//   * `doorbell_ring(d_seq)` (`elementwise.cu:209-212`, `layer.cpp:389`) - the fallback when the fused
//     publish is off: increments the ring only.  Same protocol, payload copied with memcpy nodes.
//   * `doorbell_wait(d_flag, d_seq)` (`elementwise.cu:214-218`, `session.cpp:873`) - THE ONE THAT SPINS.  A
//     one-thread kernel that loops until the host-written mapped `d_flag` equals the ring value, then
//     fences so everything the host wrote before the flag is visible to the nodes that follow.  WHO WAITS:
//     the device, inside the graph.  WHO ANSWERS: the host, after it has run the pool.
//   * `doorbell_publish_res` / `doorbell_publish_value` (`elementwise.cu:308-367`) - variants: a residency
//     check on the ids, and a STORE of a caller-known ring value instead of a read-modify-write over PCIe.
//     Both are `verify.cpp` (the P6 verifier) only, not the forward path.
//
// THE PROTOCOL, as the host loop actually runs it (`session.cpp:662-775`):
//
//     1. launch pre[l]  (contains doorbell_publish: writes x/ids/weights to host memory, then rings)
//     2. host spins on the mapped ring (volatile read) until it changes
//     3. host reads the published x/ids/weights and runs the CPU pool
//     4. host writes its answer into mapped memory (y_miss)
//     5. host launches post[l], which contains doorbell_wait (the device waits for step 4's flag), then
//        copy_from_mapped (the answer into device memory)
//
// So the handshake is a BIDIRECTIONAL producer/consumer pair with the CPU in the middle, and the ordering
// each side needs is exactly: "the ring's increment is ordered after the payload writes" (device -> host),
// and "the consumer's read is ordered after the host's answer" (host -> device).
//
// ============================================================================================================
// THE REPLACEMENT, AND WHY IT IS THIS ONE
// ============================================================================================================
//
// Vulkan has no equivalent of a kernel spinning on host memory, and the plan FORBIDS one: on the display
// card a hung compute kernel is a KMD timeout at best and a Battlemage wedge at worst (`PORT-PLAN.md`
// 2.3/§"Do not translate the device spin-wait").  So the two directions are replaced SEPARATELY, each by
// the mechanism its own side actually requires:
//
//   DEVICE -> HOST (replaces `doorbell_publish` / `doorbell_ring`).  The ring exists so the host can learn
//   that the payload has landed.  In CUDA the ring is needed because the graph is asynchronous: the device
//   writes mapped memory and increments a counter, and the host must poll.  In this backend the submission
//   that produces the payload is FENCED AND WAITED (`Ctx::dispatch` -> `end_oneshot_and_wait`,
//   `vk_compute.cpp:998-1012`), so the host's "the publish is done" observation IS the fence: when
//   `sync_publish` returns, every byte the shader wrote is visible to the host.  That is exactly the
//   ordering `__threadfence_system()` was providing, and it needs no counter to poll and (because the
//   handoff buffers are HOST_VISIBLE|HOST_COHERENT, the type `Ctx::alloc` requires - `vk_compute.cpp:398`)
//   no flush.  THE FENCE IS THE RING.  `sync_ring()` still exists and still counts, because the host loop
//   reads the ring through ONE place and a later increment may re-submit the step asynchronously (then the
//   count is what a timeline semaphore's value would carry).
//
//   HOST -> DEVICE (replaces `doorbell_wait`).  This is the direction that must never spin.  It does not
//   need a device-side wait at all: the host owns the step.  The host produces the answer, writes it into a
//   host-visible buffer, and only THEN submits the consumer.  The device is never asked to wait for the
//   host, because the device is not running - the consumer's submission is issued after the answer exists.
//   This is the "host-side handoff" the plan names, and it is the STRUCTURAL change the plan warns about:
//   the recorded step is SPLIT at the handoff (a publish submission, then a consume submission), because a
//   recorded command buffer cannot contain a host operation.  A translating spin would put the consumer in
//   the SAME submission as the publish and wait for the host's answer inside it - but the host's answer is
//   produced only after that submission ends, so it could never complete.  That is the deadlock this design
//   removes, and it is why the proof case is split across two submissions with the host between them.
//
// WHY NOT A TIMELINE SEMAPHORE.  A timeline semaphore is the right primitive for an ASYNCHRONOUS,
// re-submitted step: it carries a monotonic value a host can wait on with a timeout.  This backend's
// submission path is synchronous (every submit waits its fence before returning), so a semaphore would be a
// second ordering mechanism for an ordering the fence already gives, and it would require enabling
// `timelineSemaphore` at device creation (absent from `vk_compute.cpp`'s feature chain).  The seam is kept
// narrow so that swap is local: `sync_publish` is the ONLY place that observes "the device's writes are
// done", so when the step becomes a real asynchronous re-submission, that one function becomes the
// semaphore signal + `vkWaitSemaphores` and nothing else moves.  The DECISION (fence now; semaphore when
// the step is asynchronous) and this reasoning are recorded in the commit as well as here.
//
// ============================================================================================================
// THE ORDERING CONTRACT THIS API GUARANTEES
// ============================================================================================================
//
//   sync_publish(h, src, bytes) returns ONLY after a submission that copies `src` into the host-visible
//       payload has been fenced and waited on.  On return every payload byte is host-visible, and the ring
//       has advanced by one.  (Replaces publish + ring.)
//   sync_read_payload(...) is the host reading the payload: valid only after sync_publish returned.
//   sync_write_answer(...)  is the host writing its answer into the host-visible answer buffer.
//   sync_consume(h, dst, bytes) dispatches "answer -> dst" and waits its fence, so on return the device has
//       consumed exactly the answer the host wrote BEFORE this call.  (Replaces doorbell_wait + the
//       copy_from_mapped that followed it.)
//
// The counterpart of that contract is the FALSIFICATION the gate runs: if the consumer is dispatched
// BEFORE the answer is written, it reads the answer buffer's sentinel - the ordering, not the values, is
// what makes the handoff correct.  A translating spin does not have that failure mode; it has a hang.
#pragma once

#include "vk_compute.hpp"          // the adopted device layer: Ctx, Buf

#include <cstdint>
#include <string>

namespace strata::vulkan {

// One host/device handoff, the replacement for a `Doorbell` (`include/strata/core/layer.hpp`).
//
// `payload` and `answer` are the two mapped regions the CUDA handshake used: the device writes the payload
// for the host to read, and the host writes the answer for the device to consume.  Both are HOST_VISIBLE
// and HOST_COHERENT, which is what `Ctx::alloc` selects - the same property the CUDA `cudaHostAlloc(Mapped)`
// buffers had, and the reason no flush appears anywhere in this file.
struct Handoff {
    Ctx* ctx = nullptr;
    Buf payload{};                 // device -> host: what `doorbell_publish` published
    Buf answer{};                  // host -> device: what the pool answered
    uint32_t seq = 0;              // the ring: publications this host has OBSERVED
    std::string spv_dir;           // where copy.spv lives (the publish/consume copy is a shader, not a memcpy)
};

// Open a handoff on `ctx` with a payload and an answer region of the given byte sizes.  Neither side is
// zeroed by the handoff itself beyond what `Ctx::alloc` already does, and the answer's initial contents are
// the SENTINEL the falsification arm reads.  Refuses (prints and exits) if a region cannot be allocated -
// the port's rule.
Handoff* sync_open(Ctx& ctx, uint64_t payload_bytes, uint64_t answer_bytes, std::string spv_dir);
void sync_close(Handoff* h);

// ---- DEVICE -> HOST ---------------------------------------------------------------------------------------
// Dispatch `src -> payload` and return only once the submission is fenced, which is the ordering that
// replaces `__threadfence_system()` + the ring.  Raises the ring by one.  `src` is a DEVICE buffer (the
// shader that produced it ran earlier; this is the publish copy into host memory).
void sync_publish(Handoff& h, const Buf& src, uint64_t bytes);

// The ring, as the host loop reads it.  `doorbell_reset` zeroes the CUDA ring; here the count lives where
// it is read (the host), so a "reset" is a new Handoff.
uint32_t sync_ring(const Handoff& h);

// HOST side of the payload.  Valid only after `sync_publish` returned (see the contract above).
void sync_read_payload(const Handoff& h, void* dst, uint64_t bytes);

// ---- HOST -> DEVICE ---------------------------------------------------------------------------------------
// The host writes its answer.  The buffer is coherent, so this is a plain write; the visibility to the
// device is established by the NEXT submission (`sync_consume`), which is what removes the device-side wait.
void sync_write_answer(Handoff& h, const void* src, uint64_t bytes);

// Dispatch `answer -> dst` and wait its fence.  On return the device has consumed the answer as it stood
// when this was called.  This is what `doorbell_wait` + the `copy_from_mapped` that followed it did - but
// the ORDER is established by the host calling this, not by a kernel spinning for the host.
void sync_consume(Handoff& h, const Buf& dst, uint64_t bytes);

}  // namespace strata::vulkan
