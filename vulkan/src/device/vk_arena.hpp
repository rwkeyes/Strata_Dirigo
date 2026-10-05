// vulkan/src/device/vk_arena.hpp - the ENGINE's device layer, built on the ADOPTED port device layer
// (vk_compute.hpp, copied from ports/vulkan/harness/ - see BACKEND-INTEGRATION.md I1).
//
// The port's device layer answers "load this SPIR-V, bind these N buffers, push these bytes, dispatch G
// groups" for a GATE case, which holds a `Buf` handle for everything.  The ENGINE does not: it holds RAW
// DEVICE POINTERS (`float* kv`, `X + t0 * K`) and does pointer arithmetic on them, exactly as it did against
// cudaMalloc.  This header is the seam that turns those pointers back into bindable `(VkBuffer, offset)`
// pairs, and it does it with ONE allocation:
//
//     THE ARENA.  The engine's plan is a single DEVICE_LOCAL buffer, carved by BYTE OFFSETS.  `arena_alloc`
//     bump-allocates from it (aligned to the device's `minStorageBufferOffsetAlignment`, so every descriptor
//     view it hands out is bindable), and `arena_resolve` maps a raw device pointer back to the arena and the
//     byte offset the shader must read from.
//
// WHY A SYNTHETIC ADDRESS SPACE RATHER THAN THE MAPPING'S ADDRESS: on a discrete card (the engine's path) the
// arena is DEVICE_LOCAL and NOT MAPPABLE, so there is no host address to hand out - the device pointer is a
// token, not something the host can dereference (it could not under CUDA either).  So the arena has a fixed
// synthetic base and a device pointer is `kArenaBase + byte_offset`.  The engine only ever does arithmetic on
// it and passes it back, so the value only has to be RESOLVABLE, not readable - and resolution is a range
// check plus a subtraction, which is what makes a wrong pointer a loud refusal.
#pragma once

#include "vk_compute.hpp"                 // the adopted port device layer: Ctx, Buf, view()

#include <cstdint>
#include <string>

namespace strata::vulkan {

// One device, one arena, one recorded step (PORT-PLAN).  The engine's opaque `void* stream` names this.
struct Stream {
    // The adopted device layer.  `owns_ctx` distinguishes a Stream that created its own Ctx (the engine's
    // path) from one BORROWED from a caller that already has a device up (the gate's path, whose reserve and
    // descriptor pools must not be doubled by a second VkDevice).
    Ctx* ctx = nullptr;
    bool owns_ctx = false;

    // THE ARENA: one buffer carved by byte offsets (see the header note).
    Buf arena{};
    uint64_t arena_bytes = 0;
    uint64_t bump = 0;            // the next free byte offset; never decreases

    // Where this backend's .spv files are (fwht256.spv and, in later increments, the rest).
    std::string spv_dir;

    // THE CONTROL-VECTOR TABLES `cvec_apply` reads (I2-continued).  The engine's cvec.cu owns these on a CUDA
    // build; a Vulkan build compiles no cvec.cu, so this backend answers the cvec `host` row and places the
    // tables HERE.  They live and die with the stream - the same lifetime rule the arena follows - so a
    // reopened stream can never inherit a stale direction.  `gen` is the module generation they were built at;
    // cvec_apply replaces them when it no longer matches (an upload, a replicate or a switch change).
    struct CvecTables {
        Buf dir{}, s{}, on{}, dummy{};
        bool valid = false;
        uint64_t gen = 0;
    };
    CvecTables cvec_tables;

    // THE I-QUANT GRID TABLES the `native_mmvq` composite's IQ arms read (I3).  On a CUDA build they are
    // `__constant__` device globals inside native_mmvq.cu; on this build each is a storage buffer the shader
    // indexes, and the ENGINE's TU (matvec_vk.cpp) places them HERE, lazily, the same way cvec_apply's tables
    // live with the stream.  A per-dispatch upload would be both wasteful and fatal: `arena_alloc` never
    // decreases, so a per-layer grid allocation would exhaust the arena.  Four grids (iq1s 2048, iq2s 2048,
    // iq3s 512, iq3xxs 256 uint32) cover every IQ arm the port ships.
    struct IqGrids {
        Buf iq1s{}, iq2s{}, iq3s{}, iq3xxs{}, iq2xxs{}, iq2xs{};
        bool valid = false;
    } iq_grids;

    // A NON-NULL SENTINEL for the descriptors a Vulkan bind cannot leave empty: the two rope shaders ALWAYS
    // bind an mrope table even when `mrope == 0` and its contents are never read (a null descriptor is illegal
    // here).  It lives with the stream for the same reason the tables above do - `arena_alloc` never decreases,
    // so a per-dispatch sentinel would exhaust the arena.  Placed lazily by qsa_vk.cpp.
    Buf dummy{};

    // The synthetic device-address base and the alignment every carving starts on.  The alignment is the
    // device's OWN storage-buffer-offset limit raised to 256, so a view computed from an allocation is
    // bindable on every implementation the port runs on (the Arc measures 4 bytes; the limit is still a
    // DEVICE property and is read from it rather than assumed).
    static constexpr uintptr_t kArenaBase = 0x700000000000ull;
    static constexpr uint64_t kAlign = 256;
};

// ---- lifetime -------------------------------------------------------------------------------------------
// `borrowed_ctx` must outlive the Stream and must already have `configure_display_reserve()` called.  The
// arena is then allocated from it.  Returns null (and prints why) if the arena cannot be allocated - the
// port's rule is refuse, never degrade.
Stream* stream_borrow(Ctx* borrowed_ctx, uint64_t arena_bytes, std::string spv_dir);
// Opens its own device (STRATA_VK_DEVICE / the first suitable one).  The engine's own path; the gate borrows.
Stream* stream_open(uint64_t arena_bytes, std::string spv_dir);
// Frees the arena and, when it owns it, the device.  Idempotent.
void stream_close(Stream* s);

// ---- the engine's raw-pointer surface --------------------------------------------------------------------
// Carve `bytes` from the arena and return it as a device pointer (`kArenaBase + offset`).  Exits with a
// message when the arena is exhausted - an out-of-arena pointer would be a silent wrong answer otherwise.
// Aligned to the device's storage-buffer-offset limit, so a view from it binds.
void* arena_alloc(Stream& s, uint64_t bytes);
template <typename T>
T* arena_alloc(Stream& s, uint64_t count) {
    return reinterpret_cast<T*>(arena_alloc(s, count * sizeof(T)));
}

// POINTER -> BUFFER.  Maps a raw device pointer handed out by `arena_alloc` (or derived from one by the
// engine's own arithmetic, `p + k`) back to the arena and the byte offset a descriptor must bind.  Returns
// false - and binds nothing - when the pointer is not inside the LIVE region of the arena, so a handle from
// another device, a stale arena, or a freed Stream is a refusal rather than a wrong read.
bool arena_resolve(const Stream& s, const void* p, uint64_t bytes, Buf& out);

// Host <-> arena.  `dev` is a device pointer; the transfer goes through the mapping when the arena has one
// and through staging otherwise (the adopted layer decides, per its own rule).
void stream_write(Stream& s, void* dev, const void* host, uint64_t bytes);
void stream_read(const Stream& s, const void* dev, void* host, uint64_t bytes);

// ---- MAPPED HOST REGIONS: the `copy_from_mapped` seam -----------------------------------------------------
// A Vulkan SHADER cannot dereference a HOST pointer.  `cudaHostAlloc` in this shim returns the mapping of a
// HOST_VISIBLE | HOST_COHERENT DEVICE BLOCK, so the bytes the host stores at `host` are the bytes the buffer
// at `buf` holds - the mapping and the device view are ONE allocation, not two that need synchronising.  So a
// symbol whose CUDA body reads MAPPED, PINNED host memory (elementwise.cu:226 `copy_from_mapped_kernel`) is
// answered by BINDING THE REGION'S DEVICE-VISIBLE BUFFER, not the host address.  Registration happens at
// `cudaHostAlloc` (the shim) and is looked up here by `copy_from_mapped` (the kernel TU).
//
// THE ORDERING CONTRACT, stated where it is relied on: the host's store into the mapping IS the publish, and
// its edge to the device is the NEXT SUBMISSION.  A recorded step re-reads the buffer at SUBMIT time, so the
// store must sit AFTER `capture_end`/the recording and BEFORE each launch - NEVER at capture.  A store made
// only at capture is the STALE BLOCK this API exists to prevent (a replay would copy the capture-time bytes).
bool mapped_register(void* host, const Buf& buf, uint64_t bytes);   // false if `host` is already live
void mapped_unregister(void* host);
// Resolve a mapped host pointer (or a pointer the caller derived from one) to its device-visible buffer.
// `bytes` must lie inside the live region.  False - binding nothing - when it is not a live region, so a
// caller cannot get a token for bytes this layer cannot show the device.
bool mapped_resolve(const void* host, uint64_t bytes, Buf& out);
// How many regions are live: the gate's leak instrument, and the FIFO handshake's own count.
uint64_t mapped_live_regions();

}  // namespace strata::vulkan
