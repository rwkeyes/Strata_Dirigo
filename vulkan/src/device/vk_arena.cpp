// vulkan/src/device/vk_arena.cpp - the engine device layer's implementation (see vk_arena.hpp).
//
// No exceptions cross this file: like the adopted port layer, every failure prints and exits, because a
// backend that cannot build its device must not look like a backend whose kernel was right.
#include "vk_arena.hpp"

#include "strata/vulkan/vk_backend.hpp"   // StrataStream's declarations: stream_of()

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::vulkan {

namespace {

// LIVE-STREAM REGISTRY.  The engine's `void* stream` is opaque and, on this backend, holds a `Stream*`.  A
// wrong handle must be REFUSED, not dereferenced: the port's rule.  So `stream_of` answers from a registry of
// streams this backend created and is still holding - membership is a comparison, so the check itself cannot
// read an address that was never ours (a `magic` field would dereference the garbage to test it).
std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}
std::vector<Stream*>& registry() {
    static std::vector<Stream*> v;
    return v;
}
void registry_add(Stream* s) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    registry().push_back(s);
}
void registry_remove(Stream* s) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto& v = registry();
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == s) {
            v[i] = v.back();
            v.pop_back();
            return;
        }
    }
}

const uint64_t kSpvDefaultArena = 64ull * 1024 * 1024;   // a small arena: I1 needs bytes, not a model

// THE DEFAULT STREAM's storage (see vk_backend.hpp).  The compat shim's `g_current` IS a reference to this
// object, so "the shim's current stream" and "the stream a null handle resolves to" cannot drift apart.
Stream* g_default_stream = nullptr;

}  // namespace

Stream*& default_stream_ref() { return g_default_stream; }
Stream* default_stream() { return g_default_stream; }

Stream* stream_of(void* stream) {
    // CUDA'S DEFAULT STREAM.  A null handle is not an error and not garbage: it is the legacy default stream,
    // which on this one-stream-per-device backend is the shim's current stream (see vk_backend.hpp, and
    // generate.cpp:3893 where the engine passes exactly this).  Resolved HERE, once, so no per-symbol wrapper
    // has to remember it and so a null stream cannot mean two things.
    if (stream == nullptr) return default_stream();
    std::lock_guard<std::mutex> lk(registry_mutex());
    for (Stream* s : registry()) {
        if (s == stream) return s;
    }
    return nullptr;
}

// The arena's alignment is the device's OWN storage-buffer-offset limit raised to kAlign, so a view computed
// from any allocation this function hands out is bindable on the device that will run it.  The limit is read
// from the device rather than assumed (the Arc measures 4 bytes, but it is a device property).
static uint64_t arena_align(const Stream& s) {
    const uint64_t dev = s.ctx ? s.ctx->info().min_storage_offset_align : 0;
    return dev > Stream::kAlign ? dev : Stream::kAlign;
}

static Stream* finish_stream(Ctx* ctx, bool owns_ctx, uint64_t arena_bytes, std::string spv_dir) {
    Stream* s = new Stream();
    s->ctx = ctx;
    s->owns_ctx = owns_ctx;
    s->spv_dir = std::move(spv_dir);
    // DEVICE_LOCAL where the device has it (the engine's path); the adopted layer stages transfers where the
    // type has no mapping, so write/read below are correct on every implementation.
    s->arena = ctx->alloc_device(arena_bytes);
    s->arena_bytes = s->arena.bytes;
    s->bump = 0;
    registry_add(s);
    return s;
}

Stream* stream_borrow(Ctx* borrowed_ctx, uint64_t arena_bytes, std::string spv_dir) {
    if (borrowed_ctx == nullptr) {
        std::fprintf(stderr, "strata::vulkan: stream_borrow given a null device\n");
        return nullptr;
    }
    return finish_stream(borrowed_ctx, /*owns_ctx=*/false, arena_bytes, std::move(spv_dir));
}

Stream* stream_open(uint64_t arena_bytes, std::string spv_dir) {
    int device_index = -1;
    if (const char* e = std::getenv("STRATA_VK_DEVICE"); e && *e) device_index = std::atoi(e);
    Ctx* ctx = new Ctx(device_index, /*need_16bit=*/false);
    ctx->configure_display_reserve();
    return finish_stream(ctx, /*owns_ctx=*/true, arena_bytes, std::move(spv_dir));
}

void stream_close(Stream* s) {
    if (s == nullptr) return;
    registry_remove(s);
    if (s->ctx != nullptr && s->arena.mem != VK_NULL_HANDLE) s->ctx->free(s->arena);
    if (s->owns_ctx) delete s->ctx;
    delete s;
}

void* arena_alloc(Stream& s, uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    const uint64_t align = arena_align(s);
    const uint64_t off = (s.bump + align - 1) / align * align;
    if (off > s.arena_bytes || bytes > s.arena_bytes - off) {
        std::fprintf(stderr,
                     "strata::vulkan: arena exhausted - wanted %llu bytes (aligned to %llu) at offset %llu of a "
                     "%llu-byte arena; raise the arena size\n",
                     (unsigned long long) bytes, (unsigned long long) align, (unsigned long long) off,
                     (unsigned long long) s.arena_bytes);
        std::exit(2);
    }
    s.bump = off + bytes;
    return reinterpret_cast<void*>(Stream::kArenaBase + off);
}

bool arena_resolve(const Stream& s, const void* p, uint64_t bytes, Buf& out) {
    if (p == nullptr) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if (a < Stream::kArenaBase) return false;
    const uint64_t off = (uint64_t) (a - Stream::kArenaBase);
    // The pointer must be inside the LIVE region (allocated, not merely inside the buffer): a slice whose end
    // the engine never carved is a bug, and binding past `bump` would read bytes the engine never wrote.
    if (off > s.bump || bytes > s.bump - off) return false;
    out = view(s.arena, off);   // the view binds THIS buffer FROM this byte offset
    return true;
}

void stream_write(Stream& s, void* dev, const void* host, uint64_t bytes) {
    Buf v;
    if (!arena_resolve(s, dev, bytes, v)) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(dev);
        std::fprintf(stderr, "strata::vulkan: stream_write given a pointer that is not in the arena "
                             "(off=%llu bytes=%llu bump=%llu)\n",
                     (unsigned long long) (a >= Stream::kArenaBase ? a - Stream::kArenaBase : a),
                     (unsigned long long) bytes, (unsigned long long) s.bump);
        std::exit(2);
    }
    s.ctx->write(v, host, bytes, v.offset);
}

void stream_read(const Stream& s, const void* dev, void* host, uint64_t bytes) {
    Buf v;
    if (!arena_resolve(s, dev, bytes, v)) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(dev);
        std::fprintf(stderr, "strata::vulkan: stream_read given a pointer that is not in the arena "
                             "(off=%llu bytes=%llu bump=%llu)\n",
                     (unsigned long long) (a >= Stream::kArenaBase ? a - Stream::kArenaBase : a),
                     (unsigned long long) bytes, (unsigned long long) s.bump);
        std::exit(2);
    }
    s.ctx->read(v, host, bytes, v.offset);
}

// ---- MAPPED HOST REGIONS (see vk_arena.hpp) ----------------------------------------------------------------
namespace {
struct MappedRegion {
    void* host = nullptr;   // the mapping cudaHostAlloc returned
    Buf buf{};              // the HOST_VISIBLE | HOST_COHERENT device block behind it
    uint64_t bytes = 0;
};
std::vector<MappedRegion>& mapped_regions() {
    static std::vector<MappedRegion> v;
    return v;
}
}  // namespace

bool mapped_register(void* host, const Buf& buf, uint64_t bytes) {
    if (host == nullptr) return false;
    for (const MappedRegion& r : mapped_regions())
        if (r.host == host) return false;               // already live: a double registration is a defect
    mapped_regions().push_back(MappedRegion{host, buf, bytes});
    return true;
}

void mapped_unregister(void* host) {
    auto& v = mapped_regions();
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i].host == host) {
            v[i] = v.back();
            v.pop_back();
            return;
        }
    }
}

bool mapped_resolve(const void* host, uint64_t bytes, Buf& out) {
    if (host == nullptr || bytes == 0) return false;
    for (const MappedRegion& r : mapped_regions()) {
        if (r.host == host) {
            // The view must fit INSIDE the region the shim handed out: binding the whole block and letting the
            // shader read past `bytes` would read bytes the caller never published.  Refuse rather than bind a
            // view the caller's size does not cover.
            if (bytes > r.bytes) {
                std::fprintf(stderr,
                             "strata::vulkan: mapped_resolve: %llu bytes asked of a %llu-byte mapped region - "
                             "refusing rather than reading past the published block\n",
                             (unsigned long long) bytes, (unsigned long long) r.bytes);
                return false;
            }
            out = r.buf;                                // offset 0: the region's own base
            return true;
        }
    }
    return false;
}

uint64_t mapped_live_regions() { return (uint64_t) mapped_regions().size(); }

}  // namespace strata::vulkan
