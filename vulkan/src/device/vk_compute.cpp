// ports/vulkan/harness/vk_compute.cpp - see the header.  No exceptions thrown out of here: every failure
// prints and exits, because a gate that cannot build its device must not look like a gate that passed.
#include "vk_compute.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace strata::vulkan {

#define VK_CHECK(x)                                                                        \
    do {                                                                                   \
        const VkResult _r = (x);                                                           \
        if (_r != VK_SUCCESS) {                                                            \
            std::fprintf(stderr, "Vulkan call failed: %s -> %d at %s:%d\n", #x, (int) _r,   \
                         __FILE__, __LINE__);                                              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> out((size_t) n);
    if (n > 0 && !f.read((char*) out.data(), n)) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::exit(1);
    }
    return out;
}

}  // namespace

// ---- STRATA_VK_DISP_STAT: the live dispatch/submit accounting -----------------------------------------------
// The prompt path calls `Ctx::dispatch`, which is a FULL submit+fence ROUND TRIP per dispatch, with a command
// buffer allocated and freed and a fence created and destroyed around it.  This instrument prices each of those
// steps separately, plus the encode, so the next change is aimed at the step that actually costs - not at a
// theory.  Measurement-only: off unless the env var is set, and it never changes a dispatch.
namespace {
struct DispStat {
    bool on = false;
    uint64_t n = 0;                 // live (non-capture) dispatches
    uint64_t recorded = 0;          // dispatches ENCODED into a recorded step (capture/replay)
    uint64_t submits = 0, waits = 0, cb_allocs = 0, fences_created = 0, sets_alloc = 0, pools_made = 0, batches = 0;
    // BY ARM: a live batch flush (the prompt path), a transfer (begin_oneshot), a recorded-step submit (a replay,
    // i.e. the decode/verify arm) and a segment submit (that replay cut at a handshake boundary).
    uint64_t sub_live = 0, sub_transfer = 0, sub_rec = 0, sub_seg = 0, replays = 0;
    uint64_t seg_disp = 0;                        // recorded dispatches EXECUTED by segment submits
    uint64_t barriers = 0;                        // COMPUTE->COMPUTE chain barriers emitted (one per chained dispatch)
    double t_seg_wait = 0, t_seg_submit = 0;      // the segment path's own submit + fence wait (ms)
    double t_alloc = 0, t_encode = 0, t_fence = 0, t_submit = 0, t_wait = 0, t_free = 0;   // ms
    std::vector<std::pair<std::string, uint64_t>> by_pipe;   // per-spv dispatch counts
    // THE RECORDED ARM'S OWN COMPOSITION.  A recorded command buffer is re-executed once per segment submit, so its
    // composition IS the composition of the work the replay arm executes each round - which `by_pipe` above cannot
    // show, because it pools the (one-shot) prefill with the (repeated) replay.  Counted once per ENCODE.
    std::vector<std::pair<std::string, uint64_t>> by_pipe_rec;
};
DispStat g_ds;
// ---- STRATA_VK_NOBARRIER: MEASUREMENT-ONLY, UNSAFE.  Every dispatch carries a full COMPUTE -> COMPUTE pipeline
// barrier (`encode_dispatch`'s `chain_barrier`), which is what ORDERS one kernel's write against the next
// kernel's read inside a recorded step.  This switch ELIDES it so the barrier can be PRICED - the answers are
// then WRONG wherever a real hazard exists.  Off unless the env var is set; never a shipping mode.
bool g_nobarrier = false;
// ---- STRATA_VK_NOBARRIER_REC: MEASUREMENT-ONLY, UNSAFE, and the LIVE path keeps its barriers.  `STRATA_VK_NOBARRIER`
// above cannot price the REPLAY arm's barriers at all: the prompt path needs them for correctness and the engine's own
// guard (`prefill: routed id out of range`) aborts the run inside the prefill, before a single decode round runs -
// measured, `/tmp/em/nobar199.log`.  This narrower switch elides the chain barrier ONLY where the dispatches were
// ENCODED into a recorded step (`fresh_set`, i.e. `record_dispatch`), so the prefill runs normally and the verify
// window's replay is barrier-free: the decode's `sync`/`wait` then PRICES those barriers.  The decode's answers are
// expected to be wrong; this is a diagnostic, never a candidate.
bool g_nobarrier_rec = false;
// ---- STRATA_VK_BARRIER_HAZARD: MEASUREMENT-ONLY hazard narrowing, so the chain barrier can be PRICED.
// The unconditional rule is one full COMPUTE -> COMPUTE pipeline barrier after EVERY chained dispatch.  The
// device layer has no read/write information - a binding is just a buffer - so "a real data hazard exists" is
// approximated CONSERVATIVELY: every bound region counts as both read and written, and a barrier is emitted only
// when the dispatch about to run touches a (buffer, byte-range) REGION that a dispatch since the last barrier
// also touched.  Otherwise the region is only remembered.  This can miss a hazard, which is why it prices the
// barrier rather than shipping; the ids are the guard.  Off unless the env var is set.
bool g_hazard_only = false;
struct HazardRange {
    VkBuffer buf;
    uint64_t off, end;
};
std::vector<HazardRange> g_haz;
constexpr size_t kHazardCap = 8192;   // force a barrier rather than grow without bound
uint64_t g_haz_barriers = 0;          // barriers actually EMITTED in the hazard mode
inline double vk_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void disp_stat_count(const std::string& spv) {
    for (auto& kv : g_ds.by_pipe) {
        if (kv.first == spv) { ++kv.second; return; }
    }
    g_ds.by_pipe.push_back({spv, 1});
}
void disp_stat_count_rec(const std::string& spv) {
    for (auto& kv : g_ds.by_pipe_rec) {
        if (kv.first == spv) { ++kv.second; return; }
    }
    g_ds.by_pipe_rec.push_back({spv, 1});
}
bool g_ds_printed = false;
void disp_stat_dump() {   // callable from both ~Ctx and atexit, so a leaked Ctx still reports
    if (!g_ds.on || g_ds_printed) return;
    g_ds_printed = true;
    const double total = g_ds.t_alloc + g_ds.t_encode + g_ds.t_fence + g_ds.t_submit + g_ds.t_wait + g_ds.t_free;
    std::fprintf(stderr,
                 "vk disp stat: %llu live dispatches | %llu recorded dispatches | %llu submits | %llu host waits | "
                 "%llu live batches | %llu cb allocs | %llu fences created | %llu descriptor sets | %llu pools\n",
                 (unsigned long long) g_ds.n, (unsigned long long) g_ds.recorded, (unsigned long long) g_ds.submits,
                 (unsigned long long) g_ds.waits, (unsigned long long) g_ds.batches,
                 (unsigned long long) g_ds.cb_allocs, (unsigned long long) g_ds.fences_created,
                 (unsigned long long) g_ds.sets_alloc, (unsigned long long) g_ds.pools_made);
    std::fprintf(stderr,
                 "vk disp stat ms: total %.0f = cb-alloc %.0f + encode %.0f + fence-create %.0f + submit %.0f + "
                 "wait %.0f + cb-free/fence-destroy %.0f  (%.4f ms/dispatch)\n",
                 total, g_ds.t_alloc, g_ds.t_encode, g_ds.t_fence, g_ds.t_submit, g_ds.t_wait, g_ds.t_free,
                 g_ds.n ? total / (double) g_ds.n : 0.0);
    std::vector<std::pair<std::string, uint64_t>> v = g_ds.by_pipe;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string line;
    for (size_t i = 0; i < v.size() && i < 14; ++i) {
        const size_t slash = v[i].first.find_last_of('/');
        line += " " + v[i].first.substr(slash == std::string::npos ? 0 : slash + 1) + " " +
                std::to_string(v[i].second);
    }
    std::fprintf(stderr, "vk disp stat by shader (top):%s\n", line.c_str());
    {
        std::vector<std::pair<std::string, uint64_t>> r = g_ds.by_pipe_rec;
        std::sort(r.begin(), r.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        uint64_t tot = 0;
        for (const auto& kv : r) tot += kv.second;
        std::string l2;
        for (size_t i = 0; i < r.size() && i < 24; ++i) {
            const size_t slash = r[i].first.find_last_of('/');
            l2 += " " + r[i].first.substr(slash == std::string::npos ? 0 : slash + 1) + " " +
                  std::to_string(r[i].second);
        }
        std::fprintf(stderr, "vk disp stat RECORDED arm (%llu dispatches encoded; each RECORDED command buffer is "
                             "re-executed once per segment submit, so this IS the replay arm's per-round "
                             "composition):%s\n",
                     (unsigned long long) tot, l2.c_str());
    }
    std::fprintf(stderr, "vk disp stat by arm: live-batch %llu | transfer %llu | recorded-submit %llu (replays %llu) | "
                         "segment %llu (%llu recorded dispatches, %llu chain barriers, submit %.0f ms, wait %.0f ms)\n",
                 (unsigned long long) g_ds.sub_live, (unsigned long long) g_ds.sub_transfer,
                 (unsigned long long) g_ds.sub_rec, (unsigned long long) g_ds.replays,
                 (unsigned long long) g_ds.sub_seg, (unsigned long long) g_ds.seg_disp,
                 (unsigned long long) g_ds.barriers, g_ds.t_seg_submit,
                 g_ds.t_seg_wait);
    if (g_hazard_only)
        std::fprintf(stderr,
                     "vk disp stat[HAZARD]: %llu of %llu chain barriers EMITTED (the rest were elided where no "
                     "bound REGION overlapped a dispatch since the last barrier)\n",
                     (unsigned long long) g_haz_barriers, (unsigned long long) g_ds.barriers);
}
}  // namespace

// ---- STRATA_VK_XFER_STAT: WHICH CALL SITE issues each staging transfer ---------------------------------------
// A "transfer" is `stage_upload`/`stage_download`: its own command buffer, its own fence, its own submit and its
// own wait.  The decode arm issues ~430 of them per token, so this instrument names each one's CALL SITE (a
// backtrace, symbolized by module-relative offset) and its direction and size, so the per-token count can be
// attributed to code instead of guessed.  Measurement-only: off unless the env var is set, and no transfer's
// bytes or ordering change.  Offsets are relative to the module base (or to a resolved symbol), so they resolve
// with `addr2line -f -C -e <binary> <off>` against the same build.
namespace {
struct XferSite {
    std::string key;
    uint64_t n = 0;
    uint64_t bytes = 0;
    uint64_t n_dec = 0;       // calls after the first capture_begin (the decode phase begins)
    uint64_t bytes_dec = 0;
};
struct XferStat {
    bool on = std::getenv("STRATA_VK_XFER_STAT") != nullptr;
    bool decode = false;      // set at the run's first capture_begin: the verify/decode arm starts there
    uint64_t up = 0, down = 0, up_bytes = 0, down_bytes = 0;
    uint64_t up_dec = 0, down_dec = 0, up_bytes_dec = 0, down_bytes_dec = 0;
    std::vector<XferSite> sites;
    void note(bool is_up, uint64_t bytes) {
        if (is_up) { ++up; up_bytes += bytes; if (decode) { ++up_dec; up_bytes_dec += bytes; } }
        else { ++down; down_bytes += bytes; if (decode) { ++down_dec; down_bytes_dec += bytes; } }
        void* fr[10];
        const int n = backtrace(fr, 10);
        std::string key = is_up ? "UP" : "DOWN";
        // frame 0 is `note`, frame 1 is `stage_*`; the call site is frame 2 and outward.
        for (int i = 1; i < n && i < 7; ++i) {
            Dl_info inf{};
            char b[320];
            if (dladdr(fr[i], &inf) && inf.dli_fbase) {
                unsigned long long off =
                    (unsigned long long) ((uintptr_t) fr[i] - (uintptr_t) inf.dli_fbase);
                const char* mod = inf.dli_fname ? inf.dli_fname : "?";
                if (inf.dli_sname != nullptr && inf.dli_saddr != nullptr) {
                    std::snprintf(b, sizeof b, "%s %s+0x%llx", mod, inf.dli_sname,
                                  (unsigned long long) ((uintptr_t) fr[i] - (uintptr_t) inf.dli_saddr));
                } else {
                    std::snprintf(b, sizeof b, "%s +0x%llx", mod, off);
                }
            } else {
                std::snprintf(b, sizeof b, "?%p", fr[i]);
            }
            if (i > 1) key += " <- ";
            key += b;
        }
        for (XferSite& s : sites) {
            if (s.key == key) {
                ++s.n; s.bytes += bytes;
                if (decode) { ++s.n_dec; s.bytes_dec += bytes; }
                return;
            }
        }
        XferSite ns{key, 1, bytes, 0, 0};
        if (decode) { ns.n_dec = 1; ns.bytes_dec = bytes; }
        sites.push_back(ns);
    }
};
XferStat g_xs;
bool g_xs_printed = false;
void xfer_stat_dump() {   // callable from both ~Ctx and atexit, so a leaked Ctx still reports
    if (!g_xs.on || g_xs_printed) return;
    g_xs_printed = true;
    std::fprintf(stderr,
                 "vk xfer stat: uploads %llu (%.2f MiB) | downloads %llu (%.2f MiB) | call sites %zu\n",
                 (unsigned long long) g_xs.up, g_xs.up_bytes / 1048576.0, (unsigned long long) g_xs.down,
                 g_xs.down_bytes / 1048576.0, g_xs.sites.size());
    std::fprintf(stderr, "vk xfer stat decode-phase (after the first capture_begin): up %llu (%.2f MiB) | down %llu (%.2f MiB)\n",
                 (unsigned long long) g_xs.up_dec, g_xs.up_bytes_dec / 1048576.0,
                 (unsigned long long) g_xs.down_dec, g_xs.down_bytes_dec / 1048576.0);
    std::sort(g_xs.sites.begin(), g_xs.sites.end(),
              [](const XferSite& a, const XferSite& b) { return a.n > b.n; });
    for (size_t i = 0; i < g_xs.sites.size(); ++i) {
        std::fprintf(stderr, "vk xfer site %2zu: n=%-7llu bytes=%-11llu dec_n=%-7llu dec_bytes=%-11llu %s\n", i,
                     (unsigned long long) g_xs.sites[i].n, (unsigned long long) g_xs.sites[i].bytes,
                     (unsigned long long) g_xs.sites[i].n_dec, (unsigned long long) g_xs.sites[i].bytes_dec,
                     g_xs.sites[i].key.c_str());
    }
}
}  // namespace

// ---- STRATA_VK_FLUSH_STAT: what TRIGGERS each live-batch flush, and what it costs ---------------------------
// The live dispatch path BATCHES.  A batch is flushed (one submit + one wait) when it fills, or when something
// must OBSERVE the device.  This records the flush's CALL SITE (a backtrace) with its count and the time spent
// in submit+wait, so "where does the decode's wait sit" is attributed to the dispatch that forced it.  Same
// measurement-only rule as the two instruments above.
namespace {
struct FlushSite {
    std::string key;
    uint64_t n = 0;
    uint64_t disp = 0;
    double wait_ms = 0;
    double submit_ms = 0;
    uint64_t n_dec = 0;
    uint64_t disp_dec = 0;
    double wait_ms_dec = 0;
};
bool g_fs_on_env = false;
// NAMESPACE-SCOPE, NOT A FUNCTION-LOCAL STATIC.  A function-local static's destructor is registered when it is
// first built, i.e. AFTER the atexit handler registered in `Ctx::Ctx`; at exit the handlers run in REVERSE order,
// so the vector would be destroyed BEFORE the dump touched it - which is exactly the `std::bad_alloc` this
// produced on its first outing.  A namespace-scope object is built at static-init (registered first, destroyed
// LAST), so the dump always runs against a live container.
std::vector<FlushSite> g_flush_sites;
uint64_t g_fs_n = 0, g_fs_disp = 0, g_fs_n_dec = 0, g_fs_disp_dec = 0;
double g_fs_wait = 0, g_fs_submit = 0, g_fs_wait_dec = 0;
std::string bt_key(int skip, int maxframes) {
    void* fr[12];
    const int n = backtrace(fr, 12);
    std::string key;
    for (int i = skip; i < n && i < skip + maxframes; ++i) {
        Dl_info inf{};
        char b[320];
        if (dladdr(fr[i], &inf) && inf.dli_fbase) {
            const char* mod = inf.dli_fname ? inf.dli_fname : "?";
            if (inf.dli_sname != nullptr && inf.dli_saddr != nullptr) {
                std::snprintf(b, sizeof b, "%s %s+0x%llx", mod, inf.dli_sname,
                              (unsigned long long) ((uintptr_t) fr[i] - (uintptr_t) inf.dli_saddr));
            } else {
                std::snprintf(b, sizeof b, "%s +0x%llx", mod,
                              (unsigned long long) ((uintptr_t) fr[i] - (uintptr_t) inf.dli_fbase));
            }
        } else {
            std::snprintf(b, sizeof b, "?%p", fr[i]);
        }
        if (!key.empty()) key += " <- ";
        key += b;
    }
    return key;
}
// one flush, charged to the call site that triggered it (and to the decode phase when it is one)
void flush_note(const std::string& key, uint64_t ndisp, double submit_ms, double wait_ms, bool decode) {
    bool found = false;
    for (FlushSite& f : g_flush_sites) {
        if (f.key == key) {
            ++f.n; f.disp += ndisp; f.submit_ms += submit_ms; f.wait_ms += wait_ms;
            if (decode) { ++f.n_dec; f.disp_dec += ndisp; f.wait_ms_dec += wait_ms; }
            found = true;
            break;
        }
    }
    if (!found) {
        FlushSite ns{key, 1, ndisp, wait_ms, submit_ms, 0, 0, 0.0};
        if (decode) { ns.n_dec = 1; ns.disp_dec = ndisp; ns.wait_ms_dec = wait_ms; }
        g_flush_sites.push_back(ns);
    }
    ++g_fs_n; g_fs_disp += ndisp; g_fs_submit += submit_ms; g_fs_wait += wait_ms;
    if (decode) { ++g_fs_n_dec; g_fs_disp_dec += ndisp; g_fs_wait_dec += wait_ms; }
}
bool g_flush_printed = false;
void flush_stat_dump() {
    if (!g_fs_on_env || g_flush_printed) return;
    g_flush_printed = true;
    std::fprintf(stderr, "vk flush stat: %llu live-batch flushes, %llu dispatches, submit %.0f ms, wait %.0f ms\n",
                 (unsigned long long) g_fs_n, (unsigned long long) g_fs_disp, g_fs_submit, g_fs_wait);
    std::fprintf(stderr,
                 "vk flush stat decode-phase (after the first capture_begin): %llu flushes, %llu dispatches, "
                 "wait %.0f ms\n",
                 (unsigned long long) g_fs_n_dec, (unsigned long long) g_fs_disp_dec, g_fs_wait_dec);
    std::vector<FlushSite> v = g_flush_sites;
    std::sort(v.begin(), v.end(), [](const FlushSite& a, const FlushSite& b) { return a.wait_ms > b.wait_ms; });
    for (size_t i = 0; i < v.size() && i < 20; ++i)
        std::fprintf(stderr,
                     "vk flush site %2zu: n=%-6llu disp=%-8llu submit=%.0fms wait=%.0fms | dec_n=%-6llu "
                     "dec_disp=%-7llu dec_wait=%.0fms  %s\n",
                     i, (unsigned long long) v[i].n, (unsigned long long) v[i].disp, v[i].submit_ms, v[i].wait_ms,
                     (unsigned long long) v[i].n_dec, (unsigned long long) v[i].disp_dec, v[i].wait_ms_dec,
                     v[i].key.c_str());
}
}  // namespace

// The desktop reserve, as a pure function so the policy can be tested without a GPU.  Rules: at least the
// floor (a compositor needs something, so a caller asking for nothing still gets the floor), at most
// `cap_percent_of_heap` of the card (a small card must stay usable for the engine at all), and a request in
// between is taken as given.
bool gemm_shape_ok(uint32_t m, uint32_t n, uint32_t k, uint32_t tile_m, uint32_t tile_n, uint32_t tile_k) {
    if (m == 0 || n == 0 || k == 0) return false;
    if (tile_m == 0 || tile_n == 0 || tile_k == 0) return false;
    return m % tile_m == 0 && n % tile_n == 0 && k % tile_k == 0;
}

// The attention window: 1..256 live keys (the engine's "width in [1, max_context]" with max_context <= 256) and a
// cache at least that wide, because the kernel's padding story assumes rows past `width` exist and are zero.
bool attn_short_shape_ok(uint32_t width, uint32_t capacity) {
    return width >= 1 && width <= 256 && capacity >= 256;
}

ReserveDecision compute_desktop_reserve(uint64_t requested_bytes, uint64_t heap_total_bytes, uint64_t floor_bytes,
                                       uint32_t cap_percent_of_heap) {
    ReserveDecision d{};
    uint64_t r = requested_bytes;
    if (r < floor_bytes) {
        r = floor_bytes;
        d.raised_to_floor = true;
    }
    if (heap_total_bytes) {
        const uint64_t cap = heap_total_bytes / 100ull * (uint64_t) cap_percent_of_heap;
        if (r > cap) {
            r = cap;
            d.clamped_by_cap = true;
        }
    }
    d.reserve_bytes = r;
    return d;
}

PlanVerdict plan_fit(const PlanItem* items, size_t n, uint64_t budget_bytes) {
    PlanVerdict v{};
    v.budget = budget_bytes;
    // AN EMPTY PLAN DOES NOT FIT.  There are no weights in it, so there is nothing to run, and answering
    // "true" would be an assertion over an empty input - the failure mode this port's rules exist for.
    if (items == nullptr || n == 0) return v;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t want = items[i].bytes;
        if (v.resident + want <= budget_bytes) {
            v.resident += want;
            continue;
        }
        if (items[i].droppable) {
            v.dropped += want;
            ++v.dropped_items;
            continue;
        }
        v.first_overflow = (int) i;
        return v;   // the engine cannot start: everything after this line is moot
    }
    v.fits = true;
    return v;
}

static bool device_has_extension(VkPhysicalDevice pd, const char* want) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> eps(n);
    if (n) vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, eps.data());
    for (const auto& e : eps) {
        if (std::strcmp(e.extensionName, want) == 0) return true;
    }
    return false;
}

static std::string fs_basename(const std::string& p) {
    const size_t at = p.find_last_of('/');
    return at == std::string::npos ? p : p.substr(at + 1);
}

// VK_KHR_cooperative_matrix, asked as two questions: does the extension+feature exist, and is there a config we
// can actually run?  Source of the second fact is the driver's own property list.  A null instance skips the
// query rather than guessing.
static void fill_coopmat(VkInstance inst, DeviceInfo& di, VkPhysicalDevice pd) {
    if (inst == VK_NULL_HANDLE) return;
    uint32_t ec = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, nullptr);
    std::vector<VkExtensionProperties> exts(ec);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, exts.data());
    bool has_ext = false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) has_ext = true;
    }
    if (!has_ext) return;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{};
    cmf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 q2{};
    q2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    q2.pNext = &cmf;
    vkGetPhysicalDeviceFeatures2(pd, &q2);
    di.cooperative_matrix = cmf.cooperativeMatrix == VK_TRUE;
    if (!di.cooperative_matrix) return;

    auto get_props = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
                         vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!get_props) return;
    uint32_t n = 0;
    get_props(pd, &n, nullptr);
    if (n == 0) return;
    std::vector<VkCooperativeMatrixPropertiesKHR> props(n);
    for (auto& c : props) {
        c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        c.pNext = nullptr;
    }
    get_props(pd, &n, props.data());
    // The tile is the DEVICE's, and the M dimension differs by generation (measured: M16 on RADV/WMMA, M8 on
    // Intel XMX).  Collect what exists, then select: M16 first, because that is the kernel this port verified on
    // a Radeon, and M8 as the fallback that makes the Arc's matrix units reachable at all.
    bool have_m8 = false;
    for (const auto& c : props) {
        if (c.scope != VK_SCOPE_SUBGROUP_KHR) continue;
        if (c.AType != VK_COMPONENT_TYPE_FLOAT16_KHR || c.BType != VK_COMPONENT_TYPE_FLOAT16_KHR) continue;
        if (c.CType != VK_COMPONENT_TYPE_FLOAT32_KHR) continue;
        if (c.MSize == 16 && c.NSize == 16 && c.KSize == 16) {
            di.cm_f16_f32 = true;
            di.cm_m = 16;
            di.cm_n = 16;
            di.cm_k = 16;
        } else if (c.MSize == 8 && c.NSize == 16 && c.KSize == 16) {
            have_m8 = true;
        }
    }
    if (!di.cm_f16_f32 && have_m8) {
        di.cm_m = 8;
        di.cm_n = 16;
        di.cm_k = 16;
    }
}

static void fill_info(DeviceInfo& di, VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    di.name = props.deviceName;
    di.vendor_id = props.vendorID;
    di.device_id = props.deviceID;
    di.api = props.apiVersion;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDevice8BitStorageFeatures f8q{};
    f8q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDevice16BitStorageFeatures f16{};
    f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    f16.pNext = &f8q;
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    f2.pNext = &f16;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    di.storage_buffer_16bit = f16.storageBuffer16BitAccess;
    di.storage_buffer_8bit = f8q.storageBuffer8BitAccess;
    di.shader_int16 = f2.features.shaderInt16;
    di.shader_float64 = f2.features.shaderFloat64;

    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    di.subgroup_size = sg.subgroupSize;
    // A descriptor offset must be a multiple of this.  Recorded here because it is the limit that decides whether
    // the engine's `X + t0 * K` row slices can be bound directly (see Buf::offset).
    di.min_storage_offset_align = p2.properties.limits.minStorageBufferOffsetAlignment;

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) di.heap_device_local_bytes += mp.memoryHeaps[i].size;
    }
}

std::vector<DeviceInfo> Ctx::list_devices() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return {};
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n) vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<DeviceInfo> out;
    for (auto pd : pds) {
        DeviceInfo di{};
        fill_info(di, pd);
        fill_coopmat(inst, di, pd);
        out.push_back(di);
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

Ctx::Ctx(int want_device, bool need_16bit) {
    g_ds.on = std::getenv("STRATA_VK_DISP_STAT") != nullptr;
    if (g_ds.on) std::atexit(disp_stat_dump);
    g_nobarrier = std::getenv("STRATA_VK_NOBARRIER") != nullptr;
    g_nobarrier_rec = std::getenv("STRATA_VK_NOBARRIER_REC") != nullptr;
    if (g_nobarrier_rec)
        std::fprintf(stderr, "vk_compute[MEASUREMENT]: STRATA_VK_NOBARRIER_REC=1 - the chain barrier is elided ONLY "
                             "inside a RECORDED step (the verify window's replay).  The LIVE path keeps its barriers, "
                             "so the run still reaches the decode; the decode's answers may be WRONG and this PRICES "
                             "the replay's barriers.  Not a shipping mode.\n");
    if (g_nobarrier)
        std::fprintf(stderr, "vk_compute[MEASUREMENT]: STRATA_VK_NOBARRIER=1 - the per-dispatch COMPUTE->COMPUTE "
                             "pipeline barrier is ELIDED.  The answers may be WRONG wherever a real hazard exists; "
                             "this PRICES the barrier, it is not a shipping mode.\n");
    g_hazard_only = std::getenv("STRATA_VK_BARRIER_HAZARD") != nullptr;
    if (g_hazard_only)
        std::fprintf(stderr, "vk_compute[MEASUREMENT]: STRATA_VK_BARRIER_HAZARD=1 - the chain barrier is emitted "
                             "only where a bound buffer REGION overlaps one touched since the last barrier.  A "
                             "hazard it does not see is a WRONG answer; this PRICES the barrier.\n");
    if (g_xs.on) std::atexit(xfer_stat_dump);
    g_fs_on_env = std::getenv("STRATA_VK_FLUSH_STAT") != nullptr;
    if (g_fs_on_env) std::atexit(flush_stat_dump);
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "strata-vulkan-port-gate";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, nullptr));
    if (n == 0) {
        std::fprintf(stderr, "no Vulkan physical devices\n");
        std::exit(1);
    }
    std::vector<VkPhysicalDevice> pds(n);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, pds.data()));

    // Pick the device: an explicit index wins; otherwise the first with a compute queue that has the
    // features the shaders need (so a lavapipe run does not silently win over the real GPU).
    int chosen = -1;
    for (uint32_t i = 0; i < n; ++i) {
        DeviceInfo di{};
        fill_info(di, pds[i]);
        fill_coopmat(instance_, di, pds[i]);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, qf.data());
        bool has_compute = false;
        for (auto& q : qf) has_compute = has_compute || (q.queueFlags & VK_QUEUE_COMPUTE_BIT);
        const bool ok16 = !need_16bit || (di.storage_buffer_16bit && di.shader_int16);
        if (want_device >= 0) {
            if ((int) i == want_device) {
                chosen = (int) i;
                break;
            }
            continue;
        }
        if (has_compute && ok16) {
            chosen = (int) i;
            break;
        }
    }
    if (chosen < 0 && want_device < 0) {
        std::fprintf(stderr, "no Vulkan device with a compute queue and the required features\n");
        std::exit(1);
    }
    if (chosen < 0) chosen = want_device;
    phys_ = pds[(size_t) chosen];
    device_index_ = chosen;
    fill_info(info_, phys_);
    fill_coopmat(instance_, info_, phys_);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            found = true;
            break;
        }
    }
    if (!found) {
        std::fprintf(stderr, "device %d has no compute queue family\n", chosen);
        std::exit(1);
    }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    // Ask for exactly what the ported kernels use, and only where the device reports it.  Requesting a
    // feature the device lacks fails vkCreateDevice outright - and `shaderFloat16` was requested here for no
    // reason at all (no ported kernel stores fp16), i.e. a pure compatibility risk on hardware that lacks it.
    // `shaderFloat64` IS now requested where the device reports it: quantize_q8_0 rounds its codes from a
    // FLOAT64 quotient (the reference divides in float64 and rints, and an f32 quotient crosses a .5 boundary
    // differently), so that kernel cannot run without this feature.  silu's double-precision form remains
    // inexpressible through glslang and is still host-side only.
    VkPhysicalDevice16BitStorageFeatures f16{};
    f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    f16.storageBuffer16BitAccess = info_.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    f2.pNext = &f16;
    VkPhysicalDevice8BitStorageFeatures f8feat{};
    f8feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    f8feat.storageBuffer8BitAccess = info_.storage_buffer_8bit ? VK_TRUE : VK_FALSE;
    f16.pNext = &f8feat;
    // A cooperative-matrix pipeline needs BOTH the feature and the extension enabled, and only where the device
    // reported them - the same rule as every other feature here.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmfeat{};
    cmfeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    cmfeat.cooperativeMatrix = info_.cooperative_matrix ? VK_TRUE : VK_FALSE;
    f8feat.pNext = &cmfeat;
    f2.features.shaderInt16 = info_.shader_int16 ? VK_TRUE : VK_FALSE;
    f2.features.shaderFloat64 = info_.shader_float64 ? VK_TRUE : VK_FALSE;

    // VK_EXT_memory_budget adds NO entry points: a capability check plus the name in the enabled list is the
    // whole wiring, and enabling it is what makes the driver report a budget instead of a raw heap size.
    const char* const kBUDGET_EXT = "VK_EXT_memory_budget";
    const bool want_budget_ext = device_has_extension(phys_, kBUDGET_EXT);
    std::vector<const char*> want_exts;
    if (want_budget_ext) want_exts.push_back(kBUDGET_EXT);
    if (info_.cooperative_matrix) want_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t) want_exts.size();
    dci.ppEnabledExtensionNames = want_exts.empty() ? nullptr : want_exts.data();
    VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
    vkGetDeviceQueue(dev_, queue_family_, 0, &queue_);

    // The Linux compatibility facts.  Collected once, after the device is chosen and before anything is sized.
    env_ = detect_host_env();
    fill_driver_info(phys_, env_);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    VK_CHECK(vkCreateCommandPool(dev_, &pci, nullptr, &cmd_pool_));

    // 256 storage-buffer descriptors per pool, 64 sets each - and MORE POOLS when those run out (set_alloc).
    // A FIXED ceiling here is a silent limit on how many cases the gate can hold: it surfaces as
    // VK_ERROR_OUT_OF_POOL_MEMORY at the end of a long run, on whichever implementation has the most to do.
    desc_pools_.push_back(new_desc_pool());
    desc_pool_ = desc_pools_.back();

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    mem_types_.clear();
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        MemTypeInfo t{};
        t.index = i;
        t.heap = mp.memoryTypes[i].heapIndex;
        t.heap_bytes = mp.memoryHeaps[t.heap].size;
        t.heap_device_local = (mp.memoryHeaps[t.heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        t.device_local = (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        t.host_visible = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        t.host_coherent = (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        t.host_cached = (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
        mem_types_.push_back(t);
    }
    mem_type_ = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((mp.memoryTypes[i].propertyFlags & want) != want) continue;
        if (mem_type_ == UINT32_MAX) mem_type_ = i;
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            mem_type_ = i;
            break;
        }
    }
    if (mem_type_ == UINT32_MAX) {
        std::fprintf(stderr, "no host-visible coherent memory type\n");
        std::exit(1);
    }

    // ---- stage 4's two types, chosen from the table above and then ASSERTED by the gate -----------------
    // VRAM: DEVICE_LOCAL and, where the device offers it, NOT host-visible - that last part is what makes a
    // transfer necessary at all, and on a discrete card it is the difference between VRAM and a BAR window.
    // Two passes rather than one, so a device that has both does not end up on the mappable one by order of
    // enumeration (which is exactly how the graphics-oriented drivers list them).
    for (int pass = 0; pass < 2 && vram_type_ == UINT32_MAX; ++pass) {
        for (const MemTypeInfo& t : mem_types_) {
            if (!t.device_local) continue;
            if (pass == 0 && t.host_visible) continue;
            vram_type_ = t.index;
            // Pass 0 succeeded => the type we picked has NO mapping.  Recorded here because it is the one place
            // that knows, and a later `device_local && !host_visible` re-derivation reads the type table again -
            // which is how llvmpipe (one type, device-local AND mappable) was mistaken for a device with real
            // VRAM by an earlier version of this accessor.
            vram_unmappable_ = (pass == 0);
            break;
        }
    }
    // STAGING: host-visible and coherent first, and preferring a heap that is NOT device-local - a transfer
    // buffer is not model memory, so on a card with a system heap it must not come out of VRAM.  Where the only
    // heap is device-local (llvmpipe) there is no choice, and the account rule says so out loud.
    for (int pass = 0; pass < 2 && staging_type_ == UINT32_MAX; ++pass) {
        for (const MemTypeInfo& t : mem_types_) {
            if (!t.host_visible || !t.host_coherent) continue;
            if (pass == 0 && t.heap_device_local) continue;
            staging_type_ = t.index;
            break;
        }
    }
    if (staging_type_ == UINT32_MAX) staging_type_ = mem_type_;

    // HOST ALLOCATIONS (`cudaHostAlloc`/`cudaMallocHost`): host-visible and coherent, and - the defect this
    // selection fixes - in a heap that is NOT device-local, i.e. SYSTEM RAM, wherever the device offers one.
    // The old rule reused `mem_type_`, which PREFERS a DEVICE_LOCAL host-visible type; on the Arc that is the
    // BAR-mapped VRAM type (heap 0), so a "host" buffer came out of VRAM.  Two consequences, both measured:
    // the engine's PCIe probe (`probe_pcie_h2d_gbps`) times an H2D copy whose SOURCE is a `cudaMallocHost`
    // block, so it timed a BAR read of VRAM (~0.06 GB/s) instead of a link transfer (~1.8 GB/s from system RAM)
    // and produced `pcie_frac 0.00` - a WRONG decision input where the engine's own default (0.55) is right; and
    // every host-tier allocation silently consumed device memory.  The selection mirrors `staging_type_` and
    // FALLS BACK to `mem_type_` only where no non-device-local host type exists (llvmpipe: one heap, device-local
    // and mappable), so nothing that used to work is left without a type.
    for (int pass = 0; pass < 2 && host_type_ == UINT32_MAX; ++pass) {
        for (const MemTypeInfo& t : mem_types_) {
            if (!t.host_visible || !t.host_coherent) continue;
            if (pass == 0 && t.heap_device_local) continue;
            host_type_ = t.index;
            break;
        }
    }
    if (host_type_ == UINT32_MAX) host_type_ = mem_type_;

    // WHICH TYPE WENT WHERE, OBSERVABLE (STRATA_VK_MEM_TRACE=1).  This is the guard for the "a setting that
    // silently does nothing" class: the three chosen types decide whether a `cudaHostAlloc` block is system RAM
    // or VRAM behind the BAR, and that choice is not visible anywhere else.  A probe that times a copy out of a
    // BAR-mapped block reads ~0.06 GB/s where the same copy from system RAM reads ~1.95 GB/s (measured on the
    // Arc B70) - so the choice must be inspectable, not assumed.  No behaviour changes: this only prints.
    {
        const char* mt = std::getenv("STRATA_VK_MEM_TRACE");
        if (mt != nullptr && *mt != '\0') {
            for (const MemTypeInfo& t : mem_types_)
                std::fprintf(stderr,
                             "vk_mem[%u]: heap %u (%s, %.2f GiB) device_local=%d host_visible=%d "
                             "host_coherent=%d host_cached=%d\n",
                             t.index, t.heap, t.heap_device_local ? "DEVICE_LOCAL" : "host",
                             (double) t.heap_bytes / 1073741824.0, (int) t.device_local, (int) t.host_visible,
                             (int) t.host_coherent, (int) t.host_cached);
            std::fprintf(stderr,
                         "vk_mem: arena/vram type %u | cudaHostAlloc type %u (heap %u: %s) | alloc() type %u | "
                         "staging type %u\n",
                         vram_type_, host_type_, mem_types_[host_type_].heap,
                         mem_types_[host_type_].heap_device_local
                             ? "DEVICE_LOCAL - a host buffer from VRAM (the defect)"
                             : "system RAM (fixed)",
                         mem_type_, staging_type_);
        }
    }

    const char* fs = std::getenv("STRATA_VK_FORCE_STAGING");
    force_staging_ = fs != nullptr && *fs && std::strcmp(fs, "0") != 0;
}

Ctx::~Ctx() {
    disp_stat_dump();
    xfer_stat_dump();
    flush_stat_dump();
    flush_live();   // the last batch must reach the device before the device goes away
    if (live_fence_ && dev_) vkDestroyFence(dev_, live_fence_, nullptr);
    if (live_cb_ && cmd_pool_) vkFreeCommandBuffers(dev_, cmd_pool_, 1, &live_cb_);
    if (live_pool_ && dev_) vkDestroyDescriptorPool(dev_, live_pool_, nullptr);
    if (dev_ != VK_NULL_HANDLE) vkDeviceWaitIdle(dev_);
    for (Pipe& pv : pipes_) {
        if (pv.pipe) vkDestroyPipeline(dev_, pv.pipe, nullptr);
        if (pv.layout) vkDestroyPipelineLayout(dev_, pv.layout, nullptr);
        if (pv.set_layout) vkDestroyDescriptorSetLayout(dev_, pv.set_layout, nullptr);
    }
    if (rec_fence_) vkDestroyFence(dev_, rec_fence_, nullptr);   // the recorded step's fence: one fence for every submission
    for (VkDescriptorPool pool : desc_pools_) vkDestroyDescriptorPool(dev_, pool, nullptr);
    desc_pools_.clear();
    if (cmd_pool_) vkDestroyCommandPool(dev_, cmd_pool_, nullptr);
    if (dev_) vkDestroyDevice(dev_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Ctx::query_budget() {
    budget_.heap_total = info_.heap_device_local_bytes;   // the cap needs a heap size to take a fraction of
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    std::vector<uint32_t> local_heaps;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local_heaps.push_back(i);
    }

    if (!force_no_budget_ext_) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{};
        bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp2{};
        mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp2.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(phys_, &mp2);
        uint64_t b = 0, u = 0;
        for (uint32_t h : local_heaps) {
            b += bp.heapBudget[h];
            u += bp.heapUsage[h];
        }
        budget_.from_driver = true;
        budget_.heap_budget = b;
        budget_.heap_usage = u;
    } else {
        // The FALLBACK is a heap size, which knows nothing about who else is using the card - including the
        // desktop.  It is labelled, never silent: a caller that treats it as free memory will fill the card.
        budget_.from_driver = false;
        std::fprintf(stderr, "vk_compute: VK_EXT_memory_budget unavailable (or disabled) - free memory is a "
                             "LEDGER (heap total minus this process), not a measurement\n");
    }
    if (forced_budget_bytes_) {
        // An explicit ceiling is NOT a driver measurement, and the logs must not pretend otherwise.
        budget_.from_driver = true;
        budget_.from_explicit_limit = true;
        budget_.heap_budget = forced_budget_bytes_;
        budget_.heap_usage = 0;
    }
}

void Ctx::configure_display_reserve() {
    const char* noext = std::getenv("STRATA_VK_NO_MEMORY_BUDGET");
    force_no_budget_ext_ = noext != nullptr && *noext && std::strcmp(noext, "0") != 0;

    const char* forced = std::getenv("STRATA_VK_MAX_BUDGET_MIB");
    forced_budget_bytes_ = forced ? (uint64_t) std::strtoull(forced, nullptr, 10) << 20 : 0;

    // 1024 MiB by default: enough for a compositor plus a browser doing GPU compositing at 4K, and NOT sized
    // for a game (the engine's own `--vram-reserve-mib` 700 MiB covers its graphs/scratch/head separately, so
    // the two compose rather than overlapping).
    uint64_t want_bytes = 1024ull << 20;
    const char* rsv = std::getenv("STRATA_VK_DESKTOP_RESERVE_MIB");
    if (rsv && *rsv) want_bytes = (uint64_t) std::strtoull(rsv, nullptr, 10) << 20;

    // 512 MiB floor, raised from 256 on field evidence: a KDE/Wayland B580 desktop with two displays was
    // measured holding 354 MB of GPU memory in kwin_wayland ALONE (before any application), so a 256 MiB floor
    // was a reserve smaller than a real desktop's compositor.  A card with nothing to spare still must not be
    // filled to the last byte.  Lowering the floor to 0 is a TEST hook - the numerical gate needs a few MiB of
    // buffers, and on this box the resident local model already holds the card, so even the floor would
    // (correctly) refuse the gate.
    uint64_t floor_bytes = 512ull << 20;
    const char* flr = std::getenv("STRATA_VK_RESERVE_FLOOR_MIB");
    if (flr && *flr) floor_bytes = (uint64_t) std::strtoull(flr, nullptr, 10) << 20;

    query_budget();
    reserve_decision_ = compute_desktop_reserve(want_bytes, budget_.heap_total, floor_bytes, 25);
    reserve_bytes_ = reserve_decision_.reserve_bytes;

    // THE LEDGER RULE, and the one place the port refuses on a version/driver combination rather than on a
    // request.  A free figure that comes from the heap total rather than from the driver says nothing about
    // what the desktop and other processes hold, and on a discrete display card that is the exact shape that
    // filled an RX 6800 and cost 11 tok/s (docs/AMD_HIP.md, #380/#377).  So: ledger + discrete + no explicit
    // ceiling = nothing may be allocated, with the reason and the remedy in the message.
    ledger_untrusted_ = !budget_.from_driver && looks_discrete(budget_.heap_total, env_.host_ram_bytes) &&
                        forced_budget_bytes_ == 0;

    stack_ = detect_stack();
    advisories_ = compat_advisories(env_, info_.vendor_id, info_.device_id, budget_.from_driver);
    for (const Advisory& a : stack_advisories(stack_, info_.vendor_id, info_.device_id)) advisories_.push_back(a);

    // The stack table, printed whether or not anything is wrong, because "which loader / which ICD / which
    // libdrm" is the first question whenever a GPU is missing from enumeration.
    std::fprintf(stderr, "vk_stack: loader %s", stack_.loader_version.empty() ? "(not found)"
                                                                             : stack_.loader_version.c_str());
    if (!stack_.libdrm_version.empty()) std::fprintf(stderr, " | libdrm %s", stack_.libdrm_version.c_str());
    std::fprintf(stderr, " | session %s | Level-Zero %s | OpenCL %s\n", stack_.session_type.c_str(),
                 stack_.level_zero ? "yes" : "no", stack_.opencl ? "yes" : "no");
    for (const IcdEntry& e : stack_.icds) {
        std::fprintf(stderr, "vk_stack: ICD %-22s api %-9s -> %s\n", fs_basename(e.file).c_str(),
                     e.api_version.c_str(), e.resolves() ? e.resolved.c_str() : "!! UNRESOLVED");
    }
    for (const StackFile& f : stack_.firmware) {
        std::fprintf(stderr, "vk_stack: firmware %-22s %s\n", f.name.c_str(),
                     f.present() ? f.found_path.c_str() : "(absent)");
    }
    for (const Advisory& a : advisories_) {
        std::fprintf(stderr, "vk_compat[%s] %s\n", severity_name(a.sev), a.text.c_str());
    }
    // The userspace version is printed next to the kernel one on purpose: on Intel it is the Mesa version, not
    // the kernel, that decides whether VK_EXT_memory_budget gives a real free figure.
    std::fprintf(stderr, "vk_compat: kernel %s (%s) | driver \"%s\" / \"%s\" | DRM modules: ",
                 env_.kernel.str().c_str(), env_.kernel.release.c_str(), env_.driver_name.c_str(),
                 env_.mesa_version.c_str());
    for (const auto& m : env_.drm_modules) std::fprintf(stderr, "%s ", m.c_str());
    std::fprintf(stderr, "| host RAM %.1f GiB\n", (double) env_.host_ram_bytes / 1073741824.0);

    // Refuse only where the combination is known to be impossible (today: Battlemage on a kernel that has no
    // BMG support).  Everything else is reported and left to the operator.
    if (any_fatal(advisories_)) {
        std::fprintf(stderr, "vk_compute: refusing to run on this combination (see the FATAL line above)\n");
        std::exit(4);
    }
    if (ledger_untrusted_) {
        std::fprintf(stderr,
                     "vk_compute: the free figure is a LEDGER and this is a discrete card, so it cannot be used "
                     "to size anything (%.2f GiB of %.2f GiB is unreserved, but the desktop's and other "
                     "processes' usage is not in that number).  Set STRATA_VK_MAX_BUDGET_MIB to the ceiling you "
                     "want, or install Mesa >= 26.2 for VK_EXT_memory_budget on Intel.\n",
                     (double) budget_.heap_total / 1073741824.0, (double) budget_.heap_total / 1073741824.0);
    }
    std::fprintf(stderr,
                 "vk_compute: heap total %.2f GiB | free %s %.2f GiB | desktop reserve %.2f GiB%s%s -> %.2f GiB "
                 "usable\n",
                 (double) budget_.heap_total / 1073741824.0, (budget_.from_explicit_limit ? "(explicit limit)" : (budget_.from_driver ? "(driver)" : "(LEDGER)")),
                 (double) (budget_.from_driver ? (budget_.heap_budget - budget_.heap_usage) : budget_.heap_total) /
                     1073741824.0,
                 (double) reserve_bytes_ / 1073741824.0, reserve_decision_.raised_to_floor ? " (floor)" : "",
                 reserve_decision_.clamped_by_cap ? " (clamped to 25% of the card)" : "",
                 (double) usable_bytes() / 1073741824.0);
}

uint64_t Ctx::usable_bytes() const {
    if (ledger_untrusted_) return 0;   // see configure_display_reserve: refuse, never guess
    uint64_t free_b;
    if (budget_.from_driver) {
        free_b = budget_.heap_budget > budget_.heap_usage ? budget_.heap_budget - budget_.heap_usage : 0;
    } else {
        free_b = budget_.heap_total > allocated_ ? budget_.heap_total - allocated_ : 0;
    }
    return free_b > reserve_bytes_ ? free_b - reserve_bytes_ : 0;
}

Buf Ctx::alloc_impl(uint64_t bytes, uint32_t type_index, bool vram_account, const char* what) {
    if (type_index == UINT32_MAX || type_index >= mem_types_.size()) {
        std::fprintf(stderr, "vk_compute: this device has no memory type for a %s\n", what);
        std::exit(1);
    }
    Buf b;
    b.bytes = bytes ? bytes : 4;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = b.bytes;
    // TRANSFER_SRC/DST as well as STORAGE: stage 4's path moves bytes with vkCmdCopyBuffer, and a usage flag that
    // is missing shows up as a validation error at the first transfer rather than as a wrong number.
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev_, b.buffer, &req);

    // THE DISPLAY CONTRACT, ENFORCED.  Checked against the driver's own figure (not this layer's ledger) plus
    // the reserve, before anything is allocated.  A backend that cannot fit must refuse and name the numbers -
    // allocating anyway is exactly how the card gets filled and the desktop stops compositing.  STAGE 4 DID NOT
    // RELAX THIS: what changed is which account is checked (the VRAM one) and what a staging buffer is charged
    // to - see the account rule in the header.
    const uint64_t usable = usable_bytes();
    if (vram_account && allocated_device_local_ + req.size > usable) {
        vkDestroyBuffer(dev_, b.buffer, nullptr);
        std::fprintf(stderr,
                     "vk_compute: REFUSING a %.2f MiB %s - %.2f MiB in the VRAM account, %.2f MiB usable "
                     "(free %s %.2f GiB, desktop reserve %.2f GiB).  Raise STRATA_VK_DESKTOP_RESERVE_MIB only if "
                     "the desktop can spare it.\n",
                     (double) req.size / 1048576.0, what, (double) allocated_device_local_ / 1048576.0,
                     (double) usable / 1048576.0,
                     (budget_.from_explicit_limit ? "(explicit limit)" : (budget_.from_driver ? "(driver)" : "(LEDGER)")),
                     (double) (budget_.from_driver ? (budget_.heap_budget - budget_.heap_usage) : budget_.heap_total) /
                         1073741824.0,
                     (double) reserve_bytes_ / 1073741824.0);
        std::exit(3);
    }
    allocated_ += req.size;
    if (vram_account) allocated_device_local_ += req.size;
    else allocated_host_ += req.size;

    const MemTypeInfo& t = mem_types_[type_index];
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type_index;
    VK_CHECK(vkAllocateMemory(dev_, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev_, b.buffer, b.mem, 0));

    // What the DRIVER gave us, recorded rather than assumed - and it is these fields the gate asserts against.
    b.mem_type = type_index;
    b.device_local = t.device_local;
    b.host_visible = t.host_visible;
    b.vram_account = vram_account;
    if (b.host_visible) {
        VK_CHECK(vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
        std::memset(b.mapped, 0, (size_t) b.bytes);
    }
    return b;
}

Buf Ctx::alloc(uint64_t bytes) {
    // The gate's path, and the engine arena's: charged to the VRAM account whatever heap the driver puts it in.
    return alloc_impl(bytes, mem_type_, /*vram_account=*/true, "buffer");
}

Buf Ctx::alloc_host(uint64_t bytes) {
    // `cudaHostAlloc`/`cudaMallocHost`: HOST_VISIBLE | HOST_COHERENT in the NON-device-local host type where the
    // device has one, and charged to the HOST account when it lands there - the account rule, "charged where it
    // lands", the same one `alloc_staging` follows.  Where the only mappable heap is device-local (llvmpipe) it
    // falls back to `mem_type_` and is charged to VRAM, which is the truth about that device rather than a
    // policy.  The two effects of the old type (a BAR-timed PCIe probe, and host tiers spending device memory)
    // are both consequences of landing in VRAM; this is the one decision that moves them.
    const uint32_t type = host_type_ == UINT32_MAX ? mem_type_ : host_type_;
    const bool vram = type >= mem_types_.size() || mem_types_[type].heap_device_local;
    return alloc_impl(bytes, type, vram, "host buffer");
}

Buf Ctx::alloc_device(uint64_t bytes) {
    return alloc_impl(bytes, vram_type_, /*vram_account=*/true, "device-local buffer");
}

Buf Ctx::alloc_staging(uint64_t bytes) {
    // CHARGED WHERE IT LANDS: on a device with a system heap a transfer buffer is not model memory and does not
    // spend the VRAM account; where the only heap is device-local (llvmpipe) there is nowhere else for it to be,
    // so it is charged there - which is the truth about that device rather than a policy.
    const uint32_t type = staging_type_ == UINT32_MAX ? mem_type_ : staging_type_;
    const bool vram = type >= mem_types_.size() || mem_types_[type].heap_device_local;
    return alloc_impl(bytes, type, vram, "staging buffer");
}

void Ctx::free(Buf& b) {
    // The ledger follows the driver's own allocation size, so the accounting cannot drift from reality.
    if (b.mem != VK_NULL_HANDLE) {
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, b.buffer, &req);
        allocated_ = allocated_ > req.size ? allocated_ - req.size : 0;
        uint64_t& acct = b.vram_account ? allocated_device_local_ : allocated_host_;
        acct = acct > req.size ? acct - req.size : 0;
    }
    if (b.mapped) vkUnmapMemory(dev_, b.mem);
    if (b.buffer) vkDestroyBuffer(dev_, b.buffer, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buf{};
}

// One descriptor pool.  64 sets is plenty for a case, and set_alloc() creates another when it is not - see the
// ceiling note in the header.
// A descriptor offset must be a multiple of the device's `minStorageBufferOffsetAlignment`.  An unaligned one is
// not "slightly wrong": the bind is invalid and what the shader reads is undefined - on one driver and not another,
// which is the worst kind.  The engine reaches rows by pointer arithmetic, so this check is what says whether that
// arithmetic is portable for a given row stride.
void Ctx::check_offsets(const std::vector<const Buf*>& bufs) const {
    if (info_.min_storage_offset_align == 0) return;
    for (const Buf* b : bufs) {
        if (b == nullptr || b->offset == 0) continue;
        if (b->offset % info_.min_storage_offset_align != 0) {
            std::fprintf(stderr,
                         "dispatch: descriptor offset %llu is not a multiple of the device's %u-byte alignment "
                         "(minStorageBufferOffsetAlignment) - the bind would be invalid and the read undefined\n",
                         (unsigned long long) b->offset, info_.min_storage_offset_align);
            std::exit(2);
        }
    }
}

VkDescriptorPool Ctx::new_desc_pool() {
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 256;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &pool));
    if (g_ds.on) ++g_ds.pools_made;
    return pool;
}

// A set out of the current pool, or out of a NEW one when that pool is full.  Nothing is recycled - the gate
// allocates one set per pipeline and an extra per dispatch inside a recorded step - so the only thing that keeps
// the case count from being capped is growing here.
VkDescriptorSet Ctx::set_alloc(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = desc_pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult r = vkAllocateDescriptorSets(dev_, &dsai, &set);
    if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        desc_pools_.push_back(new_desc_pool());
        desc_pool_ = desc_pools_.back();
        dsai.descriptorPool = desc_pool_;
        r = vkAllocateDescriptorSets(dev_, &dsai, &set);
        // Say what happened: a case that allocates a lot is worth knowing about, and the alternative is a
        // mysterious failure at the end of an unrelated case.
        std::fprintf(stderr, "  (descriptor pool %u created: the previous one was full)\n",
                     (unsigned) desc_pools_.size());
    }
    VK_CHECK(r);
    if (g_ds.on) ++g_ds.sets_alloc;
    return set;
}

// A set out of a GIVEN pool.  The live batch's sets must each be distinct (its dispatches run at SUBMIT, so one
// shared set would leave them all reading the last binding) but are only needed for one submission, so the pool
// is RESET at each flush and the sets are recycled instead of growing the pool per dispatch.
VkDescriptorSet Ctx::set_alloc_in(VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(dev_, &dsai, &set));
    if (g_ds.on) ++g_ds.sets_alloc;
    return set;
}

// The live batch's ONE command buffer, ONE fence and ONE (recycled) descriptor pool.
void Ctx::live_ensure() {
    if (live_cb_ != VK_NULL_HANDLE) return;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &cbai, &live_cb_));
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &live_fence_));
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = kLiveBatchMax * 16u;   // 16 = the widest binding set any shader here takes
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = kLiveBatchMax;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &live_pool_));
    if (g_ds.on) ++g_ds.pools_made;
}

// Submit the open batch and wait ONCE for all of it.  Cheap when nothing is open.  The host-read barrier is the
// one the old per-dispatch path added after every dispatch, hoisted to the end of the batch.
void Ctx::flush_live() {
    if (!live_open_) return;
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(live_cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(live_cb_));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &live_cb_;
    VK_CHECK(vkResetFences(dev_, 1, &live_fence_));
    const double _ts = vk_ms();
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, live_fence_));
    const double _tw = vk_ms();
    VK_CHECK(vkWaitForFences(dev_, 1, &live_fence_, VK_TRUE, UINT64_MAX));
    const double _td = vk_ms();
    live_open_ = false;
    if (g_ds.on) {
        g_ds.t_submit += _tw - _ts;
        g_ds.t_wait += _td - _tw;
        ++g_ds.submits; ++g_ds.waits; ++g_ds.batches; ++g_ds.sub_live;
    }
    if (g_fs_on_env) {
        const std::string key = bt_key(2, 5);   // 0 = bt_key, 1 = flush_live, 2 = the caller
        flush_note(key, (uint64_t) live_n_, _tw - _ts, _td - _tw, g_xs.decode);
    }
}

VkCommandBuffer Ctx::begin_oneshot() {
    // Every other submission path (the staging transfers) starts here: the live batch must be on the device
    // BEFORE this transfer is submitted, or the queue order the fence used to give us is lost.
    flush_live();
    const double _t0 = vk_ms();
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &cbai, &cb));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    if (g_ds.on) { g_ds.t_alloc += vk_ms() - _t0; ++g_ds.cb_allocs; }
    return cb;
}

void Ctx::end_oneshot_and_wait(VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    const double _tf = vk_ms();
    VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &fence));
    const double _ts = vk_ms();
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence));
    const double _tw = vk_ms();
    VK_CHECK(vkWaitForFences(dev_, 1, &fence, VK_TRUE, UINT64_MAX));
    const double _td = vk_ms();
    vkDestroyFence(dev_, fence, nullptr);
    vkFreeCommandBuffers(dev_, cmd_pool_, 1, &cb);
    if (g_ds.on) {
        g_ds.t_fence += _ts - _tf;
        g_ds.t_submit += _tw - _ts;
        g_ds.t_wait += _td - _tw;
        g_ds.t_free += vk_ms() - _td;
        ++g_ds.submits; ++g_ds.waits; ++g_ds.fences_created; ++g_ds.sub_transfer;
    }
}

// ---- the staging transfers (stage 4).  THE BARRIERS ARE THE WHOLE DIFFICULTY ---------------------------------
// A copy that is submitted and waited on is not automatically visible to the NEXT thing that reads the buffer:
// the fence orders the SUBMISSIONS, and availability/visibility across stages is what the barriers below are
// for.  Getting one wrong does not fail a build, and on a coherent host it does not even fail on the machine it
// was written on - which is why each direction states its own reasoning and the gate round-trips through it.
void Ctx::stage_upload(Buf& dst, const void* src, uint64_t bytes, uint64_t offset) {
    g_xs.note(/*is_up=*/true, bytes);
    Buf st = alloc_staging(bytes ? bytes : 4);
    std::memcpy(st.mapped, src, (size_t) bytes);   // HOST_COHERENT: no flush, and the submit below needs none
    VkCommandBuffer cb = begin_oneshot();
    VkBufferCopy c{};
    c.srcOffset = 0;
    c.dstOffset = offset;
    c.size = bytes;
    vkCmdCopyBuffer(cb, st.buffer, dst.buffer, 1, &c);
    // The copy's write must be available to whatever reads the destination NEXT: a shader (the ordinary case -
    // upload weights, then run a kernel over them) or the host (an upload followed by a read of the same buffer).
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
    end_oneshot_and_wait(cb);
    free(st);
}

void Ctx::stage_download(const Buf& src, void* dst, uint64_t bytes, uint64_t offset) {
    g_xs.note(/*is_up=*/false, bytes);
    Buf st = alloc_staging(bytes ? bytes : 4);
    VkCommandBuffer cb = begin_oneshot();
    // The source was written by a shader or by an earlier copy, in an earlier submission the caller fenced and
    // waited on.  Without this barrier the transfer may read the bytes before they are available to it, which is
    // the stale-read direction of the same trap.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy c{};
    c.srcOffset = offset;
    c.dstOffset = 0;
    c.size = bytes;
    vkCmdCopyBuffer(cb, src.buffer, st.buffer, 1, &c);
    // ...and the host reads the staging buffer after this submission's fence, which needs the transfer's write
    // to be available to HOST_READ.
    VkMemoryBarrier mb2{};
    mb2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb2, 0, nullptr, 0,
                         nullptr);
    end_oneshot_and_wait(cb);
    std::memcpy(dst, st.mapped, (size_t) bytes);
    free(st);
}

void Ctx::write(Buf& b, const void* src, uint64_t bytes, uint64_t offset) {
    // A host store into a buffer a PENDING batch reads must not land after the batch is submitted (the old
    // per-dispatch submit+wait made that impossible; with the batch it would be a real race).
    flush_live();
    if (offset + bytes > b.bytes) {
        std::fprintf(stderr, "write past end of buffer (%llu+%llu > %llu)\n", (unsigned long long) offset,
                     (unsigned long long) bytes, (unsigned long long) b.bytes);
        std::exit(1);
    }
    // THE MAPPING IS THE FAST PATH AND IT IS NOT ALWAYS THERE.  force_staging_ makes the gate take the slow one
    // on a device that would otherwise take the mapping, so both paths are exercised on every implementation.
    if (b.mapped != nullptr && !force_staging_) {
        std::memcpy((uint8_t*) b.mapped + offset, src, (size_t) bytes);
        return;
    }
    stage_upload(b, src, bytes, offset);
}

void Ctx::read(const Buf& b, void* dst, uint64_t bytes, uint64_t offset) {
    // The host read must see a pending batch's writes, so the batch has to be ON the device first.
    flush_live();
    if (offset + bytes > b.bytes) {
        std::fprintf(stderr, "read past end of buffer\n");
        std::exit(1);
    }
    if (b.mapped != nullptr && !force_staging_) {
        std::memcpy(dst, (const uint8_t*) b.mapped + offset, (size_t) bytes);
        return;
    }
    stage_download(b, dst, bytes, offset);
}

VkPipeline Ctx::pipeline(const std::string& spv_path, uint32_t nbufs, uint32_t push_bytes) {
    for (const Pipe& p : pipes_) {
        if (p.spv_path == spv_path && p.nbufs == nbufs && p.push_bytes == push_bytes) return p.pipe;
    }
    Pipe pv{};
    pv.spv_path = spv_path;
    pv.nbufs = nbufs;
    pv.push_bytes = push_bytes;

    std::vector<VkDescriptorSetLayoutBinding> binds(nbufs);
    for (uint32_t i = 0; i < nbufs; ++i) {
        binds[i] = {};
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = nbufs;
    slci.pBindings = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev_, &slci, nullptr, &pv.set_layout));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_bytes;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &pv.set_layout;
    plci.pushConstantRangeCount = push_bytes ? 1u : 0u;
    plci.pPushConstantRanges = push_bytes ? &pcr : nullptr;
    VK_CHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &pv.layout));

    // A descriptor set per pipeline, allocated up front and never reset: the gate runs each kernel a handful
    // of times, so recycling is a stage-3 optimisation, not a correctness one.  `set_alloc` grows the pool if
    // this many pipelines no longer fit - the case count is not a capacity limit of the harness.
    pv.set = set_alloc(pv.set_layout);

    const std::vector<uint8_t> code = read_file(spv_path);
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode = (const uint32_t*) code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(dev_, &smci, nullptr, &module));

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = pv.layout;
    VK_CHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pv.pipe));
    vkDestroyShaderModule(dev_, module, nullptr);

    pipes_.push_back(pv);
    return pv.pipe;
}

void Ctx::encode_dispatch(VkCommandBuffer cb, VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push,
                          uint32_t push_bytes, uint32_t groups, uint32_t groups_y, bool chain_barrier,
                          bool fresh_set, VkDescriptorSet forced_set) {
    // Find the pipeline layout/set that belongs to this pipeline handle.
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    for (const Pipe& p : pipes_) {
        if (p.pipe == pipe) {
            layout = p.layout;
            set = p.set;
            set_layout = p.set_layout;
        }
    }
    if (!layout) {
        std::fprintf(stderr, "dispatch: unknown pipeline\n");
        std::exit(1);
    }
    // ONE DESCRIPTOR SET PER DISPATCH IN A RECORDED STEP.  The host-side update below happens while RECORDING,
    // but the dispatches execute at SUBMIT time - so a single shared set would leave every dispatch in the step
    // reading whatever the LAST one bound.  (The same class of trap the grouped-expert wave hit with one buffer
    // pointer standing in for two.)  A real backend pools these sets; the gate allocates them out of desc_pool_,
    // which the device destroys with the pool, so a re-recording leaks a handful of sets and nothing else.
    if (forced_set != VK_NULL_HANDLE) set = forced_set;
    else if (fresh_set) set = set_alloc(set_layout);

    check_offsets(bufs);

    std::vector<VkDescriptorBufferInfo> info(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        info[i] = {};
        info[i].buffer = bufs[i]->buffer;
        info[i].offset = bufs[i]->offset;          // a view binds from here; the default is 0
        info[i].range = VK_WHOLE_SIZE;
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t) i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(dev_, (uint32_t) writes.size(), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (push_bytes) vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cb, groups, groups_y, 1);
    if (chain_barrier) {
        // One kernel's output is the next one's input inside a recorded step.  Compute -> compute, not to host:
        // the step has exactly one host-read barrier, at its end.  COUNTED whether or not it is emitted (the
        // `STRATA_VK_NOBARRIER` and `STRATA_VK_BARRIER_HAZARD` measurements).
        ++g_ds.barriers;
        bool emit = !(g_nobarrier || (g_nobarrier_rec && fresh_set));
        if (g_hazard_only) {
            bool need = g_haz.size() >= kHazardCap;
            std::vector<HazardRange> cur;
            cur.reserve(bufs.size());
            for (const Buf* b : bufs) {
                if (b == nullptr || b->buffer == VK_NULL_HANDLE || b->bytes == 0) continue;
                const uint64_t s = b->offset, e = b->offset + b->bytes;
                cur.push_back({b->buffer, s, e});
                for (const HazardRange& p : g_haz) {
                    if (p.buf == b->buffer && s < p.end && p.off < e) { need = true; break; }
                }
            }
            emit = need;
            if (need) { g_haz = cur; ++g_haz_barriers; }
            else for (const HazardRange& r : cur) g_haz.push_back(r);
        }
        if (emit) {
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &mb, 0, nullptr, 0, nullptr);
        }
    }
}

void Ctx::dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                   uint32_t groups, uint32_t groups_y) {
    // A CAPTURE RECORDS, IT DOES NOT RUN.  CUDA's stream capture makes a kernel launch a node in a graph instead
    // of a queued execution; here the same diversion routes the launch into the recorded-step encoder.  This is
    // the ONE place the graph API joins the port's recorded step - there is no second recording path.
    if (capture_) {
        record_dispatch(pipe, bufs, push, push_bytes, groups, groups_y);
        return;
    }
    live_ensure();
    if (!live_open_) {
        VK_CHECK(vkResetCommandBuffer(live_cb_, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(live_cb_, &bi));
        // The previous batch was submitted and waited on in flush_live(), so its sets are dead and the pool can
        // be reset: the sets are RECYCLED, not grown (this path allocates kLiveBatchMax per batch, forever).
        VK_CHECK(vkResetDescriptorPool(dev_, live_pool_, 0));
        live_open_ = true;
        live_n_ = 0;
    }
    VkDescriptorSetLayout sl = VK_NULL_HANDLE;
    for (const Pipe& p : pipes_) {
        if (p.pipe == pipe) { sl = p.set_layout; break; }
    }
    if (sl == VK_NULL_HANDLE) {
        std::fprintf(stderr, "dispatch: unknown pipeline\n");
        std::exit(1);
    }
    VkDescriptorSet set = set_alloc_in(live_pool_, sl);
    const double _te0 = vk_ms();
    // A compute -> compute barrier after EVERY dispatch in the batch.  The old path got the ordering from the
    // submit+wait between two dispatches; batching removes that, so the barrier must be explicit.  It cannot be
    // conditional on `live_n_ > 0`: the FIRST dispatch's writes would then be unprotected, and the very next
    // dispatch reading them (the router writing the routing ids, `pf_copy_u32` copying them out) reads garbage -
    // measured as `prefill: routed id out of range` at the first grouped layer.
    encode_dispatch(live_cb_, pipe, bufs, push, push_bytes, groups, groups_y, /*chain_barrier=*/true,
                    /*fresh_set=*/false, set);
    if (g_ds.on) {
        g_ds.t_encode += vk_ms() - _te0;
        ++g_ds.n;
        for (const Pipe& p : pipes_) {
            if (p.pipe == pipe) { disp_stat_count(p.spv_path); break; }
        }
    }
    if (++live_n_ >= kLiveBatchMax) flush_live();
    // THE HOST-VISIBLE RULE, and it is the port's documented contract rather than a tuning choice.  A dispatch
    // that touches a HOST-VISIBLE (mapped) region must be COMPLETE when `dispatch` returns, because the engine
    // and the gate read those regions DIRECTLY - with no `Ctx::read` to flush for them.  Measured when this was
    // missing: the gate's `doorbell ring` case read the ring as 0 (its pending increment then ran at the NEXT
    // flush, which was inside the capture, so "the ring moved during the capture"), the replay case read a stale
    // ring, and `sample_tokens entry (mapped out)` read the -12345 sentinel - 881 passed / 5 failed / 0 skipped
    // on the Arc.  Device-local buffers (the engine's arena) still batch.
    for (const Buf* b : bufs) {
        if (b != nullptr && b->mapped != nullptr) { flush_live(); break; }
    }
}

void Ctx::flush() { flush_live(); }

// ---- recorded steps: the CUDA-graph replacement (NEXT.md's stage-3 note) -----------------------------------
// The command buffer and fence live for the life of the Ctx and are re-submitted, never re-recorded by a replay.
void Ctx::record_begin() {
    // The recording SUBMITS on its own; a pending live batch must be on the device before the recording's
    // command buffer (its dispatches would otherwise run out of the order the caller observed).
    flush_live();
    if (rec_cb_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev_, &cbai, &rec_cb_));
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &rec_fence_));
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // NOT ONE_TIME_SUBMIT: this buffer is submitted once per replay, which is the entire point of it.
    VK_CHECK(vkBeginCommandBuffer(rec_cb_, &bi));
    recording_ = true;
    recorded_ = 0;
    recorded_copies_ = 0;
}

void Ctx::record_dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                          uint32_t groups, uint32_t groups_y) {
    if (!recording_) {
        std::fprintf(stderr, "record_dispatch: not recording (call record_begin)\n");
        std::exit(1);
    }
    encode_dispatch(rec_cb_, pipe, bufs, push, push_bytes, groups, groups_y, /*chain_barrier=*/true,
                    /*fresh_set=*/true, VK_NULL_HANDLE);
    ++recorded_;
    if (g_ds.on) {
        ++g_ds.recorded;
        for (const Pipe& p : pipes_) {
            if (p.pipe == pipe) { disp_stat_count(p.spv_path); disp_stat_count_rec(p.spv_path); break; }
        }
    }
}

void Ctx::record_end() {
    if (!recording_) {
        std::fprintf(stderr, "record_end: not recording\n");
        std::exit(1);
    }
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(rec_cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(rec_cb_));
    recording_ = false;
    have_recording_ = true;
}

void Ctx::record_end_and_submit() {
    record_end();
    submit_recorded();
}

// ---- stream capture: the CUDA-graph API's recording, over the recorded step above --------------------------
void Ctx::capture_begin() {
    g_xs.decode = true;             // STRATA_VK_XFER_STAT: the first capture is where the decode arm starts
    discard_recording();            // a fresh capture: free any stale segment handles (normally none remain)
    record_begin();                 // SEGMENT 0's command buffer + fence
    capture_ = true;
    capture_valid_ = true;
}

void Ctx::capture_copy(const Buf& dst, const Buf& src, uint64_t bytes) {
    if (!capture_) {
        std::fprintf(stderr, "capture_copy: no capture is active\n");
        std::exit(1);
    }
    if (bytes == 0) return;
    VkBufferCopy r{};
    r.srcOffset = src.offset;
    r.dstOffset = dst.offset;
    r.size = bytes;
    vkCmdCopyBuffer(rec_cb_, src.buffer, dst.buffer, 1, &r);
    // transfer -> compute (and -> any later transfer): the copy's writes are visible to the next dispatch.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(rec_cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    ++recorded_copies_;
}

bool Ctx::capture_end() {
    if (!capture_) return false;
    capture_ = false;
    uint32_t total = recorded_ + recorded_copies_;
    for (const CaptureSeg& sg : cap_segs_) total += sg.dispatches + sg.copies;
    const bool ok = capture_valid_ && total > 0;
    if (!ok) {
        // Abandon: end the (unfinished) command buffer so it can be freed; the caller discards it and refuses.
        if (recording_) { vkEndCommandBuffer(rec_cb_); recording_ = false; }
        return false;
    }
    record_end();                   // close the LAST segment
    cap_segs_.push_back(CaptureSeg{rec_cb_, rec_fence_, recorded_, recorded_copies_});
    rec_cb_ = VK_NULL_HANDLE;
    rec_fence_ = VK_NULL_HANDLE;
    recorded_ = 0;
    recorded_copies_ = 0;
    have_recording_ = false;
    return true;
}

// THE HAND SHAKE SEAM.  A `wait_flag_ge` inside the capture is not a kernel here - it is a CUT.  The segment so
// far is closed and stashed, the boundary (the HOST word and the value it must reach) is recorded, and a fresh
// segment begins.  The driver (the shim's graph launch/query/synchronize) submits the segments in order and
// POLLS the flag on the host thread between them: no kernel spins, and the device is never inside a waiting
// queue item across the engine's host loop.
void Ctx::capture_boundary(const void* flag, uint32_t value) {
    if (!capture_ || !recording_) {
        std::fprintf(stderr, "capture_boundary: no capture is active - refusing\n");
        std::exit(2);
    }
    record_end();                   // close this segment (the closing host-read barrier); NO submit
    cap_segs_.push_back(CaptureSeg{rec_cb_, rec_fence_, recorded_, recorded_copies_});
    cap_bounds_.push_back(CaptureBoundary{flag, value});
    rec_cb_ = VK_NULL_HANDLE;        // force the NEXT segment to allocate its own command buffer + fence
    rec_fence_ = VK_NULL_HANDLE;
    recorded_ = 0;
    recorded_copies_ = 0;
    have_recording_ = false;
    record_begin();
}

void Ctx::take_recording(std::vector<CaptureSeg>& segs, std::vector<CaptureBoundary>& bounds) {
    segs = std::move(cap_segs_);
    bounds = std::move(cap_bounds_);
    cap_segs_.clear();
    cap_bounds_.clear();
    ++owned_recordings_;
}

void Ctx::discard_recording() {
    if (rec_cb_ != VK_NULL_HANDLE) { vkFreeCommandBuffers(dev_, cmd_pool_, 1, &rec_cb_); rec_cb_ = VK_NULL_HANDLE; }
    if (rec_fence_ != VK_NULL_HANDLE) { vkDestroyFence(dev_, rec_fence_, nullptr); rec_fence_ = VK_NULL_HANDLE; }
    for (CaptureSeg& sg : cap_segs_) {
        if (sg.cb != VK_NULL_HANDLE) vkFreeCommandBuffers(dev_, cmd_pool_, 1, &sg.cb);
        if (sg.fence != VK_NULL_HANDLE) vkDestroyFence(dev_, sg.fence, nullptr);
    }
    cap_segs_.clear();
    cap_bounds_.clear();
    recording_ = false;
    have_recording_ = false;
    recorded_ = 0;
    recorded_copies_ = 0;
}

void Ctx::submit_segment(const CaptureSeg& seg) {
    if (seg.cb == VK_NULL_HANDLE) return;
    flush_live();
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &seg.cb;
    VK_CHECK(vkResetFences(dev_, 1, &seg.fence));   // the fence was signalled by the previous submission
    const double _ts = vk_ms();
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, seg.fence));
    const double _tw = vk_ms();
    // THE DECODE'S OWN PATH IS TIMED HERE.  A segment is a recorded block of dispatches (the verify window's
    // per-round replay), submitted and WAITED.  It is NOT a `dispatch`, so it never reached `g_ds.t_wait` /
    // `t_submit` and the counters read zero dispatches across the whole decode - which is exactly why the decode
    // arm had no attribution.  Charged here, per segment, with its dispatch count.
    VK_CHECK(vkWaitForFences(dev_, 1, &seg.fence, VK_TRUE, UINT64_MAX));
    const double _td = vk_ms();
    if (g_ds.on) {
        ++g_ds.submits; ++g_ds.waits; ++g_ds.sub_seg;
        g_ds.t_submit += _tw - _ts;
        g_ds.t_wait += _td - _tw;
        g_ds.seg_disp += seg.dispatches;
        g_ds.t_seg_wait += _td - _tw;
        g_ds.t_seg_submit += _tw - _ts;
    }
}

void Ctx::destroy_owned(const CaptureSeg& seg) {
    if (seg.cb != VK_NULL_HANDLE) vkFreeCommandBuffers(dev_, cmd_pool_, 1, &seg.cb);
    if (seg.fence != VK_NULL_HANDLE) vkDestroyFence(dev_, seg.fence, nullptr);
}

void Ctx::release_recording() {
    if (owned_recordings_ > 0) --owned_recordings_;
}

void Ctx::submit_recorded() {
    if (!have_recording_) {
        std::fprintf(stderr, "submit_recorded: nothing recorded\n");
        std::exit(1);
    }
    flush_live();
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &rec_cb_;
    VK_CHECK(vkResetFences(dev_, 1, &rec_fence_));   // the fence was signalled by the previous submission
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, rec_fence_));
    VK_CHECK(vkWaitForFences(dev_, 1, &rec_fence_, VK_TRUE, UINT64_MAX));
    if (g_ds.on) { ++g_ds.submits; ++g_ds.waits; ++g_ds.sub_rec; }
}

void Ctx::replay_recorded() { submit_recorded(); }   // re-submits the RECORDING; it never re-records

}  // namespace strata::vulkan
