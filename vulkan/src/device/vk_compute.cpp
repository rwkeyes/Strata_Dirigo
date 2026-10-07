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
#include <map>
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
    std::vector<std::pair<uint32_t, double>> seg_log;  // per segment submit: {executed dispatches, fence-wait ms}
    std::vector<uint64_t> seg_fp;                      // per segment submit: sum of bound-region bytes (FOOTPRINT)
    std::vector<double> seg_wall;                      // per segment submit: steady-clock ms, for CLOCK alignment
    double t_alloc = 0, t_encode = 0, t_fence = 0, t_submit = 0, t_wait = 0, t_free = 0;   // ms
    // The RECORDED arm's own encode, charged at record_dispatch (the old t_encode above is the LIVE path's, so the
    // recorded arm read 0 "by construction").  Its own counter so the existing per-dispatch totals do not shift.
    double t_encode_rec = 0;
    uint64_t rec_encodes = 0;
    std::vector<std::pair<std::string, uint64_t>> by_pipe;   // per-spv dispatch counts
    // THE RECORDED ARM'S OWN COMPOSITION.  A recorded command buffer is re-executed once per segment submit, so its
    // composition IS the composition of the work the replay arm executes each round - which `by_pipe` above cannot
    // show, because it pools the (one-shot) prefill with the (repeated) replay.  Counted once per ENCODE.
    std::vector<std::pair<std::string, uint64_t>> by_pipe_rec;
    // ---- STRATA_VK_GRID_STAT: THE LAUNCH GEOMETRY (measurement-only, off unless the env var is set).  An
    // earlier empty-launch sweep on this device fitted the per-dispatch cost as `~0.102 ms fixed + ~2.8 ns PER
    // WORKGROUP`, so the question "why does one dispatch cost this port 4.6x upstream's" cannot be answered by a
    // dispatch COUNT - the same count can carry a 20x different workgroup count.  `wg` is the total workgroups
    // over every encoded dispatch; `wg_hist[w]` counts dispatches that carried exactly `w` workgroups, so the
    // DISTRIBUTION is available and not only the mean.  `_rec` mirrors the RECORDED (replay) arm alone, which
    // is the decode.  Counted at the same two call sites as `by_pipe` (the live dispatch and the recorder).
    bool grid = std::getenv("STRATA_VK_GRID_STAT") != nullptr;
    uint64_t wg = 0, wg_rec = 0;
    std::vector<std::pair<std::string, uint64_t>> by_pipe_wg, by_pipe_rec_wg;
    std::map<uint64_t, uint64_t> wg_hist, wg_hist_rec;
};
DispStat g_ds;
// ---- STRATA_VK_KERNEL_TIME: PER-DISPATCH GPU TIME (measurement-only, off unless the env var is set) --------
// This is the port's half of the cross-engine per-kernel attribution the decode target needs.  The recorded
// (decode) arm's dispatches are the engine's replay, so each ENCODED dispatch is wrapped with a timestamp-query
// PAIR (TOP_OF_PIPE before vkCmdDispatch, BOTTOM_OF_PIPE after it, before the chain barrier) and the pair is
// read back after the submission's fence WAIT - which is what makes it correct across a replay: the recorded
// command buffer re-writes the same slot every time it is re-submitted, so an end-of-run read would only ever
// see the LAST replay.  Read-and-reset per submit accumulates every replay.  `fresh_set` is true ONLY at
// `record_dispatch` (the live/prefill path passes false), so the prefill's own dispatches are not timed here.
// THE INSTRUMENT'S OWN COST is the two extra commands per dispatch; it is quoted by running the same arm with
// the flag on and off (the batch's table), never assumed.
namespace {
struct KtSlot { std::string spv; uint32_t i; uint64_t wg; };
struct KtStat {
    bool on = std::getenv("STRATA_VK_KERNEL_TIME") != nullptr;
    bool ready = false, overflow = false, printed = false, dbg = false;
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t cap = 0, next = 0;
    float period_ns = 0.0f;
    std::vector<KtSlot> slots;   // EVERY recorded slot, in recording order (readings happen in the same order)
    size_t done_upto = 0;        // everything before this index has been read and accumulated
    std::vector<std::pair<std::string, uint64_t>> disp, wg;
    std::vector<std::pair<std::string, double>> ns;
    // DELIVERABLE 1 READBACK: per-shader-family WRITE FOOTPRINT and BARRIER time.  `ns` above is the KERNEL window;
    // these are the two terms the footprint-vs-price correlation needs - the barrier a link actually paid, and the
    // bytes its bound ranges carried.  Summed over every dispatch whose four timestamps were read.
    std::vector<std::pair<std::string, uint64_t>> fam_n, fam_fp;
    std::vector<std::pair<std::string, double>> fam_bar_ns;
    std::vector<uint64_t> slot_fp;   // parallel to `slots`: one footprint per recorded dispatch, in recording order
    // ---- THE FOUR-WAY SPLIT OF THE RECORDED DISPATCH (all DEVICE-SIDE, from four timestamps per dispatch) ----
    // slot i+0 = TOP_OF_PIPE before BindPipeline, i+1 = TOP_OF_PIPE before vkCmdDispatch, i+2 = BOTTOM_OF_PIPE
    // after vkCmdDispatch, i+3 = TOP_OF_PIPE after the chain barrier.  Then front-end = (i+1)-(i+0) [the binds
    // and push constants as the device executes them], kernel = (i+2)-(i+1), barrier = (i+3)-(i+2), and the
    // inter-dispatch GAP = (i+0 of N+1) - (i+3 of N), which is the only window that can hold a device-side
    // stall between two recorded dispatches.  Accumulated in ns.
    double front_ns = 0, kern_ns = 0, bar_ns = 0, gap_ns = 0;
    uint64_t gaps = 0, split_n = 0;
    int64_t last_c = -1;
    uint32_t last_i = 0;
    bool have_last = false;
    // ---- STRATA_VK_WARM_SPLIT: the WARM per-replay split.  The read-once pool above sees the COLD first replay
    // only, because the monotone `done_upto` cursor never revisits a slot.  A recorded command buffer re-writes
    // the SAME slots every time it is re-submitted, so the warm state is sitting in the pool unread.  This mode
    // records a `vkCmdResetQueryPool` at the TOP of every recorded command buffer (reset -> write -> read, per
    // replay, exactly the shape the spec guarantees) and reads each segment's own slot span after ITS fence wait,
    // so a segment submitted R times yields R splits, keyed by replay ordinal.  The engine's decode drives a
    // CAPTURED graph: the segments are recorded once and re-submitted every round, which is what makes this the
    // measurement the cold pool cannot take.
    bool warm = std::getenv("STRATA_VK_WARM_SPLIT") != nullptr;
    std::map<VkCommandBuffer, std::pair<uint32_t, uint32_t>> cb_span;   // cb -> {base slot, timed dispatches}
    std::map<VkCommandBuffer, uint32_t> cb_exec;                        // cb -> times submitted so far
    struct WRow { double front = 0, kern = 0, bar = 0, gap = 0; uint64_t n = 0, gaps = 0; };
    std::map<uint32_t, WRow> wrows;                                     // replay ordinal -> aggregate split
    struct CbRow { uint32_t n = 0, submits = 0; WRow cold, warm; };      // per command buffer: cold vs warm
    std::map<VkCommandBuffer, CbRow> cb_rows;
    uint32_t span_open = 0, span_base = 0, span_n = 0;
    uint64_t w_reads = 0, w_unavail = 0;
    void add(const std::string& s, double ns_add, uint64_t wg_add) {
        auto bump = [&](std::vector<std::pair<std::string, uint64_t>>& v, uint64_t x) {
            for (auto& kv : v) if (kv.first == s) { kv.second += x; return; }
            v.push_back({s, x});
        };
        bump(disp, 1); bump(wg, wg_add);
        for (auto& kv : ns) if (kv.first == s) { kv.second += ns_add; return; }
        ns.push_back({s, ns_add});
    }
    uint64_t get(const std::vector<std::pair<std::string, uint64_t>>& v, const std::string& s) const {
        for (const auto& kv : v) if (kv.first == s) return kv.second;
        return 0;
    }
    double getn(const std::string& s) const {
        for (const auto& kv : ns) if (kv.first == s) return kv.second;
        return 0.0;
    }
    // DELIVERABLE 1: attribute ONE read dispatch's BARRIER ns and its WRITE FOOTPRINT bytes to its shader family,
    // so the barrier price can be correlated with the footprint across every family the recording executed.
    void addbar(const std::string& s, double bar_ns, uint64_t fp) {
        auto bumpi = [&](std::vector<std::pair<std::string, uint64_t>>& v, uint64_t x) {
            for (auto& kv : v) if (kv.first == s) { kv.second += x; return; }
            v.push_back({s, x});
        };
        bumpi(fam_n, 1); bumpi(fam_fp, fp);
        for (auto& kv : fam_bar_ns) if (kv.first == s) { kv.second += bar_ns; return; }
        fam_bar_ns.push_back({s, bar_ns});
    }
    uint64_t disp_total() const {
        uint64_t t = 0;
        for (const auto& kv : disp) t += kv.second;
        return t;
    }
};
KtStat g_kt;
void kt_create(VkPhysicalDevice pd, VkDevice dev, uint32_t qfam) {
    g_kt.ready = true;                       // decided once; a failure leaves `ready` false forever
    VkPhysicalDeviceProperties pr{};
    vkGetPhysicalDeviceProperties(pd, &pr);
    g_kt.period_ns = pr.limits.timestampPeriod;
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    if (n) vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q.data());
    const uint32_t vb = qfam < n ? q[qfam].timestampValidBits : 0;
    if (vb == 0) {
        std::fprintf(stderr, "vk kernel time: queue family %u reports timestampValidBits 0 - instrument OFF\n", qfam);
        g_kt.ready = false;
        return;
    }
    g_kt.cap = 4u * 32768u;                  // 4 timestamps/DISPATCH: 32,768 timed dispatches in one recorded step
    VkQueryPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = g_kt.cap;
    if (vkCreateQueryPool(dev, &ci, nullptr, &g_kt.pool) != VK_SUCCESS) {
        std::fprintf(stderr, "vk kernel time: vkCreateQueryPool failed - instrument OFF\n");
        g_kt.ready = false;
        return;
    }
    vkResetQueryPool(dev, g_kt.pool, 0, g_kt.cap);
    std::fprintf(stderr, "vk kernel time: ON - %u timestamp queries = %u recorded dispatches, FOUR timestamps "
                         "each (front-end / kernel / barrier / gap), period %.4f ns/tick, valid bits %u, "
                         "queue family %u (the RECORDED/decode arm only)\n", g_kt.cap, g_kt.cap / 4u,
                 g_kt.period_ns, vb, qfam);
}
// read whatever the submissions so far have EXECUTED, accumulate it, and never read the same slot twice.
//
// THE SHAPE MATTERS: the engine records ALL of a capture's segments before submitting any of them, and each
// recorded command buffer is replayed many times.  So (a) a whole-pool reset inside the command buffer wipes
// the other segments' queries on every replay (measured: 183 of ~60,000 executed dispatches ever landed), and
// (b) nothing may be reset or cleared per submit.  The pool is reset ONCE, at creation; slots are numbered
// monotonically across every recording; and each read walks only the un-read prefix, stopping at the first
// query that is not available yet - the un-executed tail belongs to segments the host has not submitted.
// VK_QUERY_RESULT_WITH_AVAILABILITY_BIT: NEVER BLOCK THE HOST.  A `WAIT_BIT` read hangs the host thread that
// services the device doorbell and that is a DEVICE LOST here (measured: r=-4), not a slow read.
void kt_flush(VkDevice dev) {
    if (!g_kt.on || g_kt.pool == VK_NULL_HANDLE) return;
    if (g_kt.done_upto >= g_kt.slots.size() || g_kt.next == 0) return;
    const size_t lo_q = g_kt.slots[g_kt.done_upto].i;
    const size_t nq = (size_t) g_kt.next - lo_q;
    if (nq == 0) return;
    std::vector<uint64_t> t(nq * 2, 0);
    const VkResult r = vkGetQueryPoolResults(dev, g_kt.pool, (uint32_t) lo_q, (uint32_t) nq, t.size() * 8, t.data(), 16,
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (!g_kt.dbg) {
        g_kt.dbg = true;
        std::fprintf(stderr, "vk kernel time[dbg]: r=%d lo=%zu nq=%zu slots=%zu t0=%llu avail0=%llu\n", (int) r, lo_q,
                     nq, g_kt.slots.size(), (unsigned long long) t[0], (unsigned long long) t[1]);
    }
    // VK_NOT_READY is a NORMAL return here even with the availability flag (Mesa returns it when some queries in
    // the range are not ready), and the results buffer IS written in that case; only a real error is a failure.
    if (r != VK_SUCCESS && r != VK_NOT_READY) {
        g_kt.overflow = true;
        if (r == VK_ERROR_DEVICE_LOST) g_kt.on = false;
        return;
    }
    auto val = [&](uint32_t i) -> int64_t { return t[(size_t) (i - lo_q) * 2 + 1] != 0 ? (int64_t) t[(size_t) (i - lo_q) * 2] : -1; };
    // The four slots have two MEANINGS depending on the barrier mode (they can be told apart by the env var, which
    // is fixed for the process).  OLD: [front-start, kernel-start, kernel-end, after-barrier].  PAIR:
    // [pre-barrier, post-barrier, kernel-start, kernel-end].
    const bool pair = std::getenv("STRATA_VK_BARRIER_PAIR") != nullptr;
    while (g_kt.done_upto < g_kt.slots.size()) {
        const KtSlot& s = g_kt.slots[g_kt.done_upto];
        const int64_t v0 = val(s.i), v1 = val(s.i + 1), v2 = val(s.i + 2), v3 = val(s.i + 3);
        if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) break;     // this segment has not been submitted yet
        const double tick = (double) g_kt.period_ns;
        double front, kern, bar;
        if (pair) { bar = (double) (v1 - v0); front = (double) (v2 - v1); kern = (double) (v3 - v2); }
        else      { front = (double) (v1 - v0); kern = (double) (v2 - v1); bar = (double) (v3 - v2); }
        if (front < 0) front = 0;
        if (kern < 0) kern = 0;
        if (bar < 0) bar = 0;
        g_kt.add(s.spv, kern * tick, s.wg);                  // KERNEL (pure dispatch window)
        g_kt.front_ns += front * tick;
        g_kt.kern_ns += kern * tick;
        g_kt.bar_ns += bar * tick;
        // the inter-dispatch GAP = the next dispatch's FIRST slot minus this one's LAST slot, only where the two
        // are CONSECUTIVE in the pool (a skipped slot would otherwise read as an enormous phantom gap).
        if (g_kt.have_last && s.i == g_kt.last_i + 4u && v0 >= g_kt.last_c) {
            g_kt.gap_ns += (double) (v0 - g_kt.last_c) * tick;
            ++g_kt.gaps;
        }
        g_kt.last_c = v3; g_kt.last_i = s.i; g_kt.have_last = true; ++g_kt.split_n;
        ++g_kt.done_upto;
    }
}
// ---- STRATA_VK_WARM_SPLIT: read ONE segment's slot span after ITS fence wait, then keep it for the next replay.
// The reset is in the command buffer (see record_begin), so reset -> write -> read holds per replay and the values
// belong to exactly one execution.  The span (`cb_span`) is monotonic over the whole recording, so the k-th slot of
// a segment is `g_kt.slots[base/4 + k]` and carries the shader name the per-family table needs.
void kt_flush_seg(VkDevice dev, VkCommandBuffer cb) {
    if (!g_kt.on || !g_kt.warm || g_kt.pool == VK_NULL_HANDLE) return;
    auto it = g_kt.cb_span.find(cb);
    if (it == g_kt.cb_span.end() || it->second.second == 0) return;
    const uint32_t base = it->second.first, n = it->second.second;
    ++g_kt.w_reads;
    const size_t nq = (size_t) n * 4u;
    std::vector<uint64_t> t(nq * 2, 0);
    const VkResult r = vkGetQueryPoolResults(dev, g_kt.pool, base, (uint32_t) nq, t.size() * 8, t.data(), 16,
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY) { g_kt.overflow = true; return; }
    const uint32_t ord = g_kt.cb_exec[cb]++;
    KtStat::WRow row;
    const double tick = (double) g_kt.period_ns;
    int64_t last_c = -1;
    uint32_t last_i = 0;
    bool have = false;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t i = base + k * 4u;
        auto val = [&](uint32_t q) -> int64_t {
            const size_t idx = (size_t) (q - base) * 2;
            return t[idx + 1] != 0 ? (int64_t) t[idx] : -1;
        };
        const int64_t v0 = val(i), v1 = val(i + 1), v2 = val(i + 2), v3 = val(i + 3);
        if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) { ++g_kt.w_unavail; return; }   // not this replay: count, do not fake
        double front = (double) (v1 - v0), kern = (double) (v2 - v1), bar = (double) (v3 - v2);
        if (front < 0) front = 0;
        if (kern < 0) kern = 0;
        if (bar < 0) bar = 0;
        row.front += front * tick;
        row.kern += kern * tick;
        row.bar += bar * tick;
        ++row.n;
        if (have && i == last_i + 4u && v0 >= last_c) { row.gap += (double) (v0 - last_c) * tick; ++row.gaps; }
        last_c = v3; last_i = i; have = true;
        const size_t si = (size_t) (base / 4u) + k;
        if (si < g_kt.slots.size()) {
            g_kt.add(g_kt.slots[si].spv, kern * tick, g_kt.slots[si].wg);
            // DELIVERABLE 1: this dispatch's BARRIER price and WRITE FOOTPRINT, attributed to its family.
            g_kt.addbar(g_kt.slots[si].spv, bar * tick, si < g_kt.slot_fp.size() ? g_kt.slot_fp[si] : 0);
        }
    }
    // merge into the replay-ordinal row and this command buffer's own cold/warm rows
    auto merge = [](KtStat::WRow& d, const KtStat::WRow& s) {
        d.front += s.front; d.kern += s.kern; d.bar += s.bar; d.gap += s.gap; d.n += s.n; d.gaps += s.gaps;
    };
    merge(g_kt.wrows[ord], row);
    KtStat::CbRow& cr = g_kt.cb_rows[cb];
    cr.n = n; cr.submits = ord + 1;
    merge(ord == 0 ? cr.cold : cr.warm, row);
}
void kt_dump() {
    if (!g_kt.on || g_kt.printed) return;
    g_kt.printed = true;
    if (g_kt.ns.empty()) {
        std::fprintf(stderr, "vk kernel time: no RECORDED dispatch was timed (the decode arm was never reached)\n");
        return;
    }
    std::vector<std::pair<std::string, double>> v = g_kt.ns;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    double tot = 0;
    for (const auto& kv : v) tot += kv.second;
    const uint64_t never = g_kt.warm ? 0 : (g_kt.slots.size() - g_kt.done_upto);
    std::fprintf(stderr, "vk kernel time (RECORDED/decode arm): %zu shader families, %.1f ms GPU over %llu dispatches "
                         "(%llu recorded slots never executed%s)%s\n",
                 v.size(), tot / 1e6, (unsigned long long) g_kt.disp_total(),
                 (unsigned long long) never, g_kt.overflow ? "; READBACK ERRORS SEEN - PARTIAL" : "",
                 g_kt.warm ? " [WARM: values are summed over every replay]" : "");
    if (g_kt.warm) {
        // THE WARM PER-REPLAY SPLIT.  Each row is one replay ORDINAL of the captured graph, summed over every
        // segment that was submitted for that ordinal: row 0 is the COLD first execution of each segment, rows
        // >= 1 are the warm steady state the read-once pool never sees.  us/dispatch and us/pair, so the rows are
        // directly comparable to the cold "119.86 us" and to `t_seg_wait / seg_disp`.
        std::fprintf(stderr, "vk warm split [WARM=%d]: %zu segment command buffers, %llu per-segment reads, "
                             "%llu reads hit an unavailable slot\n",
                     (int) g_kt.warm, g_kt.cb_span.size(), (unsigned long long) g_kt.w_reads,
                     (unsigned long long) g_kt.w_unavail);
        for (const auto& kv : g_kt.wrows) {
            const KtStat::WRow& r = kv.second;
            const double n = (double) r.n, gn = (double) r.gaps;
            const double f = n ? r.front / 1e3 / n : 0.0, k = n ? r.kern / 1e3 / n : 0.0;
            const double b = n ? r.bar / 1e3 / n : 0.0, g = gn ? r.gap / 1e3 / gn : 0.0;
            std::fprintf(stderr, "vk warm replay %2u%s: n=%-8llu front %.2f + kernel %.2f + barrier %.2f + gap %.2f "
                                 "= %.2f us/disp ; barrier+gap %.2f ; %llu pairs\n",
                         kv.first, kv.first == 0 ? " COLD" : "    ", (unsigned long long) r.n, f, k, b, g, f + k + b + g,
                         b + g, (unsigned long long) r.gaps);
        }
        // the warm aggregate over replays >= 1 (the steady state the round actually lives in)
        double wf = 0, wk = 0, wb = 0, wg = 0;
        uint64_t wn = 0, wgp = 0;
        for (const auto& kv : g_kt.wrows) {
            if (kv.first == 0) continue;
            wf += kv.second.front; wk += kv.second.kern; wb += kv.second.bar; wg += kv.second.gap;
            wn += kv.second.n; wgp += kv.second.gaps;
        }
        const double n = (double) wn, gn = (double) wgp;
        const double f = n ? wf / 1e3 / n : 0.0, k = n ? wk / 1e3 / n : 0.0;
        const double b = n ? wb / 1e3 / n : 0.0, g = gn ? wg / 1e3 / gn : 0.0;
        std::fprintf(stderr, "vk warm split AGGREGATE (replays >=1, the steady state): n=%llu  front %.2f + kernel %.2f "
                             "+ barrier %.2f + gap %.2f = %.2f us/dispatch ; barrier+gap %.2f\n",
                     (unsigned long long) wn, f, k, b, g, f + k + b + g, b + g);
        // PER COMMAND BUFFER: each segment's own cold (submit 0) against its warm mean (submits >= 1).  This is the
        // comparison the ordinal table cannot make, because different command buffers appear in different ordinals.
        for (const auto& kv : g_kt.cb_rows) {
            const KtStat::CbRow& c = kv.second;
            auto row = [](const KtStat::WRow& r, double& f, double& k, double& b, double& g, double& tot) {
                const double n = (double) r.n, gn = (double) r.gaps;
                f = n ? r.front / 1e3 / n : 0.0; k = n ? r.kern / 1e3 / n : 0.0;
                b = n ? r.bar / 1e3 / n : 0.0; g = gn ? r.gap / 1e3 / gn : 0.0; tot = f + k + b + g;
            };
            double cf, ck, cb2, cg, ct, wf2, wk2, wb2, wg2, wt;
            row(c.cold, cf, ck, cb2, cg, ct);
            row(c.warm, wf2, wk2, wb2, wg2, wt);
            std::fprintf(stderr,
                         "vk warm cb n=%-6u submits=%-3u | COLD %.2f+%.2f+%.2f+%.2f=%.2f | WARM %.2f+%.2f+%.2f+%.2f=%.2f "
                         "us/disp (front+kernel+barrier+gap)\n",
                         c.n, c.submits, cf, ck, cb2, cg, ct, wf2, wk2, wb2, wg2, wt);
        }
        // DELIVERABLE 1: THE FOOTPRINT-vs-PRICE TABLE.  One line per shader family (ordered by footprint): the
        // dispatches read, the MEAN bound-region bytes one dispatch carries - the barrier's flush surface - and the
        // MEAN barrier it paid.  The relation between the last two columns is the test of "the barrier costs what a
        // link WRITES"; a flat barrier column against a 3-decade footprint spread REFUTES the flush hypothesis.
        std::fprintf(stderr, "vk fp vs barrier (per shader family, WARM replays>=1; fp = mean bound-region bytes per "
                             "dispatch, bar = mean barrier us/dispatch):\n");
        {
            auto gp = [&](const std::vector<std::pair<std::string, uint64_t>>& v, const std::string& s) -> uint64_t {
                for (const auto& kv : v) if (kv.first == s) return kv.second;
                return 0;
            };
            auto gd = [&](const std::vector<std::pair<std::string, double>>& v, const std::string& s) -> double {
                for (const auto& kv : v) if (kv.first == s) return kv.second;
                return 0.0;
            };
            std::vector<std::pair<uint64_t, std::string>> rows;   // fp_total, name  (ascending: the relation shows)
            for (const auto& kv : g_kt.fam_n) rows.push_back({gp(g_kt.fam_fp, kv.first), kv.first});
            std::sort(rows.begin(), rows.end());
            for (const auto& r : rows) {
                const std::string& s = r.second;
                const uint64_t n = gp(g_kt.fam_n, s);
                if (!n) continue;
                const size_t slash = s.find_last_of('/');
                const std::string bs = s.substr(slash == std::string::npos ? 0 : slash + 1);
                std::fprintf(stderr, "vk fp fam %-34s n=%-8llu fp=%-12.0f B bar_us=%-8.3f kern_us=%.3f\n",
                             bs.c_str(), (unsigned long long) n, (double) r.first / (double) n,
                             gd(g_kt.fam_bar_ns, s) / 1e3 / (double) n, g_kt.getn(s) / 1e3 / (double) n);
            }
        }
    } else {
        // THE FOUR-WAY SPLIT, in us per timed dispatch.  `split_n` counts dispatches whose four timestamps were
        // read; the gap is counted only between CONSECUTIVE timed dispatches, so its own denominator is `gaps`.
        const double n = (double) g_kt.split_n, gn = (double) g_kt.gaps;
        const double f = n ? g_kt.front_ns / 1e3 / n : 0.0, k = n ? g_kt.kern_ns / 1e3 / n : 0.0;
        const double b = n ? g_kt.bar_ns / 1e3 / n : 0.0, g = gn ? g_kt.gap_ns / 1e3 / gn : 0.0;
        const char* mode = std::getenv("STRATA_VK_BARRIER_PAIR") != nullptr ? "PAIR" : "WIDE";
        std::fprintf(stderr, "vk kernel time SPLIT [%s] (us/dispatch, device-side, %llu timed): "
                             "front-end %.2f + kernel %.2f + barrier %.2f + gap %.2f = %.2f ; barrier+gap = %.2f\n",
                     mode, (unsigned long long) g_kt.split_n, f, k, b, g, f + k + b + g, b + g);
        std::fprintf(stderr, "vk kernel time SPLIT ms (whole run): front-end %.1f + kernel %.1f + barrier %.1f + "
                             "gap %.1f (over %llu consecutive pairs)\n",
                     g_kt.front_ns / 1e6, g_kt.kern_ns / 1e6, g_kt.bar_ns / 1e6, g_kt.gap_ns / 1e6,
                     (unsigned long long) g_kt.gaps);
    }
    for (size_t i = 0; i < v.size() && i < 64; ++i) {
        const std::string& s = v[i].first;
        const size_t slash = s.find_last_of('/');
        const std::string b = s.substr(slash == std::string::npos ? 0 : slash + 1);
        const uint64_t d = g_kt.get(g_kt.disp, s), w = g_kt.get(g_kt.wg, s);
        std::fprintf(stderr, "vk kernel time %-40s disp=%-8llu wg=%-11llu gpu_ms=%-10.3f us/disp=%-9.3f ns/wg=%.2f\n",
                     b.c_str(), (unsigned long long) d, (unsigned long long) w, v[i].second / 1e6,
                     d ? v[i].second / 1e3 / (double) d : 0.0, w ? v[i].second / (double) w : 0.0);
    }
}
}  // namespace
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
// ---- STRATA_VK_BARRIER_NARROW: an OPT-IN, MEASURED-AND-REFUTED change to the RECORDED arm's chain barrier ----
// Task 1's four-timestamp split prices the recorded arm's per-dispatch cost as front-end 0.02 + kernel 5.3 +
// barrier 57.4 + gap 56.7 us (cold first replay, n=2), i.e. the device-side CHAIN BARRIER and the inter-dispatch
// gap, NOT the host bind / descriptor front-end.  So the recorded chain barrier is REPLACED here by one
// `VkBufferMemoryBarrier` per bound buffer (`[offset, offset+bytes)`, `SHADER_WRITE -> SHADER_READ`) instead of a
// single GLOBAL `VkMemoryBarrier`.  IT IS NEUTRAL: measured 65.91-65.95 (global) against 65.95-65.97 us/dispatch
// (narrow) at n=3, ranges overlapping, and the split's barrier term is unchanged (57.40 -> 57.34 us), so the
// recorded barrier's cost is its EXECUTION DEPENDENCY (a pipeline flush), not its memory range.  It is therefore
// OPT-IN (off by default: the shipped path keeps the global barrier) and exists so the falsification is
// reproducible.  The LIVE/prefill arm never narrows (fresh_set false): it is not the metric.
bool g_barrier_narrow = std::getenv("STRATA_VK_BARRIER_NARROW") != nullptr;
// ---- STRATA_VK_BARRIER_PAIR: the chain barrier decided PER CONSECUTIVE PAIR, from the REAL bound byte ranges.
// The whole-binding range test is too coarse to find the independence that is actually there: a chain barrier is
// needed only between two dispatches whose bound regions INTERSECT, and the port already knows each dispatch's
// regions (`[Buf::offset, offset+Buf::bytes)`, exactly what it binds).  This mode emits, at the START of each
// recorded dispatch, one `VkBufferMemoryBarrier` per (buffer, `[max(off), min(end))`) intersection against the
// ranges still pending a barrier; if NOTHING intersects, NO barrier is emitted and the two dispatches may run
// concurrently.  Emitted at the START because a barrier placed AFTER a dispatch cannot order that dispatch's own
// reads.  Opt-in; the shipped path keeps the per-dispatch global barrier.  `g_pair_seen` counts pairs that had
// something pending, `g_pair_indep` how many of them intersected NOTHING (no barrier), `g_pair_emit` the rest.
bool g_barrier_pair = std::getenv("STRATA_VK_BARRIER_PAIR") != nullptr;
// ---- STRATA_VK_DIRECT_UPLOAD: upload STRAIGHT INTO a mappable device buffer instead of staging through host RAM.
// Two halves, and both are needed for the win:
//   * `alloc_device()` takes `mem_type_` (HOST_VISIBLE|HOST_COHERENT, device-local heap) instead of `vram_type_`
//     (the unmappable one).  Same heap, so the VRAM ACCOUNT is unchanged and the account rule is not bent - only
//     whether the CPU may map it.
//   * `stage_upload()` then has nothing to stage: the host store IS the transfer, so the copy command, the submit,
//     the fence and its wait all disappear for that call.
// MEASURED BASIS (Arc Pro B70, port's own instruments, 2026-10-07): host store into the mapped type 5.64 GB/s
// against the staged path's 1.92 (probe_mem), and the GPU reads a mappable allocation ~3% slower (probe_gpuread:
// 499.71 vs 512.49 GB/s).  Applied to the 53.7 GB of cold-start uploads the flag is worth roughly -18 s of a
// ~163 s startup for ~0.4% of the prefill.  OPT-IN ON PURPOSE: making model memory mappable is the port's parked
// policy question, not a tuning knob, so the shipped default keeps the staged path and the gate keeps asserting
// the types it asserts today.
bool g_direct_upload = std::getenv("STRATA_VK_DIRECT_UPLOAD") != nullptr;
struct PairRange {
    VkBuffer buf;
    uint64_t off, end;
};
std::vector<PairRange> g_pend;
uint64_t g_pair_seen = 0, g_pair_indep = 0, g_pair_emit = 0;
struct HazardRange {
    VkBuffer buf;
    uint64_t off, end;
};
std::vector<HazardRange> g_haz;
constexpr size_t kHazardCap = 8192;   // force a barrier rather than grow without bound
uint64_t g_haz_barriers = 0;          // barriers actually EMITTED in the hazard mode
uint64_t g_narrow_barriers = 0, g_wide_barriers = 0;   // Task 3: recorded-arm barriers by FORM (narrow vs global)
// ---- STRATA_VK_FOOTPRINT: PER-DISPATCH WRITE FOOTPRINT (measurement-only, HOST-SIDE arithmetic only) ----------
// The chain barrier's cost is the hypothesis that each link waits for its WRITES to become visible.  The hazard
// machinery already computes, for every recorded dispatch, the bound regions `[Buf::offset, offset+Buf::bytes)`
// the barrier has to publish, so a dispatch's write footprint is a READBACK of numbers the decision already used,
// not a new analysis.  This instrument SUMS those regions per recorded dispatch and attributes them to the
// command buffer (always) and to the shader family (with STRATA_VK_KERNEL_TIME), so the footprint can be
// correlated with the drain the fence wait actually measured.  IT ADDS NO DEVICE COMMAND AND NO GPU WORK: it is
// pure host arithmetic at RECORD time, so the decode arm cannot move under it (unlike KERNEL_TIME, whose own cost
// is -7.8% and which is quoted wherever its numbers are used).
bool g_fp_on = std::getenv("STRATA_VK_FOOTPRINT") != nullptr;
struct FpStat {
    std::map<VkCommandBuffer, uint64_t> cb_fp;   // per recorded command buffer: sum of bound-region bytes
    std::map<VkCommandBuffer, uint64_t> cb_n;    // ...and how many dispatches contributed
    uint64_t rec_fp = 0, rec_n = 0;
};
FpStat g_fp;
// ---- STRATA_VK_TRIVIAL_REC: DELIVERABLE 2 - A TRIVIAL-KERNEL SEGMENT AT THE SAME DISPATCH COUNT ---------------
// `record_dispatch` normally encodes the engine's real kernel.  With this flag it encodes ONE workgroup of the
// trivial `scale` kernel (x[i] *= s) over a 4 KiB scratch instead, keeping the SAME dispatch count, the SAME
// per-dispatch fresh descriptor set and the SAME chain barrier, through the SAME recording and the SAME segment
// submit path.  The drain it produces is therefore turnaround + barrier with almost no work and almost no write
// footprint - which is what separates DEVICE TURNAROUND from DEPENDENCY COST without the elision arm's confounds.
// MEASUREMENT-ONLY: the decode's answers are garbage by construction and the id is expected to move.
bool g_trivial_rec = std::getenv("STRATA_VK_TRIVIAL_REC") != nullptr;
// STRATA_VK_TRIVIAL_REC_FAMILY: the PER-FAMILY form of the same trick.  A comma/space-separated list of shader
// family names (the basename with or without `.spv`, e.g. `native_router_top10` or `native_k_mmvq.spv`).  When
// set, ONLY the named families' recorded dispatches are replaced by the trivial work and EVERY OTHER dispatch is
// encoded exactly as it always was - so the round's wall clock under this flag differs from a baseline run by
// exactly the named family's cost.  That is what turns the trivial-kernel trick into a MARGINAL-price instrument:
// run F-trivial, subtract the baseline, and the difference IS F's marginal price for the round.  With the list
// EMPTY and STRATA_VK_TRIVIAL_REC=1 the whole recorded arm is trivialised (the original deliverable-2 form).
std::string vk_trivial_norm(const std::string& spv) {
    std::string b = spv;
    const size_t slash = b.find_last_of('/');
    if (slash != std::string::npos) b = b.substr(slash + 1);
    if (b.size() > 4 && b.compare(b.size() - 4, 4, ".spv") == 0) b = b.substr(0, b.size() - 4);
    return b;
}
std::vector<std::string> g_trivial_fam;
bool g_trivial_fam_set = false;
std::string g_trivial_fam_str;
bool vk_trivial_fam_init() {
    const char* f = std::getenv("STRATA_VK_TRIVIAL_REC_FAMILY");
    if (f == nullptr || *f == '\0') return false;
    std::string s(f);
    for (char& c : s) if (c == ',' || c == '\t' || c == ';') c = ' ';
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(' ', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) { const std::string n = vk_trivial_norm(s.substr(i, j - i));
                     g_trivial_fam.push_back(n);
                     if (!g_trivial_fam_str.empty()) g_trivial_fam_str += " ";
                     g_trivial_fam_str += n; }
        i = j + 1;
    }
    g_trivial_fam_set = !g_trivial_fam.empty();
    return g_trivial_fam_set;
}
// Does THIS recorded dispatch's family get substituted?  The family list, when present, is the whole rule; the
// all-of-arm flag applies only when no list was given.
bool trivial_rec_hit(const std::string& spv) {
    if (g_trivial_fam_set) {
        const std::string n = vk_trivial_norm(spv);
        for (const std::string& x : g_trivial_fam) if (x == n) return true;
        return false;
    }
    return g_trivial_rec;
}
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
// STRATA_VK_GRID_STAT: one dispatch's workgroup count (`groups * groups_y`) into the pooled totals, the
// per-arc histogram and (for the recorded arm) the replay-only histogram.  `rec` is true only at the recorder.
void disp_stat_grid(const std::string& spv, uint64_t w, bool rec) {
    if (!g_ds.grid) return;
    auto bump = [&](std::vector<std::pair<std::string, uint64_t>>& v) {
        for (auto& kv : v) {
            if (kv.first == spv) { kv.second += w; return; }
        }
        v.push_back({spv, w});
    };
    bump(g_ds.by_pipe_wg);
    ++g_ds.wg_hist[w];
    g_ds.wg += w;
    if (rec) {
        bump(g_ds.by_pipe_rec_wg);
        ++g_ds.wg_hist_rec[w];
        g_ds.wg_rec += w;
    }
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
        for (size_t i = 0; i < r.size() && i < 64; ++i) {
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
    // PER-SEGMENT DRAIN: did the fence wait scale with the SEGMENT'S DISPATCH COUNT (links) or is it fixed?  One
    // line per segment submit so the relation is visible, not asserted.
    if (!g_ds.seg_log.empty()) {
        std::fprintf(stderr, "vk seg drain (per segment submit: dispatches, fence-wait ms, us/dispatch, bound-region "
                             "bytes = the FOOTPRINT, bytes/link, steady-clock ms):\n");
        for (size_t i = 0; i < g_ds.seg_log.size(); ++i) {
            const uint32_t d = g_ds.seg_log[i].first;
            const double w = g_ds.seg_log[i].second;
            const uint64_t fp = i < g_ds.seg_fp.size() ? g_ds.seg_fp[i] : 0;
            const double wl = i < g_ds.seg_wall.size() ? g_ds.seg_wall[i] : 0.0;
            std::fprintf(stderr,
                         "vk seg drain i=%-3zu disp=%-6u wait_ms=%-9.3f us/disp=%-8.2f fp=%-13llu fp/link=%-10.0f "
                         "wall=%.0f\n",
                         i, d, w, d ? w * 1e3 / (double) d : 0.0, (unsigned long long) fp,
                         d ? (double) fp / (double) d : 0.0, wl);
        }
    }
    std::fprintf(stderr, "vk disp stat RECORDED encode: %.1f ms over %llu encodes (%.4f ms/encode) - charged at "
                         "record_dispatch; the OLD `encode` term read 0 on this arm BY CONSTRUCTION\n",
                 g_ds.t_encode_rec, (unsigned long long) g_ds.rec_encodes,
                 g_ds.rec_encodes ? g_ds.t_encode_rec / (double) g_ds.rec_encodes : 0.0);
    if (g_narrow_barriers || g_wide_barriers)
        std::fprintf(stderr, "vk disp stat BARRIER FORM: %llu recorded-arm chain barriers NARROWED "
                             "(VkBufferMemoryBarrier per region) | %llu global (VkMemoryBarrier)\n",
                     (unsigned long long) g_narrow_barriers, (unsigned long long) g_wide_barriers);
    if (g_pair_seen || g_pair_indep || g_pair_emit)
        std::fprintf(stderr, "vk disp stat BARRIER PAIR: %llu recorded dispatch pairs had ranges pending; "
                             "%llu were INDEPENDENT (no intersecting (buffer,region) => NO barrier emitted, the "
                             "two dispatches may OVERLAP); %llu emitted VkBufferMemoryBarrier(s)\n",
                     (unsigned long long) g_pair_seen, (unsigned long long) g_pair_indep,
                     (unsigned long long) g_pair_emit);
    if (g_hazard_only)
        std::fprintf(stderr,
                     "vk disp stat[HAZARD]: %llu of %llu chain barriers EMITTED (the rest were elided where no\n"
                     "bound REGION overlapped a dispatch since the last barrier)\n",
                     (unsigned long long) g_haz_barriers, (unsigned long long) g_ds.barriers);
    if (g_ds.grid) {
        // THE LAUNCH GEOMETRY, as a distribution and per shader.  The replay (recorded) arm is the decode.
        auto dump_hist = [](const char* tag, const std::map<uint64_t, uint64_t>& h) {
            uint64_t nd = 0, wtot = 0;
            for (const auto& kv : h) { nd += kv.second; wtot += kv.first * kv.second; }
            std::string line;
            for (const auto& kv : h) {
                if (line.size() > 380) { line += " ..."; break; }
                line += " " + std::to_string(kv.first) + "wg:" + std::to_string(kv.second);
            }
            std::fprintf(stderr, "vk grid %s: %llu dispatches, %llu workgroups, mean %.1f wg/dispatch\n",
                         tag, (unsigned long long) nd, (unsigned long long) wtot,
                         nd ? (double) wtot / (double) nd : 0.0);
            std::fprintf(stderr, "vk grid %s hist (workgroups-per-dispatch:dispatches):%s\n", tag, line.c_str());
        };
        dump_hist("live+rec", g_ds.wg_hist);
        dump_hist("RECORDED (decode)", g_ds.wg_hist_rec);
        auto dump_pipe = [](const char* tag, std::vector<std::pair<std::string, uint64_t>> v) {
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string line;
            for (size_t i = 0; i < v.size() && i < 16; ++i) {
                const size_t slash = v[i].first.find_last_of('/');
                line += " " + v[i].first.substr(slash == std::string::npos ? 0 : slash + 1) + " " +
                        std::to_string(v[i].second);
            }
            std::fprintf(stderr, "vk grid %s workgroups by shader (top):%s\n", tag, line.c_str());
        };
        dump_pipe("live+rec", g_ds.by_pipe_wg);
        dump_pipe("RECORDED (decode)", g_ds.by_pipe_rec_wg);
    }
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
    if (g_kt.on) std::atexit(kt_dump);
    if (g_fp_on)
        std::fprintf(stderr, "vk_compute[MEASUREMENT]: STRATA_VK_FOOTPRINT=1 - per-recorded-dispatch write footprint "
                             "(sum of bound-region bytes) attributed to the command buffer and, with KERNEL_TIME, "
                             "to the shader family.  HOST ARITHMETIC ONLY: no device command, no GPU work, so the "
                             "decode arm cannot move under it.\\n");
    if (g_trivial_rec)
        std::fprintf(stderr, "vk_compute[MEASUREMENT]: STRATA_VK_TRIVIAL_REC=1 - recorded dispatches are replaced "
                             "by ONE workgroup of `scale` (same count, same chain barrier, same submit path).  The "
                             "decode's answers are WRONG by construction; deliverable 2 only.\\n");
    if (vk_trivial_fam_init())
        std::fprintf(stderr, "vk_compute[MEASUREMENT]: STRATA_VK_TRIVIAL_REC_FAMILY=%s - ONLY these recorded "
                             "families' dispatches are replaced by ONE workgroup of `scale` (same count, same chain "
                             "barrier, same submit path); the REST OF THE ROUND IS UNTOUCHED, so the round's "
                             "wall-clock delta IS that family's MARGINAL price.  The decode's answers are WRONG by "
                             "construction; this is a PRICE, not a candidate.\n", g_trivial_fam_str.c_str());
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
    //
    // THE BAR-HEAP VARIANT WAS TRIED AND THE PORT'S OWN ACCOUNT RULE REFUSES IT (2026-10-06).  On a ReBAR card
    // the host-visible type this port picks for `mem_type_` is ALSO device-local, and `tools/probe_submit.cpp`
    // measures a host write into it at **4.65 GB/s** against **1.75-2.02 GB/s** for the staged path - a 2.5x
    // that would take the ~30 s expert load toward ~12 s.  Selecting that type for `alloc_staging` does not
    // reach the first byte: `alloc_staging` charges by HEAP, the arena already holds 28,560 of the 28,589 MiB
    // usable, and the run dies at the first upload with
    //   `vk_compute: REFUSING a 256.00 MiB staging buffer - 28560.00 MiB in the VRAM account, 28589.00 MiB usable`
    // (RUN_RC=3, 0 dispatches, `/tmp/gap/gap5_bar199.log`).  That refusal is the contract working, not a bug: a
    // staging buffer in VRAM is model-resident memory by that rule.  The change a next batch would have to make
    // is therefore a POLICY one - exempt transient staging from the VRAM account, or leave the arena headroom -
    // and it is deliberately NOT made here.  What is measured and safe to quote is the 4.65 vs 1.9 GB/s
    // difference itself; the upload path is unchanged.
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
    kt_dump();
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
    if (trivial_buf_.buffer != VK_NULL_HANDLE) free(trivial_buf_);   // STRATA_VK_TRIVIAL_REC's 4 KiB scratch
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
    // STRATA_VK_DIRECT_UPLOAD: hand back the MAPPABLE device-local type so an upload can be a host store instead
    // of a staged copy.  TWO GUARDS, both necessary: the remap only happens (a) when the flag is set and (b) when
    // that type really is DEVICE_LOCAL - on a device whose only mappable type is system RAM (llvmpipe, and any
    // card without a ReBAR-style window) `mem_type_` would hand the engine host memory dressed as device memory,
    // so the flag stays inert there.  Same heap as `vram_type_` on the cards it does apply to, so the VRAM
    // account is unchanged; the only cost is the measured ~3% GPU read (tools/probe_gpuread.cpp).
    const bool mappable_vram = g_direct_upload && mem_type_ < mem_types_.size() && mem_types_[mem_type_].device_local;
    Buf b = alloc_impl(bytes, mappable_vram ? mem_type_ : vram_type_, /*vram_account=*/true, "device-local buffer");
    // The TYPE is mappable, but the mapping must NOT be persistent - see Buf::map_on_demand for the whole reason
    // (a persistent mapping makes `dispatch` flush the live batch per dispatch, which cost the prefill 47%).  A
    // device buffer is written by the host once and read by the GPU; the host never reads it back through the
    // mapping, and a host READ of this type measures 0.06 GB/s anyway (tools/probe_mem.cpp), so on-demand loses
    // nothing.  Coherent memory, so an unmap here cannot lose a store.
    if (mappable_vram && b.mapped != nullptr) {
        vkUnmapMemory(dev_, b.mem);
        b.mapped = nullptr;
        b.map_on_demand = true;
    }
    return b;
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
    // ---- STRATA_VK_DIRECT_UPLOAD: when the DESTINATION is mappable, the host store IS the transfer -------------
    // No staging buffer, no copy command, no submit, no fence wait.  Measured on the Arc Pro B70 with the port's
    // own instruments: a host store into the mapped device-local type runs at 5.64 GB/s against the staged path's
    // 1.92 GB/s (tools/probe_mem.cpp) - 2.9x on the 53.7 GB of cold-start uploads, i.e. ~29 s -> ~10 s of a
    // ~163 s startup.  The cost of the trade is measured too: the GPU READS a mappable allocation ~3% slower
    // (tools/probe_gpuread.cpp, 499.71 vs 512.49 GB/s), which is why this is opt-in and not a default - it is the
    // port's parked policy question (a staging buffer in VRAM is model memory by the account rule), so the DEFAULT
    // path below is untouched and the choice stays with the operator.
    // Host-coherent is a precondition, not an assumption: `mem_type_` was selected for HOST_VISIBLE|HOST_COHERENT,
    // so per the Vulkan memory model the stores are available to the device without a flush.  The barrier the
    // staged path emits orders TRANSFER_WRITE before SHADER_READ/HOST_READ; with a direct coherent store there is
    // no device write to order, and the dispatch path's own chain barrier still orders this layer's writes against
    // the kernels that read them (the engine relies on that ordering for its mapped grouping tables too).
    if (g_direct_upload && bytes != 0 && (dst.mapped != nullptr || dst.map_on_demand)) {
        // Map ON DEMAND when the port holds no persistent mapping (the device-buffer case): the store itself is the
        // transfer, and leaving no mapping behind is what keeps `dispatch`'s host-visible rule off this buffer.
        if (dst.mapped == nullptr) {
            void* m = nullptr;
            VK_CHECK(vkMapMemory(dev_, dst.mem, 0, VK_WHOLE_SIZE, 0, &m));
            std::memcpy((uint8_t*) m + offset, src, (size_t) bytes);   // HOST_COHERENT: no flush needed
            vkUnmapMemory(dev_, dst.mem);
        } else {
            std::memcpy((uint8_t*) dst.mapped + offset, src, (size_t) bytes);
        }
        return;
    }
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
    std::string spv_path;
    for (const Pipe& p : pipes_) {
        if (p.pipe == pipe) {
            layout = p.layout;
            set = p.set;
            set_layout = p.set_layout;
            spv_path = p.spv_path;
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

    // STRATA_VK_FOOTPRINT: the sum of the BOUND regions this dispatch carries - the byte ranges the chain barrier
    // has to publish, which is exactly what the hazard machinery's decision already walks.  Host arithmetic only
    // (see g_fp_on): no device command, no GPU work.  Computed whenever the footprint or the kernel-time
    // instrument needs it, so `g_kt.slot_fp` stays parallel to `g_kt.slots` BY CONSTRUCTION rather than by luck.
    uint64_t fp_here = 0;
    if (g_fp_on || (g_kt.on && fresh_set)) {
        for (const Buf* b : bufs) if (b != nullptr) fp_here += b->bytes;
    }
    if (g_fp_on && fresh_set) {
        g_fp.cb_fp[cb] += fp_here; ++g_fp.cb_n[cb];
        g_fp.rec_fp += fp_here;    ++g_fp.rec_n;
    }

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

    // ---- STRATA_VK_BARRIER_PAIR: the conditional per-pair chain barrier, at the START of THIS dispatch -------
    const bool pair_on = g_barrier_pair && fresh_set && chain_barrier;
    bool pair_here = false;
    uint32_t pair_i = 0;
    if (pair_on) {
        if (g_kt.on && !g_kt.ready) kt_create(phys_, dev_, queue_family_);
        if (g_kt.on && g_kt.ready && g_kt.next + 4 <= g_kt.cap) {
            pair_i = g_kt.next; g_kt.next += 4;
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_kt.pool, pair_i);   // P0: before the barrier
            pair_here = true;
        } else if (g_kt.on && g_kt.ready) {
            g_kt.overflow = true;
        }
        std::vector<VkBufferMemoryBarrier> bmbs;
        std::vector<PairRange> cur;
        cur.reserve(bufs.size());
        for (const Buf* b : bufs) {
            if (b == nullptr || b->buffer == VK_NULL_HANDLE || b->bytes == 0) continue;
            const uint64_t s = b->offset, e = b->offset + b->bytes;
            cur.push_back({b->buffer, s, e});
            for (const PairRange& p : g_pend) {
                if (p.buf != b->buffer || !(s < p.end && p.off < e)) continue;
                VkBufferMemoryBarrier mb{};
                mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                mb.buffer = b->buffer;
                mb.offset = s > p.off ? s : p.off;
                const uint64_t e2 = e < p.end ? e : p.end;
                mb.size = e2 > mb.offset ? e2 - mb.offset : 0;
                if (mb.size) bmbs.push_back(mb);
            }
        }
        if (!g_pend.empty()) {
            ++g_pair_seen;
            if (bmbs.empty()) ++g_pair_indep; else ++g_pair_emit;
        }
        if (!bmbs.empty()) {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, (uint32_t) bmbs.size(), bmbs.data(), 0, nullptr);
            g_pend = cur;                                            // these writes are now ordered
        } else {
            for (const PairRange& r : cur) g_pend.push_back(r);       // still pending a barrier
        }
        ++g_ds.barriers;   // the pair barrier replaces the per-dispatch global one (counted the same way)
        if (pair_here) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_kt.pool, pair_i + 1);  // P1: after barrier
    }

    // STRATA_VK_KERNEL_TIME: FOUR timestamps around THIS recorded dispatch (see the KtStat comment) - slot0
    // TOP_OF_PIPE here (the front-end start), slot1 TOP_OF_PIPE after the binds/push and before vkCmdDispatch,
    // slot2 BOTTOM_OF_PIPE after vkCmdDispatch, slot3 TOP_OF_PIPE after the chain barrier.  `fresh_set` is true
    // only for the recorded step (the decode's replay); the live (prefill) path passes false and is not timed.
    // NO RESET IS EMITTED HERE.  The pool is reset ONCE, at creation; a reset inside the command buffer is
    // re-executed on every replay and wipes the OTHER segments' queries (measured: only 183 of ~60,000 executed
    // dispatches ever landed).  Slots are numbered monotonically and read once each (see kt_flush).
    bool kt_here = false;
    uint32_t kt_i = 0;
    if (g_kt.on && fresh_set && !pair_on && !g_kt.ready) kt_create(phys_, dev_, queue_family_);
    if (g_kt.on && fresh_set && !pair_on && g_kt.ready && g_kt.next + 4 <= g_kt.cap) {
        kt_i = g_kt.next;
        g_kt.next += 4;
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_kt.pool, kt_i);        // slot0: front-end start
        kt_here = true;
    } else if (g_kt.on && fresh_set && !pair_on && g_kt.ready) {
        g_kt.overflow = true;   // the pool is full for this recorded step: report it rather than pretend
    }

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (push_bytes) vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    if (kt_here) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_kt.pool, kt_i + 1);   // slot1: kernel start
    if (pair_here) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_kt.pool, pair_i + 2);  // P2: kernel start
    vkCmdDispatch(cb, groups, groups_y, 1);
    if (kt_here) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_kt.pool, kt_i + 2);   // slot2: kernel end
    if (pair_here) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_kt.pool, pair_i + 3);  // P3: kernel end
    if (chain_barrier && !pair_on) {
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
            // TASK 3: the NARROWED chain barrier for the RECORDED arm (see g_barrier_wide).  One buffer barrier per
            // bound region instead of one global memory barrier; `STRATA_VK_BARRIER_WIDE=1` restores the global
            // barrier (the control).  If every region is unusable the global barrier is emitted, so this can only
            // ever narrow, never elide.
            bool narrowed = false;
            if (fresh_set && g_barrier_narrow && !g_hazard_only) {
                std::vector<VkBufferMemoryBarrier> bmbs;
                bmbs.reserve(bufs.size());
                for (const Buf* b : bufs) {
                    if (b == nullptr || b->buffer == VK_NULL_HANDLE || b->bytes == 0) continue;
                    VkBufferMemoryBarrier mb{};
                    mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    mb.buffer = b->buffer;
                    mb.offset = b->offset;
                    mb.size = b->bytes;
                    bmbs.push_back(mb);
                }
                if (!bmbs.empty()) {
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                         (uint32_t) bmbs.size(), bmbs.data(), 0, nullptr);
                    ++g_narrow_barriers;
                    narrowed = true;
                }
            }
            if (!narrowed) {
                if (fresh_set) ++g_wide_barriers;
                VkMemoryBarrier mb{};
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                     &mb, 0, nullptr, 0, nullptr);
            }
        }
    }
    if (kt_here) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_kt.pool, kt_i + 3);   // slot3: after the barrier
        g_kt.slots.push_back(KtSlot{spv_path, kt_i, (uint64_t) groups * (uint64_t) groups_y});
        g_kt.slot_fp.push_back(fp_here);   // parallel to `slots` (see fp_here above)
    }
    if (pair_here) {
        g_kt.slots.push_back(KtSlot{spv_path, pair_i, (uint64_t) groups * (uint64_t) groups_y});
        g_kt.slot_fp.push_back(fp_here);   // same parallel push in PAIR mode, so the vectors never diverge
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
            if (p.pipe == pipe) { disp_stat_count(p.spv_path); disp_stat_grid(p.spv_path, (uint64_t) groups * groups_y, false); break; }
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
    // STRATA_VK_WARM_SPLIT: arm this command buffer's timestamp slots.  The pool is created up front (rather than
    // lazily at the first recorded dispatch) so the reset command can be recorded here, at the TOP; on EVERY
    // replay the device then executes reset -> the per-dispatch writes, and the host reads this segment's span
    // after THIS submit's fence (see kt_flush_seg).  Segments are submitted strictly one at a time - every submit
    // is followed by a fence wait - so a whole-pool reset here can only wipe values already read.
    if (g_kt.on && g_kt.warm) {
        if (!g_kt.ready) kt_create(phys_, dev_, queue_family_);
        g_kt.span_open = g_kt.next;
        if (g_kt.ready && g_kt.pool != VK_NULL_HANDLE)
            vkCmdResetQueryPool(rec_cb_, g_kt.pool, 0, g_kt.cap);
    }
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
    // ---- STRATA_VK_TRIVIAL_REC (deliverable 2): the SAME dispatch count, the SAME chain barrier, the SAME fresh
    // descriptor set and the SAME submit path - but ONE workgroup of the trivial `scale` kernel (x[i] *= s) over a
    // 4 KiB scratch, so the work and the write footprint are both negligible.  What remains in the drain is device
    // TURNAROUND + the barrier; the decode's answers are garbage by construction and the id is expected to move.
    if (g_trivial_rec || g_trivial_fam_set) {
        std::string spv_here;
        for (const Pipe& p : pipes_) if (p.pipe == pipe) { spv_here = p.spv_path; break; }
        if (trivial_rec_hit(spv_here)) {
            static const float pcs[2] = {1.0f, 1.0f};
            if (trivial_pipe_ == VK_NULL_HANDLE) {
                const char* d = std::getenv("STRATA_VK_SPV_DIR");
                const std::string dir = (d != nullptr && *d != '\0') ? d : "ports/vulkan/shaders";
                trivial_buf_ = alloc(4096);
                trivial_pipe_ = pipeline(dir + "/scale.spv", 1, 8);
                if (g_trivial_fam_set)
                    std::fprintf(stderr, "vk TRIVIAL_REC_FAMILY: recording `scale` (1 workgroup, 4 KiB scratch, "
                                         "same chain barrier) ONLY for family [%s] - the REST of the round is "
                                         "untouched, so the wall-clock delta IS that family's MARGINAL price; the "
                                         "decode's answers are GARBAGE by construction\n", g_trivial_fam_str.c_str());
                else
                    std::fprintf(stderr, "vk TRIVIAL_REC: recording `scale` (1 workgroup, 4 KiB scratch, same chain "
                                         "barrier) instead of the engine kernel - the decode's answers are GARBAGE "
                                         "by construction\n");
            }
            std::vector<const Buf*> tb{ &trivial_buf_ };
            encode_dispatch(rec_cb_, trivial_pipe_, tb, pcs, 8, 1, 1, /*chain_barrier=*/true, /*fresh_set=*/true,
                            VK_NULL_HANDLE);
            ++recorded_;
            return;
        }
    }
    const double _te0 = vk_ms();
    encode_dispatch(rec_cb_, pipe, bufs, push, push_bytes, groups, groups_y, /*chain_barrier=*/true,
                    /*fresh_set=*/true, VK_NULL_HANDLE);
    const double _te1 = vk_ms();
    ++recorded_;
    if (g_ds.on) {
        g_ds.t_encode_rec += _te1 - _te0;
        ++g_ds.rec_encodes;
        ++g_ds.recorded;
        for (const Pipe& p : pipes_) {
            if (p.pipe == pipe) { disp_stat_count(p.spv_path); disp_stat_count_rec(p.spv_path);
                                  disp_stat_grid(p.spv_path, (uint64_t) groups * groups_y, true); break; }
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
    // STRATA_VK_WARM_SPLIT: register this command buffer's slot span for the per-replay read.
    if (g_kt.on && g_kt.warm) {
        g_kt.span_base = g_kt.span_open;
        g_kt.span_n = (g_kt.next - g_kt.span_open) / 4u;
        g_kt.cb_span[rec_cb_] = { g_kt.span_base, g_kt.span_n };
    }
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
    // STRATA_VK_KERNEL_TIME: read this replay's per-dispatch timestamps.  WARM mode reads THIS segment's span
    // into its replay-ordinal row; the read-once form reads the un-read prefix once (the COLD first replay).
    if (g_kt.on && g_kt.warm) kt_flush_seg(dev_, seg.cb);
    else kt_flush(dev_);
    if (g_ds.on) {
        ++g_ds.submits; ++g_ds.waits; ++g_ds.sub_seg;
        g_ds.t_submit += _tw - _ts;
        g_ds.t_wait += _td - _tw;
        g_ds.seg_disp += seg.dispatches;
        g_ds.t_seg_wait += _td - _tw;
        g_ds.t_seg_submit += _tw - _ts;
        g_ds.seg_log.push_back({ seg.dispatches, _td - _tw });
        g_ds.seg_fp.push_back(g_fp.cb_fp.count(seg.cb) ? g_fp.cb_fp[seg.cb] : 0);
        g_ds.seg_wall.push_back(_td);
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
    if (g_kt.on && g_kt.warm) kt_flush_seg(dev_, rec_cb_);
    else kt_flush(dev_);   // STRATA_VK_KERNEL_TIME: read this replay's per-dispatch timestamps, then reset the pool
    if (g_ds.on) { ++g_ds.submits; ++g_ds.waits; ++g_ds.sub_rec; }
}

void Ctx::replay_recorded() { submit_recorded(); }   // re-submits the RECORDING; it never re-records

}  // namespace strata::vulkan
