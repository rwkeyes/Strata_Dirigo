// vulkan/src/kernels/rope_vk.cpp - the RoPE HOST ROWS the layer body crosses, answered by the Vulkan backend.
//
// ============================================================================================================
// WHY THESE LIVE HERE, NOT IN THE ENGINE TREE
// ============================================================================================================
// The plan (`ports/vulkan/plan/BACKEND-INTEGRATION.md` §1) lists four rope rows among the eighteen
// device-crossing host rows; the layer body reaches two of them at `session_init`/graph-build time:
//
//   build_rope_table  (layer.cpp:692)  the float64 cos/sin table build.  On a CUDA build it is defined in
//                                      `src/kernels/cuda/rope.cu` - a CUDA TU, so a Vulkan build compiles no
//                                      copy of it.  It is PURE HOST code (std::pow / std::cos in double), and the
//                                      engine builds the table on the HOST because the reference computes its
//                                      frequencies in float64 and doing it on device would need double-precision
//                                      pow/cos that need not agree with the host libm (layer.cpp's own comment).
//   rope_table_set    (layer.cpp:698)  registers the device table the analytic rotation reads when
//                                      STRATA_ROPE_TABLE=1.  On a CUDA build it is defined in
//                                      `src/kernels/cuda/native_rope.cu`, again a CUDA TU.
//
// THE BODIES ARE THE ENGINE'S OWN LOOPS, TRANSCRIBED (rope.cu:48-86) - not re-derived.  The table's layout,
// the `pow(theta, -2i/n_rot)` order (inv, then ang, then cos/sin), the YaRN ramp on `ext_factor` and the
// `*mscale` fold are all that source's, and `rope_parity.cpp` on a CUDA build holds exactly these values to a
// float64 reference.  A transcription is the honest port here BECAUSE the rule is a host arithmetic contract,
// not a device kernel: there is no shader to compare against, and the gate case below cross-checks the table
// against the port's DEVICE `native_rope_apply` shader (a second, independent implementation of the same
// angles) as well as against its own float64 transcription.
//
// ============================================================================================================
// THE STORAGE IS ONE PROCESS TABLE, AND THAT IS A MEASURED CHOICE
// ============================================================================================================
// native_rope.cu keys the table by the "current device" (`cudaGetDevice`) because a layer split runs the rope
// kernels on several GPUs at once.  This backend's arena is per-Stream and the decode path is one device, so
// the CUDA's per-device array has no analogue to preserve: a single process table is the same object the
// single-GPU engine sees.  If a multi-Stream split is ever ported, THIS is the one place that changes.
//
// ============================================================================================================
// THE ANALYTIC PATH DOES NOT CONSUME IT, AND SAYS SO IN CODE
// ============================================================================================================
// `rope_table_for()` returns the table only when STRATA_ROPE_TABLE=1 AND it was built with the caller's scaling
// (exactly native_rope.cu's rule).  This backend has no table-reading rope SHADER (native_rope_apply.spv
// computes the angle on device from `theta_scale`/`rope_scaled_angle`), so when the table IS active the port
// cannot honour it - and `native_rope_apply` (qsa_vk.cpp) REFUSES loudly in that one configuration rather than
// silently returning the analytic angle the engine's table path is documented to differ from by ~0.0014 rad at
// 32K (mrope.hpp's note).  The default (STRATA_ROPE_TABLE unset) is the analytic path and is bit-for-bit the
// engine's `<false>` branch.
#if !defined(STRATA_ENABLE_VULKAN)
#error "rope_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/rope.hpp"           // build_rope_table (both overloads)
#include "strata/kernels/mrope.hpp"          // RopeTab / rope_table_set / rope_table_release / rope_table_for
#include "strata/kernels/rope_scaling.hpp"   // RopeScaling / rope_yarn_ramp

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace strata::kernels {

namespace {

/// The image-path (M-RoPE) table pointer: `mrope_table_set`/`mrope_table`.  native_rope.cu keeps one per
/// device; same single-process argument as the rope table above.
std::atomic<const int32_t*> g_mrope_tab{nullptr};

/// The registered angle table plus the scaling it was built with (native_rope.cu's RopeReg, one entry).
struct RopeReg {
    RopeTab tab;
    RopeScaling scaling;
};
RopeReg g_rope_tab;

/// native_rope.cu's `same_scaling`, field for field.
bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
           a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}

/// native_rope.cu's opt-in: STRATA_ROPE_TABLE=1.  Read ONCE (the env cannot change under the process's own
/// capture); the table's angles differ from the fast-math ones in the last bits, so the outputs move.
bool rope_table_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_ROPE_TABLE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

}  // namespace

// mrope.hpp: the image path's per-position table (null = the identity).
void mrope_table_set(const int32_t* device_table) { g_mrope_tab.store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table() { return g_mrope_tab.load(std::memory_order_relaxed); }

// ---- build_rope_table (rope.cu:48-86, verbatim) ------------------------------------------------------------
// The unscaled form: `inv = pow(theta, -2i/n_rot)`, `ang = pos * inv`, cos/sin in float64 then stored f32.
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

// The scaled form.  `none` is the original loop verbatim; linear/YaRN fold ggml's freq_scale and the ramp.
void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);   // the original loop, verbatim
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;   // ggml: the correction rides on ext_factor, not the type
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;    // the trained angle, ggml's theta_extrap
            const double interp = fs * extrap;         // ggml's theta_interp
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

// ---- rope_table_set / _release / _for (native_rope.cu:103-115, the single-device slice) ---------------------
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    g_rope_tab = RopeReg{RopeTab{cos_tab, sin_tab, max_pos}, scaling};
}
void rope_table_release(const float* cos_tab) {
    if (cos_tab != nullptr && g_rope_tab.tab.cos == cos_tab) g_rope_tab = RopeReg{};
}
RopeTab rope_table_for(const RopeScaling& scaling) {
    if (!rope_table_enabled()) return {};
    const RopeReg& r = g_rope_tab;
    return r.tab.cos != nullptr && same_scaling(r.scaling, scaling) ? r.tab : RopeTab{};
}

}  // namespace strata::kernels
