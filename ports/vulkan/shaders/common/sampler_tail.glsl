// ports/vulkan/shaders/common/sampler_tail.glsl - the sampler chain's TAIL, in ONE place.
//
//     top_p (shortest prefix whose probability reaches p, never shorter than min_keep)
//       -> min_p (then the descending prefix within log(min_p) of the head)
//       -> temperature on the SURVIVORS ONLY, softmax over them
//       -> ONE Philox draw over the cumulative walk
//
// It is the arithmetic of sampler_kernel.comp (`sampler_kernel`, src/kernels/cuda/sampler.cu), moved here because
// the port now has THREE callers of the same tail: the block-per-token kernel's portable f32 sibling
// (sampler_kernel_f32), the split sampler's merge stage (sampler_split) and the coupled draft merge
// (coupled_sample).  One source of truth for the tail is also what makes the split's token EQUAL the one-block
// token by construction rather than by a careful copy - which is the engine's own claim for its three sampled
// paths ("pick the same token, bit for bit").
//
// TWO VARIANTS, and only one of them is always available:
//   * `sampler_tail_f64` is the engine's arithmetic (the sums and the exps in DOUBLE).  It needs shaderFloat64.
//   * `sampler_tail_f32` is the PORTABLE sibling - the same chain in float - for the hardware the f64 one cannot
//     run on.  The gate MEASURES the two against each other rather than asserting the gap is nothing.
//   The f64 half is compiled ONLY where the shader defines PORT_SAMPLER_WANT_F64 before including this file, so a
//   device without fp64 does not carry the capability through a shared include it never calls.
//
// REQUIRES in the including shader (a GLSL include cannot take an SSBO as an argument, so this is a documented
// contract rather than a hidden one):
//   * the logits/history buffers are NOT needed here - the tail runs over the selection list only;
//   * nothing else: the selection list is this file's own shared arrays, below.
// The SELECTION (common/sampler_select.glsl) writes `sc_sel_id` / `sc_sel_lg` and must be included after this one.

#ifndef PORT_SAMPLER_TAIL_GLSL
#define PORT_SAMPLER_TAIL_GLSL

#include "philox.glsl"

#define SC_KMAX 64                       // the widest shortlist, `sampler_kernel`'s KMAX / the engine's kSelMax

// The selection list, in SELECTION ORDER: value descending, ties to the lower id.  Declared HERE because the tail
// is the reader and the selection is the writer, and one owner is better than two declarations that can drift.
shared int sc_sel_id[SC_KMAX];
shared float sc_sel_lg[SC_KMAX];

#ifdef PORT_SAMPLER_WANT_F64
#include "double_math.glsl"

// The engine's tail, term for term.  Every invocation computes it; the caller writes the token from lane 0.
// `prob` is the pick's probability under the final (post-temperature) distribution - the coupled drafter needs it,
// the sampler itself ignores it.
void sampler_tail_f64(int k, float temperature, float top_p, float min_p, int min_keep,
                      uint slo, uint shi, uint clo, uint chi, int t, out int pick, out float prob) {
    const float inv_t = (temperature > 0.0) ? (1.0 / temperature) : 0.0;

    int n_keep = k;
    float mx = sc_sel_lg[0];
    for (int i = 1; i < k; ++i) mx = max(mx, sc_sel_lg[i]);

    if (top_p < 1.0) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += double_exp(double(sc_sel_lg[i]) - double(mx));
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += double_exp(double(sc_sel_lg[i]) - double(mx)) / sum;
            if (cum >= double(top_p)) { cut = i + 1; break; }
        }
        if (cut < min_keep) cut = (min_keep < k) ? min_keep : k;
        n_keep = cut;
    }

    if (min_p > 0.0) {
        // In logit space: `sel_logit[0] + log(min_p)` is `p >= min_p * p_max` without the overflow of exp(raw).
        // The head always survives (`exp(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
        const float thresh = sc_sel_lg[0] + log(min_p);
        for (int i = 0; i < n_keep; ++i) {
            if (sc_sel_lg[i] < thresh) { n_keep = i; break; }
        }
    }

    float smx = sc_sel_lg[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = max(smx, sc_sel_lg[i] * inv_t);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += double_exp(double(sc_sel_lg[i] * inv_t) - double(smx));

    const float u = philox_uniform(slo, shi, clo + uint(t), chi);
    double cum = 0.0;
    int pi = n_keep - 1;
    int pk = sc_sel_id[pi];
    for (int i = 0; i < n_keep; ++i) {
        const double e = double_exp(double(sc_sel_lg[i] * inv_t) - double(smx)) / sum;
        cum += e;
        if (double(u) < cum) { pk = sc_sel_id[i]; pi = i; break; }
    }
    pick = pk;
    prob = (n_keep > 0) ? float(double_exp(double(sc_sel_lg[pi] * inv_t) - double(smx)) / sum) : 1.0;
}
#endif  // PORT_SAMPLER_WANT_F64

// The PORTABLE sibling: the identical chain in float, for a device with no shaderFloat64 (Intel Arc, per Intel's
// own support article 000089817 - the target hardware).  The gate runs it on EVERY device and measures its
// disagreement with the f64 chain; the two are exact wherever the arithmetic is (a one-survivor shortlist, and
// equal survivors at temperature 0, where the softmax cancels out of the walk).
void sampler_tail_f32(int k, float temperature, float top_p, float min_p, int min_keep,
                      uint slo, uint shi, uint clo, uint chi, int t, out int pick, out float prob) {
    const float inv_t = (temperature > 0.0) ? (1.0 / temperature) : 0.0;

    int n_keep = k;
    float mx = sc_sel_lg[0];
    for (int i = 1; i < k; ++i) mx = max(mx, sc_sel_lg[i]);

    if (top_p < 1.0) {
        float sum = 0.0;
        for (int i = 0; i < k; ++i) sum += exp(sc_sel_lg[i] - mx);
        float cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += exp(sc_sel_lg[i] - mx) / sum;
            if (cum >= top_p) { cut = i + 1; break; }
        }
        if (cut < min_keep) cut = (min_keep < k) ? min_keep : k;
        n_keep = cut;
    }

    if (min_p > 0.0) {
        const float thresh = sc_sel_lg[0] + log(min_p);
        for (int i = 0; i < n_keep; ++i) {
            if (sc_sel_lg[i] < thresh) { n_keep = i; break; }
        }
    }

    float smx = sc_sel_lg[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = max(smx, sc_sel_lg[i] * inv_t);
    float sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp(sc_sel_lg[i] * inv_t - smx);

    const float u = philox_uniform(slo, shi, clo + uint(t), chi);
    float cum = 0.0;
    int pi = n_keep - 1;
    int pk = sc_sel_id[pi];
    for (int i = 0; i < n_keep; ++i) {
        const float e = exp(sc_sel_lg[i] * inv_t - smx) / sum;
        cum += e;
        if (u < cum) { pk = sc_sel_id[i]; pi = i; break; }
    }
    pick = pk;
    prob = (n_keep > 0) ? (exp(sc_sel_lg[pi] * inv_t - smx) / sum) : 1.0;
}

#endif  // PORT_SAMPLER_TAIL_GLSL
