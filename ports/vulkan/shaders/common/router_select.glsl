// ports/vulkan/shaders/common/router_select.glsl - the MoE router's selection, shared by both arithmetic variants.
//
// SEMANTICS, transcribed from ref/moe.py::router (itself from llama-graph.cpp):
//
//     p   = softmax(logits)                    over ALL experts, not over the selected subset
//     ids = stable argsort(-p)[:k]             ties break by INDEX, ASCENDING
//     w   = gather(p, ids)
//     s   = max(sum(w), 2**-14)                ggml_clamp(..., 2**-14, INF)
//     return ids, w / s
//
// The two-step structure matters: computing the softmax over the selected subset instead is mathematically
// identical (softmax is shift-invariant) but it MOVES the renormalisation, and the clamp is part of the
// renormalisation - so the gather form is what puts the clamp in the same place.
//
// THE SELECTION IS k PASSES OF A BLOCK-WIDE ARGMAX, and the tie rule is the whole correctness argument: scanning
// `e` ASCENDING with a strict `>` keeps the LOWEST index on a tie, which is exactly the "stable descending
// argsort, ties by ascending index" the reference spells out. The source replaced an O(n^2) rank-by-counting
// loop with this (10x512 compares instead of 512x512) and the comment records that the O(n^2) STRUCTURE was the
// cost, not the arithmetic inside it.
//
// The combine is `ov > bv || (ov == bv && oi < bi)` - "maximum, ties to the lowest index" - which is associative
// and commutative, so the result does NOT depend on the order the reduction visits things in. That is why the
// port can reduce with a single deterministic scan over the invocations instead of the source's shuffle
// butterfly: same answer, no subgroup-width assumption.
// GLSL has no INFINITY, and a literal like -1e38 is NOT a substitute: the router's degenerate row is all
// -infinity, where max(-1e38, -inf) would be -1e38 and the softmax would then produce finite probabilities
// instead of NaN. The bit pattern keeps -inf as -inf.
float rs_neg_inf() { return uintBitsToFloat(0xFF800000u); }

const uint RS_MAX_EXPERTS = 512u;
shared float rs_p[RS_MAX_EXPERTS];          // the probabilities, as the source's float expression produces them
// `uint`, not a byte type: an include cannot declare the extensions a `uint8_t` would need (#extension must come
// before any non-preprocessor token), and 2 KB of shared memory is not the constraint here.
shared uint rs_taken[RS_MAX_EXPERTS];       // one word per expert: already selected
shared float rs_bv[256];                    // per-invocation scratch for the two block-wide reductions
shared int rs_bi[256];

// the block-wide MAX. Order-independent (a max of floats is associative and exact), so it is bit-identical to the
// source's warp tree - which its comment relies on for the softmax's stability term.
float rs_max_all(float v, uint lid, uint lsize) {
    rs_bv[lid] = v;
    barrier();
    if (lid == 0u) {
        float b = rs_neg_inf();
        for (uint i = 0u; i < lsize; ++i) b = max(b, rs_bv[i]);
        rs_bv[0] = b;
    }
    barrier();
    return rs_bv[0];
}

void rs_clear_taken(uint n_expert, uint lid, uint lsize) {
    for (uint e = lid; e < n_expert; e += lsize) rs_taken[e] = 0u;
    barrier();
}

/// The highest-ranked expert still untaken, or `n_expert` when none is left. The sentinel matters: the source
/// writes NOTHING for that rank, so the output buffer keeps what it held - and the renormalisation divides
/// whatever the buffer holds. That is a behaviour, not an oversight, and the gate tests it.
int rs_rank_max(uint n_expert, uint lid, uint lsize) {
    float bv = rs_neg_inf();
    int bi = int(n_expert);                       // loses to every real index
    for (uint e = lid; e < n_expert; e += lsize) {
        if (rs_taken[e] != 0u) continue;
        const float pe = rs_p[e];
        if (pe > bv) {                            // STRICT: the lowest index wins a tie
            bv = pe;
            bi = int(e);
        }
    }
    rs_bv[lid] = bv;
    rs_bi[lid] = bi;
    barrier();
    if (lid == 0u) {
        float b = rs_neg_inf();
        int ix = int(n_expert);
        for (uint t = 0u; t < lsize; ++t) {
            const float ov = rs_bv[t];
            const int oi = rs_bi[t];
            if (ov > b || (ov == b && oi < ix)) { b = ov; ix = oi; }
        }
        rs_bi[0] = ix;
    }
    barrier();
    return rs_bi[0];
}
