// ports/vulkan/shaders/common/sampler_select.glsl - the sampler chain's SELECTION, shared by the split sampler
// and the coupled draft merge.  Ported from `sampler_split_part_kernel` + `sampler_split_merge_kernel`
// (src/kernels/cuda/sampler.cu), whose whole content is the partition-then-merge argument:
//
//   * a 4096-logit PARTITION keeps its own top-k (the first k of a union are within the first k of each part);
//   * the parts are merged pairwise into their first k, and a merge of ordered lists is ordered.
//
// So the row's list is EXACT - the selection order is a strict total order (value descending, ties to the lower
// id), which is what makes the merge well defined.  The engine runs the parts as a grid over the whole GPU and the
// merge on one warp per row with `__shfl_xor`; this port's reductions are barrier trees (the design decision in
// wg_reduce.glsl - the target is Intel, where the driver picks the subgroup width per kernel), so ONE workgroup
// covers ONE row and walks its partitions, and the barrier tree is the merge's "first of the heads".
//
// REQUIRES in the including shader (documented, because a GLSL include cannot take an SSBO as an argument):
//   * `common/sampler_tail.glsl` included FIRST (it owns `sc_sel_id` / `sc_sel_lg`, the list this file writes);
//   * `LO_` - the logits storage buffer, `layout(...) readonly buffer LOGITS { float x[]; } LO_;`
//   * `HI_` - the history storage buffer, only when PORT_SAMPLER_USE_HIST is defined;
//   * a push-constant block `Pc pc` with: n_vocab, history_len, penalty_last_n, penalty_repeat, penalty_freq,
//     penalty_present.  (n_vocab is the candidate space; the coupled shader passes it as the draft subset size.)
// The workgroup must be 256 invocations (the port's kLocalSize) and every invocation must reach this function
// (a barrier tree).

#ifndef PORT_SAMPLER_SELECT_GLSL
#define PORT_SAMPLER_SELECT_GLSL

#define SC_SPAN 4096                     // logits per partition, the engine's kSplitBlockSpan
#define SC_LANES 256                     // the workgroup width this include documents
#define SC_PER_LANE (SC_SPAN / SC_LANES) // the partition logits one lane holds: its register slice

shared int sc_tmp_id[SC_KMAX];           // the current partition's top-k
shared float sc_tmp_lg[SC_KMAX];
shared int sc_mrg_id[SC_KMAX];           // the merge's output (a forward in-place merge would clobber entries
shared float sc_mrg_lg[SC_KMAX];         // still to be read once a partition entry has been taken)
shared float sc_rv[256];                 // the barrier tree's scratch
shared int sc_ri[256];

// top_k 1..64 as given; 0 ("off") and anything wider keep the widest shortlist, and never more than the vocabulary.
int sc_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < SC_KMAX) ? top_k : SC_KMAX;
    return k > n_vocab ? n_vocab : k;
}

#ifdef PORT_SAMPLER_USE_HIST
// The penalty pair, transcribed from the engine (`apply_penalties`): the repeat penalty MULTIPLIES for a
// non-positive logit and DIVIDES for a positive one, and the presence penalty is `float(count > 0)` - a boolean.
float sc_penalized(float logit, int count, float prepeat, float pfreq, float ppresent) {
    if (count <= 0) return logit;
    if (logit <= 0.0) logit *= prepeat;
    else              logit /= prepeat;
    logit -= float(count) * pfreq + 1.0 * ppresent;
    return logit;
}
#endif

// The row `t`'s top-k into `sc_sel_id` / `sc_sel_lg`, in selection order.
void sampler_row_topk(int t, int k) {
    const uint lid = gl_LocalInvocationIndex;
    const int nvb = pc.n_vocab;
    const uint row = uint(t) * uint(nvb);
#ifdef PORT_SAMPLER_USE_HIST
    int hlen = 0, hbase = 0;
    if (pc.history_len > 0) {
        hlen = min(pc.penalty_last_n, pc.history_len);
        if (hlen < 0) hlen = 0;
        hbase = t * pc.history_len + (pc.history_len - hlen);    // the window is the TAIL
    }
#endif
    const int n_blocks = (nvb + SC_SPAN - 1) / SC_SPAN;
    int ncur = 0;                         // the running list's length (meaningful in lane 0)
    for (int b = 0; b < n_blocks; ++b) {
        const int lo = b * SC_SPAN;
        const int hi = min(lo + SC_SPAN, nvb);

        // ---- the partition's PENALISED logits, computed ONCE per element into the lane's registers ----
        // A lane's candidates are `lo + lid + SC_LANES*j`, j in [0, SC_PER_LANE) - its SLICE of the partition.
        // This is the engine's own shape (`s[kSplitPerLane]`): the penalty (and the logit read) happens once
        // per element, and the k rounds scan the CACHED values.  The port's first form re-read the logit and
        // re-scanned the whole penalty window on EVERY round - O(k * span * hlen), where the engine is
        // O(span + hits * hlen).  Measured on the Arc at vocab 248320 / n_tokens 1 / k 64 / window 64:
        // 348.95 ms with the per-round rescan against 12.66 ms with the penalty scan removed entirely, so
        // this hoist is the one that matters (the k-round scan itself is the remainder).
        float sc_pv[SC_PER_LANE];
        int sc_pi[SC_PER_LANE];
        for (int j = 0; j < SC_PER_LANE; ++j) {
            const int v = lo + int(lid) + SC_LANES * j;
            float s = uintBitsToFloat(0xff800000u);    // past the partition's end: -inf with the sentinel id
            int id = nvb;
            if (v < hi) {
                s = LO_.x[row + uint(v)];
                id = v;
#ifdef PORT_SAMPLER_USE_HIST
                int cnt = 0;
                for (int q = 0; q < hlen; ++q) if (HI_.h[hbase + q] == v) ++cnt;
                s = sc_penalized(s, cnt, pc.penalty_repeat, pc.penalty_freq, pc.penalty_present);
#endif
            }
            sc_pv[j] = s;
            sc_pi[j] = id;
        }

        // ---- the partition's top-k: k rounds of a workgroup argmax over the not-yet-taken ----
        float prev_v = uintBitsToFloat(0x7f800000u);   // +inf, id -1: round 0 takes every logit
        int prev_i = -1;
        for (int i = 0; i < k; ++i) {
            float bv = uintBitsToFloat(0xff800000u);   // -inf
            int best = nvb;                            // the sentinel loses every comparison
            for (int j = 0; j < SC_PER_LANE; ++j) {
                const float s = sc_pv[j];
                const int v = sc_pi[j];
                // `j` ascends with `v`, and the comparison is strict, so the lowest id still wins a tie.
                if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
            }
            sc_rv[lid] = bv;
            sc_ri[lid] = best;
            barrier();
            for (int off = 128; off > 0; off >>= 1) {
                if (int(lid) < off) {
                    const float ov = sc_rv[lid + uint(off)];
                    const int oi = sc_ri[lid + uint(off)];
                    if (ov > sc_rv[lid] || (ov == sc_rv[lid] && oi < sc_ri[lid])) {
                        sc_rv[lid] = ov;
                        sc_ri[lid] = oi;
                    }
                }
                barrier();
            }
            // The round's result is in shared memory, so EVERY lane can advance its own threshold without a
            // second broadcast - and every lane needs it, since the next round's candidate test uses it.
            const float rv0 = sc_rv[0];
            const int ri0 = (sc_ri[0] < nvb) ? sc_ri[0] : 0;   // a round that found nothing: id 0, -inf logit
            if (lid == 0u) { sc_tmp_id[i] = ri0; sc_tmp_lg[i] = rv0; }
            prev_v = rv0;
            prev_i = ri0;
            barrier();
        }

        // ---- merge the partition's list (length k) into the running list (length ncur) ----
        // INTO A SEPARATE ARRAY, then copy back.  A forward MERGE IN PLACE is NOT safe here: the write index
        // `o = a + c` runs ahead of the read index `a` as soon as a partition entry is taken (`c > 0`), so a
        // forward pass overwrites running-list entries that a later `a` still has to read - measured, and the
        // reason the arms below carry a multi-partition row.
        if (lid == 0u) {
            int a = 0, c = 0, o = 0;
            const int nn = min(k, ncur + k);
            while (o < nn) {
                bool take_a;
                if (c >= k)         take_a = (a < ncur);
                else if (a >= ncur) take_a = false;
                else take_a = (sc_sel_lg[a] > sc_tmp_lg[c]) ||
                              (sc_sel_lg[a] == sc_tmp_lg[c] && sc_sel_id[a] < sc_tmp_id[c]);
                if (take_a) { sc_mrg_id[o] = sc_sel_id[a]; sc_mrg_lg[o] = sc_sel_lg[a]; ++a; }
                else        { sc_mrg_id[o] = sc_tmp_id[c]; sc_mrg_lg[o] = sc_tmp_lg[c]; ++c; }
                ++o;
            }
            for (int i = 0; i < nn; ++i) { sc_sel_id[i] = sc_mrg_id[i]; sc_sel_lg[i] = sc_mrg_lg[i]; }
            ncur = nn;
        }
        barrier();
    }
}

#endif  // PORT_SAMPLER_SELECT_GLSL
