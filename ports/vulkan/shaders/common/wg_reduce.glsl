// ports/vulkan/shaders/common/wg_reduce.glsl - the workgroup-wide sum, in ONE place.
//
// Three kernels need it (rms_norm_weighted, PLE's grouped_norm, PLE's gate), so it lives here rather than being
// copied three times: two separate silent defects were found in this reduction while porting rms_norm, and a
// third copy is a third place for them to come back.
//
// The two defects it encodes:
//
//   1. `MAX_SUBGROUPS` is sized for the NARROWEST subgroup the hardware may use - 256 threads / 8 lanes = 32.
//      A smaller fixed count is the classic silent failure: entries nobody wrote get summed into the result.
//      The second stage reads exactly `gl_NumSubgroups` entries and the write is bounds-guarded.
//   2. **STAGE 2 MUST LOOP.** It first read `partial[gl_SubgroupInvocationID]` for
//      `gl_SubgroupInvocationID < gl_NumSubgroups`, which is correct only while gl_NumSubgroups <= the subgroup
//      width - true at width 64 (4 entries) and at 32 (8), FALSE at width 8 where there are 32 entries and a
//      subgroup has 8 lanes. It silently dropped 24 of 32 subgroup sums: the sum came out ~1/4 of the truth and
//      every value was ~2x off (measured worst relative error 1.13 on llvmpipe, subgroup 8, against 2.35e-07 on
//      RADV, subgroup 64). Invisible on the only device it was first tested on.
//
// `wg_sum` RETURNS the total to every invocation rather than leaving it in shared memory, so a caller that needs
// the sum can do its own post-processing without another barrier and without a second shared variable. The
// redundant recomputation in the other lanes is a few flops.
//
// Call it only from a position that is UNIFORM ACROSS THE WORKGROUP: it contains barriers, and the row/stream
// guards in the callers are what make that legal.
const uint MAX_SUBGROUPS = 32u;
shared float rp_partial[MAX_SUBGROUPS];
shared float rp_total;

float wg_sum(float v) {
    // **THE LEADING BARRIER IS NOT DECORATION ONCE THIS IS CALLED MORE THAN ONCE PER INVOCATION.**  The total is
    // read out of shared memory and a later call reuses the arrays, so without this a fast invocation can write
    // the next total before a slow one has read the previous one.  The CUDA equivalent (`block_sum` in ple.cu)
    // carries the same barrier and its comment gives the same reason - "invisible in most runs and a slightly
    // different norm when it fires".  The single-call kernels never reach it; the multi-token MMVF reduces once
    // per activation row and does.
    barrier();
    const float sub = subgroupAdd(v);
    if (gl_SubgroupInvocationID == 0u && gl_SubgroupID < MAX_SUBGROUPS) rp_partial[gl_SubgroupID] = sub;
    barrier();
    if (gl_SubgroupID == 0u) {
        float t = 0.0;
        for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize) t += rp_partial[i];
        t = subgroupAdd(t);
        if (gl_SubgroupInvocationID == 0u) rp_total = t;
    }
    barrier();
    return rp_total;
}
