// ports/vulkan/shaders/common/wg_reduce.glsl - the workgroup-wide sum, in ONE place.
//
// WHY THIS IS NO LONGER A SUBGROUP TREE - AND THE HONEST VERSION OF THAT STORY.  It was rewritten because a NaN
// in the expert tier was attributed to `subgroupAdd` over a partially executing wave.  **THAT ATTRIBUTION WAS
// WRONG.**  With the tier's real defect fixed (see the `act_f16_at` note in s2_row_dot.glsl - the activation's
// fp16 scale was being read out of the EXPERT BLOB), the subgroup form produces the correct answer on all six of
// the tier's cases, on RADV (width 64) and on llvmpipe (width 8), at both 80 chunks < 256 lanes and 320 chunks
// > 256 lanes.  Re-measured, not argued: see the commit that added `case_s2expert_tier`.
//
// So this is a PORTABILITY CHOICE, not a bug fix, and it is recorded as one:
//   * the target hardware is Intel, where the driver picks the subgroup (SIMD) width per kernel and the port
//     cannot verify from the host which width a dispatch actually ran at - a workgroup-sized tree has no width
//     to be wrong about, and needs no subgroup-arithmetic capability at all;
//   * the price is log2(256) = 8 barrier rounds per reduction instead of 2 subgroup rounds, and the price on Arc
//     is UNMEASURED.  If that price turns out to matter, reverting is a one-file change and the subgroup form is
//     proven green at two widths.
//
// WHAT THIS VERSION REQUIRES THAT THE SUBGROUP ONE DID NOT: the CALL must be reached by every invocation in the
// workgroup (barriers).  A caller with fewer ITEMS than lanes is fine - idle lanes contribute 0.0, which is what
// `s2_row_dot` returns for an invocation with no chunk - but a caller that calls the reduction inside divergent
// control flow is a hang or undefined behaviour here, where the subgroup form would have returned a wrong number.
// Every caller in this tree is workgroup-uniform: the row/stream guards are the workgroup id or a workgroup-wide
// bound, which is what makes the requirement satisfiable and checked (the gate's census requires the barriers).
//
// THE TWO OLDER DEFECTS THIS FILE ENCODES, both silent, both found by the gate:
//
//   1. The array must be sized for the WIDEST workgroup, and the tree must not read entries nobody wrote.
//      `MAX_SUBGROUPS`-style sizing for the narrowest subgroup (256 threads / 8 lanes = 32) is the classic
//      silent failure: entries nobody wrote get summed into the result.
//   2. **STAGE 2 MUST COVER EVERY ENTRY.**  The subgroup version once read one entry per lane, which is correct
//      only while the entry count is <= the subgroup width - true at width 64 (4 entries) and at 32 (8), FALSE
//      at width 8 where there are 32 entries and a subgroup has 8 lanes.  It silently dropped 24 of 32 subgroup
//      sums: every value came out ~2x off (measured worst relative error 1.13 on llvmpipe, subgroup 8, against
//      2.35e-07 on RADV, subgroup 64).  Invisible on the only device it was first tested on.  A barrier tree
//      cannot have this failure: every entry is folded, once, by construction.
//
// The tree's summation ORDER is fixed (pairwise), where the driver chose the subgroup tree's - so the last bits
// of a total are this file's and not the driver's, which is the port's preference everywhere else too.
//
// `wg_sum` RETURNS the total to every invocation rather than leaving it in shared memory, so a caller that
// needs the sum can do its own post-processing without another barrier and without a second shared variable.
// All invocations read `rp_partial[0]` after the final barrier, so the value they get is identical.
//
// NOTHING HERE IS CONFIGURED FROM `gl_SubgroupSize` OR `gl_NumSubgroups`, deliberately: the driver picks a
// subgroup width per kernel on Intel (8/16/32, free to differ from the device-wide default), so any host-side
// calculation that divides by the reported width is unsafe there - the `rms_norm` lesson.
const uint WG_SUM_MAX_SIZE = 256u;               // the host's kLocalSize, checked by the gate against every shader
shared float rp_partial[WG_SUM_MAX_SIZE];

float wg_sum(float v) {
    // **THE LEADING BARRIER IS NOT DECORATION ONCE THIS IS CALLED MORE THAN ONCE PER INVOCATION.**  The array is
    // reused across calls, so without this a fast invocation can overwrite the previous total before a slow one
    // has read it.  The CUDA equivalent (`block_sum` in ple.cu) carries the same barrier and its comment gives
    // the same reason - "invisible in most runs and a slightly different norm when it fires".  The single-call
    // kernels never reach it; the multi-token MMVF reduces once per activation row and does.
    barrier();
    rp_partial[gl_LocalInvocationIndex] = v;
    barrier();
    // Pairwise tree over the WHOLE workgroup.  The `if` is divergent and that is fine: every barrier is outside
    // it, so all invocations reach every barrier - which is the only requirement shared-memory exchange has.
    for (uint step = gl_WorkGroupSize.x >> 1u; step > 0u; step >>= 1u) {
        if (gl_LocalInvocationIndex < step) {
            rp_partial[gl_LocalInvocationIndex] += rp_partial[gl_LocalInvocationIndex + step];
        }
        barrier();
    }
    return rp_partial[0];
}
