// ports/vulkan/shaders/common/gemm_prefill_staged.glsl - the PREFILL GEMM on the matrix units, WITH THE
// OPERAND TILES STAGED IN SHARED MEMORY.
//
//     Y[T x ldy] = X[T x K] . W[N x K]^T          (f16 operands, f32 accumulate)
//
// SAME CONTRACT, SAME ARITHMETIC as common/gemm_prefill.glsl (the tile is the device's, k is walked in
// increasing order, f32 accumulate, no ragged edge); only the MEMORY SCHEDULE differs, and that is the whole
// point of this file.  Read both headers together: this one exists because the other one `coopMatLoad`s its
// operands STRAIGHT FROM GLOBAL MEMORY, and that is the property llama.cpp's XMX kernels do not have.
//
// ============================================================================================================
// WHAT THE REFERENCE DOES, AND WHY THIS PORT DID NOT
// ============================================================================================================
// `~/llama-050/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` (the same structure in `mul_mmq.comp` and in the
// `dequant_*` shaders) stages both operands in shared memory before the cooperative-matrix load:
//
//     shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];
//     shared FLOAT_TYPEV2 buf_b[BN * SHMEM_STRIDE];
//     ... load_a_to_shmem(...); load_b_to_shmem(...); barrier();
//     ... coopMatLoad(cache_a, buf_a, a_shmem_index(...), a_shmem_stride(), RowMajor);
//         coopMatLoad(cache_b, buf_b, (warp_c * WN + cm_col * TN) * SHMEM_STRIDE + i / 2, SHMEM_STRIDE, ColumnMajor);
//     ... barrier();
//
// The matrix-unit load then comes from SHARED, not from global.  Three things follow, and all three are
// properties of the XMX load path rather than of the arithmetic:
//
//   1. THE GLOBAL LOADS COALESCE.  A cooperative-matrix load from global is the driver's own gather: what a
//      lane reads is the shape of the TILE, not the shape of the memory.  Staging through shared lets 256
//      consecutive lanes read 256 consecutive global addresses (contiguous along K, which is the reduction axis
//      of BOTH operands in this layout), and only the shared read is tile-shaped.
//   2. A TILE READ BY SEVERAL SUBGROUPS IS READ FROM GLOBAL ONCE.  In this kernel a workgroup's subgroups share
//      the same 8 token rows and differ only in column tile, so the X tile was previously loaded EIGHT TIMES (once
//      per subgroup) from global; it is now loaded once and read eight times from shared.
//   3. THE RE-READ IS BOUNDED BY SHARED BANDWIDTH, not by global latency.  There is nothing to poll and no
//      barrier the GPU must retire other than the two per K step below, which is the same barrier discipline the
//      port's own tiled FMA kernel (`gemm_prefill_fma.comp`) already ships.
//
// WHAT THIS IS NOT: it is not a bigger tile, it is not more rows per workgroup, and it does not change the
// summation order of a single output element.  A workgroup still owns exactly TM token rows and BN columns, K is
// still walked in increasing order, and every accumulation is still f32.  The proof that matters is the same one
// the gate already applies: the f16 operands widen exactly, the products are exact, and the SUM order per output
// is unchanged, so the numbers are identical to the global-load kernel's at every shape the gate exercises.
//
// ============================================================================================================
// THE TILE, AND WHY THE BLOCK IS (TM rows) x (BN columns) x (BK reduction)
// ============================================================================================================
//   TM  = CM_M   the device's cooperative-matrix row count (8 on BMG/Xe2, 16 on RADV/WMMA) - the OUTPUT row
//                count of one block, and it must equal CM_M because the accumulator's M dimension is the tile's.
//   TN  = CM_N   the device's column count (16 on every device measured here).
//   TK  = CM_K   the device's reduction count (16).
//   BN  = CM_BN  columns per WORKGROUP, a multiple of TN.  THE SUBGROUP COUNT IS THE REASON IT IS 128: a
//                workgroup is 256 lanes and this port's devices are 32-wide, so there are eight subgroups, and
//                BN/TN must be at least that many or a subgroup has no column tile to own.  128 = 8 x 16 gives
//                each subgroup exactly one 8x16 tile, which is also what the global-load kernel computes per
//                subgroup - the work per workgroup is IDENTICAL, only the route the operands take changed.
//   BK  = CM_BK  reduction per shared-memory step (32 = 2 x TK).  Two barriers per step, so a larger BK is fewer
//                barriers; it is a compile-time knob for exactly the reason `gemm_prefill_fma.comp` records for
//                its own TK: at a small K the barrier pair is the cost.
//
// Shared memory: TM*(BK+8) + BN*(BK+8) f16 = (8 + 128) x 40 x 2 = 10,880 bytes at the defaults.  This card
// reports 131,072 bytes (measured in DeviceInfo), so the staging is bounded by the tile, not by the device.
// The +8 pad on the K stride keeps the eight subgroups' shared rows off the same bank.
//
// DISPATCH: one workgroup per (row block, column block) PAIR, laid out row-major over the block grid; a
// workgroup whose block index is past the grid returns immediately.  The host dispatches an UPPER BOUND (see
// `gemm_f16` in prefill_vk.cpp and the gate's own formula) and surplus workgroups exit - over-dispatch costs
// nothing, under-dispatch silently drops output, which is the rule the global-load kernel's header states and
// this file keeps.
#ifndef CM_M
#define CM_M 8
#endif
#ifndef CM_N
#define CM_N 16
#endif
#ifndef CM_K
#define CM_K 16
#endif
#ifndef CM_BN
#define CM_BN 128
#endif
#ifndef CM_BK
#define CM_BK 32
#endif

layout(local_size_x = 256) in;

// X: activations, [T x K] row-major.  W: weights, [N x K] row-major.  Y: [T x ldy] row-major.
layout(set = 0, binding = 0, std430) coherent readonly buffer Xbuf { float16_t x[]; } X;
layout(set = 0, binding = 1, std430) coherent readonly buffer Wbuf { float16_t w[]; } W;
layout(set = 0, binding = 2, std430) coherent buffer Ybuf { float y[]; } Y;

layout(push_constant) uniform Pc {
    uint t;      // tokens in X (rows of Y) - a multiple of CM_M, or the tail is not computed
    uint n;      // output features (rows of W)
    uint k;      // reduction length
    uint ldy;    // row stride of Y in floats (>= n)
} pc;

const uint TM = CM_M;
const uint TN = CM_N;
const uint TK = CM_K;
const uint BN = CM_BN;
const uint BK = CM_BK;
const uint XST = BK + 8u;          // shared row stride of the X tile, padded (see the note above)
const uint WST = BK + 8u;          // shared row stride of the W tile

shared float16_t xs[TM * XST];
shared float16_t ws[BN * WST];

void main() {
    const uint blocks_n = (pc.n + BN - 1u) / BN;
    const uint blocks_t = pc.t / TM;               // the no-ragged-edge contract: the tail is not computed here
    const uint block = gl_WorkGroupID.x;
    if (block >= blocks_t * blocks_n) return;      // surplus workgroup: exit, never duplicate a block

    const uint b_t = block / blocks_n;
    const uint b_n = block % blocks_n;
    const uint row0 = b_t * TM;                    // first token row of this block
    const uint col0 = b_n * BN;                    // first output feature of this block
    const uint sg = gl_SubgroupID;                 // this subgroup's column tile inside the block
    const uint tid = gl_LocalInvocationIndex;
    const uint nthreads = gl_WorkGroupSize.x;
    // A workgroup is 256 lanes and BN/TN is 8, so a 32-wide subgroup gives every subgroup exactly one column
    // tile.  A WIDER subgroup (a 64-wide device reporting an M8 config) would leave the block's high columns
    // uncomputed, and a NARROWER one would read past the staged W tile - so the column tile is explicitly
    // CONDITIONAL here rather than assumed: an inactive subgroup loads nothing, accumulates nothing and stores
    // nothing.  It still reaches both barriers, because a barrier inside a divergent branch is the hang this
    // port does not allow.  The caller keeps such a device OFF this kernel entirely
    // (`prefill_vk.cpp` requires subgroup_size <= 32 for the matrix-unit path); this is the belt to that pair
    // of braces, and it is why the staging can never be read out of bounds.
    const bool has_tile = (sg * TN) < BN;

    coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> amat;
    coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> bmat;
    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc;
    acc = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);

    // K is walked in BK chunks.  Within a chunk the two operands are staged ONCE per workgroup - X[TM x BK] and
    // W[BN x BK] - and every subgroup then loads its own tile of them from SHARED.  Both staging loops read the
    // contiguous axis (K) with consecutive lanes, so the global side is coalesced; the shared side is written at
    // the same linear index it was read from, so the staging costs no addressing arithmetic either.
    for (uint k0 = 0u; k0 < pc.k; k0 += BK) {
        for (uint jx = 0u; jx < (TM * BK) / 256u; ++jx) {
            const uint xi = tid + jx * nthreads;
            const uint r = xi / BK;
            const uint c = xi % BK;
            // A row past T or a K past the end stages a ZERO, which contributes exactly nothing to an f32 sum.
            const bool live = (row0 + r) < pc.t && (k0 + c) < pc.k;
            xs[r * XST + c] = live ? X.x[(row0 + r) * pc.k + (k0 + c)] : float16_t(0.0);
        }
        for (uint jw = 0u; jw < (BN * BK) / 256u; ++jw) {
            const uint wi = tid + jw * nthreads;
            const uint r = wi / BK;
            const uint c = wi % BK;
            const bool live = (col0 + r) < pc.n && (k0 + c) < pc.k;
            ws[r * WST + c] = live ? W.w[(col0 + r) * pc.k + (k0 + c)] : float16_t(0.0);
        }
        barrier();

        // A is RowMajor over X[TM x K]: element (r, c) is at xs[(r)*XST + (kk + c)], i.e. offset kk, stride XST.
        // B is COLUMN-major over W[BN x K] - element (r, c) at buf[offset + r + c*stride] - which is what turns
        // W's rows into B's columns: B[r][c] = W[col0 + sg*TN + c][k0 + kk + r], offset sg*TN*WST + kk, stride
        // WST.  Reading W RowMajor here would be the transposed-operand bug: it maps B[r][c] to
        // W[col0 + r][kk + c], which is a different matrix unless K == N.
        if (has_tile) {
            for (uint kk = 0u; kk < BK; kk += TK) {
                coopMatLoad(amat, xs, kk, XST, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(bmat, ws, sg * TN * WST + kk, WST, gl_CooperativeMatrixLayoutColumnMajor);
                acc = coopMatMulAdd(amat, bmat, acc);
            }
        }
        barrier();                                  // the staged tiles are consumed before the next chunk writes
    }

    // A column tile past N belongs to a block the grid rounded up: it was computed against zeros and is not an
    // output.  The ROWS are always in range - pc.t is a multiple of TM by contract, or blocks_t rounded it down.
    if (!has_tile || col0 + sg * TN >= pc.n) return;
    coopMatStore(acc, Y.y, row0 * pc.ldy + col0 + sg * TN, pc.ldy, gl_CooperativeMatrixLayoutRowMajor);
}
