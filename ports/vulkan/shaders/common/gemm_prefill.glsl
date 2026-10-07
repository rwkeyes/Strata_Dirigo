// ports/vulkan/shaders/common/gemm_prefill.glsl - the PREFILL GEMM in the layout the engine actually calls.
//
//     Y[T x ldy] = X[T x K] . W[N x K]^T          (f16 operands, f32 accumulate)
//
// This is `Gemm::f16` in src/prefill/gemm.cu, whose cuBLAS call is (CUBLAS_OP_T, CUBLAS_OP_N) with m = N, n = T,
// k = K: THE WEIGHT IS THE TRANSPOSED OPERAND, so an output row is an output FEATURE and the reduction runs over
// the contiguous axis of both inputs.  That is a different layout from gemm_coopmat.comp / gemm_fma.comp, which
// compute C[MxN] = A[MxK].B[KxN] and are the SHAPE and no-CMA fallbacks rather than the prefill path.
//
// Getting the layout backwards is the classic silent GEMM bug - every number comes out plausible - so it is
// stated here, each loader below says which layout it reads, and the gate compares against a double-precision
// reference at shapes where N != K, so a transposed read is wrong by more than any tolerance can absorb.
//
// THE TILE IS THE DEVICE'S: CM_M rows is 8 on Intel XMX and 16 on RADV/WMMA (measured with
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR - see gemm_coopmat_m8.comp).  CM_M is a compile-time knob
// because a GLSL cooperative matrix is sized at compile time; the includer sets it, and the caller selects the
// pipeline from the device's own config list.  CM_N and CM_K are 16 on every device measured so far.
//
// THE SHAPE PRECONDITION IS THE CALLER'S, and it is not optional: this kernel has NO ragged-edge path.  T must be
// a multiple of CM_M and N, K multiples of CM_N/CM_K, or the tail rows are simply not computed (the tile count
// rounds down) and the output is left as the caller found it.  portvk::gemm_shape_ok(t, n, k, CM_M, CM_N, CM_K)
// is the rule; rows the tile cannot cover are the FMA kernel's job (gemm_prefill_fma.comp), which is why that
// kernel exists and why the gate exercises the SPLIT (aligned rows here, remainder there) rather than each half
// alone.
//
// DISPATCH: one tile per SUBGROUP, and how many tiles a workgroup covers comes from the shader's own constants
// (gl_WorkGroupSize / gl_SubgroupSize), never from a width the host looked up.  The host dispatches an UPPER
// BOUND and the surplus exits - over-dispatch costs nothing, under-dispatch silently drops output.
#ifndef CM_M
#define CM_M 8
#endif
#ifndef CM_N
#define CM_N 16
#endif
#ifndef CM_K
#define CM_K 16
#endif

// CM_CT: OUTPUT TILES PER SUBGROUP along N.  Added 2026-10-07 because the SYCL/CUDA trees get far more out of the
// same XMX hardware (8-39x in the phases that use it) while this port's cooperative-matrix kernel measured SLOWER
// than its own FMA path -- and the reason is structural rather than a device property: at CM_CT = 1 a subgroup
// holds ONE accumulator, so its K loop is a serial chain of dependent coopMatMulAdds with a single B load each,
// and every loaded A element is reused only TM times (the staged sibling measured 7.5 MACs per staged element
// against the reference's 16+).  CM_CT > 1 gives a subgroup CM_CT INDEPENDENT accumulators and reuses each A load
// CM_CT times, i.e. TM*CM_CT MACs per A element.  It is a COMPILE-TIME knob (a cooperative matrix is sized at
// compile time) and the caller must divide N the same way -- an N that CM_CT cannot divide falls back to the FMA
// kernel instead of silently dropping columns.
#ifndef CM_CT
#define CM_CT 1
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
const uint TC = CM_CT;       // output tiles per SUBGROUP along N (see the CM_CT note above)

void main() {
    // CM_CT > 1: each subgroup covers CM_CT CONSECUTIVE N-tiles, so the tile grid it indexes is narrower by that
    // factor.  The caller divides `n` the same way (portvk::gemm_shape_ok takes CM_CT), so an `n` that TC cannot
    // divide never reaches this shader -- it falls back to the FMA kernel rather than silently dropping columns.
    const uint tiles_n = (pc.n / TN) / TC;         // N-tile bands, one per subgroup
    const uint tiles_t = pc.t / TM;
    const uint total = tiles_t * tiles_n;
    const uint per_wg = max(1u, gl_WorkGroupSize.x / gl_SubgroupSize);
    const uint tile = gl_WorkGroupID.x * per_wg + gl_SubgroupID;
    if (tile >= total) return;                     // surplus workgroups exit; no duplicate tile is computed

    const uint t0 = (tile / tiles_n) * TM;         // token rows of the output tile, row-major over the tile grid
    const uint n0 = (tile % tiles_n) * TN * TC;    // FIRST output feature of this subgroup's band

    coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> amat;
    coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> bmat[TC];
    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc[TC];
    for (uint j = 0u; j < TC; ++j) acc[j] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);

    // K is walked in TK chunks.  A is RowMajor over X[T x K]: element (r, c) is at X[t0*K + kk + r*K + c], i.e.
    // offset t0*K + kk with stride K.  B is COLUMN-major over W[N x K] - element (r, c) at buf[offset + r +
    // c*stride] - which is what turns W's rows into B's columns: B[r][c] = W[n0 + c][kk + r], offset n0*K + kk,
    // stride K.  Reading W RowMajor here would be the transposed-operand bug: it maps B[r][c] to
    // W[n0 + r][kk + c], which is a different matrix unless K == N.
    //
    // THE TC INNER LOOPS ARE THE WHOLE POINT OF CM_CT.  The TC accumulators are INDEPENDENT, so the TC
    // coopMatMulAdds of one kk do not serialise on each other - the XMX pipeline can hold several in flight -
    // and the ONE amat load feeds all TC of them, which is what raises the arithmetic intensity per loaded A
    // element from TM MACs to TM*TC (the staged sibling's header measured 7.5 MACs/staged element at TC = 1, the
    // reason it loses to a global-load kernel).  At TC = 1 this reproduces the shipped loop's BEHAVIOUR AND
    // TIMING (bench gate/up T=199: 3.3351 ms against the recorded 3.334) but NOT its exact SPIR-V - the array
    // form grows the module 4,544 -> 6,000 bytes at TC = 1, so the pristine body is not byte-for-byte recoverable
    // from this file and a default-path change here needs the gate, not a size comparison.
    for (uint kk = 0u; kk < pc.k; kk += TK) {
        coopMatLoad(amat, X.x, t0 * pc.k + kk, pc.k, gl_CooperativeMatrixLayoutRowMajor);
        for (uint j = 0u; j < TC; ++j) {
            coopMatLoad(bmat[j], W.w, (n0 + j * TN) * pc.k + kk, pc.k, gl_CooperativeMatrixLayoutColumnMajor);
        }
        for (uint j = 0u; j < TC; ++j) {
            acc[j] = coopMatMulAdd(amat, bmat[j], acc[j]);
        }
    }

    for (uint j = 0u; j < TC; ++j) {
        coopMatStore(acc[j], Y.y, t0 * pc.ldy + n0 + j * TN, pc.ldy, gl_CooperativeMatrixLayoutRowMajor);
    }
}
