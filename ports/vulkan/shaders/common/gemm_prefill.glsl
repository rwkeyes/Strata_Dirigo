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

void main() {
    const uint tiles_n = pc.n / TN;
    const uint tiles_t = pc.t / TM;
    const uint total = tiles_t * tiles_n;
    const uint per_wg = max(1u, gl_WorkGroupSize.x / gl_SubgroupSize);
    const uint tile = gl_WorkGroupID.x * per_wg + gl_SubgroupID;
    if (tile >= total) return;                     // surplus workgroups exit; no duplicate tile is computed

    const uint t0 = (tile / tiles_n) * TM;         // token rows of the output tile, row-major over the tile grid
    const uint n0 = (tile % tiles_n) * TN;         // output features

    coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> amat;
    coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> bmat;
    coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc;
    acc = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);

    // K is walked in TK chunks.  A is RowMajor over X[T x K]: element (r, c) is at X[t0*K + kk + r*K + c], i.e.
    // offset t0*K + kk with stride K.  B is COLUMN-major over W[N x K] - element (r, c) at buf[offset + r +
    // c*stride] - which is what turns W's rows into B's columns: B[r][c] = W[n0 + c][kk + r], offset n0*K + kk,
    // stride K.  Reading W RowMajor here would be the transposed-operand bug: it maps B[r][c] to
    // W[n0 + r][kk + c], which is a different matrix unless K == N.
    for (uint kk = 0u; kk < pc.k; kk += TK) {
        coopMatLoad(amat, X.x, t0 * pc.k + kk, pc.k, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(bmat, W.w, n0 * pc.k + kk, pc.k, gl_CooperativeMatrixLayoutColumnMajor);
        acc = coopMatMulAdd(amat, bmat, acc);
    }

    coopMatStore(acc, Y.y, t0 * pc.ldy + n0, pc.ldy, gl_CooperativeMatrixLayoutRowMajor);
}
