// ports/vulkan/shaders/common/iq1m_dot.glsl - ONE IQ1_M WEIGHT PART against ONE q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_iq1_m_q8_1` (src/kernels/cuda/iq_kernels.cu, itself llama.cpp's vecdotq.cuh), which
// is the format the resident `coder-iq1_m` model's experts are stored in: 256 values per block, 8 dot calls per
// block, one activation block per call.
//
// THE LAYOUTS, byte for byte, from third_party/ggml/ggml-common.h - and they are documented here because every
// offset below is arithmetic on these two structs:
//
//     block_iq1_m, 56 bytes per 256 weights:
//         qs[32]     at  0    grid index, LOW 8 bits - one byte per 32-value part
//         qh[16]     at 32    grid index HIGH 3 bits, plus the shift bit, two parts per byte
//         scales[8]  at 48    3-bit sub-block scales AND the fp16 block scale, assembled from the same bytes
//
//     block_q8_1, 36 bytes per 32 values:
//         half2 ds   at  0    (d, s); **this dot uses d ONLY** - `s` is not read, which is why the gate fills it
//                             with a sentinel value that would change every result by ~1e4 if it were read
//         int8 qs[32] at  4
//
// HOW THE SCALES SHARE THE BYTES, because that is the part that looks like it cannot work: the block's fp16
// scale is assembled from the HIGH NIBBLE of scales[1], [3], [5] and [7], while an 8-part sub-scale pair comes
// out of `sc[iqs/2] >> (6 * (iqs % 2))` where `sc` is those same eight bytes read as uint16. Those two uses do
// not overlap (each byte's high nibble feeds only the fp16 scale, its low bits feed only the sub-scales), which
// is why a case can construct BOTH deliberately - and it must, because a random byte pattern would assemble an
// inf or NaN fp16 scale and the case would be measuring that instead of the dot.
//
// `iqs` (0..7) selects the part: the weight's 4-byte qs group at `4*iqs`, the qh bytes `2*iqs` and `2*iqs+1`, and
// the activation block for that part, which the caller has already indexed (part k of the row is `k`, since one
// 256-value block covers exactly 8 activation blocks).
const float IQ1M_DELTA = 0.125;                  // third_party/ggml/ggml-common.h: #define IQ1M_DELTA 0.125f

// THE INCLUDING SHADER MUST DECLARE, with these names:
//     w_b      the IQ1_M weight blocks (uint8 storage buffer, rows of `nb * 56` bytes)
//     act_b    the q8_1 activation blocks (uint8 storage buffer, 36 bytes per 32 values)
//     grid_b   the IQ1_S grid table (uint32 storage buffer, 2048 entries) - see harness/iq1s_grid.hpp
// **THE ACTIVATION'S WORDS, AND THE NAME SAYS WHICH BUFFER FOR A REASON.**  This helper used to be called
// `iq1m_read_int4` and read `w_b`, the WEIGHT buffer, while its only call site passes an ACTIVATION offset - the
// CUDA's `get_int_b4(bq8_1[iqs].qs, ...)` reads the activation.  Both mistakes compile and both read a legal
// address: the device then computes a plausible, finite, wrong sum (measured: oracle 0.440796 against device
// -0.395386 for the first part of the first row, every row of the case outside tolerance).  It is the same trap
// as `f16_at` in s2_row_dot.glsl, one format later: a CUDA pointer-taking accessor becomes a buffer-bound
// byte-offset accessor in GLSL, and the buffer has to be part of the helper's NAME so the call site can be read.
int iq1m_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

// `__dp4a`, written out: its operands are reinterpreted as SIGNED bytes and accumulated in int32 (include/
// strata/kernels/dp4a.hpp says the same of the CUDA fallback, and it is bit-exact for these call sites). The
// shifts sign-extend because GLSL's right shift on a signed type is arithmetic.
int iq1m_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

float iq1m_dot(uint wblk, uint ablk, uint iqs) {
    int sumi[2];
    float sumf[2];
    sumi[0] = 0; sumi[1] = 0;
    sumf[0] = 0.0; sumf[1] = 0.0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int half_i = l0 / 4;
        // the grid index: 8 bits from qs, 3 high bits (and the shift bit, bit 3) from qh
        const uint qh = uint(w_b.b[wblk + 32u + uint(2 * iqs + l0 / 4)]);
        const uint qhl = qh >> uint(4 * ((l0 / 2) % 2));
        const uint g = grid_b.v[uint(w_b.b[wblk + 4u * iqs + uint(l0 / 2)]) | ((qhl & 7u) << 8u)];
        const int grid0 = int((g >> 0u) & 0x0F0F0F0Fu);
        const int grid1 = int((g >> 4u) & 0x0F0F0F0Fu);
        // the activation words: bytes 4*(l0) and 4*(l0)+4 of this part's q8_1 block
        const uint ub = ablk + 4u + 4u * uint(l0);
        const int u0 = iq1m_act_int4(ub);
        const int u1 = iq1m_act_int4(ub + 4u);
        sumi[half_i] = iq1m_dp4a(grid0, u0, sumi[half_i]);
        sumi[half_i] = iq1m_dp4a(grid1, u1, sumi[half_i]);
        // the -1 bias folded into a delta, with the qh shift bit replacing delta's sign
        const float delta = -1.0 + IQ1M_DELTA - float(qhl & 0x08u) * (2.0 * IQ1M_DELTA / 8.0);
        int sumy = 0;
        sumy = iq1m_dp4a(u0, 0x01010101, sumy);
        sumy = iq1m_dp4a(u1, 0x01010101, sumy);
        sumf[half_i] += delta * float(sumy);
    }
    // the block's fp16 scale: the high nibble of scales[1], [3], [5], [7], assembled low nibble first
    const uint sc0 = uint(w_b.b[wblk + 48u]) | (uint(w_b.b[wblk + 49u]) << 8u);
    const uint sc1 = uint(w_b.b[wblk + 50u]) | (uint(w_b.b[wblk + 51u]) << 8u);
    const uint sc2 = uint(w_b.b[wblk + 52u]) | (uint(w_b.b[wblk + 53u]) << 8u);
    const uint sc3 = uint(w_b.b[wblk + 54u]) | (uint(w_b.b[wblk + 55u]) << 8u);
    const uint scale_u16 = (sc0 >> 12u) | ((sc1 >> 8u) & 0x00F0u) | ((sc2 >> 4u) & 0x0F00u) | (sc3 & 0xF000u);
    const float dscale = unpackHalf2x16(scale_u16).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    // the part's own sub-scale pair, out of the same eight bytes
    const uint sci = uint(w_b.b[wblk + 48u + (iqs >> 1u)]) | (uint(w_b.b[wblk + 48u + (iqs >> 1u) + 1u]) << 8u);
    const uint tmp = sci >> (6u * (iqs % 2u));
    const int sc_a = 2 * int(tmp & 7u) + 1;
    const int sc_b = 2 * int((tmp >> 3u) & 7u) + 1;
    return (dscale * dact) * ((float(sumi[0]) + sumf[0]) * float(sc_a) + (float(sumi[1]) + sumf[1]) * float(sc_b));
}
