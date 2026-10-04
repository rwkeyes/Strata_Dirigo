// ports/vulkan/shaders/common/iq3xxs_dot.glsl - ONE IQ3_XXS WEIGHT PART against ONE q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_iq3_xxs_q8_1` (src/kernels/cuda/iq_kernels.cu:185).  IQ3_XXS (type 18) is the resident
// model's SECOND most common expert gate/up format - 17 of its 48 layers, behind IQ2_S's 20.
//
// block_iq3_xxs, 98 bytes per 256 values (third_party/ggml/ggml-common.h):
//     0..1    half d
//     2..97   qs[96] = 64 bytes of grid indices (8 per part) + 32 bytes of aux words (4 per part)
//
// block_q8_1, 36 bytes per 32 values: half2 ds at 0 (d at 0, the SUM at 2 - THIS DOT READS d ONLY), int8 qs[32]
// at 4.  The activation block index for part k of a row is k: one 256-value weight block covers eight 32-value
// q8_1 blocks, so the caller indexes `arow + k*36` and this helper never sees `kbx`.
//
// HOW THE PART'S PIECES ARE ADDRESSED (the format's `step` is 2, so `iqs` is even, 0..14):
//     * the eight grid-index BYTES are the 8 bytes at `qs + 4*iqs` - the CUDA reads them as two `get_int_b2`
//       words (`4*iqs` and `4*(iqs+1)`) and reinterprets the pair as bytes, which is the same eight bytes;
//     * the 4-byte `aux32` word is at `qs + 64 + 2*iqs` (the CUDA's `get_int_b2(qs, QK_K/16 + iqs/2)` with
//       QK_K/16 = 16: FOUR bytes at `4*(16 + iqs/2)`);
//     * the top nibble of `aux32` is the part's sub-scale `ls`; the bits below it carry the signs, SEVEN PER STEP
//       (`aux32 >> (7*l0/2)`, truncated to its low byte).
//
// TWO STEPS THAT ARE NOT THE FORMULA THEY LOOK LIKE:
//   1. `unpack_ksigns(v)`: the sign byte is made EVEN-parity by `v ^ ((popcount(v) & 1) << 7)` and then replicated
//      into all four bytes.  It is arithmetic, not a table (`ksigns_iq2xs` in the engine's header is the CPU-side
//      equivalent), so nothing extra is uploaded.  `bitCount` is core GLSL.
//   2. The final combination has TWO TRUNCATING INTEGER DIVISIONS - `(ls * sumi + sumi/2) / 2` - and `sumi` is
//      negative about half the time, so the port implements C's truncation-toward-zero rather than inheriting
//      SPIR-V's unspecified signed division.
//
// The including shader must declare, with these names:
//     w_b              the IQ3_XXS weight blocks (uint8 storage buffer, rows of `nb * 98` bytes)
//     act_b            the q8_1 activation blocks (uint8 storage buffer)
//     iq3xxs_grid_b    the IQ3_XXS grid (uint32 storage buffer, 256 entries) - harness/iq_grids.hpp
// The path is bare because this file sits beside it in shaders/common/ and glslc resolves includes relative to the
// INCLUDING file (a sibling from inside common/ is "perbyte_sign.glsl"; only shaders/*.comp say "common/...").
#include "perbyte_sign.glsl"

int iq3xxs_weight_int4(uint off) {
    return int(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u) | (uint(w_b.b[off + 2u]) << 16u) |
               (uint(w_b.b[off + 3u]) << 24u));
}

int iq3xxs_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

// __dp4a: signed bytes, int32 accumulate.
int iq3xxs_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

// C's signed division, truncating toward zero, with a positive divisor.  SPIR-V does not promise this.
int iq3xxs_div_trunc(int a, int d) {
    const int q = abs(a) / d;
    return (a < 0) ? -q : q;
}

// `unpack_ksigns`: force the sign byte to even parity, then replicate it into all four bytes.  The parity fix
// matters because the four sign bits are read out of bit 7 below - an odd byte would flip the mask's meaning.
uint iq3xxs_unpack_ksigns(uint v) {
    const uint parity = bitCount(v) & 1u;
    return (v ^ (parity << 7u)) * 0x01010101u;
}

float iq3xxs_dot(uint wblk, uint ablk, uint iqs) {
    const uint QS = wblk + 2u;                     // qs[96]
    const uint o = QS + 4u * iqs;                  // this part's eight grid-index bytes
    const uint q3_0 = uint(w_b.b[o + 0u]), q3_1 = uint(w_b.b[o + 1u]);
    const uint q3_2 = uint(w_b.b[o + 2u]), q3_3 = uint(w_b.b[o + 3u]);
    const uint q3_4 = uint(w_b.b[o + 4u]), q3_5 = uint(w_b.b[o + 5u]);
    const uint q3_6 = uint(w_b.b[o + 6u]), q3_7 = uint(w_b.b[o + 7u]);
    const uint aux32 = uint(iq3xxs_weight_int4(QS + 64u + 2u * iqs));
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint q3l = (l0 == 0) ? q3_0 : ((l0 == 2) ? q3_2 : ((l0 == 4) ? q3_4 : q3_6));
        const uint q3h = (l0 == 0) ? q3_1 : ((l0 == 2) ? q3_3 : ((l0 == 4) ? q3_5 : q3_7));
        const int gx = int(iq3xxs_grid_b.v[q3l]);
        const int gy = int(iq3xxs_grid_b.v[q3h]);
        const uint ks = iq3xxs_unpack_ksigns((aux32 >> uint(7 * l0 / 2)) & 0xFFu);
        const int signs0 = perbyte_ne_zero(int(ks) & 0x08040201);
        const int signs1 = perbyte_ne_zero(int(ks) & 0x80402010);
        const int grid_l = perbyte_sign_flip(gx, signs0);
        const int grid_h = perbyte_sign_flip(gy, signs1);
        const uint ub = ablk + 4u + 4u * uint(l0);
        const int u0 = iq3xxs_act_int4(ub);
        const int u1 = iq3xxs_act_int4(ub + 4u);
        sumi = iq3xxs_dp4a(grid_l, u0, sumi);
        sumi = iq3xxs_dp4a(grid_h, u1, sumi);
    }
    const int ls = int(aux32 >> 28u);
    sumi = iq3xxs_div_trunc(ls * sumi + iq3xxs_div_trunc(sumi, 2), 2);
    const float dw = unpackHalf2x16(uint(w_b.b[wblk]) | (uint(w_b.b[wblk + 1u]) << 8u)).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    return (dw * dact) * float(sumi);
}
