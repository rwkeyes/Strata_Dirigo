// ports/vulkan/shaders/common/iq2s_dot.glsl - ONE IQ2_S WEIGHT PART against ONE q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_iq2_s_q8_1` (src/kernels/cuda/iq_kernels.cu).  **IQ2_S IS THE RESIDENT
// `coder-iq1_m` MODEL'S MOST COMMON EXPERT GATE/UP FORMAT** (20 of its 48 layers, from the pack's own
// `native_experts.txt`; IQ3_XXS is 17, IQ3_S 10, IQ4_XS 1 - the "IQ1_M" in that model's name is the GGUF its
// dense weights come from, not its experts).
//
// block_iq2_s, 82 bytes per 256 values (third_party/ggml/ggml-common.h):
//     0..1    half d
//     2..65   qs[64]       low grid-index bits, packed 2 per byte, AND the sign bytes at +32
//     66..73  qh[8]        the grid index's high 2 bits, one byte per 2 parts
//     74..81  scales[8]    two 4-bit sub-scales per byte
//
// block_q8_1, 36 bytes per 32 values: half2 ds at 0 (d at 0, the SUM at 2 - THIS DOT READS d ONLY),
// int8 qs[32] at 4.
//
// THREE THINGS THE PORT HAS TO WRITE OUT, each of which the CUDA gets from an intrinsic:
//
//   1. **`get_int_b2` READS FOUR BYTES, not two.**  It is llama.cpp's misnomer (`uint16_t x16[2*i32] |
//      uint16_t x16[2*i32+1] << 16`), so `get_int_b2(qs, iqs/2)` is the 4-byte word at byte offset `2*iqs`, and
//      that is what gives the loop its `qs[0..3]`.  Reading it as 2 bytes would take the grid index from the
//      wrong bits and produce plausible garbage - the failure mode this port keeps meeting.
//   2. `__vcmpne4` / `__vsub4` are per-byte integer operations with no carry between bytes: the CUDA sign-flips a
//      grid word with `(g ^ s) - s` where each byte of `s` is 0xFF or 0x00.
//   3. The final combination uses TWO TRUNCATING INTEGER DIVISIONS:
//      `sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2) / 4`.  SPIR-V's signed division does not promise a
//      rounding direction, so the port implements C's truncation-toward-zero itself rather than inheriting
//      whatever the driver does - the operands can be negative.
//
// `iqs` (0, 2, 4 .. 14 - the format's step is 2) selects the part: the weight's qs word at `2*iqs`, the qh byte
// at `iqs/2`, the scales byte at `iqs/2`, and the activation block, which the caller has already indexed (part k
// of a row is activation block k, since one 256-value weight block covers eight 32-value q8_1 blocks).
//
// The including shader must declare, with these names:
//     w_b            the IQ2_S weight blocks (uint8 storage buffer, rows of `nb * 82` bytes)
//     act_b          the q8_1 activation blocks (uint8 storage buffer)
//     iq2s_grid_b    the IQ2_S grid (uint32 storage buffer, 1024 entries as low/high halves) - harness/iq_grids.hpp
int iq2s_weight_int4(uint off) {
    return int(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u) | (uint(w_b.b[off + 2u]) << 16u) |
               (uint(w_b.b[off + 3u]) << 24u));
}

int iq2s_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

// __dp4a: signed bytes, int32 accumulate (the same helper shape as the other I-quant dots in this port).
int iq2s_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

// __vcmpne4 / __vsub4 - the per-byte sign mask and the conditional negation - live in common/perbyte_sign.glsl,
// because IQ3_XXS writes the identical idiom. Do not merge anything else: the two q8_1 quantisers are the same
// quantity under two different rules and stay separate.
// ------------------------------------------------------------------------------------------------
// NOTE ON THE PATH: this file is inside shaders/common/, and glslc resolves an include RELATIVE TO THE
// INCLUDING FILE - so a sibling here is named bare ("perbyte_sign.glsl"), not "common/perbyte_sign.glsl".
// The full-relative form only works from shaders/*.comp, which is where the dotes are included from.
#include "perbyte_sign.glsl"

// C's signed division: truncate toward zero, with a positive divisor.  Written out because SPIR-V does not
// promise the driver's OpSDiv rounds that way, and these operands are negative about half the time.
int iq2s_div_trunc(int a, int d) {
    const int q = abs(a) / d;
    return (a < 0) ? -q : q;
}

float iq2s_dot(uint wblk, uint ablk, uint iqs) {
    const uint QS = wblk + 2u;                    // qs[64]
    const uint QH = wblk + 66u;                   // qh[8]
    const uint SC = wblk + 74u;                   // scales[8]
    const uint qs0 = uint(w_b.b[QS + 2u * iqs + 0u]);
    const uint qs1 = uint(w_b.b[QS + 2u * iqs + 1u]);
    const uint qs2 = uint(w_b.b[QS + 2u * iqs + 2u]);
    const uint qs3 = uint(w_b.b[QS + 2u * iqs + 3u]);
    const uint qh = uint(w_b.b[QH + iqs / 2u]);
    // the sign bytes: 4 bytes at qs + 32 + 2*iqs, read through the same 4-byte helper
    const int signs_packed = iq2s_weight_int4(QS + 32u + 2u * iqs);
    const uint sc_byte = uint(w_b.b[SC + iqs / 2u]);
    const int ls0 = int(sc_byte & 0x0Fu);
    const int ls1 = int(sc_byte >> 4u);
    int sumi0 = 0, sumi1 = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint qsv = (l0 == 0) ? qs0 : ((l0 == 2) ? qs1 : ((l0 == 4) ? qs2 : qs3));
        // the grid index: this part's byte, plus the 2 high bits qh carries for it
        const uint gidx = qsv | ((qh << uint(8 - l0)) & 0x300u);
        const int grid_lo = int(iq2s_grid_b.v[2u * gidx]);          // the low 32 bits of the grid point
        const int grid_hi = int(iq2s_grid_b.v[2u * gidx + 1u]);     // its high 32 bits
        const int sp = (signs_packed >> (8 * (l0 / 2))) & 0xFF;
        const int signs0 = perbyte_ne_zero(((sp & 0x03) << 7) | ((sp & 0x0C) << 21));
        const int signs1 = perbyte_ne_zero(((sp & 0x30) << 3) | ((sp & 0xC0) << 17));
        const int grid_l = perbyte_sign_flip(grid_lo, signs0);
        const int grid_h = perbyte_sign_flip(grid_hi, signs1);
        const uint ub = ablk + 4u + 4u * uint(l0);
        const int u0 = iq2s_act_int4(ub);
        const int u1 = iq2s_act_int4(ub + 4u);
        if (l0 < 4) {
            sumi0 = iq2s_dp4a(grid_l, u0, sumi0);
            sumi0 = iq2s_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = iq2s_dp4a(grid_l, u0, sumi1);
            sumi1 = iq2s_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = iq2s_div_trunc(sumi0 * ls0 + sumi1 * ls1 + iq2s_div_trunc(sumi0 + sumi1, 2), 4);
    const float dw = unpackHalf2x16(uint(w_b.b[wblk]) | (uint(w_b.b[wblk + 1u]) << 8u)).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    return (dw * dact) * float(sumi);
}
