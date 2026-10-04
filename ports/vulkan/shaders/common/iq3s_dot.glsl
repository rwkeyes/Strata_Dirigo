// ports/vulkan/shaders/common/iq3s_dot.glsl - ONE IQ3_S WEIGHT PART against ONE q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_iq3_s_q8_1` (src/kernels/cuda/iq_kernels.cu:211).  IQ3_S (type 21) is the gate/up
// format of 10 of the resident model's 48 layers - the last gate/up format after IQ4_XS's single layer.
//
// block_iq3_s, 110 bytes per 256 values (third_party/ggml/ggml-common.h):
//     0..1      half d
//     2..65     qs[64]      grid indices, low 8 bits, one byte per value
//     66..73    qh[8]       the index's 9th bit, one byte per two parts
//     74..105   signs[32]   sign bits, four per byte
//     106..109  scales[4]   IQ3S_N_SCALE = QK_K/64 = 4 bytes, two nibble-scales per part-group
//
// `Fmt<21>` is { qk 256, ipb 8, step 2 }, so `iqs` = 2*(k%8) and part k of a row is weight block k/8 and
// activation block k, exactly as IQ2_S and IQ3_XXS are.
//
// THE ONE STRUCTURAL NOVELTY against IQ3_XXS (its closest relative) is WHERE THE SIGN BITS COME FROM: IQ3_XXS
// packs them into the top bits of a per-part aux word, IQ3_S keeps a separate `signs` array of four sign bits per
// byte, addressed at `2*iqs` - and the two halves of a byte feed the two grid words separately.  The grid index is
// NINE bits (512 entries) and the 9th bit arrives at a DIFFERENT SHIFT for the two words of a pair
// (`8 - l0` for the first, `7 - l0` for the second), which is easy to "tidy" into one shift and get wrong.
//
// The sub-scale is an integer multiply with no division: `sumi *= 1 + 2*((scales[iqs/4] >> ((iqs << 1) & 4)) & 15)`.
//
// The including shader must declare, with these names:
//     w_b           the IQ3_S weight blocks (uint8 storage buffer, rows of `nb * 110` bytes)
//     act_b         the q8_1 activation blocks (uint8 storage buffer)
//     iq3s_grid_b   the IQ3_S grid (uint32 storage buffer, 512 entries) - harness/iq_grids.hpp
#include "perbyte_sign.glsl"

int iq3s_weight_int4(uint off) {
    return int(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u) | (uint(w_b.b[off + 2u]) << 16u) |
               (uint(w_b.b[off + 3u]) << 24u));
}

int iq3s_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

int iq3s_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

float iq3s_dot(uint wblk, uint ablk, uint iqs) {
    const uint QS = wblk + 2u;                     // qs[64]
    const uint QH = wblk + 66u;                    // qh[8]
    const uint SG = wblk + 74u;                    // signs[32]
    const uint SC = wblk + 106u;                   // scales[4]
    const uint o = QS + 4u * iqs;                  // this part's eight grid-index bytes
    const uint cl = uint(w_b.b[o + 0u]), ch = uint(w_b.b[o + 1u]);
    const uint dl = uint(w_b.b[o + 2u]), dh = uint(w_b.b[o + 3u]);
    const uint el = uint(w_b.b[o + 4u]), eh = uint(w_b.b[o + 5u]);
    const uint fl = uint(w_b.b[o + 6u]), fh = uint(w_b.b[o + 7u]);
    const uint qh = uint(w_b.b[QH + iqs / 2u]);
    const int sp0 = int(iq3s_weight_int4(SG + 2u * iqs));       // four sign bytes, one per l0/2 step
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint lo_b = (l0 == 0) ? cl : ((l0 == 2) ? dl : ((l0 == 4) ? el : fl));
        const uint hi_b = (l0 == 0) ? ch : ((l0 == 2) ? dh : ((l0 == 4) ? eh : fh));
        // the 9th index bit: `8 - l0` for the first word of the pair, `7 - l0` for the second
        const int gx = int(iq3s_grid_b.v[lo_b | ((qh << uint(8 - l0)) & 0x100u)]);
        const int gy = int(iq3s_grid_b.v[hi_b | ((qh << uint(7 - l0)) & 0x100u)]);
        const int sp = (sp0 >> (8 * (l0 / 2))) & 0xFF;
        const int signs0 = perbyte_ne_zero(((sp & 0x03) << 7) | ((sp & 0x0C) << 21));
        const int signs1 = perbyte_ne_zero(((sp & 0x30) << 3) | ((sp & 0xC0) << 17));
        const int grid_l = perbyte_sign_flip(gx, signs0);
        const int grid_h = perbyte_sign_flip(gy, signs1);
        const uint aq = ablk + 4u + 4u * uint(l0);
        sumi = iq3s_dp4a(grid_l, iq3s_act_int4(aq), sumi);
        sumi = iq3s_dp4a(grid_h, iq3s_act_int4(aq + 4u), sumi);
    }
    const uint sc_byte = uint(w_b.b[SC + iqs / 4u]);
    sumi *= int(1u + 2u * ((sc_byte >> ((iqs << 1u) & 0x04u)) & 0x0Fu));
    const float dw = unpackHalf2x16(uint(w_b.b[wblk]) | (uint(w_b.b[wblk + 1u]) << 8u)).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    return (dw * dact) * float(sumi);
}
