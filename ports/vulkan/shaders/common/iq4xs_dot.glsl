// ports/vulkan/shaders/common/iq4xs_dot.glsl - ONE IQ4_XS WEIGHT PART against ONE q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_iq4_xs_q8_1` (src/kernels/cuda/iq_kernels.cu:290).  IQ4_XS (type 23) is the gate/up
// format of exactly ONE of the resident model's 48 layers - but it is one of the three formats left that keep any
// layer from being complete, so it is small and it is worth having.
//
// block_iq4_xs, 136 bytes per 256 values: half d at 0, uint16_t scales_h at 2, uint8_t scales_l[4] at 4,
// uint8_t qs[128] at 8 (two 4-bit values per byte, 32 values per 16 bytes).
//
// `Fmt<23>` is { qk 256, ipb 8, step 4 }: eight parts per 256-value block with a step of FOUR, so `iqs` is
// 0, 4, 8 .. 28 and each part covers 32 values (four `j` steps of 8).  Part k of a row is weight block k/8 and
// activation block k (`iqs / 4` inside the block is `k % 8`), as in every other 256-value format here.
//
// IT USES THE SAME NIBBLE MACHINERY AS IQ4_NL, whose packing order the engine's CPU AVX-2 kernel states outright
// (`src/kernels/cpu/iq_avx2.cpp`: "values 0..15" / "values 16..31" - the de-interleaved packing, low nibble of
// byte j is value j, high nibble is value j+16).  The two dots are structured identically: `v.x` (the low nibbles
// of the four bytes) multiplies the activation at `+0`, `v.y` (the high nibbles) the one at `+16`.  So the
// codebook and the packing are shared with `iq4nl_dot.glsl` in spirit but the OFFSETS ARE NOT - this format
// indexes the activation at `4j` and `4j+16` inside the block, with `iqs` stepping by 4, where IQ4_NL steps by 2
// inside an 18-byte block.  Two formats, one rule, different addressing: the rule is written out in both rather
// than factored into a helper that would hide which one a call site is in.
//
// The sub-scale is not a nibble-of-a-byte like IQ3_S's: it is split across two arrays -
// `((scales_l[iqs/8] >> (iqs & 4)) & 15) | (((scales_h >> (iqs/2)) & 3) << 4)` - and then offset by 32.
//
// The including shader must declare, with these names:
//     w_b      the IQ4_XS weight blocks (uint8 storage buffer, rows of `nb * 136` bytes)
//     act_b    the q8_1 activation blocks (uint8 storage buffer)
int iq4xs_weight_int4(uint off) {
    return int(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u) | (uint(w_b.b[off + 2u]) << 16u) |
               (uint(w_b.b[off + 3u]) << 24u));
}

int iq4xs_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

int iq4xs_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

// kvalues_iq4nl, verbatim from ggml-common.h (the same table as iq4nl_dot.glsl's).
int iq4xs_value(uint nib) {
    const int tbl[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10,
                                1, 13, 25, 38, 53, 69, 89, 113);
    return tbl[nib & 0x0Fu];
}

int iq4xs_pack4(int b0, int b1, int b2, int b3) {
    return ((b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24));
}

float iq4xs_dot(uint wblk, uint ablk, uint iqs) {
    const uint QS = wblk + 8u;                     // qs[128]
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const uint aux = uint(iq4xs_weight_int4(QS + 4u * (iqs + uint(j))));
        const int vx = iq4xs_pack4(iq4xs_value(aux & 0x0Fu), iq4xs_value((aux >> 8u) & 0x0Fu),
                                   iq4xs_value((aux >> 16u) & 0x0Fu), iq4xs_value((aux >> 24u) & 0x0Fu));
        const int vy = iq4xs_pack4(iq4xs_value((aux >> 4u) & 0x0Fu), iq4xs_value((aux >> 12u) & 0x0Fu),
                                   iq4xs_value((aux >> 20u) & 0x0Fu), iq4xs_value((aux >> 28u) & 0x0Fu));
        const uint aq = ablk + 4u + 4u * uint(j);
        sumi = iq4xs_dp4a(vx, iq4xs_act_int4(aq), sumi);
        sumi = iq4xs_dp4a(vy, iq4xs_act_int4(aq + 16u), sumi);
    }
    const uint sl = uint(w_b.b[wblk + 4u + iqs / 8u]);
    const uint sh = uint(w_b.b[wblk + 2u]) | (uint(w_b.b[wblk + 3u]) << 8u);
    const int ls = int(((sl >> (iqs & 0x04u)) & 0x0Fu) | (((sh >> (iqs / 2u)) & 0x03u) << 4u));
    sumi *= (ls - 32);
    const float dw = unpackHalf2x16(uint(w_b.b[wblk]) | (uint(w_b.b[wblk + 1u]) << 8u)).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    return (dw * dact) * float(sumi);
}
