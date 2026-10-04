// ports/vulkan/shaders/common/q2_0_dot.glsl - ONE Q2_0 WEIGHT PART against ONE q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_q2_0_q8_1` (src/kernels/cuda/iq_kernels.cu:72).  Q2_0 (type 42) is the DOWN format of
// 9 of the resident model's 48 layers; with IQ4_NL it finishes the down side of every layer.
//
// block_q2_0, 18 bytes per 64 values: ggml_half d at 0, uint8_t qs[16] at 2 (2 bits per value).
// `Fmt<42>` is { qk 64, ipb 2, step 1 }: the block is 64 values, so it spans TWO q8_1 blocks and takes two
// 32-value parts, `iqs` in {0, 1}.  Part k of a row is weight block k/2 and ACTIVATION BLOCK k - the source's
// `x + kbx*(qk/32)` is an offset of TWO q8_1 blocks per weight block, so the pair (kbx, iqs) lands on block k.
//
// THE FOUR `__byte_perm`s ARE A BYTE TABLE LOOKUP.  `__byte_perm(0x020100FF, 0x020100FF, q)` builds an 8-byte
// table whose first four bytes are 0xFF, 0x00, 0x01, 0x02 (the same four repeated as the second operand, so the
// selector's high bit cannot change the answer) and indexes it with the 2-bit codes packed in `q`.  The values are
// therefore the SIGNED bytes -1, 0, 1, 2 for codes 0..3 - which is Q2_0's codebook.  Written out as a table here,
// with the two selection steps (`qe`/`qo` then `qx`/`qy`) kept as the byte gathers they are, because that order is
// what decides which activation byte multiplies which weight value.
//
// The including shader must declare, with these names:
//     w_b      the Q2_0 weight blocks (uint8 storage buffer, rows of `nb * 18` bytes)
//     act_b    the q8_1 activation blocks (uint8 storage buffer)
int q2_0_weight_int2(uint off) {
    return int(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u));
}

int q2_0_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

int q2_0_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

// the perm's table: code 0 -> -1, 1 -> 0, 2 -> 1, 3 -> 2 (as bytes)
int q2_0_code_byte(uint code) {
    const int tbl[4] = int[4](0xFF, 0x00, 0x01, 0x02);
    return tbl[code & 0x03u];
}

int q2_0_pack4(int b0, int b1, int b2, int b3) {
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

// The four 2-bit codes of ONE byte, in field order: this is what the CUDA's `__byte_perm(0x5140)` and
// `__byte_perm(0x7362)` gather resolves to, and it is worth spelling out why.  `__byte_perm` uses only the low
// THREE bits of each selector nibble, and the selector nibbles here are the weight byte spliced with its
// neighbour (`q` is a 16-bit read of two bytes), so the high bit of each 2-bit code selects [the second copy of
// the table] - which the source makes identical to the first (`0x020100FF` in both operands) exactly so that it
// cannot matter.  Working the chain through: `qe` = the even-indexed fields of both bytes, `qo` = the
// odd-indexed ones, then `0x5140` picks byte0 of the first and second words (fields 0 and 1 of byte 0) and
// byte1 of each (fields 2 and 3 of byte 0) - the four fields of the FIRST byte in order - and `0x7362` does the
// same for the second byte.  So the pairing is simply "the codes of byte j against the 4 activations at 8j", and
// the first version of this file had it as an interleave, which the case's second oracle caught.
int q2_0_codes_of_byte(uint byte) {
    return q2_0_pack4(q2_0_code_byte(byte), q2_0_code_byte(byte >> 2u), q2_0_code_byte(byte >> 4u),
                      q2_0_code_byte(byte >> 6u));
}

float q2_0_dot(uint wblk, uint ablk, uint iqs) {
    const uint QS = wblk + 2u;
    const uint off = QS + 8u * iqs;               // the part's 8 weight bytes
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const uint w0 = uint(w_b.b[off + 2u * uint(j)]);
        const uint w1 = uint(w_b.b[off + 2u * uint(j) + 1u]);
        const int qx = q2_0_codes_of_byte(w0);    // the four values packed in the first byte
        const int qy = q2_0_codes_of_byte(w1);    // and in the second
        const uint aq = ablk + 4u + 8u * uint(j);
        const int u = q2_0_act_int4(aq);
        const int v = q2_0_act_int4(aq + 4u);
        sumi = q2_0_dp4a(u, qx, sumi);
        sumi = q2_0_dp4a(v, qy, sumi);
    }
    const float dw = unpackHalf2x16(uint(w_b.b[wblk]) | (uint(w_b.b[wblk + 1u]) << 8u)).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    return (dw * dact) * float(sumi);
}
