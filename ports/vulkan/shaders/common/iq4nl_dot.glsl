// ports/vulkan/shaders/common/iq4nl_dot.glsl - ONE IQ4_NL WEIGHT PART against the q8_1 ACTIVATION BLOCK.
//
// Transcribed from `vec_dot_iq4_nl_q8_1` (src/kernels/cuda/iq_kernels.cu:271).  IQ4_NL (type 20) is the DOWN
// format of 39 of the resident model's 48 layers - the single most valuable format left in the wave, because no
// layer's expert path is complete until its down projection is too.
//
// block_iq4_nl, 18 bytes per 32 values (third_party/ggml/ggml-common.h):  half d at 0, uint8_t qs[16] at 2.
// block_q8_1: half2 ds at 0, int8 qs[32] at 4.
//
// This format is SIMPLER than its siblings in three ways, and each one is a place a copy of the others' shape
// would have been wrong:
//   1. THE BLOCK IS 32 VALUES, NOT 256.  `Fmt<20>` is { qk 32, ipb 2, step 2 }: two dot calls per block, 16 values
//      each.  So part k of a row is weight block k/2 and ACTIVATION BLOCK k/2 (not k - the caller indexes
//      `arow + (k/2)*36`), with `iqs` = 2*(k%2) in {0, 2}.
//   2. NO SUB-SCALE AND NO SIGN HANDRY: the value is a 4-bit codebook index and the block has ONE scale, so there
//      is no `ls`, no per-part scale, and no integer division in the tail.  `d * sumi` is the whole result.
//   3. THE NIBBLE PACKING IS DE-INTERLEAVED, and this is the one that decides whether the dot is right at all:
//      the LOW nibble of byte j is value j, the HIGH nibble is value j+16 - not the adjacent-pair packing that
//      Q4_0/Q4_K use.  That is why the CUDA reads the activation at `q8[l]` and `q8[l+4]` (the low-nibble values
//      and the high-nibble values) rather than at consecutive positions.
//
// WHERE THE PACKING ORDER COMES FROM - deliberately NOT from the CUDA.  `get_int_from_table_16` reaches its
// answer through a chain of `__byte_perm` selections (and, on the AMD branch, `__builtin_amdgcn_perm`), which
// states the ORDER only implicitly.  The engine's own CPU AVX-2 kernel says it outright:
//
//     src/kernels/cpu/iq_avx2.cpp:  const __m128i lo = _mm_and_si128(bits, m4b);      // values 0..15
//                                   const __m128i hi = ..._mm_srli_epi16(bits, 4);... // values 16..31
//
// Two independent implementations in the engine agree, one of them in a comment, so the port follows that and the
// case's oracle is written from it.  `kvalues_iq4nl` is the same 16-byte constant `sform_decode.glsl` carries for
// the S-family's Iq4Nl codebook - the same source constant, but TWO DIFFERENT DECODES (that one is the canonical
// form's per-value decode, this one is the mmvq nibble table), so they stay separate.
//
// The including shader must declare, with these names:
//     w_b      the IQ4_NL weight blocks (uint8 storage buffer, rows of `nb * 18` bytes)
//     act_b    the q8_1 activation blocks (uint8 storage buffer)
int iq4nl_weight_int4(uint off) {
    return int(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u) | (uint(w_b.b[off + 2u]) << 16u) |
               (uint(w_b.b[off + 3u]) << 24u));
}

int iq4nl_act_int4(uint off) {
    return int(uint(act_b.b[off]) | (uint(act_b.b[off + 1u]) << 8u) | (uint(act_b.b[off + 2u]) << 16u) |
               (uint(act_b.b[off + 3u]) << 24u));
}

// __dp4a: signed bytes, int32 accumulate.  The codebook holds negative values, so the sign-extension matters.
int iq4nl_dp4a(int a, int b, int c) {
    return c + ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24) +
           ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

// kvalues_iq4nl, verbatim from ggml-common.h.
int iq4nl_value(uint nib) {
    const int tbl[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10,
                                1, 13, 25, 38, 53, 69, 89, 113);
    return tbl[nib & 0x0Fu];
}

// four codebook values as four SIGNED bytes in one int32 (`& 0xFF` because the values are negative)
int iq4nl_pack4(int b0, int b1, int b2, int b3) {
    return ((b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24));
}

float iq4nl_dot(uint wblk, uint ablk, uint iqs) {
    int sumi = 0;
    for (int l = 0; l < 2; ++l) {
        const uint aux = uint(iq4nl_weight_int4(wblk + 2u + 4u * (iqs + uint(l))));
        // the low nibble of each byte: values 4l+0..3 of this call's 16
        const int vx = iq4nl_pack4(iq4nl_value(aux & 0x0Fu), iq4nl_value((aux >> 8u) & 0x0Fu),
                                   iq4nl_value((aux >> 16u) & 0x0Fu), iq4nl_value((aux >> 24u) & 0x0Fu));
        // the high nibble of each byte: values 4l+16..19 - which is why the activation is read at +16 bytes
        const int vy = iq4nl_pack4(iq4nl_value((aux >> 4u) & 0x0Fu), iq4nl_value((aux >> 12u) & 0x0Fu),
                                   iq4nl_value((aux >> 20u) & 0x0Fu), iq4nl_value((aux >> 28u) & 0x0Fu));
        const uint aq = ablk + 4u + 4u * (iqs + uint(l));
        sumi = iq4nl_dp4a(vx, iq4nl_act_int4(aq), sumi);
        sumi = iq4nl_dp4a(vy, iq4nl_act_int4(aq + 16u), sumi);
    }
    const float dw = unpackHalf2x16(uint(w_b.b[wblk]) | (uint(w_b.b[wblk + 1u]) << 8u)).x;
    const float dact = unpackHalf2x16(uint(act_b.b[ablk]) | (uint(act_b.b[ablk + 1u]) << 8u)).x;
    return (dw * dact) * float(sumi);
}
