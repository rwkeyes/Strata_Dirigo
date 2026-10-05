// ports/vulkan/shaders/common/k_dots.glsl - the K-quant (Q4_K / Q5_K / Q6_K) x q8_1 dots, ONE definition.
//
// WHY THIS IS A SHARED INCLUDE RATHER THAN THREE SHADERS (or eleven).  The dense projections a `--native`
// pack serves from its GGUF shard are, in `coder-iq1_m`, Q6_K x128, Q4_K x47 and Q5_K x35 - 210 of the 300
// eligible dense tensors.  The three dots differ in how the block's 4/5/6-bit codes and its scale (and, for
// Q4_K/Q5_K, min) are unpacked, but the surrounding MMVQ shape is identical: one workgroup per output row, a
// strided walk over (block, part), the workgroup barrier-tree sum, `y[c*n_out + row]`.  The port ALREADY had
// the Q5_K dot - `native_q5_k_f32.comp` (the native HEAD's matvec, from `src/core/native_head.cpp:78`) - so
// the honest move is to lift that dot out HERE and let both the head's per-type shader and the new generic
// dense MMVQ shader (`native_k_mmvq.comp`) include it.  The Q4_K and Q6_K dots are transcribed, term for
// term, from the engine's own `q4_q8_dot` / `q6_q8_dot` (`src/kernels/cuda/native_mmvq.cu:544` / `:638`) and
// their `*_impl` bodies - the same source the CUDA the engine runs on is built from.
//
// THE ONE NON-OBVIOUS ARITHMETIC.  Q6_K's per-16 scales are SIGNED and its codes are centred by
// `__vsubss4(v, 0x20202020)`.  `__vsubss4` is a per-BYTE saturating signed subtract; the operands here are
// bytes in 0..63, so 0..63 - 32 lands in -32..31 and the saturation NEVER fires - but it is still a PER-BYTE
// subtract, so a plain 32-bit `vi - 0x20202020` would BORROW across a byte boundary and be silently wrong.
// The loop below does it per byte, which is the source's semantics exactly.
//
// THE INCLUDING SHADER MUST DECLARE (the dots read these directly, the way the port's other shared dots do):
//   layout(set = 0, binding = 0, std430) readonly buffer W { uint8_t b[]; } w_b;   // the weight rows
//   layout(set = 0, binding = 1, std430) readonly buffer A { uint8_t b[]; } a_b;   // q8_1 activation blocks
// and require GL_EXT_shader_explicit_arithmetic_types_int8 + GL_EXT_shader_8bit_storage.

const uint KD_Q4K_BLOCK = 144u;   // dm(4) + scales(12) + qs(128)
const uint KD_Q5K_BLOCK = 176u;   // dm(4) + scales(12) + qh(32) + qs(128)
const uint KD_Q6K_BLOCK = 210u;   // ql(128) + qh(64) + scales(16, int8) + d(2, half)
const uint KD_Q8_1_BLOCK = 36u;   // half2 ds(4) + int8 qs[32]

// ---- the shared byte readers (little-endian, bit-preserving) -------------------------------------------------
uint q5_u8(uint off) { return uint(w_b.b[off]); }
uint q5_u16(uint off) { return q5_u8(off) | (q5_u8(off + 1u) << 8u); }
// a 4-byte integer read from the WEIGHT buffer, little-endian, bit-preserving (the source's `*(const int*)`)
int q5_i32(uint off) {
    const uint u = q5_u8(off) | (q5_u8(off + 1u) << 8u) | (q5_u8(off + 2u) << 16u) | (q5_u8(off + 3u) << 24u);
    return int(u);
}
uint kd_wu32(uint off) {
    return q5_u8(off) | (q5_u8(off + 1u) << 8u) | (q5_u8(off + 2u) << 16u) | (q5_u8(off + 3u) << 24u);
}
// the same read from the ACTIVATION buffer - a separate helper because the CUDA's `u[]` is int8 read out of
// `bq8i->qs` (the activation), NOT the weight block.  Collapsing the two is the `f16_at` defect's exact class.
int q5_a_i32(uint off) {
    const uint u = uint(a_b.b[off]) | (uint(a_b.b[off + 1u]) << 8u) | (uint(a_b.b[off + 2u]) << 16u) |
                   (uint(a_b.b[off + 3u]) << 24u);
    return int(u);
}
// fp16 reads: the weight buffer's half at a byte offset, and the activation's.
float kd_whalf(uint off) { return unpackHalf2x16(uint(w_b.b[off]) | (uint(w_b.b[off + 1u]) << 8u)).x; }
float kd_ahalf(uint off) { return unpackHalf2x16(uint(a_b.b[off]) | (uint(a_b.b[off + 1u]) << 8u)).x; }

// four int8 multiply-accumulates, the source's `__dp4a` (exact, so identical)
int q5_sb(int a, uint k) {   // signed byte k of a packed int (explicit sign-extend)
    const int v = (a >> int(8u * k)) & 0xFF;
    return (v << 24) >> 24;
}
int q5_dp4a(int a, int b, int c) {
    int r = c;
    for (uint k = 0u; k < 4u; ++k) {
        r += q5_sb(a, k) * q5_sb(b, k);
    }
    return r;
}

// ---- Q5_K: the packed (scale, min) pair.  `native_q5_k_f32.spv`'s dot, lifted here verbatim -----------------
// ONE (Q5_K block, part) item.  `blk` is the block's byte base, `abase` the block's first q8_1 byte base,
// `p` = `iqs/2` in 0..15.  `bq8_offset = 2*(p/4)`, `nib = p%4`, exactly the engine's `kqs = VDR*(tid%(QI/VDR))`.
float q5_q8_dot(uint blk, uint abase, uint p) {
    const uint bq8_offset = 2u * (p / 4u);
    const uint nib = p % 4u;

    const uint ql = blk + 48u + 16u * bq8_offset + 4u * nib;   // (int*)(qs + 16*bq8_offset + 4*((iqs/2)%4))
    const uint qh = blk + 16u + 4u * nib;                      // (int*)(qh + 4*((iqs/2)%4))
    const int vl0 = q5_i32(ql);
    const int vl1 = q5_i32(ql + 16u);
    const int vh0 = q5_i32(qh) >> int(bq8_offset);
    const int vh1 = q5_i32(qh + 16u) >> int(bq8_offset);

    const int j = int(bq8_offset) / 2;
    const int jm = j & 1;
    const uint sbase = blk + 4u + 2u * uint(jm);
    const uint s0 = q5_u16(sbase);
    const uint s2 = q5_u16(sbase + 4u);
    const uint s4 = q5_u16(sbase + 8u);
    const uint him = (j >= 2) ? 0xFFFFFFFFu : 0u;
    const uint aux0 = ((s0 & 0x3f3fu) & ~him) | ((((s4 >> 0u) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2u)) & him);
    const uint aux1 = ((s2 & 0x3f3fu) & ~him) | ((((s4 >> 4u) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2u)) & him);
    const int sc0 = int(aux0 & 0xFFu), sc1 = int((aux0 >> 8u) & 0xFFu);   // sc[0], sc[1]
    const int m0 = int(aux1 & 0xFFu), m1 = int((aux1 >> 8u) & 0xFFu);     // m[0], m[1]

    float sumf_d = 0.0, sumf_m = 0.0;
    for (uint i = 0u; i < 2u; ++i) {
        const uint b8 = abase + (bq8_offset + i) * KD_Q8_1_BLOCK;
        const float d8 = kd_ahalf(b8);
        const int ua = q5_a_i32(b8 + 4u + 4u * nib);            // ((int*)qs)[nib]     == u[2i]
        const int ub = q5_a_i32(b8 + 4u + 4u * nib + 16u);      // ((int*)qs)[nib + 4] == u[2i+1]
        const int v0 = ((vl0 >> int(4u * i)) & 0x0f0f0f0f) | (((vh0 >> int(i)) << 4) & 0x10101010);
        const int v1 = ((vl1 >> int(4u * i)) & 0x0f0f0f0f) | (((vh1 >> int(i)) << 4) & 0x10101010);
        const int dot1 = q5_dp4a(v0, ua, q5_dp4a(v1, ub, 0));
        const int dot2 = q5_dp4a(0x01010101, ua, q5_dp4a(0x01010101, ub, 0));
        const int scv = (i == 0u) ? sc0 : sc1;
        const int mv = (i == 0u) ? m0 : m1;
        sumf_d += d8 * float(dot1 * scv);
        sumf_m += d8 * float(dot2 * mv);
    }
    const vec2 dm5 = unpackHalf2x16(q5_u16(blk) | (q5_u16(blk + 2u) << 16u));   // half2 dm = (d, min)
    return dm5.x * sumf_d - dm5.y * sumf_m;
}

// ---- Q4_K: same packed (scale, min), no `qh` (4 bits per code) -----------------------------------------------
// `q4_q8_dot` (native_mmvq.cu:544): `bq8_offset = 2*(p/4)`, the two `qs` ints at `4*nib` and `+16`, the same
// `aux`/`hi` unpack, and the same `dm.x*sumf_d - dm.y*sumf_m`.
float q4_q8_dot(uint blk, uint abase, uint p) {
    const uint bq8_offset = 2u * (p / 4u);
    const uint nib = p % 4u;

    const uint ql = blk + 16u + 16u * bq8_offset + 4u * nib;   // bq4->qs (offset 16) + the source's expression
    const int v0 = q5_i32(ql);
    const int v1 = q5_i32(ql + 16u);

    const int j = int(bq8_offset) / 2;
    const int jm = j & 1;
    const uint sbase = blk + 4u + 2u * uint(jm);
    const uint s0 = q5_u16(sbase);
    const uint s2 = q5_u16(sbase + 4u);
    const uint s4 = q5_u16(sbase + 8u);
    const uint him = (j >= 2) ? 0xFFFFFFFFu : 0u;
    const uint aux0 = ((s0 & 0x3f3fu) & ~him) | ((((s4 >> 0u) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2u)) & him);
    const uint aux1 = ((s2 & 0x3f3fu) & ~him) | ((((s4 >> 4u) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2u)) & him);
    const int sc0 = int(aux0 & 0xFFu), sc1 = int((aux0 >> 8u) & 0xFFu);
    const int m0 = int(aux1 & 0xFFu), m1 = int((aux1 >> 8u) & 0xFFu);

    float sumf_d = 0.0, sumf_m = 0.0;
    for (uint i = 0u; i < 2u; ++i) {
        const uint b8 = abase + (bq8_offset + i) * KD_Q8_1_BLOCK;
        const float d8 = kd_ahalf(b8);
        const int u0 = q5_a_i32(b8 + 4u + 4u * nib);
        const int u1 = q5_a_i32(b8 + 4u + 4u * nib + 16u);
        const int v0i = (v0 >> int(4u * i)) & 0x0f0f0f0f;
        const int v1i = (v1 >> int(4u * i)) & 0x0f0f0f0f;
        const int dot1 = q5_dp4a(v1i, u1, q5_dp4a(v0i, u0, 0));
        const int dot2 = q5_dp4a(0x01010101, u1, q5_dp4a(0x01010101, u0, 0));
        const int scv = (i == 0u) ? sc0 : sc1;
        const int mv = (i == 0u) ? m0 : m1;
        sumf_d += d8 * float(dot1 * scv);
        sumf_m += d8 * float(dot2 * mv);
    }
    const vec2 dm4 = unpackHalf2x16(q5_u16(blk) | (q5_u16(blk + 2u) << 16u));   // half2 dm = (d, min)
    return dm4.x * sumf_d - dm4.y * sumf_m;
}

// ---- Q6_K: SIGNED per-16 scales, 6-bit codes, NO min, a single `d` half --------------------------------------
// `q6_q8_dot` (native_mmvq.cu:638).  `p` = `iqs` in 0..31 (Q6_K is VDR=1/QI=32, so 32 parts).
float q6_q8_dot(uint blk, uint abase, uint p) {
    const uint bq8_offset = 4u * (p / 16u) + (p % 16u) / 8u;    // 0..3
    const uint scale_offset = 8u * (p / 16u) + (p % 16u) / 4u;  // 0..15, an int8 index into scales
    const uint vh_shift = 2u * ((p % 16u) / 8u);                // 0 or 2

    const uint vl = kd_wu32(blk + 4u * p);                       // load_int_b2(w->ql, p): bytes 4p..4p+3
    const int vh = int(kd_wu32(blk + 128u + 4u * (8u * (p / 16u) + p % 8u))) >> int(vh_shift);
    const uint sb = blk + 192u + scale_offset;                   // w->scales is int8[16] at offset 192

    float sumf = 0.0;
    for (uint i = 0u; i < 2u; ++i) {
        const int sc = int(int8_t(w_b.b[sb + 4u * i]));          // SIGNED scale (the source's `scales[4*i]`)
        const int vil = (int(vl) >> int(4u * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> int(4u * i)) << 4) & 0x30303030;
        const uint vi0 = uint(vil | vih);
        // __vsubss4(vi0, 0x20202020): per-BYTE subtract of 32 (values 0..63 -> -32..31, no saturation, but a
        // 32-bit subtract would borrow across bytes).
        uint viu = 0u;
        for (uint b = 0u; b < 4u; ++b) {
            const int bytev = int((vi0 >> (8u * b)) & 0xFFu) - 32;
            viu |= (uint(bytev) & 0xFFu) << (8u * b);
        }
        const uint aoff = abase + (bq8_offset + 2u * i) * KD_Q8_1_BLOCK;
        const float d8 = kd_ahalf(aoff);
        const int u = q5_a_i32(aoff + 4u + 4u * (p % 8u));
        sumf += d8 * float(q5_dp4a(int(viu), u, 0) * sc);
    }
    const float d = kd_whalf(blk + 208u);                        // w->d, half at offset 208
    return d * sumf;
}
