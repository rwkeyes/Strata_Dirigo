// ports/vulkan/shaders/common/iq_dequant.glsl - the STANDALONE I-quant / BF16 dequantiser, one 256-value
// superblock at a time.
//
// This is the decode half of `dq_dispatch<float>` (src/kernels/cuda/iq_kernels.cu), which is itself llama.cpp's
// dequantize_block_* (ggml-cuda/dequantize.cuh).  THE ENGINE'S DECODER IS THE ORACLE: every offset, every shift
// and every scale here is taken from the CUDA body that the engine's `iq_dequant_f32` and `iq_embed_rows`
// actually launch, and the port keeps the engine's OWN thread mapping (`tid` 0..31, `il = tid/8`, `ib = tid%8`,
// the inner `j` loop and the output position each case writes) rather than re-deriving an element -> byte map.
// A re-derived layout that happens to agree is the port's worst bug class; this file does not have one.
//
// WHERE THE PORT ALREADY HAS A FORMAT, this is the same decode the `iq*_mmvq` dots use (shaders/common/iq*_dot.glsl) -
// but here it stands ALONE, so the row-body write is the whole job rather than a multiply-accumulate.  Formats the
// port does not decode STANDALONE yet (IQ2_XXS ggml type 16, IQ2_XS 17 - their grids are not in the port's
// generated table) are deliberately absent: `iq_dequant_f32` REFUSES them by name rather than computing a
// plausible wrong row.  See ports/vulkan/PORT-MAP.tsv.
//
// THE INCLUDING SHADER MUST DECLARE, with these names:
//     w_b        the raw block bytes                (uint8 storage buffer, std430)
//     iq1s_g     the IQ1_S grid, 2048 uint32        (IQ1_M)         - harness/iq_grids.hpp
//     iq2s_g     the IQ2_S grid, 2048 uint32 (1024 points as lo/hi halves)
//     iq3xxs_g   the IQ3_XXS grid, 256 uint32
//     iq3s_g     the IQ3_S grid, 512 uint32
//     OU_        the destination, float             (writeonly storage buffer, member .y)
//
// and pass `ty` = the ggml type id, `wbase` = the byte offset the superblock's ROW starts at (so the flat form
// uses 0 and the embedding gather uses tokens[t]*row_bytes), `ibs` = the superblock index within that row,
// `tid` = the lane (0..31), `obase` = the output element the superblock's 256 values start at.

const float IQ1M_DELTA_GLSL = 0.125;              // third_party/ggml/ggml-common.h: #define IQ1M_DELTA 0.125f
const int   K_IQ4NL[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10,
                                  1, 13, 25, 38, 53, 69, 89, 113);   // kvalues_iq4nl, verbatim
const int   K_MASK[8] = int[8](1, 2, 4, 8, 16, 32, 64, 128);        // kmask_iq2xs
// ksigns_iq2xs, verbatim from ggml-common.h.  NOT recomputed: this 7-bit index -> 8-bit sign pattern is a table
// in the source, and a formula that "looks equivalent" is a second copy of kernel-critical data.
const uint  K_SIGNS[128] = uint[128](
    0u, 129u, 130u, 3u, 132u, 5u, 6u, 135u, 136u, 9u, 10u, 139u, 12u, 141u, 142u, 15u,
    144u, 17u, 18u, 147u, 20u, 149u, 150u, 23u, 24u, 153u, 154u, 27u, 156u, 29u, 30u, 159u,
    160u, 33u, 34u, 163u, 36u, 165u, 166u, 39u, 40u, 169u, 170u, 43u, 172u, 45u, 46u, 175u,
    48u, 177u, 178u, 51u, 180u, 53u, 54u, 183u, 184u, 57u, 58u, 187u, 60u, 189u, 190u, 63u,
    192u, 65u, 66u, 195u, 68u, 197u, 198u, 71u, 72u, 201u, 202u, 75u, 204u, 77u, 78u, 207u,
    80u, 209u, 210u, 83u, 212u, 85u, 86u, 215u, 216u, 89u, 90u, 219u, 92u, 221u, 222u, 95u,
    96u, 225u, 226u, 99u, 228u, 101u, 102u, 231u, 232u, 105u, 106u, 235u, 108u, 237u, 238u, 111u,
    240u, 113u, 114u, 243u, 116u, 245u, 246u, 119u, 120u, 249u, 250u, 123u, 252u, 125u, 126u, 255u);

uint iq_b(uint off) { return uint(w_b.b[off]); }
uint iq_u16(uint off) { return iq_b(off) | (iq_b(off + 1u) << 8u); }
int  iq_i8(uint off) { const int b = int(iq_b(off)); return (b << 24) >> 24; }   // sign-extend a byte
float iq_h2f(uint off) { return unpackHalf2x16(iq_u16(off)).x; }                  // ggml_half -> f32 (exact)
// the 8 bytes of a uint64 grid point, stored as a lo/hi uint32 pair (no shaderInt64 needed)
uint iq_g64_byte(uint lo, uint hi, uint j) { return (j < 4u) ? ((lo >> (8u * j)) & 0xFFu)
                                                             : ((hi >> (8u * (j - 4u))) & 0xFFu); }
int  iq_g32_byte(uint w, uint j) { return int((w >> (8u * j)) & 0xFFu); }

// get_scale_min_k4 (ggml-common.h / dequantize.cuh), out d and m.
void iq_scale_min_k4(uint j, uint scbase, out uint d, out uint m) {
    if (j < 4u) { d = iq_b(scbase + j) & 63u; m = iq_b(scbase + j + 4u) & 63u; }
    else { d = (iq_b(scbase + j + 4u) & 0xFu) | ((iq_b(scbase + j - 4u) >> 6u) << 4u);
           m = (iq_b(scbase + j + 4u) >> 4u) | ((iq_b(scbase + j - 4u) >> 6u) << 4u); }
}

// ------------------------------------------------------------------------------------------------
// ONE superblock.  `out` writes go to OU_.y[obase + <the position the CUDA case writes>].
void iq_dq_256(int ty, uint wbase, uint ibs, uint tid, uint obase) {
    const uint il = tid / 8u, ib = tid % 8u;
    if (ty == 30) {                                              // BF16: 8 raw halves per thread, widened exactly
        const uint x0 = wbase + ibs * 512u + tid * 16u;
        for (uint j = 0u; j < 8u; ++j) OU_.y[obase + tid * 8u + j] = uintBitsToFloat(iq_u16(x0 + j * 2u) << 16u);
    } else if (ty == 20) {                                       // IQ4_NL: 18 bytes per block, 8 blocks per 256
        const uint bb = wbase + ibs * 144u + ib * 18u;
        const float d = iq_h2f(bb);
        for (uint j = 0u; j < 4u; ++j) {
            const uint byte = iq_b(bb + 2u + 4u * il + j);
            OU_.y[obase + 32u * ib + 4u * il + j]      = d * float(K_IQ4NL[byte & 0xFu]);
            OU_.y[obase + 32u * ib + 4u * il + j + 16u] = d * float(K_IQ4NL[byte >> 4u]);
        }
    } else if (ty == 23) {                                       // IQ4_XS: 136 bytes, one per 256
        const uint bb = wbase + ibs * 136u;
        const uint sh = iq_u16(bb + 2u);
        const int ls = int((iq_b(bb + 4u + ib / 2u) >> (4u * (ib % 2u))) & 0xFu) | (int((sh >> (2u * ib)) & 3u) << 4);
        const float d = iq_h2f(bb) * float(ls - 32);
        const uint qs = bb + 8u + 16u * ib + 4u * il;
        for (uint j = 0u; j < 4u; ++j) {
            const uint byte = iq_b(qs + j);
            OU_.y[obase + 32u * ib + 4u * il + j]      = d * float(K_IQ4NL[byte & 0xFu]);
            OU_.y[obase + 32u * ib + 4u * il + j + 16u] = d * float(K_IQ4NL[byte >> 4u]);
        }
    } else if (ty == 8) {                                        // Q8_0: 34 bytes, 8 blocks per 256
        const uint bb = wbase + ibs * 272u + ib * 34u;
        const float d = iq_h2f(bb);
        for (uint j = 0u; j < 8u; ++j)
            OU_.y[obase + 32u * ib + 8u * il + j] = float(iq_i8(bb + 2u + 8u * il + j)) * d;
    } else if (ty == 6) {                                        // Q5_0: d, qh[4], qs[16]; -16 bias, one delta
        const uint bb = wbase + ibs * 176u + ib * 22u;
        const float d = iq_h2f(bb);
        const uint qh = iq_b(bb + 2u) | (iq_b(bb + 3u) << 8u) | (iq_b(bb + 4u) << 16u) | (iq_b(bb + 5u) << 24u);
        const uint qs = bb + 6u;
        for (uint j = 0u; j < 4u; ++j) {
            const uint iqs = 4u * il + j;
            const uint v0 = (iq_b(qs + iqs) & 0xFu) | (((qh >> (iqs + 0u)) << 4u) & 0x10u);
            const uint v1 = (iq_b(qs + iqs) >> 4u) | (((qh >> (iqs + 12u))) & 0x10u);
            OU_.y[obase + 32u * ib + iqs]      = (float(v0) - 16.0) * d;
            OU_.y[obase + 32u * ib + iqs + 16u] = (float(v1) - 16.0) * d;
        }
    } else if (ty == 7) {                                        // Q5_1: d,m, qh[4], qs[16]; a min, no -16
        const uint bb = wbase + ibs * 192u + ib * 24u;
        const float dmx = iq_h2f(bb), dmy = iq_h2f(bb + 2u);
        const uint qh = iq_b(bb + 4u) | (iq_b(bb + 5u) << 8u) | (iq_b(bb + 6u) << 16u) | (iq_b(bb + 7u) << 24u);
        const uint qs = bb + 8u;
        for (uint j = 0u; j < 4u; ++j) {
            const uint iqs = 4u * il + j;
            const uint v0 = (iq_b(qs + iqs) & 0xFu) | (((qh >> (iqs + 0u)) << 4u) & 0x10u);
            const uint v1 = (iq_b(qs + iqs) >> 4u) | (((qh >> (iqs + 12u))) & 0x10u);
            OU_.y[obase + 32u * ib + iqs]      = float(v0) * dmx + dmy;
            OU_.y[obase + 32u * ib + iqs + 16u] = float(v1) * dmx + dmy;
        }
    } else if (ty == 42) {                                       // Q2_0: 18 bytes per 64, 4 blocks per 256
        const uint b = tid / 8u, part = tid % 8u;
        const uint bb = wbase + ibs * 72u + b * 18u;
        const float d = iq_h2f(bb);
        for (uint j = 0u; j < 8u; ++j) {
            const uint i = part * 8u + j;
            const uint code = (iq_b(bb + 2u + i / 4u) >> ((i % 4u) * 2u)) & 3u;
            OU_.y[obase + b * 64u + i] = d * (float(code) - 1.0);
        }
    } else if (ty == 12) {                                       // Q4_K: dm, scales[12], qs[128]
        const uint bb = wbase + ibs * 144u;
        const float dall = iq_h2f(bb), dmin = iq_h2f(bb + 2u);
        uint sc0, m0, sc1, m1;
        iq_scale_min_k4(2u * il, bb + 4u, sc0, m0);
        iq_scale_min_k4(2u * il + 1u, bb + 4u, sc1, m1);
        const float d1 = dall * float(sc0), m1v = dmin * float(m0);
        const float d2 = dall * float(sc1), m2v = dmin * float(m1);
        const uint q = bb + 16u + 32u * il + 4u * ib;
        for (uint l = 0u; l < 4u; ++l) {
            const uint qb = iq_b(q + l);
            OU_.y[obase + 64u * il + 4u * ib + l]      = d1 * float(qb & 0xFu) - m1v;
            OU_.y[obase + 64u * il + 4u * ib + l + 32u] = d2 * float(qb >> 4u) - m2v;
        }
    } else if (ty == 13) {                                       // Q5_K: dm, scales[12], qh[32], qs[128]
        const uint bb = wbase + ibs * 176u;
        const float dall = iq_h2f(bb), dmin = iq_h2f(bb + 2u);
        for (int tt = int(tid); tt < 64; tt += 32) {             // the CUDA folds q5_K's 64 threads onto 32
            const uint iil = uint(tt) / 16u, iir = uint(tt) % 16u;
            uint sc0, m0, sc1, m1;
            iq_scale_min_k4(2u * iil, bb + 4u, sc0, m0);
            iq_scale_min_k4(2u * iil + 1u, bb + 4u, sc1, m1);
            const float d1 = dall * float(sc0), m1v = dmin * float(m0);
            const float d2 = dall * float(sc1), m2v = dmin * float(m1);
            const uint ql = bb + 48u + 32u * iil + 2u * iir;
            const uint qh = bb + 16u + 2u * iir;
            uint hm = 1u << (2u * iil);
            const uint ob = obase + 64u * iil + 2u * iir;
            OU_.y[ob + 0u]  = d1 * (float(iq_b(ql + 0u) & 0xFu) + (((iq_b(qh + 0u) & hm) != 0u) ? 16.0 : 0.0)) - m1v;
            OU_.y[ob + 1u]  = d1 * (float(iq_b(ql + 1u) & 0xFu) + (((iq_b(qh + 1u) & hm) != 0u) ? 16.0 : 0.0)) - m1v;
            hm <<= 1u;
            OU_.y[ob + 32u] = d2 * (float(iq_b(ql + 0u) >> 4u) + (((iq_b(qh + 0u) & hm) != 0u) ? 16.0 : 0.0)) - m2v;
            OU_.y[ob + 33u] = d2 * (float(iq_b(ql + 1u) >> 4u) + (((iq_b(qh + 1u) & hm) != 0u) ? 16.0 : 0.0)) - m2v;
        }
    } else if (ty == 11) {                                       // Q3_K: hmask[32], qs[64], scales[12], d
        const uint bb = wbase + ibs * 110u;
        const float d_all = iq_h2f(bb + 108u);
        for (int tt = int(tid); tt < 64; tt += 32) {             // 64 threads folded onto 32
            const int r = tt / 4, t2 = r / 2, is0 = r % 2;
            const int l0 = 16 * is0 + 4 * (tt % 4);
            const int n = t2 / 4, j = t2 - 4 * n;
            const uint m = 1u << uint(4 * n + j);
            const int is = 8 * n + 2 * j + is0;
            const int shift = 2 * j;
            const int us = (is < 4)  ? int((iq_b(bb + 96u + uint(is)) & 0xFu) | (((iq_b(bb + 96u + uint(is) + 8u)) & 3u) << 4u))
                         : (is < 8)  ? int((iq_b(bb + 96u + uint(is)) & 0xFu) | (((iq_b(bb + 96u + uint(is) + 4u) >> 2u) & 3u) << 4u))
                         : (is < 12) ? int((iq_b(bb + 96u + uint(is - 8)) >> 4u) | (((iq_b(bb + 96u + uint(is)) >> 4u) & 3u) << 4u))
                                     : int((iq_b(bb + 96u + uint(is - 8)) >> 4u) | (((iq_b(bb + 96u + uint(is - 4)) >> 6u) & 3u) << 4u));
            const float dl = d_all * float(us - 32);
            const uint ob = obase + 128u * uint(n) + 32u * uint(j);
            const uint q = bb + 32u * uint(n);
            for (int l = l0; l < l0 + 4; ++l)
                OU_.y[ob + uint(l)] = dl * float(int((iq_b(q + uint(l)) >> uint(shift)) & 3u) - (((iq_b(bb + uint(l)) & m) != 0u) ? 0 : 4));
        }
    } else if (ty == 18) {                                       // IQ3_XXS: d, qs[96]; two 4-byte grids per part
        const uint bb = wbase + ibs * 98u;
        const uint qs = bb + 2u + 8u * ib;
        const uint g1 = iq3xxs_g.v[iq_b(qs + 2u * il + 0u)];
        const uint g2 = iq3xxs_g.v[iq_b(qs + 2u * il + 1u)];
        const uint gas = bb + 66u + 4u * ib;
        const uint aux32 = iq_u16(gas) | (iq_u16(gas + 2u) << 16u);
        const float d = iq_h2f(bb) * (0.5 + float(aux32 >> 28u)) * 0.5;
        const uint signs = K_SIGNS[(aux32 >> (7u * il)) & 127u];
        for (uint j = 0u; j < 4u; ++j) {
            OU_.y[obase + 32u * ib + 8u * il + j]     = d * float(iq_g32_byte(g1, j)) * ((signs & uint(K_MASK[j])) != 0u ? -1.0 : 1.0);
            OU_.y[obase + 32u * ib + 8u * il + j + 4u] = d * float(iq_g32_byte(g2, j)) * ((signs & uint(K_MASK[j + 4])) != 0u ? -1.0 : 1.0);
        }
    } else if (ty == 21) {                                       // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
        const uint bb = wbase + ibs * 110u;
        const uint qh = iq_b(bb + 66u + ib);
        const uint g1 = iq_b(bb + 2u + 8u * ib + 2u * il + 0u) | ((qh << (8u - 2u * il)) & 256u);
        const uint g2 = iq_b(bb + 2u + 8u * ib + 2u * il + 1u) | ((qh << (7u - 2u * il)) & 256u);
        const uint w1 = iq3s_g.v[g1], w2 = iq3s_g.v[g2];
        const uint sc = iq_b(bb + 106u + ib / 2u);
        const float d = iq_h2f(bb) * (1.0 + 2.0 * float((sc >> (4u * (ib % 2u))) & 0xFu));
        const uint signs = iq_b(bb + 74u + 4u * ib + il);
        for (uint j = 0u; j < 4u; ++j) {
            OU_.y[obase + 32u * ib + 8u * il + j]     = d * float(iq_g32_byte(w1, j)) * ((signs & uint(K_MASK[j])) != 0u ? -1.0 : 1.0);
            OU_.y[obase + 32u * ib + 8u * il + j + 4u] = d * float(iq_g32_byte(w2, j)) * ((signs & uint(K_MASK[j + 4])) != 0u ? -1.0 : 1.0);
        }
    } else if (ty == 22) {                                       // IQ2_S: d, qs[64], qh[8], scales[8]
        const uint bb = wbase + ibs * 82u;
        const uint qh = iq_b(bb + 66u + ib);
        const uint gidx = iq_b(bb + 2u + 4u * ib + il) | ((qh << (8u - 2u * il)) & 0x300u);
        const uint g0 = iq2s_g.v[2u * gidx], g1 = iq2s_g.v[2u * gidx + 1u];
        const uint signs = iq_b(bb + 2u + 32u + 4u * ib + il);
        const float d = iq_h2f(bb) * (0.5 + float((iq_b(bb + 74u + ib) >> (4u * (il / 2u))) & 0xFu)) * 0.25;
        for (uint j = 0u; j < 8u; ++j)
            OU_.y[obase + 32u * ib + 8u * il + j] = d * float(iq_g64_byte(g0, g1, j)) * ((signs & uint(K_MASK[j])) != 0u ? -1.0 : 1.0);
    } else if (ty == 29) {                                       // IQ1_M: qs[32], qh[16], scales[8]
        const uint bb = wbase + ibs * 56u;
        const uint sc0 = iq_u16(bb + 48u), sc1 = iq_u16(bb + 50u), sc2 = iq_u16(bb + 52u), sc3 = iq_u16(bb + 54u);
        const uint scale_u16 = (sc0 >> 12u) | ((sc1 >> 8u) & 0x00F0u) | ((sc2 >> 4u) & 0x0F00u) | (sc3 & 0xF000u);
        const uint ib16 = 2u * ib + il / 2u;
        const uint scv = iq_u16(bb + 48u + 2u * (ib16 / 4u));
        const float d = unpackHalf2x16(scale_u16).x * (2.0 * float((scv >> (3u * (ib16 % 4u))) & 7u) + 1.0);
        const uint qh = iq_b(bb + 32u + 2u * ib + il / 2u);
        const float delta = ((qh & (0x08u << (4u * (il % 2u)))) != 0u) ? (-1.0 - IQ1M_DELTA_GLSL) : (-1.0 + IQ1M_DELTA_GLSL);
        const uint gidx = iq_b(bb + 4u * ib + il) | (((qh >> (4u * (il % 2u))) & 7u) << 8u);
        const uint g = iq1s_g.v[gidx];
        const uint q0 = g & 0x0F0F0F0Fu;
        const uint q1 = (g >> 4u) & 0x0F0F0F0Fu;
        for (uint j = 0u; j < 4u; ++j) {
            OU_.y[obase + 32u * ib + 8u * il + j]     = d * (float((q0 >> (8u * j)) & 0xFu) + delta);
            OU_.y[obase + 32u * ib + 8u * il + j + 4u] = d * (float((q1 >> (8u * j)) & 0xFu) + delta);
        }
    }
    // any other type is a caller error: the host entry points refuse it before dispatch
}
