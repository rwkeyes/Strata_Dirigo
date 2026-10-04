// ports/vulkan/shaders/common/q8_1_store.glsl - ONE q8_1 BLOCK from 32 floats, in the CUDA's own order.
//
// Transcribed from `q8_1_store` (src/kernels/cuda/iq_kernels.cu), which is what `quantize_q8_1_kernel` and
// `swiglu_q8_1_entries_kernel` there reduce to, and what `native_quantize_q8_1_kernel` in native_mmvq.cu
// reproduces.  A q8_1 block is 36 bytes:
//
//     half2 ds    at 0    d = the block's scale, s = the SUM of the 32 ORIGINAL floats
//     int8 qs[32] at 4
//
// **THE SUM IS THE BUTTERFLY ORDER'S, WHICH IS WHY THIS IS NOT `wg_sum`.**  The CUDA accumulates it with
// `__shfl_xor_sync` at offsets 16, 8, 4, 2, 1 - five rounds, every lane adding its partner's PARTIAL - and then
// stores the result as fp16.  A different addition order gives a different last bit, and the fp16 storage can
// turn that into a different stored half, so the port needs THIS block's BYTES, not a correct sum of them.  That
// is why the exchange below is a ping-pong pair in shared memory with one barrier per round, reproducing the
// shuffle's order exactly, rather than the pairwise tree every other kernel here uses.
//
// THE ROUNDING RULE IS `roundf`: half AWAY FROM ZERO.  GLSL's `round()` leaves the tie direction to the
// implementation, so the rule is written out as `floor(t + 0.5)` / `ceil(t - 0.5)`, which is exact for the range
// this quantiser can produce (|t| <= 127).  The engine's CPU path is explicit that this is the rule and that
// round-half-to-EVEN is the wrong one - "it would quietly change the activation on the ties" - and its SSE path
// (cpu/expert.cpp, `_MM_FROUND_TO_NEAREST_INT`) is the one place in the engine that gets this wrong.
//
// THE CLAMP IS #606's, and it is the difference between a finite approximation and a NaN: a block whose SUM
// exceeds 65504 - one large activation among the 32 is enough - rounds to inf as fp16, and d does past
// amax = 8.3M; the dot products then read inf * 0 = NaN.  `q8_1_finite` clamps to the largest finite half, so
// such a block yields an approximate product instead of a NaN.  Every block that was finite before is stored bit
// for bit as before, and a NaN stays NaN (the comparison is false for it).
//
// The including shader must declare, with these names:
//     q8_1_out   the destination (uint8 storage buffer of 36-byte q8_1 blocks)
const uint WG_Q8_1_SIZE = 256u;                  // the host's kLocalSize, checked by the gate against every shader
shared float q81_amax[2][WG_Q8_1_SIZE];
shared float q81_sum[2][WG_Q8_1_SIZE];

float q8_1_finite(float v) {
    return (abs(v) > 65504.0) ? ((v < 0.0) ? -65504.0 : 65504.0) : v;
}

float q8_1_roundf(float t) {
    return (t >= 0.0) ? floor(t + 0.5) : ceil(t - 0.5);
}

// `in_range` exists so a workgroup whose last 32-lane block is out of range can still take part in the exchange
// (a barrier is workgroup-wide) while writing nothing; an out-of-range lane must not have its offset used to
// write, because the block it would land on belongs to the NEXT column.  It is NOT named `active`: that is a GLSL
// reserved word, and the failure is a parse error pointing at the declaration rather than at the name.
void q8_1_store(float xi, uint out_off, bool in_range) {
    const uint lid = gl_LocalInvocationIndex;
    q81_amax[0][lid] = abs(xi);
    q81_sum[0][lid] = xi;
    barrier();
    uint cur = 0u;
    for (uint o = 16u; o > 0u; o >>= 1u) {
        const uint nxt = 1u - cur;
        // the XOR stays inside this lane's 32-lane block for every offset <= 16, which is what the CUDA's
        // `__shfl_xor_sync` mask of 32 lanes also restricts it to
        q81_amax[nxt][lid] = max(q81_amax[cur][lid], q81_amax[cur][lid ^ o]);
        q81_sum[nxt][lid] = q81_sum[cur][lid] + q81_sum[cur][lid ^ o];
        barrier();
        cur = nxt;
    }
    const float amax = q81_amax[cur][lid];
    const float sum = q81_sum[cur][lid];
    if (in_range) {
        // `amax == 0` is an all-zero block: the CUDA returns a zero code and never divides
        const float d = q8_1_finite(amax / 127.0);
        const float r = q8_1_roundf((amax == 0.0) ? 0.0 : (xi / d));
        const float c = (r > 127.0) ? 127.0 : ((r < -127.0) ? -127.0 : r);
        const int q = (amax == 0.0) ? 0 : int(c);
        const uint elem = lid & 31u;
        q8_1_out.b[out_off + 4u + elem] = uint8_t(uint(q) & 0xFFu);
        if (elem == 0u) {
            const uint16_t dh = f16_from_f32_port(q8_1_finite(d));
            const uint16_t sh = f16_from_f32_port(q8_1_finite(sum));
            q8_1_out.b[out_off + 0u] = uint8_t(uint(dh) & 0xFFu);
            q8_1_out.b[out_off + 1u] = uint8_t((uint(dh) >> 8u) & 0xFFu);
            q8_1_out.b[out_off + 2u] = uint8_t(uint(sh) & 0xFFu);
            q8_1_out.b[out_off + 3u] = uint8_t((uint(sh) >> 8u) & 0xFFu);
        }
    }
    // the shared arrays are reused by the next call in the same invocation, so this must be the last act
    barrier();
}
