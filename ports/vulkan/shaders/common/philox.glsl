// ports/vulkan/shaders/common/philox.glsl - the sampler's random number generator.
//
// Transcribed from src/kernels/cuda/sampler.cu's `philox4x32_round` / `philox_uniform`: Philox4x32 with TEN rounds,
// key = the 64-bit seed, counter = the 64-bit counter, and a 24-bit mantissa taken from the FIRST output word so the
// result is uniform in [0,1) with no rounding to 1.0.
//
// **THE CONSTANTS ARE THE ENGINE'S, NOT Random123's.**  The canonical Random123 philox4x32 uses M0 = 0xD2511F53 and
// M1 = 0xCD9E8D57; this engine uses 0x9E3779B9 and 0xBB67AE85.  Swapping them yields a perfectly good generator that
// produces different numbers, which is exactly the kind of change that looks like "the model got a bit worse" rather
// than like a bug - so the case pins the stream against a host transcription of THESE constants.
//
// The 64x32 multiply needs the high word, which GLSL gives through 64-bit integers (GL_EXT_shader_explicit_
// arithmetic_types_int64) rather than a `mulhi` builtin.  A `uint64_t` multiply of two 32-bit values cannot overflow
// 64 bits, so the shift is exact.

#ifndef PORT_PHILOX_GLSL
#define PORT_PHILOX_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

const uint PHILOX_M0 = 0x9E3779B9u;
const uint PHILOX_M1 = 0xBB67AE85u;

void philox4x32_round(inout uint c0, inout uint c1, inout uint c2, inout uint c3, uint k0, uint k1) {
    const uint hi0 = uint((uint64_t(PHILOX_M0) * uint64_t(c0)) >> 32);
    const uint hi1 = uint((uint64_t(PHILOX_M1) * uint64_t(c2)) >> 32);
    const uint lo0 = PHILOX_M0 * c0;
    const uint lo1 = PHILOX_M1 * c2;
    const uint n0 = hi1 ^ c1 ^ k0;
    const uint n1 = lo1;
    const uint n2 = hi0 ^ c3 ^ k1;
    const uint n3 = lo0;
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
}

// The engine's draw: ten rounds keyed by (seed_lo, seed_hi), counted by (counter_lo, counter_hi).
float philox_uniform(uint seed_lo, uint seed_hi, uint counter_lo, uint counter_hi) {
    uint c0 = counter_lo, c1 = counter_hi, c2 = seed_lo, c3 = seed_hi;
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, uint(i), 0u);
    }
    // 24 bits of mantissa: uniform in [0,1) and never rounds up to 1.0.
    return float(c0 >> 8) * (1.0 / 16777216.0);
}

#endif
