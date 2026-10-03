// ports/vulkan/shaders/common/f16_bits.glsl - the engine's fp16 conversion, in ONE place.
//
// Transcribed from `strata/kernels/f16_bits.hpp`, and held bit-exact by the `f32_to_f16` gate case against that
// header.  Three kernels convert to fp16 (the conversion kernel, the q8 append's scale, the q8 gather's values),
// and three copies of a bit-exact function is three chances to diverge - the audit found exactly that pattern
// once already with the converter helpers.
//
// IT IS NOT `packHalf2x16`.  Measured on RADV: the builtin SATURATES to the largest finite half (0x7BFF) where
// this returns infinity (0x7C00), and its tie/subnormal rounding is not the engine's nearest-even.  In the q8
// append that showed up as one KV group whose maximum exceeded fp16 range getting a finite scale instead of an
// infinite one - every code in it came out 127 instead of 0.  Reading a half BACK needs no such care:
// `unpackHalf2x16` is a pure widening conversion with no rounding to get wrong.
//
// The rounding is ROUND-TO-NEAREST-EVEN with an explicit subnormal path, which is what numpy's float16 does and
// what the engine's header names as its oracle.
uint16_t f16_from_f32_port(float f) {
    uint x = floatBitsToUint(f);
    uint sign = (x >> 16) & 0x8000u;
    uint rawexp = (x >> 23) & 0xFFu;
    int exp = int(rawexp) - 127 + 15;
    uint man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return uint16_t(sign | 0x7C00u | (man != 0u ? 0x200u : 0u));   // inf / quiet NaN
    if (exp >= 31) return uint16_t(sign | 0x7C00u);                                    // finite overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);                                          // underflow -> signed zero
        man |= 0x800000u;
        uint sh = uint(14 - exp);
        uint h = (man >> sh) & 0x3FFu;
        uint rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u) != 0u)) h = h + 1u;
        return uint16_t(sign | h);
    }
    uint16_t h = uint16_t(sign | (uint(exp) << 10) | (man >> 13));
    uint rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (uint(h) & 1u) != 0u)) h = uint16_t(uint(h) + 1u);
    return h;
}
