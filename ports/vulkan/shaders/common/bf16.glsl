// ports/vulkan/shaders/common/bf16.glsl - bf16 -> f32, which is a pure WIDENING; and f32 -> bf16.
//
// `bf16_bits.hpp`'s `f32_from_bf16(h)` is `bitcast((uint32) h << 16)`, so there is no rounding in the widening -
// unlike the fp16 direction, where the port's `f16_bits.glsl` exists precisely because a builtin got the
// saturation and tie rules wrong.
float bf16_to_f32(uint h) { return uintBitsToFloat(h << 16u); }

// f32 -> bf16, round-to-nearest-EVEN, KEPT AS AN F32.  The widening above is exact, so a bf16 value held in an
// f32 is the same number the engine's uint16 widens to, and a kernel that must store a bf16 activation can
// store this value in an f32 buffer without a uint16 buffer or a second conversion.
//
// The rule is `gr_parity.cpp::to_bf16` / the engine's `bf16_from_f32`: `i = (i + ((i>>16)&1) + 0x7FFF) & 0xFFFF0000`.
// It is round-half-to-even (the `>>16 & 1` carries the tie to even), NOT the round-half-away the naive
// `bits + 0x8000` produces - they agree except on exact ties, which is exactly the class a numeric test misses.
float bf16_round(float x) {
    uint u = floatBitsToUint(x);
    u = (u + ((u >> 16u) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return uintBitsToFloat(u);
}
