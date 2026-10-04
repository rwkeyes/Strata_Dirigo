// ports/vulkan/shaders/common/bf16.glsl - bf16 -> f32, which is a pure WIDENING.
//
// `bf16_bits.hpp`'s `f32_from_bf16(h)` is `bitcast((uint32) h << 16)`, so there is no rounding here and no
// regime to get wrong - unlike the fp16 direction, where the port's `f16_bits.glsl` exists precisely because a
// builtin got the saturation and tie rules wrong.
float bf16_to_f32(uint h) { return uintBitsToFloat(h << 16u); }
