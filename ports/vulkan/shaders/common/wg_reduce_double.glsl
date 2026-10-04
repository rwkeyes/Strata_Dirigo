// ports/vulkan/shaders/common/wg_reduce_double.glsl - the workgroup-wide sum in DOUBLE.
//
// WHY THIS IS A SEPARATE FILE FROM `wg_reduce.glsl`. That one is built on `subgroupAdd`, and the subgroup
// arithmetic extension has NO double overload - double is not in its type list, only float and the integer widths.
// So the double reduction is a shared-memory TREE with barriers: deterministic, and its own summation order, which
// is not CUDA's warp tree. The difference lives in the last bits of the double total, and after a sigmoid and a
// cast to f32 it does not reach the output - which is why the f64 variants can still be held to bit-exactness on
// their FLOAT results.
//
// Only the f64 variants may include this: it makes the shader require the device's shaderFloat64.
shared double rpd_partial[256];

double wg_sum_d(double v) {
    barrier();                                   // the array is reused across calls: see the note in wg_reduce.glsl
    rpd_partial[gl_LocalInvocationIndex] = v;
    barrier();
    for (uint s = gl_WorkGroupSize.x >> 1u; s > 0u; s >>= 1u) {
        if (gl_LocalInvocationIndex < s) rpd_partial[gl_LocalInvocationIndex] += rpd_partial[gl_LocalInvocationIndex + s];
        barrier();
    }
    return rpd_partial[0];
}
