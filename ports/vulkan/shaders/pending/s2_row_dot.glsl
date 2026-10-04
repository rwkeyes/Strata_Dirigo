// ports/vulkan/shaders/common/s2_row_dot.glsl - ONE S2 ROW AGAINST A Q8_0 ACTIVATION, the expert tier's primitive.
//
//     acc = sum over 32-element chunks of  dw * dx * sum_e (code[e] - 1) * x[e]
//
// **THE BIAS IS FOLDED INTO AN INTEGER IDENTITY**: `sum((code-1)*x) == sum(code*x) - sum(x)`, so the kernel
// computes TWO integer sums per chunk and subtracts them - the code word against the activations (`s`), and a
// word of ones against the activations (`hx`). That is what the CUDA source's `dp4a` pair computes, and it is why
// the -1 bias costs nothing.
//
// THE TWO INTEGERS ARE EXACT AND THEREFORE IDENTICAL TO THE CUDA'S. The source uses `__dp4a` (four int8 MACs in
// one instruction) and GLSL has no such intrinsic, so the four lane products are written out here - but integer
// arithmetic is exact, so `s` and `hx` are the SAME integers either way. Only the instruction count differs, and
// the source's own comment says the dp4a form is "the same integers from fewer instructions".
//
// ONE SCALE PER 64 ELEMENTS, which is per TWO 32-element chunks - hence `(c >> 1)`.
//
// `use_xscales` SELECTS THE ACTIVATION'S MULTIPLIER, and it exists because the CPU and GPU paths disagreed:
// the fp32 `ActQ::scale` of the CPU pool against the block's fp16 `d` differed by 4.761e-04 relative on
// **80 of 80 chunks**, which is what made a cache HIT compute a different expert from a cache MISS. Passing the
// fp16 form keeps the previous behaviour exactly, so the parity test still measures the kernel it always did.
//
// The including shader must declare, with these names:
//     blob_b    the expert blob  (uint8 storage buffer)
//     act_b     the activation   (uint8, block_q8_0)
//     xscale_v  fp32 scales      (float buffer, only read when use_xscales is true)
float f16_at(uint byte_off) {
    return unpackHalf2x16(uint(blob_b.b[byte_off]) | (uint(blob_b.b[byte_off + 1u]) << 8u)).x;
}

float s2_row_dot(uint code_off, uint scale_off, uint x_off, uint n_chunks, uint x_scale_off, bool use_xscales) {
    float acc = 0.0;
    const uint lid = gl_LocalInvocationIndex;
    const uint lsize = gl_WorkGroupSize.x;
    for (uint c = lid; c < n_chunks; c += lsize) {
        const uint cb = code_off + c * 8u;              // 8 code bytes = 32 elements
        const uint xb = x_off + c * 34u;                // one block_q8_0
        const float dx = use_xscales ? xscale_v.v[x_scale_off + c] : f16_at(xb);
        int s = 0;                                      // sum of code * x
        int hx = 0;                                     // sum of x
        for (uint j = 0u; j < 8u; ++j) {
            const uint cbyte = uint(blob_b.b[cb + j]);
            for (uint k = 0u; k < 4u; ++k) {
                const int code = int((cbyte >> (2u * k)) & 3u);
                const int raw = int(act_b.b[xb + 2u + 4u * j + k]);
                const int xv = raw > 127 ? raw - 256 : raw;     // sign-extend by hand
                s += code * xv;
                hx += xv;
            }
        }
        const float dw = f16_at(scale_off + (c >> 1u) * 2u);
        acc += dw * dx * float(s - hx);
    }
    return acc;
}
