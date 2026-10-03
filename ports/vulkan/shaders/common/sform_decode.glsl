// ports/vulkan/shaders/common/sform_decode.glsl - the S-family canonical decode, in ONE place.
//
// docs/pack-format.md gives S2, S4 and S8 ONE decode:
//
//     value = decode(code) * scale + offset        decode(code) = code + bias        (Affine)
//                                                                  = iq4nl[code & 15] (Iq4Nl)
//
// That uniformity is the point of the canonical form - thirteen source types collapse to three code widths and
// a handful of attributes - so the decode lives here rather than being written once per kernel. A second decode
// of the canonical form is a second thing to get wrong.
//
// THE BIAS IS APPLIED TO THE CODE, IN THE INTEGER DOMAIN, before the multiply. The offset is different: it
// belongs to the WEIGHT and is applied AFTER the scale (`code*scale + offset`), and it is the caller's job to
// apply it there - writing `acc += code*scale*x + offset*x` computes a mathematically equal expression that
// ROUNDS DIFFERENTLY, two multiplications and an addition where the correct form performs one of each. The
// no-offset types cannot tell the difference, which is how that survived until a Q4_K case was added.
//
// `kvalues_iq4nl`, verbatim from ggml-common.h. THE SOURCE KEEPS ITS COPY IN `__constant__` MEMORY AND THAT WAS
// A 2.12x MISTAKE THERE, which its own comment records: constant memory is fast when the access is UNIFORM and a
// non-linear codebook means every lane reads a DIFFERENT index, the pattern it handles worst. The port has no
// constant memory to misuse - the table is a literal in the shader, which the compiler folds into immediate
// values or registers - so the trap does not arise here at all.
float sform_decode(uint code, int bias, int codebook) {
    if (codebook == 1) {
        const int tbl[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10,
                                    1, 13, 25, 38, 53, 69, 89, 113);
        return float(tbl[int(code) & 0x0F]);
    }
    return float(int(code) + bias);
}
