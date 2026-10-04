// ports/vulkan/shaders/common/perbyte_sign.glsl - the per-byte integer ops the I-quant dots share.
//
// `__vcmpne4(a, b)` and `__vsub4(a, b)` are CUDA's PER-BYTE comparison and subtraction: no carries between the
// four bytes, no sign, no float. Two formats in this port use them, for the SAME thing: build a 4-byte mask out of
// a sign nibble, then conditionally negate each byte of a grid word with `(g ^ s) - s`, where every byte of `s` is
// 0xFF (negate) or 0x00 (keep).
//
// ONE RULE USED TWICE, which is why it is one helper: the source writes the identical idiom in
// `vec_dot_iq2_s_q8_1` and `vec_dot_iq3_xxs_q8_1`. This is NOT the case the two q8_1 quantisers are - those are
// the same quantity under two DIFFERENT rules, and the port deliberately carries both (see `q8_1_store.glsl`).
// Before merging two lookalikes, check which of the two situations you are in.
//
// **The harness keeps its OWN copies of these.** The oracle has to be independent of the thing it checks; sharing
// a helper would make a wrong helper agree with itself and pass. That is the same reason the gate's buffer
// addresses differ per binding rather than sharing one offset table.
//
// **`out` IS A GLSL KEYWORD** (a parameter qualifier): `int out = 0;` fails with "unexpected OUT" at the
// declaration. So are `in`, `active`, `filter`, `patch`, `precise`, `input`, `output`, `sample`. The port's skill
// reference lists them; read that list BEFORE writing a new kernel, not after the compile fails.
#ifndef PERBYTE_SIGN_GLSL
#define PERBYTE_SIGN_GLSL

// __vcmpne4(t, 0): 0xFF in every byte of `t` that is non-zero, 0x00 where it is zero.
int perbyte_ne_zero(int t) {
    int mask = 0;
    for (int b = 0; b < 4; ++b) {
        const int shift = 8 * b;
        mask |= ((((t >> shift) & 0xFF) != 0) ? 0xFF : 0x00) << shift;
    }
    return mask;
}

// __vsub4(g, s) as a per-byte conditional negation: every byte of `s` is 0xFF (negate) or 0x00 (keep).
int perbyte_sign_flip(int g, int s) {
    int flipped = 0;
    for (int b = 0; b < 4; ++b) {
        const int shift = 8 * b;
        const int gb = (g >> shift) & 0xFF;
        const int sb = (s >> shift) & 0xFF;
        flipped |= (((gb ^ sb) - sb) & 0xFF) << shift;
    }
    return flipped;
}

#endif  // PERBYTE_SIGN_GLSL
