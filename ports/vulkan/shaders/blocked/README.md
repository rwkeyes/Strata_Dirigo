# Blocked shaders (now empty of blockers)

`silu_fp64.comp` used to live here. It was the port's first attempt at a double-precision SiLU, parked because
glslang has no `exp(double)` and the port recorded it as inexpressible through this toolchain.

**IT WAS EXPRESSIBLE, so the file is gone rather than kept as a trophy.** `common/double_math.glsl` builds the
double exp from `roundEven(double)`, `ldexp(double, int)` and an 18-term series with the argument reduced to
|r| <= ln2/2. `swiglu_f64.comp` is the shipped kernel: gated BIT-EXACT against the host's double reference, and
demonstrably a different arithmetic from a float silu (665 of 2560 elements differ). The router needed the same
piece for the same reason - its exponentials are double too.

The lesson generalises past this file: "the toolchain cannot express X" deserves the same scepticism as "the
hardware cannot do X", and both deserve a test that tries. What actually remains is a CONSTRAINT rather than a
blocker - a double-precision kernel requires the device's shaderFloat64, which Intel Arc does not have, so
`swiglu_f32.comp` is what runs there and the gate measures the difference.
