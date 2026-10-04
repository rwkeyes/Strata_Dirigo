// ports/vulkan/shaders/common/double_math.glsl - the double-precision pieces a device without shaderFloat64
// cannot use, kept in ONE place so the port has a single answer for "this engine computes in double".
//
// **THIS FILE EXISTS BECAUSE THE ENGINE'S NUMERICS ARE DOUBLE IN MORE THAN ONE PLACE, AND INTEL ARC HAS NO
// shaderFloat64.** Intel's own support article 000089817: "Integrated GPUs included with 11th Gen Intel
// processors and the upcoming Intel Arc discrete GPUs don't support shaderFloat64." The devices it affects are
// the ones this port targets, so every double-arithmetic kernel needs a PORTABLE SIBLING, and the two are
// compared by measurement rather than by assertion.
//
// glslang has no `exp(double)` (nothing in the double-precision transcendental set beyond sqrt and the simple
// genDType functions), so one is BUILT here from the pieces it does have: `roundEven(double)`, `ldexp(double,int)`
// and a series, with the argument reduced so the series converges quickly and the reduction stays exact (ln2 is
// split in two so `k * ln2` does not lose the low bits).
double double_exp(double x) {
    const double LN2_HI = 0.6931471803691238;
    const double LN2_LO = 1.9082149292705877e-10;
    const double k = roundEven(x / 0.6931471805599453);
    const double r = (x - k * LN2_HI) - k * LN2_LO;      // |r| <= ln2/2, and the two-term ln2 keeps it exact
    double t = 1.0, s = 1.0;
    for (int i = 1; i <= 18; ++i) {                      // 18 terms: |r| <= 0.347, so the tail is far below 2^-53
        t *= r / double(i);
        s += t;
    }
    return ldexp(s, int(k));
}
