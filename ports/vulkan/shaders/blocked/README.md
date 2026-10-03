# Blocked shaders (kept, not built)

## silu_fp64.comp — BLOCKED BY THE TOOLCHAIN, not by the port

`silu_inplace` in `src/kernels/cuda/elementwise.cu` computes `(double)x / (1 + exp(-(double)x))` and casts to
f32, because `ref/gdn.py`'s numpy does and the reference is the oracle.

GLSL has `double` (both the `double` type and double literals compile), but **no double overload of the math
builtins reaches SPIR-V on this host's glslang**:

    error: 'exp' : no matching overloaded function found

Reproduced with `glslc` (shaderc 2023.8 / glslang 14.0.0) and with `glslangValidator` 15.1.0, with
`GL_ARB_gpu_shader_fp64` require, `GL_EXT_shader_explicit_arithmetic_types_float64` require, and both
together.  The double TYPE is fine; the genDType builtin table is what is missing.

The port therefore ships `shaders/silu_f32.comp` and the gate MEASURES its gap against the double reference
(`vk_gate --selftest` prints `silu_inplace (f32 fallback) ... worst <r>`) rather than asserting it away, which
is what the engine's own parity test does with the same pair.

To restore the fp64 kernel: a glslang with the double builtins for SPIR-V, naga/rust-gpu, or a hand-written
double `exp`.  Not a blocker for the first slice — silu is elementwise and its tolerance is visible.
