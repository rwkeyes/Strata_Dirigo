# Pending shaders - written, NOT gated, and therefore NOT shipped

These four files are the resident expert tier (`s2_expert_grouped.cu`): the grouped gate+up projection, the
SwiGLU, the down projection, and the S2-over-q8_0 row dot they share. They compile and validate, and they are
moved here rather than committed because their gate case is not passing, and an unverified shader in `shaders/` is
a shader the build would exercise without evidence.

**WHAT IS ESTABLISHED.** The `gu` stage passed a full numeric comparison on its first run - 3840/3840 against a
double reference, which is the check that matters most here because the source records TWO layout traps in this
exact kernel, both of which produce finite, plausible numbers when wrong:

* the gate/up rows are INTERLEAVED (row-slot `i` is gate row `i/2` when even, up row `(i-1)/2` when odd), and
* the output is GATE-MAJOR, because `swiglu` and `quantize_q8_0` both walk a contiguous range - a per-hit
  [gate | up] layout makes them read hit 0's up rows where they wanted hit 1's gate rows. **"With one hit it
  would have passed"**, which is why the case uses three hits with permuted `slot_index` and gapped `dst_index`.

The `swiglu` stage passed too (1920/1920). The `down` stage did not, and the cause is in the TEST rather than
demonstrated to be in the kernel: its oracle was reading the wrong activation buffer at first (fixed - it must
read the quantizer's output, which is what the previous stage actually produced), and after that fix the
intermediate showed `d = NaN`.

**WHAT IS OPEN, AND IT IS A HOLE IN THE GATE RATHER THAN IN THIS FILE.** Making the comparisons NaN-safe
(`!(ratio <= 1.0)` instead of `ratio > 1.0`, since a NaN comparison is false and the old form silently ACCEPTED a
NaN) turns up NaN err/tol ratios in cases that currently report "worst 0" - and in this tier's `gu` stage every
ratio was NaN while the raw comparison of 3840 values had passed. So either a device output is NaN, or the
oracle's `sum|terms|` is, or `gemv_bound` is. Until that is settled, `!(ratio <= 1.0)` cannot be adopted
everywhere, and these kernels cannot be called verified.

Next session: settle the NaN's provenance (print `isnan` for the device's output, the oracle's value AND the
oracle's `sum|terms|` separately - the last one was not checked), then re-run this tier's case.
