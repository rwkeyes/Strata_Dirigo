# Pending shaders - and the file this directory used to hold

**STATUS: EMPTY OF KERNELS. The four drafted files moved into the build on 2026-10-04 and are now gated.**

`s2expert_gu.comp`, `s2expert_down.comp` and `s2expert_swiglu.comp` are in `shaders/`, and `s2_row_dot.glsl` is in
`shaders/common/`; `case_s2expert_tier` in `harness/vk_gate.cpp` runs six arms over them (three gate/up, two down,
one SwiGLU) at the shapes that matter. The gate reports them separately - 126 passed / 0 failed / 0 skipped on
RADV, 120 / 0 / 1 on llvmpipe. **[SNAPSHOT at the time the tier landed; the suite has grown since - NEXT.md's
RESUME HERE carries the current state.]**

The filing rule that put them here still stands: an unverified shader in `shaders/` is one the build exercises as
though it were evidence, so a kernel stays in this directory until a case covers it. What is left here is
`probe_gu_reads.comp`, a diagnostic that was run once and is not a kernel.

## What the case found, and the diagnosis it replaces

**The NaN was a WRONG-BUFFER READ, not the reduction.** With `use_xscales` false, `s2_row_dot`'s `dx` was read
through `f16_at(xb)`, which takes a byte offset into the **expert blob**, where the CUDA's `f16_at` takes a
POINTER and was reading the **activation**. Every row's multiplier was therefore arbitrary code-byte content,
and code-byte pairs that decode as a NaN fp16 turn the row into NaN multiplied through 80 chunks. Fixed with
`act_f16_at`, which reads `act_b`.

**THE "SETTLED BY BISECTION" SECTION BELOW IS WRONG AND IS KEPT ONLY AS A RECORD.** It concluded that
`subgroupAdd` over a partially executing wave produced the NaN, and that the shared reduction was the fixed
point of the defect. With the `dx` fix in place the old subgroup reduction passes all six tier cases on RADV
(width 64) and on llvmpipe (width 8), at 80 chunks < 256 lanes and at 320 chunks. The reduction is a barrier tree
in the shipped tree, but as a portability choice for Intel and not as a bug fix.

Why the wrong diagnosis survived: the probe verified its own reads (`probe_gu_reads.comp` computes the offsets
itself and never calls `f16_at`), so it could not see the shader reading the wrong buffer; and the bisection
proved the raw per-lane dots were FINITE, which is not the same as correct - they were finite and wrong, because
no arm of that run compared anything against an oracle.

## The gate hole this exposed, now closed

`ratio > 1.0` is FALSE for a NaN, so a NaN err/tol ratio passed every comparison in the harness. The NaN-safe
form `!(ratio <= 1.0)` is in at all six sites in the shipped cases, `gemv_bound` carries a 1e-30 floor (a row
whose oracle value and sum|terms| are both zero made it 0.0, and 0/0 is NaN), and the tier's own case prints a
NaN/Inf census next to the worst ratio so an all-NaN result is reported as 3840 NaN rather than "worst 0".

## The layout facts the cases are built on (measured against the engine, not inferred from names)

The blob is four regions, in this order (`src/kernels/cuda/s2_expert_grouped.cu`):

    [0, O_D_CODES)             gate/up codes, row-SLOT i at i*ROW_GU                    ROW_GU = H/4
    [O_D_CODES, O_GU_SCALES)   down codes,     row r at O_D_CODES + r*ROW_D              ROW_D = FF/4
    [O_GU_SCALES, O_D_SCALES)  gate/up scales, row-SLOT i at O_GU_SCALES + i*SC_GU*2     SC_GU = H/64
    [O_D_SCALES, BLOB_BYTES)   down scales,    row r at O_D_SCALES + r*SC_D*2            SC_D = FF/64

At H 2560 / FF 640: 819200 / 1228800 / 1331200 / 1382400 bytes. Row-SLOT `i` is gate row `i/2` when even and up
row `(i-1)/2` when odd, and the output is GATE-MAJOR - both traps that produce finite, plausible numbers when
wrong, which is why the case's blobs give even and odd slots scales 1000x apart.
