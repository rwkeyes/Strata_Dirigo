# Pending shaders - written, NOT verified, and therefore NOT shipped

Four files from `s2_expert_grouped.cu`: the grouped gate+up projection, the SwiGLU, the down projection, and the
S2-over-q8_0 row dot they share. They compile and validate. They live here rather than in `shaders/` because an
unverified shader in the build is one the build exercises without evidence.

## A RETRACTION, first, because the record matters

An earlier commit and its report claimed **"gu passed 3840/3840 against a double reference on its first run"**.
**That claim is withdrawn.** It was an artifact of the hole described below: `ratio > 1.0` is false for a NaN, so
when the kernel produced NaN the comparison reported "0 differ, worst 0" and the case called it a pass.

## The actual state

The `gu` kernel produces **NaN on the device** where the oracle, from the same bytes, produces a finite value:

    gu: worst err/tol 0, |oracle| mass 3.75517e+06 ; got[0] nan (isnan 1)
                                                     want[0] 1417.59375 (isnan 0), abs_sum[0] 5514.59375

So it is a real defect in the kernel or in how the case drives it, not a test artifact. What has been RULED OUT:

* the blob and activation bytes are finite - every scale is written from a real value through `f16_from_f32`, and
  the code bytes are integers masked to two bits, which cannot produce NaN;
* the offsets agree between the shader and the oracle: `O_D_CODES` 819200, `O_GU_SCALES` 1228800, the down
  regions ending exactly at `BLOB_BYTES` 1382400, with the row indexing `i*ROW_GU` and `i*SC_GU*2` matching;
* the integer path cannot overflow - `s` is at most 8*4*3*127 and `hx` at most 32*127.

Not yet ruled out: where the NaN enters. The next probe should dump the first row's eight code bytes, its scale,
and the activation block **as the device reads them** (a tiny debug output buffer), rather than reasoning about
which offset is right.

## What was learned from the source while drafting them

Two layout traps in `gu_kernel`, both of which produce FINITE, PLAUSIBLE numbers when wrong, and the second of
which the source says "with one hit it would have passed":

* the gate/up rows are INTERLEAVED - row-slot `i` is gate row `i/2` when even, up row `(i-1)/2` when odd;
* the output is GATE-MAJOR, because `swiglu` and `quantize_q8_0` both walk a contiguous range, so a per-hit
  [gate | up] layout makes them read hit 0's up rows where they wanted hit 1's gate rows.

And the primitive's own trick: the -1 bias is folded into an integer identity,
`sum((code-1)*x) == sum(code*x) - sum(x)`, so two exact integer sums are computed and subtracted.

## The gate hole this exposed, now closed

`ratio > 1.0` is FALSE for a NaN, so a NaN err/tol ratio passed every comparison in the harness. The NaN-safe form
`!(ratio <= 1.0)` is now in at all six sites in the shipped cases, and `gemv_bound` carries a 1e-30 floor (a row
whose oracle value and sum|terms| are both zero made it 0.0, and 0/0 is NaN). Both changes are in the gate and both
are worth having: a NaN result should never pass a check.

## THE FIRST STEP, WRITTEN OUT: `probe_gu_reads.comp`

Dump what the DEVICE reads, not what the source says it reads. `probe_gu_reads.comp` computes the same offsets
from the same push constants as `s2expert_gu`. It writes, for one row-slot: the 8 code bytes, the 2 scale bytes,
the activation block's 2 `d` bytes and its first 8 code bytes, then `dx`, `dw`, the chunk-0 integers `s` and `hx`,
and the chunk-0 term `dw * dx * (float)(s - hx)`.

Invocation: move the three kernels and the include back into `shaders/` (they are compiled only there - the build
is flat over `shaders/*.comp`), add the census arms for `s2expert_gu` and `s2expert_down` (barrier-only, no
subgroup ops), compile this one by hand, and dispatch one workgroup with the tier's push constants plus
`probe_slot`/`probe_row`. Compare each dumped field against the host's own reading of the same bytes. The first
field that disagrees is the bug.

IT IS UNRUN AND THEREFORE IT IS HERE. A diagnostic in `shaders/` would be exercised by the build as though it were
a kernel; an unrun artifact belongs where the build cannot mistake it for a verified one - which is the same rule
that put the four kernels above in this directory.

## THE PROBE RAN - and it eliminates the reads

`probe_gu_reads.comp` was run against the same data the failing case uses. **Every field agrees between the device
and the host's own reading of the same bytes:**

    code bytes    device 11 94 17 9a 1d a0 23 a6   host 11 94 17 9a 1d a0 23 a6
    scale bytes   device 00 28                      host 00 28
    act d bytes   device 00 34                      host 00 34
    dx device 0.25 host 0.25 ; dw device 0.03125 host 0.03125
    s device -643 host -643 ; hx device -648 host -648
    term chunk 0: device 0.0390625 host 0.0390625

So by the rule this probe was built to apply, THE ADDRESSING AND THE TERM ARITHMETIC ARE CORRECT, and the NaN
enters DOWNSTREAM of the reads: the reduction (`wg_sum`) or the store (`gate_up.v[base + r] = s`).

That is a real narrowing, and it also removes the most expensive hypothesis first - every offset in the tier's
addressing was hand-checked twice before this and agreed, which is exactly why the useful move was to observe it
rather than check it a third time.

WHAT THE NEXT PROBE SHOULD DO, in order of cost:

1. Probed fields for chunk 0 only. Extend to all 80 chunks and to `h > 0` (the failing case had n_hits = 3 and
   the probe had 1) - a row is a sum over 80 chunks and only the first was inspected.
2. Compare the REDUCED sum against the same sum computed from the per-lane partials, dumped one per invocation.
   `wg_sum` is used by six shipped kernels that all pass, which argues against it - but it is called here with
   `n_chunks` (80) SMALLER than the workgroup width (256), a configuration NO SHIPPED KERNEL USES: every other
   caller has more work per invocation than invocations, or loops over the whole row. That difference is worth
   ruling out before anything else, and it is the kind of "right for the data it was written against" mismatch
   this port keeps finding.
3. If both are clean, the store index `base + r` with `i` odd (`n_hits*FF + h*FF + r`) is the last candidate.

## ALSO RULED OUT: the harness's buffer lifetime

`Buf` is a VALUE type - `Ctx::alloc` returns a struct holding its own `bytes` and `VkBuffer`, by value, and each
allocation creates its own buffer. So taking `&buf` for a dispatch cannot be invalidated by later allocations, and
the tier's larger number of allocations is not the cause. (Checked because "the data is finite and the reads are
correct" left the infrastructure as the last thing that had not been looked at rather than the last thing that had.)

## WHAT THAT LEAVES, AND THE FIRST STEP IS NOW TO REPRODUCE RATHER THAN EXPLAIN

Finite data + correct addressing + a sound harness means the arithmetic in `s2_row_dot` cannot produce a NaN for
row-slot 0 - every chunk's read is in bounds and every term is finite. So before hunting further, the OBSERVATION
itself needs reproducing:

    re-run the tier's case and dump, alongside got[0]: the raw uints around that word, the workgroup count, and
    the first eight outputs. If got[0] is NaN again, print the buffer BEFORE the dispatch too - an unwritten word
    and a NaN from the kernel look identical at the comparison and are different bugs.

That is the same discipline this port has needed three times already: an oracle, a bound and a gate rule were each
right for the data they were written against and wrong for what they were run on, and in two of those cases the
"failure" was in the measurement rather than the kernel. A NaN that the arithmetic cannot produce is a claim to
reproduce before it is a bug to chase.

## THE NAN IS REPRODUCED, AND ITS SIGNATURE POINTS AT THE SHARED REDUCTION - NOT THE ARITHMETIC

Run with the output buffer pre-filled with `0x11111111` so an unwritten word would be visible:

    repro: 3840 words | still the fill pattern 0 | NaN 3840 | Inf 0
    first 8 words (hex): 7fdf6000 7fdf6000 7fdf6000 7fdf6000 7fdf6000 7fdf6000 7fdf6000 7fdf6000
    oracle: element 0: want 1417.59375

Three facts, and together they are decisive:

* **every element was written** - not one still holds the fill pattern, so this is not an unwritten buffer;
* **all 3840 hold the SAME NaN**, bit for bit, `0x7fdf6000` - which is not even the canonical quiet NaN
  (0x7FC00000), it is a payload that nothing in this kernel's arithmetic computes;
* and the oracle reads 1417.59375 from the same bytes.

**A value identical across 3840 elements and independent of the data cannot come from arithmetic over finite
inputs.** The probe already established that every read is in bounds and every chunk-0 term is finite, and every
scale and activation block in the blob is written from a finite value. So the NaN is not computed - it is READ,
from a shared-memory location before anything writes it, and shared memory starts with the same contents in every
workgroup, which is exactly why the value is identical everywhere.

The one location in this path that fits is the reduction's shared state - `rp_partial` / `rp_total` in
`common/wg_reduce.glsl`. That helper is used by six SHIPPED kernels and all six pass, so what is different here is
the configuration rather than the code: **`gu` calls it with `n_chunks` (80) smaller than the workgroup width
(256)**, so most invocations contribute 0.0 and most of the shared array is written by... nothing. Every other
caller has more work per invocation than invocations, or loops over a whole row.

### THE ONE-LINE EXPERIMENT FOR NEXT SESSION, IN THIS ORDER

1. **Confirm the mechanism without touching the shipped helper:** copy `wg_sum` into `s2_row_dot.glsl` under a
   different name with `rp_total` and `rp_partial` explicitly written (by invocation 0) before the first barrier,
   have `s2expert_gu` call the copy, and re-run this repro. If the NaN disappears, the mechanism is confirmed and
   the fix is the same initialisation in `common/wg_reduce.glsl`.
2. ONLY THEN change `common/wg_reduce.glsl` - it backs six passing kernels and a change there has to keep them
   green, which is a re-run of the whole gate rather than this one case.
3. If the NaN survives the copy, the remaining candidate is the store (`gate_up.v[base + r]`), and the same repro
   answers it: dump `base`, `r` and `s` for one odd and one even row-slot.

DO NOT "fix" the shipped helper first. Six kernels depend on it and this session ends with it untouched and the
gate green.
