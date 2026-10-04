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
