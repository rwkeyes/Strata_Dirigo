# Start here next session

The gate **prints its own totals** (`bash ports/vulkan/gates/run_gate.sh`) — do not quote a number here, it went
stale twice in one day. Wave 1 (elementwise / conversions / norm / silu / gdn) and both GEMM paths and rope are
done and gated.

## Done since this file was last written

- `gemm_coopmat.comp` — the cooperative-matrix GEMM (matrix units). Verified on RADV on three shapes; a device
  without a usable config is a loud SKIP, never a pass.
- `gemm_fma.comp` — the plain-FMA GEMM with **no shape precondition**, so it is both the decode path (M=1 is
  structurally impossible for the CMA path) and the fallback where no CMA config exists (llvmpipe). The engine
  selects by **shape first**, device capability second.
- `rope_neox.comp` — NEOX partial RoPE, one thread per row, table passed in from the host.
- `kv_q8_append.comp` — the 8-bit KV append (the storage half of kv_q8). Verified **byte-exact over the whole
  destination image** against a transcription of the engine's own quantisation formula, so a wrong ROW is as
  visible as a wrong code: four cases (VRAM page resident; page absent, where a sentinel image must stay
  untouched; and the host copy in the identity layout, written unconditionally in both table states), covering the
  all-zero group (scale 0), all-negative, values on ± the group maximum, and a maximum past fp16 range.

## mrope: SETTLED, by reading the source rather than assuming

`mrope_pos(tab, pos, pair)` in `include/strata/kernels/mrope.hpp` is `tab ? tab[pos * 3 + pair % 3] : pos`:

- **null table (the default) means the identity** — the position the caller passes IS the rotary position. Text.
- set, `pos` is a **CELL index** into an int32 `[cells][3]` (t, h, w) table, and pair `i` takes sector `i % 3`.
  That is the vision path (qwen4exp's interleaved M-RoPE, `dimension_sections` 11/11/10/0).

A Vulkan descriptor cannot be null, so the kernel takes an explicit `mrope` flag instead.

**DEVIATION from this file's original interface table** (which specified push `{int rows; int head_dim; int
n_rot;}`): the push is `{rows, head_dim, n_rot, mrope}` = **16 bytes**. The alternative — a host-built identity
table — was rejected: it allocates and fills an N×3 array just to express "no table", and it routes the text path
through the vision path's indexing.

## The buffer registry: DECIDED, deliberately deferred

Keep loose buffers plus per-kernel push constants for now; do **not** build the registry yet. This file's own
prediction is the reason: the interface convention gets its first real test in the **GEMV wave** (weight layouts,
not elementwise), and a registry designed before that would encode elementwise assumptions as though they were
general. Its value is at ~44 kernels, and its caller is the engine integration (PORT-PLAN stage 4/6). Written
down so the deferral is a decision rather than an oversight.

## rope_neox: the gate case, and one tolerance decision worth carrying forward

The case follows this file's spec — host oracle is `build_rope_table`'s float64 loop **copied**, NEOX pairing from
`rope_neox_pair`, **bit-exact** tail sentinels, adversarial positions — with one change: the rotation bound is
2^-20 of the near-cancellation form, not the 2^-7 written here. Those constants were for kernels that *compute*
their angles on device (fast-math sin/cos differ in the last bits); this port **passes the table**, so the only
difference left is float-vs-double in one multiply-add. Measured worst err/tol ratio **0.11** — the tightened
bound still has ~9× headroom, while the loose one would have accepted a rotation wrong by ~100×.

Two traps hit while writing it, both worth not repeating:

- **`half` is a reserved word in GLSL.** `const int half = n_rot / 2;` fails to compile; it is `nhalf` here.
- **Bit-exactness is only for the parts the kernel COPIES.** The in-place case first compared every element
  exactly and reported 494/512 "failures" — the reference is computed in double, so the rotated elements *must*
  differ in the last bits. Split the comparison: exact for the copied tail, err/tol for the arithmetic. The
  shader was correct throughout.

## ONE FINDING TO CARRY INTO EVERY fp16-KERNEL PORT: `packHalf2x16` is NOT the engine's conversion

Measured on RADV while porting kv_q8. `packHalf2x16` **saturates to the largest finite half** (0x7BFF) where the
engine's `f16_from_f32` returns **infinity** (0x7C00), and its tie and subnormal rounding is not the engine's
nearest-even either. The effect is silent and small: a KV group whose maximum exceeds fp16 range gets a finite
scale instead of an infinite one, so its codes come out 127 instead of 0 — one group, in a branch normal data
never reaches. The gate caught it as exactly 73 differing bytes.

Use the port's own `f16_from_f32` (transcribed from `strata/kernels/f16_bits.hpp`, and held bit-exact by the
`f32_to_f16` case) in every kernel that converts. Reading a half BACK is a pure widening conversion with no
rounding to get wrong, so `unpackHalf2x16` is exact and stays.

## kv_q8 is COMPLETE (append, gather, round trip)

`kv_q8_append.comp` and `kv_q8_gather.comp` are both in and gated, and the pair is verified as a **round trip**
(floats -> append -> gather -> compare with the originals). Measured worst `|x'-x| / scale` = **0.53**, which is
what a correct round-to-nearest quantiser gives: below one code step, so the quantiser is optimal rather than
merely inside a loose bound. The round trip is also bit-exact against the host oracle, which only holds if the
scale the append WROTE is exactly the scale the gather READS.

The gather also settles a conversion question the append raised: a group whose scale lands in the fp16
**subnormal** range round-trips exactly, so `unpackHalf2x16` and the engine's `f32_from_f16` agree there - it is
only the PACKING direction where the builtin differs (see the packHalf2x16 section above).

The engine's fp16 converter now lives in ONE place, `shaders/common/f16_bits.glsl`, included by the three kernels
that convert (`#include` works in glslc, resolved relative to the including file). Three copies of a bit-exact
function were three chances to diverge, and `shaders/*.comp` deliberately does not glob into `common/`.

## quantize_act: the q8_0 trio is in, q8_K is next

`quantize_q8_0.comp` (ggml's bytes), `quantize_q8_0_scaled.comp` (this engine's CPU path), `dequant_q8_0.comp`.
Every comparison is `==` on the 34-byte block, because this is a reproduction of someone else's quantiser.

**TWO quantisers for ONE layout is the design, not duplication:** the first reproduces GGML's bytes (what the
pack holds, what `moe_hit_parity` checks), the second reproduces THIS ENGINE's CPU reference, because a hit and a
miss for the same expert must produce the same number. They round differently ON PURPOSE - and the gate proves
both rather than one: a synthetic block whose maximum is exactly 127 makes `d32` exactly 1.0, so eight of its
values sit on EXACT rounding ties, and the two rules disagree on **19 of 32 codes** there (half-even vs
half-away). A single oracle would have tested the wrong contract for one of them.

Two things worth carrying forward:

* **`precise` is the portable `__fmul_rn`.** GLSL's `precise` emits `OpDecorate NoContraction`, which is how a
  rounded multiply is pinned against being contracted into an FMA - the same thing the CUDA source does
  explicitly, and what `quantize_q8_K` needs for its `nearest_int` magic.
* **"One code step" is the wrong error bound when the stored scale is SUBNORMAL.** The codes round against the
  fp32 `d32` but the block stores fp16 `d16`, so the error is
  `0.5*|d32| + |q| * |d16 - d32|`. The second term is negligible while `d16` is normal and dominates once it is
  subnormal (the fp16 grid step there is fixed at ~6e-8, so a small `d32` is represented coarsely). Measured: the
  naive bound failed by 1.15x on exactly that block, and the two-term bound is hit at a ratio of 1.00 - tight, not
  loose.

## quantize_act is COMPLETE (five kernels, all byte-exact)

`quantize_q8_K.comp` and `dequant_q8_K.comp` finish the file: 292 bytes per 256 elements,
`{ f32 d ; int8 qs[256] ; int16 bsums[16] }`, the other half of `VEC_DOT_TYPE` and the format the numerically
SENSITIVE weights use. Byte-exact over all six blocks (1816/1816 bytes, guard region included) plus the 16 int16
sums per block.

Three traps that were transcribed from the source and are now pinned by the gate:

* `iscale = -127/max`, NOT -128. The -128 version sits in the source COMMENTED OUT with a note that IQ2_XXS needs
  it for an awkward AVX path; using it is a 0.79% scaling error, the same order as the quantisation step, so it
  produces an activation that looks fine and that ggml never sees.
* `max` is the SIGNED value at the largest magnitude and the comparison is STRICTLY greater, so a tie keeps the
  FIRST element. What that means in practice: `code(x) = round(-127 * x/mx)`, so the element HOLDING the maximum
  lands on -127 whatever its sign, and the opposite extreme lands on +127. A test written from the intuitive
  phrasing ("a positive max maps to -127") gets a negative maximum backwards - it did, and the byte-exact check
  overruled it. The rule is now its own four-element check.
* The rounding is `nearest_int`'s round-half-to-EVEN via the 12582912.0f magic, NOT `round`'s half-away. The tie
  block separates them on **159 of 256 codes** - the sharpest evidence yet that a rule was implemented rather
  than approximated.

The zero block is written in full (d, qs AND bsums). ggml's `continue` leaves bsums unwritten, which a dot product
cannot tell from zero but a byte comparison can; the source records the divergence deliberately and the gate holds
it. `min(127, v)` likewise has no lower counterpart - defensive, since `|iscale*x| <= 127` by construction.

### One comparison rule learned twice in this file

A kernel's f32 result can only be compared against a HIGHER-PRECISION oracle when the result is exactly
representable. A q8_0 dequant multiplies a 7-bit code by an 11-bit fp16 (18 bits - exact in f32, so a `double`
oracle agreed by luck). q8_K multiplies the same 7-bit code by a full 24-bit fp32 scale: up to 31 bits, NOT
representable, so the device's correctly rounded f32 product differs from the exact double one and 687 of 1536
values "failed". Round the oracle to f32 the way the kernel rounds before comparing, and confine bit-exactness to
where it is genuinely exact rather than merely convenient.

## `ple` is COMPLETE, and it is the port's first COMPOSITION test

Six kernels - `ple_gnorm`, `ple_gate`, `ple_bcast`, `ple_conv`, `add3`, `ple_history_advance` - plus a new shared
`shaders/common/wg_reduce.glsl` that the three reducers use (rms_norm_weighted, ple_gnorm, ple_gate), so the two
silent reduction defects have one home instead of three.

Every earlier case checked ONE kernel. Composition is where a convention shared between kernels stops being
checkable by either alone, and this one found three bugs - **all three in the test harness, none in a kernel**,
which is itself the finding: each kernel passed its own stage while the chain was wrong three times.

1. **The source's separate DESTINATIONS are not optional.** The CUDA calls are `gnorm(d_key -> d_key)`,
   `gnorm(hidden -> d_query)` and `gnorm(d_gated -> d_norm)`: two of the three write somewhere other than their
   input. The port's kernel is in place, so those become a copy then an in-place call. Skipping the copy for the
   query normalised `hidden` itself - which broke the gate and everything after it, while `gnorm` alone scored
   0.21 err/tol on its own stage. `ple_block`'s scratch comment says the same thing about its five hc_dim buffers
   ("two buffers of the same size look like an obvious saving and the only thing it saves is 40 KB"), and the test
   harness then reproduced that exact bug: reusing the convolution-weight slot for the key weights clobbered
   `w_conv` and showed up TWO STAGES LATER as 8.6e7 err/tol.
2. **A composition test must SNAPSHOT each stage as it passes.** Reading the buffers at the end of the chain
   compares whatever ran last into them: the `gated` buffer held the NORMALIZED values by then and the stage
   reported 0/10240 correct.
3. **A relative tolerance on a cancelling output measures CONDITIONING, not correctness.** The conv "failed" 16
   of 10240 elements at up to 6x the bound while the worst ABSOLUTE deviation anywhere in the array was 7.2e-07
   on inputs of order 1 - those 16 outputs had cancelled to ~5e-4. The bounds now carry an absolute floor
   relative to the input scale (a decade above the measurement), which still leaves a real misindexing four or
   five orders of magnitude clear. Print the worst absolute deviation next to the ratio, so this is measured
   rather than assumed.

Two smaller traps: **`out` is a GLSL reserved word** (an output qualifier) and cannot name a variable - the same
class as `half`; and `blah.comp.spv` is the wrong artifact name, because the gate strips `.comp` when it writes
the SPIR-V.

Verified: the key/query/normalized reductions sit at 0.19-0.21 of a 1e-6 relative bound, the gate at 0.06, and the
history advance is bit-exact over 92176 floats with the guard past the state untouched. The history advance is
in place and one thread owns a whole column - which is what makes the shift well-defined without a barrier.

## The GEMV wave: first two kernels in

`s2_gemv_q8.comp` (the S2 weight format over a Q8_0 activation - PLE's key projection) and `bf16_mmvf_f32.comp`
(the single-token BF16 matrix-vector product with an fp32 activation - PLE's value projection, and ssm_alpha/beta
plus the QSA indexer projections). Both are the source's own shape: one workgroup per output row, the source's
per-thread accumulator structure, then one workgroup reduction.

**THE FUSED MAC IS NOW A CHECKED PROPERTY, NOT A COMMENT.** `bf16_mmvf_f32` uses `fma()` to match the source's
`__fmaf_rn`, and the difference between a fused multiply-add and a separate multiply-then-add is *below the
reduction-order noise* - it cannot be caught numerically. So the gate's census rule now requires
`OpExtInst ... Fma` in that shader's SPIR-V, the same way it requires a subgroup reduction in the three reducers.
An FMA emitted as Mul+Add is more accurate, which is exactly why it is the wrong answer: it changes the last bits
of every term.

Two testing techniques from this wave worth keeping:

* **A layout probe makes a swap VISIBLE.** `bf16_mmvf_f32` reads each weight row as 32-bit pairs (element 2p low,
  2p+1 high). In a sum over random data, swapping the halves is a rounding-level error and invisible; one row is
  therefore built with its low halves at ~1e3 and its high halves at ~1e-3, which turns the same swap into a
  three-orders-of-magnitude error.
* **A ratio of exactly 0 is either exact agreement or a comparison of zeros.** `s2_gemv_q8` reported
  `worst err/tol 0` on all rows, and the numbers now printed beside it are what make that a result rather than a
  suspicion: `y[0] = -46391.4` with `max |y| = 46391.4`. The cases also FAIL a vacuous comparison (all expected
  values ~0) explicitly.

Also scored: distinct per-group scales and distinct per-block fp16 scales, so a wrong `q >> 4` or `(q*4)/32`
index is an O(1) error; negative S2 scales; a zero-scale activation block and a large one; every 2-bit code value
appearing; and the minimum legal geometry (n_in = 64, one S2 group; n_in = 2, a single weight pair).

Tolerances are ranked against the double oracle: the S2 dot products came out at ratio 0 (the dominant terms are
few and large), the BF16 products at 0.09 and 0.005 of a 1e-5 relative bound.

### THE REST OF THE WAVE - and its size, which is the number that matters

This is the biggest remaining area by a wide margin. Measured source sizes:

| source | bytes | what it is |
|---|---|---|
| `iq_kernels.cu` | 76,615 | the IQ-family quants and their dot products |
| `native_mmvq.cu` | 65,324 | the pinned CUDA quantized matvec (Q2_0/IQ3_XXS/IQ4_XS/Q8_0 paths) |
| `s2_expert_grouped.cu` | 64,436 | the MoE expert GEMV, grouped by expert |
| `s_gemv.cu` | 40,810 | the S-family (S2/S4/S8) GEMV and its sub-block variants |
| `s2_gemv_fast.cu` | 7,297 | the dp4a/fast S2 path (the source itself defers this) |
| `dequant_bf16.cu` | 12,762 | bf16 dequantisation |
| `native_bf16.cu` | 8,428 | **partly done** - the multi-row MMVF (1..8 tokens) is not ported |

~275 KB of CUDA in total, against ~40 KB ported so far this wave. **This is a scope decision, not a step**: the
next natural slice is the multi-row MMVF (small, and the prompt path needs it), then `s_gemv`'s S2/S4/S8 family
(the expert path), and `native_mmvq`/`iq_kernels` only if the port is meant to serve the quantized experts rather
than the S-family ones. Worth stating before starting rather than after.

Same shape as every case so far: read the CUDA source first, build the oracle from the engine's own function
(never from a description of it), sentinel every range the kernel must not touch, and give each branch of the
source an adversarial case. Tolerances come from measurement - print the err/tol ratio and keep it visible.
