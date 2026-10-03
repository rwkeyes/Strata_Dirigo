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

## Next: the second half of kv_q8, then the rest of the wave

`kv_gather_q8` — the dequantising reader (one thread per 4 values, `fp16(code * scale)` into the scratch the
attention kernels read). With BOTH entry points in place the pair can be tested as a **round trip** (append then
gather, compared against the original within the 8-bit bound), which is stronger than either half alone.

Then `quantize_act.cu`, `ple.cu`, then the GEMV wave.

Then `quantize_act.cu`, `ple.cu`, then the GEMV wave (`native_bf16.cu`, `s_gemv.cu`, `s2_gemv*.cu`) — that wave
is where throughput lives and where the interface convention gets its first real test.

Same shape as every case so far: read the CUDA source first, build the oracle from the engine's own function
(never from a description of it), sentinel every range the kernel must not touch, and give each branch of the
source an adversarial case. Tolerances come from measurement — print the err/tol ratio and keep it visible.
