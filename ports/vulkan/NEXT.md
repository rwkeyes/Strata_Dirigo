# Start here next session

Wave 1 is done and gated (`bash ports/vulkan/gates/run_gate.sh` → 14/14). This file is the shortest path to
wave 2, written while the source is in mind, so no re-discovery is needed.

## Next kernel: `rope_neox_apply` (`src/kernels/cuda/rope.cu` → `shaders/rope_neox.comp`)

It is the right next one because almost nothing about it needs deciding:

* **The table is already host-side.** `build_rope_table()` computes cos/sin in float64 on the CPU and the
  kernel is a pure rotation over it — so the port passes the SAME table buffer and the rotation is the only
  thing that moved. (The source's own reason for building it on the host is that device `pow`/`cos` in double
  are slow and may not agree with libm; that reason survives the port unchanged.)
* **One thread per row, no reduction, no shared memory** — `rows = batch * heads`, `head_dim = 256`,
  `n_rot = 64`. Grid stride is flat.
* **Partial rotation**: only `d < n_rot` is touched, the tail `[n_rot, head_dim)` is copied through.
* **NEOX pairing** `(i, i + n_rot/2)`, signs `(a*c - b*s, a*s + b*c)` — use `rope_neox_pair`'s arithmetic
  verbatim; the adjacent-pair ("NORMAL") reading is the classic silent-defect option and produces
  correctly-shaped scrambled output.

Interface to implement (the standing convention):

| | |
|---|---|
| binding 0 | `x` — float, readonly, `rows * head_dim` |
| binding 1 | `out` — float, write, `rows * head_dim` (may alias `x`; the host must not bind the same buffer twice) |
| binding 2 | `cos_tab` — float, readonly, `max_pos * (n_rot/2)` |
| binding 3 | `sin_tab` — float, readonly, `max_pos * (n_rot/2)` |
| binding 4 | `pos` — int, readonly, `rows` (DEVICE array — a graph bakes host scalars in, so this must stay a buffer) |
| binding 5 | `mtab` — int, readonly, the image path's multi-resolution position table |
| push | `{ int rows; int head_dim; int n_rot; }` |

The only open question is `mrope_pos(mtab, pos[r], i)` (`src/kernels/mrope.hpp`): read what it does when the
table is absent/identity, and either pass an identity table from the host or fold the two cases into the
shader. Decide it by reading the source, not by assuming.

## The gate case to add first (`harness/vk_gate.cpp`, `case_rope`)

* Reference: call the engine's OWN `rope_neox_pair` semantics — `out[i] = a*c - b*s`,
  `out[half+i] = a*s + b*c` — in double, on the CPU, with the table built by the same float64 loop as
  `build_rope_table` (copy it; it is 8 lines and it is the oracle).
* Compare with a **near-cancellation tolerance** (`max(|want| * 2^-7, (|a| + |b|) * 2^-9)`), not a plain
  relative one: `a*c - b*s` cancels for some pairs, and a relative-only test chases a transcendental
  difference that is not a defect. This exact trap is written up in `vulkan-compute-shader-porting`.
* **Assert the tail is untouched with ZERO tolerance** — fill `[n_rot, head_dim)` with sentinels and require
  them copied bit-for-bit. A port that rotates the full head_dim looks right on the first `n_rot` values.
* Adversarial cases: `i = half-1` (the last pair), a row whose position is 0, and a batch where two rows
  share a position.

## Then, in order

`kv_q8.cu` (127 lines, quantised KV append), `quantize_act.cu`, `ple.cu`, then the GEMV wave
(`native_bf16.cu`, `s_gemv.cu`, `s2_gemv*.cu`) — that last one is where throughput lives and where the
interface convention will need its first real test (weight layouts, not elementwise).

## Two things to do BEFORE writing more kernels

1. **Read `mrope.hpp`** and settle the multi-resolution table question above.
2. **Decide the `pos`/`mtab` buffers' story for the engine**: at some point the port must stop taking loose
   buffers and own a registry keyed by kernel name — that is stage 2's real content, and it is cheaper to
   design it now, while there are 9 kernels, than at 44.
