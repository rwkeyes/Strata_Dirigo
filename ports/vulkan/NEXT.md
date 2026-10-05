# Start here next session

## THE 7900 XTX'S TWO FAILURES - A WRONG CASE BOUND AND A LAVAPIPE DRIVER BUG - **DONE AND VERIFIED 2026-10-04**

The gate run on `z820b` (RX 7900 XTX, RADV gfx1100, Mesa 26.0.8) found two failures and neither was explained.
Both are resolved; the box now reads **`radeon_icd 268/0/1`, `lvp_icd 260/0/3`, `nvidia_icd 263/0/2`** (the XTX's one
skip is the pre-existing M8 cooperative-matrix case, `prefill split`).  The Arc on `vega` is unchanged at
**272/0/0**; llvmpipe 260/0/3, radeon-iGPU 263/0/2.

**Verb: `cd ~/strata-vulkan-wt && STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 bash ports/vulkan/gates/run_gate.sh`**

| failure (before) | where | cause | fix | falsified by |
|---|---|---|---|---|
| `kv_q4 round trip ... 3071/3072, worst 0.414` | `radeon_icd` (the target) | the case's bound `|d|/2` is wrong for a `d*[-8,+7]` code range - a clipped element errs up to `|d|` | the CASE: per-element bound `|d|` + a host replay of the rule | inject the gather's `-8` offset away -> `FAIL 0/3072 worst 8.06` |
| `quantize_q8_0 (ggml bytes) 256/268, 12 bytes` | `lvp_icd` (Mesa 26.0.8) | lavapipe's `roundEven(double)` is half-toward-zero on ties, not half-to-even; the engine's `rint` is definite | the SHADER: explicit ties-to-even (`floor` + parity), not `roundEven` | inject half-toward-+inf -> `FAIL 257/268 worst 11` |

Both falsifications are run by **`gates/inject-verify.sh <name>`** (`q4-gather-offset`, `q8-round-half-up`), which
prints `ANCHOR MISSED` / `DID NOT COMPILE` rather than running a stale binary and restores the tree on every exit.

**Finding 1 - the case's bound, not the shader.**  `kv_q4_gather.comp` is a lone multiply (`float(kc-8)*kd`) and an
f16 conversion, so contraction was never in play and no `precise` was added (the handoff already said this; the new
diagnostic proves it: the failing element - **cell 5 head 0 dim 193**, from the box's own pool bytes `d16=0x3642
code=15 -> (code-8)*d16 = 2.73779` - has host-rule code `plain=15 fma=15`, and `dev-vs-rule mismatch 0` over all
3072 elements).  The rule is `d = mval/-8` from the SIGNED extreme, `code = clamp(trunc(x/d+8.5),0,15)`; the codes
are `d*[-8,+7]`, an ASYMMETRIC 16-level range whose +8 end does not exist, so an element that wants the +8 end
(`x/d = 7.96` here) is clipped to 15 and errs up to `|d|` - twice `|d|/2`.  The case now bounds **each element
against its OWN group's `|d|`** instead of comparing a global worst against a global bound, and it PRINTS the
offender (index, device value, expected, the group's `|d|`, the device's own `d16`/code and the host rule's code)
so it can never again report `worst` without `where`.  A second-order point worth keeping: the fixture comes from
a shared RNG that device-conditional SKIPS shift, so the same binary gives the Arc (no skips) a different fixture
than llvmpipe/radeon (which skip `gemm_coopmat`) - the old global-vs-global test passed on the Arc at ratio 0.995
and failed on the box at 1.22 for that reason.  The per-group form is immune.

**Finding 2 - a driver bug the shader leaned on; the case was right.**  `quantize_q8_0.comp` spells `roundEven`
exactly; on lavapipe / Mesa 26.0.8 that builtin is NOT ties-to-even for `double`.  Every one of the 12 wrong bytes
is in the exact-tie block (`amax = 127 -> d32 = 1.0`, so the double quotient IS the value): the device wrote
`3.5 -> 3`, `-3.5 -> -3`, `1.5 -> 1`, i.e. round-half-toward-zero, where the engine (`src/kernels/cuda/
quantize_act.cu`, `rint`) and the case's oracle (`std::nearbyint`) both want `4 / -4 / 2`.  So the engine's rule
admits only ONE answer, the case was not over-tight, and it was NOT loosened - the shader was brought off the
driver builtin.  The `quantize_q8_0_scaled` sibling was already independent of it (it uses the CPU's half-away
rule), and this was the only f64 use of `roundEven` in the port.  **Honest limit:** this is a property of that
Mesa build, measured here and not reproduced upstream; the shader no longer depends on it either way.

## THE SPLIT SAMPLER - the DEFAULT sampled path, and the ordered merge - **DONE AND VERIFIED 2026-10-05**

`sampler_split.comp`, from `sampler_split_part_kernel` + `sampler_split_merge_kernel` (src/kernels/cuda/sampler.cu),
with the tail from `sampled_tail_warp`.  This is the path `sample_tokens` takes by default for a sampled request
(the one-block `sampler_kernel` is the `STRATA_OLD_SAMPLER=1` reference), and it is a different SELECTION from that
kernel's: a row is cut into **4096-logit partitions**, each keeps its own `top_k`, and the ordered lists are
**merged** - exact, because the first k of a union lie within the first k of each part and a merge of ordered lists
is ordered.  The engine asserts its three sampled paths "pick the same token, bit for bit", and that parity is a
gate arm here.

**The shared machinery, so the port's three tails cannot drift.**  `common/sampler_tail.glsl` owns the chain's tail
(top_p cut, min_p prefix, temperature on the survivors, softmax, one Philox draw) in BOTH arithmetic variants -
`sampler_tail_f64` (the engine's, needs shaderFloat64) and `sampler_tail_f32` (the portable sibling, compiled by the
same file) - and the selection-list shared arrays.  `common/sampler_select.glsl` owns the partition+merge selection.
The f64 half is compiled only where a shader defines `PORT_SAMPLER_WANT_F64`, so the f32 sibling does not carry the
capability through an include it never calls.  The split and (next) the coupled merge share both files; the parity
this buys is BY CONSTRUCTION rather than a careful copy.

**A measured defect the multi-partition arms found, and it is a real class.**  The first merge was a forward pass
IN PLACE.  That is wrong: the write index `o = a + c` runs ahead of the read index `a` as soon as a partition entry
is taken (`c > 0`), overwriting running-list entries a later `a` still has to read.  It read `dev 4200 want 100`
on the top_p arm - a plausible token from a corrupted list - and only the arms whose top logits sit in DIFFERENT
partitions could see it (a one-partition row exercises no merge at all).  Fixed by merging into a separate array
(`sc_mrg_*`) and copying back; all five arms then read `bit-exact`, worst dev-vs-rule 0.

**Falsified.**  `gates/inject-verify.sh sampler-split-merge-drop-parts` makes the merge drain the running list
before the partition's (`else take_a = (a < ncur)`, so every partition after the first appends to the tail and is
lost): `FAIL  sampler_split: three partitions: the merge interleaves the lists   9/   16  worst 0`.

**Measured.**  vega, this commit: **Intel Arc (BMG G31) 340 passed / 0 failed / 0 skipped** (`run_gate.sh` exit 0),
intel_icd 340/0/0, llvmpipe 328/0/3, radeon-iGPU 331/0/2.  That is **+7 verdicts** on every implementation (five
split arms + the parity arm + the group arm).  The parity arm reads **60/60** (the split's token equals
`sampler_kernel`'s on every seed of every arm), which is the engine's own bit-for-bit claim, measured.

**What it still needs.**  The split is f64-only (its tail accumulates in double, like the engine's), so it carries
the same fp64 requirement as `sampler_kernel` - the portable f32 sibling settles that next.



## M-A: the standalone dequantiser's LAST TWO FORMATS - IQ2_XXS and IQ2_XS - **DONE AND VERIFIED 2026-10-05**

`iq_dequant_f32` covered 14 formats and REFUSED the remaining two `is_iq` types by name: IQ2_XXS (ggml type 16) and
IQ2_XS (17), because their grids (`iq2xxs_grid` 256 points, `iq2xs_grid` 512) were not in the port's generated
table.  Both now decode, and the hole is closed: **`check_port_map.py` still reads 77 symbols - 28 kernel, 49 host,
0 todo** (the formats live in a shader that was already `kernel`; no map row changed, and `make_port_map.py`
regenerates `PORT-MAP.tsv` byte-identically).

**What changed.** `tools/gen-iq-tables.py` extracts the two `uint64_t` grids from the engine's own
`third_party/ggml/ggml-common.h` verbatim, as low/high `uint32` pairs like `iq2s_grid` (no `shaderInt64`);
`harness/iq_grids.hpp` gains `kIq2xxsGrid[512]` and `kIq2xsGrid[1024]`; `common/iq_dequant.glsl` gains the two
`tid`-mapped bodies; `iq_dequant_f32.comp` and `iq_embed_rows.comp` declare the two new grid buffers (bindings 5/6,
the output moving to 7/8); and `harness/vk_gate.cpp` gains ONE ARM PER FORMAT whose oracle is a transcription of
the engine's `dq_iq2_xxs` / `dq_iq2_xs` (`src/kernels/cuda/iq_kernels.cu`).

**The two layouts, from the engine's decoder (not re-derived).** Both store `d` at byte 0 and `qs` as 32 `uint16`
at byte 2; the per-lane part is `qs + 4*ib`, i.e. 8 bytes (`ib = tid%8`, `il = tid/8`).
* **IQ2_XXS (66 bytes):** the four LOW bytes of the part are the four lanes' GRID INDICES (`aux8[il]`, one packed
  byte each); the four high bytes are `aux32`, whose top nibble is the block scale's high part and whose low 28 bits
  carry four 7-bit `ksigns_iq2xs` selectors.  `d = f16 * (0.5 + (aux32>>28)) * 0.25`.
* **IQ2_XS (74 bytes):** the part's four `uint16` words each hold a 9-BIT grid index (`& 511`) with the 7-bit sign
  selector ABOVE it (`>> 9`); the scale is `scales[ib]`'s nibble `4*(il/2)`.  `d = f16 * (0.5 + nibble) * 0.25`.

**A finding worth keeping: the fixture MARGIN was not serving the arm.** The first version of the IQ2_XS injection
(drop the grid index's 9th bit, `& 511` -> `& 255`) was **NOT FALSIFIED** - the arm stayed 768/768 green.  Cause: the
shared fixture's byte pattern `(k*37+13)&0xFF` makes every odd-offset byte EVEN, and the grid-index word's high byte
is always at an odd offset here, so bit 8 of `q2[il]` was **deterministically 0** at the dequant arm's seed: the
upper half of the 512-point grid was never indexed and the injection could not move a value.  The fixture now forces
that bit to split across both halves (`| (ib & 1)`) for IQ2_XS, so the whole grid is exercised - and only then does
the injection bite.  This is the "a fixture that cannot move under the wrong rule is decorative" rule, found again.

**Falsified (both, on vega).** `gates/inject-verify.sh` gained two registrations:
* `iq-dequant-iq2xxs-grid-index` (read the neighbour lane's grid byte) -> `FAIL  iq_dequant_f32: IQ2_XXS  300/768  worst 3.79e+05`.
* `iq-dequant-iq2xs-grid-high` (drop the 512-point grid's 9th bit) -> `FAIL  iq_dequant_f32: IQ2_XS  530/768  worst 3.82e+05`.

**Measured, bit-exact.** Both arms read `bit-exact 768/768` on the dequantiser and `1536/1536` on `iq_embed_rows`
(the same decode through the row gather).  Totals **+4 verdicts on every implementation**:
* vega: **Arc Pro B70 333 passed / 0 failed / 0 skipped**; llvmpipe 321/0/3; radeon-iGPU 323/1/2 where the single
  failure is the documented intermittent `budget: independent requery agrees` (the iGPU figure drift - recorded,
  not chased; a clean radeon run reads 323/0/2).
* z820b (7900 XTX): **radeon_icd 329/0/1** (the 1 skip is the pre-existing M8 `prefill split`), lvp 321/0/3,
  nvidia (K620) 324/0/2.  The box's script exits 1 on that pre-existing skip, as at HEAD.

## M-A: the decode path's last kernels - the ORDER, and 1/3: `cvec_apply`

**The order, derived from the decode path (`src/core/`), not invented.** The ten `todo` symbols were ordered by
two things: where the decode step meets them, and **what each one depends on**. Three carry NO unported
precondition, and they come first, in the order a decode step reaches them:

    1. cvec_apply         every layer, src/core/layer.cpp:1330-1336 (block_layer_post) - the steering vector
    2. gather_rows        MtpDrafter::bind, src/core/mtp.cpp:450 - the MTP draft head's token subset
    3. scatter_rows_f32   PeerExperts::run, src/core/peer_experts.cpp:241 - the peer experts' write-back

then the pair that shares the primitive the port does NOT have - `iq_dequant_f32` -> `iq_embed_rows`, which need a
standalone IQ/BF16 dequantiser where the port has only the FUSED `iq*_mmvq` form (the port map's own reason for
that row) - then the head's matvec `native_q5_k_f32`, which needs a **Q5_K dot** the `iq*_mmvq` family does not
contain (`native_q5_k_f32` = `quantize_q8_1` + `native_q5_k_mmvq`). The MoE half follows in the order the hit path
issues it: `moe_hit_select` -> `moe_hit_grouped_s2` (with `moe_grouped_s2` its resident sibling) -> `moe_hit_add`.
**The embedding kernels are first IN TIME in a token but are the largest single increment** - a dequantiser over
the 15 `is_iq` types plus BF16 - and nothing else on the list depends on them; that, and not convenience, is why
these three are the first increments.

**What was ported.** `cvec_apply.comp`, from `cvec_kernel` (`src/kernels/cuda/cvec.cu:46`); the CONTRACT is the
engine's own test, `src/kernels/cvec_parity.cpp`, and that is what the gate reproduces arm for arm:

| arm | rule | required by |
|---|---|---|
| project (mode 0) | `h' = h - s (h.v) v`; s = 2 REFLECTS the component, s = 1 removes it | the engine: **1e-4 absolute** vs double |
| add (mode 1) | `h' = h + d` | the engine: **BITWISE** |
| a layer without a direction (s == 0) | R untouched | the engine: **BITWISE** |
| switch off, no pending write | R untouched | the engine: **BITWISE** |
| switch off, pending write | `h' = fma(bo, w, h)`, `w = 2 sigmoid(inj/hc)` | **BITWISE** at `inj == 0` |
| on + write | the write, THEN the projection | **1e-4** vs the oracle |

Two lessons from this port's own history are load-bearing here:

* **The CUDA uses `fmaf` on both paths**, so the GLSL calls `fma()` and is deliberately **NOT `precise`**: the
  fused op rounds ONCE where a separate multiply-and-add rounds twice, and `fmaf` is the engine's answer. (The
  opposite of the embedding gather, where the engine wanted the TWO-rounding form and `precise` was the fix.)
* **One workgroup per (stream, token), reduction by the barrier tree** (`common/wg_reduce.glsl`) - the port's
  design rule, because the target is Intel where the driver picks the subgroup width per kernel. The tree's
  summation ORDER is therefore the port's, which is exactly why the two PROJECTIONS cannot be bitwise and use the
  engine's own 1e-4, while every arm without a reduction in it is bitwise. `run_gate.sh`'s barrier census list
  gains `cvec_apply` (its SPIR-V carries the tree's barriers; a shader that lost them must fail).

**The write arm is bitwise BY CONSTRUCTION, not by luck**: `inj == 0` makes `2 sigmoid(0/hc) == 1.0` exactly for
any `hc`, so the fold is `fl(bo + h)` and the case demands it bitwise - and separately requires that the write
actually MOVED values (`write w=1: 6144 of 6144 values moved`), so a no-op cannot pass.

**Falsified.** `gates/inject-verify.sh cvec-apply-drop-scale` drops the per-layer factor `s` from the reduced dot
(`dot = wg_sum(dot) * s` -> `dot = wg_sum(dot)`), and the named case fails:
`FAIL  cvec_apply: project removes s(h.v)v  12/14  worst 0.844` (the correct form reads `worst 4.73e-07`).

**Measured.** vega, this commit: **Intel Arc (BMG G31, default) 278 passed / 0 failed / 0 skipped**, intel_icd
278/0/0, llvmpipe **266/0/3**, radeon-iGPU 269/0/2 (the iGPU's `budget: independent requery agrees` is the
documented intermittent driver-figure drift - 2 fail / 1 pass in 3 consecutive runs of this binary, recorded, not
a port defect). z820b (7900 XTX): **radeon_icd 274/0/1** (the 1 skip is the pre-existing M8 cooperative-matrix
case), lvp 266/0/3, nvidia (K620) 269/0/2; the box's script exits 1 on that pre-existing skip, as it does at HEAD.

**Honest limit.** The engine's `sigmoidf_` is `__expf`-based and the port's is the driver's `exp`, so with
`inj != 0` the port's `w` differs from the CUDA's in the last bits. That is why no arm compares a non-zero-`inj`
write bitwise: the rule (direction, magnitude, the `hc` division, the fold and its order) is gated; the last bits
of `__expf` are not claimed.

## M-A 2/3: `gather_rows` - the MTP draft head's opaque-byte row gather

`gather_rows.comp`, from `gather_rows` / `gather_rows_kernel` (`src/kernels/cuda/verify_kernels.cu:394`, `:303`);
its decode-path call site is `MtpDrafter::bind` (`src/core/mtp.cpp:450`), which builds the draft head by keeping
only the token subset `dvocab` of the head's weight table, one whole `row_bytes` row at a time.

    for i in [0, n*row_bytes):  r = i / row_bytes;  o = i - r*row_bytes
                                dst[i] = src[ ids[r]*row_bytes + o ]

**The CUDA's element WIDTH is a performance choice, and it is the shape a port breaks.**  It templates on `uint4` /
`uint32_t` / `uint8_t` by `row_bytes % 16 / % 4`, and `row_e = row_bytes / sizeof(E)` then stands in for
`row_bytes` in BOTH the divisor and the source stride.  A port that keeps `row_e` in one place and `row_bytes` in
the other computes a plausible, WRONG row - and nothing about the output looks wrong.  This kernel indexes in
BYTES, where both alignments are one mapping, and the gate says so with two arms: `row_bytes = 64` (the `uint4`
path) and `42` (the byte path).

**The fixture is built so an identity gather cannot pass.**  `ids` is a DERANGEMENT (`ids[r] != r` for every r),
and every source byte is distinct (`(k*31 + j*7 + 1) & 0xFF`), so a wrong source row and a wrong offset each land
on a different byte.  64 guard bytes past the written range are compared against the sentinel the fixture wrote,
so a kernel that wrote past its count fails as well.

**Falsified.**  `gates/inject-verify.sh gather-rows-identity` ignores `ids[r]` (`src[ids[r]*row_bytes + o]` ->
`src[r*row_bytes + o]`): `FAIL  gather_rows: 16-byte-aligned rows  64/576` (the faithful form reads 576/576).

**Measured.**  vega: Intel Arc **281 passed / 0 failed / 0 skipped** (default and intel_icd), llvmpipe
**269/0/3**, radeon-iGPU **272/0/2** (its budget-requery drift again: 1 fail in 3 runs of this binary).  z820b:
radeon_icd (7900 XTX) **277/0/1**, lvp 269/0/3, nvidia (K620) 272/0/2; the box's script exits 1 on its
pre-existing M8 skip, as at HEAD.

**No honest limit to state**: the rule is a pure byte copy and the case compares bytes, so the arms are exact
rather than tolerance-bearing.

## M-A 3/3: `scatter_rows_f32` - the peer experts' row write-back

`scatter_rows_f32.comp`, from `scatter_rows_f32` / `scatter_rows_kernel` (`src/kernels/cuda/elementwise.cu:263`,
`:255`); call site `PeerExperts::run` (`src/core/peer_experts.cpp:241`).

    for r in [0, n):  for i in [0, width):  dst[ rows[r]*width + i ] = src[ r*width + i ]

Two rules, both of them traps this port has already met once elsewhere:

* **`r` is a POSITION and `rows[r]` is the DESTINATION row** - the same two-roles confusion the KV gather's
  `ids[id]` carries.  The fixture is a PERMUTATION WITH NO FIXED POINT with distinct row contents, so an identity
  write fails on every row.
* **A destination row `rows[]` does not name is NOT WRITTEN AT ALL.**  The destination is filled with a sentinel
  and the unnamed rows - plus a `strays` count of zero - must survive, which is what catches a port that writes
  every row.

**The CUDA's precondition is its VECTORIZATION, not the rule**: it casts to `float4` and therefore `exit(1)`s
unless `width % 4 == 0` and both pointers are 16-byte aligned.  This kernel reads f32 words and carries neither
requirement - and that is GATED rather than asserted: the second arm runs `width = 6`, where the CUDA would refuse.

**Falsified.**  `gates/inject-verify.sh scatter-rows-identity` drops the row indirection
(`dst[rows[r]*width + i]` -> `dst[r*width + i]`): `FAIL  scatter_rows_f32: permutation ... 511/3072`.

**Measured.**  vega, this commit: **Intel Arc 284 passed / 0 failed / 0 skipped** (default and intel_icd),
llvmpipe 272/0/3, radeon-iGPU 275/0/2 - `run_gate.sh` **exit 0**.  z820b: radeon_icd (7900 XTX) **280/0/1**,
lvp 272/0/3, nvidia (K620) 275/0/2; the script exits 1 there on the PRE-EXISTING M8 cooperative-matrix skip (a
skip is not a pass), exactly as it does at HEAD.

**M-A is now 3 of the ten.**  Still `todo`: `iq_dequant_f32`, `iq_embed_rows`, `native_q5_k_f32`,
`moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`, `moe_hit_grouped_s2`.  The next increment in the derived order
is the **`iq_dequant_f32` -> `iq_embed_rows` pair**: the standalone IQ/BF16 dequantiser the embedding path needs
(15 `is_iq` types plus BF16 - the port has only the FUSED dot form today), a format at a time with one codebook's
values per arm, then the row gather built on it.

## M-A 4/10 and 5/10: `iq_dequant_f32` + `iq_embed_rows` - the STANDALONE IQ/BF16 dequantiser, and the row gather on it

**The pair the decode path needs first, and why it is a pair.** `iq_dequant_f32` is the engine's standalone
I-quant / BF16 decoder - `dq_dispatch<float>` through `dequant_flat_kernel` (`src/kernels/cuda/iq_kernels.cu:1643`,
`:1848`) - and `iq_embed_rows` (`embed_rows_kernel`, `:1830`) is the token-embedding gather built on it: rows
`tokens[t]` of a GGUF table, `row_bytes` apart, dequantised to fp32. The port had only the FUSED `iq*_mmvq` dots;
this is the standalone form, and `iq_embed_rows` needs no decode of its own.

**THE ENGINE'S DECODER IS THE ORACLE, and the port keeps its thread mapping rather than re-deriving a layout.**
Every offset, shift and scale comes from the CUDA body that `iq_dequant_f32` and `iq_embed_rows` actually launch,
and the GLSL keeps the engine's own `tid` 0..31 mapping (`il = tid/8`, `ib = tid%8`, the inner `j` loop, and the
output position each case writes) - the port's worst bug class is a re-derived element->byte map that agrees with
itself. `shaders/common/iq_dequant.glsl` holds that decode ONCE, shared by both shaders; each launches with 32
active lanes inside a 256-lane workgroup so `run_gate.sh`'s single workgroup-size rule still holds.

**Coverage, stated exactly.** Every `is_iq` type whose grid the port's generated table carries, plus BF16 - **14
formats**: BF16, IQ4_NL, IQ4_XS, Q8_0, Q5_0, Q5_1, Q2_0, Q4_K, Q5_K, Q3_K, IQ3_XXS, IQ3_S, IQ2_S, IQ1_M. **IQ2_XXS
(ggml 16) and IQ2_XS (17) are NOT ported**: their grids (`iq2xxs_grid`, `iq2xs_grid`) are not in
`harness/iq_grids.hpp`, and neither shader claims them - an unproven format is recorded, not approximated.

**One arm per format; the oracle is a transcription of the RULE.** The case reproduces each `dq_*` body on the
host, element by element, and compares the device's 256 values against it. The fixture puts a KNOWN, FINITE fp16
scale in every place the format stores one - a random byte pattern would assemble an inf/NaN scale and the case
would measure that instead of the decode - and the scale AND the pattern vary with the superblock and with the row
seed, so a decode that read the wrong superblock or the wrong row lands on a DIFFERENT value, not an equal one.
`iq_embed_rows` uses a DERANGEMENT token list (`tokens[t] != t`), so an identity gather fails on every token, and
gates the row STRIDE the caller passes with two wide-row arms (`n_embd = 512`, two superblocks per row).

**Measured.** vega, this commit: **Intel Arc (BMG G31, default) 314 passed / 0 failed / 0 skipped**, intel_icd
314/0/0, llvmpipe **302/0/3**, radeon-iGPU 305/0/2 - `run_gate.sh` **exit 0**. z820b: radeon_icd (7900 XTX)
**310/0/1** (its 1 skip is the pre-existing M8 `prefill split`), lvp 302/0/3, nvidia (K620) 305/0/2; the box's
script exits 1 on that pre-existing skip, exactly as it does at HEAD. That is **+30 verdicts on every
implementation** (14 dequant + 16 embed arms), all green, and the device's values are **bit-exact** against the
host oracle on all 30 (worst dev/tol 0).

**Falsified, and one change the falsification forced.** `gates/inject-verify.sh iq-dequant-iq1m-grid-high`
(misplace the IQ1_M grid high bit, `<< 8` -> `<< 7`) -> `FAIL  iq_dequant_f32: IQ1_M  310/768  worst 2.34e+05`.
`iq-embed-rows-identity` (gather the row at the POSITION instead of the token) -> `FAIL  iq_embed_rows: BF16 ...
0/3072  worst 6.58e+04`. The first targets a shared INCLUDE, which has no `#version` and cannot be compiled alone,
so `inject-verify.sh` now compiles the shader that INCLUDES a changed `common/` file (and says so) instead of
reporting `DID NOT COMPILE` - the same rule the port learned when three injections silently tested nothing.

**Honest limit.** The row base is computed in 64-bit and truncated to the 32-bit byte index the storage buffers
take, so a table larger than 4 GiB is outside this arm; the engine's `size_t` row arithmetic would still be
correct, and the port's embedding tables are far below that. `iq_dequant_f32` also stops at the 14 formats above -
IQ2_XXS and IQ2_XS stay `todo` in `PORT-MAP.tsv` for the grid reason stated.

**M-A is now 5 of the ten.** Still `todo`: `native_q5_k_f32`, `moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`,
`moe_hit_grouped_s2`. The next increment in the derived order is **`native_q5_k_f32`** (= `quantize_q8_1` +
`native_q5_k_mmvq`, the head's Q5_K matvec - `q5_q8_dot` in `src/kernels/cuda/native_mmvq.cu`, a pinned
integer-dot order the `iq*_mmvq` family does not contain).

## M-A 6/10: `native_q5_k_f32` - the native head's Q5_K matvec, and the ONE dot whose scale and min are PACKED

`native_q5_k_f32.comp`, from `native_q5_k_mmvq_kernel` / `q5_q8_dot` (`src/kernels/cuda/native_mmvq.cu:239`,
`:196`).  `native_q5_k_f32` (:1311) is `native_quantize_q8_1` PLUS this, and the decode head's type-13 path is
exactly that pair (`src/core/native_head.cpp:78`), so this is the last matvec between the port and the head.

**The rule is a pinned integer-dot order, and its shape is the point.**  One workgroup per row; each lane owns a
strided set of the row's `(Q5_K block, 16-value part)` items, where part `p = iqs/2` in 0..15,
`bq8_offset = 2*(p/4)`, `nib = p%4` - the engine's own `kqs = VDR*(tid % (QI/VDR))` decomposition.  Every part's
VALUE is the source's f32 expression term for term (two `dp4a` chains: `dot1` against the codes, `dot2` against
ones for the `sum(x)` term); the reduction that adds the parts is the port's barrier tree, so the comparison is a
double reference, not bit-for-bit.

**`aux` is what the `iq*_mmvq` family does not have.**  Q5_K's 12-byte `scales` packs SIX 6-bit scales and SIX
6-bit mins, and which six bits are the scale and which the min depends on the block half: `j = bq8_offset/2`,
`jm = j&1` (which `uint16` to read), and `hi`, the all-ones mask that switches between groups 0..2 and 3..5.  The
port transcribes the source's masks and shifts literally rather than re-deriving them - a "tidier" read is a
plausible wrong number, which is the class the falsification targets.

**THREE TRAPS, ALL FOUND BY THIS CASE AND ALL SILENT.**
1. **The activation's `u` is read out of the q8_1 block, not the weight.**  `q5_q8_dot` reads four int8 from
   `bq8i->qs` and four from the Q5_K block's `qs`; collapsing the two 4-byte helpers into one buffer (the source's
   pointer argument loses its buffer when the port makes it a byte offset) sent every activation read into the
   weights - the same `f16_at` defect class, and it read a legal address.  The port keeps two named helpers
   (`q5_i32` for the weight, `q5_a_i32` for the activation).
2. **A Q5_K block's activation group is `kby = kbx*8`, not the row start.**  Each 256-value Q5_K block dots the
   EIGHT q8_1 blocks at `kbx*8`; using the row base for every block computed a finite, wrong row.
3. **`unpackHalf2x16` needs the full 32-bit half2.**  `unpackHalf2x16(q5_u16(blk))` passes only the LOW half, so
   `.y` (the block's `min`) came back 0 and the whole `- min*sumf_m` term vanished - a per-row error of 1-13%,
   every value finite and close.  The first two traps were caught by the arm failing at 1e2-1e5 of tolerance; this
   one needed the shader's own `sumf_m` exposed to be seen, and it is recorded because "close but wrong" is the
   signature a value comparison should have caught and a per-part dump did.

**Fixtures.**  `q5_k_fill_blob` varies EVERY byte of `scales`, `qh` and `qs` (so both `qh` nibble halves, both
`aux` branches and every `dp4a` byte position are exercised), with `dm = (d, min)` FINITE (a random byte pattern
would assemble an inf half and the case would measure the inf instead of the decode).  Three arms: `n_in=2560,
n_out=8` (1280 parts over 256 lanes), `n_in=256, n_out=1` (16 parts, most lanes idle), and `n_in=10240, n_out=4,
ncols=2` (the widest Q5_K pitch and a two-column activation).  Each compares the row against a host double
reference, requires finite outputs, a guard region past the buffer, and a live oracle mass.

**Falsified.**  `gates/inject-verify.sh native-q5k-aux-half` drops the packed-scale half switch
(`him = (j>=2) ? ~0 : 0` -> `him = 0`), so every group reads groups 0..2's fields:
`FAIL  native_q5_k_f32 (n_in=2560, n_out=8, ncols=1, ...)  0/8  worst 5.62e+04`.

**Measured.**  vega, this commit: **Intel Arc 317 passed / 0 failed / 0 skipped**, llvmpipe **305/0/3**,
radeon-iGPU 307/1/2 (its `budget: independent requery agrees` drift again - the documented intermittent figure).
z820b (7900 XTX): **radeon_icd 313/0/1** (the 1 skip is the pre-existing M8 cooperative-matrix case),
lvp 305/0/3, nvidia (K620) 308/0/2; the box's script exits 1 on that pre-existing skip, as at HEAD.
That is **+3 verdicts on every implementation**.

**Honest limit.**  The arms supply the q8_1 activation directly (the boundary every other `iq*_mmvq` arm draws),
so the oracle is INDEPENDENT of the quantiser rather than reading bytes the quantiser wrote; the quantiser half of
`native_q5_k_f32` is gated byte-exactly by `case_quantize_q8_1`, and the composed quantiser->dot chain is the
`case_quant_prefill_chain` pattern, already gated for iq2s.  What is NOT claimed here is the engine's exact
4-warp summation order - the parts' values are, the order that adds them is the port's tree, and the tolerance is
the double reference's.

**M-A is now 6 of the ten.** Still `todo`: `moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`,
`moe_hit_grouped_s2`.  The next in the derived order is **`moe_hit_select`**.

## M-A 7/10: `moe_hit_select` - which routed experts are resident, and the ROUTING POSITION each hit fills

`moe_hit_select.comp`, from `hit_select_kernel` (`src/kernels/cuda/s2_expert_grouped.cu:623`), whose decode-path
call site is `src/core/session.cpp:866` - the token graph's first step.  One warp; `lane < k`: `e = ids[lane]`,
`s = (0 <= e < n_expert) ? res_row[e] : -1`; `hit = ballot(s >= 0)`; a hit at `at = popc(hit & ((1<<lane)-1))`
writes `slot[at] = s` and `dst[at] = lane`; lane 0 writes `count = popc(hit)`.

**TWO RULES, and the second is the one a paraphrase drops.**  Only a RESIDENT expert is a hit (`res_row[e]` is
negative for one the cache does not hold), and the hits compact in ASCENDING routing order - and `dst[at]` is the
ROUTING POSITION (`lane`), NOT the expert or the slot.  Writing the slot there hands the downstream add a
cache-order row instead of the router's; it is the same two-roles confusion `gather_rows`'s `ids[r]` and the KV
gather's `ids[id]` carry.

**A ballot is a subgroup op, so the port writes the compaction SERIALLY from one lane.**  The rule is exact
integer bookkeeping, so a different shape computes the same output - and this is why the shader carries no
barrier, no subgroup op and no atomic, which `run_gate.sh`'s census requires of it (it is deliberately NOT on the
barrier whitelist).  The output buffers are sized for the CAPACITY (k = 32), the grid the engine records.

**Fixture and arms.**  Four arms: the decode shape (10 routed, 8 experts, 5 resident with SCRAMBLED slots, one id
out of range, one negative), `k = 32` (the CUDA's maximum), `k = 1`, and a fully non-resident row (zero hits).
Entries past `count` must keep a sentinel (so a kernel that wrote the whole capacity fails), and the `res` buffer
is padded past `n_expert` with a RESIDENT value so dropping the `e < n_expert` bound counts an out-of-range id as a
hit.  Scrambled slots make `slot != position` on every hit, which is what makes `dst` falsifiable.

**Falsified.**  `gates/inject-verify.sh moe-hit-select-residency` drops the residency test (`if (s >= 0)` ->
always write), so every non-resident entry is written with `slot = -1` and the count includes it:
`FAIL  moe_hit_select (k=10, n_expert=8, decode shape, ...): 6 hits  21/32`.

## M-A 8/10: `moe_hit_grouped_s2` - the per-hit S2 expert ENTRY, which is a COMPOSITION

`moe_hit_grouped_s2` (`src/kernels/cuda/s2_expert_grouped.cu:579`) is not one kernel: it launches gate+up
(`gu_kernel`/`gu_pair_kernel`), the SwiGLU, `quantize_q8_0` on the intermediate, then the down projection.  All
four are already in the port (`s2expert_gu`, `s2expert_swiglu`, `quantize_q8_0`, `s2expert_down`) and each is
gated against its OWN oracle by `case_s2expert_tier`; **this increment is the case that gates the WIRING between
them**, which no single-kernel case can see - the gate-major layout `s2expert_gu` writes is what the SwiGLU and the
quantiser both assume, `quantize_q8_0` reads the first `n_hits*FF` floats the SwiGLU left, and `s2expert_down`
reads them back at `h*(FF/32)*34`.  So no new shader, and the PORT-MAP row names the four it composes.

**The chain runs on the device end to end** (`gu -> swiglu -> q8_0 -> down`, 3 hits, scrambled `slot_index` and
`dst_index`), and the oracle is INDEPENDENT of the device's quantiser and down projection: (a) the UP rows (the
half the SwiGLU leaves alone) are compared against the blob, pinning the gate-major base; (b) the device's
intermediate must DECODE to the device's own post-SwiGLU floats at the same position - a value check, not a byte
check, because the stored `d16` is the fp32 scale in fp16; and (c) the down rows are compared against a double
reference from the DEVICE's intermediate at the scrambled `dst_index`.  A defect in (b) or (c) also has to be seen
with the other half present, which is exactly what a wiring case is for.

**One silent failure the fixture produced, recorded because it is a class.**  The gu arm's blob makes the GATE rows
~1e5 (its `*1000` is what makes a gate/up mis-pairing visible to `case_s2expert_tier`); through `silu(gate)*up`
that overflows fp16, so the intermediate's `d16` became inf and EVERY down row was NaN - and the case's own oracle
computed NaN too, so the two AGREED.  The case counts a NaN output as a failure (correctly), which is what kept it
from being a green run over garbage.  The chain now carries its own modest scales.

**Falsified.**  `gates/inject-verify.sh moe-hit-grouped-s2-hit0-intermediate` makes `s2expert_down` read hit 0's
intermediate for every hit (`x_off = 0`): `FAIL  moe_hit_grouped_s2 (...)  473/867  worst 1.3e+06`.

**Measured (this commit covers 7/10 + 8/10).**  vega: **Intel Arc 322 passed / 0 failed / 0 skipped**, llvmpipe
**310/0/3**, radeon-iGPU 313/0/2 (`run_gate.sh` **exit 0**).  z820b (7900 XTX): **radeon_icd 318/0/1** (the 1 skip
is the pre-existing M8 cooperative-matrix case), lvp 310/0/3, nvidia (K620) 313/0/2; the box's script exits 1 on
that pre-existing skip.  Positive controls: `moe_hit_select` adds **4** verdicts and `moe_hit_grouped_s2` **1**.

**M-A is now 8 of the ten.**  Still `todo`: `moe_grouped_s2` (the grouped S2 MoE, `s2_expert_grouped.cu:1099`) and
`moe_hit_add` (the hit accumulator, `:1138`) - the batch after this one, and then M-A is closed.  The port map
reads **77 decode-path symbols - 26 kernel, 49 host, 2 todo** and passes.

## M-A 9/10 + 10/10 (ONE commit - the two share `vk_gate.cpp`): `moe_grouped_s2` + `moe_hit_add` - **M-A CLOSED**

**These two land as one commit because their cases and their registrations are the SAME file region of
`harness/vk_gate.cpp`** - the precedent `4/10 + 5/10` and `7/10 + 8/10` used, and it is stated here for that reason.

### 9/10 `moe_grouped_s2` - the GROUPED S2 expert entry: a COMPOSITION, but two of its four launches are NEW shaders

`moe_grouped_s2` (`src/kernels/cuda/s2_expert_grouped.cu:1099`) is the resident sibling of `moe_hit_grouped_s2`:
the same four-launch chain (`gu -> swiglu -> quantize_q8_0 -> down`).  **Whether it "composes" was CHECKED before
writing anything, as the sibling's own increment instructs.**  The middle two launches ARE the per-hit chain's
kernels (`s2expert_swiglu`, `quantize_q8_0`), but the gu and down halves are the GROUPED kernels
(`gu_grouped_kernel` / `down_grouped_kernel`, `:783` / `:845`), and those are NOT what `s2expert_gu` /
`s2expert_down` implement: the per-hit kernels take ONE activation row and a `slot_index[h]` into a single base
buffer, while the grouped ones take a LIST OF GROUPS - each with its own blob (`grp_ptr[g]`) - and EVERY ENTRY reads
ITS OWN token's activation row (`ent_tok[e]`) and writes entry-major (`ent_dst[e]`).  Neither is expressible with
the per-hit kernels, so this increment adds **`s2expert_gu_grouped.comp`** and **`s2expert_down_grouped.comp`**
(the S2 dot itself is the unchanged `common/s2_row_dot.glsl`) and reuses the other two; the port-map row names all
four.

The grouped `grp_ptr` is an array of DEVICE POINTERS and Vulkan has none, so the port takes the SAME interface
decision `native_gu_iq2s.comp` already made: ONE weights buffer plus a per-group BYTE OFFSET table (`grp_off[g]`),
so `grp_ptr[g] + off` becomes `grp_off[g] + off`.  The group count lives on the device, so the grid strides in y
(`for (g = WorkGroupID.y; g < ng; g += gl_NumWorkGroups.y)`) exactly as the source does - and the case arms the
stride three ways.

**Two rules the case pins that the per-hit sibling has no counterpart for**: the per-ENTRY activation token, and the
CAPACITY-strided gate-major split (the up half starts at `cap_entries*n_ff`, NOT the live entry count, because the
SwiGLU that follows pairs over `cap_entries*n_ff`).  The fixture is four groups (one EMPTY), six entries, capacity 8
above the live count 6, per-entry tokens, scrambled destinations, and three grid.y arms (== the count, BELOW it -
the stride - and above it).

**The oracle is independent of the device's quantiser and down step**: (a) the UP rows (the half the SwiGLU leaves
alone) against the group's blob at the ENTRY's token - pins grp_off, the per-entry activation and the capacity
split; (b) the device's intermediate must DECODE to the device's own post-SwiGLU floats; (c) the down rows against
a double reference from the DEVICE's intermediate at the scrambled `ent_dst[e]`.  The gate/up scales are overridden
to a MODEST value here: the shared `s2expert_blob` makes gate rows ~1e5, which overflows fp16 through `silu*up` -
the NaN trap the per-hit increment recorded.

**Falsified.**  `gates/inject-verify.sh moe-grouped-s2-entry-token` makes every entry read token 0's activation
(`tok = ent_tok.v[e]` -> `tok = 0u`): `FAIL  moe_grouped_s2 (...)  5270/5638  worst 3.78e+05` (the faithful form
reads 5638/5638, worst 0.0525).

### 10/10 `moe_hit_add` - the hit accumulator

`add_hits_kernel` (`src/kernels/cuda/s2_expert_grouped.cu:666`), the last step of the device hit path:
`parts[dst[h]*n_embd + i] += hit_out[dst[h]*n_embd + i]` for every LIVE hit.  Two rules, both paraphrase traps:
the row is `dst[h]` (the ROUTING POSITION, not `h` - the same two-roles confusion the KV gather's `ids[id]` and
`moe_hit_select`'s `dst[at]` carry), and the operator is `+=` (parts holds what the CPU left there).  The live count
is on the DEVICE, so the grid is the CAPACITY and rows past it return unwritten.

Four arms: count < cap (rows past the count and rows the list skips keep a sentinel), count == cap, `n_embd = 37`
(the workgroup's stride loop), and count 0 (nothing may be touched).  Every named row carries a NON-ZERO prior
value, so `+=` -> `=` is caught on the first row.

**One defect the arms produced, recorded because it is a class - and it was the ARM, not the kernel.**  The count-0
arm first read `FAIL ... 10304/10304 worst 0`: "every element bad" beside "worst deviation 0" is self-
contradictory, and the cause was the shared liveness guard - "the kernel MOVED something" - which the count-0
contract INVERTS (its claim is that nothing moved).  The guard is now `moved == 0` for count 0 and `moved > 0`
otherwise.  The arm was NOT loosened: count 0 still requires the whole buffer bit-identical to the sentinel, and it
is LIVE only when nothing moved.

**Falsified.**  `gates/inject-verify.sh moe-hit-add-accumulate` drops the accumulate (`+=` -> `=`):
`FAIL  moe_hit_add (cap=4, count=3, ...)  2624/10304  worst 0`.

### M-A IS CLOSED

`python3 ports/vulkan/tools/check_port_map.py` reads **77 decode-path symbols - 28 kernel, 49 host, 0 todo**: the
decode path's `todo` column is **ZERO**.  (The generator `tools/make_port_map.py` had drifted - it still classified
the two landed IQ rows as `todo` - and is fixed in this commit, so it regenerates `PORT-MAP.tsv` exactly.)

**Measured, this commit.**  vega: **Intel Arc (BMG G31) 329 passed / 0 failed / 0 skipped** (`run_gate.sh` exit 0),
intel_icd 329/0/0, llvmpipe **317/0/3**, radeon-iGPU **320/0/2**.  z820b (7900 XTX): **radeon_icd 325/0/1** (the 1
skip is the pre-existing M8 `prefill split (aligned + remainder)` case - "this device offers no M8 cooperative-matrix
config"), lvp **317/0/3**, nvidia (K620) **320/0/2**; the box's script exits 1 on that pre-existing skip, as at HEAD.
Positive controls: `moe_grouped_s2` adds **3** verdicts and `moe_hit_add` **4** (322 -> 329 on the Arc).



## STAGE 3: recorded command buffers (the CUDA-graph replacement) - **DONE AND VERIFIED 2026-10-04**

**Closed the same day the API was written.**  `case_recorded_step` (`harness/vk_gate.cpp`, six verdicts, one
commit) now exercises the whole record/replay API, and it was **proven able to fail** before it was trusted - the
two things this port's own rules ask for ("a case that only compiles is not evidence"; "a case that cannot fail
is not an oracle"):

| verdict | what it proves | Intel BMG G31 | llvmpipe | RADV (AMD iGPU) |
|---|---|---|---|---|
| `record: three dispatches` | `recorded_dispatches() == 3` at the end of the recording, `has_recording()` after `record_end_and_submit()` | PASS | PASS | PASS |
| `record: chain is byte-exact` | the ORACLE itself: three chained copies through `dispatch()` reproduce the source bit for bit | PASS | PASS | PASS |
| `record: equals single-shot` | the recorded chain == the same chain through `dispatch()`, 1024/1024 bytes | PASS | PASS | PASS |
| `record: replay reads new bytes` | new host bytes into the recording's source, then `replay_recorded()` - no re-record, no `dispatch` - produce the new bytes | PASS | PASS | PASS |
| `record: replay fixture differs` | the control for the arm above: the new source really differs (1024/1024 elements) | PASS | PASS | PASS |
| `record: replay ignores new binds` | a LATER single-shot dispatch on the SAME pipeline - which rewrites that pipeline's shared descriptor set - does not change what the replay reads | PASS | PASS | PASS |

**FALSIFICATION, which is why arm C is evidence rather than decoration.**  With `record_dispatch`'s `fresh_set`
flipped to `false` (one shared descriptor set for the whole recorded step - the trap the encoding comment names),
**three of the six verdicts fail**: `equals single-shot` 0/1024, `replay reads new bytes` 0/1024, `replay ignores
new binds` 0/1024, totals 162/0/1 -> **159 passed / 3 failed / 1 skipped**.  The flip was reverted; the file is as
committed.  So the case detects the exact defect the per-dispatch-set rule exists to prevent - and arm C is the
one that sees a shared set that a re-recording bug would otherwise hide.

One finding from writing the API, worth keeping:

* **A recorded step needs ONE DESCRIPTOR SET PER DISPATCH.**  The host-side `vkUpdateDescriptorSets` happens at
  RECORD time while the dispatches execute at SUBMIT time, so one shared set leaves every dispatch in the step
  reading whatever the LAST one bound - silently wrong output, the same class as the grouped-expert wave's single
  pointer standing in for two buffers.  `encode_dispatch`'s `fresh_set` parameter allocates a set per recorded
  dispatch (a real backend pools them).
* **`rec_fence_` IS DESTROYED BY `~Ctx` now** (it was not, and that was the loose end this block used to name).
  The command buffer goes with `cmd_pool_`; the fence does not, and a backend that owns a long-lived recording has
  to release it explicitly.

### The design, as implemented - kept because the REASONS are the useful part

The engine's decode step is a fixed sequence of dispatches re-issued every token, and a CUDA graph is how the
engine avoided re-issuing it.  The Vulkan equivalent is ONE command buffer recorded once and RE-SUBMITTED.  What
this needs, worked out against the code on 2026-10-04 and written the same day (the rule that a case must exist is
the reason the API and its case landed together rather than a half-API first):

* `harness/vk_compute.cpp`'s `dispatch()` (line ~653) allocates a command buffer per call with
  `ONE_TIME_SUBMIT`, records ONE dispatch, inserts a shader-write -> host-read barrier, submits, waits, frees.
  That is the gate's shape and it stays the single-shot path.
* **Refactor**: lift the encoding half (the `pipes_` lookup for layout/set, the descriptor update, the binds, the
  push constants, `vkCmdDispatch`) into a private `encode_dispatch(cb, pipe, bufs, push, push_bytes, groups,
  groups_y, chain_barrier)`, and have `dispatch()` call it.  `chain_barrier` adds a compute -> compute barrier,
  which a multi-dispatch STEP needs (one kernel's output is the next one's input) and a single dispatch does not.
* **New API on `Ctx`**: `record_begin()`, `record_dispatch(...)`, `record_end_and_submit()`, `replay_recorded()`,
  `recorded_dispatches()`, `has_recording()`.  Private state: one persistent `rec_cb_` + `rec_fence_` created
  once, `recorded_`, `recording_`, `have_recording_`.  Two details differ from the single-shot path ON PURPOSE:
  the buffer is NOT `ONE_TIME_SUBMIT` (it is submitted more than once) and the host-read barrier goes at the END
  of the recorded step, not after each dispatch.  `replay_recorded()` re-submits the recorded buffer and waits -
  it must not re-record anything, which is the whole point.
* **The gate case** (`case_recorded_step` in `vk_gate.cpp`; the verdicts are named `record: ...` and the six of
  them are listed in the table at the top of this file): chain the harness's own trivial COPY kernel three times
  (idempotent and byte-exact, which is why it is the right kernel for this).  Record the chain once; run the same
  three copies through the single-shot `dispatch()` path as the reference; require the two outputs byte-identical
  (and the reference itself byte-exact against the source, so the oracle is checked too).  Then PROVE the replay
  is the recording rather than a re-record: write NEW bytes into the source buffer on the host, `replay_recorded()`,
  and require the destination to equal the new source.  The third arm - an interfering single-shot dispatch on the
  same pipeline between recording and replay - is the one that catches a SHARED descriptor set; flipping
  `fresh_set` to false was measured to fail 3 of the 6 verdicts, which is what makes the case an oracle rather
  than an agreement.  No HIP device is needed to see any of it.
* **Why this oracle and not the plan's**: PORT-PLAN stage 3 says "compare tokens to the HIP path", which this box
  can no longer produce (see the plan's note).  The single-shot path is the substitute, and it is a stronger
  oracle for this particular question, because it isolates exactly the thing that changed - the recording - with
  everything else held equal.

## STAGE 4: device-local memory, staging and fit accounting - **DONE AND VERIFIED 2026-10-04**

**What it was for.** Stage 1-3's layer allocates host-visible coherent memory on purpose (a correctness device:
staging plus fences would only add ways for the GATE to be wrong).  The engine's backend cannot do that - the
things that live on the card go in memory with no mapping at all, and every byte in and out goes through a
staging buffer.  That path now exists, is exercised by the gate, and the engine's VRAM plan is fitted against the
driver's own numbers instead of being described.

**What the three implementations actually offer**, printed by the case rather than assumed (this is the reason
the layer enumerates types instead of answering "is there device-local memory?"):

| device | memory types | device-local | of those, UNMAPPABLE | staging lands in |
|---|---|---|---|---|
| Arc Pro B70 (ANV) | 7 | 5 | **3** (types 0/1/4 - real VRAM), the other 2 mappable (the BAR) | heap 1, system RAM (34.7 GiB) |
| AMD iGPU (RADV) | 11 | 6, **all of them on the heap RADV calls device-local** (it flags GTT that way) | **3** (types 0/1/7) | a mappable heap-0 type |
| llvmpipe | **1** | 1 | **0** - its single type is device-local AND mappable | its one heap (nowhere else exists) |

**The arms, all six on all three implementations** (the totals below are the whole suite):

| verdict | what it proves | Arc | llvmpipe | RADV |
|---|---|---|---|---|
| `stage: device-local chosen` | the flags on the buffer match the memory type, not what we asked for | PASS | PASS | PASS |
| `stage: vram is not mappable` | the chosen VRAM type has NO mapping where the device has one | PASS | **SKIP** (no such type) | PASS |
| `stage: staging is mappable` | the transfer buffer is HOST_VISIBLE + HOST_COHERENT | PASS | PASS | PASS |
| `stage: staging charged by heap` | a transfer buffer is charged by the heap it lands in, not by what it is for | PASS (host) | PASS (VRAM: single heap) | PASS |
| `stage: device round trip` | pattern -> staging -> copy kernel -> staging -> host, byte-exact | PASS | PASS | PASS |
| `stage: forced-staging trip` | the same with the staging path FORCED, so it runs where a mapping exists | PASS | PASS | PASS |

`plan_fit` is also gated as a PURE function (five arms, no device): fits inside budget, an exact fit is a fit,
the item that crosses is named and the walk stops, a droppable cache is dropped rather than fatal, and neither an
empty plan nor a zero-byte budget "fits" (both would be assertions over an empty input).  On the device the plan
is fitted against the driver's figure, printed, and cross-checked by a recount written in the case:

```
INFO  vram plan                    budget 28.64 GiB (driver free minus reserve)
INFO                                 weights resident   wanted  14.32 GiB
INFO                                 expert cache       wanted  14.32 GiB  (droppable)
INFO                                 kv cache           wanted   3.58 GiB
INFO                               FITS: 17.90 GiB resident, 14.32 GiB dropped in 1 item(s), first overflow -1
```

**FALSIFICATION, and one honest negative.**  A four-byte offset error in the upload's copy (`c.srcOffset = 4`)
makes BOTH round trips fail (0/1024, 174/0/1 -> 172/2/1), so the round trip is not vacuous.  **Deleting the
post-copy barrier changes NO verdict on the Arc or on llvmpipe** - measured, not assumed - so the case proves the
transfer (direction, offsets, mapping, and a barrier whose absence is observable), NOT that each barrier is
necessary; the barriers are required by the spec and are not gated on these drivers.  Recorded here so the next
reader does not mistake a green run for coverage of that defect class.

**A bug the new arm caught in the new code, worth keeping as a lesson.**  `has_nonvisible_device_local()` was
written as "a device-local type exists" and then used to decide whether to assert that the chosen type is
unmappable.  llvmpipe has exactly one type, it is device-local AND mappable, so the arm FAILED there (171/0/2 ->
171/1/2) and reported a defect the port did not have.  The accessor now records what the SELECTION did (pass 0 of
the two-pass choice succeeded).  The general form: an accessor that re-derives a property from the table it was
chosen out of can disagree with the choice, and the disagreement surfaces as a failure on one implementation only.

**The account rule, stated once** (it is in the header too): every `alloc()`/`alloc_device()` is charged to the
VRAM account whatever heap the driver puts it in - conservative on purpose, so the refusal rule cannot change with
which memory type a driver happens to prefer - and a staging buffer is charged by the heap it lands in, because a
transfer buffer is not model memory.  Stage 4 did not relax the display contract: the refusal now names the
account it refused in.



## STAGE 5 (started): the matrix path on Intel - **THE ARC'S MATRIX UNITS ARE REACHABLE, AND WERE BEING MISREPORTED**

**The finding, measured with the driver's own property list** (`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`,
all three implementations, 2026-10-04):

| device | extension | configs | the floating-point one |
|---|---|---|---|
| Intel BMG G31 (XMX) | advertised, revision 2, feature true | 6 | **M8 N16 K16**, f16/f16 -> f32, subgroup scope |
| RADV Raphael (RDNA2) | **not advertised** | (14 returned by an entry point the driver does not advertise) | all M16 N16 K16 |
| llvmpipe | not advertised | none | - |

So the port's `gemm_coopmat` case demanded **M16** N16 K16 - a rule written against the departed Radeon's list - and
therefore reported the Arc as having "no usable config", which reads as "the matrix units are unreachable here".
They are not: **BMG's op is 8 rows wide.** The tile is a property of the DEVICE, per generation, and the port now
selects the pipeline by shape for exactly that reason.

**What changed.** `shaders/gemm_coopmat_m8.comp` (the sibling of the M16 kernel, 8x16 tile, same dispatch shape and
bounds guard); `DeviceInfo::cm_m/cm_n/cm_k` hold the SELECTED config (M16 preferred where a device offers both,
because that is the kernel this port verified on a Radeon, else M8); `gemm_shape_ok` takes the tile as parameters,
so the precondition is one rule with the device's numbers in it rather than two rules that drift; the case picks
the pipeline by shape, PRINTS which one it ran, and gained a PROMPT-SHAPED case (128x256x256, a real prefill tile
grid rather than 64x64).

**Evidence.** Intel: **180 passed / 0 failed / 0 skipped** - the Arc now runs the matrix path and no longer skips
anything, including all four coopmat shapes against a double-precision reference (worst relative error 0.0101 on
the prompt-shaped grid, and that is on elements whose absolute error is under the 1e-4 floor - the case prints
both).  llvmpipe 172 / 0 / 2, RADV 175 / 0 / 1 (both still skip coopmat: their drivers do not advertise the
extension, and skipping a device that does not advertise an extension is correct - unlike skipping one that does).

**Falsification, and it is the interesting kind.** Forcing the M16 kernel on the Arc - ignoring the device's tile -
does NOT fail pipeline creation.  It **runs and computes wrong numbers**: 2560/4608 elements correct in the square
case, 16896/33280 in the prompt-shaped one, 176 passed / 4 failed.  So a wrong tile selection is a silent wrong
answer on this driver, not a loud one, which is precisely why the pipeline has to come from the property list and
why the case prints the shape it used.

**What this settles for the rest of stage 5.** The prefill GEMM on the target card has matrix units at **M8 N16
K16**, so the port's own GEMM tile is an 8-row tile with a plain-FMA fallback for devices with no config (llvmpipe,
and any driver that does not advertise the extension) - and the selection has to be per device, never per vendor.



## STAGE 5: the PREFILL GEMM - **DONE FOR THE LAYOUT THE ENGINE CALLS, WITH ITS OPERAND CONVERSION**

**What "the GEMM" means here.**  The engine's prompt path calls `Gemm::f16` / `Gemm::bf16`
(`src/prefill/gemm.cu`), whose cuBLAS call is `(CUBLAS_OP_T, CUBLAS_OP_N)` with m = N, n = T, k = K:

```
Y[T x ldy] = X[T x K] . W[N x K]^T          f16 operands, f32 accumulate
```

**The WEIGHT is the transposed operand** - an output row is an output feature, and the reduction runs over the
contiguous axis of both inputs.  That is a different layout from `gemm_coopmat.comp` / `gemm_fma.comp` (which are
`C[MxN] = A[MxK].B[KxN]`, the shape and no-CMA fallbacks), and getting it backwards is the classic SILENT GEMM
bug: every number stays plausible.  So it is stated in the kernel, each loader says which layout it reads, and the
gate compares at shapes where `N != K` - a transposed read cannot survive that.

**Three new kernels.**

| shader | what it is |
|---|---|
| `bf16_to_f16.comp` | the engine's own operand conversion (`prefill/gemm.cu`'s `bf16_to_f16_kernel`): widen, **CLAMP finite values past fp16's range to +-65504 rather than letting them become Inf** (the #540 rule), then round-to-nearest-even with the port's gated converter. This is the route from the engine's bf16 storage to the matrix units on Intel too, where the driver's list has no bf16 config. |
| `gemm_prefill_f16_m8.comp` | the GEMM on XMX at the tile Intel actually has (8x16x16). The body lives in `common/gemm_prefill.glsl`, where the tile is a compile-time knob - so the RADV/WMMA M16 sibling is a three-line file when a Radeon is in the box. Not shipped: an untested tile variant is exactly the claim this port does not make, and the FMA kernel below covers those devices correctly. |
| `gemm_prefill_fma.comp` | the same contract with **no shape precondition at all** - one invocation per output element, f16 operands read as fp16 and widened (exact), no cooperative matrix, no subgroup op, no barrier. It is the decode row, the ragged prompt, and the devices without a usable config. |

**Why there are two, and why the case tests the SPLIT.**  The tile path has no ragged edge: `tiles_t = t / TM`
rounds DOWN, so rows the tile cannot cover are silently not computed.  A real prefill of a prompt whose length is
not a multiple of the tile therefore runs BOTH kernels - aligned rows on the matrix units, the remainder on FMA -
and the case exercises exactly that (`T = 37 = 4 x 8 + 5 rows`) rather than each half alone, including a check that
the tile dispatch left the remainder rows untouched.

**Evidence (Arc Pro B70, this box):** **189 passed / 0 failed / 0 skipped**, ten new verdicts - `bf16_to_f16`
bit-exact over 1024 patterns with the clamp measured as exercised on 437 of them, three FMA shapes (1x64x64 decode
row, 17x13x5 ragged with `ldy > n`, 40x64x96 mid-prompt), three matrix-unit shapes (one tile, 64x512x512,
128x256x256 with `ldy > n`), and the split.  `ldy > n` is exercised on purpose: a kernel that assumed `ldy == n`
would smear the columns between n and the stride, and the case leaves those columns NaN and requires them to
survive.

**Falsification.**  Reading the weight RowMajor instead of ColumnMajor - the transposed-operand bug itself - fails
all four matrix-unit arms (`0/128`, `14/32768`, `16/32768`, `320/2368`, totals 189/0/0 -> 185/4/0) while the FMA
arms stay green, which is what makes this a test of the LAYOUT rather than of the arithmetic.  And the clamp's own
arm counts the entries the clamp changes, so it cannot be deleted with the case still green.

**A DEVICE-LAYER GAP THIS FOUND, and it blocks stage 6 rather than stage 5.**  The engine reaches a row slice of Y
by POINTER arithmetic (`f16(tc_x_, tc_w_, Y + t0 * ldy, ...)` in `gemm.cu`, and `X + t0 * K` for the activations in
row slices), because cuBLAS takes pointers.  The harness binds **offset 0** on every descriptor, so the case has to
give each half its own X and Y buffer instead of pointing both at one.  A Vulkan backend needs
`VkDescriptorBufferInfo::offset` (or one buffer per slice, which is what the engine's own conversion buffers
amount to) before it can serve a real prompt.  Recorded here because it is invisible until someone tries to pass a
slice.



## STAGE 5b: the QUANTISED prefill path - **and a finding that needed no new kernel**

**Reading the engine first paid off again.**  The prompt path has TWO routes to quantised weights, and only one of
them is a GEMM-shaped kernel:

* **`Gemm::native`** (what the prefill calls): it does NOT use an MMQ kernel at all.  It **dequantises** the ggml
  blocks to f16 (`strata::kernels::dequant_f16`) and then calls the ordinary f16 GEMM - the same one this port
  gated an hour earlier.  So the quantised prefill needs an IQ *dequantiser*, not a new GEMM.
* **`iq_mmvq`** (the expert tier's multi-token path): the row kernels, which the engine calls with a `ncols`
  argument and specialises at 1 / 2 / 4 / 8 columns (`mmvq_multi_kernel`).

**The port's row kernels already take `ncols`** - `iq1m_mmvq.comp` and its siblings loop
`for (c = 0; c < pc.ncols; ++c)` - so the quantised multi-token path needed **no new kernel**: it needed arming.
The existing arms sat at ncols 1 and 2, i.e. the single-token and barely-multi-token cases.  Added, for all seven
of the resident model's formats: an arm at **ncols = 8** (the widest count the engine specialises for, so this is
the real multi-token expert path) and, for IQ2_S, one at **ncols = 32**, past the engine's specialisations, where
an activation-row stride that is right for a narrow row and wrong for a wide one shows up.

**A composed case for the interface, because the halves agreeing with their own oracles does not make the wiring
right:** `case_quant_prefill_chain` runs the DEVICE's `quantize_q8_1` and feeds its output to the DEVICE's
`iq2s_mmvq`, with the oracle reading those same bytes.  IQ2_S is the format to hang it on: 20 of the model's 48
layers store their gate/up experts in it.

**AND THE CASE'S FIRST ORACLE COULD NOT FAIL - which is the finding worth keeping.**  The injection was the
obvious one: delete the quantiser's per-column block stride so every column's blocks land in column 0's region.
The quantiser's own case caught it (`ncols=2` arm: 2105 of 5760 bytes) - and **the chain case PASSED**, because an
oracle that reads the device's own bytes follows the device's own mistake: both sides then read the same misplaced
data, and the columns that read nothing compare equal to an expectation of nothing.  Fixed by asking a question the
bytes cannot answer by themselves - **per-column liveness**: with random activations, a column that read nothing
returns zeros, so the case now requires that no column comes back all-zero.  With that, the same injection fails
the case (7 of 8 columns dead) and the real tree passes.  The general form: **an oracle built from the thing under
test is blind exactly where that thing is systematically wrong**, and a liveness/coverage assertion is what closes
it - the same class as the existing `mass > 1e-3` guard, one level down.

**What is still missing for quantised prefill:** the IQ **dequantiser** (`dequant_f16` / `iq_dequant_f16`, which
covers ggml types 16/17/18/21/22/29 plus the Q2_0, IQ4_NL/IQ4_XS and k-quant families through a second path).  The
port's per-format dot includes already encode each block layout FOR A DOT, so a hand-written dequantiser would be a
second copy of that layout - the shape of bug this port keeps finding.  The non-duplicating route, recorded here
for whoever takes it: **derive the dequantiser from the dot by a one-hot activation** (a q8_1 block with a single
code set to 127 and d = 1/127 gives exactly 1.0 at one position and 0 elsewhere), so the dequantised value IS the
dot and there is nothing to drift; slow, but this is a verification path, and the same trick gives one dequantiser
per format for free.



## THE ATTENTION BLOCK, KERNEL 1: the short-step decode attention - **the engine's contract, re-derived**

**What was ported.**  `native_flash_attn_short_step` (`src/kernels/cuda/native_flash_attn.cu`, called from
`src/core/layer.cpp`), which the engine's own header describes precisely enough that nothing had to be guessed:
geometry **Q24 x 256, KV2 x 256**, one query/sequence, **scale = 1/16**, no ALiBi/softcap/sink; q/out contiguous
`[24, 256]` f32; k/v `[capacity, 2, 256]` f16 with only the first `width` rows initialised; an optional **additive**
f16 mask of 256 logits broadcast over the heads, `-inf` meaning masked.  The 24 query heads share 2 KV heads - the
GQA grouping the CUDA encodes as `kv = head / 12`.

**What is NOT claimed, up front.**  This is not the pinned CUDA bit for bit, and it does not try to be: that kernel
is a VECTOR flash attention pinned to one compiled binary - 128 threads, a lane/float2 mapping, an `__shfl_xor`
tree over 8 lanes, a streaming online-max pass, and an fma order chosen to reproduce that binary's rounding (its
comments say so outright).  The port's reduction rule bans subgroup ops anyway.  So the port carries the ALGORITHM -
score, additive mask, softmax, weighted sum - onto one workgroup per query head with one output dim per thread, and
measures the difference against a double-precision reference instead of asserting it away.  Engine-token parity for
this kernel is a stage-6 question with its own tolerance argument.

**Design, and why it is not the CUDA's.**  Each thread OWNS one key's score (thread c does the whole dot for cell c
serially over 256 dims), so the max and the sum are **two workgroup reductions per token per head** - not one per
key, which is what a per-cell reduction would cost.  `wg_reduce.glsl` gained `wg_max` for it, sharing the existing
array behind its own leading barrier.  The softmax is two-pass (max, then exp and sum) rather than streaming: both
are exact formulations of the same quantity and differ only in f32 rounding order, which the arms measure.

**A shape predicate, for the same reason the GEMM has one.**  `attn_short_shape_ok(width, capacity)` = width in
[1, 256] and capacity >= 256.  The kernel drops every key past `width`, so a wrong width is a silently wrong token
rather than an error - the engine carries a status word for exactly this, and the case's first arm is the contract
itself (width 0 and width 257 must be REFUSED, one live key and a full window must be allowed).

**Evidence.**  Arc: **204 passed / 0 failed / 0 skipped**.  Five arms - 1 live key (the first decode step, which is
bit-exact against the reference because there is no softmax to round), 3 live keys, a full 256-key window, a full
window with an additive mask (-inf on a third of the keys), and a masked half window - each 24 heads x 256 dims, each
against the double oracle.  The two KV heads carry DIFFERENT data in every arm, which is what makes the GQA grouping
falsifiable.  Tolerance is a relative bound with an absolute floor, and the case prints the worst RELATIVE and worst
ABSOLUTE deviation beside the reference magnitudes: a masked softmax over 170-odd live keys spreads the weights over
many orders of magnitude, so the worst relative error (2.25e-2) sits on an output whose absolute deviation is under
the floor - measured, not assumed.

**Falsification.**  Changing the grouping to the plausible wrong one (`head % KV_HEADS`, interleaved instead of the
engine's 12-blocks) fails all five arms with exactly half the values correct (3072 of 6144) - the 12 heads that map
to the same KV head under both rules.  A grouping bug is therefore not something this case can miss, which was the
point of giving the two KV heads different data.

**What the attention block still needs.**  The prompt path (`qsa_prompt_attn.cu`, 74 KB, and `qsa_select.cu`), the
QSA selection and indexer logic (`native_qsa*.cu`), the KV gather/write path at f16 (the port has the q8_1 append
and gather already), and a tiled version of this kernel - the serial 256-dim loops are the right trade for a
correctness kernel and the wrong one for throughput.



## THE ATTENTION BLOCK, KERNEL 2: the f16 KV window - and the engine's launch rule that a recorded buffer makes real

**What was ported.**  `kv_gather_kernel` / `kv_gather_step` (`src/kernels/cuda/qsa.cu`): the F16 sibling of
`kv_q8_gather`, same addressing, no dequantisation, because the pool itself is f16.  It is the producer of the
`[id][kv_head][head_dim]` window `attn_decode_short` consumes - the other half of a decode step.

**The pool layout is not what a fixture wants to assume.**  The pool's ROW INDEX CARRIES THE HEAD:

    row = (page * kv_heads + h) * page_size + (cell % page_size)     then  pool[row * head_dim + d]

i.e. `[page][kv_head][page_size][dim]`, NOT `[row][kv_head][dim]`.  A fixture written the second way - which this
one was, first, and it failed 6144 of 6144 values against a kernel that was right - is the mistake worth recording,
because the two layouts have the SAME TOTAL SIZE and every index in them is plausible.

**`ids[id]`, not `id`.**  The destination row index is the position in the selection; the cell it holds is the
selection's value.  Confusing them yields a window that is a PERMUTATION of a correct one, and a softmax over the
keys cannot see a permutation - it is symmetric in the key index.  That is why the composed gather-then-attend arm
runs with a POSITION-DEPENDENT MASK: with one, the permutation moves a different mask value onto each cell and the
output changes.  Measured: the injected `cell = id` fails the composed arm at worst rel error 9.19e+03.

**THE GRID IS THE CAPACITY, and the port now has a measurement for it.**  The engine's launcher says so in as many
words - "a captured layer would allocate room for token 1 and then index past it at token 500" - because a GROUP
COUNT IS BAKED INTO A RECORDED COMMAND BUFFER while a buffer is re-read at dispatch time.  The port has had recorded
command buffers since stage 3, so the rule is not hypothetical here.  The case records the gather, changes `n_ids`
from 6 to 8 in the step buffer, and replays:

    replay after n_ids 6 -> 8, capacity grid:           8 of 8 rows present
    replay after n_ids 6 -> 8, NEGCTRL live-count grid: 6 of 8 rows present

The NEGCTRL arm asserts the DEFECT (it passes only if the live-count grid really does lose rows), in the idiom the
port already uses for `f32_to_f16_trunc`.  Two rows is what a token-1 grid costs at token 8 - and it is silent.

**A case that described the rule without testing it.**  The q8 sibling's case sized BOTH its window and its grid
from `n_ids`, so the guard region its own verdict text called "incl. the guard region" did not exist: the window WAS
the live count.  Fixed to size from a capacity of 8 and dispatch over it, which is what makes the guard region real
(and the case now compares 4224 values per side instead of 3072).

**Evidence.**  Arc: **209 passed / 0 failed / 0 skipped** (five new verdicts).  The f16 gather's window is compared
element for element against the pool rows the selection names, over 16 pages in REVERSE order, ids that span pages
and page offsets and include cell 0 and cell 255; the 250 surplus rows of the 256-row window must keep their
sentinel, which is the guard; the composed arm puts 6 gathered rows through the attention and matches a
double-precision oracle computed from the POOL at worst rel 1.67e-06.

**What the KV path still needs.**  The append half at f16 (the port has the q8 append), `kv_append_q4`/`kv_gather_q4`
(the int4 pool with its FWHT-256 rotation - `fwht256_kernel` is a separate piece of arithmetic), the paged ring
table and the host staging (`kv_stream.cu`: `kv_ring_table`, `kv_stage_from_host`), and the SELECTION that fills
`ids` in the first place (`qsa_select`, `native_qsa*`) - the gather takes `ids` as an input and the case supplies
them, which is the boundary the port has drawn so far.



## THE ATTENTION BLOCK, KERNEL 3: the selection - and the first chain that runs the whole decode step

**What was ported.**  `block_scores_kernel` and `block_topk_kernel` (`src/kernels/cuda/qsa_select.cu`; the engine keeps
the second as `qsa_block_topk_ref`), i.e. the QSA indexer's block scores and the weighted top-k that produces the very
`ids` the gather now consumes.  The decode step's data path is complete end to end: pool -> scores -> selection ->
gather -> attention, all four in the port, all four gated.

**The contract, from the engine's header, is unusually complete and the port follows it literally:**

    scores: one warp per (query, block); relu PER INDEXER HEAD, summed; the tail block n_bid scores the `dead` key
            (not its pooled row) and takes +1e9 WHEN IT HAS CELLS (n_kv % R != 0).  R = 4 cells per block,
            IDX_DIM = 128, IDX_HEADS = 4.
    ids:    [nq, cap], CELLS, ASCENDING; the `width` cells with the largest block scores, each block WEIGHTED BY ITS
            CELL COUNT, ties to the lowest cell index.  While n_kv <= width the selection is the identity.

**THE WEIGHT IS THE CELL COUNT, NOT R.**  A complete last block holds `n_kv % R == 0` cells: it is SKIPPED everywhere,
so its score cannot make it selectable, and a 1-cell tail contributes 1 to the budget, not 4.  Getting this wrong
selects cells that do not exist - the danger is silent, because those ids then index a cache row beyond the live
range.  Both are gate arms, and the injection "every block weighs R" fails four arms plus the chain.

**The ordering is part of the contract.**  The ids come out ascending because the emission loop walks blocks upward
and each thread owns a contiguous range with an exclusive prefix.  Downstream, the POSITION is the window row and the
attention's mask is indexed by it, so an id list with the right cells in the wrong order is a different output.  The
injection "cells at the threshold taken from the TOP of the block" fails five topk arms and all three chain arms.

**Two things the port had to decide, both documented in the shaders.**  `order_key`'s `s + 0.0f` in the source
cancels the sign of zero, so -0.0 and +0.0 compare EQUAL; the port spells that out as a comparison rather than
leaving it to an optimiser that may fold the addition away, and the gate's oracle orders by the same keys (otherwise
a tie in f32 becomes a coin flip between two implementations that are both right).  And the scores' reduction is a
barrier tree over the workgroup instead of the source's `__shfl_xor` tree - this port's rule - so a score can differ
from the CUDA's in its last bits: measured against a double oracle at worst rel 1.32e-06, which is the same caveat
the engine's own tensor-core variant carries ("another summation order: not bitwise").

**Evidence.**  Arc: **221 passed / 0 failed / 0 skipped**.  Scores: 3 queries x 24 blocks, the dead key deliberately
far from every pooled row, the blocks past n_bid checked for having been left alone.  Selection: SEVEN arms - a
budget landing inside a block, three equal keys at the boundary, a zero-weight tail with the biggest possible score,
a 1-cell tail, both identity cases (n_kv < width and n_kv == width), and the everything-but-one-cell case - each
compared entry for entry against the rule implemented directly, with the ids past `width` required to stay sentinel.
Chain: scores -> 6 ids -> gather -> attention, with the ids, the gathered window and the attention output all
compared against oracles built from the POOL.

**Falsified (three injections, each applying to every site):**  weight = R for all blocks -> 4 arms + 3 chain arms;
`+1e9` without the has-cells test -> the scores arm (worst rel 8.55e+06); ties from the top of the block -> 5 arms +
3 chain arms.  One caution recorded for the port's own falsification scripts: an anchor that appears more than once
must be replaced at EVERY site, and the script must fail loudly if it is not - the first run of the weight injection
silently patched nothing and reported the unmodified kernel passing, which is a false negative wearing the costume of
a clean result.

**What the selection still needs.**  `qsa_block_scores_tc` / `_wmma` (the tensor-core variants, prompt path only -
the engine's header notes they are NOT bitwise with the warp kernel), the register and cluster top-k variants
(`block_topk_reg_kernel`, `block_topk_wide_kernel`, `block_topk_cluster_kernel` - the engine asserts they produce the
SAME ids, which is exactly the parity the port would gate them against), and `qsa_index_kernel` / `kv_q8_append`'s
siblings in the per-token path.



## THE ATTENTION BLOCK, KERNEL 4: the KV append - the cache's write half, and the first chain that closes the loop

**What was ported.**  `kv_append_kernel` (`src/kernels/cuda/qsa.cu`): one token's current K/V into the paged f16
cache.  This is the WRITE half of the pair whose read half is `kv_f16_gather`, and the two share one row formula:

    pos  = step[kStepPos]                                    the cell being written
    page = table[pos / page_size]
    row  = (page * kv_heads + h) * page_size + (pos % page_size)
    pool[row * head_dim + d] = fp16(cur[h * head_dim + d])

The decode step's data path is now closed at both ends: a token's K/V is written into the pool by this kernel, the
selection picks cells, the gather copies those cells into the window, and the attention reads it.

**Four things the port had to get right, all of them gated:**

1. **The conversion is the arithmetic.**  The current K/V arrives as f32 and the pool is f16, so the engine's
   bit-exact conversion (`common/f16_bits.glsl`) does the work and the case compares pool BYTES against it - not a
   tolerance.  A wrong rounding here is invisible until it moves a logit.
2. **A negative page means NO write at all** - not a write to row 0, and not a skipped host copy.  Measured: the
   injection "treat a non-resident block as page 0" is caught by exactly this arm and by nothing else, which is the
   whole argument for having an arm per rule rather than one broad comparison.
3. **Two destinations, one kernel** - the same choice the q8 append already makes in this port.  The VRAM row uses
   the PHYSICAL page from the table; the host identity layout uses the LOGICAL page index, always written.  They are
   two formulas selected by `host_layout`, not two kernels kept in step by hand.
4. **Re-appending a position overwrites the same cell**, which is what a recomputed decode step needs - and the grid
   is fixed geometry (`kv_heads * head_dim`), so a recorded launch is replay-safe by construction.

**Evidence.**  Arc: **226 passed / 0 failed / 0 skipped** (five new verdicts).  Three positions (a page's middle, its
first cell, its last cell) each leave every other element of BOTH pools at the sentinel - 786432 comparisons, which
is what makes "one cell, the right row" a measurement rather than a claim.  Both host-layout branches of the
non-resident case (nothing written vs the identity row written anyway, with the untouched count printed).  And the
chain: six tokens appended at cells 31..36 - deliberately across a page boundary, so two page-table entries are in
play - gathered into the window with 0 mismatches, attended, and matched against a double-precision oracle at worst
rel 8.37e-06.

**Falsified (three injections, each verified to have applied at exactly one site):**  the VRAM row using the LOGICAL
page index instead of the table's entry -> 5 arms, chain at worst rel 1.57e+05; a non-resident block treated as page
0 -> the non-resident arm; both halves of the grid writing K -> 4 arms incl. the chain.

**A harness bug worth recording, because its symptom was a confident lie.**  The oracle lambda closed over the
REVERSED page table even in the non-resident arm, so it predicted a write the kernel was right to skip, and the
arm's own diagnostic then announced "the identity row written anyway" for `host_layout = 0`.  The kernel was correct
throughout.  Two lessons: an oracle must take the actual bound inputs as parameters (a closed-over fixture is a
second source of truth), and a diagnostic derived from the same faulty fixture misreports in a confident voice.

**THE HARNESS HAD A CASE-COUNT CEILING, and this increment hit it.**  The first full run after the append case came
back `FAIL radeon_icd` with no numbers at all - `vkAllocateDescriptorSets -> VK_ERROR_OUT_OF_POOL_MEMORY` at the END
of the run, because `maxSets` was a hardcoded 64, one set per pipeline, nothing recycled.  Four new cases had taken
the gate past it.  It looked exactly like a kernel failure on RADV and was not one: the same binary passed on the
Arc, and the pool is the harness's, not the kernel's.  Fixed by growing: `set_alloc()` allocates a fresh pool when
the current one is full and SAYS SO on stderr (`descriptor pool 2 created`), so the next time a case pushes past 64
sets the reason is printed instead of inferred.  Worth remembering as a class: **a fixed capacity in the test harness
is a limit on the test suite**, and it fails at whichever implementation is exercised last.



## THE ATTENTION BLOCK, KERNEL 5: the Q4_0 KV path - a rotation the design depends on, and a design claim gated

**What was ported.**  Three pieces of `src/kernels/cuda/kv_q4.cu`: `fwht256_kernel` (the orthonormal 256-point
Walsh-Hadamard rotation), `kv_append_q4_kernel` with its `q4_group`/`q4_store` pair (ggml's Q4_0 group, with the
engine's deterministic tie rule), and `kv_gather_q4_kernel` (the dequantising reader).  The KV storage formats now
match the engine's three - f16, q8 and q4_0.

**THE DESIGN IS THE INTERESTING PART, and the gate tests the design, not just the kernels.**  From `kv_q4.hpp`: K
and V are rotated by H before quantisation, the QUERY is rotated too, so `<Hq, Hk> = <q, k>` and the scores are
unchanged, and the attention output - a convex mix of rotated values - is rotated back once at the end.  That is why
the gather does NOT inverse-rotate: the whole path lives in the rotated basis, and the only reason it is legal is
that H is ORTHONORMAL.  A rotation that merely "spreads outliers" would change every score.

**The rotation is gated three ways, because one of them cannot see the most likely bug:**

    the numbers against an EXPLICIT Hadamard matrix  H[i][j] = (-1)^popcount(i & j) / 16
    H(Hx) = x           (||H(Hx) - x||/||x|| <= 1e-6 measured 1.1e-07)
    <Hx,Hx> = <x,x>     (2.26e-08 relative)
    + the design claim itself, through the real attention kernel (below)

A double-precision rerun of the butterflies would share the port's own STRUCTURE, and the two invariants cannot see
a flipped sign convention either - the other convention is ALSO orthogonal and self-inverse, merely a different
basis with wrong scores.  Only the explicit matrix pins the convention, and the injection proves the point: flipping
the butterfly branches fails that arm while both invariants still pass.

**The group rule, and the one detail a paraphrase gets wrong:** the scale comes from the SIGNED EXTREME
(`d = mval / -8`, where `mval` is the value of largest magnitude and, among equals, the LARGER value), not from
`|max|`.  The sign is what maps the extreme to the bottom of the code range with an offset of 8; taking `|max|`
gives a positive scale and a wrong code set - "a bit more quantisation noise" to a perplexity run, a wrong basis in
the cache.  Codes are `trunc(x/d + 8.5)` clamped to 0..15 (TRUNCATED, not rounded), the tie rule is what makes the
result independent of a fold order (the engine added it on merge), and - unlike this port's q8 append, which
quantises against the STORED f16 scale - the codes here use the EXACT f32 scale while the stored one is f16.  Two
sibling kernels, two conventions, both faithful to their sources; the gate pins each where it belongs.

**Evidence.**  Arc: **233 passed / 0 failed / 0 skipped**.  Five special groups compared bit for bit against a host
transcription of the rule (a plain group, a `+3.5/-3.5` tie, a subnormal scale, an all-zero group, one whose codes
clamp), with the 288 bytes of the cell verified by POSITION and 0 stray bytes elsewhere.  The tie arm re-presents
the same values in the opposite order and requires the SCALE to be identical - deliberately not the whole block,
because reordering a tie moves values between elements and each element's own code legitimately changes (the first
version of this arm compared all 18 bytes and failed a correct kernel).  Round trip: the gathered window is within
**0.4719** of the rotated original against the group's own bound `|d|/2 = 0.4742`.  And the design claim, run
through the actual attention kernel: rotate q, quantise+rotate K/V to Q4_0, attend, de-rotate - worst absolute
deviation **0.644** from the unrotated attention, against a window that is itself off by up to 0.474.

**Falsified:** the scale from `|max|`; codes rounded instead of truncated; the butterfly's sign convention flipped
(caught by the matrix arm alone - see above); the nibble halves swapped in the reader.

**A fixture bug worth recording, because its symptom looked like a kernel failure.**  The Q4 pool was sized
`rows * bytes_per_head`, but the row index CARRIES the head (`(page*kv_heads + h)*page_size + off`), so the pool
needs `rows * kv_heads * bytes_per_head` - the appends then wrote entirely beyond the buffer, a driver silently
dropped them, and the gate reported a pool that was never written.  The f16 and q8 pools in this port had the same
row formula and did NOT have this bug, because their element counts happen to include the head factor; a
byte-per-head layout is where the factor becomes easy to lose.

**What the Q4 path still needs.**  The prompt-path batch append (`kv_append_q4_batch_kernel`, with its staging pool),
`kv_hybrid` (q4 for V, q8 for K - `kv_hybrid_parity.cpp` is the engine's own reference for it), the ring/staging
machinery in `kv_stream.cu`, and the per-token q4 path's rotation call sites in `prefill.cpp` (which is where the
query and the attention output get their FWHT).



## THE ATTENTION BLOCK, KERNEL 6 (no new kernel): the hybrid mode - K in INT8, V in rotated Q4_0

**What was added.**  Nothing to `shaders/`.  The hybrid mode IS the wiring: `kv_q8_append`/`kv_q8_gather` for K,
`fwht256` + `kv_q4_append`/`kv_q4_gather` for V, `attn_decode_short` over the two windows, and one `fwht256` on the
output - plus a gate that pins the asymmetry.  This is `KV_MODE 3` of the engine's own reference
(`src/kernels/kv_hybrid_parity.cpp`), whose header describes it in one line: **INT8 K, rotated Q4_0 V**, then the
output's inverse Hadamard.

**THE ASYMMETRY IS THE POINT.**  Only V is rotated.  K keeps eight bits and does not need to be spread, and because
K stays in the original basis the QUERY does too (`<Hq,Hk> = <q,k>` only has to hold for the side that moved).  So:
rotate what you quantise to four bits, leave the other side and the query alone, and take the output back with one
Hadamard at the end.  A port that "helpfully" rotated K as well would change every score; one that skipped the output
rotation would return a correctly-scored answer in the wrong basis.  Both are what the arms exist to catch.

**Evidence.**  Arc: **238 passed / 0 failed / 0 skipped** (five new verdicts), and the arms are deliberately of
different kinds because one cannot do another's job:

    K window vs the INT8 rule on the RAW rows        3072 values, 0 differ - "K is not rotated"
    V window vs the Q4_0 rule on the ROTATED rows    3072 values, 0 differ
    the mixed-basis attention vs an oracle on the same two windows      worst rel 1.68e-06 (tight)
    the output is H applied ONCE to it (the de-rotation alone)          worst rel 1.19e-04 (tight)
    vs the fp32-TRUE attention from the unquantized rows                bounded by the V-side |d|/2

**The engine's own parity file does the same two-step thing** (a tight host attention over the SAME dequantized
window, then a bounded comparison against the true one) and the reason is worth keeping: the tight comparison cannot
see what the quantisation cost, and the bounded one cannot see a basis error inside its own slack.

**A live demonstration of the "oracle built from the thing under test" trap, stumbled into rather than staged.**  The
first version bound the q4 gather's UNUSED K scratch output to the INT8 K window.  Both scratch outputs are written
unconditionally, so the q4 reader overwrote K with q4-dequantised V - and the composed arm still read **6143 of 6144
as fine**, because the attention and the oracle were both reading the same corrupted window.  What caught it was the
arm that does not go through the attention at all: **the K window check, 3072 of 3072 differing**.  Two lessons, both
already in the skill and both re-earned here: an unused binding needs somewhere harmless (the unused POOL in the
append is the same problem), and an arm that compares a kernel against its own output cannot be the only arm.

**THE CROSS-IMPLEMENTATION ARM CAUGHT A RACE IN THE CASE, and that is the find of this increment.**  The q4 append's
unused K half was bound to the SAME pool as its V half, so two writers raced on the same bytes with different data
(the K half carried the raw V row).  The Arc ran the case green - the K half's write happened to be the one the V
window check expected - while llvmpipe wrote the raw row's codes and the arm read **3071 of 3072 differing**.  A single
implementation would have shipped this.  Fixing it took two goes and the second one is the lesson: the first fix sent
the unused half to a throwaway pool but left the ROTATED row in the K input binding, which broke all three
implementations at once - because the kernel reads `is_v ? VC_ : KC_`, so the V half reads VC_.  **Read the
half-to-binding mapping off the kernel; "it passed" while two writers shared a buffer is not evidence that the
wiring was right.**

**Falsified (wiring is the mode, so the injections are in the case):**  the de-rotation applied twice and the
de-rotation omitted.  Both were *first written wrong and fixed*, which is the more useful record: the "twice" version
dispatched the same input again (a no-op that passed, and would have been reported as "the case cannot catch this"),
and the "omit" version left a push constant unused so `-Werror` failed the build and the run printed the STALE
binary's numbers.  An injection must change behaviour and must compile, and a script that does not check either is
producing evidence about nothing.  With both fixed, each injection fails the de-rotation arm (0 of 6144, worst rel
1.4e+03) and the bounded arm (12 of 6144), while the window arms and the mixed-basis arm stay green - and the two
injections produce the SAME numbers, which is itself the check: H is an involution, so applying it twice returns the
input, which is exactly what omitting it returns.

**What this leaves.**  The KV storage modes now cover what the engine ships (f16, int8, q4_0, k8v4) at the level of
a decode step.  The prompt path (`kv_append_q4_batch_kernel` and its staging pool, the tensor-core score variants,
`kv_hybrid`'s own prompt pass), the ring/staging machinery of `kv_stream.cu` and the per-token call sites that
decide which mode a step uses are still open - see `RUN-ON-B70.md` for what of it is on the critical path to a
running engine and what is deferrable.



## THE SAMPLER, KERNEL 1: the greedy argmax - the first TOKEN this port emits

**What was ported.**  `sampler_greedy_kernel` (`src/kernels/cuda/sampler.cu`) with the `history_count` /
`apply_penalties` pair it calls.  This is the `--temp 0` path, and it is the first sampler kernel here; the general
top-k/top-p kernel and the Philox draw come next, on the same penalties.  With this, the port has a complete
deterministic generation path: logits in, a token out.

    out[t] = argmax over v of apply_penalties(logit[t][v], count(v))
    count(v) = occurrences of v in the TAIL of the token's history, min(plen, history_len) long
    ties     = the LOWEST index wins; 0 when no candidate is above -inf

**THE PENALTY RULES ARE WHERE A PARAPHRASE INVERTS THE MODEL.**  The repeat penalty MULTIPLIES for a non-positive
logit and DIVIDES for a positive one - "divide by the repeat penalty" is the natural reading and it inverts the
penalty on half the vocabulary.  The presence penalty is `float(count > 0)`, a boolean, NOT the count, while the
frequency penalty carries the count.  The window is the TAIL of the history, so a token occurring only in the head of
a longer history is not penalised.  Each of those is an arm, and each fixture is built so the WRONG rule lands on a
different token.

**THE TIE RULE IS THE SERIAL SCAN'S.**  The source walks `v` ascending with a strict `>` and merges with "larger
value, or on equality the SMALLER index"; this port keeps that total order in a barrier tree (subgroup ops are banned
here) with the source's `n_vocab` sentinel for a thread with no elements, and answers 0 for an all -inf/NaN row.

**ONE PORT DECISION, stated rather than hidden.**  The source builds a shared BITMAP of the history's ids so
membership is O(1) and only the hits pay a count scan (~31 KB of shared at this model's 248,320-token vocabulary).
This port scans the window per candidate instead - what that source did before the bitmap, with identical counts -
because the bitmap and the reduction tree do not both fit the port's 32 KB shared budget at that vocabulary, and a
fast wrong membership test is worse than a slow right one.  The bitmap is the optimisation to add.

**Evidence.**  Arc: **253 passed / 0 failed / 0 skipped** (eleven new verdicts).  Ten arms against a transcription of
the rule, and every arm states the token it is BUILT to produce, which is checked as well as the oracle: a plain
argmax with a NaN and a -inf present; a two-way tie; all -inf; all NaN; the repeat penalty on a positive logit (it
must lose); on a negative logit (it must lose); the frequency count (4 hits at 2.0 drops 9.0 to 1.0, so an
unpenalised 3.0 wins); the presence penalty as a boolean rather than the count (9-2 beats 8-2, but 9-6 would not);
the TAIL window (a head-only hit is unpenalised); and the real 248,320-token vocabulary with its strided scan.

**FALSIFIED, and the fourth injection is the finding:**  the repeat branch divided unconditionally -> the negative-
logit arm; the tie rule preferring the highest index -> the tie arm; the window read as the head -> the tail arm; and
the presence penalty as the count -> the presence arm.  **The first version of that last arm PASSED under the
injection**, because its fixture let the penalised token lose under both rules: the arm was decorative, and only a
falsification could say so.  Rebuilding it (600 also in the history, once) makes the correct rule keep 500 at 9-2
against 6 and the wrong one drop it to 3.  The `expect` field earned its place the same way: with every arm stating
its token, a fixture that no longer means what it claims fails immediately - and one of them did, on the first run
after I touched it (arm 6's history still hit the runner-up once, so its count changed nothing).

**What the sampler still needs.**  `sampler_kernel` (the general path: temperature, top-k, top-p, and the cumulative
walk with the Philox draw - the RNG and its reference implementation come with it), the split-warp and
coupled/draft-staging variants (speculative decoding), and the `sample_tokens` entry point that picks between them.

## THE SAMPLER, KERNEL 2: the general path - top-k, top-p, min-p, temperature, and the draw

**What was ported.**  `sampler_kernel` (`src/kernels/cuda/sampler.cu`) and the Philox it draws with, as a shared
`shaders/common/philox.glsl` (the split-warp and draft kernels will need the same generator).  The chain is the
source's, which is llama.cpp's:

    penalties (once, on the RAW logits, over the TAIL of the history)
      -> top_k   : k rounds of a block argmax over the not-yet-taken; the list is in SELECTION order
      -> top_p   : the shortest prefix whose probability reaches top_p, never shorter than min_keep
      -> min_p   : then the descending prefix within log(min_p) of the head
      -> temperature on the SURVIVORS only, softmax over them, and ONE Philox draw

**TWO THINGS A CAREFUL PORT WOULD "FIX", both arms.**  The penalties belong on the raw logits, ONCE, before the
filters - the source's own comment records that they used to be applied a second time after the temperature scaling.
And **`temperature == 0` is NOT greedy**: the source sets `inv_t = 1/temperature` when temperature > 0 and ZERO
otherwise, so zero scales every survivor to zero and the draw is UNIFORM over the shortlist.  Greedy is its own path
(`sampler_greedy.comp`).  "Fix" either one and every zero-temperature token changes.

**THE PHILOX CONSTANTS ARE THE ENGINE'S, NOT Random123's.**  Random123's philox4x32 uses M0 = 0xD2511F53 and
M1 = 0xCD9E8D57; this engine uses 0x9E3779B9 and 0xBB67AE85.  A swap yields a perfectly good generator that produces
different numbers - the kind of change that reads as "the model got a bit worse" rather than as a bug.

**THE STREAM IS PINNED THROUGH THE SHORTLIST, not by reading the RNG back.**  Sixty-four EQUAL survivors at
temperature 0 make `floor(u * 64)` directly observable in the token the kernel returns, so sixteen seeds pin six bits
of the stream each - about 96 bits, which is what catches a swapped constant, nine rounds instead of ten, or the seed
and counter swapped into the key.  That arm is EXACT, not approximate: equal probabilities make the softmax cancel
out of the cumulative walk, so no `exp` difference can reach it.

**THREE FIXTURES WERE STRENGTHENED WHILE FALSIFYING, because the planned injections would have passed invisibly:**
equal logits make `inv_t` irrelevant (the temperature arm needs ONE dominant survivor, or "temp 0 is uniform" and
"temp 0 is greedy" give the same spread); a membership check cannot see too FEW survivors (the min_keep arm needs its
three survivors CLOSE, or one survivor is trivially inside a set of three); and a head that is REACHABLE is not a head
that is ORDERED - a doubled penalty, the source's own recorded bug, passed 184/184 under the weak form, which is why
that arm is now `top_k = 1` (a one-token shortlist makes the draw irrelevant and the order exact) plus an arm for the
temperature being applied to the survivors rather than to what the cut sees.

**THE DEVICE QUESTION, recorded as a disagreement.**  `shaders/common/double_math.glsl` says Intel's own support
article reports Arc has no shaderFloat64; this box's ANV reports `fp64 1` in the gate's `--list`, and the faithful
kernel runs here.  The portable f32 sibling that every other double kernel in this port has is therefore still to
write, and the case skips (not passes) on a device without fp64.

**WHAT THE SAMPLER STILL NEEDS.**  The split-warp and coupled/draft-staging variants (speculative decoding), and
`sample_tokens`, which picks between the paths - plus that f32 sibling.

## THE PORT MAP: the decode path's remaining work, as a CHECKED LIST

**What it is.**  `PORT-MAP.tsv` classifies **every** `kernels::` symbol `src/core/` calls - the decode path - as
`kernel` (GPU work with a shader in this tree), `host` (the engine's own host side: a size, a capability check, a
table, a sync primitive) or `todo` (GPU work this port has NOT done).  `tools/make_port_map.py` writes it and
`tools/check_port_map.py` - run by `gates/run_gate.sh` on every gate - fails on an invented symbol, on a `kernel` row
naming a shader that is not built, and on any `src/core/` symbol the table does not mention.  Falsified three ways:
an added row for a name the engine does not contain, a dropped row for a symbol `src/core/` calls, and a `kernel` row
pointing at a shader that does not exist.

**WHAT IT SAYS, which is the useful part: 77 symbols - 17 kernel, 50 host, 10 todo.**  Half the surface a reader
might assume needs porting is the engine's own host side, and the decode path's real remaining GPU work is ten
kernels, not the ~250 KB of prefill and fused-MoE code:

    cvec_apply  embedding_gather  gather_rows  scatter_rows_f32  iq_dequant_f32
    native_q5_k_f32  moe_grouped_s2  moe_hit_add  moe_hit_select  moe_hit_grouped_s2

`penalty_rows` is the eleventh thing that looks like a hole and is not: this port applies the penalties INSIDE the two
samplers, which is why it is classified `host` with that reason rather than `todo`.  `native_mmvq` is covered by the
seven `iq*_mmvq` shaders, and the `moe_hit_grouped_s2` pair by the `s2expert_*` / `moe_combine_*` / `scalar_gate_*`
set.

**The BACKEND SEAM, recorded here because it decides the shape of the integration.**  The engine selects backends by
COMPILE-TIME macros - `STRATA_ENABLE_CUDA`, `STRATA_ENABLE_HIP`, `STRATA_ENABLE_SYCL`, and no `STRATA_ENABLE_VULKAN`
- and each GPU entry point is a thin wrapper in a header whose body calls the backend's implementation
(`kv_q4.hpp`: `fwht256_inplace_cuda(...) { fwht256_cuda(...); }`).  A Vulkan backend therefore plugs in the same way:
a `_vulkan` body per wrapper, selected by a new macro, dispatching through this port's device layer.

## NEXT KERNEL: `embedding_gather` - contract PINNED, shader not written

The first-token path's gather, and the smallest of the ten holes.  What the engine's own reference
(`src/kernels/elementwise_parity.cpp`, the parity test that drives it) fixes, so no guessing is needed:

    out[i] = (code_i + bias) * scales[row * row_groups + i / group] + (offsets ? offsets[...] : 0.0f)

* `code_i` is the `bits`-wide code UNPACKED from the row's packed bytes (`bits` and `bias` are parameters: S2/S4/S8
  embeddings share this kernel), `row_bytes` is per row, `row_groups` the scales' stride per row.
* **THE REFERENCE COMPARES BITWISE, AND AGAINST THE SEPARATE MULTIPLY-AND-ADD, NOT THE FUSED FORM.**  The test builds
  `want = code * scale` then `+ offset` and `memcmp`s it against the kernel's output, while the fused
  `std::fma(code, scale, offset)` is only COUNTED (`fma_diff`) rather than required.  A port that reaches for `fma`
  as the "better" instruction fails that comparison wherever the two differ - so the port must compute the product
  and the sum as two operations, and the case must carry the same fma-difference count as a reported datum.
* The engine's fixture shape is worth copying: three rows, each run twice (with and without offsets), guard values
  just outside the written range, plus a scale applied after the gather to show it uses the caller's stream - which in
  this port is the descriptor-offset path, already gated.
* `iq_embed_rows` (the IQ/BF16 table form, `include/strata/kernels/iq_kernels.hpp`) is a SECOND kernel on the same
  path and is easy to mistake for host-side bookkeeping; it is not - the map said `host` until the header was read.

## THE EMBEDDING GATHER LANDED - and `precise` is load-bearing

**What was ported.**  `embedding_gather.comp`, from `verify_kernels.cu`'s `embedding_gather_dev_kernel` and the host
reference in `elementwise_parity.cpp`: packed codes + per-group scales (+ optional offsets) -> a float row.

    per_byte = 8 / code_bits;  mask = (1 << code_bits) - 1
    code  = (codes[tok*row_codes + i/per_byte] >> ((i % per_byte) * code_bits)) & mask    <- LSB FIRST in the byte
    group = i / group_elems
    out[tok*n + i] = (float)(code + code_bias) * scale[group] + (offsets ? offset[group] : 0)

**THE FINDING, and it is a general one for this port: A DRIVER MAY FUSE A MULTIPLY AND AN ADD UNLESS THE RESULT IS
`precise`.**  The SPIR-V contains no fma op, but GLSL's default permits *contraction* - so the driver's own backend
may still fuse, and one fused op rounds ONCE where the engine's `__fmul_rn` + `__fadd_rn` round TWICE.  The engine
compares bitwise against the two-rounding form, so a fused result is wrong wherever the two disagree.  **Measured:
four of the six arms failed before `precise`, and the two that passed were the ones whose products happen to be
exact** - which is what identified the cause.  The shader now qualifies the product, the offset and the sum, and the
SPIR-V carries six `NoContraction` decorations.

**The fixture was built to make that catchable**, rather than hoping: the host side SEARCHES for (code, scale,
offset) triples where the fused and two-step forms differ (e.g. code -7 with scale 0.1 and offset 0.1 gives
-0.599999964 two-step against -0.600000024 fused) and the probe arm puts one of them at element 0, reporting the
count of differing values - the same datum the engine tracks as `fma_diff`.

**Evidence.**  Six arms, all bitwise against the engine's rule: S8 without offsets; S8 with offsets; S4 (two codes
per byte, low bits first); S2 (four per byte) with a non-zero code bias; the rounding probe; and the group boundary.
A multi-token dispatch uses the device token array and the grid's y dimension, as the CUDA grid does, and a guard
region after the last row must stay untouched.  Falsified five ways: contractable arithmetic, MSB-first codes, the
group index as the element-within-group, the bias dropped, and the token array ignored.

## RESUME HERE (state as of the last commit)

**THE QUANTIZED-EXPERT WAVE IS DONE: ALL SIX FORMATS AND THE GROUPED PAIR.** 66 kernels, 18 shared includes, one
generated table file (four IQ grids). The gate prints its own totals - `bash ports/vulkan/gates/run_gate.sh`,
which compiles every shader from source - and this line has gone stale three times in two days, so run it rather
than quote it. The last two boxes it ran on: a Radeon RX 7900 XTX host (160 / 0 / 0 on RADV and on radeon, 154 /
0 / 1 on llvmpipe), and, after that card was swapped for an **Arc Pro B70**, this one.

**RUN ON INTEL HARDWARE (Arc Pro B70 "BMG G31", Mesa 25.2.8 / ANV, Vulkan 1.4.318, subgroup 32), 2026-10-04 after
stages 3 and 4 plus the M8 matrix path landed: **180 passed / 0 failed / 0 skipped** - the Arc skips nothing now**,
with llvmpipe **172 / 0 / 2** and the radeon ICD (now the AMD iGPU, since the discrete card is gone) at 175 / 0 / 1
on a good run - **174 / 1 / 1 when the iGPU's intermittent requery case fires**.  The two implementations that
still skip do so on cooperative matrix, which their drivers do not advertise - correct, and a different thing from
the Arc's skip, which was the port's own criterion being written around the departed Radeon's M16 config.  llvmpipe's
other skip is stage 4's `stage: vram is not mappable`: it has one memory type, device-local and mappable at once,
so there is no unmappable VRAM type to check.  The verdicts added by stages 3, 4 and the matrix path are the only
difference from the 156/0/1 these ICDs read before them.  **Read the totals, not an
exit code, and say which:** `vk_gate` itself returns 0 on a skip-only run (skips are not failures in the binary);
it is `run_gate.sh`'s assertion that makes a skipped case fail the RUN, which is the port's rule and stays.

**Two things measured about the runner on this box, both worth knowing before quoting it:**

* **`run_gate.sh` used to STOP at the Intel skip and never reach the cross-implementation arm** - so since the Arc
  swap, two of the three available implementations were silently not being exercised on this box, which is the
  opposite of what that arm exists for.  The skip now records the failure in `rc` and the arm still runs; the final
  exit status is unchanged (non-zero).  Any earlier quotation of per-ICD totals came from runs made by hand.
* **The AMD iGPU's `budget: independent requery agrees` is INTERMITTENT, not deterministic: 1 failure in 3
  consecutive runs** (161/1/1 and 162/0/1 from the same binary on the same device).  That is consistent with the
  cause already documented - an integrated device's free figure is system RAM shared with the OS - but it means a
  single radeon run can read either way.  The budget family passes on the Arc every run.

**THE FIRST DEVICE-SPECIFIC DEFECT THIS PORT HAS FOUND (fixed 2026-10-04).** `quantize_q8_K` was off by one byte
on the Arc - the low byte of block 5's scale - and the cause was measured rather than argued: printing the
device's value as a hex float beside the candidates gave `-0x1.ea9c58p-105` where the source's `1.0f/(-127/mx)`
gives `-0x1.ea9c5ap-105`, i.e. the driver FOLDS the division into `mx/-127` and rounds that. The codes were never
wrong. The case now carries both forms as images (the treatment `quantize_q8_1` already had for its codes),
demands a byte-exact match to one, and prints which: **Arc -> folded (0 of 1752 differing), iGPU -> source (0 of
1752)**. Both forms are real, on hardware in one box - one image was never going to be enough.

WHAT IS DONE, in the expert tier: the per-format row kernels for IQ1_M, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up),
IQ4_NL and Q2_0 (down) - i.e. **every format the resident `coder-iq1_m` model's 48 layers use, so each layer now
has both halves of its expert path** - plus the two q8_1 quantisers and
`native_gu_iq2s.comp` / `native_down_iq4nl.comp`, the GROUPED pair the tier actually launches. See section (b)
and (c) below for what each case checks and the one interface decision the grouped kernels needed.

WHAT IS NOT DONE, stated by name so the next session does not have to re-derive it:
  * THE OTHER FORMATS THE ENGINE HAS, which this model does not use: the per-format row kernels for IQ2_XXS (type
    16), IQ2_XS (17), Q4_K (12), Q5_K (13) and the small family Q5_0 (6) / Q5_1 (7) / Q8_0 (8).  Those are what
    the OTHER packs on the ladder need - the engine's own comment for Q4_K/Q5_K says "Unsloth's UD-Q4_K_XL
    experts", so `coder-iq3_xxs` and the q4_k_xl pack are the models waiting on them.  Seven kernels, each one
    file plus a dot include, with seven host oracles and their value-by-value partners.
  * THE FOUR OTHER GROUPED INSTANTIATIONS (`native_gu_iq3xxs`, `_iq3s`, `_iq4xs`, `native_down_q2_0`) and the
    `_multi` GRP_NC variants - mechanical copies of the gated pair (see (b)).
  * `native_mmvq.cu`'s OWN DISPATCHER (`native_iq4_nl_mmvq`, `native_q4_0_mmvq`, ... case 20 etc.), which is a
    SECOND RULE for types the expert path already has - the port carries the expert-path rule (from
    `iq_kernels.cu`), because that is what the resident model runs.  Same quantity, two rules; do not merge them.
  * WAVES 2-6 of the kernels (KV, rope, the remaining GEMVs, attention, MoE, the prefill GEMM) and the engine
    integration, neither of which this wave touched.  **The kernels ran on Intel silicon for the first time on
    2026-10-04** (the Arc run in RESUME HERE above) - but nothing has run through the ENGINE on any device, and
    the port has no engine integration at all.  See STATUS.md's last section.

**THE REAL DEFECT WAS A WRONG-BUFFER READ, AND THE "MASKED WAVE" ROOT CAUSE BELOW IS WITHDRAWN.**  In
`shaders/common/s2_row_dot.glsl`, `dx` (the activation's own fp16 scale) was read with `f16_at(xb)`, and the
port's `f16_at` takes a byte offset into the EXPERT BLOB whereas the CUDA's takes a pointer and was reading the
ACTIVATION. So with `use_xscales` false - the GPU path, the one `moe_hit_parity` exercises - every row's
multiplier was arbitrary code-byte content: code-byte pairs that decode as a NaN or a near-65504 fp16 turn the
whole row into NaN multiplied through 80 chunks, which is exactly the "same non-canonical NaN in every workgroup"
signature that was read as a reduction defect. Fixed by `act_f16_at` (reads `act_b`); six new cases cover it.

**The A/B that withdraws the masked-wave claim, measured rather than argued:** with the `dx` fix in place and the
OLD subgroup reduction restored, all six tier cases pass on RADV (subgroup 64) AND on llvmpipe (subgroup 8), at
both 80 chunks < 256 lanes and 320 chunks > 256 lanes. There is no configuration in this port where the subgroup
form was measured to fail. The reduction IS a barrier tree in the shipped tree, but as a **portability choice**
(Intel picks the subgroup width per kernel) and **not** as a bug fix - the full note and the one-file revert
recipe are at the top of `shaders/common/wg_reduce.glsl`.

WHAT MADE THE OLD DIAGNOSIS SURVIVE TWO ROUNDS, worth carrying into any future bisection:
* `probe_gu_reads.comp` verified ITS OWN reads against the host and both agreed. It never exercised `f16_at`, so
  it could not see the shader's read being from the wrong buffer - a probe that re-derives the offsets is a
  second copy of the claim, not a check of it.
* the bisection that "exonerated the dot" showed the raw per-lane dots were FINITE (0.0645924). They were finite
  and wrong. **Finiteness is not correctness, and a run with no oracle comparison cannot tell the two apart.**
* initialising the reduction's shared array changed nothing - correctly, because the cause was never shared
  memory. That non-result was read as "the hypothesis is refuted, so the cause is the caller's mask".

NEXT, in order:
1. **The quantized-expert wave** (22 shaders from `iq_kernels.cu` / `native_mmvq.cu`) - STARTED 2026-10-04:
   `iq1m_mmvq.comp` (the IQ1_M row the resident `coder-iq1_m` model's experts are stored in) is in the build and
   gated, with `common/iq1m_dot.glsl`, the grid table generated from the engine by `tools/gen-iq1s-grid.py`
   (the gate now fails if it is stale), and three cases over sub-width and above-width part counts plus two
   columns. NEXT IN THIS WAVE, in this order:
   (a) the q8_1 quantizer pair - **DONE**: `quantize_q8_1.comp` (the IQ path's, clamped) and
       `swiglu_quantize_q8_1.comp` (the shared expert path's, fused silu+quantize) are both in the build and gated.
       TWO FINDINGS FROM IT, both about the ENGINE rather than the port:
       * **THE ENGINE HAS ONE ACTIVATION QUANTISER THAT CLAMPS AND ONE THAT DOES NOT.**  `q8_1_store`
         (iq_kernels.cu, used by `quantize_q8_1_kernel` and `swiglu_q8_1_entries_kernel`) clamps `d` and the sum
         to the largest finite half - #606 - while `native_swiglu_quantize_q8_1_kernel` (native_mmvq.cu) stores
         `make_half2(d, sum)` RAW, and IT IS THE ONE THE SHARED EXPERT PATH CALLS (src/kernels/cuda/shared_expert.cu,
         two sites).  So a shared-expert block whose scale or sum passes fp16 range stores inf where the IQ path
         stores 65504, and the dot that reads it then computes inf * 0.  The port carries both rules as the source
         has them, with the difference at the call site (`clamp_606`), and the case measures a block that hits it
         (products of 3000.0: both the scale and the sum past range).  **ASKED UPSTREAM on 2026-10-04:**
         [Niko1221/Strata#606 comment 5982766696](https://github.com/Niko1221/Strata/issues/606#issuecomment-5982766696)
         - the maintainer's own 0.1.39 note names "both q8_1 activation quantizers (iq_kernels.cu and
         native_mmvq.cu)"; this is the third, and it is in that same file.  FOLLOW-UP DUTY: when it is fixed,
         comment on THAT thread (not a new issue), report what changed empirically, and flip this port's
         `q8_1_store(..., clamp_606)` to true on both paths - the swiglu case's "products past fp16 range" arm
         then expects 65504 where it now expects inf.
       * **THE DIVISION IS THE DRIVER'S.**  See (a2) - measured, and still unresolved.
   (a2) **A DECISION WORTH TAKING DELIBERATELY: which division the quantiser uses.** The port writes the source's
       `roundf(xi / d)`. MEASURED on RADV: of 2560 codes, 2184 match a correctly rounded division and 120 match
       `roundf(x*(127/amax))` - i.e. the driver's `/` is not correctly rounded, and the engine's own CPU AVX512
       path multiplies by a reciprocal too. The case accepts EITHER form (and fails on "neither"), so the port is
       portable, but the codes it produces for a tie can differ between vendors. Writing the reciprocal form
       explicitly would make it deterministic everywhere and match the CPU path; it would also stop matching the
       CUDA source's expression. Unresolved on purpose - it needs a decision, not a default.
   (a3) The swiglu quantiser's silu is `__fdividef`/`__expf` in the source and cannot be reproduced off the CUDA
       hardware, so its case verifies byte-exactness on ordinary data and a **1-ulp envelope** on tie-critical
       data. Measured with every product on a tie: 640 of 2560 codes differ from the unperturbed oracle, 0 fall
       outside the envelope. On ordinary data, 0 bytes differ.
   (b) **the grouped `native_gu` / `native_down` - DONE for the two formats that cover the most layers**
       (2026-10-04; 5 arms; RADV and radeon green, llvmpipe green).  This is the shape the expert tier actually
       launches: a list of GROUPS (the experts this batch hit), each with a range of ENTRIES (which token hit it).
       * `shaders/native_gu_iq2s.comp` (`native_gu_kernel<TG>` with IQ2_S, the model's most common gate/up format
         at 20 layers) and `shaders/native_down_iq4nl.comp` (`native_down_kernel<TD>` with IQ4_NL, the down format
         of 39).  The other four instantiations are the same file with a different dot include and three
         constants, exactly as the six per-format row kernels are.
       * **THE ONE INTERFACE DECISION: `grp_ptr` IS AN ARRAY OF DEVICE POINTERS AND VULKAN HAS NONE.** The port
         takes one weights storage buffer plus a per-group BYTE OFFSET table (`grp_off[g]`), so `grp_ptr[g] + off`
         becomes `grp_off[g] + off` and nothing else changes.  The alternative, `VK_EXT_descriptor_indexing` with
         one descriptor per expert, buys nothing here (the buffer is the same size) and would spend a
         device-extension requirement on it - the trade this port has refused everywhere else (no `shaderInt64`,
         no subgroup ops in reductions, no vendor intrinsics).  Recorded at the top of `native_gu_iq2s.comp`.
       * **THE GROUP COUNT IS A DEVICE VALUE, so the grid cannot be sized to it**: the source strides
         (`for (g = blockIdx.y; g < ng; g += gridDim.y)`) and so does the port, which needed one harness change -
         `Ctx::dispatch` grew an optional `groups_y` (it was x-only, "the y/z dims are 1"), used by these two
         kernels and nothing else.  THE CASES ARM THE STRIDE THREE WAYS (grid.y = n_groups, below it, above it),
         plus an EMPTY GROUP, because a case that only launches grid.y == n_groups cannot tell whether the stride
         works at all.
       * **What the cases test is the GROUPING, not the dots**: their oracles call the already-gated per-format
         host dots, so a failure points at the group/entry walk, the per-group offset, the `r*gu_row` /
         `up_off + r*gu_row` row addressing, the entry-major output indexing, or - on the down side - the fact
         that the activation row is the ENTRY index (`hq`) while the destination row is `ent_dst[e]`, a TOKEN.
         That last swap is the one worth a case: both indices are small integers and either compiles.
       * **`swiglu_entries_kernel` NEEDS NO NEW SHADER.** The expert tier's non-fused silu is the port's existing
         `swiglu_f32` (`(x / (1 + exp(-x))) * up`) applied to a flat entries x n_ff buffer - the same formula, and
         its case is already gated against a double reference.  For the record the engine has THREE silu spellings
         in this area - `swiglu_kernel` (double), `native_swiglu_kernel` (`__fdividef` + `__expf`), and
         `swiglu_entries_kernel` (plain `/` + `__expf`) - and the port carries the double one (`swiglu_f64`) plus
         the float one (`swiglu_f32`), both from wave 1.
       * **WHAT REMAINS IN THIS WAVE, and it is mechanical**: the four other grouped instantiations
         (`native_gu_iq3xxs`, `_iq3s`, `_iq4xs`, `native_down_q2_0`) and the `_multi` variants
         (`native_gu_multi_kernel` / `native_down_multi_kernel`), which take GRP_NC entries per pass with the
         activations staged in shared memory - the same numbers as the non-multi kernels, so the port should gate
         them for EQUALITY against their non-multi siblings rather than against a new oracle.
   (c) **the remaining formats, one shader each - ALL SIX THAT THE RESIDENT MODEL USES ARE DONE** (2026-10-04;
       3 arms each; RADV and radeon green, llvmpipe green).  Every one of the model's 48 layers now has both
       halves of its expert path ported, format by format:
       * `iq4nl_mmvq.comp` + `common/iq4nl_dot.glsl` (type 20) - the DOWN format of 39 layers.  Geometry differs
         from the 256-value formats (`Fmt<20>` is { qk 32, ipb 2, step 2 }): part k is weight block k/2 AND
         activation block k/2, two 16-value parts per block, and no sub-scale, no sign handry, no integer division
         in the tail.
       * `q2_0_mmvq.comp` + `common/q2_0_dot.glsl` (type 42) - the down format of the other 9 layers, and the one
         that closes the down side.  Geometry: 64 values per block spanning two q8_1 blocks, so part k is weight
         block k/2 and activation block k.  Its `__byte_perm` chain is a byte-table lookup ({-1,0,1,2}); the
         gather order was DERIVED by working the chain through (`__byte_perm` uses three selector bits, so the
         duplicated table operand makes each code's second bit irrelevant) and the first version had it wrong.
       * `iq3s_mmvq.comp` + `common/iq3s_dot.glsl` (type 21, 10 layers) - 110-byte blocks, a separate `signs[32]`
         array rather than IQ3_XXS's aux-word packing, and a NINE-bit grid index whose high bit arrives at a
         different shift for the two words of a pair (`8 - l0` vs `7 - l0`).
       * `iq4xs_mmvq.comp` + `common/iq4xs_dot.glsl` (type 23, 1 layer) - 136-byte blocks, `Fmt<23>` step 4, the
         IQ4_NL nibble machinery with different addressing, and a scale split across `scales_l`/`scales_h`
         (`ls - 32`).
       * **EVERY IQ CASE NOW CARRIES TWO ORACLES** and compares them to each other before comparing either to the
         device (rel gap 0, printed every run): the transcribed dp4a form and a value-by-value form in canonical
         order.  They caught three defects in the cases themselves - a per-part/per-block granularity error, a
         wrong gather order, and a table value multiplied as +255 instead of -1 (0xFF is a signed byte, and only
         the value form notices, because the dp4a paths sign-extend for free) - and none of them was reported as a
         kernel failure.  See the skill's "Testing an assumption the kernel and its oracle share".
       * **NEXT: the grouped kernels in (b)** - `native_gu` / `native_down`, the shape the expert tier actually
         launches.  They need one interface decision first: the source passes `grp_ptr`, an array of DEVICE
         POINTERS to per-expert weight blobs, and Vulkan has no equivalent - the port's options are one storage
         buffer plus an offset table (portable, recommended) or descriptor indexing (a device extension).
       * **THE PER-BYTE HELPERS ARE SHARED, DELIBERATELY - AND ONLY THOSE.** `common/perbyte_sign.glsl` holds
         `__vcmpne4`/`__vsub4` because IQ2_S and IQ3_XXS write the IDENTICAL idiom (one rule used twice). The two
         q8_1 quantisers are the same quantity under two DIFFERENT rules and stay separate. The harness keeps its
         OWN copies: an oracle that shares a helper with the kernel cannot catch a wrong helper.
       * **THE ORACLE IS THE SUSPECT TOO.** IQ2_S's first run failed with every row off by percent at the right
         magnitude, and both bad offsets were in my double oracle - the shader was right. `get_int_b2(qs, iqs/2)`
         is FOUR bytes at `2*iqs`, not two bytes at `iqs`. Tiebreaker is the helper's definition, not its name
         (written up in the skill under "Names that lie"). IQ3_XXS then passed its first numeric run.
       * **THE NIBBLE PACKING IS DE-INTERLEAVED** (low nibble of byte j = value j, high = value j+16) - not the
         adjacent-pair packing Q4_0/Q4_K use, which is why IQ4_NL and IQ4_XS read the activation at `q8[l]` and
         `q8[l+4]`. The port took that order from the engine's CPU AVX-2 kernel (`src/kernels/cpu/iq_avx2.cpp`
         says `// values 0..15` / `// values 16..31`), NOT from the CUDA's `__byte_perm` chain, which states it
         only implicitly.
2. The cross-implementation arm's `budget: independent requery agrees` case used to report a FALSE FAIL while
   the resident local model held the card. **FIXED, with the measurement**: the flake was `budget 81920 bytes,
   usage 0 bytes` - 80 KiB of driver budget bookkeeping with usage unchanged - so the discrete-card comparison
   now carries 1 MiB / 0.01%-of-heap tolerance, printed on failure, while case (c) keeps the requirement that
   the figure moves with an allocation. Do not read a future failure here as automatically environmental.

### WITHDRAWN: the masked-wave diagnosis (kept because the reasoning is worth seeing fail)

**UPDATE (bisected, not reasoned):** the `gu` NaN is no longer a mystery. The raw dot is correct - 640 of 640 up-row
dots finite - and `wg_sum` returns the same non-canonical NaN in 640 of 640 gate rows. The dot is exonerated; the
shared reduction produces it, in the one configuration no shipped caller uses (`n_chunks` 80 with 256 threads). See
`shaders/pending/README.md`, section "SETTLED BY BISECTION". The two-step fix is now justified, not speculative -
and it is still two steps, because six passing kernels depend on that helper.

The branch is GREEN: the gate prints its own totals and they were 120 passed, 0 failed, 0 skipped on RADV
(112/0/1 on llvmpipe, the skip being the cooperative-matrix case), 40 kernels plus 5 shared includes. **[A SNAPSHOT
FROM WHEN THIS WAS WRITTEN, while the withdrawn diagnosis was still being argued - NOT the current total. The
current one is in RESUME HERE at the top of this file, or better, in your own run's last line.]**

THE ONE OPEN TASK, scoped to a single case re-add:

1. `shaders/pending/` holds four finished, compiling kernels from `s2_expert_grouped.cu` - `s2expert_gu`,
   `s2expert_swiglu`, `s2expert_down` and the `s2_row_dot` include they share - plus a README with what is
   established and what is not. They are outside `shaders/*.comp` so the build does not exercise them.
2. **THE BLOCKER IS A DEVICE NaN IN `gu`, AND AN EARLIER "PASSED 3840/3840" WAS RETRACTED.** The gu kernel
   produces NaN where the oracle reads 1417.59375 from the same bytes:

       gu: got[0] nan (isnan 1)   want[0] 1417.59375 (isnan 0)   abs_sum[0] 5514.59375

   The old comparison (`ratio > 1.0`, false for NaN) reported that as "0 differ, worst 0" and the case called it a
   pass - so the claim is withdrawn and the NaN-safe comparison is now in the gate at all six sites.

   FIRST STEP, and it is a probe rather than a re-read: dump the first row's eight code bytes, its scale and its
   activation block AS THE DEVICE READS THEM (a small debug output buffer). Reasoning about which offset is right
   has already failed twice in this port. Ruled out already, in the README: the data is finite, the offsets agree
   between shader and oracle, and the integer path cannot overflow.

   THEN the `down` stage, whose own oracle bug is recorded (it must read the QUANTIZER'S OUTPUT, not the gate/up
   activation image - the third instance in this port of an oracle right for the data it was written against and
   wrong for the data it was run on).
3. The gate-wide NaN question that blocked that work is SETTLED: it was 0/0 from `gemv_bound` on a zero row, not a
   kernel. The bound now has a 1e-30 floor and the NaN-safe comparison is in at all six sites.

FILING RULE LEARNED THE HARD WAY, worth keeping: an unverified shader in `shaders/` is one the build exercises
without evidence, so drafted kernels live in `shaders/pending/` with a README that says exactly why.

THEN the rest of that file (the pair kernels, the activation correction, and the CPU-order parity kernels), and
after that the port's remaining blocks are listed under "THE REST OF THE WAVE" below.

The gate **prints its own totals** (`bash ports/vulkan/gates/run_gate.sh`) — do not quote a number here, it went
stale twice in one day. Wave 1 (elementwise / conversions / norm / silu / gdn) and both GEMM paths and rope are
done and gated.

## Done since this file was last written

- `gemm_coopmat.comp` / `gemm_coopmat_m8.comp` — the cooperative-matrix GEMM (matrix units) at the TWO tile shapes
  real devices advertise: **M16 N16 K16** (RADV/WMMA, verified on a Radeon) and **M8 N16 K16** (Intel XMX, verified
  on the Arc 2026-10-04 - see the step-5 block at the top of this file).  The case selects by the device's own
  property list, prints which pipeline it ran, and SKIPS loudly where no config exists - and since the M8 file
  landed that is a device property (llvmpipe, and RADV here, which does not advertise the extension), not a port
  limitation.
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

## `bf16_mmvf_f32_multi` is in - and the source's bit-identity claim holds

Up to 8 activation rows sharing one pass over the weight row (the prompt path; the weight row is the expensive
side). The source promises something checkable, so the gate checks exactly that: **"each output is bit-identical
to a bf16_f32_mmvf_kernel launch of its own"**. Every row of the multi-row result is compared BIT-FOR-BIT against
the single-row shader run on that row alone, and the worst difference is 0 across all three shapes - including a
shape with **padded `ldx` and `ldy`**, where a token row read at `k * n_in` instead of `k * ldx` would read 1e30
sentinel padding.

Two things the port changed about the source's shape, both deliberate:

* **The NT template parameter is gone.** `native_bf16.cu` instantiates the kernel at NT = 4 or 8 because its
  shared array is `partials[NT][32]` and NT must be a compile-time size. The port's shared reduction is sized by
  the SUBGROUP count, not by the number of tokens, so one shader covers 1..8 rows. The accumulator loop is still
  8 wide with the row count as a bound, so the per-token arithmetic is unchanged.
* **The shader CLAMPS `n_tok` to 8** rather than indexing a fixed array out of range: a shader cannot refuse to
  run, so the host's validation is documented as the host's.

And one real bug the source's own warning caught in the port's shared code: **`wg_sum` now has its leading
barrier**, because the multi-token kernel reduces once PER ROW, and without it a fast invocation can write the
next total before a slow one has read the previous - "invisible in most runs and a slightly different norm when it
fires", in ple.cu's words about the identical barrier in `block_sum`. The single-call kernels never reached it.
This is the shared include earning its place: one fix, five callers.

### THE BOUND FOR A DOT PRODUCT IS NOT RELATIVE TO ITS RESULT

A measured row of the multi-row MMVF exceeded `1e-5 * |y|` by 1.34x - while being provably correct, because the
same row matched the single-row shader bit for bit. The bound was wrong, not the kernel: an f32 accumulation of n
terms has an error bounded by `(log2(n)+1) * eps * sum|terms|`, and with cancellation `sum|terms|` is FAR larger
than `|result|`. That row: `sum|terms| ~ 6.4e5` against a result of 3.9e4, which predicts ~0.5 of absolute error
against the ~0.5 measured - the kernel sat exactly on the theoretical bound.

The three GEMV comparisons now use `rtol*|result| + 16*2^-24*sum|terms|`, with the oracle reporting both sums.
The previously failing row reads 0.0078 of the bound (170x headroom) instead of 1.34, and the bound is DERIVED
rather than fitted - a real bug moves the result by orders of magnitude more than rounding error can.

## `s_gemv_q8_split` is in - the S-family canonical decode, over both quantized activations

The heart of this is ONE decode for S2, S4 and S8 (`docs/pack-format.md`): `value = decode(code) * scale +
offset`, with the codebook choosing between the affine `code + bias` and ggml's non-linear `kvalues_iq4nl`. It now
lives in `shaders/common/sform_decode.glsl` rather than once per kernel, because a second decode of the canonical
form is a second thing to get wrong.

The kernel is one shader for BOTH quantized activation kinds - `block_q8_K` (292 bytes / 256 elements, an f32
scale) and `block_q8_0` (34 bytes / 32, an fp16 scale) - exactly as the source is one kernel templated on the
format. The only difference IS the loader; the weight decode, the octet loop, the codebook and the reduction are
identical.

**`Q8_0` IS STRUCTURAL, NOT AN OPTIMISATION CHOICE.** `ffn_down_shexp` is IQ4_NL/Q4_0/Q5_0/Q8_0 in every layer
with `n_in = 640`, and 640 is a multiple of 32 but not of 256 - so Q8_K is impossible for it rather than merely
unimplemented. The gate runs that shape, and the source's header is explicit that describing this as "Q8_K is not
implemented" was wrong for several rounds.

The forms tested are the ones the PACK contains, because the attribute vector is what the kernel takes as
arguments - and one of them is the reason to test attributes rather than arbitrary numbers:

* S2 / Q8_K / group 64 - the Q2_0 shapes, 31.64 GiB of the pack.
* S4 / Q8_K / bias -8 - Q4_0's attributes.
* S8 / Q8_0 / n_in 640 - `ffn_down_shexp`.
* IQ4_NL codebook / Q8_0 - the non-linear table, a separate decode path.
* **A form WITH AN OFFSET** - Q4_K's. This is the only case where the offset's POSITION is observable: the offset
  belongs to the weight, so the term is `(code*scale + offset) * x`. Writing the mathematically equal
  `code*scale*x + offset*x` performs two multiplications and an addition where the correct form performs one of
  each, so it ROUNDS DIFFERENTLY - and every no-offset type cannot tell. The source records that this was wrong
  until a Q4_K case existed.
* S2 / Q8_K / group 16 - a second group size, so the shift is not tested at one value only.

TWO SHAPE DIFFERENCES FROM THE SOURCE, both deliberate:

* The CUDA kernel is WARP per output row and ends in a 32-lane shuffle butterfly. The port cannot size anything
  from a device-wide subgroup default (the `rms_norm` lesson: the driver may compile a kernel at another width
  and the host would dispatch too few rows), so it is one WORKGROUP per row with the shared reduction. The
  per-thread accumulator structure and the term arithmetic are the source's; the lane-to-element mapping is not,
  which is why this comparison is against a double reference rather than bit-for-bit.
* The port has no `__constant__` memory to misuse. The source's own comment records that keeping `kvalues_iq4nl`
  in constant memory cost **2.12x** - constant memory is fast when the access is uniform, and a non-linear
  codebook means every lane reads a different index, the pattern it handles worst. The port's table is a literal
  in the shader, so the trap cannot arise.

## `s_gemv_split` is in - the fp16 activation, and the kernel `attn_output`/`shared_expert` use

Both activation kinds of the S-family GEMV now exist: `s_gemv_q8_split.comp` (Q8_K / Q8_0) and
`s_gemv_split.comp` (fp16), sharing `common/sform_decode.glsl`. Both are one workgroup per output row with the
source's per-thread structure: QE=4 for the fp16 kernel (four consecutive elements share one code word, one scale
and one offset, because every group size the format defines is a multiple of four), four accumulators combined as
`(acc0+acc1)+(acc2+acc3)`, then the workgroup reduction.

The fp16 activation is read as **32-bit pairs** (`unpackHalf2x16`), which is the source's own `__half2` view, and
the gate's activation values walk the conversion's paths deliberately: the smallest SUBNORMAL half, a zero, a
large value, negatives, small normals. Group 16 is included because it is the smallest group the format allows
and it is what makes the quad's "one group per quad" assumption tight.

ONE MORE CUDA-TO-GLSL TRAP, in the same family as `half` and `out`: **`float2` is not a GLSL type, it is
`vec2`.** The source's `__half22float2` returns a float2 and writing that name in a shader parses as an
undeclared identifier.

### STILL IN `s_gemv.cu` (the file is 40,810 bytes; both activation kinds are now covered) (the file is 40,810 bytes; this kernel is the quantized-activation one)

`s_gemv_kernel` (the naive fp16 reference, one thread per row) and `s_gemv_q8k_kernel` (the naive Q8_K one).
Both exist in the source as the REFERENCE the split kernels are checked against - "the naive one is the reference
the split one is checked against" - and the port has a stricter reference in its double oracle, so porting them
buys the same cross-check the source has rather than new coverage. Then the host wrappers including the `_async`
forms, and `s2_gemv_fast.cu` (7,297) with `s2_gemv_quads`/`s2_gemv_fast`, which the source itself defers to a
later phase ("the speed win is `__dp4a` ... and that belongs to Phase 3 once the numerics are settled").

## The MoE router - and a HARDWARE finding that shapes it

TWO ARITHMETIC VARIANTS, because the engine's router computes its exponentials AND its sum in DOUBLE and the
target hardware does not have double:

* `router_top10_f64.comp` - faithful, requires the device's `shaderFloat64`.
* `router_top10_f32.comp` - portables, float exp with **Kahan-compensated** sums, and NO `Float64` capability.

**INTEL ARC HAS NO shaderFloat64.** Intel's own support article 000089817: "Integrated GPUs included with 11th Gen
Intel processors and the upcoming Intel Arc discrete GPUs don't support shaderFloat64." That is the target
hardware, so the faithful variant cannot run there AT ALL - and this is the router, which by the source's own
measurement is 3.39 ms of a 289 ms token at 48 layers, **56% of the whole forward pass**. The port's response is
not to pretend the difference is negligible: the gate RUNS BOTH and measures the disagreement.

The result, MEASURED: over 120 rank selections - random logits at three sharpnesses, a row of exact ties, a
degenerate all `-inf` row, and near-ties at gaps of 1e-10, 1e-9, 1e-8, 3e-8, 1e-7, 1e-6 and 1e-5 - **the portable
variant selected the IDENTICAL experts as the host's double-precision reference, and so did the faithful one.**
The near-ties probe both ends of the band: ABOVE the float ULP of a probability the two variants keep the same
order, and BELOW it both round the two probabilities to the same float and both fall back to the index rule. The
band in between is where they could differ, and it is narrower than any realistic router logits produce.

THE FAITHFUL VARIANT NEEDED A DOUBLE `exp` BUILT BY HAND. glslang has no `exp(double)` - the port learned that
while porting `silu` - but it does have `roundEven(double)` and `ldexp(double, int)`, which with an 18-term series
and the argument reduced to |r| <= ln2/2 make one. Verified indirectly: the faithful variant's ids and weights
match the host reference's, and its weights match with 0.109 of a 1e-6 bound.

THE FP64 SPLIT IS NOW A STRUCTURAL GATE RULE, not a comment: `router_top10_f32` FAILS the gate if its SPIR-V
carries an `OpCapability Float64`, and `router_top10_f64` fails if it does not. The portable variant exists
precisely for devices without the capability, so its absence from the SPIR-V is the property that keeps it
runnable there.

Sharing: the semantics (softmax over ALL experts, stable argsort with ties to the ASCENDING index, the
`2**-14` clamp applied to the gathered weights) live in `common/router_select.glsl`, used by both variants - they
are semantics, not arithmetic. One behaviour is material and tested: a rank the selection does NOT write keeps
whatever the output buffer held, and the renormalisation divides those stale values too. That is the source's
behaviour, not an oversight, and the gate pre-fills the buffer to prove it.

## THE DOUBLE-ARITHMETIC FAMILY, and the port's policy for it

**A PORT-WIDE FINDING, not a per-kernel one.** More than one kernel in this engine computes in DOUBLE, and the
target hardware has no `shaderFloat64` (Intel support article 000089817). Found so far: the router's exponentials
and its sum, `moe_combine`'s accumulation, `swiglu`'s silu, and `scalar_gate`'s dot. Each one therefore needs a
PORTABLE SIBLING, and the port's policy is now explicit:

* the FAITHFUL variant is held to **bit-exactness** against the host's double reference - it claims to be the same
  arithmetic, so it is required to be;
* the PORTABLE variant is held to a **bound taken from measurement**, printed on every run, and the gate FAILS if
  its SPIR-V carries `OpCapability Float64` (it exists for devices that do not have it);
* where the portable variant's error is a CONDITIONING question (a sum of signed terms: `moe_combine`, the GEMVs)
  the bound is relative to the TERMS; where it is a product (silu) a relative bound is right.

`common/double_math.glsl` holds the one piece that has to be built by hand - a double `exp` from
`roundEven(double)`, `ldexp(double,int)` and an 18-term series with the argument reduced to |r| <= ln2/2 - because
glslang has none. The router and the SwiGLU both use it now.

## `moe_combine` and `swiglu`, in both arithmetic variants - and an unblocking

`moe_combine_f64`/`_f32` (the MoE block's final combination) and `swiglu_f64`/`_f32` (the shared expert's
activation). Measured on the real geometry (n_embd 2560, k 10):

* `moe_combine_f64` is **bit-exact** against the double reference, both with the shared expert added and without;
* `moe_combine_f32` differs on 934/2560 (worst 0.07 of the term-relative bound) and 1452/2560 (0.075);
* `swiglu_f64` is **bit-exact**, and it differs from a plain float silu on **665 of 2560** elements - so the
  double silu is not a formality, it is a different arithmetic;
* `swiglu_f32` differs from the double reference on 1040/2560 with a worst relative error of 9.01e-07.

**IT UNBLOCKS A SHADER THE PORT HAD SHELVED.** `shaders/blocked/silu_fp64.comp` was parked early on because
glslang has no `exp(double)`. It is expressible after all - `common/double_math.glsl` builds one - and the router
needed the identical piece. A blocked item turned out to be a missing helper rather than a missing capability.

Two semantics that are behaviours rather than numerics, both tested: the shared expert's output is added to the
routed result **PLAIN** (not router-weighted, not renormalised against the routed sum), and the SwiGLU rounds the
silu to f32 BEFORE multiplying by `up` - moving that cast changes the value.

THIRD RESERVED-WORD TRAP: **`shared` is a GLSL keyword** (the shared-memory qualifier), so a buffer cannot be
named `shared` any more than a variable can be named `half` or `out`.

## The shared expert is complete on the GPU side (`scalar_gate`, `scale_rows`)

`scalar_gate_f64`/`_f32` (the per-token gate: sigmoid of a bf16 dot in double) and `scale_rows` (the per-token
scale). With `moe_combine`, `swiglu` and these, every kernel `shared_expert.cu` needs is ported - what remains
there is the host orchestration, which belongs to integration.

`scale_kernel` needed no work: the port's wave-1 `scale.comp` does the same multiply, with the scalar arriving as
a push constant instead of a `g[0]` buffer read. Same value, same rounding.

THE SCALAR GATE'S FAILURE MODE IS QUIET and that is why it gets its own case: sigmoid bounds the output to [0,1],
so a gate that should be 0.5 and reads 1.0 scales the shared expert by 2x and produces finite, plausible logits.
The zero-dot row cannot hide anything - the answer there is exactly 0.5 - so it is its own check, and the
saturated row (a gate of exactly 1.0, or exactly 0.0 on the other side) is another.

### A SUBNORMAL BOUNDARY IS NOT A CORRECTNESS BOUNDARY, and the first run of this case failed on exactly that

With full-range inputs the dot is ~+-30, so the sigmoid returns values like **2.8e-45** - the smallest SUBNORMAL
floats - and the faithful variant produced 0 where the reference produced 2.8e-45. Both are "zero" in any
meaningful sense; the difference is that at the subnormal boundary the LAST BIT OF THE DOUBLE decides the float's
step. Scaling the activation so the gates land in the normal range made the faithful variant **bit-exact 4/4**,
which is the experiment that identifies the cause.

So the case's data is deliberately scaled out of that regime, and the note says why: requiring bit-exactness in a
regime where the answer is subnormal is requiring the double's last bit, not the arithmetic. The f32 variant's gap
over the same rows is 0.254 of the bound.

Two bugs of my own found here, both the same shape - a reference that was right for the data it was written
against and wrong for the data it was run on:

* the oracle was computed once for the random weight set and then reused for the all-zero and large-weight runs,
  which reported 0/4 and 2/4 for the two runs that should be the easiest to get right;
* and the subnormal regime above.

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

## THE NEXT WORKSTREAM: the quantized experts (decision taken 2026-10-04: port them)

The port is re-based onto upstream v0.1.39 and this branch no longer carries the fork's AVX1 floor (upstream
shipped it). The next wave is the quantized-expert path, chosen because the resident model is
`qwen3.8-flash-next-coder-iq1_m`.

**ITS EXPERTS ARE NOT IQ1_M, AND THE NAME IS WHY THIS WAS WRONG FOR THREE SESSIONS.**  "IQ1_M" is the GGUF the
model's DENSE weights come from; the experts live in the native pack beside it and the pack states their types
per layer.  From `Strata-data/packs/coder-iq1_m/native_experts.txt` (48 layers, 256 experts, verified
2026-10-04), counted over its `gu_type`/`d_type` columns:

    gate/up   IQ2_S (22)  20 layers | IQ3_XXS (18)  17 | IQ3_S (21)  10 | IQ4_XS (23)  1
    down      IQ4_NL (20) 39 layers | Q2_0 (42)      9

So the port's format order is now IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up) and IQ4_NL, Q2_0 (down) - NOT IQ1_M
first.  The IQ1_M dot already ported stays: it is the format of the OTHER packed models on the ladder, and of the
IQ1_M GGUF's own expert tensors where a model uses them.  A format census is worth re-running per model rather
than inferring from a filename; `native_experts.txt`'s first line names its own columns.

CORRECTED SIZES - an earlier note said "~141 KB"; the measured figures are:

    src/kernels/cuda/iq_kernels.cu     136,615 bytes   17 __global__ entry points
    src/kernels/cuda/native_mmvq.cu     71,856 bytes   11 __global__ entry points
    ---------------------------------------------------------------------
    total                              208,471 bytes   28 kernels
       for comparison, everything ported so far is ~128 KB and 40 shaders

Upstream GREW both files in v0.1.39 (+1095 lines iq_kernels.cu, +130 native_mmvq.cu), which is why the re-base
had to happen before this work rather than after.

The 28, in file order:

  iq_kernels.cu: mmvq, mmvq_multi, native_gu, native_gu_multi, swiglu_entries, native_down, native_down_multi,
                 quantize_q8_1, swiglu_q8_1_entries, dequant_flat, dequant_gu, embed_rows, native_gu_amd,
                 native_down_amd, native_gu_lds, native_down_lds, native_gu_fused
  native_mmvq.cu: native_quantize_q8_1, native_swiglu_quantize_q8_1, native_q5_k_mmvq, native_q2_0_mmvq,
                  native_q3_k_mmvq, native_iq4_xs_mmvq, native_q4_k_mmvq, native_q6_k_mmvq, native_small_mmvq,
                  native_mmvq_multi, native_mmvq_wave

NOT YET SCOPEU: how many of these are actually portable. A quick grep for HIP-isms (`__shfl`, `warpSize`, `LDS`)
labelled `native_gu_lds_kernel` "portable-looking", which is obviously wrong, so that grep was thrown away rather
than quoted. The `_amd`, `_lds` and `_fused` families look HIP-specific by name and probably have no Vulkan form;
the count of shaders to write is therefore somewhere between 20 and 28 and must come from reading the kernels, not
from a pattern match. DO THAT READ FIRST, before writing any of them.

## SCOPED (the read the note above asked for): 28 kernels -> 22 shaders

Read the six suspected ones. They are not HIP-specific by using exotic intrinsics; they are hardcoded to AMD
WAVEFRONT WIDTHS, which is a different and more decisive problem:

    native_gu_amd_kernel     lane = threadIdx.x masked to 63               wave64
    native_down_amd_kernel   lane = threadIdx.x masked to 63               wave64
    native_mmvq_wave_kernel  lane = threadIdx.x masked to 63               wave64
    native_gu_lds_kernel     lane/warp split at 32 + sgrid/LDS_NT/LDS_RB   wave32 + LDS staging
    native_down_lds_kernel   lane/warp split at 32 + sh_raw/LDS_RB         wave32 + LDS staging
    native_gu_fused_kernel   lane/warp split at 32 + LDS across two waves  wave32 + LDS staging

A kernel that hardcodes the lane mask cannot behave the same on an Intel GPU (subgroups of 8/16/32), so these are
SPECIALISATIONS the engine selects per architecture - not code the port needs to carry. The portable variants
(`native_gu_kernel`, `native_down_kernel`, `mmvq_kernel`, ...) are what this port targets; a Vulkan
specialisation is a later, separate decision.

So: 28 - 6 = 22 shaders to write, and some of those pair up (the `_multi` variants differ by an activation-row
count, exactly like bf16_mmvf_f32 / bf16_mmvf_f32_multi did here, which the port already folds into one kernel
with a parameter) - so expect ~18-20 files, not 22.

## THE MASKED-WAVE AUDIT: the defect is in the FAMILY, and one green case is UB

Root cause (see shaders/pending/README.md) is a wave reaching `subgroupAdd` with a partial execution mask, which is
what happens when the lane-distributed work count is SMALLER than the workgroup width. All 40 shaders declare 256,
so the test per caller is: can its lane loop's bound be below 256?

Every caller of the shared reduction, with the bound measured from the shader and the case that exercises it:

    caller                lane bound            safe while      verdict
    rms_norm              cols = n_embd 2560    always          SAFE
    scalar_gate_f32/f64   n = n_embd 2560       always          SAFE
    ple_gate              n = hc_dim 10240      always          SAFE
    ple_gnorm             cols = n_embd 2560    always          SAFE
    bf16_mmvf_f32         n_pairs = n_in / 2    n_in >= 512     CONDITIONAL
    bf16_mmvf_f32_multi   n_pairs = n_in / 2    n_in >= 512     CONDITIONAL
    s2_gemv_q8            n_quads = n_in / 4    n_in >= 1024    CONDITIONAL - AND ITS CASE IS ALREADY UB
    s_gemv_q8_split       unverified (QE loop)  unverified      UNVERIFIED
    s_gemv_split          unverified (QE loop)  unverified      UNVERIFIED
    s2expert_gu           H / 32 = 80           never           BROKEN (this is the NaN)

**The s2_gemv_q8 line is the one that matters.** Its case builds shapes with
`const int n_in = shape == 0 ? 2560 : 64`, so shape 1 gives n_quads = 16: lanes 16..255 do no work and the wave is
masked at the reduction - the same undefined behaviour as `gu`, in a case that currently PASSES with error/tol 0.
That pass is luck about which garbage sat in the masked lanes' registers, not evidence of correctness, and it can
flip with a recompile (it is the same code path that yields NaN in the expert tier).

So the fix belongs in the reduction or its contract, not only in the expert tier: any caller whose lane bound can
fall below the workgroup width is exposed, and two of them are already in the build.

NEXT: (1) fix the reduction shape once (a barrier-based reduce removes the class); (2) re-check the two UNVERIFIED
s_gemv lanes against the same test; (3) leave the s2_gemv_q8 shape-1 case in place but stop reading its pass as
evidence - or better, keep it as the regression that must still pass AFTER the fix.
