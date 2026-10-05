# Status — what is done, what is verified, what is not

## Done and verified in this session

Everything below is backed by a command that exits non-zero on failure. Re-run it with:

    bash ports/vulkan/gates/run_gate.sh          # compiles the shaders from source, validates the SPIR-V,
                                                 # checks each shader's declared local size, then runs the gate

**Result: the gate prints its own totals and those are the authority. On 2026-10-04, after the Radeon RX 7900 XTX
was swapped for an Arc Pro B70 and after stages 3, 4 and the prefill GEMM landed, the box's GPU run was
**189 passed / 0 failed / 0 skipped** on the Intel ICD (`Intel(R) Graphics (BMG G31)`, Mesa 25.2.8 / ANV, Vulkan
1.4.318, subgroup size 32) - 177 / 0 / 3 on llvmpipe and 180 / 0 / 2 on the radeon ICD, which now picks the AMD
iGPU because the discrete card is gone (both skip cooperative matrix, whose driver does not advertise the
extension, and the prefill SPLIT, which needs the M8 tile).  **The Arc has no skips at all:** the last one
(`gemm_coopmat`) was the port misreading the device - BMG's matrix config is M8 N16 K16, not the M16 the criterion
demanded - and since then the matrix path RUNS on XMX, including the prefill GEMM.  On the iGPU, 180/0/2 becomes
179/1/2 when its intermittent budget-requery case fires.  Before the swap the same gate read 160 / 0 / 0 on RADV
and on radeon. That count has gone stale three times in two days; read the last line of your own run.** All three
available implementations are exercised again by `run_gate.sh`: it used to stop at the Intel skip, which meant the
cross-implementation arm never ran on this box after the swap (`NEXT.md`). 58 kernels, 17 shared includes, one
generated table file (`harness/iq_grids.hpp`,
holding the IQ1_S, IQ2_S, IQ3_XXS and IQ3_S grids). TWO RECONCILIATION NOTES, both verified against a full run:
the ``PASS`` LINE COUNT IS ONE LESS than the case total, because the transcendental probe prints `INFO` while
counting as a pass; and one line ("gemm shape contract") covers seven cases. Neither is a discrepancy - but if the
numbers ever stop reconciling this way, something is wrong with the harness rather than with a kernel.

**Stage 3 of the port plan - recorded command buffers, the CUDA-graph replacement - is DONE and VERIFIED
(2026-10-04).** The engine's decode step re-issues a fixed sequence of dispatches every token, and a CUDA graph is
how it avoided that; the Vulkan equivalent is ONE command buffer recorded once and RE-SUBMITTED.
`Ctx::record_begin` / `record_dispatch` / `record_end_and_submit` / `replay_recorded` are in
`harness/vk_compute.*`, and `case_recorded_step` in `harness/vk_gate.cpp` is the case that exercises them: **six
verdicts, all green on the Arc, on llvmpipe and on the Radeon iGPU.** It was proven able to fail before it was
trusted - flipping `record_dispatch`'s `fresh_set` to false (ONE shared descriptor set for the whole recorded step,
the trap the encoding comment names) makes three of the six verdicts fail, 162/0/1 -> **159/3/1**. `NEXT.md`'s top
block carries the arm-by-arm table.

The table below is the original wave-1 set and has not been re-listed as the suite grew - every case since is
gated the same way and is described where it is defined.

**Stage 4 of the port plan - device-local memory, staging and fit accounting - is DONE and VERIFIED (2026-10-04).**
`Ctx::alloc_device()` allocates memory with NO mapping where the device offers it (real VRAM: 3 of the Arc's 7
memory types are device-local and unmappable, on llvmpipe there is no such type at all), `Ctx::alloc_staging()` is the host-visible transfer
buffer, and `write()`/`read()` stage automatically over `vkCmdCopyBuffer` so a case written for stage 1's path runs
unchanged on the engine's. `plan_fit()` answers the other half - which line of the engine's VRAM plan does not fit,
by name, and what a droppable cache buys - and the gate fits and PRINTS the plan against the driver's own figure
(28.64 GiB on the Arc). Six device arms plus five pure-policy arms, green on all three implementations, and the
round trip was falsified with a four-byte offset error (174/0/1 -> 172/2/1). Full detail, including the ONE defect
class the round trip does NOT cover (an absent barrier: deleting the post-copy barrier changes no verdict on the
Arc or llvmpipe) is in `NEXT.md`'s stage-4 block.

**Stage 5's GEMM is DONE AND VERIFIED (2026-10-04) for the layout the engine calls** - `Y[T x ldy] = X[T x K] .
W[N x K]^T` in f16 with an f32 accumulator, on Intel's real matrix tile (8x16x16) and in FMA for the ragged shapes,
plus the engine's `bf16 -> f16` conversion with its clamp.  Ten verdicts (189/0/0 on the Arc; the FMA half green on
all three implementations), with the transposed-operand bug falsified to fail four of them, and an `ldy > n` arm
that leaves the stride columns NaN and requires them to survive.  `NEXT.md`'s stage-5 block carries the layout
proof, the two-kernel split (aligned rows on the matrix units, the remainder on FMA) and the device-layer gap it
found: the harness binds descriptors at offset 0, and the engine reaches row slices by pointer arithmetic.

| Case | Verdict | Method |
|---|---|---|
| harness self-test (copy) | PASS 1024/1024 | bit-exact — proves bindings, push range and host-read barrier |
| `scale_inplace` | PASS 1000/1000 | bit-exact vs the CUDA arithmetic |
| `add_inplace` | PASS 1000/1000 | bit-exact |
| `f32_to_bf16_bulk` | PASS 1024/1024 | **bit-exact** vs `bf16_bits.hpp` (round-to-nearest-even, NaN quieted) |
| `f32_to_f16_bulk` | PASS 1024/1024 | **bit-exact** vs `f16_bits.hpp` (RNE, subnormals, overflow→inf) |
| negative control (truncating f16) | PASS (484/1024 differ) | proves the bit-exact gate can FAIL |
| `gdn_gate` (1 and 3 tokens) | PASS, worst rel **4.25e-07** | vs double-precision softplus; tol 5e-6 |
| `rms_norm_weighted` (4 shapes, incl. rows=2) | PASS, worst rel **1.79e-07** | vs double reference **and** a NaN-padded tail that fails if the row guard is missing |
| `silu_inplace` | PASS, worst rel **7.58e-07** | vs the engine's double-precision reference |
| transcendental probe (`exp`, `log`) | INFO | driver `exp` ≈ 9.1e-07, `log` near 1 ≈ 1.5e-07 abs error |
| memory budget: independent requery agrees | PASS | the test re-queries `VK_EXT_memory_budget` itself and compares |
| memory budget: free − reserve == usable | PASS | against the free figure the test computed independently |
| memory budget: usage tracks an 8 MiB alloc | PASS | the driver's number is a measurement, not a constant |
| desktop reserve policy (5 cases + 2 controls) | PASS | floor, 25%-of-card clamp, and the in-between case |
| over-budget allocation REFUSED | PASS | a child process with an 8 MiB card must exit 3 and say why |
| compat: advisory rules (11 cases + boundary) | PASS | kernel/Mesa/driver combinations, each case discriminating against its neighbour |
| compat: kernel + Mesa parsed live | PASS | `uname` release and the driver-supplied Mesa string, parsed and checked |
| ledger + discrete + no ceiling REFUSED | PASS | child process; would report 24 GiB usable if the rule were removed |
| explicit ceiling unblocks the ledger | PASS | child process; `STRATA_VK_MAX_BUDGET_MIB` is the documented remedy |

The bf16/f16 conversion fixtures deliberately include exact rounding ties at both precisions, subnormals,
the fp16 overflow point (65504 → 65536), infinities, a signalling NaN pattern and signed zero. The
`rms_norm` case `rows = 2` is the QSA case in the source's own comment: the launcher rounds the grid up, and
a missing `if (row >= rows) return;` overwrites the buffer that follows — so the case pads the allocation
with NaN and requires it to survive.

## Audit

`ports/vulkan/AUDIT.md` — a full pass against the standing code guideline plus the Arc/Vulkan and performance
rules. Nine code-guideline findings (duplicated reference implementation, dead code, parallel containers, a
hardcoded path, a constant repeated nine times, `-Werror`) and six compatibility findings — two of them real
defects (**`rms_norm` coupled the host's dispatch to the driver's subgroup width**, which would skip tail rows
silently on Intel where the driver picks a SIMD width per kernel; and **an fp16 device feature requested with
no fp16 in any kernel**, which fails `vkCreateDevice` on hardware that lacks it). All fixed and re-verified,
five new automated checks added, each proven able to fail — including a CROSS-IMPLEMENTATION arm, which was
added after the same shaders measured worst-case relative error 1.13 on llvmpipe (subgroup 8) while passing on
RADV (subgroup 64): a reduction combine stage assumed `gl_NumSubgroups <= gl_SubgroupSize`. Fixed and verified
green on both implementations.

**Display contract:** the Vulkan layer queries `VK_EXT_memory_budget` (heap usage, not this process's ledger),
holds back a 1024 MiB desktop reserve (floor 256 MiB, capped at 25% of the card), and refuses an allocation that
would cross it — naming the numbers. Measured live on this box: with the resident local model holding the
7900 XTX, RADV reports 0.19 GiB free of 24 GiB and the harness **refused to allocate**, which is the behaviour
that would have prevented the recorded AMD incident (`docs/AMD_HIP.md`, #380/#377).

**Verified on three Vulkan implementations, every run:** Intel Arc Pro B70 / ANV (subgroup 32), llvmpipe / CPU
(subgroup 8) and the AMD iGPU / RADV (subgroup 64) - the discrete Radeon is gone, so the radeon ICD now picks the
iGPU. Before the swap the arm read 14/14 on RADV/7900 XTX and llvmpipe; the suite has grown since and prints its
own totals. **Two caveats on this box, both measured 2026-10-04** (`NEXT.md`): `run_gate.sh` used to stop at the
Intel skip and so never reached this arm at all after the swap, and the iGPU's `budget: independent requery
agrees` is INTERMITTENT - 1 failure in 3 consecutive runs of the same binary on the same device - rather than the
deterministic failure it was first documented as.

## Three defects found by the gate (all real, all fixed, all measured)

0. **The expert tier's `dx` came from the wrong buffer.** `s2_row_dot`'s `f16_at` takes a byte offset into the
   EXPERT BLOB; the CUDA's takes a pointer and was reading the ACTIVATION. So with `use_xscales` false every
   row's multiplier was arbitrary code-byte content, and a code-byte pair decoding as a NaN fp16 turned the row
   to NaN through all 80 chunks. This is the defect that was misdiagnosed for a day as a masked-wave reduction
   fault - see `NEXT.md`, and `case_s2expert_tier`, whose six arms were written for the misdiagnosis and found
   the real one. Fixed by `act_f16_at`.
0b. **The same class recurred in the IQ1_M port and the new case caught it in one run.** `iq1m_read_int4` read
   `w_b` (the weights) while its only call site passed an ACTIVATION offset, so the activation words came out of
   the weight buffer. The case's diagnostic identified it in a single comparison: oracle part 0 `0.440796`
   against device `-0.395386`. Fixed by renaming the helper after the buffer it reads (`iq1m_act_int4`), which is
   the rule this port now applies to every CUDA pointer-taking accessor: the buffer belongs in the name, because
   both mistakes compile and both read a legal address. **A CUDA helper that takes a pointer loses its argument
   when the port makes it a byte offset, and nothing but a case with an oracle notices.**

1. **`gdn_gate` first version: softplus lost 4.3e-05 relative.** The delegated translation wrote
   `log(1.0f + exp(x))`; the source uses `log1pf(expf(x))`. A Kahan-form `log1p` did **not** fix it — the
   failure number was identical to five figures, which said the diagnosis was wrong. The actual cause,
   established with a probe shader: the driver's `log` carries ~2^-22 *absolute* error near `u = 1`, which
   breaks the cancellation a Kahan `log1p` depends on, and for negative `x` softplus **is** `exp(x)`, so the
   whole error lands in the result. Replaced with an 18-term series below `z = 0.5` and the driver's `log`
   above it: **4.26e-05 → 4.25e-07**, and the tolerance is now 5e-6 instead of 1e-4.
2. **fp64 `silu` is not expressible with this host's toolchain.** `silu_inplace` computes in double in CUDA.
   GLSL has `double`, but neither glslc (glslang 14.0) nor `glslangValidator` 15.1 can resolve a double
   overload of `exp` for SPIR-V. The port ships the f32 shader and **measures** its gap against the double
   reference (7.58e-07) instead of asserting it away. See `shaders/blocked/README.md`.

## One finding that was NOT a kernel defect: the port misread the device (2026-10-04)

`gemm_coopmat` skipped on the Arc with "no usable config: needs M16 N16 K16 subgroup-scope with f16 A/B and an f32
accumulator", which reads as the device having no matrix units.  Enumerating the driver's own config list showed
otherwise: **BMG's floating-point config is M8 N16 K16.**  The criterion had been written against the departed
Radeon's list, where every config is M16, and a tile is a property of the device per generation - so the port was
reporting a limitation of its own rule as a limitation of the hardware.  `gemm_coopmat_m8.comp` now exists, the
case selects the pipeline from the property list and prints which one ran, and the Arc reads **180 / 0 / 0** with
the matrix path exercised on all four shapes including a prompt-shaped 128x256x256 grid.

Two lessons, both general: **a skip message is a claim about the device and has to be derivable from the device**
(this one was derivable only from RADV); and **the wrong tile is silent** - forcing the M16 kernel on BMG does not
fail pipeline creation, it computes wrong numbers (2560 of 4608 elements correct in the square case), which is why
the case prints the shape it selected rather than only PASS.

## Not done (and how much of the job each is)

* **Waves 2-6 of the kernels** (KV, rope, the GEMVs, attention, MoE, the prefill GEMM): the remaining CUDA
  kernel files, plus the 8 prefill files and the 7 `cuBLASLt` call sites (which have no Vulkan equivalent and
  must be written by hand). The method and the gate are proven; this is a grind, not an unknown. **The resident
  expert tier - `s2_expert_grouped.cu`'s gate/up, SwiGLU and down - is DONE and gated (2026-10-04).** The next
  workstream is the quantized-expert wave: 22 shaders from `iq_kernels.cu` and `native_mmvq.cu`. **ORDERED BY THE
  RESIDENT MODEL'S OWN PACK, not by its filename**: `coder-iq1_m`'s experts are IQ2_S (20 of 48 layers), IQ3_XXS
  (17), IQ3_S (10), IQ4_XS (1) for gate/up and IQ4_NL (39), Q2_0 (9) for down - see `NEXT.md`, which carries the
  census and the reason the filename misleads. Done so far in this wave: the IQ1_M row dot, the two q8_1
  quantisers, and **ALL SIX of the model's expert formats - IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up) and IQ4_NL,
  Q2_0 (down)**, each with its dot in `shaders/common/` and its own gated case. **Every one of the model's 48
  layers now has BOTH halves of the expert path ported** (before the IQ4_NL step that count was zero: no layer is
  complete without its down projection, which is why the down formats were taken first). The grouped kernels in
  (b) are now DONE for the two formats that cover the most layers - `native_gu_iq2s.comp` and
  `native_down_iq4nl.comp`, the shape the expert tier actually launches - with the four other instantiations and
  the `_multi` variants left as mechanical copies (see `NEXT.md`).
* **The engine integration**: no `STRATA_ENABLE_VULKAN`, no arena, no kernel registry, and no recorded decode step
  IN THE ENGINE. `harness/vk_compute.*` is the seed of the device layer and now carries the single-shot `dispatch()`,
  the recorded-step API (stage 3), and the engine-shaped memory path — device-local allocation, staging transfers
  and `plan_fit` fit accounting (stage 4) — each verified by the gate. What it still is NOT: the engine's backend.
  There is no `setup.py`/`CMakeLists`/`core/device.hpp` change on this branch, so nothing in the engine can call any
  of it yet.
* **Anything on Intel hardware — NO LONGER UNVERIFIED (2026-10-04).** An Arc Pro B70 is now the discrete card in
  this host (the 7900 XTX is out; the Ryzen iGPU drives the display) and the gate runs on it: **180 / 0 / 0** —
  no skips at all, including the matrix path, which was the port's own criterion misreading the device rather
  than the device lacking anything (see the M8 finding above), with
  one device-specific defect found and fixed (the folded division in `quantize_q8_K` - NEXT.md's RESUME HERE).
  What stays unverified on Intel is everything the gate does not cover: the engine path, real token shapes, and
  the stability question below.
* **The budget requery on an INTEGRATED device.** `budget: independent requery agrees` fails on the AMD iGPU and
  passes on the Arc: an iGPU's free-memory figure is system RAM shared with the OS, so two queries can disagree by
  construction. Left red on purpose - the port's rule is to report an implementation that cannot satisfy a check
  rather than soften the check around it. **Measured 2026-10-04: it is intermittent, not deterministic** (1 failure
  in 3 consecutive runs of one binary on one device), so a single radeon run can read 161/1/1 or 162/0/1 - quote
  the device and the run, not "the radeon ICD always fails one".
* **The Battlemage stability question (§4 of the plan) — TESTED ON THE CARD, 2026-10-04.** The xe compute wedge is
  an open driver bug this port cannot fix or test around, which is why the plan makes a smoke test the FIRST thing
  to do on Battlemage hardware. Done, with `gates/smoke-arc.sh` (added the same day): **718 rounds of 8 concurrent
  instances — 5744 suite runs, ~890k case executions — over 480 s: 0 kernel cases failed, 0 hangs, 0 xe errors in
  the kernel log, slowest round 1 s, no latency creep**, after 502 further runs one instance at a time, likewise
  clean. The card did not wedge. The honest limit of that claim is the plan's own bar - "an hour of inference" -
  so what is measured is "has not wedged under 8 minutes of concurrent small-kernel load", not "survives sustained
  inference"; an inference-shaped arm on the same driver is what would close that gap.
* **Two test-side defects the smoke run found, both fixed rather than relabelled.** The first concurrent run
  exposed `case_firmware_variants` writing and deleting a FIXED `/tmp` path, so one instance's cleanup removed the
  file another was reading (2 of 8 jobs; 502/502 when run one at a time) - now per-process. And the runner learned
  to CLASSIFY the two checks that read the DRIVER's view of a busy card, because treating them as failures aborted
  the loop at round 1 and hid the sustained result. Both are in `gates/smoke-arc.sh` and the commit that added it.

## Estimated remaining cost

* **Kernel waves 2-6 (37 files):** the first wave's 8 kernels took ~40 minutes wall-clock end to end
  including the two defects above, most of it gating. At a similar rate, and with generation offloaded to the
  local model, expect **~2-3 focused sessions**. The long tail (MoE + the 7 quant formats) is the larger half.
* **The GEMM + prefill path: the GEMM IS DONE (2026-10-04) for the layout the engine calls.**  `Y[T x ldy] =
  X[T x K] . W[N x K]^T` in f16 with an f32 accumulator (`Gemm::f16`/`Gemm::bf16` in `src/prefill/gemm.cu`, whose
  cuBLAS call is (OP_T, OP_N) - the weight is the transposed operand), on the matrix units at Intel's real tile
  (8x16x16) and in plain FMA for every ragged shape and every device without a config, plus the engine's own
  `bf16 -> f16` operand conversion (with its clamp, not an overflow into Inf).  Ten verdicts, green on the Arc and
  the FMA half green on all three implementations, with the transposed-operand bug falsified to fail four of them.
  What remains in this area: **`src/prefill/kernels.cu` (53 KB), `moe_fused.cu`, `moe_fused_iq.cu` and
  `moe_mmq.cu`** - the fused prefill kernels and the quantised-weight MMQ path (`Gemm::native` over ggml types),
  which is the same porting grind as the expert tier with a different reduction; and the device-layer gap recorded
  in `NEXT.md` (descriptor bindings are offset-0, and the engine passes row slices by pointer).
* **Engine integration to a served model:** ~1 session for a CPU-verifiable path, then a GPU window on the
  hardware that is actually being targeted.
* Cheaper alternative worth considering first: the port buys *compatibility* (Arc, and any Vulkan GPU). If the
  goal is only "Arc runs Strata at a useful speed", the fastest legitimate route is the one Intel validates —
  their LLM Scaler container. This port is the route where the engine stays ours and the model stays GGUF.
