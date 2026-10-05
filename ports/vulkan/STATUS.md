# Status — what is done, what is verified, what is not

## Done and verified in this session

Everything below is backed by a command that exits non-zero on failure. Re-run it with:

    bash ports/vulkan/gates/run_gate.sh          # compiles the shaders from source, validates the SPIR-V,
                                                 # checks each shader's declared local size, then runs the gate

**Result: the gate prints its own totals and those are the authority. On 2026-10-04, after the Radeon RX 7900 XTX
was swapped for an Arc Pro B70 and after stages 3, 4, the prefill GEMM, the quantised multi-token arms, the
short-step decode attention, the f16 KV gather, the QSA selection, the f16 KV append, the Q4_0 KV path (with its
Walsh-Hadamard rotation), the hybrid K8V4 mode, descriptor OFFSETS and the first SAMPLER kernel landed, the box's GPU
run was **265 passed / 0 failed / 0 skipped** on the Intel ICD (`Intel(R) Graphics (BMG G31)`, Mesa 25.2.8 / ANV,
Vulkan 1.4.318, subgroup size 32) - 253 / 0 / 3 on llvmpipe and 256 / 0 / 2 on the radeon ICD, which now picks the AMD
iGPU because the discrete card is gone (both
skip cooperative matrix, whose driver does not advertise the extension, and the prefill SPLIT, which needs the M8
tile).  **The Arc has no skips at all:** the last one (`gemm_coopmat`) was the port
misreading the device - BMG's matrix config is M8 N16 K16, not the M16 the criterion demanded - and since then the
matrix path RUNS on XMX, including the prefill GEMM.  On the iGPU, 256/0/2 becomes 255/1/2 when the flaky budget-requery case fires
when the budget-requery case fires - and as of this increment it fires on EVERY run, not intermittently: the case
compares two queries of the driver's free-memory figure and RADV's moves ~2.8 MB against the 1.7 MB tolerance
(`requery delta: budget 2793472 bytes, usage 0 bytes`).  **That is the box, not the port: the PREVIOUS commit's
binary fails it identically, 2 runs of 2**, and the usage delta is 0 - it is the budget figure drifting on an iGPU
that shares system memory with everything else.  Verified by building the previous commit in a worktree and running
the same ICD.  Before the swap the same gate read 160 / 0 / 0 on RADV and on radeon. That count has gone
stale seven times in two days; read the last line of your own run.** All three available implementations are exercised
again by `run_gate.sh`: it used to stop at the Intel skip, which meant the cross-implementation arm never ran on this
box after the swap (`NEXT.md`). 68 kernels, 19 shared includes, one generated table file (`harness/iq_grids.hpp`,
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

**The quantised multi-token path needed no new kernel, and is now armed and composed (2026-10-04).**  Reading the
engine showed the prompt path reaches quantised weights two ways: `Gemm::native` **dequantises** the ggml blocks to
f16 and calls the GEMM above (so the quantised prefill needs a DEQUANTISER, not a GEMM), and `iq_mmvq` is the
multi-token row-kernel path (the engine specialises 1/2/4/8 columns).  The port's seven per-format row kernels
already take `ncols` - they were gated only at 1 and 2 - so the multi-token path is now armed at 8 for all seven and
at 32 for IQ2_S, and a new COMPOSED case runs the device's `quantize_q8_1` into the device's `iq2s_mmvq` with the
oracle reading the device's own quantised bytes.  That composed case is also the session's sharpest lesson: its
first version PASSED under an injected per-column-stride bug in the quantiser (an oracle built from the thing under
test follows that thing's mistakes), and it only became an oracle once it asked the question the bytes cannot
answer - per-column liveness.  `NEXT.md`'s stage-5b block has the injection and the fix.

**The ATTENTION BLOCK has started: the short-step decode attention is ported and gated (2026-10-04).**
`attn_decode_short.comp` carries the engine's own contract for `native_flash_attn_short_step` - Q24x256, KV2x256,
scale 1/16, `[capacity,2,256]` f16 KV cache, additive f16 mask broadcast over the heads, GQA 12:1 - re-derived as an
algorithm rather than micro-ported, because the CUDA is a vector flash attention pinned to one compiled binary
(128 threads, lane/float2 mapping, `__shfl_xor` tree, a chosen fma order) and this port's reductions are barrier
trees by design.  Five arms (1 live key, 3 keys, a full window, a masked full window, a masked half window) against
a double-precision softmax oracle, 24 heads x 256 dims each, plus a shape contract that REFUSES width 0 and
width 257.  `wg_reduce.glsl` gained `wg_max`.  Falsified: the plausible wrong GQA grouping fails all five arms at
exactly half the values.  What the block still needs - the prompt path (`qsa_prompt_attn.cu`, `qsa_select.cu`), the
QSA selection/indexer, and a tiled version of this kernel - is listed in `NEXT.md`.

**The f16 KV gather landed with it: the window the attention reads, and the engine's launch rule measured
(2026-10-04).**  `kv_f16_gather.comp` is the F16 sibling of `kv_q8_gather` (`src/kernels/cuda/qsa.cu`), the producer
of the `[id][kv_head][head_dim]` window the attention consumes.  Three things came out of it that are worth more than
the kernel: the POOL's row index CARRIES the head (`[page][kv_head][page_size][dim]`, not `[row][kv_head][dim]` - a
fixture written the intuitive way has the same total size and fails every value); `ids[id]` is a POSITION and `id` is
an INDEX, and a softmax over the keys cannot see a permutation of them, which is why the composed gather-then-attend
arm runs with a position-dependent mask (injected `cell = id` -> worst rel 9.19e+03); and **THE GRID IS THE CAPACITY,
NOT THE LIVE COUNT** - the engine's own launcher comment, now measured through this port's record/replay path: a
capacity grid replays at a grown `n_ids` and writes 8 of 8 rows, the live-count grid writes 6 of 8 and loses two
silently.  That arm is a NEGCTRL: it passes only if the defect reproduces.  The q8 sibling's case had described
this rule in its verdict text while sizing BOTH its window and its grid from the live count, so the guard region it
claimed to check did not exist; fixed to a capacity of 8.  Arc 209/0/0.

**The selection landed next, and with it the first END-TO-END decode chain (2026-10-04).**  `qsa_block_scores.comp`
and `qsa_block_topk.comp` (`src/kernels/cuda/qsa_select.cu`; the engine keeps the second as `qsa_block_topk_ref`) are
the QSA indexer's block scores and the weighted top-k - the producer of the very `ids` the gather consumes.  The gate
now runs the whole decode data path: **pool -> scores -> selection -> gather -> attention**, each stage against its
own oracle and the chain end to end (ids compared, the gathered window compared against the pool rows the selection
named, then the attention output, all four verdicts).  The two rules worth remembering are that a block's WEIGHT IS
ITS CELL COUNT - a complete last block weighs 0 and must be skipped even with an enormous score, and a 1-cell tail
contributes 1, not R, because getting this wrong selects cells that do not exist - and that the ids being ASCENDING
is part of the contract, since downstream the position is the window row and the mask is indexed by it.  Seven
selection arms (a budget cut inside a block, three-way ties at the boundary, a zero-weight tail, a 1-cell tail, both
identity cases, and everything-but-one-cell), three falsifications.  Arc 221/0/0.

**The KV append closed the decode loop, and the harness had a ceiling this increment hit (2026-10-04).**
`kv_f16_append.comp` (`src/kernels/cuda/qsa.cu`) is the WRITE half of the gather's pair - a token's f32 K/V converted
and stored into the paged f16 cache at the row the page table names.  Four rules, four arms: the conversion is
compared as BYTES (it is the arithmetic), a negative page writes NOTHING at all (the injection "treat a non-resident
block as page 0" is caught by that arm and by nothing else), the two layouts (physical page vs identity host copy)
are one kernel with a flag, and re-appending a position overwrites it.  The chain runs append -> gather -> attention
over six tokens appended across a page boundary, at worst rel 8.37e-06.  **And the first full run came back
`FAIL radeon_icd` with no numbers: `VK_ERROR_OUT_OF_POOL_MEMORY`, because the harness's descriptor pool had a
hardcoded `maxSets = 64` and one set per pipeline.**  That is a ceiling on the CASE COUNT, not on a kernel, and it
fails at whichever implementation runs last - fixed by growing the pool on demand and printing when it does
(`descriptor pool 2 created`).  Arc 226/0/0.

**The Q4_0 KV path landed, and with it the first gate arm that tests a DESIGN CLAIM rather than a kernel
(2026-10-04).**  `fwht256.comp` (the orthonormal 256-point Walsh-Hadamard rotation), `kv_q4_append.comp` (ggml's Q4_0
group with the engine's deterministic tie rule) and `kv_q4_gather.comp` (the dequantising reader) - the engine's three
KV storage formats are now all in the port.  The design (rotate K, V AND the query, quantise in the rotated basis,
de-rotate the output once) is legal only because H is orthonormal, so the rotation is gated against an EXPLICIT
Hadamard matrix, against `H(Hx) = x` and `<Hx,Hx> = <x,x>`, and finally THROUGH the real attention kernel: rotate q,
quantise+rotate K/V to Q4_0, attend, de-rotate, and compare with the unrotated attention - worst absolute deviation
0.644 against a window that is itself off by up to 0.474.  **The invariants cannot see a flipped butterfly sign
convention** (the other convention is also orthogonal and self-inverse - it is simply a different basis), and the
injection proves it: flipping the branches fails only the explicit-matrix arm.  Round trip 0.4719 against the group
bound |d|/2 = 0.4742.  Arc 233/0/0.

**The hybrid K8V4 mode landed, with no new kernel - and llvmpipe caught a race in the case that the Arc did not
(2026-10-04).**  The mode IS the wiring (`KV_MODE 3` of the engine's own `kv_hybrid_parity.cpp`): INT8 K, rotated
Q4_0 V, the attention in the mixed basis, and one inverse Hadamard on the output.  The asymmetry is the content -
ONLY V is rotated, so the query is not either - and the arms pin it in five ways, of which the interesting two are a
tight oracle over the same two windows (worst rel 1.68e-06) and the de-rotation tested ALONE against `H` applied
once (worst rel 1.19e-04), because a tight comparison cannot see what the quantisation cost and a bounded one cannot
see a basis error inside its slack.  **The cross-implementation arm earned its keep:** the q4 append's unused K half
was bound to the same pool as its V half, so two writers raced on the same bytes - harmless on the Arc, **3071 of
3072 wrong on llvmpipe**.  One implementation would have shipped it.  Two wiring injections (the de-rotation applied
twice, and omitted) fail exactly the arms they should, and both needed fixing first: the "twice" version dispatched
the same input again (a no-op), and the "omitted" version left a push constant unused so `-Werror` failed the build
and the run printed the STALE binary's numbers.  Arc 238/0/0.  See `RUN-ON-B70.md` for where this sits on the path to
a running engine.

**Descriptor OFFSETS landed: the recorded blocker between the port and the engine is retired (2026-10-04).**
The engine reaches a row slice by pointer arithmetic (`Y + t0 * ldy`, `X + t0 * K`) while the port bound every
descriptor at byte 0 - which forced the GEMM case into a workaround of one buffer per half.  `Buf::offset` (a copy of
the handle with the offset set, default 0 so every existing call site is unchanged) is now honoured at the single
descriptor-write site, with four gate arms: a view reads from its offset, the offset-0 binding still reads from zero,
neither dispatch writes past its count, and - the one that matters - **the L(i)=B(i) lookalike control**, because an
IGNORED offset returns the offset-0 data and looks like a plausible answer rather than an error.  The device's
`minStorageBufferOffsetAlignment` is now recorded per device and printed by the case: **the Arc reports 4 bytes**
(the port's earlier comment guessed 64), which is what makes the engine's slices bindable as they stand - on a device
with a 64-byte limit that arithmetic would need padding.  An unaligned offset is refused by name before it reaches
the driver.  Arc 242/0/0.  **Honest limit:** the aligned path is gated on three implementations, but the refusal path
was NOT reproduced out of tree - the probe's device contract differs from the gate's (it reads the LEDGER budget
instead of the driver's and refuses the allocation first), and chasing it was not worth the time; what is verified is
that the refusal machinery fires and names its cause.

**The sampler is nearly complete, and the port can emit a TOKEN two ways (2026-10-04).**  `sampler_greedy.comp` carries
`sampler_greedy_kernel` (`src/kernels/cuda/sampler.cu`) with the penalty pair it calls - the `--temp 0` path - so the
port has a complete deterministic generation path: logits in, a token out.  The rules a paraphrase inverts are all
arms: the repeat penalty MULTIPLIES for a non-positive logit and DIVIDES for a positive one (dividing unconditionally
inverts it on half the vocabulary); the presence penalty is a boolean, not the count, while the frequency penalty
carries the count; the window is the TAIL, so a head-only hit is not penalised; and ties go to the LOWEST index, with
0 for an all -inf/NaN row.  Ten arms, each stating the token it is built to produce, checked against a transcription
of the rule as well.  **The falsification found a decorative arm:** the presence arm PASSED under the injection
because its fixture let the penalised token lose under both rules - only an injection can say that.  Rebuilt, all
four injections now fail exactly the arm built for them.  Arc 253/0/0.

**The general sampler landed too - top-k, top-p, min-p, temperature and the Philox draw.**  `sampler_kernel.comp`
plus `common/philox.glsl`, so every path a decode step needs is ported.  The chain is llama.cpp's, and two of its
details are things a careful port would "fix": the penalties belong on the RAW logits, ONCE, before the filters (the
source records fixing a second application after the temperature), and **`temperature == 0` is not greedy** - it
scales every survivor to zero, making the draw UNIFORM over the shortlist.  The engine's Philox constants are its own
(0x9E3779B9 / 0xBB67AE85), NOT Random123's, so a swap yields a valid generator with different numbers.

**The RNG is pinned through the shortlist rather than read back:** 64 EQUAL survivors at temperature 0 make
`floor(u*64)` observable in the returned token, so 16 seeds pin ~96 bits of the stream - and that arm is exact,
because equal probabilities make the softmax cancel out of the cumulative walk.

**Nine injections, all caught, and two of them changed the case.**  The volume of the penalty bitmap, the round count,
the constants and the key/counter assignment do not go unnoticed: the RNG arm returns 0/16 for each.  But the doubled
penalty (the source's own recorded bug) initially passed 184/184, because the arm asserted the head was REACHABLE and
a two-token shortlist makes that true under either order; the fix was a `top_k = 1` arm, where a one-token shortlist
makes the draw irrelevant and the order exact - it now returns -1/16.  Its fixture then had to be tightened again
(2.0 against 1.5, not 0.2), because the margin that made the majority test safe made the flip invisible.  Separately,
three fixtures were strengthened BEFORE injecting, on noticing that equal logits make `inv_t` irrelevant and that a
membership check cannot see too FEW survivors.  Three of the nine injections also needed the compile target corrected
to the shader that INCLUDES the file rather than the include itself.

**The gate now checks the PORT MAP before it runs a single kernel.**  `tools/check_port_map.py` verifies
`PORT-MAP.tsv` - the classification of every `kernels::` symbol the DECODE path (`src/core/`) calls into kernel /
host / todo - against the engine's sources and the built shaders, and fails on an invented symbol, on a `kernel` row
naming a shader that is not built, and on any decode-path symbol the map does not mention.  That last rule is the
point: a new call site cannot join the decode path unnoticed.  It says 77 symbols - 17 kernel, 50 host, **10 todo** -
so the decode path's remaining GPU work is ten named kernels rather than the ~250 KB of prefill and fused-MoE code.
Falsified three ways before landing.

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
