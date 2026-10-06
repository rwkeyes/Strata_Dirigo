# Status — what is done, what is verified, what is not

## THE PREFILL'S EXPERT-PATH FLUSH WAS A MAPPED GRID, NOT A KERNEL (2026-10-06, `vega`, Arc Pro B70)

**DONE.** The IQ grid tables are now placed with `Ctx::alloc_device` instead of `Ctx::alloc`:
`vulkan/src/kernels/iq_vk.cpp`'s `iq_grids()` (six grids) and `vulkan/src/kernels/matvec_vk.cpp`'s `grid_for()`
(four). `Ctx::alloc` picks `mem_type_` — HOST_VISIBLE|HOST_COHERENT, preferring a device-local heap, i.e. the
BAR-mapped VRAM type on this card — so every grid came back MAPPED, and `Ctx::dispatch`
(`vk_compute.cpp:1520`) flushes the live batch on any mapped binding. `iq_dequant_f32.spv` binds all six grids,
so every dequant dispatch flushed, and every dispatch after it was swept into a ~5-dispatch batch and paid a
submit+wait. **The host-visible rule itself is unchanged** (a grid is a shader-READ constant the host writes once
and never reads back; on llvmpipe `alloc_device` lands in the same device-local+mappable type, so nothing there
can change). No shader changed; no bound was widened; no case was skipped or dropped.

**VERIFIED.** Gate on the Arc `intel_icd` **886 passed / 0 failed / 0 skipped** — the case count did NOT fall
(886 at HEAD and 886 here), nothing is skipped, no bound was widened. lvp 872/0/4 (the same four documented
skips, including the device-local-and-mappable path my change touches) and radeon 875/2/2 — both failures are the
documented moving-failing-set intermittent (`bf16_gemv_fp32_mmvf_cols`, and `ple_block entry` 10239/10240 worst 1,
the same one-element intermittent `/tmp/meas/gate_perf4` records on the Intel arm) and neither is a case this
change touches; smoke 60/0/0 (20/0/0 per ICD). Logs: `/tmp/meas2/gate_arc.log` and
`ports/vulkan/harness/build/icd-{intel,lvp,radeon}_icd.log`. Ids: `output  :` line md5 **`56a0b28d2de6`**
(199-token) and **`3aed108cceee`** (8-token) in every one of the TEN runs of this batch (the three pre-change arms
and the seven post-change ones), by the same convention the file has always used (md5 of the line including its
newline). `check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (no
`kernels::` symbol changed).

**MEASURED.** 199-token arm, same binary/prompt/flags: prefill **16.02 → 19.74 tok/s** (12,359.3 ms → 10,128.6 /
9,939.4, n=2) and flushes **13,619 → 3,021** (submit 69 → 16 ms); 8-token arm prefill 4.45-5.28 → **6.63** tok/s
(1,056.4 ms). Decode UNMOVED: 8.36/8.34/8.35 before, 8.34/8.31/8.22/8.35 after. The cause was named by
`addr2line` over `STRATA_VK_FLUSH_STAT`'s own call-site backtraces: the four `iq_dequant*` sites carried 10,737 of
the 13,619 flushes at ~4.9 dispatches each, while `native_gdn_out_norm <- prefill::gdn_recurrence` showed exactly
128 per flush (the batch limit) — so the ordinary arena dispatches never tripped the rule.

**THE RE-TEST, AND WHY THE TILED PATH IS NOT THE DEFAULT.** `STRATA_VK_PREFILL_TILED=1` re-measured (the 46-54 us
dispatch cost that made it lose is now 4.6-5.4 us): **1.49x faster on the PRE-narrowing binary** (8,294.1 ms /
23.87 tok/s) but **indistinguishable on the narrowed one** (19.52 tok/s, n=3, against untiled 19.74, n=2). The
1.49x was the flush, not the tile — the untiled GEMM's ~7,700 dispatches per chunk each landed in a flushed
~5-dispatch batch; the tiled kernel's few hundred did not. So the shipped default is UNCHANGED and the env arms
stay opt-in. `STRATA_VK_PREFILL_COOPMAT=1` = **18.02 tok/s** pre-narrowing (10,989.3 ms) and **20.14 tok/s** on the
narrowed binary (9,829.9 ms), ids identical in both — a wash against untiled's 19.74 and tiled's 19.52, so all
three GEMM arms are within ~3% and none is claimed as a win.

**NOT DONE.** The expert matmul is still dequant-to-FP16-then-GEMM rather than the reference's MMQ shape
(quantized operands in shared memory, packed integer dot behind a runtime `VK_KHR_shader_integer_dot_product`
check — this card reports `int dot: 0` — the scale applied once, no separate dequantized buffer). After the
narrowing, `dequant` (2,054 ms, 21%) is larger than the GEMM (596 + 616 ms, ~12%), and `iq_dequant_f32` costs
~0.55 ms per dispatch at ~24 GB/s of traffic — dispatch count and occupancy (224 of each 256-lane workgroup's
lanes idle), not bandwidth. **Also open:** the narrowed binary reports 59,640 live dispatches against 56,468
before (+3,172, same prompt, same GEMM choices) — unexplained, recorded.

## THE PER-ROUND PREDICTION IS FALSIFIED, AND SO IS THE BARRIER — the delivered decode gap is SPECULATIVE WASTE (2026-10-06, `vega`, Arc Pro B70)

**DONE.** The verify window's five per-token loops are batched to ONE dispatch per ROUND: `gdn_conv_l2_multi`,
`gdn_ab_multi`, `native_router_top10_multi`, `shared_expert_multi`, `native_moe_combine_multi`.  The binding is
`vulkan/src/kernels/vk_multi.hpp`; the round forms (`*_n`) live in `gdn_vk.cpp` / `ple_vk.cpp` / `qsa_vk.cpp`.
Two are HOST-ONLY (`native_router_top10.spv` already had the token dimension; `shared_expert_multi` uses the
module's own `ncols` column layout, which is the single-column layout replicated, so token t's column is bitwise
the single-token call).  Three shaders gained a token dimension: `native_moe_combine` (flat `n_embd*n_tok` grid),
`fused_gdn_ab` (token = `gl_WorkGroupID.y`), and `fused_gdn_conv_l2`, whose running window is now
TOKEN-INVARIANT (`stream[a..a+3]` of `[history(3) | qkv]` — the CUDA multi's own `win[j]` rule) with
`write_hist = 0` for the multi's "history is NOT written" contract.  New measurement-only instruments:
`STRATA_VK_NOBARRIER`, `STRATA_VK_BARRIER_HAZARD`, and a chain-barrier counter in the disp stat.  The four
descriptor-offset SKIPs on the `_multi` arms are RETIRED (no per-token descriptor exists any more); the harness's
six direct dispatches carry the new push-constant sizes, and its case-(G) `nw.q8_1` fixture is sized for `n_tok`
columns as `shared_expert.hpp` requires.

**VERIFIED.** Gate: Arc `intel_icd` **886 passed / 0 failed / 0 skipped** — the count has NOT moved, nothing
skipped, no bound widened; **lvp_icd 872/0/4** (868/0/6 before — the four retired `_multi` alignment SKIPs now
RUN, and the four remaining skips are the documented pre-existing ones) and **radeon_icd 876/1/2** (the
documented intermittent `bf16_gemv_fp32_mmvf_cols`); smoke 20/0/0 on all three.  The `_multi` arms are bitwise
against their single-token oracles on all three ICDs.  Ids `3aed108cceee` / `56a0b28d2de6` in all EIGHT
before/after runs (`/tmp/meas/before2.*.log`, `/tmp/meas/after.*.log`).  `check_port_map.py` passes;
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically
(`168 — 97 kernel, 0 shader, 47 host, 0 todo, 24 refused`).

**THE PREDICTION IS FALSIFIED.** Recorded dispatches fell **13.0-16.5%** (81,154 -> 73,126 and 91,060 -> 78,952
on the 8-token arm; 68,999 -> 59,111 and 75,005 -> 62,669 on the 199-token arm) while the verify window's `sync`
moved **1.2-1.4%** (197.275 -> 194.971 and 240.720 -> 237.764) and decode **+1.0-1.5%** (8.12 -> 8.21,
6.67 -> 6.74, 8.25 -> 8.36, 7.25 -> 7.36).  Per REMOVED dispatch that is **4.6-5.4 us**, against the port's own
bench figure of 46-54 us — so the five loops' dispatches are real, were really per-draft (the count DID fall:
422.5 dispatches/round removed at T=2, 637.2 at T=4 = 105.6/106.2 per window token), and are CHEAP.  The
`--spec 4` prediction of ~14 tok/s did not happen; the measured value is 6.74.

**THE BARRIER IS FALSIFIED TOO.** 21,826 chain barriers at T=2 / 26,668 at T=4 per run (~1,149 / ~1,404 per
round — about 30% of recorded dispatches, NOT one per dispatch).  Elision is impossible
(`STRATA_VK_NOBARRIER=1` breaks the prompt path: `prefill: routed id out of range`), and a conservative
bound-region hazard rule still required **21,825 of 21,826** barriers with `sync` unchanged (194.882 vs 194.971;
237.662 vs 237.764).  Ids identical.  A barrier is therefore < ~5 us, bounded by the removal, not the barrier.

**THE DECODE GAP, RE-FRAMED.** The `--spec 4` 8-token window computes 6 draft tokens in 240.6 ms = **40.1 ms per
computed token (~24.9 tok/s raw compute)**; **1.68 tokens/round are accepted**, so a delivered token costs
`240.6/1.68 = 143.2 ms` (6.98 tok/s) and **72% of the window's compute is discarded**.  Against llama.cpp's
36.52 tok/s (27.4 ms/token) the port's RAW compute is **~1.47x** off while the delivered 4.4x is speculative
WASTE — an engine/MTP property, not Vulkan overhead.  Acceptance is flat at 1.68 across T=2/4/6 (observation,
not chased).

**THE GATE GAP IS CLOSED.** `strata_vk_cudart_smoke` was a CMake target NOTHING ran, so the `cudaStreamQuery`
fix was unguarded.  `run_gate.sh` now sources `gates/smoke_sources.sh`, builds the target, runs it per ICD, and
FOLDS its cases into the totals (the count RISES; a build failure or a no-device run is a FAILURE, not a skip).
`inject-verify.sh cudart-stream-query-nofix` is the other direction and PROVEN (`outstanding before/after = 1/1`
-> FAIL; `1/0` -> PASS 20/0/0).  One arm was over-strict and is fixed at the cause: on llvmpipe EVERY arena
buffer is host-visible+coherent (`STRATA_VK_MEM_TRACE=1`: one heap, `host_visible=1 host_coherent=1`) so
`vk_compute.cpp:1521`'s rule flushes at every dispatch and nothing is queued — the arm now asserts the RULE
(queued OR flushed-because-mapped) and prints which condition held, 20/20 on intel, lvp and radeon.

**NOT DONE.** The ~21.7 ms/draft was NOT attributed: the remaining per-token work is the QSA branch's per-token
loops (`verify.cpp:778-900`) and none of it is measured (`STRATA_VERIFY_PROFILE`/`_TRACE` are closed by design).
The barrier's absolute cost is bounded, not measured.  On llvmpipe the visibility arm is vacuous.  A batched form
for the QSA per-token family is the next honest lever.  `submit_recorded` is still ZERO and unexplained.

## THE DECODE: 150 ms/token ATTRIBUTED TO THE VERIFY WINDOW'S GPU EXECUTION; `--spec 2` IS A MEASURED 1.22x (2026-10-06, `vega`, Arc Pro B70)

**DONE.** The decode got the phase table it never had. Two decode-side marks in `Verifier::run` (`ms_launch`,
`ms_sync`; `src/core/verify.cpp` + `include/strata/core/verify.hpp`) and three new instruments
(`STRATA_VK_XFER_STAT` transfer-by-call-site with a decode split, `STRATA_VK_FLUSH_STAT` flush-by-trigger,
`STRATA_VK_QUERY_STAT`/`_NOFIX`, plus segment accounting inside `STRATA_VK_DISP_STAT`) priced it. The 8-token arm
(`--spec 4`): `verify window ... host(stage) 0.986  launch 4.903  sync 240.623  commit 4.122 ms/round` over 19
rounds = **4,762 ms, 99.1% of the 4,803 ms decode; `sync` alone 4,572 ms (95.2%)** — and the segment counter says
what that sync is waiting on: **57 segment submits, 91,060 recorded dispatches, submit 1 ms, wait 4,742 ms**, i.e.
the GPU executing ~4,792 recorded dispatches per round at the port's own measured small-dispatch cost (~46–54 µs).

**VERIFIED.** Gate on the Arc **886 passed / 0 failed / 0 skipped** — the SAME case count, nothing skipped, no bound
widened; lvp 868/0/6; radeon 876/1/2 (the documented `bf16_gemv_fp32_mmvf` family). Ids `3aed108cceee` (8-token) and
`56a0b28d2de6` (199-token), identical across `--spec 2/4/6` and before/after the `cudaStreamQuery` fix. **The measured
win: `--spec 2` → 8.12 tok/s decode vs `--spec 4`'s 6.66 (ids identical; tokens/round flat at 1.68; the window pays
~110 ms fixed + ~21.7 ms PER DRAFT).** 199-token arm same session: 8.27 vs the documented 7.25.

**A RETRACTION, RECORDED.** The previous section called the decode "the transfer path" (~430 transfers/token).
**Wrong, and the instrument said so**: `STRATA_VK_XFER_STAT` shows the 13,774 transfers are the LOAD (site 0 =
`fill_slot_blocking`, 12,288 calls / 23.42 GiB), and the **decode-phase transfer count is 0**. The division by 32
decode tokens divided a one-off model load by a token count.

**THE LAST WORKER'S OPEN RISK, MEASURED AND FIXED.** `cudaStreamQuery` DID report a pending live batch as complete:
a new arm in `vulkan/tests/cudart_smoke.cpp` measures outstanding-before/after = **1/1 with `"no error"`** under
`STRATA_VK_QUERY_NOFIX=1` (a wrong "complete") and **1/0** with the fix. `cudaStreamQuery` now flushes the pending live
batch before answering. Latent in the shipped decode (38 calls, 0 with a batch pending).

**NOT DONE.** The ~2x lever is named and priced but NOT attempted: the per-draft cost is ~261 recorded dispatches
(≈5 per layer), from five per-token LOOPS in `vulkan/src/kernels/verify_vk.cpp`
(`gdn_conv_l2_multi`, `gdn_ab_multi`, `native_router_top10_multi`, `native_moe_combine_multi`, `shared_expert_multi`);
batching them per round targets sync 240.6 → ~110 ms at T=4 (decode ~14.0 tok/s). The prefill's 4,293 flushes
(~3.4 dispatches/flush, the fill-128 never reached, decode: 0) are a PREFILL lever, re-labelled as such and not
narrowed. `STRATA_VERIFY_PROFILE`/`_TRACE` are closed by design (`gpu_stamp` is an unported diagnostic that refuses
loudly) and were not chased.

## THE DISPATCH LAYER: PREFILL 5.14 → 16.17 tok/s, AND DECODE UNMOVED AT 7.25 (2026-10-06, `vega`, Arc Pro B70)

**DONE.** A new instrument (`STRATA_VK_DISP_STAT=1`) priced the submission layer: every dispatch was its own
command buffer, fence, submit and wait — 249,878 of each for 233,768 dispatches, with the fence `wait` 88.7% of the
dispatch layer and 50.8% of dispatches coming from `Gemm::native`'s per-token loop. Two changes landed:
`Gemm::native` batched to three dispatches per projection (the shaders already walked `ncols`; bit-identical per
column), and the live dispatch path batched into one command buffer flushed at every observer. Prefill
38,513.8 ms / 5.14 tok/s → 12,245.7 ms / **16.17 tok/s** (11,391 ms / 17.38 in the run before the gate fix); TTFT
38.9 → 12.8 s; submits 249,878 → 29,771.

**VERIFIED.** Gate on the Arc **886 passed / 0 failed / 0 skipped** — the SAME case count as before the change
(nothing skipped, no bound widened); lvp 868/0/6; radeon 875/2/2 (the documented `bf16_gemv_fp32_mmvf` family).
Equivalence: `198 1 198 1 …` md5 **`56a0b28d2de6`** and the 8-token arm `4653 8 15 …` md5 **`3aed108cceee`**, both
identical to the pre-change runs. The red gate on the way (881/5/0, `doorbell ring` ×3 + `sample_tokens (mapped
out)` ×2) and its three causes are recorded in `NEXT.md` and `PERFORMANCE-B70-2026-10-06.md`.

**NOT DONE — AND IT IS THE BIG ONE.** **Decode is 7.25 tok/s in every run tonight**, 5.04× off the same-card
reference (llama.cpp Vulkan, 36.52), and untouched by any of the above. `submit_recorded` is called ZERO times; the
by-arm submit split shows the decode cost is ~13,774 `begin_oneshot` transfers (≈430 per decode token) still paying
the per-call round trip removed for dispatches. Also unmeasured: the per-arm `wait` split, a decode phase table
(the timing instrument is prefill-only), and `cudaStreamQuery`'s treatment of a pending batch (reasoned, not
measured).

## THE ARITHMETIC IS CHECKED AGAINST AN INDEPENDENT IMPLEMENTATION (not a transcription): 10 mmvq formats vs ggml-cpu's own `vec_dot` on the pack's REAL weight rows (`coder-iq1_m`, Intel Arc Pro B70) (2026-10-06, `vega`)

**WHAT IS DONE.** The gate's numeric oracles for the quantised mmvq shaders are HOST TRANSCRIPTIONS of the same
CUDA dots the shaders were transcribed from (`iq2s_dot_host`, `iq3xxs_dot_host`, `q4/q5/q6_dot_host`), on synthetic
bytes - so a green gate there means two transcriptions of one expression agree. The new MEASUREMENT-ONLY tool
`ports/vulkan/ref/ref_vs_ggml.cpp` (built by `ref/build_ref.sh`; on no engine path, touching no engine TU) drives
the port's shaders through the port's own device layer and compares each against **ggml-cpu's own function**,
`ggml_get_type_traits_cpu(ty)->vec_dot`, in `~/llama-050/build-vulkan`, plus a second reference from ggml's
`to_float` (dequantise, then an exact double dot). Weights are REAL rows read at `native_experts.txt`'s offsets; one
synthetic activation per format is quantised once by ggml and fed to both sides (int8 values identical; fp16 scale
identical - the reference's scale is rounded to the same fp16).

**WHAT IS VERIFIED.** `10 passed / 0 failed / 0 skipped`, 32 real rows per format (`/tmp/ref_run2.log`): IQ3_XXS
(blk.0 gate, 6.4e-07 of sum|terms|), IQ2_S (blk.1 gate, 2.1e-06), IQ3_S (blk.17 gate, 9.4e-09), IQ4_XS (blk.47
gate, 1.3e-08), IQ4_NL (blk.0 down, 2.3e-08), Q2_0 (blk.1 down, 1.3e-08), Q4_K (blk.0 attn_gate, 2.5e-08), Q5_K
(blk.1 attn_qkv, 1.3e-08), Q6_K (output.weight, 1.2e-08), Q8_0 (blk.47 ffn_down_shexp, 2.2e-08). The bound is
`1e-5 x sum|terms| + 1e-6 x |value|` PLUS, for IQ2_S and IQ3_XXS ONLY, the per-part integer-truncation allowance
`sum over parts of d_w x d_a` read from the bytes - those two round `(ls*sumi + sumi/2)/2` per part where ggml
accumulates exactly, which is exactly why they are the two at ~1e-6 while everything else is at 1e-8. The worst case
reaches 15% of the bound; no bound was widened. The positive control (eight spread weight bytes flipped per row
between the two sides) must FAIL: all ten formats moved, `ctl-unchanged=0`. On this pack Q8_K exists only as an
ACTIVATION type (no type-15 tensor), stated so nobody hunts for a Q8_K weight row.

**A REAL MULTI-TOKEN PROMPT.** 19 ids from the pack's own tokenizer (`def fib(n):\n    if n < 2:\n        return n\n
   return`, `decode(encode(s)) == s` exact), `--max-new 4`, output ids `15336 1393 8 198` -> ` fib(n)\n`, completing
`return fib(n)` (`/tmp/ref_prompt.log`); prefill 18 tokens 37,633.6 ms (0.48 tok/s), decode 4 tokens 1,346.7 ms
(2.97 tok/s, 337 ms/token), `CPU experts 0.00`, `RAM 0 blobs, files 0 blobs`, `pcie experts 0.00`, `100% VRAM
resident`. **This shows the chain conditions on a longer prompt and decodes more than one step; it does NOT show the
text is correct or coherent** - one prompt to a high-probability Python continuation, greedy, `--spec 4` with 0 of 9
drafts accepted.

**WHAT IS NOT.** (1) THE END-TO-END CPU-HYBRID A/B. The port CANNOT reach the engine's CPU expert kernels:
`native_quant_act`/the row kernels are `refuse_cpu_row` (`vulkan/src/kernels/native_expert_vk.cpp:152`) and every
non-all-resident config needs a non-resident expert computed on the CPU, so the refusal IS the observed behaviour.
Running the A/B would mean writing the CPU-hybrid expert path into a shipped port TU; the instruction for that case
is to stop and report, so it is reported and NOT attempted. The kernel-level half is done and is the same
arithmetic (`src/kernels/cpu/native_expert.cpp`: "Nothing here is Strata arithmetic: the activation quantizers and
the row dot products are ggml-cpu's" - the comparison runs against exactly those functions). Unmeasured: the
whole-model end to end (router, PLE, attention, KV cache, sampler, logits/top-k). (2) A CORRECT TOKEN: the prompt
above is a chain-executes-and-conditions result, not a semantic one. (3) The `verify.cpp:943` device-plan arm, the
`pools 2..61` growth, and `z820b` - all still untouched.

## `resident_plan` PORTED, THE P6 VERIFY WINDOW CAPTURES AND LAUNCHES, AND THE ENGINE PRODUCES ITS FIRST REAL-CONTENT TOKEN (`coder-iq1_m`, Intel Arc Pro B70) (2026-10-06, `vega`)

**WHAT IS DONE.**  The LAST unported symbol between the all-resident verify window and the launch is wired:
`strata::kernels::resident_plan` (`verify.cpp:938`, the `if (all_resident_)` arm of the window's per-group plan) is
DEFINED in `vulkan/src/kernels/verify_vk.cpp` over the new shader `ports/vulkan/shaders/resident_plan.comp` - the
CUDA `resident_plan_kernel`'s device-side group-by over the routed ids (`ptr[grp] = cache_base + slot_off[slot]`,
the per-entry (thread,token) map, and the counts/terminators), read and written as lo/hi uint32 word PAIRS because
glslang has no 64-bit buffer index; the plan buffer is bound WHOLE so the ptr region's 8-byte-aligned offset is never
a descriptor offset.  The refusal (`refusals_vk.cpp`), the `REFUSED` set and the `todo` row in `make_port_map.py`,
and the run_gate.sh census all moved with it.  **THE WORKGROUP SIZE was reconciled at the cause:** the CUDA block is
`kResidentPlanMax == 128`; the shader originally declared 128 and the gate FAILED it
(`LocalSize 128 1 1 does not match the host's kLocalSize=256`) - the harness sizes every kernel to 256 - so the
shader now declares 256 with shared arrays sized 256 and every `tid`-guarded loop inert past `n_entries`.

**WHAT IS VERIFIED.**  (1) **THE RUN.**  On `coder-iq1_m`, with
`STRATA_VK_ARENA_MIB=28560 STRATA_VK_DESKTOP_RESERVE_MIB=256` and `--expert-cache 12288 --mmap-experts`, the window
prints `100% VRAM resident: zero-doorbell graph`, `captured the 1-token window`, the graph LAUNCHES and the engine
decodes **token id 20** (`output  : 20`, greedy) in 143.0 ms / 6.99 tok/s, exit 0 (`/tmp/tok_run7.log`, post-fix);
the SAME id 20 in a SECOND run (`/tmp/tok_run6.log`) and in a third (`/tmp/tok_run4.log`, pre-fix).  Decoded with the
PACK'S OWN tokenizer, id 20 is `5`.  **BOUNDS, same breath:** real weights but a 2-token prompt; PLE file-backed;
`--spec 4`; `pcie_frac 0.05` (at which partial residency still sends ~95% of misses to the CPU path this port
forbids, so all-resident is the only CPU-free route); 12288/12288 resident; `--mmap-experts`; arena 28,560 MiB
against a 28,593 ceiling (the reserve FLOOR - `STRATA_VK_DESKTOP_RESERVE_MIB=256` is load-bearing; without it the
port correctly REFUSES the arena, `/tmp/tok_run3.log`).  **Per-kernel numerics are the GATE's job, not this run's: a
token id is not a correct token - this is evidence the CHAIN EXECUTES.**  (2) **THE CASE**
`case_resident_plan_entry` (14 verdicts, green on the Arc, llvmpipe and Ryzen iGPU): the plan vs an INDEPENDENT host
transcription of the CUDA group-by, **bitwise on the pointer VALUES**; rivals that MOVE (changed `slot_off`,
changed `cache_base`, permuted ids, a duplicated expert); a VACUITY arm (a non-resident expert writes NOTHING) plus
the anti-vacuity arm that forbids a do-nothing implementation; the skip word; a CAPTURE arm (records, does not run;
replay == direct BITWISE); and a `ptr_to_off` arm proving the pointers reach the launcher as the right byte
offset+window.  **The first failing arm was the ninth WRONG ORACLE** (the transcription summed a group's count for
every entry instead of `s_cnt = is_first ? count : 0`); the kernel was right.  Four registered injections, ALL BITE
(`...-drop-slot-offset` worst 3.28e+04, `...-all-first` 1.23e+14, `...-entry-order`, `...-base-off-by-8` worst
exactly 8).  (3) **GATE** (vega, `/tmp/gate_b15.log`, background): **intel_icd 883 passed / 0 failed / 0 skipped**
(was 869; +14), lvp `865/0/6`, radeon `871/3/2` (the three are the documented intermittent family - `bf16_gemv`,
`fused_gr_read entry`, `bf16_gemv_fp32_mmvf_cols` - NONE is `resident_plan`; `run_gate.sh` exits 1 because of
radeon, the Arc read being the port's green).  The census prints `OK resident_plan LocalSize 256 1 1`.  **MAP: `168
= 97 kernel + 0 shader + 47 host + 0 todo + 24 refused`** (146 shaders built) - ONE row moved
(`resident_plan: refused -> kernel`); refusal count **25 -> 24**.  `check_port_map.py` passes; `make_port_map.py`
regenerates byte-identically.  **ENGINE BAR: 0 undefined - 0 BY CONSTRUCTION**, not a porting gain.  (4) The
corrected `resident_plan` refusal text is now UNREACHABLE (the symbol is DEFINED); the engine prints NO
`resident_plan` line (grepped 0 hits in the runs).

**WHAT IS NOT.**  **A CORRECT TOKEN.**  The run proves the chain executes, not that it computes the right thing:
per-kernel numerics are the gate's, prompt conditioning is untested, and the single greedy token from a 2-token
prompt is not a correctness signal.  The `verify.cpp:943` device-plan arm (`STRATA_VERIFY_DEVICE_PLAN`) is carried
by the same definition but NOT executed.  The descriptor-pool growth (`pools 2..61`, the 2048-slot config) stays
OPEN as recorded; `z820b` untouched (no XTX/K620 number).

# Status — what is done, what is verified, what is not

## THE TYPE FIX IS SHIPPED AND THE ALL-RESIDENT FIT CLOSES; the window reaches its ALL-RESIDENT ARM and stops at `resident_plan` (2026-10-06, `vega`)

**WHAT IS DONE.**  (1) `cudaHostAlloc` no longer hands out DEVICE_LOCAL (BAR VRAM) memory: a new `Ctx::alloc_host`
(`vulkan/src/device/vk_compute.{hpp,cpp}`) selects a HOST_VISIBLE | HOST_COHERENT type in a NON-device-local heap
(system RAM), falls back to the old type only where no such type exists, and charges it to the HOST account; the
choice stays observable via `STRATA_VK_MEM_TRACE=1`, which now also prints each allocation's size and heap.
(2) `STRATA_VK_ARENA_MIB` (+ a fractional `STRATA_VK_ARENA_GIB`) is added (`vulkan/src/compat/cuda_runtime.cpp`)
because the all-resident fit (28,379.07 MiB, then 28,530.24 MiB) cannot be expressed in whole GiB between the 27 GiB
step and the 28,593 MiB reserve-bounded ceiling.  (3) The stale refusal text/comment for `resident_plan` is
corrected - `all_resident_` IS reached with `--expert-cache 12288`.

**WHAT IS VERIFIED.**  Gate (vega, `/tmp/gate_b13.log`, this commit): **intel_icd == 869 passed, 0 failed, 0 skipped**
(the port's green), lvp `853/0/5`, radeon `857/3/2`; `run_gate.sh` exits 1 BECAUSE of the radeon arm (documented -
the Arc read is the port's green).  The three radeon failures are `budget: independent requery` `0/1`,
`fused_gdn_ab entry` `95/96`, `bf16_gemv_fp32_mmvf_cols entry` `2492/2496` - the documented platform intermittent
family, and the set MOVES run to run.  `check_port_map.py` passes; `make_port_map.py` regenerates byte-identically.
Map unchanged: `168 = 96 kernel + 0 shader + 47 host + 0 todo + 25 refused`.  Engine bar 0, by construction.  THE
PROBE, both ends: the independent oracle `probe_mem` reads `h2d_from_alloc_host() 160.6 ms -> 1.67 GB/s` against the
SAME run's `h2d_from_alloc() 4833.0 ms -> 0.06 GB/s` and `h2d_from_sysram 1.71 GB/s`; the engine's own probe reads
`1.9 GB/s -> pcie_frac 0.05` (was `0.1 GB/s -> 0.00`).  Device memory freed: **~602 MiB (0.588 GiB)** of the
device-local heap, the host tier's summed footprint.

**DELIVERABLE D - THE radeon `1367/2880` INSTANCE IS THE INTERMITTENT, SETTLED BY NON-REPRODUCTION.**  The previous
batch's `fused_gr_read_multi entry: a recorded block REPLAYS bitwise equal to direct execution` `1367/2880` did NOT
reproduce: on THIS gate run the case read `0/2880` (`PASS`) on radeon, and radeon failed three DIFFERENT cases
instead.  A deterministic replay defect would fail the SAME case at the SAME count; the failing SET moving between
runs is the platform-level intermittent's signature (it already moves between `bf16_gemv_fp32_mmvf_cols`,
`bf16_gemv_fp32_mmvf_multi`, `cvec_apply`, `fused_gr_read`, `budget: independent requery`).  So the ~half-the-words
magnitude is the same intermittent at a larger count, NOT a new replay defect, and the case needs no arm of its own
beyond this.  **DESCRIPTOR POOLS - NOW A NUMBER, AND IT DEPENDS ON THE ARM:** the ALL-RESIDENT window creates exactly
ONE extra pool (`descriptor pool 2 created`); the earlier `pools 2..93` were the `--expert-cache 2048` (non-
all-resident) config, whose doorbell/fetch machinery the all-resident arm BYPASSES (`zero-doorbell graph`).  The
`2..93` count for THAT config stays OPEN as recorded.

**WHAT IS NOT.**  A TOKEN - **one unported symbol short.**  The all-resident arm now runs: `12288 of 12288 slots`,
`token graph hit path: 12288 resident experts`, `prefill ... 947.5 / 1064.2 ms ... experts streamed 0`,
`100% VRAM resident: zero-doorbell graph`, then `resident_plan` REFUSES.  The host-side blocker (a 23.42 GiB anon
expert arena + the 27.7 GiB VRAM arena = two OOM kills, `anon-rss 41.86 GB` under `memguard 40G` and `40.5 GB`
global) is solved with the engine's own `--mmap-experts` (anon 18.2 GiB -> 0.6 GiB, file-backed).  The type fix did
NOT widen the arena (the arena is allocated before the host tier) - the ARENA SIZE did, and 28,560 MiB of the
28,593 MiB ceiling fits.  At `pcie_frac 0.05` a partial-residency run still sends ~95% of missed experts to the CPU
path the port forbids, so all-resident remains the only CPU-free route.  The radeon `1367/2880` instance did NOT
reproduce (see DELIVERABLE D) and is settled as the intermittent; the `2..93` pool count for the 2048-slot config
stays OPEN.  `z820b` untouched.

# Status — what is done, what is verified, what is not

## THE BINDING CONDITION IS THE PINNED HOST TIER, NOT THE SHARE OR A KERNEL — the PCIe probe is a BAR read, measured (2026-10-06, `vega`)

**WHAT IS DONE.**  The stop at LAUNCH is now NAMED and its two conditions settled.  The PLAN branch IS taken
(`STRATA_POOL_TRACE=1`: `pool trace: layer 0 publish 0 / fetch 0`), so a missed expert stays `kind = -1` on this
port.  The two conditions for `kind = 1` were tested apart: `--pcie-frac 1.0` (`pcie_num = 256`, `m = nmiss`) still
gives `publish 0 / fetch 0`, and `pcie_layer()` itself requires the same registration as `pinned()`, so REGISTRATION
alone binds both.  The PCIe probe's method was established from the engine's code and its number MEASURED two ways.

**WHAT IS VERIFIED.**  `STRATA_VK_MEM_TRACE=1` (new instrument, `vk_compute.cpp`) prints, on the real device through
the port: `cudaHostAlloc type 3` = heap 0 `DEVICE_LOCAL|HOST_VISIBLE` (BAR VRAM), `arena/vram type 0`,
`staging type 2`.  The independent oracle `ports/vulkan/tools/probe_mem.cpp` (built against the port's device layer,
Arc Pro B70, 256 MiB): CPU read of the mapped `cudaMallocHost` block `4280.7 ms -> 0.06 GB/s`; the probe's exact op
`4537.9 ms -> 0.06 GB/s`; the same copy from system RAM `137.9 ms -> 1.95 GB/s`.  So the engine's `0.1 GB/s` times a
BAR read of VRAM, not a link - **the probe is a wrong decision input; the default 0.55 is right.**  Map unchanged:
`168 = 96 kernel + 0 shader + 47 host + 0 todo + 25 refused`, 145 shaders; `check_port_map.py` passes and
`make_port_map.py` is byte-identical.  Engine bar 0, by construction.  GATE (vega, `/tmp/gate_b12.log`): **intel
`869/0/0`** (green), lvp `853/0/5`, radeon `857/3/2` - the documented intermittent family
(`bf16_gemv_fp32_mmvf_cols` 2491/2496, `bf16_gemv_fp32_mmvf_multi` 622/624) plus ONE NEW instance,
`fused_gr_read_multi entry: a recorded block REPLAYS bitwise equal to direct execution` **1367/2880** (Ryzen
iGPU; ~half the words, not the 1-6-word documented shape; intel read 0 failed in the same run) - recorded with
its count, not chased.

**WHAT IS NOT.**  A TOKEN.  **The staging path (`kind = 1`) is UNREACHABLE BY CONSTRUCTION** while the engine's
expert arena is host memory outside the single arena `VkBuffer`: `fetch_blobs` and `native_expert_grouped` rebase
every source by `ptr - kArenaBase` and window into that buffer, and `VK_EXT_external_memory_host` (present on this
driver) would yield a second buffer the rebase mis-binds.  Reaching it needs a region-ID extension of the pointer
scheme, not a one-line registration.  Even a correct probe does not reach a token: the engine's 0.55 leaves 45% CPU,
and all-resident is ~2 GiB short.  The probe-type fix (hand out the non-device-local host type in `cudaHostAlloc`)
is recorded for its own batch, not shipped.  The intermittent `prefill: routed id out of range` and the descriptor
pools (2..93) are RECORDED, not chased.  `z820b` untouched.

# Status — what is done, what is verified, what is not

## THE RECORDING COMPLETES AND THE GRAPH CAPTURES; the stop moved to the engine's host expert verb at LAUNCH (2026-10-05, `vega`)

**WHAT IS DONE.**  The P6 verify window's RECORDING now runs to completion and the token graph is captured -
`strata verify: captured the 1-token window (upload no error, sync no error)` (`/tmp/run_real_pool9.log`).  Three
symbols were closed to get here: `fetch_blobs`/`rebase_ptrs` (`verify.cpp:1053/:1054`, the PCIe staging - ON PATH
under the default `--pcie-mode auto` -> `pcie_mode == 2`; carried as DEVICE-side shaders, `fetch_blobs.spv` over
the pointer table with the 4 GiB window, `rebase_ptrs.spv` the lo/hi rewrite, so the device count is read at
submit and the `pcie_frac 0.00` case is a true no-op), and `copy_from_mapped` (`:678`, the PLE snapshot) whose
source is an ARENA (device) buffer and whose wrapper demanded a MAPPED one.  A fourth fix was measured:
`sample_tokens` (the window's own sampling, `verify.cpp:1170`) takes a MAPPED `out` (`m_out_`) - the engine reads
the token id back on the host - and the wrapper now accepts either kind.

**WHAT IS VERIFIED.**  Gate (vega): intel_icd `?/?/?`, lvp `?/?/?`, radeon `?/?/?` (see the gate log at the
commit).  `case_blob_stage_entry` proves `fetch_blobs`/`rebase_ptrs` against an independent host transcription of
the CUDA loops, the device count against the capacity (sentinel slots), rivals that MOVE, a VACUITY arm for each
(`*n == 0` writes nothing, and the same call with `*n == NB` is required to have written everything), and the 4
GiB WINDOW SPLIT driven directly at a 4 KiB window.  `case_copy_from_mapped_entry` gains an ARENA-source arm
(the `:678` shape) with a rival that re-publishes the source; `case_sample_tokens_entry` gains a MAPPED-`out` arm
(`verify.cpp:1170`).  Five new registered injections.

**WHAT IS NOT.**  A TOKEN.  The refusal at LAUNCH is the engine's own CPU expert-activation
(`strata::kernels::cpu::native_quant_act`, `expert_source.cpp:2147`) - reached only when the per-expert `kind[i]`
array has a `-1` (an expert the CPU computes), which the PLAN branch never produces.  The port has no
CPU-hybrid path and refuses it, correctly.  Two honest routes are named in `NEXT.md`: all-resident
(`--expert-cache 12288` is 0.62 GiB short of the 26/27 GiB arena) or the plan branch.  The window's descriptor
pools grow to `pool 90` (RECORDED, not chased).

# Status — what is done, what is verified, what is not

## THE P6 HANDSHAKE SEAM IS CARRIED HOST-SIDE AND GATE-PROVEN; the window's RECORDING passes `wait_flag_ge` and now stops at `fetch_blobs` (2026-10-05, `vega`)

**WHAT IS DONE.**  `wait_flag_ge` (`verify_kernels.cu:496`, a 1-thread SPIN on host-mapped memory) is carried
HOST-SIDE, not translated: `Ctx::capture_boundary` CUTS the captured window into SEGMENTS at each wait, and the
shim submits each next segment only once the mapped handshake word has been raised - polled on the HOST THREAD
between split submissions, driven by the engine's own `cudaStreamQuery`/`cudaStreamSynchronize`.  No kernel waits;
a boundary that is never satisfied is a LOUD REFUSAL, never a hang.  Two reachable symbols were ported:
`copy_rows_from_mapped` (`verify.cpp:1071`, shader `copy_rows_from_mapped.spv`) and `copy_indexed` (`:1311`,
shader `copy_indexed.spv`).

**WHAT IS VERIFIED.**  Gate (vega): `intel_icd == 853 passed, 0 failed, 0 skipped` (was 842); lvp `837/0/5`;
radeon `842/2/2` (both failures the documented intermittent `bf16_gemv_fp32_mmvf` family).  The new case
`case_verify_seam_entry` proves, on ALL THREE ICDs: `copy_indexed` == the engine rule (device index selects the
row; a rival index MOVES; a negative index writes nothing); `copy_rows_from_mapped` == the engine rule (hit rows
-> 0, the rest from mapped memory; the hit rows MOVE; count 0 copies every row); and THE SEAM - after a launch the
ops BEFORE a boundary have run and the ops AFTER it have NOT (a sentinel read, so a vacuous arm fails), raising the
flag + synchronizing runs the rest, and a REPLAY cuts and advances again.  Three registered injections, all
BITING (results in the commit).

**WHAT IS NOT.**  A TOKEN.  The run's RECORDING now passes the three waits and `copy_rows_from_mapped` but stops at
`fetch_blobs` (`verify.cpp:1053`) - **ON PATH** under the default `--pcie-mode auto` -> `set_pcie_mode(2)`
(`generate.cpp:5168/:7848`), which the previous batch mis-classified off-path from `pcie_frac`.  `fetch_blobs`/
`rebase_ptrs` gather from DEVICE-HELD POINTERS a shader cannot dereference; with `pcie_frac 0.00` the CUDA pair is
an empty no-op, and carrying that empty case faithfully is the next increment.  On a `--pcie-mode dma` probe the
stop moves to `copy_from_mapped` (`verify.cpp:678`, the PLE-history snapshot) whose source is not a live mapped
region - undiagnosed.  **The engine capture is a RECORD here, so the window's GPU body has still not been
launched; no token.**

# Status — what is done, what is verified, what is not

## THE P6 VERIFY WINDOW RUNS ELEVEN SYMBOLS DEEP; `wait_flag_ge` IS THE NEXT STOP AND IT IS A HOLE (2026-10-05, `vega`)

**WHAT IS DONE.**  Nine symbols of the window's body were ported in `verify.cpp` call order, as DECODE-equivalent
arithmetic (per-token loops over already-gated decode kernels for the `_multi` variants):
`broadcast_streams`/`add_streams_broadcast` (ONE shader, `mode` selects the rule), `gdn_conv_l2_multi` (a per-token
loop over the gated `fused_gdn_conv_l2` over a WORKING COPY of the history), `gdn_ab_multi` (per-token
`gdn_ab`), `gdn_step_norm_multi` (a single shader; BOTH halves - the commit's own write and the verify window's
`n_keep`/`t_out_begin`/state-untouched form), `native_router_top10_multi` and `native_moe_combine_multi` (per-token
loops), `shared_expert_multi` (per-token projections + ONE batched scalar-gate + ONE batched row-scale).

**WHAT IS VERIFIED.**  Gate: `intel_icd == 842 passed, 0 failed, 0 skipped`; lvp `826/0/5`; radeon's 4 failures are
the documented intermittent set.  The case `case_verify_window` proves each symbol against the engine's own rule
with an independent double oracle, the SINGLE-TOKEN kernel called directly (bitwise) where the arithmetic is the
same, a capture arm (record, replay, require bitwise equality) for every wrapper that records, rivals that MOVE, and
a vacuity arm for each multi ("the single-token arm wrote every row").  Seven registered injections, all FALSIFYING
(`/tmp/inject_final.log`).

**WHAT IS NOT.**  The run stops at `wait_flag_ge` (verify.cpp:1042, layer 0's post) - a 1-thread SPIN on host-mapped
memory, which this port's NO-KERNEL-SPINS rule forbids; the fix is a host-side poll between SPLIT SUBMISSIONS (the
handshake seam).  `native_moe_combine_multi` (1083), the second `fused_gr_read_multi` (1142), `gdn_conv_commit`
(1293) and `copy_indexed` (1311) are gate-proven but have NOT yet executed in a real run (they sit after the stop).
No token yet: **the GPU has not produced a model token through the verify window.**

# Status — what is done, what is verified, what is not

## THE FUSED HYPER-CONNECTION READ IS WIRED; the P6 verify window opens and stops at `broadcast_streams` (2026-10-05, `vega`)

**THE `verify.cpp:336` CAPABILITY REFUSAL IS CLEARED AND THE WINDOW RUNS ITS BODY.**  On `coder-iq1_m` (no
`--no-pool`, `--expert-profile /tmp/expert-profile-coder-built.bin`, `--expert-cache 2048`) the run reaches
`strata verify: window up to 6 tokens, 74.0 MiB of device buffers` — `Verifier::init` SUCCEEDED past
`fused_gr_supported` — and stops at the NEXT named refusal in the window's body (`/tmp/run_real_pool4.log`, `RC=2`,
two runs identical):

```
strata::kernels::broadcast_streams: NOT PORTED and NOT REACHED by the shipped configuration - REFUSING.
  The only configuration that reaches it: the P6 verifier (verify.cpp); Verifier::init refuses (layer.cpp:476-491)
```

`broadcast_streams` is the window's embedding broadcast (`verify.cpp:592/606`; CUDA `broadcast_streams_kernel`,
`verify_kernels.cu:256`) — a flat `R[(t*hc+c)*n_embd+d] = x[t*n_embd+d]`.  **Its "NOT REACHED" text is now STALE,
and that is the next increment.**  **NO TOKEN.**

**PORTED (NEW TU `vulkan/src/kernels/fused_gr_vk.cpp` + 4 shaders):** `fused_gr_read` / `fused_gr_read_multi`, the
verify window's per-layer GR read (`verify.cpp:693/:1142`) and the decode path's (`layer.cpp:1253/1276`).  The fused
read = the previous half's write FOLDED IN + **FP32 ACTIVATIONS** (`fused_gr.hpp:15`) — `gr.cu`'s
`<true>`/`<float>` branch, NOT the ported unfused `gr_read`'s `<uint16_t>` one, so the two are **NOT bitwise-equal
and the case MEASURES them apart (rel-L1 2.52e-3)**.  Four dispatches per token over the port's ONE barrier tree and
ONE bf16 widen: `fused_gr_rs` (fold + per-stream RMS + `rs`), `fused_gr_down` (`lo`), `fused_gr_mix` (gate+mean
FUSED), `fused_gr_inject`.  The activation is NOT materialised (the CUDA recomputes it; so does this port).  **One
defect the gate found and fixed at the cause:** with `apply == false` `R_out` is untouched, so the stages must read
`a.R` — bound explicitly now, not assumed (`R_out == R` holds in the engine but the case passes distinct buffers).

**`fused_gr_supported` NOW ANSWERS THE ENGINE'S OWN PREDICATE** (`2560,4,320`, `fused_gr.cu:1164-1166`) — TRUE
because the kernels exist and are gate-proven.  It has been wrong in BOTH directions now: the rule while the kernel
was missing selected an unwired branch; FALSE (last batch) refused a native pack's only decode path at `:336`.
**Paths that select it:** DECODE (`layer.cpp:1188`, `:1328` stops calling `gr_write` for interior layers) and the
VERIFY WINDOW.  `gr_write` stays for the last layer / head.  **The DECODE selection is gate-proved, not run-proved**
(a native pack breaks the token loop, `generate.cpp:7578-7579`).

**PROVEN.**  `case_fused_gr_read_entry` (10 verdicts) + a reworked `case_fused_gr_supported_entry`: wrapper == ported
shader path **BITWISE** 13128/13128; wrapper vs the engine's own rule (double, FP32 act) 13124/13124 worst 0.000889;
multi == 3 x single **BITWISE** 39384/39384; the final mixer / `apply == false` / rivals / **3 CAPTURE arms**
(records-not-runs, replay == direct bitwise, a replay re-reads moved `R`) / a child-process refusal.  Rivals MOVE
(whole-stack RMS 8.36e-1, SUM 3.00e+0, fold dropped 3.06e-1, BF16 act 2.52e-3).  **SIX injections, ALL BITE**
(`fused-gr-supported-false`, `-rs-drop-fold`, `-mix-sum-not-mean`, `-down-swap-halves`, `-inject-drop-rs`,
`-multi-token0-args`).  **The multi arm was VACUOUS on first writing** — an injection that read token 0's args for
every token did NOT falsify it, because unwritten slots kept the singles' values; every output the multi must write
is now sentinelled.  `fused-gr-supported-true` is RETIRED (the rule is the truth now).

**MAP:** `168 = 83 kernel + 0 shader + 46 host + 0 todo + 39 refused` (was `81/0/46/0/41`; the two fused-read rows
moved `refused -> kernel`); `check_port_map.py` passes, `make_port_map.py` regenerates byte-identically.  **ENGINE
BAR: 0 undefined — 0 BY CONSTRUCTION**, not a porting gain.  **GATE (vega): Arc `intel_icd` 816/0/0** (was 806),
llvmpipe 804/0/3; the radeon arm carried the documented platform-level non-determinism (`budget: independent
requery`, `cvec_apply entry point` 6047/6144 — RECORDED, new instance, not chased, and not this batch's case:
Arc read 0 failed in the same run).  **A LATER RUN put the SAME class on THIS batch's case**: radeon
`fused_gr_read entry (apply=true): engine wrapper == the ported shader path, BITWISE` **13127/13128 (ONE word)** —
recorded with kernel name and element count, Arc 0 failed in that run too, llvmpipe green; NOT chased.
`z820b` untouched.

## THE PREFILL PATH RUNS TO COMPLETION on the REAL pack; the stop is the verifier's residency table (2026-10-05, `vega`)

**8 -> 5 prefill refusals, and the prompt path now executes every stage.**  `coder-iq1_m` with
`--spec 4 --prefill 1 --tokens "1,2" --max-new 1 --max-context 8` prints
`prefill 1 tokens in 1 chunks, 29680.6 ms (0.0 tok/s); experts streamed 480 ...; PLE 5.5 ms` and only THEN stops at
`generate.cpp:7818` — `--spec needs the device residency table (--expert-profile, --expert-cache and the token graph)`
(`/tmp/run_pf9.log`, `RC=2`).  **NO TOKEN**, and the bounds are unchanged (real weights, 2-token prompt, PLE
file-backed, `--spec 4`, `pcie_frac 0.00`; per-kernel numerics are the GATE's job, not this run's).

**PORTED (DECODE-equivalent per-token shaping):** `prefill::kv_append` (loop over the gated `kv_f16_append`/
`kv_q8_append`), `native_qsa_indexer_append` + `_batch` (NEW shader `pf_indexer_native.comp`, transcribed from
`native_qsa_indexer.cu:46-106`; the batch loops the single append), `qsa_decode_attn_batch` (per-query loop over the
gated decode attention), `qsa_prompt_attn_batch` -> FALSE (a capability; the engine falls back to
`qsa_decode_attn_batch`).  `native_qsa_indexer_enabled()` is now a REAL flag.  **REFUSED honestly** (not on this path):
`kv_append_q4`, `kv_stage_from_host`, `native_ple_postops_batch`, `blob_dequant_f16`, `round_f16`.

**PROVEN, not assumed.**  `case_prefill_prompt_path` gates the six shaders the last batch left gate-compiled-only
(`pf_swiglu16` both modes, `pf_moe_combine`, `pf_gu_interleave_f16`, `pf_split_q`, `pf_gate_attn`, `pf_copy_u32`) plus
`kv_f16_append` (through `prefill::kv_append`) and `pf_indexer_native` (through `native_qsa_indexer_append`) — nine new
arms, each against the engine's own rule (double where that is the only reference).  Seven registered injections, ALL
FALSIFY.  **The first failing case was the ORACLE, not the kernel** (a relative-to-result tolerance on `moe_combine`
blown up by cancellation) — the port's fifth wrong oracle.

**THE VERIFIER PRECONDITION, from the code:** the message is `thits.d_res == nullptr` (`generate.cpp:7817`); `d_res` is
built only with `hit_fn && !profile.empty() && !no_pool` (`:4267`) and a capturable token graph (`:4268`).
`--expert-profile P` reads a `profile.bin` from `tools/make_profile.py` (a real producer needing a routing trace);
`--expert-cache` must be non-empty; and the token graph is captured only when `!native_pack && !multi_gpu` (`:4334`),
so a native pack has none by design.  Bigger than one batch.

**MAP:** `168 = 81 kernel + 0 shader + 45 host + 0 todo + 42 refused` (two decode-path rows moved `refused -> kernel`);
`check_port_map.py` passes, `make_port_map.py` regenerates byte-identically.  **ENGINE BAR: 0 undefined — 0 BY
CONSTRUCTION.**  GATE (vega): Arc `intel_icd` **802/0/0 (exit 0)**.  `z820b` untouched.

## THE PREFILL PATH LANDS 23 OF ITS 40 ENTRY POINTS; the real pack RUNS the prompt path and stops at `gdn_recurrence` (2026-10-05, `vega`)

**THE STOPPING POINT MOVED OFF `Gemm::init_external`.**  New TU `vulkan/src/kernels/prefill_vk.cpp` + 8 new
shaders port **23 of the 40** prefill entry points: the whole `Gemm` class (`init_external`/`init`/`rebind`/`f16`/
`bf16`/`native`), the 8-op hyper-connection family, the GDN gates/conv/L2, the thin elementwise/copy set
(`to_f16`/`to_bf16`/`copy_f32_wide`/`copy_i32`/`gather_rows16`/`rms_rows`/`route`), and `qsa_block_scores_tc`
(answered FALSE - a capability predicate, so the engine uses the ported `qsa_block_scores`).  **17 remain** (the
ordered list with measured sizes is in `NEXT.md`).

**THE RUN.**  `--prefill 1 --tokens "1,2"` on `coder-iq1_m` runs the prompt path through: GEMM init -> the chunk
embedding -> `gr_broadcast` -> the PLE block -> `gr_norm_rs` -> 3 x `Gemm::bf16` -> `gr_silu` -> `gr_mix_r` ->
2 x `Gemm::native` -> `gdn_gates` -> `gdn_conv`+L2, then exits 2 at
`strata::prefill::gdn_recurrence: NOT PORTED on the Vulkan backend - REFUSING.`  (`/tmp/run_pf3.log`.)  A ONE-token
prompt never enters the prefill block (`n_prompt > 1` guards it), so `--tokens 1` still gives `decode 0 tokens` -
measured (`/tmp/run_pf.log`).

**A DEFECT THE RUN FOUND.**  `iq_embed_rows` refused the token-embedding table because a native pack holds it in
MAPPED HOST memory and the prefill's batched gather is the first caller to hand it a mapped pointer.  Fixed at the
cause: the table now binds through `mapped_resolve` when `arena_resolve` fails (`vulkan/src/kernels/iq_vk.cpp`).

**NO TOKEN.**  The verifier's two preconditions (`--expert-profile`, `native_qsa_indexer_enabled()`) are still
BEHIND the whole prefill, so they were not reached and were not touched.  ENGINE BAR: the program LINKS, 0
undefined - **0 BY CONSTRUCTION** (the refusals define the unported symbols), NOT a porting gain.  MAP:
**`168 = 79 kernel + 0 shader + 45 host + 0 todo + 44 refused`** - UNCHANGED and correctly so: `PORT-MAP.tsv` is
the DECODE path's set, and the prefill's 40 symbols are not in it.  **`refused` is NOT a capability**; the prefill
refusal COUNT moved 40 -> 17, which the map does not carry.  Gate (vega): **Arc `intel_icd` 793/0/0 (exit 0)**,
llvmpipe 781/0/3, Ryzen iGPU 783/1/2 (the one radeon failure is the DOCUMENTED intermittent
`bf16_gemv_fp32_mmvf_cols entry`).  Five new injections registered; three RUN and all FALSIFY.

## THE NATIVE-DENSE K-QUANTS ARE PORTED (one generic shader), the LAUNCHER'S `view()` DEFECT IS FOUND AND FIXED, and the real pack stops at the PROMPT PREFILL (2026-10-05, `vega`)

**No token from the real pack, and the stopping point is now a NAMED, MEASURED subsystem, not a flag.**  The
run loads every dense projection and comes up (`1406 MiB ... 302 canonical tensors skipped: served natively`,
`300 native projection matrices, 2018.88 MiB of weights`, `session is up`), then prints `output :` EMPTY with
`decode 0 tokens in 0.0 ms`, exit 0.

**WHY EMPTY, ESTABLISHED FROM THE CODE.**  A native pack's decode loop BREAKS on its first iteration
(`generate.cpp:7578-7579`: `if (native_pack) { spec_pos = pos; break; }`) — its only decode path is the verify
window — and the window's guard is `spec_pos > 0` (`:7806`), while `spec_pos = pos_start` is 0 (`:7472`) unless
the PREFILL block set it (`:7564`).  A 1-token prompt skips that block, so nothing runs.  Any prompt with >1
token needs `--prefill CHUNK` (`:2165`) — and the prompt path is NOT PORTED: `--tokens "1,2" --prefill 1` exits
2 at `strata::prefill::Gemm::init_external: NOT PORTED on the Vulkan backend - REFUSING.`  **PREFILL IS THE
NEXT INCREMENT: 40 unported GPU entry points (~150 KB of CUDA: kernels.cu 53 KB, gemm.cu 25 KB, moe_fused.cu
23 KB, moe_fused_iq.cu 35 KB, moe_mmq.cu 12 KB).**  Behind it stand `Verifier::init`'s two preconditions: the
`--expert-profile`/`--expert-cache` residency table (`verify.cpp:324`, built only when a profile is supplied,
`generate.cpp:4267`) and `layer_verify_compatible()` — FALSE here because `native_qsa_indexer_enabled()` is
FALSE (the unported `native_qsa_indexer_append`).

**DELIVERABLE A — ONE generic K-quant MMVQ.**  `shaders/native_k_mmvq.comp` covers ggml 12 Q4_K / 13 Q5_K /
14 Q6_K (210 of the pack's 300 eligible dense tensors) behind a `ty` push constant; the three dots are ONE
definition in `shaders/common/k_dots.glsl`, which the port's EXISTING per-type `native_q5_k_f32.comp` now also
includes (its Q5_K dot was lifted out verbatim).  Q4_K/Q6_K are transcribed from the engine's own
`q4_q8_dot`/`q6_q8_dot` (`native_mmvq.cu:544`/`:638`).  `native_mmvq` gained case 12/13/14; `native_mmvq_supported`
answers TRUE for them.  **Proven** (`case_native_k_mmvq`): generic == engine wrapper BITWISE per type; generic
(ty=13) == `native_q5_k_f32.spv` BITWISE; generic vs the engine's dot (double) terms-bounded.  A first version
failed the per-type arm last-bit because a runtime parts/stride defeated the compiler's folding — each type now
has its own uniform arm.  **No K-quant dot needed inventing.**

**DELIVERABLE C — the flagged gaps, and the DEFECT they found.**
* `q8_0_mmvq` is gated (`case_q8_0_mmvq`, bitwise vs the wrapper + the engine's rule).
* The 4 GiB window boundary is exercised: `ptr_to_off.spv` for 0/wb-1/wb/wb+5/2wb+7, AND the launcher reads the
  SAME expert through window 0 (offset 0) and window 1 (offset 4 313 715 712) with BITWISE-equal output.
* The launcher's wrapper case is registered with SIX FALSIFYING injections, including two for the launcher.
* Independent arms for the generic shaders at the formats the real pack uses (`native_gu_any` 18/21/23,
  `native_down_any` 42).

**THE DEFECT: `view(b, off)` SET the offset instead of ADDING the base's.**  So `native_expert_grouped`'s four
`view(b_scr, k*fa)` scratch regions bound at the ARENA BASE: the gate/up/h/hq staging landed on the first
expert's blob, the raw-pointer q8_1 quantiser wrote to the caller's scratch, and the down stage read a region
that was neither.  **Measured:** zeroing an expert's ENTIRE gate/up half left `out` bitwise UNCHANGED (0/2560).
That is a silently wrong expert for every native expert.  Fixed to `v.offset = b.offset + off` (both copies);
new arms D (gate/up-only perturbation moves the output, 2560/2560) and E (zeroing it gives exactly 0) PIN it,
and `view-absolute-offset` restores the defect to FAIL arm D.

**RESULTS (vega).**  Gate: **Arc intel_icd 781/0/0 (exit 0), llvmpipe 769/0/3, Ryzen iGPU 768/4/2.**  The lvp
arm had ABORTED (a bare line, no counts) because the launcher case first asked for a 5 GiB arena and llvmpipe's
`vkCreateBuffer` returned `VK_ERROR_OUT_OF_DEVICE_MEMORY`, which the device layer exits on; the arena is now the
minimum that crosses one window (kWinBytes + 24 MiB, < 4 GiB) and lvp reports 769/0/3.  The 4 radeon failures
are ALL on the documented intermittent list; `native_quantize_q8_1 entry` (1151/1152) appeared once, touches
no shader this batch changed, passed on Arc and llvmpipe in the same run and did not recur — recorded as a new
instance of the platform non-determinism, not chased.  Map: `168 = 79 kernel + 0 shader + 45 host + 0 todo +
44 refused`; `check_port_map.py` passes, `make_port_map.py` regenerates byte-identically.  **ENGINE BAR: the
program LINKS, 0 undefined — 0 BY CONSTRUCTION (the refusals define the unported symbols), NOT a porting
gain.**  **`z820b` untouched.**

## THE REAL PACK, AND THE NATIVE-EXPERT GATE — the IQ-pack load defect is FIXED at the cause, the port owns the native expert GEOMETRY, and the stopping point is a NAMED missing kernel (2026-10-05, `vega`)

**No token from the real pack, and it is not a token's worth of guessing: the engine's own load path refuses
`coder-iq1_m` on LAYER 0 because the port has no GROUPED native-expert kernel.**  Both runs (identical, `RC=1`)
print exactly one line before anything is allocated:

```
strata generate: layer 0's experts are IQ3_XXS/IQ4_NL (ggml types 18/20), which this engine has no GPU kernels for
```

**THE DEFECT THAT STOOD IN FRONT OF IT, fixed at the cause.**  `option(STRATA_NATIVE_EXPERTS ...)` was declared at
`CMakeLists.txt:977`, AFTER the Vulkan block's `return()` - the same ordering trap `STRATA_VERSION` documents.  A
Vulkan configure never defined the macro, so `expert_layout.cpp` took its no-native branch and refused every IQ
pack ("this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS").  The option is
now declared above the backend blocks and `strata_vulkan_kernels_cpu` carries the define; the ggml-cpu half stays
excluded, and the port supplies the geometry itself (below).

**THE PORT NOW OWNS THE ENGINE'S OWN EXPERT ADDRESSING.**  `vulkan/src/kernels/native_expert_vk.cpp` (new) gives
`native_fmt` as REAL geometry - `gu_row`/`d_row` from the port's one `iq_row_bytes` table, `up_off = gu_row*n_ff`,
`down_off = 2*up_off`, `bytes = down_off + d_row*n_embd`, `vec_dot_type` per ggml-cpu's traits table - so the
engine's OWN load check (`expert_layout.cpp:285`, `f.bytes == the pack's blob column`) passes for ALL 48 layers.
`iq_row_bytes` gained Q8_K (15, 292 B / 256).  `native_expert_layout` / `native_expert_scratch_bytes` are real
(transcribed from `iq_kernels.cu:1869`/`:1879`); the four ggml-cpu ROW kernels refuse LOUDLY (a CPU-hybrid path
this port forbids).  `native_expert_supported` is no longer a blanket `return false`: it answers from the grouped
shader INVENTORY (`native_gu_iq2s` = IQ2_S gate/up, `native_down_iq4nl` = IQ4_NL down - 2 of the 7 (gu,d) pairs
this pack uses) AND the still-missing launcher `native_expert_grouped` (the reference passes DEVICE POINTERS a
Vulkan shader cannot dereference), so the honest answer is FALSE per pair and the hole is NAMED.

**THE FIT, MEASURED.**  dense 1.374 GiB + experts 23.419 GiB = **24.793 GiB**, against **27.3-27.4 GiB usable** on
the B70.  Through the port's own device layer: a **27 GiB** arena ALLOCATES; **28 GiB FAILS** (exit 3).  The
weights fit with ~2.2-2.6 GiB for the dense arena's alignment/state, the KV/QSA/GDN state, the MoE/PLE/GR
workspaces, staging and descriptors.  **Residency is not the problem; the kernel is.**

**`--no-ple` IS IMPOSSIBLE WITH `--native`** (`generate.cpp:1789`/`:1830`) - the PLE key is native too, so the runs
carry PLE ON (file-backed, never in VRAM).  A real pack's decode REQUIRES the P6 verify window
(`Verifier::run`), which `Verifier::init` refuses today on the expert cache/profile AND `layer_verify_compatible`.

**A DIAGNOSTIC (capability forced `true`, then REVERTED byte-identically) names what is behind the expert gate:**
the device inits, the native pack loads its header ("largest blob 2.66 MB, embedding IQ4_XS, 322 MiB"), then exit 1
at **`blk.0.attn_gate.weight: this pack holds the tensor only in its GGUF form (run with --native SHARD1)`** - the
native DENSE coverage of the non-expert projections, BEFORE a single expert is read.  Also measured: the PCIe
probe reads **0.1 GB/s -> pcie_frac 0.00**, which disables the GPU's PCIe expert share.

**RESULTS (vega).**  Gate (`run_gate.sh`, background): Arc `intel_icd` **752/0/0** (exit 0; was 750 - the two new
cases: `case_native_expert_capability_entry`, two-sided), llvmpipe **740/0/3**, Ryzen iGPU **739/4/2** - the four
are the documented platform-level non-deterministic wrong-value defects, not this batch's cases.
`check_port_map.py` passes; `make_port_map.py` regenerates byte-identically.  **Map: `168 = 78 kernel + 0 shader +
45 host + 0 todo + 45 refused`** (two rows `refused -> host`; `refused` is NOT a capability).  **ENGINE BAR: 0
remaining engine-API undefineds (0 `strata::kernels::`, 0 `cuda*`) - 0 BY CONSTRUCTION (the refusals define the
unported symbols), NOT a porting gain.**  **`z820b` untouched.**



## FIRST TOKEN FROM THE INTEL ARC PRO B70 — the doorbell RING is a recorded device op, the STREAM SEAM is fixed, and the synthetic ZERO pack emits token 0 (2026-10-05, `vega`)

**A TOKEN CAME OUT OF THE B70.**  `strata_vulkan` on `strata-synth-pack-zero` (synthetic, all weight MATRICES
zero) ran the whole 48-layer decode and `sample_tokens` (greedy argmax) emitted **token id `0`**, exit 0,
**5.11 tok/s**, in 195.6 ms decode.  The token graph captured and launched (48 layers, one launch per token)
and the engine's own 248320-logit isfinite scan PASSED.  **BOUNDS: zero weights -> the CONTENT is trivial and
meaningless (zero logits, argmax index 0); this certifies the PIPELINE, not the model, and nothing about layer
numerics.  PLE off, CPU pool unused, prefill bypassed, expert streaming not exercised.  The real model is NOT
this run.**

**TWO BLOCKERS FIXED, both at the seam.**  (A) `doorbell_ring`/`doorbell_publish` raised the ring with a
capture-time HOST read-modify-write, so a recorded token graph replayed a stale literal and `session_run_token`
would have spun forever; the ring is now a RECORDED DISPATCH (`ring_inc.comp` via `Ctx::dispatch`, one
definition for the whole family).  (B) the DEFAULT decode passes `stream = nullptr` (generate.cpp:3893) and
CUDA resolves NULL to the default stream; `stream_of(nullptr)` now returns the device layer's default stream
(the shim's `g_current`, the SAME object), fixed ONCE for all 13 kernel TUs.  (C) `doorbell_wait` is a NO-OP
under capture with its loud refusal intact outside it.

**THE FIXTURE LIMIT, reported as such.**  The ORIGINAL random pack now runs the whole decode and stops at
`248320 of 248320 logits are not finite at position 0`.  Zeros are FINITE, so the composed 48-layer chain is
arithmetically SOUND and the divergence is the fixture.  A `--weight-scale 0.02` pack (MEASURED: quant scales
50x smaller, mean 2.97e-4 vs 1.49e-2) STILL diverges - random weights of any nonzero magnitude overflow this
architecture.  **No synthetic run certifies layer numerics either way.**  Real weights need the native-expert
path (`coder-iq1_m`: no pack experts.bin; 25.1 GB from the GGUF shards) - the next item.

**RESULTS:** gate `intel_icd` **750 passed / 0 failed / 0 skipped** (exit 0; the 9 doorbell arms stay green,
+7 ring +2 stream); **4 new injections all `FALSIFIED`** (2 ring-literal, 1 wait-records-node, 1
stream-null-not-default); map `168 = 78 kernel + 0 shader + 43 host + 0 todo + 47 refused` (the four doorbell
device ops `host -> kernel`); `check_port_map.py` passes, `make_port_map.py` byte-identical.  **ENGINE BAR: 0
remaining engine-API undefineds (0 `strata::kernels::`, 0 `cuda*`) - 0 BY CONSTRUCTION (the refusals DEFINE
the unported symbols), NOT a porting gain.**

## `copy_from_mapped` WIRED + the PUBLISH HANDSHAKE; THE PROGRAM LINKS; the `refused` MAP KIND (2026-10-05, `vega`)

**THE ENGINE BAR: `134` raw / `55` distinct / `41` `strata::kernels::` / `0` cuda → `0` / `0` / `0` / `0`** (the
exact whole-archive recipe: the four engine libs vs the backend `cudart`/`device`, `-lvulkan -lpthread`).  **The
`strata` PROGRAM LINKS** (`make strata_vulkan` exit 0).  That is linkability, NOT a run - see the RUN line below.

**THE MAP: `168 = 74 kernel + 0 shader + 47 host + 0 todo + 47 refused`** (was `75/5/61/27`).  `copy_from_mapped`
moved `host → kernel` (shader `copy`); a FIFTH KIND **`refused`** now marks every symbol the backend defines ONLY
as a loud refusal - enforced by `check_port_map.py` (a `refused` row with no definition FAILS).  A `refused` symbol
is a HOLE, NEVER a capability.  `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.

**DELIVERABLE A - `copy_from_mapped` (`session.cpp:875`, the captured per-layer parts copy).**  A Vulkan shader
cannot dereference mapped host memory, so the backend binds the mapped region's DEVICE-VISIBLE buffer (the shim
maps a HOST_VISIBLE|HOST_COHERENT block in `cudaHostAlloc`; `mapped_register`/`mapped_resolve` in `vk_arena.*`).
THE HANDSHAKE: the host's store into the mapping is the publish, and it must sit AFTER capture and BEFORE each
launch - a store made only at capture is the STALE BLOCK.  **A latent defect of exactly that class was found and
fixed:** `copy_i32_from_mapped` (`layer.cpp:913/914`, also captured) did a host-staged write unconditionally,
which under capture records NOTHING; it now RECORDS a `vkCmdCopyBuffer` re-reading the region.  **Proved by
`case_copy_from_mapped_entry`** (6 verdicts, green on Arc / llvmpipe / Ryzen iGPU): direct == engine's rule; wrapper
== the port's shader path (rival buffer); a recorded replay after a re-publish sees the new bytes; no stale block;
live publish (the capture-freeze falsifier); a non-mapped source REFUSED.

**DELIVERABLE B - THE LOUD REFUSALS.**  `refusals_vk.cpp` gains the ~40 kernels-namespace holes (drafter class C,
P6 verifier class D, the A/B arm, the `kernels_cpu` prefetch); `refusals_engine_vk.cpp` (new) answers
`RemoteExpertOpt` + defines `device_code_error` as its REAL answer; `refusals_prefill_vk.cpp` (new) answers the
prompt-path kernels.  Unconditionally-run ctors/dtors are REAL empty bodies, not refusals.  `pinned.cu` (0 kernels;
host code) is WIRED as C++ against the shim - the MODEL-LOAD path.

**DELIVERABLE C - THE RUN.**  `strata_vulkan --help` runs (program start); `--pack DIR --tokens 1,2,3` stops at
**MODEL LOAD**: the engine loads a PACK (`<dir>/index.txt`) and this box holds only raw IQ1_M GGUF shards (58 GB);
exact error `strata generate: cannot open /home/bob/strata-models/IQ1_M/index.txt` (exit 1).  **NO TOKEN.**  Packing
was not attempted (26 GB free on `/` for a 58 GB model; "no artifact bakes").

**DELIVERABLE D - the instrument fix.**  `run_gate.sh` AND `inject-verify.sh` compiled the gate from a HAND-KEPT
TU list that went stale three batches running; both now source `gates/harness_sources.sh`, which GLOBS
`vulkan/src/**/*.cpp` and fails LOUDLY if the count on disk differs from the list.  The guard is proven to fire
(a `#error` TU in a new `vulkan/src/guardprobe/` subdirectory made `build_harness` fail with it).

**Gate (vega):** Arc `intel_icd` **741/0/0** (exit 0); llvmpipe **729/0/3**; Ryzen iGPU **729/3/2** - the 3 FAILs
are the documented platform-level non-deterministic wrong-value defect (`budget: independent requery`,
`bf16_gemv_fp32_mmvf_cols`, `bf16_gemv_fp32_mmvf_multi`), not this batch's cases.  **`z820b` PENDING.**  Detail in
`NEXT.md`.


## `sample_tokens` WIRED and the ORDERED DECODE-PATH LIST (2026-10-05, `vega`)

**THE MAP now reads 168 = 75 kernel + 5 shader + 61 host + 27 todo** (was 74/6/61/27): `sample_tokens` — the step
that PRODUCES A TOKEN — moved `shader → kernel` and `vulkan/src/kernels/sampler_vk.cpp` defines it.  The other
five `shader` rows are NOT on a single-token decode and were classified, not stubbed.

**THE ENGINE BAR (`138 → 134` raw / `56 → 55` distinct / `42 → 41` `strata::kernels::` / CUDA stays `0`).**  The
ONE-LAYER-BODY bar is UNCHANGED (`18` raw / `0` kernels-ns).

**THE ORDERED DECODE-PATH LIST — TWO of the 41 remaining kernels-namespace symbols are reached by a single-token
decode, in this order:** (1) **`copy_from_mapped`** (`session.cpp:875`, inside the captured per-layer block,
UNCONDITIONAL; a DEVICE kernel reading MAPPED host memory, mis-kinded `host`; UNPORTED — the #1 next item, and a
handshake seam rather than a wrapper, because a Vulkan shader cannot dereference host memory) and (2)
**`sample_tokens`** (`generate.cpp:7742`, the decode tail; WIRED here).  Everything else is a non-selected
configuration (drafter, verifier, remote/peer tiers, load) or the `kernels_cpu` half.  Full table + classes in
`plan/DECODE-PATH-TRIAGE.md`.

**`sample_tokens` PROVED** by `case_sample_tokens_entry`: three arms (sampled/split, greedy flag, temperature 0)
each bitwise against the SHADER PATH and against the ENGINE'S OWN RULE, plus a CAPTURE ARM (records → replays
bitwise).  Gate on `vega`: Arc **735/0/0** (exit 0), llvmpipe **723/0/3**, Ryzen iGPU **724/2/2 / 723/3/2 / 722/4/2**
across three runs — the radeon failures are the documented platform-level non-deterministic wrong-value defect
(`qsa_block_scores`, `bf16_gemv`, `bf16_gemv_fp32_mmvf{,_cols,_multi}`, `ple_block`, a DIFFERENT set each run),
NOT this batch's cases; all six `sample_tokens entry` verdicts pass on all three arms.  `check_port_map.py` passes;
`make_port_map.py` regenerates byte-identically.  The new `sample-tokens-entry-temp0-to-sampled` injection bites
(and `inject-verify.sh`'s stale `rebuild_harness` TU list is fixed).  **`z820b` PENDING.**  Detail in `NEXT.md`.

## THE CUDA GRAPH API OVER THE PORT'S OWN RECORDED STEP — the recorder COMPILES and REPLAYS (2026-10-05, `vega`)

**THE NEW BAR (the engine executable, not just the layer body): `154` undefined references / `64` distinct,
and `0` CUDA-runtime symbols** — the shim now answers the whole CUDA-runtime surface the engine library
references, the graph API included.  The four recorder TUs `src/core/{graph,session,mtp,verify}.cpp` COMPILE
against the shim; the one-layer-body link is unchanged (`18` / `0` kernels-namespace); the remaining 51
kernels-namespace symbols are the port's unported forward-path work, not the graph API.

**A graph IS the port's recorded step** (no second mechanism): the SAME `Ctx::encode_dispatch` / `vkQueueSubmit`
+ fence wait, with `capture_begin` diverting `Ctx::dispatch` to `record_dispatch` so a capture RECORDS.
`EndCapture` closes it; `Instantiate` takes ownership; `Launch` = `Ctx::submit_owned`; D2D copies record as
`vkCmdCopyBuffer`.  Preserved: record-once/replay-many, live-buffer re-read, capture-does-not-run, bitwise
replay==direct.  NOT preserved (stated in the shim header): the multi-stream model, and H2D/D2H copies + memsets
inside a capture, which are `cudaErrorStreamCaptureUnsupported`.  Full analysis + mapping:
`plan/CUDA-GRAPH-MAPPING.md`.

**The record of the engine's usage** (`src/core/session.cpp`, NOT the unused `GraphRegistry`): per layer
`cudaStreamCreate` → `cudaStreamBeginCapture(ThreadLocal)` → run the layer body → `cudaStreamEndCapture` →
`cudaGraphInstantiate` → `cudaGraphDestroy`; per token `cudaGraphLaunch` per layer.  It is recording, not
graph-level semantics; between replays only the pinned staging CONTENTS change (graph.hpp NOTE 2).

**Proved on the Arc** by `case_cuda_graph_entry` (11 new verdicts): capture records and does not run; the graph
holds N dispatch nodes; replay == direct execution BITWISE; host mutations between replays are SEEN; replay
twice with no contamination; 6 instantiate/destroy cycles replay correctly, leak no arena bytes, and leave no
recordings owned; a H2D copy and a memset inside a capture are each REFUSED loudly; a D2D copy records and
replays.  **Three injections bite**: `graph-drop-last-node`, `graph-replay-stale`, `graph-exec-destroy-leak`
(the last observed by the new `Ctx::owned_recordings()` instrument).

**Verified on `vega`:** gate Arc **693/0/0** (exit 0), llvmpipe 680/0/3, Ryzen iGPU 683/0/2; a LATER run's radeon
cross-arm carried the documented `budget` flake + the open `bf16_gemv` wrong-value defect (neither is this
batch's cases); `check_port_map.py` passes (`168 — 66 kernel, 14 shader, 61 host, 27 todo`); `make_port_map.py`
byte-identical; `strata_vk_cudart_smoke` and `strata_vk_entry_smoke` RUN PASS.  **`z820b` PENDING.**  The full
`strata_vulkan` PROGRAM is not yet linkable (it needs a wider CUDA surface the recorder does not); reported, not
the graph API.  Full detail in `NEXT.md`.

## THE LAST FIVE NAMES, THE LINK's kernels PART AT ZERO, and M-B RUNS (2026-10-05, `vega`)

**THE BAR MOVED `41 → 18` raw / `7 → 0` distinct full-signature `strata::kernels::` symbols / `5 → 0`
name-only.**  The five names are wired with real definitions and gate cases: `bf16_gemv_fp32_mmvf_cols`
(`matvec_vk.cpp`), `build_rope_table` + `rope_table_set` (+`_release`/`_for`/the mrope pair) in the new
`rope_vk.cpp`, `copy_i32_from_mapped` (`elementwise_vk.cpp`), and `PleTable::{collect,is_open,issue}` - the
engine's OWN `src/kernels/ngram.cpp` (+ `ple_reader.cpp` + `direct_file.cpp`) is now linked into
`strata_vulkan_kernels` (`ple_vk.cpp`'s transcribed `ngram_rows` deleted, so the engine's definition is the
ONE).  **The residual 18 raw references are NOT the five names and NOT kernels-namespace:** they are the
engine's own `strata::core::` cross-TU symbols `layer.cpp` references (`LayerView::name`, `WeightTable::find`,
`native_embed`, `NativeEmbed::gather_one`) plus `main`, whose homes are `layout.cpp`/`weights.cpp`/
`native_head.cpp`/an engine executable.

**A LATENT DEFECT FOUND AND FIXED WHILE WIRING:** the port's `native_rope_apply` IGNORED the rope table
`layer.cpp:698` registers, so under `STRATA_ROPE_TABLE=1` it would have silently differed from the engine's
table path (~0.0014 rad at 32K).  It now REFUSES loudly in exactly that configuration; the default is
bit-for-bit the engine's `<false>` branch.

**M-B RUNS.**  The new `strata_vk_layer_smoke` target (`vulkan/tests/layer_smoke.cpp`, EXCLUDE_FROM_ALL) links
`layer.cpp` + `layout.cpp`/`weights.cpp`/`native_head.cpp` + the backend with `--gc-sections` (dropping
`native_head`'s unwired `iq_embed_rows`/`iq_dequant_f32`), writes a synthetic RANDOM-weight pack, loads it
through the engine's own `WeightTable::load`, and runs the GDN mixer `gdn_layer` on the Arc.  Evidence: out
256/256 finite, range `[-131.279, 148.152]`, variance `3178.08`, a different activation moves 256/256 elements,
two runs from a re-zeroed state are 256/256 bitwise equal.  **It does NOT prove numerical correctness** (no
cheap whole-layer reference exists); per-kernel correctness is the gate's job.

**Verified on `vega`:** gate Arc **682/0/0** (exit 0), llvmpipe **670/0/3** (documented skips), Ryzen iGPU
**673/0/2**; `check_port_map.py` passes (`168 — 66 kernel, 14 shader, 61 host, 27 todo`); `make_port_map.py`
regenerates `PORT-MAP.tsv` byte-identically (one row: `bf16_gemv_fp32_mmvf_cols` shader → kernel);
`strata_vk_layer_smoke` builds + RUNS.  **`z820b` PENDING** (suspended, no WoL).  Full detail in `NEXT.md`.

## THE TWO SPLIT GEMVs + `shared_expert`, NINE REACHABILITY VERDICTS, and THE INSTRUMENT FIX (2026-10-05, `vega`)

**THE BAR MOVED `64 → 41` raw / `19 → 7` full-signature / `17 → 5` name-only.**  Landed: `s_gemv_q8k_split` and
`s_gemv_q8_0_split` (the pair that BLOCKED `shared_expert`; the shader `s_gemv_q8_split` was already in the tree -
only the engine definition was missing, the `indexer_key_append` shape), and `shared_expert` itself, wired in full
in a new `vulkan/src/kernels/shared_expert_vk.cpp`.  The nine remaining no-shader symbols were each settled from
the engine's own code and are UNREACHABLE under the shipped configuration, so each is a LOUD REFUSAL
(`vulkan/src/kernels/refusals_vk.cpp`).  **One of them was mis-classified: `fused_gr_read` IS reachable** under the
shipped `--native` launch (`g_fused_gr` is true; the claim that the backend forces it false was false of the code) -
fixed at the cause by answering `fused_gr_supported` false, which keeps the PORTED `gr_read` on the path.
**The instrument:** PORT-MAP's `kernel` kind now means "a shader exists **AND** the backend defines the symbol";
a new `shader` kind states "a shader exists, the backend does not define it" (15 rows, 14 of them newly surfaced).

Verified on `vega`: gate Arc **669/0/0**, llvmpipe **657/0/3** (documented skips), Ryzen iGPU **659/1/2** (the open
`budget: independent requery agrees` flake, not this batch).  Four new falsifications bite.  `check_port_map.py`
passes; `make_port_map.py` byte-identical; all changed `.cu`-derived rules oracled against the engine's own body.
**`z820b` PENDING.**  Full detail in `NEXT.md`'s top section and `plan/DECODE-PATH-TRIAGE.md`.

## THE MoE / QSA / GR / PLE TAIL — six more entry points, and TWO CLASSIFICATION FINDINGS (2026-10-05, `vega`)

**THE BAR MOVED `75 → 64` raw / `25 → 19` full-signature / `23 → 17` name-only.**  The six, in the order the
engine's own body reaches them, wired across `vulkan/src/kernels/qsa_vk.cpp` (two) and `ple_vk.cpp` (four),
engine headers unchanged, each proved by a new `case_*_entry` (raw lines below):

| # | symbol | call site | kind | proof: wrapper == shader path (bitwise) | wrapper vs the oracle |
|---|---|---|---|---|---|
| 1 | `qsa_step_fill` | layer.cpp:908 | PURE HOST | — (no shader) | 32/32 int32 vs the `qsa.cu:699` rule; 3 rivals MOVE |
| 2 | `indexer_key_append` | layer.cpp:948 | kernel | 640/640 + spare key, w 0 | 256/256, worst 0.0655 of the terms/mean bound |
| 3 | `fused_gr_supported` | layer.cpp:1189/:1328 | PURE HOST | — | 6/6 geometries vs `fused_gr.cu:1164` |
| 4 | `shared_expert_scratch_bytes` | layer.cpp:347 | PURE HOST | — | 7/7 widths vs `shared_expert.cu:234`; 2 rivals MOVE |
| 5 | `moe_combine` | layer.cpp:464 | kernel | 2560+2560+37, w 0 | w 0.078 / 0 / 0.0588 (terms bound) |
| 6 | `ngram_rows` | layer.cpp:1293 | PURE HOST | — | 304/304 **vs the EXTERNAL `ref/ngram.py` vectors** |

**FINDING 1 — the parent's no-shader list is wrong for one symbol.**  `qsa_step_fill` takes an
`int32_t* host_step` and writes HOST memory (`qsa.cu:699`); it has no shader because it needs none.  Wiring it
is a wrapper job, and it was still undefined in this tree.  (PORT-MAP has always called it `host`.)

**FINDING 2 — `indexer_key_append` was recorded "LANDED" but had NO DEFINITION here.**  The class-A triage
table (`plan/DECODE-PATH-TRIAGE.md`) says "LANDED 2026-10-05"; the shader and a shader case exist, but the
`strata::kernels::indexer_key_append` wrapper did not — so the layer link still showed it undefined.  This is
the same shape as the norm-weight-indexing defect: **a symbol asserted ported without checking that a
definition existed for its CONTRACT.**  Now written and proved.

**THE DISCIPLINES.**  (a) *A shader/oracle shared mistake must not pass.*  `ngram_rows`' oracle is
`ple_oracle_vectors.inc`, GENERATED from `ref/ngram.py` — not from this engine — so a mistake shared between
the transcription and a transcription-built oracle cannot pass; the four named rivals (XOR→sum, `%vocab`→mask,
`prev` reversed, the EOS cut applied AFTER the store) each MOVE on 5–6 of the 6 external cases.  Every other
oracle is the engine's OWN rule (the `.cu` body), and each rival has a host-side margin proving it MOVES:
`indexer_key_append` norm-before-pool **5.30e-01** / rotate-at-last **8.22e-02**; `moe_combine`'s shared
ROUTER-WEIGHTED rival; `qsa_step_fill`'s ceil-`n_bid` / unclamped-width / `n_bid`-on-`pos`.
(b) *The latent-defect class.*  Checked the six against the three named shapes: no parameter used as something
it does not mean (each wrapper carries the CUDA's own argument contract); no in-place kernel called where the
CUDA writes a separate destination (`moe_combine`/`indexer_key_append` are both correct as written); no
element-size mismatch (the two kernels' regions are f32, matching their shaders).  ONE mis-kind was found in
the map, reported below.

**REPORTED, NOT STUBBED — the remaining 19, and the point M-B is stuck at.**
**`N = 19` symbols remain; **`8` are WRAPPABLE** and **`11` have NO SHADER in this tree** (a shader-port job,
not a wrapper job).**  The 8: `bf16_gemv_fp32_mmvf_cols` (shader exists, verifier-only), `build_rope_table`,
`rope_table_set`, `copy_i32_from_mapped`, `PleTable::{collect,is_open,issue}` (all pure host) and **`shared_expert`
— PARTIALLY**: its CANONICAL path dispatches `s_gemv_q8_0_split` / `s_gemv_q8k_split` for the K-quant gate/up
and the legacy Q8_0-activation down projection, and NEITHER has a shader, so those two no-shader symbols block
it.  The 11 no-shader: `qsa_attend_step`, `qsa_index_step`, `topk_512_step`, `native_qsa_indexer_append`,
`native_flash_attn_short_step` (all PORT-MAP `todo`), `s_gemv_q8_0_split` / `s_gemv_q8k_split` (todo),
`fused_gr_read` (contract-removed) and **`kv_ring_table` / `kv_stream_reset` / `kv_stream_resolve` — which
PORT-MAP kinds `host` but `kv_stream.cu` LAUNCHES (`reset_kernel<<<128,256>>>`, `ring_kernel<<<64,256>>>`,
`resolve_kernel`/`copy_kernel`)**: a mis-kind of the same family as `gr_read`/`fused_gr_read`, left in the map
(byte-identical) and reported here.

**RESULTS (vega).**  Gate: Arc (`intel_icd`) **655/0/0**, llvmpipe **643/0/3** (+28 verdicts per arm, 0 failed).
Ryzen iGPU (`radeon_icd`): 645/1/2, 644/2/2, 645/1/2 across three runs — the documented intermittent
`budget` flake and the open wrong-value defect, now seen in **`bf16_gemv entry n_in=2560 n_out=128` (1 of 512
elements) and `fused_gdn_ab h_v=48 n=2560` (1 of 96)** — **neither is one of this batch's cases.**  On the
record as the brief asked: the intermittent defect has now appeared in a THIRD kernel (`fused_gdn_ab`) as well
as `bf16_gemv`/`bf16_gemv_split` and `ple_block`, so it is device/driver-level, not one kernel's bug.
`check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 112 shaders built, 93 claimed`);
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  `strata_vk_entry_smoke` builds + runs PASS.
**`z820b` PENDING** (suspended, no WoL — no XTX/K620 number is claimed).  The CUDA graph API was NOT touched.

## THE PLE / GR SHARED STAGES + THE MoE ROUTING ROWS — the next six entry points, and a WORKSPACE-LAYOUT DEFECT FIXED (2026-10-05, `vega`)

**The GR pair is the SHARED stage (every layer, 48 of 48); the PLE stage is layer-1; the two MoE rows complete the
block.**  Wired in the new `vulkan/src/kernels/ple_vk.cpp`, in the SOURCE order the engine's body reaches them:
`gr_write` (`layer.cpp:1261`/`:1195`/`:1329`), `ple_block` (`:1208`), `ple_history_advance` (`:1222`), `gr_read`
(`:1255`), `router_top10` (`:373`, `moe_route`'s generic router) and `native_moe_combine` (`:463`, the DEFAULT
combine - `native_moe_combine_enabled()` answers true).  Engine headers unchanged.  Plus the `host` rows
`gr_workspace_init` and `ple_block_scratch_bytes`, and the PLE/GR branch policy (`ple_native_bf16_enabled` /
`ple_native_postops_enabled` -> FALSE).  Each proved by a new `case_*_entry` through the ENGINE WRAPPER, BITWISE
against the port's shader path AND against the engine's own `.cu` rule, `EnginePin`-pinned:

| kernel | shader(s) | wrapper == shader (bitwise), worst | wrapper vs engine rule, worst |
|---|---|---|---|
| `gr_write` | gr_write | 10240/10240 + 192/192, w 0 | w 0.609 / 0.613 (zero-inject EXACT) |
| `ple_block` | q8_0+s2 / bf16 + gnorm/gate/bcast/conv/add3 | all 7 exports w 0 (key 10240, result 10240) | w 6.58e-04 (ple.cu chain, device-fed) |
| `ple_history_advance` | ple_history_advance | 92160/92160, w 0 | bit-exact (a copy) |
| `gr_read` | gr_norm/gr_down/gr_gate/gr_mean/gr_inject | 66/66 + 19/19, w 0 | w 9.31e-03 / 2.43e-04 |
| `router_top10` | router_top10_f32 | 80/80, w 0 | ids exact; w 1.44e-07 |
| `native_moe_combine` | native_moe_combine | 2560+2560+37, w 0 | w 9.25e-02 / 0 / 3.23e-02 |

**A CROSS-CUTTING (WORKSPACE-LAYOUT) DEFECT FOUND AND FIXED.**  The port's GR shaders store the bf16 activations
as **f32** (`gr_norm.comp` binding 3 is `float v[]`) while the engine's `GrWorkspace` sizes `xq`/`lq` as **uint16**,
so wiring `gr_read` made the shader write 4 bytes into a 2-byte region.  `gr_workspace_init` now sizes `xq`/`lq` for
f32 (legal: `GrWorkspace` is an opaque pointer struct and `gr_workspace_bytes` is the allocation authority).  A
second defect - `ple_block`'s wrapper writing `gnorm(gated)` back onto `gated` (the port's gnorm is in place) - was
caught as a `0/10240` bitwise disagreement and fixed at the source.  **Every rival reading has a host-side margin
proving it MOVES the reference**; `native_moe_combine`'s k=1 "first term as a sum" rival is stated as GENUINELY
indistinguishable (`0.0f + x == x`) rather than asserted.

**THE LINK PROGRESS - the one-layer-body link: `91 -> 75` undefined references / `33 -> 25` distinct full-signature
`strata::kernels::` symbols / `31 -> 23` name-only.**  The attention/QSA/MoE/GR/PLE/rope group falls **25 -> 17**;
glue 0, matvec/GEMV/KV 6, GDN mixer 0, other 2.  Measured with `$HOME/vkbuild-vulkan` reconfigured + rebuilt from
the tree first (never `-G Ninja`).

**RESULTS (vega).**  Gate: Arc (`intel_icd`) **639/0/0** (exit 0), llvmpipe **627/0/3**, Ryzen iGPU (`radeon_icd`)
**630/0/2** - **+27 verdicts per arm**, 0 failed.  `strata_vk_entry_smoke` builds + RUNS four of the six wrappers
(PASS, exit 0).  `check_port_map.py` passes (`168 - 78 kernel, 61 host, 29 todo; 112 shaders built, 93 claimed`);
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **`z820b` PENDING.**  CUDA graph API untouched.


## NOT DETERMINISTIC ON THE Ryzen iGPU — a characterised open defect (2026-10-05)

**A green `run_gate.sh` on `vega` does not prove determinism.** The same commit, binary and fixture seed fails
on the radeon ICD in ~11% of runs and passes in the rest: the bad case is one row of `bf16_gemv` /
`bf16_gemv_split` at `n_in=2560 n_out=512` (and once `fused_gdn_ab`), by 1.23–664× the terms-derived bound.
Measured: radeon 10 bad runs in 59 (and 11 in 102 with a diagnostic build) · **intel 0 in 84 · lvp 0 in 47**.
A standalone probe against the shipping `bf16_gemv.spv` gives **0 bad dispatches in 80,000 on the Arc and
1/8,000 (quiet) to 73/20,000 (under load) on the iGPU**, and re-dispatching the identical buffers is always
correct — so the kernel's arithmetic is right and the transient is device/driver-level (RADV
RAPHAEL_MENDOCINO). It is NOT a tolerance question, NOT a fixture/RNG drift, NOT descriptor-pool growth
(pool 2 is created in every run, passing ones included), NOT an aliasing bug (one row, not many), and NOT a
stale input read (a single-element mutation of the previous/zero/byte-zeroed form does not reproduce it, and
a stale word would persist on re-dispatch). **The case is not skipped on radeon and the bound is not
widened.** Full detail and evidence paths in `NEXT.md`'s top section.

## THE DEFAULT QSA DECODE ATTENTION — `qsa_decode_attn_step` WAS A REACHABLE HOLE, NOW PORTED (2026-10-05, `vega`)

**The default decode attention had NO shader in this tree — a HOLE of the class the previous two batches found.**
The symbol `qsa_decode_attn_step` (`layer.cpp:980`) is reached inside
`if (g_fast_attn && !native_flash_attn_short && dump == nullptr)` — every input the SHIPPED configuration's
default: `g_fast_attn` **true** (`layer.cpp:42`), `native_flash_attn_short` **false** (`layer.cpp:92`; set only by
`--native-flash-attn-short`, NOT by `--native`), `dump` **nullptr** on the decode path.  It reads the KV POOLS
through the PAGE TABLE — a DIFFERENT kernel from `attn_decode_short` (which its own header says is
`native_flash_attn_short_step`, a gathered `[cap,2,256]` f16 window).  PORT-MAP's mapping and the reachability
audit's row were both wrong; both are corrected.

**Ported:** `ports/vulkan/shaders/qsa_decode_attn.comp` — the pools read directly through the page table (no
gather), one workgroup per query head, online softmax; mode 0 = f16 pools (the shipped `--kv fp16` default),
mode 1 = int8 codes + fp16 scale/64; q4_0 and K8V4 refused loudly.  The engine's chunk+merge CUDA is re-derived
onto one workgroup with barrier-tree reductions (subgroup ops banned) and needs no scratch — bit-identity with
the chunked CUDA is NOT claimed; the gate MEASURES the difference against a DOUBLE transcription of the rule.
Wired in `vulkan/src/kernels/qsa_vk.cpp` (9th entry point) + the `host` row `qsa_decode_attn_scratch_floats`.

| case | rule (the engine's OWN CUDA = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `qsa_decode_attn` (3 arms: page_size 4/1/8) | `qsa_decode_attn.cu`: `row=(page*kv_heads+kvh)*page_size+(cell%page_size)`, `s=(Σ q·k)/sqrt(head_dim)`, softmax over resident cells, `attn=Σ w·v` | wrapper == shader path **6144/6144 w 0** x3; shader vs rule **w 9.65e-04 / 2.01e-04 / 5.92e-04** (abs floor 1e-5) | `qsa-decode-attn-drop-kv-head` -> **FAIL 0/6144 w 2.26e+04** |

**THE LINK PROGRESS — the one-layer-body link: `94 → 91` undefined references / `35 → 33` distinct
full-signature `strata::kernels::` symbols / `33 → 31` name-only.**  The attention/QSA/MoE/GR/PLE/rope group falls
**27 → 25** (glue 0 · matvec/GEMV/KV 6 · GDN mixer 0 · other 2); the two symbols resolved are
`qsa_decode_attn_step` and `qsa_decode_attn_scratch_floats`.  Measured with `$HOME/vkbuild-vulkan` reconfigured +
rebuilt from the tree first (never `-G Ninja`); recipe + group table in `NEXT.md`'s top section.

**RESULTS (vega).**  Gate: Arc (intel_icd) **612/0/0** (exit 0), llvmpipe **600/0/3**, Ryzen iGPU (radeon_icd)
**603/0/2** — **+9 verdicts per arm**, 0 failed.  `strata_vk_entry_smoke` RUNS the new wrapper (PASS, worst rel
2.92e-05).  `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 112 shaders built, 93 claimed`);
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (one row: `qsa_decode_attn_step → qsa_decode_attn`).
Bench (Arc, SOLO row): **0.38 / 1.29 / 2.52 ms** at n_ids 256 / 1024 / 2048 — linear in the selection width, the
honest cost of the correctness form (the CUDA's chunk form is the fast one and was not ported).  **`z820b`
PENDING.**  CUDA graph API untouched.

## INCREMENT I5 (ENGINE) — the ATTENTION / QSA / rope entry points (the non-GDN body's next eight) + A CROSS-CUTTING WEIGHT-INDEXING DEFECT FOUND AND FIXED (2026-10-05, `vega`)

**THE EIGHT, in the order the non-GDN body `qsa_layer` (`src/core/layer.cpp:876`) reaches them** (the plan's list is
not an order): `native_qsa_rms_norm_weighted` (:879), `native_rope_apply` (:881), `rope_neox_apply` (:882),
`qsa_block_scores` (:970), `qsa_block_topk` (:971), `native_qsa_gate_apply` (:1010), `qsa_gate_apply_f32` (:1011)
and `native_router_top10` (:370, the expert routing).  Wired in the new `vulkan/src/kernels/qsa_vk.cpp`, engine
headers unchanged; each proved by a new `case_*_entry` through the ENGINE WRAPPER, BITWISE against the port's shader
path AND against the case's explicit oracle, `EnginePin`-pinned:

| kernel | shader | wrapper == shader (bitwise), worst | wrapper vs oracle, worst |
|---|---|---|---|
| `native_qsa_rms_norm_weighted` | native_qsa_rms_norm_weighted.spv | 20480/20480, w 0 | w 2.2e-07 (the norm rule, double) |
| `native_rope_apply` | native_rope_apply.spv | 49152/49152, w 0 | w 1.6e-05 (analytic rule, tol 3e-3) |
| `rope_neox_apply` | rope_neox.spv | 768/768, w 0 | w 0.106 (NEOX rule, bit-exact tail) |
| `qsa_block_scores` | qsa_block_scores.spv | 72/72, w 0 | 72/72 w 5.56e-07 (relu + dead + 1e9) |
| `qsa_block_topk` | qsa_block_topk.spv | 128/128 x 3, w 0 | 128/128 x 3 w 0 (selection rule, ids exact) |
| `native_qsa_gate_apply` | native_qsa_gate_apply.spv | 6144/6144, w 0 | w 7.87e-07 (native gate rule) |
| `qsa_gate_apply_f32` | qsa_gate_apply_f32.spv | 6144/6144, w 0 | w 7.75e-07 (gate rule) |
| `native_router_top10` | native_router_top10.spv | 20/20, w 0 | w 1.86e-07 (ids exact, weights tol 1e-5) |

**THE LATENT DEFECT (a parameter used as something it does not mean), FOUND AND FIXED.**  `rms_norm.comp` and
`native_qsa_rms_norm_weighted.comp` indexed the norm WEIGHT by the ELEMENT index `i = row*cols + c` instead of the
COLUMN index `c`.  The engine's contract is per-column (`native_qsa.hpp`: "gamma[n_cols] broadcast over rows";
`native_qsa.cu` `gamma[col]`; `elementwise.cu` `r[c] = (r[c]*w[c])*inv`) - so every row past the first read the
wrong weight, and a cols-long engine weight is read OUT OF BOUNDS.  Three cases' oracles reproduced the shader's
element-wise indexing (the "oracle built from the thing under test" trap), masking it.  Fixed the two shaders
(`w.v[c]` / `gamma.v[c]`), the three oracles (column index), and `elementwise_vk.cpp`'s weight resolution
(`cols` floats, not `rows*cols`).  This would have produced a SILENTLY WRONG token on every QSA layer and every
legacy `rms_norm_weighted` call.  Also fixed: the router case's multi-token arm formed a `weights + t*10` (40-byte)
device pointer, unaligned on llvmpipe's 16-byte descriptor-offset limit - the cause of the lvp arm EXITING without
totals; it now completes.

**THE `host` ROW:** `rope_scaling()` (+ `rope_scaling_set`) - the rope constants, owned by `rope_scaling.cu` on a
CUDA build.  No capability flag flipped (qsa/rope were already TRUE).

**REPORTED, NOT WIRED:** `qsa_decode_attn_step` (:980) - PORT-MAP maps it to `attn_decode_short`, but that shader is
`native_flash_attn_short_step`'s (a gathered f16 window) while this symbol reads the KV POOLS through a page table -
a NO SHADER row, a shader-port job, not stubbed.  Same class: `native_flash_attn_short_step` (:995),
`qsa_attend_step`, `qsa_index_step`, `topk_512_step`, `native_qsa_indexer_append`.

**THE LINK PROGRESS - the one-layer-body link: `106 → 94` undefined references / `44 → 35` distinct full-signature
`strata::kernels::` symbols / `42 → 33` name-only.**  The attention/QSA/MoE/GR/PLE/rope group falls **36 → 27**;
glue 0, matvec/GEMV/KV 6, GDN mixer 0, other 2.  All 35 remaining are referenced by `layer.cpp` itself; the only
other structural blocker is the deferred CUDA graph API (I5).  Recipe + group table: `NEXT.md`'s I5 section.
MEASURED with `$HOME/vkbuild-vulkan` reconfigured + rebuilt from the current tree first (never `-G Ninja`).

**RESULTS (vega).**  Gate: Arc (intel_icd) **603/0/0** (exit 0), llvmpipe 591/0/3, Ryzen iGPU (radeon_icd)
**593/1/2** - **+34 verdicts per arm**, 0 failed on the Arc.  ON THE RECORD: the radeon arm's first run carried the
documented `budget` flake and the second the open RADV-only `bf16_gemv_split` defect; a third read 594/0/2.
`strata_vk_entry_smoke` builds + runs on the Arc.  `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo;
111 shaders built, 92 claimed`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **`z820b`
PENDING.**  CUDA graph API untouched; plan not re-scoped.

## INCREMENT I3 — the first eight MATVEC / GEMV / KV entry points (the weight-side math + the KV cache) + THE STANDARDISED LINK PROGRESS BAR (2026-10-05, `vega`)

**THE EIGHT, in the order `src/core/layer.cpp` and its siblings reach them** (the plan's list is not an order):
`quantize_q8_K` (:236), `quantize_q8_0` (:237), `native_quantize_q8_1` (:150), `native_mmvq` (:151, the
COMPOSITE - six `*_mmvq` shaders by ggml type), `kv_append_q4_step` (:935), `kv_gather_q4_step` (:985),
`quantize_q8_0_scaled` (session.cpp:868) and `native_q5_k_f32` (native_head.cpp:78).  Wired in the new
`vulkan/src/kernels/matvec_vk.cpp`, engine headers unchanged; each proved by a new `case_*_entry` through the
ENGINE WRAPPER, BITWISE against the port's shader path AND against the case's explicit oracle, `EnginePin`-pinned:

| kernel | shader | wrapper == shader (bitwise), worst | wrapper vs oracle |
|---|---|---|---|
| `quantize_q8_K` | quantize_q8_K.spv | 876/876, w 0 | 876/876 w 0 (one of the two scale forms) |
| `quantize_q8_0` | quantize_q8_0.spv | 102/102, w 0 | 102/102 w 0 (ggml's image) |
| `native_quantize_q8_1` | quantize_q8_1.spv | 1152/1152, w 0 | 1152/1152 w 0 (one of the two division forms) |
| `native_mmvq` | iq*_mmvq.spv (6) | 4/4, w 0 (IQ2_S) | 4/4 w 0.0435 (IQ2_S dot, terms bound) |
| `kv_append_q4_step` | kv_q4_append.spv | 147456/147456, w 0 | 272/272 w 0 (Q4_0 group rule, bit-exact) |
| `kv_gather_q4_step` | kv_q4_gather.spv | 8192/8192, w 0 | 4096/4096 w 0 (Q4_0 reader rule, fp16 exact) |
| `quantize_q8_0_scaled` | quantize_q8_0_scaled.spv | 114/114, w 0 | 114/114 w 0 (CPU rule + fp32 scales) |
| `native_q5_k_f32` | quantize_q8_1 + native_q5_k_f32 | 3/3, w 0 | 3/3 w 0.00796 (packed Q5_K dot, terms bound) |

**THE `host` ROWS THIS TU ANSWERS (not a bare bind):** `iq_row_bytes` (the per-format row stride the `*_mmvq`
shaders take as `row_bytes` - a quantisation constant table), `native_mmvq_supported` (the CAPABILITY CHECK that
gates the composite: TRUE for exactly the six types with a shader, FALSE otherwise, so the native loader cannot
route to an unported kernel), `native_mmvq_weight_bytes` and `native_q8_1_bytes` (the sizes the wrappers
range-check); plus the IQ grid tables placed LAZILY per stream in `Stream::iq_grids`, from the new
`vulkan/src/kernels/iq_grids_vk.hpp` (a verbatim copy of the port's generated harness header), because a
per-dispatch upload would exhaust the arena.  There is NO `*_enabled()` in this family to flip, so the caps case
needed no new arm.

**THE LINK PROGRESS - the one-layer-body link: `188 → 170` undefined references / `59 → 53` distinct
full-signature `strata::kernels::` symbols / `57 → 51` name-only.**  Six of the eight move THIS link (the two
reached by sibling TUs do not, as I2c's non-`layer.cpp` symbols did not).  The matvec/GEMV/KV group falls
**21 → 15**; glue 0, GDN mixer 0, attention/QSA/MoE/GR/PLE/rope 36, other 2.  Recipe + group table: `NEXT.md`'s
I3 section.  MEASURED with `$HOME/vkbuild-vulkan` reconfigured + rebuilt from the current tree first.

**RESULTS (vega).**  Gate: Arc (intel_icd) **552/0/0** (exit 0), llvmpipe 540/0/3, Ryzen iGPU (radeon_icd)
**543/0/2** - **+16 verdicts per arm**, 0 failed.  ON THE RECORD: the radeon arm's first run read **540/3/2** (the
documented `budget` flake plus this batch's own `native_q5_k_f32` case bug, since fixed); the re-run read
543/0/2.  `strata_vk_entry_smoke` builds + runs on the Arc.  `check_port_map.py` passes (`168 — 78 kernel, 61
host, 29 todo; 111 shaders built, 92 claimed`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.
**`z820b` PENDING.**  CUDA-runtime host surface untouched; plan not re-scoped.

## INCREMENT I2e — the remaining eight GDN / DeltaNet MIXER entry points, COMPLETING the mixer + THE STANDARDISED LINK PROGRESS BAR (2026-10-05, `vega`)

**THE MIXER IS COMPLETE: all fourteen entry points `gdn_layer` reaches are wired and proved** (I2d's six + I2e's
eight), in `vulkan/src/kernels/gdn_vk.cpp`.  The GDN / DeltaNet mixer group in the grouped table goes **8 → 0**.

**THE EIGHT, in the order `gdn_layer`'s body reaches them** (fused → native → legacy within each stage):
`native_gdn_beta_gate` (:296), `native_gdn_gate` (:297), `gdn_beta_gate` (:299), `native_gdn_step` (:308),
`gdn_step` (:309), `fused_gdn_step_norm` (:322), `native_gdn_out_norm` (:324), `gdn_out_norm` (:325).  Each proved
by a new `case_*_entry` through the ENGINE WRAPPER, BITWISE against the port's shader path AND against the case's
explicit oracle, `EnginePin`-pinned; the step/norm cases compare the mutated STATE bitwise too:

| kernel | shader | wrapper == shader (bitwise), worst | wrapper vs oracle, worst |
|---|---|---|---|
| `native_gdn_beta_gate` | native_gdn_beta_gate.spv | 48/48, w 0 | 48/48 w 1.42e-06 |
| `native_gdn_gate` | native_gdn_gate.spv | 48/48+5/5+300/300, w 0 | w 2.13e-07 / 6.54e-08 / 3.83e-07 |
| `gdn_beta_gate` | gdn_beta_gate.spv | 48/48, w 0 | 48/48 w 1.42e-06 |
| `native_gdn_step` | native_gdn_step.spv | 792576/792576 (+state), w 0 | 792576/792576 w 2.15e-03 |
| `gdn_step` | gdn_step.spv | 792576/792576 (+state), w 0 | 792576/792576 w 4.47e-03 |
| `fused_gdn_step_norm` | fused_gdn_step_norm.spv | 792576/792576 (+state), w 0 | 792576/792576 w 0.13 |
| `native_gdn_out_norm` | native_gdn_out_norm.spv | 6144/6144, w 0 | 6400/6400 w 5.89e-07 |
| `gdn_out_norm` | gdn_out_norm.spv | 6144/6144, w 0 | 6400/6400 w 8.08e-07 |

**NO `host` row was needed, MEASURED:** the only row the mixer reaches is `native_gdn_enabled()`, already answered
TRUE by `native_caps_vk.cpp`.  The GDN headers carry no `*_scratch_bytes` and no shape accessor - `GdnShapes` is a
by-value POD passed through, a TYPE not a symbol - and the one-layer-body link drops by exactly the eight symbols
with NO new undefined reference.  NONE of the eight is a module-state reader like `cvec_apply`; each is a bare
bind-and-dispatch carrying the CUDA wrapper's argument contract (`S == 128`, `h_v % h_k == 0`, `cols == 128`) as a
loud refusal.

**THE STANDARDISED LINK PROGRESS BAR.**  One stable build directory, **`$HOME/vkbuild-vulkan`**, RECONFIGURED AND
REBUILT FROM THE CURRENT TREE BEFORE MEASURING - the old recipe named a `<build>` placeholder and a stale library
once reported I2c's `204/73` after I2d's real `196/67`; a silently stale bar is worse than none, because
"unchanged" is a plausible reading.  Measured (total references / YOUR full-signature distinct / the PARENT's
simpler name-only distinct): **196 → 188** / **67 → 59** / **65 → 57** (the whole-file name-only variant: 81 → 73).
The reproducing recipe, both regexes quoted, and the stale hazard are at the top of `NEXT.md`'s I2e section.  The
remaining 59 group as: matvec/GEMV/KV 21, attention/QSA/MoE/GR/PLE/rope 36, **GDN mixer 0**, other 2.

**RESULTS (vega).**  Gate: Arc (intel_icd) **536/0/0** (exit 0), llvmpipe 524/0/3, Ryzen iGPU (radeon_icd)
**527/0/2** — **+20 verdicts on every arm**, 0 failed; the radeon arm's pre-existing `budget`/`bf16_gemv` flakes
appeared on early attempts (none of the eight new cases) and cleared on a re-run reading 527/0/2.
`strata_vk_entry_smoke` builds + runs on the Arc.  `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29
todo; 111 shaders built, 92 claimed`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (the eight
were already `kernel` rows).  **`z820b` PENDING.**  CUDA-runtime host surface untouched; plan not re-scoped.

## INCREMENT I2 (CONTINUED FURTHER-STILL) — the first six GDN / DeltaNet MIXER entry points + THE LINK PROGRESS BAR (2026-10-05, `vega`)

**THE LINK PROGRESS — the measured one-layer-body link: `204 → 196` undefined references / `73 → 67` distinct
`strata::kernels::` symbols** (also `71 → 65` under the parent's simpler name-only regex, which drops the two
signatures carrying a nested `strata::kernels::GdnShapes`).  This batch owns ALL SIX of the drop.  The remaining 67,
grouped by subsystem (**glue 0 · matvec/GEMV/KV 21 · attention/QSA/MoE/GR/PLE/rope 36 · GDN mixer 8 · other 2**),
the reproducing command and the full group table are at the top of `NEXT.md`'s I2-continued-further-still section.

**THE SIX.**  In the order the mixer's own body (`gdn_layer`, `src/core/layer.cpp:223`, called from
`block_layer_pre`) reaches them - NOT the plan's list: `fused_gdn_conv_l2` (:250), `native_gdn_conv_silu` (:253),
`gdn_conv_step` (:255), `native_gdn_l2_norm` (:266/267), `gdn_l2_norm` (:269/270) and `fused_gdn_ab` (:287).  Wired in
the new `vulkan/src/kernels/gdn_vk.cpp`, engine headers unchanged; each proved by a new `case_*_entry` through the
ENGINE WRAPPER, BITWISE against the port's shader path AND against the case's explicit oracle, each pinned to the
harness device (`EnginePin`):

| kernel | shader | wrapper == shader (bitwise) | wrapper vs oracle |
|---|---|---|---|
| `fused_gdn_conv_l2` | fused_gdn_conv_l2.spv | 40960/40960 + 2048/2048, worst 0 | 10240/10240 w 2.42e-04; 512/512 w 6.63e-06 |
| `native_gdn_conv_silu` | native_gdn_conv_silu.spv | 12800/12800 + 120/120, worst 0 | 5120/5120 w 1.24e-05; 48/48 w 2.49e-07 |
| `gdn_conv_step` | gdn_conv_step.spv | 96/96 + 1200/1200, worst 0 | 24/24 w 2.04e-06; 300/300 w 1.15e-06 |
| `native_gdn_l2_norm` | native_gdn_l2_norm.spv | 128/128 + 2048/2048, worst 0 | 128/128 w 1.18e-07; 2048/2048 w 1.68e-07 |
| `gdn_l2_norm` | gdn_l2_norm.spv | 2048/2048 + 384/384, worst 0 | 2048/2048 w 1.26e-07; 384/384 w 0 |
| `fused_gdn_ab` | fused_gdn_ab.spv | 96/96 + 8/8, worst 0 | 96/96 w 3.48e-07; 8/8 w 9.31e-08 |

**NO `host` ROW WAS NEEDED.**  The only `host` row the six reach is `native_gdn_enabled()`, already answered in
`native_caps_vk.cpp`; the GDN headers carry no `*_scratch_bytes` / shape-accessor symbol.  None of the six reads
module state (unlike `cvec_apply`); they carry the CUDA wrappers' argument CONTRACTS as loud refusals (`d_conv != 4`,
`channels % 128`, `n_embd % 8`).

**RESULTS.**  `strata_vk_entry_smoke` RUNS the six new wrappers and PASSES on the Arc.  Gate on `vega`: **Arc
516/0/0 (exit 0), llvmpipe 504/0/3, radeon iGPU 507/0/2** - **+24 verdicts per arm**, 0 failed; the radeon `budget`
flake did not fire.  `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 111 shaders built, 92 claimed`)
and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (the six were already `kernel` rows).  **`z820b`
PENDING** (no XTX/K620 number).  The CUDA-runtime host surface was NOT touched and the plan was NOT re-scoped.
**Left: the GDN mixer's remaining 8 symbols** (the beta/gate and step/norm stages, including `fused_gdn_step_norm`).

## INCREMENT I2 (CONTINUED FURTHER) — the next five glue entry points + THE LINK PROGRESS BAR (2026-10-05, `vega`)

**THE LINK PROGRESS — the measured one-layer-body link: `214 → 204` undefined references / `80 → 73` distinct
`strata::kernels::` symbols.**  This batch owns TWO of the seven (`cvec_apply`, `cvec()` — the only two of the
five it wired that `layer.cpp` itself reaches); the other five dropped from I2/I2b's glue/`doorbell_*` that
`layer.cpp` also calls and that the README's "backend answers 4" baseline predates.  Reproducing command and the
remaining 73 grouped by subsystem (glue 0 · matvec/GEMV/KV 21 · attention/QSA/MoE/GR/PLE/rope 36 · GDN mixer 14 ·
other 2) are at the top of `NEXT.md`'s I2-continued-further section.

**THE FIVE.**  In the order the forward path reaches them (NOT the plan's list): `add_inplace`
(`expert_source.cpp:2353`), `scatter_rows_f32` (`peer_experts.cpp:241`), `cvec_apply` (`layer.cpp:1330` — the only
one `layer.cpp` calls DIRECTLY), `gather_rows` (`mtp.cpp:450`, the MTP drafter) and `f32_to_f16_bulk` (NO
`src/core/` call site).  Wired in `vulkan/src/kernels/elementwise_vk.cpp`, engine headers unchanged; each proved
by a new `case_*_entry` through the ENGINE WRAPPER, BITWISE against the port's shader path AND against the case's
explicit oracle, each pinned to the harness device (`EnginePin`):

| kernel | shader | wrapper == shader (bitwise) | wrapper vs oracle |
|---|---|---|---|
| `add_inplace` | add.spv | 1000/1000, worst 0 | 1000/1000 == d+s |
| `scatter_rows_f32` | scatter_rows_f32.spv | 3072/3072 + 30/30, worst 0 | 3072/3072 + 30/30 vs the rule |
| `cvec_apply` | cvec_apply.spv | 6144/6144, worst 0 | 6144/6144, worst 4.74e-07 (tol 1e-4) |
| `gather_rows` | gather_rows.spv | 576/576 + 316/316, worst 0 | 576/576 + 316/316 vs ids[r] source rows |
| `f32_to_f16_bulk` | f32_to_f16.spv | 1024/1024, worst 0 | 1024/1024 == f16_from_f32 |

**`cvec_apply` IS NOT A THIN WRAPPER.**  Its wrapper reads the engine's control-vector MODULE
(`strata::kernels::cvec()`), owned by `src/kernels/cuda/cvec.cu`, which a Vulkan build does not compile — so the
backend also answers the cvec `host` row (`cvec`, `cvec_upload`, `cvec_replicate`, `cvec_set_enabled`,
`cvec_enabled`) and places the device tables lazily in each `Stream`'s arena (`Stream::cvec_tables`,
`vulkan/src/device/vk_arena.hpp`).  The four others are thin.

**RESULTS.**  `strata_vk_entry_smoke` RUNS all five (plus the earlier wrappers/doorbell) and PASSES on the Arc.
Gate on `vega`: **Arc 492/0/0 (exit 0), llvmpipe 480/0/3, radeon iGPU 483/0/2** — +14 verdicts per arm (5 new
cases), 0 failed; the radeon `budget` flake did not fire.  `check_port_map.py` passes (`168 — 78 kernel, 61 host,
29 todo; 111 shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.
**`z820b` PENDING** (no XTX/K620 number).  The CUDA-runtime host surface was NOT touched (still the un-approved
re-scope I2b reported).

## INCREMENT I2 — THE DOORBELL REPLACEMENT + THE FIRST THREE GLUE ENTRY POINTS (2026-10-05, `vega`)

The ENGINE half of I2 (`plan/BACKEND-INTEGRATION.md` §3). Measured on `vega`: Arc B70 (ANV), Ryzen iGPU (RADV)
and llvmpipe.

**THE DOORBELL.** `src/kernels/cuda/elementwise.cu`'s handshake is a kernel that SPINS on host memory ordered
by `__threadfence_system()` (`doorbell_publish` :293, `doorbell_ring` :209, `doorbell_wait` :214; call sites
`layer.cpp:380`/`:389` and `session.cpp:873`). Vulkan has no equivalent and this port forbids a waiting
kernel, so `vulkan/src/device/sync.*` (new) replaces the two directions SEPARATELY: DEVICE->HOST becomes the
submission FENCE (every submit in the device layer is fenced and waited, so `sync_publish` returning IS "every
payload byte has landed" - that is the `__threadfence_system()`+ring ordering), and HOST->DEVICE becomes a
HOST-DRIVEN SPLIT SUBMISSION (the consumer is submitted only after the host has written the answer, so the
device never waits).  The result is that the recorded step is SPLIT at the handoff - the structural change the
plan's I2 risk note names - and a literal translation (a kernel waiting for the host inside the SAME submission
that must finish before the host can answer) could never complete.  `case_sync_handoff` pins the ORDERING, not
the values: a consume BEFORE the answer reads the zero sentinel (1024/1024), the SAME call after the answer
reads it (1024/1024), and the ring counts device publications (1 then 2).  The deadlock arm is reasoned, NOT
executed: executing it means submitting a spinning kernel, which this port forbids and which risks the display
card.  PASS on all three devices.

**THE THREE GLUE ENTRY POINTS.** The plan's I2 list is NOT an order.  Read from `src/core/layer.cpp`'s layer
body, the first three glue kernels `gdn_layer` (the mixer for 36 of 48 layers) reaches are `silu_inplace`
(:257), `scale_inplace` (:276) and `f32_to_bf16_bulk` (:290); all three are wired in
`vulkan/src/kernels/elementwise_vk.cpp` and proved by `case_*_entry` through the ENGINE WRAPPER, bitwise against
the port's own shader path AND against the explicit oracle: scale 1000/1000, silu 1000/1000 bitwise (worst
7.97e-07 vs the double reference), f32_to_bf16 1024/1024.  The engine's headers are unchanged.

**I2 CONTINUED (2026-10-05, same `vega`).**  The rest of I2's engine half, plus the engine-program measurement.
(a) The five `doorbell_*` symbols `layer.cpp:380/389` and `session.cpp:873` call are answered in the new
`vulkan/src/kernels/doorbell_vk.cpp` from `sync.*`: DEVICE->HOST is the fenced publish (the ring is the host's
own count - "the fence is the ring"); HOST->DEVICE is `doorbell_wait`, which SUBMITS NOTHING (the host writes
the answer first; an un-answered handoff is a loud refusal, never a spin), so no kernel ever waits.
`sync_copy_fenced` was exposed from `sync.*` as the ONE copy primitive both the Handoff operations and the
doorbell symbols use.  `case_sync_handoff` gained arms E rather than a parallel case: publish device->host
ordered 528/528 with ring == 1, ring/publish_value 2/2, `doorbell_wait`+consumer 1024/1024.
(b) The next three glue kernels the layer body reaches - `gdn_gate` (:300), `rms_norm_weighted` (:880),
`embedding_gather` (:1083) - are wired and proved by `case_*_entry` through the ENGINE WRAPPER, bitwise against
the shader path AND against the oracle (gdn_gate 48/48+144/144, worst rel 2.7e-07 vs softplus; rms_norm_weighted
512/512+8192/8192+12288/12288, worst rel 1.4e-07 vs the double ref; embedding_gather 512/512 on four arms,
bitwise vs the two-rounding rule).  The `rms_norm_weighted` wrapper supplies ones when the CUDA contract's
null weight is passed (Vulkan has no null descriptor).
(c) THE ENGINE-PROGRAM GAP, MEASURED: under `STRATA_ENABLE_VULKAN=ON` the top-level `CMakeLists.txt` `return()`s
before `src/` is configured, so the engine's own program is never compiled.  CONFIGURE 0.16 s builds the backend
only; COMPILE of `src/core/*.cpp` + `src/program/generate.cpp` is clean except `STRATA_VERSION` (a top-level
`add_compile_definitions` after the `return()`); LINK of one layer body fails with 214 undefined references (80
distinct `strata::kernels::`, 4 answered; 12 CUDA-runtime symbols - the engine calls the CUDA runtime directly
in 18 host files); RUN - only `strata_vk_entry_smoke` links and runs.  **This is reported as a RE-SCOPE for the
user to approve, NOT as I5's wiring**: the plan prices the engine side as kernels-namespace symbols and has no
line item for a CUDA-runtime host shim / 18 migrated host files.  See `NEXT.md`'s I2-continued section.  Gate on
`vega`: Arc 477/0/0, llvmpipe 465/0/3, radeon non-deterministic (the pre-existing `budget` flake and the open
`bf16_gemv` defect; a green run is 468/0/2).  Port map unchanged and byte-identical on regeneration.

**A GATE HAZARD, FOUND AND FIXED IN THE CASE (NOT A KERNEL BUG).** The gate's own `case_icd_resolution`
(~:4673) calls `unsetenv("VK_ICD_FILENAMES")` and never restores it, so every later `*_entry` case's second
`VkInstance` enumerated EVERY ICD and took the Intel Arc while the harness `ctx` sat on the arm's ICD.  The
same `silu_f32.spv` on the same input differs by exactly 1 ULP between the Arc and llvmpipe (x=-3.44161081:
Arc bits bddaa466, llvmpipe bddaa465), which failed the bitwise arm on llvmpipe (535/1000) and RADV (798/1000).
The entry cases now pin `STRATA_VK_DEVICE` to the HARNESS DEVICE's NAME before opening the engine stream
(`EnginePin`, in the appended block); the bitwise claim is KEPT, not loosened, and all arms are green.
`case_fwht256_entry` has the same exposure and never noticed, because fwht256 is bitwise identical on every
device.

**THE W26 ALLOCATION/VISIBILITY SEAM - CHECKED; THE CHARACTERISATION STANDS.** The two questions the defect
asks of the device layer, answered by reading it.  (1) **Does it fence before releasing/reusing a buffer?**
`Ctx::free` (`vk_compute.cpp:684-697`) does NOT fence, but it does not need to: EVERY submission the layer
makes is fenced and waited BEFORE it returns (`end_oneshot_and_wait` :774-788, used by `dispatch` and both
staging directions; `submit_recorded` :1064-1076 - the file has exactly two `vkQueueSubmit` sites, each
followed by `vkWaitForFences(..., UINT64_MAX)`), so no submission that references a buffer can be in flight
when `free()` runs.  The release/reuse hazard is RULED OUT for this layer's call patterns.  (2) **Is an upload
barrier missing that the fence does not cover?** No.  A staged upload carries a `TRANSFER_WRITE ->
SHADER_READ|HOST_READ` memory barrier AND the fence (`stage_upload` :795-815); a mapped write is a plain
memcpy into `HOST_VISIBLE|HOST_COHERENT` memory (the type `Ctx::alloc` requires, :398), whose visibility to
the next dispatch is Vulkan's implicit host-write ordering at `vkQueueSubmit`.  So the seam is not the cause,
and the defect is left as characterised rather than guessed at - `NEXT.md`'s two standing hypotheses (a
stale/partial read at the write/dispatch boundary, or a subgroup-width shared-memory read on the 64-wide RADV
implementation) are untouched.  One residual, stated rather than hidden: freeing a buffer that a RECORDED
command buffer still binds is a use-after-free the synchronous API cannot prevent (the recording holds raw
`VkBuffer` handles), and no gate pattern reaches it because every replay is fenced before returning.

**BUILD.** `cmake -DSTRATA_ENABLE_VULKAN=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF
-DSTRATA_ENABLE_SYCL=OFF` CONFIGURE 0.15 s, BUILD 0.94 s on `vega` (the option `return()`s before the CUDA
engine, as I1 left it, so this builds the backend: device layer + sync + the four kernel TUs + the smoke
target).  `strata_vk_entry_smoke` RUNS on the Arc, llvmpipe and the Ryzen iGPU: fwht256, the three glue
wrappers and the four handoff arms all PASS.  Gate on `vega`: **Arc 455/0/0, llvmpipe 443/0/3, radeon iGPU
446/0/2, exit 0**; `check_port_map.py` passes (`168 - 78 kernel, 61 host, 29 todo`) and `make_port_map.py`
regenerates `PORT-MAP.tsv` byte-identically; the radeon arm's own caveat (a green run means "no failure
observed", not deterministic) still stands.  **The box `z820b` is PENDING** - no XTX/K620 number here.

## THE SAMPLER, MEASURED PROPERLY — the engine's DEFAULT is the SPLIT, and the PENALTY HOIST — the performance tier's first target (2026-10-05)

The one number in the port above 100 ms, with a caveat the release notes carried: `sampler_kernel_f32` **546.9 ms
(Arc) / 202.0 ms (XTX)** is the ONE-BLOCK **fallback**, not the engine's default.  Both halves are now settled.

**THE DEFAULT, BY CALL SITE.**  `sample_tokens` (`src/kernels/cuda/sampler.cu:977`): greedy or `temp <= 0` ->
the argmax (`:988`); `sampled_path() == Old` (`STRATA_OLD_SAMPLER=1`) -> the one-block `sampler_kernel` (`:1000`);
else the default `:1013`
`sampled_path() == Split && n_blocks <= 64 && n_tokens <= 64 && !stream_capturing(stream)` -> the **SPLIT**
(`sampler_split_part_kernel` + `sampler_split_merge_kernel`), with `sampler_one_block_kernel` as the FALLBACK.
`sampled_path()` is `Split` unless `STRATA_OLD_SAMPLER`/`STRATA_SAMPLER_ONE_BLOCK` is set (`:854-859`).  At the
model's vocab 248,320, `n_blocks = ceil(248320/4096) = 61 <= 64`, so **the split is what the engine runs, and the
546.9/202.0 figure was never the default's cost.**

| device (vega) | `sampler_kernel_f32` **fallback** (tokens=1) | `sampler_split` **DEFAULT** (tokens=1) | split/block | DEFAULT, tokens=64 |
|---|---:|---:|---:|---:|
| Arc B70 (intel_icd) | 546.81 ms | **13.50 ms** | **0.025** | 13.97 ms |
| Ryzen iGPU (radeon_icd) | 301.55 ms | **9.06 ms** | **0.030** | 74.91 ms |
| llvmpipe (lvp_icd) | 1171.46 ms | **67.64 ms** | **0.058** | 391.29 ms |

Shipped `bench/vk_bench.cpp` rows, `reps=5`, one dispatch per batch; full tables in `bench/README.md`.

**WHY IT WAS SLOW, MEASURED BEFORE FIXING.**  The split cost **348.95 ms (Arc) / 161.62 ms (iGPU) / 665.46 ms
(llvmpipe)** pre-fix.  Not bandwidth (the 993 KB row loads in microseconds); not the one-workgroup-per-row
decomposition (64 rows cost ≈ what 1 row cost: Arc 348.95 -> 384.54 ms); it WAS a serial scan in the wrong place —
removing the penalty window took it to **12.66 ms (27.6×)**.  `sampler_row_topk` re-read each logit and re-scanned
the whole history window on EVERY k round (`O(k · span · hlen)`), where the engine builds a 4,096-bit block bitmap
and penalises each element ONCE into registers (`src/kernels/cuda/sampler.cu:690-706`).

**THE FIX.**  `common/sampler_select.glsl` caches each partition element's penalised logit once in the lane's
registers (the engine's `s[kSplitPerLane]` shape, `SC_PER_LANE = 16`) and runs the k rounds over the cache; the
selection is byte-for-byte unchanged.  **348.95 -> 13.50 ms Arc (25.8×), 161.62 -> 9.06 iGPU (17.8×), 665.46 ->
67.64 llvmpipe (9.8×).**  The include is shared, so `coupled_sample.comp` gets the hoist too.

| case | rule | measured (vega Arc) | falsified by |
|---|---|---|---|
| `sampler_split` (SIX arms now; the new 6th exercises a real penalty hit) | partition top-k + ordered merge, tail in double; the 6th arm: id 100 × 8 in the window, `penalty_repeat = 4` -> head 9.0/4 = 2.25, selection moves to id 4200 | 6 arms bit-exact vs the transcribed chain; parity with `sampler_kernel` unchanged | `sampler-select-penalty-drop` -> **FAIL `sampler_split: the repeat penalty is applied ONCE per partition (the hoist)` 2/16**; `sampler-split-merge-drop-parts` still 9/16 |

**Gate, after the change: vega Arc 445/0/0, lvp 433/0/3, radeon-iGPU 436/0/2 (`run_gate.sh` exit 0)** — +1 verdict
per implementation (the new arm).  Map unchanged: `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo`),
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **Box `z820b`: PENDING** — the box is suspended
(`100% packet loss`, `No route to host`, no WoL), so the default's cost on the XTX is NOT measured and no cross-host
number is claimed; 202.0 ms stands as the XTX fallback figure.  Full detail in `NEXT.md`'s top section.

---

## THE BF16-PROJECTION PAIR — `bf16_gemv` + `bf16_gemv_split` (one shared shader) — and the 8 SPECULATIVE-DRAFTER ROWS LABELLED CLASS C (2026-10-05)

The SECOND soft edge the batch-5 REACHABILITY AUDIT raised is CLOSED by porting BOTH members of the
`native_bf16_projections` toggle that `project_bf16` (`src/core/layer.cpp:94-100`) selects between.  The setting
(`layer.cpp:91`) DEFAULT is **false** and the port pins it nowhere, so the DEFAULT selected the UNPORTED
`bf16_gemv_split` (layer.cpp:291/:292/:367) and `bf16_gemv` (:918/:962) on the MAIN forward path — the same shape
as the `native_qsa_indexer_append` hole.  `--native` sets the setting true (generate.cpp:1805 → :2286) and the
TRUE branch was already the ported `bf16_gemv_fp32_mmvf`; porting both sides closes the edge for EITHER value.
Pinning `layer_set_native_bf16(true)` was rejected (the setter is engine host code the port does not fork).

| case | rule (the engine's OWN body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `bf16_gemv` (4 arms 2560/512, 2560/128, 128/64, 2/1) | `bf16_gemv.cu`: `y[o]=Σ f32(x[i])·f32(w[o*n_in+i])`, products exact in f32, row read as 32-bit PAIRS, plain `acc+=a*b` (NOT `__fmaf_rn`) | **515/515 w 1.22e-02**, 131/131 w 5.73e-03, 67/67 w 3.2e-02, 4/4 w 1.84e-03 (terms-derived `gemv_bound`) | `bf16-gemv-swap-halves` → FAIL 4/515 w 1.54e+05 |
| `bf16_gemv_split` (same shader; 4 arms 2560/512, 2560/48, 64/32, 2/1) | `bf16_gemv_split_kernel`/`bf16_gemv_warp_kernel` — same RULE, different reduction; both entry points rendered as ONE WORKGROUP per row (subgroup ops banned; the CUDA's warp shapes are a strategy, not the rule) | **515/515 w 6.99e-03**, 51/51 w 6.37e-03, 35/35 w 1.42e-02, 4/4 w 9.61e-03 | `bf16-gemv-row-base` → FAIL 4/515 w 2.62e+34 |

**A MEASUREMENT THAT CHANGED THE SHIPPED KERNEL.**  The naive one-thread-per-row decomposition (a real CUDA path)
was built and timed FIRST: at the engine's shapes it is **14–18x slower** than the workgroup form (Arc
`split/serial 0.073` at n_out=512, 0.057 at 48; Ryzen iGPU 0.103/0.055) because it is uncoalesced — the CUDA's own
comment says so, and CUDA takes it only below n_out=64.  The port therefore SHIPS the workgroup-per-row rendering
and the naive variant is not in the tree.  ONE shader serves both engine symbols (the
`bf16_gemv_fp32_mmvf`/`_cols` precedent).

**THE MEASUREMENT (ported `bf16_gemv` vs the ported native `bf16_gemv_fp32_mmvf`; same workgroup-per-row
decomposition, only the activation precision differs → a WASH is expected; full table in `bench/README.md`).**
A **WASH**: native/ported `bf16_gemv` **1.000** (Arc, n_out=512), **1.006** (Arc, 48), **0.996 / 0.997** (Ryzen
iGPU), **0.933 / 0.956** (llvmpipe).

**THE DRAFTER ROWS.**  `add_streams_broadcast`, `fused_gr_read_multi`, `window_ids`, `qsa_decode_attn_batch`,
`moe_group_resident`, `row_top_prob`, `map_ids`, `mtp_select` are labelled **CLASS C**: reachable only under the
`--spec 4 --mtp` draft loop, which this port does not select (`Verifier::init` refuses under
`layer_verify_compatible()`, which needs `g_fused_gr` false→forced, `native_qsa_indexer_enabled()` false, and
`native_bf16_projections` unpinned).  The flag chain that WOULD enable them is `--spec 4 --mtp` AND a
verifier-compatible native stack.  The TSV keeps kind `todo` but carries the class-C reason; the classification is
in `plan/DECODE-PATH-TRIAGE.md`.

**THE MAP MOVES BY TWO ROWS:** `168 — 76 kernel, 61 host, 31 todo` → **`168 — 78 kernel, 61 host, 29 todo`**;
`check_port_map.py` passes (`111 shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the change: vega Arc 444/0/0, lvp 432/0/3, radeon-iGPU 435/0/2 (exit 0)**;
**box `z820b`: NOT RUN — the box was UNREACHABLE this batch** (`No route to host`, ARP `INCOMPLETE`, no WoL path
from vega), so the box run is PENDING a whole-tree sync + `run_gate.sh`/`run_bench.sh` there.  Both injections
BIT (raw lines in `NEXT.md`).  Full detail in `NEXT.md`'s top section and `plan/DECODE-PATH-TRIAGE.md`.

---

## THE QSA GATE'S NATIVE MEMBER — class B batch 5 — `native_qsa_enabled()` FLIPS TO TRUE + the REACHABILITY AUDIT (2026-10-05)

`native_qsa_gate_apply` — the LAST symbol `native_qsa_enabled()` gates — is ported, gated against the engine's OWN
native body (`src/kernels/cuda/native_qsa.cu`'s `gate`, `:69-77`), and MEASURED against the legacy
`qsa_gate_apply_f32` it replaces.  With both gated symbols now having a shader, the flag's invariant is satisfied
and **`native_qsa_enabled()` answers TRUE**.  This batch also carries the **REACHABILITY AUDIT** of every remaining
`todo` row (`plan/DECODE-PATH-TRIAGE.md`) and **fixes the one hole it found**.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `native_qsa_gate_apply` (3 arms 24/256, 4/12, 2/8) | `native_qsa.cu` `gate`: `out = attn * (1/(1+expf(-raw)))`, `raw` = the SECOND half of each head's 2*head_dim block, all f32 | **6152/6152 w 7.89e-07**, 56/56 w 7.54e-07, 24/24 w 7.08e-07 | `native-qsa-gate-first-half` → FAIL 8/6152 w 4.13e+10 |
| `native capabilities: qsa flag` | `native_qsa_enabled()` == "every gated symbol has a built shader"; both QSA shaders now built | **3/3** — flag **TRUE** | under-claim `native-caps-qsa-false` → FAIL 2/3; over-claim (remove the `.spv`) → FAIL 1/3 |

**The rule is the SAME as the legacy member's; the arithmetic is not.**  The native body computes the sigmoid and
the product in **f32** (`expf`); the legacy CUDA in **f64**.  The target has no shaderFloat64, so this port's legacy
shader already computes in f32 — hence the native-vs-legacy pair is a **WASH** on all six devices measured
(**1.001 Arc / 0.984 iGPU / 1.018 lvp / 1.019 XTX / 0.904 K620 / 1.120 lvp**).  There is no dispatch chain (the QSA
gate is one dispatch either way), and that is stated rather than invented.  Full table in `bench/README.md`.

**THE FLIP WAS CHECKED.**  `native_qsa_enabled()` gates exactly two symbols — `native_qsa_rms_norm_weighted`
(`layer.cpp:879`, `mtp.cpp:488/491/514`, `verify.cpp:767`; ported batch 1) and `native_qsa_gate_apply`
(`layer.cpp:1010`, `verify.cpp:775/883/887`; ported here) — and the verify sites are unreachable because the P6
verifier cannot init (`layer_verify_compatible` still demands `native_bf16_projections` and `g_fused_gr` false, and
`native_qsa_indexer_enabled()` false).  Nothing unported becomes reachable.

**THE HOLE AND ITS FIX.**  The audit found `native_qsa_indexer_append` (`layer.cpp:944`, the MAIN forward path of
all 12 QSA layers every token) reachable because the shipped `--native` sets `o.native_qsa_indexer = true`
(`generate.cpp:1807`) and calls `native_qsa_indexer_set_enabled` (`:2294`) — while the Vulkan backend defined
**NEITHER the getter nor the setter**, though the contract named the required answer (`false`).  The backend now
defines both, answering **false** → the ported `indexer_key_append`.  The map row stays `todo`; the hole was
reachability.

**THE MAP MOVES BY ONE ROW:** `168 — 75 kernel, 61 host, 32 todo` → **`168 — 76 kernel, 61 host, 31 todo`**;
`check_port_map.py` passes (`110 shaders built, 91 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the commit: vega Arc 436/0/0, lvp 424/0/3, radeon-iGPU 427/0/2 (exit 0)**, +4
verdicts per arm.  **Box `z820b`: XTX 432/0/1 (the pre-existing M8 skip, so the script exits 1), K620 427/0/2,
llvmpipe 424/0/3 — 0 failed on every arm.**  The retired injection `native-caps-qsa-true` is recorded in `NEXT.md`.

**THE AUDIT IN ONE LINE:** of the 32 `todo` rows audited, 1 is now `kernel`, **11 are reachable-but-unported** (8
speculative-drafter symbols reachable only under the shipped `--spec 4 --mtp`; `native_qsa_indexer_append`, now
flag-closed; and `bf16_gemv`/`bf16_gemv_split` as a soft edge on the unnamed `native_bf16_projections` setting),
and 20 are unreachable under the current capability answers — each with its deciding call site named.

---

## THE PERFORMANCE TIER'S FUSED GDN PATHS — class B batch 4 — and `native_gdn_enabled()` FLIPS TO TRUE (2026-10-05)

The three fused **GDN paths** — `fused_gdn_conv_l2`, `fused_gdn_ab`, `fused_gdn_step_norm` — the LAST symbols
`native_gdn_enabled()` gates.  Each is gated and oracled against the engine's OWN fused body
(`src/kernels/cuda/fused_gdn.cu`), each MEASURED against the multi-dispatch chain it replaces and against the
non-fused native kernel(s) where a comparison exists.  With ALL NINE gated symbols now having a shader, the
invariant `case_native_capabilities` asserts is satisfied and **`native_gdn_enabled()` answers TRUE**.

| case | rule (the engine's OWN fused body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `fused_gdn_conv_l2` (3 arms C=10240/512/384) | `fused_gdn.cu` `gdn_conv_l2_kernel`: 4-tap conv (no zero-bias) + SiLU + slide, then a per-head L2 `rsqrt(sum y^2 + eps)` on the q/k heads only | **51200/51200 w 2.66e-05**, 2560/2560 w 9.46e-06, 1920/1920 w 2.68e-05 (Arc); slid history bit-exact | `fused-gdn-conv-l2-drop-norm` → FAIL 49149/51200 w 1.93e+05 |
| `fused_gdn_ab` (3 arms h_v=48 n=2560 / h_v=4 n=64 / h_v=3 n=512) | `fused_gdn.cu` `gdn_ab_kernel`: BF16 mat-vec (32-bit pairs, low=2p/high=2p+1) then `beta=sigmoid(acc)`, `gate=softplus(acc+dt)*ssm_a` | **99/99 w 4.25e-07**, 11/11 w 1.33e-07, 9/9 w 7.79e-08 (Arc); terms bound, per-row margins | `fused-gdn-ab-swap-bf16-halves` → FAIL 3/99 w 25.9 |
| `fused_gdn_step_norm` (3 arms S=128; h_v=48/8/9) | `fused_gdn.cu` `gdn_step_norm_kernel`: folded-decay recurrence (contract vs UNDECAYED state) + readout vs UPDATED state + fused `1/sqrt(S)` + closing RMS (eps on the MEAN) with gamma and sigmoid(z) | **792580/792580 err/tol 0.319**, 132100/132100 w 0.0508, 148612/148612 w 0.0109 (Arc) | `fused-gdn-step-norm-silu-not-sigmoid` → FAIL 788501/792580 w 9.34e+04 |
| `native capabilities: gdn flag` | `native_gdn_enabled()` == "every gated symbol has a built shader"; all 9 now built | **4/4** — flag **TRUE** | `native-caps-gdn-false` → FAIL 3/4 (the old `native-caps-gdn-true` is retired — true is now the truth) |

**THE MEASUREMENT (fused/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
Every fused path beats the multi-dispatch chain it replaces on every device measured: `fused_gdn_conv_l2` vs
conv_silu+2× l2_norm **0.452 Arc / 0.712 iGPU / 0.560 lvp**; `fused_gdn_ab` vs 2× mmvf+beta_gate+gate **0.398 /
0.878 / 0.744**; `fused_gdn_step_norm` vs step+out_norm **0.929 / 0.987 / 0.930**.  The fused-vs-`native_gdn_step`
pair is a wash (1.021 / 0.998 / 1.092 — the fused kernel does that recurrence PLUS the norm's work), and the
fused-vs-`native_gdn_out_norm` row (6.2–63.7) is **not a like-for-like ratio** (the fused dispatch does the whole
step while out_norm alone is a ~70× smaller elementwise pass).  The win is in REMOVING DISPATCHES, as batches 2–3
found; a wash and a non-like-for-like row are both reported.

**THE FLAG FLIPS, and the fused paths' extra gates are SETTINGS, not capabilities.**  `fused_gdn_step_norm` is
selected by `g_fused_gdn && native_gdn_enabled() && state==128` (layer.cpp:306) — **no `native_bf16_projections`
term** — and `g_fused_gdn` **defaults true** (layer.cpp:42), so the flag flip alone makes that path reachable on
the default config: the shader HAD to exist before the flip (the strict invariant's point).  `fused_gdn_conv_l2` /
`fused_gdn_ab` additionally need `native_bf16_projections` (layer.cpp:247), a host setting defaulting false, set
by `--native-bf16`/`--native` (generate.cpp:2286), whose dependency `bf16_mmvf_f32` IS ported and gated.  The P6
verifier stays unreachable after the flip (`g_fused_gr` false, `native_qsa_indexer_enabled()` false,
`native_bf16_projections` false), so the router/moe `_multi` symbols stay off the path.

**THE MAP MOVES BY THREE ROWS:** `168 — 72 kernel, 61 host, 35 todo` → **`168 — 75 kernel, 61 host, 32 todo`**;
`check_port_map.py` passes (`109 shaders built, 90 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the commit: vega Arc 432/0/0, lvp 420/0/3, radeon-iGPU 423/0/2 (exit 0)** — the
radeon run's sole `budget: independent requery agrees` failure was the documented intermittent flake (a direct
re-run read 423/0/2 twice).  **Box `z820b`: XTX 428/0/1 (the 1 is the pre-existing M8 skip, so `run_gate.sh`
exits 1 there), K620 423/0/2, llvmpipe 420/0/3 — 0 failed on every arm**; the XTX's first run carried the same
`budget` flake (two direct re-runs read 428/0/1).  Four falsification injections all BIT.  **NOTE for the next
batch: the box's `vulkan/` must be synced too (the brief's sync command omits it) — a stale
`native_caps_vk.cpp` fails the flag arm 3/4.**

**TWO FIXTURE FINDINGS (both fixed, both recorded in `NEXT.md`).**  (1) The dropped-readout-scale rival is NOT an
exact invariance under the closing RMS (measured 0.109–0.19 rel-L1 — it is a moving margin, asserted > 0.05; the
first case asserted the opposite and the gate caught it).  (2) The vector rel-L1 softplus margin in `fused_gdn_ab`
was swamped by the forced branch row, so the case now uses a per-row max and a fixture whose `acc` lands in the
softplus-sensitive band.  The device matched the rule throughout — the fixtures were the defect.

---

## THE PERFORMANCE TIER'S GDN / DELTANET MIXER, the remaining three native kernels — class B batch 3 (2026-10-05)

The LAST three native **GDN / DeltaNet mixer** kernels — **COMPLETING the six** — `native_gdn_gate`,
`native_gdn_step`, `native_gdn_out_norm`, each replacing a legacy kernel already ported and gated, each oracled
against the engine's OWN native body (`src/kernels/cuda/native_gdn_preprocess.cu` / `native_gdn.cu`), each
MEASURED against that legacy kernel at the same shape on the same device (and, for `native_gdn_step`, against
the two-dispatch legacy chain it replaces), with **`native_gdn_enabled()` left answering FALSE** (the flag also
gates the three unported `fused_gdn_*` paths) and `case_native_capabilities` enforcing it.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc / box XTX) | falsified by |
|---|---|---|---|
| `native_gdn_gate` (3 arms h_v=48/5/300) | `native_gdn_preprocess.cu` `gate_softplus`: `softplus(alpha+dt) * ssm_a`, per-head one token; the threshold-20 branch | **48/48 w 2.79e-07**, 5/5 w 1.04e-07, 300/300 w 3.06e-07 (Arc); raw identity checked host-side to move | `native-gdn-gate-drop-ssm-a` → FAIL 0/48 w 8.89 |
| `native_gdn_step` (2 arms 128/16/48, 128/4/8) | `native_gdn.cu` `step`: decay FOLDED into the rank-1 update, contract against the UNDECAYED state, `1/sqrt(S)` readout scale FUSED; one thread per column (see the decomposition finding) | **792576/792576 err/tol 0.00155**, 132096/132096 w 0.00107 (Arc); INTERLEAVE + dropped-scale margins | `native-gdn-step-drop-readout-scale` → FAIL 786434/792576 w 2.03e+04 |
| `native_gdn_out_norm` (3 arms h_v=48/4/3, cols=128) | `native_gdn_preprocess.cu` `out_norm`: `(rms_norm(o) * gamma) * sigmoid(z)`, eps on the MEAN, cols==128 | **6400/6400 w 7.04e-07**, 768/768 w 6.13e-07, 640/640 w 4.92e-07 (Arc); NaN row-guard | `native-gdn-out-norm-silu-instead-of-sigmoid` → FAIL 2321/6400 w 21 |
| `native capabilities: gdn flag` | `native_gdn_enabled()` == "every gated symbol has a built shader"; the 6 native shaders exist | **4/4** — flag **FALSE** (the 3 unported fused shaders absent) | `native-caps-gdn-true` → FAIL 3/4 |

**THE MEASUREMENT (native/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
`native_gdn_step` is a **WIN — 0.917 Arc / 0.776 iGPU / 0.866 lvp (pair)** and **0.867 / 0.771 / 0.829 against
the legacy `scale_inplace`+`gdn_step` chain**; it does LESS memory traffic than the legacy kernel (the legacy's
first pass STORES the decayed state).  `native_gdn_gate` (0.901–1.044) and `native_gdn_out_norm` (0.974–1.144)
are **WASHES** — the same work per element and, at the layer's one-token / one-row shape, no algorithmic
difference to win.  A native kernel is not required to be faster, and a wash is reported as a finding.  **On the
box's XTX the three cases read `native_gdn_gate` 48/48 w 3.38e-07, `native_gdn_out_norm` 6400/6400 w 6.06e-07 and
`native_gdn_step` 792576/792576 err/tol 0.00154 — all PASS, 0 failed on every arm.**

**A FINDING THE MEASUREMENT PRODUCED, and it CHANGED THE SHIPPED KERNEL.**  The native `step` body uses ONE
32-LANE WARP per column; subgroup ops are banned here (Intel picks the SIMD width per kernel), so the warp has
two non-subgroup renderings.  The workgroup-per-column **barrier-tree** version was built and MEASURED against
the legacy kernel at this shape — **1.564x Arc, 8.328x Ryzen iGPU, 43.501x llvmpipe** (chain 1.471 / 8.078 /
41.597): 6144 workgroups each doing two 8-round barrier trees with adjacent invocations `h_v*S` floats apart is
a large REGRESSION.  So the SHIPPED port keeps the **coalesced one-thread-per-column serial** decomposition the
legacy port already uses (the CUDA's warp shape is a parallelism strategy, not the rule) and carries the native
arithmetic and its fused readout scale.  The variant is recorded in `NEXT.md` and `bench/README.md`, not shipped.

**CAPABILITY CONTRACT.**  `native_gdn_enabled()` gates NINE symbols: the six native kernels (now ALL ported) +
the three `fused_gdn_*` paths (also gated on `g_fused_gdn` + `native_bf16_projections`), which have no shader.
The backend answers **false**, and the gdn arm asserts the flag equals "every gated symbol has a built shader"
AND that the six ported shaders exist — a strict invariant (a "reachable" formulation was considered and
rejected: it would let the engine route to an unported symbol when a setting changed).

**THE MAP MOVES BY THREE ROWS:** `168 — 69 kernel, 61 host, 38 todo` → **`168 — 72 kernel, 61 host, 35 todo`**;
`check_port_map.py` passes (`106 shaders built, 87 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the commit: vega Arc 423/0/0, lvp 411/0/3, radeon-iGPU 414/0/2 (exit 0); box
`z820b` XTX 419/0/1 (the 1 is the pre-existing M8 skip), lvp 411/0/3, K620 414/0/2 — 0 failed on every arm**
(+8 verdicts on each); a re-run of the box's XTX cross-arm, after the documented intermittent `budget` flake,
read 419/0/1.  Full detail in `NEXT.md`'s top section.

---

## THE PERFORMANCE TIER'S GDN / DELTANET MIXER, first three native kernels — class B batch 2 (2026-10-05)

The first three native **GDN / DeltaNet mixer** kernels — the mixer is **36 of the model's 48 layers** — each
replacing a legacy kernel already ported and gated, each oracled against the engine's OWN native body
(`src/kernels/cuda/native_gdn_preprocess.cu`), each MEASURED against that legacy kernel at the same shape on the
same device, with **`native_gdn_enabled()` left answering FALSE** (the flag gates six more unported symbols) and
`case_native_capabilities` extended to enforce it.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc / box XTX) | falsified by |
|---|---|---|---|
| `native_gdn_conv_silu` (3 arms C=2560/24/300) | `native_gdn_preprocess.cu` `conv_silu`: the four-tap conv **PLUS the SiLU in ONE kernel**, BOTH outputs (raw + SiLU), the zero-bias fold, the slide | **12800/12800 w 1.87e-06** (Arc), 12800/12800 w **3.39e-05** (XTX), 120/120, 1500/1500; slid state **BIT-EXACT**; terms-derived bound | `native-gdn-conv-silu-drop-silu` → FAIL 11509/12800 |
| `native_gdn_l2_norm` (3 arms cols=128) | `native_gdn_preprocess.cu` `l2_norm`: f32 sums, `rsqrtf(partial/S + eps/S)` (eps on the MEAN), folded `scale_after=1/sqrt(S)` | **392/392 w 1.54e-07**, 2312/2312 w 2.16e-07, 648/648 w 1.17e-07 (Arc); XTX 1.17e-07/1.49e-07/1.74e-07; NaN row-guard | `native-gdn-l2-norm-drop-folded-scale` → FAIL 264/392 w 10.3 |
| `native_gdn_beta_gate` (48 heads) | `native_gdn_preprocess.cu` `beta_sigmoid`: `1/(1+expf(-x))` — the SAME expression as the legacy `sigmoid_f` | **48/48 w 1.42e-06** (both boxes); raw identity checked host-side to move | `native-gdn-beta-gate-sign-flip` → FAIL 0/48 w 7.2e+10 |
| `native capabilities: gdn flag` | `native_gdn_enabled()` == "every gated symbol has a built shader"; the 3 ported shaders exist | **4/4** — flag **FALSE** (6 unported gated shaders absent) | `native-caps-gdn-true` → FAIL 3/4 |

**THE MEASUREMENT (native/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
`native_gdn_conv_silu` **0.938** Arc / **0.909** iGPU / **0.694** XTX / 0.936 K620 (a win per dispatch even
though it does the SiLU and a second output too); against the legacy **chain** (`gdn_conv_step`+`silu_f32`, the
two dispatches the layer actually runs) **0.574 / 0.764 / 0.544 / 0.720** — a 1.3–2.1x win.
`native_gdn_l2_norm` **0.997 / 1.013 / 0.938 / 0.942** and `native_gdn_beta_gate` **1.011 / 1.045 / 1.068 /
0.865** are **WASHES: the same work per element as the legacy kernel, no algorithmic difference to win, and
reported as findings rather than tuned away.**  A native kernel is not required to be faster.

**CAPABILITY CONTRACT.**  `native_gdn_enabled()` gates NINE symbols: the three ported here + `native_gdn_gate`
/ `native_gdn_step` / `native_gdn_out_norm` and the three `fused_gdn_*` paths (the latter also gated on
`g_fused_gdn` + `native_bf16_projections`) — six with no shader.  The backend
(`vulkan/src/kernels/native_caps_vk.cpp`) answers **false**, and `case_native_capabilities` gains a gdn arm
asserting the flag equals "every gated symbol has a built shader" AND that the three ported shaders exist —
an invariant, not a hard-coded boolean.

**THE MAP MOVES BY THREE ROWS:** `168 — 66 kernel, 61 host, 41 todo` → **`168 — 69 kernel, 61 host, 38 todo`**;
`check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **Gate, after
the commit: vega Arc 415/0/0, lvp 403/0/3, radeon-iGPU 406/0/2 (exit 0); box `z820b` RADV XTX 411/0/1,
lvp 403/0/3, nvidia K620 406/0/2 — 0 failed on every arm on both boxes** (+8 verdicts on each).  One finding
the case produced: the first run read 12798/12800 on the iGPU at worst 1.09e-05 because two four-tap sums
CANCEL — fixed in the FIXTURE's bound (`gemv_bound`, the port's "bound a reduction by its TERMS" rule), not by
loosening a tolerance.  Full detail in `NEXT.md`'s top section.

## THE PERFORMANCE TIER'S FIRST FOUR KERNELS — class B, the NATIVE fast paths (2026-10-05)

The first increment of the PERFORMANCE tier: the **four class-B capability-gated native fast paths** are
PORTED, GATED, and **MEASURED against the legacy kernel each replaces at the same shape on the same device**,
with the Vulkan backend's **capability answers wired so the engine can take the native branch**.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `native_rope_apply` (3 arms: hd=256/128 none, hd=256 YaRN×2) | `native_rope.cu` `apply<false>`: device f32 angle (`powf`/`rope_scaled_angle`), one thread per (row,PAIR), NEOX pairing, tail copied | **49152/49152 worst 2.61e-05**, 8192/8192 w 2.77e-05, 24576/24576 w 3.16e-05 (row-relative vs a host transcription, tol 3e-3; tail BIT-EXACT) | `native-rope-adjacent-pairing` → FAIL 37307/49152 w 1.96 |
| `native_router_top10` (5 arms) | `native_router.cu` `route`: softmax+sum in plain **f32**, top-10 with ties by lowest index, renormalise by the ten's sum (clamp 2⁻¹⁴) | **10/10, 80/80, 40/40, 40/40, 40/40** ids exact; weights worst 2.28e-07 (rel 1e-5) vs `router_top10_parity`'s double rule | `native-router-top10-tie-high-index` → FAIL 0/40 |
| `native_moe_combine` (4 shapes × shared on/off) | `native_moe.cu` `combine`: pure f32, first term a PRODUCT, shared added PLAIN | all 2560/2560 (and 2500, 37) within the term-relative bound; **904-1347/2560 differ from the double rule** — the measured arithmetic change | `native-moe-combine-drop-shared` → FAIL 0/2560 w 5.21e+05 |
| `native_qsa_rms_norm_weighted` (3 shapes × in-place/out-of-place) | `native_qsa.cu` `norm`: block-per-row, `scale*x*gamma`; in-place is the engine's call | **25608/25608 w 2.21e-07**, 1032/1032, 5128/5128 (double rule, tol 3e-3 + row-guard NaN) | `native-qsa-rms-norm-eps-on-sum` → FAIL 5128/25608 w 0.98 |
| `native capabilities (vulkan backend)` | the four capability answers + each ported symbol's `.spv` present | **4/4** answers exact | `native-caps-qsa-true` → FAIL 2/4 |

**THE MEASUREMENT (native/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
`native_rope_apply` **0.301** on the Arc, **0.118** on the box XTX and 0.040 on the Ryzen iGPU (where the legacy
one-thread-per-row kernel launches 2 workgroups); `native_router_top10` **0.078** (Arc), **0.082** (XTX), 0.078
(iGPU); `native_moe_combine` **0.998** (a wash); `native_qsa_rms_norm_weighted` **1.007** on the Arc, 0.988 on
the XTX, and **1.117 on the iGPU / 1.795 on the K620**.  **The last two are NOT wins and are reported as
findings, not tuned away** — they are elementwise / same-shape kernels with no algorithmic difference, and on
the K620 the native block-per-row tree's shared-memory traffic actively hurts.

**THE CAPABILITY CONTRACT, wired in `vulkan/src/kernels/native_caps_vk.cpp` (new).**  The Vulkan backend answers
the checks itself (a Vulkan build compiles none of `src/kernels/cuda/native_*.cu`): `native_rope_enabled()`,
`native_router_enabled()`, `native_moe_combine_enabled()` → **true** (each gates only a ported, forward-path
symbol); **`native_qsa_enabled()` → false**, because that ONE flag ALSO gates the UNPORTED
`native_qsa_gate_apply` (layer.cpp:1010, the main QSA path, 12 of 48 layers) — so the port stays on the legacy
branch until that sibling lands.  **A symbol-at-a-time truth, not a blanket `true`.**  Check it with the gate's
`case_native_capabilities` (asserts the four answers AND the built shaders) or by reading that TU; the
`native_*_set_enabled` setters are no-ops here so a `--native` launch cannot select an unported symbol.

**THE MAP MOVES BY FOUR ROWS:** `168 — 62 kernel, 61 host, 45 todo` → **`168 — 66 kernel, 61 host, 41 todo`**;
`check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **Gate, after
the last commit: vega Arc 407/0/0, lvp 395/0/3, radeon-iGPU 397/1/2 (the known `budget` flake); box `z820b`
RADV XTX 403/0/1, lvp 395/0/3, nvidia K620 398/0/2 — 0 failed on every arm on both boxes.**  Two README fixes
came with it: `run_gate.sh`'s greedy barrier-count `sed` misread 11 barriers as 1 (now `grep -oE`), and the new
cases had to be APPENDED to `main()` because the gate's one shared RNG makes a new earlier case change a later
existing case's fixture.  Full detail in `NEXT.md`'s top section.

## THE PERFORMANCE TIER'S HARNESS LANDED (2026-10-05)

The port had **no throughput harness at all**, so no performance claim could be made or checked.  This
increment adds one — **`ports/vulkan/bench/`** (`vk_bench.cpp` + `run_bench.sh` + `README.md`) — and hands
back the baseline the native/fused kernels will be judged against.  It is **separate and opt-in**: it does not
change what `gates/run_gate.sh` does, and the numeric gate was re-run green on both boxes after the commit
(see `NEXT.md`'s top section for the totals line).

**Method (full text in `bench/README.md`):** each kernel is timed as a batch of `K` dispatches recorded into
one command buffer (compute→compute barrier between dispatches), warmed 2–3 replays, timed over `reps`
replays of the batch (median per-dispatch, min/max printed, `reps=9`, sampler 5).  Synchronisation is **wall
clock around the fence** — the device layer exposes no timestamp path — so the number includes one
submit+fence per batch and its floor is ~5–15 µs on the fast GPUs; only the **kernel dispatch** is timed
(inputs resident, no transfer in the loop).  `run_bench.sh` compiles the measured kernels from source and runs
the binary under **every ICD** that reports a device.

**Baseline, medians (reps=9; sampler 5):**

| kernel | vega Arc B70 | vega Ryzen iGPU | box XTX | box K620 | llvmpipe |
|---|---:|---:|---:|---:|---:|
| `gdn_step` (S=128 h_v=48) | 0.0469 ms / 50.3 GMAC/s | 0.5313 | 0.1050 / 22.5 GMAC/s | 0.4859 | 0.5015 |
| `iq2s_mmvq` n_out=2048 | 0.0464 ms / 112.9 GMAC/s | 0.9762 | 0.0198 / **264.3 GMAC/s** | 0.6887 | 6.2754 |
| `iq_dequant_f32` BF16 | 0.0130 ms | 0.0195 | 0.0140 | 0.0391 | 0.1444 |
| `quantize_q8_K` (65536) | 0.0875 ms | 0.2039 | 0.0732 | 0.2394 | 0.1938 |
| `sampler_kernel_f32` (vocab 248320) | **546.9 ms** | 299.6 | **202.0 ms** | 353.8 | 1164.7 |

The GDN chain's other five kernels are 0.0026–0.030 ms everywhere; the Q8_0/ Q8_1 quantisers and the other
IQ dequantiser arms are in `bench/README.md`.  **The baseline names a real problem:** the one-block sampler is
**202 ms/token on the XTX and 547 ms on the Arc**, because its top-k is `k` full-vocabulary rounds with an
inner loop over taken ids and history (~ `k · vocab · (k + history)`).  It is the engine's *non-default* path
(the default is the split sampler, `sampler_split.comp`, not yet measured here) and it is the performance
tier's first target.

**Distinguishability, measured:** cross-ICD ordering is physical on both boxes with ratios far outside noise
(box `iq2s_mmvq` XTX vs llvmpipe = **1201×**; vega Arc vs llvmpipe = **135×**); the 4×-work sizing arm scales
≈4× where the work exceeds the timer floor (Ryzen iGPU `iq2s` **3.90×**) and reads ~1× on the fastest GPUs
where it does not — stated as the honest limit of a fence-clock timer, not hidden.

**One shared-file change, and why it is safe:** `harness/vk_compute.cpp`'s `Ctx::set_alloc` now treats
`VK_ERROR_FRAGMENTED_POOL` as pool exhaustion too (Mesa 26.0.8's code for it, vs Mesa 25.2.8's
`OUT_OF_POOL_MEMORY`), so the grow-on-demand path that the header promises fires on both.  The numeric gate
never fills a descriptor pool, so no gate verdict depends on it; the gate was re-run to prove that.

## CLASS A IS CLOSED, and M-A IS RE-DEFINED (2026-10-05)

The last three class-A forward-path kernels are LANDED and gated: **`indexer_key_append`** (the QSA indexer
pair's legacy member; contract `native_qsa_indexer_enabled() == false`), **`gr_write`** (the hyper-connection
write — reached on BOTH fused/unfused branches, so no dodge), and **`gr_read`** (the unfused five-stage
hyper-connection read: `gr_norm`/`gr_down`/`gr_gate`/`gr_mean`/`gr_inject`).  The `gr_read` / `fused_gr_read`
question left open by the triage is SETTLED — both were mis-kinded `host` (each launches kernels), so the honest
device-op count is **54, not 52** — and the GR pair is closed by CONTRACT (`gr_set_native_mmvf(false)` +
`layer_set_fused_gr(false)`, which selects the unfused `gr_read`+`gr_write` and removes `fused_gr_read`).

| case | rule | oracle | measured (vega Arc) | falsified by |
|---|---|---|---|---|
| `indexer_key_append` (2 arms 128/4/64, 32/4/8) | raw tail; on completion `pooled[b]=rope(rms_norm(mean), pos_base+b*r)`; spare `rope(rms_norm(raw[0]),0)` | `qsa_parity.cpp` indexer reference, double | **384/384 err/tol 4.94e-02** (bound 1.0), spare key worst 1.02e-07 (tol 1e-5) | `indexer-key-append-rotate-last` → FAIL 274/384 w 9.46e+05 |
| `gr_write` (3 arms 2560/4, 64/3, 16/2) | `out[i]=R[i]+block_out[d]·2·sigmoid(inject[c]/hc)`, in place; zero inject ⇒ `w=1` EXACTLY | `gr_parity.cpp` §6 + the rule in double | **20480/20480 err/tol 2.4e-01**, zero-inject property EXACT | `gr-write-drop-two-centring` → FAIL w 6.68e+06 |
| `gr_read` (3 arms 2560/4/320, 64/2/8, 16/3/4) | 5 stages: per-stream RMSNorm → down+silu(`/hc` inside) → gate(sigmoid) → mean over streams → inject | `gr_parity.cpp::reference`, double, BF16 activation | **23364/23364 err/tol 3.0e-01**; xq==bf16(xn) exact | `gr-read-mean-vs-sum` → FAIL 20804/23364 w 2.99e+03 |

**HONEST PARITY GAPS (measured):** `indexer_key_append`'s CUDA reduces the sum of squares in double; the target
has no `shaderFloat64`, so the port accumulates in f32 and the case MEASURES the gap (worst err/tol 4.94e-02 on
the Arc / 7.87e-02 on lvp+radeon) — the spare key is therefore NOT bit-exact here, unlike the CUDA's.
`gr_read`'s stage oracles are fed the DEVICE's own input for that stage and bounded by the stage's TERMS
(`rel·|want| + 16·2^-24·Σ|terms|`), because comparing a bf16-ROUNDED `lo` against an unrounded double oracle
puts the odd element a whole bf16 ulp away and moved a near-zero `mixed` by 9.7 RELATIVE on RADV.

**THE MILESTONE, RE-DEFINED.**  `todo = 0` is neither achievable nor meaningful: the map covers every
kernels-namespace symbol the decode path reaches, including the `native_*` siblings and the verify/MTP/tooling
helpers.  **M-A (re-defined):** every symbol the forward path reaches ON THE BRANCH THE CAPABILITY CONTRACT
SELECTS has a shader and a gated case.  **Class-A remaining: 0.  MET: YES**, with two stated soft edges — the
shipped `--spec 4` loop's MTP symbols (class D by the brief's definition: **a judgement, not a measurement**),
and the `gr_read`-vs-`fused_gr_read` branch choice.  Full statement: `plan/DECODE-PATH-TRIAGE.md` → "THE
RE-DEFINED MILESTONE M-A"; the batch's detail: `NEXT.md`'s top section.

**The map moves by four rows:** `168 — 59 kernel, 63 host, 46 todo` → **`168 — 62 kernel, 61 host, 45 todo`**
(the 45 = `11 capability-off + 4 B + 8 C + 22 D`); `check_port_map.py` passes, `make_port_map.py` regenerates it
byte-identically.  **Gate: vega Arc 384/0/0, lvp 372/0/3, radeon 375/0/2, exit 0.  Box `z820b`: radeon (XTX)
380/0/1, lvp 372/0/3, nvidia 375/0/2 — 0 failed on every arm; exit 1 only for the pre-existing M8 skip.**  (A
later full-gate run on vega read its radeon-iGPU arm 374/1/2 — the documented intermittent
`budget: independent requery agrees` flake, not this change.)

## THE GDN MIXER CHAIN IS COMPLETE, plus the first QSA gate member — class A #4-6 (2026-10-05)

This increment lands the next three class-A forward-path kernels: **`gdn_step`** (the delta-rule state update),
**`gdn_out_norm`** (the closing norm) — together **completing the GDN / DeltaNet mixer chain** for the 36 GDN
layers — and **`qsa_gate_apply_f32`**, the legacy member of the QSA gate pair. All under the branch policy
`native_gdn_enabled() == false` / `native_qsa_enabled() == false` (see `NEXT.md`'s top section for the full
reasoning and the per-symbol completeness check).

| case | rule | oracle | measured (vega Arc) | falsified by |
|---|---|---|---|---|
| `gdn_step` (3 arms 128/16/48, 16/4/8, 8/2/4) | `dec=exp(gate)`; decay, contract vs k, `d=(v−sk)·β`, rank-1 update, readout vs q; `src=h%h_k` | `gdn_parity.cpp` §1 `ref_step`, transcribed to the DEVICE layout (S,h_v,S) in double | **792576/792576 err/tol 0.00733**, 2176/2176 w 0.00112, 288/288 w 0.000345 | `gdn-step-head-pairing` → FAIL 66887/792576 w 6.13e+04 |
| `gdn_out_norm` (3 arms 48/128, 4/16, 3/8) | `y = rms_norm(o)·ssm_norm·sigmoid(z)`, ONE RMS per head (`sum/S + eps`) | `gdn_parity.cpp` §4, double | **6400/6400 w 8.2e-07**, 96/96 w 2.41e-07, 40/40 w 2.18e-07 | `gdn-out-norm-eps-on-sum` → FAIL 2321/6400 w 0.912 |
| `qsa_gate_apply_f32` (3 arms 24/256, 4/12, 2/8) | `attn · sigmoid(q_full[h·2·head_dim + head_dim + d])` (SECOND half) | `qsa_parity.cpp` `ref_gate`, double | **6152/6152 w 7.88e-07**, 56/56 w 7.64e-07, 24/24 w 7.42e-07 | `qsa-gate-first-half` → FAIL 8/6152 w 4.18e+10 |

Each case pins its rule's named traps host-side (modulo-vs-interleave head pairing and decay-order for
`gdn_step`; sigmoid-vs-SiLU and eps-on-the-mean for `gdn_out_norm`; second-half and sigmoid-vs-SiLU for
`qsa_gate_apply_f32`), and a NaN-padded tail + surplus dispatched groups make a missing guard DETECTED.
**HONEST PARITY GAPS (measured, not hidden):** the target device has no `shaderFloat64`, so `gdn_out_norm`
sums `o²` in **f32** where the CUDA sums in double (measured gap 8.2e-07), and `qsa_gate_apply_f32` forms the
product and the sigmoid in **f32** where the CUDA uses double (gap 7.88e-07). `gdn_step`'s CUDA already
accumulates in f32 (as the state is f32); its gap is against the double oracle only.

**The map dropped by three:** `PORT-MAP.tsv` `168 — 56 kernel, 63 host, 49 todo` → **`168 — 59 kernel, 63 host,
46 todo`**; `check_port_map.py` passes and `make_port_map.py` regenerates it byte-identically.
**Measured: vega Arc 376/0/0, llvmpipe 364/0/3, radeon-iGPU 367/0/2, exit 0; box XTX 372/0/1, lvp 364/0/3,
nvidia K620 367/0/2, exit 1 (the M8 skip)** — +9 verdicts on every arm. Full detail in `NEXT.md`'s top section.

## THE GDN (DeltaNet) MIXER'S FIRST THREE KERNELS LANDED — class A of the decode-path triage (2026-10-05)

The corrected map's **19 class-A forward-path holes** are dominated by the **GDN / DeltaNet mixer**, which runs
on 36 of the model's 48 layers and which the port's plan never enumerated. This increment lands the FIRST THREE
kernels of the mixer's chain, in the order `gdn_layer` runs them: `gdn_conv_step` (the four-tap conv),
`gdn_l2_norm` (the L2 norm, eps on the SQUARED NORM) and `gdn_beta_gate` (`beta = sigmoid(beta)`).

**THE BRANCH POLICY, and the contract on the backend.** The class-A GDN symbols are native/legacy PAIRS
(`if (native_gdn_enabled()) native_gdn_X else gdn_X`); the port implements the **LEGACY** member of each and
requires the Vulkan backend to answer **`native_gdn_enabled() == false`**. That one answer makes ONE
implementation per pair sufficient: the layer takes every `else`, and it also removes the three fused GDN paths
(`fused_gdn_conv_l2` / `fused_gdn_ab` / `fused_gdn_step_norm`, gated on `native_gdn_enabled()`) from the path
entirely. **Re-checked the parent's count:** the 19 symbols are **15 native/legacy pair members (11 GDN + 4
QSA) + 3 fused + 1 unconditional (`gr_write`)** = 19; collapsed under the policy they need **9 kernel
implementations** (6 GDN legacy — one of them, `gdn_gate`, already ported —, 2 QSA, `gr_write`), so **8 are
still to write**. The `layer_verify_compatible()` contract: forcing these flags off makes the P6 verifier refuse
to init (speculative verification off, class D); a `--spec 0` run is unaffected.

**The three kernels and their proofs** (shaders `gdn_conv_step.comp`, `gdn_l2_norm.comp`, `gdn_beta_gate.comp`;
cases `case_gdn_conv_step` / `case_gdn_l2_norm` / `case_gdn_beta_gate`; oracles transcribed from the engine's
own rule — `src/kernels/cuda/gdn.cu` plus `src/kernels/gdn_parity.cpp` — not invented):

| case | rule | measured | falsified by |
|---|---|---|---|
| `gdn_conv_step` (3 arms) | `out[c]=Σ conv_state·kW + x·kW`, state slides oldest→newest | **96/96 w 7.09e-08**, 20/20 w 0, 1200/1200 w 3.62e-06 | `gdn-conv-tap-order` → FAIL 72/96 w 2.28 |
| `gdn_l2_norm` (3 arms) | `x *= 1/sqrt(Σx² + eps)` (eps on the **squared norm**) | **392/392 w 0**, 2312/2312 w 1.66e-07, 648/648 w 1.56e-07 | `gdn-l2-norm-eps-on-mean` → FAIL 265/392 w 0.446 |
| `gdn_beta_gate` | `beta = 1/(1+exp(-beta))` in place | **48/48 w 1.42e-06** | `gdn-beta-gate-drop-sigmoid` → FAIL 0/48 w 1.8e+12 |

The conv case compares the slid state **bit for bit** (the kernel only moves/appends values) and its fixture
labels the state rows so the slide direction is observable; the l2 case's oracle is double and the shader sums
in **f32** (the target has no shaderFloat64), so the printed `worst` IS the measured gap — an honest limit, not a
bit-exactness claim.

**The map dropped by three:** `PORT-MAP.tsv` `168 — 53 kernel, 63 host, 52 todo` → **`168 — 56 kernel, 63 host,
49 todo`**; `check_port_map.py` passes and `make_port_map.py` regenerates it byte-identically.
**Measured: vega Arc 367/0/0, llvmpipe 355/0/3, radeon-iGPU 357/1/2** (+7 verdicts on each; the 1 is the known
intermittent `budget: independent requery` flake). Full detail in `NEXT.md`'s top section.

## THE PORT MAP'S BLIND SPOT IS CLOSED — and it was 91 symbols, not 4 (2026-10-05)

`tools/check_port_map.py` keyed on the `kernels::` qualifier, so kernels-namespace symbols that
`src/core/` calls BARE (through `using namespace strata::kernels;`) were invisible — the map could
read `todo 0` while such a symbol was unported.  The checker now discovers both forms, via one shared
`tools/port_map_lib.py` (bare names come from the engine's own declarations in
`include/strata/kernels/**`, attributed only to sources that have the namespace in scope).  **The
map moves from `77 symbols — 28 kernel, 49 host, 0 todo` to `168 symbols — 53 kernel, 63 host,
52 todo`.**  The four known symbols (`coupled_draft_sample`, `coupled_draft_stage` and their shaders
`coupled_penalize`/`coupled_sample`) are classified — their shaders are now claimed, so the unclaimed
shader count falls 37 → 19 — but the same scan surfaces **52 GPU symbols this port has NOT done**
(the GDN family and their `native_*`/`fused_*` siblings, `native_rope_apply`, `native_router_top10`,
`native_moe_combine`, `native_qsa_*`, `bf16_gemv*`, the QSA prompt/indexer path, `gr_write`,
`fused_gr_read_multi`, `moe_group_resident`, and the verify/P6 device helpers).  **So M-A's `todo = 0`
was measured on a 77-symbol map, not on the decode path**; the dated "port map ... 0 todo" lines
lower down are records of the map as it then stood, not the current count.  Full detail, the rule and
the falsification (the HEAD checker passes the same file the new one fails) are in `NEXT.md`'s top
section.  **M-A is therefore NOT closed**, and the corrected map's 52 `todo` rows are triaged per symbol in
`plan/DECODE-PATH-TRIAGE.md`: **19 are genuine forward-path holes with no ported fallback** (the GDN / DeltaNet
mixer for 36 of the 48 layers, the QSA gate and indexer for the 12 QSA layers, and `gr_write`), 4 are
capability-gated with a ported fallback, 7 are the non-native configuration the shipped `--native` launch does
not select, and 22 are the P6-verifier / speculative-drafter / tooling path.

## The engine integration has STARTED: increment I1 (device layer + arena + the first entry point) — DONE AND VERIFIED 2026-10-05

`ports/vulkan/plan/BACKEND-INTEGRATION.md` §3's **I1**, the first increment that BUILDS the engine.  It adopts
`ports/vulkan/harness/vk_compute.*` as `vulkan/src/device/` (namespace renamed `portvk` -> `strata::vulkan` so one
TU can hold the port's device layer and the engine's without an ODR clash), builds **THE ARENA** (one device-local
buffer carved by byte offsets; a device pointer is `kArenaBase + offset` in a synthetic address space, which is
what an unmappable VRAM arena can offer) and the **pointer -> buffer resolution** `arena_resolve` (base subtract,
live-range check, `view(arena, offset)` — a pointer outside the arena is refused, not bound), and finishes
`strata::vulkan::fwht256` so increment 0's entry point RUNS (pipeline cache keyed as the engine dispatches: 2
storage buffers + a 4-byte push constant).  **How the engine's sources join the build:** `vulkan/CMakeLists.txt`
gained `strata_vulkan_resolve()`, mirroring `../sycl/CMakeLists.txt`'s `strata_resolve` — a source comes from
`vulkan/` when a migrated copy is there, else from the engine tree.  I1 needs no engine-root source (the wrapper is
a header) and the top-level `return()` stands; the first engine-root sources are I2's.  The target that proves the
wiring, `strata_vk_entry_smoke`, builds only under `-DSTRATA_ENABLE_VULKAN=ON` (measured **configure 0.14 s, build
1.37 s**, 9 objects) and runs PASS on the Arc.  A full engine build was not needed and was not done — the CUDA
configuration never compiles `vulkan/`.

**The proof, and it is the first WRAPPER this port has gated rather than a shader.**  `case_fwht256_entry` runs the
same input through the port's shader path and through the ENGINE wrapper (`strata::kernels::fwht256_cuda`) on the
backend's arena, and compares as 32-bit words: **PASS 1024/1024, worst 0**.  Falsified by
`gates/inject-verify.sh fwht-entry-wrong-view-offset` (shift the view the DISPATCH binds by one float) ->
`FAIL ... 128/ 1024 worst 896`.  Recorded honestly: a UNIFORM shift inside `arena_resolve` does NOT bite (transfer
and dispatch move together and cancel) — the dispatch's offset is the one that decides.

**Gate totals: vega Arc 360/0/0, llvmpipe 348/0/3, radeon-iGPU 351/0/2, exit 0, wall 56.03 s.  Box `z820b`
post-commit: default XTX 356/0/1 (the 1 is the pre-existing M8 `prefill split` skip), lvp 348/0/3, nvidia
351/0/2, exit 1 (the skip).  The first box run read 355/1/1: `budget: independent requery agrees` failed on the
XTX — the KNOWN intermittent flake (that same run's named `radeon_icd` arm read 356/0/1 clean, and the re-run
cleared it), so 0 failed is the result.  New case 1024/1024 bitwise on every box arm.  Port map **77
decode-path symbols — 28 kernel, 49 host, 0 todo** - that `0 todo` was read on the qualifier-only map; CORRECTED
2026-10-05 to **168 — 53 kernel, 63 host, 52 todo**, and **M-A is not closed** (see the top section).**

## Done and verified in this session

Everything below is backed by a command that exits non-zero on failure. Re-run it with:

    bash ports/vulkan/gates/run_gate.sh          # compiles the shaders from source, validates the SPIR-V,
                                                 # checks each shader's declared local size, then runs the gate

**Result: the gate prints its own totals and those are the authority. On 2026-10-04, after the Radeon RX 7900 XTX
was swapped for an Arc Pro B70 and after stages 3, 4, the prefill GEMM, the quantised multi-token arms, the
short-step decode attention, the f16 KV gather, the QSA selection, the f16 KV append, the Q4_0 KV path (with its
Walsh-Hadamard rotation), the hybrid K8V4 mode, descriptor OFFSETS and the first SAMPLER kernel landed, the box's GPU
run was **284 passed / 0 failed / 0 skipped** on the Intel ICD (`Intel(R) Graphics (BMG G31)`, Mesa 25.2.8 / ANV,
Vulkan 1.4.318, subgroup size 32) - 272 / 0 / 3 on llvmpipe and 275 / 0 / 2 on the radeon ICD, which now picks the AMD
iGPU because the discrete card is gone (both
skip cooperative matrix, whose driver does not advertise the extension, and the prefill SPLIT, which needs the M8
tile).  **The Arc has no skips at all:** the last one (`gemm_coopmat`) was the port
misreading the device - BMG's matrix config is M8 N16 K16, not the M16 the criterion demanded - and since then the
matrix path RUNS on XMX, including the prefill GEMM.  The count now stands at **314 / 0 / 0 on the Intel ICD**
(302 / 0 / 3 on llvmpipe, 305 / 0 / 2 on the radeon iGPU) after the M-A 4/10 + 5/10 batch.  All three available
implementations are exercised by `run_gate.sh` (it once stopped at the Intel skip, so the cross-implementation arm
never ran on this box after the swap - `NEXT.md`), and the radeon iGPU's `budget: independent requery agrees` is
INTERMITTENT rather than deterministic: the driver's free figure drifts ~2.8 MB against the 1.7 MB tolerance (1
failure in 3 consecutive runs of one binary on one device; an older commit reproduces it).  Re-run it and record it;
do not chase it.  **75 kernels, 20 shared includes**, one generated table file (`harness/iq_grids.hpp`,
holding the IQ1_S, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_XXS and IQ3_S grids). TWO RECONCILIATION NOTES, both verified against a full run:
the ``PASS`` LINE COUNT IS ONE LESS than the case total, because the transcendental probe prints `INFO` while
counting as a pass; and one line ("gemm shape contract") covers seven cases. Neither is a discrepancy - but if the
numbers ever stop reconciling this way, something is wrong with the harness rather than with a kernel.

**M-A has started: the decode path's last kernels, in the order the decode path needs them (2026-10-04).**  The
ten `todo` symbols were ordered by (a) where the decode step meets them and (b) what each depends on.  The three
with NO unported precondition come first, in the order a decode step reaches them - `cvec_apply` (every layer,
`block_layer_post`), `gather_rows` (the MTP draft head's token subset), `scatter_rows_f32` (the peer experts'
write-back) - and **all three - `cvec_apply`, `gather_rows` and `scatter_rows_f32` - are landed** (3 of the ten).
`cvec_apply.comp` reproduces the
engine's OWN test
(`src/kernels/cvec_parity.cpp`) arm for arm: the projections at the engine's 1e-4 against a double oracle, the
add and BOTH untouched cases bitwise, and the pending write bitwise by construction (`inj == 0` makes
`2 sigmoid(0/hc) == 1.0` exactly, so the fold is `fl(bo + h)`).  Its SPIR-V carries the barrier tree's ops, so
`run_gate.sh`'s census list gained `cvec_apply`.  Falsified by `gates/inject-verify.sh cvec-apply-drop-scale`
(drop the per-layer factor `s`) -> `FAIL  cvec_apply: project removes s(h.v)v  12/14  worst 0.844`.  **`gather_rows`**
(the MTP draft head's token subset, `src/kernels/cuda/verify_kernels.cu:394`) is the second: a byte-level row
gather whose CUDA element WIDTH (`uint4`/`uint32`/`uint8` by `row_bytes % 16 / % 4`) is a performance choice, not
the rule - its fixture is a DERANGEMENT so an identity gather fails on every row, and both the 16-byte-aligned and
the unaligned `row_bytes` are gated.  Falsified by `gates/inject-verify.sh gather-rows-identity`
(`src[ids[r]*row_bytes + o]` -> `src[r*row_bytes + o]`) -> `FAIL  gather_rows: 16-byte-aligned rows  64/576`.
**`scatter_rows_f32`** (the peer experts' write-back, `src/kernels/cuda/elementwise.cu:263`) is the third: `r` is a
POSITION and `rows[r]` is the DESTINATION, an unnamed destination row must survive a sentinel, and the CUDA's
`width % 4 == 0` + 16-byte-alignment precondition is its `float4` cast rather than the rule - the case GATES that
by running `width = 6`, where the CUDA would refuse.  Falsified by `gates/inject-verify.sh scatter-rows-identity`
-> `FAIL  scatter_rows_f32: permutation ... 511/3072`.  **Arc 284/0/0** (272 + the batch's twelve verdicts, and
`run_gate.sh` **exit 0**); box `radeon_icd 280/0/1`, lvp 272/0/3, nvidia 275/0/2.  **M-A 4/10 and 5/10 landed the
STANDALONE IQ/BF16 dequantiser and the token-embedding gather on it** (`iq_dequant_f32`, `iq_embed_rows`;
`shaders/common/iq_dequant.glsl`, `shaders/iq_dequant_f32.comp`, `shaders/iq_embed_rows.comp`).  The decoder is the
engine's own `dq_dispatch<float>` - the port keeps its `tid` 0..31 thread mapping rather than re-deriving a block
layout - and the case reproduces each `dq_*` body on the host as the oracle, ONE ARM PER FORMAT over the **14
formats** the port's generated grids cover (BF16, IQ4_NL, IQ4_XS, Q8_0, Q5_0, Q5_1, Q2_0, Q4_K, Q5_K, Q3_K,
IQ3_XXS, IQ3_S, IQ2_S, IQ1_M); **IQ2_XXS (16) and IQ2_XS (17) stay `todo`** - their grids are not in
`harness/iq_grids.hpp` and neither shader claims them.  `iq_embed_rows` gates the row stride and a DERANGEMENT
token list, so an identity gather fails every token.  Falsified by `gates/inject-verify.sh
iq-dequant-iq1m-grid-high` -> `FAIL  iq_dequant_f32: IQ1_M  310/768  worst 2.34e+05` and `iq-embed-rows-identity`
-> `FAIL  iq_embed_rows: BF16 ... 0/3072  worst 6.58e+04`; the first targets a shared INCLUDE, so the script now
compiles the shader that INCLUDES a changed `common/` file rather than reporting `DID NOT COMPILE` on a file with
no `#version`.  **+30 green verdicts on every implementation, bit-exact against the oracle**: vega **Arc 314/0/0**
(`run_gate.sh` exit 0), llvmpipe 302/0/3, radeon-iGPU 305/0/2; box `radeon_icd` (7900 XTX) **310/0/1**, lvp
302/0/3, nvidia 305/0/2.  **M-A is 5 of the ten**; the five `todo` symbols are `native_q5_k_f32`, `moe_grouped_s2`,
`moe_hit_add`, `moe_hit_select`, `moe_hit_grouped_s2`.

**M-A 6/10 landed the NATIVE HEAD's Q5_K matvec** (`native_q5_k_f32.comp`, from `native_q5_k_mmvq_kernel` /
`q5_q8_dot`, `src/kernels/cuda/native_mmvq.cu`).  It is the one dot in the port whose scale and min are PACKED:
Q5_K's 12-byte `scales` carries six 6-bit scales and six 6-bit mins, and an `hi` mask decides which six bits are
which - the port transcribes the source's masks and shifts verbatim, because a "tidier" read is a plausible wrong
number.  The case carries three arms (`n_in=2560` 1280 parts, `n_in=256` 16 parts, `n_in=10240` x2 columns)
against a host double reference, and the increment found THREE silent defects in the transcription, all recorded
in `NEXT.md`: the activation's `u` was read out of the weight buffer (the `f16_at` class), the per-block
activation group must be `kbx*8` not the row base, and `unpackHalf2x16` fed only the low 16 bits silently zeroed
`min` (a 1-13% per-row error - every value finite and close).  Falsified by `gates/inject-verify.sh
native-q5k-aux-half` (drop the packed-scale half switch) -> `FAIL  native_q5_k_f32 ... 0/8  worst 5.62e+04`.
**+3 verdicts on every implementation**: vega **Arc 317/0/0** (`run_gate.sh` exit 0), llvmpipe 305/0/3,
radeon-iGPU 307/1/2 (the intermittent budget-requery drift); box `radeon_icd` (7900 XTX) **313/0/1**, lvp
305/0/3, nvidia 308/0/2.  **M-A is now 6 of the ten**; the four `todo` symbols are `moe_grouped_s2`,
`moe_hit_add`, `moe_hit_select`, `moe_hit_grouped_s2`.

**M-A 7/10 + 8/10 landed the MoE hit selection and the per-hit S2 expert entry** (one commit, as 4/10 + 5/10
were).  `moe_hit_select.comp` (`hit_select_kernel`, `s2_expert_grouped.cu:623`, from `src/core/session.cpp:866`)
is the token graph's first step: only a resident expert is a hit, the hits compact in ascending routing order, and
`dst[at]` is the ROUTING POSITION (`lane`), not the slot - the same two-roles confusion `gather_rows` carries.  A
ballot is a subgroup op, so the port writes the same exact compaction SERIALLY from one lane, and the shader
therefore carries no barrier (it is deliberately NOT on the census whitelist).  Four arms (decode shape with
scrambled slots, `k=32`, `k=1`, a fully non-resident row), with sentinels past `count` and the `res` buffer padded
past `n_expert` so an unguarded read fails.  Falsified by `gates/inject-verify.sh moe-hit-select-residency` ->
`FAIL  moe_hit_select ... 21/32`.  **`moe_hit_grouped_s2` is a COMPOSITION** (`gu -> swiglu -> q8_0 -> down`,
`s2_expert_grouped.cu:579`), so it needed NO new shader: its increment is the case that gates the WIRING between
the four already-ported kernels, with an oracle that is independent of the device's quantiser and down projection
(the device's intermediate must DECODE to its own post-SwiGLU floats at the same position, and the up rows are
checked against the blob).  Falsified by `gates/inject-verify.sh moe-hit-grouped-s2-hit0-intermediate` ->
`FAIL  moe_hit_grouped_s2 ... 473/867  worst 1.3e+06`.  Its fixture first made the intermediate's `d16` inf (the
gu arm's gate scales are `*1000`), which turned EVERY down row into NaN that the oracle also computed - the case
counts a NaN output as a failure, so it did not pass over garbage; the chain now carries its own modest scales.
**vega: Arc 322/0/0** (`run_gate.sh` exit 0), llvmpipe 310/0/3, radeon-iGPU 313/0/2; box `radeon_icd` (7900 XTX)
**318/0/1**, lvp 310/0/3, nvidia 313/0/2.  **M-A is 8 of the ten**; the two `todo` symbols are `moe_grouped_s2`
and `moe_hit_add`.  The port map reads **77 symbols - 26 kernel, 49 host, 2 todo**.

**M-A's LAST TWO *VISIBLE* SYMBOLS landed together - but M-A is NOT closed (CORRECTED 2026-10-05, `7c317c4`).**
The "CLOSED" reading below was taken on the qualifier-only 77-symbol map, which could not see the bare-name call
sites; the corrected map reads **168 symbols - 53 kernel, 63 host, 52 todo**, of which **19 are class-A
forward-path holes** (`plan/DECODE-PATH-TRIAGE.md`).  The two symbols that landed here did land, and the increment
is real - it is the "*last two visible*" that is now the historical claim.  What follows is that record.
**the decode path's last two symbols landed together (9/10 + 10/10, one commit, sharing
`harness/vk_gate.cpp` - the `4/10 + 5/10` precedent).**  **`moe_grouped_s2`** turns out NOT to be a reuse of the
per-hit chain: it is the same four-launch composition, but its gu and down halves are the GROUPED kernels, and the
per-hit `s2expert_gu`/`s2expert_down` cannot express a per-GROUP blob or a per-ENTRY activation row - so the
increment adds `s2expert_gu_grouped.comp` and `s2expert_down_grouped.comp` (one weights buffer + a per-group byte
offset table replacing the source's `grp_ptr` device pointers) and reuses `s2expert_swiglu` and `quantize_q8_0`.
The case gates the WIRING with an oracle independent of the device's quantiser and down step (up rows against the
group's blob at the ENTRY's token; the device's intermediate decoded against the device's own post-SwiGLU floats;
down rows from the device's intermediate at the scrambled `ent_dst[e]`), over three grid.y arms including the
device-count stride and an EMPTY group; falsified by `moe-grouped-s2-entry-token` -> `FAIL ... 5270/5638 worst
3.78e+05`.  **`moe_hit_add`** (`add_hits_kernel`, `s2_expert_grouped.cu:666`) is `parts[dst[h]] += hit_out[dst[h]]`
for every LIVE hit - `dst[h]` is the ROUTING POSITION and the operator is `+=`; four arms (count<cap, count==cap,
`n_embd=37`, count 0), falsified by `moe-hit-add-accumulate` (`+=` -> `=`) -> `FAIL ... 2624/10304 worst 0`.  The
count-0 arm itself was the one defect found (a liveness guard that assumed something MOVED, which that contract
inverts); the guard now reads `moved == 0` for count 0 and the arm was not loosened.  The port map now reads
**77 symbols - 28 kernel, 49 host, 0 todo** - that `0 todo` was the qualifier-only map; CORRECTED 2026-10-05 to
**168 - 53 kernel, 63 host, 52 todo**, and the drifted generator `tools/make_port_map.py` was fixed to regenerate
it exactly (see the top section and `plan/DECODE-PATH-TRIAGE.md`).  **M-A is not closed.**  **Measured: vega Arc 329/0/0** (`run_gate.sh` exit 0), intel_icd 329/0/0, llvmpipe 317/0/3,
radeon-iGPU 320/0/2; box `radeon_icd` (7900 XTX) **325/0/1** (the pre-existing M8 `prefill split` skip), lvp
317/0/3, nvidia 320/0/2.  The two new symbols add 7 verdicts (322 -> 329 on the Arc).

**The standalone dequantiser's LAST TWO FORMATS landed - IQ2_XXS (ggml type 16) and IQ2_XS (17) (2026-10-05).**
Those two `is_iq` types were the ones `iq_dequant_f32` refused BY NAME because their `uint64` grids were not in the
generated table; both now decode.  `tools/gen-iq-tables.py` derives `iq2xxs_grid` (256 points) and `iq2xs_grid`
(512) from the engine's own `ggml-common.h` as low/high `uint32` pairs (so no `shaderInt64`), `harness/iq_grids.hpp`
carries them, `common/iq_dequant.glsl` gains the two `dq_*` bodies, and the harness gains ONE BIT-EXACT ARM PER
FORMAT against the engine's own decoder (768/768 bit-exact; the same decode is 1536/1536 through `iq_embed_rows`).
Falsified twice (`iq-dequant-iq2xxs-grid-index` -> `FAIL IQ2_XXS 300/768 worst 3.79e+05`; `iq-dequant-iq2xs-grid-high`
-> `FAIL IQ2_XS 530/768 worst 3.82e+05`).  A finding: the shared fixture left IQ2_XS's 9th grid bit
DETERMINISTICALLY clear, so the upper half of the 512-point grid was never indexed and the first injection was
INVISIBLE - the fixture now splits that bit across both halves.  **+4 verdicts on every implementation:** vega
**Arc 333/0/0**, llvmpipe 321/0/3, radeon-iGPU 323/0/2 (a clean run; one run had the documented intermittent
`budget: independent requery` failure); box `radeon_icd` (7900 XTX) **329/0/1**, lvp 321/0/3, nvidia 324/0/2.  The
port map still reads **77 symbols - 28 kernel, 49 host, 0 todo**, and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.

**THE SPLIT SAMPLER landed - the engine's DEFAULT sampled path (2026-10-05).**  `sampler_split.comp` is
`sampler_split_part_kernel` + `sampler_split_merge_kernel` (`src/kernels/cuda/sampler.cu`) with the `sampled_tail_warp`
tail: a row is cut into 4096-logit partitions, each keeps its own top_k, and the ordered lists are merged (exact).
This is the path `sample_tokens` takes by default for a sampled request; the one-block `sampler_kernel` is the
`STRATA_OLD_SAMPLER=1` reference.  The tail now lives ONCE in `common/sampler_tail.glsl`, in both arithmetic
variants (`sampler_tail_f64`, and the portable `sampler_tail_f32` for the device the f64 one cannot run on), and the
partition+merge selection in `common/sampler_select.glsl` - so the split and the coupled merge share them and the
parity is by construction.  Falsified by `gates/inject-verify.sh sampler-split-merge-drop-parts` -> `FAIL
sampler_split: three partitions ... 9/16`.  **+7 verdicts on every implementation, and the parity arm reads 60/60
(the split's token EQUALS `sampler_kernel`'s on every seed of every arm, the engine's own bit-for-bit claim): vega
Arc 340/0/0** (`run_gate.sh` exit 0), llvmpipe 328/0/3, radeon-iGPU 331/0/2.  The increment also FOUND a real class:
the merge's first version was a forward IN-PLACE pass, whose write index runs ahead of its read index once a
partition entry is taken - it read `dev 4200 want 100` on the top_p arm, and only the multi-partition arms could see
it.

**THE COUPLED DRAFT PATH (speculative decoding) landed (2026-10-05).**  `coupled_penalize.comp` +
`coupled_sample.comp` (`coupled_penalize_kernel` / `coupled_merge_kernel`, `src/kernels/cuda/sampler.cu`; host side
`include/strata/core/coupled_draft.hpp`).  In coupled mode the MTP draft layer samples its draft with the target's
own chain and the target's Philox draw; four rules are arms - the counter is `cell + 1` (NOT `cell`), the penalty
window is the ring's `[cap+j-h, cap+j)`, a history id maps through `id_to_sub`, and the pick maps through `sub_to_id`
and is appended at `ring[cap+j]`.  Falsified by `gates/inject-verify.sh coupled-draft-counter-off-by-one` -> `FAIL
coupled_draft: the counter ... 1/8` and `coupled-draft-window-start` -> `FAIL coupled_draft: the window ... 0/4`.
**+5 verdicts: vega Arc 345/0/0** (`run_gate.sh` exit 0), llvmpipe 333/0/3, radeon-iGPU 336/0/2; box `radeon_icd`
(7900 XTX) 341/0/1, lvp 333/0/3, nvidia 336/0/2.  Honest note: `coupled_draft_sample`/`coupled_draft_stage` are
called unqualified in `src/core/mtp.cpp`, so `check_port_map.py` (which keys on `kernels::`) neither lists them nor
the two coupled shaders - recorded, not papered over.

**THE `sample_tokens` CHOICE landed (2026-10-05).**  Both sampled paths and the greedy kernel were gated; the CHOICE
between them was not.  `case_sample_tokens` pins it (greedy || temp 0 -> GREEDY; n_blocks<=64 && n_tokens<=64 ->
SPLIT, the default; else -> the one-block ONEBLOCK fallback) as a pure predicate AND by running the chosen kernel
for six requests, including the task's named case: **temperature 0 routes to the ARGMAX, not the sampled path's
uniform draw**.  Falsified by `gates/inject-verify.sh sample-tokens-choice-temp0-to-sampled` -> `FAIL sample_tokens:
temperature 0 ... 3/4`; this injection is in the HARNESS, so `inject-verify.sh` now rebuilds the gate for a
non-shader change (and refused an earlier form as `DID NOT COMPILE (harness)`).  **+7 verdicts: vega Arc 352/0/0**
(`run_gate.sh` exit 0), llvmpipe 340/0/3, radeon-iGPU 343/0/2; box `radeon_icd` (7900 XTX) 348/0/1, lvp 340/0/3,
nvidia 343/0/2.

**THE SAMPLER'S PORTABLE f32 SIBLING landed (2026-10-05).**  `sampler_kernel_f32.comp` - the double-arithmetic
sampler's sibling for the devices WITHOUT shaderFloat64 (Intel's own article; the target hardware), which every
other double kernel in this port already had.  The chain and selection are the faithful kernel's; only the tail is
float, from `common/sampler_tail.glsl`'s `sampler_tail_f32`.  `run_gate.sh`'s fp64 rule now requires it to carry NO
`Float64` capability (measured 0).  Five arms: two EXACT where no rounding reaches it (one-survivor shortlists), one
EXACT because the softmax cancels (64 equal at temp 0, RNG pinned), two MEMBERSHIP on real distributions - and the
case MEASURES the gap: `f32 vs f64: 0 of 48 seeds differ`.  Falsified by `gates/inject-verify.sh
sampler-kernel-f32-top-p-boundary` -> `FAIL sampler_kernel_f32: one survivor (top_p cut of one) 2/8`.  **+6
verdicts: vega Arc 358/0/0** (`run_gate.sh` exit 0), llvmpipe 346/0/3, radeon-iGPU 349/0/2; box `radeon_icd` (7900
XTX) 354/0/1, lvp 346/0/3, nvidia 349/0/2.  Port map still 77 symbols - 28 kernel, 49 host, 0 todo, regenerating
identically.

**The 7900 XTX run's two failures are RESOLVED (2026-10-04).**  On `z820b` (RX 7900 XTX, RADV gfx1100, Mesa 26.0.8)
the gate now reads **`radeon_icd 280 passed / 0 failed / 1 skipped`**, `lvp_icd 272/0/3`, `nvidia_icd (K620)
275/0/2`.  Both failures were the cross-implementation arm earning its keep, and each went a different way: the
`kv_q4 round trip` failure was the CASE's bound (`|d|/2`, wrong for a `d*[-8,+7]` code range whose +8 end clips -
the shader was faithful, `dev-vs-rule mismatch 0`); the `quantize_q8_0 (ggml bytes)` failure was a lavapipe/Mesa
26.0.8 bug (`roundEven(double)` rounds ties toward zero), so the CASE was right and the SHADER was changed to an
explicit ties-to-even.  Full evidence and the two falsifying injections (`gates/inject-verify.sh`) are in
`NEXT.md`'s new top block and `HANDOFF.md` section 6.1.  The XTX's single skip is the pre-existing M8
cooperative-matrix case (the XTX's config is M16), not a failure.

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
point: a new call site cannot join the decode path unnoticed.  (It read 77 symbols - 17 kernel, 50 host, **10 todo**
when it landed; the count is now **168 - 53 kernel, 63 host, 52 todo**, because the scan was later extended to the
BARE-name call sites a `using namespace` makes legal - see the top section.)  Falsified three ways before landing.

**THE EMBEDDING GATHER LANDED, and `precise` turned out to be LOAD-BEARING.**  `embedding_gather.comp` (packed
codes + per-group scales -> float rows, from `verify_kernels.cu`'s `_dev` kernel and the parity reference).  The
finding generalises beyond this kernel: **a driver may FUSE a multiply and an add unless the result is `precise`.**
The SPIR-V carries no fma op, but GLSL's default permits contraction, so the backend can still fuse - and one fused op
rounds once where the engine's `__fmul_rn` + `__fadd_rn` round twice, which the engine compares bitwise against.
Measured: four of six arms failed before `precise`, and the two that passed were those whose products happen to be
exact - that signature is what identified the cause.  The shader qualifies the product, the offset and the sum; the
SPIR-V now carries six `NoContraction` decorations.  The fixture was built to SEE this: the host searches for
(code, scale, offset) triples where the fused and two-step forms differ and puts one at element 0, reporting the
count - the engine's own `fma_diff` datum.  Five injections all bite: contractable arithmetic, MSB-first codes, the
group as the element-within-group, the bias dropped, and the token array ignored.  The port map's hole list is now
ten kernels.

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
* **The engine integration**: **STARTED — I1 landed 2026-10-05** (`vulkan/` carries `STRATA_ENABLE_VULKAN` from
  increment 0; I1 added `vulkan/src/device/` — the adopted port device layer plus the arena, the pointer->buffer
  resolution and the pipeline cache — and one entry point, `strata::kernels::fwht256_cuda`, that RUNS and is gated
  bitwise against the port's shader path; see the section at the top). What is still NOT there: the engine's own
  sources under `STRATA_ENABLE_VULKAN` (I1's target is a smoke executable, not the engine program), the other 39
  entry points, and the recorded decode step IN THE ENGINE — so `src/` cannot yet drive the backend and there is
  still no `setup.py`/engine `CMakeLists` change. `harness/vk_compute.*` remains the seed from which
  `vulkan/src/device/` was adopted, and it carries the single-shot `dispatch()`, the recorded-step API (stage 3)
  and the engine-shaped memory path (stage 4), each verified by the gate.
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
