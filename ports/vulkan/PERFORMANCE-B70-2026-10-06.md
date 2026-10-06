# Measured performance on the Intel Arc Pro B70 — 2026-10-06

## THE DEQUANT PHASE IS NOT THE DEQUANT KERNEL — the idle-lane fix is worth ~7% of the kernel and 0% of the prefill, and the default GEMM arm is ~12% slower than coopmat (later, same day, `vega`, Arc Pro B70)

**THIS SECTION FALSIFIES THE PREMISE OF THE PREVIOUS ONE, WITH THE PORT'S OWN BENCH.** The previous section's
timeline put `dequant` at 2,054 ms (21%) and named `iq_dequant_f32` as the cost at "~0.55 ms per dispatch". The
kernel's own benchmark (`ports/vulkan/bench/`, which times a kernel in a stream of dispatches, not a phase of a
wall clock) says the kernel is **~0.30-0.60 s end to end for the whole prompt, ~3-5% of an 11.3 s prefill**:

| `iq_dequant_f32`, intel ICD, median of 9 reps | unmodified | idle lanes fixed | delta |
|---|---:|---:|---:|
| BF16, 256 superblocks | 0.0128 ms | 0.0127 ms | -0.8% |
| IQ4_NL, 256 | 0.0131 | 0.0128 | -2.3% |
| IQ2_S, 256 | 0.0146 | 0.0145 | -0.7% |
| IQ2_S, 1024 | 0.0157 | 0.0161 | +2.5% |
| **IQ2_S, 12,800 (one expert's projection)** | **0.0554** | **0.0518** | **-6.5%** |
| **IQ4_NL, 12,800 (the pack's down type)** | **0.0298** | **0.0272** | **-8.7%** |

The two small arms are **dispatch-bound** (4x the work costs 1.09x the time), so they cannot see an occupancy
change at all; the 12,800-superblock arms are the engine's own shape (`n_ff*n_embd/256 = 1280*2560/256`, three such
calls per expert = the 10,833 dispatches the histogram counts) and are what was added to the bench this batch. On
the other two implementations the same change is large - **llvmpipe 2.81x / 3.10x / 2.74x** and the **Ryzen iGPU
2.45x / 1.20x** - which is exactly the signature of idle lanes costing real work where lanes are real work.

**THE ENGINE, ONE BINARY AGAINST ITSELF, 199-token arm, `--spec 2 --prefill 256`, default (untiled) GEMM:**

| | prefill ms | tok/s | live dispatches | `dequant` phase | `gemm gate/up` phase | ids md5 |
|---|---:|---:|---:|---:|---:|---|
| HEAD (no change) | 11,429.9 | 17.32 | 56,468 | 28 ms | 4,041 ms | `56a0b28d2de6` |
| + the idle-lane fix | 11,534.4 | 17.17 | 56,468 | 143 ms | 4,015 ms | `56a0b28d2de6` |
| HEAD, `STRATA_VK_PREFILL_COOPMAT=1` | 10,046.7 | 19.71 | 59,640 | 2,059 ms | 591 ms | `56a0b28d2de6` |

**A wash, in the wrong direction, and the fix was reverted.** The one number worth keeping from that table is the
`dequant` phase itself: **the identical dequant work (10,833 dispatches, 3 per expert) is charged 28 ms on the
default-GEMM arm and 2,059 ms on the coopmat arm.** A phase that charges the same work 73x differently is a phase
that is measuring **where the host ran out of enqueue work and blocked on the fence** - the phase marks are host
wall-clock on this shim - not the kernel. On the default arm that waiting lands in `gemm gate/up` instead, which is
why the two arms' phase tables look inverted.

**TWO MORE NUMBERS RECONCILED, AND BOTH NAMED.** "~0.55 ms per dispatch" is `STRATA_VK_DISP_STAT`'s own
`ms/dispatch` for the SUBMISSION layer over a whole run (measured **0.6800 / 0.7377 / 0.7758** here), and **76-80%
of that total is `wait`** (33,246 of 43,807 ms on the clean arm); multiplying it by one kernel's dispatch count
double-counts every other kernel's GPU work. And the kernel's own traffic floor agrees with the bench: 10,833 x
13.1 MB of f32 output = **142 GB written** (+11 GB of packed reads) against a ~456 GB/s class card = **0.31 s**.
No mapped buffer is on that path any more on the Arc (the six IQ grids are `alloc_device`; the flush stat is 3,021,
down from 13,619).

**THE +3,172 LIVE DISPATCHES: THE COOPMAT GEMM SPLIT, NOT A REGRESSION.** `STRATA_VK_DISP_STAT=1` by shader, same
prompt, same binary family: the 59,640-dispatch runs are the ones with `gemm_prefill_f16_m8` **3,840** in the
histogram and `gemm_prefill_fma_small` 7,166; the 56,468-dispatch runs have **0** `f16_m8` and 7,702 `fma_small`.
`+3,840 - 536 = +3,304`, LESS `132` = the coopmat run's 22-fewer-expert routing difference (6 dispatches per
expert) = **exactly +3,172**. Coopmat has no ragged edge, so each GEMM is split into a CM part and an FMA
remainder - one GEMM becomes two dispatches. The reason the two counts were ever compared: **`STRATA_VK_PREFILL_COOPMAT=1` was inherited by the
"after" runs of the previous batch**, so its `untiled` / `tiled` label rows are coopmat rows.

**WHICH IS THE MOVED TARGET, AND IT IS ~100x LARGER THAN THE DEQUANT KERNEL.** With the environment controlled, the
default untiled GEMM arm measures **11,429.9 / 11,534.4 ms (17.32 / 17.17 tok/s, n=2)** against coopmat's
**9,895.8-10,375.3 ms (19.46-20.14 tok/s, n=5, from the earlier logs)** - **~12% in coopmat's favour**, with the
same ids. That needs a clean repeated A/B on one binary before the default moves; a default is a claim.

**GATE + IDS.** Arc `intel_icd` **886 passed / 0 failed / 0 skipped** (lvp `872/0/4`, radeon `876/1/2` - the one
failure is the documented `bf16_gemv_fp32_mmvf_cols` intermittent; smoke 60/0/0). Ids `56a0b28d2de6` (199-token) in
every run above and `3aed108cceee` (8-token). The tree diff this batch is `ports/vulkan/bench/vk_bench.cpp` only
(two measurement arms). Logs: `/tmp/meas2/bench2_{before,after}.log`, `/tmp/meas2/{before_def_199,deq8_199,base_hist_199}.log`,
`/tmp/meas2/gate_after.log`.

## THE EXPERT PATH'S `dequant` WAS A MAPPED GRID, NOT A KERNEL — 13,619 FLUSHES → 3,021, PREFILL 16.02 → 19.74 tok/s, AND THE TILED GEMM'S 1.49x WAS THAT FLUSH (later, same day, `vega`)

**THE MEASUREMENT THAT FOUND IT.** `STRATA_VK_FLUSH_STAT=1` on the 199-token arm at the merged HEAD printed
**13,619 live-batch flushes, 56,468 dispatches, submit 69 ms, wait 11,622 ms**. That instrument already dumps 20
flush CALL SITES as backtraces; they were unresolved addresses on the shipped binary, so `addr2line` was run over
the binary and NAME them:

```
site  0: n=3563  disp=17479  wait=5109ms   iq_dequant_gu_f16 <- Prefill::run_impl::{lambda}   (4.9 disp/flush)
site  1: n=102   disp=13056  wait=2403ms   native_gdn_out_norm <- prefill::gdn_recurrence     (128 disp/flush = the batch limit)
site  2: n=48    disp=4206   wait=2067ms   Prefill::run_impl
site  4: n=3563  disp=7126   wait=393ms    iq_dequant_f16    <- Prefill::run_impl::{lambda}
site  5: n=3563  disp=3563   wait=296ms    iq_dequant_gu_f16 <- Prefill::run_impl::{lambda}
```

The `iq_dequant` call sites carry **10,737 of the 13,619 flushes at ~4.9 dispatches each** — they ARE the flush.
The batch limit is 128 (`kLiveBatchMax`) and site 1 shows exactly 128 per flush, so the ordinary arena dispatches
never trip the rule. (The "~613 per prompt token" figure carried in this file earlier was the 8-TOKEN arm's 4,293
divided by its 7 prompt tokens; on the 199-token arm it is 13,619 / 198 = 68.8 per prompt token.)

**WHY IT FIRED.** `Ctx::dispatch` (`vulkan/src/device/vk_compute.cpp:1520`) flushes the live batch whenever any
bound buffer is MAPPED — the port's documented contract, "a dispatch touching a host-visible region completes when
`dispatch()` returns", because the engine and the gate read those regions directly. The buffers were the **IQ grid
tables**: `iq_vk.cpp`'s `iq_grids()` and `matvec_vk.cpp`'s `grid_for()` placed them with `Ctx::alloc`, whose memory
type is `mem_type_` — HOST_VISIBLE|HOST_COHERENT, preferring a device-local heap, i.e. on this card the **BAR-mapped
VRAM type**. `iq_dequant_f32.spv` binds all six grids, so every dequant dispatch was a mapped dispatch and flushed.

**THE CHANGE IS A NARROWING, NOT A WEAKENING.** Both sites now use `Ctx::alloc_device` (the device-local type). A
grid is a shader-READ constant the host writes ONCE through `Ctx::write` and never reads back, so the mapping
bought nothing and cost a submit+wait per dequant dispatch. The host-visible rule is untouched; on a device whose
only heap is device-local AND mappable (llvmpipe) `alloc_device` lands in that same type, so its behaviour cannot
change. **13,619 → 3,021 flushes** (submit 69 → 16 ms); the remaining 3,021 are the batch limit plus the host
`stream_write` sites (`Ctx::write`/`read` flush by contract).

**THE CHEAP TEST, RE-RUN FIRST — THE OLD REASON FOR LOSING IS DEAD, AND THE WIN IT SHOWED WAS THE FLUSH.** Same
card, same prompt, same flags, ids identical in every arm:

| 199-token arm | prefill ms | tok/s | decode tok/s | flushes | ids md5 |
|---|---:|---:|---:|---:|---|
| HEAD default (untiled) | 12,359.3 | 16.02 | 8.36 | 13,619 | `56a0b28d2de6` |
| HEAD `STRATA_VK_PREFILL_TILED=1` | 8,294.1 | 23.87 | 8.34 | 13,619 | `56a0b28d2de6` |
| HEAD `STRATA_VK_PREFILL_COOPMAT=1` | 10,989.3 | 18.02 | 8.35 | 13,551 | `56a0b28d2de6` |
| **+ the grid narrowing, untiled (n=2)** | **10,128.6 / 9,939.4** | **19.55 / 19.92** | 8.34 / 8.31 | **3,021** | `56a0b28d2de6` |
| + the grid narrowing, tiled (n=3) | 10,375.3 / 9,895.8 / 10,177.0 | 19.08 / 20.01 / 19.46 | 8.35 | 3,021 | `56a0b28d2de6` |
| + the grid narrowing, coopmat (n=1) | 9,829.9 | 20.14 | 8.35 | 3,021 | `56a0b28d2de6` |
| 8-token arm, shipped default | 1,056.4 | 6.63 | 8.22 | 381 | `3aed108cceee` |

**THE TILED PATH IS NOT MADE THE DEFAULT, AND THE REASON IT LOOKED LIKE A WIN IS ITSELF THE FIX.** The tiled kernel
is still 6.2x faster per dispatch in isolation, and it is NOT slower end to end (19.52 tok/s against untiled's
19.74, n=3 vs n=2 — inside this box's spread). The 1.49x it showed BEFORE the narrowing was the flush, not the
tile: with a mapped grid bound, every dequant dispatch flushed, so each dispatch that followed it was swept into a
~5-dispatch batch and paid a submit+wait; the untiled GEMM issues ~7,700 dispatches per chunk against the tiled
kernel's few hundred, so it paid that cost ~15x more often. Remove the serialisation and the whole gap goes with
it. **The shipped default is unchanged**; `STRATA_VK_PREFILL_TILED=1` / `=COOPMAT=1` remain the opt-in arms.
`coopmat`, the arm that measured slower than the tiled FMA kernel at EVERY shape in isolation (it loads its
operands from global with no staging), is also a wash end to end once the flush is out of the way: **20.14 tok/s
(9,829.9 ms) on the narrowed binary** against untiled's 19.74 and tiled's 19.52 — so the header's claim that it
loses is a claim about the SHADER, and the cheap test says the FMA default is not measurably better either. All
three keep their default-off status; none is claimed as a win.

**THE SAME NUMBERS AGAINST THE REFERENCE ON THIS CARD** (llama.cpp Vulkan, `pp512 913.36 +/- 289.06`,
`tg128 36.52 +/- 0.02`): 199-token prefill **19.74 / 913.36 = 2.16%** (was 1.77%), decode **8.34 / 36.52 = 22.8%**
(unchanged); 8-token arm prefill **6.63 / 913.36 = 0.73%** (was 0.57%), decode **8.22 / 36.52 = 22.5%**. Flushes
per prompt token: **13,619 / 198 = 68.8 → 3,021 / 198 = 15.3** on the 199-token arm, and **4,293 / 7 = 613 →
381 / 7 = 54.4** on the 8-token arm (the "~613" figure that opened this lead was the 8-token arm's).

**WHAT THE SAME NUMBERS SAY ABOUT WHERE THE PREFILL'S TIME NOW IS (narrowed binary, 199-token arm, `dequant`
2,054 ms / `gdn recurrence` 2,754 / `host grouping` 1,885 / `qsa proj` 768 / `gemm gate/up` 596 / `gemm down` 616
of a 9,964 ms host timeline).** The expert matmul itself is now ~12% of the prefill and `dequant` — the FP16
staging in front of it — is ~21%. **The MMQ-shaped expert matmul (quantized operands staged in shared memory,
packed integer dot behind a runtime `VK_KHR_shader_integer_dot_product` check, the scale applied once, no separate
dequantized buffer) is NOT landed in this batch.** What is measured about it: `iq_dequant_f32` costs ~0.55 ms per
dispatch over 10,833 dispatches, which at 24 GB/s of effective traffic is nowhere near this card's bandwidth — so
its cost is dispatch count and occupancy (the shader decodes one 256-value superblock with a 32-lane subgroup and
leaves 224 of its 256 lanes idle), not memory. Those are the next two numbers to attack, in that order.

## THE PER-ROUND BATCHING: 13-17% FEWER DISPATCHES, 1.2-1.4% OF `sync` — THE PREDICTION IS FALSIFIED, AND THE DELIVERED DECODE GAP IS SPECULATIVE WASTE (later, same day, `vega`)

**THE CHANGE.** The verify window's five per-token loops (`gdn_conv_l2_multi`, `gdn_ab_multi`,
`native_router_top10_multi`, `shared_expert_multi`, `native_moe_combine_multi`) now issue ONE dispatch per ROUND
instead of one per draft.  Two are host-only (`native_router_top10.spv` already carried the token dimension;
`shared_expert_multi` drives the module's own multi-column `ncols` path); three shaders gained a token dimension
(`native_moe_combine`, `fused_gdn_ab`, and `fused_gdn_conv_l2`, whose running history window is now read as
`stream[a..a+3]` instead of being slid token by token).  `sync`/`launch`/`commit`/`host` are the port's own
decode phase table; the dispatches are `STRATA_VK_DISP_STAT`'s segment counter.  Logs: `/tmp/meas/before2.*.log`
and `/tmp/meas/after.*.log`.

| arm | config | recorded dispatches / run | per round | verify `sync` ms/round | `launch` | `commit` | decode tok/s | ids md5 |
|---|---|---|---|---|---|---|---|---|
| 8-token | before `--spec 2` | 81,154 | 4,271.3 | 197.275 | 4.077 | 3.950 | 8.12 | `3aed108cceee` |
| 8-token | **after** `--spec 2` | 73,126 | 3,849.5 | **194.971** | 3.994 | 4.022 | **8.21** | `3aed108cceee` |
| 8-token | before `--spec 4` | 91,060 | 4,792.6 | 240.720 | 4.914 | 4.100 | 6.67 | `3aed108cceee` |
| 8-token | **after** `--spec 4` | 78,952 | 4,155.4 | **237.764** | 4.872 | 4.128 | **6.74** | `3aed108cceee` |
| 199-token | before `--spec 2` | 68,999 | 4,928.5 | 263.669 | 5.333 | 4.844 | 8.25 | `56a0b28d2de6` |
| 199-token | **after** `--spec 2` | 59,111 | 4,222.2 | **260.153** | 5.289 | 4.833 | **8.36** | `56a0b28d2de6` |
| 199-token | before `--spec 4` | 75,005 | 5,357.5 | 300.602 | 6.044 | 4.989 | 7.25 | `56a0b28d2de6` |
| 199-token | **after** `--spec 4` | 62,669 | 4,476.4 | **296.348** | 5.950 | 4.942 | **7.36** | `56a0b28d2de6` |

**THE PREDICTION WAS `sync 240.6 -> ~110 ms, decode 6.66 -> ~14 tok/s` AT T=4.  It is FALSIFIED**: the dispatch
count fell 13.0-16.5% and `sync` moved 2.30 ms (T=2) / 2.96 ms (T=4), i.e. **4.6-5.4 us per REMOVED dispatch**
against this port's own bench figure of 46-54 us.  The five loops' dispatches are real and were really per-draft
(the count DID fall, 105.6/106.2 fewer dispatches per WINDOW TOKEN on the 8-token arm), they are simply cheap;
the ~21.7 ms/draft measured at T=2/4/6 is elsewhere in the window.  Ids are identical in all eight runs and the
gate's `_multi` arms are bitwise, so the change is safe — it is just not the lever.

**THE CHAIN BARRIER IS NOT THE COST EITHER.**  `STRATA_VK_DISP_STAT` now counts them: **21,826 chain barriers at
T=2 and 26,668 at T=4 per run (~1,149 / ~1,404 per round — ~30% of the recorded dispatches, not one each)**.
`STRATA_VK_NOBARRIER=1` cannot even be measured: the prompt path breaks in the first stages
(`prefill: routed id out of range`, the engine's own guard).  A conservative bound-region hazard rule
(`STRATA_VK_BARRIER_HAZARD=1`) still required **21,825 of 21,826** barriers and left `sync` unchanged
(194.882 vs 194.971; 237.662 vs 237.764), ids identical — the barriers are at real region overlaps, and a
barrier is < ~5 us by the removal bound above.

**THE DECODE GAP, IN FOUR NUMBERS.**  The `--spec 4` 8-token window computes **6 draft tokens in 240.6 ms =
40.1 ms per computed token (~24.9 tok/s of raw compute)**; only **1.68 tokens/round are accepted**, so one
DELIVERED token costs `240.6/1.68 = 143.2 ms` (6.98 tok/s; the measured 6.67-6.74 adds launch/commit/host);
**72% of the window's compute is discarded** on unaccepted drafts; and against llama.cpp Vulkan on the same card
(36.52 tok/s = 27.4 ms/token) the port's **RAW compute is ~1.47x off**, while the delivered 4.4x gap
(36.52 / 8.27 at `--spec 2` on the 199-token arm) is dominated by speculative waste — an engine/MTP draft-quality
property, not Vulkan overhead.  `--spec 2` wins for exactly this reason (fewer drafts computed per round for the
same ~1.68 accepted): **1.14x** from the dispatch batching and **1.22x** from `--spec 2` over the engine's
`--spec 4` default, both id-verified.

## THE DECODE, ATTRIBUTED — the sync is the GPU executing the verify window; `--spec 2` is 1.22x (later, same day, `vega`)

**THE TRANSFER HYPOTHESIS IS RETRACTED, AND THE INSTRUMENT THAT KILLED IT IS ITS OWN.** The section below says
"the decode arm is the TRANSFER path... ~430 `begin_oneshot` transfers per decode token (13,774 ÷ 32)". That was
wrong. A new instrument, `STRATA_VK_XFER_STAT` (measurement-only, `vulkan/src/device/vk_compute.cpp`), prices every
staging transfer by CALL SITE and splits them by decode phase:

```
vk xfer stat: uploads 13725 (53613.30 MiB) | downloads 49 (62.54 MiB) | call sites 36
vk xfer stat decode-phase (after the first capture_begin): up 0 (0.00 MiB) | down 0 (0.00 MiB)
site 0: n=12288  bytes=25146163200  ExpertCache::fill_slot_blocking   <- the initial expert fill, 23.42 GiB
site 1: n=784    WeightTable::load        site 2: n=300  NativeDense::load     site 3: n=124  fill_slot_queued
```

13,725 uploads + 49 downloads = **13,774, exactly the by-arm `transfer` figure** — and it is the model LOAD, not a
per-token rate. **The decode issues zero transfers.** 13,774 ÷ 32 decode tokens divided a one-off load by a token
count; five mechanism guesses have now died the same way on this port, and this one was mine.

**THE DECODE'S 150 ms/token, MEASURED.** The prefill's 20-phase timeline (`STRATA_PREFILL_TIMING`) had no decode twin,
so the decode got its own marks at the decode-side sites — `ms_launch` (the recording's launch + the segment submits)
and `ms_sync` (the blocking window sync) in `Verifier::run`, printed by the CLI's window line:

```
8-token arm, --spec 4:
verify window  wait for rings 0.000  pool 0.000  host(stage) 0.986  launch 4.903  sync 240.623  commit 4.122 ms/round
19 rounds of 6, 1.68 tokens/round  ->  250.6 ms/round x 19 = 4,762 ms = 99.1% of the 4,803 ms decode
```

`sync` alone is **4,572 ms = 95.2% of decode** (~143 ms of the 150.1 ms/token). And the dispatch counter's new
segment row says what the sync waits on:

```
vk disp stat by arm: ... | segment 57 (91060 recorded dispatches, submit 1 ms, wait 4742 ms)
```

**57 segment submits, 91,060 recorded dispatches, 4,742 ms of fence wait, 1 ms of submit.** The decode is 3 segment
submits per round of a captured window; the wait is the GPU executing ~4,792 dispatches per round at the port's own
measured small-dispatch cost (its bench measures `iq2s_mmvq` at **0.0464 ms** on this card; 4,792 × ~0.05 ms ≈ 240
ms). The boundary loop itself is cheap: `advance_inflight` polls the mapped boundary word with a plain volatile read
and submits the next segment once it is served. **There is no drain-per-boundary waste to remove — the time is
kernels, and the fix is fewer dispatches per round.**

**THE MEASURED WIN, `--spec 2` (same binary, same prompt, ids identical at every setting):**

| `--spec` | drafts/round | verify `sync` ms/round | `launch` ms/round | decode tok/s | ids md5 |
|--:|--:|--:|--:|--:|---|
| 2 | 4 | 197.166 | 4.069 | **8.12** | `3aed108cceee` |
| 4 (engine default) | 6 | 240.623 | 4.903 | 6.66 | `3aed108cceee` |
| 6 | 8 | 285.399 | 5.775 | 5.63 | `3aed108cceee` |

`sync` fits **~110 ms fixed + ~21.7 ms per draft** (deltas +43.5 and +44.8 ms per +2 drafts) while tokens accepted per
round is **FLAT at 1.68** in all three. So drafting past 2 is pure cost on this prompt. `--spec 1` is refused by the
engine for a native pack (`generate.cpp:2167`), i.e. **the verify window IS the native pack's only decode path** — the
seam has to be made cheap, not designed away. `run_vk_perf.sh` now uses `--spec 2` with the reason in the header; the
engine's default is unchanged.

**AGAINST THE REFERENCE (same card, llama.cpp Vulkan `pp512 913.36` / `tg128 36.52`):**

| arm | configuration | decode tok/s | vs 36.52 | prefill tok/s | vs 913.36 |
|---|---|---:|---:|---:|---:|
| 8-token | `--spec 4` (before) | 6.66 | 18.2% | 5.19 | 0.57% |
| 8-token | `--spec 2` (after) | **8.12** | **22.2%** | — | — |
| 199-token | `--spec 4` (before, documented) | 7.25 | 19.9% | 16.17 | 1.77% |
| 199-token | `--spec 2` (after, same session) | **8.27** | **22.6%** | 16.05 | 1.76% |

**THE PREFILL FLUSH LEVER, RE-LABELLED (it is NOT the decode).** `STRATA_VK_FLUSH_STAT` (new) attributes every
live-batch flush to its trigger:

```
vk flush stat: 4293 live-batch flushes, 14742 dispatches, submit 17 ms, wait 1081 ms
vk flush stat decode-phase: 0 flushes, 0 dispatches, wait 0 ms
```

**4,293 flushes / 14,742 dispatches ≈ 3.4 dispatches per flush (~613 per prompt token), decode 0** — the fill-128
batch is never reached on the prompt path, and the triggers are the dequant/rope/embed-family dispatches whose bound
buffers are host-visible (`iq_dequant_gu_f16` 1,328 flushes / 6,480 dispatches / 433 ms wait; `iq_dequant_f16`
1,328 / 2,656 / 150 ms; one more 1,328 / 1,328 / 111 ms). The host-visible rule stays as documented; narrowing WHICH
mapped buffers count is a prefill lever.

**`submit_recorded` ZERO IS EXPECTED HERE.** A native (IQ) pack's CLI breaks to the verify path (`generate.cpp:7579`),
so the decode is the captured window re-submitted through `submit_segment` (57 segments). `submit_recorded` is the
unsegmented recorded-submit, reachable only from `record_end_and_submit`/`replay_recorded`, which nothing calls; the
per-token `TokenGraph` is the non-native path. `token graph hit path: 12288 resident experts` is the RESIDENCY
decision, not a captured per-token graph.

**`cudaStreamQuery` — MEASURED AND FIXED.** The last worker's "reasoned, not measured" risk was real. New arm in
`vulkan/tests/cudart_smoke.cpp`, against the device layer's own `Ctx::live_pending()`:

```
STRATA_VK_QUERY_NOFIX=1 : outstanding before/after = 1/1, cudaStreamQuery -> "no error"  -> FAIL (a wrong "complete")
fix                     : outstanding before/after = 1/0, cudaStreamQuery -> "no error"  -> PASS
```

`cudaStreamQuery` now flushes a pending live batch before answering. In the shipped decode it is latent: 38 calls, 0
with a live batch pending.

**GATE.** Arc **886 / 0 / 0** (same case count, nothing skipped, no bound widened); lvp 868/0/6; radeon 876/1/2 (the
documented `bf16_gemv_fp32_mmvf` family). Ids `3aed108cceee` / `56a0b28d2de6` throughout.

## THE DISPATCH LAYER — 5.14 → 16.17 tok/s prefill, and decode did not move (later, same day, `vega`)

**THE MEASUREMENT THAT DECIDED IT, AND IT WAS NOT ARITHMETIC.** `STRATA_PREFILL_TIMING=1` on the 199-token arm
(`/tmp/perf_timing_199.log`, 198 tokens, GPU timeline 36,976 ms, host staging **52 ms**):

```
embed+steps 145 (0.4%)  hc read 1249 (3.4%)  gdn 6198 (16.8%)  qsa proj 4054 (11.0%)  qsa indexer 486 (1.3%)
qsa select 4  qsa attn 648 (1.8%)  router+shared 8976 (24.3%)  host grouping 16 (0.0%)  gather 32 (0.1%)
wait copy 0  dequant 2175 (5.9%)  gemm gate/up 5513 (14.9%)  gemm down 2804 (7.6%)  combine 17
ple 256 (0.7%)  gdn conv+gates 16  gdn recurrence 1553 (4.2%)  gdn out proj 2831 (7.7%)
host: chunk setup 145 ms, waiting for each chunk 0 ms, after each chunk 0 ms, PLE 257 ms
```

The 8-token arm (`/tmp/perf_timing_8.log`, 7 tokens, 2,509 ms) says the rest: `router+shared` 317 ms, `gdn` 222,
`qsa proj` 135, `gdn out proj` 83 all scale **exactly** with token count (28.0×, 27.9×, 30×, 34× from 7 to 198
tokens), while `dequant` scales with EXPERT count (3.5×) — so most of the prompt path is a per-token loop of
decode kernels, and the largest single phase (`router+shared`, 24.3%) was **waiting**, not computing.

**THE NUMBER THAT NAMED THE CAUSE: `STRATA_VK_DISP_STAT=1` (NEW, this batch; `vulkan/src/device/vk_compute.cpp`),
prices the submission layer itself and prints at exit.** Priced on the Arc Pro B70, 199-token arm:

```
vk disp stat: 233768 live dispatches | 249878 recorded dispatches | 249878 submits | 249878 host waits |
              249878 cb allocs | 249878 fences created | 14684 descriptor sets | 250 descriptor pools
vk disp stat ms: total 61662 = cb-alloc 198 + encode 158 + fence-create 274 + submit 5982 + wait 54674 +
                 cb-free/fence-destroy 376   (0.2638 ms/dispatch)
by shader (top): pf_f16_to_f32 59400  quantize_q8_1 59400  native_k_mmvq 41580  iq_dequant_f32 10833
                 iq4nl_mmvq 9306  iq4xs_mmvq 8316  gemm_prefill_fma_small 7702  native_gdn_out_norm 7128
                 native_gdn_step 7128  pf_swiglu16 3659  f32_to_f16 3647  pf_gu_interleave_f16 3611
```

**EVERY dispatch was its own command buffer, its own fence, its own submit AND its own wait — 249,878 of each for
233,768 dispatches** (`Ctx::dispatch` → `begin_oneshot` + `end_oneshot_and_wait`). The fence `wait` is **88.7%**
of the dispatch layer; and **50.8% of all dispatches were two pure-overhead shaders** (`pf_f16_to_f32` +
`quantize_q8_1`, 59,400 each) from one loop: `prefill::Gemm::native` quantised and GEMV'd **one token at a time**,
so the 300 dense projections of a 198-token chunk cost 160,380 dispatches (68.7% of the run).

**THE TWO FIXES, EACH MEASURED, AND `SUB/Ms` IS NOT WHAT THE SPEED CAME FROM.**

| | prefill 198 tok | tok/s | decode | dispatches | submits |
|---|---:|---:|---:|---:|---:|
| before | 38,513.8 ms | **5.14** | 7.25 | 233,768 | 249,878 |
| + batched `Gemm::native` (3 dispatches per matrix, T columns per dispatch — the shaders already walked `ncols`) | 15,095.9 ms | **13.12** | 7.24 | 56,468 | 72,620 |
| + the live dispatch BATCH (one command buffer, flushed at observers) | 11,391.2 ms | **17.38** | 7.25 | 56,468 | 19,048 |
| + the host-visible completion rule (the gate fix below) | 12,245.7 ms | **16.17** | 7.25 | 56,468 | 29,771 |

TTFT 38.9 s → 12.8 s. Against the same-card reference (llama.cpp Vulkan, pp512 913.36, tg128 36.52): prefill
**16.17/913.36 = 1.8%** (56× off, was 157×), decode **7.25/36.52 = 19.9%** (5.04× off, UNCHANGED). 8-token arm:
prefill 2.79 → **4.45** tok/s, decode 6.65.

### THE RED GATE, ON THE RECORD — 881/5/0 before 886/0/0

The FIRST batching version was **`== 881 passed, 5 failed, 0 skipped`** on the Arc against 886/0/0 in this
batch's own earlier runs, all five in the machinery the change touches:

```
FAIL sample_tokens entry (mapped out): the id lands in the MAPPED out (verify.cpp:1170 shape)   0/1  worst -1.23e+04
FAIL sample_tokens entry (mapped out): a second call reproduces the id (a live write)            0/1
FAIL doorbell ring: direct increments the ring                 0/2  worst 0  the ring did not read 1 then 2
FAIL doorbell ring: capture records, does not run              0/1        the ring moved during the capture
FAIL doorbell ring: a replayed block advances the ring each replay 0/3    the ring did not read 3, 6, 9
```

THREE CAUSES, each a removed ordering, each fixed at the cause (not retried, not skipped):

1. **`chain_barrier` was conditional on `live_n_ > 0`**, so the FIRST dispatch in each batch had no compute→compute
   barrier. The router wrote `m.ids`, the next dispatch (`pf_copy_u32`) copied it out unsynchronised, and the host
   read garbage: **`prefill: routed id out of range`**, twice out of two runs, against a known-good 13.12 tok/s
   baseline. The engine's own validation caught it. Fix: a barrier after every dispatch in the batch. (The
   pre-A gate runs before this fix were NOT the documented interstitial — `NEXT.md:490` — and are not recorded as
   one: 2 of 2 failing before, 2 of 2 clean after, same binary otherwise.)
2. **`cudaStreamSynchronize`/`cudaDeviceSynchronize` were vacuous** — they assumed "every dispatch submits with a
   fence and waits". With batching they must FLUSH the pending batch; the prefill's `ids_h` host read depends on it.
3. **THE HOST-VISIBLE RULE (the gate fix).** A dispatch that touches a **mapped host region** keeps the documented
   contract — COMPLETE when `dispatch` returns — because the engine and the gate read those regions DIRECTLY, with
   no `Ctx::read` to flush for them. Without it, `doorbell_ring`'s pending increment ran at the NEXT flush (inside
   the capture: "the ring moved during the capture"), the replay case read a stale ring, and `sample_tokens` read
   its `-12345` sentinel. Device-local buffers (the engine's 27.9 GiB arena) still batch. Cost of the rule: 17.38 →
   16.17 tok/s prefill. **The 17.38 was real but it was not a real increment** — it depended on breaking a
   documented contract, so the landed number is 16.17.

**EQUIVALENCE, AND IT HELD THROUGH THE CONTRACT FIX.** `output  : 198 1 198 1 ...` is md5 **`56a0b28d2de6`** on
the pre-change run and on both post-fix runs (`/tmp/perf_before_199.log`, `/tmp/perf_A_199_r3.log`,
`/tmp/perf_final2_199.log`); the 8-token arm's `4653 8 15 15 ...` is **`3aed108cceee`** before and after
(`/tmp/perf_before_8.log`, `/tmp/perf_final2_8.log`). A faster path that changes the answer is a defect.

**THE HONEST HALF — DECODE, AND THE ARM BREAKDOWN.** `STRATA_VK_DISP_STAT` now splits submits by arm:

```
199-token arm:  live-batch 13619 | transfer 16110 | recorded-submit 0 (replays 0) | segment 42
8-token arm:    live-batch  4293 | transfer 13774 | recorded-submit 0 (replays 0) | segment 57
```

**Decode is 7.25 tok/s in EVERY run tonight — before fix A, after fix A, after the contract fix.** Its cost is not
the recorded path: `submit_recorded` is called **ZERO** times. The 8-token arm (7 prefill + 32 decode tokens) shows
**13,774 `begin_oneshot` TRANSFERS — ≈430 per decode token** — each still paying a command-buffer allocate, a
fence create, a submit and a wait. That is the same per-operation round trip just removed for dispatches, still
present on the transfer path, and it is the named next target (not fix #1 above, which it superficially resembles).
Unmeasured: the per-arm split of `wait` ms; the decode arm's phase table (the instrument is prefill-only);
`cudaStreamQuery` still reports a pending batch as done (no live batch is pending on the paths it is used from —
verify.cpp's captured handshake — but that is REASONED, not measured).

This is the measured performance record for the Vulkan backend on the Intel Arc Pro B70 (`BMG G31`), on branch
`vulkan-arc-port`, from commit `2cc38c8` (the tag `v0.1.39-with-arc`) through `b98e2ba`. Every number here was
taken on vega with the card otherwise idle; each one names what it was measured on. Nothing in this file is
projected.

## Performance on the Arc Pro B70 — and the same card running llama.cpp

The port produces a token; the next question is how fast, and what "fast" even means on this card. Both sides below
are measured on the same machine, the same GPU, the same driver and the same Vulkan loader, on a mixture-of-experts
of the same class, so the comparison is like for like.

| | prefill | decode |
|---|---:|---:|
| this port, all-resident, 198-token prompt (`--prefill 256`) | **5.81 tok/s** — 198 tokens, 1 chunk, 34 069 ms, time to first token **34.4 s** | **7.24 tok/s** — 32 tokens, 4 421 ms |
| this port, 8-token prompt | 2.79 tok/s — 7 tokens, 1 chunk, 2 511 ms | 6.64 tok/s |
| llama.cpp **Vulkan**, same card, Qwen3.5-35B-A3B Q4\_K\_M (20.49 GiB, 34.66 B params, ~3 B active) | **913.36 ± 289.06 tok/s** | **36.52 ± 0.02 tok/s** |

| this port against that reference | prefill | decode |
|---|---:|---:|
| | **157× slower** | **5.0× slower** |

So **Vulkan is not the limit** — llama.cpp's Vulkan backend reaches 913 tok/s of prefill on this silicon and 36.5
tok/s of decode — and the gap is in this port's kernels and its dispatch, not in the API. For scale: this is the
same card that produced token id **20** for `prompt 1 2` in four runs, and the milestone's numbers stand as
published above.

### What the measurements ruled out (each was a plausible diagnosis first)

| diagnosis | what the measurement said |
|---|---|
| "the prefill matmul is untiled — one invocation per output element, so it re-reads its operands" | Tiling it made the kernel **6.1× / 6.9× faster in isolation** (gate/up 6.426 → 1.053 ms, down 3.291 → 0.477 ms at T=199) and the **engine slower**: 5.81 → **3.88** and **3.25** tok/s across four runs. Shipped opt-in (`STRATA_VK_PREFILL_TILED=1`), not default. |
| "the dequant-to-FP16 staging pass is the cost" | It is **~0.18 ms per expert, ≈4%** of the per-expert budget. Removing it would buy almost nothing. |
| "cooperative matrix is free headroom on this silicon" | The shaders are dispatchable (`coopmat=1 cm=8x16x16`) but measure **slower at every shape** than the tiled FMA, because they load straight from global with no shared-memory staging. That is this port's shader, not the feature. |

### What it points at instead

The one-chunk prefill costs **~2.9–3.6 ms per token-layer**, and decode costs **~2.9 ms per token-layer** — the same
number. The batched prefill is therefore not batching anything: both are bound by the **per-token expert path**
(48 layers × ~10 experts × 3 matmuls ≈ **1440 small dispatches per token**, ≈96 µs each), not by arithmetic. At
~6.5 GB/s of effective expert read — about **1.5%** of this card's VRAM bandwidth — decode is not bandwidth-bound
either.

The instrument that makes the next number trustworthy is already built in: `STRATA_PREFILL_TIMING=1` prints a
**20-phase GPU timeline** (`router+shared`, `host grouping`, `gather`, `wait copy`, `dequant`, `gemm gate/up`,
`gemm down`, `combine`, `ple`, `qsa attn`, `gdn …`) plus the host's own share (`host_sync_ms`, `host_chunk_ms`,
`host_setup_ms`). It is switched off, not unbuilt — so the next step is a measurement, not a rewrite. Everything
that is compared here holds the gate green and the injections biting: a faster path that changes the output tokens
is a defect, not a win.

### What other engines do on this silicon

| engine | the technique | in this port |
|---|---|---|
| llama.cpp Vulkan | tiled `mul_mm`; **MMQ** (`mul_mmq.comp`) — a tiled matmul that dequantizes **in registers, in one pass**, straight from the quantized weights; flash attention (cm1/cm2, dequant, split-k) | no — this port dequantizes to FP16 and then matmuls |
| llama.cpp Vulkan, Intel specifically | **cooperative matrix enabled for Xe2 on purpose** (PR #14001, detected by `minSubgroupSize` 8 → 16): Lunar Lake pp512 **154 → 398 tok/s**, where the same change *regressed* A-series (A770 **961 → 261**). Quantized GGUFs still fall back to **dp4a** rather than XMX. | shaders shipped, **not dispatched**; would need staging first |
| llama.cpp SYCL / IPEX-LLM / OpenVINO / vLLM-XPU | the vendor path, via oneAPI and Level Zero. Issue #22413 is titled *"brutally bad SYCL performance on Battlemage"* and its reporter adds OpenVINO and vLLM-XPU were "similarly badly" affected, with recovery reported on oneAPI 2025.3.3 / B60. | being measured on this card with the installed oneAPI 2026.1 |

The card's own capability line bounds all of it (llama.cpp's probe):

```
Intel(R) Graphics (BMG G31) | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 131072 | int dot: 0 | matrix cores: KHR_coopmat
```

### Not measured, not run, or open

- The **minefield / hallucination / instruction-following / tool-use batteries** have **not been run** against this
  engine. The instruments exist and self-test (69 / 44 / 30 cases); the runs do not.
- The `--kv q4_0` and `--mtp` arms: **not run** — unstarted, not failed.
- The **platform intermittent is not RADV-specific.** On the prefill-tiling batch it appeared on the **Intel arm**
  (`gate_perf4`: `fused_gdn_step_norm`, one element out of ~132 k, while the identical binary returned 886/0/0 in
  the runs after it). Recorded with the kernel, shape and count; not chased. **A green arm means "no failure
  observed in that run."**
- One binary, five gate runs: Arc **883/0/0**, **884/2/0**, 886/0/0, 886/0/0, and one run stopped at a build error
  while the harness was mid-edit — the gate refuses to report a result on a broken build rather than guessing.

## Provenance — the exact runs behind these numbers

Card, driver and API level (from llama.cpp's own probe, which also bounds every fast path available here):

```
Intel(R) Graphics (BMG G31) | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 131072 | int dot: 0 | matrix cores: KHR_coopmat
vk_stack: ICD intel_icd.json api 1.4.318 -> /usr/lib/x86_64-linux-gnu/libvulkan_intel.so
```

The reference — llama.cpp's Vulkan backend, build `7fe450e19`, `-ngl 99`, three repetitions:

```
llama-bench -m Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 3
  pp512 = 913.36 +/- 289.06 t/s     tg128 = 36.52 +/- 0.02 t/s      -> /tmp/ref_llamacpp_vulkan.log
```

This port — all-resident (12288 of 12288 expert slots, 27.891 GiB arena, `--mmap-experts`), same card, same
session:

```
STRATA_VK_ARENA_MIB=28560 STRATA_VK_DESKTOP_RESERVE_MIB=256 /tmp/memguard_swap.sh 45G 16G \
  ~/vkbuild-vulkan/vulkan/strata_vulkan --pack <coder-iq1_m> --native <IQ1_M shard 1> \
  --spec 4 --prefill 256 --tokens <199-token prompt> --max-new 32 --max-context 512 \
  --expert-profile /tmp/expert-profile-coder-built.bin --expert-cache 12288 --mmap-experts

prefill 198 tokens in 1 chunks, 34069.3 ms (5.8 tok/s)    decode 32 tokens in 4420.6 ms -> 7.24 tok/s   -> /tmp/perf_before_199.log
prefill 198 tokens in 1 chunks, 50999.3 ms (3.9 tok/s)                                                 -> /tmp/perf_after_199.log   (tiled, opt-in)
prefill 198 tokens in 1 chunks, 60953.6 ms (3.3 tok/s)                                                 -> /tmp/perf_after2_199.log  (tiled, opt-in)
```

Run ledger (one JSON row per run, appended at the time): `/home/bob/forktest/perf-arc-2026-10-06.jsonl`.
The isolated GEMM probe: `/home/bob/forktest/vk_gemm_probe.cpp`, sweep in `/tmp/gemm_sweep.log`.

The gate — the numbers the port's "green" rests on, same binary across five runs:

| run | Arc (`intel_icd`) | llvmpipe | radeon iGPU |
|---|---|---|---|
| `gate_perf1` | 883 / 0 / 0 | 865/0/6 | 872/2/2 |
| `gate_perf3` | — build error mid-edit; the gate refused to report | — | — |
| `gate_perf4` | **884 / 2 / 0** — the intermittent | 865/3/6 | 874/3/2 |
| `gate_perf5` | 886 / 0 / 0 | 868/0/6 | 875/2/2 |
| `gate_perf6` | 886 / 0 / 0 | 868/0/6 | 875/2/2 |

Falsification, re-run by the parent rather than taken on report:

```
bash ports/vulkan/gates/inject-verify.sh pf-gemm-fma-wrong-ldy
  FALSIFIED (pf-gemm-fma-wrong-ldy): FAIL prefill Gemm::f16 entry: Y[T,ldy]=X.W^T (ragged T=3 N=5 K=8, ldy>N) vs a double reference
```

Equivalence, without which the batched number would not be trustworthy: `--prefill 1` (7 chunks, 9137.2 ms) and
`--prefill 256` (1 chunk, 2582.1 ms) produce the **same output token ids in the same order** on the 8-token
prompt, for 3.7x better time to first token. A batched path that changes the answer is a defect, not a win.

### The SYCL path, measured — it is not the way on this card

The obvious alternative to hand-written Vulkan is Intel's own toolchain, so it was built and run on the same card,
same model, same flags (`oneAPI DPC++/C++ 2026.1.1`, ggml `bdff91b14`, `-ngl 99 -p 512 -n 128 -r 3`):

| configuration | pp512 | tg128 |
|---|---:|---:|
| llama.cpp **Vulkan** (the reference above) | **913.36 +/- 289.06 tok/s** | **36.52 +/- 0.02 tok/s** |
| SYCL, default | not run - SIGSEGV | not run - SIGSEGV |
| SYCL + `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | not run - SIGSEGV | not run - SIGSEGV |
| SYCL + `GGML_SYCL_ENABLE_VMM=0` | not run - SIGSEGV | not run - SIGSEGV |
| SYCL + `GGML_SYCL_FORCE_MMQ=1` | not run - compile-time define, rebuild required | - |

`sycl-ls` sees the card (`[level_zero:gpu] Intel(R) Arc(TM) Pro B70 Graphics`), the build succeeds, and **every**
run dies `RC=139`: one at device enumeration, one deep in prompt processing (`ggml_sycl_get_rows` on
`conv_states-0`), with `general protection fault ... in libc.so.6` in the kernel log and **no xe engine reset** — a
userspace fault, not a GPU hang. No SYCL throughput number exists; none is inferred, none is written as 0.

### The mechanism that actually closes the gap (read from the source, not guessed)

| engine | mechanism | in this port |
|---|---|---|
| llama.cpp Vulkan, **expert matmul** | **batched into ONE dispatch**: `mul_mm_id_funcs.glsl` loads the routing row-ids once, uses `gl_WorkGroupID.z` as the expert index, `subgroupBallot` for the counts | **no** — this port dispatches per expert (`matvec_vk.cpp:192`, one workgroup per output row) |
| llama.cpp Vulkan, **quantized matmul** | **MMQ** (`mul_mmq.comp`, BM/BN 64, BK 32): stages the quantized operands in shared memory, accumulates with the packed integer dot, applies the scale **once at the end** — the weights are never dequantized into a separate buffer | **no** — this port runs a dequant-to-FP16 pass (`iq_vk.cpp:189-216`) and then GEMMs |
| llama.cpp Vulkan, prefill | tiled `mul_mm` / `mul_mmq` with shared-memory staging | **opt-in** (`STRATA_VK_PREFILL_TILED=1`); untiled is the default because tiling measured *slower* end-to-end (`ee69697`) |
| llama.cpp Vulkan, cooperative matrix | `mul_mm_cm2.comp`, device-gated (`ggml-vulkan.cpp:82`; Xe2 = `minSubgroupSize == 16`), enabled for Xe2 by PR #14001 | shaders shipped, **off**; measured slower at every shape (they load from global with no staging) |
| llama.cpp Vulkan, attention | `flash_attn*` (cm1/cm2/dequant/split-k) | **no** — `qsa_decode_attn.comp`, `attn_decode_short.comp`, no tiled flash attention |
| llama.cpp SYCL, expert matmul | **not** one dispatch: decode is a per-hit GEMV (`ggml_sycl_mul_mat_id_mmvq_fused`), prefill a host-side counting sort into per-expert slices, then batched GEMM | n/a |
| llama.cpp SYCL, XMX | **explicitly unused** — `ggml-sycl/common.hpp:99-102`: *"define for XMX in Intel GPU / TODO: currently, it's not used for XMX really"* | n/a |

Two things follow. **The lever is the expert dispatch, not arithmetic** — which is what the per-token-layer
measurement said independently (prefill ~2.9-3.6 ms/token-layer, decode ~2.9 ms). And **XMX is not the lever for
this pack**: the vendor's own SYCL backend leaves it unused, the reference's quantized path uses the integer dot
instead, and this card's own probe reports `int dot: 0` — so a ported MMQ must test
`VK_KHR_shader_integer_dot_product` first and keep a non-packed accumulation path.
