# Measured performance on the Intel Arc Pro B70 — 2026-10-06

## THE EMPTY LAUNCH IS NOT IN THE OTHER LAUNCHERS — IT IS A SECOND TIME IN THE SAME ONE: `native_k_mmvq` has no window loop OR y-grid (and the "41,580 dispatch" premise is STALE — it is < 1,152 now), `moe_grouped_s2` is a refusal and `peer_experts` only calls the launcher that already had the cap; the grouped launcher's WINDOW-INDEPENDENT stages (SwiGLU + q8_1) ran `nwin`=8 times per call and are now HOISTED — window **227.9 → 225.0 ms/round**, decode **9.49 → 9.62 tok/s**; `fused_gr_mix` priced a REDUCTION-BOUND NO (2026-10-06, `vega`, Arc Pro B70)

### 1. THE AUDIT THE BRIEF ASKED FOR — AND THE PATTERN IS IN NONE OF THE NAMED LAUNCHERS

| launcher | window loop? | y-grid? | verdict |
|---|---|---|---|
| `native_k_mmvq` | **no** (`matvec_vk.cpp:254-272`) | **no** (1-D grid `n_out`) | no empty launch to cap; headline count STALE |
| `moe_grouped_s2` | n/a | n/a | **loud refusal** (`refusals_vk.cpp:187`); non-native packs only |
| `peer_experts` / `remote_experts` | n/a (calls `native_expert_grouped`) | inherits the `gy` cap | no own launcher; **not dispatched** in an all-resident run |
| `verify::fetch_blobs` | yes | 1-D, grid `cap ≤ 64` | non-resident path; absent from every histogram |

**The `native_k_mmvq` number is stale.** The brief's "41,580 dispatches in one prefill histogram, ~50% of the
dispatch layer" is the **per-token** prefill form: 210 K-quant dense tensors × 198 tokens. The batched native
GEMM (`prefill_vk.cpp::Gemm::native`, the `beta == 0 && ldy == N` arm) already collapsed that to **three
dispatches per PROJECTION** — `f16_to_f32` + `quantize_q8_1` + `native_k_mmvq` — so 210 instead of 41,580.
The engine's own current histogram agrees: **`native_k_mmvq.spv` is absent from the top-14 of the run's 59,640
live dispatches**, whose 14th entry is 1,152. New arm `native_k_mmvq_engine` (Q6_K, n_in 2560, n_out 1280,
device-local, ONE process):

| row | grid | median | per token | dispatches for 198 tokens |
|---|---|---:|---:|---:|
| `k_mmvq_batched` | `(n_out, 1)` | **5.9188 ms** | **0.0299 ms** | **1** |
| `k_mmvq_pertoken` | `(n_out, 1)` | 0.0408 ms | 0.0408 ms | 198 |
| `k_mmvq_gy8_spurious` | `(n_out, 8)` | 0.2006 ms | 0.2006 ms | 1 (8× the WORK) |

The last row is the point: the shader reads no `gl_WorkGroupID.y`, so a y-grid multiplies **workgroups**, not
guarded-and-empty ones. `gy` is not a knob this launcher has. Batched vs per-token: 1.36× the per-token cost and
**198× the dispatch count** — a dispatch-count result, not an empty-launch one.

### 2. WHAT THE AUDIT FOUND INSTEAD — THE SAME WASTE ONE LAYER UP, IN THE LAUNCHER THAT ALREADY HAD THE CAP

The window loop exists **only** for the weight read (a 32-bit storage-buffer index reaches one 4 GiB window).
**Two of the four stages read no weights** — the SwiGLU over the gate/up scratch and the q8_1 quantise of its
result — and both sat **inside** the loop, so a call ran **8 SwiGLU launches and 8 q8_1 launches, 7 of each
recomputing the byte-identical result** over the identical scratch. New arm `native_grouped_hoist` (30 groups =
3 tokens × top-10, n_embd 2560, n_ff 1280, the arena's 8 windows, device-local):

| row | dispatches/call | median ms/call |
|---|---:|---:|
| `grp_call_current` (8 × (gu + swiglu + q8_1 + down)) | 32 | **4.1465** |
| `grp_call_hoist` (8 × gu + swiglu + q8_1 + 8 × down) | 18 | **4.0780** |
| `grp_swiglu1` (one SwiGLU alone) | 1 | 0.0205 |
| `grp_q8_1` (one q8_1 alone) | 1 | 0.0216 |

Delta **0.0685 ms over 14 removed dispatches = ~4.9 µs/dispatch**, the record's own live-batch marginal
(4.6-5.4 µs). The fix is a reordering, not a kernel change: **gu loop → SwiGLU → q8_1 → down loop**; `gu`/`down`
stay window-bound and `down` must follow the single q8_1 image. Bit-identical by construction.

### 3. THE ENGINE A/B — interleaved, n=3 per arm, against a saved copy of the previous binary

`--spec 2 --prefill 256 --max-new 32`, one config per invocation, detached and polled; every arm logs its own
env, shader sha256s and binary sha256 (`/tmp/gyfix/`). prev = `c59e8687…` (HEAD 4f4aea7), new = `662d2547…`.

| arm | bin | segments | recorded disp | barriers | sync ms/round | wait ms | decode | id |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `prev199`  | prev  | 42 | 59,111 | 71,005 | 227.945 | 3,322 | 9.48 | `56a0b28d2de6` |
| `prev199b` | prev  | 42 | 59,111 | 71,005 | 227.901 | 3,322 | 9.50 | `56a0b28d2de6` |
| `prev199c` | prev  | 42 | 59,111 | 71,005 | 228.047 | 3,323 | 9.50 | `56a0b28d2de6` |
| `new199`   | hoist | 42 | 49,703 | 68,989 | 225.085 | 3,281 | 9.62 | `56a0b28d2de6` |
| `new199b`  | hoist | 42 | 49,703 | 68,989 | 224.882 | 3,278 | 9.62 | `56a0b28d2de6` |
| `new199c`  | hoist | 42 | 49,703 | 68,989 | 224.988 | 3,279 | 9.63 | `56a0b28d2de6` |

Window sync **227.96 → 224.99 ms/round (−1.3%)**, decode **9.49 → 9.62 tok/s (+1.4%)**, segment wait
3,322 → 3,279 ms; **disjoint ranges** (prev 227.901-228.047 / 9.48-9.50; new 224.882-225.085 / 9.62-9.63). The
**encoded dispatch composition falls by exactly the predicted 2,016**: `swiglu_f32` **1,296 → 288**,
`quantize_q8_1` **1,875 → 867**, `native_gu_any` / `native_down_any` / `fused_gr_mix` and every other kernel
**UNCHANGED**; chain barriers **71,005 → 68,989**. Id `56a0b28d2de6` in every arm. The 8-token arm
(`1 2 3 4 5 6 7 8` → id `3aed108cceee`) reproduces it: `prev8` / `prev8b` **172.186 / 172.424 ms** and 9.26 /
9.24 tok/s against `new8` / `new8b` **169.366 / 169.657** and 9.39 / 9.38 — disjoint ranges, **-1.6% sync /
+1.5% decode**.

### 4. TARGET 2 — `fused_gr_mix` (68.3 µs, the largest priced single kernel) IS A PRICED **NO**

New arm `fused_gr_mix_pass` varies `hc` on the SAME shader (N=2560, LR=320):

| row | median | barrier reductions per column |
|---|---:|---:|
| `mix_hc4` (the artifact) | **0.0683 ms** | 4 |
| `mix_hc2` | 0.0455 ms | 2 |
| `mix_hc1` | 0.0338 ms | 1 |

Each of the four `wg_sum` reduction trees costs ~**11.5 µs (17% of 68.3)**; the four are ~46 µs = **67% of the
kernel**, over a body of 5 MACs per thread. The only lever that removes a reduction changes the summation
**tree** — the last bits, and therefore the ids the contract pins — and the gate **bans subgroup reductions**
(`run_gate.sh`'s census). Spread options change the tree the same way. **Priced NO; the kernel is unchanged.**

### 5. THE LEVER NOT TAKEN (Task 3, priced not built)

The launcher still issues 8 launches per call — 7 `gu` + 7 `down` empty at `gy`=1. Removing them needs a 64-bit
buffer index (glslang 15.1 rejects it, measured), buffer-device-address, or an 8-binding weight switch (each
binding one 4 GiB arena view, selected by `grp_win[g]`). All three are real; none was attempted.

## THE "GROUPED GEMV ncols" LEVER IS FALSIFIED, AND THE EMPTY LAUNCH IT EXPOSED IS WORTH 12.2% OF THE VERIFY WINDOW — `native_expert_grouped`'s y-grid capped, window 260.4 → 228.0 ms/round, decode 8.34 → 9.46 tok/s, ids unmoved (2026-10-06, `vega`, Arc Pro B70)

**THE ONE PARAGRAPH.** The queued brief sized the next lever as *"batch tokens per weight-reading dispatch: `iq1m_mmvq` n_out=1280 at ncols=3 is 51.8 µs against 111.0 µs at ncols=1 — 3x the work for HALF the cost — so the window's 2,304 `native_gu_any`/`down_any` dispatches should go ~265 ms → ~124 ms."* **Both halves of that premise are wrong and the measurement says so.** (1) The 51.8-vs-111.0 ratio is a **memory-type confound**: ncols=3 was measured in the arena's DEVICE_LOCAL type and ncols=1 in the harness's MAPPED type, and `iq1m_controlled` — the same four rows, one process, interleaved — gives **mapped 111.0 → 58.2 µs** (ncols 1 → 3) but **device-local 28.5 → 58.2 µs**, i.e. within the memory type the engine actually uses, three columns cost **2.07x one column for 3x the work — a 1.45x per-dispatch gain, not 6.4x**, and the `ncols=1` device-local row (26.8-30.1 µs) is already the FAST one. (2) The port's grouped expert kernel does not dispatch per token at all: the brief's own `native_gu_any` runs **one launch per arena WINDOW for ALL groups at once** (grid `2*n_ff × gy`, and each workgroup strides every entry of its group), and the engine-shaped rows say the token-batching alternative is **slower**: `gu_1grp30` (30 entries on one expert, one dispatch) 1.4725 ms against `gu_port8` (the port's 8-window form) 0.4932 ms/dispatch. **THERE IS NO SEPARATE GROUPED-GEMV CHANGE TO MAKE, AND NONE WAS MADE.** What the same arm DID find is that **an EMPTY launch is 65% of the port's `native_expert_grouped` call**: a layer's groups are contiguous in the pack and live in ONE arena window, yet the launcher loops all 8 windows, so the engine runs 1 working launch and **7 launches whose every workgroup `continue`s on `grp_win[g] != win_id`** — measured **0.3168 ms** for gu and **0.1606 ms** for down per empty launch. The empty launch is the launch of the caller's grid (`2*n_ff × cap_groups` = 76,800 workgroups), so **the fix is the launch shape, not the kernel**: cap the y-grid at 1 (`gy` only strided how many groups a launch covers — results do not depend on it, proven by the gate's own `grid_groups=1` arm and `native_grouped_parity`'s 0/1/2/3/4/cap+3 sweep). The full launch is **also** faster at gy=1 (1.5883 vs 1.6912 ms), and per CALL the grouped path goes **3.91 → 2.35 ms (gu)**. In the engine: window **260.427 → 228.020 ms/round**, per recorded dispatch **64.05 → 56.26 µs**, decode **8.34 → 9.46 tok/s (+13.4%)**, with the segment count, the 59,111 recorded dispatches and the 71,005 chain barriers all UNCHANGED and the id `56a0b28d2de6` unmoved.

### 1. THE FALSIFIED PREMISE, MEASURED IN ONE PROCESS (the port's own "a row's position decides its value" trap)

`iq1m_mmvq` n_out=1280, batch 8, reps 9, `intel_icd`, all six rows of ONE arm (`iq1m_controlled`) so an ordering or memory-type effect cannot be read as an ncols effect:

| row (one process, interleaved) | ncols=1 | ncols=3 | 3 cols / 1 col |
|---|---:|---:|---:|
| **mapped** (the harness's default) | 0.1446 ms (min 0.1108) | 0.0588 ms | 0.41x |
| **device-local** (the ARENA's type) | 0.0301 ms (0.0298) | 0.0585 ms | **1.94x** |
| repeat: mapped / device-local | — | 0.0586 | — |
| repeat: device-local ncols=1 | 0.0268 ms | — | — |

The committed `51.8-vs-111.0` pair is `dev ncols=3` against `mapped ncols=1`. The honest pair is the device-local one, **and the direction is the opposite of the brief's**: 3 columns cost 1.94x the time, so per dispatch the family gains **1.45x**, and ncols=1 device-local (26.8-30.1 µs) is faster per column than ncols=3 (19.5 µs/column). The engine's arena is device-local, so the device-local row is the one that applies — and it says the 265 → 124 ms projection had already assumed the answer.

### 2. THE PORT'S GROUPED GEMV ALREADY BATCHES — AND AT THE ENGINE'S SHAPE IT BEATS TOKEN-BATCHING

New arm `native_grouped_engine` (IQ3_XXS gate/up, IQ4_NL down, n_embd 2560, n_ff 1280, 30 groups = the `--spec 2` window's 3 tokens × top-10, DEVICE_LOCAL, batch 8). `gu_*`/`dn_*` are **per dispatch**; a "call" is the launcher's 8-window loop:

| row | what it is | ms/dispatch | ms per CALL |
|---|---|---:|---:|
| `gu_1win` | 1 dispatch, all 30 groups in ONE window (the CUDA's shape) | 1.6918 | 1.69 |
| `gu_port8` | 8 dispatches, groups spread over the 8 windows | 0.4932 | **3.95** |
| `gu_engine1` | 8 dispatches, all 30 groups in ONE window (the ENGINE's real case) | 0.4817 | **3.85** |
| `gu_1grp30` | 1 dispatch, 30 entries on ONE expert (the "batch tokens" idea) | 1.4725 | 1.47 |
| `dn_1win` / `dn_engine1` | the down side, same two shapes | 1.1825 / 0.2853 | 1.18 / **2.28** |
| `gu_empty` / `dn_empty` | ONE dispatch, no group in its window | **0.3168** / **0.1606** | 7 × → 2.22 / 1.12 |

**Two things fall out.** (a) **Token-batching is slower, not faster**: folding 30 entries onto one expert (`gu_1grp30`, 1.4725) is 3x the port's per-dispatch form (0.4932) and the same as the single-window launch — the port's grouping is already the efficient shape. (b) **The empty launch is 65% of the gu call and 49% of the down call** (7 × 0.3168 = 2.22 of 3.85; 7 × 0.1606 = 1.12 of 2.28). It is not the kernel — the shader exits at `grp_win[g] != win_id` on its first load — it is the **workgroup launch itself**.

### 3. THE EMPTY LAUNCH, DECOMPOSED BY THE ONE KNOB THAT SIZES IT

`gy` (the caller's `grid_groups`) is how many groups a launch covers side by side; the shaders stride `g += gl_NumWorkGroups.y`, so results are invariant and only the WORKGROUP COUNT moves. Swept in the same process (`gy` = 1, 2, 4, 8, 16, 30):

| gy | gu work | gu empty | **gu per call** | dn empty | dn work |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.5883 | 0.1092 | **2.3527** | 0.0619 | — |
| 2 | 1.5807 | 0.1156 | 2.3898 | 0.0638 | — |
| 4 | 1.5804 | 0.1317 | 2.5022 | 0.0688 | — |
| 8 | 1.5983 | 0.1575 | 2.7011 | 0.0795 | — |
| 16 | 1.6375 | 0.2106 | 3.1116 | 0.1075 | — |
| 30 (the caller's) | 1.6912 | 0.3167 | **3.9082** | 0.1606 | 1.1825 |

An empty launch fits **~0.102 ms FIXED + ~2.8 ns per workgroup** (2,560 workgroups at gy=1, 76,800 at gy=30). The fixed term is the same species as this record's **F = 67 µs per submit**; the per-workgroup term is the launch throughput. **gy=1 is optimal on both sides** — it is also the fastest WORKING launch (1.5883 vs 1.6912), because 2,560 workgroups each striding 30 groups beats 76,800 workgroups doing one each here. Per call the grouped path falls **3.91 → 2.35 ms (gu, -40%)** and the down side falls proportionally.

### 4. THE CHANGE (four lines, no shader, no binding, no dispatch-count change)

`vulkan/src/kernels/native_expert_grouped_vk.cpp`: `gy = min(the caller's request, STRATA_VK_GROUPED_GYMAX)` with the cap defaulting to **1** (`STRATA_VK_GROUPED_GYMAX=0` restores the caller's request exactly, so one binary A/Bs both). Nothing else moves: the same 8 windows are dispatched, the same descriptors bound, the same `ptr_to_off` rebase, the same shaders — only the y-extent of each launch.

### 5. THE ENGINE A/B (199-token and 8-token ids; each arm its own env, shader hashes and binary sha256)

`--spec 2 --prefill 256`, one config per invocation (`setsid nohup`, polled), `STRATA_VK_DISP_STAT=1` + `STRATA_VK_FLUSH_STAT=1` in every run. `gy0` is the **same changed binary** with `STRATA_VK_GROUPED_GYMAX=0`, i.e. the OLD grid — the control that separates the change from the rebuild:

| arm | binary | gy | segments | recorded disp | barriers | window sync ms/round | segment wait ms | µs/dispatch | decode tok/s | id |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base199` | HEAD | caller (30) | 42 | 59,111 | 71,005 | 260.427 | 3,786 | 64.05 | 8.34 | `56a0b28d2de6` |
| **`gy1_199`** | **changed** | **1** | 42 | 59,111 | 71,005 | **228.020** | **3,326** | **56.26** | **9.46** | `56a0b28d2de6` |
| `gy0_199` | changed | 0 → caller | 42 | 59,111 | 71,005 | 260.327 | 3,785 | 64.03 | 8.35 | `56a0b28d2de6` |
| `base8` | HEAD | caller | 57 | 73,126 | 21,826 | 195.334 | 3,863 | 52.83 | 8.18 | `3aed108cceee` |
| **`gy1_8`** | **changed** | **1** | 57 | 73,126 | 21,826 | **172.344** | **3,417** | **46.73** | **9.21** | `3aed108cceee` |
| `gy0_8` | changed | 0 → caller | 57 | 73,126 | 21,826 | 195.003 | — | — | 8.20 | `3aed108cceee` |

**The window's GPU time falls 12.5% (199-token) and 11.8% (8-token); decode rises 13.4% / 12.6%; the per-dispatch cost falls 64.05 → 56.26 µs and 52.83 → 46.73 µs. The dispatch count, the segment count and the barrier count are IDENTICAL — the change removed GPU EXECUTION, not dispatches, which is exactly the shape a launch-size fix should have.** The control arms (`gy0_*`) reproduce the HEAD built from a different binary to within 0.04% (260.427 vs 260.327; 195.334 vs 195.003), so the delta is the grid, not the rebuild.

### 6. TARGET 2 CLOSED: THE `fused_gr_*` GROUP IS PRICED, AND IT WAS NEVER THE 112.7 µs RESIDUAL

New arm `gr_pricing` at the artifact's geometry (N=2560 HC=4 LR=320, DEVICE_LOCAL, batch 8; `fused_gr_read_multi` is four dispatches per token):

| kernel | ms/dispatch | | chain | ms/token | ms/dispatch |
|---|---:|---|---|---:|---:|
| `fused_gr_rs` | 0.0147 | | `fused_gr_read_multi` T=3 (12 disp) | 0.1114 | 0.0279 |
| `fused_gr_down` | 0.0222 | | `fused_gr_read_multi` T=8 (32 disp) | 0.0955 | 0.0239 |
| `fused_gr_mix` | 0.0683 | | | | |
| `fused_gr_inject` | 0.0151 | | | | |

The four `fused_gr_*` groups are **2,304 of the window's 11,365 recorded dispatches (21%), and they price at 69.3 ms — 10% of the 687.6 ms the named histogram takes, NOT the 112.7 µs each (259.5 ms) the previous mixture model's residual assigned them.** **The residual was a missing row and it disappears when the row exists** (`/tmp/gap2/mixture_model2.py`, 99% of the histogram priced). The same rebuild also shows the engine-shaped gu/down rows OVER-subscribe the window (983.5 ms = 143%), i.e. the synthetic fixture's absolute scale overstates the engine's real per-call work (its 30 IQ3_XXS groups × 2,560 rows is heavier than the real mix) — **the RATIO across gy is what this fixture is for, and the absolute is not a window cost**; the engine A/B above is the authority for the absolute.

## THE 3-12x PER-DISPATCH "GAP" IS CLOSED: it was a comparison error — the engine's MIXTURE AVERAGE against the bench's LIGHT-kernel marginals — and the like-for-like measurement says the window costs what its own kernels cost, to within 11% (2026-10-06, `vega`, Arc Pro B70)

**THE ONE PARAGRAPH.** The queued brief called "replay 64.0 µs, live 152.5 µs against the bench's in-stream marginal 5-20 µs" **the largest unexplained number in the record**, and asked for the per-dispatch overhead that must cause it. There is no such overhead, and the comparison is the defect: **64.0 µs is an average over the verify window's whole kernel mixture, and that mixture contains ~2,300 weight-reading MoE GEMVs per round — a kernel family this bench had never measured.** Adding the missing row (`iq1m_mmvq`, the window's own IQ1_M format) prices that family at **99.5 / 111.0 / 119.2 µs** per dispatch at the engine's shapes in **device-local** memory — 5-24x the "5-20 µs" the average was held against, and squarely around the 64 µs average. A mixture model built only from measured bench rows prices **77% of the window's dispatches at 38.6 µs each**, leaving **112.7 µs** for the 23% it cannot price (2,304 of which are the `fused_gr_*` hyper-connection read group) — i.e. the window's 64 µs is exactly what a mixture of ~110 µs weight-reading GEMVs and ~3-9 µs elementwise kernels should cost. **The four candidate overheads were also tested one at a time, in ONE process, and all four are falsified**: a different pipeline every dispatch (identical work, identical grid, identical buffer), a device-local buffer, 128 descriptor targets spread over 1 GiB, and all three at once each move the per-dispatch marginal by **<20%** (3.7-4.4 µs against a 3.8 µs uniform baseline). The harness's own fixed cost is **F = 67 µs per submit and c = 3.3 µs marginal per dispatch**, measured by a 1→1,408 sweep — and the engine pays the *same* fixed cost: a single-dispatch live flush in the real run costs **70 µs** (`flush site 11`, n=2,364), which is the F = 67 µs the bench measures, in a different process, on a different instrument. **This is the same class of error as the pooled 1.20/0.30 barriers-per-dispatch: two phases merged into one ratio.** What the window's 64 µs is NOT is host submit time — the decode issues **42** submits for 59,111 dispatches, and the engine's own live-batch `vkQueueSubmit` is **4.6 µs**. A separate, real finding came out of the same counters and is reported as its own thing: the **expert-load** path spends ~12-14 s of host time in 16,114 small upload submits, and that one is **bandwidth-bound, not submit-bound** — a 2 GiB staged-upload probe is **flat at 1.75-2.02 GB/s across 1/4/16/64 MiB chunks** with a ~37 µs per-call overhead, so batching the submits cannot move it; the lever there is the 2.4x between the staged path (1.9 GB/s) and the mapped path (4.65 GB/s).

### 1. THE HARNESS ARM: one structural property at a time, in ONE process

New arm `dispatch_gap` (`ports/vulkan/bench/vk_bench.cpp`), Arc `intel_icd`, `reps 9 warmups 3`, every row the same instrument as the rest of the file (wall clock around a recorded-batch fence, median/batch). `scale` and `add` do the **same work at the same grid** (one f32 element per thread, n = 16384) and differ only in their shader binary and binding count, so "alt-shader" changes the pipeline and nothing else:

| row | what it moves | med µs/dispatch | batch |
|---|---|---:|---:|
| `gap_uniform` | K copies of `scale`, ONE mapped buffer (cold: first arm in the process) | 17.2 | 128 |
| `gap_altshader` | `scale` <-> `add`: a DIFFERENT PIPELINE every dispatch | 4.4 | 256 |
| `gap_devlocal` | K copies of `scale`, a DEVICE_LOCAL buffer (the arena type) | 3.8 | 128 |
| `gap_altbuffer` | `scale` on 128 views of ONE 1 GiB device-local buffer, 8 MiB apart | 3.9 | 128 |
| `gap_engineshape` | alt-shader + device-local + 128 rotating 1 GiB views, all three | 3.7 | 256 |

**FOUR HYPOTHESES, FALSIFIED.** Pipeline diversity, the arena's memory type, and descriptor-target address spread each move the per-dispatch marginal by **<20%**, and together by nothing (3.7 vs a 3.8 µs settled uniform). The one large number in the table is `gap_uniform`, which is **not** a hypothesis result: it is the **first arm in the process**, and the same configuration re-measured later (`gap_batch K=128`) reads **3.8 µs** against its 17.2 µs median (min 4.8). That 4.5x order effect is the harness's own cold regime — the same confound `gdn_step_probe` documented — and it is why the first row of any process here must not be quoted.

**THE FIXED/MARGINAL SPLIT, from the same process** (`gap_batch`, one command buffer, K dispatches of `scale`):

| K | 1 | 2 | 8 | 32 | 128 | 512 | 1,408 |
|---|---:|---:|---:|---:|---:|---:|---:|
| µs/dispatch | 70.3 | 36.9 | 11.7 | 5.4 | 3.8 | 3.4 | 3.3 |

`per-dispatch = F/K + c` fits every row: **F = 67 µs per submit, c = 3.3 µs marginal per dispatch.** The engine's recorded segments carry ~1,400 dispatches each, so F is worth 0.05 µs/dispatch there — **the fixed cost is not the replay's 64 µs, and no submission-granularity change can be.** And the engine pays the SAME F: `hist199`'s flush-site table has a site with **n=2,364 flushes of exactly ONE dispatch each, wait 165 ms = 70 µs per single-dispatch submit** — the bench's F, measured inside the engine, in a different process.

### 2. THE MISSING ROW: the window's own kernel family, priced

The bench had rows for elementwise kernels and for IQ2_S, and **no row for the format the window's dominant kernels read**. New arm `iq1m_mmvq` (IQ1_M, `n_in=2560`, one workgroup per row, the engine's gate/up `n_out=1280` and down `n_out=2560` shapes), batch 8, `reps 9`:

| row | wbytes | med µs/dispatch | GMAC/s |
|---|---:|---:|---:|
| `iq1m_mmvq/dev` n_out=2560 ncols=1 | 1,433,600 | **119.2** | 55.0 |
| `iq1m_mmvq/mapped` n_out=2560 ncols=1 | 1,433,600 | 136.0 | 48.2 |
| `iq1m_mmvq/mapped` n_out=1280 ncols=1 | 716,800 | **111.0** | 29.5 |
| `iq1m_mmvq/mapped` n_out=512 ncols=1 | 286,720 | 99.5 | 13.2 |
| `iq1m_mmvq/dev` n_out=1280 **ncols=3** | 716,800 | **51.8** | **189.7** |
| `iq2s_mmvq` n_out=2048 (the previously smallest published row) | – | 46.4 | 112.3 |

**99.5-136 µs, not 5-20 µs.** Device-local (the engine's arena type) is *faster* than the mapped type here, 119.2 vs 136.0 — the opposite of what "the arena is the difference" would predict, and consistent with §1's `gap_devlocal`.

### 3. THE MIXTURE MODEL: what the window SHOULD cost, from measured rows only

`tools`-side arithmetic (`/tmp/gap/mixture_model.py`, printed to `/tmp/gap/mixture_model.out`) takes the engine's `vk disp stat RECORDED arm` histogram — the replay's own composition — and prices each entry against a named bench row (measured / proxy / UNPRICED, no guessing):

* the window's **weight-reading GEMVs alone** (`native_gu_any` 1,152 + `native_down_any` 1,152 + `native_k_mmvq` 633 + `iq4nl_mmvq` 141 + `iq4xs_mmvq` 126 = **3,204 dispatches, 30% of the count**) price at **365.1 ms of the 687.6 ms** the window's 10,743 named dispatches take — **53% of the time from 30% of the dispatches**, all in the 99.5-136 µs band judged above;
* the harness can price **77%** of the named mixture; those alone average **38.6 µs/dispatch**;
* the residual 2,424 UNPRICED dispatches (2,304 of them the four `fused_gr_*` hyper-connection kernels, which have no bench row) must carry **112.7 µs each** to reach the engine's measured 64.0 µs average — i.e. they are the same class of kernel as the ones already priced.

**So the 64.0 µs average is what this mixture costs, to within ~11%.** There is no residue for a per-dispatch overhead to occupy. The 152.5 µs live-prefill average is the same statement over a different mixture (the prefill's own histogram is `iq_dequant_f32` 10,767 + `gemm_prefill_fma_small` 7,166 + `gemm_prefill_f16_m8` 3,840 + … , and this file already carries GEMM rows of **0.10-9.0 ms** at the engine's shapes).

### 4. THE MATCHED SUBMIT COMPARISON (a defined operation, three paths, one process)

The brief asked whether the per-dispatch overhead is host-side submit. It is not, and the engine's own counters separate the three submit paths cleanly:

| path | submits | host time in `vkQueueSubmit` | per submit | source |
|---|---:|---:|---:|---|
| live batch (the prefill's flushes) | 3,021 | **14 ms** | **4.6 µs** | `gap2_base199` `vk flush stat` |
| recorded segment (the decode window) | 42 | **1 ms** | **24 µs** | `gap2_base199` `vk disp stat by arm` |
| transfer (the expert load, one-shot) | 16,114 | ~14.4 s | **~0.89 ms** | `gap2_base199` `vk disp stat ms` |

The decode issues **42** submits for 59,111 dispatches of GPU execution — **3,783 ms of fence wait at 64.0 µs per dispatch**. Host submit time in the decode is **1 ms for the entire run**. The bench's own recorded submit is bounded by its K=1 replay total (70.3 µs, submit + wait), i.e. **no larger than the engine's**. **The 3-12x is not a submit-path problem.**

### 5. THE ONE REAL FINDING ON THAT PATH — AND IT IS NOT FIXABLE BY BATCHING

The 0.75-0.89 ms per submit lives **entirely in the `transfer` arm — the expert load**, 16,114 one-shot staged uploads of ~4.16 MB each (~53.6 GiB), which are ~12-14 s of host submit + ~18.5 s of wait of a ~30 s cold start. **It is a cold-load cost with a different cause from the decode's, and it is not merged with it here.** The obvious fix is to submit fewer, larger copies, so it was measured rather than assumed (new `ports/vulkan/tools/probe_submit.cpp`, Arc `intel_icd`, 2 GiB staged into a device-local buffer):

| chunk | submits | total | per call | achieved |
|---:|---:|---:|---:|---:|
| 1 MiB | 2,048 | 1,224.6 ms | 597.9 µs | **1.75 GB/s** |
| 4 MiB | 512 | 1,062.4 ms | 2,074.9 µs | **2.02 GB/s** |
| 16 MiB | 128 | 1,176.1 ms | 9,188.7 µs | **1.83 GB/s** |
| 64 MiB | 32 | 1,149.9 ms | 35,934.2 µs | **1.87 GB/s** |
| (mapped host memcpy, 16 MiB) | – | 461.4 ms | – | **4.65 GB/s** |

**The total is FLAT across a 64x change in chunk size: the cost is bytes, not calls.** Fitting `per-call = F + bytes/B` gives **F ≈ 37 µs** and B ≈ 1.87 GB/s — so the per-call overhead in a clean process is 37 µs, not 750. **Batching the uploads into larger submits cannot move the load.** The engine's own transfer arithmetic agrees: 4.16 MB per call at (0.89 ms submit + 1.15 ms wait) = **2.0 GB/s**, against this probe's 2.02 GB/s for the same 4 MiB call — the same path at the same rate. **The lever on the load is therefore the 2.4x between the STAGED path (1.9 GB/s) and the MAPPED path (4.65 GB/s), not the submission count.** And the follow-up was attempted rather than left as a suggestion: `alloc_staging` was pointed at the host-visible **device-local** type for one arm (`gap5_bar199`, 199-token, same binary, same token list) and **the port's own account rule refuses it before the first byte** —

```
vk_compute: REFUSING a 256.00 MiB staging buffer - 28560.00 MiB in the VRAM account, 28589.00 MiB usable
RUN_RC=3, ARM_VERDICT decoded=0   (/tmp/gap/gap5_bar199.log)
```

`alloc_staging` charges by HEAP, the arena already holds 28,560 of 28,589 MiB usable, and a staging buffer in VRAM is model-resident memory by that rule. **That is the contract working, not a bug, and it is reported as a CONFLICT rather than worked around.** The change a next batch would need is a *policy* one — exempt transient staging from the VRAM account, or leave arena headroom — and it is deliberately not made here. (The arm's `decoded=0` is exactly the failure mode the new `ARM_VERDICT` line exists to catch: a run that dies at startup still exits 0 for a while.) The upload path is unchanged; what is measured and quotable is the **4.65 vs 1.9 GB/s** difference itself, and the probe that separates it.

### 6. THE NAMED LEVER: FEWER DEPENDENT PAIRS — MEASURED, AND SMALLER THAN IT LOOKS

The brief's lever was the replay's long chain of small dispatches. The fusion it names — `quantize_q8_1` (1,875) -> `swiglu_f32` (1,296) — is priced here at **17.1 + 11.8 ms of the window's 687.6 ms = 4.2%**, so fusing that pair correctly is worth **at most ~4%** of the window. **The measured large version of "fewer dependent pairs" is not fusion but BATCHING TOKENS PER WEIGHT-READING DISPATCH**, and the new row prices it: **`iq1m_mmvq` n_out=1280 at `ncols=3` is 51.8 µs against 111.0 µs at `ncols=1`** — 3x the work for **half** the per-dispatch cost, **6.4x** the throughput per dispatch (189.7 vs 29.5 GMAC/s). Applied to the 2,304 `native_gu_any`/`native_down_any` dispatches, that is the difference between **265 ms and ~124 ms** of the window. It is a kernel change to the grouped expert GEMV (the engine's own grouping, not the barrier), it is NOT attempted here, and it is the honest successor to this batch.

### GATE, IDS, LOGS

Arc `intel_icd` **895 passed / 0 failed / 0 skipped** — the count did NOT fall and nothing was skipped (`/tmp/gap/gate_final.log`). Ids: **`56a0b28d2de6`** (199-token, `gap2_base199`, decoded 32 tokens) and **`3aed108cceee`** (8-token). The bench-only changes in this batch touch no engine source, so both are guards rather than expectations.

Logs, exact: `/tmp/gap/bench_gap_intel.log` (the `dispatch_gap` arm: uniform / alt-shader / device-local / alt-buffer / engine-shape / the 1→1,408 batch sweep — ONE process), `/tmp/gap/bench_iq1m_intel.log` (the `iq1m_mmvq` rows + `iq2s_mmvq` re-run, one config per invocation), `/tmp/gap/mixture_model.py` + `/tmp/gap/mixture_model.out` (the model, per-entry price and source), `/tmp/gap/probe_submit.log` (the chunk sweep), `/tmp/gap/gap2_base199.log` (199-token ids + the flush/dispatch counters), `/tmp/gap/gap3_base8.log` (8-token id), `/tmp/gap/gap_base199.log` (**a FAILED arm, kept on purpose**: the token list was a single id, it decoded **0** tokens and still exited 0 — which is why the driver now writes `ARM_VERDICT … decoded=N` and an arm with 0 is a failure, not a result), `/tmp/gap/gate_final.log` (the gate), `/tmp/gap/gap5_base199.log` (the staging A/B baseline, same binary), `/tmp/gap/gap5_bar199.log` (the BAR-staging arm: **REFUSED by the VRAM account**, 0 dispatches).

### NOT DONE / DELIBERATELY LEFT

(i) **The `fused_gr_*` group is UNPRICED** — 2,304 dispatches, 21% of the window's count, carrying 112.7 µs each in the model's residual. A bench row for those four kernels would close the last 23% of the model; it is not attempted. (ii) **`swiglu_f32` has no bench row** and is priced at the `quantize_q8_1` elementwise row as a proxy. (iii) **The token-batching lever in §6 is measured in the harness, NOT in the engine** — whether `native_gu_any`/`native_down_any` can carry `ncols>1` for the entries the window actually routes is unverified, and no engine change was made. (iv) **The engine's ~0.89 ms `vkQueueSubmit` inside the transfer path is measured, not attributed**: this probe's per-call overhead is 37 µs in a clean process, so the 0.89 ms is a property of the engine's live submission state (the 27 GiB arena's live BO set) that no measurement here separates from the copy itself. (v) The load-time `submit` figure varies across runs (**9.1 / 10.3 / 14.4 s**) and is reported as a range, not a constant. (vi) The `gap_uniform` 17.2 µs is an ORDER effect, not a structural result, and is reported as such.


## THE VERIFY WINDOW'S WAIT STRUCTURE, MEASURED: the 42 segment submits are worth **1 ms for the whole run**, collapsing them to ONE command buffer per window is worth **0.05%**, and the replay's cost is a **faulted dispatch count** — with the chain barrier priced (upper bound) at **~35% of the per-dispatch time** by an arm whose answer is WRONG (2026-10-06, `vega`, Arc Pro B70)

**THE ONE PARAGRAPH.** The queued brief asked for the verify window's waits to be collapsed, from "~57 segments, each ending in `vkSubmit` -> `vkWaitForFences`", against a baseline of `sync 240.623 ms/round`. Measured first, on this card with the same `coder-iq1_m` pack: the whole run has **42 segment submits** (14 rounds x 3 — the window is cut in **two** by its single host boundary, plus **one** for the commit graph), they cost **1 ms of submit in total**, and the `sync` the brief wants to remove is **260.2 ms/round of fence wait that the GPU is executing**, not a round trip: 59,111 recorded dispatches execute for **3,783 ms** of wait. **The change was built anyway** (`STRATA_VK_WINDOW_ONE_CB=1`: the driver publishes the PLE rows before the window's launch, so the recorded window needs no host boundary and is ONE command buffer) and it is **neutral**: segments 42 -> **28**, submit 1 -> 0 ms, segment wait **3,783 ms unchanged**, window `launch+sync` 265.44 -> **265.41 ms/round (0.05%)**, decode 8.36 -> 8.31, ids `56a0b28d2de6` **unmoved**. It ships **opt-in** because a default is a claim and this is not a win. What the counters then say, and it is the same conclusion the port reached in the previous batch by a different route: **the replay's cost is ~64.0 µs per recorded dispatch** (3,783 ms / 59,111) against an in-stream marginal cost of 5-20 µs in the port's own bench, the **submits are 1 ms**, and **the chain barrier is the one thing in that path that removal does improve — by ~35% per dispatch, in an arm that produces the WRONG answer** (`5c30ca20`), which is why it is a bound and not a candidate. The structure is exhausted; the residue is the shaders' own execution.

### 1. THE STRUCTURE, PRICED BEFORE IT WAS TOUCHED (and it is not the structure)

Every arm below runs the 199-token prompt (`--spec 2 --prefill 256 --max-new 32 --max-context 512`, the same arena, the same expert profile), **one config per invocation**, each arm logging its own `env | grep -i strata_`, the sha256 of the shaders it will load and the sha256 of the binary, all detached with `setsid nohup`, and the two guarded ids are checked as the md5 of the engine's own `output  :` line:

| arm | configuration | segments | recorded dispatch | chain barriers | segment submit | segment wait | window `launch`/`sync` | decode tok/s | ids md5 |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `base199` | HEAD, `--spec 2` | 42 | 59,111 | 71,005 | 1 ms | 3,783 ms | 5.276 / 260.166 | 8.33 | `56a0b28d2de6` |
| `hist199` | + `FLUSH_STAT=1` | 42 | 59,111 | 71,005 | 1 ms | 3,784 ms | 5.297 / 260.237 | 8.36 | `56a0b28d2de6` |
| `onecb199` | + `STRATA_VK_WINDOW_ONE_CB=1` | **28** | 59,111 | 71,005 | **0 ms** | **3,783 ms** | **265.411 / 0.000** | 8.31 | `56a0b28d2de6` |
| `base8` | 8-token prompt, `--spec 2` | 57 | 73,126 | 21,826 | 1 ms | 3,859 ms | 4.127 / 195.064 | 8.20 | `3aed108cceee` |
| `nobarrec199` | + `STRATA_VK_NOBARRIER_REC=1` | 51 | 65,372 | 66,724 | 1 ms | **2,699 ms** | 3.058 / 154.926 | 11.69 | **`5c30ca20` — WRONG** |
| `nobar199` | + `STRATA_VK_NOBARRIER=1` | **did not run — the prefill aborts** (`prefill: routed id out of range`, `RUN_RC=1`) | | | | | | | |

* **The brief's "~57 segments" is the 8-token arm's whole-run count** (19 rounds x 3), and the 199-token arm's is 42 (14 rounds x 3). Per round it is always **three**: the window is cut into **two** command buffers by its **one** all-resident host boundary (the PLE `wait_flag_ge` at layer 1, `verify.cpp:638`), and the per-round commit graph is the third. They are **not** 57 tiny submissions — each carries ~1,407 dispatches.
* **The submit cost is the instrument's own answer and it is 1 ms for the entire run.** `vk disp stat by arm: ... segment 42 (59111 recorded dispatches, ... submit 1 ms, wait 3783 ms)`. The `wait` is the fence, and the fence is the GPU running the recorded dispatches: `sync 260.166 ms/round` x 14 rounds = 3,642 ms against the segment path's **3,783 ms** total. Nothing is idle at a boundary that could be reclaimed.
* **`STRATA_VK_NOBARRIER=1` cannot price the replay's barriers at all**, and the log says why on its own line: the **prompt** path needs them — the engine's own guard fires (`prefill: routed id out of range`, `/tmp/em/nobar199.log:52`) with only **6 live-batch flushes** done, i.e. it dies inside the prefill, before a single decode round. It is recorded as **did not run**, not as a number. (`vk_compat`'s 640 ms GuC line is printed by **every** run — it is the startup note, `base199.log` has it too — and it is NOT this arm's failure.)

### 2. THE ONE-COMMAND-BUFFER WINDOW, BUILT AND MEASURED: 42 -> 28 segments, 0.05% of the round

`STRATA_VK_WINDOW_ONE_CB=1` (`src/core/verify.cpp`) moves the **host** half of the PLE — the table gather — to **before** the window's launch, so the recorded window needs no handshake at all: the recorded `copy_from_mapped` binds the mapped region's BUFFER (`elementwise_vk.cpp:328-334`) and re-reads what the host published before the submit. It is the technique the engine already uses for the captured per-layer block (*"the driver calls `ple_stage_token` once per token, BEFORE the graphs"*, `layer.cpp:1186`). Segments **42 -> 28** (3/round -> 2/round: window 1, commit 1), recorded dispatches **59,111 unchanged**, segment submit **1 -> 0 ms**, segment wait **3,783 ms unchanged**, and the round's window cost is the same **265.41 vs 265.53 ms**: the `sync` did not disappear, it moved into `launch`, because `submit_segment` waits for its fence and there is now only one segment. Ids `56a0b28d2de6` on both sides of the change. **It is neutral, it ships opt-in, and it is reported as neutral.**

### 3. THE REPLAY'S PER-DISPATCH COST, AGAINST THE LIVE PATH, FROM THE PORT'S OWN COUNTERS

| path | dispatches | wall in the path | per dispatch | source |
|---|---:|---:|---:|---|
| **replay (decode)** | 59,111 over 14 rounds | 3,783 ms of segment fence wait | **64.0 µs** | `disp stat` segment row, `base199` |
| replay, 8-token arm | 73,126 over 19 rounds | 3,859 ms | **52.8 µs** | `base8` |
| **live batch (prefill)** | 59,640 in 3,021 flushes (19.7/flush) | 9,096 ms | **152.5 µs** | `flush stat`, `hist199` |
| replay, barriers elided (WRONG output) | 65,372 | 2,699 ms | **41.3 µs** | `nobarrec199` |
| bench, in-stream marginal | one dispatch in a 128-dispatch replay | — | **5-20 µs** | `ports/vulkan/bench` |

* **The replay is 2.4x CHEAPER per dispatch than the live prefill path**, so there is no replay-specific anomaly: both arms sit ~3-12x above the bench's in-stream marginal, and that gap is a property of the engine's real one-shot arena access against the bench's L2-hot harness buffers — the same unexplained factor the `gdn recurrence` batch named (**0.17 ms/dispatch** in the engine against **0.0199** in the bench).
* **IT IS NOT THE BARRIER COUNT.** The recorded arm carries **exactly one chain barrier per recorded dispatch** in every arm (`11,365/11,365` at 199 tokens, `7,084/7,084` at 8). The pooled figures that look like 1.20/dispatch (`71,005 / 59,111`) and 0.30/dispatch (`21,826 / 73,126`) are pooling the **live prefill's** barriers with the replay's — `71,005 = 59,640 live + 11,365 recorded` and `21,826 = 14,742 live + 7,084 recorded`, exactly. A 4x difference in the pooled ratio is the prefill's size, not the replay's structure. The one-CB arm keeps the same 71,005.
* **THE BARRIER IS, HOWEVER, THE ONE THING IN THE REPLAY THAT REMOVAL MOVES — by ~35% per dispatch.** `STRATA_VK_NOBARRIER_REC=1` (new, measurement-only) elides the chain barrier **only in recorded steps** (`fresh_set`, i.e. `record_dispatch`), so the live prefill keeps its barriers and the run reaches the decode. The wait per recorded dispatch falls **64.0 -> 41.3 µs**, the window's `launch+sync` falls **265.4 -> 158.0 ms/round (-40%)**, decode 8.33 -> **11.69 tok/s** — **and the output id is `5c30ca20`, not `56a0b28d2de6`.** The arm also did **more** work (4,669 dispatches/round against 4,222) with a different trajectory, so **the 35% is an UPPER BOUND from an arm that produces wrong output, not a price a correct variant can claim.** Statement to keep: removing the barriers buys ~40% and breaks the answer. The port's only *correct* barrier experiment (`STRATA_VK_BARRIER_HAZARD=1`) left the barrier count at 21,825 of 21,826 and moved `sync` by **1.2%** — because in this arena the bound regions DO overlap, so the hazards are real.
* **LIKE-FOR-LIKE AGAINST UPSTREAM, CORRECTED.** Upstream's `verify window ... commit 3.858 ms/round` is ONE sub-step of their round, not their round: at 73.4 tok/s and 2.91 tokens/round their round is **~40 ms**. This port's is **~265 ms/round** — **~6.6x, not 62x** — and with the barriers removed (upper bound, wrong output) it would still be **~158 ms/round, ~3.9x**. **The remaining ~3.9x is the shaders' own execution time inside the window, not its submission structure, its submit count, or its segment count.**

### 4. WHERE THE WINDOW'S DISPATCHES ACTUALLY ARE (new instrument: the RECORDED arm's composition)

`STRATA_VK_DISP_STAT`'s `by shader` line pools the one-shot prefill with the replayed window, so the port could not see what the replay is *made of*. `vk disp stat RECORDED arm` (new, `vk_compute.cpp`) counts the dispatches **encoded into recorded steps**; because each recorded command buffer is re-executed once per segment submit, **its composition IS the replay arm's per-round composition**. 11,365 encoded dispatches (all captured window sizes plus the commit graphs), top of the list: `quantize_q8_1` **1,875**, `swiglu_f32` **1,296**, `native_down_any` **1,152**, `native_gu_any` **1,152**, `native_k_mmvq` 633, `fused_gr_rs`/`fused_gr_down`/`fused_gr_mix`/`fused_gr_inject` **576 each**, `bf16_mmvf_f32` 318, `router_top10_f32` 288, `f32_to_bf16` 150, `gdn_step_norm_multi`/`resident_plan`/`native_moe_combine`/`scalar_gate_f32`/`scale_rows`/`ptr_to_off` 144 each. **The replay is a long chain of small, latency-bound dispatches** — activation quantize/silu and the four-kernel hyper-connection read group are ~40% of its dispatch count — and **1.20-ish barriers per dispatch is 1 per dispatch**, so the remaining structural idea is *fewer dependent pairs* (fuse or batch the per-row kernels), not conditional barriers.

**GATE + IDS.** Arc `intel_icd` **895 passed / 0 failed / 0 skipped** (the count did NOT fall, nothing skipped); lvp `879/0/4` (the four documented skips); smoke `60/0/0`. Ids **`56a0b28d2de6`** (199-token) in `base199`, `hist199` and `onecb199`, and **`3aed108cceee`** (8-token) in `base8` — the two 11,365/73,126-arm checks that the one-CB change did not move the answer. Logs, exact: `/tmp/em/base199.log` (baseline), `/tmp/em/hist199.log` (flush attribution + the RECORDED-arm composition), `/tmp/em/onecb199.log` (the one-CB window), `/tmp/em/base8.log` (the 8-token id guard), `/tmp/em/nobarrec199.log` (the barrier bound, wrong ids), `/tmp/em/nobar199.log` (did not run), `/tmp/em/gate_final.log` (the gate).

**NOT DONE / DELIBERATELY LEFT.** (i) **The measure of a *correct* cheaper barrier is not made.** `nobarrec` bounds the chain barrier at ~35% of the replay's per-dispatch time but changes the answer; a candidate would have to re-run with a *different barrier form* (a split `vkCmdSetEvent`/`vkCmdWaitEvents`, or a narrower stage pair) and hold both ids, and that experiment is not attempted. The port's own note stands: on Vulkan a strict serial chain still needs an execution dependency between consecutive dispatches, so the plausible correct lever is **fewer dependent dispatches**, not a cheaper barrier. (ii) **The taller coopmat row block and the opt-in tiled FMA GEMM are still not attempted** (both were named, neither was measured). (iii) The `nobarrec` arm's 11.69 tok/s is **contaminated** by its own different trajectory and is reported as a bound, never as a rate. (iv) The **~3-12x gap between the engine's per-dispatch cost and the bench's in-stream marginal cost is still unexplained**, and it is now measured on **both** paths (replay 64.0 µs, live 152.5 µs, bench 5-20 µs); it is the largest single number in this file. (v) The **PCIe probe disagreement** — this port's own probe reads **2.0 GB/s** host->device in `hist199` (`pcie_frac 0.05`) where the SYCL port measured 6.6-7.0 GB/s on the same card — is a fact about the instruments, not about the card, and is not chased. (vi) The MTP draft path is **not** enabled: see the queued brief's target 1 and `NEXT.md`.

## THE STEP KERNEL'S SERIAL WALK, MEASURED AND THEN UNROLLED BIT-EXACT: 2.46x/2.73x on the kernel, **3.40%** on the prefill — and the falsification of "the phase is the step kernel" (2026-10-06, `vega`, Arc Pro B70)

**THE SHAPE, MEASURED BEFORE IT WAS TOUCHED.** New bench arm `gdn_step_probe`, intel ICD:

```
PROBE gdn_step shape | S=128 h_k=16 h_v=48 (3 MiB state) | grid = 24 workgroups x 256 lanes = 6144 threads |
card = 32 Xe2 cores x 128 = 4096 fp32 lanes (B70/BMG-G31)
```

* **The card** (from `vulkaninfo` and `lspci`, not assumed): `Intel(R) Graphics (BMG G31)`, PCI `8086:e223`
  `Battlemage G31 [Arc Pro B70]`, **32 Xe2 cores / 4,096 fp32 lanes**, **24 MB L2**, 256-bit GDDR6 at
  **608 GB/s**, subgroup 32.
* **The ceiling against the phase's measured row.** 6,144 threads is **1.5 waves** of 4,096 lanes, and at one
  workgroup per core **8 of 32 cores get no workgroup**. The card's lane rate is ~8.2 TFMA/s at ~2 GHz; the
  kernel's own row is **55 GMAC/s = 0.7% of it**. Not ALU-bound.
* **The memory floor.** 3 MiB state read + 3 MiB written (pass 2's re-read hits L2: 3 MiB in 24 MB) = 6 MiB at
  608 GB/s = **10.3 µs**; measured **61.8 µs** with the engine's real footprint — **6x off bandwidth**, i.e.
  latency-bound.
* **The footprint, isolated in one process:** `native_gdn_step hot128` **0.0345 ms** against `cold36host`
  (36 distinct 3 MiB states cycled, 108 MiB) **0.0619 ms** — **1.79x**. The same 36 states in DEVICE_LOCAL
  memory read 0.0636 ms, so the allocation TYPE is not the difference; the footprint is. The probe also caught
  an instrument artifact worth keeping: a batch-8 row read **0.2641 ms** the FIRST time it ran in the process
  and **0.0428 ms** twice later, which is the whole explanation of `gdn_step_pair`'s 0.1072 against the batch
  sweep's 0.0427.

**THE FIX AND ITS THREE-IMPLEMENTATION PROOF.** `native_gdn_step.comp` walks its row axis in chunks of `KU`
(`#define KU 16`), ONE accumulator per pass, terms added in the same ascending `i` order. The new
`gdn_step_unroll` arm builds KU=1 (rolled), KU=8 and SHIPPED from the SAME source (`run_bench.sh`) and compares
STATE and `o` BITWISE on identical inputs, naming the first mismatching index:

| build (batch 128) | Arc hot | Arc cold36 | lvp hot | lvp cold36 | radeon hot | radeon cold36 |
|---|---:|---:|---:|---:|---:|---:|
| KU=1 (rolled) | 0.0330 | 0.0603 | 0.3827 | 0.5939 | 0.4340 | 0.4747 |
| KU=8 | 0.0148 | 0.0231 | 0.2554 | 0.4584 | 0.4312 | 0.4714 |
| **KU=16 (SHIPPED)** | **0.0134** | **0.0221** | 0.2580 | 0.4574 | 0.4320 | 0.4704 |
| shipped/rolled | **2.46x** | **2.73x** | 1.48x | 1.30x | 1.005x | 1.009x |

`BITEXACT ... 0/786432 differ` and `0/6144 differ` on **intel, lvp AND radeon**. The check is demonstrated to
fail: putting the legacy `gdn_step.spv` (same rule, different rounding order) in the reference slot gives
`615347/786432` and `6144/6144` differs with the index named. **A parallel scan was considered and rejected** —
associative in exact arithmetic is not associative in f32, and this recurrence is chaotic, so a scan changes the
answer.

**THE ENGINE A/B — AND WHY IT IS A FALSIFICATION.** 199-token prompt, `--spec 2 --prefill 256`, n=3 per arm
interleaved, one config per invocation, each arm logging its own env and the sha256 of the shader it placed:

| arm | mode | prefill ms | tok/s | `gdn recurrence` ms | ids md5 |
|---|---|---|---:|---:|---:|---|
| `b1,b2,b3` | base (rolled) | 9,940.6 / 10,072.5 / 10,181.9 | 20.02 / 19.76 / 19.54 | 2,677 / 2,678 / 2,670 | `56a0b28d2de6` |
| `n1,n2,n3` | KU=8 | 9,742.9 / 9,884.1 / 9,765.7 | 20.43 / 20.13 / 20.38 | 2,552 / 2,540 / 2,555 | `56a0b28d2de6` |
| `x1,x2` | **KU=16 (SHIPPED)** | **9,736.3 / 9,723.9** | **20.44 / 20.47** | **2,537 / 2,547** | `56a0b28d2de6` |

Medians: prefill **10,072.5 -> 9,730.1 ms (3.40% less time)**, tok/s **19.76 -> 20.45 (+3.5%)**, `gdn recurrence`
**2,677 -> 2,542 ms (-5.0%)**; the ranges **do not overlap**. **THE FALSIFICATION:** a 2.7x kernel speedup saving
135 ms of a 2,677 ms phase puts the kernel's own cost at ~**212 ms (~8%)**; the other **~92%** is the engine's
per-dispatch cost in the recurrence's 128-dispatch batches (**0.16-0.17 ms/dispatch**), which the harness does
not reproduce (**0.0221 ms/dispatch** cold, 7.6x lower). **The step kernel's serial walk is not the phase's
bottleneck.**

**UPSTREAM (READ-ONLY).** `0c86bbec`'s `rg == 0` shuffle fix and `ae3b249f`'s double-buffered multi were read:
**our port does not carry that bug** (subgroup ops are banned; every barrier sits outside the lane-conditional
— `common/wg_reduce.glsl:59-65` and the four GDN shaders' line numbers above). The live `fused_gdn_ab entry`
radeon failure is the documented RADV intermittent, not that bug: the case compares two dispatches of the SAME
`.spv`, so an in-kernel reduction bug could not separate them, and four radeon repeats of the full gate gave
**6 / 3 / 1 / 2 failures with a moving set**. Left documented, not speculatively patched.

**GATE + IDS.** Arc `intel_icd` **895/0/0 — the count did NOT fall, nothing skipped**; `lvp_icd` 879/0/4;
`radeon_icd` 882/2/2 (both in the documented moving-failing-set family); smoke 60/0/0; ids `56a0b28d2de6` in all
eight arms. Logs: `/tmp/gdnsweep/{b1,b2,b3,n1,n2,n3,x1,x2}.log`, `/tmp/gdnsweep/driver.log`,
`/tmp/gdnsweep/gate_final.log`, `/tmp/gdnsweep/radeon_rep{1,2,3,4}.log`, `/tmp/gdnsweep/{probe,unroll}.log`.

## THE `gdn recurrence` PHASE, MEASURED, AND THE MEASUREMENT FALSIFIED THE OBVIOUS FIX: it is GPU-bound and LATENCY-bound, but it is NOT dispatch-count-bound — so the port's own fused kernel stays an OPT-IN 3.7%-of-the-phase win, and the wrapper finally has a gate arm (2026-10-06, `vega`, Arc Pro B70)

**THE ONE PARAGRAPH.** The phase was measured before it was touched, and it is **2,675 ms = 27.3% of a 9,786 ms GPU timeline** — the largest phase this port owns — made of **`native_gdn_step` 7,164 + `native_gdn_out_norm` 7,164 dispatches** (199 tokens x 36 GDN layers) plus `f32_to_f16` 3,644. It is **NOT host blocking**: the fence wait charged to the flush batches whose trigger sits inside `prefill::gdn_recurrence` is **2,651 ms of that phase over 15,616 dispatches**, while the host's ENTIRE prefill costs ~55 ms of encode and 15 ms of submit. It is **latency-bound, not throughput-bound** — a strict serial chain (state[t] <- state[t-1]) of 24-workgroup dispatches whose own bench row is 55 GMAC/s. So the obvious fix is FEWER DEPENDENT DISPATCHES, and the port already had the fused form of exactly this pair. **It was built, A/B'd in the engine, and the A/B falsified the hypothesis**: fusing the pair (2 dispatches -> 1, 14,328 -> 7,164) moved the phase **2,674 -> 2,574 ms (medians, ranges disjoint, ids identical in all six arms)** — **3.7% of the phase, not the ~50% a dispatch-count-bound phase would owe** — because `native_gdn_step` carries ~90% of the pair's cost and the fusion removes the other kernel. The end-to-end prefill cannot even resolve that much: a separate session's plain arms are **10,126.2 / 9,829.1 ms** on the fused path against **10,046.6 ms** on the chain. So `STRATA_PF_GDN_REC_FUSED=1` ships as **opt-in**, the chain stays the default exactly as `STRATA_VK_PREFILL_TILED` did, and the real target is now named: the step kernel's own serial walk, which the bench cannot justify changing.

### 1. WHAT THE PHASE IS MADE OF (before touching it)

199-token arm, `--spec 2 --prefill 256`, one config per invocation, `STRATA_PREFILL_TIMING=1` + `STRATA_VK_FLUSH_STAT=1` + `STRATA_VK_DISP_STAT=1` in the same run (`/tmp/gdn/r1_full.log`):

```
strata prefill timing: 199 tokens, GPU timeline 9786 ms, wall 9810 ms, host staging 326 ms:
  ... dequant 1890 (19.3%) | gemm gate/up 715 (7.3%) | gemm down 633 (6.5%) | host grouping 1922 (19.6%) |
  gdn recurrence 2675 (27.3%) | ...
vk flush stat: 3030 live-batch flushes, 59870 dispatches, submit 15 ms, wait 9300 ms
```

* **Which kernels.** The histogram is the proof, not a reading of the source: `native_gdn_step.spv 7164`, `native_gdn_out_norm.spv 7164`, `f32_to_f16.spv 3644`. One pair per token per GDN layer.
* **The phase is the GPU, and the flush `wait` is the number that says so.** The flush-site backtraces are raw addresses in this build (the executable exports no dynamic symbols), so `addr2line` names them: the two sites whose trigger chain runs through `prefill::gdn_recurrence` carry **n=71 disp=9088 wait=1546 ms** and **n=51 disp=6528 wait=1105 ms** — **2,651 ms of the 9,300 ms total flush wait, over 15,616 dispatches.** The host's share of the whole prefill is **15 ms of submit and ~55 ms of encode**.
* **The kernels' own in-stream cost, from the bench, AT THE ENGINE'S OWN BATCH.** Every other GDN row in `bench/` is measured at ONE batch; `kLiveBatchMax` is 128 dispatches = 64 pairs. The new `gdn_rec_batch_sweep` arm (`/tmp/gdn/bench_full.log`) sweeps it:

| batch (pairs) | dispatches/replay | chain ms/pair | ms/dispatch | fused ms/pair | fused/chain |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.1069 | 0.0534 | 0.1028 | 0.962 |
| 8 | 16 | 0.0472 | 0.0236 | 0.0438 | 0.928 |
| 32 | 64 | 0.0407 | 0.0204 | 0.0373 | 0.916 |
| **64** | **128** | **0.0398** | **0.0199** | **0.0363** | **0.912** |
| 128 | 256 | 0.0393 | 0.0197 | 0.0359 | 0.913 |
| 199 | 398 | 0.0390 | 0.0195 | 0.0355 | 0.912 |

  Two things fall out. (a) **The port's batch-8 rows were 19% pessimistic and its batch-1 rows 2.7x** — a batch-8 row does NOT represent the engine's unit, and `bench/README.md` now says so. (b) **At the engine's batch the fused pair is 8.8% cheaper per token-layer**, worth ~42 ms of in-stream work (7,164 x 0.0059) — so the fusion could never have been the lever the phase's 2,675 ms suggests.
* **The step kernel is LATENCY-bound, not throughput-bound, and that is where the phase actually lives.** `groups_for(h_v*S)` = **24 workgroups for 6,144 threads**, each thread walking S=128 rows twice with a dependent load->FMA chain whose trip count comes from a push constant (so the backend cannot unroll it). Its own row: 3 MiB read twice and written once in 0.043 ms, **143 Melem/s / 55 GMAC/s** — a small-grid latency figure, not a bandwidth one. The fused pair (0.0363 ms) against the step alone (0.0428 ms, batch 8) says the norm is ~12% of the pair and the step ~90%.
* **And the engine's per-dispatch cost is 8.5x the bench's in-stream marginal at the same batch** (2,641 ms / 15,616 = 0.169 ms against 0.0199). That is NOT resolved here: the bench replays L2-hot harness-owned buffers and the engine runs one-shot against a 27 GiB arena. It is why the engine A/B, not the bench, was the decider.

### 2. THE A/B, AND ITS HONEST READING: A FALSIFICATION, NOT A WIN

`vulkan/src/kernels/prefill_vk.cpp::gdn_recurrence` gained a switch: `STRATA_PF_GDN_REC_FUSED=1` dispatches `fused_gdn_step_norm` (ONE dispatch per token) instead of the step + closing norm pair; the chain stays the DEFAULT. Both shaders' STATE update is `s = g*state + k*delta` in the same order and the closing norm does not touch the state, so the state trajectory is **bitwise identical** between the paths — which is what the ids depend on. What differs is the closing norm's reduction TREE, a different association in the mean-square; that is why the ids (and the new gate case's bounded `y` arm) are the guards.

199-token prompt, `--spec 2 --prefill 256`, n=3 per arm, interleaved, one config per invocation, `setsid nohup` + poll, EVERY arm logging its own `env | grep -iE 'strata_(vk|prefill|pf)'`, the sha256 of the four shaders it depends on and its own histogram (`/tmp/gdn/c{1,2,3}.log`, `/tmp/gdn/f{1,2,3}.log`):

| arm | mode | prefill ms | tok/s | `gdn recurrence` ms | recurrence histogram | ids md5 |
|---|---|---:|---:|---:|---|---|
| `c1` | chain (default) | 10,280.6 | 19.36 | 2,965 | step 7,164 + out_norm 7,164 | `56a0b28d2de6` |
| `c2` | chain | 10,208.0 | 19.49 | 2,668 | 7,164 + 7,164 | `56a0b28d2de6` |
| `c3` | chain | 9,891.7 | 20.12 | 2,674 | 7,164 + 7,164 | `56a0b28d2de6` |
| `f1` | fused | 9,834.7 | 20.23 | 2,574 | `fused_gdn_step_norm` 7,164 | `56a0b28d2de6` |
| `f2` | fused | 9,888.9 | 20.12 | 2,581 | 7,164 | `56a0b28d2de6` |
| `f3` | fused | 9,832.7 | 20.24 | 2,574 | 7,164 | `56a0b28d2de6` |

* **The phase: 2,674 -> 2,574 ms median, ranges DO NOT OVERLAP** (chain 2,668 / 2,674 / 2,965 against fused 2,574 / 2,574 / 2,581; the fused band is 0.3% wide, the chain band 11%). **The saving is 100 ms = 3.7% of the phase.**
* **THAT IS THE FALSIFICATION.** A phase bound by the NUMBER of dependent dispatches would have nearly halved when the dispatch count halved (14,328 -> 7,164). It moved 3.7%, which is what removing a kernel worth ~12% of the pair predicts. **The phase is carried by `native_gdn_step` itself, and no dispatch-count change will reach it.**
* **The end-to-end cannot resolve the 100 ms.** In the A/B session the prefill medians were 10,208.0 -> 9,834.7 ms (one chain arm, 9,891.7, landed inside the fused band); in a SEPARATE session the plain arms are **10,126.2 / 9,829.1 ms** on the fused path against **10,046.6 ms** on the chain — overlapping bands both times.
* **So the fusion stays OPT-IN and the chain stays shipped.** A 3.7%-of-the-phase win that the end-to-end prefill cannot resolve does not move a default; that is the same call `STRATA_VK_PREFILL_TILED` got. The ids are identical in all six arms, and each arm's own histogram is what makes its label evidence rather than an assumption.

### 3. THE WRAPPER HAD NO GATE ARM AT ALL — NOW IT HAS FIVE

`case_prefill_gdn_recurrence` (`ports/vulkan/harness/vk_gate.cpp`, appended last for the shared-RNG reason every batch names) drives `strata::prefill::gdn_recurrence` at the real geometry (S=128, HK=16, HV=48, T=3, the 3 MiB state) in BOTH modes — the wrapper reads `STRATA_PF_GDN_REC_FUSED` on every call precisely so one process can exercise both — and asserts five things: the two modes' **STATE is BITWISE identical**; the **STATE trajectory matches the native rule transcribed in double**; each mode's **`y` matches the same rule, bounded**; the fixture **MOVES** (an INTERLEAVED head-pairing rival changes `y` by rel-L1 1.29, so the passing arms are not passing on a blind fixture); and **a call with the variable UNSET is the CHAIN, the shipped path, BYTE FOR BYTE** (804,864/804,864) — "the default is a claim" made falsifiable, in the direction that matters now that the chain is shipped.

**FALSIFICATION.** The new registered injection `fused-gdn-step-norm-head-pairing` (INTERLEAVE the fused kernel's head pairing) bites: `FALSIFIED (fused-gdn-step-norm-head-pairing): FAIL prefill gdn_recurrence: chain and fused leave a BITWISE identical STATE 65536/786432`. The tree is restored on every exit path.

### 4. TWO FIXTURE DEFECTS FOUND AT THE CAUSE, WHICH IS WHERE THEY WERE FIXED

The case was RED twice before it was green, and both times the ORACLE was wrong and the kernel was right — the tenth and eleventh instances of the port's most-repeated finding:

1. **The fixture was CHAOTIC.** With raw N(0,1) k heads, `||k||^2 ~ 128`, so the rank-1 update's eigenvalue `g*(1 - beta*||k||^2)` is about **-19 per token** and the recurrence amplifies any rounding difference by `|lambda|^T`. The state diverged ~6e3x from a double oracle across three tokens while the two shader paths agreed with each other BITWISE. **The engine's own conv+L2 stage L2-NORMALISES the q and k heads before the recurrence**; the fixture now does the same and the double reference is meaningful. The bound was NOT widened.
2. **The operand slices were SWAPPED.** The oracle was transcribed from `case_native_gdn_step`, where q, k and v are separate buffers. Here they are three slices of one row — `q = h`, `k = h + HK*S`, `v = h + 2*HK*S` — and the first version read the **q** slice where the kernel reads **k** and vice versa. Every value stayed finite and plausible, both kernels agreed with each other, and only the double reference caught it (it failed 1 element of `y` at `t=1 h=24 j=33`: kernel `1.83692908`, oracle `0.00525862537`). Fixed at the cause with the arithmetic in the comment; a failing element now NAMES ITSELF (index and both values) on any red run and prints nothing when green.

### 5. THE BENCH INSTRUMENT: THE ARM LEDGER, AND WHY IT EXISTS

The bench that measured this phase is the one that DIED silently first: against a hand-built `.spv` dir it printed 62 rows, then `cannot open .../gemm_prefill_f16_m8_staged.spv` among the output, and exited 1. The message named a FILE, not the arm, and it was invisible in a tail. `vk_bench` now has an **arm ledger**: every arm is named, prints `-- arm <name> ...` (flushed) before it runs and `-- arm <name> OK (<rows>, <bytes>)` after, an arm that **printed nothing** is a named FAILURE, the run ends with `== arms: R ran | S skipped | F failed` and exits 1 if anything failed, and `--only <name>` runs exactly one arm (an unknown name lists the arms and exits 2). **The ledger's own first version counted only `ROW` lines and therefore called `gdn_rec_batch_sweep` an empty arm** — a check that fails on a good arm is as bad as one that cannot fail — so evidence is now **bytes on stdout** (`ftell`, ROW delta as fallback). Both new failure paths are DEMONSTRATED, not asserted: an unknown `--only` exits 2 with the list, and a missing `.spv` leaves `-- arm gemm_prefill ...` as the last line before `cannot open`. The arm count is part of the record: **36 arms ran, 0 skipped, 0 failed, 95 ROW + 6 SWEEP + 30 XPAIR lines.**

### 6. GATE, IDS, MAP, PATHS.

Arc `intel_icd` **895 passed / 0 failed / 0 skipped** — **the count RISES by exactly six (889 -> 895)**, the six being this case's arms; nothing removed, nothing skipped, no bound widened. `lvp_icd` **879/0/4** (the same four documented skips). `radeon_icd` **882/2/2**, with **BOTH** failures in the documented RADV moving-failing-set family and **neither** a case this batch touches: `bf16_gemv entry (n_in=2560 n_out=128)` 511/512 and `bf16_gemv_fp32_mmvf_cols entry (n_in=2560 n_out=48 ncols=13)` 2493/2496 — the one-row-of-512 intermittent W26 characterised. Smoke 60/0/0. Ids **`56a0b28d2de6`** (199-token) in all six A/B arms. `check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (no symbol was added: `fused_gdn_step_norm` and `gdn_recurrence` are both existing rows). Logs, exact: `/tmp/gdn/r1_full.log` (the phase measurement), `/tmp/gdn/bench_full.log` (the ledgered bench, 36 arms), `/tmp/gdn/bench_sweep.log` (the first, truncated run — kept as the record of the instrument failure), `/tmp/gdn/c{1,2,3}.log` + `/tmp/gdn/f{1,2,3}.log` (the A/B), `/tmp/gdn/ab_driver.log` (the sequencing), `/tmp/gdn/gate_final.log` (the gate), `/tmp/gdn/gate_probe.log` (the Arc-only case loop while the oracle was being fixed).

**NOT DONE / DELIBERATELY LEFT.** (i) **The taller-row-block staged coopmat GEMM (the named Target 2) and the T-threshold rule (Target 3) were NOT REACHED** — the batch's budget went to the measurement, the switch and its A/B, the gate case that had to exist, and two fixture defects; neither was attempted and neither is claimed. (ii) **The step kernel's serial walk is now the named target and is NOT attempted**: the bench cannot justify changing it (its 3 MiB state is L2-hot on replay, so a register-carrying redesign would look WORSE in the harness than in the engine), so it needs an engine A/B first. (iii) The engine's per-dispatch cost being **8.5x** the bench's in-stream marginal at the same batch is **unexplained** — no measurement here separates the L2-hot harness buffers from the engine's one-shot arena access. (iv) The 0.169 ms/dispatch figure is an attribution via the flush trigger site, not a device timestamp. (v) `f32_to_f16` (3,644 dispatches) sits inside the same phase and was not separated. (vi) The A/B is **n=3 per arm**.

## THE PREFILL GEMM: COOPMAT IS 17.4% FASTER (n=5 A/B, EACH ARM PROVING ITS OWN CONFIG) AND IS NOW THE DEFAULT; THE SHARED-MEMORY STAGING IS A 2.5-3.2x LOSS PER DISPATCH AND 1.62x END TO END (later, same day, `vega`, Arc Pro B70)

**THE A/B, AND WHY THE LABELS ARE NOW EVIDENCE.** Ten sequential engine arms, 199-token prompt,
`--spec 2 --prefill 256`, `STRATA_VK_DISP_STAT=1`, one config per invocation, interleaved def/cm, and **every arm's
own log carries both `env | grep -i strata_vk` as it was launched and its per-shader dispatch histogram** - the
previous batch lost a whole set of arms to an inherited `STRATA_VK_PREFILL_COOPMAT=1`.

| arm | prefill ms | tok/s | `gemm_prefill_f16_m8.spv` | `gemm_prefill_fma_small.spv` |
|---|---:|---:|---:|---:|
| `t1_def_1` default | 11,613.1 | 17.14 | absent | 7,716 |
| `t1_def_2` default | 12,338.8 | 16.13 | absent | 7,716 |
| `t1_def_3` default | 11,632.0 | 17.11 | absent | 7,7xx |
| `t1_def_4` default | 12,084.4 | 16.47 | absent | 7,7xx |
| `t1_def_5` default | 11,629.5 | 17.11 | absent | 7,7xx |
| `t1_cm_2` coopmat | **9,905.6** | **20.09** | **3,850** | 7,150 |
| `t1_cm_3` coopmat | 10,082.4 | 19.74 | 3,850 | 7,150 |
| `t1_cm_4` coopmat | 9,916.2 | 20.07 | 3,850 | 7,150 |
| `t1_cm_5` coopmat | 9,881.6 | 20.14 | 3,850 | 7,150 |
| `t1_cm_6` coopmat | 9,884.9 | 20.13 | 3,850 | 7,150 |

**median 11,632.0 ms / 17.11 tok/s (default) against 9,905.6 ms / 20.09 tok/s (coopmat): 1.174x, and the ranges do
not overlap** (the slowest coopmat arm beats the fastest default arm by 13%). Spread: 6.2% default, 2.0% coopmat.
One arm (`t1_cm_1`) is excluded because its own log shows it refused the arena while the previous arm held the card;
it was re-run, not relabelled. Ids `56a0b28d2de6` in all ten.

**THE DEFAULT MOVED, AND IT IS VERIFIED PER RUN.** Four arms with NO `STRATA_VK_PREFILL_COOPMAT` in their
environment at all: **9,884.3 / 9,925.8 / 9,918.2 / 9,886.0 ms** (20.13 / 20.05 / 20.06 / 20.13 tok/s, median
9,900.2 ms / 20.09 tok/s) - each one's histogram showing `gemm_prefill_f16_m8.spv` 3,850, i.e. the matrix units ran
with nothing set. `STRATA_VK_PREFILL_COOPMAT=0` still forces the FMA path; the shape precondition
(`t%8==0 && n%16==0 && k%16==0`, no ragged edge) is the small-T rule, so every T<8 and every ragged shape stays on
the FMA kernels.

**THE REFERENCE'S SHARED-MEMORY STAGING, MEASURED THREE WAYS - AND IT LOSES.** llama.cpp's `mul_mm.comp` /
`mul_mmq.comp` stage the operand tiles in shared memory before the cooperative-matrix load; this port loaded
straight from global. That staging is now implemented (`shaders/common/gemm_prefill_staged.glsl`, a workgroup
stages `X[8 x 32]` and `W[128 x 32]` into 10,880 B of the card's 131,072 B with two barriers per K step), gated,
falsified twice, and measured:

**(a) bench, in-stream marginal cost** (one dispatch per replay, batch 8, median of 9):

| shape | cm-global ms | cm-staged ms | staged/global |
|---|---:|---:|---:|
| gate/up T=8 N=1280 K=2560 | 0.4017 | 1.2930 | **3.219** |
| gate/up T=16 | 0.4057 | 1.2985 | **3.201** |
| gate/up T=64 | 1.2511 | 3.4052 | **2.722** |
| gate/up T=199 | 3.3341 | 9.0432 | **2.712** |
| down T=8 N=2560 K=640 | 0.1099 | 0.3345 | **3.045** |
| down T=64 | 0.5358 | 1.3594 | **2.537** |
| down T=199 | 1.6094 | 4.9149 | **3.054** |

**(b) engine, end to end** (the staged kernel under the shipped file name in its own SPV dir; each arm logs the
sha256 of the kernel it will load - `8f8d743d...` staged, `67a05a3e...` global):

| arm | kernel | prefill ms | tok/s |
|---|---|---:|---:|
| `q_t3g_1` | global (shipped) | 10,120.2 | 19.66 |
| `q_t3g_2` | global (shipped) | 9,896.4 | 20.11 |
| `q_t3s_1` | staged | **16,033.5** | **12.41** |
| `q_t3s_2` | staged | **16,012.3** | **12.43** |

**(c) phase table** (`STRATA_PREFILL_TIMING=1`, HOST WALL-CLOCK - used as a ranking, not as GPU time): `gemm
gate/up` + `gemm down` = **718+626 = 1,344 ms** on the shipped kernel (13.5% of a 9,981 ms timeline) against
**1,615+1,365 = 2,980 ms** (18.7% of 15,858 ms) staged; the GPU timeline itself goes 9,756-9,981 ms -> 15,858 ms.
The same pair of phases on the FMA path (`q_t3f_1`, `STRATA_VK_PREFILL_COOPMAT=0`) is **4,053+454 = 4,507 ms**
(39.4% of 11,449 ms) - so the table ranks the three arms the way the token rate does: matrix units, then FMA, then
staged.

**WHY, AND IT IS THE TILE, NOT THE IDEA.** A workgroup here covers `TM = 8` token rows - one cooperative-matrix row
block, because the engine's expert row-batches are ~8 tokens - so 128 staged W columns feed a single row block:
**7.5 MACs per staged element**. `mul_mm.comp` stages a 64x64 output block and gets **16 MACs per element**, which
is what pays for the staging, the barriers and the shared round trip on Xe2. Fixed by a TALLER output block
(BM 32-64, guarded store through shared memory, as the reference does for partial tiles) - a named next increment,
not attempted here. The staged kernel is built, gated and bench-armed and **nothing dispatches it**.

**FOR SCALE.** The same bench, same shapes: the untiled FMA kernel is the FASTEST of the four at T=8 (0.195 ms
against coopmat's 0.402 and the tiled FMA's 0.238), and the tiled FMA kernel is the fastest at T>=64 (gate/up T=199:
1.056 ms against coopmat 3.334 and untiled 6.330) - yet the engine's end-to-end A/B (the authority) has coopmat
17.4% ahead of the untiled path and the tiled path a wash (19.52 against 19.74 tok/s, n=3 vs n=2). An isolated row
ranks kernels; it does not decide the default.

**GATE + IDS.** Arc `intel_icd` **889 passed / 0 failed / 0 skipped** (the count RISES by the three new arms, 886 ->
889; nothing skipped, no bound widened), the intel default arm likewise `889/0/0`; lvp `873/0/4` and radeon
`875/3/2` in the full run with **`878/0/2` on an immediate re-run of the same binary** (the three failures - none of
them a case this batch touches - CLEARED: the documented RADV moving-failing-set), smoke `60/0/0` - all in
`/tmp/gemm/gate_final.log` and `/tmp/gemm/q_radeon_recheck.log`. Three injections FALSIFY the new arithmetic and the
new grid (`prefill-cm-grid-short` 1152/2176 worst 1e+30; the arm had to be widened from N=64 to N=128 first, because
at N=64 the SHIPPED kernel's eight-tiles-per-workgroup mapping is still covered by a one-short grid - see NEXT.md).
Ids `56a0b28d2de6` (199-token) in all nineteen 199-token arms and `3aed108cceee` (8-token) in both 8-token arms
(decode 8.20 tok/s, unmoved). Logs: `/tmp/gemm/t1_{def,cm}_*.log`, `/tmp/gemm/q_def_*.log`,
`/tmp/gemm/q_t3{g,s,f}_*.log`, `/tmp/gemm/e_def_*.log`, `/tmp/gemm/bench.log`,
`ports/vulkan/bench/build/icd-*.log`, `/tmp/gemm/gate_final.log`.

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
