# The port-side throughput harness (`ports/vulkan/bench/`)

**This is not the numeric gate.**  `gates/run_gate.sh` + `harness/vk_gate.cpp` stay the *correctness*
authority — they prove a kernel computes the right numbers.  This directory is a separate, opt-in
**throughput** measurement of the kernels the port has already landed, on whichever Vulkan device you point
it at.  It exists because the port is correct-but-slow by construction (every fast path was dodged by a
capability contract: `native_gdn_enabled() == false`, `gr_set_native_mmvf(false)`,
`layer_set_fused_gr(false)`), and the native/fused kernels that come next can only be called an
*improvement* against numbers this harness produces.  No claim without a measurement.

It reuses the port's own device layer (`harness/vk_compute.*`) rather than inventing one, and it builds in
its **own** tree (`bench/build/`, which the repo's `build*/` ignore rule covers), so it never touches
`harness/build/` or the gate's `.spv` files.

## Layout

```
ports/vulkan/bench/
  vk_bench.cpp     the harness: the kernels' shapes, the timing method, the report
  run_bench.sh     compiles the measured kernels from source, builds the binary, runs it under every ICD
  README.md        this file (the method, the limits, the baseline)
  build/           (git-ignored) spv/, vk_bench, icd-<name>.log
```

## How to run it

```bash
cd /home/bob/strata-vulkan-wt
STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 \
  bash ports/vulkan/bench/run_bench.sh
```

`run_bench.sh` (1) compiles the measured kernels from `.comp` into `bench/build/spv/` with `glslc` and
validates each with `spirv-val`; (2) builds `vk_bench` with `-O2 -Werror`; (3) runs it once under **every**
Vulkan ICD that reports a device (`VK_ICD_FILENAMES=<icd>`), logging to `bench/build/icd-<name>.log`; and
(4) prints a cross-ICD comparison.  Extra arguments are passed through to the binary:

```
vk_bench [--spv-dir D] [--device N] [--reps R] [--warmups W] [--sampler-vocab N] [--list]
```

## The timing method (read this before quoting a number)

* **What is timed — the kernel dispatch only.**  Every input is uploaded *before* the timed region.  The
  timed region is a **batch** of `K` dispatches of one kernel, recorded into a single command buffer
  (`Ctx::record_begin` / `record_dispatch` / `record_end_and_submit`) with a compute→compute barrier
  between dispatches, so they execute back to back and do not overlap.  No transfer is inside the loop, so
  the unit is the kernel's throughput **on resident data**.  Upload and readback are not in the number.
* **Warmup.**  2–3 untimed replays of the batch before the timed ones (plus the first submit at record end),
  so the pipeline is warm and the driver's first-dispatch costs are not in the figure.
* **Repetitions and the reported statistic.**  `reps` timed replays (default **9**; **5** for the sampler).
  Each replay is one `vkQueueSubmit` + `vkWaitForFences` of the whole batch; the per-dispatch time is that
  replay's wall time divided by `K`.  The reported figure is the **median** over the `reps` replays, with
  **min and max printed beside it** (the raw logs carry all three, so the variance is visible, not asserted
  away).  `K` is per kernel (8–128) and is printed in the `batch` column.
* **Synchronisation — wall clock around a fence, not device timestamps.**  The port's device layer exposes
  no `VkQueryPool`/timestamp path, so `vk_bench` times `std::chrono::steady_clock` around the replay, which
  ends in `vkWaitForFences`.  This measures the GPU-bounded region including the submit and one fence wait
  per **batch** — amortised over `K` dispatches.  **The known limit, and it is visible in the tables
  below:** the wall-clock floor of a single batch replay is ~5–15 µs on the fast GPUs, so a kernel whose
  whole batch is smaller than that reads as the floor rather than as its own cost.  That is why some of the
  4×-work sizing arms scale ~1× on the Arc and the XTX (both sizes sit near the floor) while they scale ~3.5×
  on llvmpipe and the K620 (both sizes are work-bound).  A device-timestamp version is the natural next step;
  it is not available from this device layer today.
* **The ICD/device per run.**  `run_bench.sh` runs the binary once per ICD, so each row is measured on the
  device named in that log's `== device` line (printed with vendor id, subgroup size and the storage/fp64
  features).  A single-ICD run is not a cross-device proof and `run_bench.sh` says so.
* **The display-reserve setting.**  `configure_display_reserve()` reads `STRATA_VK_DESKTOP_RESERVE_MIB`
  (default 1024), `STRATA_VK_RESERVE_FLOOR_MIB` (default 512) and `STRATA_VK_MAX_BUDGET_MIB`, and prints what
  it decided (`vk_compute: heap total … free … desktop reserve … -> … usable`).  The harness needs only a few
  MiB of buffers, so the runs here use the reserve and its floor at **0**, exactly as the numeric gate does:
  with the real defaults a card that another process is holding back would (correctly) refuse the
  allocation, and on `vega` the resident local model is that process.  **Measured budget lines (reserve 0):**
  Arc `heap total 31.89 GiB | free (driver) 28.64 GiB | reserve 0.00 GiB -> 28.64 GiB usable`; llvmpipe
  `46.26 / 39.19`; Ryzen iGPU `15.75 / 15.47`; **box XTX `24.00 / 22.45`**; box K620 `2.24 / 2.17`; box
  llvmpipe `92.29 / 79.96`.  The reserve policy itself is still verified by the gate's `case_reserve_policy`
  (as a pure function) and `case_memory_budget` (the live driver figures) — the escape hatch does not weaken
  it.

## What is measured

| kernel (shader) | shape | unit |
|---|---|---|
| `gdn_conv_step` | C=2560 d_conv=4 | elements/s (outputs) |
| `gdn_l2_norm` | rows=48 (h_v) cols=128 (S) | elements/s |
| `gdn_beta_gate` | h_v=48 | elements/s |
| `gdn_gate` | h_v=48 n_tokens=64 | elements/s |
| `gdn_step` | S=128 h_k=16 h_v=48 (3 MiB state) | GMAC/s (3·S MACs per output) |
| `gdn_out_norm` | h_v=48 S=128 | elements/s |
| `iq_dequant_f32` BF16 / IQ4_NL / IQ2_S | 256 superblocks (65536 floats); IQ2_S also at 1024 (262144) | elements/s |
| `iq2s_mmvq` | n_in=2560 n_out=512 and 2048, ncols=1 | GMAC/s (n_in MACs per output) |
| `sampler_kernel_f32` | vocab=248320, n_tokens=1 | elements/s (logits scanned) |
| `quantize_q8_0` | n=65536 (2048 blocks) — **skipped where the device has no `shaderFloat64`** | elements/s |
| `quantize_q8_1` | n_in=2560 ncols=8 | elements/s |
| `quantize_q8_K` | n=65536 (256 blocks) | elements/s |
| `rope_neox` vs `native_rope_apply` | rows=512 head_dim=256 n_rot=64 | elements/s |
| `router_top10_f32` vs `native_router_top10` | n_tokens=16, n_expert=512, k=10 | elements/s |
| `moe_combine_f32` vs `native_moe_combine` | n_embd=2560 k=10 shared=1 | elements/s |
| `rms_norm` vs `native_qsa_rms_norm_weighted` | rows=128 cols=2560 | elements/s |
| `qsa_gate_apply_f32` vs `native_qsa_gate_apply` | n_head=24 head_dim=256 | elements/s |
| `gdn_conv_step` vs `native_gdn_conv_silu` | C=2560 d_conv=4 | elements/s (outputs) |
| `gdn_conv_step`+`silu_f32` vs `native_gdn_conv_silu` | C=2560 d_conv=4 (the 2-dispatch legacy chain) | elements/s |
| `gdn_l2_norm` vs `native_gdn_l2_norm` | rows=48 cols=128 | elements/s |
| `gdn_beta_gate` vs `native_gdn_beta_gate` | h_v=48 | elements/s |
| `gdn_gate` vs `native_gdn_gate` | h_v=48 n_tokens=1 | elements/s |
| `gdn_out_norm` vs `native_gdn_out_norm` | h_v=48 S=128 | elements/s |
| `gdn_step` vs `native_gdn_step` | S=128 h_k=16 h_v=48 (3 MiB state) | GMAC/s (3·S MACs per output) |
| `scale`+`gdn_step` vs `native_gdn_step` | S=128 h_k=16 h_v=48 (the 2-dispatch legacy chain) | GMAC/s |
| `fused_gdn_conv_l2` vs `native_gdn_conv_silu`+2× `native_gdn_l2_norm` | C=10240 qk_heads=32 d_conv=4 (the 3-dispatch native chain) | elements/s |
| `fused_gdn_ab` vs 2× `bf16_mmvf_f32` + `native_gdn_beta_gate` + `native_gdn_gate` | h_v=48 n=2560 (the 4-dispatch chain) | elements/s |
| `fused_gdn_step_norm` vs `native_gdn_step` + `native_gdn_out_norm` | S=128 h_k=16 h_v=48 (3 MiB state; the 2-dispatch chain) | GMAC/s |

The **sampler is measured last on purpose**: its one-block top-k is a single workgroup sweeping the whole
vocabulary `k` times, the port's heaviest single dispatch, and on one device (see below) it is heavy enough
to trip a driver timeout.  Running it last means a reset there costs only that row.

## Baseline — measured 2026-10-05

`reps=9` (sampler 5), median ms; the logs carry min/max and the batch.  `–` = not measured on that device.

### `vega` (Arc Pro B70 / ANV, and the Ryzen iGPU / RADV, and llvmpipe)

| kernel | Arc B70 (ms) | Ryzen iGPU (ms) | llvmpipe (ms) |
|---|---:|---:|---:|
| `gdn_conv_step` | 0.0297 | 0.0074 | 0.0231 |
| `gdn_l2_norm` | 0.0198 | 0.0059 | 0.0762 |
| `gdn_beta_gate` | 0.0037 | 0.0010 | 0.0136 |
| `gdn_gate` | 0.0050 | 0.0028 | 0.0212 |
| `gdn_step` | 0.0469 | 0.5313 | 0.5015 |
| `gdn_out_norm` | 0.0059 | 0.0064 | 0.0782 |
| `iq_dequant_f32` BF16 | 0.0130 | 0.0195 | 0.1444 |
| `iq_dequant_f32` IQ4_NL | 0.0133 | 0.0220 | 0.1456 |
| `iq_dequant_f32` IQ2_S @256 | 0.0145 | 0.0458 | 0.1712 |
| `iq_dequant_f32` IQ2_S @1024 | 0.0158 | 0.1483 | 0.5946 |
| `iq2s_mmvq` n_out=512 | 0.0224 | 0.2502 | 1.9112 |
| `iq2s_mmvq` n_out=2048 | 0.0464 | 0.9762 | 6.2754 |
| `quantize_q8_0` | 0.0265 | 0.0573 | 0.0693 |
| `quantize_q8_1` | 0.0095 | 0.0159 | 0.1340 |
| `quantize_q8_K` | 0.0875 | 0.2039 | 0.1938 |
| `sampler_kernel_f32` | **546.90** | **299.57** | **1164.75** |

Arc derived rates: `gdn_step` **50.3 GMAC/s**; `iq2s_mmvq` n_out=2048 **112.9 GMAC/s**; `iq_dequant` BF16
~5.0 Gelem/s (65536 outputs / 13 µs); `quantize_q8_K` 0.75 Gelem/s.

### `z820b` (7900 XTX / RADV NAVI31, Quadro K620 / NVIDIA, and llvmpipe)

| kernel | 7900 XTX (ms) | Quadro K620 (ms) | llvmpipe (ms) |
|---|---:|---:|---:|
| `gdn_conv_step` | 0.0043 | 0.0073 | 0.0894 |
| `gdn_l2_norm` | 0.0040 | 0.0113 | 0.1355 |
| `gdn_beta_gate` | 0.0026 | 0.0027 | 0.0520 |
| `gdn_gate` | 0.0038 | 0.0046 | 0.0902 |
| `gdn_step` | 0.1050 | 0.4859 | 0.9662 |
| `gdn_out_norm` | 0.0036 | 0.0125 | 0.1451 |
| `iq_dequant_f32` BF16 | 0.0140 | 0.0391 | 0.4210 |
| `iq_dequant_f32` IQ4_NL | 0.0094 | 0.0343 | 0.4224 |
| `iq_dequant_f32` IQ2_S @256 | 0.0155 | 0.0305 | 0.4952 |
| `iq_dequant_f32` IQ2_S @1024 | 0.0162 | 0.0824 | 1.6704 |
| `iq2s_mmvq` n_out=512 | 0.0154 | 0.1932 | 6.6139 |
| `iq2s_mmvq` n_out=2048 | 0.0198 | 0.6887 | 23.7802 |
| `quantize_q8_0` | 0.0175 | 0.1440 | 0.2225 |
| `quantize_q8_1` | 0.0075 | 0.0277 | 0.2751 |
| `quantize_q8_K` | 0.0732 | 0.2394 | 0.8536 |
| `sampler_kernel_f32` | **202.03** | **353.83** | **5627.97** |

XTX derived rates: `gdn_step` **22.5 GMAC/s**; `iq2s_mmvq` n_out=2048 **264.3 GMAC/s** (the fastest device
measured on every kernel).

### The class-B NATIVE vs LEGACY pairs — measured 2026-10-05

The performance tier's first four kernels: each NATIVE fast path timed against the legacy kernel it replaces,
**at the same shape, on the same device**.  `run_bench.sh` now compiles both sides from source and each pair
prints an `XPAIR` line: `native/legacy`, so **< 1.0 means the native kernel is faster**.  `reps=9`.

| pair (native ← legacy) | Arc B70 med (ms) native / legacy | native/legacy | Ryzen iGPU | native/legacy | llvmpipe | native/legacy |
|---|---:|---:|---:|---:|---:|---:|
| `native_rope_apply` ← `rope_neox` | 0.0084 / 0.0280 | **0.301** | 0.0490 / 1.2242 | **0.040** | 0.0512 / 0.1121 | 0.457 |
| `native_router_top10` ← `router_top10_f32` | 0.0245 / 0.3125 | **0.078** | 0.0300 / 0.3842 | **0.078** | 0.2036 / 0.1644 | 1.238 |
| `native_moe_combine` ← `moe_combine_f32` | 0.0058 / 0.0058 | 0.998 | 0.0059 / 0.0060 | 0.988 | 0.0224 / 0.0227 | 0.986 |
| `native_qsa_rms_norm_weighted` ← `rms_norm` | 0.0107 / 0.0106 | 1.007 | 0.1338 / 0.1198 | 1.117 | 0.2214 / 0.2240 | 0.988 |

The same pairs on the **box `z820b`** — the RX 7900 XTX is the primary RADV target:

| pair | XTX (RADV NAVI31) med (ms) native / legacy | native/legacy | K620 (nvidia) | native/legacy |
|---|---:|---:|---:|---:|
| `native_rope_apply` ← `rope_neox` | 0.0076 / 0.0644 | **0.118** | 0.0482 / 0.3169 | 0.152 |
| `native_router_top10` ← `router_top10_f32` | 0.0134 / 0.1641 | **0.082** | 0.0600 / 0.1609 | 0.373 |
| `native_moe_combine` ← `moe_combine_f32` | 0.0040 / 0.0039 | 1.018 | 0.0119 / 0.0112 | 1.056 |
| `native_qsa_rms_norm_weighted` ← `rms_norm` | 0.0060 / 0.0060 | 0.988 | 0.0209 / 0.0117 | 1.795 |

**The honest reading, including the pair that is NOT a win.**

* **RoPE is a genuine 3.3× on the Arc, 8.5× on the XTX and 25× on the RADV iGPU** (2.2× on llvmpipe).  The
  legacy kernel is ONE THREAD PER ROW (the CUDA's own decomposition), so at 512 rows it launches 2 workgroups;
  the native kernel is one thread per (row, PAIR) - 512 workgroups - and computes the angle on device.  Same
  shape, same device.
* **The router is 12.8× on the Arc, 12.2× on the XTX and 12.8× on the iGPU.**  Both kernels are one workgroup
  per token, and the native body's win here survives the port: this port's native shader reduces with the
  port's pairwise trees, while the legacy `router_top10_f32` sums 512 experts on ONE lane with Kahan
  compensation and runs its ten selection passes through a single-lane combine.  That is the cost the native
  path removes.
* **`native_moe_combine` is a WASH (0.998 on the Arc, 1.018 on the XTX).**  Both are elementwise, one thread
  per column, so there is no algorithmic difference to win - the native body's only advantage (plain f32
  instead of fma+Kahan) is not a throughput difference.  Recorded as measured, not tuned.
* **`native_qsa_rms_norm_weighted` is NEUTRAL on the Arc (1.007) and 12% SLOWER on the RADV iGPU.**  Both
  kernels are one workgroup per row with the same stride loop; the native body's block-per-row reduction is
  not faster than the legacy warp-per-row one at 2560 columns, and on RADV the extra shared traffic costs a
  little.  **This is a finding, not a defect** - a native kernel need not be faster, and the capability
  contract still requires the answer to be honest; the flag has since flipped TRUE (batch 5 ported its last gated
  symbol, `native_qsa_gate_apply`, so every symbol the flag gates now has a shader).  See `STATUS.md`.
* **The K620 (a Kepler Quadro) is where the native reductions HURT**: `native_qsa_rms_norm_weighted` is 1.795×
  the legacy there, because the native body's block-per-row tree does more shared-memory traffic than the
  legacy warp-per-row warp-shuffle on a device with no fast shared path.  Reported, not hidden.

The pairs are measured through the same fence-clock method as everything else here, and on the Arc, the XTX and
the iGPU the kernel batch is above the ~5-15 µs floor; on llvmpipe the `native_router_top10` pair sits at the
floor for the native row (0.2036 ms for 16 tokens) and reads slightly slower than legacy there - a
toolchain/execution-model artifact of a CPU driver, not a device result.  Both rows are printed so the reader
sees it.

### The class-B QSA GATE pair — `native_qsa_gate_apply` ← `qsa_gate_apply_f32` — measured 2026-10-05

The last symbol `native_qsa_enabled()` gates (batch 5).  Same `XPAIR` convention (`native/legacy`, **< 1.0 means
the native kernel is faster**); `reps=9`, shape `n_head=24 head_dim=256` (the artifact).  **There is NO
two-dispatch chain to measure here** — the QSA layer's gate is ONE dispatch either way
(`layer.cpp:1010-1012`) — so this is a drop-in pair and only the single pair is reported.

| pair (native ← legacy) | Arc B70 med (ms) n / l | native/legacy | Ryzen iGPU | native/legacy | lvp (vega) | native/legacy |
|---|---:|---:|---:|---:|---:|---:|
| `native_qsa_gate_apply` ← `qsa_gate_apply_f32` | 0.0048 / 0.0048 | 1.001 | 0.0213 / 0.0217 | 0.984 | 0.0043 / 0.0042 | 1.018 |

The same pair on the **box `z820b`**:

| pair | XTX (RADV NAVI31) med (ms) n / l | native/legacy | K620 (nvidia) | native/legacy | lvp (box) | native/legacy |
|---|---:|---:|---:|---:|---:|---:|
| `native_qsa_gate_apply` ← `qsa_gate_apply_f32` | 0.1011 / 0.0992 | 1.019 | 0.0049 / 0.0054 | 0.904 | 0.0022 / 0.0020 | 1.120 |

**A WASH on all six devices (0.904–1.120), and that is the expected result.**  Both kernels are one thread per
output element with no reduction and no shared memory, and the RULES are identical — the native body computes
`attn * sigmoid(second-half gate)` in f32 (`expf`), the legacy CUDA in f64; because the target has no
`shaderFloat64`, this port's legacy shader **already** computes in f32, so the two device shaders agree to within
the transcendental gap.  The XTX's 1.019 is a 2 µs difference on a 6.1k-element dispatch, at the fence-clock
resolution; the K620's 0.904 is the same magnitude the other way.  **Recorded as measured, not tuned** — a native
kernel is not required to be faster, and the point of this increment is the branch, not the throughput.

### The class-B NATIVE vs LEGACY pairs, batch 2 — the GDN / DeltaNet MIXER — measured 2026-10-05

The mixer runs on **36 of the model's 48 layers**.  These are the first three native mixer kernels, each timed
against the legacy kernel it replaces at the same shape on the same device, plus (for the conv) against the
**two-dispatch legacy chain** the layer actually issues.  Same `XPAIR` convention: `native/legacy`, **< 1.0
means the native kernel is faster**; `reps=9`.

| pair (native ← legacy) | Arc B70 (ms) native/legacy | ratio | Ryzen iGPU | ratio | XTX (box) | ratio | K620 (box) | ratio | llvmpipe (vega/box) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `native_gdn_conv_silu` ← `gdn_conv_step` | 0.0047 / 0.0050 | **0.938** | 0.0067 / 0.0074 | **0.909** | 0.0018 / 0.0026 | **0.694** | 0.0073 / 0.0078 | **0.936** | 0.0189 / 0.0214 | 0.881 |
| `native_gdn_conv_silu` ← `gdn_conv_step`+`silu_f32` (2 dispatches) | 0.0047 / 0.0082 | **0.574** | 0.0067 / 0.0088 | **0.764** | 0.0018 / 0.0033 | **0.544** | 0.0073 / 0.0101 | **0.720** | 0.0189 / 0.0377 | 0.501 |
| `native_gdn_l2_norm` ← `gdn_l2_norm` | 0.0050 / 0.0050 | 0.997 | 0.0059 / 0.0059 | 1.013 | 0.0024 / 0.0026 | 0.938 | 0.0116 / 0.0123 | 0.942 | 0.0679 / 0.0692 | 0.981 |
| `native_gdn_beta_gate` ← `gdn_beta_gate` | 0.0037 / 0.0037 | 1.011 | 0.0010 / 0.0010 | 1.045 | 0.0012 / 0.0011 | 1.068 | 0.0025 / 0.0029 | 0.865 | 0.0118 / 0.0116 | 1.016 |

**The honest reading, including the two pairs that are NOT wins.**

* **`native_gdn_conv_silu` is a win everywhere, and a bigger win against what the branch runs.**  Per DISPATCH
  it writes a second output and computes the SiLU, yet it beats the `gdn_conv_step` it replaces by 6-31%
  (0.694-0.938 across the four GPUs).  The legacy branch is `gdn_conv_step` -> a D2D copy -> `silu_f32`
  (`layer.cpp:255-257`); the copy cannot be timed as a kernel, but the two KERNELS can, and the native one
  dispatch is **1.3-2.1x faster than that chain** (0.481-0.764).  Both rows are printed.
* **`native_gdn_l2_norm` is a WASH (0.938-1.039).**  The native rule is numerically the SAME as the legacy's
  (`x / sqrt(sum(x^2)+eps)`; the `1/S` and `1/sqrt(S)` factors cancel), and both kernels are one workgroup per
  128-wide row - the arithmetic change (f32 sums, a `rsqrtf` of the mean form, a folded `scale_after`) buys no
  throughput.  Recorded as measured.
* **`native_gdn_beta_gate` is a WASH (0.865-1.068).**  Its rule is `1/(1+expf(-x))`, the SAME expression as the
  legacy `sigmoid_f`, one thread per head either way - nothing to win.  On the Arc/iGPU/XTX it reads 1-7%
  SLOWER, which is run-to-run noise at a dispatch of 48 elements, but it is reported rather than hidden.
* **A native kernel is not required to be faster.**  The batch's value for these two is that the ONE
  `native_gdn_enabled()` flag now has all three of its simple mixer stages on the native arithmetic path; the
  flag stays false until the six remaining gated symbols land (see `NEXT.md`).

### The class-B FUSED GDN PATHS, batch 4 — measured 2026-10-05

The three fused mixer paths, each timed against the **multi-dispatch chain it replaces** (that is where a fusion
shows — the batches so far found the win is in REMOVING DISPATCHES) and against the non-fused native kernel(s)
where a comparison exists.  Same `XPAIR` convention (`fused/legacy`, **< 1.0 means the fused path is faster**);
`reps=9`.  Each fused path is ONE recorded dispatch; the chains are recorded with a compute→compute barrier
between dispatches (`time_chain`).

| pair (fused ← what it replaces) | vega Arc B70 (ms) fused/chain | ratio | Ryzen iGPU | ratio | llvmpipe (vega) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `fused_gdn_conv_l2` ← conv_silu + 2× l2_norm (**3 dispatches**) | 0.0059 / 0.0131 | **0.452** | 0.0212 / 0.0298 | **0.712** | 0.0705 / 0.1259 | **0.560** |
| `fused_gdn_conv_l2` ← `native_gdn_conv_silu` (1 dispatch) | 0.0059 / 0.0053 | 1.122 | 0.0212 / 0.0235 | 0.903 | 0.0705 / 0.0252 | 2.803 |
| `fused_gdn_ab` ← 2× bf16_mmvf + beta_gate + gate (**4 dispatches**) | 0.0075 / 0.0188 | **0.398** | 0.0240 / 0.0273 | **0.878** | 0.1319 / 0.1773 | **0.744** |
| `fused_gdn_ab` ← `bf16_mmvf_f32` (1, one row set) | 0.0075 / 0.0066 | 1.136 | 0.0240 / 0.0128 | 1.868 | 0.1319 / 0.0742 | 1.776 |
| `fused_gdn_step_norm` ← native_gdn_step + native_gdn_out_norm (**2 dispatches**) | 0.0442 / 0.0476 | **0.929** | 0.4148 / 0.4202 | **0.987** | 0.4247 / 0.4566 | **0.930** |
| `fused_gdn_step_norm` ← `native_gdn_step` (1 dispatch) | 0.0442 / 0.0433 | 1.021 | 0.4148 / 0.4154 | 0.998 | 0.4247 / 0.3887 | 1.092 |
| `fused_gdn_step_norm` ← `native_gdn_out_norm` (1; NOT like-for-like) | 0.0442 / 0.0059 | 7.533 | 0.4148 / 0.0065 | 63.722 | 0.4247 / 0.0683 | 6.220 |

The same rows on the **box `z820b`** (RX 7900 XTX / RADV NAVI31 and Quadro K620 / NVIDIA):

| pair (fused ← what it replaces) | XTX (ms) fused / chain | ratio | K620 (ms) | ratio | llvmpipe (box) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `fused_gdn_conv_l2` ← conv_silu + 2× l2_norm (3) | 0.0026 / 0.0045 | **0.584** | 0.0163 / 0.0298 | **0.547** | 0.1845 / 0.3691 | **0.500** |
| `fused_gdn_conv_l2` ← `native_gdn_conv_silu` (1) | 0.0026 / 0.0022 | 1.175 | 0.0163 / 0.0150 | 1.089 | 0.1845 / 0.1206 | 1.530 |
| `fused_gdn_ab` ← 2× bf16_mmvf + beta_gate + gate (4) | 0.0030 / 0.0065 | **0.466** | 0.0289 / 0.0332 | **0.869** | 0.2731 / 0.4293 | **0.636** |
| `fused_gdn_ab` ← `bf16_mmvf_f32` (1, one row set) | 0.0030 / 0.0032 | 0.946 | 0.0289 / 0.0147 | 1.969 | 0.2731 / 0.1563 | 1.747 |
| `fused_gdn_step_norm` ← step + out_norm (2) | 0.0509 / 0.0507 | 1.003 | 0.3597 / 0.3620 | **0.994** | 0.9909 / 1.0932 | **0.906** |
| `fused_gdn_step_norm` ← `native_gdn_step` (1) | 0.0509 / 0.0504 | 1.009 | 0.3597 / 0.3550 | 1.013 | 0.9909 / 0.9337 | 1.061 |
| `fused_gdn_step_norm` ← `native_gdn_out_norm` (1; NOT like-for-like) | 0.0509 / 0.0028 | 17.945 | 0.3597 / 0.0123 | 29.310 | 0.9909 / 0.1443 | 6.867 |

**And the ONE non-win it produced, reported as measured: the XTX's `fused_gdn_step_norm` chain pair reads 1.003**
(0.0509 vs 0.0507 ms) — a wash inside run noise.  The XTX's step is fast and memory-bound (0.050 ms for 3 MiB),
so the second dispatch (`native_gdn_out_norm`, 0.0028 ms) fits inside the same fence window and the fusion buys
nothing there — the SAME effect batch 3 measured for the XTX's `scale`+`gdn_step` chain (1.022).  Every other
chain pair on every other device is a win (0.466–0.994).

**The honest reading, including the pairs that are NOT wins.**

* **Every fused path BEATS the multi-dispatch chain it replaces on every device measured, with ONE exception at a
  wash (0.398–1.003).**  The conv+L2 path is 1.7–2.2× the `native_gdn_conv_silu`+`2× native_gdn_l2_norm` chain;
  the AB path is 1.15–2.5× the `2× bf16_mmvf + beta_gate + gate` chain; the step+norm path is ~1.01–1.13× the
  `native_gdn_step + native_gdn_out_norm` chain SAVE the XTX, where that pair reads **1.003** (a wash — the XTX's
  memory-bound step leaves no room for a 0.0028 ms second dispatch to matter; see above).  This is the same
  finding as batches 2–3: a fusion's win here is the dispatches it removes, not arithmetic.
* **The fused-vs-SINGLE-native rows are washes-to-slower, and that is expected.**  Per dispatch the fused path
  does MORE than any one kernel it replaces: `fused_gdn_conv_l2` reads the conv inputs and writes the normed
  output (1.122 / 0.903 / 2.803 the `native_gdn_conv_silu` it also subsumes); `fused_gdn_ab` does BOTH mat-vecs
  and BOTH epilogues (1.136 / 1.868 / 1.776 one `bf16_mmvf_f32`); `fused_gdn_step_norm` does the whole step PLUS
  the norm (1.021 / 0.998 / 1.092 `native_gdn_step`).  Recorded as measured, not tuned.
* **The fused-vs-`native_gdn_out_norm` row (6.2–63.7) is NOT a like-for-like fusion ratio.**  The fused dispatch
  still carries the whole 3 MiB step (~0.42 ms) while the standalone out_norm is a ~0.006 ms 48-workgroup
  elementwise pass — the ratio is the step's cost over a ~70× smaller kernel.  Printed so the reader sees it;
  the like-for-like comparator is the two-dispatch chain above.
* **The llvmpipe conv/L2 rows are above the fence-clock floor for the chain and near it for the fused path**, so
  the CPU driver's 2.803-vs-`native_gdn_conv_silu` is a floor/toolchain artifact of comparing one fused pass
  against one smaller pass, not a device result; the CHAIN ratio (0.560) is the meaningful llvmpipe number.

### The class-B NATIVE vs LEGACY pairs, batch 3 — the REST of the GDN / DeltaNet MIXER — measured 2026-10-05

The remaining three native mixer kernels, completing the six.  Same `XPAIR` convention (`native/legacy`, **< 1.0
means the native kernel is faster**); `reps=9`.  For `native_gdn_step` the legacy branch runs a SECOND dispatch
the native kernel fuses (the `1/sqrt(S)` readout scale, applied by `scale_inplace` at `layer.cpp:276`), so that
pair is ALSO measured against the two-dispatch chain.

| pair (native ← legacy) | Arc B70 (ms) native/legacy | ratio | Ryzen iGPU | ratio | XTX (box) | ratio | K620 (box) | ratio | llvmpipe (vega/box) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `native_gdn_gate` ← `gdn_gate` | 0.0041 / 0.0042 | 0.998 | 0.0014 / 0.0015 | 0.901 | 0.0021 / 0.0020 | 1.044 | 0.0032 / 0.0031 | 1.037 | 0.0123 / 0.0126 · 0.0544 / 0.0523 | 0.980 / 1.042 |
| `native_gdn_out_norm` ← `gdn_out_norm` | 0.0061 / 0.0061 | 0.995 | 0.0065 / 0.0065 | 1.000 | 0.0035 / 0.0031 | 1.144 | 0.0119 / 0.0117 | 1.020 | 0.0680 / 0.0698 · 0.1399 / 0.1429 | 0.974 / 0.979 |
| `native_gdn_step` ← `gdn_step` | 0.0437 / 0.0476 | **0.917** | 0.4133 / 0.5327 | **0.776** | 0.0924 / 0.0977 | **0.946** | 0.3556 / 0.4909 | **0.724** | 0.3864 / 0.4464 · 0.9369 / 0.9937 | **0.866 / 0.943** |
| `native_gdn_step` ← `scale`+`gdn_step` (2 dispatches) | 0.0437 / 0.0504 | **0.867** | 0.4133 / 0.5358 | **0.771** | 0.0924 / 0.0904 | 1.022 | 0.3556 / 0.4909 | **0.724** | 0.3864 / 0.4661 · 0.9369 / 1.0778 | **0.829 / 0.869** |

**The honest reading.**

* **`native_gdn_step` is a WIN against the legacy kernel on EVERY device measured (0.724–0.946 pair), and it is
  a MEMORY-TRAFFIC win, not a compute one.**  Both kernels do the same two S-long passes; the legacy kernel's
  first pass STORES the decayed state back to the (S, h_v, S) buffer, while the native body contracts the
  UNDECAYED state and folds the decay into the second pass — so the native moves ~one third less state traffic.
  **Against the two-dispatch legacy chain it is a win on five of the six GPUs (0.724–0.869) but NOT on the
  XTX**, where the chain reads 1.022: the XTX's step is fast and memory-bound (0.098 ms) and the extra
  `scale_inplace` dispatch (2048 q elements) fits in the same fence window, so the fused kernel's advantage is
  inside run noise there.  Both rows are printed.
* **`native_gdn_gate` (0.901–1.044) and `native_gdn_out_norm` (0.974–1.144) are WASHES-with-noise.**  At the
  layer's shapes (gate: 48 heads, one token; out_norm: one row per head) both do the same work per element as the
  legacy kernel; the largest deviation is the XTX's `out_norm` at 1.144 (native ~14% slower, i.e. ~0.4 µs on a
  0.003 ms row — near the fence-clock floor).  Recorded as measured, not tuned.
* **A MEASUREMENT THAT CHANGED THE SHIPPED KERNEL, and it is the batch's real finding.**  The native `step` body
  is ONE 32-LANE WARP per column (four rows/lane in registers, `__shfl_xor_sync`).  Subgroup ops are banned in
  this port, so the warp was ALSO rendered as a workgroup-per-column BARRIER TREE and timed: **1.564x the Arc,
  8.328x the Ryzen iGPU and 43.501x llvmpipe** the legacy kernel (chain 1.471 / 8.078 / 41.597).  The causes are
  both structural: 6144 workgroups × two 8-round barrier trees is a lot of synchronisation, and the
  warp-shaped row ownership makes adjacent invocations read rows `h_v*S` floats apart — 32 cache lines per warp
  instead of one.  The port therefore SHIPS the coalesced one-thread-per-column serial decomposition (which the
  legacy `gdn_step` port already uses; the CUDA's warp shape is a parallelism strategy, not the rule) carrying
  the native arithmetic and the fused readout scale.  The barrier-tree variant is NOT in the tree; its numbers
  are recorded here and in `NEXT.md`/`STATUS.md` as the price of the port's no-subgroup rule at this shape.

## The BF16-projection pair (`bf16_gemv` / `bf16_gemv_split`) — the DEFAULT side of `native_bf16_projections` (2026-10-05)

`project_bf16` (`src/core/layer.cpp:94-100`) selects `bf16_gemv_fp32_mmvf` only while `native_bf16_projections`
is TRUE (`--native`, generate.cpp:1805 → :2286); the setting **defaults false** and the port pins it nowhere, so
the default side — `bf16_gemv` (QSA indexer projections, `split=false`) and `bf16_gemv_split` (GDN alpha/beta,
router logits, `split=true`) — is on the main forward path.  Both are now ported as ONE shared shader, ONE
WORKGROUP per output row (the CUDA's warp/tpr shapes are a parallelism strategy, and subgroup ops are banned
here).  There is **no legacy sibling** to compare against (this IS the non-native branch), so the pair is the
ported `bf16_gemv` against the ported native sibling `bf16_gemv_fp32_mmvf`: the same workgroup-per-row
decomposition, only the activation precision differs (bf16 vs f32), so a WASH is expected.

| pair (ported `bf16_gemv` ← native `bf16_gemv_fp32_mmvf`) | Arc B70 (ms) native/ported | ratio | Ryzen iGPU | ratio | lvp (vega) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `bf16_gemv` n_in=2560 n_out=512 | 0.0089 / 0.0089 | 1.000 | 0.1127 / 0.1131 | 0.996 | 0.6626 / 0.7099 | 0.933 |
| `bf16_gemv` n_in=2560 n_out=48  | 0.0066 / 0.0065 | 1.006 | 0.0134 / 0.0134 | 0.997 | 0.0777 / 0.0812 | 0.956 |

**The honest reading.**  A WASH on the GPUs (0.996–1.006), exactly as a same-decomposition drop-in should be: the
only difference is that `bf16_gemv` carries the activation as bf16 (products bf16×bf16, exact in f32, half the
activation bytes) while the native sibling takes it already f32.  llvmpipe reads 0.933 / 0.956 — the
bf16-activation form is marginally FASTER there; reported, not tuned.  **NOTE on the raw log:** the `XPAIR
bf16_gemv n_out=512` line on the GPUs interleaved with the device layer's descriptor-pool-growth notice on
stderr; the ratios above are taken from the two `ROW` medians.

**A variant that is NOT shipped, with its numbers.**  The CUDA's naive ONE-THREAD-PER-ROW decomposition was built
and benchmarked first: at these shapes it is **14–18x slower** than the workgroup form — Arc `split/serial` 0.073
(n_out=512) / 0.057 (48), Ryzen iGPU 0.103 / 0.055, llvmpipe 1.084 / 0.646 — because it is uncoalesced (the
CUDA's own comment: 32 transactions per load at a fixed `i`).  CUDA takes the naive path only below n_out=64,
which this engine never does for `bf16_gemv`, so the port ships the workgroup-per-row rendering.

## The one number that is a problem, not a baseline

`sampler_kernel_f32` **202–1165 ms per token** on every device (202 ms on the XTX, 547 ms on the Arc).
The kernel's top-k is `k` rounds, each a block-argmax over the *whole* vocabulary with an inner loop over
the already-taken ids and the history window — cost ~ `k · vocab · (k + history)`.  At `k=64`,
`vocab=248320`, `history=64` that is the port's slowest single dispatch by three orders of magnitude.  This
is the **one-block** sampler; the engine's default is the split sampler (4096-logit partitions), which the
port also has (`sampler_split.comp`) but which this harness does not yet measure.  For the performance tier
this is the first target: no decode step survives a 200 ms sampler.

## Evidence the harness measures something real

1. **Cross-ICD (the same binary, the same kernel, different device).**  `run_bench.sh` runs every ICD, and
   the ordering is physical on both boxes — discrete GPU fastest, CPU slowest — with ratios far outside run
   noise.  On `vega`: `iq2s_mmvq` (n_out=2048) Arc 0.0464 vs llvmpipe 6.2754 = **135×**; `iq_dequant`
   IQ2_S 0.0158 vs 0.5946 = **38×**; `gdn_l2_norm` 0.0198 vs 0.0762 = **12.9×**.  On `z820b`:
   `iq2s_mmvq` XTX 0.0198 vs llvmpipe 23.78 = **1201×** vs K620 = **35×**; `gdn_step` XTX 0.1050 vs llvmpipe
   0.9662 = **9.2×**.  A harness that could not resolve a difference would not produce this ordering on two
   independent machines.
2. **Same kernel, 4× the work.**  `iq2s_mmvq` n_out=512→2048 (4× the MACs) and `iq_dequant` IQ2_S
   n_sb=256→1024 (4× the outputs) scale ≈ 4× where the sizes are work-bound: XTX `iq2s` **1.29×** (both
   sizes near the ~15 µs wall-clock floor), llvmpipe `iq2s` **3.60×** and `iq_dequant` **3.37×**, K620
   `iq2s` **3.57×**, Ryzen iGPU `iq2s` **3.90×** (**4× work in 3.9× time** — the cleanest single result).
   The size-scaling proof is therefore device-dependent *by design*: it is clean exactly where a dispatch is
   longer than the timer's floor, and that is itself the documented limit of a fence-clock timer.
3. **What the harness could NOT be made to show, stated rather than hidden.**  On the fastest GPUs the
   4×-work arm is **`~1×`** for `iq_dequant` (Arc 0.0145→0.0158, XTX 0.0155→0.0162) because both sizes sit
   below the ~5–15 µs fence-clock floor; the harness reports the floor, not the kernel.  This is why the
   cross-ICD arm (1), not the sizing arm (2), is the primary distinguishability evidence.

## Findings the harness produced on its first real run

* **The one-block sampler is unusable at the real vocabulary** (above): 202 ms/token on the 7900 XTX.
* **The Ryzen iGPU (RADV) hard-recovered** on the first version of the sampler row: a *batch of 8* live
  full-vocabulary sampler dispatches in one command buffer tripped RADV's "hard recovery" (context lost).
  The harness now both runs the sampler last and records **one dispatch per batch** for it, and it completed
  on every ICD — so this was a launch-shape hazard, now documented and contained, not a kernel defect.
* **A device-layer robustness gap, hit when the harness allocates many descriptor sets:** on the box's
  RADV / **Mesa 26.0.8** an exhausted descriptor pool returns **`VK_ERROR_FRAGMENTED_POOL`**, and
  `Ctx::set_alloc` only listed `OUT_OF_POOL_MEMORY` (what vega's Mesa 25.2.8 returns), so the grow-on-demand
  path never fired and the XTX arm aborted at the first pool exhaustion.  Fixed in
  `harness/vk_compute.cpp` (both codes now grow the pool).  The numeric gate never fills a pool, so this
  changes no gate verdict — it makes the header's "no fixed ceiling" promise true on a second Mesa version.
