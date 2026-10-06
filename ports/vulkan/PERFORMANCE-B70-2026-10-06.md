# Measured performance on the Intel Arc Pro B70 — 2026-10-06

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
