# Measured performance on the Intel Arc Pro B70 — 2026-10-06

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
