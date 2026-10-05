# Running the strata-supplied model on the B70 — the shortest path, correctness first

**The goal, stated so it can be checked:** the engine loads a strata-supplied GGUF on this box and emits tokens on the
Intel Arc Pro B70, correctness first.  Optimisation is explicitly deferred — the prompt path, the fused variants and
the remaining quant instantiations do not have to land for this.

**The model that has to work:** `~/strata-models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf` —
**58 GB in two parts against 32 GB of VRAM.**  That is not a detail to work around: the engine's own expert file-tier
and resident/low-RAM modes are the reason this model runs on this card at all, so the integration has to include the
expert streaming path, and a first run cannot be a "load it all into VRAM" run.

## The ladder

| | Milestone | What decides it | Verified by |
|---|---|---|---|
| **M0** | This plan + the two inventories below | the engine's own call surface, the port's shader list | done (this file) |
| **M1** | The port is the engine's **backend**: a `vulkan` option in `setup.py`/CMake and an implementation of the kernel API the decode path calls | `setup.py` builds per backend today (`--backend hip`, CUDA via nvcc, `-DSTRATA_*` flags); there is no `STRATA_ENABLE_VULKAN` | the engine links and starts, **on llvmpipe first** |
| **M2** | The **decode path** is complete: the sampler, the embedding gather, `cvec_apply`, and the shape variants the model actually uses | the inventory below; **the sampler's first kernel is done** (`sampler_greedy.comp`, the `--temp 0` path, 253 verdicts green on the Arc) - `sampler_kernel` (temperature/top-k/top-p with the Philox draw) is the remainder, and with the greedy path the port can already turn logits into a token | a decode step produces a token, on llvmpipe |
| **M3** | The **expert tier streams** the 58 GB model from NVMe | the engine's resident/file-tier modes; the port has the primitives (device-local + staging + `plan_fit`, stage 4) but not the engine's expert cache management (host-side) | the model loads and a layer mixes experts |
| **M4** | **First token on llvmpipe** with the real model | M1–M3 | tokens in stdout, no GPU |
| **M5** | **First token on the B70**, then measure | M4 + a GPU-visible run | tokens/s on the Arc, no xe wedge |

Only after M5 does the deferred work matter (below).

**CORRECTION 2026-10-05.**  The **M2** row above reads "the decode path is complete", and `RUN-ON-B70.md`'s
inventory counted only the `kernels::`-qualified symbols.  The corrected port map (`7c317c4`) and the per-symbol
triage (`plan/DECODE-PATH-TRIAGE.md`) show the decode path is **not** complete: 19 class-A forward-path holes
remain, chiefly the **GDN / DeltaNet mixer** on 36 of the 48 layers and the **QSA gate and indexer** on the other
12 - none of which `layer.cpp`'s bare-name `gdn_layer`/`qsa_layer` calls let the old scan see.

## What the port already covers for a decode step

The gate reports **238 cases, 0 failed, 0 skipped on the Arc**.  Among them, the whole decode data path and both
ends of the KV: norms, RoPE (`rope_neox`), the quantisers/dequantisers, the router, all six expert formats the model's
pack uses (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0) plus two grouped variants, the GEMV splits, the short-step
attention, the QSA selection, the f16/q8/q4 KV append+gather, the FWHT rotation, the hybrid K8V4 mode, the MoE
combines, PLE, and the prefill GEMM (not needed for M4 — see the bypasses).

## The gap, from two inventories

`src/core/layer.cpp` mentions **55 distinct `kernels::` symbols; 23 have a name match in the port.**  Of the other 32,
most are not kernels at all: types (`Codebook`, `FusedGrArgs`, `GrShapes`, `KvHostPools`, `KvStreamMap`, `SForm`,
`cvec`), constants (`KV_Q8_GROUP`, `NG_HC_DIM`, `NG_N_EMBD`, `kKvCtlInts`, `kStepCount`, `kTopkMaxCells`), and
byte/workspace sizing (`gr_workspace_bytes`, `kv_block_bytes`, `kv_q8_bytes_per_cell`, `gr_workspace_init`) — all
host-side and unchanged by a backend swap.  **The real device-side remainder is small:** `embedding_gather`,
`cvec_apply`, the RoPE table (`build_rope_table` / `rope_table_set` / `rope_scaling` — check whether host), the
`kv_*_q4_step` launchers (their kernels are ported; the wrappers are host), `kv_ring_table` (the streaming ring — part
of M3), and **`sampler.cu` (60 KB), which is not mentioned in `layer.cpp` at all** because sampling happens outside
the layer loop — and without it there is no token.  `fused_gr_supported()` and `native_gdn_enabled()` show the engine
carries its own fallbacks, so the fused gate-RoPE family is optional.

## Bypasses that keep M4 close

* **The batched prefill is not on the path.**  A prompt can be fed through the decode path token by token, so
  `prefill/kernels.cu`, `moe_fused*.cu`, `moe_mmq.cu`, the 7 `cuBLASLt` call sites and `fused_gr.cu` — roughly 250 KB
  of porting — are all deferrable.  Slow prompt ingestion, which "not fully optimized" already allows.
* **`--kv f16`** avoids q4/q8/hybrid entirely for the first run (all three are ported anyway).
* **Speculative decoding off** keeps `ngram_rows`/the draft head off the path.
* **llvmpipe first** — a CPU Vulkan implementation.  Integration bugs (descriptor sets per dispatch, the arena, the
  recorded step) are then debuggable with no GPU in the loop, and the gate already exercises llvmpipe on every run.

## What M1 has to build, concretely

The port's `harness/vk_compute.*` is the seed, not the backend: it has device selection, device-local allocation,
staging transfers, `plan_fit`, single-shot dispatch and recorded steps — each gated — but:
* **descriptor offsets.**  The engine passes row slices by pointer arithmetic (`Y + t0*ldy`, `X + t0*K`), and the
  harness binds every descriptor at offset 0.  `VkDescriptorBufferInfo::offset` closes this; it is the one known
  blocker recorded in `NEXT.md`.
* **the arena.**  The engine plans one buffer carved by byte offsets (`qsa_buffers_init` and friends show the shape);
  the port allocates buffers per case.  The arena is what makes the engine's `bytes_per_cell`-style arithmetic hold.
* **the kernel registry.**  The engine's host code calls `strata::kernels::name(...)`; the port needs a table from
  those names to pipelines + a dispatch that takes the engine's arguments, not the gate's.
* **the recorded decode step in the engine.**  The engine re-issues a fixed sequence of dispatches per token; the port
  has the recorded-step API (stage 3) but nothing in the engine calls it.

## Cost, honestly

The port's own estimate stands: the remaining kernel waves are a grind at a measured rate (the first 8 kernels took
~40 minutes end to end including gating), the sampler is a session on its own, and M1 is the piece with unknown
corners because it is the first code in this tree that the *engine* drives.  M4 (a token on llvmpipe) is the first
milestone worth showing; M5 is a hardware window after that.

## M5 REACHED, MEASURED (2026-10-05, `vega`)

**A token came out of the Intel Arc Pro B70.**  This is M5 on a SYNTHETIC pack, not on the real model (M3/M4's
real-artifact path is still open - see below).

Exact command (run 1):

```
STRATA_VK_SPV_DIR=/home/bob/strata-vulkan-wt/ports/vulkan/shaders \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json STRATA_VK_ARENA_GIB=16 \
  ~/bin/memguard 40G ~/vkbuild-vulkan/vulkan/strata_vulkan \
  --pack /media/bob/3d651e2c-e9a4-4758-ba77-863725fe3731/strata-synth-pack-zero \
  --tokens "1" --max-new 1 --no-pool --mmap-experts --no-ple --max-context 8
```

Raw output (the tail; the load prints ~900 upload progress lines first, ~9.5 min):

```
strata generate: 5890 MiB of weights loaded from .../strata-synth-pack-zero (0 canonical tensors skipped: served natively)
strata generate: GPU 0: Intel(R) Graphics (BMG G31), compute capability 0.0
strata generate: session is up (engine 0.1.39)
strata generate: sampling greedy
strata generate: token graph captured (48 layers, one launch per token)
strata generate: position 0, token 1 (prompt)
prompt  : 1
output  : 0
decode                   1 tokens in 195.6 ms  ->  5.11 tok/s
RUN_RC=0
```

- **Device / ICD:** Intel Arc Pro B70 (`Intel(R) Graphics (BMG G31)`), `intel_icd.json`, Mesa 25.2.8, kernel
  7.0.0-34.
- **Sampler path:** `sample_tokens` (generate.cpp:7742) with `sp.greedy = true` (the DEFAULT; no `--seed`) -> the
  greedy argmax shader `sampler_greedy.spv`.  The log line `sampling greedy` is that choice.
- **Pack geometry:** synthetic, the engine's canonical geometry (48 layers, n_embd 2560, 512 experts); dense.bin
  6.448 GiB + embd.bin 0.666 GiB + `experts.bin` SYMLINKED (never read under `--no-pool --mmap-experts`);
  `index.txt` 1073 tensors, engine pool 5.752 GiB.  **Its weight MATRICES are ALL ZERO** (`make_synth_pack.py
  --zero`): zero in -> finite zero logits -> the greedy argmax is index 0.
- **BOUNDS, in the same breath:** the weights are zero, so the token's CONTENT is meaningless and trivial; this
  certifies the PIPELINE (open -> load -> capture -> one launch -> sample), NOT the model, and NOTHING about
  layer numerics (the gate's 750 per-kernel cases are the only numeric evidence).  PLE is OFF (`--no-ple`), the
  CPU expert pool is UNUSED (`--no-pool`), prefill is BYPASSED (`--prefill` defaults to 0; the prompt is one
  token fed through the decode path), the expert streaming tier is NOT exercised (`--mmap-experts` on a symlink
  that is never read), multi-GPU/speculative/verify are off, and the `iq3`/`iq1_m` quant families are not on
  this pack (it is Q8_0/BF16/F32 only).

**Two runs are recorded**: the ZERO pack (token 0, above; a second identical run repeats token 0 exactly), and
the ORIGINAL random pack, which now runs the whole decode and stops at the engine's own
`248320 of 248320 logits are not finite at position 0`.  The zeros run DISCRIMINATES that: a zero pack is
finite, so the composed 48-layer chain is arithmetically sound and the non-finiteness is the FIXTURE (random
weights overflow this architecture).  A `--weight-scale 0.02` pack - measured to have quant scales 50x smaller,
mean 2.97e-4 vs 1.49e-2 - STILL diverges: **an unresolved FIXTURE LIMIT, not a port defect.**  The real model
(`coder-iq1_m`, 1.4 GB dense + 25.1 GB experts read from the GGUF shards, NO pack `experts.bin`) needs the
native-expert path this port has not exercised; that is the next item.

