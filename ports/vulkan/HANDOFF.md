# Handoff - the Vulkan port after the sampler, the embedding gather, and the move to the 7900 XTX

Written 2026-10-04 at the end of a long session, and it SUPERSEDES the earlier handoff on this host (whose machine
facts still hold: `vega` has the Intel **Arc Pro B70**, the display is on the **Ryzen iGPU**).  **Keep the earlier one
to hand** - it is commit `d595375`, readable with `git show d595375:ports/vulkan/HANDOFF.md`, and it carries what this
document does not: the Arc swap and why it happened, the BOOT FAILURE and its cause, the NVMe staging that is a husk
of symlink stubs (`~/strata-gguf-iq3/flat/`), `~/start-strata-flash-next.sh` and its preflight, and the measurement
caveats.  This document is about the PORT.  It assumes you know nothing about this session, so it states what is true,
what was measured, and what is genuinely unknown.  Read this first, then `NEXT.md`'s top block, then `STATUS.md`, then
`PORT-MAP.tsv`.

Repo: `/home/bob/strata-vulkan-wt`, branch `vulkan-arc-port`, tree clean at `d1237ee`.
Gate on this box: **Arc 272 passed / 0 failed / 0 skipped**, llvmpipe 260/0/3, radeon-iGPU 263/0/2, `run_gate.sh` exit 0.

---

## 1. What this session added (all committed)

| commit | what |
|---|---|
| `dd9fd17` | **descriptor offsets** - the recorded blocker between the port and the engine, retired.  `Buf` gained a view offset, the single descriptor-write site honours `VkDescriptorBufferInfo::offset`, and the measured alignment is **4 bytes** on the Arc (not the 64 a comment claimed), so the engine's `X + t0*K` row slices are bindable as they stand. |
| `5da8ba4` | **`sampler_greedy.comp`** - the `--temp 0` path with the penalty pair, and the first TOKEN this port can emit. |
| `5066380` | **`sampler_kernel.comp` + `common/philox.glsl`** - the general path: penalties, top-k, top-p, min-p, temperature, the softmax, and one Philox draw. |
| `11a84f4`, `0925a58` | **`PORT-MAP.tsv`** - every `kernels::` symbol the DECODE path calls, classified, with a checker wired into the gate. |
| `a304dad` | the map corrected: `iq_embed_rows` is GPU work, not host-side bookkeeping (found by reading its header). |
| `d1237ee` | **`embedding_gather.comp`** - packed codes + per-group scales -> float rows, and the `precise` finding. |

Gate totals moved 238 -> 272 verdicts.  Everything landed with the usual discipline: a gated case, falsification by
injection, three implementations, and the docs kept current.

---

## 2. THREE FINDINGS A FRESH SESSION MUST NOT RE-DERIVE

1. **`precise` is load-bearing.**  A driver may FUSE a multiply and an add unless the result is qualified `precise`,
   even though the SPIR-V contains no fma op.  GLSL's default permits *contraction*; one fused op rounds ONCE where
   the engine's `__fmul_rn` + `__fadd_rn` round TWICE, and the engine compares bitwise.  Measured: four of six
   embedding arms failed before `precise`, and the two that passed were those whose products happen to be exact --
   that signature is what identified the cause.  **This applies to any kernel ported to match an engine that controls
   its rounding.**
2. **The sampler's two "fixable" behaviours.**  The penalties belong on the RAW logits, exactly ONCE, before the
   filters (the source records having applied them a second time after the temperature).  And **`temperature == 0` is
   NOT greedy**: `inv_t` is zero, so every survivor scales to zero and the draw is UNIFORM over the shortlist; greedy
   is the separate path.  "Fix" either and every zero-temperature token changes.  Also: the engine's Philox uses
   **its own constants** (`0x9E3779B9 / 0xBB67AE85`), not Random123's (`0xD2511F53 / 0xCD9E8D57`) - a swap yields a
   perfectly good generator producing different numbers, which reads as "the model got a bit worse".
3. **The grid is the CAPACITY, not the live count** - the engine's own rule, measured through the record/replay path
   (capacity grid -> 8/8 rows after replay; live-count grid -> 6/8).

---

## 3. THE TARGET GPU MOVED: the 7900 XTX IS IN z820b NOW

* **`vega`** = the Intel **Arc Pro B70** (the port's development box) + the Ryzen iGPU driving the display.  There is
  **no AMD dGPU here any more** - the 7900 XTX was moved out, which is exactly what a fresh session must not assume
  (the earlier handoff's machine section is stale on this point).
* **`z820b`** = `192.168.1.116` (`bob`, ssh key works) - HP Z820, 2x E5-2687W (**AVX1-only, no AVX2**), 92 GB RAM,
  **AMD Radeon RX 7900 XTX 24 GiB as "AMD Radeon RX 7900 XTX (RADV NAVI31)"**, RADV, Mesa 26.0.8, api 1.4.335, plus
  a Quadro K620 on the NVIDIA driver.  `/opt/rocm` exists.  **It suspends when idle (~2700 s) and needs a manual
  wake** - keep work brisk, use ssh keepalives.
* **The port is already copied there**: `~/strata-vulkan-wt` (7.8 MB: `include/`, `src/`,
  `ports/vulkan/{harness,shaders,gates,tools}` + the docs).
* **THE COPY NEEDS `third_party/ggml/ggml-common.h`.**  Without it the gate's generated-table check
  (`tools/gen-iq-tables.py --check`) fails and the run ABORTS BEFORE ANY KERNEL - measured on the first attempt.
  Copy it together with the rest:
  `tar czf - include src third_party/ggml/ggml-common.h ports/vulkan/{harness,shaders,gates,tools} ports/vulkan/*.md ports/vulkan/*.tsv | ssh bob@192.168.1.116 'tar xzf - -C ~/strata-vulkan-wt'`.
  Do **not** copy `ports/vulkan/logs` (246 MB of gate logs) and do not bother with `*.spv` (gitignored; the gate
  compiles and validates them itself).
* **Run the gate there:**
  `cd ~/strata-vulkan-wt && STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 bash ports/vulkan/gates/run_gate.sh`
  The cross-implementation arm will pick up radeon (the XTX), llvmpipe, and possibly the K620 - an old device failing
  cases there is a DATA POINT, not a defect in the port.

---

## 4. HOW MUCH FURTHER TO SOMETHING THAT GENERATES TOKENS

`PORT-MAP.tsv` (checked by the gate) classifies **every** kernels-namespace symbol the decode path (`src/core/`)
reaches, written `kernels::X` or BARE `X`.  **Corrected 2026-10-05 (`7c317c4`): the map reads
`168 symbols - 53 kernel, 63 host, 52 todo`** - the earlier `77 symbols - 28 kernel, 49 host, 0 todo` was measured
by a scan that keyed on the `kernels::` qualifier and could not see the bare-name call sites; see `NEXT.md`'s top
section and `plan/DECODE-PATH-TRIAGE.md`.  The port has landed a lot of the inference core and it is gated:
attention, the KV cache in all four modes (f16 / q8 / q4 / hybrid), the QSA block selection, the embedding gather,
the router top-10, the quantised matvec family, rms_norm / rope / silu / swiglu - and **both samplers, so the port
can turn logits into a token**.  But the corrected map shows **M-A is NOT closed**: 52 rows are `todo`, of which
**19 are genuine forward-path holes with no ported fallback** - the GDN / DeltaNet mixer (36 of the 48 layers),
the QSA gate and indexer (12 layers), and `gr_write`.

**The ten holes section 4 used to list are all landed** (`cvec_apply`, `gather_rows`, `scatter_rows_f32`,
`iq_dequant_f32`, `iq_embed_rows`, `native_q5_k_f32`, `moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`,
`moe_hit_grouped_s2`) - but they were only the visible tip, and the class-A 19 above are the real remainder.

**Three things stand between here and tokens, in order:**

1. **Those ten kernels.**  Roughly one increment each (shader + gated case + falsification + docs) at the rate this
   port has been moving.  Five or six of them are the MoE/expert half.
2. **The backend does not exist in the engine.**  Backend selection is COMPILE-TIME macro driven
   (`STRATA_ENABLE_CUDA` / `_HIP` / `_SYCL`; there is no `_VULKAN`), and each GPU entry point is a thin wrapper in a
   header calling the backend's implementation (`kv_q4.hpp`: `fwht256_inplace_cuda(...) { fwht256_cuda(...); }`), with
   per-backend translation units behind it.  This is **integration**, 3-5 increments of a different kind: the numeric
   gate proves KERNELS, it says nothing about a PROGRAM, and that is where the surprises are.  The port's
   `harness/vk_compute.*` is the seed of the device layer (arena, grow-on-demand descriptor pool, view offsets).
3. **58 GB of weights against 24 GiB of VRAM.**  The engine's expert file-tier streaming is required, not optional.
   It is host-side and already in the engine (the map's `kv_stream_*` rows), but the Vulkan path has to satisfy its
   residency assumptions.  Feed the prompt through the DECODE path so the batched-prefill port (~250 KB) stays
   deferred - slow but real.

**Milestones worth judging:**

* **M-A** - the decode path's last kernels gated.  **RE-DEFINED 2026-10-05** (the old form, "the map's `todo`
  column reaches zero for the forward-path work", is not achievable or meaningful - the map also covers the
  `native_*` siblings, the fused alternatives a flag removes, and the verify/MTP/tooling helpers).  The
  re-defined M-A is: **every symbol the forward path reaches ON THE BRANCH THE CAPABILITY CONTRACT SELECTS has a
  shader and a gated case**, under `native_gdn_enabled() == false` + `native_qsa_enabled() == false` +
  `native_qsa_indexer_enabled() == false` + `native_rope_enabled() == false` + `native_router_enabled() == false`
  + `native_moe_combine_enabled() == false` + `gr_set_native_mmvf(false)` + `layer_set_fused_gr(false)`.
  **MET: YES** (class-A remaining 0; the 45 `todo` rows are `11 capability-off + 4 B + 8 C + 22 D`), with two
  stated soft edges (the shipped `--spec 4` MTP symbols - a judgement - and the `gr_read`/`fused_gr_read` branch
  choice).  Full statement: `plan/DECODE-PATH-TRIAGE.md`.
* **M-B** - **ONE LAYER, end to end, on the GPU with random weights.**  The first thing that proves the BACKEND
  rather than the kernels, and it needs no model at all.  Make it a gate case: a Vulkan-backed single-layer forward
  pass.
* **M-C** - the real model emitting a token, on llvmpipe first (deterministic, no VRAM ceiling), then on the 7900 XTX.

**Estimate: ~2-3 more sessions of this size.**  The kernels are the predictable half (~1); the integration is 1-2 and
carries the risk.

**A shortcut that was considered and REJECTED by the user: the CPU hybrid.**  The engine's CPU expert/MoE
implementations (`src/kernels/cpu/`: expert, native_expert, iq_avx2/512, kq_avx1/2, q2_avx2, router, pool) are
always compiled in and used unconditionally for layout and routing, so a "GPU attention + CPU experts" mode is
architecturally plausible and would skip five or six kernels.  The user's instruction was explicit: **no CPU path -
the target is the 7900 XTX.**  Do not re-open it.

---

## 5. THE DISCIPLINE (unchanged - it is what makes the numbers mean anything)

* **Every case is falsified**: inject the wrong rule, confirm the arm fails, revert, and record honest negatives.
  An injection that silently does not apply is worse than none - the script must print `ANCHOR MISSED` or
  `DID NOT COMPILE` rather than run the stale binary, and must recompile the shader that INCLUDES a changed file.
* **The fixture's margin must serve the arm's CLAIM.**  Reachable is not ordered (a head that appears under either
  ordering proves nothing - use a one-token shortlist); a majority test is noise-dominated once probabilities
  compress; a membership check cannot see too FEW survivors; equal logits make a scaling factor irrelevant.  Give
  every arm a stated expected value and check it against the oracle too, so a stale fixture fails loudly.
* **Three implementations, every run**, plus the target: the cross-implementation arm always runs.
* **The gate's own printed totals are the authority**, and a skipped case is not a passing one.
* **Known environment, not a defect:** the radeon ICD intermittently fails `budget: independent requery agrees`
  (the driver's figure drifts ~2.8 MB against a 1.7 MB tolerance; an older commit reproduces it).  Re-run; record it.

---

## 6. OPEN ITEMS

1. **THE 7900 XTX RUN'S TWO FAILURES ARE RESOLVED (2026-10-04).**  The run found `radeon_icd 267/1/1`,
   `lvp_icd 259/1/3`, `nvidia_icd (Quadro K620) 263/0/2`, EXIT=1.  Both failures are now fixed and the box reads
   `radeon_icd 268/0/1`, `lvp_icd 260/0/3`, `nvidia_icd 263/0/2` (the XTX's one skip is the pre-existing M8
   cooperative-matrix case).  One was a wrong CASE BOUND, the other a driver bug the SHADER leaned on:

   * **`kv_q4 round trip` - the CASE's bound was wrong, not the shader.**  The failing element is printed now:
     cell 5 head 0 dim 193 (the box's own bytes: `d16=0x3642 code=15 -> (code-8)*d16 = 2.73779`, host rule code
     `plain=15 fma=15`, `dev-vs-rule mismatch 0` over all 3072).  The rule is `d = mval/-8` from the SIGNED extreme
     then `code = clamp(trunc(x/d+8.5),0,15)`, so the codes are `d*[-8,+7]` - an ASYMMETRIC range whose +8 end does
     not exist.  An element that needs the +8 end (`x/d = 7.96` here) is clipped to code 15 and is off by up to
     `|d|`, TWICE the `|d|/2` the case allowed (that is the bound of a round-to-nearest rule in a symmetric range).
     Fixed: the case now bounds each element against its OWN group's `|d|` and replays the rule on the host for
     every element (so it can SHOW the shader faithful).  Contraction was never involved - the gather is still a
     lone multiply, no `precise` added.  The old verdict compared a GLOBAL worst against a GLOBAL bound, which is
     fixture-sensitive: the fixture is generated from a shared RNG that device-conditional SKIPS shift (the Arc
     runs `gemm_coopmat`, llvmpipe/radeon skip it), so the Arc passed at ratio 0.995 where the box failed at 1.22.
   * **`quantize_q8_0 (ggml bytes)` - a lavapipe/Mesa 26.0.8 driver bug, and the case was RIGHT.**  All 12 differing
     bytes are in the exact-tie block (`amax = 127 -> d32 = 1.0`, so the quotient IS the value): the device wrote
     `3.5 -> 3`, `-3.5 -> -3`, `1.5 -> 1`, i.e. `roundEven(double)` rounds half-toward-zero there.  The engine's
     rule (`src/kernels/cuda/quantize_act.cu`, `rint`) is ties-to-EVEN and admits ONLY that answer, so the case is
     not over-tight and was not loosened.  Fixed in the SHADER: `roundEven(double)` is replaced by an explicit
     ties-to-even (`floor` + parity), which the older lavapipe and every other implementation already agreed on.
     A skipped case would be a different matter; this one is a real divergence, recorded.
2. **The sampler's remaining variants** (CLOSED - see item 3) and the ten kernels of section 4 (all landed - but
   they were not the whole hole: **M-A is NOT closed**; corrected 2026-10-05 the map's `todo` column reads 52, of
   which 19 are class-A forward-path holes, see `plan/DECODE-PATH-TRIAGE.md`).
3. **The sampler's remaining variants - CLOSED 2026-10-05.**  All four landed, each with a shader, a gated case, a
   registered falsification and docs, and the full gate green on both boxes:
   * **the split sampler** (`sampler_split.comp`, the engine's DEFAULT sampled path: 4096-logit partitions, each
     its own top_k, merged) - `2539912`;
   * **the coupled/draft-staging path** (`coupled_penalize.comp` + `coupled_sample.comp`, speculative decoding;
     the counter is `cell + 1`) - `334ca16`;
   * **`sample_tokens`'s CHOICE** (`case_sample_tokens`: greedy || temp 0 -> the argmax, else the split, with the
     one-block fallback) - `8fa6816`;
   * **the portable f32 sibling** (`sampler_kernel_f32.comp`, the sampler's variant for the devices WITHOUT
     shaderFloat64 that Intel's own article describes) - `06ae153`.
   The tail and the partition+merge selection now live ONCE in `common/sampler_tail.glsl` /
   `common/sampler_select.glsl`, so the three sampled paths cannot drift.  `PORT-MAP.tsv` still reads 77 symbols -
  28 kernel, 49 host, **0 todo**, and regenerates identically.  `NEXT.md`'s top four sections carry the rule,
  the traps, the evidence and every falsification.  The sampler item is DONE.  (**CORRECTED 2026-10-05,
  `7c317c4`:** that `0 todo` was read on the qualifier-only map; the map actually reads **168 symbols - 53
  kernel, 63 host, 52 todo**, and **M-A is NOT closed** - 19 of the 52 are class-A forward-path holes with no
  ported fallback, triaged per symbol in `plan/DECODE-PATH-TRIAGE.md`.  The ten holes §4 lists were landed, but
  they were not the whole hole; §4 above is corrected.  What remains on this list is the backend integration of
  §4.2 plus the class-A 19, two different efforts.)
4. **The ten kernels** of section 4, in whatever order the first-token path wants them.

---

## 7. WHERE THE DOCS LIVE

* `ports/vulkan/NEXT.md` - the resume point at the top, then one section per landed increment (each states the rule
  ported, the traps, the evidence, and what the falsification found).
* `ports/vulkan/STATUS.md` - the gate's totals per implementation and the notable entries.
* `ports/vulkan/PORT-MAP.tsv` + `ports/vulkan/tools/{make,check}_port_map.py` - the checked classification of the
  decode path; the gate fails if it drifts from the engine or from the built shaders.
* `ports/vulkan/RUN-ON-B70.md` - the plan from "kernels" to "a running engine" (retitle it for the XTX).
* `ports/vulkan/plan/PORT-PLAN.md` - the six stages of the port itself.
* The skill **`vulkan-compute-shader-porting`** - the accumulated lessons (contraction and `precise`, the
  decorative-arm trap, the fixture-margin rules, glslang's `ull` and subscript quirks, pinning an RNG through an
  observable decision, and the "improvements a careful port would make" that need arms).
