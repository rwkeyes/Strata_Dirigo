# Start here next session

## THE REAL PACK REACHES THE NATIVE-EXPERT GATE: the CMake ordering defect that refused every IQ pack is fixed, the port owns the native (IQ) expert GEOMETRY, and the stopping point is now a NAMED missing kernel (2026-10-05, `vega`)

**THE PACK LOADS AND THE LAYOUT IS VALIDATED.  `coder-iq1_m` no longer dies at "built without
STRATA_NATIVE_EXPERTS"; it now stops at `generate.cpp:2007` with an exact, per-layer name.**  Raw output, both runs
identical, `RC=1` (the full raw output is ONE line - nothing was allocated, nothing was dispatched):

```
strata generate: layer 0's experts are IQ3_XXS/IQ4_NL (ggml types 18/20), which this engine has no GPU kernels for
```

Exact command (run 1 and run 2, `RC=1` both, `/tmp/probe4.log` and `/tmp/run2.log`):

```
STRATA_VK_SPV_DIR=/home/bob/strata-vulkan-wt/ports/vulkan/shaders \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json STRATA_VK_ARENA_GIB=8 \
  ~/vkbuild-vulkan/vulkan/strata_vulkan \
  --pack /media/bob/.../strata-packs/coder-iq1_m \
  --native ~/strata-models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --spec 4 --tokens 1 --max-new 1 --max-context 8
```

**`--no-ple` IS IMPOSSIBLE WITH `--native`, MEASURED.**  `generate.cpp:1789` requires the PLE key for `--native`
(the key is native too), so `--native` + `--no-ple` exits 2 at option validation
("--native requires --ple-gguf ...").  `--no-ple` and `--ple-gguf` are mutually exclusive (`:1830`).  The runs
above therefore carry PLE ON; the PLE table (shard 2) is opened file-backed and NEVER held in VRAM, so it is not
part of the fit.  **A real pack's decode REQUIRE the P6 verify window** (`generate.cpp:2165`: `--native`,
`--spec >= 2`, `--prefill` or a 1-token prompt), i.e. `Verifier::run` - not the token-graph path.

### DELIVERABLE A - THE FIT, MEASURED ON THE B70

The model needs **dense 1.374 GiB + experts 23.419 GiB = 24.793 GiB** (sum over the 48 layers of
`n_expert 256 x blob_bytes` from `native_experts.txt`; the header's own `total 25146163200` agrees).  A single
device-local ARENA of that order was allocated on the Arc **through the port's own device layer** (the shim's one
`VkDeviceMemory`, `strata_vulkan` + `STRATA_VK_ARENA_GIB=N`):

| `STRATA_VK_ARENA_GIB` | arena line printed | driver free / usable |
|---|---|---|
| 24 / 26 / 27 | **YES** (`one arena of N GiB`) | 28.30-28.43 GiB free, **27.3-27.4 GiB usable** |
| **28 / 29** | **NO** (exit 3, no arena line) | same free figure |

**So the weights FIT: 24.793 GiB against 27.3-27.4 GiB of usable device heap, with ~2.2-2.6 GiB left over.**  The
difference is consumed by the dense arena's 1079 tensors' 256-byte-aligned footprint + metadata, the KV/QSA/GDN
session state, the MoE/PLE/GR workspaces, the PCIe staging ring and the descriptor pools - NOT by the experts,
which are not staged.  **The resident arm is viable; the missing piece is the kernel, not the space.**

### DELIVERABLE B - THE ENGINE'S OWN EXPERT ADDRESSING, IN THE PORT

* **`vulkan/src/kernels/native_expert_vk.cpp` (new)** supplies the six native-expert symbols a Vulkan build needs
  and the ggml half does not provide.  **`native_fmt` is REAL geometry**, not a stub: `gu_row`/`d_row` come from
  the port's ONE row table (`iq_row_bytes`), `up_off = gu_row*n_ff`, `down_off = 2*up_off`,
  `bytes = down_off + d_row*n_embd`, and `gu_act`/`d_act` are ggml's `vec_dot_type` map (i-quants -> Q8_K,
  IQ4_NL/Q2_0 -> Q8_0), transcribed from `ggml-cpu.c`'s traits table in this tree.  **The engine's own load check
  measures it**: `expert_layout_load` (`expert_layout.cpp:284-289`) requires `f.bytes == the pack's blob column`
  for EVERY layer, and the run now gets past it - so all 48 layers' layouts match the pack.
* **`iq_row_bytes` gained type 15 (Q8_K, 292 B / 256 values)** - the one table entry the geometry needs
  (`block_q8_K`: `float d` + `int8 qs[256]` + `int16 bsums[16]`, confirmed by ggml-common.h's own static_assert).
  It was a loud refusal before ("no row layout for ggml type 15"), which is how the gap was found.
* **`native_expert_layout` and `native_expert_scratch_bytes` are now their REAL host rows** (transcribed from
  `iq_kernels.cu:1869` / `:1879`), replacing refusals.  The MAP moves them `refused -> host`.
* **The four CPU-hybrid rows refuse LOUDLY** (`native_quant_act/_h`, `native_gu_rows`, `native_down_rows`): they
  are ggml-cpu's vec_dot rows, i.e. a CPU-hybrid execution path this port forbids; a silent zero there is a wrong
  token.
* **THE STAGING SEAM, NAMED.**  The engine gets expert bytes to the device through its PCIe fetch
  (`GpuPlanSink::fetch` -> `cudaMemcpyAsync` on a device staging slot, `expert_source.cpp:2083-2102`), which the
  shim already stages.  What is missing is the kernel that READS the staged blob: **`native_expert_grouped`**,
  still a loud refusal, because the reference passes `grp_ptr` as an array of DEVICE POINTERS (a Vulkan shader
  cannot dereference one) while the port's `native_gu_iq2s.comp` takes a per-group BYTE OFFSET.
* **THE GROUPED SHADER INVENTORY IS 2 OF 7 PAIRS.**  The port ships `native_gu_iq2s.spv` (IQ2_S gate/up) and
  `native_down_iq4nl.spv` (IQ4_NL down).  The Coder pack uses GU in {IQ3_XXS(18), IQ3_S(21), IQ2_S(22),
  IQ4_XS(23)} and DOWN in {IQ4_NL(20), Q2_0(42)}.  **Layer 0's gate/up is IQ3_XXS: no shader.**

### DELIVERABLE C - THE STOPPING POINT, AND THE ONE BEHIND IT (a DIAGNOSTIC, reverted)

`native_expert_supported` was a blanket `return false` with no code behind it.  It now answers from TWO NAMED
COMPONENTS - the grouped shader inventory above and whether the launcher is wired - and the honest composed answer
is **FALSE for every pair** while `native_expert_grouped` is a refusal.  The engine asks it BEFORE allocating
anything, so the pack stops on **layer 0**.  Pinned by `case_native_expert_capability_entry` (two-sided: inventory
`1 true / 6 false`, composed answer `0 of 7`, the wrong-stride rival moves `7/7`).

**A DIAGNOSTIC RUN (the capability temporarily forced `true`, then REVERTED byte-identically) names the NEXT
blocker - it is NOT the experts:** the run inits the device, prints
`native pack: ... experts (largest blob 2.66 MB), token embedding IQ4_XS in mapped host memory (322 MiB)`, then
exits 1 at **`blk.0.attn_gate.weight: this pack holds the tensor only in its GGUF form (run with --native SHARD1)`**
- i.e. the **native DENSE coverage** of the non-expert projections, before a single expert is read.  Two more
firsts: the PCIe probe reads **0.1 GB/s host->device -> pcie_frac 0.00** (worth checking against the shim's
staging copies, since 0.00 disables the GPU's PCIe expert share), and the run needs `--expert-profile` +
`--expert-cache` before `Verifier::init` will even build its windows (`verify.cpp:324`).

### RESULTS (vega)

Gate (`run_gate.sh`, background): **Arc `intel_icd` 752 passed / 0 failed / 0 skipped** (exit 0; was 750 - the two
new cases), llvmpipe **740/0/3**, Ryzen iGPU **739/4/2** - the four are the DOCUMENTED platform-level
non-deterministic wrong-value defects (`budget: independent requery`, `fused_gdn_ab entry`,
`bf16_gemv_fp32_mmvf_cols entry`, `bf16_gemv_fp32_mmvf_multi entry`), NOT this batch's cases, and the new case
passes on all three arms.  Map: **`168 = 78 kernel + 0 shader + 45 host + 0 todo + 45 refused`** (`refused` is NOT
a capability); `check_port_map.py` passes and `make_port_map.py` regenerates byte-identically.  Engine bar: the
`strata_vulkan` program LINKS, 0 undefined - **0 BY CONSTRUCTION (the refusals define the unported symbols), not a
porting gain.**  **`z820b` untouched (no XTX/K620 number claimed).**



## THE FIRST TOKEN from the Intel Arc Pro B70: the DOORBELL RING is a recorded device op, `doorbell_wait` is a no-op under capture, and the STREAM SEAM (a NULL handle is CUDA's default stream) is fixed once (2026-10-05, `vega`)

**A TOKEN CAME OUT.  `strata_vulkan` --pack <synthetic ZERO-weight pack> ran the whole 48-layer decode ON THE
INTEL ARC PRO B70 and the sampler emitted token id `0`, exit 0, 5.11 tok/s.**  Raw tail (run 1, exact command in
`RUN-ON-B70.md` and below):

```
strata generate: token graph captured (48 layers, one launch per token)
strata generate: position 0, token 1 (prompt)
prompt  : 1
output  : 0
decode                   1 tokens in 195.6 ms  ->  5.11 tok/s
RUN_RC=0
```

**THE WEIGHTS ARE ZERO, SO THE TOKEN IS CONTENT-FREE** - zero in gives a finite zero logits vector and the
greedy argmax is index 0.  **What the token certifies is the PIPELINE** (open -> load -> capture -> one launch
-> sample), not a working model: nothing about layer numerics is certified by it (the 750 gate cases are), PLE
is off, the CPU pool is unused, and prefill is bypassed.

### DELIVERABLE A - the DOORBELL RING is a RECORDED DEVICE operation

`src/core/layer.cpp:383-389` states the requirement: the sequence value is read from the HOST copy and
incremented, "which is what makes the write idempotent across replays of the same graph - a captured literal
would ring the same number forever and the host would never see a change."  The port's ring was a capture-time
HOST read-modify-write, so a captured block replayed a stale literal and `session_run_token` would have spun on
`h_seq` forever.  **Fixed**: `ports/vulkan/shaders/ring_inc.comp` (one lane, `ring.seq = store ? value :
ring.seq + 1`), DISPATCHED through `Ctx::dispatch` in `vulkan/src/kernels/doorbell_vk.cpp::ring_raise` - which
RECORDS while a capture is active and submits-and-waits otherwise.  One definition; the whole `doorbell_*`
family (`publish`, `_res`, `_value`, `ring`) flows through it, and the payload copies were already recorded
dispatches (`sync_copy_fenced` -> `Ctx::dispatch`).  The map RE-KINDS the family `host -> kernel`
(`doorbell_wait` stays `host`: it submits nothing).

### DELIVERABLE B - `doorbell_wait` is a NO-OP under capture

`if (s->ctx->capturing()) return;` sits BEFORE any read, so a capture records no node, touches no buffer and
refuses nothing - the host answers later, per the engine's contract.  The LOUD REFUSAL (flag < ring, exit 2)
is UNCHANGED for the genuinely-unanswered non-capture case, and `case_doorbell_ring_replay`'s child proves it
still exits 2 with the message.

### DELIVERABLE C - PROOF + FALSIFICATION (all on Arc `intel_icd`)

`case_doorbell_ring_replay` (7 verdicts) proves the ring ADVANCES ON EVERY REPLAY: direct 1 then 2; a capture of
3 `doorbell_ring` calls leaves the ring at 0 and holds 3 nodes; three replays read 3, 6, 9; a captured
`[doorbell_wait, copy_from_mapped]` block reads the host's NEW answer on each replay; and `doorbell_wait` alone
records nothing (the capture refuses).  **Four injections BITE** (`gates/inject-verify.sh`, all `FALSIFIED`):
`doorbell-ring-captured-literal` (host raise at capture -> "capture records, does not run" FAILS),
`doorbell-ring-shader-literal` (store instead of increment -> "direct increments" FAILS),
`doorbell-wait-records-node` (a device node where the wait must submit nothing -> "records NOTHING" FAILS).

### THE NEW STOPPING POINT THE RING FIX REVEALED - the STREAM SEAM

With the ring fixed, the run captured the token graph and stopped at `embedding_gather: the stream handle is not
a live Vulkan stream`.  **Root cause is the seam, not the symbol**: `src/program/generate.cpp:3893` is
`void* token_stream = o.stream_token ? main_cs : nullptr`, so the DEFAULT path reaches `embed_row ->
embedding_gather` with a NULL stream, and CUDA resolves NULL to the legacy DEFAULT stream.  Every one of the 13
kernel TUs refused it.  **Fixed ONCE at resolution**: `vk_arena.cpp::stream_of(nullptr)` now returns the device
layer's DEFAULT STREAM, which is the compat shim's `g_current` (a REFERENCE to the same object, so a null handle
and the shim's current stream cannot disagree).  `case_null_stream_default` proves a null handle runs and equals
the explicit handle BITWISE (child), and that a BOGUS non-null handle is STILL refused (child exit 2); the
injection `stream-null-not-default` makes it FAIL.

### DELIVERABLE D - THE RUN, AND WHAT IT FOUND

**Run 1 (synthetic ZERO pack) -> TOKEN 0, exit 0** (above).  **Run 2 (the ORIGINAL random pack)**: the ring fix
and the seam fix carried it all the way to `248320 of 248320 logits are not finite at position 0` - the decode
EXECUTED (48-layer token graph captured, logits computed) and the engine's own isfinite scan refused.  **The
zeros run discriminates the cause**: a zero pack gives a FINITE zero logits vector, so **the composed 48-layer
chain is arithmetically SOUND - the non-finiteness is the FIXTURE, not a port defect.**  A `--weight-scale 0.02`
pack (measured: its quant scales are 50x smaller, mean 2.97e-4 vs 1.49e-2, so the scale DID reach the quant
planes) STILL diverges - random weights of any nonzero magnitude overflow this 48-layer architecture (48 layers
of residual accumulation with hc mixing and a delta-rule recurrence).  **THIS IS AN UNRESOLVED FIXTURE LIMIT,
reported as such**: the synth pack cannot produce finite NONZERO logits, and no synthetic run certifies layer
numerics either way.

### NEXT ITEM - THE REAL PACK NEEDS THE EXPERT PATH

`/media/bob/.../public/strata-gguf/strata-packs/` holds four PRODUCTION packs; `coder-iq1_m` (dense 1407 MB,
`native_experts.txt`: n_expert 256, 25.1 GB) has **NO experts.bin** - its experts are read from the original
GGUF shards at runtime, a path this port has NOT exercised (`--mmap-experts` reads a pack `experts.bin`).  25.1
GB + 1.4 GB against 27.6 GiB usable MAY fit resident, but **whether it fits is a measurement for the next
batch, not a conclusion.**  That is the road to a token from REAL weights.

### RESULTS (vega)

Gate (`intel_icd`): **750 passed, 0 failed, 0 skipped** (exit 0; was 741 - the 9 doorbell arms stay green, +7
ring +2 stream).  Map: `168 = 78 kernel + 0 shader + 43 host + 0 todo + 47 refused` (the four doorbell device
ops moved `host -> kernel`; `refused` is NOT a capability); `check_port_map.py` passes and `make_port_map.py`
regenerates byte-identically.  Engine bar: 0 remaining engine-API undefined symbols (0 `strata::kernels::`, 0
`cuda*`); the 210 left are libc/libstdc++ - **0 BY CONSTRUCTION because the refusals define the unported
symbols, not a porting gain.**  `z820b` untouched (no XTX/K620 number claimed).

## `copy_from_mapped` WIRED (THE LAST REACHED SYMBOL) + THE PUBLISH HANDSHAKE; THE PROGRAM LINKS (0 undefined); THE MAP GAINS A `refused` KIND (2026-10-05, `vega`)

**THE ENGINE BAR MOVED `134` raw / `55` distinct / `41` `strata::kernels::` / `0` cuda → `0` / `0` / `0` / `0`.**
Measured with the exact whole-archive recipe (four engine libs - `strata_vulkan_engine/core/kernels/kernels_cpu` -
vs the backend `cudart`/`device`; `-lvulkan -lpthread`).  **THE `strata` PROGRAM NOW LINKS** (`make strata_vulkan`
→ exit 0, `strata_vulkan` 2.5 MB).  The drop is the refusals (below) + `pinned.cu`; it is NOT a claim that the
engine RUNS (see DELIVERABLE C).

**THE MAP now reads `168 = 74 kernel + 0 shader + 47 host + 0 todo + 47 refused`** (was `75/5/61/27`).  Two moves:
(1) **`copy_from_mapped` is `host → kernel`** (shader `copy`): its CUDA body is a DEVICE kernel, the port DEFINES
it as a shader dispatch, and `case_copy_from_mapped_entry` proves it.  (2) **A FIFTH KIND, `refused`** - every
symbol the backend defines ONLY as a loud refusal.  `check_port_map.py` enforces it (a `refused` row whose symbol
the backend does NOT define FAILS, like a `kernel` row with no definition), and `make_port_map.py` carries the
`REFUSED` set.  **NEVER read `refused` as a capability - it works for nothing.**  The five former `shader` rows
(`coupled_draft_sample`, `moe_grouped_s2`, `moe_hit_grouped_s2_cpu_order`, `native_expert_grouped`,
`shared_expert_multi`) had to be re-kinded because the checker refuses a defined `shader` row; two `kernel` rows
(`embedding_gather_dev`, `native_flash_attn_short_step`) were ALREADY lies of this class (kinded `kernel` with no
working definition - the `embedding_gather_dev` "engine-header inline" exception was false: the linker named it).

## DELIVERABLE A - `copy_from_mapped` and the PUBLISH HANDSHAKE it needs

`vulkan/src/kernels/elementwise_vk.cpp` defines `strata::kernels::copy_from_mapped` (`session.cpp:875`, the
captured per-layer parts copy, the #1 reached symbol).  **A VULKAN SHADER CANNOT DEREFERENCE MAPPED HOST MEMORY,
so this is not a wrapper - it is the handshake `sync.hpp` kept the seam for.**  The shim's `cudaHostAlloc` maps a
HOST_VISIBLE|HOST_COHERENT device block, so the region's `host` pointer and its device view are ONE allocation.
The port registers each region (`mapped_register`, in `vk_arena.*`, called from `cudaHostAlloc`) and
`copy_from_mapped` binds **the region's device-visible buffer** as the shader's source (`copy.spv`), NOT the host
address.  A host pointer the shim did not hand out, or one whose published size does not cover `n` floats, is
REFUSED (proved by a child process).

**WHERE THE PUBLISH SITS, and what the wrong side is.**  The host's store into the mapping IS the publish; its
edge to the device is the NEXT SUBMISSION.  A recorded step re-reads its BUFFERS at submit time (the `graph.hpp`
property), so the store must sit **AFTER `capture_end` and BEFORE each `cudaGraphLaunch`** - never at capture.
Wrong side = a store made only at capture: the recorded node reads the mapping LIVE, so a replay would copy the
CAPTURE-TIME bytes (a stale block).  The trap is not hypothetical: the OLD `copy_i32_from_mapped` did a fenced
HOST-STAGED `stream_write` unconditionally, which under capture executes ONCE and records NOTHING - the exact
"EVERY TOKEN AFTER THE FIRST REPLAYED POSITION 0" the engine's own `session.cpp` comment describes.  **FIXED
while wiring: under capture `copy_i32_from_mapped` now RECORDS a `vkCmdCopyBuffer`** (`Ctx::capture_copy`)
re-reading the mapped region; outside capture it keeps the fenced host-staged copy.  Both are BYTE copies, so the
int32 payload is bit-exact either way (copy.spv is float-typed and an int bit-pattern must not pass a float
load/store).

**PROVED BY `case_copy_from_mapped_entry`** (6 verdicts, all green on Arc `intel_icd`, llvmpipe and the Ryzen
iGPU): direct == the engine's own rule bitwise; the wrapper == the port's own shader path bitwise (a SEPARATE
rival buffer, so a wrapper that wrote nothing cannot pass); **a recorded graph replayed after a re-publish sees
the NEW bytes**; replay twice with no stale block; the live publish (not the capture-time block) - the arm that
BITES if the publish had been frozen at capture; and a non-mapped source REFUSED (child exit 2 + the message).

## DELIVERABLE B - THE LOUD REFUSALS, so the PROGRAM links

Three refusal TUs, all with the established discipline (a real definition whose only behaviour is to abort,
naming the flag chain): `refusals_vk.cpp` gains the **~40 kernels-namespace holes** the program link wanted
(the drafter class C, the P6-verifier class D, the `--expert-cache-cpu-order` A/B arm, the `kernels_cpu`
`RouterLookahead` prefetch); **`refusals_engine_vk.cpp`** (new) answers `strata::core::RemoteExpertOpt` (~11
methods, the remote/peer tier, `--expert-cache-remote`/`--peer-device`) and defines `device_code_error` as its
REAL answer (`""` - a Vulkan device has no CUDA arch to mismatch); **`refusals_prefill_vk.cpp`** (new) answers the
prompt-path kernels (`src/prefill/*.cu`) after `src/prefill/prefill.cpp` was WIRED into the engine target (it
compiles against the shim once `cudaMemcpyPeerAsync` is refused there).  Constructors/destructors that the
request path runs UNCONDITIONALLY (`Prefill`, `Gemm`, `RemoteExpertOpt`) are REAL EMPTY bodies, not refusals -
refusing them would abort a decode that never used the object.  **`pinned.cu` is WIRED, not refused**: 0 kernels,
0 launches, host code that compiles as C++ against the shim - it is the MODEL-LOAD path (`load_experts_*`,
`PinnedArena`, `fnv1a64`).  Every refused map symbol has its triage class + flag chain in
`plan/DECODE-PATH-TRIAGE.md` and its chain IN the `refuse_unreachable` call.

## DELIVERABLE C - THE RUN: how far it got

`strata_vulkan` LINKS (0 undefined).  RUN: **program start reached** - `--help` prints the CLI, arg parsing works,
and the engine then stops at **MODEL LOAD**: it needs a PACK directory (`--pack DIR`, `WeightTable::load` opens
`<dir>/index.txt`), and **this box holds only raw IQ1_M GGUF shards** (`~/strata-models/IQ1_M/…-of-00002.gguf`,
58 GB); there is no `pack/`.  Exact output: `strata generate: cannot open /home/bob/strata-models/IQ1_M/index.txt`
(exit 1).  **NO TOKEN.**  Packing was NOT attempted: `tools/iq_pack.py` needs ~the model's size in free space and
`/` has **26 GB free on a 58 GB model** (and "no artifact bakes" is a batch constraint).  So the stopping point is
**program start → model open/load (no pack)**, and the tokens `1,2,3`, the sampler, capture and every decode step
were NOT exercised.

## DELIVERABLE D - the instrument fix at the cause

`run_gate.sh` AND `inject-verify.sh` each compiled the gate from a HAND-KEPT TU list, which went stale three
batches running (most recently `iq_vk.cpp`/`moe_vk.cpp`/`sampler_vk.cpp`): an injection into a TU not in the list
rebuilt a gate that could not link, ran the OLD binary, and reported "NOT FALSIFIED".  **Both now source
`gates/harness_sources.sh`**, which GLOBS `vulkan/src/**/*.cpp` and FAILS LOUDLY if the count on disk differs from
the count in the list it hands the compiler.  **The guard is PROVEN to fire**: a `#error` TU dropped into a NEW
`vulkan/src/guardprobe/` subdirectory (a blind spot of the old literal list) made `build_harness` fail with that
`#error`, i.e. the new TU was COMPILED.

## RESULTS (vega)

Gate (`run_gate.sh`, background): Arc (`intel_icd`) **741/0/0** (exit 0); llvmpipe **729/0/3**; Ryzen iGPU
(`radeon_icd`) **729/3/2** - the 3 FAILs are the documented platform-level non-deterministic wrong-value defect
(`budget: independent requery`, `bf16_gemv_fp32_mmvf_cols`, `bf16_gemv_fp32_mmvf_multi`), NOT this batch's cases;
all six `copy_from_mapped entry` verdicts PASS on all three arms.  `check_port_map.py` passes (`168 - 74 kernel,
0 shader, 47 host, 0 todo, 47 refused`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.
**`z820b` is PENDING** (suspended, no WoL - no XTX/K620 number claimed).

## `sample_tokens` WIRED — THE STEP THAT PRODUCES A TOKEN — AND THE ORDERED DECODE-PATH LIST THAT SAYS ONLY TWO OF THE 41 REMAINING kernels-NAMESPACE SYMBOLS ARE ON A SINGLE-TOKEN DECODE (2026-10-05, `vega`)

**THE MAP now reads 168 = 75 kernel + 5 shader + 61 host + 27 todo** (was 74/6/61/27): `sample_tokens` moved
`shader → kernel` and the backend now DEFINES it (`vulkan/src/kernels/sampler_vk.cpp`).  The other five `shader`
rows (`coupled_draft_sample`, `moe_grouped_s2`, `moe_hit_grouped_s2_cpu_order`, `native_expert_grouped`,
`shared_expert_multi`) stay `shader` with the classes the ORDERED LIST gives them — none is on a single-token
decode (below), so none was stubbed.

**THE ENGINE BAR (`138 → **134` raw / `56 → **`55` distinct / `42 → **`41` `strata::kernels::` / CUDA stays `0`**)** —
`sample_tokens` is the drop.  Measured with the exact whole-archive recipe (four engine libs vs the backend
`cudart`/`device`; `-lvulkan -lpthread`).  **THE ONE-LAYER-BODY bar is UNCHANGED (`18` raw / `0` kernels-ns** —
`sample_tokens` is not referenced by `layer.cpp`).

**THE ORDERED DECODE-PATH LIST (the batch's central output) — TWO symbols, in the order the engine reaches
them.**  Traced from the engine's own code (`generate.cpp`'s token loop → `session.cpp`'s captured block →
`layer.cpp`), under the shipped `--native` launch and this backend's capability answers:

1. **`copy_from_mapped`** — `src/core/session.cpp:875`, INSIDE the captured per-layer block, UNCONDITIONAL.
   PORT-MAP kinds it `host`, but `elementwise.cu:226` launches `copy_from_mapped_kernel` — a float4 DEVICE kernel
   reading MAPPED host memory (the same mis-kind as `copy_i32_from_mapped`).  **UNPORTED and the #1 next item** —
   and NOT a wrapper: a Vulkan shader cannot dereference host memory, so it must be answered by the port's
   split-submission handshake (the host re-publishes `parts_dev` before each launch), the seam `sync.hpp` kept for
   exactly this swap.  See `plan/DECODE-PATH-TRIAGE.md`'s new section.
2. **`sample_tokens`** — `generate.cpp:7742`, the decode tail.  **WIRED THIS BATCH.**

**Everything else of the 41 is a NON-SELECTED configuration** (the drafter's `--spec 4 --mtp` loop, which the
contract refuses at `Verifier::init`; the `--expert-cache-cpu-order` A/B arm, default false; the P6 verifier; the
multi-GPU remote/peer expert tiers; model load: `embed_type_supported`), or the `kernels_cpu` half
(`strata::kernels::cpu::bf16_rows_dot_multi{,_avx1}` — the name-only pattern collapses both to `cpu`; the file-tier
`RouterLookahead` prefetch, off on the packed-image launch).  The full ordered table, each symbol's class and the
flag/default chain that selects it, is in `plan/DECODE-PATH-TRIAGE.md`.

## DELIVERABLE A — `sample_tokens`, the step that PRODUCES A TOKEN

`vulkan/src/kernels/sampler_vk.cpp` defines `strata::kernels::sample_tokens` over the port's ALREADY-GATED shaders,
following the engine's own path selection (`sampler.cu:1006-1011`): `greedy || temperature <= 0` → the argmax
(`sampler_greedy.spv`); otherwise the SPLIT (`sampler_split.spv`) — the port's split is ONE workgroup per row with
the merge in-shader, so it needs NO scratch and NO `cudaMalloc`, and it RECORDS under capture (unlike the CUDA,
which falls back under capture because ITS split needs a `cudaMalloc`); the f32 one-block sibling
(`sampler_kernel_f32.spv`) on a device with no `shaderFloat64`.  Engine headers unchanged.

**PROVED BY `case_sample_tokens_entry`** (6 verdicts: three arms × two facts, plus the capture arm), all green on
the Arc (`intel_icd`), llvmpipe and the Ryzen iGPU:

| arm | wrapper == shader path (bitwise) | wrapper == the engine's own rule |
|---|---|---|
| sampled / split (t=1), 12288 vocab, top_k 4, ids 100/4200/8300 in 3 partitions | **1/1, w 0** | **1/1, w 0** (the transcription's pick) |
| greedy FLAG | **1/1, w 0** | **1/1, w 0** (`sampler_want`) |
| temperature 0 WITHOUT the flag | — (must equal the argmax) | **1/1, w 0** — `sampler.cu:1006` routes temp 0 to the argmax, NOT the sampled kernel's uniform draw |
| **CAPTURE ARM** | **4/4, w 0** — records a block containing the wrapper, replays, bitwise equal | |

**THE OTHER FIVE, each classified from the engine's own code (why not finished):** `coupled_draft_sample` and
`moe_grouped_s2` are CLASS C (the `mtp.cpp` drafter, a configuration the port does not select — `Verifier::init`
refuses under the contract); `moe_hit_grouped_s2_cpu_order` is CLASS C (the `--expert-cache-cpu-order` A/B arm,
**default false** → the PORTED `moe_hit_grouped_s2`); `native_expert_grouped` is CLASS D/remote (the P6 verifier +
the `--expert-cache-remote`/`--peer-device` tiers); `shared_expert_multi` is CLASS D (`verify.cpp:980` only).  A
definition for any of them would be a wiring for a branch nothing selects — reported, not stubbed.

## RESULTS (vega)

Gate (`run_gate.sh`, background): Arc (`intel_icd`) **735/0/0** (exit 0); llvmpipe **723/0/3** (the documented
coopmat/vram skips); Ryzen iGPU (`radeon_icd`) **724/2/2**, then **723/3/2**, then **722/4/2** across three runs —
the ramp is the open, characterised platform-level non-deterministic wrong-value defect, and the failing CASES
differ each run, which is the point: `qsa_block_scores entry` (71/72), `bf16_gemv_fp32_mmvf_cols entry`
(2494-2495/2496), `bf16_gemv entry` (510/512 + its oracle arm), `bf16_gemv_fp32_mmvf entry` (255/256),
`bf16_gemv_fp32_mmvf_multi entry` (620/624), `ple_block entry` (10239/10240) — the same class the README records
for `bf16_gemv`, `bf16_gemv_split`, `fused_gdn_ab`; this batch adds `qsa_block_scores`, `bf16_gemv_fp32_mmvf*` and
`ple_block` to that record.  **NONE is one of this batch's cases.**  All six `sample_tokens entry` verdicts PASS on
all three arms.  `check_port_map.py` passes (`168 — 75 kernel, 5 shader, 61
host, 27 todo`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (one row: `sample_tokens`).
**`z820b` is PENDING** (suspended, no WoL — no XTX/K620 number claimed).  The CUDA graph API was NOT touched.
**Also fixed:** `gates/inject-verify.sh`'s `rebuild_harness` had gone stale AGAIN (missing `iq_vk.cpp`,
`moe_vk.cpp`, `sampler_vk.cpp` — an engine-side injection would have rebuilt a gate that cannot link and reported
"NOT FALSIFIED"); it now carries the full TU list, and the new `sample-tokens-entry-temp0-to-sampled` injection
BITES (`FALSIFIED ... FAIL sample_tokens entry (temperature 0) 1/2`).

## THE 14 `shader` ROWS: EIGHT WIRED (+ their capture arms) AND THE CUDA SURFACE THAT LETS generate.cpp COMPILE (2026-10-05, `vega`)

**THE MAP now reads 168 = 74 kernel + 6 shader + 61 host + 27 todo** (was 66/14/61/27): eight of the fourteen
`shader` rows are now `kernel` — `bf16_gemv_fp32_mmvf_multi`, `iq_dequant_f32`, `iq_embed_rows`, `moe_hit_select`,
`moe_hit_add`, `moe_hit_grouped_s2`, `moe_hit_grouped_s2_dev`, `s_gemv_split_async` — each with an ENGINE-WRAPPER
gate case (wrapper == the port's own shader path bitwise, AND the engine's own rule, AND a CAPTURE arm).  The
remaining six `shader` rows did NOT get a definition THIS batch (a missing one is worse than none; see the report):
`coupled_draft_sample`, `moe_grouped_s2`, `moe_hit_grouped_s2_cpu_order`, `native_expert_grouped`,
`sample_tokens`, `shared_expert_multi`.  New backend TUs `vulkan/src/kernels/{iq_vk,moe_vk}.cpp`;
`bf16_gemv_fp32_mmvf_multi` + `s_gemv_split_async` landed in `matvec_vk.cpp`.

**THE ENGINE BAR (`154` → **`138` raw / `64` → `56` distinct / `50` → `42` kernels-ns / CUDA stays `0`**)** — the
eight symbols are the drop.  The ONE-LAYER-BODY bar is UNCHANGED (`18` raw / `0` kernels-ns).  `generate.cpp`
now COMPILES against the shim (was 34 errors).

**THE CUDA SURFACE (deliverable C)** — `vulkan/include/cuda_compat/cuda_runtime.h` + `.../compat/cuda_runtime.cpp`
add `cudaDeviceProp`/`cudaGetDeviceProperties`/`cudaDeviceGetAttribute`/`cudaDevAttr*`, `cudaMallocHost`, a typed
`cudaMalloc` template, `cudaRuntimeGetVersion`+`CUDART_VERSION`, and default `stream = nullptr` on
`cudaMemcpyAsync`/`cudaMemsetAsync`/`cudaEventRecord`.  **THE DEVICE-PROPERTY ANSWER, stated in the header:** a
Vulkan device has no SM count and no clock rate, so `cudaDevAttrMultiProcessorCount` and `cudaDevAttrClockRate`
are **REFUSED** (`cudaErrorInvalidValue`) rather than fabricated; `prop.name`/`totalGlobalMem` are the REAL
deviceName / DEVICE_LOCAL heap; `major`/`minor` are 0 (no CUDA compute capability).  The only consumer of the two
refused attributes is the multi-GPU `--layer-split` AUTO heuristic (`generate.cpp:2759-2766`), unreachable on this
one-device backend (`cudaGetDeviceCount() == 1`), and the engine already has the `std::max(1.0, …)` / 1.8 GHz
fallbacks if it were reached.

**THE CAPTURE DISCIPLINE.**  Every wrapper uses `Ctx::dispatch`, which RECORDS under capture, so all eight record;
`moe_hit_grouped_s2_dev` is the exception — its contract needs the DEVICE's live hit count and the port's
`s2expert_gu`/`s2expert_down` take the count as a PUSH CONSTANT, so it reads the count on the host and REFUSES
LOUDLY under capture (invalidates the recording → `cudaErrorStreamCaptureUnsupported`).  Each of the eight is
proved by a case that records a block containing it, replays, and requires bitwise equality with direct execution;
the `_dev` refusal is its own arm.

## THE CUDA GRAPH API OVER THE PORT'S OWN RECORDED STEP — the recorder COMPILES and REPLAYS (2026-10-05, `vega`)

**THE NEW BAR (the engine executable, not just the layer body): `154` undefined references / `64` distinct =
`51` `strata::kernels::` full-signature + `13` `strata::core::` + others, and `0` CUDA-runtime symbols.**  The
`0` is the point: the shim now answers the ENTIRE CUDA-runtime surface the engine library references, **the graph
API included** (`grep -oP "undefined reference to \`\K[^']+" ... | grep -c '^cuda'` → 0).  Measured by
whole-archiving the four engine libraries against the backend (`strata_vk_engine/core/kernels/kernels_cpu` vs
`cudart/device`); the four recorder TUs `src/core/{graph,session,mtp,verify}.cpp` now **COMPILE** against the shim
(`g++ -std=c++20 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device -DSTRATA_ENABLE_VULKAN=1 -c`
→ exit 0 each).  The one-layer-body link is UNCHANGED (`18` raw / `0` kernels-namespace).  The remaining 51
kernels-namespace symbols are the port's unported forward-path work (the MoE grouping, the drafter, the
sampler multi-forms, …), NOT the graph API.

## DELIVERABLE A — how the engine USES capture (read from its own code), and what "replay-only" would break

**The recorder is `src/core/session.cpp`** (the same shape in `mtp.cpp` / `verify.cpp`); the `GraphRegistry` in
`include/strata/core/graph.hpp` is **referenced by NO engine host code** (only `graph.cpp` and the `sycl/` copy
name it).  Per layer, in order (`session.cpp:221-258`):

    cudaStreamCreate(&cs) -> cudaStreamBeginCapture(cs, ThreadLocal) -> block_layer_pre/post(..., (void*) cs)
      -> cudaStreamEndCapture(cs, &graph) -> cudaStreamDestroy(cs) -> cudaGraphInstantiate(out, graph, 0)
      -> cudaGraphDestroy(graph)
    // per token, per layer (session.cpp:282-289):
    stage_token(...) ; cudaGraphLaunch(gr.execs[l], cs)

`mtp.cpp` adds `cudaGraphUpload(exec, cs)` + `cudaStreamSynchronize(cs)`; `verify.cpp` adds the diagnostic
`cudaGraphGetNodes`/`cudaGraphNodeGetType`/`cudaGraphKernelNodeGetParams` (under `STRATA_VERIFY_NODES`); the
destructors call `cudaGraphExecDestroy`.

**It is RECORDING, not graph-level semantics.**  No hand-added nodes, no cross-stream capture, no event/wait
nodes inside a capture (events are recorded only BETWEEN two launches — `session.hpp`), no
`cudaGraphExecUpdate`, no memory-node aliasing.  The graph-level facts it DOES rely on are `graph.hpp`'s two
measurements: **(1) a replay re-reads its input BUFFERS but not its kernel ARGUMENTS** ("a kernel argument is
copied into the node at capture and is never re-read … a position, a page-table base, a token id must be DATA in
a device buffer"), and **(2) fixed addresses are the caller's obligation.**  A block is ~43 nodes
(`session.hpp`: "the 43-node block graph measured 1.585 ms against 2.393 ms for direct launches"), two graphs
(`pre`/`post`) per layer.  Between replays `stage_token` (`session.cpp:177`) rewrites the PINNED staging buffers
`host_step`/`host_pos`; the body reads them through the `copy_i32_from_mapped` KERNEL into `st.step`/`st.pos_dev`
(`layer.cpp:911-914`, `g_publish_kernel` default **true**) — no pointer changes.

**What a naive replay-only implementation breaks.**  (a) A capture must RECORD, not RUN: this backend's
`dispatch` submits AND waits, so letting it execute would advance every GDN recurrent state and the residual
once before the first token.  (b) A HOST↔DEVICE copy or a memset inside a capture cannot be recorded (this
shim stages both through host memory); CUDA records a memcpy node, the port must REFUSE.  Both are handled and
stated in the shim header.

## DELIVERABLE B — the mapping: a graph IS the port's recorded step (no second mechanism)

`ports/vulkan/plan/CUDA-GRAPH-MAPPING.md` (new) has the full table.  In one line: **a `cudaGraph_t`/`cudaGraphExec_t`
is one of the port's recorded steps** — the SAME `Ctx::encode_dispatch` (chain barrier between dispatches, a fresh
descriptor set per dispatch) and the SAME `vkQueueSubmit` + fence wait — made `capture_begin` divert `Ctx::dispatch`
to `record_dispatch`, so a capture RECORDS.  `EndCapture` closes the recording (no submit) and the shim owns it;
`Instantiate` takes ownership; `Launch` = `Ctx::submit_owned`; `Destroy` frees.  D2D copies record as
`vkCmdCopyBuffer`; `Upload` is a no-op; `GetNodes`/`NodeGetType` report counts; `KernelNodeGetParams` is
`cudaErrorNotSupported`.  **Preserved:** record-once/replay-many, live-buffer re-read, capture-does-not-run,
bitwise replay==direct.  **NOT preserved (stated in the header):** one queue (no multi-stream), and H2D/D2H
copies + memsets inside a capture are `cudaErrorStreamCaptureUnsupported`.

## DELIVERABLE C — the proof (Arc, `intel_icd`), and the three injections

`case_cuda_graph_entry` (`vk_gate.cpp`, appended) opens the ENGINE stream, records 6 `silu_inplace` dispatches
through the shim, and compares to direct execution.  Raw lines:

    PASS  cuda graph: capture records, does not run  1024/ 1024   worst 0   captured elements moved
    PASS  cuda graph: the graph holds N dispatch nodes     6/    6   worst 6   nodes
    PASS  cuda graph: replay == direct execution (bitwise)  1024/ 1024   worst 0   elements differ
    PASS  cuda graph: host mutations between replays are SEEN  1024/ 1024   worst 0   elements differ
    PASS  cuda graph: replay twice, no cross-replay contamination  1024/ 1024   worst 0   elements differ
    PASS  cuda graph: 6 instantiate/destroy cycles replay correctly     6/    6   worst 0   cycles failed
    PASS  cuda graph: instantiate/destroy leaks no arena bytes     1/    1   worst 0   arena bytes grown
    PASS  cuda graph: no instantiation left live (recordings return to 0)     1/    1   worst 0   recordings still owned
    PASS  cuda graph: a H2D copy inside a capture is REFUSED     1/    1   worst 0   refused loudly
    PASS  cuda graph: a memset inside a capture is REFUSED     1/    1   worst 0   refused loudly
    PASS  cuda graph: a D2D copy records and replays  1024/ 1024   worst 0   elements differ

**The three injections BIT** (`gates/inject-verify.sh graph-…`; the script's `rebuild_harness` was also fixed —
it had gone stale, missing `rope_vk.cpp` + the engine host sources + `-lpthread`):

    FALSIFIED (graph-drop-last-node):     FAIL  cuda graph: replay == direct execution (bitwise)     0/ 1024   elements differ
    FALSIFIED (graph-replay-stale):       FAIL  cuda graph: host mutations between replays are SEEN     0/ 1024   elements differ
    FALSIFIED (graph-exec-destroy-leak):  FAIL  cuda graph: no instantiation left live (recordings return to 0)     0/  1   worst 7

The leak injection is observed by a NEW instrument, `Ctx::owned_recordings()` (++ when a recording is taken, --
when destroyed; the arena never moves for a command-buffer leak).  A NEW latent-defect guard was added with it:
the case asserts the counter returns to 0, so a leaked/double-owned instantiation cannot pass.

## DELIVERABLE D/E/F — results (vega)

`run_gate.sh` (background, exit 0 on the cleaned run): **Arc `intel_icd` 693/0/0**, llvmpipe `680/0/3` (the
documented coopmat/vram skips), radeon iGPU `683/0/2` — **+11 verdicts per arm**, 0 failed; every `cuda graph:`
arm green on the Arc.  **ON THE RECORD:** a later full run's radeon cross-arm carried the documented intermittent
`budget: independent requery agrees` flake AND the open RADV-only wrong-value defect (`bf16_gemv entry n_in=2560
n_out=128`, 510/512) — NEITHER is one of this batch's cases; the Arc arm read 693/0/0 both times.
`check_port_map.py` passes (`168 — 66 kernel, 14 shader, 61 host, 27 todo`); `make_port_map.py` regenerates
`PORT-MAP.tsv` **byte-identically** (no new symbol — the shim's `cuda*` names are not `strata::kernels::`).
`strata_vk_cudart_smoke` and `strata_vk_entry_smoke` build + RUN PASS.  **`z820b` PENDING** (suspended, no WoL).
The full `strata_vulkan` PROGRAM (`generate.cpp`) is NOT yet linkable — it needs a wider CUDA surface the RECORDER
does not (`cudaDeviceProp`/`cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, typed-pointer `cudaMalloc`,
`cudaMallocHost`); that is a separate increment, reported, not the graph API.

## THE LAST FIVE NAMES, THE LINK's kernels PART AT ZERO, and M-B RUNS (2026-10-05, `vega`)

**THE BAR (the running line): `41 → 18` raw / `7 → 0` distinct full-signature `strata::kernels::` symbols /
`5 → 0` under the parent's name-only pattern.**  The five names are WIRED (below).  **The residual 18 raw
references are NOT the five names and NOT kernels-namespace:** they are the engine's own `strata::core::`
cross-TU symbols that `layer.cpp` references - `LayerView::name` (8 refs + 1 "more undefined references" line),
`WeightTable::find` (6), `native_embed`, `NativeEmbed::gather_one` - plus `main`; their homes are the engine's
`layout.cpp` / `weights.cpp` / `native_head.cpp` and an engine executable, none of which is a backend symbol.
Measured with the CURRENT STANDARD recipe (`$HOME/vkbuild-vulkan` is a **Makefiles** build dir, so **never pass
`-G Ninja`**; reconfigured + rebuilt from the current tree first).  **The recipe now also needs `-lpthread`** (the
engine's `ngram.cpp` / `ple_reader.cpp` / `direct_file.cpp` are in the kernels lib - see `PleTable`):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lpthread -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 18   (was 41)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 0    (was 7)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 0    (was 5)

**THE GROUP TABLE (the kernels-namespace symbols remaining: none).**

| subsystem | n | symbols |
|---|---:|---|
| **glue** | **0** | all answered |
| **matvec / GEMV / KV** | **0** | `bf16_gemv_fp32_mmvf_cols` WIRED (was the last) |
| **attention / QSA / MoE / GR / PLE / rope** | **0** | `build_rope_table`, `rope_table_set`, `PleTable::{collect,is_open,issue}` WIRED |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **0** | `copy_i32_from_mapped` WIRED |

## DELIVERABLE A — the five names, each with its proof

| # | symbol | where | proof (gate case, `vega` Arc) |
|---|---|---|---|
| 1 | `bf16_gemv_fp32_mmvf_cols` | `matvec_vk.cpp` | `case_bf16_gemv_fp32_mmvf_cols_entry`: n_in=2560 n_out=48 **ncols=13** (crosses the CUDA's block-of-8) wrapper == shader path **2496/2496 bitwise**, vs the terms-bound oracle 624/624 w 0.0103; the wrong column stride **MOVES** (1.18e+05) |
| 2 | `build_rope_table` | `rope_vk.cpp` (new) | `case_build_rope_table_entry`: table == an **exp/log float64 transcription** (different arithmetization) 512/512 w 2.96e-08; == the port's **DEVICE `native_rope_apply` shader** (an independent implementation) 192/192 abs w 2.26e-05; YaRN arm 512/512 w 5.96e-08; rivals (halved exponent / cos-sin swap) MOVE (1.43) |
| 3 | `rope_table_set` (+`_release`/`_for`/mrope) | `rope_vk.cpp` | `case_rope_table_set_entry`: the default arm (opt-in off) returns NO table even when registered; the enabled arm runs in a **CHILD process** (`--expect-rope-table`, the env is read once) and proves store + scaling-match + release-clears (`ROPE_TABLE_OK`, exit 0) |
| 4 | `copy_i32_from_mapped` | `elementwise_vk.cpp` | `case_copy_i32_from_mapped_entry`: the mapped host image arrives bitwise 216/216, the sentinel past `n` intact, and a one-element source shift MOVES 200/200 |
| 5 | `PleTable::{collect,is_open,issue}` | `src/kernels/ngram.cpp` (engine, LINKED) | `case_ple_table_entry`: `is_open` false on a fresh table, `collect` before `issue` refused (`"without issue"`), `issue` arms, `collect` consumes (zero-filled), a second `collect` refused. The DATA path needs a GGUF pack (`ple_parity`'s job); this proves the state machine `layer.cpp:1294/1300` drives and that the SYMBOL resolves in the Vulkan link |

**`PleTable` IS THE ENGINE'S OWN CODE, not a reimplementation.**  `vulkan/CMakeLists.txt` now resolves
`src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp` and `src/platform/direct_file.cpp` into
`strata_vulkan_kernels`; `ple_vk.cpp`'s transcribed `ngram_rows` was DELETED (the engine's `ngram.cpp` defines
it; keeping both was a duplicate-symbol link error).  `ngram_rows`'s gate case is unchanged and still passes
against the external `ref/ngram.py` oracle.

**A LATENT DEFECT FOUND AND FIXED WHILE WIRING (the "a setting silently ignored" class).**  The port's
`native_rope_apply` (`qsa_vk.cpp`) computes the angle ON DEVICE from `theta_scale`/`rope_scaled_angle` and
IGNORED the table `layer.cpp:698` registers, so under `STRATA_ROPE_TABLE=1` it would have produced the analytic
angle the engine's table path is documented to differ from (~0.0014 rad at 32K) - silently.  It now REFUSES
loudly when `rope_table_for(scaling)` is non-empty (the `native_mmvq_supported` discipline: a backend with no
table-reading shader must own that answer in code).  The DEFAULT (opt-in unset) is bit-for-bit the engine's
`<false>` branch.

## DELIVERABLE B — M-B: ONE LAYER BODY, RANDOM WEIGHTS, ON THE ARC

`vulkan/tests/layer_smoke.cpp` → the new **`strata_vk_layer_smoke`** target (EXCLUDE_FROM_ALL, built by name).
It links `layer.cpp` + the three cross-TU homes (`layout.cpp`/`weights.cpp`/`native_head.cpp`) + the backend,
with `-ffunction-sections -Wl,--gc-sections` so `native_head`'s unwired `iq_embed_rows`/`iq_dequant_f32` deps
are DROPPED (it is a LAYER-BODY link, not the whole engine).  It:

1. opens the engine stream (device + arena) and sets the shim's "current stream" (`cuda_compat_set_stream` - the
   stand-in for CUDA's current device, which the loader's `cudaHostAlloc` needs);
2. writes a **SYNTHETIC RANDOM PACK** (`dense.bin` + `index.txt`) for exactly the nine tensors `gdn_layer`
   resolves and loads it through the engine's OWN `WeightTable::load`.  `WeightTable`'s storage is private and
   its only public constructor needs a pack, so the loader's own format IS the mechanism for "random weights, no
   model"; the loader's segment check refuses a mis-sized plane rather than loading a wrong offset;
3. carves `GdnBuffers` (from `gdn_buffers_bytes`) and zeroes the recurrent + conv state;
4. calls `strata::core::gdn_layer(...)`.

**IT RUNS (Arc, `intel_icd`).**  Geometry `n_embd=256 C=1024 S=128 h_k=2 h_v=4 V=512`; pool 611,328 B; 9
tensors loaded; `gdn_layer` returned 1 (success) with `err=""`.  Evidence:

| check | result |
|---|---|
| shape | out = 256 floats = n_embd |
| finiteness | **256/256 finite** |
| non-degenerate | range `[-131.279, 148.152]`, mean `-2.45868`, variance `3178.08` (> 0) |
| input dependence | a DIFFERENT activation moves **256/256** output elements |
| repeatability | two runs from a re-zeroed state are **256/256 bitwise equal** |

**WHAT IT DOES NOT PROVE: numerical correctness of the layer.**  It executed and produced finite,
non-degenerate, repeatable output of the right shape; there is **NO reference comparison** (no cheap whole-layer
oracle exists here, and a restatement of the implementation would not be one), so agreement with the engine's
CPU/CUDA path is NOT established.  Per-kernel numerical correctness remains the gate's job.

## WHAT THE ATTEMPT MET (the stopping points that were cleared, and the one that remains)

* **Cleared:** link (`--gc-sections` keeps the layer-body closure); the shim's current-stream seam
  (`cuda_compat_set_stream`); the Q8_K block contract (n_embd must be a multiple of 256, because `gdn_layer`
  always builds `x_q8k`); finite fixture values (raw random BYTES make NaN/Inf f32 scales, and a positive
  `ssm_a` overflows the recurrence's `exp` - both are FIXTURE bugs, not kernel ones).
* **Remains:** `WeightTable` has no public insertion API, so "random weights" MUST go through `load()` and a
  pack-format file.  That is the design, not a defect - but it means M-B is not a "no filesystem writes" test.

## RESULTS (vega)

Gate (see the totals line at the top of `STATUS.md`): Arc `682/0/0` (exit 0), llvmpipe `670/0/3`, Ryzen iGPU
`673/0/2`; `check_port_map.py` passes (`168 — 66 kernel, 14 shader, 61 host, 27 todo`); `make_port_map.py`
regenerates `PORT-MAP.tsv` byte-identically (one row: `bf16_gemv_fp32_mmvf_cols` shader → kernel);
`strata_vk_layer_smoke` builds + RUNS.  **`z820b` is PENDING** (suspended, no WoL).  The CUDA graph API was NOT
touched.

## THE TWO SPLIT GEMVs + `shared_expert`, THE NINE SETTLED REACHABILITY VERDICTS, and THE INSTRUMENT FIX (2026-10-05, `vega`)

**THE BAR (the running line): `64 → 41` undefined references / `19 → 7` distinct full-signature
`strata::kernels::` symbols / `17 → 5` under the parent's name-only pattern.**  The matvec / GEMV / KV group falls
**6 → 1**; attention / QSA / MoE / GR / PLE / rope falls **12 → 5**; `other` is 1; glue 0 and GDN 0.  Measured
with the CURRENT STANDARD recipe (`$HOME/vkbuild-vulkan` is a **Makefiles** build dir, so **never pass `-G
Ninja`**; reconfigured + rebuilt from the current tree first):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 41   (was 64)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 7    (was 19)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 5    (was 17)

**THE GROUP TABLE (the remaining 7 distinct full-signature symbols).**

| subsystem | n | symbols |
|---|---:|---|
| **glue** | **0** | all answered |
| **matvec / GEMV / KV** | **1** | `bf16_gemv_fp32_mmvf_cols` (shader exists — a verifier-only row) |
| **attention / QSA / MoE / GR / PLE / rope** | **5** | `build_rope_table`, `rope_table_set`, `PleTable::{collect,is_open,issue}` |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **1** | `copy_i32_from_mapped` |

**THE WRAPPABLE / NO-SHADER SPLIT, updated.**  Of the previous batch's 19: **8 WRAPPABLE → 7**
(`bf16_gemv_fp32_mmvf_cols`, `build_rope_table`, `rope_table_set`, `copy_i32_from_mapped`,
`PleTable::{collect,is_open,issue}`) and **11 NO-SHADER → 0** — the two split GEMVs were PORTED (and their
shader was already in the tree; only the engine definition was missing) and the other nine are answered by a
LOUD REFUSAL each.  `shared_expert` — the 8th "wrappable, partially" — is now wired in FULL.

## DELIVERABLE A — the split GEMV pair, and `shared_expert`

**THE FINDING FIRST, because it is the batch's second instrument lesson.**  PORT-MAP carried `s_gemv_q8_0_split`
and `s_gemv_q8k_split` as `todo` / "no shader in this tree yet", and `ple_vk.cpp` recorded `shared_expert` as
un-wirable for the same reason.  **The shader `s_gemv_q8_split` WAS already in the tree** — built by
`run_gate.sh`, gated by `case_s_gemv_q8_split`.  What was missing was the ENGINE-SIDE definition, exactly the
`indexer_key_append` shape the triage records ("ported" is a claim about the SYMBOL the layer links against, not
about a shader or a plan row).  The map's `s_gemv_split_async → s_gemv_q8_split` row was also wrong (the fp16
split's shader is `s_gemv_split`).

* **Wired** `strata::kernels::s_gemv_q8k_split` / `s_gemv_q8_0_split` in `matvec_vk.cpp` (the CUDA's ONE
  `s_gemv_q8_split_kernel<CODE_BITS, Q8K>`; the port carries the activation kind in the push constant `q8k`),
  one workgroup per output row.  Engine headers unchanged.
* **Proved** by `case_s_gemv_q8k_split_entry` / `case_s_gemv_q8_0_split_entry` (three forms each: S4/S8 Q8_K
  incl. the offset form; S8/IQ4_NL/S4 Q8_0 incl. the `ffn_down_shexp` n_in=640 shape).  Each runs the SHADER
  PATH on the harness `ctx` and the ENGINE WRAPPER on its own `EnginePin`-pinned stream, requires them
  **BITWISE** equal, and requires the wrapper equal to the ENGINE'S OWN RULE (`s_gemv_q8_host_row`, the double
  transcription of `s_gemv.cu`) bounded by the SUM OF TERMS.  Raw gate lines (vega/Arc): `... bitwise 32/32
  worst 0` and `... vs the engine's own rule 8/8 worst 0.00766` (S8/Q8_K) — the deepest arm, the offset form,
  is `8/8 worst 0`.
* **`shared_expert` wired** in a new TU `vulkan/src/kernels/shared_expert_vk.cpp`: the canonical chain of
  `shared_expert.cu:242-361` — `gemv(gate)`, `gemv(up)`, `swiglu`, quantise-to-the-down-weight's-contract,
  `gemv(down)`, the BF16 scalar gate, the per-row scale.  The native-projection branches delegate to the
  port's already-wired `native_quantize_q8_1` + `native_mmvq` (the shipped `--native` launch CAN make
  `ffn_down_shexp`'s IQ4_NL native_data non-null); the native-BF16 scalar gate is answered off by this TU's own
  `shared_expert_native_bf16_enabled() == false` (the `native_*_enabled` pattern).  Proved by
  `case_shared_expert_entry`: wrapper vs a DOUBLE transcription of the whole chain, **256/256, worst 0.0475**,
  with TWO rivals given their own observables and host-side margins proving they MOVE the reference (SILU on UP:
  379.3; the scalar gate dropped: 6325.0).
* **A LATENT DEFECT FOUND AND FIXED WHILE WIRING (this batch's instance of the class).**  The port's `gemv`
  lambda CLOSED OVER the input activation (`x_q8_0`/`x_q8k`), so the DOWN projection read the input image
  instead of the `h_q8_0`/`h_q8k` buffer the two lines above it had just produced — the CUDA passes the
  activation as a PARAMETER.  The error was ~1e3× the oracle and perfectly finite; the case said so on its first
  real run.  Fixed at the source (the lambda takes `act80`/`actq8k`), not by loosening an arm.
* **Falsified, and both bite:** `s-gemv-q8k-flag-flip` (engine: hardwire `q8k=0`) → `FAIL s_gemv_q8k_split
  entry ... 0/32`; `s-gemv-q8-split-wrong-act-block` (shader: the Q8_0 block stride for the Q8_K image) →
  `FAIL ... 0/8 worst 5.34e+30`; `shared-expert-silu-on-up` → `FAIL shared_expert entry ... 0/256 worst 5.1e+05`.
* **Bench (Arc, new rows):** `s_gemv_q8_split` at the shared expert's shapes — **0.0185 ms** (Q8_K 2560→640
  g64), **0.0195 ms** (Q8_K 2560→640 g32), **0.0440 ms** (Q8_0 640→2560, `ffn_down_shexp`); llvmpipe 1.87 / 2.04
  / 6.43 ms; Ryzen iGPU 0.20 / 0.27 / 1.02 ms.  Honest per-dispatch costs of the shared expert's three GEMVs.

## DELIVERABLE B — the nine no-shader symbols, each SETTLED from the engine's own code

**All nine are UNREACHABLE under the shipped configuration, so each is a LOUD REFUSAL** (a definition that lets
the layer LINK and, if ever reached, names the flag chain and exits 2): `vulkan/src/kernels/refusals_vk.cpp`.
The per-symbol call site, enclosing condition, every flag, its default and what the shipped launch sets are in
`plan/DECODE-PATH-TRIAGE.md` ("THE NINE REACHABILITY VERDICTS", this batch), and quoted in the refusal's own
message: `native_flash_attn_short_step` & `qsa_attend_step` (layer.cpp:978/989/1002; `native_flash_attn_short`
default **false**, only `--native-flash-attn-short`), `qsa_index_step` & `topk_512_step` (layer.cpp:968;
`g_fast_select` default **true**), `native_qsa_indexer_append` (layer.cpp:944; the backend ANSWERS
`native_qsa_indexer_enabled()` **false**), and `kv_ring_table` / `kv_stream_reset` / `kv_stream_resolve`
(layer.cpp:707/709/757; `g_kv_resident` default **0** → mode 0).

**THE FINDING THE MANDATORY ANALYSIS PRODUCED — `fused_gr_read` is the `qsa_decode_attn_step` SHAPE AGAIN.**
The previous batch classed it C and wrote that "the backend's init calls `layer_set_fused_gr(false)`".  **No such
call exists in this tree**, and the shipped `--native` launch sets `gr_native_mmvf = true` (generate.cpp:1804)
with `no_fused_gr` false, so `layer_set_fused_gr(true)` runs (generate.cpp:2284) and `g_fused_gr` IS TRUE.  With
`fused_gr_supported(2560,4,320)` true the layer reaches `fused_gr_read` — the UNPORTED branch — and the ported
`gr_read` is the NON-selected one: **a HOLE, mis-labelled by a claim true of a plan and false of the code.**
**Fixed at the cause**: the branch is selected by `fused_gr_supported`, which the BACKEND defines, and a backend
with no `fused_gr` shader must answer it FALSE (the `native_mmvq_supported` discipline).  `ple_vk.cpp` now
returns false, the engine takes the ported `gr_read`, and `fused_gr_read` is genuinely unreachable → its refusal.
`case_fused_gr_supported_entry` now asserts BOTH readings distinctly (backend false everywhere; the CUDA
geometry rule true at (2560,4,320)); falsified by `fused-gr-supported-true` → `FAIL fused_gr_supported entry
... 5/6`.

## DELIVERABLE C — THE INSTRUMENT: `kernel` now states TWO facts

`PORT-MAP.tsv`'s `kernel` kind meant only "a shader exists", and was twice read as "the backend answers the
symbol the engine calls" (`indexer_key_append`, `qsa_decode_attn_step`).  It now means **a shader exists AND the
Vulkan backend DEFINES the symbol**, checked by `tools/check_port_map.py` scanning the backend's own TUs; a
`kernel` row with no definition FAILS the gate.  A new kind **`shader`** states the middle fact (a shader is
built, the backend does NOT define the symbol); a `shader` row that IS defined also FAILS (a stale row).  Applying
it found **14 more** unwired-but-shadered symbols (`sample_tokens`, `moe_hit_*`, `native_expert_grouped`,
`iq_dequant_f32`, `iq_embed_rows`, `coupled_draft_sample`, the `_multi` pair, …) which are now kinded `shader`,
plus `s_gemv_split_async` (whose shader was mis-attributed to the q8 one).  The map reads
**`168 = 65 kernel + 15 shader + 61 host + 27 todo`** (was `78/0/61/29`).  **Falsified:** renaming the backend's
`s_gemv_q8_0_split` definition makes the checker exit 1 with the named FAIL; restoring it returns to exit 0.
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (verified by sha256).

## RESULTS (vega)

Gate: Arc (`intel_icd`) **669/0/0**; llvmpipe **657/0/3** (the 3 documented coopmat/vram skips); Ryzen iGPU
(`radeon_icd`) **659/1/2** — the ONE failure is the documented, open `budget: independent requery agrees` flake,
**not one of this batch's cases** (the 3 skips are the coopmat/vram ones).  `check_port_map.py` passes;
`make_port_map.py` byte-identical; `strata_vk_entry_smoke` builds.  All four new falsifications bite.
**`z820b` is PENDING** (suspended, no WoL — no XTX/K620 number is claimed).  The CUDA graph API was NOT touched.

## THE MoE / QSA / GR / PLE TAIL — six more entry points, TWO CLASSIFICATION FINDINGS, and how far M-B is from a LINK (2026-10-05, `vega`)

**THE BAR (the running line): `75 → 64` undefined references / `25 → 19` distinct full-signature
`strata::kernels::` symbols / `23 → 17` under the parent's name-only pattern.**  The attention / QSA / MoE / GR /
PLE / rope group falls **17 → 12**; matvec/GEMV/KV is unchanged at 6 and `other` falls **2 → 1** (glue 0, GDN
mixer 0).  Measured with the CURRENT STANDARD recipe (`$HOME/vkbuild-vulkan` is a **Makefiles** build dir, so
**never pass `-G Ninja`**; reconfigured + rebuilt from the current tree first):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 64   (was 75)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 19   (was 25)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 17   (was 23)

**THE GROUP TABLE (the remaining 19 distinct full-signature symbols).**

| subsystem | n | symbols |
|---|---:|---|
| **glue** | **0** | all answered |
| **matvec / GEMV / KV** | **6** | `bf16_gemv_fp32_mmvf_cols`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve` |
| **attention / QSA / MoE / GR / PLE / rope** | **12** | `build_rope_table`, `rope_table_set`, `fused_gr_read`, `shared_expert`, `native_flash_attn_short_step`, `native_qsa_indexer_append`, `qsa_attend_step`, `qsa_index_step`, `topk_512_step`, `PleTable::{collect,is_open,issue}` |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **1** | `copy_i32_from_mapped` |

**HOW FAR M-B IS FROM A LINK — THE NUMBER THE BRIEF ASKS FOR.**  **19 symbols remain; 8 are WRAPPABLE and 11
have NO SHADER in this tree.**  Exhausting the wrappable set lands on 11 symbols whose only remaining work is
**SHADERS PORTED, not wrappers written** — a different work shape.  The 8 wrappable: `bf16_gemv_fp32_mmvf_cols`
(shader exists — verifier-only), `build_rope_table`, `rope_table_set`, `copy_i32_from_mapped`,
`PleTable::{collect,is_open,issue}` (pure host) and **`shared_expert` — PARTIALLY**: its canonical path needs
`s_gemv_q8_0_split`/`s_gemv_q8k_split`, two of the no-shader eleven.  The 11 no-shader: `qsa_attend_step`,
`qsa_index_step`, `topk_512_step`, `native_qsa_indexer_append`, `native_flash_attn_short_step`,
`s_gemv_q8_0_split`, `s_gemv_q8k_split`, `fused_gr_read`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve`.

**THE SIX, IN THE ORDER THE ENGINE'S OWN BODY REACHES THEM** (SOURCE order of the call sites; within a
native/legacy pair the branches are alternatives):

| # | symbol | call site | kind | shader(s) | TU |
|---|---|---|---|---|---|
| 1 | `qsa_step_fill` | layer.cpp:908 | PURE HOST | — (writes HOST memory) | qsa_vk.cpp |
| 2 | `indexer_key_append` | layer.cpp:948 | kernel | indexer_key_append.spv | qsa_vk.cpp |
| 3 | `fused_gr_supported` | layer.cpp:1189/:1328 | PURE HOST | — (a geometry predicate) | ple_vk.cpp |
| 4 | `shared_expert_scratch_bytes` | layer.cpp:347 | PURE HOST | — (a size) | ple_vk.cpp |
| 5 | `moe_combine` | layer.cpp:464 | kernel | moe_combine_f32.spv | ple_vk.cpp |
| 6 | `ngram_rows` | layer.cpp:1293 | PURE HOST | — (the PLE hash) | ple_vk.cpp |

Engine headers unchanged.  Each proved by a new `case_*_entry`, `EnginePin`-pinned where it opens its own
stream.  Raw lines (vega, `intel_icd`/Arc B70):

| # | symbol | wrapper == shader path (bitwise) | wrapper vs the oracle |
|---|---|---|---|
| 1 | `qsa_step_fill` | — (no shader) | 32/32 int32 + three rivals MOVE (ceil `n_bid` / unclamped `width` / `n_bid` on `pos`) |
| 2 | `indexer_key_append` | **640/640 + spare, w 0** | 256/256, worst 0.0655 (terms/mean bound); margins 0.530 / 0.082 |
| 3 | `fused_gr_supported` | — | 6/6 geometries (5 false arms falsify "always true") |
| 4 | `shared_expert_scratch_bytes` | — | 7/7 widths + two rivals MOVE (Q8_K sized as Q8_0 / no alignment) |
| 5 | `moe_combine` | **2560 + 2560 + 37, w 0** | w 0.078 / 0 / 0.0588 (terms bound); shared-router-weighted rival MOVE |
| 6 | `ngram_rows` | — | 304/304 **vs the EXTERNAL `ref/ngram.py` vectors**; 4 rivals MOVE on 5–6 of 6 cases |

**FINDING 1 — `qsa_step_fill` IS NOT A NO-SHADER ROW.**  The brief listed it with the shader-port work; its
signature is `void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes&)` and `qsa.cu:699` writes
HOST memory.  It needs no shader and never did.  (PORT-MAP has always kinded it `host`.)  Wired and proved.

**FINDING 2 — `indexer_key_append` WAS RECORDED "LANDED" WITHOUT A DEFINITION.**  `plan/DECODE-PATH-TRIAGE.md`
carries it as "LANDED 2026-10-05" in the class-A table; the shader (`indexer_key_append.spv`) and a SHADER case
exist, but no `strata::kernels::indexer_key_append` wrapper did — so the layer link showed it undefined.  The
same defect shape as the QSA norm-weight-indexing bug: **an assertion of "ported" that checked a shader existed
and not that a definition did.**  Written here, proved by `case_indexer_key_append_entry`.

**A MAP MIS-KIND FOUND (reported, map left byte-identical).**  PORT-MAP kinds `kv_ring_table`,
`kv_stream_reset` and `kv_stream_resolve` as `host`, but `src/kernels/cuda/kv_stream.cu` LAUNCHES kernels
(`reset_kernel<<<128,256>>>` :199, `ring_kernel<<<64,256>>>` :226, `resolve_kernel` + `copy_kernel` :204-222).
They are DEVICE ops with no shader here — the same family as the `gr_read`/`fused_gr_read` mis-kind the triage
settled.  They are therefore counted in the 11 no-shader, not the 8 wrappable.

**THE DISCIPLINES.**  Every oracle is the engine's OWN rule (a `.cu` body), except `ngram_rows`' which is the
EXTERNAL generated vector file — so a shader/oracle SHARED mistake cannot pass, and each rival reading gets its
own observable and a host-side margin proving it MOVES (or, where genuinely indistinguishable, is stated: the
`moe_combine` k=1 "first term as a sum" rival is `0.0f + x == x` and is stated, not asserted — carried over from
the native member's case).  The latent-defect class (a parameter used as something it does not mean; an
in-place kernel where the CUDA writes a separate destination; an element-size mismatch between the engine's
layout and the port's shader) was checked against all six; none of the three shapes occurs in them (the two
kernels' regions are f32 in both the engine and the shader).

**RESULTS (vega).**  Gate: Arc (`intel_icd`) **655/0/0**, llvmpipe **643/0/3** — **+28 verdicts per arm**, 0
failed.  Ryzen iGPU (`radeon_icd`): **645/1/2, 644/2/2, 645/1/2** across three runs — the documented
intermittent `budget` flake and the open non-deterministic wrong-value defect, this session seen in
**`bf16_gemv entry n_in=2560 n_out=128` (1 of 512)** and **`fused_gdn_ab h_v=48 n=2560` (1 of 96)** — neither
is one of this batch's cases.  **New datum on the record:** the intermittent defect has now appeared in a THIRD
kernel (`fused_gdn_ab`), as well as `bf16_gemv`/`bf16_gemv_split` and `ple_block (key)` — the device/driver
characterisation stands.  `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 112 shaders built, 93
claimed`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  `strata_vk_entry_smoke` builds +
runs PASS.  **`z820b` is PENDING** (suspended, no WoL — no XTX/K620 number is claimed).  The CUDA graph API was
NOT touched.

## THE PLE / GR SHARED STAGES + THE MoE ROUTING ROWS — the next six entry points, and a WORKSPACE-LAYOUT DEFECT FOUND AND FIXED (2026-10-05, `vega`)

**THE BAR (the running line): `91 → 75` undefined references / `33 → 25` distinct full-signature `strata::kernels::`
symbols / `31 → 23` under the parent's name-only pattern.**  The attention / QSA / MoE / GR / PLE / rope group
falls **25 → 17**; every other group is unchanged (glue 0, matvec/GEMV/KV 6, GDN mixer 0, other 2).  Measured with
the CURRENT STANDARD recipe (`$HOME/vkbuild-vulkan` is a **Makefiles** build dir, so **never pass `-G Ninja`**;
reconfigured + rebuilt from the current tree first):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 75   (was 91)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 25   (was 33)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 23   (was 31)

**THE GROUP TABLE (the remaining 25 distinct full-signature symbols).**

| subsystem | n | symbols |
|---|---:|---|
| **glue** | **0** | all answered |
| **matvec / GEMV / KV** | **6** | unchanged (`bf16_gemv_fp32_mmvf_cols`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve`) |
| **attention / QSA / MoE / GR / PLE / rope** | **17** | (was 25) the six device symbols + the two host rows below moved to answered |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

**THE SIX, IN THE ORDER `block_layer_pre`/`block_layer_post` REACH THEM** (the plan's §3 list is not an order; the
order is the SOURCE order of the call sites - within a native/legacy pair the branches are alternatives):

| # | symbol | call site | shader(s) | TU |
|---|---|---|---|---|
| 1 | `gr_write` | layer.cpp:1261 (run2 unfused), :1195/:1329 | gr_write.spv | ple_vk.cpp |
| 2 | `ple_block` | layer.cpp:1208 (layer 1) | quantize_q8_0 + s2_gemv_q8 + f32_to_bf16 + bf16_gemv + ple_gnorm/gate/bcast/conv + add3 | ple_vk.cpp |
| 3 | `ple_history_advance` | layer.cpp:1222 | ple_history_advance.spv | ple_vk.cpp |
| 4 | `gr_read` | layer.cpp:1255 (run0 unfused) | gr_norm + gr_down + gr_gate + gr_mean + gr_inject | ple_vk.cpp |
| 5 | `router_top10` | layer.cpp:373 (`moe_route`'s generic router) | router_top10_f32.spv | ple_vk.cpp |
| 6 | `native_moe_combine` | layer.cpp:463 (`moe_combine_parts`, the DEFAULT combine) | native_moe_combine.spv | ple_vk.cpp |

New TU `vulkan/src/kernels/ple_vk.cpp` (added to the CMake kernel list and to `run_gate.sh`'s link line).  Engine
headers unchanged.  The two `host` rows the TU answers (NOT a bare bind): `gr_workspace_init`
(`block_buffers_init`, layer.cpp:1140) and `ple_block_scratch_bytes` (layer.cpp:1203/1316), both transcribed from
`gr.cu` / `ple.cu`; plus the PLE/GR BRANCH POLICY (`ple_native_bf16_enabled`/`ple_native_postops_enabled` → FALSE,
`gr_set_native_mmvf`/`gr_set_fp32_activations` → no-ops), the `native_gdn_enabled()` pattern: the backend reports
what it implements, so a `--native` launch cannot route at a kernel this tree lacks.

**EACH PROVED BY ITS CASE THROUGH THE ENGINE WRAPPER - BITWISE vs THE SHADER PATH AND vs THE ENGINE'S OWN RULE.**
Six new `case_*_entry` in the port's gate, `EnginePin`-pinned.  Raw lines (vega, `intel_icd`/Arc B70):

| # | symbol | wrapper == shader path (bitwise) | wrapper vs the engine's rule |
|---|---|---|---|
| 1 | `gr_write` | **10240/10240 + 192/192, w 0** | w 0.609 / 0.613 (gr.cu rule, double; zero-inject EXACT) |
| 2 | `ple_block` | **key 10240, value 2560, gate 4, gated 10240, normalized 10240, conv 10240, result 10240 - all w 0** | w 6.58e-04 (ple.cu chain fed the device's key/value) |
| 3 | `ple_history_advance` | **92160/92160, w 0** | bit-exact (a copy) |
| 4 | `gr_read` | **66/66 + 19/19, w 0** | w 9.31e-03 / 2.43e-04 (gr_parity rule, double) |
| 5 | `router_top10` | **80/80, w 0** | ids exact; w 1.44e-07 (router_top10.cu, tol 1e-5) |
| 6 | `native_moe_combine` | **2560 + 2560 + 37, w 0** | w 9.25e-02 / 0 / 3.23e-02 (the native f32 expression, terms bound) |

**WRITTEN SO A SHARED SHADER/ORACLE MISTAKE CANNOT PASS.**  Every oracle is a transcription of the engine's OWN
rule (the `.cu` body), not of the shader, so a mistake shared with the shader would have to live in the RULE.  Each
rival reading gets its OWN observable and a HOST-SIDE margin proving it MOVES the reference:
* `gr_write`: drop-the-2, drop-the-`/hc`, and block_out read PER STACK ELEMENT (a distinct `bo_stack`) - three
  rivals, three observables, all shown to move the output.
* `ple_block`: the conv's row-MAJOR history (`hist[c + hc_dim*row]`) and `gated` (not `normalized`) as the conv
  input - both shown to move the rule (the row-major rival by 1.02e+04 err/tol).
* `ple_history_advance`: the flat-memmove (row-major) reading - shown to scramble the channels.
* `gr_read`: whole-stack RMS (vs per-stream) and SUM-over-streams (vs MEAN) - both shown to move `mixed`.
* `router_top10` / `native_moe_combine`: exact ties exercise the lowest-index rule; the shared row weighted vs
  added-plain moves the combine.  **STATED, NOT ASSERTED:** for `native_moe_combine` `k == 1`, "first term as a
  sum" (`0 + parts*w[0]`) is GENUINELY indistinguishable from the product (`0.0f + x == x`), so no legal fixture
  can move it; the arm says so rather than asserting a decorative margin.

**A CROSS-CUTTING DEFECT FOUND AND FIXED WHILE WIRING (the class the last two batches found).**  The port's GR
shaders store the bf16 activations as **f32** - `gr_norm.comp` binding 3 is `float v[]` and it writes
`bf16_round(x)` AS AN F32 ("bf16(xn) as f32"); `gr_down`/`gr_gate`/`gr_inject` read it the same way.  But the
engine's `GrWorkspace` sizes `xq`/`lq` as **uint16** (`gr.cu`'s `gr_down_kernel<uint16_t>`), so wiring `gr_read`
with the engine's own workspace made the shader write 4 bytes where the region held 2 - into `lq`.  `gr_workspace_init`
now sizes `xq`/`lq` for f32 (which is legal: `GrWorkspace` is an opaque pointer struct, `gr_workspace_bytes` is the
AUTHORITY `block_buffers_init` allocates by, and `gr_read` reads the same pointers), and the values are identical -
only a byte-layout comparison against the CUDA would see a difference, and that is stated.  **The gate's first run
caught it as `gr_read entry ... 0/66` on every device.**

**THE STANDARD RECIPE, AND THE CASE BUGS THE BITWISE ARM CAUGHT.**  `ple_block`'s first wrapper wrote
`gnorm(gated)` back ONTO `gated` (the port's `ple_gnorm` is in place) - clobbering it before `add3` and the export
and leaving `d_norm` stale.  The bitwise arm read **0/10240 on every element** (a whole-region disagreement, not a
tolerance), which is the signature of the wrong memory, and the fix is the copy-into-`d_norm`-first the CUDA's
distinct-destination call implies.  Both were fixed at the source, not by loosening an arm.

**RESULTS (vega).**  Gate: Arc (`intel_icd`) **639/0/0** (exit 0), llvmpipe **627/0/3**, Ryzen iGPU (`radeon_icd`)
**630/0/2** - **+27 verdicts on every arm**, 0 failed.  `strata_vk_entry_smoke` builds + RUNS four of the six
wrappers on the Arc (PASS, exit 0) and pulls the whole TU into the link.  `check_port_map.py` passes (`168 - 78
kernel, 61 host, 29 todo; 112 shaders built, 93 claimed`); `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **`z820b` is PENDING** (suspended, no WoL - no XTX/K620 number is claimed).  The CUDA graph API
was NOT touched.

**LEFT IN THE GROUP (17).**  Reported, NOT stubbed: `qsa_attend_step`, `qsa_index_step`, `qsa_step_fill`,
`topk_512_step`, `native_qsa_indexer_append`, `native_flash_attn_short_step` (the diagnostic gathered-window
branch) have NO shader in this tree; `fused_gr_read`/`fused_gr_supported` are the GR-fused alternative the
`gr_set_native_mmvf(false)` contract removes; `build_rope_table`/`rope_table_set`/`ngram_rows`/`PleTable::{collect,
is_open,issue}` are host rows the engine's own tables own; `shared_expert(+_scratch_bytes)` and `moe_combine` are
the remaining MoE rows (`shared_expert` is on-path every layer; `moe_combine` is the off-path legacy combine,
since `native_moe_combine_enabled()` answers true); `ple_block_projected` stays `todo`.

## AN INTERMITTENT, NON-DETERMINISTIC FAILURE ON THE Ryzen iGPU — CHARACTERISED, NOT FIXED (2026-10-05)

**This is an OPEN DEFECT and it outranks the performance work below.** A green `run_gate.sh` on `vega` does
**not** currently prove determinism: the same commit, the same binary, the same fixture seed (`std::mt19937
g_rng(11)`) fails on the radeon ICD in ~11% of runs and passes in the rest. The failure was found by the
parent tier's own verification run (radeon `434 passed, 2 failed, 2 skipped`; the other failure is the
documented `budget: independent requery agrees` flake) and was reported as
`FAIL bf16_gemv n_in=2560 n_out=128 130/131 worst 36.7`, against `131/131 worst 0.00573` for the SAME case in
the SAME run on the Arc — a ~6000× gap, so it is not a tolerance question.

**THE RATE AND THE ARMS (radeon only, `vega`; same binary, ICD pinned with `VK_ICD_FILENAMES`).**

| measurement | runs | runs with a bad case | arms seen |
|---|---:|---:|---|
| gate, radeon (`--spv-dir shaders`) | 59 | 10 | `bf16_gemv_split n=2560 n_out=512` ×5, `bf16_gemv n=2560 n_out=512` ×3, `bf16_gemv_split n=2560 n_out=48` ×1, `fused_gdn_ab h_v=48 n=2560` ×1 |
| gate, radeon (diagnostic build, +23 lines) | 102 | 11 | `bf16_gemv_split n_out=512` ×10, `bf16_gemv n_out=512` ×1 |
| gate, **intel** (Arc, ANV) | 84 | **0** | — |
| gate, **lvp** (llvmpipe) | 47 | **0** | — |

Every bad case is ONE row out of 512 (occasionally 2–5); the reported `worst` spans 1.23 to 664 × the
terms-derived bound. Deviation of the device's output from a hand-written f32 emulation of the shader's own
summation (256-lane strided accumulation, pairwise tree): **3e-6 to 6e-4 absolute, i.e. a few ULP of the
result**, on one row.

**WHAT IT IS NOT — each ruled out by measurement, not by argument.**

* **Not the kernel's arithmetic, and not a stale or partial input read.** A focused probe
  (`/tmp/repro3.cpp`, built against the port's own `harness/vk_compute.*` and the shipping `bf16_gemv.spv`)
  alternates TWO complete fixtures A/B so every element differs between consecutive dispatches, and on a bad
  dispatch searches EVERY single-element mutation of the failing row against the device's value bit for bit
  (previous-fixture value, zero, low byte zeroed, high byte zeroed). It matches on a few, and in most failures
  **no single-element mutation of any of those forms reproduces the value** — and the deviation is a few ULP,
  orders of magnitude below one element's contribution. A stale or partially-written word would move the sum
  by that word's contribution (~0.6) and would PERSIST on re-dispatch, because the bytes stay stale.
* **Not write/dispatch ordering.** Re-dispatching the IDENTICAL buffers with no host rewrite returns the
  correct value: 10 of the 11 gate hits, and 100% of the probe's hits (one needed a second re-dispatch).
  A host-write-visibility race would survive the re-dispatch.
* **Not descriptor-pool growth.** `(descriptor pool 2 created: the previous one was full)` appears at the same
  point (line ~460) in EVERY run, including the ~89% that pass; and the standalone probe reproduces with no
  pool growth at all (one pipeline, one descriptor set).
* **Not an aliasing/binding bug.** Only one row (one workgroup) is ever wrong; a wrong descriptor slice or an
  overlapping binding corrupts many rows.
* **Not the fixture or the RNG.** With the fixture fixed, passing runs reproduce their `worst` EXACTLY
  (e.g. `0.00699` / `0.00722` for the two `n_in=2560` arms, run after run). An earlier independent build also
  saw `fused_gdn_ab`, so the failing set is not one shader.
* **Not a GPU reset.** No `amdgpu` ring timeout or reset in `dmesg` during any of these runs (the device HAS
  two historical resets and one page fault on record, both hours earlier and on other binaries).

**ITS SHAPE, MEASURED.** The probe reproduces only with a FRESHLY ALLOCATED buffer per dispatch (mode C:
2 bad dispatches in 24,000 under 4 concurrent probe processes) and never with a reused buffer
(modes A/B: **0 in 48,000**). The rate tracks DEVICE/HOST LOAD, not process count: 4 concurrent probe
processes on the radeon gave 2–5 bad dispatches per 20,000, while one radeon probe run alongside four Arc
probe processes and two gate loops gave **73 in 20,000** (`/tmp/repro3-single-radeon.log` — so it is NOT a
clean single-process baseline; the quiet single-process figure is 1 in 8,000). **The same probe on the Arc
(ANV) gave 0 bad dispatches in 80,000** — same binary, same shaders, same fixtures, same host code.

**THE HONEST VERDICT.** The evidence rules out the port's own arithmetic: the shader's f32 result matches a
hand emulation of its own summation order bit-for-bit on the vast majority of rows on BOTH devices, re-dispatch
of the identical buffers is always correct, and the Arc — which has none of the iGPU's shared-memory /
coherency path — never shows it in 80,000 dispatches. **Two hypotheses are still standing, and neither is
closed:**

1. **A stale/partial read, ordered by the write/dispatch or the dispatch/dispatch boundary.** Favoured by the
   shape of the evidence (fresh buffers only, rate scaling with load, self-correcting on re-dispatch) and
   disfavoured by the magnitude (a few ULP, where a stale word would move the sum by ~0.6). Not reproduced by
   the single-element mutation search; the ordering could still be at a coarser granularity than one element.
2. **A subgroup-width-dependent shared-memory read on the 64-wide RADV implementation.** Favoured by the
   device split (64-wide RADV only; 32-wide ANV and 8-wide llvmpipe clean over 131 gate runs) and by the
   affected shaders all using the workgroup `wg_sum` barrier tree; disfavoured by the fact that the tree has
   no width to be wrong about by construction.

**Until one of them is closed, treat a green `run_gate.sh` on `vega` as "no failure was observed", NOT as
"the suite is deterministic", and read the radeon arm's FAILURE LIST, not just its total.** Nothing was
suppressed for this: the case is NOT skipped on radeon, the bound is NOT widened, and the failure is NOT
deleted from the README.

Evidence kept: `/tmp/rgdiag-hit-*.log` (11 radeon gate hits with re-dispatch diagnostics),
`/tmp/rg200-fail-*.log` (10 pre-diagnostic hits), `/tmp/repro2-*.log` (the fma vs mul+add emulation split),
`/tmp/repro3-*.log`, `/tmp/repro3-intel-*.log`, `/tmp/repro3-single-radeon.log`,
`/tmp/evidence-bf16gemv-radeon-FAIL.log`. The two probes are `/tmp/repro2.cpp` and `/tmp/repro3.cpp` (not
committed - they are diagnostic scratch, built against `harness/vk_compute.*` and the shipping `bf16_gemv.spv`).

**DESCOPED BY RE-PRIORITISATION (2026-10-05).** The performance tier's two remaining items — the sampler's
next-cost attack and the PER-TOKEN BUDGET table — are NOT done. The user re-ranked engine integration
(tokens produced on the B70) above further performance work, so this increment stops at the correctness
finding. What IS on the record, because it was measured before the re-ranking: the bench now measures the
split sampler at the ENGINE'S REAL parameters (top_k 20, the penalty window DISABLED) beside the old
(k 64 / window 64) row this harness used to quote (see `bench/README.md`), and the engine's real sampler
parameters are `k = sampled_k(top_k, n_vocab)` = **20** at the default `top_k = 20`, `hlen =
min(penalty_last_n, history_len)` = **0** at the default `penalty_last_n = 0`, `kSplitMaxBlocks =
kSplitMaxRows = 64`, stage 1 = 128 threads over `kSplitBlockSpan` 4096, stage 2 = one 32-thread warp per row.
The port renders both stages as ONE 256-thread workgroup per row (`SC_PER_LANE = 16`), which is the port's
shape, NOT the engine's `kSplitPerLane = 32`.

**BOX `z820b`: PENDING.** The box is suspended (`100% packet loss`, `No route to host`, no Wake-on-LAN), so
the whole-tree sync and `run_gate.sh` / `run_bench.sh` there are PENDING and **no XTX or K620 number is
claimed anywhere in this section or the ones below it**. Every measured number above is from `vega`; the
sampler rows are from `vega`'s Arc (intel_icd), Ryzen iGPU (radeon_icd) and llvmpipe (lvp_icd).


## THE DEFAULT QSA DECODE ATTENTION — `qsa_decode_attn_step` IS REACHABLE, WAS A HOLE, AND IS NOW PORTED (2026-10-05, `vega`)

**THE VERDICT, FROM THE ENGINE'S OWN CODE: REACHABLE — and it was a HOLE of the class the previous two batches
found.**  The call site (`src/core/layer.cpp:978-980`):

    if (g_fast_attn && !native_flash_attn_short && dump == nullptr) {
        const strata::kernels::QsaAttnPools pools = qsa_attn_pools(st);
        strata::kernels::qsa_decode_attn_step(b.qcur, pools, b.ids, st.step, cap, s, b.attn_scratch, b.attn, stream);
    } else { ... the gathered window + `qsa_attend_step` ... }

Every flag in that condition, its DEFAULT, and what the shipped launch sets:

| flag | default | what the shipped launch sets | effect on the condition |
|---|---|---|---|
| `g_fast_attn` | **true** (`layer.cpp:42`) | `layer_set_fast_attn(!o.no_fast_attn)` (`generate.cpp:2280`); `o.no_fast_attn` defaults **false** (`generate.cpp:418`), set only by `--no-fast-attn` (`:1519`) | TRUE -> the `if` branch, this symbol |
| `native_flash_attn_short` | **false** (`layer.cpp:92`) | `layer_set_native_flash_attn_short(o.native_flash_attn_short)` (`generate.cpp:2287`); `o.native_flash_attn_short` defaults **false** (`generate.cpp:272`), set only by `--native-flash-attn-short` (`:1329`) — **NOT by `--native`** | `!false` = TRUE |
| `dump` | the caller's argument | `block_layer(...)` passes `bb.dump` (`layer.cpp:1259`); the decode path passes **nullptr** — only the P6 verifier / debug supplies one | `nullptr == nullptr` = TRUE |

All three are TRUE on the shipped launch, so `qsa_decode_attn_step` **IS the default decode attention**.  The
REACHABILITY AUDIT (`plan/DECODE-PATH-TRIAGE.md`) recorded the selected branch as *"the ported
`qsa_decode_attn_step`"* and only the `else` member (`qsa_attend_step`) as class C — but this symbol had **NO
shader in this tree** (PORT-MAP pointed it at `attn_decode_short`, which its own header says is
`native_flash_attn_short_step`: a gathered `[cap,2,256]` f16 WINDOW, NOT the pool+page-table kernel — a different
kernel).  **So the audit's row was WRONG and the symbol was a HOLE.**  Both the audit row and PORT-MAP are
corrected here.

**THE PORT — the shader-port job the last batch reported, now done.**  `ports/vulkan/shaders/qsa_decode_attn.comp`:
the KV POOLS read DIRECTLY through the PAGE TABLE (no gather copy), ONE workgroup per query head, online softmax.
Mode 0 = f16 pools (the shipped `--kv fp16` default, `generate.cpp:310`), mode 1 = int8 codes + fp16 scale per 64
values; the q4_0 pool and the K8V4 hybrid are **REFUSED loudly** (this shader does not implement their storage).
The engine's chunk+merge CUDA is RE-DERIVED onto one workgroup with barrier-tree reductions (subgroup ops are
banned here) and needs NO scratch; **bit-identity with the CUDA's chunked form is NOT claimed** (the softmax
summation order differs) — the gate MEASURES the difference against a DOUBLE transcription of the engine's own
rule.  Wired in `vulkan/src/kernels/qsa_vk.cpp` as `strata::kernels::qsa_decode_attn_step` (its 9th entry point)
plus the `host` row `qsa_decode_attn_scratch_floats` (transcribed from `qsa_decode_attn.cu`: CHUNK=64, HD=256).
`strata_vk_entry_smoke` runs it too.

**PROVED BY `case_qsa_decode_attn`** (3 arms; the artifact's geometry 24 query heads x 2 KV heads x head_dim 256;
a PERMUTED page table and two `-1` pages; the shader path vs the ENGINE WRAPPER BITWISE, and both vs the rule),
**FALSIFIED by `gates/inject-verify.sh qsa-decode-attn-drop-kv-head`.**

| arm | wrapper == shader path (bitwise) | shader path vs the engine's rule (double) |
|---|---|---|
| page_size=4, 12 ids, 2 pages masked | **6144/6144, w 0** | **w 9.65e-04** (abs floor 1e-5) |
| page_size=1 (unpaged), 8 ids, none masked | **6144/6144, w 0** | **w 2.01e-04** |
| page_size=8, 16 ids, 2 pages masked | **6144/6144, w 0** | **w 5.92e-04** |

**FALSIFICATION:** `qsa-decode-attn-drop-kv-head` -> **`FAIL qsa_decode_attn (page_size=4, 12 ids, 2 pages
masked): shader path vs the engine's rule (double) 0/6144 worst 2.26e+04`** (the pool row's `kvh` term dropped, so
both KV heads read head 0's data).  The margin arms (host-side) confirm the fixture SEPARATES the rule from the
identity-page and head-major rivals.

**THE BAR (the running line): `94 → 91` undefined references / `35 → 33` distinct full-signature
`strata::kernels::` symbols / `33 → 31` under the parent's name-only pattern.**  The
attention / QSA / MoE / GR / PLE / rope group falls **27 → 25**; every other group is unchanged.  Measured with
the CURRENT STANDARD recipe (`$HOME/vkbuild-vulkan` is a Makefiles build dir — never `-G Ninja`; reconfigured +
rebuilt from the tree first); the two symbols resolved are `qsa_decode_attn_step` and
`qsa_decode_attn_scratch_floats`.

**THE GROUP TABLE (the remaining 33 distinct full-signature symbols).**

| subsystem | n | symbols |
|---|---:|---|
| **glue** | **0** | all answered |
| **matvec / GEMV / KV** | **6** | unchanged |
| **attention / QSA / MoE / GR / PLE / rope** | **25** | (was 27) `qsa_decode_attn_step` + `qsa_decode_attn_scratch_floats` moved to answered |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

**THE MEASUREMENT (bench, Arc; a SOLO row — no second implementation existed to pair against).**  At the artifact's
geometry `qsa_decode_attn (default)` reads **0.3797 ms** at `n_ids=256`, **1.2946 ms** at 1024, and **2.5203 ms**
at the engine's real selection width **2048** (`qsa_selection_width`, `idx_top_k = 2048`) — LINEAR in `n_ids`,
because the port's correctness re-derivation is ONE barrier-tree reduction PER CELL.  That is the honest cost of
the correctness form; the CUDA's CHUNK+merge decomposition is the performance form and was NOT ported (this task
is the missing DEFAULT symbol, not its speed).

**RESULTS (vega).**  Gate: Arc (intel_icd) **612/0/0** (exit 0), llvmpipe **600/0/3**, Ryzen iGPU (radeon_icd)
**603/0/2** — **+9 verdicts on every arm**, 0 failed.  `strata_vk_entry_smoke` RUNS the new wrapper (PASS, worst
rel 2.92e-05).  `check_port_map.py` passes (`168 - 78 kernel, 61 host, 29 todo; 112 shaders built, 93 claimed`)
and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (ONE row changed:
`qsa_decode_attn_step  kernel  qsa_decode_attn`).  **`z820b` is PENDING** (no XTX/K620 number).  The CUDA graph
API was NOT touched.

## INCREMENT I5 (ENGINE) — THE ATTENTION / QSA / rope ENTRY POINTS: THE NON-GDN BODY'S NEXT EIGHT, AND A CROSS-CUTTING WEIGHT-INDEXING DEFECT FOUND AND FIXED (2026-10-05, `vega`)

**THE BAR (the running line): `106 → 94` undefined references / `44 → 35` distinct full-signature
`strata::kernels::` symbols / `42 → 33` under the parent's name-only pattern.**  The
attention / QSA / MoE / GR / PLE / rope group falls **36 → 27**; every other group is unchanged.  MEASURED with the
CURRENT STANDARD recipe (`$HOME/vkbuild-vulkan` is a **Makefiles** build dir, so **never pass `-G Ninja`**;
reconfigured + rebuilt from the current tree first):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 94   (was 106)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 35   (was 44)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 33   (was 42)

**THE EIGHT, IN THE ORDER THE ENGINE'S OWN NON-GDN BODY REACHES THEM.**  The non-GDN layer body is `qsa_layer`
(`src/core/layer.cpp:876`) - the analogue of `gdn_layer` for the 12 QSA layers (`qsa_interval = 4`), reached from
`block_layer_pre` (`layer.cpp:1259`).  The plan's §3 list is NOT an order; the order is the source order of the call
sites (within a stage the native/legacy branches are alternatives, so the order is the order the calls appear):

| # | symbol | call site | stage | shader |
|---|---|---|---|---|
| 1 | `native_qsa_rms_norm_weighted` | layer.cpp:879 | `normalize_rotate`, the NATIVE norm | native_qsa_rms_norm_weighted.spv |
| 2 | `native_rope_apply` | layer.cpp:881 | `normalize_rotate`, the NATIVE rope | native_rope_apply.spv |
| 3 | `rope_neox_apply` | layer.cpp:882 | `normalize_rotate`, the LEGACY rope | rope_neox.spv |
| 4 | `qsa_block_scores` | layer.cpp:970 | the `g_fast_select` block scores | qsa_block_scores.spv |
| 5 | `qsa_block_topk` | layer.cpp:971 | the `g_fast_select` weighted top-k | qsa_block_topk.spv |
| 6 | `native_qsa_gate_apply` | layer.cpp:1010 | the gate, NATIVE member | native_qsa_gate_apply.spv |
| 7 | `qsa_gate_apply_f32` | layer.cpp:1011 | the gate, LEGACY member | qsa_gate_apply_f32.spv |
| 8 | `native_router_top10` | layer.cpp:370 (`moe_route`) | the expert routing | native_router_top10.spv |

All eight live in the new `vulkan/src/kernels/qsa_vk.cpp` (attention + rope ride with the QSA body; the router is the
next stage `block_layer_pre` run4 reaches after run1's `qsa_layer`).  Engine headers unchanged.  Each is a THIN
bind-and-dispatch over the shader its gate case already proved; `native_rope_apply` derives `theta_scale` and the
`RopeScaling::kernel_args` constants (exactly as the source does), and `native_qsa_rms_norm_weighted` carries the
`(n_cols, n_rows)` argument order the header declares.  A per-dispatch mrope sentinel (a Vulkan binding cannot be
null) is cached in `Stream::dummy` (`vulkan/src/device/vk_arena.hpp`), lazily, because `arena_alloc` never
decreases.

**EACH PROVED BY ITS CASE THROUGH THE ENGINE WRAPPER - BITWISE vs THE SHADER PATH AND vs THE EXPLICIT ORACLE.**
A new `case_*_entry` in the port's gate runs the port's EXISTING case's fixture through (A) the shader path and (B)
the ENGINE WRAPPER `strata::kernels::<symbol>` on its own engine stream (`EnginePin`-pinned), then asserts (C)
wrapper == shader path **BITWISE** and (D) == the case's explicit oracle.  Raw lines (vega, `intel_icd`/Arc B70;
the same cases pass on llvmpipe and the Ryzen iGPU):

| # | symbol | wrapper == shader path (bitwise), worst | wrapper vs oracle, worst |
|---|---|---|---|
| 1 | `native_qsa_rms_norm_weighted` | **20480/20480 + 512/512 + 3072/3072, w 0** | **w 2.2e-07 / 1.69e-07 / 2.38e-07** (the norm rule, double, tol 3e-3) |
| 2 | `native_rope_apply` | **49152/49152 + 24576/24576, w 0** | **w 1.6e-05 / 1.9e-05** (row-relative analytic rule, tol 3e-3; None + YaRN2) |
| 3 | `rope_neox_apply` | **768/768 + 1024/1024, w 0** | **w 0.106 / 0.088** (the NEOX rule, bit-exact tail) |
| 4 | `qsa_block_scores` | **72/72, w 0** | **72/72 w 5.56e-07** (relu per head + the dead key + 1e9) |
| 5 | `qsa_block_topk` | **128/128 x 3 arms, w 0** | **128/128 x 3 w 0** (the selection rule, ids exact) |
| 6 | `native_qsa_gate_apply` | **6144/6144 + 48/48, w 0** | **w 7.87e-07 / 7.3e-07** (the native gate rule, tol 1e-5) |
| 7 | `qsa_gate_apply_f32` | **6144/6144 + 48/48, w 0** | **w 7.75e-07 / 7.78e-07** (the gate rule, tol 1e-5) |
| 8 | `native_router_top10` | **20/20 + 80/80, w 0** | **w 1.86e-07 / 2.27e-07** (ids exact, weights tol 1e-5) |

**THE LATENT DEFECT - THE CLASS THE LAST BATCH NAMED (A PARAMETER USED AS SOMETHING IT DOES NOT MEAN), FOUND HERE
AND FIXED: the RoPE/norm WEIGHT was indexed by the ELEMENT index instead of the COLUMN index.**  The engine's
contract (and its CUDA) is per-column: `include/strata/kernels/native_qsa.hpp` says "gamma[n_cols] broadcast over
rows", and `native_qsa.cu`'s `norm` computes `output[col] = scale * input[col] * gamma[col]` with `input`/`output`
advanced by the row offset and gamma NOT advanced; the legacy `elementwise.cu` `rms_norm_weighted_kernel` is
`r[c] = (r[c] * w[c]) * inv`.  But BOTH port shaders read `gamma.v[i]` / `w.v[i]` with `i = row*cols + c` - the
ELEMENT index.  For row 0 the two coincide; for every later row it reads the wrong weight, and with a cols-long
weight (which is what the engine passes) it reads OUT OF BOUNDS.  **The defect was invisible because THREE cases'
oracles were written to reproduce the shader's element-wise indexing instead of the rule** - the "oracle built from
the thing under test" trap: `case_rms_norm`, `case_rms_norm_weighted_entry` and
`case_native_qsa_rms_norm_weighted` all fed an n-long weight and an element-wise oracle.  FIXED: `rms_norm.comp`
and `native_qsa_rms_norm_weighted.comp` now read `w.v[c]` / `gamma.v[c]`; the three oracles now use the column
index (the RNG draws are unchanged, so no later case's fixture moves); and `elementwise_vk.cpp`'s
`rms_norm_weighted` now range-checks `cols` floats (not `rows*cols`) and supplies `cols` ones - so a cols-long
engine weight is accepted rather than refused.  This defect would have produced a SILENTLY WRONG token on every
QSA layer (and every legacy `rms_norm_weighted` call) the moment the engine ran; the integration case caught it
because it compares against the ENGINE'S contract rather than the shader's habit.

**THE FIXTURE DEFECTS FOUND WHILE PROVING (both were the CASE's, both fixed, neither weakened):** (1) the entry
case's gamma view was one `cols` long, which exposed the out-of-bounds read above; (2) the router case's multi-token
arm formed a `weights + t*10` device pointer - a **40-byte** offset, which is NOT a multiple of llvmpipe's 16-byte
`minStorageBufferOffsetAlignment` (the Arc's is 4), so the lvp arm DIED on the bind.  The arm now uses per-token
arena buffers; the engine's own `logits + t*512` (2048 B) is aligned.  NOTE ON THE lvp ARM: it had been EXITING
(no totals) in the first two runs of this batch - that was this case's misaligned test offset, not a device
problem; it now completes.

**THE `host` ROW: `rope_scaling()`** - the rope CONSTANTS (`rope_scaling.hpp`).  On a CUDA build
`src/kernels/cuda/rope_scaling.cu` owns the storage; a Vulkan build compiles no such file, and `layer.cpp:881`
calls `rope_scaling()` at every analytic rope launch.  `qsa_vk.cpp` answers `rope_scaling()` and its setter
(`rope_scaling_set`), default-constructed to the identity (the engine's unscaled decode).  No capability flag was
turned on: `native_qsa_enabled()` and `native_rope_enabled()` were already TRUE (both symbols each gates already
had a shader), so `case_native_capabilities` needed no new arm and still passes.

**REPORTED, NOT WIRED - A SHADER-PORT JOB, NOT A STUB.**  `qsa_decode_attn_step` (`layer.cpp:980`, the DEFAULT
decode attention) sits at #6 in the derived order but has NO shader in this tree: PORT-MAP maps it to
`attn_decode_short`, whose own header says it IS `native_flash_attn_short_step` - a GATHERED `[capacity,2,256]` f16
WINDOW - while `qsa_decode_attn_step`'s contract (`qsa_decode_attn.hpp`) reads the KV POOLS through the page table
with an int8/q4 option and takes a scratch.  That is a different kernel, so `qsa_decode_attn_step` is a
SHADER-PORT job (the `s_gemv_q8_0_split` class), reported rather than stubbed; its sibling
`native_flash_attn_short_step` (:995) and the `todo` rows `qsa_attend_step` (:1002), `qsa_index_step` /
`topk_512_step` (:973) and `native_qsa_indexer_append` (:945) are the same class.  Also NOT in this increment: the
PLE and GR stages of `block_layer_pre` (`ple_block`, `ple_history_advance`, `gr_read`, `gr_write`, `fused_gr_*`) -
they run for EVERY layer, GDN or QSA, so they are not "the non-GDN body".

**THE REMAINING 35 `strata::kernels::` SYMBOLS, GROUPED.**

| subsystem | n | symbols |
|---|---:|---|
| **glue (the I2 elementwise set)** | **0** | all answered |
| **matvec / GEMV / KV** | **6** | `bf16_gemv_fp32_mmvf_cols`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve` |
| **attention / QSA / MoE / GR / PLE / rope** | **27** | `build_rope_table`, `rope_table_set`, `gr_read`, `gr_write`, `gr_workspace_init`, `fused_gr_read`, `fused_gr_supported`, `moe_combine`, `native_moe_combine`, `shared_expert(+_scratch_bytes)`, `ple_block(+_scratch_bytes)`, `ple_history_advance`, `ngram_rows`, `PleTable::{collect,is_open,issue}`, `qsa_attend_step`, `qsa_decode_attn_step`, `qsa_decode_attn_scratch_floats`, `qsa_index_step`, `qsa_step_fill`, `topk_512_step`, `native_qsa_indexer_append`, `native_flash_attn_short_step`, `router_top10` |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

(`rope_scaling` is ANSWERED by this batch (the one `host` row), so it is not among the 27.)

**THE REMAINING DISTANCE TO A LAYER THAT LINKS.**  The one-layer-body link (`src/core/layer.cpp` vs the backend +
the shim) carries **35 undefined `strata::kernels::` symbols**, ALL of them referenced by `layer.cpp` itself (the
link compiles only `layer.cpp`, so every one it names is on this link) - that is the integration's remaining work.
It also carries the FIVE engine cross-TU symbols (`main`, `strata::core::LayerView::name`,
`WeightTable::find`, `native_embed`, `NativeEmbed::gather_one`), which have homes in the targets the shim batch
added.  Beyond the symbols, the ONLY other structural blocker is the DEFERRED CUDA GRAPH API that I5 owns:
`include/strata/core/graph.hpp:81` needs `cudaGraph_t`/`cudaGraphExec_t` (which the shim, per the user's I5
deferral, does not declare), so `src/core/graph.cpp`, `src/core/session.cpp` and `src/program/generate.cpp` cannot
compile - which is what keeps `strata_vulkan_core` / `strata_vulkan_engine` / `strata_vulkan` from building.  The
one-layer body needs the 35 kernels ANDed; a whole PROGRAM additionally needs I5's graph declarations.  The graph
API was NOT ported.

**RESULTS (vega).**  Engine CONFIGURE + BUILD under `-DSTRATA_ENABLE_VULKAN=ON` clean.  Gate on `vega`:
**Arc (intel_icd) 603/0/0 (exit 0), llvmpipe 591/0/3, Ryzen iGPU (radeon_icd) 593/1/2** - **+34 verdicts on every
arm** (8 new cases: 3+2+2+2+3+2+2+2 arms = 17 arms x 2 verdicts = 34), 0 failed on the Arc.  ON THE RECORD: the
radeon arm's first run carried the documented intermittent `budget: independent requery agrees` flake and the
second carried the open RADV-only `bf16_gemv_split n_in=2560 n_out=512` defect; a third read **594/0/2**.  The lvp
arm had been EXITING without totals in the first two runs of this batch - the misaligned router-case offset above
- and now completes 591/0/3.  `check_port_map.py` passes (`168 - 78 kernel, 61 host, 29 todo; 111 shaders built,
92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (the eight were already `kernel`
rows and `rope_scaling` a `host` row, so the map does not move).  **`z820b` is PENDING** (no XTX/K620 number).  The
CUDA graph API was NOT ported and the plan was NOT re-scoped.

## INCREMENT I4 — THE NEXT EIGHT MATVEC / GEMV / KV ENTRY POINTS (THE BF16 GEMVs, THE S2 GEMV, THE FP16/INT8 KV CACHE) AND THE STANDARDISED LINK PROGRESS BAR (2026-10-05, `vega`)

**THE BAR (the running line): `118 → 106` undefined references / `53 → 44` distinct full-signature
`strata::kernels::` symbols / `51 → 42` under the parent's name-only pattern.**  The matvec/GEMV/KV group falls
**15 → 6**; every other group is unchanged.  MEASURED with the CURRENT STANDARD recipe (the shim is in the link
line; `$HOME/vkbuild-vulkan` is a **Makefiles** build dir, so **never pass `-G Ninja`** - CMake refuses and the
stale library silently reproduces the PREVIOUS batch's numbers):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release     # reconfigure
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 106  (was 118)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 44   (was 53)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 42   (was 51)

**THE EIGHT, IN THE ORDER THE LAYER BODY NAMES THEM** (source order in `src/core/layer.cpp`; the plan's §3
list is NOT an order).  All eight are reached by `layer.cpp` or a function it defines (`project_bf16`,
`gemv_quantized`, `qsa_layer`); their first NAMING line is the order:

| # | symbol | call site | stage | shader |
|---|---|---|---|---|
| 1 | `bf16_gemv_fp32_mmvf` | layer.cpp:97 | `project_bf16` native branch (`--native`) | bf16_mmvf_f32.spv |
| 2 | `bf16_gemv_split` | layer.cpp:98 | `project_bf16` split branch (gdn alpha/beta :291, router :367) | bf16_gemv.spv |
| 3 | `bf16_gemv` | layer.cpp:99 | `project_bf16` plain branch (qsa indexer k/q :918/:962) | bf16_gemv.spv |
| 4 | `s2_gemv_q8` | layer.cpp:166 | `gemv_quantized` S2 branch (code_bits == 2) | s2_gemv_q8.spv |
| 5 | `kv_append_q8_step` | layer.cpp:934 | qsa_layer - the INT8/K8V4 KV cache APPEND | kv_q8_append.spv |
| 6 | `kv_append_step` | layer.cpp:943 | qsa_layer - the FP16 KV cache APPEND | kv_f16_append.spv |
| 7 | `kv_gather_q8_step` | layer.cpp:983 | qsa_layer - the INT8/K8V4 KV cache GATHER | kv_q8_gather.spv |
| 8 | `kv_gather_step` | layer.cpp:989 | qsa_layer - the FP16 KV cache GATHER | kv_f16_gather.spv |

All eight live in `vulkan/src/kernels/matvec_vk.cpp` (the KV TU is that file - there is no separate KV file).

**EACH PROVED BY ITS CASE THROUGH THE ENGINE WRAPPER - BITWISE vs THE SHADER PATH AND vs THE EXPLICIT ORACLE.**
A new `case_*_entry` in the port's gate runs the port's EXISTING case's fixture through (A) the shader path and
(B) the ENGINE WRAPPER `strata::kernels::<symbol>` on its own engine stream (`EnginePin`-pinned), then asserts
(C) the wrapper equals the shader path **BITWISE** and (D) equals the case's explicit oracle.  Raw lines (vega,
`intel_icd`/Arc B70 arm; the same cases pass on llvmpipe and the Ryzen iGPU):

| # | symbol | wrapper == shader path (bitwise), worst | wrapper vs oracle, worst |
|---|---|---|---|
| 1 | `bf16_gemv_fp32_mmvf` | **256/256, w 0** | **65/65 w 0.00682** (terms bound + output guard) |
| 2 | `bf16_gemv_split` | **192/192, w 0** | **49/49 w 0.00612** (terms bound + output guard) |
| 3 | `bf16_gemv` | **512/512, w 0** | **129/129 w 0.00724** (terms bound + output guard) |
| 4 | `s2_gemv_q8` | **32/32, w 0** | **8/8 w 0** (the S2-over-Q8_0 rule, terms bound) |
| 5 | `kv_append_q8_step` | **16896/16896, w 0** | **16896/16896 w 0** (the KV-Q8 group rule, bit-exact; + a host-copy arm **33280/33280**) |
| 6 | `kv_append_step` | **262144/262144, w 0** | **262144/262144 w 0** (the append row rule, fp16 exact) |
| 7 | `kv_gather_q8_step` | **8448/8448, w 0** | **8448/8448 w 0** (the int8 reader rule, fp16 exact) |
| 8 | `kv_gather_step` | **8448/8448, w 0** | **8448/8448 w 0** (the fp16 row copy, exact) |

**THE `host` ROWS THIS BATCH ANSWERS.**  One, and it is not a bare bind: **`kv_block_bytes(s, fmt)`** - the
BYTES OF ONE BLOCK (page) of K and V together, the size the KV-streaming path sizes its pinned host copy with
(layer.cpp:659).  A pure function of the shapes and the storage format, transcribed from
`src/kernels/cuda/kv_stream.cu:192` so the two agree byte for byte.  **The other three KV-stream rows are NOT
answered, and the reason is measured**: `kv_stream_reset` (layer.cpp:707/738), `kv_ring_table` (:709) and
`kv_stream_resolve` (:757) are the STREAMING RESIDENT TIER (`--kv-resident`, `p.mode != 0`; the DEFAULT is mode 0,
fully resident) - the engine calls the first two with a **NULL stream** and this backend has no default stream to
fall back on, and the third needs the resolve/copy kernels whose shaders this tree does not build.

**CAPABILITIES: NONE TURNED ON, AND THAT IS WHY THE CAPS CASE IS UNTOUCHED.**  None of the eight is gated by a
`*_enabled()` capability (`bf16_gemv*`'s branch is the `native_bf16_projections` SETTING, not a getter), so
`case_native_capabilities` needed no new arm and its invariant stands unchanged - and it passed (569/0/0).

**A LATENT DEFECT IN I3'S Q4 KV APPEND, FIXED HERE.**  `kv_append_q4_step` dispatched ONCE binding the POOL with
`host_layout = host ? 1 : 0`.  But `host_layout` selects the ROW RULE, not whether to write: the CUDA writes the
VRAM page (page-table row) ALWAYS and the host copy (identity row) additionally when a host pool is present.  So
under KV streaming I3's wrapper wrote ONLY the host row and left the pool stale.  It is now TWO dispatches: the
pool always (host_layout 0), and - when `host->k_q4 != nullptr` - the host buffers with host_layout 1.  The same
shape is what the new `kv_append_q8_step` / `kv_append_step` implement, and the q8 append case PROVES it with a
table that maps block 0 to page 1 while the identity row is page 0 (so the two images differ): the host-copy arm
reads 33280/33280.

**THE REMAINING 44 `strata::kernels::` SYMBOLS, GROUPED.**

| subsystem | n | symbols |
|---|---:|---|
| **glue (the I2 elementwise set)** | **0** | all answered |
| **matvec / GEMV / KV** | **6** | `bf16_gemv_fp32_mmvf_cols`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve` |
| **attention / QSA / MoE / GR / PLE / rope** | **36** | unchanged |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

**WHAT IS LEFT OF THE MATVEC / GEMV / KV GROUP, AND WHY EACH OF THE SIX STAYS.**  `bf16_gemv_fp32_mmvf_cols`
(layer.cpp:414) is only inside `moe_route_window`, which `verify.cpp:927` calls - the P6 verifier, not the decode
path.  `s_gemv_q8_0_split` and `s_gemv_q8k_split` are PORT-MAP `todo` rows ("no shader in this tree yet"), and a
wrapper cannot be proved against a case that does not exist.  The three `kv_stream_*`/`kv_ring_table` rows are the
streaming tier above.

**THE ENGINE'S CROSS-TU SYMBOLS - ALL FIVE NOW HAVE A HOME; NONE IS MISSING FROM A SOURCE LIST.**  The five the
shim batch listed (`LayerView::name`, `WeightTable::find`, `native_embed`, `NativeEmbed::gather_one`, `main`) are
named in the targets the shim batch added, so no further one-liner is owed:

| symbol | file | target that names it |
|---|---|---|
| `LayerView::name` | `src/core/layout.cpp:137` | `strata_vulkan_core` |
| `WeightTable::find` | `src/core/weights.cpp:474` | `strata_vulkan_core` |
| `native_embed` | `src/core/native_head.cpp:101` | `strata_vulkan_engine` |
| `NativeEmbed::gather_one` | `src/core/native_head.cpp:194` | `strata_vulkan_engine` |
| `main` | `src/program/generate.cpp:1228` | `strata_vulkan` (the executable) |

**Each of the three underlying TUs COMPILES clean under the shim** (`layout.cpp`, `weights.cpp`,
`native_head.cpp`): linking those three objects into the one-layer body drops its raw undefined references
**106 → 66** and resolves `LayerView::name`, `WeightTable::find`, `native_embed` and `gather_one` (the one
remaining line is inside `gather_one`'s own body, not the symbol).  The reason the whole targets do not build is
NOT a source-list gap: each target also carries a sibling TU that needs the DEFERRED CUDA graph API -
`strata_vulkan_core` carries `src/core/graph.cpp:81` (`cudaGraph_t`/`cudaGraphExec_t`), `strata_vulkan_engine`
carries `src/core/session.cpp` (same), `strata_vulkan` carries `src/program/generate.cpp` (which also needs
`cudaLaunchHostFunc`).  That declaration work is **I5's** and was not touched.

**RESULTS (vega).**  Engine CONFIGURE + BUILD under `-DSTRATA_ENABLE_VULKAN=ON` clean.  Gate on `vega`:
**Arc (intel_icd) 569/0/0 (exit 0), llvmpipe 557/0/3, Ryzen iGPU (radeon_icd) 560/0/2** - **+17 verdicts on every
arm** (8 new cases: `bf16_gemv_fp32_mmvf` 2, `bf16_gemv_split` 2, `bf16_gemv` 2, `s2_gemv_q8` 2,
`kv_append_q8_step` 3, `kv_append_step` 2, `kv_gather_q8_step` 2, `kv_gather_step` 2), 0 failed; the radeon arm's
characterised intermittent defect did NOT fire this run.  `check_port_map.py` passes (`168 - 78 kernel, 61 host,
29 todo; 111 shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (all
twelve were already `kernel`/`host` rows, so the map does not move).  **`z820b` is PENDING** (no XTX/K620 number).
The CUDA graph API was NOT ported; the plan was NOT re-scoped.

## THE CUDA-RUNTIME SHIM + THE ENGINE'S OWN TARGETS (the approved hybrid route, deliverables A+B) — AND HOW FAR ONE LAYER GETS TOWARD RUNNING (2026-10-05, `vega`)

**THE APPROVED ROUTE, IMPLEMENTED.** The user chose *"Hybrid: shim the ~13 symbols needed to link and run one
layer now, design the graph mapping later at I5"*.  This increment delivers the shim (A), the build-system
change (B), and takes one layer body as far toward RUNNING as the still-unported kernels allow (C).  The CUDA
graph API and stream capture stay DEFERRED (D).

**DELIVERABLE A — THE SHIM.** `vulkan/include/cuda_compat/cuda_runtime.h` (put FIRST on the include path, the
`hip_compat/` mechanism) + `vulkan/src/compat/cuda_runtime.cpp`, a `cuda_runtime.h`-compatible surface over the
port's device layer.  The engine's headers and sources are NOT edited.  The FINAL symbol list is the predicted
fourteen — `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpy2DAsync`, `cudaMemsetAsync`
(+`cudaMemset`), `cudaHostAlloc`, `cudaHostGetDevicePointer`, `cudaFreeHost`, `cudaEventCreate`
(+`cudaEventCreateWithFlags`), `cudaEventRecord`, `cudaEventElapsedTime` (+`cudaEventSynchronize`/`Destroy`),
`cudaDeviceSynchronize`, `cudaPeekAtLastError` — plus `cudaGetLastError`, `cudaGetErrorString`, `cudaMemGetInfo`,
`cudaStreamSynchronize` and the host-compile decorations (`__host__`/`__device__`/`__forceinline__`), which the
compiler demanded.  NOT in the predicted fourteen and ADDED because the linker/compiler demanded them: none of
the nineteen *functions* are referenced by the one-layer-body link beyond the twelve below, so `cudaMalloc`,
`cudaFree`, `cudaMemset`, `cudaGetErrorString`, `cudaMemGetInfo`, `cudaStreamSynchronize` and the two extra event
forms are implemented for the WIDER program but are NOT on this link's path.  **The one-layer-body link's actual
runtime demand is TWELVE symbols** (`cudaDeviceSynchronize`, `cudaEventCreate`, `cudaEventElapsedTime`,
`cudaEventRecord`, `cudaFreeHost`, `cudaHostAlloc`, `cudaHostGetDevicePointer`, `cudaMemcpy`, `cudaMemcpy2DAsync`,
`cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaPeekAtLastError`) — `cudaMalloc`/`cudaFree` of the predicted fourteen
are NOT reached by `layer.cpp`.

**WHERE CUDA COULD NOT BE HONOURED EXACTLY, AND WHAT THE NUMBER THEREFORE CANNOT MEAN** (all stated in the header
too): `cudaEventElapsedTime` is a **HOST wall-clock** figure (`std::chrono`) — this device layer enables NO
Vulkan timestamp query, so it CANNOT mean device execution time; `cudaDeviceSynchronize` is the submission fence
and is VACUOUS here (every `Ctx::dispatch` already fences and waits), so it does not wait for async work because
this backend has none; `cudaFree` releases NOTHING (the arena is a bump allocator) — memory returns only at
`stream_close`, so an alloc/free loop exhausts the arena; a mapped host pointer is **NOT device-dereferenceable**
(the arena is DEVICE_LOCAL/unmappable, so `cudaHostGetDevicePointer` returns a host pointer the SHIM resolves;
the engine's zero-copy handshake is answered by `src/device/sync.*`, not by that pointer); a device->device
`cudaMemcpy` is STAGED through the host (two transfers, not a device-side copy); `cudaMemsetAsync` is a staged
fill (no memset shader).  `cudaMemGetInfo` is HONEST (the device's own `VK_EXT_memory_budget` figure minus the
reserve); the error enum and `cudaGetErrorString` are real, and `cudaPeekAtLastError`/`cudaGetLastError` carry the
shim's OWN last error, not a constant.

**DELIVERABLE B — THE BUILD-SYSTEM CHANGE.** `vulkan/CMakeLists.txt` now defines the ENGINE'S OWN targets,
mirroring `../sycl/CMakeLists.txt`'s `strata_resolve` shape through `strata_vulkan_resolve()`:
`strata_vulkan_core`, `strata_vulkan_engine`, `strata_vulkan_spec`, `strata_vulkan_kernels_cpu` and the
`strata_vulkan` EXECUTABLE — so a `-DSTRATA_ENABLE_VULKAN=ON` configure has an engine binary and not just the
backend library.  The `STRATA_VERSION` ordering defect is fixed by moving
`add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")` ABOVE the backend-option blocks in the top-level
`CMakeLists.txt` (each of them `return()`s before the old line, so the define never reached the Vulkan/SYCL
configuration — the exact reason `src/program/generate.cpp` failed to compile).  The mutual-exclusion
`FATAL_ERROR` behaviour is unchanged.  **EXCLUDED to configure** (reported as asked): the three `src/core/*.cu`
(`device.cu`, `pinned.cu`, `remote_expert_opt.cu`), the whole PREFILL path (`src/prefill/*`), the parity/bench
TUs (`src/kernels/*_parity.cpp`, `bench/`), and the native-expert/ggml-cpu half of `strata_kernels_cpu`
(`native_expert.cpp`, `iq_avx*.cpp`).  **WALL TIMES (vega):** CONFIGURE **0.23 s** (was 0.16 s; the engine
targets added ~0.07 s of CMake graph), BUILD of `strata_vk_cudart_smoke` + `strata_vk_entry_smoke` **3.45 s**.

**DELIVERABLE C — THE LINK BAR AND HOW FAR THE LAYER GOT.** The bar is measured with the STANDARD recipe
(`$HOME/vkbuild-vulkan` reconfigured and rebuilt from the current tree first) and the shim in the link line:

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels strata_vulkan_cudart -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include/cuda_compat -Ivulkan/include -Ivulkan/src/device \
        -DSTRATA_ENABLE_VULKAN=1 -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_cudart.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                       # -> 118  (was 170)
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                                # -> 53   (unchanged)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                              # -> 51   (unchanged)

**THE BAR MOVED `170 -> 118` undefined references; the 53 distinct `strata::kernels::` symbols are UNCHANGED and
the 12 CUDA-runtime symbols are ALL RESOLVED (0 left).**  Two readings are recorded because the port's total is a
`grep -c`, and GNU ld BATCHES repeated undefined references ("...: more undefined references to `X' follow"), so
the raw-reference total is not a pure linear count: at this HEAD the twelve CUDA symbols carry **46 raw reference
mentions** (45 `undefined reference to` lines + one batched `more` line for `cudaMemsetAsync`), and resolving
them also re-batched `LayerView::name`'s messages (39 -> 32 prefix lines) — hence 170 -> 118 is **-52 raw lines
for -12 distinct symbols**.  The task's quoted bar (`188`, the I2e HEAD) is pre-I3: the shim removes the same
twelve; on I3's HEAD (`6747bd5`) the measured before/after is **170 -> 118**.

**HOW FAR THE LAYER GOT TOWARD RUNNING.** The shim RUNS: `strata_vk_cudart_smoke` executes every entry point on
the Arc and PASSES (raw line `strata_vk_cudart_smoke: PASS`, exit 0) — allocate, host->device->host bit-exact,
mapped-host upload, `cudaMemcpy2DAsync` de-pitch, a staged `cudaMemsetAsync`, event timing, `cudaMemGetInfo`
(28.64 GiB free / 31.89 GiB total), and the real error path.  A FULL SINGLE-LAYER FORWARD DOES NOT RUN, and the
blocker is precise: the one-layer-body link still carries **53 unresolved `strata::kernels::` entry points**
(matvec/GEMV/KV 15 + attention/QSA/MoE/GR/PLE/rope 36 + other 2) plus the engine's own cross-TU symbols
(`LayerView::name`, `WeightTable::find`, `NativeEmbed::gather_one`, `native_embed`, `main`).  A layer cannot run
until those entry points are ported (the plan's I4/I5).  What the shim removes is the ENTIRE
runtime-not-kernels blocker: with it in the link line there are ZERO `cuda*` undefined references left.

**DELIVERABLE D — THE DEFERRED GRAPH CALLS ON THE LAYER'S CALL PATH (for I5's design).** `src/core/layer.cpp`,
the layer BODY itself, calls **NONE** of `cudaGraph*`/`cudaStreamBeginCapture`/`EndCapture` — the layer is
capture-clean.  The graph calls live one level up, on the DRIVER/STEP-RECORDER path: `src/core/graph.cpp`
(`cudaStreamBeginCapture:49`, `cudaStreamEndCapture:55`, `cudaGraphInstantiate:65`, `cudaGraphLaunch:77`,
`cudaGraphGetNodes:59`, `cudaGraphDestroy:41`, `cudaGraphExecDestroy:42`), `src/core/session.cpp`
(`cudaStreamBeginCapture:228`, `cudaStreamEndCapture:245`, `cudaGraphInstantiate:252`, `cudaGraphLaunch:284`,
`cudaGraphDestroy:256`, `cudaGraphExecDestroy:489`) and `src/program/generate.cpp` (`cudaLaunchHostFunc:5165`).
So I5's graph mapping is a STEP-RECORDER problem (`graph.cpp`/`session.cpp`), not a layer-body one — the layer
can be driven by the port's already-built recorded-step mechanism (`Ctx::record_begin/record_dispatch/replay`).

**RESULTS (vega).**  Engine CONFIGURE under `-DSTRATA_ENABLE_VULKAN=ON` is CLEAN and now yields the engine
targets; the backend + the shim + both smoke targets BUILD clean; `strata_vulkan_spec` and
`strata_vulkan_kernels_cpu` COMPILE + LINK.  `strata_vulkan_core` and `strata_vulkan_engine` CONFIGURE but their
COMPILE stops at the DEFERRED CUDA graph API: `include/strata/core/graph.hpp:81` needs `cudaGraph_t` /
`cudaGraphExec_t` (which the shim, per the user's I5 deferral, deliberately does NOT declare), so `graph.cpp`
and `session.cpp` do not compile under the shim.  That is the precise declaration work I5 must add - it is NOT a
defect of the build-system change (the targets exist and configure).  The shim smoke PASSES on the Arc.  Port gate on `vega`: **Arc (intel_icd) 552/0/0 (exit 0), llvmpipe 540/0/3** — IDENTICAL to the I3
baseline (no shader, kernel or case changed).  The radeon iGPU arm read **542/1/2**; the one failure is the
**pre-existing** characterised `bf16_gemv n_in=2560 n_out=128 130/131 worst 2.36` defect (the non-deterministic
RADV bug at the top of this file), NOT this increment — no shader or case was touched.  `check_port_map.py`
passes (`168 — 78 kernel, 61 host, 29 todo; 111 shaders built, 92 claimed`) and `make_port_map.py` regenerates
`PORT-MAP.tsv` byte-identically.  `z820b` is PENDING (no XTX/K620 number).  The plan was NOT re-scoped.

## INCREMENT I3 — THE FIRST EIGHT MATVEC / GEMV / KV ENTRY POINTS (the weight-side math + the KV cache) AND THE STANDARDISED LINK PROGRESS BAR (2026-10-05, `vega`)

**THE BAR (the running line): `188 → 170` undefined references / `59 → 53` distinct full-signature
`strata::kernels::` symbols / `57 → 51` under the parent's name-only pattern.**  The matvec/GEMV/KV group falls
**21 → 15**; every other group is unchanged.  MEASURED with the STANDARD recipe - ONE build directory,
**`$HOME/vkbuild-vulkan`**, RECONFIGURED AND REBUILT FROM THE CURRENT TREE FIRST (no stale library; the hazard
I2e recorded):

    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels -j"$(nproc)"
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include -Ivulkan/src/device -DSTRATA_ENABLE_VULKAN=1 \
        -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # -> 170
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 53 (full signature)
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 51 (name only)

**THE EIGHT, IN THE ORDER `src/core/layer.cpp` AND ITS SIBLINGS REACH THEM** (the plan's I3 list is NOT an
order).  Six are reached by `layer.cpp` itself, two by a sibling file:

| # | symbol | call site | stage | shader |
|---|---|---|---|---|
| 1 | `quantize_q8_K` | layer.cpp:236 | gdn_layer stage 1 - the K-quant activation image | quantize_q8_K.spv |
| 2 | `quantize_q8_0` | layer.cpp:237 | gdn_layer stage 1 - the Q8_0 activation image | quantize_q8_0.spv |
| 3 | `native_quantize_q8_1` | layer.cpp:150 | gdn_layer stage 2 - the native qkv projection's activation | quantize_q8_1.spv |
| 4 | `native_mmvq` | layer.cpp:151 | gdn_layer stage 2 - the native projection (COMPOSITE, six shaders) | iq1m/iq2s/iq3s/iq3xxs/iq4nl/iq4xs_mmvq.spv |
| 5 | `kv_append_q4_step` | layer.cpp:935 | qsa_layer - the Q4_0 KV cache APPEND | kv_q4_append.spv |
| 6 | `kv_gather_q4_step` | layer.cpp:985 | qsa_layer - the Q4_0 KV cache GATHER | kv_q4_gather.spv |
| 7 | `quantize_q8_0_scaled` | session.cpp:868 (also expert_source.cpp:2327) | the MoE routed-expert activation | quantize_q8_0_scaled.spv |
| 8 | `native_q5_k_f32` | native_head.cpp:78 | the head's native Q5_K matvec (the native sibling of #4) | quantize_q8_1.spv + native_q5_k_f32.spv |

The increment's other two are DEFERRED, and that is a reachability statement rather than a choice:
**`quantize_q8_1_rows`** is the PEER-expert pool's activation (`peer_experts.cpp:230`, `remote_experts.cpp:307`
- the peer tier, not the single-GPU path) and **`s_gemv_split_async`** appears only in the standalone driver
mains `overlap_main.cpp:118/142` and `concurrent_main.cpp:141`, never in `src/core/`.  Both have shaders and
map rows; they are simply later in reach order than these eight.

**EACH PROVED BY ITS CASE THROUGH THE ENGINE WRAPPER - BITWISE vs THE SHADER PATH AND vs THE EXPLICIT ORACLE.**
A new `case_*_entry` in the port's gate runs the port's EXISTING case's fixture through (A) the shader path and
(B) the ENGINE WRAPPER `strata::kernels::<symbol>` on its own engine stream (`EnginePin`-pinned to the harness
device), then asserts (C) the wrapper's answer equals the shader path's **BITWISE**, and (D) equals the case's
explicit oracle.  Raw lines (vega, `intel_icd`/Arc B70 arm):

| # | symbol | wrapper == shader path (bitwise), worst | wrapper vs oracle, worst |
|---|---|---|---|
| 1 | `quantize_q8_K` | **876/876, w 0** | **876/876 w 0** (the Q8_K rule; one of the two scale forms) |
| 2 | `quantize_q8_0` | **102/102, w 0** | **102/102 w 0** (ggml's Q8_0 image) |
| 3 | `native_quantize_q8_1` | **1152/1152, w 0** | **1152/1152 w 0** (one of the two division forms) |
| 4 | `native_mmvq` (IQ2_S) | **4/4, w 0** | **4/4 w 0.0435** (the IQ2_S dot, terms bound) |
| 5 | `kv_append_q4_step` | **147456/147456, w 0** | **272/272 w 0** (the Q4_0 group rule, bit-exact) |
| 6 | `kv_gather_q4_step` | **8192/8192, w 0** | **4096/4096 w 0** (the Q4_0 reader rule, fp16 exact) |
| 7 | `quantize_q8_0_scaled` | **114/114, w 0** | **114/114 w 0** (the CPU Q8_0 rule + fp32 scales) |
| 8 | `native_q5_k_f32` | **3/3, w 0** | **3/3 w 0.00796** (the packed Q5_K dot, terms bound) |

**THE `host` ROWS THIS TU ANSWERS** (not a bare bind, and each with a measured reason):

* **`iq_row_bytes(type, n)`** - the per-format ROW STRIDE the six `*_mmvq` shaders take as `row_bytes`, and the
  weight-buffer size the wrapper range-checks.  A quantisation CONSTANT table (Q4_0/Q5_0/Q5_1/Q8_0 18/22/22/34 B
  per 32; Q2_0 18 B per 64; IQ4_NL 18 B per 32; IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/Q3_K/Q4_K/Q5_K/Q6_K/
  IQ4_XS 56/66/74/82/98/110/110/144/176/210/136 B per 256).  A type with no layout is a loud refusal, not a
  guessed stride.
* **`native_mmvq_supported(type)`** - THE CAPABILITY CHECK THAT GATES THE COMPOSITE, and the one this batch had
  to keep honest: `native_mmvq` dispatches by ggml type and the port ships shaders for exactly SIX types, so the
  check answers TRUE for those six and FALSE for every other type.  Answering TRUE for a type with no shader
  would route the engine (`native_dense.cpp:53/166`, `native_head.cpp:34`) at an unported kernel.  There is NO
  `*_enabled()` flag in this family to turn on (the six `native_caps_vk.cpp` getters are unrelated), so the caps
  case needed no new arm - but the per-type answer is the same "capability == every gated symbol has a shader"
  rule, and `case_native_mmvq_entry` exercises its dispatch.
* **`native_mmvq_weight_bytes(type, n_in, n_out)`** and **`native_q8_1_bytes(n_in, ncols)`** - the weight and
  q8_1-activation sizes the wrappers range-check against the arena (`native_q5_k_f32` sizes its internal
  quantise scratch with the latter).
* **The IQ grid tables** (`native_mmvq`'s four grid-taking arms) live in `Stream::iq_grids`
  (`vulkan/src/device/vk_arena.hpp`), placed LAZILY from `vulkan/src/kernels/iq_grids_vk.hpp` - a verbatim copy
  of the port's generated harness header, adopted exactly as I1 adopted `vk_compute.*`.  A per-dispatch upload
  would EXHAUST the arena (`arena_alloc` never decreases), so it is a stream-lifetime cache, the cvec_apply
  precedent.

**THE BAR MOVE, HONESTLY ATTRIBUTED.**  Of the eight, SIX move the one-layer-body link (the six `layer.cpp`
reaches); `quantize_q8_0_scaled` (session.cpp) and `native_q5_k_f32` (native_head.cpp) do NOT - they are reached
by sibling TUs, exactly as I2c's `add_inplace`/`gather_rows`/`scatter_rows_f32`/`f32_to_f16_bulk` did not move
it.  The raw-reference drop is larger than six because `quantize_q8_K` has 6 references and `quantize_q8_0` 4
from `layer.cpp` alone.

**THE REMAINING 53 `strata::kernels::` SYMBOLS, GROUPED.**

| subsystem | n | symbols |
|---|---:|---|
| **glue (the I2 elementwise set)** | **0** | all answered |
| **matvec / GEMV / KV** | **15** | `bf16_gemv`, `bf16_gemv_split`, `bf16_gemv_fp32_mmvf`, `bf16_gemv_fp32_mmvf_cols`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `s2_gemv_q8`, `kv_append_step`, `kv_append_q8_step`, `kv_gather_step`, `kv_gather_q8_step`, `kv_block_bytes`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve` |
| **attention / QSA / MoE / GR / PLE / rope** | **36** | unchanged |
| **GDN / DeltaNet mixer** | **0** | COMPLETE |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

(The 170 total also carries the 12 CUDA-runtime symbols and the engine's cross-TU `strata::core`/`main`
references - the CUDA-runtime surface is the un-approved RE-SCOPE and this batch did NOT touch it.)

**TWO CASE-CONSTRUCTION BUGS FOUND AND FIXED WHILE PROVING (recorded, because each was caught by the case and
not hidden):** (1) the `quantize_q8_K` case's guard arm indexed the WRAPPER's buffer past its allocation (an
out-of-bounds read that showed as `875/876` on the bitwise arm while the oracle arm passed) - the buffer is now
allocated `blk + slack` and both guards are compared.  (2) `native_q5_k_f32`'s case first fed the shader path an
INDEPENDENTLY quantised activation, which fails on RADV because the q8_1 quantiser's division form is chosen per
pipeline instance (the same fact `case_quantize_q8_1` records) - the shader path is now fed the WRAPPER's OWN
activation bytes, so the arm is a genuine dot-kernel comparison and the quantiser half is gated by
`case_native_quantize_q8_1_entry`.  Neither case was weakened; both were corrected.

**RESULTS (vega).**  Engine CONFIGURE + BUILD under `-DSTRATA_ENABLE_VULKAN=ON` clean.  Gate on `vega`:
**Arc (intel_icd) 552/0/0 (exit 0), llvmpipe 540/0/3, Ryzen iGPU (radeon_icd) 543/0/2** - **+16 verdicts on every
arm** (8 new cases x 2 verdicts), 0 failed.  ON THE RECORD: the radeon arm's FIRST run of this batch read
**540/3/2** - the documented intermittent `budget: independent requery agrees` flake PLUS this batch's own
`native_q5_k_f32` case bug (before fix 2 above); after the fix a re-run read **543/0/2**.  `strata_vk_entry_smoke`
builds and runs on the Arc.  `check_port_map.py` passes (`168 - 78 kernel, 61 host, 29 todo; 111 shaders built,
92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (the eight were already `kernel`
rows, so the map does not move).  **`z820b` is PENDING** (no XTX/K620 number).  The CUDA-runtime host surface was
NOT touched, and the plan was NOT re-scoped.

**WHAT IS LEFT OF I3.**  The matvec/GEMV/KV group's remaining 15: the deferred pair above plus the `bf16_gemv`
family, the two `s_gemv_*_split` siblings, `s2_gemv_q8`, the two other `kv_append_*`/`kv_gather_*` siblings and
the four KV host rows (`kv_block_bytes`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve`).

## INCREMENT I2e — THE REMAINING EIGHT GDN / DELTANET MIXER ENTRY POINTS, **COMPLETING THE MIXER**, AND THE STANDARDISED LINK PROGRESS BAR (2026-10-05, `vega`)

**THE MIXER IS COMPLETE: all FOURTEEN entry points `gdn_layer` reaches are wired and proved** (I2d's six + this
batch's eight), in `vulkan/src/kernels/gdn_vk.cpp`.  The mixer runs on 36 of the model's 48 layers
(`gdn_layer`, `src/core/layer.cpp:223`), so this closes the GDN / DeltaNet subsystem: the grouped table's
**GDN mixer group goes 8 → 0**.

**THE EIGHT, IN THE ORDER `gdn_layer`'s OWN BODY REACHES THEM.**  Read from the call sites, NOT from the plan's
list - and within a stage the branches are alternatives, so "the order the body reaches them" is the order the
calls appear: **fused, then native, then legacy**.  The beta/gate stage (layer.cpp:285-302) then the recurrence
(:304-313) then the z-gate/norm stage (:318-327):

| # | symbol | call site | stage | shader |
|---|---|---|---|---|
| 7 | `native_gdn_beta_gate` | layer.cpp:296 | beta/gate, native | native_gdn_beta_gate.spv |
| 8 | `native_gdn_gate` | layer.cpp:297 | beta/gate, native | native_gdn_gate.spv |
| 9 | `gdn_beta_gate` | layer.cpp:299 | beta/gate, legacy | gdn_beta_gate.spv |
| 10 | `native_gdn_step` | layer.cpp:308 | recurrence, native | native_gdn_step.spv |
| 11 | `gdn_step` | layer.cpp:309 | recurrence, legacy | gdn_step.spv |
| 12 | `fused_gdn_step_norm` | layer.cpp:322 | z-gate/norm, fused | fused_gdn_step_norm.spv |
| 13 | `native_gdn_out_norm` | layer.cpp:324 | z-gate/norm, native | native_gdn_out_norm.spv |
| 14 | `gdn_out_norm` | layer.cpp:325 | z-gate/norm, legacy | gdn_out_norm.spv |

(The beta/gate stage's SEVENTH symbol, `gdn_gate` at :300, was already answered in `elementwise_vk.cpp` with the
other glue kernels, so it is not part of these eight and is not repeated here.)

**EACH PROVED BY ITS CASE THROUGH THE ENGINE WRAPPER - BITWISE vs THE SHADER PATH AND vs THE EXPLICIT ORACLE.**  A
new `case_*_entry` in the port's gate (`harness/vk_gate.cpp`) runs the port's EXISTING case's fixture through (A)
the shader path and (B) the ENGINE WRAPPER `strata::kernels::<symbol>` on its own engine stream (`EnginePin`
pinned to the harness device), then asserts (C) the wrapper's answer equals the shader path's **BITWISE**, and (D)
equals the case's explicit oracle.  The step/norm cases compare the mutated **STATE** bitwise as well.  Raw lines
(vega, default/Arc arm; the same eight pass on llvmpipe and the Ryzen iGPU):

| # | symbol | wrapper == shader (bitwise), worst | wrapper vs oracle, worst |
|---|---|---|---|
| 7 | `native_gdn_beta_gate` | **48/48, w 0** | **48/48 w 1.42e-06** (double sigmoid, tol 2e-6) |
| 8 | `native_gdn_gate` | **48/48 + 5/5 + 300/300, w 0** | **48/48 w 2.13e-07; 5/5 w 6.54e-08; 300/300 w 3.83e-07** (double softplus, tol 5e-6) |
| 9 | `gdn_beta_gate` | **48/48, w 0** | **48/48 w 1.42e-06** (double sigmoid, tol 2e-6) |
| 10 | `native_gdn_step` | **792576/792576 (+state), w 0** | **792576/792576 w 2.15e-03** (native rule, tol 2e-4 rel + 1e-5 abs) |
| 11 | `gdn_step` | **792576/792576 (+state), w 0** | **792576/792576 w 4.47e-03** (legacy rule, same tol) |
| 12 | `fused_gdn_step_norm` | **792576/792576 (+state), w 0** | **792576/792576 w 0.13** (fused rule, same tol) |
| 13 | `native_gdn_out_norm` | **6144/6144, w 0** | **6400/6400 w 5.89e-07** (+ row-guard tail untouched) |
| 14 | `gdn_out_norm` | **6144/6144, w 0** | **6400/6400 w 8.08e-07** (+ row-guard tail untouched) |

(The oracle is a double transcription of the engine's OWN body, as every GDN case already uses:
`native_gdn_preprocess.cu`'s `beta_sigmoid`/`gate_softplus`/`out_norm`, `native_gdn.cu`'s `step`, `gdn.cu`'s
`gdn_step`/`out_norm`, `fused_gdn.cu`'s `gdn_step_norm_kernel`.  `native_gdn_step`'s oracle is the NATIVE rule -
the decay folded into the rank-1 update and the `1/sqrt(S)` readout scale FUSED; `gdn_step`'s is the LEGACY rule -
decay-first, no readout scale; `fused_gdn_step_norm`'s is the fused rule - decay folded, scale fused, the closing
RMS norm with `sigmoid(z)`.  The wrappers compute the folded scale (`native_gdn_step`) and fix `S = 128`
(`fused_gdn_step_norm`, the fused operator's own constant); those are the only arithmetic they add.)

**NO `host` ROW WAS NEEDED, AND THAT IS MEASURED RATHER THAN ASSERTED.**  The only `host` row the mixer reaches is
`native_gdn_enabled()` (layer.cpp:247/253/265/295/306/324), already answered TRUE by
`vulkan/src/kernels/native_caps_vk.cpp`.  The GDN headers carry **no `*_scratch_bytes` and no shape accessor**: the
one shared shape, `GdnShapes` (by value, `include/strata/kernels/gdn.hpp`), is a POD the wrapper passes through -
a TYPE, not an undefined symbol.  Verified by the link measurement below: wiring the eight drops the one-layer-body
link by **exactly eight** symbols and introduces **no new undefined reference**.

**NONE OF THE EIGHT IS A MODULE-STATE READER LIKE `cvec_apply`.**  Each is a bare bind-and-dispatch over an
already-gated shader.  What they carry is the CUDA wrapper's argument CONTRACT, kept as the port's loud refusal
(never a silently wrong binding): `native_gdn_step` refuses `S != 128` and `h_v % h_k != 0`; `gdn_step` and
`fused_gdn_step_norm` refuse `h_v % h_k != 0`; `native_gdn_out_norm` refuses `cols != 128` (gamma is 128 floats);
`fused_gdn_step_norm` fixes `S = 128` as the fused body does.

**RESULTS (vega).**  Engine CONFIGURE + BUILD under `-DSTRATA_ENABLE_VULKAN=ON` clean.  Gate on `vega`:
**Arc (intel_icd) 536/0/0 (exit 0), llvmpipe 524/0/3, Ryzen iGPU (radeon_icd) 527/0/2** - **+20 verdicts on every
arm** (8 new cases: 6 emit 2 verdicts, `native_gdn_gate` emits 6, the two out_norm cases 2 each), 0 failed.  The
radeon arm's first three attempts showed the two PRE-EXISTING characterised defects (the `budget: independent
requery` flake and `bf16_gemv_split`/`fused_gdn_ab` - none of the eight new cases); a re-run read **527/0/2**,
0 failed, the flake cleared.  `strata_vk_entry_smoke` builds and runs on the Arc.  `check_port_map.py` passes
(`168 - 78 kernel, 61 host, 29 todo; 111 shaders built, 92 claimed`) and `make_port_map.py` regenerates
`PORT-MAP.tsv` byte-identically (the eight were already `kernel` rows, so the map does not move).  **`z820b` is
PENDING** (no XTX/K620 number).  The CUDA-runtime host surface was NOT touched, and the plan was NOT re-scoped.

**THE LINK PROGRESS BAR, STANDARDISED - ONE BUILD DIRECTORY, REBUILT BEFORE MEASURING.**  The bar silently read
STALE numbers because the recipe named a `<build>` placeholder and the parent picked the wrong `/tmp/vkbuild-*`
directory: it reproduced I2c's `204/73` from a library built BEFORE I2d, where I2d's real bar was `196/67`.
Unchanged progress is a PLAUSIBLE reading, so a stale bar is worse than no bar.  THE STANDARD RECIPE names ONE
stable absolute directory, **`$HOME/vkbuild-vulkan`**, and **it must be reconfigured and rebuilt from the current
tree before any number is read** (the block below does exactly that; there is no placeholder):

    # THE BUILD.  One stable directory.  RECONFIGURE + REBUILD from the CURRENT tree FIRST - a stale
    # libstrata_vulkan_kernels.a silently reports the PREVIOUS batch's bar (that is the hazard this fixes).
    cmake -S . -B "$HOME/vkbuild-vulkan" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME/vkbuild-vulkan" --target strata_vulkan_kernels -j"$(nproc)"

    # THE BAR.  Compile the one layer body, link it against the FRESH library, count.
    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include -Ivulkan/src/device -DSTRATA_ENABLE_VULKAN=1 \
        -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a" \
        "$HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a" -lvulkan -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                      # total references -> 188
    # YOUR pattern (the tree's): the FULL demangled signature, distinct.
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                               # -> 59
    # THE PARENT's simpler NAME-ONLY pattern: the name up to the first '('.  It drops the two signatures carrying
    # a nested `strata::kernels::GdnShapes` (`gdn_step`/`native_gdn_step`), so it reads one or two LOWER - the
    # same-regex DELTA is the signal, not the absolute value.
    grep -oP "undefined reference to \`\Kstrata::kernels::[A-Za-z_0-9]+" /tmp/link.log \
        | sort -u | wc -l                                                             # -> 57

**THE BAR MOVE THIS BATCH OWNS ALL EIGHT OF.**  `layer.cpp` reaches exactly these eight kernels, so each one it
answers removes its own reference(s) from THIS link:

| measurement | before (e12072d, I2d) | after (this batch, I2e) | delta |
|---|---:|---:|---:|
| `grep -c "undefined reference"` (the link's total) | **196** | **188** | **-8** |
| YOUR pattern: full-signature distinct `strata::kernels::` | **67** | **59** | **-8** |
| PARENT's name-only pattern: name-up-to-`(` distinct | **65** | **57** | **-8** |
| (whole-file `grep -oP 'strata::kernels::[A-Za-z_0-9]+'` distinct, for reference) | 81 | 73 | -8 |

(The parent's cited `87/65` pair: **65 reproduces exactly** as the name-only distinct on the I2d library, and **87
reproduces as the WHOLE-FILE name-only count on the PRE-I2d (I2c) library** - i.e. the two figures come from the two
name-only variants at different points, which is itself the stale-reading trap; both variants move by the same 6
per batch, which is the signal.  The I2e build was measured with `$HOME/vkbuild-vulkan` rebuilt from the current
tree; the I2d baseline was reproduced by rebuilding HEAD's `gdn_vk.cpp` into the same library, giving 196/67/65 -
the bar I2d recorded.)

**THE REMAINING 59 `strata::kernels::` SYMBOLS, GROUPED (so the next batches plan from a number).**

| subsystem | n | symbols |
|---|---:|---|
| **glue (the I2 elementwise set)** | **0** | all answered |
| **matvec / GEMV / KV** | **21** | unchanged |
| **attention / QSA / MoE / GR / PLE / rope** | **36** | unchanged |
| **GDN / DeltaNet mixer** | **0** | **COMPLETE** - all fourteen answered in `gdn_vk.cpp` (+ `gdn_gate` in `elementwise_vk.cpp`) |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

(The 188 total also carries the 12 CUDA-runtime symbols and the engine's own cross-TU `strata::core`/`main`
references - the CUDA-runtime surface is the un-approved RE-SCOPE I2b reported and this batch did not touch.)

**WHAT IS LEFT OF I2.**  The matvec/GEMV/KV group (21) and the attention/QSA/MoE/GR/PLE/rope group (36), plus the
two `other` rows.  The GDN / DeltaNet mixer contributes **nothing**.

## INCREMENT I2 (CONTINUED FURTHER-STILL) — THE FIRST SIX GDN / DELTANET MIXER ENTRY POINTS, AND THE LINK PROGRESS BAR (2026-10-05, `vega`)

**LINK PROGRESS (the port's progress bar toward a layer that LINKS): `204 → 196` undefined references, `73 → 67`
distinct `strata::kernels::` symbols** (and `71 → 65` under the parent's simpler name-only regex that drops the two
signatures carrying a nested `strata::kernels::GdnShapes` - the `gdn_step` / `native_gdn_step` pair; the full-signature
NEXT.md pattern below is the one this tree uses).  Re-run the one-layer-body link with (build = the
`-DSTRATA_ENABLE_VULKAN=ON` CMake build; measured with `/tmp/vkbuild-gdn`):

    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include -Ivulkan/src/device -DSTRATA_ENABLE_VULKAN=1 \
        -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o $HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a $HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a -lvulkan \
        -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                     # -> 196
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                              # -> 67

**OF THE SIX-SYMBOL DROP THIS BATCH OWNS ALL SIX.**  `layer.cpp` reaches exactly the six GDN kernels wired here, so
each one it answers removes its own reference(s) from THIS link (8 raw lines: `native_gdn_l2_norm` and `gdn_l2_norm`
are called twice each, at :266/267 and :269/270).

**THE REMAINING 67 `strata::kernels::` SYMBOLS, GROUPED (so the next batches plan from a number).**

| subsystem | n | symbols |
|---|---:|---|
| **glue (the I2 elementwise set)** | **0** | all answered |
| **matvec / GEMV / KV** | **21** | unchanged: `bf16_gemv`, `bf16_gemv_split`, `bf16_gemv_fp32_mmvf`, `bf16_gemv_fp32_mmvf_cols`, `native_mmvq`, `native_quantize_q8_1`, `quantize_q8_0`, `quantize_q8_K`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `s2_gemv_q8`, `kv_*` (11) |
| **attention / QSA / MoE / GR / PLE / rope** | **36** | unchanged: the attention/QSA/MoE/GR/PLE/rope set |
| **GDN / DeltaNet mixer** | **8** | `gdn_beta_gate`, `gdn_step`, `gdn_out_norm`, `native_gdn_beta_gate`, `native_gdn_gate`, `native_gdn_step`, `native_gdn_out_norm`, `fused_gdn_step_norm` |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

(The 196 total also carries the 12 CUDA-runtime symbols and the engine's own cross-TU `strata::core`/`main`
references - the CUDA-runtime surface is the un-approved RE-SCOPE I2b reported and this batch did not touch.)

**THE SIX, IN THE ORDER THE MIXER'S OWN BODY REACHES THEM.**  Read from `gdn_layer` (`src/core/layer.cpp:223`, the
mixer for 36 of the 48 layers, called from `block_layer_pre` at `:1259`), NOT from the plan's §3 list (which names
the GDN rows in class-B/class-A prose and is not an order).  Within a mixer STAGE the branches are alternatives, so
the order is the order the call sites appear: fused, then native, then legacy.

| # | symbol | call site | shader | wrapper == shader (bitwise) | wrapper vs oracle |
|---|---|---|---|---|---|
| 1 | `fused_gdn_conv_l2` | layer.cpp:250 (conv stage, fused) | fused_gdn_conv_l2.spv | **40960/40960 + 2048/2048, worst 0** | **10240/10240 w 2.42e-04; 512/512 w 6.63e-06** (terms bound) |
| 2 | `native_gdn_conv_silu` | layer.cpp:253 (native) | native_gdn_conv_silu.spv | **12800/12800 + 120/120, worst 0** | **5120/5120 w 1.24e-05; 48/48 w 2.49e-07** (terms bound) |
| 3 | `gdn_conv_step` | layer.cpp:255 (legacy) | gdn_conv_step.spv | **96/96 + 1200/1200, worst 0** | **24/24 w 2.04e-06; 300/300 w 1.15e-06** |
| 4 | `native_gdn_l2_norm` | layer.cpp:266/267 (native) | native_gdn_l2_norm.spv | **128/128 + 2048/2048, worst 0** | **128/128 w 1.18e-07; 2048/2048 w 1.68e-07** (tol 1e-4) |
| 5 | `gdn_l2_norm` | layer.cpp:269/270 (legacy) | gdn_l2_norm.spv | **2048/2048 + 384/384, worst 0** | **2048/2048 w 1.26e-07; 384/384 w 0** (tol 1e-5) |
| 6 | `fused_gdn_ab` | layer.cpp:287 (beta/gate stage, fused) | fused_gdn_ab.spv | **96/96 + 8/8, worst 0** | **96/96 w 3.48e-07; 8/8 w 9.31e-08** (terms bound) |

All measured on `vega`'s Arc (intel_icd) through the port's gate, each case pinned to the harness device
(`EnginePin`).  The engine wrapper's output is compared BITWISE (as 32-bit words) to the port's shader path over the
same fixture, and separately against the case's explicit oracle (a double transcription of the engine's OWN body -
`fused_gdn.cu` / `native_gdn_preprocess.cu` / `gdn.cu`).  The conv cases also compare the SLID state bitwise.

**NO `host` ROW WAS NEEDED, and that is stated rather than assumed.**  The only `host` row the mixer's six reach is
`native_gdn_enabled()` (`layer.cpp:247`/`:253`/`:265`/`:295`/`:306`/`:324`), already answered by
`vulkan/src/kernels/native_caps_vk.cpp`.  The GDN headers carry NO `*_scratch_bytes`-style symbol and no shape
accessor - the only shared shape, `GdnShapes`, is a by-value POD passed to `gdn_step` (not in this batch) - so the
six are self-contained and the link introduced no new undefined reference.

**NONE OF THE SIX IS A MODULE-STATE READER LIKE `cvec_apply`.**  Each is bind-and-dispatch over an already-gated
shader; what they DO carry is the CUDA wrapper's argument CONTRACT, kept as the port's customary loud refusal (never
a silently wrong binding): `fused_gdn_conv_l2` refuses `channels % 128 != 0` or `qk_heads > channels/128` (the CUDA
body's own check), `native_gdn_conv_silu` refuses `d_conv != 4` (four taps), `fused_gdn_ab` refuses `n_embd % 8 != 0`
(the shader's 8-element activation chunk).  One wrapper detail worth recording: the port's `native_gdn_l2_norm`
shader divides the push `eps` by `cols` ITSELF, so this wrapper passes the RAW engine epsilon and
`1/sqrt(cols)` - matching the gate case's push bit for bit (the CUDA passes `eps/S` and `1/sqrt(S)`; identical at
`cols == S == 128`).

**RESULTS (vega).**  Engine CONFIGURE + BUILD under `-DSTRATA_ENABLE_VULKAN=ON` clean;
`strata_vk_entry_smoke` RUNS the six new wrappers on the Arc and PASSES.  Gate on `vega`: **Arc 516/0/0 (exit 0),
llvmpipe 504/0/3, radeon iGPU 507/0/2** - **+24 verdicts on every arm** (6 cases x 2 arms x 2 verdicts), 0 failed;
the radeon arm's `budget` flake did NOT fire this run.  `check_port_map.py` passes (`168 - 78 kernel, 61 host, 29
todo; 111 shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically (the six
were already `kernel` rows, so the map does not move).  No new falsification injection: the wrapper proof IS the
check (wrapper == shader path BITWISE), and the wrong-view-offset class is falsified by
`gates/inject-verify.sh fwht-entry-wrong-view-offset`.  **`z820b` is PENDING** (no XTX/K620 number).  The CUDA-runtime
host surface was NOT touched, and the plan was NOT re-scoped.

**WHAT IS LEFT.**  The GDN mixer's remaining **8** symbols: the beta/gate and step/norm stages - `native_gdn_beta_gate`,
`gdn_beta_gate`, `native_gdn_gate`, `native_gdn_step`, `gdn_step`, `native_gdn_out_norm`, `gdn_out_norm`, and the
fused `fused_gdn_step_norm` - which would complete the mixer and take the GDN group to 0.

## INCREMENT I2 (CONTINUED FURTHER) — the next five glue entry points, and THE LINK PROGRESS BAR (2026-10-05, `vega`)

**LINK PROGRESS (the port's progress bar toward a layer that links): `214 → 204` undefined references, `80 → 73`
distinct `strata::kernels::` symbols.**  Re-run the one-layer-body link with (build = the `-DSTRATA_ENABLE_VULKAN=ON`
CMake build; measured with `/tmp/vkbuild-i2c`):

    g++ -std=c++20 -O0 -Iinclude -Ivulkan/include -Ivulkan/src/device -DSTRATA_ENABLE_VULKAN=1 \
        -c src/core/layer.cpp -o /tmp/layer.o
    g++ /tmp/layer.o $HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_kernels.a $HOME/vkbuild-vulkan/vulkan/libstrata_vulkan_device.a -lvulkan \
        -o /tmp/layer-link 2> /tmp/link.log ; true
    grep -c "undefined reference" /tmp/link.log                                     # -> 204
    grep -oP "undefined reference to \`\K[^']+" /tmp/link.log | grep "strata::kernels::" \
        | sed 's/strata::kernels:://' | sort -u | wc -l                              # -> 73

**OF THE SEVEN-SYMBOL DROP, THIS BATCH OWNS TWO.**  `layer.cpp` reaches exactly two of the five this batch wired
(`cvec_apply` and the cvec `host` accessor `cvec()`); the other five of the drop (`gdn_gate`, `rms_norm_weighted`,
`embedding_gather`, `doorbell_publish`, `doorbell_ring`) landed in I2/I2b and were already answered before this
batch — the README's "214 / 80, the backend answers 4" baseline predates them.  The other three of the five
(`add_inplace`, `gather_rows`, `scatter_rows_f32`) and `f32_to_f16_bulk` are called from `expert_source.cpp` /
`peer_experts.cpp` / `mtp.cpp` / nowhere, so they do NOT move THIS link — they are wired and proved, and will move
the link for the full program.

**THE REMAINING 73 `strata::kernels::` SYMBOLS, GROUPED (so the next batches plan from a number).**

| subsystem | n | symbols |
|---|---:|---|
| **glue (the I2 elementwise set)** | **0** | all answered — `silu/scale/f32_to_bf16/gdn_gate/rms_norm_weighted/embedding_gather/add_inplace/gather_rows/scatter_rows_f32/cvec_apply/f32_to_f16_bulk` |
| **matvec / GEMV / KV** | **21** | `bf16_gemv`, `bf16_gemv_split`, `bf16_gemv_fp32_mmvf`, `bf16_gemv_fp32_mmvf_cols`, `native_mmvq`, `native_quantize_q8_1`, `quantize_q8_0`, `quantize_q8_K`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `s2_gemv_q8`, `kv_append_step`, `kv_append_q4_step`, `kv_append_q8_step`, `kv_gather_step`, `kv_gather_q4_step`, `kv_gather_q8_step`, `kv_block_bytes`, `kv_ring_table`, `kv_stream_reset`, `kv_stream_resolve` |
| **attention / QSA / MoE / GR / PLE / rope** | **36** | `native_flash_attn_short_step`, `native_rope_apply`, `rope_neox_apply`, `rope_scaling`, `rope_table_set`, `build_rope_table`, `native_router_top10`, `router_top10`, `moe_combine`, `native_moe_combine`, `shared_expert(+_scratch_bytes)`, `ple_block(+_scratch_bytes)`, `ple_history_advance`, `PleTable::{collect,is_open,issue}`, `ngram_rows`, `qsa_attend_step`, `qsa_block_scores`, `qsa_block_topk`, `qsa_decode_attn_scratch_floats`, `qsa_decode_attn_step`, `qsa_gate_apply_f32`, `qsa_index_step`, `qsa_step_fill`, `native_qsa_{gate_apply,rms_norm_weighted,indexer_append}`, `topk_512_step`, `gr_read`, `gr_write`, `gr_workspace_init`, `fused_gr_read`, `fused_gr_supported` |
| **GDN / DeltaNet mixer** | **14** | `gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`, `gdn_step`, `gdn_out_norm` and their `native_gdn_*` siblings, and `fused_gdn_conv_l2`, `fused_gdn_ab`, `fused_gdn_step_norm` |
| **other** | **2** | `copy_i32_from_mapped`, `indexer_key_append` |

(The 204 total also carries the 12 CUDA-runtime symbols and the engine's own cross-TU `strata::core`/`main`
references — the CUDA-runtime surface is the un-approved RE-SCOPE I2b reported and this batch did not touch.)

**THE FIVE ENTRY POINTS, in the order the forward path reaches them.**  Read from the decode path, NOT from the
plan's list (the plan's list is alphabetical in part).  `layer.cpp`'s body reaches exactly ONE of the five
directly; the others are the same forward path through its sibling files:

| # | symbol | call site | shader | wrapper == shader (bitwise) | wrapper vs oracle |
|---|---|---|---|---|---|
| 1 | `add_inplace` | `expert_source.cpp:2353` (MoE hit combine) | add.spv | **1000/1000, worst 0** | **1000/1000 == d+s, worst 0** |
| 2 | `scatter_rows_f32` | `peer_experts.cpp:241` | scatter_rows_f32.spv | **3072/3072 + 30/30, worst 0** | **3072/3072 + 30/30 vs the rule, worst 0** |
| 3 | `cvec_apply` | `layer.cpp:1330/1333/1336` | cvec_apply.spv | **6144/6144, worst 0** | **6144/6144 vs the project(2·s) oracle, worst 4.74e-07** (tol 1e-4) |
| 4 | `gather_rows` | `mtp.cpp:450` (MTP drafter) | gather_rows.spv | **576/576 + 316/316, worst 0** | **576/576 + 316/316 vs ids[r] source rows, worst 0** |
| 5 | `f32_to_f16_bulk` | NO `src/core/` site | f32_to_f16.spv | **1024/1024, worst 0** | **1024/1024 == f16_from_f32, worst 0** |

All measured on `vega`'s Arc (intel_icd) through the port's gate, each case pinned to the harness device
(`EnginePin`).  `strata_vk_entry_smoke` runs all five too.

**`cvec_apply` IS THE ONE THAT IS NOT THIN.**  Its engine wrapper reads MODULE state (`strata::kernels::cvec()`),
which `src/kernels/cuda/cvec.cu` owns and a Vulkan build does not compile; the plan's §1 classifies `cvec` as a
`host` row ("the table the cvec_apply kernel reads") but it is a device-crossing row like the 18 it lists — the
backend must answer it.  So this batch also implements `cvec`, `cvec_upload`, `cvec_replicate`,
`cvec_set_enabled`, `cvec_enabled`, and places the device tables LAZILY into each `Stream`'s arena
(`Stream::cvec_tables` in `vulkan/src/device/vk_arena.hpp` — the tables live and die with the stream, so a
reopened stream cannot inherit a stale direction).  The published `Cvec::dir/s/on` are non-null sentinels: the
engine only tests them for null, and `covers()` reads the host `steered` vector.

**RESULTS (vega).**  Engine CONFIGURE 0.2 s / BUILD ~2 s (`-DSTRATA_ENABLE_VULKAN=ON`; the option `return()`s
before the CUDA engine, so this builds the backend + smoke).  `strata_vk_entry_smoke` PASSES on the Arc, running
all eleven wrappers + five doorbell + the handoff.  Gate on `vega`: **Arc 492/0/0 (exit 0), llvmpipe 480/0/3,
radeon iGPU 483/0/2** — **+14 verdicts on every arm** (5 new cases × 2–4 verdicts), 0 failed; the radeon arm's
`budget` flake did NOT fire this run.  `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 111
shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **`z820b` is
PENDING** (no XTX/K620 number).  No new falsification injection was added: the wrapper proof IS the check
(wrapper == shader path BITWISE), and the wrong-view-offset class is already falsified by
`gates/inject-verify.sh fwht-entry-wrong-view-offset`.

## INCREMENT I2 (CONTINUED) — the `doorbell_*` symbols answered, the next three glue kernels, AND THE ENGINE-PROGRAM GAP (2026-10-05, `vega`)

The rest of I2's engine half, plus the measurement the increments never had: **what it actually takes to compile
the ENGINE'S OWN program under `STRATA_ENABLE_VULKAN=ON`.**  Every number here was measured on `vega`; the box
`z820b` is down, so no XTX/K620 number is claimed.

**THE ENGINE-PROGRAM GAP — MEASURED, NOT ESTIMATED (deliverable A).** Under `STRATA_ENABLE_VULKAN=ON` the
top-level `CMakeLists.txt` `return()`s after `add_subdirectory(vulkan)`, so none of `src/` is configured. The
gap was measured by compiling the engine's own host translation units with the backend's flags
(`g++ -std=c++20 -Iinclude -Ivulkan/include -I vulkan/src/device -DSTRATA_ENABLE_VULKAN=1`):

1. **CONFIGURE — clean, 0.16 s.**  The backend target only.  The top-level block returns before
   `add_compile_definitions(STRATA_VERSION=...)`, `strata_core`, `strata_kernels`, `strata_engine`,
   `strata_prefill`, `strata`, `strata_spec` and `strata_kernels_cpu`.
2. **COMPILE — clean but for ONE define.**  All of `src/core/*.cpp` and `src/program/generate.cpp` compile as
   ordinary C++20.  `generate.cpp` is the only failure: `STRATA_VERSION` (used at :958, :3867, :5700) is not
   defined, because the `add_compile_definitions` at top-level line 37 sits AFTER the `return()`.  No engine
   header assumes CUDA in a way that blocks the compile — a system `cuda_runtime.h` (`/usr/include`) satisfies
   the `#include`, and the engine's own CUDA headers are declarations only.
3. **LINK — the wall.**  A minimal engine-linked target (one layer body, `src/core/layer.cpp`, linked against
   `libstrata_vulkan_kernels.a` + `libstrata_vulkan_device.a`) fails with **214 undefined references**, of which
   **80 distinct `strata::kernels::` symbols** (the backend answers 4: `fwht256_cuda`, `silu_inplace`,
   `scale_inplace`, `f32_to_bf16_bulk`; I3/I4/I5 own the rest) and **12 CUDA RUNTIME symbols** —
   `cudaMalloc`, `cudaMemcpy`, `cudaMemcpy2DAsync`, `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaHostAlloc`,
   `cudaHostGetDevicePointer`, `cudaFreeHost`, `cudaEventCreate/Record/ElapsedTime`, `cudaDeviceSynchronize`,
   `cudaPeekAtLastError`.  Across `src/core/*.cpp` + `src/program/*.cpp` the engine calls the CUDA runtime
   API directly in **18 host files** (≈100 distinct API names: streams, events, graphs, peer access, pinned/mapped
   host memory, `cudaMemGetInfo`).
4. **RUN — the deepest engine-linked target that links and runs is `strata_vk_entry_smoke` (I1/I2)**: it PASSES
   on the Arc, now running fwht256 + six glue wrappers + the five `doorbell_*` symbols.  One layer body does NOT
   link, so no token can be produced.

**VERDICT — THIS IS A RE-SCOPE THE USER MUST APPROVE, NOT I5's WIRING.**  The plan (§1) prices the engine side
as *kernels-namespace symbols*: 53 `kernel` + 18 device-crossing `host` rows + 19 class-A = 90 entry points, and
I5 is "2 kernel rows + wiring the recorded decode step into `generate.cpp`".  The measurement shows a SECOND,
uncounted surface the plan does not name and I5 does not cover: **the CUDA runtime API called directly by 18
engine host `.cpp` files**, plus the engine's non-GPU libraries (`strata_core`'s `device.cu`/`pinned.cu`/
`graph.cpp`, `strata_kernels_cpu` + ggml, `strata_prefill`, `strata_spec`) and the `STRATA_VERSION` define.  The
SYCL port's route for exactly this was to MIGRATE the host files that call the runtime (`sycl/CMakeLists.txt`:
"only host files that called the CUDA runtime were migrated") and ship a `cuda_runtime` shim.  The Vulkan plan
says it does not edit engine headers and prices "implement N symbols" — it has no line item for a runtime shim
or 18 migrated host files.  **Per the batch's instruction, this is NOT re-scoped here; it is measured and
reported for the user to approve.**  What is certain: I5 as written (2 kernel rows + step wiring) is NOT
sufficient to produce a token.

**THE `doorbell_*` SYMBOLS ANSWERED (deliverable B).**  New `vulkan/src/kernels/doorbell_vk.cpp` answers the
five symbols `layer.cpp:380/389` and `session.cpp:873` call, each mapped onto the split submission
`vulkan/src/device/sync.hpp` designed:

| symbol | direction | the mapping |
|---|---|---|
| `doorbell_publish` | device→host | the fenced publish copy of x/ids/weights (`sync_copy_fenced`, the SAME primitive `sync_publish` uses), then the ring raised by one.  THE FENCE IS THE RING: the fenced submit returning IS "the host may read the payload". |
| `doorbell_publish_value` | device→host | the same publish; the ring STORED (the HIP `STRATA_DOORBELL_STORE=1` variant). |
| `doorbell_publish_res` | device→host | `ids` copied always, `x` only when a selected id is a miss — the miss decision is a pure function, so it is evaluated on the host and the copy stays a device→host transfer.  P6-verifier-only. |
| `doorbell_ring` | device→host | the ring alone (the fallback when the fused publish is off). |
| `doorbell_wait` | host→device | **SUBMITS NOTHING.**  The CUDA form is a one-thread kernel that SPINS on host memory; this backend forbids a waiting kernel.  The host writes the answer BEFORE calling it, the consumer is a later fenced submission, and `doorbell_wait` enforces only the ordering contract — an un-answered handoff is a LOUD REFUSAL, never a hang. |

The ring is the HOST's own count (sync.hpp's stated choice: "the fence is the ring", the count "lives where it is
read").  `sync_copy_fenced` was exposed from `sync.*` so both directions of the handshake have ONE definition.
**PROOF (extended `case_sync_handoff`, arms E — not a parallel case):** `doorbell_publish` device→host ordered
**528/528**, ring reads 1; `doorbell_ring`/`publish_value` increment-then-store **2/2**; `doorbell_wait` then the
host-submitted consumer reads the host's answer **1024/1024**.  The falsification (calling `doorbell_wait` before
the host answers) is a refusal, REASONED not executed — a submitted spin is what this port forbids.

**THE NEXT THREE GLUE KERNELS (deliverable C).**  Added to `vulkan/src/kernels/elementwise_vk.cpp`, engine
headers unchanged, in the order the layer body reaches them — `gdn_gate` (`layer.cpp:300`), `rms_norm_weighted`
(`:880`), `embedding_gather` (`:1083`).  Each is proved by a new `case_*_entry` through the ENGINE WRAPPER,
bitwise against the port's own shader path AND against the explicit oracle, each pinned to the harness device
(`EnginePin`):

| kernel | shader | wrapper == shader path | wrapper vs oracle |
|---|---|---|---|
| `gdn_gate` | gdn_gate.spv | 48/48 + 144/144 bitwise | 48/48 + 144/144, worst rel 2.71e-07 / 2.27e-07 (tol 5e-6) |
| `rms_norm_weighted` | rms_norm.spv | 512/512 + 8192/8192 + 12288/12288 bitwise | 512/8192/12288 all pass, worst rel 1.4e-07 (tol 3e-3) |
| `embedding_gather` | embedding_gather.spv | 512/512 ×4 arms bitwise | 512/512 ×4 bitwise vs the two-rounding rule |

The `rms_norm_weighted` wrapper supplies ones for the CUDA contract's null weight (Vulkan has no null descriptor);
the unweighted arm is exercised and still agrees BITWISE with the shader path.

**RESULTS.**  Engine CONFIGURE 0.16 s / BUILD 1.95 s (`-DSTRATA_ENABLE_VULKAN=ON`; the option `return()`s before
the CUDA engine, so this builds the backend + smoke).  `strata_vk_entry_smoke` RUNS on the Arc and passes every
arm (fwht256 + six glue wrappers + the five `doorbell_*` + the handoff).  Gate on `vega`: **Arc 477/0/0,
llvmpipe 465/0/3**; the radeon iGPU is NON-DETERMINISTIC, and three consecutive runs gave **467/1/2** (the
documented `budget: independent requery` flake), **466/2/2** (the open `bf16_gemv` defect), and **468/0/2** —
the pre-existing characterised defects, unchanged by this batch (Arc +22 and lvp +22 verdicts over I2's totals).
`check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 111 shaders built, 92 claimed`) and
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **`z820b` PENDING.**

## INCREMENT I2 — THE DOORBELL REDESIGN + THE FIRST THREE GLUE ENTRY POINTS — **DONE 2026-10-05** (`vega`)

The engine half of I2.  The plan's I2 list is NOT an order, and the doorbell was the increment's named biggest
unknown; both are settled by reading the engine rather than guessing.

**THE DOORBELL, READ FIRST.** `src/kernels/cuda/elementwise.cu`'s handshake is a kernel that SPINS on host
memory ordered by `__threadfence_system()`: `doorbell_publish` (:293, called from `layer.cpp:380` - the GPU
copies the router's `x`/`ids`/`weights` into mapped pinned host memory and increments a mapped ring),
`doorbell_ring` (:209, `layer.cpp:389` - the ring alone), `doorbell_wait` (:214, `session.cpp:873` - a
one-thread kernel that waits for the host-written mapped flag).  So it is a BIDIRECTIONAL CPU-in-the-middle
handoff, and each direction needs a different replacement.  `vulkan/src/device/sync.*` (new) gives:
DEVICE->HOST = the submission FENCE (`Ctx::dispatch` submits and waits a fence before returning, so
`sync_publish` returning is exactly "the payload is visible to the host" - the `__threadfence_system()`+ring
ordering, needing no counter to poll and no flush, because the handoff buffers are `HOST_VISIBLE|HOST_COHERENT`
the type `Ctx::alloc` requires); HOST->DEVICE = a HOST-DRIVEN SPLIT SUBMISSION (the consumer is submitted only
AFTER the host has written the answer, so the device never waits).  The consequences for I3/I4, stated so the
next increment is not surprised: **the recorded decode step is SPLIT at the handoff** (phase 1 = the publish,
phase 2 = the consume), and the host's per-token loop becomes submit-publish -> read payload -> run the pool ->
write answer -> submit-consume.  A timeline semaphore was considered and NOT used (the submission path here is
synchronous, so the fence already gives the ordering, and enabling `timelineSemaphore` would be a device-create
change); the swap is local to `sync_publish` when the step becomes a genuinely asynchronous re-submission.
`case_sync_handoff` pins the ORDERING (consume before the answer reads the sentinel, the same call after it
reads the answer; the ring counts publications).  The deadlock arm is reasoned, NOT executed - executing it
means submitting a spinning kernel, which is forbidden and risks the display card.  Where the engine's
`doorbell_*` symbols get answered from `sync.*` is I2's NEXT piece (the device-crossing host rows), together
with `build_rope_table`, `rope_table_set/release`, `kv_ring_table`, `kv_stream_reset`, `gr_workspace_init`.

**THE THREE GLUE ENTRY POINTS.** Read from `src/core/layer.cpp`'s layer body (`block_layer` -> `gdn_layer`, the
mixer for 36 of the 48 layers), the first three glue kernels it reaches are `silu_inplace` (:257, the legacy
conv+SiLU), `scale_inplace` (:276, the legacy 1/sqrt(S) on q) and `f32_to_bf16_bulk` (:290, the alpha/beta
activation, reached by DEFAULT because `native_bf16_projections` defaults false).  Wired in
`vulkan/src/kernels/elementwise_vk.cpp`; each proved by `case_*_entry` through the ENGINE WRAPPER, bitwise
against the port's own shader path and against the explicit oracle (scale 1000/1000, silu 1000/1000, bf16
1024/1024).  **The NEXT three, same reading, are `gdn_gate` (:300), `rms_norm_weighted` (:880) and
`embedding_gather` (:1083).**

**A GATE HAZARD FOUND WHILE PROVING THEM, AND ITS FIX.** The gate's own `case_icd_resolution` (~:4673) calls
`unsetenv("VK_ICD_FILENAMES")` and never restores it, so any `*_entry` case's SECOND `VkInstance` enumerates
every ICD and takes the Intel Arc while the harness `ctx` sits on the arm's ICD.  The same `silu_f32.spv` on
the same input differs by exactly 1 ULP between the Arc and llvmpipe (x=-3.44161081: bddaa466 vs bddaa465).
The entry cases now pin `STRATA_VK_DEVICE` to the harness device's NAME (`EnginePin`); the bitwise claim was
KEPT, not loosened.  `case_fwht256_entry` shares the exposure and never noticed because fwht256 is bitwise
identical on every device - worth remembering for any future cross-instance case.

**THE W26 ALLOCATION/VISIBILITY SEAM — CHECKED; THE TOP SECTION'S CHARACTERISATION STANDS.** (1) `Ctx::free`
(`vulkan/src/device/vk_compute.cpp:684-697`) does NOT fence, but it does not need to: the file has exactly TWO
`vkQueueSubmit` sites and each is followed by `vkWaitForFences(..., UINT64_MAX)` (`end_oneshot_and_wait`
:774-788, used by `dispatch` and both staging directions; `submit_recorded` :1064-1076), so no submission that
references a buffer is ever in flight when `free()` runs - the release/reuse hazard is RULED OUT for this
layer's call patterns.  (2) No upload barrier is missing: a staged upload carries a `TRANSFER_WRITE ->
SHADER_READ|HOST_READ` barrier AND the fence (:795-815), and a mapped write goes into `HOST_VISIBLE|
HOST_COHERENT` memory whose visibility to the next dispatch is Vulkan's implicit host-write ordering at
`vkQueueSubmit`.  The defect is therefore left as characterised, not guessed at.  The one residual worth
knowing: freeing a buffer that a RECORDED command buffer still binds is a use-after-free the synchronous API
cannot prevent (a recording holds raw `VkBuffer` handles); no gate pattern reaches it because every replay is
fenced before returning.

**RESULTS.** Engine CONFIGURE 0.15 s / BUILD 0.94 s (`-DSTRATA_ENABLE_VULKAN=ON`, CUDA/HIP/SYCL OFF; the option
`return()`s before the CUDA engine, so this builds the backend + smoke target).  `strata_vk_entry_smoke` RUNS
on the Arc, llvmpipe and the Ryzen iGPU and passes every arm.  Gate on `vega`: **Arc 455/0/0, llvmpipe
443/0/3, radeon iGPU 446/0/2, exit 0**; `check_port_map.py` passes and `PORT-MAP.tsv` regenerates
byte-identically.  **`z820b` is PENDING** (no XTX/K620 number).  Not measured here: the engine's own PROGRAM
(the layer/session loop) - the `doorbell_*` symbols are not yet answered, so "one layer end to end" is M-B,
not this increment.

## THE SAMPLER, MEASURED PROPERLY — the DEFAULT is the SPLIT, its real cost, and the PENALTY-HOIST fix — **DONE 2026-10-05**
The performance tier's named first target, and the only number in the port above 100 ms.  The baseline carried
`sampler_kernel_f32` **546.9 ms (Arc) / 202.0 ms (XTX)** with a CAVEAT: it is the ONE-BLOCK **fallback**, not the
engine's default.  This increment names the default, measures it over the real vocabulary, finds WHY it was slow,
and fixes the hot step.

**WHICH PATH THE ENGINE ACTUALLY TAKES (the call site and the condition).**  `sample_tokens`
(`src/kernels/cuda/sampler.cu:977`) branches, in order: `p.greedy || p.temperature <= 0.0f` -> the GREEDY argmax
(`:988`); `sampled_path() == SampledPath::Old` -> the one-block `sampler_kernel` (`:1000`, `STRATA_OLD_SAMPLER=1`);
else the default (`:1005`), where `:1013`
`if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows && !stream_capturing(stream))`
selects the **SPLIT** (`sampler_split_part_kernel` + `sampler_split_merge_kernel`); anything else — a wider
vocabulary, more rows, a captured stream, or a scratch that failed to allocate — falls to
`sampler_one_block_kernel`.  `sampled_path()` (`:854-859`) is **`Split`** unless `STRATA_OLD_SAMPLER` or
`STRATA_SAMPLER_ONE_BLOCK` is set; `kSplitMaxBlocks = kSplitMaxRows = 64` (`:606-608`).  **At the model's vocab
248,320, `n_blocks = ceil(248320/4096) = 61 <= 64` and a decode has `n_tokens = 1`, so a sampled request takes the
SPLIT.  The fallback is NOT the default, and 546.9/202.0 ms was never the selection's real cost.**  (The port
already pinned this choice as a predicate in `case_sample_tokens`; what was missing was the COST of the chosen
path.)

**MEASURED (vega; the shipped `bench/vk_bench.cpp` rows, `reps=5`, one dispatch per batch; `< 1.0` means the
default is faster than the fallback).**  The default takes ONE workgroup per row, so the split row is the whole
selection for that token:

| device | `sampler_kernel_f32` ONE-BLOCK fallback (tokens=1) | `sampler_split` **DEFAULT** (tokens=1) | split/block | `sampler_split` tokens=64 |
|---|---:|---:|---:|---:|
| vega Arc B70 (intel_icd) | 546.81 ms | **13.50 ms** | **0.025** | 13.97 ms |
| vega Ryzen iGPU (radeon_icd) | 301.55 ms | **9.06 ms** | **0.030** | 74.91 ms |
| vega llvmpipe (lvp_icd) | 1171.46 ms | **67.64 ms** | **0.058** | 391.29 ms |

**THE DIAGNOSIS, MEASURED NOT ASSUMED.**  The split was ported CORRECT but slow: **348.95 ms on the Arc /
161.62 ms on the iGPU / 665.46 ms on llvmpipe** at `n_tokens=1` (a focused probe of the same device layer and the
same shipped shader, one dispatch per replay, median of 5).  Three candidate causes, distinguished by measurement:
* **Not bandwidth.**  The logits row is 248,320 × 4 = **993 KB**; loading it once at any plausible bandwidth is
  microseconds, three orders below the 349 ms observed.
* **Not the one-workgroup-per-row decomposition.**  64 rows cost about what 1 row cost pre-fix (Arc **348.95 ->
  384.54 ms**, `tokens=64`) and post-fix (13.50 -> 13.97 ms): the workgroups run concurrently and the cost is
  PER ROW, not per launch.  (The engine's stage-1 grid over (61 blocks × rows) is parallel form the port did NOT
  take — that is headroom left on the table, not the bottleneck.)
* **It WAS the serial scan, in the wrong place.**  Removing the penalty window entirely took the split to **12.66 ms
  on the Arc — a 27.6× drop** (and the one-block f32 kernel 546.76 -> 210.37 ms, 2.6×).  `sampler_row_topk`
  (`common/sampler_select.glsl`) re-read each logit and re-scanned the whole history window on **every one of the
  `k` rounds** — `O(k · span · hlen)`, the port's scan-instead-of-bitmap trade applied where the engine applies a
  bitmap.  The engine's `sampler_split_part_kernel` (`src/kernels/cuda/sampler.cu:690-706`) builds a 4,096-bit
  bitmap of the block and penalises each element **once**, into the warp's registers (`s[kSplitPerLane]`), then runs
  its rounds over the cached values.

**THE FIX (the hot one, and only it).**  `sampler_row_topk` now computes each partition element's penalised logit
**once** into the lane's registers (`sc_pv[]`/`sc_pi[]`, the lane's `SC_PER_LANE = 4096/256 = 16` slice — the
engine's own `s[kSplitPerLane]` shape) and runs the k rounds over the cached values.  The value and index are the
same, the penalty count is round-invariant, and the candidate set and order are unchanged, so the selection is
byte-for-byte the old one.  The include is shared by `sampler_split.comp` and `coupled_sample.comp`, so the
coupled merge gets it too.  **Before -> after: 348.95 -> 13.50 ms on the Arc (25.8×), 161.62 -> 9.06 on the iGPU
(17.8×), 665.46 -> 67.64 on llvmpipe (9.8×).**  Nothing slower was shipped (nothing was rejected this batch — the
single candidate was a win on all three devices).

**FALSIFIED, and the case it falsifies is NEW.**  `gates/inject-verify.sh sampler-select-penalty-drop` zeroes the
hoisted penalty's count -> **`FAIL  sampler_split: the repeat penalty is applied ONCE per partition (the hoist)
    2/   16  worst 0`**.  `case_sampler_split` gained a SIXTH arm for it (a real hit on the row's head logit, id
100 × 8 with `penalty_repeat = 4`, which divides the 9.0 head to 2.25 and moves the selection to id 4200) — the
five existing arms all carry an empty window (`hist = 9999`), so NOTHING in the split case exercised the penalty
before this arm, and a hoist arm with no hit could not tell the two forms apart.  The existing
`sampler-split-merge-drop-parts` still bites (`FAIL ... 9/16`, the merge is unchanged).

**GATE TOTALS AFTER THE CHANGE (vega, whole gate, `run_gate.sh` exit 0):** **Arc (intel_icd) 445 passed / 0 failed
/ 0 skipped**, llvmpipe 433/0/3, radeon-iGPU 436/0/2 — **+1 verdict on every implementation** (the new split arm).
The pre-existing skips and the intermittent radeon `budget: independent requery` flake are unchanged (this batch
did not touch it).

**MAP:** `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo; 111 shaders built, 92 claimed`);
`make_port_map.py` regenerates `PORT-MAP.tsv` **byte-identically**.  No `kernels::` symbol changed (no new shader).

**THE RELEASE NOTE'S CAVEAT, RESTATED.**  The one-block figure was the FALLBACK's cost.  On the Arc the DEFAULT
measured 348.95 ms pre-fix, so the headline was 1.6× the default's real cost — and after this fix it is **40×**
above it (546.81 vs 13.50).  The XTX's default cost is **NOT MEASURED** (the box is suspended): **202.0 ms stands
as the XTX ONE-BLOCK FALLBACK figure, and the XTX SPLIT row is PENDING.**

**BOX `z820b`: PENDING, and deliberately un-invented.**  A probe of `192.168.1.116` times out (`100% packet loss`,
`No route to host`) with no WoL path; the whole-tree sync and `run_gate.sh`/`run_bench.sh` there are PENDING the
box being up.  No cross-host number is claimed.

## THE BF16-PROJECTION PAIR — `bf16_gemv` + `bf16_gemv_split` PORTED (one shared shader) — and the
## 8 SPECULATIVE-DRAFTER SYMBOLS LABELLED CLASS C — **DONE 2026-10-05**

This increment closes the **SECOND soft edge** the batch-5 REACHABILITY AUDIT raised.  The audit marked
`bf16_gemv` / `bf16_gemv_split` a soft edge because the setting that selects them
(`native_bf16_projections`) is not named in the port's capability contract.  Hole-hunted to the end:

**THE CHAIN.**  `project_bf16` (`src/core/layer.cpp:94-100`) is the ONLY caller of BOTH:
`native_bf16_projections ? bf16_gemv_fp32_mmvf : (split ? bf16_gemv_split : bf16_gemv)`.  The setting
(`layer.cpp:91`) DEFAULTS **false**; `layer_set_native_bf16` (`:174`) is called from `generate.cpp:2286` with
`o.native_bf16`, which `--native` sets true (`generate.cpp:1805`) and `--native-bf16` sets independently.  The
**port pins the setting nowhere** (it is a host setting, not a capability getter — `native_caps_vk.cpp` answers
neither), so with the flag off the layer dispatches the UNPORTED members on the main forward path:
`bf16_gemv_split` at `layer.cpp:291`/`:292` (GDN alpha/beta) and `:367` (router logits), `bf16_gemv` at
`:918`/`:962` (QSA indexer projections).

**VERDICT: REACHABLE BY DEFAULT → a HOLE, and BOTH MEMBERS ARE PORTED** (the preferred fix; `--native`'s TRUE
branch was already covered by the ported `bf16_gemv_fp32_mmvf`).  Pinning `layer_set_native_bf16(true)` was the
alternative and was REJECTED: the setter is engine host code (`src/core/layer.cpp`) the port does not fork, so a
pin would be a claim about a setting the port does not own — porting both sides makes the setting irrelevant.

| symbol (shader) | the rule (the engine's own body = the oracle) | case |
|---|---|---|
| `bf16_gemv` (`bf16_gemv.comp`) | `bf16_gemv.cu`: `y[o]=Σ f32(x[i])·f32(w[o*n_in+i])`, products exact in f32, the row read as 32-bit PAIRS, plain `acc += a*b` (NOT `__fmaf_rn`) | 4 arms: 2560×512, 2560×128, 128×64, 2×1 |
| `bf16_gemv_split` (SAME `bf16_gemv.comp`) | `bf16_gemv_split_kernel`/`bf16_gemv_warp_kernel` — the same RULE, a different reduction.  ONE shared shader: the CUDA's three kernels differ only in PARALLELISM STRATEGY, the engine's call sites all land on the coalesced warp paths (n_out 512/128/48, tpr 32), and subgroup ops are BANNED here — so both are rendered as ONE WORKGROUP per output row through the barrier tree (`threads_per_row` dropped).  Two map rows, one shader — the `bf16_gemv_fp32_mmvf`/`_cols` precedent. | 4 arms: 2560×512, 2560×48, 64×32, 2×1 |

**A MEASUREMENT THAT CHANGED THE SHIPPED KERNEL.**  The naive ONE-THREAD-PER-ROW decomposition was built and
benchmarked first, and it is **14–18x slower** than the workgroup form at the engine's shapes (Arc `split/serial
0.073` at n_out=512 / 0.057 at 48; Ryzen iGPU 0.103/0.055) — the CUDA's own comment says why (uncoalesced: 32
transactions per load), and CUDA uses the naive path only below n_out=64, which this engine never does for
`bf16_gemv`.  So the port ships the workgroup-per-row rendering; the naive variant is NOT in the tree.

**THE ORACLE is the double transcription of the shared rule, bounded by the row's TERMS** (`gemv_bound`,
`rtol·|want| + 16·2⁻²⁴·Σ|terms|`) — never a relative tolerance, because a cancellation row's `Σ|terms|` dominates
its result.  Fixture: row 0 is the LAYOUT PROBE (low halves ~1e3, high halves ~1e-3 → a pair-halves swap is
O(1)); row 1 is ALL-ZERO (the bound's `1e-30` floor is load-bearing); both margins (halves swap, off-by-one row)
are asserted host-side to MOVE the oracle; the output buffer carries a 0x5E guard region behind a surplus
workgroup.  Verdicts (Arc, all PASS): `bf16_gemv` 515/515 w 1.22e-02, 131/131 w 5.73e-03, 67/67 w 3.2e-02,
4/4 w 1.84e-03; `bf16_gemv_split` 515/515 w 6.99e-03, 51/51 w 6.37e-03, 35/35 w 1.42e-02, 4/4 w 9.61e-03.

**FALSIFICATION (both BIT, on the shared shader):** `bf16-gemv-swap-halves` → `FAIL bf16_gemv n_in=2560 n_out=512
4/515 w 1.54e+05`; `bf16-gemv-row-base` (weight row indexed by the OUTPUT stride) → `FAIL bf16_gemv n_in=2560
n_out=512 4/515 w 2.62e+34`.

**THE MEASUREMENT** (there is NO legacy sibling — this IS the non-native branch, so the pair is the ported
`bf16_gemv` against the PORTED native sibling `bf16_gemv_fp32_mmvf`; same workgroup-per-row decomposition, only the
activation precision differs, so a WASH is expected and measured; `XPAIR` ratio `<1.0` = first row faster, `reps=9`):

| pair (ported `bf16_gemv` ← native `bf16_gemv_fp32_mmvf`) | Arc B70 | Ryzen iGPU | lvp (vega) |
|---|---:|---:|---:|
| `bf16_gemv` n_out=512 | 1.000 | 0.996 | 0.933 |
| `bf16_gemv` n_out=48  | 1.006 | 0.997 | 0.956 |

**THE 8 SPECULATIVE-DRAFTER SYMBOLS → CLASS C (labelled, not ported).**  `add_streams_broadcast`,
`fused_gr_read_multi`, `window_ids`, `qsa_decode_attn_batch`, `moe_group_resident`, `row_top_prob`, `map_ids`,
`mtp_select` run only in the `--spec 4 --mtp` DRAFT loop, which THE PORT DOES NOT SELECT: the loop needs
`Verifier::init`, and `layer_verify_compatible()` (`layer.cpp:476-491`) demands `g_fused_gr` (forced FALSE by the
GR contract), `native_qsa_indexer_enabled()` (answered FALSE — the native append is unported) and
`native_bf16_projections` (a setting the port does not pin).  **The flag chain that would enable them is
`--spec 4 --mtp` AND a verifier-compatible native stack; the selected branch is a `--spec 0` run.**  The TSV rows
keep kind `todo` (its vocabulary is kernel/host/todo) but carry a `class C` reason string; the classification and
its chain are in `plan/DECODE-PATH-TRIAGE.md` → THE REACHABILITY AUDIT.

**THE MAP DROPS BY TWO.**  `168 — 76 kernel, 61 host, 31 todo` → **`168 — 78 kernel, 61 host, 29 todo`**;
`check_port_map.py` passes (`111 shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.

**GATE, after the change.  vega:** intel_icd (Arc B70) **444 / 0 / 0** (`run_gate.sh` exit **0**), llvmpipe
**432 / 0 / 3**, radeon_icd (Ryzen iGPU) **435 / 0 / 2** — **+8 verdicts** each (4 arms × 2 symbols), 0 failed;
the intermittent `budget: independent requery agrees` flake did not fire.  **Box (`z820b`): NOT RUN — the box
was UNREACHABLE this batch** (`ssh bob@192.168.1.116` → `No route to host`; ARP `INCOMPLETE`; the `z820b` tunnel
alias refuses; no Wake-on-LAN helper exists on vega).  The box run is **PENDING**: sync the WHOLE tree
(`tar czf - --exclude='*/build' --exclude='./.git' . | ssh bob@192.168.1.116 'tar xzf - -C ~/strata-vulkan-wt'`)
and re-run `run_gate.sh` + `run_bench.sh` there before the batch is treated as closed on both boxes.  Everything
in this section above is measured on vega.

## THE QSA GATE'S NATIVE MEMBER — class B batch 5 — **DONE 2026-10-05** — `native_qsa_enabled()` FLIPS TO TRUE,
## plus the REACHABILITY AUDIT of every remaining `todo` row and ONE HOLE FIXED

This increment ports **`native_qsa_gate_apply`** — the LAST symbol `native_qsa_enabled()` gates, and once it has a
shader that ONE flag's invariant is satisfied, so **the backend now answers `native_qsa_enabled()` TRUE**.  It also
carries the **REACHABILITY AUDIT** the task asked for: every remaining `todo` row of `plan/DECODE-PATH-TRIAGE.md`
is decided against the port's CURRENT capability answers, with its call site and flag chain — and the audit found
**one reachable-but-unported hole, which this batch FIXES**: `native_qsa_indexer_append` was reachable because its
gating flag `native_qsa_indexer_enabled()` had **no Vulkan definition at all**.

**THE SYMBOL, and the oracle it was transcribed from.**  `native_qsa_gate_apply` (`src/kernels/cuda/native_qsa.cu`
`:69-77` body `gate`, wrapper `:120-126`) replaces the legacy `qsa_gate_apply_f32` on the QSA layer's gate
(`layer.cpp:1010`, `if (native_qsa_enabled()) native_qsa_gate_apply(...) else qsa_gate_apply_f32(...)`).

| symbol (shader) | replaces | the native body's rule (oracle) | case |
|---|---|---|---|
| `native_qsa_gate_apply` | `qsa_gate_apply_f32` | `native_qsa.cu`'s `gate`: one thread per output element; `raw = q_full[h*2*head_dim + head_dim + d]` (the **SECOND** half of each head's 2*head_dim block), `out = attn * (1/(1+expf(-raw)))`, all **f32** | 3 arms (`n_head/head_dim` = 24/256 the model, 4/12, 2/8); vs a double transcription of the NATIVE rule, tol 1e-5; margins: the FIRST-half read and SiLU must each MOVE the fixture (checked host-side) |

**THE NATIVE vs THE LEGACY MEMBER: the RULES are the same, the ARITHMETIC is not.**  Both compute
`attn * sigmoid(second-half gate)`; the native CUDA forms the sigmoid with `expf` and the product in **f32**, the
legacy CUDA forms both in **f64** and rounds once.  The target has no `shaderFloat64`, so this port's legacy
shader **already** computes in f32 — so the two DEVICE shaders agree to within the transcendental gap and the
native-vs-legacy pair is a **WASH**.  That is reported, not tuned away: the case's margins therefore pin the two
plausible WRONG rules (FIRST half, SiLU), which are the ones this kernel can get wrong.

**THE MEASUREMENT — `native_qsa_gate_apply` ← `qsa_gate_apply_f32`, same shape, same device (`XPAIR` lines;
ratio is native/legacy, so < 1.0 means the native kernel is faster; `reps=9`).**  There is **NO dispatch chain**
here — the layer's QSA gate is ONE dispatch either way, unlike the fused GDN paths — so this is a drop-in pair:

| pair (native ← legacy) | Arc B70 | Ryzen iGPU | lvp (vega) | XTX (box) | K620 (box) | lvp (box) |
|---|---:|---:|---:|---:|---:|---:|
| `native_qsa_gate_apply` ← `qsa_gate_apply_f32` | 1.001 | 0.984 | 1.018 | 1.019 | 0.904 | 1.120 |
| med ms native / legacy | 0.0048 / 0.0048 | 0.0213 / 0.0217 | 0.0043 / 0.0042 | 0.1011 / 0.0992 | 0.0049 / 0.0054 | 0.0022 / 0.0020 |

**A WASH on all six devices (0.904–1.120)**, exactly as a same-shape, same-arithmetic drop-in should be: both
kernels are one thread per output element with no reduction and no shared memory.  Recorded as measured.  (The
XTX's 1.019 is a 2 µs difference on a 6.1k-element dispatch, at the fence-clock resolution.)  There is no
chain pair to report, and that is stated rather than invented.

**THE FLIP — batch 4's analysis, done BEFORE the flip.**  `native_qsa_enabled()` gates exactly TWO symbols, and
EVERY call site behind it was enumerated:

* `native_qsa_rms_norm_weighted` — `layer.cpp:879` (`normalize_rotate`), `mtp.cpp:488/491/514`, `verify.cpp:767`;
  ported in batch 1 and gated (`case_native_qsa_rms_norm_weighted`).
* `native_qsa_gate_apply` — `layer.cpp:1010`, and the verify path `verify.cpp:775` (`qb`), `:883`, `:887`;
  **ported here.**

**Nothing unported becomes reachable, and the reason is checked rather than assumed.**  The verify.cpp sites are
owned by the **P6 verifier**, which cannot init under this contract: `layer_verify_compatible()` (`layer.cpp:476-491`)
requires `native_bf16_projections` (a setting, default **false**), `g_fused_gr` (**false**, forced by the GR
contract), `g_fast_attn`/`g_fast_select` (true), the fused native GDN (**true since batch 4**) AND
`native_qsa_indexer_enabled()` (**false** — unported).  Three terms remain false, so the verifier is unreachable;
and even if it were reached, both QSA symbols it dispatches are ported.  **So the flip routes the engine only at
symbols that have a shader.**  (The flip also changes the verify path's `qb` branch, `verify.cpp:775/883`, from the
legacy gate to the native gate — both ported.)

**THE HOLE THE AUDIT FOUND, AND THE FIX.**  `native_qsa_indexer_append` (`layer.cpp:944`) is on the MAIN forward
path of all 12 QSA layers, every token; the shipped launch sets `o.native_qsa_indexer = true`
(`generate.cpp:1807` → `:2294`), and the port's contract has always *stated* the required answer
(`native_qsa_indexer_enabled() == false`) — **but the backend never defined the getter or the setter.**  The
contract was a claim without a body, and the shipped option would have selected the unported symbol (or failed to
link).  **Fix:** `vulkan/src/kernels/native_caps_vk.cpp` now defines both, answering **false**, which selects the
ported legacy `indexer_key_append`.  The map row stays `todo` (the native append is still unported) — the hole was
reachability, not the port.  Full audit in `plan/DECODE-PATH-TRIAGE.md`.

**THE CAPS CASE ENFORCES THE TRUTH, and the invariant is now falsified BOTH WAYS.**  `case_native_capabilities`
gained a **qsa arm** in the same form as the gdn arm: `native_qsa_rms_norm_weighted` and `native_qsa_gate_apply`
are marked `ported=true`, every ported symbol's `.spv` is required to exist, and `native_qsa_enabled()` must EQUAL
"every gated symbol has a built shader" — now **TRUE** (3/3).  Falsified:
* **under-claim**: `inject-verify.sh native-caps-qsa-false` (answer FALSE while both shaders exist) →
  `FAIL native capabilities: qsa flag 2/3`.
* **over-claim**: answer TRUE while a gated shader is ABSENT — reproduced by removing the gate's `.spv` and
  running the gate binary → `capability: native_qsa_gate_apply is ported but has no built shader` and
  `FAIL native capabilities: qsa flag 1/3`.
* **RETIRED**: `native-caps-qsa-true` (answer true while the gate shader was unported) — answering true is now
  the TRUTH, so it no longer falsifies anything; it prints `RETIRED` and exits 2 (the batch-4 `native-caps-gdn-true`
  precedent).  The retirement is recorded here as batch 4 recorded its own.

**THE MAP DROPS BY ONE.**  `PORT-MAP.tsv` moved `168 — 75 kernel, 61 host, 32 todo` → **`168 — 76 kernel, 61 host,
31 todo`** (`native_qsa_gate_apply` todo→kernel); `check_port_map.py` passes (`110 shaders built, 91 claimed`) and
`make_port_map.py` regenerates the file **byte-identically** (`diff -q`).

**GATE, after the change.  vega:** intel_icd (Arc B70) **436 / 0 / 0** (`run_gate.sh` exit **0**), llvmpipe
**424 / 0 / 3**, radeon_icd (Ryzen iGPU) **427 / 0 / 2** — **+4 verdicts** on every arm (3 gate arms + the new qsa
flag arm; the old combined caps verdict went 4→3 and the new qsa arm added one), 0 failed.  **Box (`z820b`):**
radeon_icd (RX 7900 XTX) **432 / 0 / 1** (the 1 is the pre-existing M8 `prefill split` skip, so `run_gate.sh` exits
**1** there), llvmpipe **424 / 0 / 3**, nvidia_icd (Quadro K620) **427 / 0 / 2** — **0 failed on every arm**.  The
Arc's new-case verdicts: `native_qsa_gate_apply` **6152/6152 w 7.89e-07**, 56/56 w 7.54e-07, 24/24 w 7.08e-07; the
box's: w 7.15e-07 / 7.1e-07 / 6.86e-07.  The shader's build line: `OK native_qsa_gate_apply OpExecutionMode %main
LocalSize 256 1 1 | census: none` (no subgroup op, barrier or atomic, as the CUDA).  The intermittent
`budget: independent requery agrees` flake did not fire on either box this run.
Every falsification was run and BIT: `native-qsa-gate-first-half` → `FAIL native_qsa_gate_apply n_head=24 head_dim=256
8/6152 w 4.13e+10`; `native-caps-qsa-false` → `FAIL native capabilities: qsa flag 2/3`; the over-claim direction
above → `FAIL 1/3`; and the batch-1 `native-qsa-rms-norm-eps-on-sum` re-run → `FAIL
native_qsa_rms_norm_weighted r=8 c=2560 in-place 5128/25608 w 0.98` (a regression check that the rebuilt gate still
bites).

**THE AUDIT'S DECISION LIST (the task's second deliverable).**  All 32 rows are decided in
`plan/DECODE-PATH-TRIAGE.md` → "THE REACHABILITY AUDIT".  Briefly: **1 is now `kernel`**; **11 are
reachable-but-unported** — the 8 speculative-drafter symbols (`add_streams_broadcast`, `fused_gr_read_multi`,
`window_ids`, `qsa_decode_attn_batch`, `moe_group_resident`, `row_top_prob`, `map_ids`, `mtp_select`), reachable
ONLY because the shipped `setup.py` writes `--spec 4 --mtp`; `native_qsa_indexer_append` (the hole, now
flag-closed); and the 2 BF16-projection rows (`bf16_gemv`, `bf16_gemv_split`), reachable only if
`native_bf16_projections` is false — a **soft edge** the port's contract table does not name; and **20 are
unreachable under the current answers**, each with its deciding condition.  **The queue's next item is the
drafter**: if the product ships `--spec 4`, those eight are forward-path dispatches.

## THE PERFORMANCE TIER'S FUSED GDN PATHS — class B batch 4 — **DONE 2026-10-05** — and `native_gdn_enabled()` FLIPS TO TRUE

This increment ports the **three fused GDN paths** — `fused_gdn_conv_l2`, `fused_gdn_ab`, `fused_gdn_step_norm`
— the LAST symbols `native_gdn_enabled()` gates.  Each is **gated and oracled against the engine's OWN fused
body** (`src/kernels/cuda/fused_gdn.cu`, via `fused_gdn.hpp`), each is **MEASURED against the multi-dispatch
chain it replaces** and against the non-fused native kernel(s) where a comparison exists (`ports/vulkan/bench/`),
and — because ALL NINE gated symbols (the six native kernels of batches 2–3 and these three) now have a shader —
**the flag's invariant is satisfied and `native_gdn_enabled()` answers TRUE.**

**THE THREE SYMBOLS, and the oracle each was transcribed from** (`fused_gdn.cu`; the layer's fused selection is
`layer.cpp:247` `fused_pre` and `:306` `fused_gdn`):

| symbol (shader) | replaces | the fused body's rule (oracle) | case |
|---|---|---|---|
| `fused_gdn_conv_l2` | `native_gdn_conv_silu` + `native_gdn_l2_norm` (q) + `native_gdn_l2_norm` (k) — 3 dispatches | `gdn_conv_l2_kernel` (`:73-93`): the four-tap conv (**NO** zero-bias fold, tap fastest) + the FP32 fast-math SiLU + the state slide, THEN a per-head L2 `y *= rsqrt(sum(y^2) + eps)` (eps on the **SQUARED NORM**) applied to the **q/k heads only** (`head < qk_heads`) | 3 arms (C=10240 qk=32 the model / 512 / 384, an ODD head count); h vs a double transcription, slid history **BIT-EXACT**, margins: the norm MOVES the q/k heads and leaves the v heads |
| `fused_gdn_ab` | `bf16_gemv_fp32_mmvf` (alpha) + `bf16_gemv_fp32_mmvf` (beta) + `native_gdn_beta_gate` + `native_gdn_gate` — 4 dispatches | `gdn_ab_kernel` (`:95-123`): one BF16 mat-vec per row (weight read as 32-bit PAIRS, LOW half = element `2p`, HIGH = `2p+1`), then `beta = sigmoid(acc)` and `gate = softplus(acc + dt) * ssm_a` | 3 arms (h_v=48 n=2560 / h_v=4 n=64 / h_v=3 n=512); gate+beta vs a double transcription, sums bounded by their TERMS (`gemv_bound`); margins: the softplus, the `ssm_a` factor and the beta sigmoid each MOVE the output (per-row max) |
| `fused_gdn_step_norm` | `native_gdn_step` + `native_gdn_out_norm` — 2 dispatches | `gdn_step_norm_kernel` (`:16-71`): the folded-decay recurrence (contract against the **UNDECAYED** state), the readout against the UPDATED state, the fused `1/sqrt(S)` readout scale, THEN the closing RMS norm (eps on the **MEAN**) with `gamma` and `sigmoid(z)` | 3 arms (S=128 h_k=16 h_v=48 / 4,8 / 3,9 — ODD h_v); y + state vs a double transcription; 4 margins: INTERLEAVE pairing, dropped decay, SiLU-vs-sigmoid, dropped readout scale each MOVE the output |

**THE MEASUREMENT — fused vs the chain it replaces, and vs the non-fused native kernel(s), same shape, same
device (`XPAIR` lines; ratio is fused/legacy, so < 1.0 means the fused path is faster; `reps=9`).**  The win is
in REMOVING DISPATCHES, exactly as batches 2–3 found; the fused-vs-native-single pairs are washes-to-slower
because the fused path replaces MORE than one dispatch.

| pair (fused ← what it replaces) | Arc B70 | Ryzen iGPU | llvmpipe (vega) |
|---|---:|---:|---:|
| `fused_gdn_conv_l2` ← `native_gdn_conv_silu` (1 dispatch) | 1.122 | 0.903 | 2.803 |
| `fused_gdn_conv_l2` ← conv_silu + 2× l2_norm (**3 dispatches**) | **0.452** | **0.712** | **0.560** |
| `fused_gdn_ab` ← `bf16_mmvf_f32` (1 dispatch, one row set) | 1.136 | 1.868 | 1.776 |
| `fused_gdn_ab` ← 2× bf16_mmvf + beta_gate + gate (**4 dispatches**) | **0.398** | **0.878** | **0.744** |
| `fused_gdn_step_norm` ← `native_gdn_step` (1 dispatch) | 1.021 | 0.998 | 1.092 |
| `fused_gdn_step_norm` ← `native_gdn_out_norm` (1 dispatch; NOT like-for-like — see below) | 7.533 | 63.722 | 6.220 |
| `fused_gdn_step_norm` ← native_gdn_step + native_gdn_out_norm (**2 dispatches**) | **0.929** | **0.987** | **0.930** |

The same pairs on the **box `z820b`** (RX 7900 XTX / RADV NAVI31, Quadro K620 / NVIDIA, llvmpipe):

| pair (fused ← what it replaces) | XTX | K620 | lvp (box) |
|---|---:|---:|---:|
| `fused_gdn_conv_l2` ← conv_silu + 2× l2_norm (3 dispatches) | **0.584** | **0.547** | **0.500** |
| `fused_gdn_conv_l2` ← `native_gdn_conv_silu` (1) | 1.175 | 1.089 | 1.530 |
| `fused_gdn_ab` ← 2× bf16_mmvf + beta_gate + gate (4 dispatches) | **0.466** | **0.869** | **0.636** |
| `fused_gdn_ab` ← `bf16_mmvf_f32` (1) | 0.946 | 1.969 | 1.747 |
| `fused_gdn_step_norm` ← native_gdn_step + native_gdn_out_norm (2 dispatches) | 1.003 | **0.994** | **0.906** |
| `fused_gdn_step_norm` ← `native_gdn_step` (1) | 1.009 | 1.013 | 1.061 |
| `fused_gdn_step_norm` ← `native_gdn_out_norm` (1; NOT like-for-like) | 17.945 | 29.310 | 6.867 |

**Every fused path beats the multi-dispatch chain it replaces (0.398–1.003), with ONE exception at a wash:** the
XTX's `fused_gdn_step_norm` chain pair reads **1.003** (0.0509 vs 0.0507 ms) — the XTX's step is fast and
memory-bound, so the second dispatch (`native_gdn_out_norm`, 0.0028 ms) fits inside the same fence window and the
fusion buys nothing there, exactly as batch 3 measured for the XTX's `scale`+`gdn_step` chain (1.022).  The
fused-vs-`native_gdn_step` pairs (1.009–1.092) are **washes**, as expected — the fused kernel does that same
recurrence PLUS the norm's work in one dispatch.  The fused-vs-`native_gdn_out_norm` rows (6.2–63.7) are
**not a like-for-like fusion ratio and should not be read as one**: the fused kernel's dispatch still does the
whole step (~0.42 ms of 3 MiB state traffic) while the standalone out_norm is a ~0.006 ms 48-workgroup
elementwise pass, so the ratio is the step's cost divided by a 70× smaller kernel.  Both rows are printed so the
reader sees why.

**THE DECOMPOSITION, and the batch-3 finding applied UP FRONT.**  Batch 3 measured a faithful warp-shaped
rendering of the native step (a workgroup-per-column barrier tree) at **1.564× the Arc, 8.33× the iGPU and 43.5×
llvmpipe**, and shipped the coalesced one-thread-per-column form instead.  That result decided all three fused
decompositions without a second rejection: `fused_gdn_conv_l2` keeps the native `conv_silu`'s coalesced
one-thread-per-channel shape and does the per-head norm as a **128-lane HALF-WORKGROUP tree** (a 256-lane group
covers exactly TWO 128-wide heads, each half reducing its own head with 7 barrier rounds); `fused_gdn_step_norm`
keeps the native `step`'s coalesced one-thread-per-column recurrence and does the closing norm as the same
half-workgroup tree; `fused_gdn_ab` renders the CUDA's one-WARP-per-row as one WORKGROUP-per-row (the
non-subgroup analogue), which is a win against its chain (0.398–0.878).  No new barrier-tree variant was built to
reject — the batch-3 measurement stands as the reason, and it is named here rather than a fresh experiment
invented for the record.

**THE FLAG FLIPS, AND WHY THE ORDER WAS FORCED — the interesting part of the batch.**  `native_gdn_enabled()`
gates NINE symbols; all nine now have a shader, so the invariant `case_native_capabilities` asserts
(`flag == "every gated symbol has a built shader"`) demands **TRUE**, and the backend
(`vulkan/src/kernels/native_caps_vk.cpp`) now answers **true**.  The fused paths' ADDITIONAL runtime gates are
engine **SETTINGS**, not capabilities this backend answers, and they differ — which is why the strict invariant
mattered:

* **`fused_gdn_step_norm` is selected by `g_fused_gdn && native_gdn_enabled() && state == 128` (layer.cpp:306) —
  NO `native_bf16_projections` term.**  `g_fused_gdn` **defaults TRUE** in the engine (layer.cpp:42).  So the
  moment the flag flipped true, the fused step+norm path became **reachable on the default configuration**.
  Flipping the flag one batch earlier (with this shader absent) would have made the engine dispatch an unported
  symbol — the precise failure the strict invariant exists to prevent.  The flag therefore could not flip before
  this shader existed, and now it must.
* **`fused_gdn_conv_l2` / `fused_gdn_ab` additionally require `native_bf16_projections` (layer.cpp:247)**, which
  is a HOST SETTING: it defaults **false** and is set from `--native-bf16` / `--native`
  (`generate.cpp:2286` → `layer_set_native_bf16`).  Their BF16 dependency is the `bf16_mmvf_f32` projection that
  setting selects, which IS ported and gated.  So the fused_pre paths are reachable only under `--native-bf16`,
  and nothing they need is unported.  **`g_fused_gdn` and `native_bf16_projections` are settings whose
  dependencies are all ported; the backend answers no capability for either.**

**A CONSEQUENCE CHECKED, not assumed: flipping the GDN flag does NOT make the P6 verifier reachable.**
`layer_verify_compatible()` (layer.cpp:476-491) requires a conjunction including `native_bf16_projections` (false
by default), `g_fused_gr` (false by default) and `native_qsa_indexer_enabled()` (false — unported), so the
verifier is still unreachable and the router/moe `_multi` symbols stay off the forward path.  The cap TU's
comment for `native_router_enabled()`/`native_moe_combine_enabled()` was updated to state the argument with the
GDN flag ON.

**THE CAPS CASE ENFORCES THE TRUTH.**  `case_native_capabilities`'s gdn arm now marks all nine gated symbols
`ported=true` and asserts (1) every ported symbol's `.spv` exists and (2) `native_gdn_enabled()` EQUALS "every
gated symbol is built" — which is now **true** (4/4).  It is still an invariant, not a hard-coded boolean: a
`true` answer cannot hide a deleted shader, and the falsification `native-caps-gdn-false` (answer FALSE while all
nine exist) must FAIL — it does (`FAIL native capabilities: gdn flag 3/4`).

**THE MAP DROPS BY THREE.**  `PORT-MAP.tsv` moved `168 — 72 kernel, 61 host, 35 todo` → **`168 — 75 kernel, 61
host, 32 todo`** (the three `fused_gdn_*` rows are now `kernel` naming their shaders); `check_port_map.py` passes
(`109 shaders built, 90 claimed`) and `make_port_map.py` regenerates the file **byte-identically** (`diff -q`).

**GATE, after the change.  vega:** intel_icd (Arc B70) **432 / 0 / 0** (`run_gate.sh` exit **0**), llvmpipe
**420 / 0 / 3**, radeon_icd (Ryzen iGPU) **423 / 0 / 2** — **+4 verdicts** on every arm (3 fused cases +
1 flag), 0 failed.  The radeon arm's first run in this batch read `budget: independent requery agrees 0/1` (the
documented intermittent flake); a direct re-run of that arm read **423 / 0 / 2** twice.  The Arc's new-case
verdicts: `fused_gdn_conv_l2` 51200/51200 w 2.66e-05, 2560/2560 w 9.46e-06, 1920/1920 w 2.68e-05;
`fused_gdn_ab` 99/99 w 3.12e-07, 11/11 w 1.48e-07, 9/9 w 8.85e-08; `fused_gdn_step_norm` 792580/792580 err/tol
0.183, 132100/132100 w 0.0108, 148612/148612 w 0.0568; gdn flag 4/4.  **Box (`z820b`):** radeon_icd (RX 7900
XTX) **428 / 0 / 1** (the 1 is the pre-existing M8 `prefill split` skip — `run_gate.sh` exits **1** on it), NVIDIA
Quadro K620 **423 / 0 / 2**, llvmpipe **420 / 0 / 3** — 0 failed on every arm, so the box is green with the
documented skip.  The XTX arm's first run also carried the intermittent `budget: independent requery agrees`
flake; two direct re-runs of that arm read **428 / 0 / 1**, the flake cleared.  (NOTE: the brief's sync command
omits `vulkan/`, so the box's `vulkan/src/kernels/native_caps_vk.cpp` must be synced TOO — without it the box
built the OLD `native_gdn_enabled() == false` against the nine new shaders and the flag arm failed 3/4; syncing
`vulkan/` fixed it.  This is called out so the next batch does not re-learn it.)
Every one of the four new falsification injections was run and BIT: `fused-gdn-conv-l2-drop-norm` →
`FAIL fused_gdn_conv_l2 C=10240 qk=32 49149/51200 w 1.93e+05`; `fused-gdn-ab-swap-bf16-halves` →
`FAIL fused_gdn_ab h_v=48 n=2560 3/99 w 25.9`; `fused-gdn-step-norm-silu-not-sigmoid` →
`FAIL fused_gdn_step_norm S=128 h_k=16 h_v=48 788501/792580 w 9.34e+04`; `native-caps-gdn-false` →
`FAIL native capabilities: gdn flag 3/4`.  (`native-caps-gdn-true` is retired — answering true is now the truth.)

**TWO FIXTURE FINDINGS THE GATE PRODUCED, both recorded rather than hidden.**
1. **The dropped-readout-scale "invariance" was WRONG.**  The first case asserted that dropping
   `fused_gdn_step_norm`'s folded `1/sqrt(S)` readout scale does NOT move `y` (because the closing RMS
   renormalises it).  That holds only while `sum oc^2 >> eps*S`; with this fixture's deep decay the readouts are
   small (`sum oc^2 ~ 1e-3` against `eps*S = 1.28e-4`), so the drop moves `y` by a measured **0.109–0.19**
   rel-L1.  The case now asserts it as a **moving margin** (> 0.05) and prints the value; the numeric comparison
   against the rule (which includes the scale) was green throughout on all three arms.
2. **A vector rel-L1 margin was swamped by the branch row.**  `fused_gdn_ab`'s softplus margin read 0 on the
   h_v=4/n=64 and h_v=3/n=512 arms because row 0 (forced onto the `v > 20` branch, where softplus IS the
   identity by the rule) contributed a large `|want|` to the rel-L1 denominator.  Switched to a **per-row max**
   (`max_rel_f`, the faithful "can this fixture see the wrong rule" statement), and the fixture's activation was
   scaled down (`x ~ N(0,0.05)`) so `acc` lands in the softplus-sensitive `|v| < ~3` band on every arm.  The
   device matched the rule on every element of both arms throughout — the FIXTURE was the defect, not the kernel.

## THE PERFORMANCE TIER'S GDN / DELTANET MIXER, the remaining three native kernels — class B batch 3 — **DONE 2026-10-05**

This increment ports the **last three native fast paths** of the model's GDN / DeltaNet mixer — the mixer runs
on **36 of the model's 48 layers** (`gdn_layer`, `src/core/layer.cpp:223`), so it is where the decode step
spends most of its layers — **COMPLETING the six**.  `native_gdn_gate`, `native_gdn_step` and
`native_gdn_out_norm` each replace a legacy kernel already ported and gated, each is **oracled against the
engine's OWN native body** (`src/kernels/cuda/native_gdn_preprocess.cu` / `native_gdn.cu`, not the legacy
rule), each is **MEASURED against that legacy kernel at the same shape on the same device** and, for the one
that fuses a second dispatch, against the **legacy dispatch chain** (`ports/vulkan/bench/`), and
**`native_gdn_enabled()` is deliberately left answering FALSE** with `case_native_capabilities` extended to
enforce that the answer keeps describing what is actually implemented.

**THE THREE SYMBOLS, and the oracle each was transcribed from:**

| symbol (shader) | replaces | the native body's rule (oracle) | case |
|---|---|---|---|
| `native_gdn_gate` | `gdn_gate` | `native_gdn_preprocess.cu`'s `gate_softplus` (`:90-97`): `gate[h] = softplus(alpha[h] + dt[h]) * ssm_a[h]`, the threshold-20 branch `x > 20 ? x : log1p(exp(x))`. **PER-HEAD, one token** (the layer calls it with `heads = ssm_v_heads`, layer.cpp:297); the legacy kernel is per-(token,head) and the legacy branch calls it with `n_tokens = 1` (layer.cpp:300), so at THAT shape the two RULES agree | 3 arms (h_v = 48 / 5 / 300); vs the native rule's double softplus; ssm_a NEGATIVE as `-exp(A_log)`, every third head past the branch; raw-identity margin checked host-side |
| `native_gdn_step` | `gdn_step` | `native_gdn.cu`'s `step` (`:46-81`): the SAME recurrence with the decay FOLDED into the rank-1 update (`s = g*s + k*delta`), the `sk` contract taken against the **UNDECAYED** state, and the `1/sqrt(S)` **readout scale FUSED** into the kernel — which the legacy branch applies in a SEPARATE `scale_inplace` launch (layer.cpp:276) | 2 arms (S=128 h_k=16 h_v=48 / h_k=4 h_v=8; the native wrapper requires S==128); o + state vs the native rule in double; INTERLEAVE-pairing and dropped-scale margins checked host-side |
| `native_gdn_out_norm` | `gdn_out_norm` | `native_gdn_preprocess.cu`'s `out_norm` (`:99-114`): `y = rms_norm(o, eps) * gamma * sigmoid(z)`, ONE RMS per head, eps on the **MEAN** (`partial/S + eps`), `(scale*value)*gamma` then `* sigmoid` — the SAME expression as the legacy kernel; the wrapper requires **cols == 128** and forwards eps DIRECTLY (unlike `l2_norm`'s eps/S) | 3 arms (h_v = 48 / 4 / 3, all cols=128); vs the native rule's double transcription; SiLU-vs-sigmoid margin (the `gdn_parity.cpp` §4 trap) checked host-side; NaN row-guard |

**THE MEASUREMENT — native vs legacy, same shape, same device (`XPAIR` lines; ratio is native/legacy, so < 1.0
means the native kernel is faster).**  The step's pair is a WIN (it does LESS memory traffic than the legacy
kernel: the legacy's first pass STORES the decayed state, the native keeps the undecayed state and folds `g`
into the second pass); the other two are washes, as expected for drop-in replacements:

| pair (native ← legacy) | Arc B70 | Ryzen iGPU | XTX (box) | K620 (box) | llvmpipe (vega/box) |
|---|---:|---:|---:|---:|---:|
| `native_gdn_gate` ← `gdn_gate` | 0.998 | 0.901 | 1.044 | 1.037 | 0.980 / 1.042 |
| `native_gdn_out_norm` ← `gdn_out_norm` | 0.995 | 1.000 | 1.144 | 1.020 | 0.974 / 0.979 |
| `native_gdn_step` ← `gdn_step` | **0.917** | **0.776** | **0.946** | **0.724** | 0.866 / 0.943 |
| `native_gdn_step` ← `scale_inplace`+`gdn_step` (the 2-dispatch chain it replaces) | **0.867** | **0.771** | 1.022 | **0.724** | 0.829 / 0.869 |

`native_gdn_step` wins against the legacy kernel on EVERY GPU (0.724-0.946) and against the two-dispatch chain on
five of the six (0.724-0.869); the exception is the XTX, where the chain reads **1.022** — the XTX's step is fast
and memory-bound (0.098 ms) and the extra `scale_inplace` dispatch (2048 q elements) fits inside the same
fence-clock window, so the fused kernel's advantage is inside run noise there.  Both rows are printed.

**THE STEP'S DECOMPOSITION, AND A MEASUREMENT THAT DECIDED IT.**  The native body uses ONE 32-LANE WARP per
column (four state rows per lane in registers, reduced with `__shfl_xor_sync`).  Subgroup ops are BANNED in this
port (Intel picks the SIMD width per kernel — see `common/wg_reduce.glsl`), so the warp has two non-subgroup
renderings: a **workgroup-per-column barrier tree**, or the **coalesced one-thread-per-column serial walk** the
legacy `gdn_step` port already uses.  The barrier-tree version was BUILT AND MEASURED against the legacy kernel
at this shape — **1.564x the Arc, 8.328x the Ryzen iGPU and 43.501x llvmpipe** (chain 1.471 / 8.078 / 41.597) —
because 6144 workgroups each doing two 8-round barrier trees, with adjacent invocations owning rows `h_v*S`
floats apart (32 cache lines per warp), is a large REGRESSION, not a wash.  So the SHIPPED port keeps the
coalesced serial decomposition (the legacy port's own note: the CUDA's warp/staging shape is a parallelism
strategy, not the rule) and carries the native ARITHMETIC and its fused readout scale; the case's oracle is
still the native RULE.  **The measurement is the finding; the barrier-tree variant is recorded here and in
`bench/README.md`, not shipped.**

**THE CAPABILITY DISCIPLINE — `native_gdn_enabled()` STILL answers FALSE, and the case enforces WHY.**  This ONE
flag gates NINE symbols: ALL SIX native GDN kernels — the three of batch 2 and the three here (layer.cpp
:253/266-267/296 and :297/308/324) — AND the three UNPORTED fused paths `fused_gdn_conv_l2` / `fused_gdn_ab` /
`fused_gdn_step_norm` (:250/287/322, the latter additionally gated on `g_fused_gdn` + `native_bf16_projections`).
Answering true would make the engine dispatch a symbol with no shader, so the backend
(`vulkan/src/kernels/native_caps_vk.cpp`) answers **false**.  `case_native_capabilities`'s **gdn arm** asserts the
flag EQUALS *"every gated symbol has a built shader"* — currently false, because the three fused shaders are
absent — AND that all six ported shaders exist (so a `false` cannot hide a deleted shader).  It is an invariant,
not a hard-coded boolean.  **The stricter "every gated symbol" form is kept deliberately** (the comment in the
case states the alternative): a more permissive *"every symbol REACHABLE under the current settings"* reading
WOULD let the flag answer true here, because the fused paths need `g_fused_gdn` + `native_bf16_projections`,
which this backend never sets — but that is a weaker invariant: it would let the engine route to an unported
symbol the moment a setting changed.  Falsified by `native-caps-gdn-true` → `FAIL native capabilities: gdn flag
3/4`.

**THE MAP DROPS BY THREE.**  `PORT-MAP.tsv` moved `168 — 69 kernel, 61 host, 38 todo` -> **`168 — 72 kernel,
61 host, 35 todo`** (the three symbols are now `kernel` rows naming their shaders); `check_port_map.py` passes
(`106 shaders built, 87 claimed`) and `make_port_map.py` regenerates the file **byte-identically** (`diff -q`).

**GATE, after the change.  vega:** intel_icd (Arc B70) **423 / 0 / 0** (`run_gate.sh` exit **0**), llvmpipe
**411 / 0 / 3**, radeon_icd (Ryzen iGPU) **414 / 0 / 2** — **+8 verdicts** on every arm (3 gate + 3 out_norm + 2
step), 0 failed.  Box (`z820b`): radeon_icd (RX 7900 XTX) **419 / 0 / 1**, llvmpipe **411 / 0 / 3**, nvidia_icd
(Quadro K620) **414 / 0 / 2** — **0 failed on every arm**; the one skip is the pre-existing M8 `prefill split`,
so `run_gate.sh` exits **1** there (a skipped case is not a passing one).  The box's cross-arm caught the
documented intermittent `budget: independent requery agrees` on one run of the XTX arm; a re-run read
**419 / 0 / 1**, the flake cleared.  The XTX's new-case verdicts: `native_gdn_gate` 48/48 w 3.38e-07, 5/5,
300/300 w 3.71e-07; `native_gdn_out_norm` 6400/6400 w 6.06e-07, 768/768, 640/640; `native_gdn_step`
792576/792576 err/tol 0.00154, 132096/132096 w 0.000883; gdn flag 4/4.  Every one of the four new falsification
injections was run and BIT: `native-gdn-gate-drop-ssm-a` -> `FAIL native_gdn_gate h_v=48 0/48 worst 8.89`;
`native-gdn-out-norm-silu-instead-of-sigmoid` -> `FAIL native_gdn_out_norm h_v=48 S=128 2321/6400 worst 21`;
`native-gdn-step-drop-readout-scale` -> `FAIL native_gdn_step S=128 h_k=16 h_v=48 786434/792576 worst 2.03e+04`;
`native-caps-gdn-true` -> `FAIL native capabilities: gdn flag 3/4`.

## THE PERFORMANCE TIER'S GDN / DELTANET MIXER, first three native kernels — class B batch 2 — **DONE 2026-10-05**

The mixer runs on **36 of the model's 48 layers** (`gdn_layer`, `src/core/layer.cpp:223`), so it is where the
decode step spends most of its layers.  This increment ports the **first three of its native fast paths** —
`native_gdn_conv_silu`, `native_gdn_l2_norm`, `native_gdn_beta_gate` — each replacing a legacy kernel already
ported and gated, each **oracled against the engine's OWN native body** (`src/kernels/cuda/native_gdn_preprocess.cu`,
not the legacy rule), each **MEASURED against that legacy kernel at the same shape on the same device**
(`ports/vulkan/bench/`), and **`native_gdn_enabled()` is deliberately left answering FALSE** with
`case_native_capabilities` extended to enforce that the answer keeps describing what is actually implemented.

**THE THREE SYMBOLS, and the oracle each was transcribed from:**

| symbol (shader) | replaces | the native body's rule (oracle) | case |
|---|---|---|---|
| `native_gdn_conv_silu` | `gdn_conv_step` | `native_gdn_preprocess.cu`'s `conv_silu` (`:53-68`): the SAME four-tap conv **PLUS the SiLU in ONE kernel**, writing BOTH the raw and the SiLU output, with the zero-bias fold `sum = __fadd_rn(sum, 0.0f)`; the legacy branch is `gdn_conv_step` -> a D2D copy -> `silu_f32` (layer.cpp:255-257), three launches | 3 arms (C=2560/24/300, d_conv=4); raw+SiLU vs a double transcription of the native rule, slid state **BIT-EXACT**; a terms-derived bound (see the finding below) |
| `native_gdn_l2_norm` | `gdn_l2_norm` | `native_gdn_preprocess.cu`'s `l2_norm` (`:70-84`): sums in **FLOAT**, `rsqrtf(partial/S + eps/S)` (eps on the **MEAN**), then a folded `scale_after = 1/sqrt(S)` — numerically the SAME rule as the legacy, a different arithmetic path | 3 arms (rows 1/16/3, cols=128); vs the native rule's double transcription; near-zero row 0 + NaN-padded tail; **HONEST: the two RULES are the same, so the printed `worst` IS the arithmetic gap** |
| `native_gdn_beta_gate` | `gdn_beta_gate` | `native_gdn_preprocess.cu`'s `beta_sigmoid` (`:86-89`): `beta[i] = 1/(1+expf(-beta[i]))` — the **SAME expression** as the legacy `sigmoid_f` | 48 heads spanning ~0/mid/~1; vs the engine's double sigmoid; the raw identity reading is checked host-side to move |

**THE MEASUREMENT — native vs legacy, same shape, same device (`XPAIR` lines; ratio is native/legacy, so < 1.0
means the native kernel is faster).**  This includes the pair that is NOT a win:

| pair (native ← legacy) | Arc B70 | Ryzen iGPU | XTX (box) | K620 (box) | llvmpipe (both) |
|---|---:|---:|---:|---:|---:|
| `native_gdn_conv_silu` ← `gdn_conv_step` | **0.938** | **0.909** | **0.694** | **0.936** | 0.881 / 0.949 |
| `native_gdn_conv_silu` ← `gdn_conv_step`+`silu_f32` (the 2-dispatch chain it replaces) | **0.574** | **0.764** | **0.544** | **0.720** | 0.501 / 0.481 |
| `native_gdn_l2_norm` ← `gdn_l2_norm` | 0.997 | 1.013 | 0.938 | 0.942 | 0.981 / 1.039 |
| `native_gdn_beta_gate` ← `gdn_beta_gate` | 1.011 | 1.045 | 1.068 | 0.865 | 1.016 / 1.002 |

**The conv+SiLU kernel is the batch's real win, and it is larger against what the branch actually runs.**
Per DISPATCH the native writes one extra output and computes the SiLU, yet it is still 6-31% faster than the
plain `gdn_conv_step` (0.694-0.938).  Measured against the legacy **chain** the layer really issues
(`gdn_conv_step` then `silu_f32`, two dispatches — a D2D copy cannot be timed as a kernel), the native one
dispatch is **1.3-2.1x faster** (0.481-0.764).  The legacy conv is one thread per channel with a serial 4-tap
dot; the native keeps that shape but fuses the second kernel's work into the same thread.

**`native_gdn_l2_norm` and `native_gdn_beta_gate` are WASHES, and that is the honest measurement.**
`l2_norm` sits at 0.938-1.039 and `beta_gate` at 0.865-1.068 across six devices — both are the same
work per element as the legacy kernel (one workgroup per 128-wide row; one thread per head), with no
algorithmic difference to win, so the arithmetic change (f32 sums / a folded scale / the same sigmoid) buys
no throughput.  **A native kernel is not required to be faster; recorded as measured, not tuned**
(`bench/README.md` has the full tables and the reading).

**A FINDING THE CASE ITSELF PRODUCED, fixed in the FIXTURE's bound rather than by loosening a tolerance.**  The
first run of `case_native_gdn_conv_silu` read **12798/12800 on the Ryzen iGPU (worst 1.09e-05)** while every
channel's ABSOLUTE deviation sat at the f32 ulp of its terms — two channels whose four-tap sums partially
CANCEL, which is exactly the case a purely relative bound fails on.  The comparison is now the port's
terms-derived bound (`gemv_bound`: `rtol·|want| + 16·2⁻²⁴·Σ|terms|`), and the same run then reads green with
the worst relative deviation unchanged — the kernel was right and the assertion was wrong (the port's own
"bound a reduction by its TERMS" rule, `vulkan-compute-shader-porting`).

**THE CAPABILITY DISCIPLINE — `native_gdn_enabled()` answers FALSE, and the case enforces WHY.**  This ONE flag
gates NINE symbols: the three ported here (`native_gdn_conv_silu`/`_l2_norm`/`_beta_gate` at
layer.cpp:253/266-267/296) AND six unported ones (the remaining native GDN kernels `native_gdn_gate`,
`native_gdn_step`, `native_gdn_out_norm` at :297/308/324, and the three fused paths `fused_gdn_conv_l2`,
`fused_gdn_ab`, `fused_gdn_step_norm` at :250/287/322 — the latter additionally gated on `g_fused_gdn` and
`native_bf16_projections`).  Answering true would make the engine dispatch a symbol with no shader, so the
backend (`vulkan/src/kernels/native_caps_vk.cpp`, new GDN block) answers **false**.  `case_native_capabilities`
gains a **gdn arm** that asserts the flag EQUALS *"every gated symbol has a built shader"* — currently false,
because the six unported gated shaders are absent — AND that this batch's three ported shaders exist (so a
`false` cannot hide a deleted shader).  It is an invariant, not a hard-coded boolean: when the remaining six
land, the flag has to be revisited.  Falsified by `native-caps-gdn-true` → `FAIL native capabilities: gdn flag
3/4`.

**THE MAP DROPS BY THREE.**  `PORT-MAP.tsv` moved `168 — 66 kernel, 61 host, 41 todo` -> **`168 — 69 kernel,
61 host, 38 todo`** (the three symbols are now `kernel` rows naming their shaders); `check_port_map.py` passes
(`103 shaders built, 84 claimed`) and `make_port_map.py` regenerates the file **byte-identically** (`diff -q`).

**GATE, after the change.  vega:** intel_icd (Arc B70) **415 / 0 / 0** (`run_gate.sh` exit **0**), llvmpipe
**403 / 0 / 3**, radeon_icd (Ryzen iGPU) **406 / 0 / 2** — **+8 verdicts** on every arm (3 conv + 3 l2_norm +
1 beta_gate + 1 gdn-flag), 0 failed.  Box (`z820b`): radeon_icd (RX 7900 XTX) **411 / 0 / 1**, llvmpipe
**403 / 0 / 3**, nvidia_icd (Quadro K620) **406 / 0 / 2** — **0 failed on every arm**; `run_gate.sh` exits **1**
there only for the pre-existing M8 `prefill split` skip.  Every one of the four new falsification injections
was run and BIT: `native-gdn-conv-silu-drop-silu` -> `FAIL native_gdn_conv_silu C=2560 d_conv=4 11509/12800`;
`native-gdn-l2-norm-drop-folded-scale` -> `FAIL native_gdn_l2_norm r=1 c=128 264/392 worst 10.3`;
`native-gdn-beta-gate-sign-flip` -> `FAIL native_gdn_beta_gate (sigmoid) 0/48 worst 7.2e+10`;
`native-caps-gdn-true` -> `FAIL native capabilities: gdn flag 3/4`.

## THE PERFORMANCE TIER'S FIRST FOUR KERNELS — class B, the NATIVE fast paths — **DONE 2026-10-05**

The port is correct-but-slow **by construction**, and the throughput harness (`ports/vulkan/bench/`) landed
last increment so a performance claim can carry a before/after.  This increment is the first to use it: the
**four class-B capability-gated native fast paths** (`plan/DECODE-PATH-TRIAGE.md`'s class B) are PORTED, GATED,
and **MEASURED against the legacy kernel each replaces, at the same shape on the same device** - and the
Vulkan backend now **answers the capability checks itself** so the engine can take the native branch.

**THE FOUR SYMBOLS, and the oracle each was transcribed from** (the engine's OWN native body, not the legacy
kernel's rule; where the native arithmetic differs, the case says so and MEASURES the gap):

| symbol (shader) | replaces | the native body's rule (oracle) | case |
|---|---|---|---|
| `native_rope_apply` | `rope_neox_apply` | `native_rope.cu`'s `apply<false>`: angle computed ON DEVICE in f32 (`powf`/`rope_scaled_angle`), one thread per (row, PAIR) — NOT the legacy host-built float64 table | 3 arms (hd=256/128, none + YaRN factor 2); rotation vs a host transcription of `rope_scaled_angle` at the parity file's 3e-3 row-relative bar, tail BIT-EXACT, NEOX-vs-adjacent and partial-vs-full margins checked |
| `native_router_top10` | `router_top10` | `native_router.cu`'s `route`: softmax + sum in **plain FLOAT** (`expf`, `sum +=`, `1/warp_sum`), not the legacy's double-Kahan | 5 arms (random, x8, all-equal ties, 12-way tie, dominant); ids EXACT vs `router_top10_parity`'s double transcription, weights rel 1e-5 |
| `native_moe_combine` | `moe_combine` | `native_moe.cu`'s `combine`: **pure f32**, first term a PRODUCT (`parts*weights[0]`), then mul-add, shared added PLAIN | 4 shapes × shared on/off; oracle is the native float expression, bounded by the row's TERMS; the double transcription is reported beside it |
| `native_qsa_rms_norm_weighted` | `rms_norm_weighted` | `native_qsa.cu`'s `norm`: block-per-row, `scale*x*gamma` (legacy is `(x*gamma)*inv`) | 3 shapes × in-place/out-of-place (the engine's call aliases input and output); vs a double transcription of the rule, eps-on-the-mean, NaN-padded tail |

**THE MEASUREMENT — native vs legacy, same shape, same device (`ports/vulkan/bench/`, `XPAIR` lines; ratio is
native/legacy, so < 1.0 is faster).**  This is the increment's deliverable, and it includes the pair that is
NOT a win:

| pair | Arc B70 | Ryzen iGPU | XTX (box) | K620 (box) | llvmpipe |
|---|---:|---:|---:|---:|---:|
| `native_rope_apply` ← `rope_neox` | **0.301** | **0.040** | **0.118** | 0.152 | 0.457 |
| `native_router_top10` ← `router_top10_f32` | **0.078** | **0.078** | **0.082** | 0.373 | 1.238 |
| `native_moe_combine` ← `moe_combine_f32` | 0.998 | 0.988 | 1.018 | 1.056 | 0.986 |
| `native_qsa_rms_norm_weighted` ← `rms_norm` | 1.007 | 1.117 | 0.988 | 1.795 | 0.988 |

**RoPE is a real 3.3× on the Arc, 8.5× on the XTX and 25× on the RADV iGPU; the router 12.8× on the Arc,
12.2× on the XTX and 12.8× on the iGPU** (the legacy rope is one-thread-per-row → 2 workgroups; the legacy
router sums 512 experts on ONE lane with Kahan).  **`native_moe_combine` is a WASH and
`native_qsa_rms_norm_weighted` is NEUTRAL on the Arc, 0.988 on the XTX and 12% slower on the iGPU (and 1.795×
on the K620)** — these are elementwise / same-shape kernels with no algorithmic difference to win, and the
K620 shows the native block-per-row tree's shared-memory traffic can HURT.  A native kernel is not required to
be faster.  **Recorded as measured, not tuned** (`bench/README.md` has the full table and the reading).

**THE CAPABILITY WIRING — the part that lets the engine take the native branch.**  A Vulkan build compiles none
of `src/kernels/cuda/native_*.cu`, so it must answer these checks itself.  **`vulkan/src/kernels/native_caps_vk.cpp`**
(new) answers them from **what this backend has actually implemented**, and the answer is a **symbol-at-a-time
truth, not a blanket `true`**:

* `native_rope_enabled()` → **true** (only `native_rope_apply` is gated by it)
* `native_router_enabled()` → **true** (reachable set is `native_router_top10`; `_multi` is verifier-only)
* `native_moe_combine_enabled()` → **true** (same)
* `native_qsa_enabled()` → **FALSE.**  This ONE flag ALSO gates the UNPORTED `native_qsa_gate_apply`
  (layer.cpp:1010, the MAIN QSA path, 12 of 48 layers).  Answering true would make the engine dispatch a symbol
  with no shader — so the port stays on the legacy branch until that sibling lands, even though
  `native_qsa_rms_norm_weighted` itself is ported and gated.

**HOW TO CHECK IT:** the gate's `case_native_capabilities` calls the four getters, requires the exact answers
above, and requires each ported symbol's `.spv` to exist - so a capability cannot answer true for a deleted
shader.  The setter (`native_*_set_enabled`, the engine's CLI plumbing) is a no-op here: the backend reports its
own implementation, so a `--native` launch cannot talk it into selecting an unported symbol.  Falsified by
`gates/inject-verify.sh native-caps-qsa-true` → `FAIL native capabilities 2/4`.

**THE README FIXES this increment found (both real, both small).**  (1) `run_gate.sh`'s barrier-count `sed` was
greedy and read a two-digit count as its last digit, so a kernel with 11 `OpControlBarrier` read as 1 and
failed the shared-memory arm while printing 11 in its census line (`native_router_top10` is the first such
kernel); now `grep -oE '[0-9]+ OpControlBarrier'`.  (2) The NEW cases had to be appended at the END of the
gate's `main()`, because the gate has ONE shared RNG (`g_rng(11)`) and a new case placed earlier moves the
fixture a LATER existing case sees — measured as `gr_write` going borderline-fail on lvp/radeon until the
cases were moved after it.

**The port map drops by four:** `168 — 62 kernel, 61 host, 45 todo` → **`168 — 66 kernel, 61 host, 41 todo`**;
`check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.

**GATE, after the last commit:** vega **Arc 407/0/0** (`run_gate.sh` exit 0), llvmpipe 395/0/3, radeon-iGPU
397/1/2 (the 1 is the documented intermittent `budget: independent requery` flake); box `z820b` **RADV XTX
403/0/1** (the 1 is the pre-existing M8 `prefill split` skip), lvp 395/0/3, nvidia K620 398/0/2 — **0 failed on
every arm on both boxes.**  Every one of the four new cases' falsification injections was run and BIT
(`native-rope-adjacent-pairing`, `native-router-top10-tie-high-index`, `native-moe-combine-drop-shared`,
`native-qsa-rms-norm-eps-on-sum`, `native-caps-qsa-true`).

## THE PERFORMANCE TIER'S HARNESS LANDED — a throughput benchmark with a real baseline — **DONE 2026-10-05**

The port is correct-but-slow **by construction** (every fast path is dodged by a capability contract:
`native_gdn_enabled() == false`, `gr_set_native_mmvf(false)`, `layer_set_fused_gr(false)`), so the native/fused
kernels are the next tier — and **no performance claim could be made or checked, because the port had no
benchmark at all**.  This increment adds one, `ports/vulkan/bench/`, and hands back the baseline the fused
kernels will be judged against.  It does **not** touch the numeric gate: `gates/run_gate.sh` +
`harness/vk_gate.cpp` remain the correctness authority, and the gate was re-run green after this commit.

**WHAT IT IS.**  `bench/vk_bench.cpp` — a runner that loads the port's own `.spv` kernels and times them per
device, built on the port's own device layer (`harness/vk_compute.*`).  A batch of `K` dispatches of one
kernel is recorded into a single command buffer (one compute→compute barrier between dispatches), warmed 2–3
replays, then timed over `reps` **replays of the batch**; the reported figure is the **median per-dispatch
time** with min/max beside it.  Timing is **wall clock around the fence** (`vkQueueFences` via
`replay_recorded`), because the device layer exposes no timestamp/`VkQueryPool` path — so the number includes
one submit + fence wait per batch, amortised over `K`, and its floor is ~5–15 µs on the fast GPUs.  What is
timed is the **kernel dispatch only** (inputs resident; no transfer in the loop).  `bench/run_bench.sh`
compiles the measured kernels from source into `bench/build/spv/` and runs the binary under **every Vulkan
ICD** that reports a device, then prints a cross-ICD comparison.  The full method, the run command, the
per-kernel shapes and units, and the two baseline tables are in **`bench/README.md`**.

**THE BASELINE (medians, reps=9, sampler 5), GMAC/s where the kernel does a reduction.**  vega Arc B70:
`gdn_step` 0.0469 ms (50.3 GMAC/s); `iq2s_mmvq` n_out=2048 0.0464 ms (112.9 GMAC/s); `iq_dequant` BF16
0.0130 ms; `quantize_q8_K` 0.0875 ms; `sampler_kernel_f32` **546.9 ms**.  Box XTX: `gdn_step` 0.1050 ms
(22.5 GMAC/s); `iq2s_mmvq` n_out=2048 0.0198 ms (**264.3 GMAC/s**, fastest on every kernel); `quantize_q8_K`
0.0732 ms; `sampler_kernel_f32` **202.0 ms**.  The GDN chain's other five kernels are 0.0026–0.030 ms
everywhere.  Full tables per device (Arc, Ryzen iGPU, llvmpipe / XTX, K620, llvmpipe) in `bench/README.md`.

**THE FIRST PROBLEM THE BASELINE NAMES: the one-block sampler.**  202 ms/token on the XTX and 547 ms on the
Arc.  `sampler_kernel_f32`'s top-k is `k` rounds, each a block-argmax over the **whole** vocabulary with an
inner loop over the taken ids and the history — cost ~ `k · vocab · (k + history)`, three orders of magnitude
above every other kernel here.  It is the **one-block** path; the engine's default is the **split** sampler
(4096-logit partitions), which the port has (`sampler_split.comp`) but this harness does not yet measure.
This is the performance tier's first target.

**THE HARNESS PROVES IT MEASURES SOMETHING REAL, two ways, both in the raw output.**  (1) Cross-ICD: the same
binary under every ICD orders physically on both boxes — on the box `iq2s_mmvq` XTX 0.0198 vs llvmpipe 23.78 =
**1201×**; on vega `iq2s_mmvq` Arc 0.0464 vs llvmpipe 6.2754 = **135×**.  (2) 4×-work sizing: the same kernel
at 1024 vs 256 superblocks / n_out 2048 vs 512 scales ≈4× exactly where the work exceeds the timer's floor —
Ryzen iGPU `iq2s` **3.90× for 4× work**, K620 **3.57×**, llvmpipe **3.60×**.  **Stated negative:** on the
fastest GPUs the sizing arm reads **~1×** (`iq_dequant` Arc 0.0145→0.0158, XTX 0.0155→0.0162) because both
sizes sit below the fence-clock floor; the cross-ICD arm, not the sizing arm, is the primary evidence.  That
is the honest limit of a fence-clock timer, and it is why a device-timestamp path is the obvious next step.

**TWO FINDINGS THE RUN ITSELF PRODUCED.**  (a) The **Ryzen iGPU (RADV) hard-recovered** on the harness's
first form of the sampler row — a *batch of 8* live full-vocabulary sampler dispatches in one command buffer.
The harness now runs the sampler **last** and records **one dispatch per batch** for it, and it completed on
every ICD; a launch-shape hazard, contained and documented, not a kernel defect.  (b) **A device-layer
robustness gap:** on the box's **RADV / Mesa 26.0.8** an exhausted descriptor pool returns
**`VK_ERROR_FRAGMENTED_POOL`**, and `Ctx::set_alloc` listed only `OUT_OF_POOL_MEMORY` (what vega's Mesa
25.2.8 returns), so the grow-on-demand path never fired and the XTX arm aborted at the first pool exhaustion.
Fixed in `harness/vk_compute.cpp` (both codes grow the pool) — the numeric gate never fills a pool, so this
changes no gate verdict; it makes the header's "no fixed ceiling" promise true on a second Mesa version.

**GATE TOTALS AFTER THE COMMIT (same tree content as the runs below).**  vega: intel (Arc) **384/0/0**,
llvmpipe **372/0/3**, radeon-iGPU **374/1/2** — the single failure is the documented intermittent
`budget: independent requery agrees` flake (that arm has read 375/0/2 on other runs).  Box `z820b`: primary
arm (XTX) **380/0/1** on re-run, lvp **372/0/3**, nvidia (K620) **375/0/2** — the primary arm's first run
read 379/1/1, the same `budget` flake, cleared on re-run; 0 failed on every arm; the only skip is the
pre-existing M8 `prefill split`.  These are identical to the pre-increment readings — the numeric gate is
undisturbed.

## CLASS A IS CLOSED — `indexer_key_append`, `gr_write`, `gr_read`, and M-A RE-DEFINED — **DONE 2026-10-05**

This increment lands the last three class-A forward-path kernels of `plan/DECODE-PATH-TRIAGE.md`, SETTLES the
`gr_read` / `fused_gr_read` question that triage left open, and **re-defines the milestone** (below), because
`todo = 0` was never achievable or meaningful.

**1. `indexer_key_append`** — the QSA indexer pair's LEGACY member (the raw tail, then on a block completion
`pooled[b] = rope(rms_norm(mean(raw[b*r..]), w_kn), pos_base + b*r)` and the spare slot
`pooled[n_bid] = rope(rms_norm(raw[0]), 0)`). Rule `qsa.cu:155-243`/`:722-739`; contract `qsa.hpp:222-250`;
call site `layer.cpp:948`. **Its own capability contract, stated:** the layer calls
`native_qsa_indexer_append` when `native_qsa_indexer_enabled()`, else this kernel — the indexer's OWN check,
separate from `native_qsa_enabled()`, and the one `layer_verify_compatible()` reads. The backend answers
**`native_qsa_indexer_enabled() == false`**. Shader `indexer_key_append.comp`; oracle the triage's own reference
in double. Case 2 arms (`idx_dim/r/n_rot` = 128/4/64, 32/4/8). **HONEST GAP:** the CUDA reduces the sum of
squares in DOUBLE; the target has no `shaderFloat64`, so this port accumulates in F32 and the case MEASURES the
gap (pooled worst err/tol 4.94e-02 on the Arc, 7.87e-02 on lvp/radeon, against a bound of 1.0; spare key worst
1.02e-07 against 1e-5) — so the spare key is NOT bit-exact here, unlike the CUDA's. Falsified by
`indexer-key-append-rotate-last` → `FAIL 274/384 worst 9.46e+05`.

**2. `gr_write`** — the hyper-connection WRITE, the one entry reached on BOTH branches of the fused/unfused
choice (`layer.cpp:1261`/`:1329` unfused, `:1195`/`:1332` fused), so no dodge exists. Rule `gr.cu:283-297`;
contract `gr.hpp:114-120`. `out[i] = R[i] + block_out[d]·2·sigmoid(inject[c]/hc)`, in place. Shader
`gr_write.comp` (no barrier, no subgroup op — recomputing `w[c]` per element is the same float the CUDA stages
into shared). Case 3 arms, and it asserts the gr_parity PROPERTY 6 **bit-exactly**: a ZERO injection gives
`w = 1` exactly, i.e. `out == R + block_out`. **Measured: PASS (20480/20480, 384/384, 64/64), worst err/tol
2.4e-01, the numerical worst, and the property arm exact.** Falsified by `gr-write-drop-two-centring` →
`FAIL worst 6.68e+06`.

**3. `gr_read`** — the hyper-connection READ, the unfused five-stage chain (norm → down+silu → gate → mean →
inject). Rule `gr.cu:135-297`/`:344-411`; contract `gr.hpp:79-112`; call sites `layer.cpp:1255`/`:1278`. FIVE
shaders: `gr_norm`, `gr_down`, `gr_gate`, `gr_mean`, `gr_inject` (the map row names all five). Oracle:
`gr_parity.cpp`'s `reference` in double, DEFAULT (BF16) activation contract. The case SNAPSHOTS EACH STAGE, fed
the DEVICE's own input for that stage, with the port's terms-derived bound (`rel·|want| + 16·2^-24·Σ|terms|`) —
both forced by the cross-implementation arm (see below). **Measured: PASS 23364/23364 on the Arc, worst err/tol
3.0e-01; lvp/radeon 0.331.** Falsified by `gr-read-mean-vs-sum` → `FAIL 20804/23364 worst 2.99e+03`.

**SETTLED: `gr_read` and `fused_gr_read` were BOTH mis-kinded `host` — a kind-table FALSE NEGATIVE.** `gr_read`
(`gr.cu:344`) launches five kernels; `fused_gr_read` (`fused_gr.cu:1168`) launches `gr_down_kernel`/`gr_up_kernel`.
`host / a workspace read` describes `gr_workspace_init`/`gr_workspace_bytes`, not the read entry. **So the honest
device-op count is 54, not 52.** Which is on the forward path is decided by
`fused = g_fused_gr && fused_gr_supported(...)` (`layer.cpp:1188`) — and `fused_gr_supported()` is a pure
GEOMETRY predicate (`fused_gr.cu:1164`: n_embd 2560/hc 4/hc_lr 320) that is TRUE at the artifact's geometry, so it
is NOT a capability a backend may answer false. The selecting input is `g_fused_gr` (from `gr_native_mmvf`, which
`--native` sets), i.e. **the shipped launch selects the FUSED read**. The port therefore closes the pair by
CONTRACT — the same shape as every other contract here (implement the legacy member, force the flag): the
backend's init calls **`gr_set_native_mmvf(false)`** and **`layer_set_fused_gr(false)`**, so the layer takes
`gr_read` + the legacy `gr_write`, and `fused_gr_read` leaves the path. Its map row is corrected from `host` to
`todo` with that reason; it is NOT ported. **Residual risk, stated: if the product must run the SHIPPED
`--native` selection bit for bit, the class-A member is `fused_gr_read`, not `gr_read`.** Full argument in
`plan/DECODE-PATH-TRIAGE.md` ("THE KIND-TABLE FALSE NEGATIVE").

**THE MILESTONE, RE-DEFINED.** `todo = 0` is neither achievable nor meaningful — the map covers every
kernels-namespace symbol the decode path reaches, including the `native_*` siblings of ported legacy members and
the verifier/MTP/tooling helpers. **M-A (re-defined):** *every symbol the forward path reaches ON THE BRANCH THE
CAPABILITY CONTRACT SELECTS has a shader and a gated case*, under the contract
`native_gdn_enabled() == false` + `native_qsa_enabled() == false` + `native_qsa_indexer_enabled() == false` +
`native_rope_enabled() == false` + `native_router_enabled() == false` + `native_moe_combine_enabled() == false`
+ `gr_set_native_mmvf(false)` + `layer_set_fused_gr(false)`. **Class-A remaining: 0. Is it MET? YES, with two
soft edges stated, not hidden:** (i) the shipped `setup.py` writes `--spec 4 --mtp`, so the MTP drafter runs;
its symbols are class D by the brief's own definition ("separable from a correct first token") — **a judgement,
not a measurement**; (ii) `gr_read` vs `fused_gr_read` above. The full statement, the 45-row decomposition
(`11 capability-off + 4 B + 8 C + 22 D`) and the 10-implementation table are in
`plan/DECODE-PATH-TRIAGE.md` → "THE RE-DEFINED MILESTONE M-A".

**THE MAP MOVES BY FOUR ROWS.** `PORT-MAP.tsv` `168 — 59 kernel, 63 host, 46 todo` → **`168 — 62 kernel, 61 host,
45 todo`** (`indexer_key_append` and `gr_write` todo→kernel; `gr_read` host→kernel; `fused_gr_read` host→todo);
`check_port_map.py` passes and `make_port_map.py` regenerates the file BYTE-IDENTICALLY.

**THE CROSS-IMPLEMENTATION ARM EARNED ITS KEEP TWICE, and both fixes are the port's own documented rules.** The
first run was green on the Arc and FAILED on llvmpipe AND RADV (identical numbers → deterministic, not driver
noise): (a) `gr_read` — comparing a bf16-ROUNDED `lo` against an unrounded double oracle puts the odd element a
whole bf16 ulp away whenever the two sides straddle a boundary (measured: 19 flips on RADV's f32 `lo`), and a
near-zero `mixed` then moved **9.7 RELATIVE**; fixed by feeding each stage's oracle the DEVICE's own input and
bounding it by the stage's TERMS. (b) `indexer_key_append` — a component whose mean CANCELS carried a large
relative error that was entirely the shared mean's, so the bound must carry the MEAN's term scale
(`inv·|w_kn[d]|·Σ_j|raw[j][d]|/r`), derived, not fitted. And the lvp/radeon-specified **first box run** then
found a THIRD defect in the FIXTURE: the "rotate at the block's LAST cell" margin was diluted by the 64
UNROTATED dims to ~0.04-0.06, so it straddled its 0.05 bar and flipped with the RNG stream (a case before it
skips on one ICD and runs on another, moving `g_rng`) — the arm was decorative on one device and not another.
The margin is now measured where the rotation acts (dims 0..n_rot-1): **1.83e-01 on the Arc, 1.64e-01 on
lvp/radeon**.

**GATE TOTALS, after the change. vega:** intel_icd (Arc B70) **384 / 0 / 0**, llvmpipe **372 / 0 / 3**, radeon_icd
(Ryzen iGPU) **375 / 0 / 2** — exit **0**; the 3 / 2 skips are pre-existing. Box (`z820b`): radeon_icd (RX 7900
XTX) **380 / 0 / 1**, llvmpipe **372 / 0 / 3**, nvidia_icd (Quadro K620) **375 / 0 / 2** — **0 failed on every
arm**; `run_gate.sh` exits **1** there only for the pre-existing M8 `prefill split` skip.  A later full-gate run
on vega read the radeon-iGPU arm **374 / 1 / 2** — the single failure is the DOCUMENTED intermittent
`budget: independent requery agrees` flake on that integrated device (it also read clean, 375/0/2, in the run
immediately before), recorded rather than chased.

**CLASS-A WORK REMAINING: none.** `fused_gr_read` (a real device op the GR contract removes), `fused_gr_read_multi`
and the `_multi`/`gdn_conv_commit` verify family are class C/D; `fused_gr_read_multi` is class D and still `todo`.

## THE GDN MIXER CHAIN IS COMPLETE, plus the first QSA gate member — class A #4-6 — **DONE 2026-10-05**

The previous increment landed the mixer's first three kernels (`gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`)
under the branch policy `native_gdn_enabled() == false`. This one lands the next two — **`gdn_step`** (the
delta-rule state update) and **`gdn_out_norm`** (the closing norm) — which **complete the GDN / DeltaNet mixer
chain**, plus one member of the QSA gate pair, **`qsa_gate_apply_f32`**. All three are class A of
`plan/DECODE-PATH-TRIAGE.md`.

**THE GDN CHAIN IS COMPLETE UNDER THE CONTRACT — checked symbol by symbol, not assumed.** `gdn_layer`
(`src/core/layer.cpp:244-327`) runs `conv -> l2_norm -> scale -> beta/gate -> step -> out_norm`. With
`native_gdn_enabled() == false` the layer takes the legacy `else` of every pair, and the three fused paths
(`fused_gdn_conv_l2` `:250`, `fused_gdn_ab` `:287`, `fused_gdn_step_norm` `:322`) are gated on
`g_fused_gdn && native_gdn_enabled() && …` (`:247`, `:306`), so they leave the forward path. Every GDN-namespace
symbol the chain then reaches has a shader: `gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`, `gdn_gate`,
`gdn_step`, `gdn_out_norm` — all `kernel` rows. The chain's remaining calls are the SHARED primitives
`silu_inplace` (`silu_f32`), `scale_inplace` (`scale`), `f32_to_bf16_bulk` (`f32_to_bf16`) and
`quantize_q8_0` / `quantize_q8_K`, already `kernel` rows. **No further GDN symbol is reachable while the
fused/`_multi` variants are off:** the `_multi` / `gdn_conv_commit` family (`gdn_ab_multi`, `gdn_conv_commit`,
`gdn_conv_l2_multi`, `gdn_step_norm_multi`) is class D, reached only from `verify.cpp`, and `gdn_gate` is the
pair member the port already had.

**THE BRANCH POLICY FOR THE QSA PICK — its own decision, stated.** `qsa_gate_apply_f32` is the legacy member of
the QSA GATE pair: `qsa_layer` (`layer.cpp:1009-1012`) calls `native_qsa_gate_apply` when `native_qsa_enabled()`,
else THIS kernel, and the native sibling is equally unported. The backend answers **`native_qsa_enabled() ==
false`** — the same shape as the GDN contract, and that flag also gates the class-B `native_qsa_rms_norm_weighted`
dodge. The QSA INDEXER pair (`native_qsa_indexer_append` / `indexer_key_append`) has its OWN capability check
(`native_qsa_indexer_enabled()`) and is a separate increment; it stays class-A `todo`. `layer_verify_compatible()`
(`layer.cpp:476-486`) demands `native_gdn && g_fused_gdn` and `native_qsa_indexer_enabled()`, so answering these
off disables the P6 verifier (class D) and a `--spec 0` run is unaffected.

**THE THREE, each with its oracle from the engine's own rule** (shaders + cases; no invented definition):

* **`gdn_step`** — the delta-rule state update, one thread per `(h, j)` column, in place, no barrier. Rule
  `gdn.cu:78-93`; contract `gdn.hpp:44-57`. Oracle: the engine's own `ref_step` (`gdn_parity.cpp` §1),
  transcribed into the **DEVICE** layout `(S, h_v, S)` in double. Case `case_gdn_step`, 3 arms
  `S/h_k/h_v = 128/16/48`, `16/4/8`, `8/2/4`; pins the two named traps host-side (MODULO vs INTERLEAVE head
  pairing; decay BEFORE vs after the update) and the state layout (the fixture encodes each cell's coordinates).
  **Measured: PASS 792576/792576 worst err/tol 0.00733, 2176/2176 w 0.00112, 288/288 w 0.000345** (bound
  2e-4 rel + 1e-5 abs). Falsified by `gates/inject-verify.sh gdn-step-head-pairing` (INTERLEAVE) →
  `FAIL gdn_step S=128 h_k=16 h_v=48 66887/792576 worst 6.13e+04`.
* **`gdn_out_norm`** — `y = rms_norm(o) * ssm_norm * sigmoid(z)`, ONE RMS per head. Rule `gdn.cu:132-149`; contract
  `gdn.hpp:90-95`. Oracle: `gdn_parity.cpp` §4, double. Shader `shaders/gdn_out_norm.comp` (one workgroup per
  head, the barrier-tree reduction in `common/wg_reduce.glsl`; added to `run_gate.sh`'s barrier-census whitelist).
  Case `case_gdn_out_norm`, 3 arms `h_v/S = 48/128`, `4/16`, `3/8`; pins SIGMOID vs SiLU and the eps on the
  **MEAN** (`sum/S + eps`, the OPPOSITE convention to `gdn_l2_norm`) — its distinguishing rule, read from this
  file's own source; a NaN-padded tail + two surplus groups make a missing row guard DETECTED. **HONEST LIMIT:**
  the CUDA sums `o²` in DOUBLE; the target has no `shaderFloat64`, so the port sums in F32 and the case MEASURES
  the gap (the `gdn_l2_norm` form). **Measured: PASS 6400/6400 worst 8.2e-07, 96/96 w 2.41e-07, 40/40 w 2.18e-07.**
  Falsified by `gdn-out-norm-eps-on-sum` → `FAIL gdn_out_norm h_v=48 S=128 2321/6400 worst 0.912`.
* **`qsa_gate_apply_f32`** — `out = attn * sigmoid(q_full[h*2*head_dim + head_dim + d])`. Rule `qsa.cu:793-810`;
  contract `qsa.hpp:314-324`. Oracle: the engine's own `ref_gate` (`qsa_parity.cpp`), double. Case
  `case_qsa_gate_apply_f32`, 3 arms `n_head/head_dim = 24/256`, `4/12`, `2/8`; pins the gate from the **SECOND**
  half (not the first — the split is not element-interleaved) and SIGMOID vs SiLU, both checked host-side to move
  the fixture. **HONEST LIMIT:** the CUDA forms the product and the sigmoid in DOUBLE; the port in F32, gap
  measured. **Measured: PASS 6152/6152 worst 7.88e-07, 56/56 w 7.64e-07, 24/24 w 7.42e-07.** Falsified by
  `qsa-gate-first-half` → `FAIL qsa_gate_apply_f32 n_head=24 head_dim=256 8/6152 worst 4.18e+10`.

**THE MAP DROPS BY THREE.** `PORT-MAP.tsv` moved `168 — 56 kernel, 63 host, 49 todo` → **`168 — 59 kernel, 63
host, 46 todo`** (the three symbols are now `kernel` rows naming their shaders); `check_port_map.py` passes and
`make_port_map.py` regenerates the file BYTE-IDENTICALLY (`diff -q` against a copy).

**GATE TOTALS, after the change. vega:** intel_icd (Arc B70) **376 / 0 / 0**, llvmpipe **364 / 0 / 3**,
radeon_icd (Ryzen iGPU) **367 / 0 / 2** — the 3 / 2 skips are pre-existing (coopmat + the M8 prefill split), and
the intermittent `budget: independent requery agrees` flake did not fire this run. The increment adds **+9
verdicts** (3 `gdn_step` + 3 `gdn_out_norm` + 3 `qsa_gate_apply_f32`) on each implementation. `run_gate.sh` exits
**0** here. **Box (`z820b`):** radeon_icd (RX 7900 XTX) **372 / 0 / 1**, llvmpipe **364 / 0 / 3**, nvidia_icd
(Quadro K620) **367 / 0 / 2** — +9 on each arm too, and `run_gate.sh` exits **1** there because of the
pre-existing M8 `prefill split` skip (a skipped case is not a passing one).

**REMAINING CLASS-A.** Of the triage's 9 kernel implementations, 7 are landed (6 GDN legacy incl. the already-
ported `gdn_gate`, 1 QSA). **2 remain: `indexer_key_append`** (the QSA indexer pair's legacy member,
`native_qsa_indexer_enabled() == false`) **and `gr_write`** (unconditional, the hyper-connection write).

## THE GDN (DeltaNet) MIXER'S FIRST THREE KERNELS — class A of the decode-path triage — **DONE 2026-10-05**

The corrected map (`plan/DECODE-PATH-TRIAGE.md`) puts **19 class-A symbols** on the shipped model's forward path
with no ported fallback on either branch, and 36 of the 48 layers run the **GDN / DeltaNet mixer**
(`gdn_layer`, `src/core/layer.cpp:223`) — the port's own plan never enumerated it. This increment lands the FIRST
THREE kernels of the mixer's chain, in the order the layer's own sequence reaches them, under the branch policy
below.

**THE BRANCH POLICY, and the contract it puts on the backend's capability checks.** Every class-A GDN symbol is
one member of a native/legacy pair written `if (native_gdn_enabled()) native_gdn_X(...) else gdn_X(...)`
(`layer.cpp:253/255`, `266/269`, `296/299`, `308/309`, `324/325`), and in most pairs BOTH members are unported.
The port implements the **LEGACY** branch and requires the Vulkan backend to answer
**`native_gdn_enabled() == false`** (at init: `strata::kernels::native_gdn_set_enabled(false)`). Three
consequences make ONE implementation per pair sufficient:

* the layer takes the `else` of every pair, so only the `gdn_*` member needs a shader;
* the **three fused paths** `fused_gdn_conv_l2` / `fused_gdn_ab` / `fused_gdn_step_norm` are gated on
  `g_fused_gdn && native_gdn_enabled() && …` (`layer.cpp:247, 306`), so the flag removes them from the path
  entirely — they are neither implemented nor needed;
* the legacy branch is the one whose contract the engine's own header documents in full (`gdn.hpp`), and whose
  beta/gate branch's second half `gdn_gate` this port ALREADY has — so that branch is completed by adding
  `gdn_beta_gate` alone, where the native branch would still need `native_gdn_gate`.

**A CONTRACT, stated because it is real:** `layer_verify_compatible()` (`layer.cpp:476-486`) demands
`native_gdn && g_fused_gdn` and `native_qsa_indexer_enabled()`, so a backend that answers these off makes the P6
verifier refuse to init — i.e. it disables speculative verification (class D). A `--spec 0` run is unaffected.
The same policy applies to the QSA pair (`native_qsa_enabled()` / `native_qsa_indexer_enabled()` answer false →
`qsa_gate_apply_f32` / `indexer_key_append`) when that increment lands.

**RE-CHECKED: the class-A count.** The parent's reading is right in substance but its arithmetic was off by two.
The accurate decomposition of the 19: **15 are native/legacy pair members** (11 GDN + 4 QSA), **3 are the fused
GDN paths** the flag removes, **1 (`gr_write`) is unconditional** — 15 + 3 + 1 = 19. Collapsed under the policy
the 19 need **9 kernel implementations**: 6 GDN legacy (`gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`,
`gdn_gate` — already ported —, `gdn_step`, `gdn_out_norm`), 2 QSA (`qsa_gate_apply_f32`, `indexer_key_append`),
and `gr_write`. **8 are still to write**; this increment lands 3 of them. Recorded in `plan/DECODE-PATH-TRIAGE.md`.

**THE THREE, in the layer's sequence.** `gdn_layer` runs conv → l2_norm → beta/gate → step → out_norm
(`layer.cpp:244-327`); the first unported step is the conv, then the norm, then the beta half of the beta/gate
pair (the gate half is already ported).

* **`gdn_conv_step`** — the four-tap causal convolution (`gdn.cu:99-111`, `gdn.hpp:59-73`), the legacy `else` at
  `layer.cpp:255`. Shader `shaders/gdn_conv_step.comp`. Oracle: the engine's own rule transcribed from
  `gdn_parity.cpp`'s `ref_conv` (double). Case `case_gdn_conv_step`, 3 arms (C/d_conv = 24/4, 10/2, 300/4); the
  fixture's state rows are LABELLED and its taps have DISTINCT magnitudes (the two traps `gdn_parity.cpp`
  names), and the slid STATE is compared **bit for bit** because the kernel only moves and appends values.
  **Measured: PASS 96/96 worst 7.09e-08, 20/20 worst 0, 1200/1200 worst 3.62e-06.** Falsified by
  `gates/inject-verify.sh gdn-conv-tap-order` (reverse the tap order) → `FAIL gdn_conv_step C=24 d_conv=4
  72/96 worst 2.28`.
* **`gdn_l2_norm`** — `x *= 1/sqrt(sum(x^2) + eps)`, the eps an ABSOLUTE floor on the SQUARED NORM
  (`gdn.cu:118-130`, `gdn.hpp:75-80`), the legacy `else` at `layer.cpp:269-270`. Shader
  `shaders/gdn_l2_norm.comp` (one workgroup per row, the barrier-tree reduction in `common/wg_reduce.glsl`).
  Oracle: `gdn_parity.cpp` §3, double. **HONEST LIMIT:** the CUDA sums in DOUBLE; the target device has no
  shaderFloat64, so this port sums in F32 and the case MEASURES the gap against the double oracle rather than
  claiming bit-exactness (the `silu_inplace` form). Case `case_gdn_l2_norm`, 3 arms; a NaN-padded tail makes a
  missing row guard DETECTED, and the rival reading (eps on the MEAN) is checked host-side to move the fixture.
  **Measured: PASS 392/392 worst 0, 2312/2312 worst 1.66e-07, 648/648 worst 1.56e-07.** Falsified by
  `gdn-l2-norm-eps-on-mean` → `FAIL gdn_l2_norm r=1 c=128 265/392 worst 0.446`.
* **`gdn_beta_gate`** — `beta = sigmoid(beta)`, in place (`gdn.cu:217-233`, `gdn.hpp:82-88`), the legacy leaf at
  `layer.cpp:299`; the fraction `gdn_step`'s contract (`d = (v-sk)*beta`) demands. Shader
  `shaders/gdn_beta_gate.comp`. Oracle: the engine's own `sigmoid_f`, double. Case `case_gdn_beta_gate`
  (48 heads spanning → ~0 / mid / → ~1). **Measured: PASS 48/48 worst 1.42e-06.** Falsified by
  `gdn-beta-gate-drop-sigmoid` → `FAIL gdn_beta_gate 0/48 worst 1.8e+12`.

**THE MAP DROPS BY THREE.** `PORT-MAP.tsv` moved `168 — 53 kernel, 63 host, 52 todo` → `168 — 56 kernel, 63
host, 49 todo` (the three symbols are now `kernel` rows naming their shaders); `check_port_map.py` passes and
`make_port_map.py` regenerates the file BYTE-IDENTICALLY. `gdn_l2_norm` was added to `run_gate.sh`'s
barrier-census whitelist, because it carries a shared-memory reduction and the gate requires >= 2 barriers.

**GATE TOTALS, after the change. vega:** intel_icd (Arc B70) **367 / 0 / 0**, llvmpipe **355 / 0 / 3**,
radeon_icd (Ryzen iGPU) **357 / 1 / 2** — the 1 is the KNOWN intermittent `budget: independent requery agrees`
flake, and the 3 / 2 skips are pre-existing (coopmat + the M8 prefill split). The increment adds **+7 verdicts**
(3 conv + 3 l2_norm + 1 beta_gate) on each implementation. `run_gate.sh` exits 1 because of the skips and the
flake, which is the documented state — the gate's printed per-implementation totals are the authority.

## THE PORT MAP'S BLIND SPOT IS CLOSED - bare-name decode-path symbols - **DONE 2026-10-05**

`tools/check_port_map.py` keyed on the `kernels::` QUALIFIER, so a kernels-namespace symbol that
`src/core/` calls BARE (legal wherever a `using namespace strata::kernels;` is in scope, and exactly
what `src/core/mtp.cpp` does for the coupled-draft entry points) was invisible: the map could read
`todo 0` while such a symbol was unported, and the two coupled shaders sat in the UNCLAIMED list
while no row named them.

**THE RULE NOW APPLIED, and why it is not a heuristic that invents symbols.**  A DECODE-PATH SYMBOL
is an identifier `src/core/` reaches into the kernels namespace, written `kernels::X` or bare `X`.
The bare names are NOT scraped from `src/core/` -- a source file is full of local identifiers, and
treating every `foo(` in it as a kernel symbol would invent names the engine does not have.  They
come from the ENGINE'S OWN DECLARATIONS: namespace-scope functions declared in
`include/strata/kernels/**`, attributed to `src/core/` only in a source that has actually brought
the namespace into scope (`using namespace strata::kernels;` / `using strata::kernels::X;`).  The
discovery lives ONCE in `tools/port_map_lib.py`, imported by BOTH the checker and the generator, so
the two cannot drift.

**THE MEASURED EFFECT - AND A FINDING, NOT A COSMETIC FIX.**  Closing the blind spot does not add
the four known symbols; it adds **91**, because the qualifier-only scan was hiding the whole
bare-name half of the decode path.  The map moves

    77 decode-path symbols - 28 kernel, 49 host,  0 todo; ...
    168 decode-path symbols - 53 kernel, 63 host, 52 todo; ...

**M-A's `todo = 0` was therefore measured on a 77-symbol map, not on the decode path.**  The two
coupled symbols ARE ported, and their shaders are now CLAIMED (unclaimed shaders fall 37 -> 19).  But
the newly-visible set also carries a large body of GPU work this port has NOT done, classified `todo`
by the map's own definition (GPU work with no shader in this tree): the **GDN family**
(`gdn_step` / `gdn_conv_step` / `gdn_l2_norm` / `gdn_out_norm` / `gdn_beta_gate` and their `native_*`
and `fused_*` siblings), **`native_rope_apply`**, **`native_router_top10(_multi)`**,
**`native_moe_combine(_multi)`**, **`native_qsa_gate_apply`/`_rms_norm_weighted`/`_indexer_append`**,
**`bf16_gemv`/`bf16_gemv_split`**, **`s_gemv_q8_0_split`/`s_gemv_q8k_split`**, the QSA prompt/indexer
path (`qsa_decode_attn_batch`, `qsa_index_step`, `qsa_attend_step`, `indexer_key_append`,
`topk_512_step`, `qsa_gate_apply_f32`), `gr_write`/`fused_gr_read_multi`, `moe_group_resident`, and
the verify/P6 device helpers (`add_streams_broadcast`, `broadcast_streams`, `fetch_blobs`,
`copy_indexed`, `gpu_stamp`, `map_ids`, `mtp_select`, `rebase_ptrs`, `resident_plan`, `row_top_prob`,
`wait_flag_ge(_or)`, `window_ids`).  The 14 new `host` rows are the bare-name host side (the
`*_enabled` capability checks, the mapped copies, `doorbell_publish_res`/`_value`,
`coupled_draft_stage`).  **This is the honest hole list M-A claimed to have emptied**, and it
supersedes the "the map's `todo` column is 0" line in `HANDOFF.md` §6.3 and `STATUS.md` — and every dated
"77 symbols … 0 todo" line below in this file, which are records of the map as it then stood.

**M-A IS NOT CLOSED, and the 52 `todo` rows are now triaged per symbol** in `plan/DECODE-PATH-TRIAGE.md`:
**19 are class A** — genuine forward-path holes with no ported fallback on either branch (the **GDN / DeltaNet
mixer**, which runs on **36 of the 48 layers**, the **QSA gate and indexer** on the other 12, and `gr_write`) —
and those 19 are the M-A remainder; **4 are class B** — capability-gated with a ported fallback the backend
satisfies by forcing `*_enabled()` false (`native_rope_apply`, `native_router_top10`,
`native_qsa_rms_norm_weighted`, `native_moe_combine`); **7 are class C** — the non-native configuration the
shipped `setup.py` launch (`--pack … --native …`, `generate.cpp:1805-1807`) does not select; **22 are class D** —
the P6 verifier, the speculative MTP drafter, and tooling.  **The headline is the GDN family: the port's plan
never enumerated it** (`PORT-PLAN.md` wave 1 lists only `gdn_gate`), the qualifier-only scan could not see the
bare `gdn_*` calls in `layer.cpp`'s `gdn_layer`, and no GDN kernel but `gdn_gate` is ported.

**FALSIFIED - and the OLD checker is shown blind on the SAME file.**  Drop the `coupled_draft_sample`
row (the blind-spot symbol):

    $ grep -v $'^coupled_draft_sample\t' PORT-MAP.tsv > /tmp/pm.tmp && cp /tmp/pm.tmp PORT-MAP.tsv
    $ python3 tools/check_port_map.py
      FAIL src/core/ reaches kernels::coupled_draft_sample, which PORT-MAP.tsv does not mention   (exit 1)
    $ git show HEAD:ports/vulkan/tools/check_port_map.py > /tmp/old.py
    $ cp /tmp/old.py tools/_old_check.py && python3 tools/_old_check.py
      port map: 167 decode-path symbols - 52 kernel, 63 host, 52 todo; ...                        (exit 0)

The second injection renames a shader a `kernel` row names (`coupled_sample.spv` ->
`coupled_sampleX.spv`): `FAIL PORT-MAP.tsv: coupled_draft_sample names shader 'coupled_sample',
which is not built` (exit 1).  Both restore clean.

**THE ONE-COMMAND REPRODUCTION (checker-only).**  `gates/inject-verify.sh` falsifies a vk_gate CASE
by injecting into a shader/harness and running the gate binary; this is a build-time Python tool, so
the injection is recorded here rather than as an `inject-verify.sh` entry:

    cd ports/vulkan && cp PORT-MAP.tsv /tmp/pm.bak \
      && grep -v $'^coupled_draft_sample\t' PORT-MAP.tsv > /tmp/pm.tmp && cp /tmp/pm.tmp PORT-MAP.tsv \
      && python3 tools/check_port_map.py; rc=$?; cp /tmp/pm.bak PORT-MAP.tsv; exit $rc
    # expect:  FAIL src/core/ reaches kernels::coupled_draft_sample ...   (exit 1)

`make_port_map.py` no longer reads a `/tmp/core_syms.txt` hand-off: it derives the symbol set from
`port_map_lib` too, and regenerates `PORT-MAP.tsv` byte-identically (`diff -q` against a copy).

## I1 - THE ENGINE BACKEND'S FIRST INCREMENT: the device layer, the arena, and the first entry point that RUNS - **DONE AND VERIFIED 2026-10-05**

Increment **I1** of `ports/vulkan/plan/BACKEND-INTEGRATION.md` (the first increment that BUILDS the engine).
Until now every case in this port proved a **shader**; this one proves a **WRAPPER** - `strata::kernels::fwht256_cuda`,
the engine symbol `include/strata/kernels/kv_q4.hpp` declares and the decode path calls through
`fwht256_inplace_cuda` - end to end through a real device layer that did not exist in the engine before.

**What was adopted and built.**
* `ports/vulkan/harness/vk_compute.*` (and the `vk_compat.*` / `vk_stack.*` it includes) is ADOPTED as
  `vulkan/src/device/`, one copy for the engine.  Its namespace was renamed `portvk` -> `strata::vulkan` so a
  single translation unit can hold BOTH the port's device layer and the engine's at once (the gate needs both);
  the port's harness copy is untouched and stays the gate's oracle.
* **THE ARENA.**  `Stream` owns ONE device-local buffer (`alloc_device`), carved by byte offsets.  `arena_alloc`
  bump-allocates, 256-byte aligned and raised to the device's `minStorageBufferOffsetAlignment`, so every view it
  hands out is bindable.  A device pointer is `kArenaBase + byte_offset` in a SYNTHETIC address space - the engine
  only does arithmetic on it and passes it back, exactly as it did against `cudaMalloc`, so it has to be
  RESOLVABLE, not readable (the arena is unmappable VRAM on the Arc, so there is no host address to hand out).
* **POINTER -> BUFFER.**  `arena_resolve` subtracts the base, range-checks the LIVE region (`bump`), and returns
  `view(arena, offset)` - the `(VkBuffer, byte offset)` a descriptor binds.  A pointer outside the arena is a loud
  refusal, not a wrong read.
* **`fwht256` finished.**  The wrapper's two raw pointers resolve to views, and the pipeline comes from the device
  layer's cache keyed as the engine dispatches (2 storage buffers + a 4-byte push constant).  A LIVE-STREAM
  REGISTRY makes `stream_of` refuse a handle this backend did not create (membership is a comparison, so the check
  never dereferences rubbish).

**How the engine's sources join the build (the decision, and it is sycl's pattern).**  `vulkan/CMakeLists.txt`
gained `strata_vulkan_resolve()`, mirroring `../sycl/CMakeLists.txt`'s `strata_resolve`: a source is taken from
`vulkan/` when a migrated copy exists there, otherwise from the engine tree `${STRATA_ROOT}`.  I1 needs **no**
engine-root source (the wrapper is a header), so the top-level `return()` stands; the first engine-root sources are
the I2 glue and are named through the same function.  The target that proves the wiring is `strata_vk_entry_smoke`
(built only under `-DSTRATA_ENABLE_VULKAN=ON`): it links the device layer + the kernel TU and calls the engine
wrapper, round-tripping against an explicit Hadamard matrix on the host.  **Measured: configure 0.14 s, build
1.37 s** (9 objects); run PASS on the Arc (BMG G31).  A FULL engine build was **not** needed for I1's claim and was
not done - the CUDA configuration never compiles `vulkan/` (the block is inside `if(STRATA_ENABLE_VULKAN)`), so it
cannot be affected.

**The proof (the plan's own condition).**  `case_fwht256_entry` in `ports/vulkan/harness/vk_gate.cpp` runs the SAME
input twice - once through the port's shader path, once through the ENGINE wrapper on the backend's own arena - and
compares the outputs as 32-bit WORDS (a float `==` calls +0 and -0 equal and misses a NaN payload).  Raw line:

```
PASS  fwht256 entry point: engine wrapper == shader path, bitwise  1024/ 1024   worst 0          words differ - the arena view offset, the pipeline or the dispatch the wrapper uses does not match the ported shader's own path
```

Falsified by `gates/inject-verify.sh fwht-entry-wrong-view-offset`, which shifts the view the **dispatch** binds by
one float:

```
FALSIFIED (fwht-entry-wrong-view-offset): FAIL  fwht256 entry point: engine wrapper == shader path, bitwise   128/ 1024   worst 896        words differ - the arena view offset, the pipeline or the dispatch the wrapper uses does not match the ported shader's own path
```

**THE INSTRUCTIVE PART: a UNIFORM shift does not bite.**  The first injection tried `+4` inside `arena_resolve`
itself and the case stayed GREEN - the transfer (write/read) and the dispatch all resolve the same pointer, so the
shift CANCELS.  The offset that decides the answer is the one the DISPATCH binds, so the registered injection
targets that.  The injection lives in `vulkan/src/kernels/fwht_vk.cpp` (an engine-side, non-shader source), so
`inject-verify.sh`'s `rebuild_harness`/`is_harness_src` were extended to link the whole backend and rebuild it -
otherwise the script would have run a stale binary.

**Gate totals after I1.**  vega: Arc **360/0/0**, llvmpipe 348/0/3, radeon-iGPU 351/0/2, `run_gate.sh` **exit 0**,
wall **56.03 s** (a second run read 55.65 s).  Box (`z820b`), the post-commit run: default XTX **356/0/1** (the 1
is the pre-existing M8 `prefill split` skip), lvp 348/0/3, nvidia (K620) 351/0/2, exit 1 (the skip).  The FIRST box
run read **355/1/1**: `budget: independent requery agrees` failed on the XTX - the KNOWN intermittent flake (the
same run's named `radeon_icd` arm read 356/0/1 clean, and the re-run cleared it), so 0 failed is the result and
the box's non-zero exit is the documented skip.  The new case is **1024/1024 bitwise on every box arm**.  The port
map still reads **77 decode-path symbols - 28 kernel, 49 host, 0 todo**, and `make_port_map.py` regenerates
`PORT-MAP.tsv` identically.

**What I1 does NOT do** (kept for the increments that own it): the engine-root sources (the glue, rope, the mapped
copies, the doorbell redesign) are I2 and nothing in `src/` is compiled under `STRATA_ENABLE_VULKAN` yet; the
row-slice alignment risk the plan names (`X + t0*K` whose stride is not a multiple of the device's 4-byte limit)
is not exercised by fwht256, which binds whole 256-float rows.

## THE 7900 XTX'S TWO FAILURES - A WRONG CASE BOUND AND A LAVAPIPE DRIVER BUG - **DONE AND VERIFIED 2026-10-04**

The gate run on `z820b` (RX 7900 XTX, RADV gfx1100, Mesa 26.0.8) found two failures and neither was explained.
Both are resolved; the box now reads **`radeon_icd 268/0/1`, `lvp_icd 260/0/3`, `nvidia_icd 263/0/2`** (the XTX's one
skip is the pre-existing M8 cooperative-matrix case, `prefill split`).  The Arc on `vega` is unchanged at
**272/0/0**; llvmpipe 260/0/3, radeon-iGPU 263/0/2.

**Verb: `cd ~/strata-vulkan-wt && STRATA_VK_DESKTOP_RESERVE_MIB=0 STRATA_VK_RESERVE_FLOOR_MIB=0 bash ports/vulkan/gates/run_gate.sh`**

| failure (before) | where | cause | fix | falsified by |
|---|---|---|---|---|
| `kv_q4 round trip ... 3071/3072, worst 0.414` | `radeon_icd` (the target) | the case's bound `|d|/2` is wrong for a `d*[-8,+7]` code range - a clipped element errs up to `|d|` | the CASE: per-element bound `|d|` + a host replay of the rule | inject the gather's `-8` offset away -> `FAIL 0/3072 worst 8.06` |
| `quantize_q8_0 (ggml bytes) 256/268, 12 bytes` | `lvp_icd` (Mesa 26.0.8) | lavapipe's `roundEven(double)` is half-toward-zero on ties, not half-to-even; the engine's `rint` is definite | the SHADER: explicit ties-to-even (`floor` + parity), not `roundEven` | inject half-toward-+inf -> `FAIL 257/268 worst 11` |

Both falsifications are run by **`gates/inject-verify.sh <name>`** (`q4-gather-offset`, `q8-round-half-up`), which
prints `ANCHOR MISSED` / `DID NOT COMPILE` rather than running a stale binary and restores the tree on every exit.

**Finding 1 - the case's bound, not the shader.**  `kv_q4_gather.comp` is a lone multiply (`float(kc-8)*kd`) and an
f16 conversion, so contraction was never in play and no `precise` was added (the handoff already said this; the new
diagnostic proves it: the failing element - **cell 5 head 0 dim 193**, from the box's own pool bytes `d16=0x3642
code=15 -> (code-8)*d16 = 2.73779` - has host-rule code `plain=15 fma=15`, and `dev-vs-rule mismatch 0` over all
3072 elements).  The rule is `d = mval/-8` from the SIGNED extreme, `code = clamp(trunc(x/d+8.5),0,15)`; the codes
are `d*[-8,+7]`, an ASYMMETRIC 16-level range whose +8 end does not exist, so an element that wants the +8 end
(`x/d = 7.96` here) is clipped to 15 and errs up to `|d|` - twice `|d|/2`.  The case now bounds **each element
against its OWN group's `|d|`** instead of comparing a global worst against a global bound, and it PRINTS the
offender (index, device value, expected, the group's `|d|`, the device's own `d16`/code and the host rule's code)
so it can never again report `worst` without `where`.  A second-order point worth keeping: the fixture comes from
a shared RNG that device-conditional SKIPS shift, so the same binary gives the Arc (no skips) a different fixture
than llvmpipe/radeon (which skip `gemm_coopmat`) - the old global-vs-global test passed on the Arc at ratio 0.995
and failed on the box at 1.22 for that reason.  The per-group form is immune.

**Finding 2 - a driver bug the shader leaned on; the case was right.**  `quantize_q8_0.comp` spells `roundEven`
exactly; on lavapipe / Mesa 26.0.8 that builtin is NOT ties-to-even for `double`.  Every one of the 12 wrong bytes
is in the exact-tie block (`amax = 127 -> d32 = 1.0`, so the double quotient IS the value): the device wrote
`3.5 -> 3`, `-3.5 -> -3`, `1.5 -> 1`, i.e. round-half-toward-zero, where the engine (`src/kernels/cuda/
quantize_act.cu`, `rint`) and the case's oracle (`std::nearbyint`) both want `4 / -4 / 2`.  So the engine's rule
admits only ONE answer, the case was not over-tight, and it was NOT loosened - the shader was brought off the
driver builtin.  The `quantize_q8_0_scaled` sibling was already independent of it (it uses the CPU's half-away
rule), and this was the only f64 use of `roundEven` in the port.  **Honest limit:** this is a property of that
Mesa build, measured here and not reproduced upstream; the shader no longer depends on it either way.

## THE SPLIT SAMPLER - the DEFAULT sampled path, and the ordered merge - **DONE AND VERIFIED 2026-10-05**

`sampler_split.comp`, from `sampler_split_part_kernel` + `sampler_split_merge_kernel` (src/kernels/cuda/sampler.cu),
with the tail from `sampled_tail_warp`.  This is the path `sample_tokens` takes by default for a sampled request
(the one-block `sampler_kernel` is the `STRATA_OLD_SAMPLER=1` reference), and it is a different SELECTION from that
kernel's: a row is cut into **4096-logit partitions**, each keeps its own `top_k`, and the ordered lists are
**merged** - exact, because the first k of a union lie within the first k of each part and a merge of ordered lists
is ordered.  The engine asserts its three sampled paths "pick the same token, bit for bit", and that parity is a
gate arm here.

**The shared machinery, so the port's three tails cannot drift.**  `common/sampler_tail.glsl` owns the chain's tail
(top_p cut, min_p prefix, temperature on the survivors, softmax, one Philox draw) in BOTH arithmetic variants -
`sampler_tail_f64` (the engine's, needs shaderFloat64) and `sampler_tail_f32` (the portable sibling, compiled by the
same file) - and the selection-list shared arrays.  `common/sampler_select.glsl` owns the partition+merge selection.
The f64 half is compiled only where a shader defines `PORT_SAMPLER_WANT_F64`, so the f32 sibling does not carry the
capability through an include it never calls.  The split and (next) the coupled merge share both files; the parity
this buys is BY CONSTRUCTION rather than a careful copy.

**A measured defect the multi-partition arms found, and it is a real class.**  The first merge was a forward pass
IN PLACE.  That is wrong: the write index `o = a + c` runs ahead of the read index `a` as soon as a partition entry
is taken (`c > 0`), overwriting running-list entries a later `a` still has to read.  It read `dev 4200 want 100`
on the top_p arm - a plausible token from a corrupted list - and only the arms whose top logits sit in DIFFERENT
partitions could see it (a one-partition row exercises no merge at all).  Fixed by merging into a separate array
(`sc_mrg_*`) and copying back; all five arms then read `bit-exact`, worst dev-vs-rule 0.

**Falsified.**  `gates/inject-verify.sh sampler-split-merge-drop-parts` makes the merge drain the running list
before the partition's (`else take_a = (a < ncur)`, so every partition after the first appends to the tail and is
lost): `FAIL  sampler_split: three partitions: the merge interleaves the lists   9/   16  worst 0`.

**Measured.**  vega, this commit: **Intel Arc (BMG G31) 340 passed / 0 failed / 0 skipped** (`run_gate.sh` exit 0),
intel_icd 340/0/0, llvmpipe 328/0/3, radeon-iGPU 331/0/2.  That is **+7 verdicts** on every implementation (five
split arms + the parity arm + the group arm).  The parity arm reads **60/60** (the split's token equals
`sampler_kernel`'s on every seed of every arm), which is the engine's own bit-for-bit claim, measured.

**What it still needs.**  The split is f64-only (its tail accumulates in double, like the engine's), so it carries
the same fp64 requirement as `sampler_kernel` - the portable f32 sibling settles that next.

## THE COUPLED DRAFT PATH (speculative decoding) - **DONE AND VERIFIED 2026-10-05**

`coupled_penalize.comp` + `coupled_sample.comp`, from `coupled_penalize_kernel` and `coupled_merge_kernel`
(src/kernels/cuda/sampler.cu); the host arithmetic is `include/strata/core/coupled_draft.hpp`.  In coupled mode
(`STRATA_SPEC_COUPLED=1`, off by default) the MTP draft layer SAMPLES its draft with the TARGET's own chain and the
SAME Philox draw the target will use for the row that verifies the draft, so the target's pick - and so the text -
does not change (verification is an exact-match against the target's sample).  `coupled_sample` reuses the split
sampler's selection and tail includes, so its chain cannot drift from the target's.

**FOUR RULES, EACH AN ARM - and the first is the one a paraphrase gets wrong.**  `coupled_draft_counter(cell) =
cell + 1`: the draft is verified by the NEXT window's row, drawn with counter `cell + 1`, so the drafter draws with
`cell + 1` too - not `cell`.  A different counter is a different (still valid) generator and a worse acceptance
rate, which reads as "the model got a bit worse".  Then: the penalty window is the ring's `[cap + j - h, cap + j)`
(`coupled_hist_start`), NOT the staged base; a history id maps through `id_to_sub` to a subset index (the draft head
owns a VOCABULARY SUBSET); and the pick maps through `sub_to_id` and is appended at `ring[cap + j]`.

**Falsified (both).**  `gates/inject-verify.sh coupled-draft-counter-off-by-one` drops the `+1`:
`FAIL  coupled_draft: the counter is cell+1: 8 equal survivors, cells 0..7    1/    8  worst 0` (the equal-survivor
arm observes the stream exactly - the softmax cancels out of the walk).  `coupled-draft-window-start` drops `j` from
the window start: `FAIL  coupled_draft: the window is the ring's [cap+j-h, cap+j)    0/    4  worst 0`.

**Measured.**  vega, this commit: **Intel Arc (BMG G31) 345 passed / 0 failed / 0 skipped**, llvmpipe 333/0/3,
radeon-iGPU 336/0/2, `run_gate.sh` exit 0.  **+5 verdicts** (four arms + the group arm).  Box (7900 XTX):
radeon_icd 341/0/1, lvp 333/0/3, nvidia 336/0/2.

**Two honest notes.**  (1) `coupled_draft_sample` / `coupled_draft_stage` are called UNQUALIFIED in `src/core/mtp.cpp`
(`using namespace strata::kernels`), so `check_port_map.py` - which keys on `kernels::` - does not list them, and the
two coupled shaders are therefore reported as UNCLAIMED by a `kernel` row rather than mapped.  The map's rule is
unchanged and still passes; the coupled entry points are a pre-existing blind spot of the symbol scan, recorded here
rather than papered over.  **CLOSED 2026-10-05 - see the top section: the checker now discovers bare-name call sites,
the coupled pair is classified, and the same scan surfaced 91 hidden symbols (52 `todo`).**  (2) `coupled_penalize` scans a subset index's occurrences in the window instead of the
engine's shared bitmap + `atomicOr`; the counts are identical (`id_to_sub` is the inverse of `sub_to_id`) and it
keeps the shader free of atomics, at O(nv x h) instead of O(h + hits x h) - the port's existing scan-instead-of-bitmap
trade, acceptable for a correctness arm with the gate's small subset sizes.

## THE `sample_tokens` ENTRY POINT: THE CHOICE, not just the two paths - **DONE AND VERIFIED 2026-10-05**

The port had both sampled paths gated (`sampler_kernel`, `sampler_split`) and the greedy kernel gated
(`sampler_greedy`), but nothing proved WHICH one a request takes - the `PORT-MAP.tsv` row named the shaders and the
choice itself was untested.  `case_sample_tokens` (`harness/vk_gate.cpp`) pins it, as a pure predicate AND by
running the chosen kernel:

    sample_choice(greedy, temperature, n_vocab, n_tokens, split_ok):
      greedy || temperature <= 0            -> GREEDY
      n_blocks <= 64 && n_tokens <= 64      -> SPLIT   (the default)
      otherwise                             -> ONEBLOCK (the fallback)

**The arm the task names is `temperature == 0`.**  `sample_tokens` routes it to the GREEDY kernel - the argmax -
NOT to the sampled path's uniform-over-the-shortlist draw (which is what the sampled kernel does when called
DIRECTLY with temp 0, gated separately by `case_sampler_kernel`).  The fixture is 8 distinct logits 8..1, so the
greedy token is id 0 while the sampled path would draw `floor(u*8)` over the 8, and the row's `sampled-path spread`
is printed as evidence the arm discriminates.

**Six rows**: the greedy flag; temperature 0; a sampled request (SPLIT); no scratch (ONEBLOCK fallback); a
vocabulary wider than the merge holds (300000 logits -> 74 partitions > 64, ONEBLOCK); and 65 rows (> 64,
ONEBLOCK).  Each runs its chosen pipeline and compares every token to that path's oracle.
**One defect the multi-row arm found in the CASE**: the sampled kernels draw `philox(seed, counter + t)`, so the
host oracle must use counter `t`, not `0` - reading `0` for every row passed the 1-row arms and failed the 65-row
one (`-87/4`: more bad than compared, the signature of a wrong oracle rather than a wrong kernel).

**Falsified.**  `gates/inject-verify.sh sample-tokens-choice-temp0-to-sampled` changes the predicate's
`temperature <= 0.0f` to `temperature < 0.0f`, so a temp-0 request falls through to the sampled path:
`FAIL  sample_tokens: temperature 0 takes the argmax, NOT the draw    3/    4  worst 0`.  This injection lives in
the HARNESS, so `inject-verify.sh` now rebuilds the gate for a non-shader change (an earlier `if (greedy)` form
was caught as `DID NOT COMPILE (harness)` on an unused-parameter `-Werror`, which is exactly the "an injection that
does not build is a stale binary" rule).

**Measured.**  vega: **Intel Arc 352 passed / 0 failed / 0 skipped**, llvmpipe 340/0/3, radeon-iGPU 343/0/2,
`run_gate.sh` exit 0.  **+7 verdicts** (six rows + the group arm).  Box (7900 XTX): radeon_icd 348/0/1, lvp 340/0/3,
nvidia 343/0/2.  The port map is unchanged: `sample_tokens` already named `sampler_greedy sampler_kernel
sampler_split`, and the case adds no `kernels::` symbol.

## THE PORTABLE f32 SIBLING for the sampler - **DONE AND VERIFIED 2026-10-05**

`sampler_kernel_f32.comp`, the sibling the faithful `sampler_kernel.comp` cannot be without.  Every other
double-arithmetic kernel in this port already had one (`router_top10_f32`, `moe_combine_f32`, `swiglu_f32`,
`scalar_gate_f32`): the engine's sampler computes its top_p cut and its softmax in DOUBLE, and **Intel Arc has no
shaderFloat64** (Intel's own support article 000089817 - the target hardware), so without the sibling the sampled
path cannot run there at all.  The chain and the SELECTION are byte-for-byte the faithful kernel's; only the tail's
arithmetic is float, and it lives in `common/sampler_tail.glsl` as `sampler_tail_f32`, so the two cannot drift.
`run_gate.sh`'s fp64 structural rule now also requires this variant to carry NO `Float64` capability (verified:
`spirv-dis | grep -c OpCapability Float64` = 0), which is the test that it really is the portable one.

**The case MEASURES the gap rather than asserting it.**  Five arms: two EXACT because no rounding can reach the
answer (a one-survivor shortlist has no exponential - top_k = 1 and a top_p cut of exactly one), one EXACT because
the softmax cancels out of the walk (64 EQUAL survivors at temperature 0, so the draw is `floor(u*64)` and the RNG
is pinned bit for bit), and two MEMBERSHIP where a real distribution is involved (the f32 token must be one of the
f64 chain's survivors).  Measured on the Arc: `f32 vs f64: 0 of 48 seeds differ` - the fixtures' margins are wide
enough that the two agree here, which is reported rather than claimed as a general result.

**Falsified.**  `gates/inject-verify.sh sampler-kernel-f32-top-p-boundary` changes the f32 tail's `>=` top_p
boundary to `>`: `FAIL  sampler_kernel_f32: one survivor (top_p cut of one): exact    2/    8  worst 0`.

**Measured.**  vega: **Intel Arc 358 passed / 0 failed / 0 skipped**, llvmpipe 346/0/3, radeon-iGPU 349/0/2,
`run_gate.sh` exit 0.  **+6 verdicts** (five arms + the group arm).  Box (7900 XTX): radeon_icd 354/0/1, lvp 346/0/3,
nvidia 349/0/2.  The port map names `sampler_kernel_f32` in the `sample_tokens` row (`make_port_map.py` regenerates
`PORT-MAP.tsv` identically; still 77 symbols - 28 kernel, 49 host, 0 todo).

**Honest limit.**  0-of-48 agreement is a statement about THESE fixtures, not about f32 in general: a distribution
whose probabilities pile up within a float ULP of a cumulative boundary can still flip a token, and this port has no
f64-free device to reproduce the hardware the variant exists for (this box's ANV reports `fp64 1`).  The variant is
gated as EXACT where the arithmetic is exact and as MEMBERSHIP elsewhere, which is the strongest claim the
measurement supports.



## M-A: the standalone dequantiser's LAST TWO FORMATS - IQ2_XXS and IQ2_XS - **DONE AND VERIFIED 2026-10-05**

`iq_dequant_f32` covered 14 formats and REFUSED the remaining two `is_iq` types by name: IQ2_XXS (ggml type 16) and
IQ2_XS (17), because their grids (`iq2xxs_grid` 256 points, `iq2xs_grid` 512) were not in the port's generated
table.  Both now decode, and the hole is closed: **`check_port_map.py` still reads 77 symbols - 28 kernel, 49 host,
0 todo** (the formats live in a shader that was already `kernel`; no map row changed, and `make_port_map.py`
regenerates `PORT-MAP.tsv` byte-identically).

**What changed.** `tools/gen-iq-tables.py` extracts the two `uint64_t` grids from the engine's own
`third_party/ggml/ggml-common.h` verbatim, as low/high `uint32` pairs like `iq2s_grid` (no `shaderInt64`);
`harness/iq_grids.hpp` gains `kIq2xxsGrid[512]` and `kIq2xsGrid[1024]`; `common/iq_dequant.glsl` gains the two
`tid`-mapped bodies; `iq_dequant_f32.comp` and `iq_embed_rows.comp` declare the two new grid buffers (bindings 5/6,
the output moving to 7/8); and `harness/vk_gate.cpp` gains ONE ARM PER FORMAT whose oracle is a transcription of
the engine's `dq_iq2_xxs` / `dq_iq2_xs` (`src/kernels/cuda/iq_kernels.cu`).

**The two layouts, from the engine's decoder (not re-derived).** Both store `d` at byte 0 and `qs` as 32 `uint16`
at byte 2; the per-lane part is `qs + 4*ib`, i.e. 8 bytes (`ib = tid%8`, `il = tid/8`).
* **IQ2_XXS (66 bytes):** the four LOW bytes of the part are the four lanes' GRID INDICES (`aux8[il]`, one packed
  byte each); the four high bytes are `aux32`, whose top nibble is the block scale's high part and whose low 28 bits
  carry four 7-bit `ksigns_iq2xs` selectors.  `d = f16 * (0.5 + (aux32>>28)) * 0.25`.
* **IQ2_XS (74 bytes):** the part's four `uint16` words each hold a 9-BIT grid index (`& 511`) with the 7-bit sign
  selector ABOVE it (`>> 9`); the scale is `scales[ib]`'s nibble `4*(il/2)`.  `d = f16 * (0.5 + nibble) * 0.25`.

**A finding worth keeping: the fixture MARGIN was not serving the arm.** The first version of the IQ2_XS injection
(drop the grid index's 9th bit, `& 511` -> `& 255`) was **NOT FALSIFIED** - the arm stayed 768/768 green.  Cause: the
shared fixture's byte pattern `(k*37+13)&0xFF` makes every odd-offset byte EVEN, and the grid-index word's high byte
is always at an odd offset here, so bit 8 of `q2[il]` was **deterministically 0** at the dequant arm's seed: the
upper half of the 512-point grid was never indexed and the injection could not move a value.  The fixture now forces
that bit to split across both halves (`| (ib & 1)`) for IQ2_XS, so the whole grid is exercised - and only then does
the injection bite.  This is the "a fixture that cannot move under the wrong rule is decorative" rule, found again.

**Falsified (both, on vega).** `gates/inject-verify.sh` gained two registrations:
* `iq-dequant-iq2xxs-grid-index` (read the neighbour lane's grid byte) -> `FAIL  iq_dequant_f32: IQ2_XXS  300/768  worst 3.79e+05`.
* `iq-dequant-iq2xs-grid-high` (drop the 512-point grid's 9th bit) -> `FAIL  iq_dequant_f32: IQ2_XS  530/768  worst 3.82e+05`.

**Measured, bit-exact.** Both arms read `bit-exact 768/768` on the dequantiser and `1536/1536` on `iq_embed_rows`
(the same decode through the row gather).  Totals **+4 verdicts on every implementation**:
* vega: **Arc Pro B70 333 passed / 0 failed / 0 skipped**; llvmpipe 321/0/3; radeon-iGPU 323/1/2 where the single
  failure is the documented intermittent `budget: independent requery agrees` (the iGPU figure drift - recorded,
  not chased; a clean radeon run reads 323/0/2).
* z820b (7900 XTX): **radeon_icd 329/0/1** (the 1 skip is the pre-existing M8 `prefill split`), lvp 321/0/3,
  nvidia (K620) 324/0/2.  The box's script exits 1 on that pre-existing skip, as at HEAD.

## M-A: the decode path's last kernels - the ORDER, and 1/3: `cvec_apply`

**The order, derived from the decode path (`src/core/`), not invented.** The ten `todo` symbols were ordered by
two things: where the decode step meets them, and **what each one depends on**. Three carry NO unported
precondition, and they come first, in the order a decode step reaches them:

    1. cvec_apply         every layer, src/core/layer.cpp:1330-1336 (block_layer_post) - the steering vector
    2. gather_rows        MtpDrafter::bind, src/core/mtp.cpp:450 - the MTP draft head's token subset
    3. scatter_rows_f32   PeerExperts::run, src/core/peer_experts.cpp:241 - the peer experts' write-back

then the pair that shares the primitive the port does NOT have - `iq_dequant_f32` -> `iq_embed_rows`, which need a
standalone IQ/BF16 dequantiser where the port has only the FUSED `iq*_mmvq` form (the port map's own reason for
that row) - then the head's matvec `native_q5_k_f32`, which needs a **Q5_K dot** the `iq*_mmvq` family does not
contain (`native_q5_k_f32` = `quantize_q8_1` + `native_q5_k_mmvq`). The MoE half follows in the order the hit path
issues it: `moe_hit_select` -> `moe_hit_grouped_s2` (with `moe_grouped_s2` its resident sibling) -> `moe_hit_add`.
**The embedding kernels are first IN TIME in a token but are the largest single increment** - a dequantiser over
the 15 `is_iq` types plus BF16 - and nothing else on the list depends on them; that, and not convenience, is why
these three are the first increments.

**What was ported.** `cvec_apply.comp`, from `cvec_kernel` (`src/kernels/cuda/cvec.cu:46`); the CONTRACT is the
engine's own test, `src/kernels/cvec_parity.cpp`, and that is what the gate reproduces arm for arm:

| arm | rule | required by |
|---|---|---|
| project (mode 0) | `h' = h - s (h.v) v`; s = 2 REFLECTS the component, s = 1 removes it | the engine: **1e-4 absolute** vs double |
| add (mode 1) | `h' = h + d` | the engine: **BITWISE** |
| a layer without a direction (s == 0) | R untouched | the engine: **BITWISE** |
| switch off, no pending write | R untouched | the engine: **BITWISE** |
| switch off, pending write | `h' = fma(bo, w, h)`, `w = 2 sigmoid(inj/hc)` | **BITWISE** at `inj == 0` |
| on + write | the write, THEN the projection | **1e-4** vs the oracle |

Two lessons from this port's own history are load-bearing here:

* **The CUDA uses `fmaf` on both paths**, so the GLSL calls `fma()` and is deliberately **NOT `precise`**: the
  fused op rounds ONCE where a separate multiply-and-add rounds twice, and `fmaf` is the engine's answer. (The
  opposite of the embedding gather, where the engine wanted the TWO-rounding form and `precise` was the fix.)
* **One workgroup per (stream, token), reduction by the barrier tree** (`common/wg_reduce.glsl`) - the port's
  design rule, because the target is Intel where the driver picks the subgroup width per kernel. The tree's
  summation ORDER is therefore the port's, which is exactly why the two PROJECTIONS cannot be bitwise and use the
  engine's own 1e-4, while every arm without a reduction in it is bitwise. `run_gate.sh`'s barrier census list
  gains `cvec_apply` (its SPIR-V carries the tree's barriers; a shader that lost them must fail).

**The write arm is bitwise BY CONSTRUCTION, not by luck**: `inj == 0` makes `2 sigmoid(0/hc) == 1.0` exactly for
any `hc`, so the fold is `fl(bo + h)` and the case demands it bitwise - and separately requires that the write
actually MOVED values (`write w=1: 6144 of 6144 values moved`), so a no-op cannot pass.

**Falsified.** `gates/inject-verify.sh cvec-apply-drop-scale` drops the per-layer factor `s` from the reduced dot
(`dot = wg_sum(dot) * s` -> `dot = wg_sum(dot)`), and the named case fails:
`FAIL  cvec_apply: project removes s(h.v)v  12/14  worst 0.844` (the correct form reads `worst 4.73e-07`).

**Measured.** vega, this commit: **Intel Arc (BMG G31, default) 278 passed / 0 failed / 0 skipped**, intel_icd
278/0/0, llvmpipe **266/0/3**, radeon-iGPU 269/0/2 (the iGPU's `budget: independent requery agrees` is the
documented intermittent driver-figure drift - 2 fail / 1 pass in 3 consecutive runs of this binary, recorded, not
a port defect). z820b (7900 XTX): **radeon_icd 274/0/1** (the 1 skip is the pre-existing M8 cooperative-matrix
case), lvp 266/0/3, nvidia (K620) 269/0/2; the box's script exits 1 on that pre-existing skip, as it does at HEAD.

**Honest limit.** The engine's `sigmoidf_` is `__expf`-based and the port's is the driver's `exp`, so with
`inj != 0` the port's `w` differs from the CUDA's in the last bits. That is why no arm compares a non-zero-`inj`
write bitwise: the rule (direction, magnitude, the `hc` division, the fold and its order) is gated; the last bits
of `__expf` are not claimed.

## M-A 2/3: `gather_rows` - the MTP draft head's opaque-byte row gather

`gather_rows.comp`, from `gather_rows` / `gather_rows_kernel` (`src/kernels/cuda/verify_kernels.cu:394`, `:303`);
its decode-path call site is `MtpDrafter::bind` (`src/core/mtp.cpp:450`), which builds the draft head by keeping
only the token subset `dvocab` of the head's weight table, one whole `row_bytes` row at a time.

    for i in [0, n*row_bytes):  r = i / row_bytes;  o = i - r*row_bytes
                                dst[i] = src[ ids[r]*row_bytes + o ]

**The CUDA's element WIDTH is a performance choice, and it is the shape a port breaks.**  It templates on `uint4` /
`uint32_t` / `uint8_t` by `row_bytes % 16 / % 4`, and `row_e = row_bytes / sizeof(E)` then stands in for
`row_bytes` in BOTH the divisor and the source stride.  A port that keeps `row_e` in one place and `row_bytes` in
the other computes a plausible, WRONG row - and nothing about the output looks wrong.  This kernel indexes in
BYTES, where both alignments are one mapping, and the gate says so with two arms: `row_bytes = 64` (the `uint4`
path) and `42` (the byte path).

**The fixture is built so an identity gather cannot pass.**  `ids` is a DERANGEMENT (`ids[r] != r` for every r),
and every source byte is distinct (`(k*31 + j*7 + 1) & 0xFF`), so a wrong source row and a wrong offset each land
on a different byte.  64 guard bytes past the written range are compared against the sentinel the fixture wrote,
so a kernel that wrote past its count fails as well.

**Falsified.**  `gates/inject-verify.sh gather-rows-identity` ignores `ids[r]` (`src[ids[r]*row_bytes + o]` ->
`src[r*row_bytes + o]`): `FAIL  gather_rows: 16-byte-aligned rows  64/576` (the faithful form reads 576/576).

**Measured.**  vega: Intel Arc **281 passed / 0 failed / 0 skipped** (default and intel_icd), llvmpipe
**269/0/3**, radeon-iGPU **272/0/2** (its budget-requery drift again: 1 fail in 3 runs of this binary).  z820b:
radeon_icd (7900 XTX) **277/0/1**, lvp 269/0/3, nvidia (K620) 272/0/2; the box's script exits 1 on its
pre-existing M8 skip, as at HEAD.

**No honest limit to state**: the rule is a pure byte copy and the case compares bytes, so the arms are exact
rather than tolerance-bearing.

## M-A 3/3: `scatter_rows_f32` - the peer experts' row write-back

`scatter_rows_f32.comp`, from `scatter_rows_f32` / `scatter_rows_kernel` (`src/kernels/cuda/elementwise.cu:263`,
`:255`); call site `PeerExperts::run` (`src/core/peer_experts.cpp:241`).

    for r in [0, n):  for i in [0, width):  dst[ rows[r]*width + i ] = src[ r*width + i ]

Two rules, both of them traps this port has already met once elsewhere:

* **`r` is a POSITION and `rows[r]` is the DESTINATION row** - the same two-roles confusion the KV gather's
  `ids[id]` carries.  The fixture is a PERMUTATION WITH NO FIXED POINT with distinct row contents, so an identity
  write fails on every row.
* **A destination row `rows[]` does not name is NOT WRITTEN AT ALL.**  The destination is filled with a sentinel
  and the unnamed rows - plus a `strays` count of zero - must survive, which is what catches a port that writes
  every row.

**The CUDA's precondition is its VECTORIZATION, not the rule**: it casts to `float4` and therefore `exit(1)`s
unless `width % 4 == 0` and both pointers are 16-byte aligned.  This kernel reads f32 words and carries neither
requirement - and that is GATED rather than asserted: the second arm runs `width = 6`, where the CUDA would refuse.

**Falsified.**  `gates/inject-verify.sh scatter-rows-identity` drops the row indirection
(`dst[rows[r]*width + i]` -> `dst[r*width + i]`): `FAIL  scatter_rows_f32: permutation ... 511/3072`.

**Measured.**  vega, this commit: **Intel Arc 284 passed / 0 failed / 0 skipped** (default and intel_icd),
llvmpipe 272/0/3, radeon-iGPU 275/0/2 - `run_gate.sh` **exit 0**.  z820b: radeon_icd (7900 XTX) **280/0/1**,
lvp 272/0/3, nvidia (K620) 275/0/2; the script exits 1 there on the PRE-EXISTING M8 cooperative-matrix skip (a
skip is not a pass), exactly as it does at HEAD.

**M-A is now 3 of the ten.**  Still `todo`: `iq_dequant_f32`, `iq_embed_rows`, `native_q5_k_f32`,
`moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`, `moe_hit_grouped_s2`.  The next increment in the derived order
is the **`iq_dequant_f32` -> `iq_embed_rows` pair**: the standalone IQ/BF16 dequantiser the embedding path needs
(15 `is_iq` types plus BF16 - the port has only the FUSED dot form today), a format at a time with one codebook's
values per arm, then the row gather built on it.

## M-A 4/10 and 5/10: `iq_dequant_f32` + `iq_embed_rows` - the STANDALONE IQ/BF16 dequantiser, and the row gather on it

**The pair the decode path needs first, and why it is a pair.** `iq_dequant_f32` is the engine's standalone
I-quant / BF16 decoder - `dq_dispatch<float>` through `dequant_flat_kernel` (`src/kernels/cuda/iq_kernels.cu:1643`,
`:1848`) - and `iq_embed_rows` (`embed_rows_kernel`, `:1830`) is the token-embedding gather built on it: rows
`tokens[t]` of a GGUF table, `row_bytes` apart, dequantised to fp32. The port had only the FUSED `iq*_mmvq` dots;
this is the standalone form, and `iq_embed_rows` needs no decode of its own.

**THE ENGINE'S DECODER IS THE ORACLE, and the port keeps its thread mapping rather than re-deriving a layout.**
Every offset, shift and scale comes from the CUDA body that `iq_dequant_f32` and `iq_embed_rows` actually launch,
and the GLSL keeps the engine's own `tid` 0..31 mapping (`il = tid/8`, `ib = tid%8`, the inner `j` loop, and the
output position each case writes) - the port's worst bug class is a re-derived element->byte map that agrees with
itself. `shaders/common/iq_dequant.glsl` holds that decode ONCE, shared by both shaders; each launches with 32
active lanes inside a 256-lane workgroup so `run_gate.sh`'s single workgroup-size rule still holds.

**Coverage, stated exactly.** Every `is_iq` type whose grid the port's generated table carries, plus BF16 - **14
formats**: BF16, IQ4_NL, IQ4_XS, Q8_0, Q5_0, Q5_1, Q2_0, Q4_K, Q5_K, Q3_K, IQ3_XXS, IQ3_S, IQ2_S, IQ1_M. **IQ2_XXS
(ggml 16) and IQ2_XS (17) are NOT ported**: their grids (`iq2xxs_grid`, `iq2xs_grid`) are not in
`harness/iq_grids.hpp`, and neither shader claims them - an unproven format is recorded, not approximated.

**One arm per format; the oracle is a transcription of the RULE.** The case reproduces each `dq_*` body on the
host, element by element, and compares the device's 256 values against it. The fixture puts a KNOWN, FINITE fp16
scale in every place the format stores one - a random byte pattern would assemble an inf/NaN scale and the case
would measure that instead of the decode - and the scale AND the pattern vary with the superblock and with the row
seed, so a decode that read the wrong superblock or the wrong row lands on a DIFFERENT value, not an equal one.
`iq_embed_rows` uses a DERANGEMENT token list (`tokens[t] != t`), so an identity gather fails on every token, and
gates the row STRIDE the caller passes with two wide-row arms (`n_embd = 512`, two superblocks per row).

**Measured.** vega, this commit: **Intel Arc (BMG G31, default) 314 passed / 0 failed / 0 skipped**, intel_icd
314/0/0, llvmpipe **302/0/3**, radeon-iGPU 305/0/2 - `run_gate.sh` **exit 0**. z820b: radeon_icd (7900 XTX)
**310/0/1** (its 1 skip is the pre-existing M8 `prefill split`), lvp 302/0/3, nvidia (K620) 305/0/2; the box's
script exits 1 on that pre-existing skip, exactly as it does at HEAD. That is **+30 verdicts on every
implementation** (14 dequant + 16 embed arms), all green, and the device's values are **bit-exact** against the
host oracle on all 30 (worst dev/tol 0).

**Falsified, and one change the falsification forced.** `gates/inject-verify.sh iq-dequant-iq1m-grid-high`
(misplace the IQ1_M grid high bit, `<< 8` -> `<< 7`) -> `FAIL  iq_dequant_f32: IQ1_M  310/768  worst 2.34e+05`.
`iq-embed-rows-identity` (gather the row at the POSITION instead of the token) -> `FAIL  iq_embed_rows: BF16 ...
0/3072  worst 6.58e+04`. The first targets a shared INCLUDE, which has no `#version` and cannot be compiled alone,
so `inject-verify.sh` now compiles the shader that INCLUDES a changed `common/` file (and says so) instead of
reporting `DID NOT COMPILE` - the same rule the port learned when three injections silently tested nothing.

**Honest limit.** The row base is computed in 64-bit and truncated to the 32-bit byte index the storage buffers
take, so a table larger than 4 GiB is outside this arm; the engine's `size_t` row arithmetic would still be
correct, and the port's embedding tables are far below that. `iq_dequant_f32` also stops at the 14 formats above -
IQ2_XXS and IQ2_XS stay `todo` in `PORT-MAP.tsv` for the grid reason stated.

**M-A is now 5 of the ten.** Still `todo`: `native_q5_k_f32`, `moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`,
`moe_hit_grouped_s2`. The next increment in the derived order is **`native_q5_k_f32`** (= `quantize_q8_1` +
`native_q5_k_mmvq`, the head's Q5_K matvec - `q5_q8_dot` in `src/kernels/cuda/native_mmvq.cu`, a pinned
integer-dot order the `iq*_mmvq` family does not contain).

## M-A 6/10: `native_q5_k_f32` - the native head's Q5_K matvec, and the ONE dot whose scale and min are PACKED

`native_q5_k_f32.comp`, from `native_q5_k_mmvq_kernel` / `q5_q8_dot` (`src/kernels/cuda/native_mmvq.cu:239`,
`:196`).  `native_q5_k_f32` (:1311) is `native_quantize_q8_1` PLUS this, and the decode head's type-13 path is
exactly that pair (`src/core/native_head.cpp:78`), so this is the last matvec between the port and the head.

**The rule is a pinned integer-dot order, and its shape is the point.**  One workgroup per row; each lane owns a
strided set of the row's `(Q5_K block, 16-value part)` items, where part `p = iqs/2` in 0..15,
`bq8_offset = 2*(p/4)`, `nib = p%4` - the engine's own `kqs = VDR*(tid % (QI/VDR))` decomposition.  Every part's
VALUE is the source's f32 expression term for term (two `dp4a` chains: `dot1` against the codes, `dot2` against
ones for the `sum(x)` term); the reduction that adds the parts is the port's barrier tree, so the comparison is a
double reference, not bit-for-bit.

**`aux` is what the `iq*_mmvq` family does not have.**  Q5_K's 12-byte `scales` packs SIX 6-bit scales and SIX
6-bit mins, and which six bits are the scale and which the min depends on the block half: `j = bq8_offset/2`,
`jm = j&1` (which `uint16` to read), and `hi`, the all-ones mask that switches between groups 0..2 and 3..5.  The
port transcribes the source's masks and shifts literally rather than re-deriving them - a "tidier" read is a
plausible wrong number, which is the class the falsification targets.

**THREE TRAPS, ALL FOUND BY THIS CASE AND ALL SILENT.**
1. **The activation's `u` is read out of the q8_1 block, not the weight.**  `q5_q8_dot` reads four int8 from
   `bq8i->qs` and four from the Q5_K block's `qs`; collapsing the two 4-byte helpers into one buffer (the source's
   pointer argument loses its buffer when the port makes it a byte offset) sent every activation read into the
   weights - the same `f16_at` defect class, and it read a legal address.  The port keeps two named helpers
   (`q5_i32` for the weight, `q5_a_i32` for the activation).
2. **A Q5_K block's activation group is `kby = kbx*8`, not the row start.**  Each 256-value Q5_K block dots the
   EIGHT q8_1 blocks at `kbx*8`; using the row base for every block computed a finite, wrong row.
3. **`unpackHalf2x16` needs the full 32-bit half2.**  `unpackHalf2x16(q5_u16(blk))` passes only the LOW half, so
   `.y` (the block's `min`) came back 0 and the whole `- min*sumf_m` term vanished - a per-row error of 1-13%,
   every value finite and close.  The first two traps were caught by the arm failing at 1e2-1e5 of tolerance; this
   one needed the shader's own `sumf_m` exposed to be seen, and it is recorded because "close but wrong" is the
   signature a value comparison should have caught and a per-part dump did.

**Fixtures.**  `q5_k_fill_blob` varies EVERY byte of `scales`, `qh` and `qs` (so both `qh` nibble halves, both
`aux` branches and every `dp4a` byte position are exercised), with `dm = (d, min)` FINITE (a random byte pattern
would assemble an inf half and the case would measure the inf instead of the decode).  Three arms: `n_in=2560,
n_out=8` (1280 parts over 256 lanes), `n_in=256, n_out=1` (16 parts, most lanes idle), and `n_in=10240, n_out=4,
ncols=2` (the widest Q5_K pitch and a two-column activation).  Each compares the row against a host double
reference, requires finite outputs, a guard region past the buffer, and a live oracle mass.

**Falsified.**  `gates/inject-verify.sh native-q5k-aux-half` drops the packed-scale half switch
(`him = (j>=2) ? ~0 : 0` -> `him = 0`), so every group reads groups 0..2's fields:
`FAIL  native_q5_k_f32 (n_in=2560, n_out=8, ncols=1, ...)  0/8  worst 5.62e+04`.

**Measured.**  vega, this commit: **Intel Arc 317 passed / 0 failed / 0 skipped**, llvmpipe **305/0/3**,
radeon-iGPU 307/1/2 (its `budget: independent requery agrees` drift again - the documented intermittent figure).
z820b (7900 XTX): **radeon_icd 313/0/1** (the 1 skip is the pre-existing M8 cooperative-matrix case),
lvp 305/0/3, nvidia (K620) 308/0/2; the box's script exits 1 on that pre-existing skip, as at HEAD.
That is **+3 verdicts on every implementation**.

**Honest limit.**  The arms supply the q8_1 activation directly (the boundary every other `iq*_mmvq` arm draws),
so the oracle is INDEPENDENT of the quantiser rather than reading bytes the quantiser wrote; the quantiser half of
`native_q5_k_f32` is gated byte-exactly by `case_quantize_q8_1`, and the composed quantiser->dot chain is the
`case_quant_prefill_chain` pattern, already gated for iq2s.  What is NOT claimed here is the engine's exact
4-warp summation order - the parts' values are, the order that adds them is the port's tree, and the tolerance is
the double reference's.

**M-A is now 6 of the ten.** Still `todo`: `moe_grouped_s2`, `moe_hit_add`, `moe_hit_select`,
`moe_hit_grouped_s2`.  The next in the derived order is **`moe_hit_select`**.

## M-A 7/10: `moe_hit_select` - which routed experts are resident, and the ROUTING POSITION each hit fills

`moe_hit_select.comp`, from `hit_select_kernel` (`src/kernels/cuda/s2_expert_grouped.cu:623`), whose decode-path
call site is `src/core/session.cpp:866` - the token graph's first step.  One warp; `lane < k`: `e = ids[lane]`,
`s = (0 <= e < n_expert) ? res_row[e] : -1`; `hit = ballot(s >= 0)`; a hit at `at = popc(hit & ((1<<lane)-1))`
writes `slot[at] = s` and `dst[at] = lane`; lane 0 writes `count = popc(hit)`.

**TWO RULES, and the second is the one a paraphrase drops.**  Only a RESIDENT expert is a hit (`res_row[e]` is
negative for one the cache does not hold), and the hits compact in ASCENDING routing order - and `dst[at]` is the
ROUTING POSITION (`lane`), NOT the expert or the slot.  Writing the slot there hands the downstream add a
cache-order row instead of the router's; it is the same two-roles confusion `gather_rows`'s `ids[r]` and the KV
gather's `ids[id]` carry.

**A ballot is a subgroup op, so the port writes the compaction SERIALLY from one lane.**  The rule is exact
integer bookkeeping, so a different shape computes the same output - and this is why the shader carries no
barrier, no subgroup op and no atomic, which `run_gate.sh`'s census requires of it (it is deliberately NOT on the
barrier whitelist).  The output buffers are sized for the CAPACITY (k = 32), the grid the engine records.

**Fixture and arms.**  Four arms: the decode shape (10 routed, 8 experts, 5 resident with SCRAMBLED slots, one id
out of range, one negative), `k = 32` (the CUDA's maximum), `k = 1`, and a fully non-resident row (zero hits).
Entries past `count` must keep a sentinel (so a kernel that wrote the whole capacity fails), and the `res` buffer
is padded past `n_expert` with a RESIDENT value so dropping the `e < n_expert` bound counts an out-of-range id as a
hit.  Scrambled slots make `slot != position` on every hit, which is what makes `dst` falsifiable.

**Falsified.**  `gates/inject-verify.sh moe-hit-select-residency` drops the residency test (`if (s >= 0)` ->
always write), so every non-resident entry is written with `slot = -1` and the count includes it:
`FAIL  moe_hit_select (k=10, n_expert=8, decode shape, ...): 6 hits  21/32`.

## M-A 8/10: `moe_hit_grouped_s2` - the per-hit S2 expert ENTRY, which is a COMPOSITION

`moe_hit_grouped_s2` (`src/kernels/cuda/s2_expert_grouped.cu:579`) is not one kernel: it launches gate+up
(`gu_kernel`/`gu_pair_kernel`), the SwiGLU, `quantize_q8_0` on the intermediate, then the down projection.  All
four are already in the port (`s2expert_gu`, `s2expert_swiglu`, `quantize_q8_0`, `s2expert_down`) and each is
gated against its OWN oracle by `case_s2expert_tier`; **this increment is the case that gates the WIRING between
them**, which no single-kernel case can see - the gate-major layout `s2expert_gu` writes is what the SwiGLU and the
quantiser both assume, `quantize_q8_0` reads the first `n_hits*FF` floats the SwiGLU left, and `s2expert_down`
reads them back at `h*(FF/32)*34`.  So no new shader, and the PORT-MAP row names the four it composes.

**The chain runs on the device end to end** (`gu -> swiglu -> q8_0 -> down`, 3 hits, scrambled `slot_index` and
`dst_index`), and the oracle is INDEPENDENT of the device's quantiser and down projection: (a) the UP rows (the
half the SwiGLU leaves alone) are compared against the blob, pinning the gate-major base; (b) the device's
intermediate must DECODE to the device's own post-SwiGLU floats at the same position - a value check, not a byte
check, because the stored `d16` is the fp32 scale in fp16; and (c) the down rows are compared against a double
reference from the DEVICE's intermediate at the scrambled `dst_index`.  A defect in (b) or (c) also has to be seen
with the other half present, which is exactly what a wiring case is for.

**One silent failure the fixture produced, recorded because it is a class.**  The gu arm's blob makes the GATE rows
~1e5 (its `*1000` is what makes a gate/up mis-pairing visible to `case_s2expert_tier`); through `silu(gate)*up`
that overflows fp16, so the intermediate's `d16` became inf and EVERY down row was NaN - and the case's own oracle
computed NaN too, so the two AGREED.  The case counts a NaN output as a failure (correctly), which is what kept it
from being a green run over garbage.  The chain now carries its own modest scales.

**Falsified.**  `gates/inject-verify.sh moe-hit-grouped-s2-hit0-intermediate` makes `s2expert_down` read hit 0's
intermediate for every hit (`x_off = 0`): `FAIL  moe_hit_grouped_s2 (...)  473/867  worst 1.3e+06`.

**Measured (this commit covers 7/10 + 8/10).**  vega: **Intel Arc 322 passed / 0 failed / 0 skipped**, llvmpipe
**310/0/3**, radeon-iGPU 313/0/2 (`run_gate.sh` **exit 0**).  z820b (7900 XTX): **radeon_icd 318/0/1** (the 1 skip
is the pre-existing M8 cooperative-matrix case), lvp 310/0/3, nvidia (K620) 313/0/2; the box's script exits 1 on
that pre-existing skip.  Positive controls: `moe_hit_select` adds **4** verdicts and `moe_hit_grouped_s2` **1**.

**M-A is now 8 of the ten.**  Still `todo`: `moe_grouped_s2` (the grouped S2 MoE, `s2_expert_grouped.cu:1099`) and
`moe_hit_add` (the hit accumulator, `:1138`) - the batch after this one, and then M-A is closed.  The port map
reads **77 decode-path symbols - 26 kernel, 49 host, 2 todo** and passes.

## M-A 9/10 + 10/10 (ONE commit - the two share `vk_gate.cpp`): `moe_grouped_s2` + `moe_hit_add` - the last two symbols the OLD map could see (**CORRECTED 2026-10-05: M-A is NOT closed** - see the triage in `plan/DECODE-PATH-TRIAGE.md`)

**These two land as one commit because their cases and their registrations are the SAME file region of
`harness/vk_gate.cpp`** - the precedent `4/10 + 5/10` and `7/10 + 8/10` used, and it is stated here for that reason.

### 9/10 `moe_grouped_s2` - the GROUPED S2 expert entry: a COMPOSITION, but two of its four launches are NEW shaders

`moe_grouped_s2` (`src/kernels/cuda/s2_expert_grouped.cu:1099`) is the resident sibling of `moe_hit_grouped_s2`:
the same four-launch chain (`gu -> swiglu -> quantize_q8_0 -> down`).  **Whether it "composes" was CHECKED before
writing anything, as the sibling's own increment instructs.**  The middle two launches ARE the per-hit chain's
kernels (`s2expert_swiglu`, `quantize_q8_0`), but the gu and down halves are the GROUPED kernels
(`gu_grouped_kernel` / `down_grouped_kernel`, `:783` / `:845`), and those are NOT what `s2expert_gu` /
`s2expert_down` implement: the per-hit kernels take ONE activation row and a `slot_index[h]` into a single base
buffer, while the grouped ones take a LIST OF GROUPS - each with its own blob (`grp_ptr[g]`) - and EVERY ENTRY reads
ITS OWN token's activation row (`ent_tok[e]`) and writes entry-major (`ent_dst[e]`).  Neither is expressible with
the per-hit kernels, so this increment adds **`s2expert_gu_grouped.comp`** and **`s2expert_down_grouped.comp`**
(the S2 dot itself is the unchanged `common/s2_row_dot.glsl`) and reuses the other two; the port-map row names all
four.

The grouped `grp_ptr` is an array of DEVICE POINTERS and Vulkan has none, so the port takes the SAME interface
decision `native_gu_iq2s.comp` already made: ONE weights buffer plus a per-group BYTE OFFSET table (`grp_off[g]`),
so `grp_ptr[g] + off` becomes `grp_off[g] + off`.  The group count lives on the device, so the grid strides in y
(`for (g = WorkGroupID.y; g < ng; g += gl_NumWorkGroups.y)`) exactly as the source does - and the case arms the
stride three ways.

**Two rules the case pins that the per-hit sibling has no counterpart for**: the per-ENTRY activation token, and the
CAPACITY-strided gate-major split (the up half starts at `cap_entries*n_ff`, NOT the live entry count, because the
SwiGLU that follows pairs over `cap_entries*n_ff`).  The fixture is four groups (one EMPTY), six entries, capacity 8
above the live count 6, per-entry tokens, scrambled destinations, and three grid.y arms (== the count, BELOW it -
the stride - and above it).

**The oracle is independent of the device's quantiser and down step**: (a) the UP rows (the half the SwiGLU leaves
alone) against the group's blob at the ENTRY's token - pins grp_off, the per-entry activation and the capacity
split; (b) the device's intermediate must DECODE to the device's own post-SwiGLU floats; (c) the down rows against
a double reference from the DEVICE's intermediate at the scrambled `ent_dst[e]`.  The gate/up scales are overridden
to a MODEST value here: the shared `s2expert_blob` makes gate rows ~1e5, which overflows fp16 through `silu*up` -
the NaN trap the per-hit increment recorded.

**Falsified.**  `gates/inject-verify.sh moe-grouped-s2-entry-token` makes every entry read token 0's activation
(`tok = ent_tok.v[e]` -> `tok = 0u`): `FAIL  moe_grouped_s2 (...)  5270/5638  worst 3.78e+05` (the faithful form
reads 5638/5638, worst 0.0525).

### 10/10 `moe_hit_add` - the hit accumulator

`add_hits_kernel` (`src/kernels/cuda/s2_expert_grouped.cu:666`), the last step of the device hit path:
`parts[dst[h]*n_embd + i] += hit_out[dst[h]*n_embd + i]` for every LIVE hit.  Two rules, both paraphrase traps:
the row is `dst[h]` (the ROUTING POSITION, not `h` - the same two-roles confusion the KV gather's `ids[id]` and
`moe_hit_select`'s `dst[at]` carry), and the operator is `+=` (parts holds what the CPU left there).  The live count
is on the DEVICE, so the grid is the CAPACITY and rows past it return unwritten.

Four arms: count < cap (rows past the count and rows the list skips keep a sentinel), count == cap, `n_embd = 37`
(the workgroup's stride loop), and count 0 (nothing may be touched).  Every named row carries a NON-ZERO prior
value, so `+=` -> `=` is caught on the first row.

**One defect the arms produced, recorded because it is a class - and it was the ARM, not the kernel.**  The count-0
arm first read `FAIL ... 10304/10304 worst 0`: "every element bad" beside "worst deviation 0" is self-
contradictory, and the cause was the shared liveness guard - "the kernel MOVED something" - which the count-0
contract INVERTS (its claim is that nothing moved).  The guard is now `moved == 0` for count 0 and `moved > 0`
otherwise.  The arm was NOT loosened: count 0 still requires the whole buffer bit-identical to the sentinel, and it
is LIVE only when nothing moved.

**Falsified.**  `gates/inject-verify.sh moe-hit-add-accumulate` drops the accumulate (`+=` -> `=`):
`FAIL  moe_hit_add (cap=4, count=3, ...)  2624/10304  worst 0`.

### M-A IS **NOT** CLOSED (corrected 2026-10-05)

`python3 ports/vulkan/tools/check_port_map.py` reads **168 decode-path symbols - 53 kernel, 63 host, 52 todo**:
the decode path's `todo` column is **52, not ZERO**.  **The "CLOSED" this heading used to read was measured on the
qualifier-only 77-symbol map**, which could not see the bare-name call sites; the corrected map and the per-symbol
triage (`plan/DECODE-PATH-TRIAGE.md`) show **19 class-A forward-path holes** (the GDN / DeltaNet mixer for 36 of
the 48 layers, the QSA gate and indexer for the 12 QSA layers, and `gr_write`), 4 dodgeable, 7 non-selected, 22
out of the forward pass.  The 9/10 + 10/10 increment below did land and is real; it was the last two symbols the
old map could see, not the last on the decode path.  (The generator `tools/make_port_map.py` had drifted - it still classified
the two landed IQ rows as `todo` - and is fixed in this commit, so it regenerates `PORT-MAP.tsv` exactly.)

**Measured, this commit.**  vega: **Intel Arc (BMG G31) 329 passed / 0 failed / 0 skipped** (`run_gate.sh` exit 0),
intel_icd 329/0/0, llvmpipe **317/0/3**, radeon-iGPU **320/0/2**.  z820b (7900 XTX): **radeon_icd 325/0/1** (the 1
skip is the pre-existing M8 `prefill split (aligned + remainder)` case - "this device offers no M8 cooperative-matrix
config"), lvp **317/0/3**, nvidia (K620) **320/0/2**; the box's script exits 1 on that pre-existing skip, as at HEAD.
Positive controls: `moe_grouped_s2` adds **3** verdicts and `moe_hit_add` **4** (322 -> 329 on the Arc).



## STAGE 3: recorded command buffers (the CUDA-graph replacement) - **DONE AND VERIFIED 2026-10-04**

**Closed the same day the API was written.**  `case_recorded_step` (`harness/vk_gate.cpp`, six verdicts, one
commit) now exercises the whole record/replay API, and it was **proven able to fail** before it was trusted - the
two things this port's own rules ask for ("a case that only compiles is not evidence"; "a case that cannot fail
is not an oracle"):

| verdict | what it proves | Intel BMG G31 | llvmpipe | RADV (AMD iGPU) |
|---|---|---|---|---|
| `record: three dispatches` | `recorded_dispatches() == 3` at the end of the recording, `has_recording()` after `record_end_and_submit()` | PASS | PASS | PASS |
| `record: chain is byte-exact` | the ORACLE itself: three chained copies through `dispatch()` reproduce the source bit for bit | PASS | PASS | PASS |
| `record: equals single-shot` | the recorded chain == the same chain through `dispatch()`, 1024/1024 bytes | PASS | PASS | PASS |
| `record: replay reads new bytes` | new host bytes into the recording's source, then `replay_recorded()` - no re-record, no `dispatch` - produce the new bytes | PASS | PASS | PASS |
| `record: replay fixture differs` | the control for the arm above: the new source really differs (1024/1024 elements) | PASS | PASS | PASS |
| `record: replay ignores new binds` | a LATER single-shot dispatch on the SAME pipeline - which rewrites that pipeline's shared descriptor set - does not change what the replay reads | PASS | PASS | PASS |

**FALSIFICATION, which is why arm C is evidence rather than decoration.**  With `record_dispatch`'s `fresh_set`
flipped to `false` (one shared descriptor set for the whole recorded step - the trap the encoding comment names),
**three of the six verdicts fail**: `equals single-shot` 0/1024, `replay reads new bytes` 0/1024, `replay ignores
new binds` 0/1024, totals 162/0/1 -> **159 passed / 3 failed / 1 skipped**.  The flip was reverted; the file is as
committed.  So the case detects the exact defect the per-dispatch-set rule exists to prevent - and arm C is the
one that sees a shared set that a re-recording bug would otherwise hide.

One finding from writing the API, worth keeping:

* **A recorded step needs ONE DESCRIPTOR SET PER DISPATCH.**  The host-side `vkUpdateDescriptorSets` happens at
  RECORD time while the dispatches execute at SUBMIT time, so one shared set leaves every dispatch in the step
  reading whatever the LAST one bound - silently wrong output, the same class as the grouped-expert wave's single
  pointer standing in for two buffers.  `encode_dispatch`'s `fresh_set` parameter allocates a set per recorded
  dispatch (a real backend pools them).
* **`rec_fence_` IS DESTROYED BY `~Ctx` now** (it was not, and that was the loose end this block used to name).
  The command buffer goes with `cmd_pool_`; the fence does not, and a backend that owns a long-lived recording has
  to release it explicitly.

### The design, as implemented - kept because the REASONS are the useful part

The engine's decode step is a fixed sequence of dispatches re-issued every token, and a CUDA graph is how the
engine avoided re-issuing it.  The Vulkan equivalent is ONE command buffer recorded once and RE-SUBMITTED.  What
this needs, worked out against the code on 2026-10-04 and written the same day (the rule that a case must exist is
the reason the API and its case landed together rather than a half-API first):

* `harness/vk_compute.cpp`'s `dispatch()` (line ~653) allocates a command buffer per call with
  `ONE_TIME_SUBMIT`, records ONE dispatch, inserts a shader-write -> host-read barrier, submits, waits, frees.
  That is the gate's shape and it stays the single-shot path.
* **Refactor**: lift the encoding half (the `pipes_` lookup for layout/set, the descriptor update, the binds, the
  push constants, `vkCmdDispatch`) into a private `encode_dispatch(cb, pipe, bufs, push, push_bytes, groups,
  groups_y, chain_barrier)`, and have `dispatch()` call it.  `chain_barrier` adds a compute -> compute barrier,
  which a multi-dispatch STEP needs (one kernel's output is the next one's input) and a single dispatch does not.
* **New API on `Ctx`**: `record_begin()`, `record_dispatch(...)`, `record_end_and_submit()`, `replay_recorded()`,
  `recorded_dispatches()`, `has_recording()`.  Private state: one persistent `rec_cb_` + `rec_fence_` created
  once, `recorded_`, `recording_`, `have_recording_`.  Two details differ from the single-shot path ON PURPOSE:
  the buffer is NOT `ONE_TIME_SUBMIT` (it is submitted more than once) and the host-read barrier goes at the END
  of the recorded step, not after each dispatch.  `replay_recorded()` re-submits the recorded buffer and waits -
  it must not re-record anything, which is the whole point.
* **The gate case** (`case_recorded_step` in `vk_gate.cpp`; the verdicts are named `record: ...` and the six of
  them are listed in the table at the top of this file): chain the harness's own trivial COPY kernel three times
  (idempotent and byte-exact, which is why it is the right kernel for this).  Record the chain once; run the same
  three copies through the single-shot `dispatch()` path as the reference; require the two outputs byte-identical
  (and the reference itself byte-exact against the source, so the oracle is checked too).  Then PROVE the replay
  is the recording rather than a re-record: write NEW bytes into the source buffer on the host, `replay_recorded()`,
  and require the destination to equal the new source.  The third arm - an interfering single-shot dispatch on the
  same pipeline between recording and replay - is the one that catches a SHARED descriptor set; flipping
  `fresh_set` to false was measured to fail 3 of the 6 verdicts, which is what makes the case an oracle rather
  than an agreement.  No HIP device is needed to see any of it.
* **Why this oracle and not the plan's**: PORT-PLAN stage 3 says "compare tokens to the HIP path", which this box
  can no longer produce (see the plan's note).  The single-shot path is the substitute, and it is a stronger
  oracle for this particular question, because it isolates exactly the thing that changed - the recording - with
  everything else held equal.

## STAGE 4: device-local memory, staging and fit accounting - **DONE AND VERIFIED 2026-10-04**

**What it was for.** Stage 1-3's layer allocates host-visible coherent memory on purpose (a correctness device:
staging plus fences would only add ways for the GATE to be wrong).  The engine's backend cannot do that - the
things that live on the card go in memory with no mapping at all, and every byte in and out goes through a
staging buffer.  That path now exists, is exercised by the gate, and the engine's VRAM plan is fitted against the
driver's own numbers instead of being described.

**What the three implementations actually offer**, printed by the case rather than assumed (this is the reason
the layer enumerates types instead of answering "is there device-local memory?"):

| device | memory types | device-local | of those, UNMAPPABLE | staging lands in |
|---|---|---|---|---|
| Arc Pro B70 (ANV) | 7 | 5 | **3** (types 0/1/4 - real VRAM), the other 2 mappable (the BAR) | heap 1, system RAM (34.7 GiB) |
| AMD iGPU (RADV) | 11 | 6, **all of them on the heap RADV calls device-local** (it flags GTT that way) | **3** (types 0/1/7) | a mappable heap-0 type |
| llvmpipe | **1** | 1 | **0** - its single type is device-local AND mappable | its one heap (nowhere else exists) |

**The arms, all six on all three implementations** (the totals below are the whole suite):

| verdict | what it proves | Arc | llvmpipe | RADV |
|---|---|---|---|---|
| `stage: device-local chosen` | the flags on the buffer match the memory type, not what we asked for | PASS | PASS | PASS |
| `stage: vram is not mappable` | the chosen VRAM type has NO mapping where the device has one | PASS | **SKIP** (no such type) | PASS |
| `stage: staging is mappable` | the transfer buffer is HOST_VISIBLE + HOST_COHERENT | PASS | PASS | PASS |
| `stage: staging charged by heap` | a transfer buffer is charged by the heap it lands in, not by what it is for | PASS (host) | PASS (VRAM: single heap) | PASS |
| `stage: device round trip` | pattern -> staging -> copy kernel -> staging -> host, byte-exact | PASS | PASS | PASS |
| `stage: forced-staging trip` | the same with the staging path FORCED, so it runs where a mapping exists | PASS | PASS | PASS |

`plan_fit` is also gated as a PURE function (five arms, no device): fits inside budget, an exact fit is a fit,
the item that crosses is named and the walk stops, a droppable cache is dropped rather than fatal, and neither an
empty plan nor a zero-byte budget "fits" (both would be assertions over an empty input).  On the device the plan
is fitted against the driver's figure, printed, and cross-checked by a recount written in the case:

```
INFO  vram plan                    budget 28.64 GiB (driver free minus reserve)
INFO                                 weights resident   wanted  14.32 GiB
INFO                                 expert cache       wanted  14.32 GiB  (droppable)
INFO                                 kv cache           wanted   3.58 GiB
INFO                               FITS: 17.90 GiB resident, 14.32 GiB dropped in 1 item(s), first overflow -1
```

**FALSIFICATION, and one honest negative.**  A four-byte offset error in the upload's copy (`c.srcOffset = 4`)
makes BOTH round trips fail (0/1024, 174/0/1 -> 172/2/1), so the round trip is not vacuous.  **Deleting the
post-copy barrier changes NO verdict on the Arc or on llvmpipe** - measured, not assumed - so the case proves the
transfer (direction, offsets, mapping, and a barrier whose absence is observable), NOT that each barrier is
necessary; the barriers are required by the spec and are not gated on these drivers.  Recorded here so the next
reader does not mistake a green run for coverage of that defect class.

**A bug the new arm caught in the new code, worth keeping as a lesson.**  `has_nonvisible_device_local()` was
written as "a device-local type exists" and then used to decide whether to assert that the chosen type is
unmappable.  llvmpipe has exactly one type, it is device-local AND mappable, so the arm FAILED there (171/0/2 ->
171/1/2) and reported a defect the port did not have.  The accessor now records what the SELECTION did (pass 0 of
the two-pass choice succeeded).  The general form: an accessor that re-derives a property from the table it was
chosen out of can disagree with the choice, and the disagreement surfaces as a failure on one implementation only.

**The account rule, stated once** (it is in the header too): every `alloc()`/`alloc_device()` is charged to the
VRAM account whatever heap the driver puts it in - conservative on purpose, so the refusal rule cannot change with
which memory type a driver happens to prefer - and a staging buffer is charged by the heap it lands in, because a
transfer buffer is not model memory.  Stage 4 did not relax the display contract: the refusal now names the
account it refused in.



## STAGE 5 (started): the matrix path on Intel - **THE ARC'S MATRIX UNITS ARE REACHABLE, AND WERE BEING MISREPORTED**

**The finding, measured with the driver's own property list** (`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`,
all three implementations, 2026-10-04):

| device | extension | configs | the floating-point one |
|---|---|---|---|
| Intel BMG G31 (XMX) | advertised, revision 2, feature true | 6 | **M8 N16 K16**, f16/f16 -> f32, subgroup scope |
| RADV Raphael (RDNA2) | **not advertised** | (14 returned by an entry point the driver does not advertise) | all M16 N16 K16 |
| llvmpipe | not advertised | none | - |

So the port's `gemm_coopmat` case demanded **M16** N16 K16 - a rule written against the departed Radeon's list - and
therefore reported the Arc as having "no usable config", which reads as "the matrix units are unreachable here".
They are not: **BMG's op is 8 rows wide.** The tile is a property of the DEVICE, per generation, and the port now
selects the pipeline by shape for exactly that reason.

**What changed.** `shaders/gemm_coopmat_m8.comp` (the sibling of the M16 kernel, 8x16 tile, same dispatch shape and
bounds guard); `DeviceInfo::cm_m/cm_n/cm_k` hold the SELECTED config (M16 preferred where a device offers both,
because that is the kernel this port verified on a Radeon, else M8); `gemm_shape_ok` takes the tile as parameters,
so the precondition is one rule with the device's numbers in it rather than two rules that drift; the case picks
the pipeline by shape, PRINTS which one it ran, and gained a PROMPT-SHAPED case (128x256x256, a real prefill tile
grid rather than 64x64).

**Evidence.** Intel: **180 passed / 0 failed / 0 skipped** - the Arc now runs the matrix path and no longer skips
anything, including all four coopmat shapes against a double-precision reference (worst relative error 0.0101 on
the prompt-shaped grid, and that is on elements whose absolute error is under the 1e-4 floor - the case prints
both).  llvmpipe 172 / 0 / 2, RADV 175 / 0 / 1 (both still skip coopmat: their drivers do not advertise the
extension, and skipping a device that does not advertise an extension is correct - unlike skipping one that does).

**Falsification, and it is the interesting kind.** Forcing the M16 kernel on the Arc - ignoring the device's tile -
does NOT fail pipeline creation.  It **runs and computes wrong numbers**: 2560/4608 elements correct in the square
case, 16896/33280 in the prompt-shaped one, 176 passed / 4 failed.  So a wrong tile selection is a silent wrong
answer on this driver, not a loud one, which is precisely why the pipeline has to come from the property list and
why the case prints the shape it used.

**What this settles for the rest of stage 5.** The prefill GEMM on the target card has matrix units at **M8 N16
K16**, so the port's own GEMM tile is an 8-row tile with a plain-FMA fallback for devices with no config (llvmpipe,
and any driver that does not advertise the extension) - and the selection has to be per device, never per vendor.



## STAGE 5: the PREFILL GEMM - **DONE FOR THE LAYOUT THE ENGINE CALLS, WITH ITS OPERAND CONVERSION**

**What "the GEMM" means here.**  The engine's prompt path calls `Gemm::f16` / `Gemm::bf16`
(`src/prefill/gemm.cu`), whose cuBLAS call is `(CUBLAS_OP_T, CUBLAS_OP_N)` with m = N, n = T, k = K:

```
Y[T x ldy] = X[T x K] . W[N x K]^T          f16 operands, f32 accumulate
```

**The WEIGHT is the transposed operand** - an output row is an output feature, and the reduction runs over the
contiguous axis of both inputs.  That is a different layout from `gemm_coopmat.comp` / `gemm_fma.comp` (which are
`C[MxN] = A[MxK].B[KxN]`, the shape and no-CMA fallbacks), and getting it backwards is the classic SILENT GEMM
bug: every number stays plausible.  So it is stated in the kernel, each loader says which layout it reads, and the
gate compares at shapes where `N != K` - a transposed read cannot survive that.

**Three new kernels.**

| shader | what it is |
|---|---|
| `bf16_to_f16.comp` | the engine's own operand conversion (`prefill/gemm.cu`'s `bf16_to_f16_kernel`): widen, **CLAMP finite values past fp16's range to +-65504 rather than letting them become Inf** (the #540 rule), then round-to-nearest-even with the port's gated converter. This is the route from the engine's bf16 storage to the matrix units on Intel too, where the driver's list has no bf16 config. |
| `gemm_prefill_f16_m8.comp` | the GEMM on XMX at the tile Intel actually has (8x16x16). The body lives in `common/gemm_prefill.glsl`, where the tile is a compile-time knob - so the RADV/WMMA M16 sibling is a three-line file when a Radeon is in the box. Not shipped: an untested tile variant is exactly the claim this port does not make, and the FMA kernel below covers those devices correctly. |
| `gemm_prefill_fma.comp` | the same contract with **no shape precondition at all** - one invocation per output element, f16 operands read as fp16 and widened (exact), no cooperative matrix, no subgroup op, no barrier. It is the decode row, the ragged prompt, and the devices without a usable config. |

**Why there are two, and why the case tests the SPLIT.**  The tile path has no ragged edge: `tiles_t = t / TM`
rounds DOWN, so rows the tile cannot cover are silently not computed.  A real prefill of a prompt whose length is
not a multiple of the tile therefore runs BOTH kernels - aligned rows on the matrix units, the remainder on FMA -
and the case exercises exactly that (`T = 37 = 4 x 8 + 5 rows`) rather than each half alone, including a check that
the tile dispatch left the remainder rows untouched.

**Evidence (Arc Pro B70, this box):** **189 passed / 0 failed / 0 skipped**, ten new verdicts - `bf16_to_f16`
bit-exact over 1024 patterns with the clamp measured as exercised on 437 of them, three FMA shapes (1x64x64 decode
row, 17x13x5 ragged with `ldy > n`, 40x64x96 mid-prompt), three matrix-unit shapes (one tile, 64x512x512,
128x256x256 with `ldy > n`), and the split.  `ldy > n` is exercised on purpose: a kernel that assumed `ldy == n`
would smear the columns between n and the stride, and the case leaves those columns NaN and requires them to
survive.

**Falsification.**  Reading the weight RowMajor instead of ColumnMajor - the transposed-operand bug itself - fails
all four matrix-unit arms (`0/128`, `14/32768`, `16/32768`, `320/2368`, totals 189/0/0 -> 185/4/0) while the FMA
arms stay green, which is what makes this a test of the LAYOUT rather than of the arithmetic.  And the clamp's own
arm counts the entries the clamp changes, so it cannot be deleted with the case still green.

**A DEVICE-LAYER GAP THIS FOUND, and it blocks stage 6 rather than stage 5.**  The engine reaches a row slice of Y
by POINTER arithmetic (`f16(tc_x_, tc_w_, Y + t0 * ldy, ...)` in `gemm.cu`, and `X + t0 * K` for the activations in
row slices), because cuBLAS takes pointers.  The harness binds **offset 0** on every descriptor, so the case has to
give each half its own X and Y buffer instead of pointing both at one.  A Vulkan backend needs
`VkDescriptorBufferInfo::offset` (or one buffer per slice, which is what the engine's own conversion buffers
amount to) before it can serve a real prompt.  Recorded here because it is invisible until someone tries to pass a
slice.



## STAGE 5b: the QUANTISED prefill path - **and a finding that needed no new kernel**

**Reading the engine first paid off again.**  The prompt path has TWO routes to quantised weights, and only one of
them is a GEMM-shaped kernel:

* **`Gemm::native`** (what the prefill calls): it does NOT use an MMQ kernel at all.  It **dequantises** the ggml
  blocks to f16 (`strata::kernels::dequant_f16`) and then calls the ordinary f16 GEMM - the same one this port
  gated an hour earlier.  So the quantised prefill needs an IQ *dequantiser*, not a new GEMM.
* **`iq_mmvq`** (the expert tier's multi-token path): the row kernels, which the engine calls with a `ncols`
  argument and specialises at 1 / 2 / 4 / 8 columns (`mmvq_multi_kernel`).

**The port's row kernels already take `ncols`** - `iq1m_mmvq.comp` and its siblings loop
`for (c = 0; c < pc.ncols; ++c)` - so the quantised multi-token path needed **no new kernel**: it needed arming.
The existing arms sat at ncols 1 and 2, i.e. the single-token and barely-multi-token cases.  Added, for all seven
of the resident model's formats: an arm at **ncols = 8** (the widest count the engine specialises for, so this is
the real multi-token expert path) and, for IQ2_S, one at **ncols = 32**, past the engine's specialisations, where
an activation-row stride that is right for a narrow row and wrong for a wide one shows up.

**A composed case for the interface, because the halves agreeing with their own oracles does not make the wiring
right:** `case_quant_prefill_chain` runs the DEVICE's `quantize_q8_1` and feeds its output to the DEVICE's
`iq2s_mmvq`, with the oracle reading those same bytes.  IQ2_S is the format to hang it on: 20 of the model's 48
layers store their gate/up experts in it.

**AND THE CASE'S FIRST ORACLE COULD NOT FAIL - which is the finding worth keeping.**  The injection was the
obvious one: delete the quantiser's per-column block stride so every column's blocks land in column 0's region.
The quantiser's own case caught it (`ncols=2` arm: 2105 of 5760 bytes) - and **the chain case PASSED**, because an
oracle that reads the device's own bytes follows the device's own mistake: both sides then read the same misplaced
data, and the columns that read nothing compare equal to an expectation of nothing.  Fixed by asking a question the
bytes cannot answer by themselves - **per-column liveness**: with random activations, a column that read nothing
returns zeros, so the case now requires that no column comes back all-zero.  With that, the same injection fails
the case (7 of 8 columns dead) and the real tree passes.  The general form: **an oracle built from the thing under
test is blind exactly where that thing is systematically wrong**, and a liveness/coverage assertion is what closes
it - the same class as the existing `mass > 1e-3` guard, one level down.

**What is still missing for quantised prefill:** the IQ **dequantiser** (`dequant_f16` / `iq_dequant_f16`, which
covers ggml types 16/17/18/21/22/29 plus the Q2_0, IQ4_NL/IQ4_XS and k-quant families through a second path).  The
port's per-format dot includes already encode each block layout FOR A DOT, so a hand-written dequantiser would be a
second copy of that layout - the shape of bug this port keeps finding.  The non-duplicating route, recorded here
for whoever takes it: **derive the dequantiser from the dot by a one-hot activation** (a q8_1 block with a single
code set to 127 and d = 1/127 gives exactly 1.0 at one position and 0 elsewhere), so the dequantised value IS the
dot and there is nothing to drift; slow, but this is a verification path, and the same trick gives one dequantiser
per format for free.



## THE ATTENTION BLOCK, KERNEL 1: the short-step decode attention - **the engine's contract, re-derived**

**What was ported.**  `native_flash_attn_short_step` (`src/kernels/cuda/native_flash_attn.cu`, called from
`src/core/layer.cpp`), which the engine's own header describes precisely enough that nothing had to be guessed:
geometry **Q24 x 256, KV2 x 256**, one query/sequence, **scale = 1/16**, no ALiBi/softcap/sink; q/out contiguous
`[24, 256]` f32; k/v `[capacity, 2, 256]` f16 with only the first `width` rows initialised; an optional **additive**
f16 mask of 256 logits broadcast over the heads, `-inf` meaning masked.  The 24 query heads share 2 KV heads - the
GQA grouping the CUDA encodes as `kv = head / 12`.

**What is NOT claimed, up front.**  This is not the pinned CUDA bit for bit, and it does not try to be: that kernel
is a VECTOR flash attention pinned to one compiled binary - 128 threads, a lane/float2 mapping, an `__shfl_xor`
tree over 8 lanes, a streaming online-max pass, and an fma order chosen to reproduce that binary's rounding (its
comments say so outright).  The port's reduction rule bans subgroup ops anyway.  So the port carries the ALGORITHM -
score, additive mask, softmax, weighted sum - onto one workgroup per query head with one output dim per thread, and
measures the difference against a double-precision reference instead of asserting it away.  Engine-token parity for
this kernel is a stage-6 question with its own tolerance argument.

**Design, and why it is not the CUDA's.**  Each thread OWNS one key's score (thread c does the whole dot for cell c
serially over 256 dims), so the max and the sum are **two workgroup reductions per token per head** - not one per
key, which is what a per-cell reduction would cost.  `wg_reduce.glsl` gained `wg_max` for it, sharing the existing
array behind its own leading barrier.  The softmax is two-pass (max, then exp and sum) rather than streaming: both
are exact formulations of the same quantity and differ only in f32 rounding order, which the arms measure.

**A shape predicate, for the same reason the GEMM has one.**  `attn_short_shape_ok(width, capacity)` = width in
[1, 256] and capacity >= 256.  The kernel drops every key past `width`, so a wrong width is a silently wrong token
rather than an error - the engine carries a status word for exactly this, and the case's first arm is the contract
itself (width 0 and width 257 must be REFUSED, one live key and a full window must be allowed).

**Evidence.**  Arc: **204 passed / 0 failed / 0 skipped**.  Five arms - 1 live key (the first decode step, which is
bit-exact against the reference because there is no softmax to round), 3 live keys, a full 256-key window, a full
window with an additive mask (-inf on a third of the keys), and a masked half window - each 24 heads x 256 dims, each
against the double oracle.  The two KV heads carry DIFFERENT data in every arm, which is what makes the GQA grouping
falsifiable.  Tolerance is a relative bound with an absolute floor, and the case prints the worst RELATIVE and worst
ABSOLUTE deviation beside the reference magnitudes: a masked softmax over 170-odd live keys spreads the weights over
many orders of magnitude, so the worst relative error (2.25e-2) sits on an output whose absolute deviation is under
the floor - measured, not assumed.

**Falsification.**  Changing the grouping to the plausible wrong one (`head % KV_HEADS`, interleaved instead of the
engine's 12-blocks) fails all five arms with exactly half the values correct (3072 of 6144) - the 12 heads that map
to the same KV head under both rules.  A grouping bug is therefore not something this case can miss, which was the
point of giving the two KV heads different data.

**What the attention block still needs.**  The prompt path (`qsa_prompt_attn.cu`, 74 KB, and `qsa_select.cu`), the
QSA selection and indexer logic (`native_qsa*.cu`), the KV gather/write path at f16 (the port has the q8_1 append
and gather already), and a tiled version of this kernel - the serial 256-dim loops are the right trade for a
correctness kernel and the wrong one for throughput.



## THE ATTENTION BLOCK, KERNEL 2: the f16 KV window - and the engine's launch rule that a recorded buffer makes real

**What was ported.**  `kv_gather_kernel` / `kv_gather_step` (`src/kernels/cuda/qsa.cu`): the F16 sibling of
`kv_q8_gather`, same addressing, no dequantisation, because the pool itself is f16.  It is the producer of the
`[id][kv_head][head_dim]` window `attn_decode_short` consumes - the other half of a decode step.

**The pool layout is not what a fixture wants to assume.**  The pool's ROW INDEX CARRIES THE HEAD:

    row = (page * kv_heads + h) * page_size + (cell % page_size)     then  pool[row * head_dim + d]

i.e. `[page][kv_head][page_size][dim]`, NOT `[row][kv_head][dim]`.  A fixture written the second way - which this
one was, first, and it failed 6144 of 6144 values against a kernel that was right - is the mistake worth recording,
because the two layouts have the SAME TOTAL SIZE and every index in them is plausible.

**`ids[id]`, not `id`.**  The destination row index is the position in the selection; the cell it holds is the
selection's value.  Confusing them yields a window that is a PERMUTATION of a correct one, and a softmax over the
keys cannot see a permutation - it is symmetric in the key index.  That is why the composed gather-then-attend arm
runs with a POSITION-DEPENDENT MASK: with one, the permutation moves a different mask value onto each cell and the
output changes.  Measured: the injected `cell = id` fails the composed arm at worst rel error 9.19e+03.

**THE GRID IS THE CAPACITY, and the port now has a measurement for it.**  The engine's launcher says so in as many
words - "a captured layer would allocate room for token 1 and then index past it at token 500" - because a GROUP
COUNT IS BAKED INTO A RECORDED COMMAND BUFFER while a buffer is re-read at dispatch time.  The port has had recorded
command buffers since stage 3, so the rule is not hypothetical here.  The case records the gather, changes `n_ids`
from 6 to 8 in the step buffer, and replays:

    replay after n_ids 6 -> 8, capacity grid:           8 of 8 rows present
    replay after n_ids 6 -> 8, NEGCTRL live-count grid: 6 of 8 rows present

The NEGCTRL arm asserts the DEFECT (it passes only if the live-count grid really does lose rows), in the idiom the
port already uses for `f32_to_f16_trunc`.  Two rows is what a token-1 grid costs at token 8 - and it is silent.

**A case that described the rule without testing it.**  The q8 sibling's case sized BOTH its window and its grid
from `n_ids`, so the guard region its own verdict text called "incl. the guard region" did not exist: the window WAS
the live count.  Fixed to size from a capacity of 8 and dispatch over it, which is what makes the guard region real
(and the case now compares 4224 values per side instead of 3072).

**Evidence.**  Arc: **209 passed / 0 failed / 0 skipped** (five new verdicts).  The f16 gather's window is compared
element for element against the pool rows the selection names, over 16 pages in REVERSE order, ids that span pages
and page offsets and include cell 0 and cell 255; the 250 surplus rows of the 256-row window must keep their
sentinel, which is the guard; the composed arm puts 6 gathered rows through the attention and matches a
double-precision oracle computed from the POOL at worst rel 1.67e-06.

**What the KV path still needs.**  The append half at f16 (the port has the q8 append), `kv_append_q4`/`kv_gather_q4`
(the int4 pool with its FWHT-256 rotation - `fwht256_kernel` is a separate piece of arithmetic), the paged ring
table and the host staging (`kv_stream.cu`: `kv_ring_table`, `kv_stage_from_host`), and the SELECTION that fills
`ids` in the first place (`qsa_select`, `native_qsa*`) - the gather takes `ids` as an input and the case supplies
them, which is the boundary the port has drawn so far.



## THE ATTENTION BLOCK, KERNEL 3: the selection - and the first chain that runs the whole decode step

**What was ported.**  `block_scores_kernel` and `block_topk_kernel` (`src/kernels/cuda/qsa_select.cu`; the engine keeps
the second as `qsa_block_topk_ref`), i.e. the QSA indexer's block scores and the weighted top-k that produces the very
`ids` the gather now consumes.  The decode step's data path is complete end to end: pool -> scores -> selection ->
gather -> attention, all four in the port, all four gated.

**The contract, from the engine's header, is unusually complete and the port follows it literally:**

    scores: one warp per (query, block); relu PER INDEXER HEAD, summed; the tail block n_bid scores the `dead` key
            (not its pooled row) and takes +1e9 WHEN IT HAS CELLS (n_kv % R != 0).  R = 4 cells per block,
            IDX_DIM = 128, IDX_HEADS = 4.
    ids:    [nq, cap], CELLS, ASCENDING; the `width` cells with the largest block scores, each block WEIGHTED BY ITS
            CELL COUNT, ties to the lowest cell index.  While n_kv <= width the selection is the identity.

**THE WEIGHT IS THE CELL COUNT, NOT R.**  A complete last block holds `n_kv % R == 0` cells: it is SKIPPED everywhere,
so its score cannot make it selectable, and a 1-cell tail contributes 1 to the budget, not 4.  Getting this wrong
selects cells that do not exist - the danger is silent, because those ids then index a cache row beyond the live
range.  Both are gate arms, and the injection "every block weighs R" fails four arms plus the chain.

**The ordering is part of the contract.**  The ids come out ascending because the emission loop walks blocks upward
and each thread owns a contiguous range with an exclusive prefix.  Downstream, the POSITION is the window row and the
attention's mask is indexed by it, so an id list with the right cells in the wrong order is a different output.  The
injection "cells at the threshold taken from the TOP of the block" fails five topk arms and all three chain arms.

**Two things the port had to decide, both documented in the shaders.**  `order_key`'s `s + 0.0f` in the source
cancels the sign of zero, so -0.0 and +0.0 compare EQUAL; the port spells that out as a comparison rather than
leaving it to an optimiser that may fold the addition away, and the gate's oracle orders by the same keys (otherwise
a tie in f32 becomes a coin flip between two implementations that are both right).  And the scores' reduction is a
barrier tree over the workgroup instead of the source's `__shfl_xor` tree - this port's rule - so a score can differ
from the CUDA's in its last bits: measured against a double oracle at worst rel 1.32e-06, which is the same caveat
the engine's own tensor-core variant carries ("another summation order: not bitwise").

**Evidence.**  Arc: **221 passed / 0 failed / 0 skipped**.  Scores: 3 queries x 24 blocks, the dead key deliberately
far from every pooled row, the blocks past n_bid checked for having been left alone.  Selection: SEVEN arms - a
budget landing inside a block, three equal keys at the boundary, a zero-weight tail with the biggest possible score,
a 1-cell tail, both identity cases (n_kv < width and n_kv == width), and the everything-but-one-cell case - each
compared entry for entry against the rule implemented directly, with the ids past `width` required to stay sentinel.
Chain: scores -> 6 ids -> gather -> attention, with the ids, the gathered window and the attention output all
compared against oracles built from the POOL.

**Falsified (three injections, each applying to every site):**  weight = R for all blocks -> 4 arms + 3 chain arms;
`+1e9` without the has-cells test -> the scores arm (worst rel 8.55e+06); ties from the top of the block -> 5 arms +
3 chain arms.  One caution recorded for the port's own falsification scripts: an anchor that appears more than once
must be replaced at EVERY site, and the script must fail loudly if it is not - the first run of the weight injection
silently patched nothing and reported the unmodified kernel passing, which is a false negative wearing the costume of
a clean result.

**What the selection still needs.**  `qsa_block_scores_tc` / `_wmma` (the tensor-core variants, prompt path only -
the engine's header notes they are NOT bitwise with the warp kernel), the register and cluster top-k variants
(`block_topk_reg_kernel`, `block_topk_wide_kernel`, `block_topk_cluster_kernel` - the engine asserts they produce the
SAME ids, which is exactly the parity the port would gate them against), and `qsa_index_kernel` / `kv_q8_append`'s
siblings in the per-token path.



## THE ATTENTION BLOCK, KERNEL 4: the KV append - the cache's write half, and the first chain that closes the loop

**What was ported.**  `kv_append_kernel` (`src/kernels/cuda/qsa.cu`): one token's current K/V into the paged f16
cache.  This is the WRITE half of the pair whose read half is `kv_f16_gather`, and the two share one row formula:

    pos  = step[kStepPos]                                    the cell being written
    page = table[pos / page_size]
    row  = (page * kv_heads + h) * page_size + (pos % page_size)
    pool[row * head_dim + d] = fp16(cur[h * head_dim + d])

The decode step's data path is now closed at both ends: a token's K/V is written into the pool by this kernel, the
selection picks cells, the gather copies those cells into the window, and the attention reads it.

**Four things the port had to get right, all of them gated:**

1. **The conversion is the arithmetic.**  The current K/V arrives as f32 and the pool is f16, so the engine's
   bit-exact conversion (`common/f16_bits.glsl`) does the work and the case compares pool BYTES against it - not a
   tolerance.  A wrong rounding here is invisible until it moves a logit.
2. **A negative page means NO write at all** - not a write to row 0, and not a skipped host copy.  Measured: the
   injection "treat a non-resident block as page 0" is caught by exactly this arm and by nothing else, which is the
   whole argument for having an arm per rule rather than one broad comparison.
3. **Two destinations, one kernel** - the same choice the q8 append already makes in this port.  The VRAM row uses
   the PHYSICAL page from the table; the host identity layout uses the LOGICAL page index, always written.  They are
   two formulas selected by `host_layout`, not two kernels kept in step by hand.
4. **Re-appending a position overwrites the same cell**, which is what a recomputed decode step needs - and the grid
   is fixed geometry (`kv_heads * head_dim`), so a recorded launch is replay-safe by construction.

**Evidence.**  Arc: **226 passed / 0 failed / 0 skipped** (five new verdicts).  Three positions (a page's middle, its
first cell, its last cell) each leave every other element of BOTH pools at the sentinel - 786432 comparisons, which
is what makes "one cell, the right row" a measurement rather than a claim.  Both host-layout branches of the
non-resident case (nothing written vs the identity row written anyway, with the untouched count printed).  And the
chain: six tokens appended at cells 31..36 - deliberately across a page boundary, so two page-table entries are in
play - gathered into the window with 0 mismatches, attended, and matched against a double-precision oracle at worst
rel 8.37e-06.

**Falsified (three injections, each verified to have applied at exactly one site):**  the VRAM row using the LOGICAL
page index instead of the table's entry -> 5 arms, chain at worst rel 1.57e+05; a non-resident block treated as page
0 -> the non-resident arm; both halves of the grid writing K -> 4 arms incl. the chain.

**A harness bug worth recording, because its symptom was a confident lie.**  The oracle lambda closed over the
REVERSED page table even in the non-resident arm, so it predicted a write the kernel was right to skip, and the
arm's own diagnostic then announced "the identity row written anyway" for `host_layout = 0`.  The kernel was correct
throughout.  Two lessons: an oracle must take the actual bound inputs as parameters (a closed-over fixture is a
second source of truth), and a diagnostic derived from the same faulty fixture misreports in a confident voice.

**THE HARNESS HAD A CASE-COUNT CEILING, and this increment hit it.**  The first full run after the append case came
back `FAIL radeon_icd` with no numbers at all - `vkAllocateDescriptorSets -> VK_ERROR_OUT_OF_POOL_MEMORY` at the END
of the run, because `maxSets` was a hardcoded 64, one set per pipeline, nothing recycled.  Four new cases had taken
the gate past it.  It looked exactly like a kernel failure on RADV and was not one: the same binary passed on the
Arc, and the pool is the harness's, not the kernel's.  Fixed by growing: `set_alloc()` allocates a fresh pool when
the current one is full and SAYS SO on stderr (`descriptor pool 2 created`), so the next time a case pushes past 64
sets the reason is printed instead of inferred.  Worth remembering as a class: **a fixed capacity in the test harness
is a limit on the test suite**, and it fails at whichever implementation is exercised last.



## THE ATTENTION BLOCK, KERNEL 5: the Q4_0 KV path - a rotation the design depends on, and a design claim gated

**What was ported.**  Three pieces of `src/kernels/cuda/kv_q4.cu`: `fwht256_kernel` (the orthonormal 256-point
Walsh-Hadamard rotation), `kv_append_q4_kernel` with its `q4_group`/`q4_store` pair (ggml's Q4_0 group, with the
engine's deterministic tie rule), and `kv_gather_q4_kernel` (the dequantising reader).  The KV storage formats now
match the engine's three - f16, q8 and q4_0.

**THE DESIGN IS THE INTERESTING PART, and the gate tests the design, not just the kernels.**  From `kv_q4.hpp`: K
and V are rotated by H before quantisation, the QUERY is rotated too, so `<Hq, Hk> = <q, k>` and the scores are
unchanged, and the attention output - a convex mix of rotated values - is rotated back once at the end.  That is why
the gather does NOT inverse-rotate: the whole path lives in the rotated basis, and the only reason it is legal is
that H is ORTHONORMAL.  A rotation that merely "spreads outliers" would change every score.

**The rotation is gated three ways, because one of them cannot see the most likely bug:**

    the numbers against an EXPLICIT Hadamard matrix  H[i][j] = (-1)^popcount(i & j) / 16
    H(Hx) = x           (||H(Hx) - x||/||x|| <= 1e-6 measured 1.1e-07)
    <Hx,Hx> = <x,x>     (2.26e-08 relative)
    + the design claim itself, through the real attention kernel (below)

A double-precision rerun of the butterflies would share the port's own STRUCTURE, and the two invariants cannot see
a flipped sign convention either - the other convention is ALSO orthogonal and self-inverse, merely a different
basis with wrong scores.  Only the explicit matrix pins the convention, and the injection proves the point: flipping
the butterfly branches fails that arm while both invariants still pass.

**The group rule, and the one detail a paraphrase gets wrong:** the scale comes from the SIGNED EXTREME
(`d = mval / -8`, where `mval` is the value of largest magnitude and, among equals, the LARGER value), not from
`|max|`.  The sign is what maps the extreme to the bottom of the code range with an offset of 8; taking `|max|`
gives a positive scale and a wrong code set - "a bit more quantisation noise" to a perplexity run, a wrong basis in
the cache.  Codes are `trunc(x/d + 8.5)` clamped to 0..15 (TRUNCATED, not rounded), the tie rule is what makes the
result independent of a fold order (the engine added it on merge), and - unlike this port's q8 append, which
quantises against the STORED f16 scale - the codes here use the EXACT f32 scale while the stored one is f16.  Two
sibling kernels, two conventions, both faithful to their sources; the gate pins each where it belongs.

**Evidence.**  Arc: **233 passed / 0 failed / 0 skipped**.  Five special groups compared bit for bit against a host
transcription of the rule (a plain group, a `+3.5/-3.5` tie, a subnormal scale, an all-zero group, one whose codes
clamp), with the 288 bytes of the cell verified by POSITION and 0 stray bytes elsewhere.  The tie arm re-presents
the same values in the opposite order and requires the SCALE to be identical - deliberately not the whole block,
because reordering a tie moves values between elements and each element's own code legitimately changes (the first
version of this arm compared all 18 bytes and failed a correct kernel).  Round trip: the gathered window is within
**0.4719** of the rotated original against the group's own bound `|d|/2 = 0.4742`.  And the design claim, run
through the actual attention kernel: rotate q, quantise+rotate K/V to Q4_0, attend, de-rotate - worst absolute
deviation **0.644** from the unrotated attention, against a window that is itself off by up to 0.474.

**Falsified:** the scale from `|max|`; codes rounded instead of truncated; the butterfly's sign convention flipped
(caught by the matrix arm alone - see above); the nibble halves swapped in the reader.

**A fixture bug worth recording, because its symptom looked like a kernel failure.**  The Q4 pool was sized
`rows * bytes_per_head`, but the row index CARRIES the head (`(page*kv_heads + h)*page_size + off`), so the pool
needs `rows * kv_heads * bytes_per_head` - the appends then wrote entirely beyond the buffer, a driver silently
dropped them, and the gate reported a pool that was never written.  The f16 and q8 pools in this port had the same
row formula and did NOT have this bug, because their element counts happen to include the head factor; a
byte-per-head layout is where the factor becomes easy to lose.

**What the Q4 path still needs.**  The prompt-path batch append (`kv_append_q4_batch_kernel`, with its staging pool),
`kv_hybrid` (q4 for V, q8 for K - `kv_hybrid_parity.cpp` is the engine's own reference for it), the ring/staging
machinery in `kv_stream.cu`, and the per-token q4 path's rotation call sites in `prefill.cpp` (which is where the
query and the attention output get their FWHT).



## THE ATTENTION BLOCK, KERNEL 6 (no new kernel): the hybrid mode - K in INT8, V in rotated Q4_0

**What was added.**  Nothing to `shaders/`.  The hybrid mode IS the wiring: `kv_q8_append`/`kv_q8_gather` for K,
`fwht256` + `kv_q4_append`/`kv_q4_gather` for V, `attn_decode_short` over the two windows, and one `fwht256` on the
output - plus a gate that pins the asymmetry.  This is `KV_MODE 3` of the engine's own reference
(`src/kernels/kv_hybrid_parity.cpp`), whose header describes it in one line: **INT8 K, rotated Q4_0 V**, then the
output's inverse Hadamard.

**THE ASYMMETRY IS THE POINT.**  Only V is rotated.  K keeps eight bits and does not need to be spread, and because
K stays in the original basis the QUERY does too (`<Hq,Hk> = <q,k>` only has to hold for the side that moved).  So:
rotate what you quantise to four bits, leave the other side and the query alone, and take the output back with one
Hadamard at the end.  A port that "helpfully" rotated K as well would change every score; one that skipped the output
rotation would return a correctly-scored answer in the wrong basis.  Both are what the arms exist to catch.

**Evidence.**  Arc: **238 passed / 0 failed / 0 skipped** (five new verdicts), and the arms are deliberately of
different kinds because one cannot do another's job:

    K window vs the INT8 rule on the RAW rows        3072 values, 0 differ - "K is not rotated"
    V window vs the Q4_0 rule on the ROTATED rows    3072 values, 0 differ
    the mixed-basis attention vs an oracle on the same two windows      worst rel 1.68e-06 (tight)
    the output is H applied ONCE to it (the de-rotation alone)          worst rel 1.19e-04 (tight)
    vs the fp32-TRUE attention from the unquantized rows                bounded by the V-side |d|/2

**The engine's own parity file does the same two-step thing** (a tight host attention over the SAME dequantized
window, then a bounded comparison against the true one) and the reason is worth keeping: the tight comparison cannot
see what the quantisation cost, and the bounded one cannot see a basis error inside its own slack.

**A live demonstration of the "oracle built from the thing under test" trap, stumbled into rather than staged.**  The
first version bound the q4 gather's UNUSED K scratch output to the INT8 K window.  Both scratch outputs are written
unconditionally, so the q4 reader overwrote K with q4-dequantised V - and the composed arm still read **6143 of 6144
as fine**, because the attention and the oracle were both reading the same corrupted window.  What caught it was the
arm that does not go through the attention at all: **the K window check, 3072 of 3072 differing**.  Two lessons, both
already in the skill and both re-earned here: an unused binding needs somewhere harmless (the unused POOL in the
append is the same problem), and an arm that compares a kernel against its own output cannot be the only arm.

**THE CROSS-IMPLEMENTATION ARM CAUGHT A RACE IN THE CASE, and that is the find of this increment.**  The q4 append's
unused K half was bound to the SAME pool as its V half, so two writers raced on the same bytes with different data
(the K half carried the raw V row).  The Arc ran the case green - the K half's write happened to be the one the V
window check expected - while llvmpipe wrote the raw row's codes and the arm read **3071 of 3072 differing**.  A single
implementation would have shipped this.  Fixing it took two goes and the second one is the lesson: the first fix sent
the unused half to a throwaway pool but left the ROTATED row in the K input binding, which broke all three
implementations at once - because the kernel reads `is_v ? VC_ : KC_`, so the V half reads VC_.  **Read the
half-to-binding mapping off the kernel; "it passed" while two writers shared a buffer is not evidence that the
wiring was right.**

**Falsified (wiring is the mode, so the injections are in the case):**  the de-rotation applied twice and the
de-rotation omitted.  Both were *first written wrong and fixed*, which is the more useful record: the "twice" version
dispatched the same input again (a no-op that passed, and would have been reported as "the case cannot catch this"),
and the "omit" version left a push constant unused so `-Werror` failed the build and the run printed the STALE
binary's numbers.  An injection must change behaviour and must compile, and a script that does not check either is
producing evidence about nothing.  With both fixed, each injection fails the de-rotation arm (0 of 6144, worst rel
1.4e+03) and the bounded arm (12 of 6144), while the window arms and the mixed-basis arm stay green - and the two
injections produce the SAME numbers, which is itself the check: H is an involution, so applying it twice returns the
input, which is exactly what omitting it returns.

**What this leaves.**  The KV storage modes now cover what the engine ships (f16, int8, q4_0, k8v4) at the level of
a decode step.  The prompt path (`kv_append_q4_batch_kernel` and its staging pool, the tensor-core score variants,
`kv_hybrid`'s own prompt pass), the ring/staging machinery of `kv_stream.cu` and the per-token call sites that
decide which mode a step uses are still open - see `RUN-ON-B70.md` for what of it is on the critical path to a
running engine and what is deferrable.



## THE SAMPLER, KERNEL 1: the greedy argmax - the first TOKEN this port emits

**What was ported.**  `sampler_greedy_kernel` (`src/kernels/cuda/sampler.cu`) with the `history_count` /
`apply_penalties` pair it calls.  This is the `--temp 0` path, and it is the first sampler kernel here; the general
top-k/top-p kernel and the Philox draw come next, on the same penalties.  With this, the port has a complete
deterministic generation path: logits in, a token out.

    out[t] = argmax over v of apply_penalties(logit[t][v], count(v))
    count(v) = occurrences of v in the TAIL of the token's history, min(plen, history_len) long
    ties     = the LOWEST index wins; 0 when no candidate is above -inf

**THE PENALTY RULES ARE WHERE A PARAPHRASE INVERTS THE MODEL.**  The repeat penalty MULTIPLIES for a non-positive
logit and DIVIDES for a positive one - "divide by the repeat penalty" is the natural reading and it inverts the
penalty on half the vocabulary.  The presence penalty is `float(count > 0)`, a boolean, NOT the count, while the
frequency penalty carries the count.  The window is the TAIL of the history, so a token occurring only in the head of
a longer history is not penalised.  Each of those is an arm, and each fixture is built so the WRONG rule lands on a
different token.

**THE TIE RULE IS THE SERIAL SCAN'S.**  The source walks `v` ascending with a strict `>` and merges with "larger
value, or on equality the SMALLER index"; this port keeps that total order in a barrier tree (subgroup ops are banned
here) with the source's `n_vocab` sentinel for a thread with no elements, and answers 0 for an all -inf/NaN row.

**ONE PORT DECISION, stated rather than hidden.**  The source builds a shared BITMAP of the history's ids so
membership is O(1) and only the hits pay a count scan (~31 KB of shared at this model's 248,320-token vocabulary).
This port scans the window per candidate instead - what that source did before the bitmap, with identical counts -
because the bitmap and the reduction tree do not both fit the port's 32 KB shared budget at that vocabulary, and a
fast wrong membership test is worse than a slow right one.  The bitmap is the optimisation to add.

**Evidence.**  Arc: **253 passed / 0 failed / 0 skipped** (eleven new verdicts).  Ten arms against a transcription of
the rule, and every arm states the token it is BUILT to produce, which is checked as well as the oracle: a plain
argmax with a NaN and a -inf present; a two-way tie; all -inf; all NaN; the repeat penalty on a positive logit (it
must lose); on a negative logit (it must lose); the frequency count (4 hits at 2.0 drops 9.0 to 1.0, so an
unpenalised 3.0 wins); the presence penalty as a boolean rather than the count (9-2 beats 8-2, but 9-6 would not);
the TAIL window (a head-only hit is unpenalised); and the real 248,320-token vocabulary with its strided scan.

**FALSIFIED, and the fourth injection is the finding:**  the repeat branch divided unconditionally -> the negative-
logit arm; the tie rule preferring the highest index -> the tie arm; the window read as the head -> the tail arm; and
the presence penalty as the count -> the presence arm.  **The first version of that last arm PASSED under the
injection**, because its fixture let the penalised token lose under both rules: the arm was decorative, and only a
falsification could say so.  Rebuilding it (600 also in the history, once) makes the correct rule keep 500 at 9-2
against 6 and the wrong one drop it to 3.  The `expect` field earned its place the same way: with every arm stating
its token, a fixture that no longer means what it claims fails immediately - and one of them did, on the first run
after I touched it (arm 6's history still hit the runner-up once, so its count changed nothing).

**What the sampler still needs.**  `sampler_kernel` (the general path: temperature, top-k, top-p, and the cumulative
walk with the Philox draw - the RNG and its reference implementation come with it), the split-warp and
coupled/draft-staging variants (speculative decoding), and the `sample_tokens` entry point that picks between them.

## THE SAMPLER, KERNEL 2: the general path - top-k, top-p, min-p, temperature, and the draw

**What was ported.**  `sampler_kernel` (`src/kernels/cuda/sampler.cu`) and the Philox it draws with, as a shared
`shaders/common/philox.glsl` (the split-warp and draft kernels will need the same generator).  The chain is the
source's, which is llama.cpp's:

    penalties (once, on the RAW logits, over the TAIL of the history)
      -> top_k   : k rounds of a block argmax over the not-yet-taken; the list is in SELECTION order
      -> top_p   : the shortest prefix whose probability reaches top_p, never shorter than min_keep
      -> min_p   : then the descending prefix within log(min_p) of the head
      -> temperature on the SURVIVORS only, softmax over them, and ONE Philox draw

**TWO THINGS A CAREFUL PORT WOULD "FIX", both arms.**  The penalties belong on the raw logits, ONCE, before the
filters - the source's own comment records that they used to be applied a second time after the temperature scaling.
And **`temperature == 0` is NOT greedy**: the source sets `inv_t = 1/temperature` when temperature > 0 and ZERO
otherwise, so zero scales every survivor to zero and the draw is UNIFORM over the shortlist.  Greedy is its own path
(`sampler_greedy.comp`).  "Fix" either one and every zero-temperature token changes.

**THE PHILOX CONSTANTS ARE THE ENGINE'S, NOT Random123's.**  Random123's philox4x32 uses M0 = 0xD2511F53 and
M1 = 0xCD9E8D57; this engine uses 0x9E3779B9 and 0xBB67AE85.  A swap yields a perfectly good generator that produces
different numbers - the kind of change that reads as "the model got a bit worse" rather than as a bug.

**THE STREAM IS PINNED THROUGH THE SHORTLIST, not by reading the RNG back.**  Sixty-four EQUAL survivors at
temperature 0 make `floor(u * 64)` directly observable in the token the kernel returns, so sixteen seeds pin six bits
of the stream each - about 96 bits, which is what catches a swapped constant, nine rounds instead of ten, or the seed
and counter swapped into the key.  That arm is EXACT, not approximate: equal probabilities make the softmax cancel
out of the cumulative walk, so no `exp` difference can reach it.

**THREE FIXTURES WERE STRENGTHENED WHILE FALSIFYING, because the planned injections would have passed invisibly:**
equal logits make `inv_t` irrelevant (the temperature arm needs ONE dominant survivor, or "temp 0 is uniform" and
"temp 0 is greedy" give the same spread); a membership check cannot see too FEW survivors (the min_keep arm needs its
three survivors CLOSE, or one survivor is trivially inside a set of three); and a head that is REACHABLE is not a head
that is ORDERED - a doubled penalty, the source's own recorded bug, passed 184/184 under the weak form, which is why
that arm is now `top_k = 1` (a one-token shortlist makes the draw irrelevant and the order exact) plus an arm for the
temperature being applied to the survivors rather than to what the cut sees.

**THE DEVICE QUESTION, recorded as a disagreement.**  `shaders/common/double_math.glsl` says Intel's own support
article reports Arc has no shaderFloat64; this box's ANV reports `fp64 1` in the gate's `--list`, and the faithful
kernel runs here.  The portable f32 sibling that every other double kernel in this port has is therefore still to
write, and the case skips (not passes) on a device without fp64.

**WHAT THE SAMPLER STILL NEEDS.**  The split-warp and coupled/draft-staging variants (speculative decoding), and
`sample_tokens`, which picks between the paths - plus that f32 sibling.

## THE PORT MAP: the decode path's remaining work, as a CHECKED LIST

**What it is.**  `PORT-MAP.tsv` classifies **every** `kernels::` symbol `src/core/` calls - the decode path - as
`kernel` (GPU work with a shader in this tree), `host` (the engine's own host side: a size, a capability check, a
table, a sync primitive) or `todo` (GPU work this port has NOT done).  `tools/make_port_map.py` writes it and
`tools/check_port_map.py` - run by `gates/run_gate.sh` on every gate - fails on an invented symbol, on a `kernel` row
naming a shader that is not built, and on any `src/core/` symbol the table does not mention.  Falsified three ways:
an added row for a name the engine does not contain, a dropped row for a symbol `src/core/` calls, and a `kernel` row
pointing at a shader that does not exist.

**WHAT IT SAYS, which is the useful part: 77 symbols - 17 kernel, 50 host, 10 todo.**  Half the surface a reader
might assume needs porting is the engine's own host side, and the decode path's real remaining GPU work is ten
kernels, not the ~250 KB of prefill and fused-MoE code:

    cvec_apply  embedding_gather  gather_rows  scatter_rows_f32  iq_dequant_f32
    native_q5_k_f32  moe_grouped_s2  moe_hit_add  moe_hit_select  moe_hit_grouped_s2

`penalty_rows` is the eleventh thing that looks like a hole and is not: this port applies the penalties INSIDE the two
samplers, which is why it is classified `host` with that reason rather than `todo`.  `native_mmvq` is covered by the
seven `iq*_mmvq` shaders, and the `moe_hit_grouped_s2` pair by the `s2expert_*` / `moe_combine_*` / `scalar_gate_*`
set.

**The BACKEND SEAM, recorded here because it decides the shape of the integration.**  The engine selects backends by
COMPILE-TIME macros - `STRATA_ENABLE_CUDA`, `STRATA_ENABLE_HIP`, `STRATA_ENABLE_SYCL`, and no `STRATA_ENABLE_VULKAN`
- and each GPU entry point is a thin wrapper in a header whose body calls the backend's implementation
(`kv_q4.hpp`: `fwht256_inplace_cuda(...) { fwht256_cuda(...); }`).  A Vulkan backend therefore plugs in the same way:
a `_vulkan` body per wrapper, selected by a new macro, dispatching through this port's device layer.

## NEXT KERNEL: `embedding_gather` - contract PINNED, shader not written

The first-token path's gather, and the smallest of the ten holes.  What the engine's own reference
(`src/kernels/elementwise_parity.cpp`, the parity test that drives it) fixes, so no guessing is needed:

    out[i] = (code_i + bias) * scales[row * row_groups + i / group] + (offsets ? offsets[...] : 0.0f)

* `code_i` is the `bits`-wide code UNPACKED from the row's packed bytes (`bits` and `bias` are parameters: S2/S4/S8
  embeddings share this kernel), `row_bytes` is per row, `row_groups` the scales' stride per row.
* **THE REFERENCE COMPARES BITWISE, AND AGAINST THE SEPARATE MULTIPLY-AND-ADD, NOT THE FUSED FORM.**  The test builds
  `want = code * scale` then `+ offset` and `memcmp`s it against the kernel's output, while the fused
  `std::fma(code, scale, offset)` is only COUNTED (`fma_diff`) rather than required.  A port that reaches for `fma`
  as the "better" instruction fails that comparison wherever the two differ - so the port must compute the product
  and the sum as two operations, and the case must carry the same fma-difference count as a reported datum.
* The engine's fixture shape is worth copying: three rows, each run twice (with and without offsets), guard values
  just outside the written range, plus a scale applied after the gather to show it uses the caller's stream - which in
  this port is the descriptor-offset path, already gated.
* `iq_embed_rows` (the IQ/BF16 table form, `include/strata/kernels/iq_kernels.hpp`) is a SECOND kernel on the same
  path and is easy to mistake for host-side bookkeeping; it is not - the map said `host` until the header was read.

## THE EMBEDDING GATHER LANDED - and `precise` is load-bearing

**What was ported.**  `embedding_gather.comp`, from `verify_kernels.cu`'s `embedding_gather_dev_kernel` and the host
reference in `elementwise_parity.cpp`: packed codes + per-group scales (+ optional offsets) -> a float row.

    per_byte = 8 / code_bits;  mask = (1 << code_bits) - 1
    code  = (codes[tok*row_codes + i/per_byte] >> ((i % per_byte) * code_bits)) & mask    <- LSB FIRST in the byte
    group = i / group_elems
    out[tok*n + i] = (float)(code + code_bias) * scale[group] + (offsets ? offset[group] : 0)

**THE FINDING, and it is a general one for this port: A DRIVER MAY FUSE A MULTIPLY AND AN ADD UNLESS THE RESULT IS
`precise`.**  The SPIR-V contains no fma op, but GLSL's default permits *contraction* - so the driver's own backend
may still fuse, and one fused op rounds ONCE where the engine's `__fmul_rn` + `__fadd_rn` round TWICE.  The engine
compares bitwise against the two-rounding form, so a fused result is wrong wherever the two disagree.  **Measured:
four of the six arms failed before `precise`, and the two that passed were the ones whose products happen to be
exact** - which is what identified the cause.  The shader now qualifies the product, the offset and the sum, and the
SPIR-V carries six `NoContraction` decorations.

**The fixture was built to make that catchable**, rather than hoping: the host side SEARCHES for (code, scale,
offset) triples where the fused and two-step forms differ (e.g. code -7 with scale 0.1 and offset 0.1 gives
-0.599999964 two-step against -0.600000024 fused) and the probe arm puts one of them at element 0, reporting the
count of differing values - the same datum the engine tracks as `fma_diff`.

**Evidence.**  Six arms, all bitwise against the engine's rule: S8 without offsets; S8 with offsets; S4 (two codes
per byte, low bits first); S2 (four per byte) with a non-zero code bias; the rounding probe; and the group boundary.
A multi-token dispatch uses the device token array and the grid's y dimension, as the CUDA grid does, and a guard
region after the last row must stay untouched.  Falsified five ways: contractable arithmetic, MSB-first codes, the
group index as the element-within-group, the bias dropped, and the token array ignored.

## RESUME HERE (state as of the last commit)

**THE QUANTIZED-EXPERT WAVE IS DONE: ALL SIX FORMATS AND THE GROUPED PAIR.** 66 kernels, 18 shared includes, one
generated table file (four IQ grids). The gate prints its own totals - `bash ports/vulkan/gates/run_gate.sh`,
which compiles every shader from source - and this line has gone stale three times in two days, so run it rather
than quote it. The last two boxes it ran on: a Radeon RX 7900 XTX host (160 / 0 / 0 on RADV and on radeon, 154 /
0 / 1 on llvmpipe), and, after that card was swapped for an **Arc Pro B70**, this one.

**RUN ON INTEL HARDWARE (Arc Pro B70 "BMG G31", Mesa 25.2.8 / ANV, Vulkan 1.4.318, subgroup 32), 2026-10-04 after
stages 3 and 4 plus the M8 matrix path landed: **180 passed / 0 failed / 0 skipped** - the Arc skips nothing now**,
with llvmpipe **172 / 0 / 2** and the radeon ICD (now the AMD iGPU, since the discrete card is gone) at 175 / 0 / 1
on a good run - **174 / 1 / 1 when the iGPU's intermittent requery case fires**.  The two implementations that
still skip do so on cooperative matrix, which their drivers do not advertise - correct, and a different thing from
the Arc's skip, which was the port's own criterion being written around the departed Radeon's M16 config.  llvmpipe's
other skip is stage 4's `stage: vram is not mappable`: it has one memory type, device-local and mappable at once,
so there is no unmappable VRAM type to check.  The verdicts added by stages 3, 4 and the matrix path are the only
difference from the 156/0/1 these ICDs read before them.  **Read the totals, not an
exit code, and say which:** `vk_gate` itself returns 0 on a skip-only run (skips are not failures in the binary);
it is `run_gate.sh`'s assertion that makes a skipped case fail the RUN, which is the port's rule and stays.

**Two things measured about the runner on this box, both worth knowing before quoting it:**

* **`run_gate.sh` used to STOP at the Intel skip and never reach the cross-implementation arm** - so since the Arc
  swap, two of the three available implementations were silently not being exercised on this box, which is the
  opposite of what that arm exists for.  The skip now records the failure in `rc` and the arm still runs; the final
  exit status is unchanged (non-zero).  Any earlier quotation of per-ICD totals came from runs made by hand.
* **The AMD iGPU's `budget: independent requery agrees` is INTERMITTENT, not deterministic: 1 failure in 3
  consecutive runs** (161/1/1 and 162/0/1 from the same binary on the same device).  That is consistent with the
  cause already documented - an integrated device's free figure is system RAM shared with the OS - but it means a
  single radeon run can read either way.  The budget family passes on the Arc every run.

**THE FIRST DEVICE-SPECIFIC DEFECT THIS PORT HAS FOUND (fixed 2026-10-04).** `quantize_q8_K` was off by one byte
on the Arc - the low byte of block 5's scale - and the cause was measured rather than argued: printing the
device's value as a hex float beside the candidates gave `-0x1.ea9c58p-105` where the source's `1.0f/(-127/mx)`
gives `-0x1.ea9c5ap-105`, i.e. the driver FOLDS the division into `mx/-127` and rounds that. The codes were never
wrong. The case now carries both forms as images (the treatment `quantize_q8_1` already had for its codes),
demands a byte-exact match to one, and prints which: **Arc -> folded (0 of 1752 differing), iGPU -> source (0 of
1752)**. Both forms are real, on hardware in one box - one image was never going to be enough.

WHAT IS DONE, in the expert tier: the per-format row kernels for IQ1_M, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up),
IQ4_NL and Q2_0 (down) - i.e. **every format the resident `coder-iq1_m` model's 48 layers use, so each layer now
has both halves of its expert path** - plus the two q8_1 quantisers and
`native_gu_iq2s.comp` / `native_down_iq4nl.comp`, the GROUPED pair the tier actually launches. See section (b)
and (c) below for what each case checks and the one interface decision the grouped kernels needed.

WHAT IS NOT DONE, stated by name so the next session does not have to re-derive it:
  * THE OTHER FORMATS THE ENGINE HAS, which this model does not use: the per-format row kernels for IQ2_XXS (type
    16), IQ2_XS (17), Q4_K (12), Q5_K (13) and the small family Q5_0 (6) / Q5_1 (7) / Q8_0 (8).  Those are what
    the OTHER packs on the ladder need - the engine's own comment for Q4_K/Q5_K says "Unsloth's UD-Q4_K_XL
    experts", so `coder-iq3_xxs` and the q4_k_xl pack are the models waiting on them.  Seven kernels, each one
    file plus a dot include, with seven host oracles and their value-by-value partners.
  * THE FOUR OTHER GROUPED INSTANTIATIONS (`native_gu_iq3xxs`, `_iq3s`, `_iq4xs`, `native_down_q2_0`) and the
    `_multi` GRP_NC variants - mechanical copies of the gated pair (see (b)).
  * `native_mmvq.cu`'s OWN DISPATCHER (`native_iq4_nl_mmvq`, `native_q4_0_mmvq`, ... case 20 etc.), which is a
    SECOND RULE for types the expert path already has - the port carries the expert-path rule (from
    `iq_kernels.cu`), because that is what the resident model runs.  Same quantity, two rules; do not merge them.
  * WAVES 2-6 of the kernels (KV, rope, the remaining GEMVs, attention, MoE, the prefill GEMM) and the engine
    integration, neither of which this wave touched.  **The kernels ran on Intel silicon for the first time on
    2026-10-04** (the Arc run in RESUME HERE above) - but nothing has run through the ENGINE on any device, and
    the port has no engine integration at all.  See STATUS.md's last section.

**THE REAL DEFECT WAS A WRONG-BUFFER READ, AND THE "MASKED WAVE" ROOT CAUSE BELOW IS WITHDRAWN.**  In
`shaders/common/s2_row_dot.glsl`, `dx` (the activation's own fp16 scale) was read with `f16_at(xb)`, and the
port's `f16_at` takes a byte offset into the EXPERT BLOB whereas the CUDA's takes a pointer and was reading the
ACTIVATION. So with `use_xscales` false - the GPU path, the one `moe_hit_parity` exercises - every row's
multiplier was arbitrary code-byte content: code-byte pairs that decode as a NaN or a near-65504 fp16 turn the
whole row into NaN multiplied through 80 chunks, which is exactly the "same non-canonical NaN in every workgroup"
signature that was read as a reduction defect. Fixed by `act_f16_at` (reads `act_b`); six new cases cover it.

**The A/B that withdraws the masked-wave claim, measured rather than argued:** with the `dx` fix in place and the
OLD subgroup reduction restored, all six tier cases pass on RADV (subgroup 64) AND on llvmpipe (subgroup 8), at
both 80 chunks < 256 lanes and 320 chunks > 256 lanes. There is no configuration in this port where the subgroup
form was measured to fail. The reduction IS a barrier tree in the shipped tree, but as a **portability choice**
(Intel picks the subgroup width per kernel) and **not** as a bug fix - the full note and the one-file revert
recipe are at the top of `shaders/common/wg_reduce.glsl`.

WHAT MADE THE OLD DIAGNOSIS SURVIVE TWO ROUNDS, worth carrying into any future bisection:
* `probe_gu_reads.comp` verified ITS OWN reads against the host and both agreed. It never exercised `f16_at`, so
  it could not see the shader's read being from the wrong buffer - a probe that re-derives the offsets is a
  second copy of the claim, not a check of it.
* the bisection that "exonerated the dot" showed the raw per-lane dots were FINITE (0.0645924). They were finite
  and wrong. **Finiteness is not correctness, and a run with no oracle comparison cannot tell the two apart.**
* initialising the reduction's shared array changed nothing - correctly, because the cause was never shared
  memory. That non-result was read as "the hypothesis is refuted, so the cause is the caller's mask".

NEXT, in order:
1. **The quantized-expert wave** (22 shaders from `iq_kernels.cu` / `native_mmvq.cu`) - STARTED 2026-10-04:
   `iq1m_mmvq.comp` (the IQ1_M row the resident `coder-iq1_m` model's experts are stored in) is in the build and
   gated, with `common/iq1m_dot.glsl`, the grid table generated from the engine by `tools/gen-iq1s-grid.py`
   (the gate now fails if it is stale), and three cases over sub-width and above-width part counts plus two
   columns. NEXT IN THIS WAVE, in this order:
   (a) the q8_1 quantizer pair - **DONE**: `quantize_q8_1.comp` (the IQ path's, clamped) and
       `swiglu_quantize_q8_1.comp` (the shared expert path's, fused silu+quantize) are both in the build and gated.
       TWO FINDINGS FROM IT, both about the ENGINE rather than the port:
       * **THE ENGINE HAS ONE ACTIVATION QUANTISER THAT CLAMPS AND ONE THAT DOES NOT.**  `q8_1_store`
         (iq_kernels.cu, used by `quantize_q8_1_kernel` and `swiglu_q8_1_entries_kernel`) clamps `d` and the sum
         to the largest finite half - #606 - while `native_swiglu_quantize_q8_1_kernel` (native_mmvq.cu) stores
         `make_half2(d, sum)` RAW, and IT IS THE ONE THE SHARED EXPERT PATH CALLS (src/kernels/cuda/shared_expert.cu,
         two sites).  So a shared-expert block whose scale or sum passes fp16 range stores inf where the IQ path
         stores 65504, and the dot that reads it then computes inf * 0.  The port carries both rules as the source
         has them, with the difference at the call site (`clamp_606`), and the case measures a block that hits it
         (products of 3000.0: both the scale and the sum past range).  **ASKED UPSTREAM on 2026-10-04:**
         [Niko1221/Strata#606 comment 5982766696](https://github.com/Niko1221/Strata/issues/606#issuecomment-5982766696)
         - the maintainer's own 0.1.39 note names "both q8_1 activation quantizers (iq_kernels.cu and
         native_mmvq.cu)"; this is the third, and it is in that same file.  FOLLOW-UP DUTY: when it is fixed,
         comment on THAT thread (not a new issue), report what changed empirically, and flip this port's
         `q8_1_store(..., clamp_606)` to true on both paths - the swiglu case's "products past fp16 range" arm
         then expects 65504 where it now expects inf.
       * **THE DIVISION IS THE DRIVER'S.**  See (a2) - measured, and still unresolved.
   (a2) **A DECISION WORTH TAKING DELIBERATELY: which division the quantiser uses.** The port writes the source's
       `roundf(xi / d)`. MEASURED on RADV: of 2560 codes, 2184 match a correctly rounded division and 120 match
       `roundf(x*(127/amax))` - i.e. the driver's `/` is not correctly rounded, and the engine's own CPU AVX512
       path multiplies by a reciprocal too. The case accepts EITHER form (and fails on "neither"), so the port is
       portable, but the codes it produces for a tie can differ between vendors. Writing the reciprocal form
       explicitly would make it deterministic everywhere and match the CPU path; it would also stop matching the
       CUDA source's expression. Unresolved on purpose - it needs a decision, not a default.
   (a3) The swiglu quantiser's silu is `__fdividef`/`__expf` in the source and cannot be reproduced off the CUDA
       hardware, so its case verifies byte-exactness on ordinary data and a **1-ulp envelope** on tie-critical
       data. Measured with every product on a tie: 640 of 2560 codes differ from the unperturbed oracle, 0 fall
       outside the envelope. On ordinary data, 0 bytes differ.
   (b) **the grouped `native_gu` / `native_down` - DONE for the two formats that cover the most layers**
       (2026-10-04; 5 arms; RADV and radeon green, llvmpipe green).  This is the shape the expert tier actually
       launches: a list of GROUPS (the experts this batch hit), each with a range of ENTRIES (which token hit it).
       * `shaders/native_gu_iq2s.comp` (`native_gu_kernel<TG>` with IQ2_S, the model's most common gate/up format
         at 20 layers) and `shaders/native_down_iq4nl.comp` (`native_down_kernel<TD>` with IQ4_NL, the down format
         of 39).  The other four instantiations are the same file with a different dot include and three
         constants, exactly as the six per-format row kernels are.
       * **THE ONE INTERFACE DECISION: `grp_ptr` IS AN ARRAY OF DEVICE POINTERS AND VULKAN HAS NONE.** The port
         takes one weights storage buffer plus a per-group BYTE OFFSET table (`grp_off[g]`), so `grp_ptr[g] + off`
         becomes `grp_off[g] + off` and nothing else changes.  The alternative, `VK_EXT_descriptor_indexing` with
         one descriptor per expert, buys nothing here (the buffer is the same size) and would spend a
         device-extension requirement on it - the trade this port has refused everywhere else (no `shaderInt64`,
         no subgroup ops in reductions, no vendor intrinsics).  Recorded at the top of `native_gu_iq2s.comp`.
       * **THE GROUP COUNT IS A DEVICE VALUE, so the grid cannot be sized to it**: the source strides
         (`for (g = blockIdx.y; g < ng; g += gridDim.y)`) and so does the port, which needed one harness change -
         `Ctx::dispatch` grew an optional `groups_y` (it was x-only, "the y/z dims are 1"), used by these two
         kernels and nothing else.  THE CASES ARM THE STRIDE THREE WAYS (grid.y = n_groups, below it, above it),
         plus an EMPTY GROUP, because a case that only launches grid.y == n_groups cannot tell whether the stride
         works at all.
       * **What the cases test is the GROUPING, not the dots**: their oracles call the already-gated per-format
         host dots, so a failure points at the group/entry walk, the per-group offset, the `r*gu_row` /
         `up_off + r*gu_row` row addressing, the entry-major output indexing, or - on the down side - the fact
         that the activation row is the ENTRY index (`hq`) while the destination row is `ent_dst[e]`, a TOKEN.
         That last swap is the one worth a case: both indices are small integers and either compiles.
       * **`swiglu_entries_kernel` NEEDS NO NEW SHADER.** The expert tier's non-fused silu is the port's existing
         `swiglu_f32` (`(x / (1 + exp(-x))) * up`) applied to a flat entries x n_ff buffer - the same formula, and
         its case is already gated against a double reference.  For the record the engine has THREE silu spellings
         in this area - `swiglu_kernel` (double), `native_swiglu_kernel` (`__fdividef` + `__expf`), and
         `swiglu_entries_kernel` (plain `/` + `__expf`) - and the port carries the double one (`swiglu_f64`) plus
         the float one (`swiglu_f32`), both from wave 1.
       * **WHAT REMAINS IN THIS WAVE, and it is mechanical**: the four other grouped instantiations
         (`native_gu_iq3xxs`, `_iq3s`, `_iq4xs`, `native_down_q2_0`) and the `_multi` variants
         (`native_gu_multi_kernel` / `native_down_multi_kernel`), which take GRP_NC entries per pass with the
         activations staged in shared memory - the same numbers as the non-multi kernels, so the port should gate
         them for EQUALITY against their non-multi siblings rather than against a new oracle.
   (c) **the remaining formats, one shader each - ALL SIX THAT THE RESIDENT MODEL USES ARE DONE** (2026-10-04;
       3 arms each; RADV and radeon green, llvmpipe green).  Every one of the model's 48 layers now has both
       halves of its expert path ported, format by format:
       * `iq4nl_mmvq.comp` + `common/iq4nl_dot.glsl` (type 20) - the DOWN format of 39 layers.  Geometry differs
         from the 256-value formats (`Fmt<20>` is { qk 32, ipb 2, step 2 }): part k is weight block k/2 AND
         activation block k/2, two 16-value parts per block, and no sub-scale, no sign handry, no integer division
         in the tail.
       * `q2_0_mmvq.comp` + `common/q2_0_dot.glsl` (type 42) - the down format of the other 9 layers, and the one
         that closes the down side.  Geometry: 64 values per block spanning two q8_1 blocks, so part k is weight
         block k/2 and activation block k.  Its `__byte_perm` chain is a byte-table lookup ({-1,0,1,2}); the
         gather order was DERIVED by working the chain through (`__byte_perm` uses three selector bits, so the
         duplicated table operand makes each code's second bit irrelevant) and the first version had it wrong.
       * `iq3s_mmvq.comp` + `common/iq3s_dot.glsl` (type 21, 10 layers) - 110-byte blocks, a separate `signs[32]`
         array rather than IQ3_XXS's aux-word packing, and a NINE-bit grid index whose high bit arrives at a
         different shift for the two words of a pair (`8 - l0` vs `7 - l0`).
       * `iq4xs_mmvq.comp` + `common/iq4xs_dot.glsl` (type 23, 1 layer) - 136-byte blocks, `Fmt<23>` step 4, the
         IQ4_NL nibble machinery with different addressing, and a scale split across `scales_l`/`scales_h`
         (`ls - 32`).
       * **EVERY IQ CASE NOW CARRIES TWO ORACLES** and compares them to each other before comparing either to the
         device (rel gap 0, printed every run): the transcribed dp4a form and a value-by-value form in canonical
         order.  They caught three defects in the cases themselves - a per-part/per-block granularity error, a
         wrong gather order, and a table value multiplied as +255 instead of -1 (0xFF is a signed byte, and only
         the value form notices, because the dp4a paths sign-extend for free) - and none of them was reported as a
         kernel failure.  See the skill's "Testing an assumption the kernel and its oracle share".
       * **NEXT: the grouped kernels in (b)** - `native_gu` / `native_down`, the shape the expert tier actually
         launches.  They need one interface decision first: the source passes `grp_ptr`, an array of DEVICE
         POINTERS to per-expert weight blobs, and Vulkan has no equivalent - the port's options are one storage
         buffer plus an offset table (portable, recommended) or descriptor indexing (a device extension).
       * **THE PER-BYTE HELPERS ARE SHARED, DELIBERATELY - AND ONLY THOSE.** `common/perbyte_sign.glsl` holds
         `__vcmpne4`/`__vsub4` because IQ2_S and IQ3_XXS write the IDENTICAL idiom (one rule used twice). The two
         q8_1 quantisers are the same quantity under two DIFFERENT rules and stay separate. The harness keeps its
         OWN copies: an oracle that shares a helper with the kernel cannot catch a wrong helper.
       * **THE ORACLE IS THE SUSPECT TOO.** IQ2_S's first run failed with every row off by percent at the right
         magnitude, and both bad offsets were in my double oracle - the shader was right. `get_int_b2(qs, iqs/2)`
         is FOUR bytes at `2*iqs`, not two bytes at `iqs`. Tiebreaker is the helper's definition, not its name
         (written up in the skill under "Names that lie"). IQ3_XXS then passed its first numeric run.
       * **THE NIBBLE PACKING IS DE-INTERLEAVED** (low nibble of byte j = value j, high = value j+16) - not the
         adjacent-pair packing Q4_0/Q4_K use, which is why IQ4_NL and IQ4_XS read the activation at `q8[l]` and
         `q8[l+4]`. The port took that order from the engine's CPU AVX-2 kernel (`src/kernels/cpu/iq_avx2.cpp`
         says `// values 0..15` / `// values 16..31`), NOT from the CUDA's `__byte_perm` chain, which states it
         only implicitly.
2. The cross-implementation arm's `budget: independent requery agrees` case used to report a FALSE FAIL while
   the resident local model held the card. **FIXED, with the measurement**: the flake was `budget 81920 bytes,
   usage 0 bytes` - 80 KiB of driver budget bookkeeping with usage unchanged - so the discrete-card comparison
   now carries 1 MiB / 0.01%-of-heap tolerance, printed on failure, while case (c) keeps the requirement that
   the figure moves with an allocation. Do not read a future failure here as automatically environmental.

### WITHDRAWN: the masked-wave diagnosis (kept because the reasoning is worth seeing fail)

**UPDATE (bisected, not reasoned):** the `gu` NaN is no longer a mystery. The raw dot is correct - 640 of 640 up-row
dots finite - and `wg_sum` returns the same non-canonical NaN in 640 of 640 gate rows. The dot is exonerated; the
shared reduction produces it, in the one configuration no shipped caller uses (`n_chunks` 80 with 256 threads). See
`shaders/pending/README.md`, section "SETTLED BY BISECTION". The two-step fix is now justified, not speculative -
and it is still two steps, because six passing kernels depend on that helper.

The branch is GREEN: the gate prints its own totals and they were 120 passed, 0 failed, 0 skipped on RADV
(112/0/1 on llvmpipe, the skip being the cooperative-matrix case), 40 kernels plus 5 shared includes. **[A SNAPSHOT
FROM WHEN THIS WAS WRITTEN, while the withdrawn diagnosis was still being argued - NOT the current total. The
current one is in RESUME HERE at the top of this file, or better, in your own run's last line.]**

THE ONE OPEN TASK, scoped to a single case re-add:

1. `shaders/pending/` holds four finished, compiling kernels from `s2_expert_grouped.cu` - `s2expert_gu`,
   `s2expert_swiglu`, `s2expert_down` and the `s2_row_dot` include they share - plus a README with what is
   established and what is not. They are outside `shaders/*.comp` so the build does not exercise them.
2. **THE BLOCKER IS A DEVICE NaN IN `gu`, AND AN EARLIER "PASSED 3840/3840" WAS RETRACTED.** The gu kernel
   produces NaN where the oracle reads 1417.59375 from the same bytes:

       gu: got[0] nan (isnan 1)   want[0] 1417.59375 (isnan 0)   abs_sum[0] 5514.59375

   The old comparison (`ratio > 1.0`, false for NaN) reported that as "0 differ, worst 0" and the case called it a
   pass - so the claim is withdrawn and the NaN-safe comparison is now in the gate at all six sites.

   FIRST STEP, and it is a probe rather than a re-read: dump the first row's eight code bytes, its scale and its
   activation block AS THE DEVICE READS THEM (a small debug output buffer). Reasoning about which offset is right
   has already failed twice in this port. Ruled out already, in the README: the data is finite, the offsets agree
   between shader and oracle, and the integer path cannot overflow.

   THEN the `down` stage, whose own oracle bug is recorded (it must read the QUANTIZER'S OUTPUT, not the gate/up
   activation image - the third instance in this port of an oracle right for the data it was written against and
   wrong for the data it was run on).
3. The gate-wide NaN question that blocked that work is SETTLED: it was 0/0 from `gemv_bound` on a zero row, not a
   kernel. The bound now has a 1e-30 floor and the NaN-safe comparison is in at all six sites.

FILING RULE LEARNED THE HARD WAY, worth keeping: an unverified shader in `shaders/` is one the build exercises
without evidence, so drafted kernels live in `shaders/pending/` with a README that says exactly why.

THEN the rest of that file (the pair kernels, the activation correction, and the CPU-order parity kernels), and
after that the port's remaining blocks are listed under "THE REST OF THE WAVE" below.

The gate **prints its own totals** (`bash ports/vulkan/gates/run_gate.sh`) — do not quote a number here, it went
stale twice in one day. Wave 1 (elementwise / conversions / norm / silu / gdn) and both GEMM paths and rope are
done and gated.

## Done since this file was last written

- `gemm_coopmat.comp` / `gemm_coopmat_m8.comp` — the cooperative-matrix GEMM (matrix units) at the TWO tile shapes
  real devices advertise: **M16 N16 K16** (RADV/WMMA, verified on a Radeon) and **M8 N16 K16** (Intel XMX, verified
  on the Arc 2026-10-04 - see the step-5 block at the top of this file).  The case selects by the device's own
  property list, prints which pipeline it ran, and SKIPS loudly where no config exists - and since the M8 file
  landed that is a device property (llvmpipe, and RADV here, which does not advertise the extension), not a port
  limitation.
- `gemm_fma.comp` — the plain-FMA GEMM with **no shape precondition**, so it is both the decode path (M=1 is
  structurally impossible for the CMA path) and the fallback where no CMA config exists (llvmpipe). The engine
  selects by **shape first**, device capability second.
- `rope_neox.comp` — NEOX partial RoPE, one thread per row, table passed in from the host.
- `kv_q8_append.comp` — the 8-bit KV append (the storage half of kv_q8). Verified **byte-exact over the whole
  destination image** against a transcription of the engine's own quantisation formula, so a wrong ROW is as
  visible as a wrong code: four cases (VRAM page resident; page absent, where a sentinel image must stay
  untouched; and the host copy in the identity layout, written unconditionally in both table states), covering the
  all-zero group (scale 0), all-negative, values on ± the group maximum, and a maximum past fp16 range.

## mrope: SETTLED, by reading the source rather than assuming

`mrope_pos(tab, pos, pair)` in `include/strata/kernels/mrope.hpp` is `tab ? tab[pos * 3 + pair % 3] : pos`:

- **null table (the default) means the identity** — the position the caller passes IS the rotary position. Text.
- set, `pos` is a **CELL index** into an int32 `[cells][3]` (t, h, w) table, and pair `i` takes sector `i % 3`.
  That is the vision path (qwen4exp's interleaved M-RoPE, `dimension_sections` 11/11/10/0).

A Vulkan descriptor cannot be null, so the kernel takes an explicit `mrope` flag instead.

**DEVIATION from this file's original interface table** (which specified push `{int rows; int head_dim; int
n_rot;}`): the push is `{rows, head_dim, n_rot, mrope}` = **16 bytes**. The alternative — a host-built identity
table — was rejected: it allocates and fills an N×3 array just to express "no table", and it routes the text path
through the vision path's indexing.

## The buffer registry: DECIDED, deliberately deferred

Keep loose buffers plus per-kernel push constants for now; do **not** build the registry yet. This file's own
prediction is the reason: the interface convention gets its first real test in the **GEMV wave** (weight layouts,
not elementwise), and a registry designed before that would encode elementwise assumptions as though they were
general. Its value is at ~44 kernels, and its caller is the engine integration (PORT-PLAN stage 4/6). Written
down so the deferral is a decision rather than an oversight.

## rope_neox: the gate case, and one tolerance decision worth carrying forward

The case follows this file's spec — host oracle is `build_rope_table`'s float64 loop **copied**, NEOX pairing from
`rope_neox_pair`, **bit-exact** tail sentinels, adversarial positions — with one change: the rotation bound is
2^-20 of the near-cancellation form, not the 2^-7 written here. Those constants were for kernels that *compute*
their angles on device (fast-math sin/cos differ in the last bits); this port **passes the table**, so the only
difference left is float-vs-double in one multiply-add. Measured worst err/tol ratio **0.11** — the tightened
bound still has ~9× headroom, while the loose one would have accepted a rotation wrong by ~100×.

Two traps hit while writing it, both worth not repeating:

- **`half` is a reserved word in GLSL.** `const int half = n_rot / 2;` fails to compile; it is `nhalf` here.
- **Bit-exactness is only for the parts the kernel COPIES.** The in-place case first compared every element
  exactly and reported 494/512 "failures" — the reference is computed in double, so the rotated elements *must*
  differ in the last bits. Split the comparison: exact for the copied tail, err/tol for the arithmetic. The
  shader was correct throughout.

## ONE FINDING TO CARRY INTO EVERY fp16-KERNEL PORT: `packHalf2x16` is NOT the engine's conversion

Measured on RADV while porting kv_q8. `packHalf2x16` **saturates to the largest finite half** (0x7BFF) where the
engine's `f16_from_f32` returns **infinity** (0x7C00), and its tie and subnormal rounding is not the engine's
nearest-even either. The effect is silent and small: a KV group whose maximum exceeds fp16 range gets a finite
scale instead of an infinite one, so its codes come out 127 instead of 0 — one group, in a branch normal data
never reaches. The gate caught it as exactly 73 differing bytes.

Use the port's own `f16_from_f32` (transcribed from `strata/kernels/f16_bits.hpp`, and held bit-exact by the
`f32_to_f16` case) in every kernel that converts. Reading a half BACK is a pure widening conversion with no
rounding to get wrong, so `unpackHalf2x16` is exact and stays.

## kv_q8 is COMPLETE (append, gather, round trip)

`kv_q8_append.comp` and `kv_q8_gather.comp` are both in and gated, and the pair is verified as a **round trip**
(floats -> append -> gather -> compare with the originals). Measured worst `|x'-x| / scale` = **0.53**, which is
what a correct round-to-nearest quantiser gives: below one code step, so the quantiser is optimal rather than
merely inside a loose bound. The round trip is also bit-exact against the host oracle, which only holds if the
scale the append WROTE is exactly the scale the gather READS.

The gather also settles a conversion question the append raised: a group whose scale lands in the fp16
**subnormal** range round-trips exactly, so `unpackHalf2x16` and the engine's `f32_from_f16` agree there - it is
only the PACKING direction where the builtin differs (see the packHalf2x16 section above).

The engine's fp16 converter now lives in ONE place, `shaders/common/f16_bits.glsl`, included by the three kernels
that convert (`#include` works in glslc, resolved relative to the including file). Three copies of a bit-exact
function were three chances to diverge, and `shaders/*.comp` deliberately does not glob into `common/`.

## quantize_act: the q8_0 trio is in, q8_K is next

`quantize_q8_0.comp` (ggml's bytes), `quantize_q8_0_scaled.comp` (this engine's CPU path), `dequant_q8_0.comp`.
Every comparison is `==` on the 34-byte block, because this is a reproduction of someone else's quantiser.

**TWO quantisers for ONE layout is the design, not duplication:** the first reproduces GGML's bytes (what the
pack holds, what `moe_hit_parity` checks), the second reproduces THIS ENGINE's CPU reference, because a hit and a
miss for the same expert must produce the same number. They round differently ON PURPOSE - and the gate proves
both rather than one: a synthetic block whose maximum is exactly 127 makes `d32` exactly 1.0, so eight of its
values sit on EXACT rounding ties, and the two rules disagree on **19 of 32 codes** there (half-even vs
half-away). A single oracle would have tested the wrong contract for one of them.

Two things worth carrying forward:

* **`precise` is the portable `__fmul_rn`.** GLSL's `precise` emits `OpDecorate NoContraction`, which is how a
  rounded multiply is pinned against being contracted into an FMA - the same thing the CUDA source does
  explicitly, and what `quantize_q8_K` needs for its `nearest_int` magic.
* **"One code step" is the wrong error bound when the stored scale is SUBNORMAL.** The codes round against the
  fp32 `d32` but the block stores fp16 `d16`, so the error is
  `0.5*|d32| + |q| * |d16 - d32|`. The second term is negligible while `d16` is normal and dominates once it is
  subnormal (the fp16 grid step there is fixed at ~6e-8, so a small `d32` is represented coarsely). Measured: the
  naive bound failed by 1.15x on exactly that block, and the two-term bound is hit at a ratio of 1.00 - tight, not
  loose.

## quantize_act is COMPLETE (five kernels, all byte-exact)

`quantize_q8_K.comp` and `dequant_q8_K.comp` finish the file: 292 bytes per 256 elements,
`{ f32 d ; int8 qs[256] ; int16 bsums[16] }`, the other half of `VEC_DOT_TYPE` and the format the numerically
SENSITIVE weights use. Byte-exact over all six blocks (1816/1816 bytes, guard region included) plus the 16 int16
sums per block.

Three traps that were transcribed from the source and are now pinned by the gate:

* `iscale = -127/max`, NOT -128. The -128 version sits in the source COMMENTED OUT with a note that IQ2_XXS needs
  it for an awkward AVX path; using it is a 0.79% scaling error, the same order as the quantisation step, so it
  produces an activation that looks fine and that ggml never sees.
* `max` is the SIGNED value at the largest magnitude and the comparison is STRICTLY greater, so a tie keeps the
  FIRST element. What that means in practice: `code(x) = round(-127 * x/mx)`, so the element HOLDING the maximum
  lands on -127 whatever its sign, and the opposite extreme lands on +127. A test written from the intuitive
  phrasing ("a positive max maps to -127") gets a negative maximum backwards - it did, and the byte-exact check
  overruled it. The rule is now its own four-element check.
* The rounding is `nearest_int`'s round-half-to-EVEN via the 12582912.0f magic, NOT `round`'s half-away. The tie
  block separates them on **159 of 256 codes** - the sharpest evidence yet that a rule was implemented rather
  than approximated.

The zero block is written in full (d, qs AND bsums). ggml's `continue` leaves bsums unwritten, which a dot product
cannot tell from zero but a byte comparison can; the source records the divergence deliberately and the gate holds
it. `min(127, v)` likewise has no lower counterpart - defensive, since `|iscale*x| <= 127` by construction.

### One comparison rule learned twice in this file

A kernel's f32 result can only be compared against a HIGHER-PRECISION oracle when the result is exactly
representable. A q8_0 dequant multiplies a 7-bit code by an 11-bit fp16 (18 bits - exact in f32, so a `double`
oracle agreed by luck). q8_K multiplies the same 7-bit code by a full 24-bit fp32 scale: up to 31 bits, NOT
representable, so the device's correctly rounded f32 product differs from the exact double one and 687 of 1536
values "failed". Round the oracle to f32 the way the kernel rounds before comparing, and confine bit-exactness to
where it is genuinely exact rather than merely convenient.

## `ple` is COMPLETE, and it is the port's first COMPOSITION test

Six kernels - `ple_gnorm`, `ple_gate`, `ple_bcast`, `ple_conv`, `add3`, `ple_history_advance` - plus a new shared
`shaders/common/wg_reduce.glsl` that the three reducers use (rms_norm_weighted, ple_gnorm, ple_gate), so the two
silent reduction defects have one home instead of three.

Every earlier case checked ONE kernel. Composition is where a convention shared between kernels stops being
checkable by either alone, and this one found three bugs - **all three in the test harness, none in a kernel**,
which is itself the finding: each kernel passed its own stage while the chain was wrong three times.

1. **The source's separate DESTINATIONS are not optional.** The CUDA calls are `gnorm(d_key -> d_key)`,
   `gnorm(hidden -> d_query)` and `gnorm(d_gated -> d_norm)`: two of the three write somewhere other than their
   input. The port's kernel is in place, so those become a copy then an in-place call. Skipping the copy for the
   query normalised `hidden` itself - which broke the gate and everything after it, while `gnorm` alone scored
   0.21 err/tol on its own stage. `ple_block`'s scratch comment says the same thing about its five hc_dim buffers
   ("two buffers of the same size look like an obvious saving and the only thing it saves is 40 KB"), and the test
   harness then reproduced that exact bug: reusing the convolution-weight slot for the key weights clobbered
   `w_conv` and showed up TWO STAGES LATER as 8.6e7 err/tol.
2. **A composition test must SNAPSHOT each stage as it passes.** Reading the buffers at the end of the chain
   compares whatever ran last into them: the `gated` buffer held the NORMALIZED values by then and the stage
   reported 0/10240 correct.
3. **A relative tolerance on a cancelling output measures CONDITIONING, not correctness.** The conv "failed" 16
   of 10240 elements at up to 6x the bound while the worst ABSOLUTE deviation anywhere in the array was 7.2e-07
   on inputs of order 1 - those 16 outputs had cancelled to ~5e-4. The bounds now carry an absolute floor
   relative to the input scale (a decade above the measurement), which still leaves a real misindexing four or
   five orders of magnitude clear. Print the worst absolute deviation next to the ratio, so this is measured
   rather than assumed.

Two smaller traps: **`out` is a GLSL reserved word** (an output qualifier) and cannot name a variable - the same
class as `half`; and `blah.comp.spv` is the wrong artifact name, because the gate strips `.comp` when it writes
the SPIR-V.

Verified: the key/query/normalized reductions sit at 0.19-0.21 of a 1e-6 relative bound, the gate at 0.06, and the
history advance is bit-exact over 92176 floats with the guard past the state untouched. The history advance is
in place and one thread owns a whole column - which is what makes the shift well-defined without a barrier.

## The GEMV wave: first two kernels in

`s2_gemv_q8.comp` (the S2 weight format over a Q8_0 activation - PLE's key projection) and `bf16_mmvf_f32.comp`
(the single-token BF16 matrix-vector product with an fp32 activation - PLE's value projection, and ssm_alpha/beta
plus the QSA indexer projections). Both are the source's own shape: one workgroup per output row, the source's
per-thread accumulator structure, then one workgroup reduction.

**THE FUSED MAC IS NOW A CHECKED PROPERTY, NOT A COMMENT.** `bf16_mmvf_f32` uses `fma()` to match the source's
`__fmaf_rn`, and the difference between a fused multiply-add and a separate multiply-then-add is *below the
reduction-order noise* - it cannot be caught numerically. So the gate's census rule now requires
`OpExtInst ... Fma` in that shader's SPIR-V, the same way it requires a subgroup reduction in the three reducers.
An FMA emitted as Mul+Add is more accurate, which is exactly why it is the wrong answer: it changes the last bits
of every term.

Two testing techniques from this wave worth keeping:

* **A layout probe makes a swap VISIBLE.** `bf16_mmvf_f32` reads each weight row as 32-bit pairs (element 2p low,
  2p+1 high). In a sum over random data, swapping the halves is a rounding-level error and invisible; one row is
  therefore built with its low halves at ~1e3 and its high halves at ~1e-3, which turns the same swap into a
  three-orders-of-magnitude error.
* **A ratio of exactly 0 is either exact agreement or a comparison of zeros.** `s2_gemv_q8` reported
  `worst err/tol 0` on all rows, and the numbers now printed beside it are what make that a result rather than a
  suspicion: `y[0] = -46391.4` with `max |y| = 46391.4`. The cases also FAIL a vacuous comparison (all expected
  values ~0) explicitly.

Also scored: distinct per-group scales and distinct per-block fp16 scales, so a wrong `q >> 4` or `(q*4)/32`
index is an O(1) error; negative S2 scales; a zero-scale activation block and a large one; every 2-bit code value
appearing; and the minimum legal geometry (n_in = 64, one S2 group; n_in = 2, a single weight pair).

Tolerances are ranked against the double oracle: the S2 dot products came out at ratio 0 (the dominant terms are
few and large), the BF16 products at 0.09 and 0.005 of a 1e-5 relative bound.

## `bf16_mmvf_f32_multi` is in - and the source's bit-identity claim holds

Up to 8 activation rows sharing one pass over the weight row (the prompt path; the weight row is the expensive
side). The source promises something checkable, so the gate checks exactly that: **"each output is bit-identical
to a bf16_f32_mmvf_kernel launch of its own"**. Every row of the multi-row result is compared BIT-FOR-BIT against
the single-row shader run on that row alone, and the worst difference is 0 across all three shapes - including a
shape with **padded `ldx` and `ldy`**, where a token row read at `k * n_in` instead of `k * ldx` would read 1e30
sentinel padding.

Two things the port changed about the source's shape, both deliberate:

* **The NT template parameter is gone.** `native_bf16.cu` instantiates the kernel at NT = 4 or 8 because its
  shared array is `partials[NT][32]` and NT must be a compile-time size. The port's shared reduction is sized by
  the SUBGROUP count, not by the number of tokens, so one shader covers 1..8 rows. The accumulator loop is still
  8 wide with the row count as a bound, so the per-token arithmetic is unchanged.
* **The shader CLAMPS `n_tok` to 8** rather than indexing a fixed array out of range: a shader cannot refuse to
  run, so the host's validation is documented as the host's.

And one real bug the source's own warning caught in the port's shared code: **`wg_sum` now has its leading
barrier**, because the multi-token kernel reduces once PER ROW, and without it a fast invocation can write the
next total before a slow one has read the previous - "invisible in most runs and a slightly different norm when it
fires", in ple.cu's words about the identical barrier in `block_sum`. The single-call kernels never reached it.
This is the shared include earning its place: one fix, five callers.

### THE BOUND FOR A DOT PRODUCT IS NOT RELATIVE TO ITS RESULT

A measured row of the multi-row MMVF exceeded `1e-5 * |y|` by 1.34x - while being provably correct, because the
same row matched the single-row shader bit for bit. The bound was wrong, not the kernel: an f32 accumulation of n
terms has an error bounded by `(log2(n)+1) * eps * sum|terms|`, and with cancellation `sum|terms|` is FAR larger
than `|result|`. That row: `sum|terms| ~ 6.4e5` against a result of 3.9e4, which predicts ~0.5 of absolute error
against the ~0.5 measured - the kernel sat exactly on the theoretical bound.

The three GEMV comparisons now use `rtol*|result| + 16*2^-24*sum|terms|`, with the oracle reporting both sums.
The previously failing row reads 0.0078 of the bound (170x headroom) instead of 1.34, and the bound is DERIVED
rather than fitted - a real bug moves the result by orders of magnitude more than rounding error can.

## `s_gemv_q8_split` is in - the S-family canonical decode, over both quantized activations

The heart of this is ONE decode for S2, S4 and S8 (`docs/pack-format.md`): `value = decode(code) * scale +
offset`, with the codebook choosing between the affine `code + bias` and ggml's non-linear `kvalues_iq4nl`. It now
lives in `shaders/common/sform_decode.glsl` rather than once per kernel, because a second decode of the canonical
form is a second thing to get wrong.

The kernel is one shader for BOTH quantized activation kinds - `block_q8_K` (292 bytes / 256 elements, an f32
scale) and `block_q8_0` (34 bytes / 32, an fp16 scale) - exactly as the source is one kernel templated on the
format. The only difference IS the loader; the weight decode, the octet loop, the codebook and the reduction are
identical.

**`Q8_0` IS STRUCTURAL, NOT AN OPTIMISATION CHOICE.** `ffn_down_shexp` is IQ4_NL/Q4_0/Q5_0/Q8_0 in every layer
with `n_in = 640`, and 640 is a multiple of 32 but not of 256 - so Q8_K is impossible for it rather than merely
unimplemented. The gate runs that shape, and the source's header is explicit that describing this as "Q8_K is not
implemented" was wrong for several rounds.

The forms tested are the ones the PACK contains, because the attribute vector is what the kernel takes as
arguments - and one of them is the reason to test attributes rather than arbitrary numbers:

* S2 / Q8_K / group 64 - the Q2_0 shapes, 31.64 GiB of the pack.
* S4 / Q8_K / bias -8 - Q4_0's attributes.
* S8 / Q8_0 / n_in 640 - `ffn_down_shexp`.
* IQ4_NL codebook / Q8_0 - the non-linear table, a separate decode path.
* **A form WITH AN OFFSET** - Q4_K's. This is the only case where the offset's POSITION is observable: the offset
  belongs to the weight, so the term is `(code*scale + offset) * x`. Writing the mathematically equal
  `code*scale*x + offset*x` performs two multiplications and an addition where the correct form performs one of
  each, so it ROUNDS DIFFERENTLY - and every no-offset type cannot tell. The source records that this was wrong
  until a Q4_K case existed.
* S2 / Q8_K / group 16 - a second group size, so the shift is not tested at one value only.

TWO SHAPE DIFFERENCES FROM THE SOURCE, both deliberate:

* The CUDA kernel is WARP per output row and ends in a 32-lane shuffle butterfly. The port cannot size anything
  from a device-wide subgroup default (the `rms_norm` lesson: the driver may compile a kernel at another width
  and the host would dispatch too few rows), so it is one WORKGROUP per row with the shared reduction. The
  per-thread accumulator structure and the term arithmetic are the source's; the lane-to-element mapping is not,
  which is why this comparison is against a double reference rather than bit-for-bit.
* The port has no `__constant__` memory to misuse. The source's own comment records that keeping `kvalues_iq4nl`
  in constant memory cost **2.12x** - constant memory is fast when the access is uniform, and a non-linear
  codebook means every lane reads a different index, the pattern it handles worst. The port's table is a literal
  in the shader, so the trap cannot arise.

## `s_gemv_split` is in - the fp16 activation, and the kernel `attn_output`/`shared_expert` use

Both activation kinds of the S-family GEMV now exist: `s_gemv_q8_split.comp` (Q8_K / Q8_0) and
`s_gemv_split.comp` (fp16), sharing `common/sform_decode.glsl`. Both are one workgroup per output row with the
source's per-thread structure: QE=4 for the fp16 kernel (four consecutive elements share one code word, one scale
and one offset, because every group size the format defines is a multiple of four), four accumulators combined as
`(acc0+acc1)+(acc2+acc3)`, then the workgroup reduction.

The fp16 activation is read as **32-bit pairs** (`unpackHalf2x16`), which is the source's own `__half2` view, and
the gate's activation values walk the conversion's paths deliberately: the smallest SUBNORMAL half, a zero, a
large value, negatives, small normals. Group 16 is included because it is the smallest group the format allows
and it is what makes the quad's "one group per quad" assumption tight.

ONE MORE CUDA-TO-GLSL TRAP, in the same family as `half` and `out`: **`float2` is not a GLSL type, it is
`vec2`.** The source's `__half22float2` returns a float2 and writing that name in a shader parses as an
undeclared identifier.

### STILL IN `s_gemv.cu` (the file is 40,810 bytes; both activation kinds are now covered) (the file is 40,810 bytes; this kernel is the quantized-activation one)

`s_gemv_kernel` (the naive fp16 reference, one thread per row) and `s_gemv_q8k_kernel` (the naive Q8_K one).
Both exist in the source as the REFERENCE the split kernels are checked against - "the naive one is the reference
the split one is checked against" - and the port has a stricter reference in its double oracle, so porting them
buys the same cross-check the source has rather than new coverage. Then the host wrappers including the `_async`
forms, and `s2_gemv_fast.cu` (7,297) with `s2_gemv_quads`/`s2_gemv_fast`, which the source itself defers to a
later phase ("the speed win is `__dp4a` ... and that belongs to Phase 3 once the numerics are settled").

## The MoE router - and a HARDWARE finding that shapes it

TWO ARITHMETIC VARIANTS, because the engine's router computes its exponentials AND its sum in DOUBLE and the
target hardware does not have double:

* `router_top10_f64.comp` - faithful, requires the device's `shaderFloat64`.
* `router_top10_f32.comp` - portables, float exp with **Kahan-compensated** sums, and NO `Float64` capability.

**INTEL ARC HAS NO shaderFloat64.** Intel's own support article 000089817: "Integrated GPUs included with 11th Gen
Intel processors and the upcoming Intel Arc discrete GPUs don't support shaderFloat64." That is the target
hardware, so the faithful variant cannot run there AT ALL - and this is the router, which by the source's own
measurement is 3.39 ms of a 289 ms token at 48 layers, **56% of the whole forward pass**. The port's response is
not to pretend the difference is negligible: the gate RUNS BOTH and measures the disagreement.

The result, MEASURED: over 120 rank selections - random logits at three sharpnesses, a row of exact ties, a
degenerate all `-inf` row, and near-ties at gaps of 1e-10, 1e-9, 1e-8, 3e-8, 1e-7, 1e-6 and 1e-5 - **the portable
variant selected the IDENTICAL experts as the host's double-precision reference, and so did the faithful one.**
The near-ties probe both ends of the band: ABOVE the float ULP of a probability the two variants keep the same
order, and BELOW it both round the two probabilities to the same float and both fall back to the index rule. The
band in between is where they could differ, and it is narrower than any realistic router logits produce.

THE FAITHFUL VARIANT NEEDED A DOUBLE `exp` BUILT BY HAND. glslang has no `exp(double)` - the port learned that
while porting `silu` - but it does have `roundEven(double)` and `ldexp(double, int)`, which with an 18-term series
and the argument reduced to |r| <= ln2/2 make one. Verified indirectly: the faithful variant's ids and weights
match the host reference's, and its weights match with 0.109 of a 1e-6 bound.

THE FP64 SPLIT IS NOW A STRUCTURAL GATE RULE, not a comment: `router_top10_f32` FAILS the gate if its SPIR-V
carries an `OpCapability Float64`, and `router_top10_f64` fails if it does not. The portable variant exists
precisely for devices without the capability, so its absence from the SPIR-V is the property that keeps it
runnable there.

Sharing: the semantics (softmax over ALL experts, stable argsort with ties to the ASCENDING index, the
`2**-14` clamp applied to the gathered weights) live in `common/router_select.glsl`, used by both variants - they
are semantics, not arithmetic. One behaviour is material and tested: a rank the selection does NOT write keeps
whatever the output buffer held, and the renormalisation divides those stale values too. That is the source's
behaviour, not an oversight, and the gate pre-fills the buffer to prove it.

## THE DOUBLE-ARITHMETIC FAMILY, and the port's policy for it

**A PORT-WIDE FINDING, not a per-kernel one.** More than one kernel in this engine computes in DOUBLE, and the
target hardware has no `shaderFloat64` (Intel support article 000089817). Found so far: the router's exponentials
and its sum, `moe_combine`'s accumulation, `swiglu`'s silu, and `scalar_gate`'s dot. Each one therefore needs a
PORTABLE SIBLING, and the port's policy is now explicit:

* the FAITHFUL variant is held to **bit-exactness** against the host's double reference - it claims to be the same
  arithmetic, so it is required to be;
* the PORTABLE variant is held to a **bound taken from measurement**, printed on every run, and the gate FAILS if
  its SPIR-V carries `OpCapability Float64` (it exists for devices that do not have it);
* where the portable variant's error is a CONDITIONING question (a sum of signed terms: `moe_combine`, the GEMVs)
  the bound is relative to the TERMS; where it is a product (silu) a relative bound is right.

`common/double_math.glsl` holds the one piece that has to be built by hand - a double `exp` from
`roundEven(double)`, `ldexp(double,int)` and an 18-term series with the argument reduced to |r| <= ln2/2 - because
glslang has none. The router and the SwiGLU both use it now.

## `moe_combine` and `swiglu`, in both arithmetic variants - and an unblocking

`moe_combine_f64`/`_f32` (the MoE block's final combination) and `swiglu_f64`/`_f32` (the shared expert's
activation). Measured on the real geometry (n_embd 2560, k 10):

* `moe_combine_f64` is **bit-exact** against the double reference, both with the shared expert added and without;
* `moe_combine_f32` differs on 934/2560 (worst 0.07 of the term-relative bound) and 1452/2560 (0.075);
* `swiglu_f64` is **bit-exact**, and it differs from a plain float silu on **665 of 2560** elements - so the
  double silu is not a formality, it is a different arithmetic;
* `swiglu_f32` differs from the double reference on 1040/2560 with a worst relative error of 9.01e-07.

**IT UNBLOCKS A SHADER THE PORT HAD SHELVED.** `shaders/blocked/silu_fp64.comp` was parked early on because
glslang has no `exp(double)`. It is expressible after all - `common/double_math.glsl` builds one - and the router
needed the identical piece. A blocked item turned out to be a missing helper rather than a missing capability.

Two semantics that are behaviours rather than numerics, both tested: the shared expert's output is added to the
routed result **PLAIN** (not router-weighted, not renormalised against the routed sum), and the SwiGLU rounds the
silu to f32 BEFORE multiplying by `up` - moving that cast changes the value.

THIRD RESERVED-WORD TRAP: **`shared` is a GLSL keyword** (the shared-memory qualifier), so a buffer cannot be
named `shared` any more than a variable can be named `half` or `out`.

## The shared expert is complete on the GPU side (`scalar_gate`, `scale_rows`)

`scalar_gate_f64`/`_f32` (the per-token gate: sigmoid of a bf16 dot in double) and `scale_rows` (the per-token
scale). With `moe_combine`, `swiglu` and these, every kernel `shared_expert.cu` needs is ported - what remains
there is the host orchestration, which belongs to integration.

`scale_kernel` needed no work: the port's wave-1 `scale.comp` does the same multiply, with the scalar arriving as
a push constant instead of a `g[0]` buffer read. Same value, same rounding.

THE SCALAR GATE'S FAILURE MODE IS QUIET and that is why it gets its own case: sigmoid bounds the output to [0,1],
so a gate that should be 0.5 and reads 1.0 scales the shared expert by 2x and produces finite, plausible logits.
The zero-dot row cannot hide anything - the answer there is exactly 0.5 - so it is its own check, and the
saturated row (a gate of exactly 1.0, or exactly 0.0 on the other side) is another.

### A SUBNORMAL BOUNDARY IS NOT A CORRECTNESS BOUNDARY, and the first run of this case failed on exactly that

With full-range inputs the dot is ~+-30, so the sigmoid returns values like **2.8e-45** - the smallest SUBNORMAL
floats - and the faithful variant produced 0 where the reference produced 2.8e-45. Both are "zero" in any
meaningful sense; the difference is that at the subnormal boundary the LAST BIT OF THE DOUBLE decides the float's
step. Scaling the activation so the gates land in the normal range made the faithful variant **bit-exact 4/4**,
which is the experiment that identifies the cause.

So the case's data is deliberately scaled out of that regime, and the note says why: requiring bit-exactness in a
regime where the answer is subnormal is requiring the double's last bit, not the arithmetic. The f32 variant's gap
over the same rows is 0.254 of the bound.

Two bugs of my own found here, both the same shape - a reference that was right for the data it was written
against and wrong for the data it was run on:

* the oracle was computed once for the random weight set and then reused for the all-zero and large-weight runs,
  which reported 0/4 and 2/4 for the two runs that should be the easiest to get right;
* and the subnormal regime above.

### THE REST OF THE WAVE - and its size, which is the number that matters

This is the biggest remaining area by a wide margin. Measured source sizes:

| source | bytes | what it is |
|---|---|---|
| `iq_kernels.cu` | 76,615 | the IQ-family quants and their dot products |
| `native_mmvq.cu` | 65,324 | the pinned CUDA quantized matvec (Q2_0/IQ3_XXS/IQ4_XS/Q8_0 paths) |
| `s2_expert_grouped.cu` | 64,436 | the MoE expert GEMV, grouped by expert |
| `s_gemv.cu` | 40,810 | the S-family (S2/S4/S8) GEMV and its sub-block variants |
| `s2_gemv_fast.cu` | 7,297 | the dp4a/fast S2 path (the source itself defers this) |
| `dequant_bf16.cu` | 12,762 | bf16 dequantisation |
| `native_bf16.cu` | 8,428 | **partly done** - the multi-row MMVF (1..8 tokens) is not ported |

~275 KB of CUDA in total, against ~40 KB ported so far this wave. **This is a scope decision, not a step**: the
next natural slice is the multi-row MMVF (small, and the prompt path needs it), then `s_gemv`'s S2/S4/S8 family
(the expert path), and `native_mmvq`/`iq_kernels` only if the port is meant to serve the quantized experts rather
than the S-family ones. Worth stating before starting rather than after.

Same shape as every case so far: read the CUDA source first, build the oracle from the engine's own function
(never from a description of it), sentinel every range the kernel must not touch, and give each branch of the
source an adversarial case. Tolerances come from measurement - print the err/tol ratio and keep it visible.

## THE NEXT WORKSTREAM: the quantized experts (decision taken 2026-10-04: port them)

The port is re-based onto upstream v0.1.39 and this branch no longer carries the fork's AVX1 floor (upstream
shipped it). The next wave is the quantized-expert path, chosen because the resident model is
`qwen3.8-flash-next-coder-iq1_m`.

**ITS EXPERTS ARE NOT IQ1_M, AND THE NAME IS WHY THIS WAS WRONG FOR THREE SESSIONS.**  "IQ1_M" is the GGUF the
model's DENSE weights come from; the experts live in the native pack beside it and the pack states their types
per layer.  From `Strata-data/packs/coder-iq1_m/native_experts.txt` (48 layers, 256 experts, verified
2026-10-04), counted over its `gu_type`/`d_type` columns:

    gate/up   IQ2_S (22)  20 layers | IQ3_XXS (18)  17 | IQ3_S (21)  10 | IQ4_XS (23)  1
    down      IQ4_NL (20) 39 layers | Q2_0 (42)      9

So the port's format order is now IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up) and IQ4_NL, Q2_0 (down) - NOT IQ1_M
first.  The IQ1_M dot already ported stays: it is the format of the OTHER packed models on the ladder, and of the
IQ1_M GGUF's own expert tensors where a model uses them.  A format census is worth re-running per model rather
than inferring from a filename; `native_experts.txt`'s first line names its own columns.

CORRECTED SIZES - an earlier note said "~141 KB"; the measured figures are:

    src/kernels/cuda/iq_kernels.cu     136,615 bytes   17 __global__ entry points
    src/kernels/cuda/native_mmvq.cu     71,856 bytes   11 __global__ entry points
    ---------------------------------------------------------------------
    total                              208,471 bytes   28 kernels
       for comparison, everything ported so far is ~128 KB and 40 shaders

Upstream GREW both files in v0.1.39 (+1095 lines iq_kernels.cu, +130 native_mmvq.cu), which is why the re-base
had to happen before this work rather than after.

The 28, in file order:

  iq_kernels.cu: mmvq, mmvq_multi, native_gu, native_gu_multi, swiglu_entries, native_down, native_down_multi,
                 quantize_q8_1, swiglu_q8_1_entries, dequant_flat, dequant_gu, embed_rows, native_gu_amd,
                 native_down_amd, native_gu_lds, native_down_lds, native_gu_fused
  native_mmvq.cu: native_quantize_q8_1, native_swiglu_quantize_q8_1, native_q5_k_mmvq, native_q2_0_mmvq,
                  native_q3_k_mmvq, native_iq4_xs_mmvq, native_q4_k_mmvq, native_q6_k_mmvq, native_small_mmvq,
                  native_mmvq_multi, native_mmvq_wave

NOT YET SCOPEU: how many of these are actually portable. A quick grep for HIP-isms (`__shfl`, `warpSize`, `LDS`)
labelled `native_gu_lds_kernel` "portable-looking", which is obviously wrong, so that grep was thrown away rather
than quoted. The `_amd`, `_lds` and `_fused` families look HIP-specific by name and probably have no Vulkan form;
the count of shaders to write is therefore somewhere between 20 and 28 and must come from reading the kernels, not
from a pattern match. DO THAT READ FIRST, before writing any of them.

## SCOPED (the read the note above asked for): 28 kernels -> 22 shaders

Read the six suspected ones. They are not HIP-specific by using exotic intrinsics; they are hardcoded to AMD
WAVEFRONT WIDTHS, which is a different and more decisive problem:

    native_gu_amd_kernel     lane = threadIdx.x masked to 63               wave64
    native_down_amd_kernel   lane = threadIdx.x masked to 63               wave64
    native_mmvq_wave_kernel  lane = threadIdx.x masked to 63               wave64
    native_gu_lds_kernel     lane/warp split at 32 + sgrid/LDS_NT/LDS_RB   wave32 + LDS staging
    native_down_lds_kernel   lane/warp split at 32 + sh_raw/LDS_RB         wave32 + LDS staging
    native_gu_fused_kernel   lane/warp split at 32 + LDS across two waves  wave32 + LDS staging

A kernel that hardcodes the lane mask cannot behave the same on an Intel GPU (subgroups of 8/16/32), so these are
SPECIALISATIONS the engine selects per architecture - not code the port needs to carry. The portable variants
(`native_gu_kernel`, `native_down_kernel`, `mmvq_kernel`, ...) are what this port targets; a Vulkan
specialisation is a later, separate decision.

So: 28 - 6 = 22 shaders to write, and some of those pair up (the `_multi` variants differ by an activation-row
count, exactly like bf16_mmvf_f32 / bf16_mmvf_f32_multi did here, which the port already folds into one kernel
with a parameter) - so expect ~18-20 files, not 22.

## THE MASKED-WAVE AUDIT: the defect is in the FAMILY, and one green case is UB

Root cause (see shaders/pending/README.md) is a wave reaching `subgroupAdd` with a partial execution mask, which is
what happens when the lane-distributed work count is SMALLER than the workgroup width. All 40 shaders declare 256,
so the test per caller is: can its lane loop's bound be below 256?

Every caller of the shared reduction, with the bound measured from the shader and the case that exercises it:

    caller                lane bound            safe while      verdict
    rms_norm              cols = n_embd 2560    always          SAFE
    scalar_gate_f32/f64   n = n_embd 2560       always          SAFE
    ple_gate              n = hc_dim 10240      always          SAFE
    ple_gnorm             cols = n_embd 2560    always          SAFE
    bf16_mmvf_f32         n_pairs = n_in / 2    n_in >= 512     CONDITIONAL
    bf16_mmvf_f32_multi   n_pairs = n_in / 2    n_in >= 512     CONDITIONAL
    s2_gemv_q8            n_quads = n_in / 4    n_in >= 1024    CONDITIONAL - AND ITS CASE IS ALREADY UB
    s_gemv_q8_split       unverified (QE loop)  unverified      UNVERIFIED
    s_gemv_split          unverified (QE loop)  unverified      UNVERIFIED
    s2expert_gu           H / 32 = 80           never           BROKEN (this is the NaN)

**The s2_gemv_q8 line is the one that matters.** Its case builds shapes with
`const int n_in = shape == 0 ? 2560 : 64`, so shape 1 gives n_quads = 16: lanes 16..255 do no work and the wave is
masked at the reduction - the same undefined behaviour as `gu`, in a case that currently PASSES with error/tol 0.
That pass is luck about which garbage sat in the masked lanes' registers, not evidence of correctness, and it can
flip with a recompile (it is the same code path that yields NaN in the expert tier).

So the fix belongs in the reduction or its contract, not only in the expert tier: any caller whose lane bound can
fall below the workgroup width is exposed, and two of them are already in the build.

NEXT: (1) fix the reduction shape once (a barrier-based reduce removes the class); (2) re-check the two UNVERIFIED
s_gemv lanes against the same test; (3) leave the s2_gemv_q8 shape-1 case in place but stop reading its pass as
evidence - or better, keep it as the regression that must still pass AFTER the fix.
