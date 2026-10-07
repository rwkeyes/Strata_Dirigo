# strata_dirigo — FINAL ANALYSIS: WHERE VULKAN'S ABILITY ENDED

**2026-10-07, `vega` (Intel Arc Pro B70 / BMG G31), branch `vulkan-arc-port` at `c123372`+.** Read with
`MOTHBALL-REPORT.md` (the state and the handoff) and `PERFORMANCE-FINAL-SURVEY.md` (the per-claim detail and
re-check commands); this document answers one question — *how far did the port get, and which wall was Vulkan's
rather than ours?* — and collects the performance numbers in one table.

**Labels:** **[verified]** = measured here and reproducible now; **[external]** = another project's or another
box's artifact; **[unverified]** = reasoned, not tested.

---

## 1. THE SHAPE OF THE ANSWER

"Vulkan's ability ended here" is only meaningful if the limits are separated, because three different kinds of
wall look identical from a benchmark table, and only one of them is the API's:

| kind of limit | can more work fix it? | how it must be reported |
|---|---|---|
| **the API's or the driver's ceiling** | no — the mechanism is absent or measured not to pay | a finding: this route is closed, and here is the number that closes it |
| **the port's unfinished work** | yes — it is transcription and scheduling | an open item, with effort and risk, not an excuse |
| **the measurement's own limits** | yes — with data that does not exist locally | *unknown*, and stated as unknown rather than as zero |

Most of the honest content of this analysis is in keeping those three apart. A report that calls unfinished work
"the API's limit" is not an analysis, and a report that calls an API ceiling "unfinished work" sends the next
session into a wall this one already hit.

---

## 2. THE PERFORMANCE NUMBERS, IN ONE PLACE

**The port, on the only model it has ever run** (Qwen3.8-Flash-Next IQ1_M, a 58.4 GB pack that does not fit the
32 GB card, so it streams experts through a 12,288-slot VRAM cache and the file tier):

| measurement | value | instrument | label |
|---|---|---|---|
| prefill, 199-token arm | **6,975 ms device / 6,988 ms wall** (~28 tok/s) | device timestamps (`9934785`) + wall clock | [verified] |
| decode | **11.6 tok/s** | server rate line | [verified] |
| output id (the guard) | `56a0b28d2de6` | the engine's own ids | [verified] |
| cold start | **188–208 s → 99 s** after `STRATA_VK_DIRECT_UPLOAD` became the default | wall clock + the dispatch accounting | [verified] |
| decode round | **~237 ms, of which 13.7 ms is GPU execution** — 94.2% is not compute | device timestamps vs wall | [verified] |
| per-dispatch cost | **~3.86 µs flat**, ~3,550 dispatch executions per round | counters | [verified] |
| the host's share of the prefill | **13 ms of 6,988 ms** (0.2%) — the prefill is DEVICE-bound | device timestamps | [verified] |
| gate | **965 passed / 0 failed / 0 skipped** (Intel; the line that matters), 949/0/4 llvmpipe, 951/3/2 radeon (a rotating, pre-existing flake), smoke 63 | the port's own gate | [verified] |

**Against the engine's own backend on the same hardware** — the only like-for-like comparison in this project:

| implementation | device time, same pack, same card, same prompt | label |
|---|---|---|
| the engine's **SYCL** backend (Level-Zero) | **515 ms** | [verified] |
| this **Vulkan** port | **6,975 ms** | [verified] |
| ratio | **13.5x** | [verified] |

The port issues **2.95x the dispatches and 2.7x the workgroups** of the SYCL backend for that same prompt
[verified]. Its *decode* kernels, by contrast, already run at the measured trivial-kernel floor (0.80–3.13
ns/workgroup) [verified] — so the decode is not slow because its kernels are slow.

**Numbers that exist but must NOT be put in that table** — every one of these compares different models, cards
or implementations, and quoting a ratio across them is how this project previously produced a "~33x" figure it
had to withdraw:

| figure | why it is not comparable | label |
|---|---|---|
| llama.cpp Vulkan: 913.36 tok/s prefill / 36.52 decode | a **different, smaller model** — `qwen35moe 35B.A3B Q4_K_M`, 20.49 GiB, `ngl 99`: it FITS the card with nothing streamed, while the port streams a 58.4 GB pack | [external] |
| the HIP control run on z820b: 74.85 / 24.07 | the *engine*, not the port, and a **different card** (RX 7900 XTX) | [external] |
| z820a's llama.cpp builds: 72.9 / 71.1 … 71.5 / 69.6 | a different box, a different model, and they compare **builds to each other** (~2% apart — the "~25% spread" once seen there was *packaging*, not version); none involves strata | [external] |

**The pattern that governs everything in this document** — a kernel win measured in isolation, and what it did
to the engine end to end [verified, all of it]:

| the isolated win | in isolation | end to end |
|---|---|---|
| `CM_CT` (independent accumulators per subgroup) | **1.53x / 1.46x** faster | still **2.06x behind** the port's own FMA kernel |
| the dequantiser's idle-lane fix | removes 87.5% idle lanes / 8x fewer workgroups | **508 vs 510 ms** — zero |
| tiled-K GEMM | **6.2x faster per dispatch** | *slower* end to end |
| the GDN step kernel | **2.46x / 2.73x** faster | moved its phase **5.0%** |
| KU-unrolled FMA (the one that landed) | per-call 1.6–5.6x | **−24.1% prefill, +31.6% tok/s** |

Four of five wins did not transfer. That is not bad luck; it is the signature of a backend whose costs are
**per-dispatch and per-boundary** rather than per-arithmetic-operation, which is exactly what the next section
is about.

---

## 3. THE HARD CEILINGS — WHERE VULKAN'S ABILITY ACTUALLY ENDED

Each of these is a mechanism that is *absent* or *measured not to pay*, with the number that closes it.

### 3.1 No vendor GEMM library — and exposing the matrix units is not enough

The engine's prompt path leans on **cuBLASLt** (7 `GemmEx` call sites) and, on newer parts, the matrix units.
Vulkan has no equivalent: the port hand-writes its GEMMs in GLSL. The hardware's matrix units *are* reachable
(cooperative matrix, `CM_CT`), and the port's XMX path was made **1.53x / 1.46x** faster than the single-accumulator
form [verified] — and it **still lost to the port's own K-unrolled FMA kernel by 2.06x** on this part, because an
8×16 tile with `TM=8` yields only 7.5 MACs per staged element at these shapes [verified].

**The finding:** the API exposes the hardware, and without a tuned library behind it, exposing it is not enough.
Closing that gap means writing a tuned GEMM, which is a library project, not a port task. **This one is the
clearest Vulkan ceiling in the project.**

### 3.2 The 4 GiB storage-buffer range — and Vulkan's own escape hatch does not pay

`maxStorageBufferRange` on this device is **4,294,967,295 bytes** [verified, `probe_limits`], so any tensor above
4 GiB must be split into windows with separate descriptors. The port's answer is a window-split shape, and
Vulkan's own answer is **buffer device address** (raw 64-bit pointers, no window). The port probes both:
`bufferDeviceAddress` is *supported* here (and `shaderInt64` is 1) [verified] — and the single-dispatch shape BDA
would enable measures **3.5x SLOWER** than the port's window split (gate/up 0.4835 vs 1.6996 ms; down 0.2858 vs
1.1851 ms) [verified].

**The finding:** the cap is the *driver's*, not an index-width quirk, and Vulkan's escape hatch exists and does
not pay on this hardware. Two independent routes — window-splitting and BDA — have been measured, and the
cheaper one is already the one in use.

### 3.3 No system-scope fence — a redesign, not a translation

The engine's `elementwise.cu` uses `__threadfence_system` for a **host↔device doorbell**: the host spins on a
device-written counter. Vulkan has no equivalent memory-scope operation for that direction, so this is not a
kernel that can be transcribed — it is a synchronisation redesign. The gate's port map counts **24 refused
entry points** [verified], and this class is among them.

### 3.4 No CUDA-graph semantics

The engine relies on graphs (~120 call sites: 41 `GraphLaunch`, 22 `BeginCapture`). The port replaced them with
one recorded command buffer replayed per token — a legitimate simplification, and one *Vulkan can express*. What
Vulkan does **not** give back is graph-level fusion of stream work: the port pays per-dispatch and per-boundary
costs that a graph would have collapsed (§4.2). This is a ceiling in the sense that *the mechanism the engine
was designed around does not exist*, not in the sense that Vulkan cannot run the workload.

### 3.5 Sustained-load instability on this generation [external]

Battlemage (Xe2) has a known driver defect: the xe KMD wedges under sustained compute load through Level-Zero,
OpenCL *and* Vulkan (`intel/compute-runtime#948`, open upstream). It is a driver problem a port cannot fix or
test around, and it is why the port's plan says to smoke-test the card first. Note the asymmetry with §3.1: this
is a ceiling that affects **all three** APIs on this part, which is a good illustration of the difference between
"Vulkan cannot do this" and "this GPU cannot be held here".

---

## 4. WHAT IS **NOT** VULKAN'S CEILING (AND SAYING SO PLAINLY)

### 4.1 The prefill kernels — the 13.5x is mostly this

The port issues 2.95x the dispatches and 2.7x the workgroups of the SYCL backend [verified]. That is
*transcription not yet done* — tedious, well-defined, low-risk and gate-verifiable, with the 24 refusals in the
gate's own port map as the worklist. It is the largest single lever in the project and it is **not** an API
limit. Reporting it as one would be the cop-out this section exists to prevent. [The one caveat: some of the 24
refusals are §3.3-class, i.e. genuinely untranslatable, so the worklist must be triaged rather than started.]

### 4.2 The decode's submission granularity — hard, not impossible

The decode round is **~237 ms of which 13.7 ms is GPU execution**: 94.2% of the round is not compute [verified].
The lever is **fewer, larger submissions and fewer host boundaries per token**. Vulkan can express that (one
command buffer per token, which is what the port already does at the graph level). What has been *measured dead*
is the two obvious ways to attack it: the per-dispatch layer is **1 ms of a 3,322 ms decode = 0.03%** [verified],
and narrowing the widest grids measured **+125.9 ms worse** [verified]. So the remaining route is a real
restructuring of how the engine's per-token work is batched — medium-high risk, on the port's
correctness-sensitive seam.

### 4.3 The measurement's own limits — unknown, not zero

* **No GPU-resident model exists here.** Every pack is 42–55 GB against 32 GB of VRAM, so the port has never
  run a fitting model and has **no small-model data point at all** [verified]. This is the number a reader most
  wants and the one thing this project never produced.
* **The instrument itself was wrong for most of the project's life.** The port's `cudaEvent*` reported the
  **host clock** until `9934785`, which produced the retracted 262x/225x and 13.2x phase ratios (they compared
  this port's host time against the SYCL tree's device time) and the retracted "the prefill is host-bound"
  claim [verified]. Every phase-level number from before that commit is void.
* **The phase table is schedule-dependent even now.** Adjacent phases exchange time between runs
  (`gdn conv+gates` 2 → 1,540 ms) [verified], so it names a neighbourhood and must never be used to pick a
  single kernel to optimise.

---

## 5. SO HOW DID WE GET HERE? THE SHAPE OF THE END

The project's path to Vulkan's ceiling was not a straight line of diminishing returns; it was a sequence of
*mechanisms proposed, measured, and eliminated* — and the eliminations are the value:

1. **Kernel-level optimisation was tried first and repeatedly failed to transfer** (§2, four of five wins). The
   reason is structural: the port's cost is dominated by how many dispatches and boundaries the engine's
   per-token work is split into, not by what each dispatch computes.
2. **That pushed the work to the submission layer** — where the two natural levers were measured dead (0.03% and
   +125.9 ms), leaving a genuine restructuring as the only route (§4.2).
3. **The hardware-specific routes were then probed directly** — matrix units (exposed, but 2.06x behind the
   port's own FMA without a tuned library: §3.1), BDA (supported, but 3.5x slower: §3.2), and the 4 GiB cap
   (the driver's, with the port already on the cheaper side of it). These are the measurements that *establish*
   the ceiling rather than assume it.
4. **What remained was the prefill transcription** (§4.1) — the one large, clean, well-defined lever, and the
   one the project ran out of time for rather than ran into a wall on.

The end of Vulkan's ability, stated precisely: **the port reached the point where every remaining structural
win requires either a tuned vendor-class GEMM library (§3.1), a mechanism the API does not provide (§3.3), or a
measurement that shows the API's own alternative is slower (§3.2) — while the largest remaining lever is not an
API limit at all but unported kernels (§4.1).** That is a *reach* project's ending, not a performance project's
failure: the port runs the engine bit-exactly on hardware where the engine's own backends cannot run at all,
which was the reason to build it.

---

## 6. WHAT WOULD HAVE TO CHANGE TO GO FURTHER

Honest and specific:

| to gain | what has to become true |
|---|---|
| the prefill's 13.5x | port the remaining kernels (§4.1) — effort, not breakthroughs; triage the 24 refusals |
| a faster decode | restructure per-token batching (§4.2) — medium-high risk on the correctness seam |
| the matrix units to pay | a tuned Vulkan GEMM (§3.1) — a library project |
| a GPU-resident comparison | a Strata pack that fits 32 GB — it does not exist on this machine (§4.3) |
| a same-API speed win | a driver in which BDA beats window-splitting (§3.2) — outside the project's control |

---

## 7. BOTTOM LINE

The port works, is bit-exact against the engine's references (965 numeric cases on Intel), and is stable. On the
only model it has ever run — 58.4 GB against a 32 GB card — it is **13.5x slower than the engine's own SYCL
backend on the same pack, same card and same prompt** [verified], and that gap is mostly unported prefill
kernels [§4.1]. Its decode is limited by dispatch and host boundaries, not arithmetic: 13.7 ms of execution in a
~237 ms round [§4.2]. It has **no GPU-resident-model number at all**, which is a gap in the evidence and is
recorded as one [§4.3].

Vulkan's ability ended where the API has no vendor GEMM to lean on, no system-scope fence for the engine's
doorbell, and an escape hatch from the 4 GiB window cap that measures 3.5x slower than living inside it — each
established by measurement, not assumed [§3]. Everything else that remains is work, and it is written down in
`MOTHBALL-REPORT.md` §8 with the numbers that make it worth doing.
