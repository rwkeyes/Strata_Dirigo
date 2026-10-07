# strata_dirigo — MOTHBALL REPORT

**Written 2026-10-07 (evening) on `vega`, Intel Arc Pro B70 / BMG G31. Branch `vulkan-arc-port`, HEAD `c123372`,
pushed to remote `dirigo`, working tree clean.** This is the document to read first. It says what the project is,
what state it is in, what is proven and what is not, what is left, and — because this project spent most of its
life measuring — which of its own earlier conclusions have since been withdrawn.

**Labels:** **[verified]** = measured and reproducible now; **[external]** = a claim from another project's own
artifacts; **[unverified]** = reasoned, not tested — do not act on one of these without testing it.

Companion documents, and which one to open for what:

| document | what it is |
|---|---|
| `PERFORMANCE-FINAL-SURVEY.md` (this directory, and `~/strata-dirigo-performance-survey.md`) | the performance survey: ranked levers, the ten closed negatives in full, and the commands to re-check every number |
| `/home/bob/strata-port-workqueue.md` | the session ledger — the day-by-day detail behind everything here |
| `NEXT.md`, `STATUS.md` | the port's own working docs (both now carry banners pointing at the survey) |
| `/home/bob/step4/z820b-hip/` | the raw HIP control-run artifacts, fetched off z820b |
| `harness/session/` | the 43 driver scripts that produced every number here, copied in-repo so they outlive the session |

---

## 1. THE DIRECT ANSWERS

1. **It works and it is stable.** [verified] 965 numeric cases pass on Intel (identical to baseline), the port is
   bit-exact against the engine's own references, and the 199-token arm reproduces its id (`56a0b28d2de6`) across
   every change made in this session.
2. **It is slow on the only model it has ever run, and the reason is architectural, not kernel-level.**
   [verified] The engine's **own SYCL backend** runs the *same pack, same card, same 198-token prompt* at
   **515 ms of device time against this port's 6,975 ms — 13.5x**. That is the only apples-to-apples comparison
   this project has, and it is the number to judge the port by.
3. **Its cold start is now ~90–110 s faster.** [verified] `STRATA_VK_DIRECT_UPLOAD` is the default as of
   `fa6f6da`: **188–208 s → 99 s**, ids unmoved, both rates inside their existing ranges.
4. **There is no valid small-model number for this port, and none can be produced from what is on the machine.**
   [verified] Every pack is 42–55 GB against 32 GB of VRAM, so the port has *never* run a GPU-resident model;
   its figures are streaming figures. Every llama.cpp figure this project quotes is on a *different, smaller*
   model that does fit (20.49 GiB). Ratios between them mean nothing — see §5.
5. **Nothing measured remains on the table.** [verified] The only open work is porting the engine's remaining
   prefill kernels (§8), which is what the 13.5x is made of.

---

## 2. WHAT THE PROJECT IS, AND WHY IT EXISTS

A **Vulkan port of the Strata inference engine** (`Strata_Dirigo`, a fork of `Niko1221/Strata`), built so the
engine runs on the Arc Pro B70 in `vega`. It is a port, not a fork-with-features: the engine's arithmetic is
transcribed into GLSL compute shaders and the CUDA runtime it calls is emulated over the Vulkan device layer.

**Why Vulkan, when the engine already has a faster SYCL backend on the same card:** *reach*. The project's own
priority order is wide compatibility first — Vulkan is present on machines where Level-Zero, ROCm and CUDA are
not, which is what a bootable appliance image has to assume. The port is therefore not trying to beat SYCL on
this hardware; it is the only way to run this engine on hardware SYCL does not reach. Judged as a speed
project on `vega`, it loses to its own SYCL backend and always will; judged as a reach project, that is the
point. Anyone resuming should decide which question they are answering before optimising anything.

---

## 3. STATE AT MOTHBALL

| fact | value | label |
|---|---|---|
| branch / HEAD | `vulkan-arc-port` at **`c123372`** | [verified] |
| remote | `dirigo/vulkan-arc-port` = `c123372` (matches local) | [verified] |
| working tree | clean | [verified] |
| gate, Intel (the line that matters) | **965 passed, 0 failed, 0 skipped** | [verified] |
| gate, llvmpipe | 949 / 0 / 4 | [verified] |
| gate, radeon iGPU | 951 / 3 / 2 — a **rotating, pre-existing** flake (recorded across runs as 948/6/2, 951/3/2, 952/2/2, 953/1/2), all in the `bf16_gemv` / `gdn_out_norm` / `qsa_decode_attn` wrapper-vs-shader set | [verified] |
| CUDA-runtime smoke | 63 passed / 0 failed, over 3 implementations | [verified] |
| port map (the gate's own line) | 169 decode-path symbols — 98 kernel, 47 host, **24 refused**; 152 shaders built, 114 claimed | [verified] |
| 199-token arm | prefill 6,988 ms wall / 6,975 ms device (~28 tok/s), decode 11.6 tok/s, id `56a0b28d2de6` | [verified] |
| cold start | 99 s (was 188–208 s before `fa6f6da`) | [verified] |
| model | Qwen3.8-Flash-Next IQ1_M, a 58.4 GB pack — **does not fit the 32 GB card**, so it streams | [verified] |

**Refresh any of it with:**

```bash
cd /home/bob/strata-vulkan-wt && bash ports/vulkan/gates/run_gate.sh        # NO arguments; never wrap in flock
cd /home/bob/step4 && bash build_product.sh                                 # rebuild the product
env STRATA_VK_BIN=/home/bob/vkbuild-vulkan/vulkan/strata_vulkan STRATA_PREFILL_TIMING=1 \
    timeout 900 flock /tmp/b70.lock -c "bash run.sh <name> \"$(printf '1 %.0s' $(seq 1 199))\""
```

---

## 4. WHAT THIS SESSION CHANGED (2026-10-07, twenty commits)

**Landed, each gate-verified:**

| change | outcome |
|---|---|
| `fa6f6da` `STRATA_VK_DIRECT_UPLOAD` default | **188–208 s → 99 s cold start**; mechanism proven by the dispatch accounting (cb-alloc 41→0, fence-create 34→0, submit 8,660→175, wait 27,427→9,586 ms); opt-out `=0`; gate at baseline |
| `9934785` `cudaEvent*` reports **device** time | real `vkCmdWriteTimestamp` into the live batch; the semantic fix (a mark is a batch boundary) verified by phases summing to the timeline ≤ wall |
| `7931b87` `CM_CT` output tiles per subgroup | **1.53x / 1.46x** on the matrix-unit GEMM path — and still **2.06x behind** the port's own FMA kernel, so no default moved |
| `625cdb0`, `a67cacb` prefill GEMM | KU-unrolled FMA is the default; **−24.1% prefill, +31.6% tok/s** |
| `fae1ba6`, `ba8f127`, `9df4b82`, `0fd180a`, `91a50de` | the direct-upload opt-in, `probe_gpuread`, the HIP launch counter, `probe_limits`, and the doc correction that the 4 GiB window cap is the **device's** limit, not the toolchain's |
| `f832950`, `ae9d561`, `98d78f6`, `c123372` | the survey, its update for the direct-upload default, banners on the two docs that carried retracted phase ratios, and the correction that the llama.cpp reference is a different model |

**Tried and reverted (a measured zero, kept as a patch):** the 8x idle-lane dequantiser fix — bit-exact, gate
965/0/0, and **zero effect** (508 vs 510 ms), because that phase is traffic-bound at 141 µs/dispatch. Patch at
`~/.backup/files/dequant_sbperwg/`.

---

## 5. PERFORMANCE POSITION

**[verified] The valid comparison, and the one to quote:**

| | device time, same pack, same card, same 198-token prompt |
|---|---|
| the engine's own SYCL backend | **515 ms** |
| this port | **6,975 ms** |
| **ratio** | **13.5x** |

**[external] The llama.cpp figures this project has been quoting are NOT a valid comparison.** The reference log
`/tmp/ref_llamacpp_vulkan.log` names what it ran: `qwen35moe 35B.A3B Q4_K_M`, 20.49 GiB, `ngl 99` — a model that
fits the 32 GB card outright with nothing streamed, giving 913.36 tok/s prefill / 36.52 tok/s decode. This port
runs a 58.4 GB pack that does *not* fit and streams experts through a 12,288-slot VRAM cache and the file tier.
Two different models, and the port's figures are streaming figures. The "~33x prefill" this project previously
carried was an artifact of that comparison and has been withdrawn.

**[external] The llama.cpp builds/branches axis is not a lever.** Four builds measured on z820a's 9B are ~2%
apart (retune fork 72.9/71.1, master 72.1/70.7, 0.5.0 72.0/70.6, build 11191 71.5/69.6), and the skill's own note
is that a ~25% spread seen on that box was *packaging*, not version. None of those runs involve strata.

**Ranked levers and the closed-negative list live in `PERFORMANCE-FINAL-SURVEY.md`** — §3 for the levers, §4 for
the negatives with the same numbers in fuller form.

---

## 6. CORRECTIONS AND RETRACTIONS (the change log)

A handoff that keeps a retracted claim is worse than no handoff, so these are stated, not buried:

1. **"The prefill is host-bound."** WITHDRAWN. [verified] Device time is 6,975 ms of a 6,988 ms wall — the host
   is 13 ms ahead. The claim came from the port's dispatch-layer counter, which buckets submit/wait and says
   nothing about where the time is. **The prefill is device-bound.**
2. **"`gdn recurrence` is 34.7% and the top lever" / the 262x and 13.2x phase ratios.** WITHDRAWN. [verified]
   Those ratios compared this port's old **host-clock** event timings against the SYCL tree's **device**
   timestamps. The port's `cudaEvent*` reported host time until `9934785`. Even with real device time, the
   per-phase split is **schedule-dependent**: adjacent phases exchange time between runs, so it names a
   neighbourhood, not a phase. Never pick a kernel to optimise from that table alone.
3. **"The port is ~33x off llama.cpp on the prefill."** WITHDRAWN — two different models (§5).
4. **"The all-zeros output means non-finite logits."** CORRECTED (earlier the same day, `d8f7bb5`): the measured
   cause is a **finite, constant zero logits row**. The `-inf`-to-0 sampler path described in older notes is real
   but was not the cause.
5. **"The `SharedSignalPool` leak and the `--spec 2` zero windows are upstream defects."** WITHDRAWN as upstream
   findings. [verified] The leak string exists only in ROCm's own `libhsa-runtime64.so` on that test box, and the
   `--spec 2` line is not in upstream main. Neither belongs to anybody in this project.

---

## 7. CLOSED NEGATIVES — DO NOT REPEAT THESE

Ten dead ends. Each is settled by a number, and this is the section most likely to save the next session a week:
the ideas below *look* like the obvious lever, one after another, and each is already disproven.

| # | the idea | verdict | the number that settles it |
|---|---|---|---|
| 1 | "the prefill is host-bound — speed up the submission path" | **DEAD** | device **6,975 ms of a 6,988 ms wall**; the host is 13 ms ahead (0.2%) |
| 2 | "the per-phase table names the top lever (`gdn recurrence`, 34.7%)" | **DEAD** | adjacent phases **exchange** time between runs (`gdn conv+gates` 2 → 1,540 ms); the table names a neighbourhood, not a phase |
| 3 | "port upstream's fused/MMQ expert path" | **DEAD as a port task** | the SYCL tree **stubs it too** — "the prompt path's MMQ plan is empty here, so `prefill.cpp` never takes the fused branch" |
| 4 | "use buffer device address / 64-bit indices to drop the 4 GiB window split" | **DEAD** | the port's window-split shape is **3.5x faster** than the single-dispatch shape BDA enables (0.4835 vs 1.6996 ms); the cap is the driver's `maxStorageBufferRange` |
| 5 | "the dequant family wastes 87.5% of its lanes — 8x the workgroups" | **DEAD** | implemented, bit-exact, gate 965/0/0 — and measured **zero** (508 vs 510 ms): traffic-bound at 141 µs/dispatch |
| 6 | "the decode's per-dispatch layer is hot" | **DEAD** | `submit` is **1 ms of a 3,322 ms** decode = 0.03% |
| 7 | "narrow the widest grids" | **DEAD** | the grid-width curve is linear with a *falling* marginal (3.625 µs + 0.7155 ns/workgroup); narrowing measured **+125.9 ms worse** |
| 8 | "kernel micro-optimisation will move the phase" | **MOSTLY DEAD** | a **2.46x/2.73x** faster step kernel moved its phase **5.0%**; tiling was 6.2x faster per dispatch and *slower* end to end |
| 9 | "cooperative matrix (XMX) is the lever" | **WASH** | `CM_CT` bought **1.53x/1.46x** in isolation and still loses to the K-unrolled FMA by **2.06x** on this part |
| 10 | "the 4 GiB window split's empty launches are the problem" | **NOT THE PROBLEM** | 2,016 empty launches/round ≈ **4.1–4.8%** of a decode round — the cheap residue of the *winning* shape |

The one-line lesson underneath rows 5, 8 and 9, which is the single most reused finding of this project: **a win
measured in isolation on this card repeatedly failed to transfer end to end.** Re-measure at the engine level
before proposing a default, and distrust any bench row offered as a reason to change one.

---

## 8. OPEN WORK

* **L1 — port the engine's remaining prefill kernels.** The 13.5x is made of this: the port issues 2.95x the
  dispatches and 2.7x the workgroups of the SYCL backend for the same prompt, while its *decode* kernels already
  run at the measured trivial-kernel floor (0.80–3.13 ns/workgroup). The 24 refused entry points in the gate's
  own port map are the worklist. Tedious, well-defined, low-risk, gate-verifiable.
* **L2 — cut the decode's dispatches and host round-trips.** [verified] The decode round is ~237 ms of which
  13.7 ms is GPU execution; ~3,550 dispatch executions per round at a flat ~3.86 µs each. The lever is fewer,
  larger submissions and fewer host boundaries per token — *not* narrower grids and *not* the dispatch layer,
  both measured dead (§7 rows 6 and 7). Medium-high risk: this is the port's correctness-sensitive seam.
* **Explicitly not a lever:** anything that showed up only in a single bench row (§7).

---

## 9. WHERE THE EVIDENCE LIVES

| artifact | path | note |
|---|---|---|
| the performance survey | `ports/vulkan/PERFORMANCE-FINAL-SURVEY.md` + `~/strata-dirigo-performance-survey.md` | the per-claim detail and re-check commands |
| **this report** | `ports/vulkan/MOTHBALL-REPORT.md` + `~/strata-dirigo-mothball-report.md` | the entry point |
| the session ledger | `/home/bob/strata-port-workqueue.md` | day-by-day detail, ~208 KB |
| **the session harnesses** | `ports/vulkan/harness/session/` (43 scripts, 192 K) | [verified] copied in-repo from `/home/bob/step4/`; scripts left in `~` do not survive a session. They expect `~/step4/` as cwd — see that dir's `README.md` |
| the HIP control-run evidence | `/home/bob/step4/z820b-hip/` (11 files, 364 K) | [verified] fetched from z820b, every file checked against the remote's own sha256; **originals left in place on z820b** (a local user is on that box) |
| the port's own working docs | `ports/vulkan/NEXT.md`, `STATUS.md`, `PERFORMANCE-B70-2026-10-06.md` | all three now banner-point to the survey and mark what is superseded |
| the port's tools | `ports/vulkan/tools/` | `probe_mem`, `probe_gpuread`, `probe_submit`, `probe_limits`, `hipcount` |
| procedural memory | skill `strata-llm-engine` | carries this session's portable traps (§10) |

---

## 10. HOW TO RESUME — AND THE TRAPS THAT COST TIME

**The traps, in the order you will hit them:**

* **`run_gate.sh` takes NO arguments.** It forwards `"$@"` to the harness and asserts the harness's own summary
  line; pass a label and you get the usage text plus "no summary line from the gate - refusing to report
  success". The run's identity is the gate's own output. **Never wrap it in `flock`** — it takes `/tmp/b70.lock`.
* **One GPU contender at a time.** Runs go through `flock /tmp/b70.lock`.
* **Compare DEVICE to DEVICE, and prefer counters over timers.** The port's pre-`9934785` records contain
  host-vs-device comparisons that must not be re-quoted.
* **Never read a Vulkan timestamp with `WAIT_BIT`** on this device — it hangs the doorbell thread and is a
  measured device-lost (r=-4). Availability bit only.
* **A timestamp mark must be a batch boundary** in this batching backend, or the phase deltas sum to more than
  the run's wall.
* **The ids are the guard.** A change that moves `56a0b28d2de6` on the 199-token arm needs a deliberate
  re-baseline decision, not a shrug.
* **An isolation win has repeatedly failed to transfer on this card** (§7).
* **Before blaming the engine or upstream, refute the alternatives by reading the artifact.** The one anomaly
  this session chased turned out to be the port's own batching semantics, after two upstream candidates were
  checked and eliminated (§6 item 5).
* **A secret-scan pattern that matches a substring hits common words.** `token` matched "tokens", `passwd`
  matched "passwordless" — 23 false positives before the real check (a search for actual `KEY=` assignments,
  which found none). Anchor the pattern and read every matched line before acting on either verdict.

---

## 11. THE HONEST BOTTOM LINE

The port works, is bit-exact, and is stable. On the only model it has ever run — one far too large for the card —
it is **13.5x slower than the engine's own SYCL backend on the same model, same card, same prompt**, and that gap
is made of prefill kernels the port has not transcribed yet, listed as 24 refusals in the gate's own port map.
Its decode is limited by dispatch and host round-trips rather than arithmetic. It has **no measured
GPU-resident-model number at all**, which is the comparison a reader most wants and the one thing this project
never produced, because no pack small enough to fit exists on the machine.

That is the state in which it is being mothballed: not misunderstood, and not overclaimed. Roughly twenty hours
of this session's measurement went into establishing what *doesn't* work, and that is recorded above and in the
survey as negatives with their numbers, so the next person starts where this one finished instead of repeating it.
