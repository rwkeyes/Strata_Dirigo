# strata_dirigo (Vulkan port of Strata) — FINAL PERFORMANCE SURVEY

**Written 2026-10-07 on `vega` (Intel Arc Pro B70, BMG G31), branch `vulkan-arc-port`, HEAD `fa6f6da`,
clean tree.** Purpose: one broad, comprehensive check of where the remaining performance is, taken immediately
before the project is mothballed. Written for whoever picks it up — including the possibility that nobody does.
(Updated once after the survey: L3 shipped as `fa6f6da`, and the numbers in sections 1 and 3 are its result.)

**Labels used throughout, and they matter:** **[verified]** = measured in this session with the instrument
named; **[measured earlier]** = in this repo's records (`NEXT.md`, `PERFORMANCE-B70-2026-10-06.md`, the
session ledger `/home/bob/strata-port-workqueue.md`), not re-run here; **[hypothesis]** = reasoned, not
tested — do not act on one of these without testing it first.

---

## 1. THE DIRECT ANSWERS

1. **Yes, there is headroom, and it is large: ~13.5x on the prefill.** [verified] Same card, same model,
   same 198-token prompt, device time on both sides: **this port 6,975 ms vs the engine's own SYCL backend
   515 ms**. That comparison is conservative, because this port is DEVICE-BOUND (wall 6,988 ms, so its wall
   is not hiding host work) while the SYCL figure is device time whose wall is *worse* than 515 ms.
2. **The headroom is NOT in the kernels' quality and NOT in the host path.** Both of the obvious answers are
   refuted below with numbers. It is in **how much work is issued, and in what shape** — the port's prefill
   does 2.7x more workgroups than SYCL for the same prompt, and its prefill kernels cost ~5x more per
   workgroup than the trivial-kernel floor its own *decode* kernels already reach.
3. **The decode's ceiling is dispatch/round-trip overhead, not arithmetic.** [measured earlier] The port's
   decode round is ~237 ms of which **13.7 ms is GPU execution** — the kernels run at 0.80–3.13 ns/workgroup,
   i.e. AT the measured trivial-kernel floor (0.7–3.8), and a **flat ~3.86 us is paid per dispatch across all
   50 families**. So ~94% of the decode round is not execution.
4. **The one ready improvement has been SHIPPED:** `STRATA_VK_DIRECT_UPLOAD` is now the DEFAULT (commit
   `fa6f6da`), opt-out with `=0`. [verified] 199-token arm, cold start: **188–208 s wall → 99 s**, ids unmoved
   (`56a0b28d2de6`), rates inside their existing ranges. The mechanism is visible in the port's own dispatch
   accounting: cb-alloc 41 → 0, fence-create 34 → 0, submit 8,660 → 175, wait 27,427 → 9,586 ms. Gate re-run
   with it as the default: **intel 965/0/0, identical to baseline** — the allocation-type assertions do not
   object, because the gate forces staging where it asserts types.
5. **Most of the work in this project is a closed negative.** Section 4 lists what is dead. It is the most
   valuable part of this document: it is ~20 hours of measurement that should not be repeated.

---

## 2. WHERE THE PORT STANDS (current numbers)

| metric | value | basis |
|---|---|---|
| prefill, 198 tokens | **6,988 ms wall / 6,975 ms device** → 27.9–29.9 tok/s | [verified] arm `dq_dev2`, bin `560ebbd9`, `STRATA_PREFILL_TIMING=1` |
| decode | **11.6 tok/s** | [verified] same arm |
| cold start | ~100–200 s wall for ~9.5 s of measured work | [measured earlier] |
| ids | `56a0b28d2de6` (199-token arm) — unmoved by every change below | [verified] |

**Yardsticks, and what each one licenses:**

| reference | number | gap | what it means |
|---|---|---|---|
| engine's own SYCL backend, same card/prompt/model, DEVICE time | 515 ms prefill | **13.5x** | the honest like-for-like target: same work, same hardware, different implementation |
| llama.cpp Vulkan, same card — **but a DIFFERENT, SMALLER MODEL**: Qwen3.5-35B-A3B Q4_K_M, 20.49 GiB, `ngl 99` (`pp512 913.36`, `tg128 36.52`) | 913 / 36.5 tok/s | **not comparable** | **THIS IS NOT A LIKE-FOR-LIKE GAP.** llama.cpp ran a 34.66 B MoE that FITS the 32 GB card outright — nothing streamed. This port runs the **58.4 GB** Qwen3.8-Flash-Next IQ1_M pack, which does not fit and therefore streams experts through a 12,288-slot VRAM cache and the file tier. The port's numbers are streaming numbers, dominated by the cache, the file tier and the upload path — not by kernel speed. Do not quote a ratio between them, and note that the "~33x" this row used to carry was an artifact of comparing two different models. |

**The port has NO model that fits this card.** Every pack available is 42–55 GB against 32 GB of VRAM
(`coder-iq1_m` 55 G native shard, `iq3_s` 49 G, `iq3_xxs` 42 G, `swift-iq3_xxs` 42 G), so there is **no measured
small-model, GPU-resident data point for this port at all** — which is the one comparison a reader is most
likely to want, and it cannot be produced without first building a pack for a smaller model.

The prefill is the whole story: the decode's cost is dispatch and host round-trips rather than arithmetic (§1.3),
while the prefill is **13.5x** off the engine's own backend measured on the same model, the same card and the
same prompt.

---

## 3. WHAT IS ACTUALLY LEFT (ranked, with evidence)

### L1 — Port the remaining engine prefill kernels faithfully. The one real lever. [hypothesis on mechanism, verified on magnitude]

- **Magnitude [verified]:** 6,975 ms device against SYCL's 515 ms device.
- **Decomposition [verified, counters only]:** the port issues **59,640 prefill dispatches / 129,098,884
  workgroups** against SYCL's **20,209 / ~47M** — 2.95x the dispatches, 2.7x the workgroups. Since the two
  backends' per-workgroup cost is the same order (57–60 ns, [measured earlier]) and the port's *decode*
  kernels sit at the trivial-kernel floor, the prefill's kernels must cost **~5x more per workgroup** —
  2.7 x 5 = the observed 13.5x.
- **Why that is expected:** the port's docs name their own remaining list — the batched/fused families the
  engine's CUDA has and this port still refuses (`refusals_prefill_vk.cpp`, `refusals_engine_vk.cpp`). The
  SYCL tree runs the *same* non-fused route (its own `moe_fused_stub.cpp` says the MMQ plan is empty there
  too), so the difference is **kernel quality**, not fusing: their dpct'd kernels are unrolled and packed
  (`sycl::vec<half,1>` loads; `for (l0 = 0; l0 < 8; l0 += 2)`), this port's are straightforward loops.
- **What to do:** work the port's own refusal list, transcribing the engine's kernels as faithfully as the
  decode side already does. Verify each with the port's numeric gate, and require the *phase* to move, not
  just the kernel — see the trap in section 4.
- **Effort/risk:** large (the biggest single body of work left). Risk is low: every kernel has a gate case
  and the ids must hold.

### L2 — Reduce the decode's dispatches and host round-trips. [measured earlier]

- 3,550 recorded dispatch executions per round carrying 10.3M workgroups at ~3.86 us of GPU cost each ≈ 13.7 ms
  of a ~237 ms round. The rest is dispatch latency and host handoff, not arithmetic.
- The port's own recorded/replay machinery (`record_begin`/`replay_recorded`) exists to submit a decode step
  as ONE command buffer; the remaining cost is the host-visible boundaries between segments
  (`drain_inflight`, the doorbell handshake).
- **Do not** chase this by narrowing grids or shrinking the dispatch layer: both are measured dead (section 4).
  The lever is *fewer, larger submissions* and *fewer host boundaries per token*.
- **Effort/risk:** medium-high; this is the port's correctness-sensitive seam.

### L3 — SHIPPED (`fa6f6da`): `STRATA_VK_DIRECT_UPLOAD` is the DEFAULT, opt-out with `=0`. [verified]

- 199-token arm cold start: **188–208 s wall → 99 s**; ids unmoved (`56a0b28d2de6`); prefill 7,136 ms / 27.75
  tok/s and decode 11.63 tok/s, both inside their existing ranges — it buys STARTUP only.
- Mechanism, from the port's own dispatch accounting (not just the clock): cb-alloc 41 → 0, fence-create
  34 → 0, submit 8,660 → 175, wait 27,427 → 9,586 ms.
- Gate with it as the default: **intel 965/0/0 identical to baseline**, lvp 949/0/4, radeon 951/3/2 (the
  rotating pre-existing flake), smoke 63 passed.
- It is map-on-demand, NOT persistent mapping: a persistent mapping trips the port's host-visible flush rule
  and turned 563 batches into 45,635 (a measured +47% prefill regression) before it was fixed.
- **CORRECTION worth keeping:** the `xfer stat` upload COUNT does NOT fall — it reads 13,702 in both modes,
  because its counter sits above the path branch. Use the dispatch accounting above to tell the paths apart;
  the older A/B's "0 uploads" came from a binary where the counter sat elsewhere.
- **Undo:** `STRATA_VK_DIRECT_UPLOAD=0` restores the staged path, and the gate keeps that path exercised
  through `set_force_staging` — which is what makes this a default rather than a one-way door.

### L4 — Anything at all in the per-phase table. [verified as unreliable]

The phase split is NOT load-bearing: across two device-time runs the totals agree but adjacent phases
exchange time (`gdn conv+gates` 2 → 1540 ms, `gdn out proj` 293 → 6 ms). The engine charges a gap to "the
phase that was waiting", and which host-labelled interval a piece of device work lands in is a scheduling
accident. **Never pick a kernel to optimise from this table alone.** Use it for totals, and use dispatch
counts and grid geometry (which are counters, not timers) for structure.

---

## 4. CLOSED NEGATIVES — DO NOT REPEAT THESE

Each of these cost real time and is settled. Listed with the number that settles it.

| # | idea | verdict | evidence |
|---|---|---|---|
| 1 | "the prefill is host-bound; speed up the submission path" | **DEAD** | [verified] device 6,975 ms of a 6,988 ms wall — the host is 13 ms ahead (0.2%). The reading came from the dispatch-layer counter, which buckets `submit`/`wait` and says nothing about where the time is. |
| 2 | "the per-phase table names the top lever (`gdn recurrence` 34.7%)" | **DEAD** | [verified] adjacent phases exchange time between runs; the table names a neighbourhood, not a phase. |
| 3 | "port upstream's fused/MMQ expert path" | **DEAD as a port task** | [verified] the SYCL tree stubs it too — "the prompt path's MMQ plan is empty here, so prefill.cpp never takes the fused branch". There is no upstream fused path to copy on this backend's terms. |
| 4 | "use buffer device address / 64-bit indices to remove the 4 GiB window split" | **DEAD** | [measured earlier] the port's window-split shape is **3.5x FASTER** than the single-dispatch shape BDA would enable (gate/up 0.4835 vs 1.6996 ms; down 0.2858 vs 1.1851). The 4 GiB cap is a DRIVER limit (`maxStorageBufferRange` 4,294,967,295), not an index-width quirk. |
| 5 | "the dequant family wastes 87.5% of its lanes (32 of 256) — 8x the workgroups" | **DEAD** | [verified] implemented, bit-exact, gate 965/0/0 — and it measured **zero**: phase 508 vs 510 ms. That phase is traffic/arithmetic-bound at 141 us/dispatch, ~37x its own dispatch cost. Reverted; patch kept at `~/.backup/files/dequant_sbperwg/`. |
| 6 | "the decode's per-dispatch layer is hot" | **DEAD** | [measured earlier] `submit` is 1 ms of a 3,322 ms decode = 0.03%. The cost is grid width and boundaries. |
| 7 | "narrow the widest grids" | **DEAD** | [measured earlier] the grid-width curve is linear with a FALLING marginal (3.625 us + 0.7155 ns/workgroup); narrowing the widest grids measured **+125.9 ms worse**. |
| 8 | "kernel micro-optimisation will move the phase" | **MOSTLY DEAD** | [measured earlier] a 2.46x/2.73x faster step kernel moved its phase **5.0%**. Same for the coopmat/K-unroll/tiling family: wins in isolation, washes end to end (tiling was 6.2x faster per dispatch and *slower* end to end). |
| 9 | "cooperative matrix (XMX) is the lever" | **WASH** | [verified] `CM_CT` (independent accumulators per subgroup) bought 1.53x/1.46x in isolation and **still loses to the K-unrolled FMA by 2.06x** on BMG at these tile sizes; `TM=8` gives only 7.5 MACs/staged element. Landed opt-in, no default moved. |
| 10 | "the 4 GiB window split's empty launches are the problem" | **NOT THE PROBLEM** | [measured earlier] 2,016 empty launches/round ≈ 4.1–4.8% of a decode round — the cheap residue of the *winning* shape. |

---

## 5. THE PORT'S OWN BACKLOG, RE-PRICED

`ports/vulkan/NEXT.md` and `STATUS.md` are current as of this session for the items above. Anything in them
claiming a *phase-level* opportunity should be read with section 3/L4 in mind, and anything claiming the
host path or the dispatch layer as the prefill's cost is superseded by section 4 items 1 and 6.

**Still open and genuinely unexplored:** the port's refusal list (L1). That is the only large body of work
this survey leaves on the table, and it is also the only one with a 13.5x-sized prize attached.

---

## 6. HOW TO RE-CHECK ANY CLAIM HERE

The instruments that produced these numbers, in the order you will need them:

```bash
# 1. the gate — the ONLY authority on whether a change is safe (never wrap it in flock)
cd /home/bob/strata-vulkan-wt && bash ports/vulkan/gates/run_gate.sh
#    expect: intel 965/0/0 (the line that matters), lvp 949/0/4, radeon ~948-953 with 1-6 rotating
#    failures in the bf16_gemv/gdn_out_norm/qsa_decode_attn wrapper-vs-shader set, smoke 63 passed.

# 2. the arm that produces the numbers in section 2 (device time now, since 9934785)
cd /home/bob/step4 && bash build_product.sh
env STRATA_VK_BIN=/home/bob/vkbuild-vulkan/vulkan/strata_vulkan STRATA_PREFILL_TIMING=1 \
    timeout 900 flock /tmp/b70.lock -c "bash run.sh <name> \"$(printf '1 %.0s' $(seq 1 199))\""

# 3. the phase line — TOTALS ONLY (see L4); `phases sum` should equal `GPU timeline` and be <= `wall`
grep -m1 'strata prefill timing' /home/bob/step4/logs/<name>.log

# 4. grid geometry and dispatch counts (counters — reliable)
#    STRATA_VK_GRID_STAT=1 for workgroups/dispatch; the arm logs carry `vk disp stat`.
```

**Instrumentation facts worth knowing before you trust a number:**
- `cudaEvent*` reports REAL device time as of `9934785` (`vkCmdWriteTimestamp` into the live batch,
  availability-bit reads, never `WAIT_BIT` — a waiting read is a measured DEVICE LOST here, r=-4).
- A mark is a BATCH BOUNDARY: without that, a mark's predecessor is not the previous mark and the phase
  deltas can sum to MORE than the run (measured: 8,043 ms of phases against a 6,976 ms wall).
- The decode's own kernel timer (`g_kt`) uses a separate pool and is genuine device time.
- Anything comparing this port to another backend must compare DEVICE to DEVICE; the port's host-clock era
  is over but its records still contain host-vs-device comparisons that should not be re-quoted.

---

## 7. BOTTOM LINE FOR A MOTHBALL

The port works, is bit-exact against the engine's references (965 numeric cases), and is stable. It runs the
same model as the engine's SYCL backend **13.5x slower on the prefill**, and its decode is limited by dispatch
and host round-trip overhead — 13.7 ms of GPU execution inside a ~237 ms round — rather than by arithmetic.
There is no valid small-model, GPU-resident number for this port at all (see section 2). The remaining
performance is in porting the engine's own prefill kernels — the tedious, well-defined
work the decode side already finished — and nowhere else that this project has been able to find after
~20 hours of measurement, most of which is recorded above as negatives.

L3 is **DONE** as of `fa6f6da` — the direct-upload default, 188–208 s → 99 s of cold start, gate at baseline —
so nothing that this project has measured remains on the table. The only open item is **L1**: porting the
engine's remaining prefill kernels, a session's worth of tedious, well-defined transcription with a 13.5x
prize at the end of it, and no other lead left that this project has been able to find.
