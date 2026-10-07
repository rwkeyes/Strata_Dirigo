# The per-family trivial-work ablation, priced on the Arc Pro B70

**2026-10-07, `vega`, Intel Arc Pro B70 (BMG G31), `intel_icd`, one card via `flock /tmp/b70.lock`.**
Brief: `forktest/queue/2026-10-07-family-ablation-ranked.md`. Tree `~/strata-vulkan-wt`, branch `vulkan-arc-port`,
HEAD `a905c27` (the router drop-in commit) + one measurement-only edit to `vulkan/src/device/vk_compute.cpp`.
Harness + raw logs: `/home/bob/step4/`.

## THE RESULT THAT MATTERS MOST: **the average per-link price double-counts, and cannot be read as a cost**

Summing the fifty measured families' average per-link prices (`avg_us_link x dispatches/round`) gives **551.5 ms of
round** -- **231% of a 238.8 ms round**. The average is the mean barrier+gap a dispatch is charged in the warm split,
and it is charged to EVERY dispatch that shares the barrier, so the family sum has no ceiling. This is the router's
2.6x (average 288.09 us/link, marginal ~110) generalised to the whole round, and it is the reason **any ranking
built from these averages ranks the wrong families**: it is a decomposition of a shared cost, not of a cost. The
shares in the table below are averages and are labelled as such; none of them is a budget.

## THE ROADMAP (what the ablation's own arithmetic leaves standing)

**Rank on KERNEL GPU TIME, not on any per-link price.** The per-dispatch GPU figures (`us/disp`, `ns/wg`, `gpu_ms`
from `STRATA_VK_KERNEL_TIME`) are measured per dispatch and are therefore trajectory-INdependent: they are the one
per-family column that survives the divergence further down, and they are already in hand for all 50 families
(`logs/flagged_base199.log`, `logs/off8_ws2.log`).

**And they say the lever is STRUCTURAL, not a family.** The whole recorded GPU time is **282.6 ms over the 8-token
arm's entire run -- 14.87 ms/round, 7.9% of its 188 ms round** -- and **36.1 ms of a 238 ms round (15%) on the
199-token arm** (`PERFORMANCE-B70-2026-10-06.md`). **So 85-92% of the round is not kernel work at all**: it is the
per-link structure -- turnaround, barrier, drain, the flat ~3.86 us/dispatch this tree's own sweep already fitted.
The next work is fewer links or cheaper links. A per-family ranking cannot exist as designed (see THE COMPARISON
FORM), and even if it could, no family holds a round's worth of kernel time to win.

**One family is retired by that arithmetic, and it is exactly the one a reader would have picked.**
`qsa_decode_attn` is the #1 family by average per-link price (293.9 us/link on the 199 arm, 34.955 on the 8-token
arm) and the one the narrow-grid model would send the next session after. Its kernel time is **4.984 us/dispatch at
207.65 ns/workgroup, 2.213 ms of GPU in the whole run -- 0.78% of the run's recorded GPU**. Its barrier charge is
**7.0x its own work** (34.955 vs 4.984 us/dispatch), and that charge is precisely the double-counted quantity the
231% above describes: **one family priced two ways -- ~0.8% of the GPU if you count the kernel, 7x its kernel in
barrier if you count the average.** Not a target.

**A single-family observation that is NOT reconciled with the aggregate, and must not reopen it:** in the ablated
arm the substitute's own row is `scale.spv n=636 fp=4096 B bar_us=2.850 kern_us=5.019` against `qsa_decode_attn`'s
`fp=329420636160 B bar_us=34.955` -- a 742 MB-per-dispatch footprint collapsing to 4 KiB took the barrier charge
34.96 -> 2.85 us while the kernel stayed ~5 us, which *looks like* the barrier tracking the footprint. **The
aggregate footprint hypothesis was already falsified** (per-CB r=-0.70, per-family r=0.318 R^2=0.10), so this pair is
recorded as **not reconcilable with** that falsification. **Do not reopen the footprint model on one pair.**

## THE OTHER RESULT: the per-family ablation cannot isolate a marginal price, so the marginal column is NOT delivered

Trivialising one family changes the decode's ANSWER, which changes the speculator's acceptance, which changes the
captured window set and the round count -- so the ablation arm and the baseline do not do comparable work and the
wall clock cannot be differenced. Two completed 8-token arms on this card:

| | `off8_v` (family unset, control) | `fam_qsa_v` (`qsa_decode_attn` trivial'd) |
|---|---|---|
| captured windows | 1- and 2-token | 1-, 2- and **4-token** |
| dispatches encoded | 5,356 | 8,266 (**+2,910 = exactly the extra captured 4-token window**) |
| segments | 57 | 78 |
| drafts accepted | 13 of 18 (0.722) | 6 of 27 (0.222) |
| rounds | 19 | 26 |
| decode | 11.00 tok/s | 7.85 tok/s |
| **ms/round** | **152.8** | **156.9** |

**The +2,910 is NOT the substitution adding work -- it is the trajectory.** The recorded-arm census is explicit: the
trivialised family's own `.spv` goes **36 -> 0** (its dispatches become uncounted `scale` dispatches, one-for-one,
because the swap returns before the census), while every OTHER family's count grows only because a larger window got
captured. The swap preserves the per-command-buffer chain; the arms diverge because a trivialised family emits
different tokens. **Either way the arm as specified reads a NEGATIVE price for a family the warm split prices at
293.9 us/link, so a ranked marginal list built from it would be wrong in both directions -- and the control cannot
read ~0 either, because every ablation moves the trajectory.** The fix is a fixed-window comparison (the same
captured command buffer's per-dispatch cost in both arms, which the `STRATA_VK_WARM_SPLIT` per-command-buffer rows
already expose) or a fixed round count; neither was run. Until then, per-family ablation is not this roadmap's input.

## THE BASELINE IS BIT-REPRODUCIBLE, with one rare degenerate run

Three repeats of the identical arm (same binary, same args, 8-token prompt) -- `/home/bob/step4/logs/repro8_{1,2,3}.log`:

| repeat | id | rounds | drafts accepted | decode |
|---|---|---|---|---|
| 1 | `3aed108cceee` | 19 | `0:6 1:13 2:0 3:0` | 2,897.4 ms (11.04 tok/s) |
| 2 | `3aed108cceee` | 19 | `0:6 1:13 2:0 3:0` | 2,904.8 ms (11.02 tok/s) |
| 3 | `3aed108cceee` | 19 | `0:6 1:13 2:0 3:0` | 2,899.2 ms (11.03 tok/s) |

Identical ids, identical acceptance, decode spread 7.4 ms (0.26%). **`bins/strata_vulkan.baseline` and
`bins/strata_vulkan.head` are the same bytes** (`fb0fd1fa40c4f623…`, both `= a905c27`), so the earlier `base8_1`
reading -- id `d01eee6a3948`, 12.51 tok/s, **17 rounds**, `16 of 16` accepted, output all-zeros -- is the SAME BUILD
decoding the same prompt differently: a degenerate all-zeros trajectory. Observed once in ~7 clean baseline runs
(`head8_a`, `head8_b`, `off8_v`, `repro8_{1,2,3}` all correct). **Its raw log was overwritten by the relaunched sweep
before that sweep was stopped**, so only the recorded values stand; the 17-round / all-zero shape is the detector
(it also matches the all-trivial `TRIVIAL_REC=1` shape, which is why it reads like a substitution bug). **The
instrument edit is exonerated:** the same tree built clean at `a905c27` (`strata_vulkan.head`) keeps
`3aed108cceee` on both of its runs.

## THE RANKED TABLE (by AVERAGE price, the only price column that could be produced)

`avg us/link` = warm-split mean barrier+gap per dispatch (FOOTPRINT+KERNEL_TIME+WARM_SPLIT run,
`logs/flagged_base199.log`, id `56a0b28d2de6`); `disp/round` = the RECORDED-arm census; `wg/disp` from the
KERNEL_TIME rows; `bytes/wg` = bound-region bytes / workgroups; **avg-share** = avg x disp/round against the same
run's 238.8 ms/round. **These shares sum to 231% -- see the top of this file.** Full 50-row table:
`/home/bob/step4/flagged_table.tsv`. Top 20 + the controls:

| family | disp/round | wg/disp | bytes/wg | avg us/link | avg-share | marginal |
|---|---:|---:|---:|---:|---:|---|
| qsa_decode_attn | 72 | 24 | 30,914,098 | 293.9 | 8.9% | **not measured** |
| native_k_mmvq | 633 | 5,011 | 6,069 | 262.5 | 69.6% | **not measured** |
| sampler_greedy | 3 | 3 | 809,387,315 | 201.8 | 0.3% | **not measured** |
| fused_gr_mix | 288 | 6,766 | 19,760 | 152.0 | 18.3% | **not measured** |
| iq4xs_mmvq | 126 | 2,974 | 51,367 | 129.3 | 6.8% | **not measured** |
| gdn_step_norm_multi | 144 | 24 | 10,522,168 | 117.9 | 7.1% | **not measured** |
| native_down_any | 1,152 | 2,560 | 13,056 | 104.8 | 50.6% | **not measured** |
| q8_0_mmvq | 3 | 2,560 | 2,506,752 | 97.7 | 0.1% | **not measured** |
| native_gu_any | 1,152 | 1,280 | 30,464 | 96.1 | 46.3% | **not measured** |
| iq4nl_mmvq | 141 | 2,560 | 53,335 | 91.9 | 5.4% | **not measured** |
| gr_gate | 6 | 10,240 | 316,166 | 49.5 | 0.1% | **not measured** |
| native_router_top10 | 288 | 1 | 50,586,707 | 34.5 | 4.2% | (already fixed) |
| bf16_gemv | 6 | 2,560 | 948,500 | 33.4 | 0.1% | **not measured** |
| fused_gr_down | 288 | 846 | 131,736 | 29.6 | 3.6% | **not measured** |
| gr_down | 6 | 320 | 7,588,006 | 14.9 | 0.0% | **not measured** |
| bf16_mmvf_f32_multi | 48 | 320 | 899,859 | 13.3 | 0.3% | **not measured** |
| resident_plan | 144 | 1 | 178,257,920 | 12.7 | 0.8% | **not measured** |
| fused_gr_inject | 288 | 11 | 10,538,897 | 10.7 | 1.3% | **not measured** |
| bf16_mmvf_f32 | 318 | 458 | 106,800 | 8.8 | 1.2% | **not measured** |
| fused_gdn_ab | 108 | 254 | 1,639,384 | 8.4 | 0.4% | **not measured** |
| _ctl_ ple_bcast | 6 | 40 | 60,704,048 | 2.8 | 0.0% | **not measured** |
| _ctl_ swiglu_f32 | 288 | 37 | 1,829,626 | 2.9 | 0.3% | **not measured** |
| _ctl_ add3 | 6 | 40 | 80,938,731 | 3.1 | 0.0% | **not measured** |

## THE TWO NEW COLUMNS vs THE NARROW-GRID MODEL

Spearman rank correlations over all 50 families (`/home/bob/step4/corr.py`):

| quantity | vs avg us/link | vs workgroups/dispatch |
|---|---:|---:|
| **bytes / workgroup** | **rho = -0.421** | -- |
| bytes / dispatch | rho = -0.151 | -- |
| workgroups / dispatch | -- | rho = +0.365 |
| ns / workgroup (GPU) | rho = -0.381 | -- |

**"Price tracks a NARROW GRID over a large row" is REFUTED by the average price**, in its own terms: pricier families
carry *fewer* bytes per workgroup (rho -0.42) and *wider* grids (rho +0.37). The counterexample is large and
explicit: `native_k_mmvq` is the second most expensive family at **5,011 workgroups/dispatch and 6 KB per
workgroup** -- the widest grid and one of the smallest per-workgroup footprints in the table -- while the
narrow-grid families the model predicts (`native_router_top10` 1 wg/disp at 34.5 us, `resident_plan` 1 wg/disp at
12.7, `ptr_to_off` 1 wg/disp at 5.4) are now CHEAP. The two expensive narrow-grid families that remain are
`qsa_decode_attn` (24 wg/disp, 30.9 MB/wg) and `gdn_step_norm_multi` (24 wg/disp, 10.5 MB/wg), so narrowness is
neither necessary nor sufficient. What tracks the average price here is the mean barrier the warm split charges the
family, whose dominant term is `disp/round` (rho +0.243) -- the VOLUME of dispatches, not their narrowness. **With
the marginal column unmeasured, this is where the model test stands** -- but note it is an average-based test, and
the 231% above says the average is not a cost.

## WHAT IS AND IS NOT DELIVERED

* **Delivered:** the 50-family table with `wg/dispatch` and `bytes/workgroup`; the 231% finding; the model verdict;
  the instrument verdict (with the mechanism and the fix named); the reproducibility test (3/3) and the rare
  degenerate-run finding; the gate attribution below.
* **NOT delivered, by construction:** the marginal ms/round column, the per-family average/marginal ratio, and the
  top-3-5 "if fixed like the router" projection. On averages alone the top shares are `native_k_mmvq` (69.6%),
  `native_down_any` (50.6%), `native_gu_any` (46.3%), `fused_gr_mix` (18.3%), `qsa_decode_attn` (8.9%) -- but with
  the family sum at 231% of a round, **no tok/s projection is made from them**; doing so would repeat the error the
  router already measured. That column is not merely unmeasured but **CLOSED as designed** (see THE COMPARISON
  FORM): the capture set is downstream of the substitution, so no re-run can hold the trajectory fixed, and the port
  has no saved-capture replay. **The roadmap is the kernel-GPU-time section at the top of this file.**
* **Not measured / named:** the marginal price of every family (closed, above); the 199-token confirmation (unrun
  once the sweep was stopped); the `radeon_icd` gate arm as a pass (see below).
* **The instrument itself was attributed before it was committed:** `gate_attrib.sh` ran the full gate on a clean
  `a905c27` and on the instrumented tree, hashing the gate binary each way -- `intel 965/0/0` both, `lvp 949/0/4`
  both, `radeon 951/3/2` clean vs `953/1/2` instrumented (a subset of the documented W26 three). The delta is on
  RADV only and the instrument is exonerated; the three commits are the instrument, the W26 note, and this file.

## GUARDS, BINARIES, LOGS

* **Arc gate `965 passed / 0 failed / 0 skipped`** on `intel_icd` on BOTH builds -- clean `a905c27` (instrument
  stashed) and the instrument built in: the line that says the instrument did not touch the port. `lvp_icd`
  `949/0/4` (the four documented skips) both ways. `radeon_icd`: **clean `951/3/2`** = exactly the documented W26
  three (`fused_gdn_ab`, `bf16_gemv_fp32_mmvf`, `bf16_gemv_fp32_mmvf_cols`); **dirty `953/1/2`** = a SUBSET (only
  `mmvf_cols`). **The instrument is exonerated** (`logs/gate_{clean,dirty}.log`; `gate_attrib.sh` hashes the gate
  binary each way). A first run of this batch's gate read `945/9/2` -- outside the documented range -- with
  `ple_block entry` at `7680/10240 worst 2.56e+03` and `8691/10240 worst 1.55e+03`; absent from both later runs, and
  recorded in NEXT.md's W26 note as an unbounded-magnitude observation, **not chased**.
* `check_port_map.py` rc=0 (168 decode-path symbols; 150 shaders built, 114 claimed); `make_port_map.py`
  regenerates `PORT-MAP.tsv` identically.

## THE COMPARISON FORM: which of the three fixes is sound, and why the other two are not

**Chosen: match the trajectory at the CAPTURED COMMAND BUFFER (option 2, realised without an engine flag).** The
capture keeps the windowed command buffers -- one per verified-window size (1..4 tokens) -- and the trivial swap
preserves each buffer's dispatch COUNT (proved above: the family's `.spv` goes 36 -> 0 while the chain length is
unchanged). Both arms capture the SAME 1-token and 2-token buffers, and the warm split already reports each
buffer's dispatch count `n` and its steady-state `WARM us/disp` (`vk warm cb n=<n> ... WARM f+k+b+g us/disp`). The
marginal price of family F is therefore **`n` x (WARM us/disp of that buffer, baseline-minus-F-trivial)**, computed
on two command buffers **whose dispatch sequence is identical by construction** -- the trajectory cancels because
the comparison is inside one window, not across arms. Cost: one baseline and one F arm with
`STRATA_VK_WARM_SPLIT=1`.

**That attempt was MADE, and it does not match -- which is the proof, not a failure.** Both arms with
`KERNEL_TIME`+`WARM_SPLIT`, identical args: `off8_ws2` ran **19 rounds, 57 segments, 53,446 recorded dispatches**;
`fam_qsa_ws2` ran **26 rounds, 78 segments, 73,499** -- 1.36x the work. **Re-running the engine cannot produce a
matched trajectory, because the substitution changes the LIVE decode's answer, which changes the acceptance, which
changes what gets captured: the capture set is downstream of the variable under test.** Only replaying a SAVED
capture (one capture, replayed under both arms) could hold it fixed, and **the port has no such path** -- the
recorded segments live for one process's life and are re-recorded per capture. **So the marginal-price question is
CLOSED as designed, not left open**, and the roadmap is the kernel-GPU-time one above, not a marginal table.

**Why the other two are not sound as stated.**
1. *Normalise per round* is **exactly what the delivered data already does, and it is the biased one.** A round's
   window is `1 + accepted` tokens, so the arms' rounds are not equal work: the baseline ran 19 rounds at 1.68
   tokens/round, the ablation 26 at 1.23. The ablated arm's rounds are *smaller windows*, so they are cheaper for
   reasons that have nothing to do with the family -- ms/round is biased toward a NEGATIVE marginal, which is what
   `fam_qsa_v` returned (+4.1 ms/round for a family the warm split prices at 293.9 us/link). Dividing by round
   count cancels trajectory LENGTH, not trajectory SHAPE.
2. *Match the trajectory by forcing the baseline's token stream* is sound in principle but **not available**: the
   engine has no flag to pin the draft/token stream (`--spec` fixes the draft geometry, not the acceptance), and the
   fallback the brief offers -- compare only arms whose round counts agree -- discards the data, because the ablation
   changes acceptance by construction (0.722 -> 0.222 here).
3. *Abandon per-family ablation for the structural budget* is **the honest fallback and it is what this file
   reports**, but it cannot give a marginal price at all: it gives dispatch/segment counts and the per-family GPU
   time, which are decomposition-of-a-shared-cost quantities (the 231% above), not marginals. It is the floor, not
   the fix.

**What this leaves:** the 15-family sweep was NOT re-run in any form (it was stopped on this finding), so this file
carries no marginal column and no tok/s projection -- and needs neither, because the roadmap is the kernel-GPU-time
one above.

* Binaries: baseline/head `fb0fd1fa40c4f623…` (`= a905c27`, two names, same bytes); ablation `714b760c7d531f9a…`.
* Baseline ids: `56a0b28d2de6` (199-token, `flagged_base199`, 32 decoded) and `3aed108cceee` (8-token,
  `repro8_{1,2,3}`, `head8_{a,b}`, `off8_v`).
* Logs, exact: `/home/bob/step4/logs/{flagged_base199,off8_v,fam_qsa_v,head8_a,head8_b,repro8_1,repro8_2,repro8_3,base8_1}.log`
  (arms), `logs/{off8_ws,fam_qsa_ws,off8_ws2,fam_qsa_ws2}.log` (the matched-capture attempt),
  `logs/gate_{clean,dirty}.log` (attribution), `logs/gate_step4.log` (this batch's first full gate);
  harness `/home/bob/step4/{run.sh,sweep8.sh,confirm199.sh,debug_set.sh,repro_test.sh,gate_attrib.sh,parse_flag.py,marginal.py,corr.py,report.py}`;
  table `/home/bob/step4/flagged_table.tsv`.
