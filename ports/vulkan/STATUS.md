# Status — what is done, what is verified, what is not

## THE SAMPLER, MEASURED PROPERLY — the engine's DEFAULT is the SPLIT, and the PENALTY HOIST — the performance tier's first target (2026-10-05)

The one number in the port above 100 ms, with a caveat the release notes carried: `sampler_kernel_f32` **546.9 ms
(Arc) / 202.0 ms (XTX)** is the ONE-BLOCK **fallback**, not the engine's default.  Both halves are now settled.

**THE DEFAULT, BY CALL SITE.**  `sample_tokens` (`src/kernels/cuda/sampler.cu:977`): greedy or `temp <= 0` ->
the argmax (`:988`); `sampled_path() == Old` (`STRATA_OLD_SAMPLER=1`) -> the one-block `sampler_kernel` (`:1000`);
else the default `:1013`
`sampled_path() == Split && n_blocks <= 64 && n_tokens <= 64 && !stream_capturing(stream)` -> the **SPLIT**
(`sampler_split_part_kernel` + `sampler_split_merge_kernel`), with `sampler_one_block_kernel` as the FALLBACK.
`sampled_path()` is `Split` unless `STRATA_OLD_SAMPLER`/`STRATA_SAMPLER_ONE_BLOCK` is set (`:854-859`).  At the
model's vocab 248,320, `n_blocks = ceil(248320/4096) = 61 <= 64`, so **the split is what the engine runs, and the
546.9/202.0 figure was never the default's cost.**

| device (vega) | `sampler_kernel_f32` **fallback** (tokens=1) | `sampler_split` **DEFAULT** (tokens=1) | split/block | DEFAULT, tokens=64 |
|---|---:|---:|---:|---:|
| Arc B70 (intel_icd) | 546.81 ms | **13.50 ms** | **0.025** | 13.97 ms |
| Ryzen iGPU (radeon_icd) | 301.55 ms | **9.06 ms** | **0.030** | 74.91 ms |
| llvmpipe (lvp_icd) | 1171.46 ms | **67.64 ms** | **0.058** | 391.29 ms |

Shipped `bench/vk_bench.cpp` rows, `reps=5`, one dispatch per batch; full tables in `bench/README.md`.

**WHY IT WAS SLOW, MEASURED BEFORE FIXING.**  The split cost **348.95 ms (Arc) / 161.62 ms (iGPU) / 665.46 ms
(llvmpipe)** pre-fix.  Not bandwidth (the 993 KB row loads in microseconds); not the one-workgroup-per-row
decomposition (64 rows cost ≈ what 1 row cost: Arc 348.95 -> 384.54 ms); it WAS a serial scan in the wrong place —
removing the penalty window took it to **12.66 ms (27.6×)**.  `sampler_row_topk` re-read each logit and re-scanned
the whole history window on EVERY k round (`O(k · span · hlen)`), where the engine builds a 4,096-bit block bitmap
and penalises each element ONCE into registers (`src/kernels/cuda/sampler.cu:690-706`).

**THE FIX.**  `common/sampler_select.glsl` caches each partition element's penalised logit once in the lane's
registers (the engine's `s[kSplitPerLane]` shape, `SC_PER_LANE = 16`) and runs the k rounds over the cache; the
selection is byte-for-byte unchanged.  **348.95 -> 13.50 ms Arc (25.8×), 161.62 -> 9.06 iGPU (17.8×), 665.46 ->
67.64 llvmpipe (9.8×).**  The include is shared, so `coupled_sample.comp` gets the hoist too.

| case | rule | measured (vega Arc) | falsified by |
|---|---|---|---|
| `sampler_split` (SIX arms now; the new 6th exercises a real penalty hit) | partition top-k + ordered merge, tail in double; the 6th arm: id 100 × 8 in the window, `penalty_repeat = 4` -> head 9.0/4 = 2.25, selection moves to id 4200 | 6 arms bit-exact vs the transcribed chain; parity with `sampler_kernel` unchanged | `sampler-select-penalty-drop` -> **FAIL `sampler_split: the repeat penalty is applied ONCE per partition (the hoist)` 2/16**; `sampler-split-merge-drop-parts` still 9/16 |

**Gate, after the change: vega Arc 445/0/0, lvp 433/0/3, radeon-iGPU 436/0/2 (`run_gate.sh` exit 0)** — +1 verdict
per implementation (the new arm).  Map unchanged: `check_port_map.py` passes (`168 — 78 kernel, 61 host, 29 todo`),
`make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **Box `z820b`: PENDING** — the box is suspended
(`100% packet loss`, `No route to host`, no WoL), so the default's cost on the XTX is NOT measured and no cross-host
number is claimed; 202.0 ms stands as the XTX fallback figure.  Full detail in `NEXT.md`'s top section.

---

## THE BF16-PROJECTION PAIR — `bf16_gemv` + `bf16_gemv_split` (one shared shader) — and the 8 SPECULATIVE-DRAFTER ROWS LABELLED CLASS C (2026-10-05)

The SECOND soft edge the batch-5 REACHABILITY AUDIT raised is CLOSED by porting BOTH members of the
`native_bf16_projections` toggle that `project_bf16` (`src/core/layer.cpp:94-100`) selects between.  The setting
(`layer.cpp:91`) DEFAULT is **false** and the port pins it nowhere, so the DEFAULT selected the UNPORTED
`bf16_gemv_split` (layer.cpp:291/:292/:367) and `bf16_gemv` (:918/:962) on the MAIN forward path — the same shape
as the `native_qsa_indexer_append` hole.  `--native` sets the setting true (generate.cpp:1805 → :2286) and the
TRUE branch was already the ported `bf16_gemv_fp32_mmvf`; porting both sides closes the edge for EITHER value.
Pinning `layer_set_native_bf16(true)` was rejected (the setter is engine host code the port does not fork).

| case | rule (the engine's OWN body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `bf16_gemv` (4 arms 2560/512, 2560/128, 128/64, 2/1) | `bf16_gemv.cu`: `y[o]=Σ f32(x[i])·f32(w[o*n_in+i])`, products exact in f32, row read as 32-bit PAIRS, plain `acc+=a*b` (NOT `__fmaf_rn`) | **515/515 w 1.22e-02**, 131/131 w 5.73e-03, 67/67 w 3.2e-02, 4/4 w 1.84e-03 (terms-derived `gemv_bound`) | `bf16-gemv-swap-halves` → FAIL 4/515 w 1.54e+05 |
| `bf16_gemv_split` (same shader; 4 arms 2560/512, 2560/48, 64/32, 2/1) | `bf16_gemv_split_kernel`/`bf16_gemv_warp_kernel` — same RULE, different reduction; both entry points rendered as ONE WORKGROUP per row (subgroup ops banned; the CUDA's warp shapes are a strategy, not the rule) | **515/515 w 6.99e-03**, 51/51 w 6.37e-03, 35/35 w 1.42e-02, 4/4 w 9.61e-03 | `bf16-gemv-row-base` → FAIL 4/515 w 2.62e+34 |

**A MEASUREMENT THAT CHANGED THE SHIPPED KERNEL.**  The naive one-thread-per-row decomposition (a real CUDA path)
was built and timed FIRST: at the engine's shapes it is **14–18x slower** than the workgroup form (Arc
`split/serial 0.073` at n_out=512, 0.057 at 48; Ryzen iGPU 0.103/0.055) because it is uncoalesced — the CUDA's own
comment says so, and CUDA takes it only below n_out=64.  The port therefore SHIPS the workgroup-per-row rendering
and the naive variant is not in the tree.  ONE shader serves both engine symbols (the
`bf16_gemv_fp32_mmvf`/`_cols` precedent).

**THE MEASUREMENT (ported `bf16_gemv` vs the ported native `bf16_gemv_fp32_mmvf`; same workgroup-per-row
decomposition, only the activation precision differs → a WASH is expected; full table in `bench/README.md`).**
A **WASH**: native/ported `bf16_gemv` **1.000** (Arc, n_out=512), **1.006** (Arc, 48), **0.996 / 0.997** (Ryzen
iGPU), **0.933 / 0.956** (llvmpipe).

**THE DRAFTER ROWS.**  `add_streams_broadcast`, `fused_gr_read_multi`, `window_ids`, `qsa_decode_attn_batch`,
`moe_group_resident`, `row_top_prob`, `map_ids`, `mtp_select` are labelled **CLASS C**: reachable only under the
`--spec 4 --mtp` draft loop, which this port does not select (`Verifier::init` refuses under
`layer_verify_compatible()`, which needs `g_fused_gr` false→forced, `native_qsa_indexer_enabled()` false, and
`native_bf16_projections` unpinned).  The flag chain that WOULD enable them is `--spec 4 --mtp` AND a
verifier-compatible native stack.  The TSV keeps kind `todo` but carries the class-C reason; the classification is
in `plan/DECODE-PATH-TRIAGE.md`.

**THE MAP MOVES BY TWO ROWS:** `168 — 76 kernel, 61 host, 31 todo` → **`168 — 78 kernel, 61 host, 29 todo`**;
`check_port_map.py` passes (`111 shaders built, 92 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the change: vega Arc 444/0/0, lvp 432/0/3, radeon-iGPU 435/0/2 (exit 0)**;
**box `z820b`: NOT RUN — the box was UNREACHABLE this batch** (`No route to host`, ARP `INCOMPLETE`, no WoL path
from vega), so the box run is PENDING a whole-tree sync + `run_gate.sh`/`run_bench.sh` there.  Both injections
BIT (raw lines in `NEXT.md`).  Full detail in `NEXT.md`'s top section and `plan/DECODE-PATH-TRIAGE.md`.

---

## THE QSA GATE'S NATIVE MEMBER — class B batch 5 — `native_qsa_enabled()` FLIPS TO TRUE + the REACHABILITY AUDIT (2026-10-05)

`native_qsa_gate_apply` — the LAST symbol `native_qsa_enabled()` gates — is ported, gated against the engine's OWN
native body (`src/kernels/cuda/native_qsa.cu`'s `gate`, `:69-77`), and MEASURED against the legacy
`qsa_gate_apply_f32` it replaces.  With both gated symbols now having a shader, the flag's invariant is satisfied
and **`native_qsa_enabled()` answers TRUE**.  This batch also carries the **REACHABILITY AUDIT** of every remaining
`todo` row (`plan/DECODE-PATH-TRIAGE.md`) and **fixes the one hole it found**.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `native_qsa_gate_apply` (3 arms 24/256, 4/12, 2/8) | `native_qsa.cu` `gate`: `out = attn * (1/(1+expf(-raw)))`, `raw` = the SECOND half of each head's 2*head_dim block, all f32 | **6152/6152 w 7.89e-07**, 56/56 w 7.54e-07, 24/24 w 7.08e-07 | `native-qsa-gate-first-half` → FAIL 8/6152 w 4.13e+10 |
| `native capabilities: qsa flag` | `native_qsa_enabled()` == "every gated symbol has a built shader"; both QSA shaders now built | **3/3** — flag **TRUE** | under-claim `native-caps-qsa-false` → FAIL 2/3; over-claim (remove the `.spv`) → FAIL 1/3 |

**The rule is the SAME as the legacy member's; the arithmetic is not.**  The native body computes the sigmoid and
the product in **f32** (`expf`); the legacy CUDA in **f64**.  The target has no shaderFloat64, so this port's legacy
shader already computes in f32 — hence the native-vs-legacy pair is a **WASH** on all six devices measured
(**1.001 Arc / 0.984 iGPU / 1.018 lvp / 1.019 XTX / 0.904 K620 / 1.120 lvp**).  There is no dispatch chain (the QSA
gate is one dispatch either way), and that is stated rather than invented.  Full table in `bench/README.md`.

**THE FLIP WAS CHECKED.**  `native_qsa_enabled()` gates exactly two symbols — `native_qsa_rms_norm_weighted`
(`layer.cpp:879`, `mtp.cpp:488/491/514`, `verify.cpp:767`; ported batch 1) and `native_qsa_gate_apply`
(`layer.cpp:1010`, `verify.cpp:775/883/887`; ported here) — and the verify sites are unreachable because the P6
verifier cannot init (`layer_verify_compatible` still demands `native_bf16_projections` and `g_fused_gr` false, and
`native_qsa_indexer_enabled()` false).  Nothing unported becomes reachable.

**THE HOLE AND ITS FIX.**  The audit found `native_qsa_indexer_append` (`layer.cpp:944`, the MAIN forward path of
all 12 QSA layers every token) reachable because the shipped `--native` sets `o.native_qsa_indexer = true`
(`generate.cpp:1807`) and calls `native_qsa_indexer_set_enabled` (`:2294`) — while the Vulkan backend defined
**NEITHER the getter nor the setter**, though the contract named the required answer (`false`).  The backend now
defines both, answering **false** → the ported `indexer_key_append`.  The map row stays `todo`; the hole was
reachability.

**THE MAP MOVES BY ONE ROW:** `168 — 75 kernel, 61 host, 32 todo` → **`168 — 76 kernel, 61 host, 31 todo`**;
`check_port_map.py` passes (`110 shaders built, 91 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the commit: vega Arc 436/0/0, lvp 424/0/3, radeon-iGPU 427/0/2 (exit 0)**, +4
verdicts per arm.  **Box `z820b`: XTX 432/0/1 (the pre-existing M8 skip, so the script exits 1), K620 427/0/2,
llvmpipe 424/0/3 — 0 failed on every arm.**  The retired injection `native-caps-qsa-true` is recorded in `NEXT.md`.

**THE AUDIT IN ONE LINE:** of the 32 `todo` rows audited, 1 is now `kernel`, **11 are reachable-but-unported** (8
speculative-drafter symbols reachable only under the shipped `--spec 4 --mtp`; `native_qsa_indexer_append`, now
flag-closed; and `bf16_gemv`/`bf16_gemv_split` as a soft edge on the unnamed `native_bf16_projections` setting),
and 20 are unreachable under the current capability answers — each with its deciding call site named.

---

## THE PERFORMANCE TIER'S FUSED GDN PATHS — class B batch 4 — and `native_gdn_enabled()` FLIPS TO TRUE (2026-10-05)

The three fused **GDN paths** — `fused_gdn_conv_l2`, `fused_gdn_ab`, `fused_gdn_step_norm` — the LAST symbols
`native_gdn_enabled()` gates.  Each is gated and oracled against the engine's OWN fused body
(`src/kernels/cuda/fused_gdn.cu`), each MEASURED against the multi-dispatch chain it replaces and against the
non-fused native kernel(s) where a comparison exists.  With ALL NINE gated symbols now having a shader, the
invariant `case_native_capabilities` asserts is satisfied and **`native_gdn_enabled()` answers TRUE**.

| case | rule (the engine's OWN fused body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `fused_gdn_conv_l2` (3 arms C=10240/512/384) | `fused_gdn.cu` `gdn_conv_l2_kernel`: 4-tap conv (no zero-bias) + SiLU + slide, then a per-head L2 `rsqrt(sum y^2 + eps)` on the q/k heads only | **51200/51200 w 2.66e-05**, 2560/2560 w 9.46e-06, 1920/1920 w 2.68e-05 (Arc); slid history bit-exact | `fused-gdn-conv-l2-drop-norm` → FAIL 49149/51200 w 1.93e+05 |
| `fused_gdn_ab` (3 arms h_v=48 n=2560 / h_v=4 n=64 / h_v=3 n=512) | `fused_gdn.cu` `gdn_ab_kernel`: BF16 mat-vec (32-bit pairs, low=2p/high=2p+1) then `beta=sigmoid(acc)`, `gate=softplus(acc+dt)*ssm_a` | **99/99 w 4.25e-07**, 11/11 w 1.33e-07, 9/9 w 7.79e-08 (Arc); terms bound, per-row margins | `fused-gdn-ab-swap-bf16-halves` → FAIL 3/99 w 25.9 |
| `fused_gdn_step_norm` (3 arms S=128; h_v=48/8/9) | `fused_gdn.cu` `gdn_step_norm_kernel`: folded-decay recurrence (contract vs UNDECAYED state) + readout vs UPDATED state + fused `1/sqrt(S)` + closing RMS (eps on the MEAN) with gamma and sigmoid(z) | **792580/792580 err/tol 0.319**, 132100/132100 w 0.0508, 148612/148612 w 0.0109 (Arc) | `fused-gdn-step-norm-silu-not-sigmoid` → FAIL 788501/792580 w 9.34e+04 |
| `native capabilities: gdn flag` | `native_gdn_enabled()` == "every gated symbol has a built shader"; all 9 now built | **4/4** — flag **TRUE** | `native-caps-gdn-false` → FAIL 3/4 (the old `native-caps-gdn-true` is retired — true is now the truth) |

**THE MEASUREMENT (fused/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
Every fused path beats the multi-dispatch chain it replaces on every device measured: `fused_gdn_conv_l2` vs
conv_silu+2× l2_norm **0.452 Arc / 0.712 iGPU / 0.560 lvp**; `fused_gdn_ab` vs 2× mmvf+beta_gate+gate **0.398 /
0.878 / 0.744**; `fused_gdn_step_norm` vs step+out_norm **0.929 / 0.987 / 0.930**.  The fused-vs-`native_gdn_step`
pair is a wash (1.021 / 0.998 / 1.092 — the fused kernel does that recurrence PLUS the norm's work), and the
fused-vs-`native_gdn_out_norm` row (6.2–63.7) is **not a like-for-like ratio** (the fused dispatch does the whole
step while out_norm alone is a ~70× smaller elementwise pass).  The win is in REMOVING DISPATCHES, as batches 2–3
found; a wash and a non-like-for-like row are both reported.

**THE FLAG FLIPS, and the fused paths' extra gates are SETTINGS, not capabilities.**  `fused_gdn_step_norm` is
selected by `g_fused_gdn && native_gdn_enabled() && state==128` (layer.cpp:306) — **no `native_bf16_projections`
term** — and `g_fused_gdn` **defaults true** (layer.cpp:42), so the flag flip alone makes that path reachable on
the default config: the shader HAD to exist before the flip (the strict invariant's point).  `fused_gdn_conv_l2` /
`fused_gdn_ab` additionally need `native_bf16_projections` (layer.cpp:247), a host setting defaulting false, set
by `--native-bf16`/`--native` (generate.cpp:2286), whose dependency `bf16_mmvf_f32` IS ported and gated.  The P6
verifier stays unreachable after the flip (`g_fused_gr` false, `native_qsa_indexer_enabled()` false,
`native_bf16_projections` false), so the router/moe `_multi` symbols stay off the path.

**THE MAP MOVES BY THREE ROWS:** `168 — 72 kernel, 61 host, 35 todo` → **`168 — 75 kernel, 61 host, 32 todo`**;
`check_port_map.py` passes (`109 shaders built, 90 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the commit: vega Arc 432/0/0, lvp 420/0/3, radeon-iGPU 423/0/2 (exit 0)** — the
radeon run's sole `budget: independent requery agrees` failure was the documented intermittent flake (a direct
re-run read 423/0/2 twice).  **Box `z820b`: XTX 428/0/1 (the 1 is the pre-existing M8 skip, so `run_gate.sh`
exits 1 there), K620 423/0/2, llvmpipe 420/0/3 — 0 failed on every arm**; the XTX's first run carried the same
`budget` flake (two direct re-runs read 428/0/1).  Four falsification injections all BIT.  **NOTE for the next
batch: the box's `vulkan/` must be synced too (the brief's sync command omits it) — a stale
`native_caps_vk.cpp` fails the flag arm 3/4.**

**TWO FIXTURE FINDINGS (both fixed, both recorded in `NEXT.md`).**  (1) The dropped-readout-scale rival is NOT an
exact invariance under the closing RMS (measured 0.109–0.19 rel-L1 — it is a moving margin, asserted > 0.05; the
first case asserted the opposite and the gate caught it).  (2) The vector rel-L1 softplus margin in `fused_gdn_ab`
was swamped by the forced branch row, so the case now uses a per-row max and a fixture whose `acc` lands in the
softplus-sensitive band.  The device matched the rule throughout — the fixtures were the defect.

---

## THE PERFORMANCE TIER'S GDN / DELTANET MIXER, the remaining three native kernels — class B batch 3 (2026-10-05)

The LAST three native **GDN / DeltaNet mixer** kernels — **COMPLETING the six** — `native_gdn_gate`,
`native_gdn_step`, `native_gdn_out_norm`, each replacing a legacy kernel already ported and gated, each oracled
against the engine's OWN native body (`src/kernels/cuda/native_gdn_preprocess.cu` / `native_gdn.cu`), each
MEASURED against that legacy kernel at the same shape on the same device (and, for `native_gdn_step`, against
the two-dispatch legacy chain it replaces), with **`native_gdn_enabled()` left answering FALSE** (the flag also
gates the three unported `fused_gdn_*` paths) and `case_native_capabilities` enforcing it.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc / box XTX) | falsified by |
|---|---|---|---|
| `native_gdn_gate` (3 arms h_v=48/5/300) | `native_gdn_preprocess.cu` `gate_softplus`: `softplus(alpha+dt) * ssm_a`, per-head one token; the threshold-20 branch | **48/48 w 2.79e-07**, 5/5 w 1.04e-07, 300/300 w 3.06e-07 (Arc); raw identity checked host-side to move | `native-gdn-gate-drop-ssm-a` → FAIL 0/48 w 8.89 |
| `native_gdn_step` (2 arms 128/16/48, 128/4/8) | `native_gdn.cu` `step`: decay FOLDED into the rank-1 update, contract against the UNDECAYED state, `1/sqrt(S)` readout scale FUSED; one thread per column (see the decomposition finding) | **792576/792576 err/tol 0.00155**, 132096/132096 w 0.00107 (Arc); INTERLEAVE + dropped-scale margins | `native-gdn-step-drop-readout-scale` → FAIL 786434/792576 w 2.03e+04 |
| `native_gdn_out_norm` (3 arms h_v=48/4/3, cols=128) | `native_gdn_preprocess.cu` `out_norm`: `(rms_norm(o) * gamma) * sigmoid(z)`, eps on the MEAN, cols==128 | **6400/6400 w 7.04e-07**, 768/768 w 6.13e-07, 640/640 w 4.92e-07 (Arc); NaN row-guard | `native-gdn-out-norm-silu-instead-of-sigmoid` → FAIL 2321/6400 w 21 |
| `native capabilities: gdn flag` | `native_gdn_enabled()` == "every gated symbol has a built shader"; the 6 native shaders exist | **4/4** — flag **FALSE** (the 3 unported fused shaders absent) | `native-caps-gdn-true` → FAIL 3/4 |

**THE MEASUREMENT (native/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
`native_gdn_step` is a **WIN — 0.917 Arc / 0.776 iGPU / 0.866 lvp (pair)** and **0.867 / 0.771 / 0.829 against
the legacy `scale_inplace`+`gdn_step` chain**; it does LESS memory traffic than the legacy kernel (the legacy's
first pass STORES the decayed state).  `native_gdn_gate` (0.901–1.044) and `native_gdn_out_norm` (0.974–1.144)
are **WASHES** — the same work per element and, at the layer's one-token / one-row shape, no algorithmic
difference to win.  A native kernel is not required to be faster, and a wash is reported as a finding.  **On the
box's XTX the three cases read `native_gdn_gate` 48/48 w 3.38e-07, `native_gdn_out_norm` 6400/6400 w 6.06e-07 and
`native_gdn_step` 792576/792576 err/tol 0.00154 — all PASS, 0 failed on every arm.**

**A FINDING THE MEASUREMENT PRODUCED, and it CHANGED THE SHIPPED KERNEL.**  The native `step` body uses ONE
32-LANE WARP per column; subgroup ops are banned here (Intel picks the SIMD width per kernel), so the warp has
two non-subgroup renderings.  The workgroup-per-column **barrier-tree** version was built and MEASURED against
the legacy kernel at this shape — **1.564x Arc, 8.328x Ryzen iGPU, 43.501x llvmpipe** (chain 1.471 / 8.078 /
41.597): 6144 workgroups each doing two 8-round barrier trees with adjacent invocations `h_v*S` floats apart is
a large REGRESSION.  So the SHIPPED port keeps the **coalesced one-thread-per-column serial** decomposition the
legacy port already uses (the CUDA's warp shape is a parallelism strategy, not the rule) and carries the native
arithmetic and its fused readout scale.  The variant is recorded in `NEXT.md` and `bench/README.md`, not shipped.

**CAPABILITY CONTRACT.**  `native_gdn_enabled()` gates NINE symbols: the six native kernels (now ALL ported) +
the three `fused_gdn_*` paths (also gated on `g_fused_gdn` + `native_bf16_projections`), which have no shader.
The backend answers **false**, and the gdn arm asserts the flag equals "every gated symbol has a built shader"
AND that the six ported shaders exist — a strict invariant (a "reachable" formulation was considered and
rejected: it would let the engine route to an unported symbol when a setting changed).

**THE MAP MOVES BY THREE ROWS:** `168 — 69 kernel, 61 host, 38 todo` → **`168 — 72 kernel, 61 host, 35 todo`**;
`check_port_map.py` passes (`106 shaders built, 87 claimed`) and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.  **Gate, after the commit: vega Arc 423/0/0, lvp 411/0/3, radeon-iGPU 414/0/2 (exit 0); box
`z820b` XTX 419/0/1 (the 1 is the pre-existing M8 skip), lvp 411/0/3, K620 414/0/2 — 0 failed on every arm**
(+8 verdicts on each); a re-run of the box's XTX cross-arm, after the documented intermittent `budget` flake,
read 419/0/1.  Full detail in `NEXT.md`'s top section.

---

## THE PERFORMANCE TIER'S GDN / DELTANET MIXER, first three native kernels — class B batch 2 (2026-10-05)

The first three native **GDN / DeltaNet mixer** kernels — the mixer is **36 of the model's 48 layers** — each
replacing a legacy kernel already ported and gated, each oracled against the engine's OWN native body
(`src/kernels/cuda/native_gdn_preprocess.cu`), each MEASURED against that legacy kernel at the same shape on the
same device, with **`native_gdn_enabled()` left answering FALSE** (the flag gates six more unported symbols) and
`case_native_capabilities` extended to enforce it.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc / box XTX) | falsified by |
|---|---|---|---|
| `native_gdn_conv_silu` (3 arms C=2560/24/300) | `native_gdn_preprocess.cu` `conv_silu`: the four-tap conv **PLUS the SiLU in ONE kernel**, BOTH outputs (raw + SiLU), the zero-bias fold, the slide | **12800/12800 w 1.87e-06** (Arc), 12800/12800 w **3.39e-05** (XTX), 120/120, 1500/1500; slid state **BIT-EXACT**; terms-derived bound | `native-gdn-conv-silu-drop-silu` → FAIL 11509/12800 |
| `native_gdn_l2_norm` (3 arms cols=128) | `native_gdn_preprocess.cu` `l2_norm`: f32 sums, `rsqrtf(partial/S + eps/S)` (eps on the MEAN), folded `scale_after=1/sqrt(S)` | **392/392 w 1.54e-07**, 2312/2312 w 2.16e-07, 648/648 w 1.17e-07 (Arc); XTX 1.17e-07/1.49e-07/1.74e-07; NaN row-guard | `native-gdn-l2-norm-drop-folded-scale` → FAIL 264/392 w 10.3 |
| `native_gdn_beta_gate` (48 heads) | `native_gdn_preprocess.cu` `beta_sigmoid`: `1/(1+expf(-x))` — the SAME expression as the legacy `sigmoid_f` | **48/48 w 1.42e-06** (both boxes); raw identity checked host-side to move | `native-gdn-beta-gate-sign-flip` → FAIL 0/48 w 7.2e+10 |
| `native capabilities: gdn flag` | `native_gdn_enabled()` == "every gated symbol has a built shader"; the 3 ported shaders exist | **4/4** — flag **FALSE** (6 unported gated shaders absent) | `native-caps-gdn-true` → FAIL 3/4 |

**THE MEASUREMENT (native/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
`native_gdn_conv_silu` **0.938** Arc / **0.909** iGPU / **0.694** XTX / 0.936 K620 (a win per dispatch even
though it does the SiLU and a second output too); against the legacy **chain** (`gdn_conv_step`+`silu_f32`, the
two dispatches the layer actually runs) **0.574 / 0.764 / 0.544 / 0.720** — a 1.3–2.1x win.
`native_gdn_l2_norm` **0.997 / 1.013 / 0.938 / 0.942** and `native_gdn_beta_gate` **1.011 / 1.045 / 1.068 /
0.865** are **WASHES: the same work per element as the legacy kernel, no algorithmic difference to win, and
reported as findings rather than tuned away.**  A native kernel is not required to be faster.

**CAPABILITY CONTRACT.**  `native_gdn_enabled()` gates NINE symbols: the three ported here + `native_gdn_gate`
/ `native_gdn_step` / `native_gdn_out_norm` and the three `fused_gdn_*` paths (the latter also gated on
`g_fused_gdn` + `native_bf16_projections`) — six with no shader.  The backend
(`vulkan/src/kernels/native_caps_vk.cpp`) answers **false**, and `case_native_capabilities` gains a gdn arm
asserting the flag equals "every gated symbol has a built shader" AND that the three ported shaders exist —
an invariant, not a hard-coded boolean.

**THE MAP MOVES BY THREE ROWS:** `168 — 66 kernel, 61 host, 41 todo` → **`168 — 69 kernel, 61 host, 38 todo`**;
`check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **Gate, after
the commit: vega Arc 415/0/0, lvp 403/0/3, radeon-iGPU 406/0/2 (exit 0); box `z820b` RADV XTX 411/0/1,
lvp 403/0/3, nvidia K620 406/0/2 — 0 failed on every arm on both boxes** (+8 verdicts on each).  One finding
the case produced: the first run read 12798/12800 on the iGPU at worst 1.09e-05 because two four-tap sums
CANCEL — fixed in the FIXTURE's bound (`gemv_bound`, the port's "bound a reduction by its TERMS" rule), not by
loosening a tolerance.  Full detail in `NEXT.md`'s top section.

## THE PERFORMANCE TIER'S FIRST FOUR KERNELS — class B, the NATIVE fast paths (2026-10-05)

The first increment of the PERFORMANCE tier: the **four class-B capability-gated native fast paths** are
PORTED, GATED, and **MEASURED against the legacy kernel each replaces at the same shape on the same device**,
with the Vulkan backend's **capability answers wired so the engine can take the native branch**.

| case | rule (the engine's OWN native body = the oracle) | measured (vega Arc) | falsified by |
|---|---|---|---|
| `native_rope_apply` (3 arms: hd=256/128 none, hd=256 YaRN×2) | `native_rope.cu` `apply<false>`: device f32 angle (`powf`/`rope_scaled_angle`), one thread per (row,PAIR), NEOX pairing, tail copied | **49152/49152 worst 2.61e-05**, 8192/8192 w 2.77e-05, 24576/24576 w 3.16e-05 (row-relative vs a host transcription, tol 3e-3; tail BIT-EXACT) | `native-rope-adjacent-pairing` → FAIL 37307/49152 w 1.96 |
| `native_router_top10` (5 arms) | `native_router.cu` `route`: softmax+sum in plain **f32**, top-10 with ties by lowest index, renormalise by the ten's sum (clamp 2⁻¹⁴) | **10/10, 80/80, 40/40, 40/40, 40/40** ids exact; weights worst 2.28e-07 (rel 1e-5) vs `router_top10_parity`'s double rule | `native-router-top10-tie-high-index` → FAIL 0/40 |
| `native_moe_combine` (4 shapes × shared on/off) | `native_moe.cu` `combine`: pure f32, first term a PRODUCT, shared added PLAIN | all 2560/2560 (and 2500, 37) within the term-relative bound; **904-1347/2560 differ from the double rule** — the measured arithmetic change | `native-moe-combine-drop-shared` → FAIL 0/2560 w 5.21e+05 |
| `native_qsa_rms_norm_weighted` (3 shapes × in-place/out-of-place) | `native_qsa.cu` `norm`: block-per-row, `scale*x*gamma`; in-place is the engine's call | **25608/25608 w 2.21e-07**, 1032/1032, 5128/5128 (double rule, tol 3e-3 + row-guard NaN) | `native-qsa-rms-norm-eps-on-sum` → FAIL 5128/25608 w 0.98 |
| `native capabilities (vulkan backend)` | the four capability answers + each ported symbol's `.spv` present | **4/4** answers exact | `native-caps-qsa-true` → FAIL 2/4 |

**THE MEASUREMENT (native/legacy, same shape, same device — < 1.0 is faster; full table in `bench/README.md`).**
`native_rope_apply` **0.301** on the Arc, **0.118** on the box XTX and 0.040 on the Ryzen iGPU (where the legacy
one-thread-per-row kernel launches 2 workgroups); `native_router_top10` **0.078** (Arc), **0.082** (XTX), 0.078
(iGPU); `native_moe_combine` **0.998** (a wash); `native_qsa_rms_norm_weighted` **1.007** on the Arc, 0.988 on
the XTX, and **1.117 on the iGPU / 1.795 on the K620**.  **The last two are NOT wins and are reported as
findings, not tuned away** — they are elementwise / same-shape kernels with no algorithmic difference, and on
the K620 the native block-per-row tree's shared-memory traffic actively hurts.

**THE CAPABILITY CONTRACT, wired in `vulkan/src/kernels/native_caps_vk.cpp` (new).**  The Vulkan backend answers
the checks itself (a Vulkan build compiles none of `src/kernels/cuda/native_*.cu`): `native_rope_enabled()`,
`native_router_enabled()`, `native_moe_combine_enabled()` → **true** (each gates only a ported, forward-path
symbol); **`native_qsa_enabled()` → false**, because that ONE flag ALSO gates the UNPORTED
`native_qsa_gate_apply` (layer.cpp:1010, the main QSA path, 12 of 48 layers) — so the port stays on the legacy
branch until that sibling lands.  **A symbol-at-a-time truth, not a blanket `true`.**  Check it with the gate's
`case_native_capabilities` (asserts the four answers AND the built shaders) or by reading that TU; the
`native_*_set_enabled` setters are no-ops here so a `--native` launch cannot select an unported symbol.

**THE MAP MOVES BY FOUR ROWS:** `168 — 62 kernel, 61 host, 45 todo` → **`168 — 66 kernel, 61 host, 41 todo`**;
`check_port_map.py` passes and `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  **Gate, after
the last commit: vega Arc 407/0/0, lvp 395/0/3, radeon-iGPU 397/1/2 (the known `budget` flake); box `z820b`
RADV XTX 403/0/1, lvp 395/0/3, nvidia K620 398/0/2 — 0 failed on every arm on both boxes.**  Two README fixes
came with it: `run_gate.sh`'s greedy barrier-count `sed` misread 11 barriers as 1 (now `grep -oE`), and the new
cases had to be APPENDED to `main()` because the gate's one shared RNG makes a new earlier case change a later
existing case's fixture.  Full detail in `NEXT.md`'s top section.

## THE PERFORMANCE TIER'S HARNESS LANDED (2026-10-05)

The port had **no throughput harness at all**, so no performance claim could be made or checked.  This
increment adds one — **`ports/vulkan/bench/`** (`vk_bench.cpp` + `run_bench.sh` + `README.md`) — and hands
back the baseline the native/fused kernels will be judged against.  It is **separate and opt-in**: it does not
change what `gates/run_gate.sh` does, and the numeric gate was re-run green on both boxes after the commit
(see `NEXT.md`'s top section for the totals line).

**Method (full text in `bench/README.md`):** each kernel is timed as a batch of `K` dispatches recorded into
one command buffer (compute→compute barrier between dispatches), warmed 2–3 replays, timed over `reps`
replays of the batch (median per-dispatch, min/max printed, `reps=9`, sampler 5).  Synchronisation is **wall
clock around the fence** — the device layer exposes no timestamp path — so the number includes one
submit+fence per batch and its floor is ~5–15 µs on the fast GPUs; only the **kernel dispatch** is timed
(inputs resident, no transfer in the loop).  `run_bench.sh` compiles the measured kernels from source and runs
the binary under **every ICD** that reports a device.

**Baseline, medians (reps=9; sampler 5):**

| kernel | vega Arc B70 | vega Ryzen iGPU | box XTX | box K620 | llvmpipe |
|---|---:|---:|---:|---:|---:|
| `gdn_step` (S=128 h_v=48) | 0.0469 ms / 50.3 GMAC/s | 0.5313 | 0.1050 / 22.5 GMAC/s | 0.4859 | 0.5015 |
| `iq2s_mmvq` n_out=2048 | 0.0464 ms / 112.9 GMAC/s | 0.9762 | 0.0198 / **264.3 GMAC/s** | 0.6887 | 6.2754 |
| `iq_dequant_f32` BF16 | 0.0130 ms | 0.0195 | 0.0140 | 0.0391 | 0.1444 |
| `quantize_q8_K` (65536) | 0.0875 ms | 0.2039 | 0.0732 | 0.2394 | 0.1938 |
| `sampler_kernel_f32` (vocab 248320) | **546.9 ms** | 299.6 | **202.0 ms** | 353.8 | 1164.7 |

The GDN chain's other five kernels are 0.0026–0.030 ms everywhere; the Q8_0/ Q8_1 quantisers and the other
IQ dequantiser arms are in `bench/README.md`.  **The baseline names a real problem:** the one-block sampler is
**202 ms/token on the XTX and 547 ms on the Arc**, because its top-k is `k` full-vocabulary rounds with an
inner loop over taken ids and history (~ `k · vocab · (k + history)`).  It is the engine's *non-default* path
(the default is the split sampler, `sampler_split.comp`, not yet measured here) and it is the performance
tier's first target.

**Distinguishability, measured:** cross-ICD ordering is physical on both boxes with ratios far outside noise
(box `iq2s_mmvq` XTX vs llvmpipe = **1201×**; vega Arc vs llvmpipe = **135×**); the 4×-work sizing arm scales
≈4× where the work exceeds the timer floor (Ryzen iGPU `iq2s` **3.90×**) and reads ~1× on the fastest GPUs
where it does not — stated as the honest limit of a fence-clock timer, not hidden.

**One shared-file change, and why it is safe:** `harness/vk_compute.cpp`'s `Ctx::set_alloc` now treats
`VK_ERROR_FRAGMENTED_POOL` as pool exhaustion too (Mesa 26.0.8's code for it, vs Mesa 25.2.8's
`OUT_OF_POOL_MEMORY`), so the grow-on-demand path that the header promises fires on both.  The numeric gate
never fills a descriptor pool, so no gate verdict depends on it; the gate was re-run to prove that.

## CLASS A IS CLOSED, and M-A IS RE-DEFINED (2026-10-05)

The last three class-A forward-path kernels are LANDED and gated: **`indexer_key_append`** (the QSA indexer
pair's legacy member; contract `native_qsa_indexer_enabled() == false`), **`gr_write`** (the hyper-connection
write — reached on BOTH fused/unfused branches, so no dodge), and **`gr_read`** (the unfused five-stage
hyper-connection read: `gr_norm`/`gr_down`/`gr_gate`/`gr_mean`/`gr_inject`).  The `gr_read` / `fused_gr_read`
question left open by the triage is SETTLED — both were mis-kinded `host` (each launches kernels), so the honest
device-op count is **54, not 52** — and the GR pair is closed by CONTRACT (`gr_set_native_mmvf(false)` +
`layer_set_fused_gr(false)`, which selects the unfused `gr_read`+`gr_write` and removes `fused_gr_read`).

| case | rule | oracle | measured (vega Arc) | falsified by |
|---|---|---|---|---|
| `indexer_key_append` (2 arms 128/4/64, 32/4/8) | raw tail; on completion `pooled[b]=rope(rms_norm(mean), pos_base+b*r)`; spare `rope(rms_norm(raw[0]),0)` | `qsa_parity.cpp` indexer reference, double | **384/384 err/tol 4.94e-02** (bound 1.0), spare key worst 1.02e-07 (tol 1e-5) | `indexer-key-append-rotate-last` → FAIL 274/384 w 9.46e+05 |
| `gr_write` (3 arms 2560/4, 64/3, 16/2) | `out[i]=R[i]+block_out[d]·2·sigmoid(inject[c]/hc)`, in place; zero inject ⇒ `w=1` EXACTLY | `gr_parity.cpp` §6 + the rule in double | **20480/20480 err/tol 2.4e-01**, zero-inject property EXACT | `gr-write-drop-two-centring` → FAIL w 6.68e+06 |
| `gr_read` (3 arms 2560/4/320, 64/2/8, 16/3/4) | 5 stages: per-stream RMSNorm → down+silu(`/hc` inside) → gate(sigmoid) → mean over streams → inject | `gr_parity.cpp::reference`, double, BF16 activation | **23364/23364 err/tol 3.0e-01**; xq==bf16(xn) exact | `gr-read-mean-vs-sum` → FAIL 20804/23364 w 2.99e+03 |

**HONEST PARITY GAPS (measured):** `indexer_key_append`'s CUDA reduces the sum of squares in double; the target
has no `shaderFloat64`, so the port accumulates in f32 and the case MEASURES the gap (worst err/tol 4.94e-02 on
the Arc / 7.87e-02 on lvp+radeon) — the spare key is therefore NOT bit-exact here, unlike the CUDA's.
`gr_read`'s stage oracles are fed the DEVICE's own input for that stage and bounded by the stage's TERMS
(`rel·|want| + 16·2^-24·Σ|terms|`), because comparing a bf16-ROUNDED `lo` against an unrounded double oracle
puts the odd element a whole bf16 ulp away and moved a near-zero `mixed` by 9.7 RELATIVE on RADV.

**THE MILESTONE, RE-DEFINED.**  `todo = 0` is neither achievable nor meaningful: the map covers every
kernels-namespace symbol the decode path reaches, including the `native_*` siblings and the verify/MTP/tooling
helpers.  **M-A (re-defined):** every symbol the forward path reaches ON THE BRANCH THE CAPABILITY CONTRACT
SELECTS has a shader and a gated case.  **Class-A remaining: 0.  MET: YES**, with two stated soft edges — the
shipped `--spec 4` loop's MTP symbols (class D by the brief's definition: **a judgement, not a measurement**),
and the `gr_read`-vs-`fused_gr_read` branch choice.  Full statement: `plan/DECODE-PATH-TRIAGE.md` → "THE
RE-DEFINED MILESTONE M-A"; the batch's detail: `NEXT.md`'s top section.

**The map moves by four rows:** `168 — 59 kernel, 63 host, 46 todo` → **`168 — 62 kernel, 61 host, 45 todo`**
(the 45 = `11 capability-off + 4 B + 8 C + 22 D`); `check_port_map.py` passes, `make_port_map.py` regenerates it
byte-identically.  **Gate: vega Arc 384/0/0, lvp 372/0/3, radeon 375/0/2, exit 0.  Box `z820b`: radeon (XTX)
380/0/1, lvp 372/0/3, nvidia 375/0/2 — 0 failed on every arm; exit 1 only for the pre-existing M8 skip.**  (A
later full-gate run on vega read its radeon-iGPU arm 374/1/2 — the documented intermittent
`budget: independent requery agrees` flake, not this change.)

## THE GDN MIXER CHAIN IS COMPLETE, plus the first QSA gate member — class A #4-6 (2026-10-05)

This increment lands the next three class-A forward-path kernels: **`gdn_step`** (the delta-rule state update),
**`gdn_out_norm`** (the closing norm) — together **completing the GDN / DeltaNet mixer chain** for the 36 GDN
layers — and **`qsa_gate_apply_f32`**, the legacy member of the QSA gate pair. All under the branch policy
`native_gdn_enabled() == false` / `native_qsa_enabled() == false` (see `NEXT.md`'s top section for the full
reasoning and the per-symbol completeness check).

| case | rule | oracle | measured (vega Arc) | falsified by |
|---|---|---|---|---|
| `gdn_step` (3 arms 128/16/48, 16/4/8, 8/2/4) | `dec=exp(gate)`; decay, contract vs k, `d=(v−sk)·β`, rank-1 update, readout vs q; `src=h%h_k` | `gdn_parity.cpp` §1 `ref_step`, transcribed to the DEVICE layout (S,h_v,S) in double | **792576/792576 err/tol 0.00733**, 2176/2176 w 0.00112, 288/288 w 0.000345 | `gdn-step-head-pairing` → FAIL 66887/792576 w 6.13e+04 |
| `gdn_out_norm` (3 arms 48/128, 4/16, 3/8) | `y = rms_norm(o)·ssm_norm·sigmoid(z)`, ONE RMS per head (`sum/S + eps`) | `gdn_parity.cpp` §4, double | **6400/6400 w 8.2e-07**, 96/96 w 2.41e-07, 40/40 w 2.18e-07 | `gdn-out-norm-eps-on-sum` → FAIL 2321/6400 w 0.912 |
| `qsa_gate_apply_f32` (3 arms 24/256, 4/12, 2/8) | `attn · sigmoid(q_full[h·2·head_dim + head_dim + d])` (SECOND half) | `qsa_parity.cpp` `ref_gate`, double | **6152/6152 w 7.88e-07**, 56/56 w 7.64e-07, 24/24 w 7.42e-07 | `qsa-gate-first-half` → FAIL 8/6152 w 4.18e+10 |

Each case pins its rule's named traps host-side (modulo-vs-interleave head pairing and decay-order for
`gdn_step`; sigmoid-vs-SiLU and eps-on-the-mean for `gdn_out_norm`; second-half and sigmoid-vs-SiLU for
`qsa_gate_apply_f32`), and a NaN-padded tail + surplus dispatched groups make a missing guard DETECTED.
**HONEST PARITY GAPS (measured, not hidden):** the target device has no `shaderFloat64`, so `gdn_out_norm`
sums `o²` in **f32** where the CUDA sums in double (measured gap 8.2e-07), and `qsa_gate_apply_f32` forms the
product and the sigmoid in **f32** where the CUDA uses double (gap 7.88e-07). `gdn_step`'s CUDA already
accumulates in f32 (as the state is f32); its gap is against the double oracle only.

**The map dropped by three:** `PORT-MAP.tsv` `168 — 56 kernel, 63 host, 49 todo` → **`168 — 59 kernel, 63 host,
46 todo`**; `check_port_map.py` passes and `make_port_map.py` regenerates it byte-identically.
**Measured: vega Arc 376/0/0, llvmpipe 364/0/3, radeon-iGPU 367/0/2, exit 0; box XTX 372/0/1, lvp 364/0/3,
nvidia K620 367/0/2, exit 1 (the M8 skip)** — +9 verdicts on every arm. Full detail in `NEXT.md`'s top section.

## THE GDN (DeltaNet) MIXER'S FIRST THREE KERNELS LANDED — class A of the decode-path triage (2026-10-05)

The corrected map's **19 class-A forward-path holes** are dominated by the **GDN / DeltaNet mixer**, which runs
on 36 of the model's 48 layers and which the port's plan never enumerated. This increment lands the FIRST THREE
kernels of the mixer's chain, in the order `gdn_layer` runs them: `gdn_conv_step` (the four-tap conv),
`gdn_l2_norm` (the L2 norm, eps on the SQUARED NORM) and `gdn_beta_gate` (`beta = sigmoid(beta)`).

**THE BRANCH POLICY, and the contract on the backend.** The class-A GDN symbols are native/legacy PAIRS
(`if (native_gdn_enabled()) native_gdn_X else gdn_X`); the port implements the **LEGACY** member of each and
requires the Vulkan backend to answer **`native_gdn_enabled() == false`**. That one answer makes ONE
implementation per pair sufficient: the layer takes every `else`, and it also removes the three fused GDN paths
(`fused_gdn_conv_l2` / `fused_gdn_ab` / `fused_gdn_step_norm`, gated on `native_gdn_enabled()`) from the path
entirely. **Re-checked the parent's count:** the 19 symbols are **15 native/legacy pair members (11 GDN + 4
QSA) + 3 fused + 1 unconditional (`gr_write`)** = 19; collapsed under the policy they need **9 kernel
implementations** (6 GDN legacy — one of them, `gdn_gate`, already ported —, 2 QSA, `gr_write`), so **8 are
still to write**. The `layer_verify_compatible()` contract: forcing these flags off makes the P6 verifier refuse
to init (speculative verification off, class D); a `--spec 0` run is unaffected.

**The three kernels and their proofs** (shaders `gdn_conv_step.comp`, `gdn_l2_norm.comp`, `gdn_beta_gate.comp`;
cases `case_gdn_conv_step` / `case_gdn_l2_norm` / `case_gdn_beta_gate`; oracles transcribed from the engine's
own rule — `src/kernels/cuda/gdn.cu` plus `src/kernels/gdn_parity.cpp` — not invented):

| case | rule | measured | falsified by |
|---|---|---|---|
| `gdn_conv_step` (3 arms) | `out[c]=Σ conv_state·kW + x·kW`, state slides oldest→newest | **96/96 w 7.09e-08**, 20/20 w 0, 1200/1200 w 3.62e-06 | `gdn-conv-tap-order` → FAIL 72/96 w 2.28 |
| `gdn_l2_norm` (3 arms) | `x *= 1/sqrt(Σx² + eps)` (eps on the **squared norm**) | **392/392 w 0**, 2312/2312 w 1.66e-07, 648/648 w 1.56e-07 | `gdn-l2-norm-eps-on-mean` → FAIL 265/392 w 0.446 |
| `gdn_beta_gate` | `beta = 1/(1+exp(-beta))` in place | **48/48 w 1.42e-06** | `gdn-beta-gate-drop-sigmoid` → FAIL 0/48 w 1.8e+12 |

The conv case compares the slid state **bit for bit** (the kernel only moves/appends values) and its fixture
labels the state rows so the slide direction is observable; the l2 case's oracle is double and the shader sums
in **f32** (the target has no shaderFloat64), so the printed `worst` IS the measured gap — an honest limit, not a
bit-exactness claim.

**The map dropped by three:** `PORT-MAP.tsv` `168 — 53 kernel, 63 host, 52 todo` → **`168 — 56 kernel, 63 host,
49 todo`**; `check_port_map.py` passes and `make_port_map.py` regenerates it byte-identically.
**Measured: vega Arc 367/0/0, llvmpipe 355/0/3, radeon-iGPU 357/1/2** (+7 verdicts on each; the 1 is the known
intermittent `budget: independent requery` flake). Full detail in `NEXT.md`'s top section.

## THE PORT MAP'S BLIND SPOT IS CLOSED — and it was 91 symbols, not 4 (2026-10-05)

`tools/check_port_map.py` keyed on the `kernels::` qualifier, so kernels-namespace symbols that
`src/core/` calls BARE (through `using namespace strata::kernels;`) were invisible — the map could
read `todo 0` while such a symbol was unported.  The checker now discovers both forms, via one shared
`tools/port_map_lib.py` (bare names come from the engine's own declarations in
`include/strata/kernels/**`, attributed only to sources that have the namespace in scope).  **The
map moves from `77 symbols — 28 kernel, 49 host, 0 todo` to `168 symbols — 53 kernel, 63 host,
52 todo`.**  The four known symbols (`coupled_draft_sample`, `coupled_draft_stage` and their shaders
`coupled_penalize`/`coupled_sample`) are classified — their shaders are now claimed, so the unclaimed
shader count falls 37 → 19 — but the same scan surfaces **52 GPU symbols this port has NOT done**
(the GDN family and their `native_*`/`fused_*` siblings, `native_rope_apply`, `native_router_top10`,
`native_moe_combine`, `native_qsa_*`, `bf16_gemv*`, the QSA prompt/indexer path, `gr_write`,
`fused_gr_read_multi`, `moe_group_resident`, and the verify/P6 device helpers).  **So M-A's `todo = 0`
was measured on a 77-symbol map, not on the decode path**; the dated "port map ... 0 todo" lines
lower down are records of the map as it then stood, not the current count.  Full detail, the rule and
the falsification (the HEAD checker passes the same file the new one fails) are in `NEXT.md`'s top
section.  **M-A is therefore NOT closed**, and the corrected map's 52 `todo` rows are triaged per symbol in
`plan/DECODE-PATH-TRIAGE.md`: **19 are genuine forward-path holes with no ported fallback** (the GDN / DeltaNet
mixer for 36 of the 48 layers, the QSA gate and indexer for the 12 QSA layers, and `gr_write`), 4 are
capability-gated with a ported fallback, 7 are the non-native configuration the shipped `--native` launch does
not select, and 22 are the P6-verifier / speculative-drafter / tooling path.

## The engine integration has STARTED: increment I1 (device layer + arena + the first entry point) — DONE AND VERIFIED 2026-10-05

`ports/vulkan/plan/BACKEND-INTEGRATION.md` §3's **I1**, the first increment that BUILDS the engine.  It adopts
`ports/vulkan/harness/vk_compute.*` as `vulkan/src/device/` (namespace renamed `portvk` -> `strata::vulkan` so one
TU can hold the port's device layer and the engine's without an ODR clash), builds **THE ARENA** (one device-local
buffer carved by byte offsets; a device pointer is `kArenaBase + offset` in a synthetic address space, which is
what an unmappable VRAM arena can offer) and the **pointer -> buffer resolution** `arena_resolve` (base subtract,
live-range check, `view(arena, offset)` — a pointer outside the arena is refused, not bound), and finishes
`strata::vulkan::fwht256` so increment 0's entry point RUNS (pipeline cache keyed as the engine dispatches: 2
storage buffers + a 4-byte push constant).  **How the engine's sources join the build:** `vulkan/CMakeLists.txt`
gained `strata_vulkan_resolve()`, mirroring `../sycl/CMakeLists.txt`'s `strata_resolve` — a source comes from
`vulkan/` when a migrated copy is there, else from the engine tree.  I1 needs no engine-root source (the wrapper is
a header) and the top-level `return()` stands; the first engine-root sources are I2's.  The target that proves the
wiring, `strata_vk_entry_smoke`, builds only under `-DSTRATA_ENABLE_VULKAN=ON` (measured **configure 0.14 s, build
1.37 s**, 9 objects) and runs PASS on the Arc.  A full engine build was not needed and was not done — the CUDA
configuration never compiles `vulkan/`.

**The proof, and it is the first WRAPPER this port has gated rather than a shader.**  `case_fwht256_entry` runs the
same input through the port's shader path and through the ENGINE wrapper (`strata::kernels::fwht256_cuda`) on the
backend's arena, and compares as 32-bit words: **PASS 1024/1024, worst 0**.  Falsified by
`gates/inject-verify.sh fwht-entry-wrong-view-offset` (shift the view the DISPATCH binds by one float) ->
`FAIL ... 128/ 1024 worst 896`.  Recorded honestly: a UNIFORM shift inside `arena_resolve` does NOT bite (transfer
and dispatch move together and cancel) — the dispatch's offset is the one that decides.

**Gate totals: vega Arc 360/0/0, llvmpipe 348/0/3, radeon-iGPU 351/0/2, exit 0, wall 56.03 s.  Box `z820b`
post-commit: default XTX 356/0/1 (the 1 is the pre-existing M8 `prefill split` skip), lvp 348/0/3, nvidia
351/0/2, exit 1 (the skip).  The first box run read 355/1/1: `budget: independent requery agrees` failed on the
XTX — the KNOWN intermittent flake (that same run's named `radeon_icd` arm read 356/0/1 clean, and the re-run
cleared it), so 0 failed is the result.  New case 1024/1024 bitwise on every box arm.  Port map **77
decode-path symbols — 28 kernel, 49 host, 0 todo** - that `0 todo` was read on the qualifier-only map; CORRECTED
2026-10-05 to **168 — 53 kernel, 63 host, 52 todo**, and **M-A is not closed** (see the top section).**

## Done and verified in this session

Everything below is backed by a command that exits non-zero on failure. Re-run it with:

    bash ports/vulkan/gates/run_gate.sh          # compiles the shaders from source, validates the SPIR-V,
                                                 # checks each shader's declared local size, then runs the gate

**Result: the gate prints its own totals and those are the authority. On 2026-10-04, after the Radeon RX 7900 XTX
was swapped for an Arc Pro B70 and after stages 3, 4, the prefill GEMM, the quantised multi-token arms, the
short-step decode attention, the f16 KV gather, the QSA selection, the f16 KV append, the Q4_0 KV path (with its
Walsh-Hadamard rotation), the hybrid K8V4 mode, descriptor OFFSETS and the first SAMPLER kernel landed, the box's GPU
run was **284 passed / 0 failed / 0 skipped** on the Intel ICD (`Intel(R) Graphics (BMG G31)`, Mesa 25.2.8 / ANV,
Vulkan 1.4.318, subgroup size 32) - 272 / 0 / 3 on llvmpipe and 275 / 0 / 2 on the radeon ICD, which now picks the AMD
iGPU because the discrete card is gone (both
skip cooperative matrix, whose driver does not advertise the extension, and the prefill SPLIT, which needs the M8
tile).  **The Arc has no skips at all:** the last one (`gemm_coopmat`) was the port
misreading the device - BMG's matrix config is M8 N16 K16, not the M16 the criterion demanded - and since then the
matrix path RUNS on XMX, including the prefill GEMM.  The count now stands at **314 / 0 / 0 on the Intel ICD**
(302 / 0 / 3 on llvmpipe, 305 / 0 / 2 on the radeon iGPU) after the M-A 4/10 + 5/10 batch.  All three available
implementations are exercised by `run_gate.sh` (it once stopped at the Intel skip, so the cross-implementation arm
never ran on this box after the swap - `NEXT.md`), and the radeon iGPU's `budget: independent requery agrees` is
INTERMITTENT rather than deterministic: the driver's free figure drifts ~2.8 MB against the 1.7 MB tolerance (1
failure in 3 consecutive runs of one binary on one device; an older commit reproduces it).  Re-run it and record it;
do not chase it.  **75 kernels, 20 shared includes**, one generated table file (`harness/iq_grids.hpp`,
holding the IQ1_S, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_XXS and IQ3_S grids). TWO RECONCILIATION NOTES, both verified against a full run:
the ``PASS`` LINE COUNT IS ONE LESS than the case total, because the transcendental probe prints `INFO` while
counting as a pass; and one line ("gemm shape contract") covers seven cases. Neither is a discrepancy - but if the
numbers ever stop reconciling this way, something is wrong with the harness rather than with a kernel.

**M-A has started: the decode path's last kernels, in the order the decode path needs them (2026-10-04).**  The
ten `todo` symbols were ordered by (a) where the decode step meets them and (b) what each depends on.  The three
with NO unported precondition come first, in the order a decode step reaches them - `cvec_apply` (every layer,
`block_layer_post`), `gather_rows` (the MTP draft head's token subset), `scatter_rows_f32` (the peer experts'
write-back) - and **all three - `cvec_apply`, `gather_rows` and `scatter_rows_f32` - are landed** (3 of the ten).
`cvec_apply.comp` reproduces the
engine's OWN test
(`src/kernels/cvec_parity.cpp`) arm for arm: the projections at the engine's 1e-4 against a double oracle, the
add and BOTH untouched cases bitwise, and the pending write bitwise by construction (`inj == 0` makes
`2 sigmoid(0/hc) == 1.0` exactly, so the fold is `fl(bo + h)`).  Its SPIR-V carries the barrier tree's ops, so
`run_gate.sh`'s census list gained `cvec_apply`.  Falsified by `gates/inject-verify.sh cvec-apply-drop-scale`
(drop the per-layer factor `s`) -> `FAIL  cvec_apply: project removes s(h.v)v  12/14  worst 0.844`.  **`gather_rows`**
(the MTP draft head's token subset, `src/kernels/cuda/verify_kernels.cu:394`) is the second: a byte-level row
gather whose CUDA element WIDTH (`uint4`/`uint32`/`uint8` by `row_bytes % 16 / % 4`) is a performance choice, not
the rule - its fixture is a DERANGEMENT so an identity gather fails on every row, and both the 16-byte-aligned and
the unaligned `row_bytes` are gated.  Falsified by `gates/inject-verify.sh gather-rows-identity`
(`src[ids[r]*row_bytes + o]` -> `src[r*row_bytes + o]`) -> `FAIL  gather_rows: 16-byte-aligned rows  64/576`.
**`scatter_rows_f32`** (the peer experts' write-back, `src/kernels/cuda/elementwise.cu:263`) is the third: `r` is a
POSITION and `rows[r]` is the DESTINATION, an unnamed destination row must survive a sentinel, and the CUDA's
`width % 4 == 0` + 16-byte-alignment precondition is its `float4` cast rather than the rule - the case GATES that
by running `width = 6`, where the CUDA would refuse.  Falsified by `gates/inject-verify.sh scatter-rows-identity`
-> `FAIL  scatter_rows_f32: permutation ... 511/3072`.  **Arc 284/0/0** (272 + the batch's twelve verdicts, and
`run_gate.sh` **exit 0**); box `radeon_icd 280/0/1`, lvp 272/0/3, nvidia 275/0/2.  **M-A 4/10 and 5/10 landed the
STANDALONE IQ/BF16 dequantiser and the token-embedding gather on it** (`iq_dequant_f32`, `iq_embed_rows`;
`shaders/common/iq_dequant.glsl`, `shaders/iq_dequant_f32.comp`, `shaders/iq_embed_rows.comp`).  The decoder is the
engine's own `dq_dispatch<float>` - the port keeps its `tid` 0..31 thread mapping rather than re-deriving a block
layout - and the case reproduces each `dq_*` body on the host as the oracle, ONE ARM PER FORMAT over the **14
formats** the port's generated grids cover (BF16, IQ4_NL, IQ4_XS, Q8_0, Q5_0, Q5_1, Q2_0, Q4_K, Q5_K, Q3_K,
IQ3_XXS, IQ3_S, IQ2_S, IQ1_M); **IQ2_XXS (16) and IQ2_XS (17) stay `todo`** - their grids are not in
`harness/iq_grids.hpp` and neither shader claims them.  `iq_embed_rows` gates the row stride and a DERANGEMENT
token list, so an identity gather fails every token.  Falsified by `gates/inject-verify.sh
iq-dequant-iq1m-grid-high` -> `FAIL  iq_dequant_f32: IQ1_M  310/768  worst 2.34e+05` and `iq-embed-rows-identity`
-> `FAIL  iq_embed_rows: BF16 ... 0/3072  worst 6.58e+04`; the first targets a shared INCLUDE, so the script now
compiles the shader that INCLUDES a changed `common/` file rather than reporting `DID NOT COMPILE` on a file with
no `#version`.  **+30 green verdicts on every implementation, bit-exact against the oracle**: vega **Arc 314/0/0**
(`run_gate.sh` exit 0), llvmpipe 302/0/3, radeon-iGPU 305/0/2; box `radeon_icd` (7900 XTX) **310/0/1**, lvp
302/0/3, nvidia 305/0/2.  **M-A is 5 of the ten**; the five `todo` symbols are `native_q5_k_f32`, `moe_grouped_s2`,
`moe_hit_add`, `moe_hit_select`, `moe_hit_grouped_s2`.

**M-A 6/10 landed the NATIVE HEAD's Q5_K matvec** (`native_q5_k_f32.comp`, from `native_q5_k_mmvq_kernel` /
`q5_q8_dot`, `src/kernels/cuda/native_mmvq.cu`).  It is the one dot in the port whose scale and min are PACKED:
Q5_K's 12-byte `scales` carries six 6-bit scales and six 6-bit mins, and an `hi` mask decides which six bits are
which - the port transcribes the source's masks and shifts verbatim, because a "tidier" read is a plausible wrong
number.  The case carries three arms (`n_in=2560` 1280 parts, `n_in=256` 16 parts, `n_in=10240` x2 columns)
against a host double reference, and the increment found THREE silent defects in the transcription, all recorded
in `NEXT.md`: the activation's `u` was read out of the weight buffer (the `f16_at` class), the per-block
activation group must be `kbx*8` not the row base, and `unpackHalf2x16` fed only the low 16 bits silently zeroed
`min` (a 1-13% per-row error - every value finite and close).  Falsified by `gates/inject-verify.sh
native-q5k-aux-half` (drop the packed-scale half switch) -> `FAIL  native_q5_k_f32 ... 0/8  worst 5.62e+04`.
**+3 verdicts on every implementation**: vega **Arc 317/0/0** (`run_gate.sh` exit 0), llvmpipe 305/0/3,
radeon-iGPU 307/1/2 (the intermittent budget-requery drift); box `radeon_icd` (7900 XTX) **313/0/1**, lvp
305/0/3, nvidia 308/0/2.  **M-A is now 6 of the ten**; the four `todo` symbols are `moe_grouped_s2`,
`moe_hit_add`, `moe_hit_select`, `moe_hit_grouped_s2`.

**M-A 7/10 + 8/10 landed the MoE hit selection and the per-hit S2 expert entry** (one commit, as 4/10 + 5/10
were).  `moe_hit_select.comp` (`hit_select_kernel`, `s2_expert_grouped.cu:623`, from `src/core/session.cpp:866`)
is the token graph's first step: only a resident expert is a hit, the hits compact in ascending routing order, and
`dst[at]` is the ROUTING POSITION (`lane`), not the slot - the same two-roles confusion `gather_rows` carries.  A
ballot is a subgroup op, so the port writes the same exact compaction SERIALLY from one lane, and the shader
therefore carries no barrier (it is deliberately NOT on the census whitelist).  Four arms (decode shape with
scrambled slots, `k=32`, `k=1`, a fully non-resident row), with sentinels past `count` and the `res` buffer padded
past `n_expert` so an unguarded read fails.  Falsified by `gates/inject-verify.sh moe-hit-select-residency` ->
`FAIL  moe_hit_select ... 21/32`.  **`moe_hit_grouped_s2` is a COMPOSITION** (`gu -> swiglu -> q8_0 -> down`,
`s2_expert_grouped.cu:579`), so it needed NO new shader: its increment is the case that gates the WIRING between
the four already-ported kernels, with an oracle that is independent of the device's quantiser and down projection
(the device's intermediate must DECODE to its own post-SwiGLU floats at the same position, and the up rows are
checked against the blob).  Falsified by `gates/inject-verify.sh moe-hit-grouped-s2-hit0-intermediate` ->
`FAIL  moe_hit_grouped_s2 ... 473/867  worst 1.3e+06`.  Its fixture first made the intermediate's `d16` inf (the
gu arm's gate scales are `*1000`), which turned EVERY down row into NaN that the oracle also computed - the case
counts a NaN output as a failure, so it did not pass over garbage; the chain now carries its own modest scales.
**vega: Arc 322/0/0** (`run_gate.sh` exit 0), llvmpipe 310/0/3, radeon-iGPU 313/0/2; box `radeon_icd` (7900 XTX)
**318/0/1**, lvp 310/0/3, nvidia 313/0/2.  **M-A is 8 of the ten**; the two `todo` symbols are `moe_grouped_s2`
and `moe_hit_add`.  The port map reads **77 symbols - 26 kernel, 49 host, 2 todo**.

**M-A's LAST TWO *VISIBLE* SYMBOLS landed together - but M-A is NOT closed (CORRECTED 2026-10-05, `7c317c4`).**
The "CLOSED" reading below was taken on the qualifier-only 77-symbol map, which could not see the bare-name call
sites; the corrected map reads **168 symbols - 53 kernel, 63 host, 52 todo**, of which **19 are class-A
forward-path holes** (`plan/DECODE-PATH-TRIAGE.md`).  The two symbols that landed here did land, and the increment
is real - it is the "*last two visible*" that is now the historical claim.  What follows is that record.
**the decode path's last two symbols landed together (9/10 + 10/10, one commit, sharing
`harness/vk_gate.cpp` - the `4/10 + 5/10` precedent).**  **`moe_grouped_s2`** turns out NOT to be a reuse of the
per-hit chain: it is the same four-launch composition, but its gu and down halves are the GROUPED kernels, and the
per-hit `s2expert_gu`/`s2expert_down` cannot express a per-GROUP blob or a per-ENTRY activation row - so the
increment adds `s2expert_gu_grouped.comp` and `s2expert_down_grouped.comp` (one weights buffer + a per-group byte
offset table replacing the source's `grp_ptr` device pointers) and reuses `s2expert_swiglu` and `quantize_q8_0`.
The case gates the WIRING with an oracle independent of the device's quantiser and down step (up rows against the
group's blob at the ENTRY's token; the device's intermediate decoded against the device's own post-SwiGLU floats;
down rows from the device's intermediate at the scrambled `ent_dst[e]`), over three grid.y arms including the
device-count stride and an EMPTY group; falsified by `moe-grouped-s2-entry-token` -> `FAIL ... 5270/5638 worst
3.78e+05`.  **`moe_hit_add`** (`add_hits_kernel`, `s2_expert_grouped.cu:666`) is `parts[dst[h]] += hit_out[dst[h]]`
for every LIVE hit - `dst[h]` is the ROUTING POSITION and the operator is `+=`; four arms (count<cap, count==cap,
`n_embd=37`, count 0), falsified by `moe-hit-add-accumulate` (`+=` -> `=`) -> `FAIL ... 2624/10304 worst 0`.  The
count-0 arm itself was the one defect found (a liveness guard that assumed something MOVED, which that contract
inverts); the guard now reads `moved == 0` for count 0 and the arm was not loosened.  The port map now reads
**77 symbols - 28 kernel, 49 host, 0 todo** - that `0 todo` was the qualifier-only map; CORRECTED 2026-10-05 to
**168 - 53 kernel, 63 host, 52 todo**, and the drifted generator `tools/make_port_map.py` was fixed to regenerate
it exactly (see the top section and `plan/DECODE-PATH-TRIAGE.md`).  **M-A is not closed.**  **Measured: vega Arc 329/0/0** (`run_gate.sh` exit 0), intel_icd 329/0/0, llvmpipe 317/0/3,
radeon-iGPU 320/0/2; box `radeon_icd` (7900 XTX) **325/0/1** (the pre-existing M8 `prefill split` skip), lvp
317/0/3, nvidia 320/0/2.  The two new symbols add 7 verdicts (322 -> 329 on the Arc).

**The standalone dequantiser's LAST TWO FORMATS landed - IQ2_XXS (ggml type 16) and IQ2_XS (17) (2026-10-05).**
Those two `is_iq` types were the ones `iq_dequant_f32` refused BY NAME because their `uint64` grids were not in the
generated table; both now decode.  `tools/gen-iq-tables.py` derives `iq2xxs_grid` (256 points) and `iq2xs_grid`
(512) from the engine's own `ggml-common.h` as low/high `uint32` pairs (so no `shaderInt64`), `harness/iq_grids.hpp`
carries them, `common/iq_dequant.glsl` gains the two `dq_*` bodies, and the harness gains ONE BIT-EXACT ARM PER
FORMAT against the engine's own decoder (768/768 bit-exact; the same decode is 1536/1536 through `iq_embed_rows`).
Falsified twice (`iq-dequant-iq2xxs-grid-index` -> `FAIL IQ2_XXS 300/768 worst 3.79e+05`; `iq-dequant-iq2xs-grid-high`
-> `FAIL IQ2_XS 530/768 worst 3.82e+05`).  A finding: the shared fixture left IQ2_XS's 9th grid bit
DETERMINISTICALLY clear, so the upper half of the 512-point grid was never indexed and the first injection was
INVISIBLE - the fixture now splits that bit across both halves.  **+4 verdicts on every implementation:** vega
**Arc 333/0/0**, llvmpipe 321/0/3, radeon-iGPU 323/0/2 (a clean run; one run had the documented intermittent
`budget: independent requery` failure); box `radeon_icd` (7900 XTX) **329/0/1**, lvp 321/0/3, nvidia 324/0/2.  The
port map still reads **77 symbols - 28 kernel, 49 host, 0 todo**, and `make_port_map.py` regenerates `PORT-MAP.tsv`
byte-identically.

**THE SPLIT SAMPLER landed - the engine's DEFAULT sampled path (2026-10-05).**  `sampler_split.comp` is
`sampler_split_part_kernel` + `sampler_split_merge_kernel` (`src/kernels/cuda/sampler.cu`) with the `sampled_tail_warp`
tail: a row is cut into 4096-logit partitions, each keeps its own top_k, and the ordered lists are merged (exact).
This is the path `sample_tokens` takes by default for a sampled request; the one-block `sampler_kernel` is the
`STRATA_OLD_SAMPLER=1` reference.  The tail now lives ONCE in `common/sampler_tail.glsl`, in both arithmetic
variants (`sampler_tail_f64`, and the portable `sampler_tail_f32` for the device the f64 one cannot run on), and the
partition+merge selection in `common/sampler_select.glsl` - so the split and the coupled merge share them and the
parity is by construction.  Falsified by `gates/inject-verify.sh sampler-split-merge-drop-parts` -> `FAIL
sampler_split: three partitions ... 9/16`.  **+7 verdicts on every implementation, and the parity arm reads 60/60
(the split's token EQUALS `sampler_kernel`'s on every seed of every arm, the engine's own bit-for-bit claim): vega
Arc 340/0/0** (`run_gate.sh` exit 0), llvmpipe 328/0/3, radeon-iGPU 331/0/2.  The increment also FOUND a real class:
the merge's first version was a forward IN-PLACE pass, whose write index runs ahead of its read index once a
partition entry is taken - it read `dev 4200 want 100` on the top_p arm, and only the multi-partition arms could see
it.

**THE COUPLED DRAFT PATH (speculative decoding) landed (2026-10-05).**  `coupled_penalize.comp` +
`coupled_sample.comp` (`coupled_penalize_kernel` / `coupled_merge_kernel`, `src/kernels/cuda/sampler.cu`; host side
`include/strata/core/coupled_draft.hpp`).  In coupled mode the MTP draft layer samples its draft with the target's
own chain and the target's Philox draw; four rules are arms - the counter is `cell + 1` (NOT `cell`), the penalty
window is the ring's `[cap+j-h, cap+j)`, a history id maps through `id_to_sub`, and the pick maps through `sub_to_id`
and is appended at `ring[cap+j]`.  Falsified by `gates/inject-verify.sh coupled-draft-counter-off-by-one` -> `FAIL
coupled_draft: the counter ... 1/8` and `coupled-draft-window-start` -> `FAIL coupled_draft: the window ... 0/4`.
**+5 verdicts: vega Arc 345/0/0** (`run_gate.sh` exit 0), llvmpipe 333/0/3, radeon-iGPU 336/0/2; box `radeon_icd`
(7900 XTX) 341/0/1, lvp 333/0/3, nvidia 336/0/2.  Honest note: `coupled_draft_sample`/`coupled_draft_stage` are
called unqualified in `src/core/mtp.cpp`, so `check_port_map.py` (which keys on `kernels::`) neither lists them nor
the two coupled shaders - recorded, not papered over.

**THE `sample_tokens` CHOICE landed (2026-10-05).**  Both sampled paths and the greedy kernel were gated; the CHOICE
between them was not.  `case_sample_tokens` pins it (greedy || temp 0 -> GREEDY; n_blocks<=64 && n_tokens<=64 ->
SPLIT, the default; else -> the one-block ONEBLOCK fallback) as a pure predicate AND by running the chosen kernel
for six requests, including the task's named case: **temperature 0 routes to the ARGMAX, not the sampled path's
uniform draw**.  Falsified by `gates/inject-verify.sh sample-tokens-choice-temp0-to-sampled` -> `FAIL sample_tokens:
temperature 0 ... 3/4`; this injection is in the HARNESS, so `inject-verify.sh` now rebuilds the gate for a
non-shader change (and refused an earlier form as `DID NOT COMPILE (harness)`).  **+7 verdicts: vega Arc 352/0/0**
(`run_gate.sh` exit 0), llvmpipe 340/0/3, radeon-iGPU 343/0/2; box `radeon_icd` (7900 XTX) 348/0/1, lvp 340/0/3,
nvidia 343/0/2.

**THE SAMPLER'S PORTABLE f32 SIBLING landed (2026-10-05).**  `sampler_kernel_f32.comp` - the double-arithmetic
sampler's sibling for the devices WITHOUT shaderFloat64 (Intel's own article; the target hardware), which every
other double kernel in this port already had.  The chain and selection are the faithful kernel's; only the tail is
float, from `common/sampler_tail.glsl`'s `sampler_tail_f32`.  `run_gate.sh`'s fp64 rule now requires it to carry NO
`Float64` capability (measured 0).  Five arms: two EXACT where no rounding reaches it (one-survivor shortlists), one
EXACT because the softmax cancels (64 equal at temp 0, RNG pinned), two MEMBERSHIP on real distributions - and the
case MEASURES the gap: `f32 vs f64: 0 of 48 seeds differ`.  Falsified by `gates/inject-verify.sh
sampler-kernel-f32-top-p-boundary` -> `FAIL sampler_kernel_f32: one survivor (top_p cut of one) 2/8`.  **+6
verdicts: vega Arc 358/0/0** (`run_gate.sh` exit 0), llvmpipe 346/0/3, radeon-iGPU 349/0/2; box `radeon_icd` (7900
XTX) 354/0/1, lvp 346/0/3, nvidia 349/0/2.  Port map still 77 symbols - 28 kernel, 49 host, 0 todo, regenerating
identically.

**The 7900 XTX run's two failures are RESOLVED (2026-10-04).**  On `z820b` (RX 7900 XTX, RADV gfx1100, Mesa 26.0.8)
the gate now reads **`radeon_icd 280 passed / 0 failed / 1 skipped`**, `lvp_icd 272/0/3`, `nvidia_icd (K620)
275/0/2`.  Both failures were the cross-implementation arm earning its keep, and each went a different way: the
`kv_q4 round trip` failure was the CASE's bound (`|d|/2`, wrong for a `d*[-8,+7]` code range whose +8 end clips -
the shader was faithful, `dev-vs-rule mismatch 0`); the `quantize_q8_0 (ggml bytes)` failure was a lavapipe/Mesa
26.0.8 bug (`roundEven(double)` rounds ties toward zero), so the CASE was right and the SHADER was changed to an
explicit ties-to-even.  Full evidence and the two falsifying injections (`gates/inject-verify.sh`) are in
`NEXT.md`'s new top block and `HANDOFF.md` section 6.1.  The XTX's single skip is the pre-existing M8
cooperative-matrix case (the XTX's config is M16), not a failure.

**Stage 3 of the port plan - recorded command buffers, the CUDA-graph replacement - is DONE and VERIFIED
(2026-10-04).** The engine's decode step re-issues a fixed sequence of dispatches every token, and a CUDA graph is
how it avoided that; the Vulkan equivalent is ONE command buffer recorded once and RE-SUBMITTED.
`Ctx::record_begin` / `record_dispatch` / `record_end_and_submit` / `replay_recorded` are in
`harness/vk_compute.*`, and `case_recorded_step` in `harness/vk_gate.cpp` is the case that exercises them: **six
verdicts, all green on the Arc, on llvmpipe and on the Radeon iGPU.** It was proven able to fail before it was
trusted - flipping `record_dispatch`'s `fresh_set` to false (ONE shared descriptor set for the whole recorded step,
the trap the encoding comment names) makes three of the six verdicts fail, 162/0/1 -> **159/3/1**. `NEXT.md`'s top
block carries the arm-by-arm table.

The table below is the original wave-1 set and has not been re-listed as the suite grew - every case since is
gated the same way and is described where it is defined.

**Stage 4 of the port plan - device-local memory, staging and fit accounting - is DONE and VERIFIED (2026-10-04).**
`Ctx::alloc_device()` allocates memory with NO mapping where the device offers it (real VRAM: 3 of the Arc's 7
memory types are device-local and unmappable, on llvmpipe there is no such type at all), `Ctx::alloc_staging()` is the host-visible transfer
buffer, and `write()`/`read()` stage automatically over `vkCmdCopyBuffer` so a case written for stage 1's path runs
unchanged on the engine's. `plan_fit()` answers the other half - which line of the engine's VRAM plan does not fit,
by name, and what a droppable cache buys - and the gate fits and PRINTS the plan against the driver's own figure
(28.64 GiB on the Arc). Six device arms plus five pure-policy arms, green on all three implementations, and the
round trip was falsified with a four-byte offset error (174/0/1 -> 172/2/1). Full detail, including the ONE defect
class the round trip does NOT cover (an absent barrier: deleting the post-copy barrier changes no verdict on the
Arc or llvmpipe) is in `NEXT.md`'s stage-4 block.

**Stage 5's GEMM is DONE AND VERIFIED (2026-10-04) for the layout the engine calls** - `Y[T x ldy] = X[T x K] .
W[N x K]^T` in f16 with an f32 accumulator, on Intel's real matrix tile (8x16x16) and in FMA for the ragged shapes,
plus the engine's `bf16 -> f16` conversion with its clamp.  Ten verdicts (189/0/0 on the Arc; the FMA half green on
all three implementations), with the transposed-operand bug falsified to fail four of them, and an `ldy > n` arm
that leaves the stride columns NaN and requires them to survive.  `NEXT.md`'s stage-5 block carries the layout
proof, the two-kernel split (aligned rows on the matrix units, the remainder on FMA) and the device-layer gap it
found: the harness binds descriptors at offset 0, and the engine reaches row slices by pointer arithmetic.

**The quantised multi-token path needed no new kernel, and is now armed and composed (2026-10-04).**  Reading the
engine showed the prompt path reaches quantised weights two ways: `Gemm::native` **dequantises** the ggml blocks to
f16 and calls the GEMM above (so the quantised prefill needs a DEQUANTISER, not a GEMM), and `iq_mmvq` is the
multi-token row-kernel path (the engine specialises 1/2/4/8 columns).  The port's seven per-format row kernels
already take `ncols` - they were gated only at 1 and 2 - so the multi-token path is now armed at 8 for all seven and
at 32 for IQ2_S, and a new COMPOSED case runs the device's `quantize_q8_1` into the device's `iq2s_mmvq` with the
oracle reading the device's own quantised bytes.  That composed case is also the session's sharpest lesson: its
first version PASSED under an injected per-column-stride bug in the quantiser (an oracle built from the thing under
test follows that thing's mistakes), and it only became an oracle once it asked the question the bytes cannot
answer - per-column liveness.  `NEXT.md`'s stage-5b block has the injection and the fix.

**The ATTENTION BLOCK has started: the short-step decode attention is ported and gated (2026-10-04).**
`attn_decode_short.comp` carries the engine's own contract for `native_flash_attn_short_step` - Q24x256, KV2x256,
scale 1/16, `[capacity,2,256]` f16 KV cache, additive f16 mask broadcast over the heads, GQA 12:1 - re-derived as an
algorithm rather than micro-ported, because the CUDA is a vector flash attention pinned to one compiled binary
(128 threads, lane/float2 mapping, `__shfl_xor` tree, a chosen fma order) and this port's reductions are barrier
trees by design.  Five arms (1 live key, 3 keys, a full window, a masked full window, a masked half window) against
a double-precision softmax oracle, 24 heads x 256 dims each, plus a shape contract that REFUSES width 0 and
width 257.  `wg_reduce.glsl` gained `wg_max`.  Falsified: the plausible wrong GQA grouping fails all five arms at
exactly half the values.  What the block still needs - the prompt path (`qsa_prompt_attn.cu`, `qsa_select.cu`), the
QSA selection/indexer, and a tiled version of this kernel - is listed in `NEXT.md`.

**The f16 KV gather landed with it: the window the attention reads, and the engine's launch rule measured
(2026-10-04).**  `kv_f16_gather.comp` is the F16 sibling of `kv_q8_gather` (`src/kernels/cuda/qsa.cu`), the producer
of the `[id][kv_head][head_dim]` window the attention consumes.  Three things came out of it that are worth more than
the kernel: the POOL's row index CARRIES the head (`[page][kv_head][page_size][dim]`, not `[row][kv_head][dim]` - a
fixture written the intuitive way has the same total size and fails every value); `ids[id]` is a POSITION and `id` is
an INDEX, and a softmax over the keys cannot see a permutation of them, which is why the composed gather-then-attend
arm runs with a position-dependent mask (injected `cell = id` -> worst rel 9.19e+03); and **THE GRID IS THE CAPACITY,
NOT THE LIVE COUNT** - the engine's own launcher comment, now measured through this port's record/replay path: a
capacity grid replays at a grown `n_ids` and writes 8 of 8 rows, the live-count grid writes 6 of 8 and loses two
silently.  That arm is a NEGCTRL: it passes only if the defect reproduces.  The q8 sibling's case had described
this rule in its verdict text while sizing BOTH its window and its grid from the live count, so the guard region it
claimed to check did not exist; fixed to a capacity of 8.  Arc 209/0/0.

**The selection landed next, and with it the first END-TO-END decode chain (2026-10-04).**  `qsa_block_scores.comp`
and `qsa_block_topk.comp` (`src/kernels/cuda/qsa_select.cu`; the engine keeps the second as `qsa_block_topk_ref`) are
the QSA indexer's block scores and the weighted top-k - the producer of the very `ids` the gather consumes.  The gate
now runs the whole decode data path: **pool -> scores -> selection -> gather -> attention**, each stage against its
own oracle and the chain end to end (ids compared, the gathered window compared against the pool rows the selection
named, then the attention output, all four verdicts).  The two rules worth remembering are that a block's WEIGHT IS
ITS CELL COUNT - a complete last block weighs 0 and must be skipped even with an enormous score, and a 1-cell tail
contributes 1, not R, because getting this wrong selects cells that do not exist - and that the ids being ASCENDING
is part of the contract, since downstream the position is the window row and the mask is indexed by it.  Seven
selection arms (a budget cut inside a block, three-way ties at the boundary, a zero-weight tail, a 1-cell tail, both
identity cases, and everything-but-one-cell), three falsifications.  Arc 221/0/0.

**The KV append closed the decode loop, and the harness had a ceiling this increment hit (2026-10-04).**
`kv_f16_append.comp` (`src/kernels/cuda/qsa.cu`) is the WRITE half of the gather's pair - a token's f32 K/V converted
and stored into the paged f16 cache at the row the page table names.  Four rules, four arms: the conversion is
compared as BYTES (it is the arithmetic), a negative page writes NOTHING at all (the injection "treat a non-resident
block as page 0" is caught by that arm and by nothing else), the two layouts (physical page vs identity host copy)
are one kernel with a flag, and re-appending a position overwrites it.  The chain runs append -> gather -> attention
over six tokens appended across a page boundary, at worst rel 8.37e-06.  **And the first full run came back
`FAIL radeon_icd` with no numbers: `VK_ERROR_OUT_OF_POOL_MEMORY`, because the harness's descriptor pool had a
hardcoded `maxSets = 64` and one set per pipeline.**  That is a ceiling on the CASE COUNT, not on a kernel, and it
fails at whichever implementation runs last - fixed by growing the pool on demand and printing when it does
(`descriptor pool 2 created`).  Arc 226/0/0.

**The Q4_0 KV path landed, and with it the first gate arm that tests a DESIGN CLAIM rather than a kernel
(2026-10-04).**  `fwht256.comp` (the orthonormal 256-point Walsh-Hadamard rotation), `kv_q4_append.comp` (ggml's Q4_0
group with the engine's deterministic tie rule) and `kv_q4_gather.comp` (the dequantising reader) - the engine's three
KV storage formats are now all in the port.  The design (rotate K, V AND the query, quantise in the rotated basis,
de-rotate the output once) is legal only because H is orthonormal, so the rotation is gated against an EXPLICIT
Hadamard matrix, against `H(Hx) = x` and `<Hx,Hx> = <x,x>`, and finally THROUGH the real attention kernel: rotate q,
quantise+rotate K/V to Q4_0, attend, de-rotate, and compare with the unrotated attention - worst absolute deviation
0.644 against a window that is itself off by up to 0.474.  **The invariants cannot see a flipped butterfly sign
convention** (the other convention is also orthogonal and self-inverse - it is simply a different basis), and the
injection proves it: flipping the branches fails only the explicit-matrix arm.  Round trip 0.4719 against the group
bound |d|/2 = 0.4742.  Arc 233/0/0.

**The hybrid K8V4 mode landed, with no new kernel - and llvmpipe caught a race in the case that the Arc did not
(2026-10-04).**  The mode IS the wiring (`KV_MODE 3` of the engine's own `kv_hybrid_parity.cpp`): INT8 K, rotated
Q4_0 V, the attention in the mixed basis, and one inverse Hadamard on the output.  The asymmetry is the content -
ONLY V is rotated, so the query is not either - and the arms pin it in five ways, of which the interesting two are a
tight oracle over the same two windows (worst rel 1.68e-06) and the de-rotation tested ALONE against `H` applied
once (worst rel 1.19e-04), because a tight comparison cannot see what the quantisation cost and a bounded one cannot
see a basis error inside its slack.  **The cross-implementation arm earned its keep:** the q4 append's unused K half
was bound to the same pool as its V half, so two writers raced on the same bytes - harmless on the Arc, **3071 of
3072 wrong on llvmpipe**.  One implementation would have shipped it.  Two wiring injections (the de-rotation applied
twice, and omitted) fail exactly the arms they should, and both needed fixing first: the "twice" version dispatched
the same input again (a no-op), and the "omitted" version left a push constant unused so `-Werror` failed the build
and the run printed the STALE binary's numbers.  Arc 238/0/0.  See `RUN-ON-B70.md` for where this sits on the path to
a running engine.

**Descriptor OFFSETS landed: the recorded blocker between the port and the engine is retired (2026-10-04).**
The engine reaches a row slice by pointer arithmetic (`Y + t0 * ldy`, `X + t0 * K`) while the port bound every
descriptor at byte 0 - which forced the GEMM case into a workaround of one buffer per half.  `Buf::offset` (a copy of
the handle with the offset set, default 0 so every existing call site is unchanged) is now honoured at the single
descriptor-write site, with four gate arms: a view reads from its offset, the offset-0 binding still reads from zero,
neither dispatch writes past its count, and - the one that matters - **the L(i)=B(i) lookalike control**, because an
IGNORED offset returns the offset-0 data and looks like a plausible answer rather than an error.  The device's
`minStorageBufferOffsetAlignment` is now recorded per device and printed by the case: **the Arc reports 4 bytes**
(the port's earlier comment guessed 64), which is what makes the engine's slices bindable as they stand - on a device
with a 64-byte limit that arithmetic would need padding.  An unaligned offset is refused by name before it reaches
the driver.  Arc 242/0/0.  **Honest limit:** the aligned path is gated on three implementations, but the refusal path
was NOT reproduced out of tree - the probe's device contract differs from the gate's (it reads the LEDGER budget
instead of the driver's and refuses the allocation first), and chasing it was not worth the time; what is verified is
that the refusal machinery fires and names its cause.

**The sampler is nearly complete, and the port can emit a TOKEN two ways (2026-10-04).**  `sampler_greedy.comp` carries
`sampler_greedy_kernel` (`src/kernels/cuda/sampler.cu`) with the penalty pair it calls - the `--temp 0` path - so the
port has a complete deterministic generation path: logits in, a token out.  The rules a paraphrase inverts are all
arms: the repeat penalty MULTIPLIES for a non-positive logit and DIVIDES for a positive one (dividing unconditionally
inverts it on half the vocabulary); the presence penalty is a boolean, not the count, while the frequency penalty
carries the count; the window is the TAIL, so a head-only hit is not penalised; and ties go to the LOWEST index, with
0 for an all -inf/NaN row.  Ten arms, each stating the token it is built to produce, checked against a transcription
of the rule as well.  **The falsification found a decorative arm:** the presence arm PASSED under the injection
because its fixture let the penalised token lose under both rules - only an injection can say that.  Rebuilt, all
four injections now fail exactly the arm built for them.  Arc 253/0/0.

**The general sampler landed too - top-k, top-p, min-p, temperature and the Philox draw.**  `sampler_kernel.comp`
plus `common/philox.glsl`, so every path a decode step needs is ported.  The chain is llama.cpp's, and two of its
details are things a careful port would "fix": the penalties belong on the RAW logits, ONCE, before the filters (the
source records fixing a second application after the temperature), and **`temperature == 0` is not greedy** - it
scales every survivor to zero, making the draw UNIFORM over the shortlist.  The engine's Philox constants are its own
(0x9E3779B9 / 0xBB67AE85), NOT Random123's, so a swap yields a valid generator with different numbers.

**The RNG is pinned through the shortlist rather than read back:** 64 EQUAL survivors at temperature 0 make
`floor(u*64)` observable in the returned token, so 16 seeds pin ~96 bits of the stream - and that arm is exact,
because equal probabilities make the softmax cancel out of the cumulative walk.

**Nine injections, all caught, and two of them changed the case.**  The volume of the penalty bitmap, the round count,
the constants and the key/counter assignment do not go unnoticed: the RNG arm returns 0/16 for each.  But the doubled
penalty (the source's own recorded bug) initially passed 184/184, because the arm asserted the head was REACHABLE and
a two-token shortlist makes that true under either order; the fix was a `top_k = 1` arm, where a one-token shortlist
makes the draw irrelevant and the order exact - it now returns -1/16.  Its fixture then had to be tightened again
(2.0 against 1.5, not 0.2), because the margin that made the majority test safe made the flip invisible.  Separately,
three fixtures were strengthened BEFORE injecting, on noticing that equal logits make `inv_t` irrelevant and that a
membership check cannot see too FEW survivors.  Three of the nine injections also needed the compile target corrected
to the shader that INCLUDES the file rather than the include itself.

**The gate now checks the PORT MAP before it runs a single kernel.**  `tools/check_port_map.py` verifies
`PORT-MAP.tsv` - the classification of every `kernels::` symbol the DECODE path (`src/core/`) calls into kernel /
host / todo - against the engine's sources and the built shaders, and fails on an invented symbol, on a `kernel` row
naming a shader that is not built, and on any decode-path symbol the map does not mention.  That last rule is the
point: a new call site cannot join the decode path unnoticed.  (It read 77 symbols - 17 kernel, 50 host, **10 todo**
when it landed; the count is now **168 - 53 kernel, 63 host, 52 todo**, because the scan was later extended to the
BARE-name call sites a `using namespace` makes legal - see the top section.)  Falsified three ways before landing.

**THE EMBEDDING GATHER LANDED, and `precise` turned out to be LOAD-BEARING.**  `embedding_gather.comp` (packed
codes + per-group scales -> float rows, from `verify_kernels.cu`'s `_dev` kernel and the parity reference).  The
finding generalises beyond this kernel: **a driver may FUSE a multiply and an add unless the result is `precise`.**
The SPIR-V carries no fma op, but GLSL's default permits contraction, so the backend can still fuse - and one fused op
rounds once where the engine's `__fmul_rn` + `__fadd_rn` round twice, which the engine compares bitwise against.
Measured: four of six arms failed before `precise`, and the two that passed were those whose products happen to be
exact - that signature is what identified the cause.  The shader qualifies the product, the offset and the sum; the
SPIR-V now carries six `NoContraction` decorations.  The fixture was built to SEE this: the host searches for
(code, scale, offset) triples where the fused and two-step forms differ and puts one at element 0, reporting the
count - the engine's own `fma_diff` datum.  Five injections all bite: contractable arithmetic, MSB-first codes, the
group as the element-within-group, the bias dropped, and the token array ignored.  The port map's hole list is now
ten kernels.

| Case | Verdict | Method |
|---|---|---|
| harness self-test (copy) | PASS 1024/1024 | bit-exact — proves bindings, push range and host-read barrier |
| `scale_inplace` | PASS 1000/1000 | bit-exact vs the CUDA arithmetic |
| `add_inplace` | PASS 1000/1000 | bit-exact |
| `f32_to_bf16_bulk` | PASS 1024/1024 | **bit-exact** vs `bf16_bits.hpp` (round-to-nearest-even, NaN quieted) |
| `f32_to_f16_bulk` | PASS 1024/1024 | **bit-exact** vs `f16_bits.hpp` (RNE, subnormals, overflow→inf) |
| negative control (truncating f16) | PASS (484/1024 differ) | proves the bit-exact gate can FAIL |
| `gdn_gate` (1 and 3 tokens) | PASS, worst rel **4.25e-07** | vs double-precision softplus; tol 5e-6 |
| `rms_norm_weighted` (4 shapes, incl. rows=2) | PASS, worst rel **1.79e-07** | vs double reference **and** a NaN-padded tail that fails if the row guard is missing |
| `silu_inplace` | PASS, worst rel **7.58e-07** | vs the engine's double-precision reference |
| transcendental probe (`exp`, `log`) | INFO | driver `exp` ≈ 9.1e-07, `log` near 1 ≈ 1.5e-07 abs error |
| memory budget: independent requery agrees | PASS | the test re-queries `VK_EXT_memory_budget` itself and compares |
| memory budget: free − reserve == usable | PASS | against the free figure the test computed independently |
| memory budget: usage tracks an 8 MiB alloc | PASS | the driver's number is a measurement, not a constant |
| desktop reserve policy (5 cases + 2 controls) | PASS | floor, 25%-of-card clamp, and the in-between case |
| over-budget allocation REFUSED | PASS | a child process with an 8 MiB card must exit 3 and say why |
| compat: advisory rules (11 cases + boundary) | PASS | kernel/Mesa/driver combinations, each case discriminating against its neighbour |
| compat: kernel + Mesa parsed live | PASS | `uname` release and the driver-supplied Mesa string, parsed and checked |
| ledger + discrete + no ceiling REFUSED | PASS | child process; would report 24 GiB usable if the rule were removed |
| explicit ceiling unblocks the ledger | PASS | child process; `STRATA_VK_MAX_BUDGET_MIB` is the documented remedy |

The bf16/f16 conversion fixtures deliberately include exact rounding ties at both precisions, subnormals,
the fp16 overflow point (65504 → 65536), infinities, a signalling NaN pattern and signed zero. The
`rms_norm` case `rows = 2` is the QSA case in the source's own comment: the launcher rounds the grid up, and
a missing `if (row >= rows) return;` overwrites the buffer that follows — so the case pads the allocation
with NaN and requires it to survive.

## Audit

`ports/vulkan/AUDIT.md` — a full pass against the standing code guideline plus the Arc/Vulkan and performance
rules. Nine code-guideline findings (duplicated reference implementation, dead code, parallel containers, a
hardcoded path, a constant repeated nine times, `-Werror`) and six compatibility findings — two of them real
defects (**`rms_norm` coupled the host's dispatch to the driver's subgroup width**, which would skip tail rows
silently on Intel where the driver picks a SIMD width per kernel; and **an fp16 device feature requested with
no fp16 in any kernel**, which fails `vkCreateDevice` on hardware that lacks it). All fixed and re-verified,
five new automated checks added, each proven able to fail — including a CROSS-IMPLEMENTATION arm, which was
added after the same shaders measured worst-case relative error 1.13 on llvmpipe (subgroup 8) while passing on
RADV (subgroup 64): a reduction combine stage assumed `gl_NumSubgroups <= gl_SubgroupSize`. Fixed and verified
green on both implementations.

**Display contract:** the Vulkan layer queries `VK_EXT_memory_budget` (heap usage, not this process's ledger),
holds back a 1024 MiB desktop reserve (floor 256 MiB, capped at 25% of the card), and refuses an allocation that
would cross it — naming the numbers. Measured live on this box: with the resident local model holding the
7900 XTX, RADV reports 0.19 GiB free of 24 GiB and the harness **refused to allocate**, which is the behaviour
that would have prevented the recorded AMD incident (`docs/AMD_HIP.md`, #380/#377).

**Verified on three Vulkan implementations, every run:** Intel Arc Pro B70 / ANV (subgroup 32), llvmpipe / CPU
(subgroup 8) and the AMD iGPU / RADV (subgroup 64) - the discrete Radeon is gone, so the radeon ICD now picks the
iGPU. Before the swap the arm read 14/14 on RADV/7900 XTX and llvmpipe; the suite has grown since and prints its
own totals. **Two caveats on this box, both measured 2026-10-04** (`NEXT.md`): `run_gate.sh` used to stop at the
Intel skip and so never reached this arm at all after the swap, and the iGPU's `budget: independent requery
agrees` is INTERMITTENT - 1 failure in 3 consecutive runs of the same binary on the same device - rather than the
deterministic failure it was first documented as.

## Three defects found by the gate (all real, all fixed, all measured)

0. **The expert tier's `dx` came from the wrong buffer.** `s2_row_dot`'s `f16_at` takes a byte offset into the
   EXPERT BLOB; the CUDA's takes a pointer and was reading the ACTIVATION. So with `use_xscales` false every
   row's multiplier was arbitrary code-byte content, and a code-byte pair decoding as a NaN fp16 turned the row
   to NaN through all 80 chunks. This is the defect that was misdiagnosed for a day as a masked-wave reduction
   fault - see `NEXT.md`, and `case_s2expert_tier`, whose six arms were written for the misdiagnosis and found
   the real one. Fixed by `act_f16_at`.
0b. **The same class recurred in the IQ1_M port and the new case caught it in one run.** `iq1m_read_int4` read
   `w_b` (the weights) while its only call site passed an ACTIVATION offset, so the activation words came out of
   the weight buffer. The case's diagnostic identified it in a single comparison: oracle part 0 `0.440796`
   against device `-0.395386`. Fixed by renaming the helper after the buffer it reads (`iq1m_act_int4`), which is
   the rule this port now applies to every CUDA pointer-taking accessor: the buffer belongs in the name, because
   both mistakes compile and both read a legal address. **A CUDA helper that takes a pointer loses its argument
   when the port makes it a byte offset, and nothing but a case with an oracle notices.**

1. **`gdn_gate` first version: softplus lost 4.3e-05 relative.** The delegated translation wrote
   `log(1.0f + exp(x))`; the source uses `log1pf(expf(x))`. A Kahan-form `log1p` did **not** fix it — the
   failure number was identical to five figures, which said the diagnosis was wrong. The actual cause,
   established with a probe shader: the driver's `log` carries ~2^-22 *absolute* error near `u = 1`, which
   breaks the cancellation a Kahan `log1p` depends on, and for negative `x` softplus **is** `exp(x)`, so the
   whole error lands in the result. Replaced with an 18-term series below `z = 0.5` and the driver's `log`
   above it: **4.26e-05 → 4.25e-07**, and the tolerance is now 5e-6 instead of 1e-4.
2. **fp64 `silu` is not expressible with this host's toolchain.** `silu_inplace` computes in double in CUDA.
   GLSL has `double`, but neither glslc (glslang 14.0) nor `glslangValidator` 15.1 can resolve a double
   overload of `exp` for SPIR-V. The port ships the f32 shader and **measures** its gap against the double
   reference (7.58e-07) instead of asserting it away. See `shaders/blocked/README.md`.

## One finding that was NOT a kernel defect: the port misread the device (2026-10-04)

`gemm_coopmat` skipped on the Arc with "no usable config: needs M16 N16 K16 subgroup-scope with f16 A/B and an f32
accumulator", which reads as the device having no matrix units.  Enumerating the driver's own config list showed
otherwise: **BMG's floating-point config is M8 N16 K16.**  The criterion had been written against the departed
Radeon's list, where every config is M16, and a tile is a property of the device per generation - so the port was
reporting a limitation of its own rule as a limitation of the hardware.  `gemm_coopmat_m8.comp` now exists, the
case selects the pipeline from the property list and prints which one ran, and the Arc reads **180 / 0 / 0** with
the matrix path exercised on all four shapes including a prompt-shaped 128x256x256 grid.

Two lessons, both general: **a skip message is a claim about the device and has to be derivable from the device**
(this one was derivable only from RADV); and **the wrong tile is silent** - forcing the M16 kernel on BMG does not
fail pipeline creation, it computes wrong numbers (2560 of 4608 elements correct in the square case), which is why
the case prints the shape it selected rather than only PASS.

## Not done (and how much of the job each is)

* **Waves 2-6 of the kernels** (KV, rope, the GEMVs, attention, MoE, the prefill GEMM): the remaining CUDA
  kernel files, plus the 8 prefill files and the 7 `cuBLASLt` call sites (which have no Vulkan equivalent and
  must be written by hand). The method and the gate are proven; this is a grind, not an unknown. **The resident
  expert tier - `s2_expert_grouped.cu`'s gate/up, SwiGLU and down - is DONE and gated (2026-10-04).** The next
  workstream is the quantized-expert wave: 22 shaders from `iq_kernels.cu` and `native_mmvq.cu`. **ORDERED BY THE
  RESIDENT MODEL'S OWN PACK, not by its filename**: `coder-iq1_m`'s experts are IQ2_S (20 of 48 layers), IQ3_XXS
  (17), IQ3_S (10), IQ4_XS (1) for gate/up and IQ4_NL (39), Q2_0 (9) for down - see `NEXT.md`, which carries the
  census and the reason the filename misleads. Done so far in this wave: the IQ1_M row dot, the two q8_1
  quantisers, and **ALL SIX of the model's expert formats - IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up) and IQ4_NL,
  Q2_0 (down)**, each with its dot in `shaders/common/` and its own gated case. **Every one of the model's 48
  layers now has BOTH halves of the expert path ported** (before the IQ4_NL step that count was zero: no layer is
  complete without its down projection, which is why the down formats were taken first). The grouped kernels in
  (b) are now DONE for the two formats that cover the most layers - `native_gu_iq2s.comp` and
  `native_down_iq4nl.comp`, the shape the expert tier actually launches - with the four other instantiations and
  the `_multi` variants left as mechanical copies (see `NEXT.md`).
* **The engine integration**: **STARTED — I1 landed 2026-10-05** (`vulkan/` carries `STRATA_ENABLE_VULKAN` from
  increment 0; I1 added `vulkan/src/device/` — the adopted port device layer plus the arena, the pointer->buffer
  resolution and the pipeline cache — and one entry point, `strata::kernels::fwht256_cuda`, that RUNS and is gated
  bitwise against the port's shader path; see the section at the top). What is still NOT there: the engine's own
  sources under `STRATA_ENABLE_VULKAN` (I1's target is a smoke executable, not the engine program), the other 39
  entry points, and the recorded decode step IN THE ENGINE — so `src/` cannot yet drive the backend and there is
  still no `setup.py`/engine `CMakeLists` change. `harness/vk_compute.*` remains the seed from which
  `vulkan/src/device/` was adopted, and it carries the single-shot `dispatch()`, the recorded-step API (stage 3)
  and the engine-shaped memory path (stage 4), each verified by the gate.
* **Anything on Intel hardware — NO LONGER UNVERIFIED (2026-10-04).** An Arc Pro B70 is now the discrete card in
  this host (the 7900 XTX is out; the Ryzen iGPU drives the display) and the gate runs on it: **180 / 0 / 0** —
  no skips at all, including the matrix path, which was the port's own criterion misreading the device rather
  than the device lacking anything (see the M8 finding above), with
  one device-specific defect found and fixed (the folded division in `quantize_q8_K` - NEXT.md's RESUME HERE).
  What stays unverified on Intel is everything the gate does not cover: the engine path, real token shapes, and
  the stability question below.
* **The budget requery on an INTEGRATED device.** `budget: independent requery agrees` fails on the AMD iGPU and
  passes on the Arc: an iGPU's free-memory figure is system RAM shared with the OS, so two queries can disagree by
  construction. Left red on purpose - the port's rule is to report an implementation that cannot satisfy a check
  rather than soften the check around it. **Measured 2026-10-04: it is intermittent, not deterministic** (1 failure
  in 3 consecutive runs of one binary on one device), so a single radeon run can read 161/1/1 or 162/0/1 - quote
  the device and the run, not "the radeon ICD always fails one".
* **The Battlemage stability question (§4 of the plan) — TESTED ON THE CARD, 2026-10-04.** The xe compute wedge is
  an open driver bug this port cannot fix or test around, which is why the plan makes a smoke test the FIRST thing
  to do on Battlemage hardware. Done, with `gates/smoke-arc.sh` (added the same day): **718 rounds of 8 concurrent
  instances — 5744 suite runs, ~890k case executions — over 480 s: 0 kernel cases failed, 0 hangs, 0 xe errors in
  the kernel log, slowest round 1 s, no latency creep**, after 502 further runs one instance at a time, likewise
  clean. The card did not wedge. The honest limit of that claim is the plan's own bar - "an hour of inference" -
  so what is measured is "has not wedged under 8 minutes of concurrent small-kernel load", not "survives sustained
  inference"; an inference-shaped arm on the same driver is what would close that gap.
* **Two test-side defects the smoke run found, both fixed rather than relabelled.** The first concurrent run
  exposed `case_firmware_variants` writing and deleting a FIXED `/tmp` path, so one instance's cleanup removed the
  file another was reading (2 of 8 jobs; 502/502 when run one at a time) - now per-process. And the runner learned
  to CLASSIFY the two checks that read the DRIVER's view of a busy card, because treating them as failures aborted
  the loop at round 1 and hid the sustained result. Both are in `gates/smoke-arc.sh` and the commit that added it.

## Estimated remaining cost

* **Kernel waves 2-6 (37 files):** the first wave's 8 kernels took ~40 minutes wall-clock end to end
  including the two defects above, most of it gating. At a similar rate, and with generation offloaded to the
  local model, expect **~2-3 focused sessions**. The long tail (MoE + the 7 quant formats) is the larger half.
* **The GEMM + prefill path: the GEMM IS DONE (2026-10-04) for the layout the engine calls.**  `Y[T x ldy] =
  X[T x K] . W[N x K]^T` in f16 with an f32 accumulator (`Gemm::f16`/`Gemm::bf16` in `src/prefill/gemm.cu`, whose
  cuBLAS call is (OP_T, OP_N) - the weight is the transposed operand), on the matrix units at Intel's real tile
  (8x16x16) and in plain FMA for every ragged shape and every device without a config, plus the engine's own
  `bf16 -> f16` operand conversion (with its clamp, not an overflow into Inf).  Ten verdicts, green on the Arc and
  the FMA half green on all three implementations, with the transposed-operand bug falsified to fail four of them.
  What remains in this area: **`src/prefill/kernels.cu` (53 KB), `moe_fused.cu`, `moe_fused_iq.cu` and
  `moe_mmq.cu`** - the fused prefill kernels and the quantised-weight MMQ path (`Gemm::native` over ggml types),
  which is the same porting grind as the expert tier with a different reduction; and the device-layer gap recorded
  in `NEXT.md` (descriptor bindings are offset-0, and the engine passes row slices by pointer).
* **Engine integration to a served model:** ~1 session for a CPU-verifiable path, then a GPU window on the
  hardware that is actually being targeted.
* Cheaper alternative worth considering first: the port buys *compatibility* (Arc, and any Vulkan GPU). If the
  goal is only "Arc runs Strata at a useful speed", the fastest legitimate route is the one Intel validates —
  their LLM Scaler container. This port is the route where the engine stays ours and the model stays GGUF.
