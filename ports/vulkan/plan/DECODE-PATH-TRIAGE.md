# Decode-path triage — the 52 `todo` rows, from the engine's own sources
#
# CURRENT 2026-10-05 (after the BF16-PROJECTION batch: `bf16_gemv` + `bf16_gemv_split` PORTED, and the eight
# SPECULATIVE-DRAFTER symbols LABELLED CLASS C): the map reads **168 = 78 kernel + 61 host + 29 todo**.  The
# class-A set is CLOSED; the performance tier's class-B native fast paths are ALL ported (batch 1's four, ALL SIX
# native GDN / DeltaNet mixer kernels, the THREE fused GDN paths, the QSA gate); and this batch ports the TWO
# BF16-PROJECTION entry points that `project_bf16` (layer.cpp:94-100) reaches on the DEFAULT side of the
# `native_bf16_projections` setting, so that setting can no longer route the engine at an unported symbol whichever
# way it answers.  The **REACHABILITY AUDIT** at the end of this file is the decision list under the port's
# capability answers; batch 5 fixed one hole (`native_qsa_indexer_append`, an UNIMPLEMENTED gating flag) and this
# batch CLOSES the second soft edge it raised (the two BF16-projection rows).  M-A is RE-DEFINED over the class-A
# set at the end of this file ("THE RE-DEFINED MILESTONE M-A").  The numbers quoted immediately below are the state
# at `7c317c4`, kept as the record the triage was written against.

Written 2026-10-05 on `vega`, branch `vulkan-arc-port`, HEAD `7c317c4`.  Companion to `PORT-MAP.tsv` and
`tools/port_map_lib.py`; it **explains** the map's `todo` column and does not rewrite it.  The map still reads

    port map: 168 decode-path symbols - 53 kernel, 63 host, 52 todo

`7c317c4` fixed the checker so it also sees the symbols `src/core/` calls BARE (a `using namespace
strata::kernels;` in scope).  The map moved from `77 - 28 kernel, 49 host, 0 todo` to `168 - 53 kernel, 63 host,
52 todo`, and **M-A was declared closed on the old reading, so M-A has to be re-established here.**

## Counts per class (from the triage below)

| class | n | what it is |
|---|---|---|
| **A** — genuine decode-path hole, no ported fallback on either branch | **19** | the decode path of the shipped model cannot run without these |
| **B** — capability-gated, shipped branch, **ported** fallback the backend can force | **4** | backend sets the `*_enabled()` to false |
| **C** — non-selected configuration: the shipped `--native` run takes the other branch, which IS ported | **7** | not needed for the shipped model as launched |
| **D** — not the forward pass: P6 verifier, speculative drafter, tooling | **22** | separable from a correct first token |
| **total** | **52** | |

**Class C is not the family you would guess.**  The brief names *the GDN family* and *the QSA indexer* as
candidates for "a model family the shipped model does not use".  They are **used**.  The shipped model is a
48-layer hybrid: `include/strata/core/layout.hpp:30` `qsa_interval = 4`, `is_qsa_layer` (`:63-65`), so **12 QSA
layers (3, 7, … 47) and 36 GDN / DeltaNet layers** — `docs/HOW_IT_WORKS.md:22` "attention and DeltaNet mixers".
`gdn_layer` is dispatched for every non-QSA layer (`src/core/layer.cpp:1259`) and `qsa_layer` for the rest, and
both are exercised every token.  So the GDN mixer and the QSA indexer are **class A**, not C.  The class-C rows
are the *other* branch of a toggle the shipped launch selects the ported side of (native vs legacy projections,
the fast select/attention defaults).

## The rule used

A symbol's branch is read from its call site in `src/core/`.  The shipped configuration is the engine's own
default launch: `setup.py:4224-4227` writes `--pack … --native <shard> --expert-config … --spec 4 --mtp …`, so
`native_preset` is set and `src/program/generate.cpp:1805-1807` turns **every** `native_*` flag on
(`native_bf16`, `native_gdn`, `native_router`, `native_qsa`, `native_qsa_indexer`, `native_rope`,
`native_moe_combine`, `native_ple_postops`), and `gr_native_mmvf = true`.

* **A** — reached on the forward path of a layer the shipped model has, and **both** branches of the toggle
  (native and legacy) are `todo` (or the symbol is unconditional), so no configuration avoids it.
* **B** — the symbol runs in the **shipped** branch and the branch the toggle would take instead reaches a map
  `kernel` (ported) symbol; the backend forces the toggle off and uses the ported symbol.
* **C** — the symbol runs only in a branch the shipped configuration does **not** select, and the selected
  branch is a map `kernel` symbol.
* **D** — reached only from the P6 verifier (`src/core/verify.cpp`) or the speculative drafter
  (`src/core/mtp.cpp`) or a tooling helper.  These are separable: `--spec` defaults to `0`
  (`generate.cpp:452`) and `RUN-ON-B70.md:52` states "Speculative decoding off keeps … the draft head off
  the path".  (The shipped `setup.py` sets `--spec 4`, so on a `--spec 4` run the drafter/verifier do run; if
  the milestone is meant to cover that loop, the class-D `mtp`/verify symbols become forward-path work — see
  *Ambiguities* below.)

## Class A — the genuine holes (19)

Every one is on the forward path of a layer the shipped model has.  `native_*` and their legacy siblings have
**no ported side** (only `gdn_gate` — the legacy one — is ported), so neither branch of the toggle helps.

### GDN / DeltaNet mixer — 36 of the 48 layers (`src/core/layer.cpp`, `gdn_layer`)

| symbol | call site(s) | the other branch is also `todo` |
|---|---|---|
| `fused_gdn_conv_l2` | `layer.cpp:250` (the shipped `fused_pre` branch) | non-fused `native_gdn_conv_silu` / `gdn_conv_step` |
| `native_gdn_conv_silu` | `layer.cpp:253` (`if (native_gdn_enabled())`) | `gdn_conv_step` |
| `gdn_conv_step` | `layer.cpp:255` (legacy branch) | `native_gdn_conv_silu` |
| `native_gdn_l2_norm` | `layer.cpp:266-267` | `gdn_l2_norm` |
| `gdn_l2_norm` | `layer.cpp:269-270` | `native_gdn_l2_norm` |
| `fused_gdn_ab` | `layer.cpp:287` | `native_gdn_beta_gate` + `native_gdn_gate` |
| `native_gdn_beta_gate` | `layer.cpp:296` | `gdn_beta_gate` |
| `native_gdn_gate` | `layer.cpp:297` | `gdn_gate` is ported, **but** the legacy branch is `{gdn_beta_gate; gdn_gate;}` and `gdn_beta_gate` is itself `todo` — the branch is not fully ported, so the dodge is unsound |
| `gdn_beta_gate` | `layer.cpp:299` | `native_gdn_beta_gate` |
| `native_gdn_step` | `layer.cpp:308` | `gdn_step` |
| `gdn_step` | `layer.cpp:309` | `native_gdn_step` |
| `fused_gdn_step_norm` | `layer.cpp:322` | `native_gdn_step` + `native_gdn_out_norm` |
| `native_gdn_out_norm` | `layer.cpp:324` | `gdn_out_norm` |
| `gdn_out_norm` | `layer.cpp:325` | `native_gdn_out_norm` |

Only `gdn_gate.comp` exists in `ports/vulkan/shaders/`; there is no other GDN shader.  The shipped config
takes the three `fused_gdn_*` branches when `g_fused_gdn && native_gdn_enabled() && native_bf16_projections &&
ssm_d_conv == 4 && ssm_state_size == 128` (`layer.cpp:247,306`) — the artifact's geometry — and the rest are
the non-fused / legacy fallbacks, all unported.

### QSA gate and indexer — 12 layers (`src/core/layer.cpp`, `qsa_layer`)

| symbol | call site(s) | why it cannot be dodged |
|---|---|---|
| `native_qsa_gate_apply` | `layer.cpp:1010` (`if (native_qsa_enabled())`) | the fallback `qsa_gate_apply_f32` is `todo` |
| `qsa_gate_apply_f32` | `layer.cpp:1011` | `native_qsa_gate_apply` is `todo` |
| `native_qsa_indexer_append` | `layer.cpp:945` (`if (native_qsa_indexer_enabled())`) | the fallback `indexer_key_append` is `todo` |
| `indexer_key_append` | `layer.cpp:948` | `native_qsa_indexer_append` is `todo` |

There is no `qsa_gate`/indexer shader in the tree.  The indexer runs unconditionally every QSA layer every
token (the append at `:917-948` precedes the selection at `:964-1005`), so it is not a prompt-only path.

### The hyper-connection write

| symbol | call site(s) | why it cannot be dodged |
|---|---|---|
| `gr_write` | `layer.cpp:1195,1261,1329,1332` (and `mtp.cpp:605`, `verify.cpp:661`) | the fused alternative is `fused_gr_read` (`layer.cpp:1253,1276`), which the map marks `host` but which is a device op with **no shader** (`src/kernels/cuda/fused_gr.cu:1168` launches `gr_down_kernel`/`gr_up_kernel`) — see *Map caveat* below |

## The implementation count behind the 19 class-A symbols (2026-10-05)

This section does not change the classification above; it records the count that makes the class tractable, and
corrects one number in the working brief.

Of the **19** class-A symbols: **15 are native/legacy pair members** — `native_gdn_conv_silu`/`gdn_conv_step`,
`native_gdn_l2_norm`/`gdn_l2_norm`, `native_gdn_beta_gate`/`gdn_beta_gate`, `native_gdn_gate` (its partner
`gdn_gate` is already ported), `native_gdn_step`/`gdn_step`, `native_gdn_out_norm`/`gdn_out_norm` (11 GDN) and
`native_qsa_gate_apply`/`qsa_gate_apply_f32`, `native_qsa_indexer_append`/`indexer_key_append` (4 QSA); **3 are
the fused GDN paths** `fused_gdn_conv_l2`/`fused_gdn_ab`/`fused_gdn_step_norm`; and **1, `gr_write`, is
unconditional**. 15 + 3 + 1 = 19. The brief's "14 of the 19 are native/legacy pairs" undercounts the pair
members by one and does not account for the 3 + 1.

Under the branch policy **`native_gdn_enabled() == false`** (and, for QSA, `native_qsa_enabled() == false` /
`native_qsa_indexer_enabled() == false`), one implementation per pair is sufficient: the layer takes the legacy
`else` of every pair, and the flag removes the three fused paths from the forward path entirely (they are gated
on `native_gdn_enabled() && …`, `layer.cpp:247, 306`). So the 19 symbols need **9 kernel implementations**:

| # | symbol | branch | state |
|---|---|---|---|
| 1 | `gdn_conv_step` | legacy | **LANDED 2026-10-05** |
| 2 | `gdn_l2_norm` | legacy | **LANDED 2026-10-05** |
| 3 | `gdn_beta_gate` | legacy | **LANDED 2026-10-05** |
| 4 | `gdn_gate` | legacy | already ported (`gdn_gate.comp`) |
| 5 | `gdn_step` | legacy | **LANDED 2026-10-05** |
| 6 | `gdn_out_norm` | legacy | **LANDED 2026-10-05** |
| 7 | `qsa_gate_apply_f32` | legacy | **LANDED 2026-10-05** |
| 8 | `indexer_key_append` | legacy | todo |
| 9 | `gr_write` | unconditional | todo |

**8 to write; 6 are now landed** (the second batch of three — `gdn_step`, `gdn_out_norm`, `qsa_gate_apply_f32` —
is recorded in `NEXT.md`'s top section). **2 remain:** `indexer_key_append` (QSA indexer pair; its own check
`native_qsa_indexer_enabled() == false`) and `gr_write` (unconditional). With `gdn_step` and `gdn_out_norm` the
**GDN / DeltaNet mixer chain is complete under the contract** — the six legacy GDN members all have shaders and
nothing else in the chain is reachable while the fused/`_multi` variants are off (the `_multi` /
`gdn_conv_commit` family is class D, `verify.cpp` only). Contract this puts on the backend's capability checks:
`native_gdn_enabled()` must answer **false** on Vulkan, and (when the QSA increments land) `native_qsa_enabled()`
and `native_qsa_indexer_enabled()` must too. `layer_verify_compatible()` (`src/core/layer.cpp:476-486`) requires the
native GDN and the native QSA indexer, so answering them off disables the P6 verify window — speculative
verification only; a `--spec 0` run is unaffected.

## Class B — capability-gated, ported fallback (4)

All four are the **shipped** branch; the backend forces the `*_enabled()` to false and the engine uses the
map-`kernel` symbol named.  `native_router_enabled` etc. are themselves `host` rows (`PORT-MAP.tsv`), i.e. the
engine only checks flags — the backend owns them.

| symbol | call site | check | fallback (map status) | what the backend must make the check return |
|---|---|---|---|---|
| `native_rope_apply` | `layer.cpp:881`; also `mtp.cpp:515`, `verify.cpp:769` | `native_rope_enabled()` | `rope_neox_apply` — **kernel**, `rope_neox` | `false` → NeXo rope |
| `native_router_top10` | `layer.cpp:371`; also `mtp.cpp:576` | `native_router_enabled()` | `router_top10` — **kernel**, `router_top10_f32/f64` | `false` → generic top-10 |
| `native_qsa_rms_norm_weighted` | `layer.cpp:879`; also `mtp.cpp:488,491,514`, `verify.cpp:767` | `native_qsa_enabled()` | `rms_norm_weighted` — **kernel**, `rms_norm` | `false` → weighted rms norm |
| `native_moe_combine` | `layer.cpp:463`; also `mtp.cpp:601` | `native_moe_combine_enabled()` | `moe_combine` — **kernel**, `moe_combine_f32/f64` | `false` → weighted combine |

**The dodge is sound for the main path but not free.**  `layer_verify_compatible()` (`layer.cpp:476-486`)
requires the native configuration for the P6 verify window, so forcing these off disables speculative
verification — which is class D anyway.  It does not require `native_rope`/`native_router`/`native_qsa`/
`native_moe_combine` to be on (`:478-483` checks `native_bf16`, `g_fused_gr`, `native_gdn`+`g_fused_gdn`,
`g_fast_attn`, `g_fast_select`, `native_qsa_indexer`), so the four here are safely dodgeable.

### Class B is now PORTED (2026-10-05)

All four class-B symbols have shaders and gated cases, so the "dodge" is retired for the ones the backend can
turn on.  The Vulkan backend answers the capability checks ITSELF (`vulkan/src/kernels/native_caps_vk.cpp`),
and the answer is a **symbol-at-a-time truth**, not a blanket `true`:

| class-B symbol | shader | capability check | the Vulkan backend answers | why |
|---|---|---|---|---|
| `native_rope_apply` | `native_rope_apply` | `native_rope_enabled()` | **true** | every caller of this check dispatches only `native_rope_apply` (layer.cpp:881, mtp.cpp:515, verify.cpp:769) |
| `native_router_top10` | `native_router_top10` | `native_router_enabled()` | **true** | the reachable set is `{native_router_top10}` (layer.cpp:370, mtp.cpp:576); the `_multi` variant is verify.cpp:916 only, and the verifier cannot init under this contract |
| `native_moe_combine` | `native_moe_combine` | `native_moe_combine_enabled()` | **true** | same: layer.cpp:463 and mtp.cpp:601 are the forward path; `_multi` (verify.cpp:1081) is verifier-only |
| `native_qsa_rms_norm_weighted` | `native_qsa_rms_norm_weighted` | `native_qsa_enabled()` | **FALSE** | this ONE flag ALSO gates the UNPORTED `native_qsa_gate_apply` (layer.cpp:1010, the MAIN QSA path, 12 of 48 layers); answering true would dispatch a symbol with no shader |

**The `native_qsa` row is the point of the "symbol-at-a-time" rule.**  The symbol is ported, gated and measured,
but the flag it belongs to is shared with an unimplemented sibling, so the honest answer is still `false` — the
port chooses the legacy branch until that sibling lands.  Setting it true would crash the QSA path, which is
exactly what the task's warning ("whatever is not ported must still answer false") is about.  See
`vulkan/src/kernels/native_caps_vk.cpp` and the gate's `case_native_capabilities`, which asserts the four
answers AND that each ported symbol's `.spv` exists.

**Measured (Arc Pro B70 / ANV, the box's RX 7900 XTX / RADV NAVI31 and Quadro K620, the Ryzen iGPU / RADV, and
llvmpipe; `ports/vulkan/bench/`):**  the native pair is timed against the legacy kernel each replaces at the
same shape: `native_rope_apply` **0.301×** of `rope_neox` on the Arc (0.118× on the XTX),
`native_router_top10` **0.078×** of `router_top10_f32` (0.082× on the XTX), `native_moe_combine` **0.998×**
(a wash), `native_qsa_rms_norm_weighted` **1.007×** (neutral; 1.117× on the iGPU and 1.795× on the K620 — a
FINDING, not a win).  The full table is in `bench/README.md`.

### Class B, batch 2: the GDN / DeltaNet MIXER's first three native fast paths (2026-10-05)

The same symbol-at-a-time discipline applies to the second batch, and it is the case the discipline was written
for.  Three native GDN kernels are now ported and gated - `native_gdn_conv_silu` (replaces `gdn_conv_step`),
`native_gdn_l2_norm` (replaces `gdn_l2_norm`) and `native_gdn_beta_gate` (replaces `gdn_beta_gate`) - each
oracled against the engine's OWN native body (`src/kernels/cuda/native_gdn_preprocess.cu`), not the legacy rule.
**`native_gdn_enabled()` still answers FALSE**, because that ONE flag also gates SIX symbols this tree has no
shader for: the remaining native GDN kernels `native_gdn_gate` / `native_gdn_step` / `native_gdn_out_norm`
(layer.cpp:297/308/324) and the three fused paths `fused_gdn_conv_l2` / `fused_gdn_ab` / `fused_gdn_step_norm`
(:250/287/322, the latter also gated on `g_fused_gdn` + `native_bf16_projections`).  `case_native_capabilities`
gains a **gdn arm** asserting the flag EQUALS "every gated symbol has a built shader" - currently false - and
that the three ported shaders exist.  Measured (`bench/README.md`): `native_gdn_conv_silu` is a win (0.694-0.938
per dispatch; 0.481-0.764 against the 2-dispatch legacy chain), while `native_gdn_l2_norm` (0.938-1.039) and
`native_gdn_beta_gate` (0.865-1.068) are WASHES - the same work per element, no algorithmic difference to win.

### Class B, batch 3: the GDN / DeltaNet MIXER's REMAINING three native fast paths (2026-10-05)

The third batch COMPLETES the six native GDN kernels.  `native_gdn_gate` (replaces `gdn_gate`),
`native_gdn_out_norm` (replaces `gdn_out_norm`) and `native_gdn_step` (replaces `gdn_step`) are each oracled
against the engine's OWN native body (`native_gdn_preprocess.cu` / `native_gdn.cu`), not the legacy rule.
`native_gdn_step` also fuses a SECOND dispatch the legacy branch runs (`scale_inplace`, layer.cpp:276), so it is
measured against that two-dispatch chain too.  **`native_gdn_enabled()` STILL answers FALSE**, because that ONE
flag also gates the THREE `fused_gdn_*` paths this tree has no shader for (`fused_gdn_conv_l2` / `fused_gdn_ab` /
`fused_gdn_step_norm` at layer.cpp:250/287/322, the latter also gated on `g_fused_gdn` +
`native_bf16_projections`).  The gdn arm of `case_native_capabilities` asserts the flag equals "every gated
symbol has a built shader" and that all SIX ported shaders exist - the strict form, deliberately: a "reachable
symbols" reading would let the flag answer true here (the fused paths need settings this backend never sets) and
could route the engine at an unported symbol when a setting changed.

Measured (`bench/README.md`): `native_gdn_step` is a WIN - **0.917 / 0.776 / 0.866** against `gdn_step` and
**0.867 / 0.771 / 0.829** against the legacy chain (Arc / Ryzen iGPU / llvmpipe) - because it moves ~a third less
state traffic (the legacy kernel's first pass stores the decayed state; the native contracts the UNDECAYED state
and folds the decay into the second pass).  `native_gdn_gate` (0.901-0.998) and `native_gdn_out_norm`
(0.974-1.000) are WASHES.  **A finding that changed the shipped kernel:** the native `step` body is one 32-lane
warp per column; with subgroup ops banned, its workgroup-per-column BARRIER-TREE rendering was built and timed at
**1.564x Arc / 8.328x iGPU / 43.501x llvmpipe** the legacy kernel, so the port ships the coalesced
one-thread-per-column serial decomposition instead, carrying the native arithmetic and the fused readout scale.
The remaining GDN holes in the map are the three `fused_gdn_*` paths and the `_multi` variants.

## Class C — the non-selected configuration (7)

Each is reached only in a branch the shipped `--native` launch does **not** select; the selected branch is a
port, and the symbol's own branch is the alternative implementation.  Naming the selecting decision is the
point: none of these is needed to run the shipped model as `setup.py` launches it.

| symbol | call site | configuration that selects it (and where it is decided) | the ported branch taken instead |
|---|---|---|---|
| `bf16_gemv` | `layer.cpp:99` | `!native_bf16_projections` in `project_bf16` (`layer.cpp:94-100`); the toggle is set by `layer_set_native_bf16(o.native_bf16)` (`generate.cpp:2286`), true under `--native` (`generate.cpp:1805`) | `bf16_gemv_fp32_mmvf` — **kernel** |
| `bf16_gemv_split` | `layer.cpp:98` | same toggle | `bf16_gemv_fp32_mmvf` — **kernel** |
| `s_gemv_q8_0_split` | `layer.cpp:172` | `!w.wants_q8k()` **and** non-native weight in `gemv_quantized` (`layer.cpp:140-172`); the shipped dense weights are native (`w.native_data`, `:142`) | `native_mmvq` — **kernel** |
| `s_gemv_q8k_split` | `layer.cpp:172`, `layer.cpp:1017` | the non-native `attn_output` branch (`layer.cpp:1016-1018`); shipped is native | `gemv_quantized` → `native_mmvq` — **kernel** |
| `qsa_attend_step` | `layer.cpp:1002` | the `else` of `g_fast_attn && !native_flash_attn_short && dump == nullptr` (`layer.cpp:978`); `g_fast_attn` defaults true (`layer.cpp:42`), `native_flash_attn_short` is not set by `--native` | `qsa_decode_attn_step` — **kernel**, `attn_decode_short` |
| `qsa_index_step` | `layer.cpp:973` | the `else` of `g_fast_select` (`layer.cpp:968`); `g_fast_select` defaults true | `qsa_block_scores` — **kernel** |
| `topk_512_step` | `layer.cpp:973` | same `g_fast_select` decision | `qsa_block_topk` — **kernel** |

## Class D — not the forward pass (22)

Reached only from the P6 verifier (`verify.cpp`), the speculative MTP drafter (`mtp.cpp`), or a tooling helper.
`verify_kernels.hpp` is the shared home of the multi-token / verify / draft kernels.

**P6 verifier only (`src/core/verify.cpp`):**
`broadcast_streams` (592, 606) · `fetch_blobs` (1053) · `gdn_ab_multi` (732) · `gdn_conv_commit` (1293, 1844) ·
`gdn_conv_l2_multi` (726, 730) · `gdn_step_norm_multi` (743, 748, 1294, 1845) · `gpu_stamp` (564, 565) ·
`native_moe_combine_multi` (1083) · `native_router_top10_multi` (920) · `ple_block_projected` (668) ·
`rebase_ptrs` (1054) · `resident_plan` (938, 943) · `wait_flag_ge` (638, 1042, 1049, 1066, 1509) ·
`wait_flag_ge_or` (1039, 1048, 1062)

**Speculative drafter only (`src/core/mtp.cpp`):**
`add_streams_broadcast` (497) · `map_ids` (644) · `moe_group_resident` (579) · `mtp_select` (710, 719, 738) ·
`row_top_prob` (643) · `window_ids` (551)

**Drafter + verifier (the multi-token window):**
`fused_gr_read_multi` (`mtp.cpp:510,571,620`; `verify.cpp:693,1142`) ·
`qsa_decode_attn_batch` (`mtp.cpp:552`; `verify.cpp:864,878`)

`gpu_stamp`/`map_ids`/`fetch_blobs`/`rebase_ptrs` are tooling/handshake helpers (profiling stamps, an
id remap, a DMA gather, a pointer rebase); `wait_flag_ge(_or)` are the host/device wait kernels the plan
refuses to translate at all.

## Corrected milestone M-A

`HANDOFF.md`, `NEXT.md`, `STATUS.md` and `plan/BACKEND-INTEGRATION.md` all read "M-A closed, `todo` = 0".  That
was measured on the qualifier-only map.  What is true:

> **M-A is NOT closed.**  The corrected map reads `168 = 53 kernel + 63 host + 52 todo`.  Of the 52, **19 are
> class A** — GPU work on the forward path of the shipped model with **no ported fallback on either branch** —
> and they are the correct M-A remainder.  Four (`B`) are capability-gated with a ported fallback and can be
> closed by *choosing the ported branch*; seven (`C`) are the non-native configuration the shipped launch does
> not select; twenty-two (`D`) are the P6 verifier / speculative drafter / tooling path, which `--spec 0`
> removes from the decode loop.  A decode-path `todo` of zero therefore requires the **19 class-A kernels**:
> the GDN / DeltaNet mixer for the 36 GDN layers, the QSA gate and indexer for the 12 QSA layers, and `gr_write`.

Derived numbers (state them, do not round): the decode-path map is `168 = 53 kernel + 63 host + 52 todo`, of
which `A = 19`, `B = 4`, `C = 7`, `D = 22`.  If "the inference half" is read to exclude the P6-verifier and
speculative-drafter machinery (class D), the forward-path remainder is `A + B + C = 30` — and of that, only the
19 class-A symbols are unavoidable.

## Corrected entry-point count for `plan/BACKEND-INTEGRATION.md` §1

§1 priced the backend from the old map as **28 kernel entry points + 12 device-crossing host rows = 40 entry
points**.  Re-derived from the corrected map:

* **ported kernel entry points: 53** (was 28) — the `kernel` rows, each an engine entry point whose body
  dispatches a shader;
* **device-crossing host rows: 18** (was 12) — the 12 §1 listed plus the bare-name rows the qualifier-only
  scan could not see: `copy_rows_from_mapped`, `copy_i32_from_mapped_unless`, `copy_indexed`,
  `doorbell_publish_res`, `doorbell_publish_value`, `coupled_draft_stage`;
* **class-A entry points still to implement: 19** — the class-A list above, GPU work with no shader, which the
  old map did not contain at all.

**So the backend's honest surface is `53 + 18 = 71` ported entry points, and the milestone owes a further 19
class-A entry points → `90` entry points** for a complete native-config backend (the old price of 40 was low by
more than half).  The class-B four are not counted: the backend satisfies them by forcing `*_enabled()` false
and dispatching the already-ported fallback.  The class-C seven are not needed for the shipped `--native`
configuration, and the class-D twenty-two are the P6/speculative/tooling path (separable via `--spec 0`).
I2–I5 are **not** re-scoped here — that is their own checkpoint; only the count and its basis change.

## Ambiguities, stated with the direction they would resolve

* **The class-D split.**  `setup.py` ships `--spec 4 --mtp …`, so the drafter and verifier DO run on the shipped
  product; `--spec` merely defaults to `0` at the binary.  I classified the drafter/verifier-only symbols D
  because they are separable from a correct first token (a `--spec 0` run is the whole model and nothing less),
  which is the brief's definition.  **If the milestone is meant to cover the shipped `--spec 4` loop, the
  `mtp.cpp`-only symbols become forward-path holes (A)** — they are the same `native_*` calls as the layer path
  and are equally unported (e.g. `mtp.cpp:515 native_rope_apply`, `:576 native_router_top10`, `:582
  moe_grouped_s2`, `:600 native_moe_combine`, `:644 map_ids`, `:710 mtp_select`).  The P6 verifier *is* blocked
  in any case: `Verifier::init` refuses unless `layer_verify_compatible()` holds, and that demands the native
  GDN and the native QSA indexer — both class A.
  **RESOLVED BY THIS BATCH (2026-10-05):** the eight `mtp.cpp`-only symbols are labelled **CLASS C** — the
  drafter config is one THIS PORT does not select (`Verifier::init` refuses under the port's answers, so the
  draft loop is never entered).  See "THE REACHABILITY AUDIT" → the class-C drafter group.
* **`native_gdn_gate`.**  Its named fallback `gdn_gate` IS ported, which would make it B; but the fallback
  *branch* is `{ gdn_beta_gate; gdn_gate; }` and `gdn_beta_gate` is `todo`, so the dodge is not sound.  Placed
  in A (the unsafe reading), as instructed.
* **`gr_write` vs the fused read.**  On the shipped config `g_fused_gr` is true (`gr_native_mmvf` true,
  `layer.cpp:42` / `generate.cpp:2284`) so the layer path would call `fused_gr_read`, not `gr_write`.  If
  `fused_gr_read` were genuinely host-side the dodge would be real and `gr_write` would be C; but
  `fused_gr.cu:1168` launches `gr_down_kernel`/`gr_up_kernel` and there is **no** gr shader, so neither branch
  is ported.  Placed in A.

## Map caveat (do NOT change the map in this commit) — a possible false NEGATIVE

The map's `todo` column is produced by `tools/port_map_lib.py`, but its **kind** column is a hand-written table
(`tools/make_port_map.py`, `TABLE`).  Two rows look mis-kinded:

* `gr_read` — `PORT-MAP.tsv` says `host` ("a workspace read"), but `src/kernels/cuda/gr.cu:344` launches
  `gr_norm_kernel` / `gr_down_kernel` / `gr_gate_kernel` / `gr_mean_kernel` / `gr_inject_kernel`.
* `fused_gr_read` — says `host` ("a workspace read"), but `src/kernels/cuda/fused_gr.cu:1168` launches
  `gr_down_kernel` / `gr_up_kernel`, and there is no `gr`/`fused_gr` shader in `ports/vulkan/shaders/`.

If they are device ops, the honest hole list is 52 + 2, not 52.  This is a **kind-table** issue, not the
bare-name rule (which produced the corrected set correctly); it is flagged here for a separate fix so this
commit leaves `PORT-MAP.tsv` byte-identical.

**No over-count found.**  The corrected rule is not inventing symbols: every one of the 52 resolves to a real
`include/strata/kernels/**` declaration (`port_map_lib.kernel_header_functions`) reached from a `src/core/`
source with the namespace in scope, and each has a real call site cited above.  The rule's one stated residual
(a local `src/core/` function shadowing a header name) does not occur here.

---

# THE KIND-TABLE FALSE NEGATIVE — `gr_read` / `fused_gr_read` (2026-10-05)

The *Map caveat* above flagged two rows as mis-kinded `host`.  This section SETTLES it by reading both entry
points and every call site (the increment the caveat asked for).

**Both are DEVICE entry points, and the `host` kind was wrong for both:**

* `gr_read` — `src/kernels/cuda/gr.cu:344-411` launches `gr_norm_kernel`, `gr_down_kernel`, `gr_gate_kernel`,
  `gr_mean_kernel`, `gr_inject_kernel` (the non-native branch), or `native_gr_rms_norm_weighted` +
  `bf16_gemv_fp32_mmvf` + `native_gr_down_silu` + `native_gr_pre_gated` (the `native_mmvf` branch).
* `fused_gr_read` — `src/kernels/cuda/fused_gr.cu:1168-1183` launches `gr_down_kernel` / `gr_up_kernel`.

`host / a workspace read` describes `gr_workspace_init` (a host hand-out) and `gr_workspace_bytes` (a size) — not
the read entry.  **So the honest device-op count is 54, not 52**, and class A is 19 without them and 20 with the
one that is on the selected branch (below).

**Which of the two is on the forward path.**  `layer.cpp:1188` and `:1328` compute
`fused = g_fused_gr && fused_gr_supported(g.n_embd, g.hc, g.hc_lr)`.  Two facts decide it:

* `fused_gr_supported()` is a pure GEOMETRY predicate (`fused_gr.cu:1164-1166`: `n_embd == 2560 && hc == 4 &&
  hc_lr == 320`), **TRUE at the artifact's geometry**.  It is NOT a capability answer — a Vulkan backend cannot
  return false without lying about the geometry.
* the selecting input is therefore **`g_fused_gr`**, set at `generate.cpp:2284` from `gr_native_mmvf`, which
  `--native` (`generate.cpp:1804`) turns on.  **The shipped launch selects the FUSED read.**

`gr_write` is reached on BOTH branches (`:1261` and `:1329` unfused; `:1195` and `:1332` fused), which is why it
is the one hyper-connection entry the port cannot dodge.  The READ has to be chosen explicitly, and this port
chooses the **unfused** member — the same shape as every other contract here (implement the legacy member, force
the flag that selects it): the backend's init calls **`gr_set_native_mmvf(false)`** and
**`layer_set_fused_gr(false)`**, so the layer takes `gr_read` at `:1255`/`:1278` and the legacy `gr_write` at
`:1261`/`:1329`, and `fused_gr_read` leaves the forward path.  This follows the port's own recorded plan
(`RUN-ON-B70.md`: `fused_gr.cu` is "deferrable"; "the engine carries its own fallbacks, so the fused gate-RoPE
family is optional").

**Consequences, stated rather than smoothed over.**  `fused_gr_read` is a real device op but a NON-SELECTED
configuration under the contract (class C — the same shape as `qsa_attend_step`); its map row is corrected from
`host` to `todo` with that reason, and it is NOT ported.  **The residual risk:** if the product must run the
SHIPPED `--native` selection bit for bit, the class-A member is `fused_gr_read`, not `gr_read`; this batch closed
the *pair* by choosing the branch, not the shipped branch.  Nothing here measures which branch the product wants.

# THE RE-DEFINED MILESTONE M-A (2026-10-05)

`todo = 0` is neither achievable nor meaningful.  The map covers EVERY kernels-namespace symbol the decode path
reaches — including the `native_*` siblings of ported legacy members, the fused alternatives a flag removes, and
the verifier/MTP/tooling helpers — so a `todo` column of zero would require porting ~45 more symbols of which
**zero** are on the forward path the port runs.  The milestone is therefore re-defined over the class-A set:

> **M-A (re-defined).**  Every kernels-namespace symbol the forward path of the shipped model reaches — **on the
> branch the backend's capability contract selects** — has a shader and a gated case.  The contract:
>
> | the backend answers | which selects |
> |---|---|
> | `native_gdn_enabled() == false` | the legacy GDN / DeltaNet mixer (`gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`, `gdn_gate`, `gdn_step`, `gdn_out_norm`) and removes the three `fused_gdn_*` paths |
> | `native_qsa_enabled() == false` | the legacy QSA gate (`qsa_gate_apply_f32`) and the class-B weighted-RMS-norm fallback |
> | `native_qsa_indexer_enabled() == false` | the legacy QSA indexer append (`indexer_key_append`) |
> | `native_rope_enabled() == false` | the ported NeXo RoPE (class B) |
> | `native_router_enabled() == false` | the ported generic top-10 (class B) |
> | `native_moe_combine_enabled() == false` | the ported weighted combine (class B) |
> | `gr_set_native_mmvf(false)` **and** `layer_set_fused_gr(false)` | the legacy unfused read (`gr_read`) and write (`gr_write`) |
>
> **SUPERSEDED IN PART (batch 5, 2026-10-05):** the GDN and QSA rows above are no longer the current answers —
> `native_gdn_enabled()` (batch 4) and `native_qsa_enabled()` (batch 5) now answer **TRUE**, and
> `native_qsa_indexer_enabled()` is now actually defined (`false`).  The CURRENT answers, and the reachability of
> every remaining `todo` row under them, are in "THE REACHABILITY AUDIT" at the end of this file.
>
> **Deliberately out of scope, with the reason:**
> * **class B (4)** — capability-gated with a PORTED fallback the backend selects by answering the check false.
> * **class C (8)** — a configuration the launch does not select, whose selected branch IS ported: `bf16_gemv`,
>   `bf16_gemv_split`, `s_gemv_q8_0_split`, `s_gemv_q8k_split`, `qsa_attend_step`, `qsa_index_step`,
>   `topk_512_step`, and `fused_gr_read` (above).
> * **class D (22)** — the P6 verifier, the speculative drafter and the tooling helpers; separable from a correct
>   first token, since a `--spec 0` run is the whole model and nothing less.
> * **the capability-off siblings (11)** — the `native_*` members and the three `fused_gdn_*` paths: reachable
>   only if a capability answers **true**, which the contract forbids.

Counts after the closure batch: **`168 = 62 kernel + 61 host + 45 todo`**, and the 45 decompose as
`11 capability-off + 4 B + 8 C + 22 D`.  **Class-A remaining: 0.**

**IS M-A (re-defined) MET?  YES — with two soft edges stated rather than hidden.**

1. **The shipped `--spec 4` loop.**  `setup.py:4224-4227` writes `--spec 4 --mtp`, so the MTP drafter DOES run on
   the shipped product and its `mtp.cpp`-only symbols (`map_ids`, `mtp_select`, `moe_group_resident`,
   `row_top_prob`, `window_ids`, `add_streams_broadcast`, `fused_gr_read_multi`, `qsa_decode_attn_batch`) are
   **CLASS C** (this batch's labelling, see the REACHABILITY AUDIT above): they are selected only by the
   `--spec 4 --mtp` draft loop, which the PORT does not enable — `Verifier::init` refuses because
   `layer_verify_compatible()` needs `g_fused_gr` (forced false), `native_qsa_indexer_enabled()` (answered false)
   and `native_bf16_projections` (unpinned).  The P6 verifier is blocked in any case, so the port's selected
   branch is a `--spec 0` run.
2. **`gr_read` vs `fused_gr_read`** (above): the pair is closed by CONTRACT, not by matching the shipped launch's
   own selection.

## The class-A implementations, and where each landed

| # | symbol | branch | shader(s) | state |
|---|---|---|---|---|
| 1 | `gdn_conv_step` | legacy | `gdn_conv_step` | LANDED |
| 2 | `gdn_l2_norm` | legacy | `gdn_l2_norm` | LANDED |
| 3 | `gdn_beta_gate` | legacy | `gdn_beta_gate` | LANDED |
| 4 | `gdn_gate` | legacy | `gdn_gate` | LANDED (pre-existing) |
| 5 | `gdn_step` | legacy | `gdn_step` | LANDED |
| 6 | `gdn_out_norm` | legacy | `gdn_out_norm` | LANDED |
| 7 | `qsa_gate_apply_f32` | legacy | `qsa_gate_apply_f32` | LANDED |
| 8 | `indexer_key_append` | legacy | `indexer_key_append` | **LANDED 2026-10-05** |
| 9 | `gr_write` | both | `gr_write` | **LANDED 2026-10-05** |
| 10 | `gr_read` | unfused | `gr_norm gr_down gr_gate gr_mean gr_inject` | **LANDED 2026-10-05** |

`gr_read` is the tenth because the pair's unfused member is what the GR contract selects; `fused_gr_read` leaves
the path and is `todo` with that reason.

---

# THE REACHABILITY AUDIT — every `todo` row, against the port's CURRENT capability answers (2026-10-05)

This is a **report**, not code. Its purpose is to convert the map's `todo` count from a number into a DECISION
LIST: for each remaining row, IS IT REACHABLE on the shipped model's decode path **under the capability answers
the backend actually gives today**, and if so, through which call site and which flag chain.

**The answers audited against** (the state after batch 4's `native_gdn_enabled() == true` flip and this batch's
QSA work):

| the backend answers | value | selects |
|---|---|---|
| `native_gdn_enabled()` | **true** (batch 4) | the native GDN kernels + the three `fused_gdn_*` paths |
| `native_qsa_enabled()` | **true** (this batch) | `native_qsa_rms_norm_weighted` + `native_qsa_gate_apply` |
| `native_qsa_indexer_enabled()` | **false** (this batch; before it, UNANSWERED — see the hole below) | the legacy `indexer_key_append` |
| `native_rope_enabled()` / `native_router_enabled()` / `native_moe_combine_enabled()` | true | their native member |
| `gr_set_native_mmvf(false)` + `layer_set_fused_gr(false)` | forced | `gr_read` + the legacy `gr_write`; removes `fused_gr_read` |

Host SETTINGS that gate paths but are not capability answers, with their defaults and whether the shipped launch
(`setup.py:4224-4227` writes `--pack … --native <shard> --spec 4 --spec-min-p 0.5 --mtp <rt>`) changes them:
`g_fused_gdn` **true** (layer.cpp:42, default); `g_fast_attn`/`g_fast_select`/`g_shared_early`/`g_publish_kernel`
**true** (layer.cpp:42); `g_fused_gr` **false**, and the GR contract forces it false; `native_bf16_projections`
**false** by default, set true by `--native` (`generate.cpp:1805-1807` → `:2286`); `native_flash_attn_short`
false (not set by `--native`).

**The one hole found and FIXED in this batch: `native_qsa_indexer_append`.**

> `src/core/layer.cpp:944` selects it with `if (native_qsa_indexer_enabled()) native_qsa_indexer_append(...)`,
> `else indexer_key_append(...)`; the indexer runs on the MAIN forward path of **all 12 QSA layers, every token**
> (the append at `:917-948` precedes the selection at `:964-1005`).  The shipped `--native` launch sets
> `o.native_qsa_indexer = true` (`generate.cpp:1807`) and calls `native_qsa_indexer_set_enabled(true)`
> (`:2294`) — and `native_qsa_indexer_append` has **no shader** in this tree.  The port's contract has ALWAYS
> named the required answer (`native_qsa_indexer_enabled() == false`, `HANDOFF.md`, `NEXT.md`, the milestone
> table), but **no Vulkan definition of the getter or the setter existed** — so the contract was a claim without
> a body, and the shipped option would have selected the unported symbol (or failed to link).  This is the same
> defect class as batch 4's flag flip: **a capability answer is a ROUTING decision.**
>
> **The fix** (this batch): `native_caps_vk.cpp` now DEFINES `native_qsa_indexer_set_enabled` (a no-op) and
> `native_qsa_indexer_enabled() == false`, which selects the ported legacy `indexer_key_append`.  The row stays
> `todo` (the native append is still unported) but it is now **unreachable** under the contract — the same shape
> as the GDN/QSA pairs.

**THE AUDIT TABLE.**  "Reachable?" is evaluated under the answers above.  A reachable-but-unported row is a HOLE
(DONE = closed this batch; QUEUE = the ordered remaining work); "no" rows state the deciding condition so the
answer can be re-checked when a default changes.

| # | symbol | class (map) | reachable now? | call site + flag chain that decides it |
|---|---|---|---|---|
| 1 | `native_qsa_gate_apply` | todo → **kernel** | — **DONE this batch** | `layer.cpp:1010` `if (native_qsa_enabled())`; shader landed, flag flipped |
| 2 | `native_qsa_indexer_append` | todo | **was a HOLE → DONE this batch** | `layer.cpp:944` `if (native_qsa_indexer_enabled())`; the flag is now ANSWERED false (was unimplemented), selecting the ported `indexer_key_append` (`:948`) |
| 3 | `bf16_gemv` | todo → **kernel** | — **DONE this batch** | `layer.cpp:99` in `project_bf16`: `native_bf16_projections ? bf16_gemv_fp32_mmvf : (split ? bf16_gemv_split : bf16_gemv)`. The setting (`layer.cpp:91`) DEFAULTS **false** and the port pins it nowhere, so the DEFAULT selects the UNPORTED member; the shipped `--native` (`generate.cpp:1805` → `:2286`) sets it true. **Porting BOTH members closes the edge for either value** — `case_bf16_gemv` / `case_bf16_gemv_split`. |
| 4 | `bf16_gemv_split` | todo → **kernel** | — **DONE this batch** | same chain; `split=true` is the `gdn_layer` alpha/beta call (`layer.cpp:291-292`) and the router logits (`:367`) |
| 5 | `s_gemv_q8_0_split` | todo (C) | **no** | `layer.cpp:172` in `gemv_quantized`, reached only when `w.native_data == nullptr`; the shipped dense weights ARE native (`:142` → `native_mmvq`) |
| 6 | `s_gemv_q8k_split` | todo (C) | **no** | `layer.cpp:172` (same) and `layer.cpp:1017` (`!w_attno->native_data`); shipped `attn_output` is native |
| 7 | `qsa_index_step` | todo (C) | **no** | `layer.cpp:973` `else` of `if (g_fast_select)`; `g_fast_select` defaults true → the ported `qsa_block_scores`/`qsa_block_topk` |
| 8 | `topk_512_step` | todo (C) | **no** | same `g_fast_select` decision (`layer.cpp:973`) |
| 9 | `qsa_attend_step` | todo (C) | **no** | `layer.cpp:1002` `else` of `if (g_fast_attn && !native_flash_attn_short && dump == nullptr)`; defaults true/false/null → `qsa_decode_attn_step` (CORRECTED 2026-10-05: this row said "the PORTED `qsa_decode_attn_step`" — but that symbol had NO shader until the correction below; the selected branch was a HOLE) |
| 10 | `fused_gr_read` | todo (C) | **no** | `layer.cpp:1253`/`:1276` `fused = g_fused_gr && fused_gr_supported(...)`; the GR contract forces `layer_set_fused_gr(false)` |
| 11 | `add_streams_broadcast` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:497` in `MtpDrafter::record_forward`, unconditional; the drafter runs when `o.spec > 0` (`generate.cpp:7796`) and the shipped launch sets `--spec 4 --mtp` |
| 12 | `fused_gr_read_multi` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:510`/`:571`/`:620`, unconditional; also `verify.cpp:693`/`:1142` (verifier only) |
| 13 | `qsa_decode_attn_batch` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:552`, unconditional; also `verify.cpp:864`/`:878` |
| 14 | `moe_group_resident` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:579`, unconditional |
| 15 | `row_top_prob` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:643`, in the drafter's sampling tail (`coupled_rec_` false branch) |
| 16 | `map_ids` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:644` `if (sub)` (a draft-vocabulary subset file exists — the shipped `--mtp` writes one) |
| 17 | `mtp_select` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:710`/`:719`/`:738` in `MtpDrafter::draft` |
| 18 | `window_ids` | todo (D) | **YES — under `--spec 4 --mtp`** | `mtp.cpp:551` `if (window_ > 0)` |
| 19 | `broadcast_streams` | todo (D) | **no** | `verify.cpp:592`/`:606`; the P6 verifier cannot init (below) |
| 20 | `fetch_blobs` | todo (D) | **no** | `verify.cpp:1053` |
| 21 | `gdn_ab_multi` | todo (D) | **no** | `verify.cpp:732` |
| 22 | `gdn_conv_commit` | todo (D) | **no** | `verify.cpp:1293`/`:1844` |
| 23 | `gdn_conv_l2_multi` | todo (D) | **no** | `verify.cpp:726`/`:730` |
| 24 | `gdn_step_norm_multi` | todo (D) | **no** | `verify.cpp:743`/`:748`/`:1294`/`:1845` |
| 25 | `gpu_stamp` | todo (D) | **no** | `verify.cpp:564`/`:565` (guarded by `prof_on_`/`trace_m_`, both off) |
| 26 | `native_moe_combine_multi` | todo (D) | **no** | `verify.cpp:1083` |
| 27 | `native_router_top10_multi` | todo (D) | **no** | `verify.cpp:920` |
| 28 | `ple_block_projected` | todo (D) | **no** | `verify.cpp:668` |
| 29 | `rebase_ptrs` | todo (D) | **no** | `verify.cpp:1054` |
| 30 | `resident_plan` | todo (D) | **no** | `verify.cpp:938`/`:943` |
| 31 | `wait_flag_ge` | todo (D) | **no** | `verify.cpp:638`/`:1042`/`:1049`/`:1066` |
| 32 | `wait_flag_ge_or` | todo (D) | **no** | `verify.cpp:1039`/`:1048`/`:1062` |

**THE VERIFIER IS UNREACHABLE, and this is WHY (it is one conjunction, checked, not assumed).**
`Verifier::init` refuses unless `layer_verify_compatible()` holds (`layer.cpp:476-491`), which requires
`native_bf16_projections` (a setting, default **false**; the shipped `--native` sets it true but the verifier is
NOT the forward path), `g_fused_gr` (**false**, and the GR contract forces it false), `g_fused_gdn &&
native_gdn_enabled()` (**true now**), `g_fast_attn && !native_flash_attn_short` (true), `g_fast_select` (true),
AND `native_qsa_indexer_enabled()` (**false** — the indexer is unported).  So rows 19-32 are off every path the
port runs.  **The batch-4 flip did not change this**: it satisfied one term of that conjunction, and three other
terms keep it false.

**THE REACHABLE-BUT-UNPORTED QUEUE, re-read after this batch.**  Both forward-path holes are now CLOSED:
`native_qsa_indexer_append` by the FLAG (batch 5) and the two BF16-projection rows by PORTING BOTH MEMBERS (this
batch).  What remains is the drafter's eight, and this batch LABELS them **class C**:

1. **The speculative drafter's eight (`mtp.cpp`) → CLASS C.**  `add_streams_broadcast` (497) →
   `fused_gr_read_multi` (510/571/620) → `window_ids` / `qsa_decode_attn_batch` (551/552) → `moe_group_resident`
   (579) → `row_top_prob` / `map_ids` (643/644) → `mtp_select` (710/719/738).  They run ONLY when the engine takes
   the `--spec 4 --mtp` DRAFT loop — a configuration THE PORT DOES NOT SELECT, because the loop needs
   `Verifier::init` to succeed and `layer_verify_compatible()` (`layer.cpp:476-491`) demands a conjunction the
   port's contract leaves false: `g_fused_gr` (forced FALSE by the GR contract), `native_qsa_indexer_enabled()`
   (answered FALSE — the native append is unported) and `native_bf16_projections` (a setting the port does not
   pin).  **The flag chain that WOULD enable them is `--spec 4 --mtp` AND a verifier-compatible native stack; the
   selected branch (a `--spec 0` / non-drafting run) is the whole model.**  That is exactly the class-C shape — a
   NON-SELECTED configuration whose selected branch IS ported.  The port's own contract is what keeps the
   verifier out, so the drafter is not a branch this backend takes.  (`fused_gr_read_multi`'s ported siblings,
   `gr_read` + `gr_write`, are already in the tree; the other seven are new kernels.)

**Counts, stated, not rounded:** of the 32 rows audited, **3 are now `kernel`** (`native_qsa_gate_apply`,
`bf16_gemv`, `bf16_gemv_split`); **8 are CLASS C** (the drafter rows above, reachable only under the non-selected
`--spec 4 --mtp` config); `native_qsa_indexer_append` stays `todo` but is **flag-closed**; and **20 are
unreachable** under the current answers, each with its deciding condition named in the table above.  3 + 8 + 1 +
20 = 32.

# THE BF16-PROJECTION PAIR — the second soft edge, CLOSED (2026-10-05)

The REACHABILITY AUDIT above raised the `bf16_gemv` / `bf16_gemv_split` SOFT EDGE and asked for it to be pinned.
This batch RESOLVED it by porting BOTH ENGINE MEMBERS, which closes the edge for EITHER value of the setting:

| # | symbol | shader | case | call sites (split) |
|---|---|---|---|---|
| 1 | `bf16_gemv` | `bf16_gemv.comp` — ONE WORKGROUP per output row through the barrier tree (the engine's `bf16_gemv` call sites are n_out = 128 and 512, where CUDA takes its WARP kernel; subgroup ops are banned here) | `case_bf16_gemv` (4 arms: 2560×512, 2560×128, 128×64, 2×1) | `layer.cpp:918`/`:962` (QSA indexer projections, `split=false`) |
| 2 | `bf16_gemv_split` | the SAME `bf16_gemv.comp` — the CUDA's split/warp kernels differ from `bf16_gemv` only in parallelism STRATEGY, and the engine calls this entry point with `threads_per_row = 32` (a warp), so the port renders it identically and DROPS `threads_per_row`.  Two map rows, one shader (the `bf16_gemv_fp32_mmvf`/`_cols` precedent). | `case_bf16_gemv_split` (4 arms: 2560×512, 2560×48, 64×32, 2×1) | `layer.cpp:291`/`:292` (GDN alpha/beta), `:367` (router logits), all `split=true` |

**A MEASUREMENT THAT CHANGED THE SHIPPED KERNEL.**  The naive ONE-THREAD-PER-ROW decomposition was built and
benchmarked first, and at the engine's shapes it is **14–18x slower** than the workgroup form (Arc `split/serial`
0.073 at n_out=512 / 0.057 at 48; Ryzen iGPU 0.103/0.055) because it is uncoalesced — the CUDA's own comment says
so, and CUDA uses the naive path only below n_out=64, which this engine never does for `bf16_gemv`.  The port
therefore SHIPS the workgroup-per-row rendering; the naive variant is not in the tree.

**THE CHAIN, in full.**  `project_bf16` (`src/core/layer.cpp:94-100`) is the ONLY caller of both:

    native_bf16_projections ? bf16_gemv_fp32_mmvf : (split ? bf16_gemv_split : bf16_gemv)

The setting `native_bf16_projections` (`layer.cpp:91`) DEFAULT **false**; `layer_set_native_bf16`
(`layer.cpp:174`) writes it, called from `generate.cpp:2286` with `o.native_bf16`, which `--native` sets true
(`generate.cpp:1805`) and `--native-bf16` sets independently.  The **port pins the setting NOWHERE**, and the
native-capability contract (`vulkan/src/kernels/native_caps_vk.cpp`) does not answer it — it is a host setting,
not a capability getter.  So the honest reading is: with the flag off, the layer dispatches the UNPORTED member
(`bf16_gemv_split` at three call sites, `bf16_gemv` at two), on the main forward path.  That is the SAME shape as
the `native_qsa_indexer_append` hole batch 5 fixed — the difference being that here the flag has a real host
definition and both values are now covered.

**Why PORTING rather than pinning.**  `layer_set_native_bf16` lives in engine host code
(`src/core/layer.cpp`), which the port does NOT fork; the backend cannot redefine it, and pinning it true would
have been a claim about a setting the port does not own.  Porting both members is stronger: whichever way the
setting answers, the engine reaches a shader.

**The oracle and the bound.**  Both kernels compute the same rule (`bf16_gemv.hpp`: the split variant is "NOT
bit-identical ... the partial sums are added in a different order") on the same fixture: `y[o] = Σ
f32_from_bf16(x[i]) · f32_from_bf16(w[o*n_in+i])`, every product exact in f32.  The comparison is the port's
TERMS-derived `gemv_bound` (never a relative tolerance — with cancellation `Σ|terms|` dominates `|result|`), the
weight row read as 32-bit PAIRS (element 2p LOW, 2p+1 HIGH).  Fixture: row 0 is the LAYOUT PROBE (low halves
~1e3, high halves ~1e-3, so a halves swap is O(1)); row 1 is ALL-ZERO (the `gemv_bound` floor is load-bearing);
four arms, including a non-square `128×64` and the degenerate `2×1`.  Both margins (the halves swap, an
off-by-one row) are asserted host-side to MOVE the oracle, and the output buffer carries a 0x5E guard region
behind a surplus dispatched group.

**Falsified, and both bit:** `bf16-gemv-swap-halves` → `FAIL bf16_gemv n_in=2560 n_out=512 4/515 worst 1.54e+05`;
`bf16-gemv-row-base` (the weight row indexed by the OUTPUT stride) → `FAIL bf16_gemv n_in=2560 n_out=512 4/515
worst 2.62e+34`.

**Measured (bench/README.md; the ported `bf16_gemv` vs the ported native `bf16_gemv_fp32_mmvf`, same
workgroup-per-row decomposition, only the activation precision differs → a WASH is expected):** native/bf16_gemv
**1.000** (Arc 512), **1.006** (Arc 48), **0.996 / 0.997** (Ryzen iGPU), **0.933 / 0.956** (llvmpipe).

**Map:** `168 = 76 kernel + 61 host + 31 todo` → **`168 = 78 kernel + 61 host + 29 todo`** (both rows
`todo → kernel`); `make_port_map.py` regenerates `PORT-MAP.tsv` byte-identically.  The eight drafter rows keep
kind `todo` in the TSV (its kind vocabulary is `kernel`/`host`/`todo`) but carry a `class C` reason string and
are classified in the AUDIT above, so the map no longer reports them as open forward-path work.

# THE THIRD HOLE — `qsa_decode_attn_step`, the DEFAULT decode attention (CORRECTED 2026-10-05)

The REACHABILITY AUDIT's row 9 recorded `qsa_attend_step` as the class-C member and *"the ported
`qsa_decode_attn_step`"* as the SELECTED branch of the fast-attention decision.  **That was wrong: the selected
branch had no shader, and it is the engine's DEFAULT attention.**  Settled from the engine's own code:

* **The call site** `src/core/layer.cpp:978-980`: `if (g_fast_attn && !native_flash_attn_short && dump == nullptr)
  { ... qsa_decode_attn_step(...); }`.
* **Every flag, its default, and what the shipped launch sets:** `g_fast_attn` **true** (`layer.cpp:42`; the
  launch writes `layer_set_fast_attn(!o.no_fast_attn)` at `generate.cpp:2280`, and `o.no_fast_attn` defaults
  **false**); `native_flash_attn_short` **false** (`layer.cpp:92`; `o.native_flash_attn_short` defaults **false**
  at `generate.cpp:272` and only `--native-flash-attn-short` sets it, NOT `--native`); `dump` is the caller's
  argument and the decode path passes **nullptr** (`bb.dump`, `layer.cpp:1259`).
* All three are the shipped defaults, so the `if` branch — `qsa_decode_attn_step` — IS reached.
* **Its contract** (`include/strata/kernels/qsa_decode_attn.hpp`) reads the KV **POOLS** through the PAGE TABLE
  with an int8/q4 option and a scratch — NOT the gathered `[cap,2,256]` f16 WINDOW of `attn_decode_short` (which
  its own header states is `native_flash_attn_short_step`).  PORT-MAP's `qsa_decode_attn_step → attn_decode_short`
  row was therefore a false mapping.

**Both are corrected.**  `ports/vulkan/shaders/qsa_decode_attn.comp` ports the pools-through-the-page-table
kernel (f16 pools — the shipped `--kv fp16` default — and int8; q4_0/K8V4 refused), wired as
`strata::kernels::qsa_decode_attn_step` in `vulkan/src/kernels/qsa_vk.cpp` plus the `host` row
`qsa_decode_attn_scratch_floats`; proved by `case_qsa_decode_attn` (bitwise wrapper==shader and both against a
double transcription of the engine's rule) and falsified by
`gates/inject-verify.sh qsa-decode-attn-drop-kv-head`.  PORT-MAP's row now reads
`qsa_decode_attn_step  kernel  qsa_decode_attn`.  The one-layer-body link moves `94 → 91` raw / `35 → 33`
full-signature / `33 → 31` name-only; the attention/QSA/MoE/GR/PLE/rope group falls **27 → 25**.  Full detail in
`NEXT.md`'s top section.

**The lesson, stated the way the previous two holes stated it:** a capability answer is a ROUTING decision, and
so is a class label — **the audit asserted a symbol was ported without checking that a shader existed for its
CONTRACT**, and the wrong PORT-MAP pairing (`attn_decode_short`) is what made the assertion look true.  A
`kernel` row's named shader must be the kernel the symbol's own header describes, not merely a shader whose file
name is `*attn*`.

