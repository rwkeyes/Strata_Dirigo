# Decode-path triage — the 52 `todo` rows, from the engine's own sources

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
