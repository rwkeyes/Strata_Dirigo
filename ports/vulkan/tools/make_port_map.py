#!/usr/bin/env python3
"""Write ports/vulkan/PORT-MAP.tsv: the DECODE PATH's kernels-namespace symbols, each classified as

    kernel  GPU work this port has, naming the shader(s)
    host    the engine's own host side (a size, a check, a table, a sync primitive) - no dispatch to port
    todo    GPU work this port has NOT done - the honest hole list

The table is checked, not decorative: gates/run_gate.sh fails if a symbol is invented (absent from the engine's
sources), if a `kernel` row names a shader that is not built, if a built shader is named by no row, or if src/core/
reaches a kernels-namespace symbol the table does not mention.  The symbol set comes from tools/port_map_lib.py,
which finds the qualified AND the BARE (using-namespace) call sites alike - the blind spot this file used to have.
"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import port_map_lib as lib

ROOT = pathlib.Path(__file__).resolve().parents[3]      # .../strata-vulkan-wt
syms = sorted(lib.decode_path_symbols())

# kind, shader(s) or the reason it is host-side.
TABLE = {
    # ---- ported: the GPU work has a shader in this tree ----
    'add_inplace':                    ('kernel', 'add'),
    'bf16_gemv_fp32_mmvf':            ('kernel', 'bf16_mmvf_f32'),
    'bf16_gemv_fp32_mmvf_cols':       ('kernel', 'bf16_mmvf_f32'),
    'bf16_gemv_fp32_mmvf_multi':      ('kernel', 'bf16_mmvf_f32_multi'),
    'coupled_draft_sample':           ('kernel', 'coupled_penalize coupled_sample'),
    'cvec_apply':                     ('kernel', 'cvec_apply'),
    'embedding_gather':               ('kernel', 'embedding_gather'),
    'embedding_gather_dev':           ('kernel', 'embedding_gather'),
    'f32_to_bf16_bulk':               ('kernel', 'f32_to_bf16'),
    'fwht256_inplace_cuda':           ('kernel', 'fwht256'),
    'gather_rows':                    ('kernel', 'gather_rows'),
    'gdn_gate':                       ('kernel', 'gdn_gate'),
    # The GDN (DeltaNet) mixer's LEGACY branch, landed under the branch policy `native_gdn_enabled() == false`
    # (see NEXT.md): the layer's own sequence reaches conv -> l2_norm -> beta_gate -> gate -> step -> out_norm.
    'gdn_conv_step':                  ('kernel', 'gdn_conv_step'),
    'gdn_l2_norm':                    ('kernel', 'gdn_l2_norm'),
    'gdn_beta_gate':                  ('kernel', 'gdn_beta_gate'),
    # The next two COMPLETE the legacy GDN mixer chain (conv -> l2_norm -> beta/gate -> step -> out_norm): the
    # delta-rule state update and the closing norm.  Same branch policy, `native_gdn_enabled() == false`.
    'gdn_step':                       ('kernel', 'gdn_step'),
    'gdn_out_norm':                   ('kernel', 'gdn_out_norm'),
    # The hyper-connection (GR) pair, under the branch policy `gr_set_native_mmvf(false)` + `layer_set_fused_gr(false)`
    # (the same shape as `native_gdn_enabled() == false`; see DECODE-PATH-TRIAGE.md and NEXT.md): the LEGACY
    # UNFUSED read (five stages) and the write.  `fused_gr_read` is a DEVICE op too - it launches
    # gr_down_kernel/gr_up_kernel - but it LEAVES the forward path under that policy, so it is `todo` with a
    # stated reason rather than `host`.  The old `host`/`a workspace read` rows for gr_read and fused_gr_read were
    # FALSE NEGATIVES: a "workspace read" is gr_workspace_init/bytes, a host hand-out, not the read entry point.
    'gr_read':                        ('kernel', 'gr_norm gr_down gr_gate gr_mean gr_inject'),
    'gr_write':                       ('kernel', 'gr_write'),
    # The QSA indexer pair's LEGACY member: `native_qsa_indexer_enabled() == false` - the indexer's OWN check,
    # separate from `native_qsa_enabled()`, and the one `layer_verify_compatible()` reads.
    'indexer_key_append':             ('kernel', 'indexer_key_append'),
    'iq_dequant_f32':                 ('kernel', 'iq_dequant_f32'),
    'iq_embed_rows':                  ('kernel', 'iq_embed_rows'),
    'kv_append_q4_step':              ('kernel', 'kv_q4_append'),
    'kv_append_q8_step':              ('kernel', 'kv_q8_append'),
    'kv_append_step':                 ('kernel', 'kv_f16_append'),
    'kv_gather_q4_step':              ('kernel', 'kv_q4_gather'),
    'kv_gather_q8_step':              ('kernel', 'kv_q8_gather'),
    'kv_gather_step':                 ('kernel', 'kv_f16_gather'),
    'moe_combine':                    ('kernel', 'moe_combine_f32 moe_combine_f64'),
    'moe_grouped_s2':                 ('kernel', 's2expert_gu_grouped s2expert_swiglu quantize_q8_0 s2expert_down_grouped'),
    'moe_hit_add':                    ('kernel', 'moe_hit_add'),
    'moe_hit_grouped_s2':             ('kernel', 's2expert_gu s2expert_swiglu quantize_q8_0 s2expert_down'),
    'moe_hit_grouped_s2_cpu_order':   ('kernel', 's2_gemv_q8 scalar_gate_f32 scalar_gate_f64 moe_combine_f32 moe_combine_f64'),
    'moe_hit_grouped_s2_dev':         ('kernel', 's2_gemv_q8 scalar_gate_f32 scalar_gate_f64 moe_combine_f32 moe_combine_f64'),
    'moe_hit_select':                 ('kernel', 'moe_hit_select'),
    'native_expert_grouped':          ('kernel', 'native_gu_iq2s native_down_iq4nl s2expert_gu s2expert_down s2expert_swiglu'),
    'native_flash_attn_short_step':   ('kernel', 'attn_decode_short'),
    'native_mmvq':                    ('kernel', 'iq1m_mmvq iq2s_mmvq iq3s_mmvq iq3xxs_mmvq iq4nl_mmvq iq4xs_mmvq'),
    'native_q5_k_f32':                ('kernel', 'native_q5_k_f32'),
    # THE PERFORMANCE TIER, class B: the native fast paths, each replacing the legacy kernel already ported.
    # The Vulkan backend answers the checks they belong to itself (vulkan/src/kernels/native_caps_vk.cpp).
    'native_rope_apply':              ('kernel', 'native_rope_apply'),        # replaces rope_neox_apply
    'native_router_top10':            ('kernel', 'native_router_top10'),      # replaces router_top10
    'native_moe_combine':             ('kernel', 'native_moe_combine'),       # replaces moe_combine
    # native_qsa_rms_norm_weighted is ported and gated, and `native_qsa_gate_apply` now is too (this batch), so
    # the backend answers `native_qsa_enabled()` TRUE - both symbols its one flag gates have a shader.  The row
    # for the gate is `kernel` below; see vulkan/src/kernels/native_caps_vk.cpp and case_native_capabilities.
    'native_qsa_rms_norm_weighted':   ('kernel', 'native_qsa_rms_norm_weighted'),   # replaces rms_norm_weighted
    'native_qsa_gate_apply':          ('kernel', 'native_qsa_gate_apply'),          # replaces qsa_gate_apply_f32
    # THE PERFORMANCE TIER, class B, batch 2: the first three native GDN / DeltaNet MIXER kernels (the mixer is
    # 36 of the model's 48 layers).  Each replaces a legacy kernel already ported (`gdn_conv_step` / `gdn_l2_norm`
    # / `gdn_beta_gate`), and all are gated by the SAME `native_gdn_enabled()` flag - which stays FALSE because
    # the flag ALSO gates the three unported `fused_gdn_*` paths.  See vulkan/src/kernels/native_caps_vk.cpp
    # and case_native_capabilities.
    'native_gdn_conv_silu':           ('kernel', 'native_gdn_conv_silu'),      # replaces gdn_conv_step (fused + SiLU)
    'native_gdn_l2_norm':             ('kernel', 'native_gdn_l2_norm'),        # replaces gdn_l2_norm
    'native_gdn_beta_gate':           ('kernel', 'native_gdn_beta_gate'),      # replaces gdn_beta_gate
    # THE PERFORMANCE TIER, class B, batch 3: the REMAINING three native GDN / DeltaNet MIXER kernels,
    # COMPLETING the six.  Each replaces a legacy kernel already ported (`gdn_gate` / `gdn_out_norm` / `gdn_step`)
    # and is gated by the same flag, which still answers FALSE because the three `fused_gdn_*` paths remain
    # unported (the flag stays false until EVERY symbol it gates has a shader).
    'native_gdn_gate':                ('kernel', 'native_gdn_gate'),           # replaces gdn_gate
    'native_gdn_out_norm':            ('kernel', 'native_gdn_out_norm'),       # replaces gdn_out_norm
    'native_gdn_step':                ('kernel', 'native_gdn_step'),           # replaces gdn_step (folds scale_inplace)
    'native_quantize_q8_1':           ('kernel', 'quantize_q8_1'),
    'ple_block':                      ('kernel', 'ple_bcast ple_conv ple_gate ple_gnorm'),
    'ple_history_advance':            ('kernel', 'ple_history_advance'),
    'qsa_block_scores':               ('kernel', 'qsa_block_scores'),
    'qsa_block_topk':                 ('kernel', 'qsa_block_topk'),
    # The QSA gate pair's LEGACY member: `native_qsa_enabled() == false` (the QSA half of the branch policy).
    'qsa_gate_apply_f32':             ('kernel', 'qsa_gate_apply_f32'),
    'qsa_decode_attn_step':           ('kernel', 'attn_decode_short'),
    'quantize_q8_':                   ('kernel', 'quantize_q8_0 quantize_q8_1 quantize_q8_K quantize_q8_0_scaled'),
    'quantize_q8_0':                  ('kernel', 'quantize_q8_0'),
    'quantize_q8_0_scaled':           ('kernel', 'quantize_q8_0_scaled'),
    'quantize_q8_1_rows':             ('kernel', 'quantize_q8_1'),
    'quantize_q8_K':                  ('kernel', 'quantize_q8_K'),
    'rms_norm_weighted':              ('kernel', 'rms_norm'),
    'rope_neox_apply':                ('kernel', 'rope_neox'),
    'router_top10':                   ('kernel', 'router_top10_f32 router_top10_f64'),
    's2_gemv_q8':                     ('kernel', 's2_gemv_q8'),
    's_gemv_split_async':             ('kernel', 's_gemv_q8_split'),
    'sample_tokens':                  ('kernel', 'sampler_greedy sampler_kernel sampler_kernel_f32 sampler_split'),
    'scale_inplace':                  ('kernel', 'scale'),
    'scatter_rows_f32':               ('kernel', 'scatter_rows_f32'),
    'shared_expert':                  ('kernel', 's2expert_gu s2expert_swiglu quantize_q8_0 s2expert_down moe_combine_f32 scalar_gate_f32'),
    'shared_expert_multi':            ('kernel', 's2expert_gu s2expert_swiglu quantize_q8_0 s2expert_down moe_combine_f32 scalar_gate_f32'),
    'silu_inplace':                   ('kernel', 'silu_f32'),
    # ---- the engine's own host side: no dispatch for the port to supply ----
    'build_rope_table':               ('host', 'the engine builds the table; rope_neox is the kernel that reads it'),
    'copy_from_mapped':               ('host', 'a mapped-buffer copy'),
    'copy_i32_from_mapped':           ('host', 'a mapped-buffer copy'),
    'copy_i32_from_mapped_unless':    ('host', 'a mapped-buffer copy'),
    'copy_indexed':                   ('host', 'a device-indexed copy'),
    'copy_or_zero_from_mapped':       ('host', 'a mapped-buffer copy'),
    'copy_rows_from_mapped':          ('host', 'a mapped-buffer copy'),
    'coupled_draft_scratch_bytes':    ('host', 'a size'),
    'coupled_draft_stage':            ('host', "the coupled round's mapped staging (a copy; the coupled kernels read it)"),
    'cpu':                            ('host', 'the engine ticks its own clock'),
    'cvec':                           ('host', 'the table the cvec_apply kernel reads'),
    'doorbell_publish':               ('host', 'a host-device sync primitive'),
    'doorbell_publish_res':           ('host', 'a host-device sync primitive'),
    'doorbell_publish_value':         ('host', 'a host-device sync primitive'),
    'doorbell_ring':                  ('host', 'a host-device sync primitive'),
    'doorbell_wait':                  ('host', 'a host-device sync primitive'),
    'embed_type_supported':           ('host', 'a capability check'),
    'f16_from_f32':                   ('host', 'a type conversion helper'),
    'f32_from_f16':                   ('host', 'a type conversion helper'),
    'fused_gr_check':                 ('host', 'a capability check'),
    'fused_gr_supported':             ('host', 'a capability check'),
    'gr_workspace_bytes':             ('host', 'a size'),
    'gr_workspace_init':              ('host', 'a workspace hand-out'),
    'iq_row_bytes':                   ('host', 'a size'),
    'k':                              ('host', 'a constant'),
    'kv_block_bytes':                 ('host', 'a size'),
    'kv_q4_bytes_per_cell':           ('host', 'a size'),
    'kv_q4_bytes_per_head':           ('host', 'a size'),
    'kv_q8_bytes_per_cell':           ('host', 'a size'),
    'kv_ring_restore':                ('host', 'ring bookkeeping'),
    'kv_ring_table':                  ('host', 'ring bookkeeping'),
    'kv_stream_map_bytes':            ('host', 'the expert file tier: a size'),
    'kv_stream_reset':                ('host', 'the expert file tier: bookkeeping'),
    'kv_stream_resolve':              ('host', 'the expert file tier: whether a row is resident'),
    'moe_hit_grouped_scratch_bytes':  ('host', 'a size'),
    'native_expert_layout':           ('host', 'a layout description'),
    'native_expert_scratch_bytes':    ('host', 'a size'),
    'native_gdn_enabled':             ('host', 'a capability check'),
    'native_mmvq_supported':          ('host', 'a capability check'),
    'native_mmvq_weight_bytes':       ('host', 'a size'),
    'native_moe_combine_enabled':     ('host', 'a capability check'),
    'native_q8_1_bytes':              ('host', 'a size'),
    'native_qsa_enabled':             ('host', 'a capability check'),
    'native_qsa_indexer_enabled':     ('host', 'a capability check'),
    'native_rope_enabled':            ('host', 'a capability check'),
    'native_router_enabled':          ('host', 'a capability check'),
    'ngram_rows':                     ('host', 'host-side row bookkeeping'),
    'penalty_rows':                   ('host', 'folded into the samplers in this port: see sampler_greedy/sampler_kernel'),
    'ple_block_scratch_bytes':        ('host', 'a size'),
    'ple_native_bf16_enabled':        ('host', 'a capability check'),
    'ple_native_postops_enabled':     ('host', 'a capability check'),
    'qsa_decode_attn_scratch_floats': ('host', 'a size'),
    'qsa_real_shapes':                ('host', 'a shape description'),
    'qsa_selection_width':            ('host', 'a shape description'),
    'qsa_step_bytes':                 ('host', 'a size'),
    'qsa_step_fill':                  ('host', "the step record's fill"),
    'rope_scaling':                   ('host', 'a table parameter'),
    'rope_table_release':             ('host', 'table bookkeeping'),
    'rope_table_set':                 ('host', 'table bookkeeping'),
    'shared_expert_native_bf16_enabled': ('host', 'a capability check'),
    'shared_expert_scratch_bytes':    ('host', 'a size'),
    # ---- GPU work this port has NOT done: the honest hole list ----
    'add_streams_broadcast':          ('todo', 'no shader in this tree yet'),
    'bf16_gemv':                      ('todo', 'no shader in this tree yet'),
    'bf16_gemv_split':                ('todo', 'no shader in this tree yet'),
    'broadcast_streams':              ('todo', 'no shader in this tree yet'),
    'fetch_blobs':                    ('todo', 'no shader in this tree yet'),
    'fused_gdn_ab':                   ('kernel', 'fused_gdn_ab'),          # replaces 2x bf16 mmvf + beta_gate + gate
    'fused_gdn_conv_l2':              ('kernel', 'fused_gdn_conv_l2'),     # replaces native_gdn_conv_silu + 2x l2_norm
    'fused_gdn_step_norm':            ('kernel', 'fused_gdn_step_norm'),   # replaces native_gdn_step + native_gdn_out_norm
    'fused_gr_read':                  ('todo', 'a DEVICE op (fused_gr.cu:1168 launches gr_down/gr_up), mis-kinded host before; the FUSED alternative the gr branch policy removes'),
    'fused_gr_read_multi':            ('todo', 'no shader in this tree yet'),
    'gdn_ab_multi':                   ('todo', 'no shader in this tree yet'),
    'gdn_conv_commit':                ('todo', 'no shader in this tree yet'),
    'gdn_conv_l2_multi':              ('todo', 'no shader in this tree yet'),
    'gdn_step_norm_multi':            ('todo', 'no shader in this tree yet'),
    'gpu_stamp':                      ('todo', 'no shader in this tree yet'),
    'map_ids':                        ('todo', 'no shader in this tree yet'),
    'moe_group_resident':             ('todo', 'no shader in this tree yet'),
    'mtp_select':                     ('todo', 'no shader in this tree yet'),
    'native_moe_combine_multi':       ('todo', 'no shader in this tree yet'),
    'native_qsa_indexer_append':      ('todo', 'no shader in this tree yet'),
    'native_router_top10_multi':      ('todo', 'no shader in this tree yet'),
    'ple_block_projected':            ('todo', 'no shader in this tree yet'),
    'qsa_attend_step':                ('todo', 'no shader in this tree yet'),
    'qsa_decode_attn_batch':          ('todo', 'no shader in this tree yet'),
    'qsa_index_step':                 ('todo', 'no shader in this tree yet'),
    'rebase_ptrs':                    ('todo', 'no shader in this tree yet'),
    'resident_plan':                  ('todo', 'no shader in this tree yet'),
    'row_top_prob':                   ('todo', 'no shader in this tree yet'),
    's_gemv_q8_0_split':              ('todo', 'no shader in this tree yet'),
    's_gemv_q8k_split':               ('todo', 'no shader in this tree yet'),
    'topk_512_step':                  ('todo', 'no shader in this tree yet'),
    'wait_flag_ge':                   ('todo', 'no shader in this tree yet'),
    'wait_flag_ge_or':                ('todo', 'no shader in this tree yet'),
    'window_ids':                     ('todo', 'no shader in this tree yet'),
}

missing = [s for s in syms if s not in TABLE]
extra = [s for s in TABLE if s not in syms]
assert not missing, f"unclassified symbols src/core/ reaches: {missing}"
assert not extra, f"table rows that src/core/ does not reach: {extra}"

rows = [(s, *TABLE[s]) for s in syms]
counts = {k: sum(1 for r in rows if r[1] == k) for k in ('kernel', 'host', 'todo')}
out = ROOT / 'ports/vulkan/PORT-MAP.tsv'
with out.open('w') as f:
    f.write("# ports/vulkan/PORT-MAP.tsv - the DECODE PATH's kernels-namespace symbols (every one src/core/ reaches), classified.\n")
    f.write("# kernel = GPU work with a shader in this tree; host = the engine's own host side; todo = GPU work not ported.\n")
    f.write("# A symbol is in scope whether src/core/ writes it `kernels::X` or bare `X` (a using-directive in scope).\n")
    f.write("# gates/run_gate.sh checks this file against the engine's sources and the built shaders: an invented\n")
    f.write("# symbol, a shader that is not built, an unclaimed shader or an unlisted src/core/ symbol all fail.\n")
    f.write("# symbol\tkind\tshader(s) or reason\n")
    for s, kind, what in rows:
        f.write(f"{s}\t{kind}\t{what}\n")
print(f"PORT-MAP.tsv: {len(rows)} decode-path symbols - {counts['kernel']} kernel, {counts['host']} host, {counts['todo']} todo")
print("the holes, named:", ' '.join(s for s, k, _ in ((r[0], r[1], None) for r in rows) if k == 'todo'))
