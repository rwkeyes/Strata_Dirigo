#!/usr/bin/env python3
"""Write ports/vulkan/PORT-MAP.tsv: the decode path's kernels:: symbols, each classified as

    kernel  GPU work this port has, naming the shader(s)
    host    the engine's own host side (a size, a check, a table, a sync primitive) - no dispatch to port
    todo    GPU work this port has NOT done - the honest hole list

The table is checked, not decorative: gates/run_gate.sh fails if a symbol is invented (absent from the engine's
sources), if a `kernel` row names a shader that is not built, if a built shader is named by no row, or if src/core/
calls a kernels:: symbol the table does not mention.  That last rule is what keeps it honest over time.
"""
import pathlib

ROOT = pathlib.Path('/home/bob/strata-vulkan-wt')
syms = sorted({line.strip().split('::', 1)[1] for line in pathlib.Path('/tmp/core_syms.txt').read_text().split()
               if '::' in line})

# kind, shader(s) or the reason it is host-side.
TABLE = {
    # ---- ported: the GPU work has a shader in this tree ----
    'add_inplace':                 ('kernel', 'add'),
    'fwht256_inplace_cuda':        ('kernel', 'fwht256'),
    'kv_append_q4_step':           ('kernel', 'kv_q4_append'),
    'kv_gather_q4_step':           ('kernel', 'kv_q4_gather'),
    'native_quantize_q8_1':        ('kernel', 'quantize_q8_1'),
    'ple_history_advance':         ('kernel', 'ple_history_advance'),
    'ple_block_scratch_bytes':     ('host', 'a size'),
    'qsa_decode_attn_step':        ('kernel', 'attn_decode_short'),
    'quantize_q8_':                ('kernel', 'quantize_q8_0 quantize_q8_1 quantize_q8_K quantize_q8_0_scaled'),
    'quantize_q8_0_scaled':        ('kernel', 'quantize_q8_0_scaled'),
    'quantize_q8_1_rows':          ('kernel', 'quantize_q8_1'),
    's_gemv_split_async':          ('kernel', 's_gemv_q8_split'),
    'sample_tokens':               ('kernel', 'sampler_greedy sampler_kernel'),
    'ple_block':                   ('kernel', 'ple_bcast ple_conv ple_gate ple_gnorm'),
    'native_mmvq':                 ('kernel', 'iq1m_mmvq iq2s_mmvq iq3s_mmvq iq3xxs_mmvq iq4nl_mmvq iq4xs_mmvq'),
    'native_expert_grouped':       ('kernel', 'native_gu_iq2s native_down_iq4nl s2expert_gu s2expert_down s2expert_swiglu'),
    'moe_hit_grouped_s2_dev':      ('kernel', 's2_gemv_q8 scalar_gate_f32 scalar_gate_f64 moe_combine_f32 moe_combine_f64'),
    'moe_hit_grouped_s2_cpu_order':('kernel', 's2_gemv_q8 scalar_gate_f32 scalar_gate_f64 moe_combine_f32 moe_combine_f64'),
    'cvec_apply':                  ('kernel', 'cvec_apply'),
    'gather_rows':                 ('kernel', 'gather_rows'),
    'scatter_rows_f32':            ('kernel', 'scatter_rows_f32'),

    # ---- the engine's own host side: no dispatch for the port to supply ----
    'cpu':                         ('host', 'the engine ticks its own clock'),
    'k':                           ('host', 'a constant'),
    'cvec':                        ('host', 'the table the cvec_apply kernel reads'),
    'copy_from_mapped':            ('host', 'a mapped-buffer copy'),
    'copy_i32_from_mapped':        ('host', 'a mapped-buffer copy'),
    'copy_or_zero_from_mapped':    ('host', 'a mapped-buffer copy'),
    'build_rope_table':            ('host', 'the engine builds the table; rope_neox is the kernel that reads it'),
    'rope_table_set':              ('host', 'table bookkeeping'),
    'rope_table_release':          ('host', 'table bookkeeping'),
    'rope_scaling':                ('host', 'a table parameter'),
    'f16_from_f32':                ('host', 'a type conversion helper'),
    'f32_from_f16':                ('host', 'a type conversion helper'),
    'fused_gr_check':              ('host', 'a capability check'),
    'fused_gr_supported':          ('host', 'a capability check'),
    'fused_gr_read':               ('host', 'a workspace read'),
    'gr_read':                     ('host', 'a workspace read'),
    'gr_workspace_bytes':          ('host', 'a size'),
    'gr_workspace_init':           ('host', 'a workspace hand-out'),
    'doorbell_publish':            ('host', 'a host-device sync primitive'),
    'doorbell_ring':               ('host', 'a host-device sync primitive'),
    'doorbell_wait':               ('host', 'a host-device sync primitive'),
    'kv_stream_map_bytes':         ('host', 'the expert file tier: a size'),
    'kv_stream_reset':             ('host', 'the expert file tier: bookkeeping'),
    'kv_stream_resolve':           ('host', 'the expert file tier: whether a row is resident'),
    'kv_ring_table':               ('host', 'ring bookkeeping'),
    'kv_ring_restore':             ('host', 'ring bookkeeping'),
    'kv_block_bytes':              ('host', 'a size'),
    'kv_q4_bytes_per_cell':        ('host', 'a size'),
    'kv_q4_bytes_per_head':        ('host', 'a size'),
    'kv_q8_bytes_per_cell':        ('host', 'a size'),
    'iq_row_bytes':                ('host', 'a size'),
    'iq_embed_rows':               ('todo', 'the IQ/BF16 token-embedding dequant, not ported - iq_kernels.hpp declares it'),
    'native_expert_layout':        ('host', 'a layout description'),
    'native_expert_scratch_bytes': ('host', 'a size'),
    'native_q8_1_bytes':           ('host', 'a size'),
    'native_mmvq_supported':       ('host', 'a capability check'),
    'native_mmvq_weight_bytes':    ('host', 'a size'),
    'native_gdn_enabled':          ('host', 'a capability check'),
    'native_qsa_indexer_enabled':  ('host', 'a capability check'),
    'embed_type_supported':        ('host', 'a capability check'),
    'ngram_rows':                  ('host', 'host-side row bookkeeping'),
    'penalty_rows':                ('host', 'folded into the samplers in this port: see sampler_greedy/sampler_kernel'),
    'qsa_real_shapes':             ('host', 'a shape description'),
    'qsa_selection_width':         ('host', 'a shape description'),
    'qsa_step_bytes':              ('host', 'a size'),
    'qsa_decode_attn_scratch_floats': ('host', 'a size'),
    'coupled_draft_scratch_bytes': ('host', 'a size'),
    'moe_hit_grouped_scratch_bytes':('host', 'a size'),
    'shared_expert_scratch_bytes': ('host', 'a size'),

    # ---- GPU work this port has NOT done: the honest hole list ----
    'embedding_gather':            ('kernel', 'embedding_gather'),
    'iq_dequant_f32':              ('todo', 'a standalone IQ dequantiser (this port has the FUSED iq*_mmvq)'),
    'native_q5_k_f32':             ('kernel', 'native_q5_k_f32'),
    'moe_grouped_s2':              ('todo', 'the grouped S2 MoE, not ported'),
    'moe_hit_add':                 ('todo', 'the hit accumulator, not ported'),
    'moe_hit_select':              ('kernel', 'moe_hit_select'),
    'moe_hit_grouped_s2':          ('kernel', 's2expert_gu s2expert_swiglu quantize_q8_0 s2expert_down'),
}

missing = [s for s in syms if s not in TABLE]
extra = [s for s in TABLE if s not in syms]
assert not missing, f"unclassified symbols in src/core/: {missing}"
assert not extra, f"table rows that src/core/ does not call: {extra}"

rows = [(s, *TABLE[s]) for s in syms]
counts = {k: sum(1 for r in rows if r[1] == k) for k in ('kernel', 'host', 'todo')}
out = ROOT / 'ports/vulkan/PORT-MAP.tsv'
with out.open('w') as f:
    f.write("# ports/vulkan/PORT-MAP.tsv - the DECODE PATH's kernels:: symbols (every one src/core/ calls), classified.\n")
    f.write("# kernel = GPU work with a shader in this tree; host = the engine's own host side; todo = GPU work not ported.\n")
    f.write("# gates/run_gate.sh checks this file against the engine's sources and the built shaders: an invented\n")
    f.write("# symbol, a shader that is not built, an unclaimed shader or an unlisted src/core/ symbol all fail.\n")
    f.write("# symbol\tkind\tshader(s) or reason\n")
    for s, kind, what in rows:
        f.write(f"{s}\t{kind}\t{what}\n")
print(f"PORT-MAP.tsv: {len(rows)} decode-path symbols - {counts['kernel']} kernel, {counts['host']} host, {counts['todo']} todo")
print("the holes, named:", ' '.join(s for s, k, _ in ((r[0], r[1], None) for r in rows) if k == 'todo'))
