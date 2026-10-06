#!/usr/bin/env python3
# ports/vulkan/ref/dump_gguf.py - list the expert tensors in a pack shard (measurement-only helper).
import sys
sys.path.insert(0, "/home/bob/llama-050/gguf-py")
from gguf import GGUFReader

path = sys.argv[1]
r = GGUFReader(path)
print("tensor count:", len(r.tensors))
want = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
shown = 0
for t in r.tensors:
    if any(w in t.name for w in want):
        if t.name.startswith(("blk.0.", "blk.1.", "blk.2.", "blk.7.", "blk.8.")):
            print(f"{t.name:44s} type={int(t.tensor_type):3d} dims={list(t.shape)} off={int(t.data_offset)} nbytes={t.n_bytes}")
            shown += 1
print("shown:", shown)
