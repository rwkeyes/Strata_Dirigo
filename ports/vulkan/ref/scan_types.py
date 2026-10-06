#!/usr/bin/env python3
# ports/vulkan/ref/scan_types.py - pick one real tensor per ggml type for the fixture set.
import sys, collections
sys.path.insert(0, "/home/bob/llama-050/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
seen = {}
for t in r.tensors:
    ty = int(t.tensor_type)
    seen.setdefault(ty, []).append((t.name, int(t.data_offset), tuple(int(x) for x in t.shape), t.n_bytes))
for ty in sorted(seen):
    names = seen[ty]
    print(f"type {ty:3d}: {len(names):4d} tensors   e.g. {names[0][0]}  off={names[0][1]} dims={names[0][2]} nbytes={names[0][3]}")
