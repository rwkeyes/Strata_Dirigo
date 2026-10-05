#!/usr/bin/env python3
"""make_synth_pack.py - write a SMALL SYNTHETIC pack the engine's own loader (`WeightTable::load`) loads.

WHY THIS EXISTS.  The Vulkan port's program (`strata_vulkan generate`) stops at MODEL LOAD because the engine
wants a PACK (`<dir>/index.txt`, written by `tools/pack_index.py`) and this box holds only raw GGUF shards.
To reach a TOKEN without the 58 GB artifact, this writes the SMALLEST pack the engine's FIXED geometry can
decode: the canonical Qwen3.8-Flash-Next geometry (48 layers, n_embd 2560, 512 experts) with RANDOM weights in
every tensor the forward path resolves.

WHAT IT IS NOT.  It is not a model.  The weights are random, so the token the engine produces is MEANINGLESS
CONTENT - what is being demonstrated is the PIPELINE, not the model.

THE FORMAT, from the engine's own code:
  * `src/core/weights.cpp::WeightTable::load` reads `<dir>/index.txt` (19 space-separated columns; see
    `tools/pack_index.py`'s header for the column list).  `tools/pack_index.py` is the engine's own tool that
    turns a `manifest.json` + the pack's `.bin` files into that index - this script writes the manifest + the
    `.bin` files and then INVOKES `tools/pack_index.py` to write `index.txt`, so the loader's format has one
    implementation (the engine's) and not two.
  * A pack is: `<dir>/{dense.bin, embd.bin, experts.bin}` + `index.txt` (+ an optional `manifest.json`), where
    each dense tensor is a set of PLANES (`codes`, `scales`, optional `offsets`) the loader copies (and
    possibly re-widens) into one device arena.
  * `experts.bin` must be exactly `n_layers * n_expert * BLOB` = 48*512*1,382,400 = 33,973,862,400 B for this
    geometry (`src/kernels/cpu/expert_layout.cpp`, `expert.hpp:BLOB`).  It is created SPARSE and never read on
    the `--no-pool --mmap-experts` run this pack is built for.

`tools/strata_pack.py build` (the GGUF -> pack builder) is the other pack tool, and it is NOT usable here: it
needs a source GGUF whose tensor types are in its MAPPINGS table, and the only GGUF on this box is the real
58 GB IQ1_M artifact (types include IQ1_M, which MAPPINGS does not cover).  So the manifest is written directly
in the pack format `tools/strata_pack.py::tensor_entry` defines, and the engine's own `pack_index.py` writes the
index.  See ports/vulkan/NEXT.md for the run and its bounds.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np

ALIGN = 64
N_VOCAB = 248320
FORMAT_VERSION = 1

# ---- geometry: the engine's canonical defaults (include/strata/core/layout.hpp) -------------------------
G = dict(n_embd=2560, n_layers=48, qsa_interval=4, ssm_state_size=128, ssm_k_heads=16, ssm_v_heads=48,
         ssm_d_conv=4, ssm_conv_channels=10240, ssm_value_dim=6144, n_head=24, n_head_kv=2, head_dim=256,
         idx_q_heads=4, idx_key_dim=128, hc=4, hc_lr=320, n_expert=512, n_ff=640)
HC_DIM = G["hc"] * G["n_embd"]                     # 10240


def align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


class Builder:
    """One output .bin, written streaming so a 6 GB pack never sits in RAM.

    `zero` writes a pack whose MATRICES are all zero (codes 0x80 -> (128-128)*scale = 0; bf16 0): the clean
    invariant for "is the composed 48-layer chain arithmetically SOUND?" - a zero in must give a finite zero
    out, and a non-finite answer from a zero pack is a PORT DEFECT, not a degenerate fixture.
    `weight_scale` multiplies the quant scales and the bf16 values (the projection/embedding matrices) so a
    random pack can be given SANE magnitudes for this architecture; the f32 rows (norms, biases, `ssm_a`) are
    left alone - they are constants, not data-driven weight matrices."""

    def __init__(self, path: pathlib.Path, zero: bool = False, weight_scale: float = 1.0):
        self.path = path
        self.fh = open(path, "wb")
        self.pos = 0
        self.zero = zero
        self.weight_scale = weight_scale
        self.entries: dict[str, dict] = {}

    def _align(self) -> None:
        pad = align_up(self.pos) - self.pos
        if pad:
            self.fh.write(b"\0" * pad)
            self.pos += pad

    def _put(self, blob: bytes) -> dict:
        off = self.pos
        self.fh.write(blob)
        self.pos += len(blob)
        return {"offset": off, "bytes": len(blob)}

    # ---- the three tensor families the index/pack distinguish ------------------------------------------
    def add_quant(self, name: str, ne0: int, ne1: int, group: int = 32, bits: int = 8) -> None:
        """A legacy Q8_0 (S8) tensor: codes then f32 scales, contiguous, no offset plane."""
        assert ne0 % group == 0, (name, ne0, group)
        assert ne0 % (8 // bits) == 0
        elems = ne0 * ne1
        codes_bytes = elems * bits // 8
        n_groups = ne0 // group
        scales_bytes = ne1 * n_groups * 4
        self._align()
        self._put_codes(codes_bytes)
        self._put_scales(scales_bytes)
        codes = {"offset": self.pos - codes_bytes - scales_bytes, "bytes": codes_bytes}
        scales = {"offset": self.pos - scales_bytes, "bytes": scales_bytes}
        self.entries[name] = {
            "file": self.path.name, "shape": [ne0, ne1], "elements": elems, "source_type": "Q8_0",
            "code_bits": 8, "code_bias": -128, "group_elems": group, "has_offset": False,
            "scales_fp16": False, "offsets_fp16": False, "codes": codes, "scales": scales,
        }

    def _put_codes(self, n: int) -> None:
        # random bytes, drawn in chunks; code byte b decodes as (b - 128) * scale.
        # ZERO mode writes 0x80: (128 - 128) * scale = 0 for every code, whatever the scale is.
        if self.zero:
            left = n
            z = b"\x80" * (1 << 24)
            while left:
                k = min(left, 1 << 24)
                self.fh.write(z[:k])
                self.pos += k
                left -= k
            return
        rng = np.random.default_rng(0x5EED1234 ^ self.pos)
        left = n
        while left:
            k = min(left, 1 << 24)
            self.fh.write(rng.integers(0, 256, size=k, dtype=np.uint8).tobytes())
            self.pos += k
            left -= k

    def _put_scales(self, n: int) -> None:
        # small positive fp32 scales (avoid NaN/Inf and keep magnitudes bounded)
        rng = np.random.default_rng(0x1234ABCD ^ self.pos)
        left = n // 4
        while left:
            k = min(left, 1 << 22)
            s = (rng.random(k, dtype=np.float32) * 0.02 + 0.005).astype("<f4")
            if self.weight_scale != 1.0:
                s = (s * np.float32(self.weight_scale)).astype("<f4")
            self.fh.write(s.tobytes())
            self.pos += 4 * k
            left -= k

    def add_bf16(self, name: str, ne0: int, ne1: int) -> None:
        """A BF16 source tensor, held in the pack PROMOTED to f32 (the loader takes the high 16 bits)."""
        elems = ne0 * ne1
        self._align()
        self._put_bf16(elems)
        values = {"offset": self.pos - elems * 4, "bytes": elems * 4}
        self.entries[name] = {
            "file": self.path.name, "shape": [ne0, ne1], "elements": elems, "source_type": "BF16",
            "values": values, "values_fp16": False,
        }

    def _put_bf16(self, n: int) -> None:
        rng = np.random.default_rng(0xABCD1234 ^ self.pos)
        left = n
        while left:
            k = min(left, 1 << 22)
            if self.zero:
                self.fh.write(b"\0" * (4 * k))
                self.pos += 4 * k
                left -= k
                continue
            v = (rng.random(k, dtype=np.float32) - 0.5) * 0.4          # ~[-0.2, 0.2)
            if self.weight_scale != 1.0:
                v = v * np.float32(self.weight_scale)
            u = v.view(np.uint32)
            u = (u & np.uint32(0xFFFF0000))                            # zero the low half: exact bf16
            self.fh.write(u.astype("<u4").tobytes())
            self.pos += 4 * k
            left -= k

    def add_f32(self, name: str, n: int, value: float = 1.0) -> None:
        self._align()
        arr = np.full(n, value, dtype="<f4")
        self._put(arr.tobytes())
        self.entries[name] = {
            "file": self.path.name, "shape": [n, 0], "elements": n, "source_type": "F32",
            "values": {"offset": self.pos - n * 4, "bytes": n * 4}, "values_fp16": False,
        }

    def add_f32_neg(self, name: str, n: int) -> None:
        """`ssm_a` must be NEGATIVE: the delta-rule `dec = exp(softplus(...) * ssm_a)` overflows on a positive draw."""
        self._align()
        rng = np.random.default_rng(0xF00D ^ self.pos)
        arr = (-(rng.random(n, dtype=np.float32) * 0.45 + 0.05)).astype("<f4")
        self._put(arr.tobytes())
        self.entries[name] = {
            "file": self.path.name, "shape": [n, 0], "elements": n, "source_type": "F32",
            "values": {"offset": self.pos - n * 4, "bytes": n * 4}, "values_fp16": False,
        }

    def close(self) -> int:
        self.fh.close()
        return self.pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack-index", default=None, help="path to tools/pack_index.py")
    ap.add_argument("--experts", action="store_true", help="also create the sparse experts.bin")
    ap.add_argument("--zero", action="store_true",
                    help="all weight matrices ZERO (the arithmetic-soundness invariant: a finite zero out)")
    ap.add_argument("--weight-scale", type=float, default=1.0,
                    help="multiply the quant scales and bf16 values (sane magnitudes for the architecture)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dense = Builder(out / "dense.bin", zero=args.zero, weight_scale=args.weight_scale)
    embd = Builder(out / "embd.bin", zero=args.zero, weight_scale=args.weight_scale)

    # ---- the per-layer set, exactly the names src/core/layer.cpp resolves and check_layer shapes --------
    for l in range(G["n_layers"]):
        qsa = (l % G["qsa_interval"] == G["qsa_interval"] - 1)
        p = f"blk.{l}."
        # hyper-connection (every layer).  w_down [hc_dim, hc_lr], w_up [hc_lr, hc_dim], w_inject [hc_dim, hc]
        dense.add_bf16(p + "hc_attn_down.weight", HC_DIM, G["hc_lr"])
        dense.add_bf16(p + "hc_attn_up.weight", G["hc_lr"], HC_DIM)
        dense.add_bf16(p + "hc_attn_inject.weight", HC_DIM, G["hc"])
        dense.add_bf16(p + "hc_ffn_down.weight", HC_DIM, G["hc_lr"])
        dense.add_bf16(p + "hc_ffn_up.weight", G["hc_lr"], HC_DIM)
        dense.add_bf16(p + "hc_ffn_inject.weight", HC_DIM, G["hc"])
        dense.add_f32(p + "hc_attn_norm.weight", HC_DIM)
        dense.add_f32(p + "hc_ffn_norm.weight", HC_DIM)
        # MoE (every layer)
        dense.add_bf16(p + "ffn_gate_inp.weight", G["n_embd"], G["n_expert"])       # router, BF16
        dense.add_quant(p + "ffn_gate_shexp.weight", G["n_embd"], G["n_ff"])
        dense.add_quant(p + "ffn_up_shexp.weight", G["n_embd"], G["n_ff"])
        dense.add_quant(p + "ffn_down_shexp.weight", G["n_ff"], G["n_embd"])
        dense.add_bf16(p + "ffn_gate_inp_shexp.weight", G["n_embd"], 1)             # scalar gate, BF16
        if not qsa:
            # GDN / DeltaNet mixer
            dense.add_quant(p + "attn_qkv.weight", G["n_embd"], G["ssm_conv_channels"])
            dense.add_quant(p + "attn_gate.weight", G["n_embd"], G["ssm_value_dim"])
            dense.add_quant(p + "ssm_out.weight", G["ssm_value_dim"], G["n_embd"])
            dense.add_f32(p + "ssm_conv1d.weight", G["ssm_d_conv"] * G["ssm_conv_channels"])
            dense.add_bf16(p + "ssm_alpha.weight", G["n_embd"], G["ssm_v_heads"])
            dense.add_bf16(p + "ssm_beta.weight", G["n_embd"], G["ssm_v_heads"])
            dense.add_f32_neg(p + "ssm_a", G["ssm_v_heads"])
            dense.add_f32(p + "ssm_dt.bias", G["ssm_v_heads"])
            dense.add_f32(p + "ssm_norm.weight", G["ssm_state_size"])
        else:
            # QSA (full attention)
            dense.add_quant(p + "attn_q.weight", G["n_embd"], 2 * G["n_head"] * G["head_dim"])
            dense.add_quant(p + "attn_k.weight", G["n_embd"], G["n_head_kv"] * G["head_dim"])
            dense.add_quant(p + "attn_v.weight", G["n_embd"], G["n_head_kv"] * G["head_dim"])
            dense.add_quant(p + "attn_output.weight", G["n_head"] * G["head_dim"], G["n_embd"])
            dense.add_bf16(p + "indexer.q_proj.weight", G["n_embd"], G["idx_q_heads"] * G["idx_key_dim"])
            dense.add_bf16(p + "indexer.k_proj.weight", G["n_embd"], G["idx_key_dim"])
            dense.add_f32(p + "attn_q_norm.weight", G["head_dim"])
            dense.add_f32(p + "attn_k_norm.weight", G["head_dim"])
            dense.add_f32(p + "indexer.q_norm.weight", G["idx_key_dim"])
            dense.add_f32(p + "indexer.k_norm.weight", G["idx_key_dim"])

    # ---- the head (lm_head + lm_head_mix) ---------------------------------------------------------------
    dense.add_bf16("output_hc_down.weight", HC_DIM, G["hc_lr"])
    dense.add_bf16("output_hc_up.weight", G["hc_lr"], HC_DIM)
    dense.add_f32("output_hc_norm.weight", HC_DIM)
    dense.add_quant("output.weight", G["n_embd"], N_VOCAB)
    # ---- the token embedding ----------------------------------------------------------------------------
    embd.add_quant("token_embd.weight", G["n_embd"], N_VOCAB)

    dense_bytes = dense.close()
    embd_bytes = embd.close()

    tensors = {}
    tensors.update(dense.entries)
    tensors.update(embd.entries)

    mode = "ZERO weight matrices (the arithmetic-soundness invariant)" if args.zero else \
           (f"random weights, weight-scale {args.weight_scale}" if args.weight_scale != 1.0 else
            "random weights")
    man = {
        "format": "strata-pack", "format_version": FORMAT_VERSION,
        "source": {"shard1": f"(synthetic: {mode}, no GGUF source)", "shard2": None},
        "n_layers": G["n_layers"], "align": ALIGN,
        "codebooks": {"IQ4NL": []},
        "tensors": tensors, "experts": {}, "n_experts_per_layer": G["n_expert"],
        "files": {"experts.bin": G["n_layers"] * G["n_expert"] * 1382400,
                  "dense.bin": dense_bytes, "embd.bin": embd_bytes},
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"wrote {man_path} ({len(tensors)} tensors)")
    print(f"  dense.bin {dense_bytes/2**30:.3f} GiB, embd.bin {embd_bytes/2**30:.3f} GiB")

    if args.experts:
        exp = out / "experts.bin"
        want = G["n_layers"] * G["n_expert"] * 1382400
        with open(exp, "wb") as f:      # sparse: truncate to size; never read on --no-pool
            f.truncate(want)
        print(f"  experts.bin {want} B (sparse), on {exp.stat().st_size}")

    pi = args.pack_index or str(pathlib.Path(__file__).resolve().parent / "pack_index.py")
    r = subprocess.run([sys.executable, pi, "--pack", str(out)], check=False)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
