#!/usr/bin/env python3
"""ports/vulkan/tools/gen-iq-tables.py - emit the IQ grid tables as C++ for the port's harness.

WHY A GENERATOR.  The I-quant dot products index lookup tables that live in the ENGINE'S OWN
`third_party/ggml/ggml-common.h` - the same tables the CUDA kernels read.  Retyping thousands of constants into
the port would be a second copy of kernel-critical data with nothing to keep the two equal, and a wrong entry
shows up as a plausible wrong number, not as a failure.  So they are DERIVED here and the generated file says so.
Re-run after any change to the engine's tables:

    python3 ports/vulkan/tools/gen-iq-tables.py           # writes harness/iq_grids.hpp
    python3 ports/vulkan/tools/gen-iq-tables.py --check   # fails if the file is stale

THE SHAPES, and why they are what they are:

  * `iq1s_grid` is 2048 uint32 - one 32-bit word per grid point, indexed directly by the kernel.
  * `iq2s_grid` is 1024 uint64 - EIGHT packed bytes per grid point.  `vec_dot_iq2_s_q8_1` reads it as a pair of
    32-bit words (`grid_pos[0]`, `grid_pos[1]`), so the port stores it as 2048 uint32 in low/high order: that
    keeps `shaderInt64` - an OPTIONAL Vulkan feature - out of the shaders entirely, for a value that is only
    ever consumed as two halves.

The shaders read these through storage buffers (a large constant array with a divergent index is the case
constant memory handles worst - the engine's own comment records a 2.12x cost for exactly that pattern).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parent
TREE = PORT.parent.parent
HEADER = TREE / "third_party" / "ggml" / "ggml-common.h"
OUT = PORT / "harness" / "iq_grids.hpp"


def extract_u32(name: str, want: int, digits: int) -> list[int]:
    text = HEADER.read_text()
    m = re.search(rf"GGML_TABLE_BEGIN\(uint32_t, {name}, N?[A-Z0-9_]*\)(.*?)GGML_TABLE_END\(\)", text, re.S)
    if not m:
        sys.exit(f"gen-iq-tables: {name} is not in {HEADER} - has it moved?")
    vals = [int(v, 16) for v in re.findall(rf"0x([0-9a-fA-F]{{{digits}}})", m.group(1))]
    if len(vals) != want:
        sys.exit(f"gen-iq-tables: {name} parsed {len(vals)} entries, {want} expected - a partial parse would "
                 "silently shift every index")
    return vals


def extract_u64_as_u32pairs(name: str, want: int) -> list[int]:
    text = HEADER.read_text()
    m = re.search(rf"GGML_TABLE_BEGIN\(uint64_t, {name}, [A-Z0-9_]+\)(.*?)GGML_TABLE_END\(\)", text, re.S)
    if not m:
        sys.exit(f"gen-iq-tables: {name} is not in {HEADER} - has it moved?")
    vals = [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]{16})", m.group(1))]
    if len(vals) != want:
        sys.exit(f"gen-iq-tables: {name} parsed {len(vals)} entries, {want} expected")
    out: list[int] = []
    for v in vals:
        out.append(v & 0xFFFFFFFF)          # grid_pos[0]
        out.append((v >> 32) & 0xFFFFFFFF)  # grid_pos[1]
    return out


def emit(lines: list[str], values: list[int], ctype: str, decl: str, note: str) -> None:
    lines += ["", note, f"inline constexpr {ctype} {decl} = {{"]
    for i in range(0, len(values), 8):
        lines.append("    " + ", ".join(f"0x{v:08x}u" for v in values[i:i + 8]) + ",")
    lines += ["};"]


def render() -> str:
    iq1s = extract_u32("iq1s_grid_gpu", 2048, 8)
    iq2s = extract_u64_as_u32pairs("iq2s_grid", 1024)
    iq3xxs = extract_u32("iq3xxs_grid", 256, 8)
    lines = [
        "// ports/vulkan/harness/iq_grids.hpp - GENERATED, do not edit.",
        "//",
        "// The I-quant grid tables, extracted verbatim from the engine's own third_party/ggml/ggml-common.h by",
        "// ports/vulkan/tools/gen-iq-tables.py. The kernels under test index these; the gate uploads them as storage",
        "// buffers, because that is what a real Vulkan backend does with a device-global lookup table. Regenerate",
        "// rather than edit:",
        "//",
        "//     python3 ports/vulkan/tools/gen-iq-tables.py --check",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace strata::vkport {",
    ]
    emit(lines, iq1s, "uint32_t", "kIq1sGrid[2048]",
         "// iq1s_grid: 2048 grid points, one uint32 each (IQ1_S / IQ1_M).")
    lines.append("inline constexpr int kIq1sGridSize = 2048;")
    emit(lines, iq2s, "uint32_t", "kIq2sGrid[2048]",
         "// iq2s_grid: 1024 grid points of EIGHT packed bytes, stored as low/high 32-bit halves "
         "(IQ2_S), so no\n// shader needs the optional shaderInt64 feature.")
    lines.append("inline constexpr int kIq2sGridSize = 1024;")
    emit(lines, iq3xxs, "uint32_t", "kIq3xxsGrid[256]",
         "// iq3xxs_grid: 256 grid points, one uint32 each (IQ3_XXS). Four signed bytes per word.")
    lines.append("inline constexpr int kIq3xxsGridSize = 256;")
    lines += ["", "}  // namespace strata::vkport", ""]
    return "\n".join(lines)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"gen-iq-tables: {OUT} is STALE (or missing) - regenerate it", file=sys.stderr)
            return 1
        print(f"gen-iq-tables: {OUT} is current (iq1s 2048 + iq2s 1024 + iq3xxs 256)")
        return 0
    OUT.write_text(text)
    print(f"gen-iq-tables: wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
