#!/usr/bin/env python3
"""ports/vulkan/tools/gen-iq1s-grid.py - emit the IQ1_S grid table as C++ for the port's harness.

WHY A GENERATOR.  The IQ1_M dot product indexes a 2048-entry table (`iq1s_grid_gpu`) that lives in the
ENGINE'S OWN `third_party/ggml/ggml-common.h` - the same table the CUDA kernel reads.  Retyping 2048 constants
into the port would be a second copy of a kernel-critical table with nothing to keep the two equal, and a wrong
entry shows up as a plausible wrong number rather than as a failure.  So the table is DERIVED here and the
generated file says so.  Re-run this after any change to the engine's table:

    python3 ports/vulkan/tools/gen-iq1s-grid.py           # writes harness/iq1s_grid.hpp
    python3 ports/vulkan/tools/gen-iq1s-grid.py --check   # fails if the file is stale

The shader reads the table through a storage buffer (a 2048-entry constant array with a divergent index is the
case constant memory handles worst - the engine's own comment records a 2.12x cost for exactly that pattern).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parent
TREE = PORT.parent.parent
HEADER = TREE / "third_party" / "ggml" / "ggml-common.h"
OUT = PORT / "harness" / "iq1s_grid.hpp"


def extract() -> list[int]:
    text = HEADER.read_text()
    m = re.search(r"GGML_TABLE_BEGIN\(uint32_t, iq1s_grid_gpu, NGRID_IQ1S\)(.*?)GGML_TABLE_END\(\)",
                  text, re.S)
    if not m:
        sys.exit(f"gen-iq1s-grid: the iq1s_grid_gpu table is not in {HEADER} - has it moved?")
    vals = [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]{8})", m.group(1))]
    if len(vals) != 2048:
        sys.exit(f"gen-iq1s-grid: parsed {len(vals)} entries, the engine declares NGRID_IQ1S = 2048 - "
                 "a partial parse would silently shift every index")
    return vals


def render(vals: list[int]) -> str:
    lines = [
        "// ports/vulkan/harness/iq1s_grid.hpp - GENERATED, do not edit.",
        "//",
        "// The IQ1_S grid table, extracted verbatim from the engine's own third_party/ggml/ggml-common.h",
        "// (GGML_TABLE_BEGIN(uint32_t, iq1s_grid_gpu, NGRID_IQ1S)) by ports/vulkan/tools/gen-iq1s-grid.py.",
        "// The kernel under test indexes this table; the gate uploads it as a storage buffer, because that is",
        "// what a real Vulkan backend does with a device-global lookup table. Regenerate rather than edit:",
        "//",
        "//     python3 ports/vulkan/tools/gen-iq1s-grid.py --check",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace strata::vkport {",
        "",
        "inline constexpr int kIq1sGridSize = 2048;",
        "",
        "inline constexpr uint32_t kIq1sGrid[kIq1sGridSize] = {",
    ]
    for i in range(0, len(vals), 8):
        chunk = ", ".join(f"0x{v:08x}u" for v in vals[i:i + 8])
        lines.append(f"    {chunk},")
    lines += ["};", "", "}  // namespace strata::vkport", ""]
    return "\n".join(lines)


def main() -> int:
    vals = extract()
    text = render(vals)
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"gen-iq1s-grid: {OUT} is STALE (or missing) - regenerate it", file=sys.stderr)
            return 1
        print(f"gen-iq1s-grid: {OUT} is current ({len(vals)} entries)")
        return 0
    OUT.write_text(text)
    print(f"gen-iq1s-grid: wrote {OUT} ({len(vals)} entries, first 0x{vals[0]:08x} last 0x{vals[-1]:08x})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
