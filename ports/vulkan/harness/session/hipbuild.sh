#!/usr/bin/env bash
# /home/bob/hipbuild.sh (runs ON z820b) - the engine's native HIP backend for gfx1100, on THIS box's CPUs.
#
# The HIP control (forktest/queue/2026-10-07-z820b-hip-control-run.md) separates a VULKAN-PORT property from an
# ENGINE property: this port pays 66.8 us per dispatch against upstream SYCL's 14.5, and the engine's own native
# backend is HIP, with gfx1100 (RX 7900 XTX) the card its hipBLASLt tuning tables are written for.
#
# TWO BOX-SPECIFIC FACTS THIS SCRIPT PINS DOWN, both found the hard way and both stated so the run is reproducible:
#
#  1. THE CPU FLOOR.  This box is an HP Z820 with a Xeon E5-2687W v0 - Sandy Bridge-EP, AVX only.  The engine's
#     ready-made CPU expert kernels need AVX2+FMA+F16C and the binary refuses to run (rc=2, "does not support
#     AVX2 with FMA and F16C").  docs/INSTALL.md's "Older CPUs" section is the sanctioned route: an experimental
#     `STRATA_ISA_FLOOR=avx` build (CMakeLists.txt:67).  CONSEQUENCE FOR THE CONTROL: the CPU half of the expert
#     path on this box is the EXPERIMENTAL slow codegen, so tok/s here is not a like-for-like against a modern
#     host - the per-dispatch cost and grid geometry are the comparable fields, which is what the brief measures.
#
#  2. THE LINK.  clang picks this box's NEWEST GCC headers (gcc/16) while CMake's link line pointed at gcc/12, so
#     the link failed on `_M_replace_cold` / `GLIBCXX_3.4.32` / `CXXABI_1.3.15` (ROCm's libamdhip64 needs 3.4.32+).
#     The box has gcc 12-16 and the runtime libstdc++ reaches GLIBCXX_3.4.35.  Pinning the search path to gcc/16
#     links cleanly - verified by hand before this script existed.
set -u
SRC="$HOME/strata-hip-wt"
BD="$SRC/build-hip"
LOG="$HOME/hipbuild.log"
cd "$SRC" || { echo "no $SRC"; exit 2; }

{
  echo "== strata HIP build $(date -Is)"
  echo "   target=strata arch=gfx1100 jobs=8 nice=19 isa_floor=avx link=-L/usr/lib/gcc/x86_64-linux-gnu/16"
  echo "   source=mirror of a67cacb838e9a2d2687f2abd09db6c5c31bd9288 (same commit the port's vega tree builds)"
} | tee "$LOG"

nice -n 19 systemd-inhibit --what=idle --why="strata HIP build (agent)" \
  cmake -S . -B build-hip -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    -DSTRATA_ISA_FLOOR=avx \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/16" 2>&1 | tail -12 | tee -a "$LOG"

nice -n 19 systemd-inhibit --what=idle --why="strata HIP build (agent)" \
  cmake --build "$BD" --target strata -j8 2>&1 | tail -60 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "BUILD_EXIT=$rc  $(date -Is)" | tee -a "$LOG"
ls -la "$BD/strata" 2>/dev/null | tee -a "$LOG"
echo "--- the binary's arch list (must contain gfx1100) ---" | tee -a "$LOG"
strings "$BD/strata" 2>/dev/null | grep -m3 'gfx11' | tee -a "$LOG"
