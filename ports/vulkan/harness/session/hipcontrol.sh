#!/usr/bin/env bash
# /home/bob/hipcontrol.sh (runs ON z820b) - THE HIP CONTROL ARM.
#
# The brief (forktest/queue/2026-10-07-z820b-hip-control-run.md) fixes the decision rule BEFORE the run:
#   * HIP's us/dispatch near SYCL's ~14.5  -> the port's 66.8 us/dispatch is a VULKAN-PORT cost.
#   * HIP also paying ~60 us/dispatch      -> it is the ENGINE's own per-dispatch structure.
#
# FIELDS IT ASKS FOR: kernels per token; us per dispatch; workgroups per dispatch (grid histogram); the phase table
# (host/launch/sync/commit ms per round); decode and prefill tok/s; tokens per round and draft acceptance; ids.
# The engine's own tables carry the phase split (STRATA_DECODE_TIMING -> the "wait for rings / pool / host(stage) /
# launch / sync / commit" line; STRATA_PREFILL_TIMING -> the prefill phase table), and the same 199-token prompt
# and flags are used as the port's arms so the two are comparable.
#
# SECOND RUN: `--ple-io ram` holds the 28.8 GB PLE table in RAM (the box has 81 GiB free).  The first run
# left it on a ROTATIONAL disk and the phase table showed what that costs: pool 184.8 + host(stage) 175.8 ms per
# round against a 9 ms expert-dispatch path - i.e. the first run measured the HARD DRIVE, not the engine.
#
# STATED SQUARELY, because the brief demands it: the SMALLER pack is `coder-iq1_m` (the same pack the port uses -
# its dense.bin is 1,475 MB; the 24.79 GiB the brief refers to is the VRAM FOOTPRINT of dense + a 12,288-slot
# expert cache, which does not fit 24 GB).  So the pack is NOT shrunk; the EXPERT CACHE is (8,192 slots instead of
# 12,288).  Per-dispatch cost and grid geometry do not depend on the cache size; tok/s DOES, so tok/s here is not
# comparable to a 12,288-slot arm.  Second caveat: this box's CPU is AVX-only, so the engine is the experimental
# STRATA_ISA_FLOOR=avx build and its CPU half is the slow one - again, the GPU-side per-dispatch numbers are the
# comparable fields.
set -u
BIN="$HOME/strata-hip-wt/build-hip/strata"
PACK="$HOME/strata-packs/coder-iq1_m"
NATIVE="$HOME/strata-models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
LOG="$HOME/hip199_ram.log"
T199="$(printf '1 %.0s' $(seq 1 199))"
[ -x "$BIN" ] || { echo "no HIP binary"; exit 2; }
cd "$HOME/strata-hip-wt/build-hip" || exit 2
{
  echo "== HIP control arm $(date -Is)"
  echo "   bin=$BIN  pack=$PACK  native=$NATIVE"
  echo "   cache=8192 mmap-experts  spec=2 prefill=256 tokens=199 max-new=32 max-context=512"
  echo "   rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')  arch=gfx1100  isa_floor=avx"
} > "$LOG"
STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1 \
  timeout 1500 "$BIN" --pack "$PACK" --native "$NATIVE" --spec 2 --prefill 256 \
    --tokens "$T199" --max-new 32 --max-context 512 \
    --expert-profile "$HOME/expert-profile-coder-built.bin" --expert-cache 8192 --mmap-experts --ple-io ram >> "$LOG" 2>&1
echo "RUN_RC=$?" >> "$LOG"
{
  echo "== the fields the control asks for =="
  grep -m1 '^prefill' "$LOG"
  grep -m1 'decode' "$LOG"
  grep -m1 -E 'wait for rings|host\(stage\)' "$LOG"
  echo "-- output (id source) --"
  grep -m1 '^output' "$LOG" | cut -c1-120
} >> "$LOG"
tail -40 "$LOG"
