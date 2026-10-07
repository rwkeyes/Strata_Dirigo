#!/usr/bin/env bash
# /home/bob/hipcontrol_count.sh (runs ON z820b) - THE HIP CONTROL ARM, WITH THE LAUNCH COUNTER ATTACHED.
#
# Same arm, same flags, same prompt as hipcontrol.sh -- only two things differ: LD_PRELOAD puts hipcount.so in
# front of libamdhip64 (counting every hipLaunchKernel / hipModuleLaunchKernel / graph node / graph replay), and
# the log is hip199_count.log so the earlier control's hip199_ram.log is not overwritten.
#
# WHAT THIS CLOSES.  The brief asks for kernels per token and us per dispatch.  The engine prints the launch TIME
# but never a count, so the port's 66.8 us/dispatch had nothing to compare against.  With the count and the
# engine's own per-round launch bucket, `us/dispatch` falls out directly:
#     us/dispatch = (launch ms/round x 1000) / (kernel submissions in one round)
# and the launch TIMELINE (/home/bob/hipcount.log) lets the prefill window and the decode window be counted
# separately, since the engine prints the prefill line before the decode line.
set -u
BIN="$HOME/strata-hip-wt/build-hip/strata"
PACK="$HOME/strata-packs/coder-iq1_m"
NATIVE="$HOME/strata-models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
LOG="$HOME/hip199_count.log"
COUNT="$HOME/hipcount.log"
T199="$(printf '1 %.0s' $(seq 1 199))"
[ -x "$BIN" ] || { echo "no HIP binary"; exit 2; }
[ -f "$HOME/hipcount.so" ] || { echo "no hipcount.so"; exit 2; }
cd "$HOME/strata-hip-wt/build-hip" || exit 2
{
  echo "== HIP control arm + launch counter $(date -Is)"
  echo "   bin=$BIN  pack=$PACK  native=$NATIVE"
  echo "   cache=8192 mmap-experts  spec=2 prefill=256 tokens=199 max-new=32 max-context=512"
  echo "   shim=$HOME/hipcount.so  timeline=$COUNT  (one line per kernel submission)"
  echo "   rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')  arch=gfx1100  isa_floor=avx"
} > "$LOG"
LD_PRELOAD="$HOME/hipcount.so" HIPCOUNT_LOG="$COUNT" \
STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1 \
  timeout 1800 "$BIN" --pack "$PACK" --native "$NATIVE" --spec 2 --prefill 256 \
    --tokens "$T199" --max-new 32 --max-context 512 \
    --expert-profile "$HOME/expert-profile-coder-built.bin" --expert-cache 8192 --mmap-experts --ple-io ram >> "$LOG" 2>&1
echo "RUN_RC=$?  (0 = the engine exited clean; 124 = the 1800 s timeout)" >> "$LOG"
{
  echo "== the fields the control asks for =="
  grep -m1 '^prefill' "$LOG"
  grep -m1 'decode' "$LOG"
  grep -m1 -E 'wait for rings|host\(stage\)' "$LOG"
  grep -m1 'hipcount:' "$LOG"
  echo "-- the launch timeline: window boundaries and counts --"
  awk -F'\t' 'NR==1{first=$1} {last=$1; c[$2]++; t[$2]=t[$2]} END{n=0; for (k in c) {printf "  %-12s %7d  %s\n", k, c[k], ""; n+=c[k]} printf "  %-12s %7d  (events between %.1f us and %.1f us)\n", "TOTAL", n, first, last}' "$COUNT"
  echo "-- first 12 events and last 6 --"
  head -12 "$COUNT" | sed 's/^/  /'; echo "  ..."; tail -6 "$COUNT" | sed 's/^/  /'
} >> "$LOG"
tail -46 "$LOG"
