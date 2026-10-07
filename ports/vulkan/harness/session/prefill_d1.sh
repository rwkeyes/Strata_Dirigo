#!/usr/bin/env bash
# /home/bob/step4/prefill_d1.sh -- the D1 falsifier (staged draft D1-1057-prefill-stager-sleep.patch).
#
#   FALSIFIER (written down before the run, ports/vulkan workqueue 2026-10-07):
#     PROVE  prefill >= 1.5x  with process CPU% falling to ~100%
#     REFUTE < 2% gain        with CPU% unchanged
#
#   Arms, interleaved, on the 199-token arm (the arm results are REPORTED from):
#     pfA_r*      the A binary  = HEAD without D1            (the spin, no env)
#     pfB_off_r*  the B binary  = HEAD + D1, STRATA_STAGER_SLEEP=0  (the spin, same binary as the sleep arm)
#     pfB_on_r*   the B binary  = HEAD + D1, STRATA_STAGER_SLEEP=1  (the sleep)
#   pfB_off is the switch-off control INSIDE one binary, so the arm pair differs by one env variable and nothing
#   else; pfA is the cross-check that the patch applied to the shipped spin is the same arm as the control.
#
#   REFUSES if the two binaries are the same bytes (a no-op build would VOID the comparison), if the id list is not
#   199 long, and it reports ARM_DEGENERATE / ARM_DIDNOTRUN straight from run.sh's own screen.
set -u
LOGD=/home/bob/step4/logs
TSV=/home/bob/step4/prefill_d1.tsv
A=${PFA_BIN:-/home/bob/step4/bins/strata_vulkan.pfA}
B=${PFB_BIN:-/home/bob/step4/bins/strata_vulkan.pfB}
REPS=${REPS:-1}
T199="$(printf '1 %.0s' $(seq 1 199))"
[ "$(echo $T199 | wc -w)" -eq 199 ] || { echo "PREFILL_D1 REFUSES: id list is $(echo $T199|wc -w), not 199"; exit 2; }
for f in "$A" "$B"; do
  [ -x "$f" ] || { echo "PREFILL_D1 REFUSES: $f is missing or not executable"; exit 2; }
done
shaA=$(sha256sum "$A" | cut -d' ' -f1); shaB=$(sha256sum "$B" | cut -d' ' -f1)
if [ "$shaA" = "$shaB" ]; then
  echo "PREFILL_D1 VOID: the A and B binaries are the SAME bytes ($shaA) - the build was a no-op"; exit 3
fi
: > "$TSV"
printf 'label\tbin\tenv\tprefill_ms\tprefill_s\tok\tload_prefill_ms\tload_ok\tttft_ms\tdecode\tid\tcpu_pct\telapsed_s\tmaxrss_kb\trc\n' >> "$TSV"

one() {  # one <name> <bin> <env-prefix-string>
  local name="$1" bin="$2" envs="$3" L="$LOGD/$1.log" tf="/tmp/pf_time_$1.txt"
  echo "[$(date '+%H:%M:%S')] ARM $name bin=$(basename "$bin") sha=$(sha256sum "$bin"|cut -c1-12) env=[$envs]"
  rm -f "$tf"
  env STRATA_VK_BIN="$bin" $envs timeout 900 flock /tmp/b70.lock -c \
      "/usr/bin/time -v -o $tf bash /home/bob/step4/run.sh $name \"$T199\"" || true
  # ---- read the RAW lines (a summary is not evidence) -------------------------------------------------
  local pms p_ok lms l_ok ttft dec id cpu el maxrss rc
  pms=$(grep -m1 '^prefill ' "$L" | grep -oE 'in [0-9.]+ ms' | grep -oE '[0-9.]+')
  p_ok=$(grep -m1 '^prefill ' "$L" | grep -oE '\-> *[0-9.]+ tok/s' | grep -oE '[0-9.]+')
  lms=$(grep -m1 '^strata generate: prefill ' "$L" | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+')
  l_ok=$(grep -m1 '^strata generate: prefill ' "$L" | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+')
  ttft=$(grep -m1 'time to first token' "$L" | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+')
  dec=$(grep -m1 '^decode ' "$L" | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+')
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)
  cpu=$(grep -m1 'Percent of CPU' "$tf" | grep -oE '[0-9]+%')
  el=$(grep -m1 'Elapsed (wall clock)' "$tf" | sed 's/.*): //')
  maxrss=$(grep -m1 'Maximum resident set size' "$tf" | awk '{print $NF}')
  rc=$(grep -m1 RUN_RC "$L" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$(basename "$bin")" "${envs:-none}" "${pms:-0}" "${p_ok:-0}" "${lms:-0}" "${l_ok:-0}" \
    "${ttft:-0}" "${dec:-0}" "${id:-none}" "${cpu:-?}" "${el:-?}" "${maxrss:-0}" "${rc:-?}" >> "$TSV"
  grep -m1 ARM_DIDNOTRUN "$L" || true
  grep -m1 ARM_DEGENERATE "$L" || true
  tail -1 "$TSV" | column -t
}

echo "== prefill_d1 $(date -Is)  REPS=$REPS  A=$(basename $A):$(cut -c1-12 <<<"$shaA")  B=$(basename $B):$(cut -c1-12 <<<"$shaB")"
for r in $(seq 1 "$REPS"); do
  one "pfA_r$r"     "$A" ""
  one "pfB_off_r$r" "$B" "STRATA_STAGER_SLEEP=0"
  one "pfB_on_r$r"  "$B" "STRATA_STAGER_SLEEP=1"
done
echo "== prefill_d1 DONE $(date -Is); table: $TSV"
column -t "$TSV"
