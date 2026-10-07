#!/usr/bin/env bash
# /home/bob/step4/sweep8.sh <families-file> [reps]
# The per-family MARGINAL-price sweep on the 8-token arm.  Interleaves, per rep:
#   base8_r  = the SAVED BASELINE binary (rebuilt-neutrality reference)
#   off8_r   = the ABLATION binary with NO family selected (== baseline behaviour; the control)
#   fam_<F>  = the ABLATION binary with STRATA_VK_TRIVIAL_REC_FAMILY=F (bound: output is garbage)
# Every GPU step is under flock /tmp/b70.lock.  One arm per process; ids counted with wc -w inside run.sh.
# NEW (the ablation build) and BASE (the saved baseline) are ASSERTED to be different bytes before any arm
# runs -- printing the two shas without comparing them is the "settings that silently do nothing" class.
set -u
FAMFILE=${1:?usage: sweep8.sh <families-file> [reps]}
REPS=${2:-3}
T8="1 2 3 4 5 6 7 8"
NEW=${SWEEP_NEW:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
BASE=${SWEEP_BASE:-/home/bob/step4/bins/strata_vulkan.head}
FAMS=$(grep -vE '^\s*#|^\s*$' "$FAMFILE" | tr '\n' ' ')
[ "$(echo $T8 | wc -w)" -eq 8 ] || { echo "SWEEP REFUSES: 8-list has $(echo $T8 | wc -w) ids (need 8)"; exit 2; }
[ -x "$NEW" ] && [ -x "$BASE" ] || { echo "SWEEP REFUSES: a binary is missing"; exit 2; }
[ -n "$FAMS" ] || { echo "SWEEP REFUSES: empty family list"; exit 2; }
NEW_SHA=$(sha256sum "$NEW" | cut -d' ' -f1)
BASE_SHA=$(sha256sum "$BASE" | cut -d' ' -f1)
# THE ASSERTION: the ablation binary and the saved baseline MUST be different bytes.  Printing the two shas
# without comparing them let a stale/mislabelled $BASE be the SAME build as $NEW, which would make every
# off8_*/fam_* row a control against itself (the "settings that silently do nothing" class).
if ! bash /home/bob/step4/assert_diff.sh "sweep8 NEW-vs-BASE binary" "$NEW_SHA" "$BASE_SHA"; then
  echo "SWEEP REFUSES: NEW == BASE (identical binaries) -- the ablation build and the saved baseline must differ" >&2
  echo "  NEW=$NEW (${NEW_SHA:-none}) BASE=$BASE (${BASE_SHA:-none}); every off8/fam_* row would be void" >&2
  exit 2
fi
echo "sweep8: $(wc -w <<<"$FAMS") families [$FAMS] x $REPS reps; NEW=$(cut -c1-16 <<<"$NEW_SHA") BASE=$(cut -c1-16 <<<"$BASE_SHA") ASSERTED-DIFFERENT"
if [ "${SWEEP_CHECK_ONLY:-0}" = 1 ]; then
  echo "SWEEP_ASSERT_ONLY the NEW != BASE assertion PASSED; arms NOT run (SWEEP_CHECK_ONLY=1)"
  exit 0
fi
one() {   # one <name> <bin> [VAR=VAL...]
  local name="$1"; local bin="$2"; shift 2
  echo "[$(date '+%H:%M:%S')] arm $name bin=$(basename "$bin") env=[$*]"
  env STRATA_VK_BIN="$bin" "$@" timeout 900 flock /tmp/b70.lock -c "bash /home/bob/step4/run.sh $name \"$T8\""
  local rc=$?
  local dec; dec=$(grep -m1 '^decode' "/home/bob/step4/logs/$name.log" 2>/dev/null | grep -oE '^decode\s+[0-9]+' | grep -oE '[0-9]+')
  echo "[$(date '+%H:%M:%S')]   rc=$rc decoded=${dec:-0} $(grep -m1 '^decode' /home/bob/step4/logs/$name.log 2>/dev/null)"
}
for r in $(seq 1 "$REPS"); do
  one "base8_$r" "$BASE"
  one "off8_$r"  "$NEW"
  for f in $FAMS; do
    one "fam_${f}_8_$r" "$NEW" "STRATA_VK_TRIVIAL_REC_FAMILY=$f"
  done
done
echo "SWEEP8 DONE $(date -Is)"
