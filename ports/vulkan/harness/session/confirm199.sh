#!/usr/bin/env bash
# /home/bob/step4/confirm199.sh <family> [<family> ...]
# Confirm the top families on the 199-token arm: the SAME shape as the 8-token sweep, interleaved n=3 against the
# saved HEAD binary, one arm per process, every GPU step under flock /tmp/b70.lock.  BOUND arms (garbage output).
set -u
[ "$#" -ge 1 ] || { echo "usage: confirm199.sh <family> [<family>...]"; exit 2; }
T199="$(printf '1 %.0s' $(seq 1 199))"
[ "$(echo $T199 | wc -w)" -eq 199 ] || { echo "CONFIRM REFUSES: 199-list has $(echo $T199|wc -w) ids"; exit 2; }
NEW=/home/bob/vkbuild-vulkan/vulkan/strata_vulkan
BASE=/home/bob/step4/bins/strata_vulkan.head
one() {  # one <name> <bin> [VAR=VAL...]
  local name="$1"; local bin="$2"; shift 2
  echo "[$(date '+%H:%M:%S')] arm $name bin=$(basename "$bin") env=[$*]"
  env STRATA_VK_BIN="$bin" "$@" timeout 900 flock /tmp/b70.lock -c "bash /home/bob/step4/run.sh $name \"$T199\""
  echo "[$(date '+%H:%M:%S')]   rc=$? $(grep -m1 '^decode' /home/bob/step4/logs/$name.log 2>/dev/null | sed 's/^decode *//') id=$(grep -m1 ARM_OUTPUT_ID /home/bob/step4/logs/$name.log 2>/dev/null | cut -d' ' -f2)"
}
echo "confirm199: families=[$*] x3 reps; NEW=$(sha256sum $NEW|cut -c1-16) BASE=$(sha256sum $BASE|cut -c1-16)"
for r in 1 2 3; do
  one "base199c_$r" "$BASE"
  for f in "$@"; do one "fam_${f}_199_$r" "$NEW" "STRATA_VK_TRIVIAL_REC_FAMILY=$f"; done
done
echo "CONFIRM199 DONE $(date -Is)"
