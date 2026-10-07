#!/usr/bin/env bash
# /home/bob/step4/debug_set.sh -- settle the two open questions with raw logs, nothing else touching the card.
#  (B) HEAD binary (a905c27, fb0fd1fa) on the 8-token arm x2  -> does it keep 3aed108cceee?
#  (A) ONE per-family arm, CLEAN (nothing may kill it)        -> does the substitution complete and price?
set -u
T8="1 2 3 4 5 6 7 8"
HEAD=/home/bob/step4/bins/strata_vulkan.head
ABL=/home/bob/vkbuild-vulkan/vulkan/strata_vulkan
echo "== debug_set $(date -Is)"
echo "== HEAD bin sha $(sha256sum $HEAD | cut -c1-16)  ABL bin sha $(sha256sum $ABL | cut -c1-16)"
run() {  # run <name> <bin> [VAR=VAL...]
  local name="$1"; local bin="$2"; shift 2
  echo "[$(date '+%H:%M:%S')] arm $name (bin $(basename $bin)) env=[$*]"
  env STRATA_VK_BIN="$bin" "$@" timeout 600 flock /tmp/b70.lock -c "bash /home/bob/step4/run.sh $name \"$T8\""
  local rc=$?
  local dec; dec=$(grep -m1 '^decode' "/home/bob/step4/logs/$name.log" 2>/dev/null)
  local id;  id=$(grep -m1 'ARM_OUTPUT_ID' "/home/bob/step4/logs/$name.log" 2>/dev/null | cut -d' ' -f2)
  local rounds; rounds=$(grep -m1 'speculation' "/home/bob/step4/logs/$name.log" 2>/dev/null | sed 's/^ *//' | cut -c1-60)
  echo "[$(date '+%H:%M:%S')]   rc=$rc id=$id | $dec | $rounds"
}
run head8_a  "$HEAD"
run fam_qsa_v "$ABL" "STRATA_VK_TRIVIAL_REC_FAMILY=qsa_decode_attn"
run head8_b  "$HEAD"
echo "== debug_set DONE $(date -Is)"
