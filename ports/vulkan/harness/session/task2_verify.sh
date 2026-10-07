#!/usr/bin/env bash
# /home/bob/step4/task2_verify.sh -- the router guard-widening falsifiers, on the NEW binary (router_ne, 3cdef9c8).
#  (a) ids: 8-token -> 3aed108cceee, 199-token -> 56a0b28d2de6 (screen for all-zeros d01eee6a3948 and repeat)
#  (b) the router's dispatch count per recorded command buffer: 144 -> 48 (DISP_STAT RECORDED arm)
#  (e) negative control: STRATA_VK_NOBARRIER_REC=1 must still give a WRONG id
set -u
BIN=/home/bob/step4/bins/strata_vulkan.router_ne
LOGD=/home/bob/step4/logs
T8="1 2 3 4 5 6 7 8"
T199="$(printf '1 %.0s' $(seq 1 199))"
one(){  # one <name> <tokens> [VAR=VAL ...]
  local name="$1"; local toks="$2"; shift 2
  echo "[$(date '+%H:%M:%S')] ARM $name env=[$*]"
  env STRATA_VK_BIN="$BIN" "$@" timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$toks\"" || true
  local L="$LOGD/$name.log"
  local id dec sync
  id=$(grep -m1 ARM_OUTPUT_ID "$L" 2>/dev/null | cut -d' ' -f2)
  dec=$(grep -m1 '^decode' "$L" 2>/dev/null | sed 's/^decode *//')
  sync=$(grep -m1 'verify window' "$L" 2>/dev/null | grep -oE 'sync [0-9.]+' | awk '{print $2}')
  local rtr; rtr=$(grep -m1 'RECORDED arm' "$L" | grep -oE 'native_router_top10.spv [0-9]+' | awk '{print $2}')
  local mmv; mmv=$(grep -m1 'RECORDED arm' "$L" | grep -oE 'bf16_mmvf_f32.spv [0-9]+' | awk '{print $2}')
  printf '    RESULT %-12s id=%s sync=%s router_rec=%s mmvf_rec=%s | %s\n' "$name" "${id:-NONE}" "${sync:-NONE}" "${rtr:-NONE}" "${mmv:-NONE}" "${dec:-NONE}"
}
echo "== task2_verify $(date -Is)  BIN=$BIN sha=$(sha256sum $BIN | cut -c1-16)"
one t2_id8_1    "$T8"
one t2_id8_2    "$T8"
one t2_id199_1  "$T199"
one t2_nobar_1  "$T8" STRATA_VK_NOBARRIER_REC=1
echo "== task2_verify DONE $(date -Is)"
