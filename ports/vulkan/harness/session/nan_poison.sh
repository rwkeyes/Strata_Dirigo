#!/usr/bin/env bash
# /home/bob/step4/nan_poison.sh -- run AFTER nan_rate_sweep.sh: prove the detector FIRES and names the poisoned stage.
# Also one plain nan3 valid arm, to show nan3 (poison fixed) still gives the 3aed108cceee baseline.
set -u
LOGD=/home/bob/step4/logs; mkdir -p "$LOGD"
BIN=/home/bob/step4/bins/strata_vulkan.nan3
T8="1 2 3 4 5 6 7 8"
OUT=/home/bob/step4/nan_poison.tsv
: > "$OUT"
printf 'label\tid\tarmed\tfirst_stage\trc\n' >> "$OUT"

one() { # $1=name $2=envprefix
  flock /tmp/b70.lock -c "STRATA_VK_BIN=$BIN STRATA_DBG_NAN=1 $2 bash /home/bob/step4/run.sh $1 \"$T8\""
  local L="$LOGD/$1.log" id armed st rc
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | awk '{print $2}')
  grep -q 'instrument ARMED' "$L" && armed=yes || armed=no
  st=$(grep -m1 'FIRST NON-FINITE STAGE' "$L" | sed 's/.*FIRST NON-FINITE STAGE=//' | awk '{print $1}')
  [ -z "$st" ] && st=-
  rc=$(grep -m1 RUN_RC "$L" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "${id:-none}" "$armed" "$st" "${rc:-?}" >> "$OUT"
  tail -1 "$OUT"
  grep -m1 'strata dbg NAN: pos0' "$L" || echo "   (no fire line)"
}

echo "[$(date -Is)] nan3 plain valid arm (baseline identity)"
one nan3_valid ""
echo "[$(date -Is)] poison controls (must name the poisoned stage)"
one pzc_head_gr_read     "STRATA_DBG_NAN_POISON=head_gr_read"
one pzc_act_quantiser    "STRATA_DBG_NAN_POISON=act_quantiser"
one pzc_head_projection  "STRATA_DBG_NAN_POISON=head_projection"
echo "[$(date -Is)] POISON DONE"
