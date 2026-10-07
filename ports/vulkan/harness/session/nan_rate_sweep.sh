#!/usr/bin/env bash
# /home/bob/step4/nan_rate_sweep.sh -- measure the all-zeros (non-finite head) rate and the first offending stage.
#   CONTROLS (instrument self-test): poison S1/S2/S3 -> the detector MUST name the poisoned stage.
#   ARM A (known trigger): STRATA_VK_TRIVIAL_REC=1            -> the shipped d01eee6a3948 all-zeros.
#   ARM B (measure)      : n=20 identical valid arms          -> the rate table.
# Every arm writes a UNIQUE, never-overwritten log ($LOGD/<name>.log).
set -u
LOGD=/home/bob/step4/logs; mkdir -p "$LOGD"
BIN=/home/bob/step4/bins/strata_vulkan.nan2
T8="1 2 3 4 5 6 7 8"
OUT=/home/bob/step4/nan_rate.tsv
: > "$OUT"
printf 'label\tid\trounds\tdecoded\ttok_s\tzeros\tarmed\tfirst_stage\trc\n' >> "$OUT"

emit() {  # $1=label  $2=logpath
  local L="$2" id rounds dec tok zeros armed stage rc
  id=$(grep -m1 'ARM_OUTPUT_ID' "$L" | awk '{print $2}')
  rounds=$(grep -m1 'speculation' "$L" | grep -oE '[0-9]+ rounds' | grep -oE '[0-9]+')
  dec=$(grep -m1 '^decode' "$L" | grep -oE '[0-9]+ tokens in' | grep -oE '[0-9]+')
  tok=$(grep -m1 '^decode' "$L" | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+')
  if grep -qE '^output  : 0( 0){31}$' "$L"; then zeros=ALLZERO; else zeros=no; fi
  if grep -q 'instrument ARMED' "$L"; then armed=yes; else armed=no; fi
  stage=$(grep -m1 'FIRST NON-FINITE STAGE' "$L" | sed 's/.*FIRST NON-FINITE STAGE=//' | awk '{print $1}')
  [ -z "$stage" ] && stage=-
  rc=$(grep -m1 'RUN_RC' "$L" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "${id:-none}" "${rounds:-0}" "${dec:-0}" "${tok:-0}" "$zeros" "$armed" "$stage" "${rc:-?}" >> "$OUT"
}

run_arm() {  # $1=name  $2=extra env string
  flock /tmp/b70.lock -c "STRATA_VK_BIN=$BIN STRATA_DBG_NAN=1 $2 bash /home/bob/step4/run.sh $1 \"$T8\""
  emit "$1" "$LOGD/$1.log"
  tail -1 "$OUT"
}

echo "[$(date -Is)] CONTROLS: poison each stage (must be named)"
run_arm pz_s1 "STRATA_DBG_NAN_POISON=head_gr_read"
run_arm pz_s2 "STRATA_DBG_NAN_POISON=act_quantiser"
run_arm pz_s3 "STRATA_DBG_NAN_POISON=head_projection"

echo "[$(date -Is)] ARM A: trivial trigger (known d01eee6a3948)"
run_arm nan_trivial_3 "STRATA_VK_TRIVIAL_REC=1"

echo "[$(date -Is)] ARM B: 20 valid arms with the NAN instrument"
for r in $(seq 1 20); do
  echo "[$(date '+%H:%M:%S')] valid run $r/20"
  run_arm "nanr_$r" ""
done
echo "[$(date -Is)] SWEEP DONE"
