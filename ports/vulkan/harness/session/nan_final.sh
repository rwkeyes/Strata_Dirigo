#!/usr/bin/env bash
# /home/bob/step4/nan_final.sh -- run AFTER nan_rate_sweep.sh on the FINAL instrument binary (nan5).
#   poisons: prove the detector FIRES and names each stage, including the sampler stage (S5).
#   valid x2 (default tail) and x2 with STRATA_VK_SAMPLER_F32=1 (D4's f32 sibling): must be bit-identical ids.
set -u
LOGD=/home/bob/step4/logs; mkdir -p "$LOGD"
BIN=/home/bob/step4/bins/strata_vulkan.nan5
T8="1 2 3 4 5 6 7 8"
OUT=/home/bob/step4/nan_final.tsv
: > "$OUT"
printf 'label\tid\tarmed\tfirst_stage\trounds\ttok_s\trc\n' >> "$OUT"

one() { # $1=name $2=envprefix
  flock /tmp/b70.lock -c "STRATA_VK_BIN=$BIN STRATA_DBG_NAN=1 $2 bash /home/bob/step4/run.sh $1 \"$T8\""
  local L="$LOGD/$1.log" id armed st rd tk rc
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | awk '{print $2}')
  grep -q 'instrument ARMED' "$L" && armed=yes || armed=no
  st=$(grep -m1 'FIRST NON-FINITE STAGE' "$L" | sed 's/.*FIRST NON-FINITE STAGE=//' | awk '{print $1}')
  [ -z "$st" ] && st=-
  rd=$(grep -m1 speculation "$L" | grep -oE '[0-9]+ rounds' | grep -oE '[0-9]+')
  tk=$(grep -m1 '^decode' "$L" | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+')
  rc=$(grep -m1 RUN_RC "$L" | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "${id:-none}" "$armed" "$st" "${rd:-0}" "${tk:-0}" "${rc:-?}" >> "$OUT"
  tail -1 "$OUT"
  grep -m1 'strata dbg NAN: pos0' "$L" || true
}

echo "[$(date -Is)] POISON CONTROLS on nan5 (each must name its stage)"
one pz2_s1_head_gr_read      "STRATA_DBG_NAN_POISON=head_gr_read"
one pz2_s2_act_quantiser     "STRATA_DBG_NAN_POISON=act_quantiser"
one pz2_s3_head_projection   "STRATA_DBG_NAN_POISON=head_projection"
one pz2_s5_sampler           "STRATA_DBG_NAN_POISON=sampler"

echo "[$(date -Is)] DEFAULT tail valid arms (nan5)"
one n5_valid_a ""
one n5_valid_b ""

echo "[$(date -Is)] F32 sibling tail (D4 STRATA_VK_SAMPLER_F32=1) valid arms"
one n5_f32_a "STRATA_VK_SAMPLER_F32=1"
one n5_f32_b "STRATA_VK_SAMPLER_F32=1"
echo "[$(date -Is)] FINAL DONE"
