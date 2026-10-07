#!/usr/bin/env bash
# /home/bob/step4/prefill_reg_ab.sh -- THREE variants in ONE launch, on ONE binary, one env var each:
#
#   A (chain)          STRATA_PREFILL_TIMING=1                                        (the shipped per-token loop)
#   B (chunk KU)       + STRATA_PF_GDN_REC_CHUNK=1                                     (gdn_rec_chunk.spv = KU unrolled)
#   R (chunk REGISTER) + STRATA_PF_GDN_REC_CHUNK=1 STRATA_PF_GDN_REC_CHUNK_SPV=gdn_rec_chunk_reg.spv
#
# WHY ONE LAUNCH: the variants differ by ~1-3% while this machine's absolute numbers drift by more than that
# BETWEEN launches (a cold first arm read 11,949 ms where the warm ones read 9,284).  Interleaving is the only way
# these differences mean anything.
#
# WHAT MUST HOLD: id `56a0b28d2de6` on EVERY arm.  All three claim the shipped step's arithmetic, element for
# element in the same ascending-i order, so an id change REFUTES that claim for that variant and is reported as
# ID_MOVED, not smoothed over.
set -u
LOGD=/home/bob/step4/logs
SPVD=/home/bob/strata-vulkan-wt/ports/vulkan/shaders
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
BASE_ID=56a0b28d2de6
ROUNDS=${ROUNDS:-2}
T199="$(printf '1 %.0s' $(seq 1 199))"
[ -x "$BIN" ] || { echo "PREFILL_REG_AB REFUSES: $BIN is missing"; exit 2; }
SHA=$(sha256sum "$BIN" | cut -c1-16)
echo "== prefill_reg_ab $(date -Is)  BIN sha=$SHA  rounds=$ROUNDS"
printf '   spv: ku=%s (gdn_rec_chunk.spv)   reg=%s (gdn_rec_chunk_reg.spv)\n' \
  "$(sha256sum "$SPVD/gdn_rec_chunk.spv" 2>/dev/null | cut -c1-16)" \
  "$(sha256sum "$SPVD/gdn_rec_chunk_reg.spv" 2>/dev/null | cut -c1-16)"
TSV=/home/bob/step4/prefill_reg_ab.tsv
: > "$TSV"

one() {  # one <name> <env-prefix>
  local name="$1" envs="$2"
  local L="$LOGD/$name.log"
  echo "[$(date '+%H:%M:%S')] ARM $name env=[$envs]"
  env STRATA_VK_BIN="$BIN" $envs timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\"" || true
  local id rec tac tl
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)
  rec=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gdn recurrence \([0-9][0-9]*\) (.*/\1/p')
  tl=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*GPU timeline \([0-9][0-9]*\) ms.*/\1/p')
  tac=$(grep -m1 '^prefill ' "$L" | sed 's/.*tokens in \([0-9.]*\) ms.*-> *\([0-9.]*\) tok.*/\1|\2/')
  grep -m1 ARM_DIDNOTRUN "$L" || true
  grep -m1 ARM_DEGENERATE "$L" || true
  if [ -n "$id" ] && [ "$id" != "$BASE_ID" ]; then echo "  ID_MOVED id=$id (baseline $BASE_ID)"; else echo "  id=$id"; fi
  printf '  gdn_recurrence=%s ms  gpu_timeline=%s ms  prefill=%s ms  %s tok/s\n' \
    "${rec:-NA}" "${tl:-NA}" "$(echo "$tac" | cut -d'|' -f1)" "$(echo "$tac" | cut -d'|' -f2)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$id" "${rec:-NA}" "$(echo "$tac" | cut -d'|' -f1)" "$(echo "$tac" | cut -d'|' -f2)" >> "$TSV"
}

for r in $(seq 1 "$ROUNDS"); do
  one "pxr_A_$r" "STRATA_PREFILL_TIMING=1"
  one "pxr_B_$r" "STRATA_PREFILL_TIMING=1 STRATA_PF_GDN_REC_CHUNK=1"
  one "pxr_R_$r" "STRATA_PREFILL_TIMING=1 STRATA_PF_GDN_REC_CHUNK=1 STRATA_PF_GDN_REC_CHUNK_SPV=gdn_rec_chunk_reg.spv"
done

echo
echo "== per-variant medians (arm, id, gdn_recurrence, prefill, tok/s) =="
awk -F'\t' '{
  v=substr($1,5,1); n[v]++; id[v]=$2; rec[v,n[v]]=$3; pf[v,n[v]]=$4; tk[v,n[v]]=$5;
}
END {
  split("A B R", vs, " ");
  for (i=1;i<=3;i++) { v=vs[i]; m=n[v]; if (m==0) continue;
    # insertion sort on the recurrence
    for (a=1;a<=m;a++) { rr[a]=rec[v,a]; pp[a]=pf[v,a]; tt[a]=tk[v,a] }
    for (a=2;a<=m;a++) { kr=rr[a]; kp=pp[a]; kt=tt[a]; b=a-1;
      while (b>=1 && rr[b]+0 > kr+0) { rr[b+1]=rr[b]; pp[b+1]=pp[b]; tt[b+1]=tt[b]; b-- }
      rr[b+1]=kr; pp[b+1]=kp; tt[b+1]=kt }
    med=(m%2)? rr[(m+1)/2] : (rr[m/2]+rr[m/2+1])/2
    medp=(m%2)? pp[(m+1)/2] : (pp[m/2]+pp[m/2+1])/2
    printf "  %-4s id=%s  n=%d  gdn_recurrence median=%s ms  prefill median=%s ms\n", v, id[v], m, med, medp
    printf "       gdn_recurrence: %s\n", join3(rr, m)
  }
}
function join3(arr, m,   s, i) { s=""; for (i=1;i<=m;i++) s = s (i>1?", ":"") arr[i]; return s }
' "$TSV"
echo "== prefill_reg_ab DONE $(date -Is)"
