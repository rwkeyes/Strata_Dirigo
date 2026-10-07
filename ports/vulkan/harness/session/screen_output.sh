#!/usr/bin/env bash
# /home/bob/step4/screen_output.sh <arm-log> <name> -- THE ALL-ZEROS / DEGENERATE SCREEN (mandatory).
#
# Reads the arm log's own `output` line and ARM_OUTPUT_ID and, when the output is ALL ZEROS, prints
#   ARM_DEGENERATE name=<arm> reason=all-zeros-output suspect=non-finite-logits id=<id> ntok=<N> ablation=<...>
#   RATE_UNRELIABLE name=<arm> note=...
#
# WHY all-zeros means a numerical fault: the shipped greedy sampler answers token 0 when no candidate beats
# `-inf` (ports/vulkan/shaders/sampler_greedy.comp:106), so an all-zeros decode is NON-FINITE HEAD LOGITS.
# Trivialising work feeds garbage to the head -- the very thing that drives logits to -inf -- so ANY arm that
# intentionally feeds garbage (STRATA_VK_TRIVIAL_REC, ...TRIVIAL_REC_FAMILY, ...NOBARRIER_REC, any bound arm)
# can return an all-zeros trajectory whose rate is the DEGENERATE fast path, NOT a ceiling.  run.sh calls this
# on EVERY arm; the tabulating drivers (parse_flag.py / marginal.py) re-derive the same verdict independently
# through arm_gate.screen_arm_output, so the flag reaches them even if this line is stripped.
#
# Exit 0 always: this is a reporter, not a gate.  Whether a degenerate arm is fatal is the driver's call -- a
# PRICE arm is garbage by construction (label it), a CEILING/control arm is not (refuse it).
set -u
LOG=${1:?usage: screen_output.sh <arm-log> <name>}
NAME=${2:-$(basename "$LOG" .log)}
DEGEN_ID=d01eee6a3948
OUT_TOKS=$(grep -m1 '^output ' "$LOG" 2>/dev/null | sed 's/^output *: *//')
NTOK=$(printf '%s' "$OUT_TOKS" | wc -w)
OID=$(grep -m1 'ARM_OUTPUT_ID ' "$LOG" 2>/dev/null | awk '{print $2}')
DEGEN=0
[ "$OID" = "$DEGEN_ID" ] && DEGEN=1
if [ "$NTOK" -gt 0 ] && [ -z "$(printf '%s' "$OUT_TOKS" | tr -d '0[:space:]')" ]; then DEGEN=1; fi
[ "$DEGEN" = 1 ] || exit 0
ABL=""
for v in STRATA_VK_TRIVIAL_REC STRATA_VK_TRIVIAL_REC_FAMILY STRATA_VK_NOBARRIER_REC STRATA_VK_NOBARRIER; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && ABL="$ABL $v=$val"
done
echo "ARM_DEGENERATE name=$NAME reason=all-zeros-output suspect=non-finite-logits id=${OID:-NONE} ntok=$NTOK ablation=${ABL:-none}"
printf 'RATE_UNRELIABLE name=%s note=the decode rate above is from an all-zeros trajectory (cause MEASURED 2026-10-07: a FINITE constant-zero logits row, not a proven non-finite value) and must NOT be quoted as a ceiling\n' "$NAME"
exit 0
