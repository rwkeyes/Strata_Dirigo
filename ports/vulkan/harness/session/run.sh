#!/usr/bin/env bash
# /home/bob/step4/run.sh <name> <tokens> [extra args...] -- the step4 (family-ablation) arm runner.
# One config per invocation.  Header carries the id COUNT (wc -w), the binary sha256, the spv shas and the env.
# REFUSES on an id count that is neither 199 nor 8 and writes ARM_DIDNOTRUN when the engine produced no decode
# line or decoded 0 tokens (a 0-token arm is a FAILED arm, never a fast one).  Also writes ARM_DEGENERATE when
# the output is all-zeros -- non-finite head logits (sampler_greedy.comp:106) -- which is MANDATORY for ablation
# arms, whose rate is then the degenerate fast path, not a ceiling.
set -u
NAME="$1"; TOKS="$2"; shift 2
LOGD=/home/bob/step4/logs; mkdir -p "$LOGD"
LOG="$LOGD/$NAME.log"
BIN="${STRATA_VK_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}"
SPVD="${STRATA_VK_SPV_DIR:-/home/bob/strata-vulkan-wt/ports/vulkan/shaders}"
n=$(echo $TOKS | wc -w)
if [ "$n" -ne 199 ] && [ "$n" -ne 8 ]; then
  echo "ARM_DIDNOTRUN name=$NAME reason=id-count-mismatch got=$n want=199-or-8"; exit 2
fi
{
  echo "== port arm $NAME | $(date -Is)"
  echo "== ids n=$n (wc -w)  first8: $(echo $TOKS | awk '{for(i=1;i<=8&&i<=NF;i++)printf "%s ",$i}')"
  echo "== args: $*"
  echo "== binary: $BIN"
  echo "== spv dir: $SPVD"
  echo "== bin sha256: $(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "== scale.spv sha256: $(sha256sum "$SPVD/scale.spv" 2>/dev/null | cut -d' ' -f1)"
  echo "== env:"; env | grep -i -E 'strata_(vk|pf|prefill|fgr|router)' | sort
  echo "== cal epoch=$(date +%s.%N) uptime=$(cut -d' ' -f1 /proc/uptime)"
  echo "== run:"
} > "$LOG" 2>&1
cd /home/bob
STRATA_VK_SPV_DIR="$SPVD" \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
STRATA_VK_ARENA_MIB=28560 STRATA_VK_DESKTOP_RESERVE_MIB=256 \
STRATA_VK_DISP_STAT=1 \
/tmp/memguard_swap.sh 45G 16G "$BIN" \
  --pack /media/bob/3d651e2c-e9a4-4758-ba77-863725fe3731/public/strata-gguf/strata-packs/coder-iq1_m \
  --native /home/bob/strata-models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --spec 2 --prefill 256 --tokens "$TOKS" --max-new 32 --max-context 512 \
  --expert-profile /tmp/expert-profile-coder-built.bin --expert-cache 12288 --mmap-experts \
  "$@" >> "$LOG" 2>&1
RC=$?
echo "RUN_RC=$RC" >> "$LOG"
DEC="$(grep -m1 '^decode' "$LOG" | grep -oE '[0-9]+ tokens' | grep -oE '[0-9]+')"
{
  echo "ARM_VERDICT name=$NAME tokens_arg=$n decoded=${DEC:-0}"
  echo "ARM_OUTPUT_ID $(grep '^output ' "$LOG" | md5sum | cut -c1-12)"
} >> "$LOG"

# --- ALL-ZEROS / DEGENERATE SCREEN (mandatory for ablation arms; applied to EVERY arm) ---
# Delegate to screen_output.sh (a testable reporter): the shipped greedy sampler answers token 0 when no
# candidate beats `-inf` (sampler_greedy.comp:106), so an all-zeros output means NON-FINITE HEAD LOGITS -- and
# trivialising work feeds garbage to the head.  The tabulating drivers (parse_flag.py / marginal.py) re-check
# this independently via arm_gate.screen_arm_output, so a rate from a flagged arm cannot be quoted as a ceiling.
bash "${SCREEN_OUTPUT:-/home/bob/step4/screen_output.sh}" "$LOG" "$NAME" >> "$LOG"

echo "ARM_END $NAME $(date -Is)" >> "$LOG"
if [ -z "${DEC:-}" ] || [ "${DEC:-0}" -eq 0 ]; then
  echo "ARM_DIDNOTRUN name=$NAME reason=no-decode-line-or-zero-tokens rc=$RC see=$LOG"
fi
