#!/usr/bin/env bash
# /home/bob/step4/gate_attrib.sh -- attribute the radeon gate arm: CLEAN a905c27 (instrument reverted) vs DIRTY
# (instrument applied).  Runs the full gate both ways (it is ~5 min) THROUGH gate_run.sh, so each run's own log
# records the gate binary's sha256 + the commit + the `git status --porcelain` state; hashes the gate binary each
# time so the comparison is attributable at the BINARY level; ASSERTS the two gate binaries DIFFER (a no-op
# `git stash` would leave 'clean == dirty' and VOID the attribution -- and the attribution is what exonerated the
# instrument); and extracts the radeon totals + the ple_block magnitudes.
set -u
R=/home/bob/strata-vulkan-wt
L=/home/bob/step4/logs
sha(){ sha256sum "$R/ports/vulkan/harness/build/vk_gate" 2>/dev/null | cut -d' ' -f1; }
SHA_CLEAN=""; SHA_DIRTY=""
run(){  # run <tag>
  local tag="$1"
  echo "[$(date '+%H:%M:%S')] gate $tag starting gate-bin-sha=$(sha)"
  GATE_REPO="$R" GATE_LOGD="$L" GATE_TIMEOUT=1200 bash /home/bob/step4/gate_run.sh "$tag"
  local rc=$?
  local s; s="$(sha)"
  echo "[$(date '+%H:%M:%S')] gate $tag rc=$rc gate-bin-sha=$s"
  if [ "$tag" = clean ]; then SHA_CLEAN="$s"; else SHA_DIRTY="$s"; fi
}
echo "== gate_attrib $(date -Is)"
echo "== tree diff BEFORE: $(cd $R && git status --short)"
# --- CLEAN: revert the instrument only ---
cd "$R" && git stash push -m "step4-gate-attrib" -- vulkan/src/device/vk_compute.cpp >/dev/null
echo "== clean tree: $(cd $R && git status --short | tr '\n' ' ')"
run clean
# --- DIRTY: re-apply the instrument ---
cd "$R" && git stash pop >/dev/null
echo "== dirty tree: $(cd $R && git status --short | tr '\n' ' ')"
run dirty
# --- THE ASSERTION THE ATTRIBUTION RESTS ON: the CLEAN and DIRTY gate binaries MUST be different bytes. ---
if ! bash /home/bob/step4/assert_diff.sh "gate_attrib clean-vs-dirty gate binary" "$SHA_CLEAN" "$SHA_DIRTY"; then
  echo "GATE_ATTRIB_VOID clean=${SHA_CLEAN:-none} dirty=${SHA_DIRTY:-none} -- the two gate binaries are" >&2
  echo "  byte-identical, so the clean-vs-dirty radeon comparison below compares a build with ITSELF: VOID." >&2
  exit 2
fi
echo "== radeon totals =="
for t in clean dirty; do
  printf '%-6s %s\n' "$t" "$(grep -E 'radeon_icd +==' $L/gate_$t.log | head -1 | sed 's/^ *//')"
done
echo "== ple_block lines =="
for t in clean dirty; do echo "--- $t ---"; grep -E "FAIL .*ple_block|FAIL .*fused_gdn_ab|FAIL .*bf16_gemv" $L/gate_$t.log | head -12; done
echo "== intel totals =="
for t in clean dirty; do printf '%-6s %s\n' "$t" "$(grep -E 'intel_icd +==' $L/gate_$t.log | head -1 | sed 's/^ *//')"; done
echo "== gate_attrib DONE $(date -Is)"
