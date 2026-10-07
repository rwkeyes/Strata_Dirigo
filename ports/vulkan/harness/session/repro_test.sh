#!/usr/bin/env bash
# /home/bob/step4/repro_test.sh -- THE REPRODUCIBILITY TEST.
# Three repeats of ONE identical arm: same binary (saved HEAD), same args, same env, 8-token prompt.
# Compares the output id AND the speculation line (rounds / drafts accepted / tokens per round) across all three,
# and REPORTS WHICH id it saw: three repeats that agree on the ALL-ZEROS id (d01eee6a3948, non-finite head
# logits) are CONSISTENTLY DEGENERATE, not consistently correct -- that is REPRO_DEGENERATE, not REPRO_OK.
# If the ids differ, the baseline is not bit-reproducible and that outranks any perf number.
set -u
T8="1 2 3 4 5 6 7 8"
BIN=/home/bob/step4/bins/strata_vulkan.head
DEGEN_ID=d01eee6a3948
ids=(); specs=()
# Self-test hook: REPRO_IDS_OVERRIDE="<id> <id> <id>" (and REPRO_SPECS_OVERRIDE) skips the engine runs and
# classifies the given values instead -- this is how selftest_degenerate.sh exercises the verdict on CPU.
if [ -n "${REPRO_IDS_OVERRIDE:-}" ]; then
  echo "== repro_test CLASSIFY-ONLY (REPRO_IDS_OVERRIDE set; no engine run)"
  read -r -a ids <<<"${REPRO_IDS_OVERRIDE}"
  read -r -a specs <<<"${REPRO_SPECS_OVERRIDE:-s s s}"
else
  echo "== repro_test $(date -Is)  bin=$(sha256sum $BIN | cut -c1-16)  bin-mtime=$(stat -c %y $BIN)"
  for r in 1 2 3; do
    echo "[$(date '+%H:%M:%S')] repeat $r"
    STRATA_VK_BIN="$BIN" timeout 600 flock /tmp/b70.lock -c "bash /home/bob/step4/run.sh repro8_$r \"$T8\""
    L=/home/bob/step4/logs/repro8_$r.log
    echo "    rc=$(grep -m1 RUN_RC $L | cut -d= -f2) | $(grep -m1 '^decode' $L | sed 's/^decode *//')"
    echo "    id=$(grep -m1 ARM_OUTPUT_ID $L | cut -d' ' -f2) | $(grep -m1 speculation $L | sed 's/^ *//')"
    echo "    accepted: $(grep -m1 'accepted per round' $L | sed 's/^ *//')"
    echo "    output: $(grep -m1 '^output ' $L | cut -c1-70)"
    ids+=("$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)")
    specs+=("$(grep -m1 speculation "$L" | sed 's/^ *//')")
  done
fi
# The comparison the header always claimed: three identical arms must agree on the id and the speculation line,
# and WHICH id they agree on decides whether that agreement means anything.
if [ "${ids[0]}" = "${ids[1]}" ] && [ "${ids[1]}" = "${ids[2]}" ] \
   && [ "${specs[0]}" = "${specs[1]}" ] && [ "${specs[1]}" = "${specs[2]}" ]; then
  if [ "${ids[0]}" = "$DEGEN_ID" ]; then
    echo "ARM_DEGENERATE name=repro reason=all-zeros-output suspect=non-finite-logits id=${ids[0]} ntok=32" >&2
    echo "REPRO_DEGENERATE ids=${ids[*]} -- the three repeats AGREE, but on the ALL-ZEROS (non-finite-logit) id:" >&2
    echo "                  consistently DEGENERATE, not consistently correct; no rate from this arm is a ceiling" >&2
    echo "== repro_test DEGENERATE $(date -Is)" >&2
    exit 2
  fi
  echo "REPRO_OK ids=${ids[*]} (id=${ids[0]}, not the all-zeros ${DEGEN_ID}) -- bit-reproducible across 3 repeats"
  echo "== repro_test DONE $(date -Is)"
else
  echo "REPRO_FAIL ids=[${ids[*]}] (saw ids ${ids[0]:-NONE}/${ids[1]:-NONE}/${ids[2]:-NONE}) spec differs -- the baseline is NOT bit-reproducible; this outranks any perf number" >&2
  echo "== repro_test FAILED $(date -Is)" >&2
  exit 1
fi
