#!/usr/bin/env bash
# /home/bob/step4/selftest_degenerate.sh -- self-test for the ALL-ZEROS / DEGENERATE screen.
# Shows the screen FIRING on an all-zeros fixture and STAYING SILENT on a normal one, at every layer that now
# carries it: arm_gate.screen_arm_output (the independent authority), screen_output.sh (what run.sh calls),
# parse_flag.py (the tabulating driver -- a degenerate arm is NOT tabulated), marginal.py (a degenerate CONTROL
# voids the run) and repro_test.sh (three repeats that agree on the all-zeros id are DEGENERATE, not correct).
set -u
cd /home/bob/step4
F=selftest_degen; rm -rf "$F"; mkdir -p "$F"
ZEROS="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
REAL="4653 4653 4653 4653 1 9 91 9 9 9 91 9 91 9 91 9 1 91 9 91 9 9 91 9 91 9 1 91 9 5 9 91"
DEGID=$(printf 'output  : %s\n' "$ZEROS" | md5sum | cut -c1-12)
mk(){ # mk <file> <name> <outputtokens> <id> <tok/s>
  { echo "== port arm $2 | 2026-10-07T00:00:00-04:00"
    echo "== bin sha256: $(printf 'B' | sha256sum | cut -d' ' -f1)"
    echo "== scale.spv sha256: $(printf 'S' | sha256sum | cut -d' ' -f1)"
    echo "output  : $3"
    echo "decode                   32 tokens in 221.3 ms  ->  $5 tok/s"
    echo "ARM_VERDICT name=$2 tokens_arg=8 decoded=32"
    echo "ARM_OUTPUT_ID $4"; } > "$1"
}
mk "$F/deg.log" deg_fixture "$ZEROS" "$DEGID" 144.58
mk "$F/ok.log"  ok_fixture  "$REAL"  3aed108cceee 11.04
echo "all-zeros id = $DEGID (the pinned d01eee6a3948)"

echo "################ ALL-ZEROS / DEGENERATE SELF-TEST ################"
echo
echo "=== T1: arm_gate.screen_arm_output on the ALL-ZEROS fixture -> MUST be degenerate ==="
python3 -c "from arm_gate import screen_arm_output as s; print(s('$F/deg.log'))"
echo
echo "=== T2: arm_gate.screen_arm_output on the NORMAL fixture -> MUST be silent (False, ...) ==="
python3 -c "from arm_gate import screen_arm_output as s; print(s('$F/ok.log'))"
echo
echo "=== T3: screen_output.sh (what run.sh calls) -- FIRES on the degenerate one, SILENT on the normal one ==="
echo "--- degenerate ---"; STRATA_VK_TRIVIAL_REC=1 bash screen_output.sh "$F/deg.log" deg_fixture; echo "T3a exit=$? (want 0, lines emitted)"
echo "--- normal ---";     STRATA_VK_TRIVIAL_REC=1 bash screen_output.sh "$F/ok.log"  ok_fixture;  echo "T3b exit=$? (want 0, NOTHING emitted)"
echo
echo "=== T4: parse_flag.py over BOTH arms -> the degenerate arm MUST be excluded and the driver MUST exit 2 ==="
python3 parse_flag.py --no-identity-gate --out /tmp/selftest_degen_rows.json "$F/ok.log" "$F/deg.log" 2>&1 | sed -n '1,12p'
echo "T4 exit=${PIPESTATUS[0]} (want 2; the row table must NOT contain deg_fixture)"
echo
echo "=== T5: marginal.py -- a degenerate CONTROL (base8_*) voids the run (exit 2) ==="
D=$F/logs; mkdir -p "$D"
mk "$D/fam_zzz_8_1.log" fam_zzz_8_1 "$REAL" 68fa2f714583 7.85
mk "$D/off8_1.log"      off8_1      "$REAL" 3aed108cceee 11.05
mk "$D/base8_1.log"     base8_1     "$ZEROS" "$DEGID" 12.51
MARGINAL_LOGD="$D" MARGINAL_JSON=/tmp/selftest_marginal.json \
  python3 marginal.py --no-identity-gate --want-bin-sha "$(printf 'B' | sha256sum | cut -d' ' -f1)" 2>&1 | sed -n '1,14p'
echo "T5 exit=${PIPESTATUS[0]} (want 2; MARGINAL_REFERENCE_DEGENERATE must be printed)"
echo
echo "=== T6: marginal.py with a NORMAL control -> no reference-degenerate, exit 0 ==="
mk "$D/base8_1.log" base8_1 "$REAL" 3aed108cceee 12.51
MARGINAL_LOGD="$D" MARGINAL_JSON=/tmp/selftest_marginal.json \
  python3 marginal.py --no-identity-gate --want-bin-sha "$(printf 'B' | sha256sum | cut -d' ' -f1)" 2>&1 | sed -n '1,12p'
echo "T6 exit=${PIPESTATUS[0]} (want 0)"
echo
echo "=== T7: repro_test.sh -- three repeats agreeing on the ALL-ZEROS id MUST be REPRO_DEGENERATE (exit 2) ==="
REPRO_IDS_OVERRIDE="$DEGID $DEGID $DEGID" bash repro_test.sh
echo "T7 exit=$? (want 2)"
echo
echo "=== T8: repro_test.sh -- three repeats on the REAL id -> REPRO_OK, naming the id (exit 0) ==="
REPRO_IDS_OVERRIDE="3aed108cceee 3aed108cceee 3aed108cceee" bash repro_test.sh
echo "T8 exit=$? (want 0)"
echo
echo "=== T9: repro_test.sh -- ids that DISAGREE -> REPRO_FAIL (exit 1) ==="
REPRO_IDS_OVERRIDE="3aed108cceee $DEGID 3aed108cceee" bash repro_test.sh
echo "T9 exit=$? (want 1)"
echo "################ END ALL-ZEROS / DEGENERATE SELF-TEST ################"
