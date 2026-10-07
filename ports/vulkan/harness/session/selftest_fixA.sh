#!/usr/bin/env bash
# /home/bob/step4/selftest_fixA.sh -- THE FIX-A SELF-TEST.
# Builds three fixture arm logs whose stamped bin/scale.spv sha256 are KNOWN, then runs the tabulating
# driver against the intended build's sha and shows it REFUSE the two wrong arms while tabulating the good one.
# Then flips the intended value so even the good arm is refused (the checker must fail when it should), and
# shows the no-expectation behaviour (refuse to tabulate at all).
set -u
cd /home/bob/step4
FIX=selftest_fixA; rm -rf "$FIX"; mkdir -p "$FIX"
INTEND_BIN=$(printf 'INTENDED-BUILD'      | sha256sum | cut -d' ' -f1)
INTEND_SPV=$(printf 'INTENDED-SCALE-SPV'  | sha256sum | cut -d' ' -f1)
WRONG_BIN=$(printf  'STALE-BUILD'         | sha256sum | cut -d' ' -f1)
WRONG_SPV=$(printf  'STALE-SCALE-SPV'     | sha256sum | cut -d' ' -f1)

mk(){ # mk <file> <bin-sha> <spv-sha> <arm-name>
  cat > "$1" <<EOF
== port arm $4 | 2026-10-07T00:00:00-04:00
== ids n=8 (wc -w)  first8: 1 2 3 4 5 6 7 8
== args:
== binary: /fixture/$4
== spv dir: /fixture/shaders
== bin sha256: $2
== scale.spv sha256: $3
== env:
STRATA_VK_BIN=/fixture/$4
== run:
vk kernel time qsa_decode_attn.spv disp=100 wg=800 gpu_ms=1.000 us/disp=10.000 ns/wg=1.250
vk fp fam qsa_decode_attn.spv n=8 fp=4096 B bar_us=12.5 kern_us=44.0
vk disp stat RECORDED arm (100 dispatches encoded; qsa_decode_attn.spv 100
EOF
}
mk "$FIX/good.log"    "$INTEND_BIN" "$INTEND_SPV" fam_selftest_good
mk "$FIX/badbin.log"  "$WRONG_BIN"  "$INTEND_SPV" fam_selftest_badbin
mk "$FIX/badspv.log"  "$INTEND_BIN" "$WRONG_SPV"  fam_selftest_badspv

echo "################ FIX A SELF-TEST ################"
echo "INTENDED bin sha = $INTEND_BIN"
echo "INTENDED spv sha = $INTEND_SPV"
echo "WRONG    bin sha = $WRONG_BIN"
echo "WRONG    spv sha = $WRONG_SPV"
echo
echo "=== T1: arm_gate.py with the correct intended build -> the 2 wrong arms MUST be refused ==="
python3 arm_gate.py --want-bin-sha "$INTEND_BIN" --want-spv-sha "$INTEND_SPV" \
        "$FIX/good.log" "$FIX/badbin.log" "$FIX/badspv.log"
echo "T1 exit=$? (want 2)"

echo
echo "=== T2: the tabulating driver parse_flag.py MUST refuse to tabulate the 2 wrong arms ==="
python3 parse_flag.py --want-bin-sha "$INTEND_BIN" --want-spv-sha "$INTEND_SPV" --out /tmp/selftest_A_rows.json \
        "$FIX/good.log" "$FIX/badbin.log" "$FIX/badspv.log"
echo "T2 exit=$? (want 2; table must contain ONLY fam_selftest_good)"

echo
echo "=== T3 (negative control): give the WRONG value as intended -> even the good arm MUST be refused ==="
python3 arm_gate.py --want-bin-sha "$WRONG_BIN" --want-spv-sha "$WRONG_SPV" \
        "$FIX/good.log"
echo "T3 exit=$? (want 2)"

echo
echo "=== T4: correct arm alone MUST be admitted, exit 0 ==="
python3 arm_gate.py --want-bin-sha "$INTEND_BIN" --want-spv-sha "$INTEND_SPV" "$FIX/good.log"
echo "T4 exit=$? (want 0)"

echo
echo "=== T5: no intended build given -> driver MUST refuse to tabulate (exit 2) ==="
python3 parse_flag.py "$FIX/good.log"
echo "T5 exit=$? (want 2)"

echo
echo "=== T6: --no-identity-gate opts out but says so loudly ==="
python3 parse_flag.py --no-identity-gate --out /tmp/selftest_A_rows.json "$FIX/good.log" | head -3
echo "T6 exit=${PIPESTATUS[0]} (want 0)"
echo "################ END FIX A SELF-TEST ################"
