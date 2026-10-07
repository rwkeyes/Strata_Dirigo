#!/usr/bin/env bash
# /home/bob/step4/selftest_assert_diff.sh -- self-test for assert_diff.sh (gap 2's assertion helper).
# gate_attrib.sh (clean-vs-dirty gate binary) and sweep8.sh (NEW-vs-BASE binary) now call this; the checker
# must FAIL when the two builds are the same bytes (a no-op `git stash`, a mislabelled baseline) and PASS when
# they differ.
set -u
cd /home/bob/step4
echo "################ assert_diff SELF-TEST ################"
A=$(printf 'build-A' | sha256sum | cut -d' ' -f1)
B=$(printf 'build-B' | sha256sum | cut -d' ' -f1)
echo "A = $A"
echo "B = $B"
echo
echo "=== T1: two DIFFERENT hashes -> MUST pass (exit 0) ==="
bash assert_diff.sh "clean-vs-dirty gate binary" "$A" "$B"; echo "T1 exit=$? (want 0)"
echo
echo "=== T2: two IDENTICAL hashes (the no-op git stash) -> MUST fail loudly (exit 2) ==="
bash assert_diff.sh "clean-vs-dirty gate binary" "$A" "$A"; echo "T2 exit=$? (want 2)"
echo
echo "=== T3: an EMPTY hash (a missing binary) -> MUST fail (exit 2), never read as 'different' ==="
bash assert_diff.sh "NEW-vs-BASE binary" "$A" ""; echo "T3 exit=$? (want 2)"
echo
echo "=== T4: BOTH empty -> MUST fail (exit 2) ==="
bash assert_diff.sh "NEW-vs-BASE binary" "" ""; echo "T4 exit=$? (want 2)"
echo
echo "=== T5: the CALLER sweep8.sh, NEW and BASE the SAME bytes -> MUST refuse (exit 2, no arm runs) ==="
S=selftest_diff; rm -rf "$S"; mkdir -p "$S"
printf 'same-build\n' > "$S/same"; cp "$S/same" "$S/same2"; chmod +x "$S/same" "$S/same2"
printf 'fam_zzz\n' > "$S/families"
SWEEP_NEW="$PWD/$S/same" SWEEP_BASE="$PWD/$S/same2" bash sweep8.sh "$PWD/$S/families" 1 2>&1 | sed -n '1,6p'
echo "T5 exit=${PIPESTATUS[0]} (want 2)"
echo
echo "=== T6: the CALLER sweep8.sh, NEW and BASE DIFFERENT -> assertion PASSES (check-only, no arm runs) ==="
printf 'other-build\n' > "$S/other"; chmod +x "$S/other"
SWEEP_NEW="$PWD/$S/other" SWEEP_BASE="$PWD/$S/same" SWEEP_CHECK_ONLY=1 bash sweep8.sh "$PWD/$S/families" 1 2>&1 | sed -n '1,6p'
echo "T6 exit=${PIPESTATUS[0]} (want 0)"
echo "################ END assert_diff SELF-TEST ################"
