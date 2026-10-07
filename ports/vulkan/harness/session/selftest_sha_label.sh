#!/usr/bin/env bash
# /home/bob/step4/selftest_sha_label.sh -- self-test for assert_sha.sh (gap 3's content pin).
# task1_arms.sh now asserts that bins/strata_vulkan.base9cd885e is what its NAME is not: the INSTRUMENTED
# ablation build (sha 714b760c..., TRIVIAL_REC_FAMILY present), not a baseline.  The checker must FAIL on a
# copy that is not what its name claims and PASS on the asserted bytes.
set -u
cd /home/bob/step4
F=selftest_sha; rm -rf "$F"; mkdir -p "$F"
printf 'not-the-baseline\n' > "$F/thing"
GOOD=$(sha256sum "$F/thing" | cut -d' ' -f1)
BASE_SHA=$(sha256sum /home/bob/step4/bins/strata_vulkan.head | cut -d' ' -f1)
echo "################ assert_sha SELF-TEST ################"
echo "fixture file sha = $GOOD"
echo "the REAL baseline binary sha (bins/strata_vulkan.head) = $BASE_SHA"
echo
echo "=== T1: correct expected sha -> MUST pass (exit 0) ==="
bash assert_sha.sh "label" "$F/thing" "$GOOD"; echo "T1 exit=$? (want 0)"
echo
echo "=== T2: a copy that is NOT what its name claims (wrong sha) -> MUST fail loudly (exit 2) ==="
bash assert_sha.sh "label" "$F/thing" 0000000000000000000000000000000000000000000000000000000000000000; echo "T2 exit=$? (want 2)"
echo
echo "=== T3: missing file -> MUST fail (exit 2) ==="
bash assert_sha.sh "label" "$F/nope" "$GOOD"; echo "T3 exit=$? (want 2)"
echo
echo "=== T4: the REAL artifact vs its ASSERTED sha (instrumented-ablation, 714b760c...) -> MUST pass (exit 0) ==="
bash assert_sha.sh "task1 bin base9cd885e = instrumented ablation" \
     /home/bob/step4/bins/strata_vulkan.base9cd885e \
     714b760c7d531f9a341ca3e20ede195933c75aff20eed32db1094d08f5d8222d
echo "T4 exit=$? (want 0)"
echo
echo "=== T5: the REAL artifact fed the BASELINE's sha -> MUST fail (exit 2): the name is not the identity ==="
bash assert_sha.sh "task1 bin base9cd885e claimed-as-baseline" \
     /home/bob/step4/bins/strata_vulkan.base9cd885e "$BASE_SHA"
echo "T5 exit=$? (want 2)"
echo
echo "=== T6: task1_arms.sh fed a file that is NOT the asserted bytes -> MUST refuse loudly, run no arm (exit 2) ==="
TASK1_BIN="$PWD/$F/thing" bash task1_arms.sh; echo "T6 exit=$? (want 2)"
echo
echo "=== T7: task1_arms.sh on the REAL artifact -> assertion PASSES (TASK1_CHECK_ONLY stops before the arms) ==="
TASK1_CHECK_ONLY=1 bash task1_arms.sh; echo "T7 exit=$? (want 0)"
echo
echo "=== T8: the assertion keys on the BYTES, not the name (fixture + its own sha) -> PASSES ==="
TASK1_BIN="$PWD/$F/thing" TASK1_BIN_SHA="$GOOD" TASK1_CHECK_ONLY=1 bash task1_arms.sh; echo "T8 exit=$? (want 0)"
echo "################ END assert_sha SELF-TEST ################"
