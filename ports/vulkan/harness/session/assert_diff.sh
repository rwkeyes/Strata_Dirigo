#!/usr/bin/env bash
# /home/bob/step4/assert_diff.sh <label> <sha_a> <sha_b> -- assert two build hashes DIFFER.
#
# WHY: `gate_attrib.sh` prints the clean-vs-dirty gate-binary shas but never compared them, so a no-op
# `git stash` would leave 'clean == dirty' and VOID the attribution (the attribution is what exonerated the
# instrument).  `sweep8.sh` prints NEW/BASE shas without asserting they differ, so the ablation and its
# baseline could be the same bytes and every off8/fam_* row would be a control against itself.
#
# Prints DIFF_ASSERT_OK and exits 0 when the two are non-empty AND different; DIFF_ASSERT_FAIL and exits 2
# when they are identical or either is empty (a missing binary hashes to "" and must not read as "different").
set -u
label=${1:?usage: assert_diff.sh <label> <sha_a> <sha_b>}
a=${2-}; b=${3-}
if [ -z "$a" ] || [ -z "$b" ]; then
  echo "DIFF_ASSERT_FAIL label=$label reason=empty-hash a='${a:-}' b='${b:-}'"
  exit 2
fi
if [ "$a" = "$b" ]; then
  echo "DIFF_ASSERT_FAIL label=$label reason=identical a=$a b=$b"
  exit 2
fi
echo "DIFF_ASSERT_OK label=$label a=$a b=$b"
exit 0
