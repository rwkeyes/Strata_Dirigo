#!/usr/bin/env bash
# /home/bob/step4/assert_sha.sh <label> <file> <want-sha256> -- assert a file's sha256 EQUALS the expected value.
#
# WHY: `bins/strata_vulkan.base9cd885e` is named like a baseline but its bytes are the INSTRUMENTED ablation
# build (sha 714b760c..., TRIVIAL_REC_FAMILY present).  A name is not evidence; this pins the CONTENT so a
# future copy that is not what its name claims fails loudly instead of silently changing what an arm measured.
#
# Prints SHA_ASSERT_OK and exits 0 on a match; SHA_ASSERT_FAIL and exits 2 on a mismatch or a missing file.
set -u
label=${1:?usage: assert_sha.sh <label> <file> <want-sha256>}
file=${2:?usage: assert_sha.sh <label> <file> <want-sha256>}
want=${3:?usage: assert_sha.sh <label> <file> <want-sha256>}
got=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1)
if [ -z "$got" ]; then
  echo "SHA_ASSERT_FAIL label=$label file=$file reason=missing-file"
  exit 2
fi
if [ "$got" != "$want" ]; then
  echo "SHA_ASSERT_FAIL label=$label file=$file reason=sha-mismatch got=$got want=$want"
  exit 2
fi
echo "SHA_ASSERT_OK label=$label file=$file sha=$got"
exit 0
