#!/usr/bin/env bash
# /home/bob/step4/build_a_b.sh -- build the two binaries for the D1 prefill A/B, with identity assertions.
#
#   A = HEAD (clean tree)                     -> bins/strata_vulkan.pfA
#   B = HEAD + D1-1057-prefill-stager-sleep   -> bins/strata_vulkan.pfB
#
#   REFUSES if the tree is dirty (one writer at a time), if the patch does not apply, if the product did not
#   rebuild, or if the two binaries end up the same bytes (a no-op build VOIDS the comparison).  It leaves the
#   tree CLEAN (patch reverted, kept staged in forktest/stage) and the live BUILD.stamp describing the clean
#   HEAD product, with a per-binary copy of each stamp beside the binary.
set -u
REPO=/home/bob/strata-vulkan-wt
BD=/home/bob/vkbuild-vulkan
PROD=$BD/vulkan/strata_vulkan
BINS=/home/bob/step4/bins
PATCH=/home/bob/forktest/stage/MINE-0.1.40.2/drafts/D1-1057-prefill-stager-sleep.patch
D1_SOURCE=src/prefill/prefill.cpp

die(){ echo "BUILD_A_B REFUSES: $*" >&2; exit 2; }

porc=$(git -C "$REPO" status --porcelain)
[ -z "$porc" ] || { echo "$porc"; die "the tree is DIRTY - one writer at a time (commit or stash first)"; }
head=$(git -C "$REPO" rev-parse HEAD)
echo "== build_a_b $(date -Is)  HEAD=$head $(git -C "$REPO" log --oneline -1 | cut -c1-60)"
[ -f "$PATCH" ] || die "patch not found: $PATCH"
git -C "$REPO" apply --check "$PATCH" || die "D1 does not apply cleanly (--check failed)"

# ---------- B: HEAD + D1 -----------------------------------------------------------------------------
git -C "$REPO" apply "$PATCH" || die "D1 apply failed after a successful --check"
if ! bash /home/bob/step4/build_product.sh --sources "$REPO/$D1_SOURCE"; then
  git -C "$REPO" checkout -- "$D1_SOURCE"; die "the B build FAILED (patch reverted; tree clean)"
fi
cp -v "$PROD" "$BINS/strata_vulkan.pfB" || die "could not save pfB"
cp -v /home/bob/step4/BUILD.stamp "$BINS/strata_vulkan.pfB.buildstamp"
shaB=$(sha256sum "$BINS/strata_vulkan.pfB" | cut -d' ' -f1)
echo "   pfB sha256=$shaB  (built from HEAD+D1, tree was dirty: expected)"

# ---------- back to HEAD (clean), then A ---------------------------------------------------------------
git -C "$REPO" checkout -- "$D1_SOURCE" || die "could not revert the D1 patch"
porc2=$(git -C "$REPO" status --porcelain)
[ -z "$porc2" ] || { echo "$porc2"; die "the tree is still dirty after reverting D1"; }
if ! bash /home/bob/step4/build_product.sh --sources "$REPO/$D1_SOURCE"; then die "the A build FAILED"; fi
cp -v "$PROD" "$BINS/strata_vulkan.pfA" || die "could not save pfA"
cp -v /home/bob/step4/BUILD.stamp "$BINS/strata_vulkan.pfA.buildstamp"
shaA=$(sha256sum "$BINS/strata_vulkan.pfA" | cut -d' ' -f1)
echo "   pfA sha256=$shaA  (built from clean HEAD)"

[ "$shaA" != "$shaB" ] || die "pfA and pfB are the SAME bytes - the D1 build was a no-op (VOID A/B)"
printf '\nBUILD_A_B ok=yes\n  A (clean HEAD) = %s  sha=%s\n  B (HEAD+D1)    = %s  sha=%s\n  tree porcelain=%s  HEAD=%s\n' \
  "$BINS/strata_vulkan.pfA" "$shaA" "$BINS/strata_vulkan.pfB" "$shaB" "$(git -C "$REPO" status --porcelain | wc -l)" "$head"
