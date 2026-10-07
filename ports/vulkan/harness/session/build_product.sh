#!/usr/bin/env bash
# /home/bob/step4/build_product.sh -- build the PRODUCT target EXPLICITLY and FAIL LOUDLY when the product
# did not change while its sources did.  Closes the "settings that silently do nothing" class for BUILDS:
#   * a product declared EXCLUDE_FROM_ALL so `cmake --build .` rebuilt every library and left the engine
#     binary untouched (next run reads the UNCHANGED engine and reports "the edit did nothing");
#   * a tree that DID NOT COMPILE, so a stale binary ran.
#
# WHAT IT DOES
#   1. builds ONLY the named target: `cmake --build <build-dir> --target <target>` (never the default target);
#   2. asserts the PRODUCT's sha256 and mtime are consistent with a fresh build -- specifically, when the
#      sources changed since the recorded stamp, the product's mtime must advance (or its sha must change):
#      it FAILS loudly when both are unchanged (target silently not rebuilt), and it FAILS on the stale tell
#      "build artifacts (libs/objects) are newer than the product" and on "a source is newer than the product";
#   3. records commit + `git status --porcelain` + product sha + mtime + source signature into a stamp file
#      (default /home/bob/step4/BUILD.stamp) that the arm identity gate (arm_gate.py) compares arms against;
#   4. prints a one-line verdict: BUILD_VERDICT ok=yes|no ...
#
# The real build tree belongs to a GPU worker: do NOT point this at /home/bob/vkbuild-vulkan while that worker
# holds it.  Use --build-cmd to exercise the comparison logic without cmake (that is how selftest_fixB.sh runs).
set -u
REPO=/home/bob/strata-vulkan-wt
BUILD_DIR=/home/bob/vkbuild-vulkan
TARGET=strata_vulkan
PRODUCT=""
STAMP=/home/bob/step4/BUILD.stamp
SPV_DIR=/home/bob/strata-vulkan-wt/ports/vulkan/shaders
BUILD_CMD=""
ALLOW_UNCHANGED=0
SOURCES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    --product) PRODUCT="$2"; shift 2;;
    --stamp) STAMP="$2"; shift 2;;
    --spv-dir) SPV_DIR="$2"; shift 2;;
    --build-cmd) BUILD_CMD="$2"; shift 2;;
    --sources) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do SOURCES+=("$1"); shift; done;;
    --allow-unchanged) ALLOW_UNCHANGED=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "build_product: unknown arg $1" >&2; exit 3;;
  esac
done
[ "$PRODUCT" ] || PRODUCT="$BUILD_DIR/vulkan/$TARGET"
if [ ${#SOURCES[@]} -eq 0 ]; then SOURCES=("$REPO/src" "$REPO/include" "$REPO/ports/vulkan"); fi

sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
evo(){ stat -c %Y "$1" 2>/dev/null || echo 0; }
eiso(){ stat -c %y "$1" 2>/dev/null || echo none; }
sig(){ # source signature = hash of (mtime,size,path) over every source file, sorted
  : > /tmp/.bp_sig.$$
  for d in "${SOURCES[@]}"; do
    [ -d "$d" ] && find "$d" -type f -printf '%T@ %s %p\n' 2>/dev/null >> /tmp/.bp_sig.$$
  done
  sort -o /tmp/.bp_sig.$$ /tmp/.bp_sig.$$
  sha256sum /tmp/.bp_sig.$$ | cut -d' ' -f1; rm -f /tmp/.bp_sig.$$
}
newest_src(){ # newest mtime among source files
  local m=0; for d in "${SOURCES[@]}"; do
    [ -d "$d" ] || continue
    local x; x=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    [ -n "$x" ] && [ "${x%.*}" -gt "$m" ] 2>/dev/null && m=${x%.*}
  done; echo "$m"; }
newest_sib(){ # newest mtime among build artifacts that are NOT the product (the "libraries" tell)
  local pm="$1" m=0
  while IFS= read -r line; do
    local t=${line%% *}; local p=${line#* }
    [ "$p" = "$pm" ] && continue
    [ "${t%.*}" -gt "$m" ] 2>/dev/null && m=${t%.*}
  done < <(find "$BUILD_DIR" -type f \( -name '*.a' -o -name '*.so' -o -name '*.so.*' -o -name '*.o' \) -printf '%T@ %p\n' 2>/dev/null)
  echo "$m"; }

echo "build_product: repo=$REPO build=$BUILD_DIR target=$TARGET product=$PRODUCT"
b_sha=$(sha "$PRODUCT"); b_mt=$(evo "$PRODUCT")
srcsig=$(sig); srcnew=$(newest_src)
commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)
porc=$(git -C "$REPO" status --porcelain 2>/dev/null || true)
porc_sha=$(printf '%s' "$porc" | sha256sum | cut -d' ' -f1)
echo "build_product: BEFORE product_sha=${b_sha:-none} product_mtime=$(evo "$PRODUCT") sources_newest=$srcnew commit=$commit dirty=$(printf '%s' "$porc" | grep -c . )"

# --- (1) build the PRODUCT target EXPLICITLY ---
if [ -z "$BUILD_CMD" ]; then
  BUILD_CMD='cmake --build "$BUILD_DIR" --target "$TARGET"'
fi
export REPO BUILD_DIR TARGET PRODUCT
set +e
bash -c "$BUILD_CMD"
rc=$?
echo "build_product: build-rc=$rc"

a_sha=$(sha "$PRODUCT"); a_mt=$(evo "$PRODUCT")
sibnew=$(newest_sib "$PRODUCT")

# --- previous stamp (baseline) ---
prev_sha=""; prev_mt=""; prev_sig=""
if [ -f "$STAMP" ]; then
  prev_sha=$(grep -E '^product_sha=' "$STAMP" 2>/dev/null | head -1 | cut -d= -f2-)
  prev_mt=$(grep -E '^product_mtime_epoch=' "$STAMP" 2>/dev/null | head -1 | cut -d= -f2-)
  prev_sig=$(grep -E '^sources_sig=' "$STAMP" 2>/dev/null | head -1 | cut -d= -f2-)
fi
base_sha="${prev_sha:-$b_sha}"; base_mt="${prev_mt:-$b_mt}"

fail=""; reason=""
[ "$rc" -ne 0 ] && { fail=1; reason="BUILD_FAILED rc=$rc"; }
[ -z "$a_sha" ] && { fail=1; reason="BUILD_NO_PRODUCT ($PRODUCT missing after build)"; }

sha_changed=no; mt_adv=no
if [ "$a_sha" != "$base_sha" ]; then sha_changed=yes; fi
if [ "${a_mt:-0}" -gt "${base_mt:-0}" ] 2>/dev/null; then mt_adv=yes; fi
src_changed=unknown
if [ -n "$prev_sig" ]; then [ "$srcsig" = "$prev_sig" ] && src_changed=no || src_changed=yes; fi

if [ -z "$fail" ]; then
  # (2a) sources changed vs stamp but the product did not move -> the target was silently not rebuilt
  if [ "$src_changed" = "yes" ] && [ "$sha_changed" = "no" ] && [ "$mt_adv" = "no" ]; then
    fail=1; reason="BUILD_PRODUCT_UNCHANGED sources changed but product sha AND mtime unchanged (target not rebuilt: EXCLUDE_FROM_ALL / wrong target?)"
  elif [ "$src_changed" = "yes" ] && [ "$mt_adv" = "no" ]; then
    fail=1; reason="BUILD_PRODUCT_MTIME_NOT_ADVANCED product sha changed but mtime did not advance"
  fi
fi
# (2b) the stale tell: libraries/objects rebuilt (new mtimes) with the product still old
if [ -z "$fail" ] && [ "$mt_adv" = "no" ] && [ "${sibnew:-0}" -gt "${a_mt:-0}" ] 2>/dev/null; then
  fail=1; reason="BUILD_PRODUCT_STALE build artifacts newer than the product (sib_mtime=$sibnew product_mtime=$a_mt) -- libs rebuilt, product not"
fi
# (2c) a source is newer than the product -> the product was not built from it
if [ -z "$fail" ] && [ "${srcnew:-0}" -gt "${a_mt:-0}" ] 2>/dev/null; then
  fail=1; reason="BUILD_PRODUCT_STALE_FROM_SOURCE newest source ($srcnew) newer than product ($a_mt)"
fi

if [ -n "$fail" ] && [ "$ALLOW_UNCHANGED" = 1 ] && [ "${reason%% *}" = "BUILD_PRODUCT_UNCHANGED" ]; then
  echo "build_product: WARNING $reason (--allow-unchanged: downgraded)"; fail=""
fi

# --- (3) stamp ---
spv_sha=$(sha "$SPV_DIR/scale.spv")
{
  echo "BUILD_STAMP_V=1"; echo "ts=$(date -Is)"
  echo "repo=$REPO"; echo "build_dir=$BUILD_DIR"; echo "target=$TARGET"; echo "product=$PRODUCT"
  echo "commit=$commit"; echo "porcelain_sha256=$porc_sha"; echo "porcelain_lines=$(printf '%s' "$porc" | grep -c . )"
  echo "product_sha=$a_sha"; echo "product_mtime_epoch=${a_mt:-0}"; echo "product_mtime_iso=$(eiso "$PRODUCT")"
  echo "sources_newest_epoch=${srcnew:-0}"; echo "sources_sig=$srcsig"
  echo "sibling_newest_epoch=${sibnew:-0}"
  echo "spv_dir=$SPV_DIR"; echo "scale_spv_sha=$spv_sha"
} > "$STAMP"

# --- (4) one-line verdict ---
printf 'BUILD_VERDICT ok=%s product_sha=%s product_mtime=%s src_changed=%s product_sha_changed=%s product_mtime_advanced=%s sib_newest=%s' \
  "$([ -z "$fail" ] && echo yes || echo no)" "${a_sha:-none}" "$(eiso "$PRODUCT")" "$src_changed" "$sha_changed" "$mt_adv" "${sibnew:-0}"
[ -n "$fail" ] && printf ' reason=%s\n' "$reason" || printf ' reason=none\n'
[ -n "$fail" ] && { echo "build_product: REFUSED -- $reason"; exit 1; }
echo "build_product: stamp written to $STAMP"
exit 0
