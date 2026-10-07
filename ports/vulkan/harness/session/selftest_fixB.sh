#!/usr/bin/env bash
# /home/bob/step4/selftest_fixB.sh -- THE FIX-B SELF-TEST.
# Builds a fixture repo + build dir + a fake product, then runs build_product.sh (with a stub builder standing
# in for cmake, because the real build tree is owned by a GPU worker) through the cases that matter:
#   C1 first build, product rebuilt          -> ok=yes
#   C2 a source changed, product NOT rebuilt -> MUST FAIL (BUILD_PRODUCT_UNCHANGED)
#   C3 a source changed, product rebuilt     -> ok=yes, mtime advanced + sha changed
#   C4 no source change, only a lib touched  -> MUST FAIL (BUILD_PRODUCT_STALE: the libraries-newer tell)
set -u
cd /home/bob/step4
F=selftest_fixB
rm -rf "$F/repo" "$F/build" "$F/BUILD.stamp" "$F/BUILD.stamp2"; mkdir -p "$F/repo/src" "$F/build/vulkan" "$F/build/lib"
printf 'int main(){return 0;}\n' > "$F/repo/src/kernel.c"
printf 'PRODUCT v1\n' > "$F/build/vulkan/strata_vulkan"; chmod +x "$F/build/vulkan/strata_vulkan"
printf 'LIB v1\n' > "$F/build/lib/libfoo.a"
touch -d '2026-10-07 00:00:00' "$F/repo/src/kernel.c" "$F/build/vulkan/strata_vulkan" "$F/build/lib/libfoo.a"

run(){ # run <STUB_MODE> "<label>"
  echo "--- $2 (STUB_MODE=$1) ---"
  STUB_MODE="$1" bash build_product.sh \
      --repo "$F/repo" --build-dir "$F/build" --target strata_vulkan \
      --stamp "$F/BUILD.stamp" --sources "$F/repo/src" \
      --build-cmd 'bash /home/bob/step4/selftest_fixB/stub_build.sh'
  echo "<<exit=$?>>"
  echo
}

echo "################ FIX B SELF-TEST (stub builder; real cmake/GPU tree NOT used) ################"
run rebuild "C1 first build, product rebuilt -> want ok=yes"
# C2: change a source, then a build that does NOT update the product.
sleep 1; printf 'int main(){return 1;}\n' > "$F/repo/src/kernel.c"
echo "== source edited at $(stat -c %y "$F/repo/src/kernel.c"); stamp product_mtime=$(grep ^product_mtime_epoch= "$F/BUILD.stamp" | cut -d= -f2)"
run noop "C2 source changed, product NOT rebuilt -> want FAIL BUILD_PRODUCT_UNCHANGED"
# C3: change a source again, then a real (product-updating) build.
sleep 1; printf 'int main(){return 2;}\n' > "$F/repo/src/kernel.c"
run rebuild "C3 source changed, product rebuilt -> want ok=yes (mtime advanced + sha changed)"
# C4: no source change; only a library gets a new timestamp and the product stays old.
sleep 1
run noop "C4 no source change, only a lib touched -> want FAIL BUILD_PRODUCT_STALE (the tell)"
# C5: prove the wrapper invokes the target EXPLICITLY (fake cmake on PATH, no --build-cmd).
chmod +x "$F/cmake"
echo "--- C5 explicit-target proof (fake cmake on PATH, default build-cmd) ---"
PATH="/home/bob/step4/selftest_fixB:$PATH" bash build_product.sh \
    --repo "$F/repo" --build-dir "$F/build" --target strata_vulkan \
    --stamp "$F/BUILD.stamp2" --sources "$F/repo/src"
echo "<<exit=$?>>"
echo
echo "################ END FIX B SELF-TEST ################"
