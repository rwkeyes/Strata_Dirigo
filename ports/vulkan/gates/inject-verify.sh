#!/usr/bin/env bash
# ports/vulkan/gates/inject-verify.sh - FALSIFY a gate case by injecting the WRONG rule into its shader.
#
# A case that cannot fail is not an oracle.  This script applies ONE named injection (the wrong rule), recompiles
# the shader from source, runs the gate on the default ICD, and REQUIRES the named case to FAIL - then restores
# the tree.  It never runs a stale binary: a missing anchor prints ANCHOR MISSED, a shader that will not compile
# prints DID NOT COMPILE (and the whole tree is restored), and restoring is on every exit path.
#
#   inject-verify.sh q4-gather-offset   kv_q4_gather.comp   drop the "-8" code offset
#                                       -> must FAIL  "kv_q4 round trip: append (rotated) -> gather ..."
#   inject-verify.sh q8-round-half-up   quantize_q8_0.comp  replace the ties-to-even rounding with half-toward-+inf
#                                       -> must FAIL  "quantize_q8_0 (ggml bytes)"
#   inject-verify.sh cvec-apply-drop-scale  cvec_apply.comp     drop the per-layer reflect factor s
#                                       -> must FAIL  "cvec_apply: project removes s(h.v)v ..."
#   inject-verify.sh gather-rows-identity   gather_rows.comp    ignore ids[r]: gather the row at the POSITION
#                                       -> must FAIL  "gather_rows: 16-byte-aligned rows ..."
#   inject-verify.sh scatter-rows-identity  scatter_rows_f32.comp  write dst row r instead of rows[r]
#                                       -> must FAIL  "scatter_rows_f32: permutation ..."
#
# Usage: inject-verify.sh <name> [icd.json]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"       # ports/vulkan
SH="$ROOT/shaders"
BUILD="$ROOT/harness/build"
GATE="$BUILD/vk_gate"
[ -x "$GATE" ] || { echo "no gate binary at $GATE - run run_gate.sh once first"; exit 4; }

name="${1:-}"; icd="${2:-}"
case "$name" in
  q4-gather-offset)
    file="$SH/kv_q4_gather.comp"; spv="kv_q4_gather"
    old=$'KS_.y[dst + j] = uint16_t(f16_from_f32(float(kc - 8) * kd));'
    new=$'KS_.y[dst + j] = uint16_t(f16_from_f32(float(kc) * kd));   // INJECTION: the -8 offset dropped'
    want="FAIL  kv_q4 round trip" ;;
  q8-round-half-up)
    file="$SH/quantize_q8_0.comp"; spv="quantize_q8_0"
    old=$'        double q = fl + ((u - fl > 0.5) ? 1.0 : ((u - fl < 0.5) ? 0.0 : odd));'
    new=$'        double q = fl + ((u - fl >= 0.5) ? 1.0 : 0.0);   // INJECTION: round half toward +inf'
    want="FAIL  quantize_q8_0 (ggml bytes)" ;;
  cvec-apply-drop-scale)
    file="$SH/cvec_apply.comp"; spv="cvec_apply"
    old=$'    if (steer && pc.mode == 0) dot = wg_sum(dot) * SL.v[pc.layer];'
    new=$'    if (steer && pc.mode == 0) dot = wg_sum(dot);   // INJECTION: the per-layer scale s dropped'
    want="FAIL  cvec_apply: project removes" ;;
  gather-rows-identity)
    file="$SH/gather_rows.comp"; spv="gather_rows"
    old=$'    DST.b[i] = SRC.b[uint(IDS.v[r]) * pc.row_bytes + o];'
    new=$'    DST.b[i] = SRC.b[r * pc.row_bytes + o];   // INJECTION: ids[r] ignored - the row at the position'
    want="FAIL  gather_rows: 16-byte-aligned rows" ;;
  scatter-rows-identity)
    file="$SH/scatter_rows_f32.comp"; spv="scatter_rows_f32"
    old=$'    const uint dbase = uint(ROWS.v[r]) * pc.width;'
    new=$'    const uint dbase = r * pc.width;   // INJECTION: the destination row indirection dropped'
    want="FAIL  scatter_rows_f32: permutation" ;;
  *) echo "unknown injection '$name'"; exit 2 ;;
esac

# 1. the anchor must be there, or we would be "injecting" into a file that no longer says what we think.
if ! grep -qF -- "$old" "$file"; then
  echo "ANCHOR MISSED: $name - '$file' does not contain the anchored text; nothing changed"
  exit 2
fi

bak="$(mktemp)"
cp "$file" "$bak"
restore() {
  cp "$bak" "$file"; rm -f "$bak"
  glslc --target-env=vulkan1.3 -fshader-stage=compute "$file" -o "$SH/$spv.spv" 2>/dev/null
  # if the changed file is a shared INCLUDE, every shader that includes it must be recompiled too
  if [ "${file#"$SH/common/"}" != "$file" ]; then
    for c in "$SH"/*.comp; do
      grep -qF "$(basename "$file")" "$c" || continue
      glslc --target-env=vulkan1.3 -fshader-stage=compute "$c" -o "${c%.comp}.spv" 2>/dev/null
    done
  fi
}
trap restore EXIT

# 2. apply it, exactly once.
python3 - "$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
n = s.count(old)
if n != 1:
    sys.exit(f"ANCHOR COUNT {n} (expected 1)")
open(p, "w").write(s.replace(old, new))
PY
[ $? -eq 0 ] || { echo "ANCHOR MISSED: $name - anchor not unique"; exit 2; }

# 3. it must COMPILE - an injection that does not build is a stale binary wearing a new timestamp.
if ! glslc --target-env=vulkan1.3 -fshader-stage=compute "$file" -o "$SH/$spv.spv" 2>"$BUILD/$spv.glslerr"; then
  echo "DID NOT COMPILE: $name"; sed -n '1,12p' "$BUILD/$spv.glslerr"; exit 3
fi
if [ "${file#"$SH/common/"}" != "$file" ]; then
  for c in "$SH"/*.comp; do
    grep -qF "$(basename "$file")" "$c" || continue
    glslc --target-env=vulkan1.3 -fshader-stage=compute "$c" -o "${c%.comp}.spv" 2>/dev/null || { echo "DID NOT COMPILE (dependent): $c"; exit 3; }
  done
fi

# 4. run the gate and require the named case to FAIL.
if [ -n "$icd" ]; then out="$(VK_ICD_FILENAMES="$icd" "$GATE" --spv-dir "$SH" 2>&1)"; else out="$("$GATE" --spv-dir "$SH" 2>&1)"; fi
line="$(grep -F -- "$want" <<<"$out" | head -1)"
if [ -n "$line" ]; then
  echo "FALSIFIED ($name): $line"
  exit 0
else
  echo "NOT FALSIFIED ($name): expected a line starting '$want' and did not get one"
  grep -E '^(PASS|FAIL|SKIP) ' <<<"$out" | grep -iF "${want#FAIL  }" | head -3
  exit 1
fi
