#!/usr/bin/env bash
# /home/bob/step4/selftest_gate_identity.sh -- self-test for the GATE-LOG IDENTITY stamp (gap 1).
# The gate is the project's ACCEPTANCE TEST and it was the one artifact with no identity recorded.
# Fixture: a throwaway git repo + a stub gate runner that "builds" a gate binary from $GATE_STUB_MARK and prints
# a gate body.  gate_run.sh runs twice (clean tree, then dirty) and the stamp is shown to record the gate
# binary's sha256 + the commit + the porcelain state; then arm_gate.py --check-gate-logs is put through the
# cases that matter, each shown FAILING when it should:
#   T1 right gate-bin sha            -> ADMITTED (exit 0)
#   T2 wrong gate-bin sha            -> GATE_REFUSED gate-bin-hash-mismatch (exit 2)
#   T3 right sha + right commit      -> ADMITTED (exit 0)
#   T4 right sha + wrong commit      -> GATE_REFUSED commit-mismatch (exit 2)
#   T5 a legacy gate log (no stamp)  -> GATE_REFUSED legacy-unstamped (exit 2)
#   T6 the same legacy log + --allow-unstamped-legacy -> ADMITTED but LABELLED GATE_LEGACY_UNSTAMPED
set -u
cd /home/bob/step4
F=selftest_gateid; rm -rf "$F"; mkdir -p "$F/repo/ports/vulkan/gates" "$F/repo/ports/vulkan/harness/build" "$F/logs"
( cd "$F/repo" && git init -q && git config user.email t@t && git config user.name t \
  && echo hello > tracked.txt && git add tracked.txt && git commit -qm init ) >/dev/null
cat > "$F/repo/ports/vulkan/gates/run_gate.sh" <<'EOF'
#!/usr/bin/env bash
# self-test stub gate runner: "build" the gate binary from $GATE_STUB_MARK, then print a gate body.
mkdir -p "$(dirname "$GATEBIN")"
printf 'gate-binary %s\n' "${GATE_STUB_MARK:-v1}" > "$GATEBIN"
echo "== compiling shaders from source ($PWD)"
echo "  OK   add   OpExecutionMode %main LocalSize 256 1 1 | census: none"
echo "== 965 passed, 0 failed, 0 skipped"
EOF
# keep the built gate binary out of the fixture's porcelain so "clean" really is clean
printf '*\n' > "$F/repo/ports/vulkan/harness/build/.gitignore"
( cd "$F/repo" && git add -A && git commit -qm ports ) >/dev/null
GITC=$(git -C "$F/repo" rev-parse HEAD)
FABS="$(pwd)/$F"
GB="$FABS/repo/ports/vulkan/harness/build/vk_gate"
RUN(){ GATE_REPO="$FABS/repo" GATE_LOGD="$FABS/logs" \
       GATE_RUNNER="$FABS/repo/ports/vulkan/gates/run_gate.sh" GATE_BIN="$GB" \
       GATE_LOCK="$FABS/lock" GATE_STUB_MARK="$1" bash gate_run.sh "$2"; }
WRONG=$(printf 'not-the-intended-gate-binary' | sha256sum | cut -d' ' -f1)

echo "################ GATE-LOG IDENTITY SELF-TEST ################"
echo "fixture commit = $GITC"
echo
echo "=== R1: gate_run.sh on a CLEAN tree (stub builds gate binary 'aaaa') ==="
RUN aaaa alpha; echo "R1 exit=$?"
echo "--- $F/logs/gate_alpha.log identity header ---"
grep -nE '^== gate ' "$F/logs/gate_alpha.log" | head -12
SHA_ALPHA=$(sha256sum "$GB" | cut -d' ' -f1)
echo "gate binary sha AFTER the run = $SHA_ALPHA"
echo
echo "=== R2: dirty the tree, re-run (stub builds gate binary 'bbbb') -- the stamp must MOVE ==="
echo scratch >> "$F/repo/tracked.txt"
RUN bbbb beta; echo "R2 exit=$?"
grep -nE '^== gate (run|commit|status sha256|status lines|bin sha256)' "$F/logs/gate_beta.log"
echo "  recorded porcelain:"; sed -n '/^== gate status --porcelain:/,/^== gate body follows/p' "$F/logs/gate_beta.log" | sed -n '2,3p'
SHA_BETA=$(sha256sum "$GB" | cut -d' ' -f1)
echo "beta gate binary sha = $SHA_BETA   (alpha was $SHA_ALPHA)"
bash assert_diff.sh "alpha-vs-beta gate binary" "$SHA_ALPHA" "$SHA_BETA" || true
echo
echo "=== T1: --check-gate-logs with the RIGHT gate-bin sha -> MUST admit (exit 0) ==="
python3 arm_gate.py --check-gate-logs --want-gate-bin-sha "$SHA_BETA" "$F/logs/gate_beta.log"
echo "T1 exit=$? (want 0)"
echo
echo "=== T2: --check-gate-logs with a WRONG gate-bin sha -> MUST refuse (exit 2) ==="
python3 arm_gate.py --check-gate-logs --want-gate-bin-sha "$WRONG" "$F/logs/gate_beta.log"
echo "T2 exit=$? (want 2)"
echo
echo "=== T3: right sha AND right commit -> MUST admit (exit 0) ==="
python3 arm_gate.py --check-gate-logs --want-gate-bin-sha "$SHA_BETA" --want-commit "$GITC" "$F/logs/gate_beta.log"
echo "T3 exit=$? (want 0)"
echo
echo "=== T4: right sha but WRONG commit -> MUST refuse (exit 2) ==="
python3 arm_gate.py --check-gate-logs --want-gate-bin-sha "$SHA_BETA" \
        --want-commit 0000000000000000000000000000000000000000 "$F/logs/gate_beta.log"
echo "T4 exit=$? (want 2)"
echo
echo "=== T5: a LEGACY gate log (the real logs/gate_clean.log, written before the stamp) -> MUST refuse (exit 2) ==="
python3 arm_gate.py --check-gate-logs --want-gate-bin-sha "$SHA_BETA" logs/gate_clean.log
echo "T5 exit=$? (want 2)"
echo
echo "=== T6: the same legacy log + --allow-unstamped-legacy -> ADMITTED but LABELLED (exit 0) ==="
python3 arm_gate.py --check-gate-logs --allow-unstamped-legacy logs/gate_clean.log
echo "T6 exit=$? (want 0 -- legacy labelled, NOT forged)"
echo "################ END GATE-LOG IDENTITY SELF-TEST ################"
