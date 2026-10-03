# Vulkan port audit — code guideline, Arc/Vulkan compatibility, performance

**Scope:** `ports/vulkan/` (the Arc port's stage-1 artifacts) — `harness/vk_compute.{hpp,cpp}`,
`harness/vk_gate.cpp`, `gates/run_gate.sh`, `shaders/*.comp`, and the three documents.
**Rubric:** `~/code-guidline-prompt-for-llm.txt` (14 lines, read in full), plus the Arc/Vulkan and
performance rules carried by the `intel-arc-llm-inference`, `vulkan-compute-shader-porting` and
`gpu-backend-integration` skills.
**Tree audited:** `~/strata-vulkan-wt` (branch `vulkan-arc-port`). Pre-fix line numbers are quoted from
revision `1bcdc91`; the fixes are in the working tree on top of `bf4b0bd`.
**Auditor date:** 2026-10-03. **Nothing in this audit has been run on Intel hardware** — see §6.

Everything marked FIXED is applied and re-verified: `bash ports/vulkan/gates/run_gate.sh` →
**14 passed, 0 failed, 0 skipped**, plus the shader checks below.

---

## 1. Verdict

| Axis (guideline) | Result |
|---|---|
| correct / complete | 8 of 44 CUDA kernels ported and gated; the rest are listed with a wave plan. No silent gaps — absent kernels SKIP loudly and the gate FAILS on a skip. |
| security | no untrusted input, no world-writable writes, no `eval`. Nothing found. |
| performance | the gate's choices are correct for a gate and wrong for throughput, now stated in the code and in §5. |
| user-friendliness | one command runs everything; it now refuses to report success on an empty or skipped run. |
| maintainability | 3 real defects fixed (duplicated reference, parallel containers, hardcoded path). |
| consistency | 1 defect fixed (one workgroup-size constant, cross-checked against every shader). |
| dead code / unused imports | 4 items removed; 3 deliberately kept, with reasons (§4). |
| DRY / YAGNI / bloat | the largest finding of this audit (§2.1) and §3.2. |
| Arc/Vulkan compatibility | 2 real defects fixed, 1 documented, 3 recorded as blocked or unfixable-in-code (§3). |

---

## 2. Code guideline findings

### 2.1 Duplicated reference implementation — DRY, and a drift risk (HIGH, FIXED)
`vk_gate.cpp:35` and `:45` (rev `1bcdc91`) hand-copied `bf16_from_f32` and `f16_from_f32` out of the engine's
headers. The tree ships both as **host-includable** (`STRATA_BF16_HD`/`STRATA_HD` expand to nothing outside
CUDA/HIP), and `f16_bits.hpp`'s own comment states the convention: *"the copy lives here and the kernel and
its TEST both include it"*.

A transcription of the reference is worse than duplication: if the engine's rounding changes, the gate keeps
testing the old rule and reports PASS against a reference that no longer exists. **Fixed** — `vk_gate.cpp:41-42`
now includes `strata/kernels/bf16_bits.hpp` and `f16_bits.hpp` and uses them directly, and `run_gate.sh` adds
`-I<TREE>/include`. The port's gate now tests the shaders against the engine's *actual* converter.

### 2.2 Dead code (HIGH, FIXED)
`Ctx::die()`, `err_` and `last_error()` (`vk_compute.hpp:71,98,100`; `vk_compute.cpp:436-437`) were written and
never called — no caller, no reader. **Removed.** Also removed: the `fp64` tolerance branch in `case_silu`
(`vk_gate.cpp:376`) which selected a tolerance for a shader that cannot be built, and the `shader_int64`
plumbing (see §3.2).

### 2.3 Two containers that must stay the same length (MEDIUM, FIXED)
`pkeys_` and `pvals_` (`vk_compute.hpp:96-97`) were parallel vectors holding key and value; a push to one and
not the other is a silent mismatch. **Fixed** — one `std::vector<Pipe>` with the key fields inside the struct,
so the two cannot drift.

### 2.4 Hardcoded absolute path as a default (MEDIUM, FIXED)
`vk_gate.cpp:441`: `std::string dir = "/home/bob/strata-vulkan-port/shaders";` baked one person's scratch
directory into the binary. **Fixed** — `STRATA_VK_SPV_DIR` overrides, the argument overrides that, the default
is the relative `shaders`, and the runner always passes the path explicitly.

### 2.5 The same constant in nine places (MEDIUM, FIXED)
`256` appeared as `LOCAL = 256` in eight cases and as the literal `'LocalSize 256 1 1'` in `run_gate.sh:27`. The
host sizes **every** dispatch from it and the shaders declare it, so a silent disagreement is a wrong grid
rather than a compile error. **Fixed** — `kLocalSize` is named once (`vk_gate.cpp:77`), `run_gate.sh` *reads it
out of the source* and asserts it equals every shader's compiled `OpExecutionMode LocalSize`, per shader.

### 2.6 Warnings could not fail the build (LOW, FIXED)
`-Wall -Wextra` without `-Werror` (`run_gate.sh:41`). **Fixed**: `-Werror`. The tree still builds clean.

### 2.7 Not changed, and why
`vk_gate.cpp` is ~480 lines in one file: cases, references and the driver are separated by section comments and
the file is the *gate*, not a library — splitting it now would be structure without a reader. Revisit when
kernel waves 4-6 land.

---

## 3. Arc / Vulkan compatibility findings

### 3.1 `rms_norm` coupled the host's dispatch to the driver's subgroup width (HIGH, FIXED)
The first version was one **subgroup** per row, which forced the host to compute
`rows-per-workgroup = local_size_x / subgroupSize`. `subgroupSize` from `VkPhysicalDeviceSubgroupProperties` is
a device-wide default; **where `VK_EXT_subgroup_size_control` is exposed, the driver may compile a given kernel
at a different width** (Intel's compiler picks a SIMD width per kernel — 8, 16 or 32). The host would then
dispatch too few workgroups and the tail rows would go unprocessed: no error, no crash, wrong numbers. This is
the Intel-specific form of the "SIMD width is the silent killer" trap, and it is invisible on RADV.

**Fixed** — rewritten as one **workgroup** per row (`shaders/rms_norm.comp`): dispatch is exactly `rows`
workgroups, the width appears only inside the kernel (`gl_NumSubgroups`, `gl_SubgroupSize` — compile-time
constants of the pipeline that actually runs), and the guard is now workgroup-uniform so the barriers are legal.
The host's subgroup arithmetic is gone.

**AND THE FIRST VERSION OF THAT FIX WAS ITSELF WRONG — see §3.1b. The correction is kept here rather than
quietly edited out, because the way it was found is the finding.**

### 3.1b The rewrite's combine stage assumed `gl_NumSubgroups <= gl_SubgroupSize` (HIGH, FIXED, found on a second implementation)

The new two-stage reduction combined **one entry per subgroup** by having the first subgroup read
`partial[gl_SubgroupInvocationID]` for `gl_SubgroupInvocationID < gl_NumSubgroups`. That is correct only while
the number of subgroups does not exceed the subgroup's lane count:

| subgroup size | subgroups (local_size_x = 256) | entries the stage could read | |
|---|---|---|---|
| 64 (RADV) | 4 | 4 | correct |
| 32 | 8 | 8 | correct |
| 16 | 16 | 16 | correct |
| **8 (llvmpipe, and a width Intel's compiler may pick)** | **32** | **8** | **drops 24 of 32 sums** |

Measured on llvmpipe: **worst relative error 1.13 on all four shapes, against 2.35e-07 on RADV.** The arithmetic
matches exactly — three quarters of the sum missing, so `rsqrt(mean)` is 2x too large and every element comes
out ~2x off. It is the same defect class as a hardcoded 32-lane assumption, one level up, and it was **invisible
on the only implementation the port had ever been run on**.

**Fixed** — the stage now accumulates with a stride (`for (i = lane; i < gl_NumSubgroups; i += gl_SubgroupSize)`),
which is correct at any ratio. Re-verified: **14/14 on RADV (subgroup 64) and 14/14 on llvmpipe (subgroup 8)**.

**Root cause of the near-miss: the port had only ever been tested on one Vulkan implementation.** `run_gate.sh`
now has a cross-implementation arm that runs the gate on **every ICD that reports a device** and fails if any of
them fails (absent hardware is reported as absent, which is honest; a present implementation is never skipped).
On this box that means RADV and llvmpipe; on the target box it will mean the Intel ICD the moment the card is
present — which is exactly the implementation whose per-kernel SIMD width motivated §3.1.

### 3.1c The display contract: a missing assignment, an unenforced reserve, and no way to test either (HIGH, FIXED)

Added on request ("the Arc card stays the video card; leave the desktop enough to composite"), and it turned up
one real bug immediately. `query_budget()` **never assigned `heap_total`**, so the 25%-of-card cap had no heap
to take a fraction of and the clamp could never fire - the policy read correctly and could not work. Found by
the new live case printing its own inputs, not by review.

The contract now lives in code rather than in a comment: the driver's `VK_EXT_memory_budget` figures (heap usage,
not this process's ledger), a 1024 MiB desktop reserve with a 256 MiB floor and a 25% cap, and an **enforced**
check that refuses an allocation crossing it — with a fork-free child process proving the refusal path exits
non-zero and says why. It composes with the engine's existing `--vram-reserve-mib` (700 MiB) instead of
overlapping it. Rationale, the AMD/HIP incident it prevents, the tunables and the non-memory display-safety
rules (no device spin-wait, bounded submissions, smoke-test first) are in PORT-PLAN.md §4b.

### 3.2 An unrequested feature could refuse device creation (HIGH, FIXED)
`vk_compute.cpp:186` (rev `1bcdc91`) set `VkPhysicalDeviceShaderFloat16Int8Features::shaderFloat16 = VK_TRUE`
unconditionally while **no ported kernel stores fp16** — on any device that does not report it,
`vkCreateDevice` fails outright and the portable-looking port would not start. `shaderInt64` was requested for
the same no-reason. **Fixed** — only `storageBuffer16BitAccess` and `shaderInt16` are requested, each gated on
the feature the device reports. This is the YAGNI finding and the compatibility finding at once.

### 3.3 The DEVICE_LOCAL heap is not a model budget on Arc (MEDIUM, DOCUMENTED)
`DeviceInfo::heap_device_local_bytes` prints 24.0 GiB here. On Intel Arc that heap is the dedicated VRAM or
BAR window (often 256 MB–512 MB), **not** the shared system memory a model is actually allocated from — Arc is
unified-memory in exactly the sense that breaks this number. Anyone sizing a model from it would refuse models
that fit. **Fixed in the code's documentation** and the gate now prints an explicit caution when the local heap
is under 4 GiB. The real accounting must come from `VK_EXT_memory_budget`
(`heap_budget - heap_usage`, DEVICE_LOCAL heaps) — stage 4, already in the plan.

### 3.4 The double-precision `silu` has no GLSL form on this toolchain (MEDIUM, RECORDED, BLOCKED)
The CUDA kernel computes in double; `exp(double)` does not reach SPIR-V on glslang 14.0 (shaderc) or
`glslangValidator` 15.1 with either fp64 extension. The port ships the f32 kernel and **measures** its gap
against the double reference (7.58e-07) rather than claiming exactness. Note for Arc specifically: fp64 runs at
a small fraction of fp32 on Arc, so restoring the double version in a per-element hot path would be a
performance regression dressed as a fidelity fix. `shaders/blocked/README.md` records all of it.

### 3.5 Battlemage stability is a driver bug this port cannot test around (HIGH, UNFIXABLE IN CODE)
The xe KMD wedges under sustained compute load through Level-Zero, OpenCL **and Vulkan** (open upstream bug
`intel/compute-runtime#948`; present on every kernel that supports BMG). Alchemist has none of those reports.
This is in the plan as the *first* thing to check on real hardware, before any scheduling work. **UNVERIFIED
here** — no Arc card is attached to this machine.

### 3.6 Xe2 matrix units are out of reach on this toolchain (MEDIUM, RECORDED)
Cooperative matrix needs glslang > 15.1; this host has 14.0/15.1. Correct on Arc is reachable today; *fast* on
Xe2 needs a toolchain bump (glslang ≥ 16, or naga/rust-gpu). Recorded in the plan, not worked around.

---

## 4. Deliberately NOT removed (with reasons)

| Item | Why it stays |
|---|---|
| `shaders/f32_to_f16_trunc.comp` | the **negative control**: a deliberately wrong converter that must FAIL the same bit-exact comparison the real one passes (it does — 484/1024 differ). Its header says it must never ship. Removing it removes the proof that the bit-exact gate can fail. |
| `shaders/blocked/silu_fp64.comp` | blocked by the toolchain, not by the port. Kept **outside** the compiled set with a README so the reason outlives the session and the work is not silently dropped. |
| `shaders/exp_probe.comp` | the diagnostic that identified the softplus defect (driver `exp` ≈ 9.05e-07, `log` near 1 ≈ 1.5e-07 absolute). It is what converts "the gate failed" into a named cause. |
| `shaders/copy.comp` | the harness self-test. A verifier that lies is worse than none. |

---

## 5. Performance findings

| # | Finding | Status |
|---|---|---|
| P1 | Host-visible coherent memory, a fresh command buffer per dispatch and a fence wait per call. Correct for a gate (it removes every way the *gate* could be wrong), wrong for throughput. | Documented in `vk_compute.hpp` and the plan; the engine's backend needs device-local memory + staging (stage 4). |
| P2 | Per-dispatch overhead is **not** the prize: the engine already replays a captured CUDA graph (~2 driver calls per token). Stage 3 must record one command buffer per decode step and re-submit it — never re-record per token. | Recorded for stage 3. |
| P3 | The pipeline cache is keyed on (path, bindings, push size). When specialization constants arrive, the workgroup size **must** join that key — pipelines are baked per size and a stale entry silently dispatches the wrong one. | Not applicable in this wave (no specialization constants); recorded. |
| P4 | softplus is now an 18-term series (18 dependent FMAs per element). Negligible next to the surrounding GEMV work; measured driver `exp` accuracy is 9.05e-07 relative, which is what the tolerance is built from. | Accepted, measured. |

---

## 6. Verification of the audit's own checks, and what remains unverified

**Every check added by this audit was proven able to fail** — a check nobody has seen fail is decoration:

| Check | How it was proven |
|---|---|
| operation census (embellishment) | injected a subgroup op into `scale.comp` with an **asserted** anchor → `FAIL scale (unexpected ops: 1 OpGroupNonUniformFAdd)`, exit 1 |
| workgroup-size agreement | changed `add.comp` to `local_size_x = 128` → `FAIL ... does not match the host's kLocalSize=256`, exit 1 |
| empty shader set | ran the gate in a copy with no `.comp` files → `refusing to report success`, exit 1 |
| a missing kernel is not a pass | removed `add.comp` → `SKIP add.spv`, then `1 case(s) SKIPPED - a skipped case is not a passing one`, exit 1 |
| the bit-exact numeric gate | the negative control shader disagrees on 484 of 1024 values |
| cross-implementation (new) | it is what found §3.1b: the same shaders, unchanged, produce worst-case relative error 1.13 on llvmpipe and 2.4e-07 on RADV |
| over-budget refusal (new) | a child process with an 8 MiB card exits 3 and names the numbers; exit 5 would mean the allocation was allowed |
| budget arithmetic (new) | the live case re-queries the driver itself and compares, and allocates 8 MiB to prove the figure moves |
| reserve policy (new) | 5 cases plus two discriminators (the clamp must actually fire, the floor must actually raise) |

**Six of this audit's own artifacts were defective, and every one was caught by testing rather than by
reading** — worth recording, because they are the exact failure modes the guideline's audit rules warn about
(two injections that proved nothing, a fix that broke an untested width, a test measuring integer truncation, a
path that resolved in another process, and a requirement that over-specified a driver):

1. The first injection **silently did not apply** (`if (i >= pc.n) return;` is not the text in
   `scale.comp`), so the census correctly reported a clean shader and the "test" proved nothing. Fixed by
   asserting the anchor matched and re-reading the file — a no-op edit that reports success is worse than one
   that fails.
2. The second attempt injected code the **compiler folds away** (`subgroupAdd(1.0) == 0.0` is uniform and
   constant), so again nothing was proven. The census can only see operations that survive optimisation —
   which is the right property, but it means an injection must change the result to test the check.
3. **The reserve-policy case compared exact bytes against an integer-truncated cap.** 25% of 24 GiB, computed
   as `heap/100*25`, lands 19 bytes off an exact quarter, so a correct policy failed a test that was measuring
   integer arithmetic rather than the rule. Fixed with a 1 MiB tolerance **plus** explicit discriminators (the
   clamp must actually fire, the floor must actually raise) so the case still cannot pass vacuously.
4. **The refusal case's child ran `dash` instead of the gate.** `popen("/proc/self/exe --expect-refusal")`
   resolves that path *in the shell popen spawns*, so the shell exec'd itself with a flag it does not know and
   exited 2 - reported as a refusal-path failure that did not exist. Resolve the path in the parent
   (`readlink`) before building the command.
5. **The budget-tracking check was over-specified twice**: exact bytes first (7.04 MiB measured for an 8 MiB
   request, which is correct coarse accounting), then a fixed bar that a software implementation failed
   non-deterministically (0 bytes on one run, 7.38e6 on the next). It now distinguishes a discrete local heap
   (hard requirement - where "free" staying at the heap size is the dangerous shape) from a device whose local
   heap is ~all of system RAM (reported with its reason, because the OS protects the desktop there).
6. **The §3.1 fix introduced §3.1b**, and no amount of self-testing on this box could have found it: the
   self-tests all ran on the same implementation. A second implementation with a different subgroup width found
   it in one run. The general rule this audit arrives at: *for a kernel whose correctness depends on a runtime
   width — subgroup, SIMD, warp — one implementation is not a test, it is an anecdote.*

Along the way a script's exit status was read from a pipeline (`... | tail -2; echo $?`) and reported `0` for a
command that had correctly exited 1 — an exit code is not a result.

**UNVERIFIED (stated as such, not as findings):** the Intel ICD itself (present but deviceless here, so the
cross-implementation arm reports it as no-device and it is genuinely untested); anything else on Intel silicon — the subgroup widths ANV actually
compiles, whether the workgroup-per-row reduction is fast enough there, `VK_EXT_memory_budget` on the Arc
driver, 8-bit storage for `embedding_gather`, cooperative matrix, and the Battlemage stability question in §3.5.
The harness cannot answer any of them without the card.
