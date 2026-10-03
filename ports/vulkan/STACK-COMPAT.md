# Linux stack compatibility, performance and recommendations — Arc/Vulkan port

*Renamed from `KERNEL-COMPAT.md`: the kernel is one component among several this port depends on, and the
others are covered in section 3.5.*

> **Pending hardware change:** this host's 7900 XTX is being replaced by an Intel Arc Pro B70. What that
> invalidates, and what to do about it, is in [`SWAP-PREFLIGHT.md`](SWAP-PREFLIGHT.md) - including one item that
> stops the local model service outright.

**For:** whoever ships or runs the Vulkan backend on Intel Arc.
**Port:** `ports/vulkan/` on branch `vulkan-arc-port`.
**Revision of this document:** 2026-10-03.
**Everything time-sensitive here is checkable in one command** — see §1.3. Nothing in this document should be
believed on its own authority; the version numbers move.

---

## 1. Summary

### 1.1 The two answers

| Question | Answer |
|---|---|
| **Best performance** | **7.2 stable** is the best you can install today. **7.3** (expected stable 2026-10-18) is the better bet on paper for this workload. The cycle after 7.3 carries the first Battlemage-flagged performance work, and it is **not installable** — see §4.3. |
| **Recommended minimum** | **6.12** on an Arc B580 / BMG-G21, **6.14** on an Arc Pro B70 / BMG-G31 — to run at all. **7.2** for a box where the Arc drives the display. There is no kernel yet that solves the stability problem in §7. |

### 1.2 What it rests on

- This port is a **userspace Vulkan client**: no device node, no ioctl, no libdrm, no version checks. Per-kernel
  *code* work is **zero lines** for every transition covered here. What varies is behaviour, features and
  stability — so the port *detects* the host, *reports* the rules, and *refuses* only where the combination is
  known to be impossible (§6).
- Steady-state throughput is set by **Mesa and the card's clocks**, not the kernel. The kernel decides what
  happens under **memory pressure**, how **buffer migration** is scheduled, and whether the card **resets** —
  and a reset costs the whole request, so stability dominates performance for unattended inference (§4.1).
- **Mesa is a separate axis from the kernel**, and for one feature it is the binding one: Intel needs
  **Mesa ≥ 26.2** for a driver-backed free-memory figure (`VK_EXT_memory_budget`), which the display reserve
  depends on (§3.4).

### 1.3 Check the version facts yourself

```bash
curl -s https://www.kernel.org/releases.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('latest stable:', d['latest_stable']['version']); \
   [print(' ', r['version'], r['moniker'], r['released']['isodate'][:10]) for r in d['releases'][:6]]"
```

As of **2026-10-03** that prints: latest stable **7.2.9** (released that day), **mainline 7.3-rc5**, longterm
lines 6.18.55 / 6.12.112 / 6.6.158 / 6.1.189 — and **no 7.4 of any kind**. A patch labelled "for 7.4" sits in
drm-next/linux-next for a cycle whose merge window has not opened; the label is a target, not a kernel. An
earlier revision of this document got that wrong and is corrected in §9.

---

## 2. Scope: what "kernel compatibility" even means here

Verified by inspection of the port, not assumed:

- The only Vulkan API-version constant in the tree is `VK_API_VERSION_1_2` (instance); the only device extension
  enabled is `VK_EXT_memory_budget`. Everything else (`storageBuffer16BitAccess`, `shaderInt16`, subgroup ops,
  `gl_NumSubgroups`) is core 1.1+ or a SPIR-V capability.
- `grep -rn "/dev/dri|ioctl|drm_|DRM_IOCTL|/sys/class/drm" harness/ gates/` returns **nothing**.

So there is no kernel ABI to track, no out-of-tree module, and no `#if LINUX_VERSION_CODE`. Everything the
kernel does for this port arrives through **Mesa**, which is where the uAPIs are consumed. The port's entire
Linux-awareness is three reads: `uname()` for the release, `/sys/module/<drv>` for which DRM module owns the
card, and `VkPhysicalDeviceDriverProperties` for the userspace driver and its version.

---

## 3. Stack compatibility

### 3.0 The kernel

### 3.1 The ladder

| Kernel | Installable? | DRM / Xe changes that matter to this port |
|---|---|---|
| 6.8 | yes (longterm-ish) | **No Battlemage support at all.** Do not downgrade hoping for stability. |
| 6.12 | yes (longterm, 6.12.112) | **First mainline kernel with Xe2 enabled out of the box** — the floor for a B580 / BMG-G21 (`8086:e20b`, and the other `e209`/`e211`/`e212` BMG-G21 ids) |
| 6.14 | yes | Also the floor our notes record for the **Arc Pro B70 / BMG-G31** (`8086:e223`); start of the compute-load crash report range |
| 6.18 | yes (longterm, 6.18.55) | CCS engine resets reported on 6.18.2 and 6.18.32 (openSUSE, Arch; May 2026) |
| 7.0 | yes | Xe SR-IOV + multi-device SVM (Battlematrix), VFIO Xe driver — **neither used by this port** |
| 7.1 | yes (stable 2026-06-14) | Xe vRAM memory-pressure / out-of-memory behaviour; Nova Lake P / Xe3P_LPG initial; DRM RAS over Netlink |
| **7.2** | **yes — current stable (7.2.9)** | **Battlemage G21 cold-boot black-screen fix**; Panther Lake Xe3 performance; Crescent Island; cache-aware scheduling (a **CPU** scheduler feature) |
| 7.3 | **not yet** — rc5 (2026-09-27), stable expected 2026-10-18 | Nova Lake-S graphics on by default, Xe3P considered stable, **TTM eviction more aggressive**, rc5 DRM fixes |
| *post-7.3* | **no — cycle not open** | Xe **cold reset recovery**, vRAM health check / degraded-memory handling, **CPU binds + ULLS on the migration queue** |

### 3.2 Difficulty per transition, relative to the version before it

Ratings are for **writing compatibility in this port** — code, not verification.

| Transition | What changed that could matter here | Code required | Verification required | Difficulty |
|---|---|---|---|---|
| **7.0 → 7.1** | vRAM memory-pressure/OOM behaviour; Nova Lake/Xe3P enablement; DRM RAS | none | re-run the gate; re-check the desktop-reserve refusal path, because memory-pressure behaviour is what it interacts with | **Very low** |
| **7.1 → 7.2** | Battlemage G21 cold-boot black-screen fix; Xe3 performance; cache-aware scheduling | none | re-run the gate. The display fix is a reason to prefer 7.2 over 7.1 on Arc, not a code change | **Very low** |
| **7.2 → 7.3** | Nova Lake-S by default; Xe3P stable; **TTM eviction more aggressive** | none | re-run the gate **and** re-verify the desktop reserve: more aggressive TTM changes what happens when the card is full, which is the one behaviour the display contract depends on | **Very low** + one **required** re-check |

### 3.3 Two version-gated features, if we ever want them

Neither is compatibility we owe; both are optional and both need a fallback.

1. **Purgeable VRAM for the expert cache.** `DRM_IOCTL_XE_MADVISE` is present in this box's **7.0** headers
   (`.../uapi/drm/xe_drm.h`, incl. `DRM_XE_VM_BIND_FLAG_MADVISE_AUTORESET`), so the interface itself predates
   7.1; what 7.1 added is a refinement we have **not** pinned to a constant (UNVERIFIED). Adopting it would let
   the kernel reclaim engine VRAM rather than failing an allocation — better than a static reserve, but it needs
   Mesa-side support and a fallback for older kernels.
2. **A real free-memory figure on Intel.** `VK_EXT_memory_budget` needs **Mesa 26.2** (2026-08-05:
   `anv: add memory heap budget tracking across VkInstance`), the same release that added
   `intel: madvise purgeable VMAs in Xe KMD` and `anv: fixup compute queue detection`. Before it, the port's
   ledger rule applies (§6.2) — which is a refusal, not a guess.

### 3.4 Kernel-side configuration and hardware

| Component | Requirement | Why |
|---|---|---|
| Kernel module | `xe` for Battlemage (i915 for Alchemist), `CONFIG_DRM_XE=m` or `=y` | i915 cannot drive Battlemage |
| Display | `CONFIG_DRM_XE_DISPLAY=y` if the card drives a display | verified present on this box |
| `CONFIG_DRM_XE_GPUSVM` | needed **only** by the SYCL/USM path, **not** by this Vulkan port | present (`=y`) on this box |
| Userspace | **Mesa ≥ 26.2** on Intel for a trustworthy free figure | see §3.3 |
| Hardware | Resizable BAR as large as the card allows (16 GB observed on a B580, 32 GB on this box's card) | it sizes the local heap the driver reports |
| Firmware | current `bmg_guc_70.bin` | stale GuC firmware is implicated in several engine-reset reports |

### 3.5 The rest of the stack: drivers, libraries, tools

Detected by `harness/vk_stack.{hpp,cpp}` (filesystem only: no device node, no shelling out, nothing requiring
privileges) and printed on every run.

| Component | Detected | Why it matters here | Rule |
|---|---|---|---|
| **Vulkan loader** | `libvulkan.so.1` symlink target (here `1.3.275`) | the **only** stack component this port links. A loader older than an ICD's advertised `api_version` is the first suspect when a documented extension vanishes, though usually harmless since the loader is a thin trampoline | Note |
| **ICD files** | `/usr/share/vulkan/icd.d/*.json`, plus `VK_DRIVER_FILES` / `VK_ICD_FILENAMES`, which **replace** the default search rather than adding to it | `library_path` is normally a **bare soname** (`libvulkan_radeon.so`) that the loader resolves through the system library path. An ICD naming a library that cannot be resolved makes its GPU **silently vanish** from enumeration | Warn |
| **Mesa (ANV / RADV)** | `VkPhysicalDeviceDriverProperties.driverInfo` | 26.2 or newer for a driver-backed free-memory figure on Intel; the same release added purgeable VMAs and the compute-queue fix | Warn (see 6.1) |
| **libdrm** | `libdrm.so.2` symlink target (here `2.125.0`) | Mesa's dependency, not the port's; its version tracks the kernel interface Mesa speaks | reported, no rule |
| **GPU firmware** | `/lib/firmware/xe/bmg_guc_70.bin`, `xe/bmg_huc.bin`, `i915/bmg_dmc.bin` | a missing GuC blob is a driver that fails to **probe** (`firmware production part check failure`, `probe ... failed with -71`), not a slow driver, and stale firmware recurs in the reset reports. The **loaded** version exists only in `dmesg` (`GuC firmware: ... version N`), so this check reports presence and **refuses to invent a version** | Warn when an Intel GPU is present and a blob is missing |
| **Level-Zero / OpenCL** | `libze_intel_gpu.so.1` (1.17.39395 here), `libOpenCL.so.1` | they serve the **SYCL/OpenCL** path, **not** this Vulkan port | Info, labelled so nobody debugs the wrong stack |
| **Session type** | `XDG_SESSION_TYPE` (x11 here) | a Wayland compositor was measured holding **354 MB** on a two-display desktop; an X11 session holds a different amount. This is what the reserve is sized against | Info |
| **Build toolchain** | `glslc`, `spirv-val`, `g++` versions **and capability probes**, run by `gates/run_gate.sh` every time | **cooperative matrix is absent on this toolchain** (glslang 14.0 via shaderc, 15.1 standalone): the only Vulkan route to Xe2's matrix units, so the performance ceiling is a *tool* limit rather than a port defect. The `fp64` math builtins are absent too (the silu-fidelity item) | probes printed each run; a capability appearing changes what the port can promise |

Two lessons are baked into the code here rather than only written down. First, **the ICD check's first version was
wrong**: it tested `library_path` as a file path and reported all nine working ICDs on this box as broken, because
bare sonames are resolved by the loader rather than found beside the JSON. The resolver now searches the standard
library directories and `$LD_LIBRARY_PATH`, accepts `.N` suffixed variants, and the case carries **both** controls,
so a stale ICD must be flagged and a resolvable one must not. Second, **an ICD scan has to respect the replace
semantics of `VK_DRIVER_FILES`**, or it reports the system's ICDs as "the" ICDs whenever that variable is set,
which is exactly what made the first negative control inspect the wrong entry.

### 3.6 Generation support: Alchemist and Battlemage

The rules are per **generation**, not per vendor, because the support history differs enormously. Source:
Intel's own KMD support tables (`dgpu-docs.intel.com/overview/supported-hardware/i915-driver-gpus.html` and
`.../xe-driver-gpus.html`, read 2026-10-03), which separate *initial support* (available experimentally, may
require `force_probe=PCI_ID`) from *full support* (enabled by default, validated). The severity rule follows from
that: **refuse below initial support** (the kernel predates the device), **warn inside the initial band**,
**silent once full support is reached**.

| Generation | Driver | Cards | Initial | Full | What this port does |
|---|---|---|---|---|---|
| **Alchemist** / Xe-HPG | **i915** | A770, A750, A580, A380, A310, Pro A40/A50/A60 | 6.0 | 6.2 | fatal below 6.0; warn in 6.0–6.1; silent from 6.2 |
| Alchemist mobile | i915 | A770M, A730M, A570M, A550M, A530M, A370M, A350M, Pro A30M/A60M | 5.19 | 6.2 | as above, floor 5.19 |
| **Battlemage** / Xe2 | **xe** | B580, B570 | 6.11 | 6.12 | fatal below 6.11; warn in 6.11; CCS-reset warning from 6.12 to 7.0 |
| Battlemage Pro | xe | B50 (6.11/6.14), B60 (–/6.15), B65 and B70 (–/6.17) | per card | | floor from our field notes (6.14) where Intel lists no initial release |

Three consequences worth stating plainly.

1. **A working Arc A770 on kernel 6.6** — a common LTS — must not be refused. The first version of these rules
   applied the Battlemage line (6.12) to *every* Intel device and would have refused it. Fixed, and the case
   table now carries the row that would have caught it.
2. **The Xe2 CCS engine-reset warning is Battlemage's.** Alchemist has none of those reports, so the warning is
   raised only for Battlemage (or for an Alchemist card that happens to be on xe). Applying it to Alchemist was
   the second half of the same bug.
3. **Firmware is per generation**: `xe/bmg_guc_70.bin` for Battlemage, `i915/dg2_guc_70.bin` for Alchemist. A
   missing-blob verdict is raised only when a generation's whole set is absent, so a drifted filename cannot be
   reported as missing firmware.

#### Is the port compatible with Alchemist?

**The API and shader layer: yes**, and Alchemist is in one respect the *easier* target. It supports subgroup
sizes 8/16/32, so a dispatch that assumed a width breaks there; this port's one-workgroup-per-row dispatch with a
strided stage-2 combine is exactly the fix for that, and the width-8 llvmpipe arm of the gate is the closest
available proxy until real Alchemist silicon is attached. Nothing in the port uses an Xe2-only feature, and the
memory-budget rule (Mesa ≥ 26.2) applies to both generations.

Two caveats, both sourced:

- **Alchemist belongs on i915.** Intel's Xe support table lists no DG2/Alchemist part at all — it begins at Lunar
  Lake and Battlemage. The xe path on DG2 is experimental with open HuC issues, and compute-runtime reports zero
  OpenCL/Level-Zero platforms on DG2 under xe (`intel/compute-runtime#905`). If xe is driving an Alchemist card,
  the port says so.
- **Cooperative matrix is a performance *regression* there**, not a win: ANV has exposed
  `VK_KHR_cooperative_matrix` since Mesa 24.0, but llama.cpp deliberately gates it to Xe2 only for that reason.
  So the plain-FMA ceiling of §4.3b stands on Alchemist even with a matrix-capable toolchain.

**Unverified**: none of this has run on Alchemist silicon (this host carries a Radeon RX 7900 XTX and the Ryzen
iGPU - there is no Intel discrete card here, and no NVIDIA card either: an earlier revision of this document said
"a 7900 XTX and a K620", which `lspci` does not support). The floors come
from Intel's tables; the behaviour on the hardware is untested, and the port's rules say so rather than implying
otherwise.

---

## 4. Performance

### 4.1 What a kernel can and cannot buy here

Steady-state throughput is Mesa and clocks — the daemon does the arithmetic, and no kernel version changes that.
The kernel enters performance in three ways, all about **sustained** running rather than peak speed:

1. **Memory management under pressure** — what happens when the card is full while the desktop is live on it.
2. **Scheduling on the migration queue** — how buffer moves are handled.
3. **Stability** — an engine reset costs the entire request. A kernel that resets is slower than one that does
   not, whatever the clocks say, and for unattended inference this dominates the other two.

### 4.2 The ladder, by what each cycle changed for Battlemage

| Kernel | Effect on performance here |
|---|---|
| 7.0 | Baseline. Nothing in it is used by this port. |
| 7.1 | vRAM memory-pressure behaviour — the path a full card takes. **But the performance reports conflict**: Phoronix measured 7.1 *helping* the Arc B580, while a community report claims a **50–90% OpenGL regression starting with 7.1**. Both exist; neither is this port's workload. Measure on the card; trust neither. |
| 7.2 | Stable. Battlemage G21 **cold-boot black-screen fix** — a display fix is a performance fix when the card drives the desktop. Its headline *cache-aware scheduling* is a **CPU** feature for multi-die CPUs and buys nothing here. |
| 7.3 | **TTM actively evicts unprotected buffers so protected allocations get VRAM** instead of falling back to system memory. That is this port's shape exactly: one large protected (resident) allocation plus a live desktop. The most promising change for this workload in the whole list — but it is rc5. |
| post-7.3 | **CPU binds + ULLS on the migration queue**, described as a big improvement for Battlemage — the first Battlemage-flagged performance work. **Not installable.** |

### 4.3 The one setting that decides whether a long generation survives

`CONFIG_DRM_XE_PREEMPT_TIMEOUT` is **640 ms** on the kernel inspected (`_MIN=1`, `_MAX=10000000`). Every
individual submission must complete inside it. Overrunning it is **an engine reset, not a slowdown** — this is
the failure reported for tensor-parallel LLM inference on Battlemage, asked about on Intel's own forum in
exactly those terms. This is why the port's design rule is **bounded work per submission** (PORT-PLAN §4b), and
it is the constraint to check first if a long run dies mid-generation.

### 4.3b The matrix-unit path (`VK_KHR_cooperative_matrix`)

**Correction, same day, after building it:** this section previously said the toolchain could not emit cooperative
matrix and that Battlemage's XMX engines were therefore unreachable. **That was wrong, and the fault was in my
check, not in the toolchain.** The probe shader I wrote omitted `GL_KHR_memory_scope_semantics`, misspelled
`coopMatMulAdd` (it is not `coopmatMulAdd`), and used `gl_MatrixLayoutRowMajor` instead of
`gl_CooperativeMatrixLayoutRowMajor`. `glslc` — shaderc 2023.8, glslang 14, well below the version the skill
blamed — emits `OpCooperativeMatrixMulAddKHR` and `spirv-val` accepts it for Vulkan 1.3. The probe now compiles
the **shipping kernel** rather than a fixture, so it cannot drift from the capability it claims.

What is true, and matters more than the toolchain question:

- **The config is hardware, and the driver's list is the only authority.** Measured on RADV / RX 7900 XTX: 14
  supported configs, all `M16 N16 K16` with subgroup scope; the only floating-point ones are `f16×f16→f16` and
  `f16×f16→f32`. **There is no fp32-operand config at all**, so an fp32 cooperative-matrix GEMM compiles and still
  cannot run. Selection must come from `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` — never from "it
  compiled", which is the same trap in a new place.
- **It is not a universal win.** On Intel it is gated to Xe2 because it *regresses* on Alchemist, so the pipeline
  is chosen by device property, never by vendor.
- **A device without it is a loud SKIP**, neither a pass nor a failure (§6), and the suite fails closed on skips.

`shaders/gemm_coopmat.comp` is the first kernel on this path (`C = A·B`, fp16 operands with an fp32 accumulator),
verified against a host reference on three shapes — 64³, 32×16×48, 16×64×32 — with worst relative error 4.5e-4,
i.e. **fp16-operand precision, the storage format's own limit rather than the kernel's**. The reference is built
from the fp16-rounded operands for exactly that reason. The dispatch is one 16×16 tile per *subgroup*, with the
per-workgroup tile count derived inside the kernel from `gl_WorkGroupSize/gl_SubgroupSize`; the host dispatches an
upper bound and the kernel discards the surplus, because over-dispatch is free while under-dispatch silently drops
output — the `rms_norm` defect, avoided by construction here.

Two of its defects were found by the shape matrix rather than by inspection: the tile→(row,col) mapping was
**transposed** (invisible on a square tile grid, where transposing still visits every tile — the 64³ case passed
with it), and the new case **perturbed the memory-budget check** by allocating before it (RADV rounds allocations
up to 2 MiB, so a few buffers move `heapUsage` by ~9 MB). Both fixed; the case now runs after the device-state
checks, with a comment saying why.

**Honest state:** the matrix-unit path works end to end and is verified on AMD silicon. What remains for the B70
is Intel-specific — the same shader on Xe2, its subgroup width (16/32 rather than 64), and the dequant step the
real engine needs, since IQ1_M weights must be dequantized to fp16 before the matrix units can consume them.

### 4.4 Honest limit

**No benchmark of this port on Arc exists.** Everything above is release-note reasoning about the driver, not a
measured number for this workload, because there is no Arc card on the machine where the port was built. The
numbers that *are* measured here (26 gate checks, the Vulkan findings in `AUDIT.md`) are on RADV and llvmpipe.

---

## 5. Recommendations

### 5.1 What to install

| Scenario | Kernel | Why |
|---|---|---|
| **Just make it run** | 6.12 (B580) / 6.14 (Arc Pro B70) | the documented floors; older is a fatal refusal and 6.8 has no support at all |
| **A workstation where the Arc drives the display** | **7.2 stable** | the oldest stable kernel with both the vRAM memory-pressure work (7.1) and the Battlemage display fix, and outside the 6.14–6.19 crash range |
| **A server / unattended inference** | **7.2 stable today; 7.3 once stable** | stability is the dominant term (§4.1). Cold reset recovery is in the post-7.3 cycle and is the first thing likely to change this row |
| **Maximum throughput, unexplained** | 7.2; measure 7.3 and 7.1 on the actual card | the 7.1 reports conflict, and 7.3 changes eviction behaviour this workload is sensitive to |
| **Kernel + userspace pairing** | 7.2 with **Mesa ≥ 26.2** | the kernel is not the binding constraint for the memory-budget feature; Mesa is (§3.4) |

### 5.2 Kernel and userspace settings the port relies on

* Module: `xe` loaded and owning the card; `CONFIG_DRM_XE=m|y`, `CONFIG_DRM_XE_DISPLAY=y` if it drives a display.
* Firmware: current `bmg_guc_70.bin` / `bmg_huc.bin` — stale GuC firmware appears in multiple reset reports.
* Resizable BAR: leave it as large as the card and BIOS allow (16 GB seen on a B580, 32 GB here).
* Userspace: Mesa ≥ 26.2 on Intel; otherwise expect the ledger refusal and set a ceiling instead (§6.2).
* Port tunables: `STRATA_VK_MAX_BUDGET_MIB` (explicit ceiling), `STRATA_VK_DESKTOP_RESERVE_MIB` (default
  **1024 MiB**), `STRATA_VK_RESERVE_FLOOR_MIB` (default **512 MiB**). The reserve is clamped in the middle by
  design: **at least the floor, at most 25% of the card's local heap** (so a small card stays usable for the
  engine at all), and a request between the two is taken as asked. The reserve exists so the desktop keeps
  compositing: a two-display KDE/Wayland session was measured holding **354 MB in `kwin_wayland` alone**, which
  is why the floor is above the old 256 MiB.

### 5.3 The whole of the per-kernel work: a recipe

```bash
bash ports/vulkan/gates/run_gate.sh      # compiles from source, validates, runs every ICD present
```
then one 60-second sustained-compute smoke test on the card while watching for `Engine reset` in `dmesg`, then
re-check `vulkaninfo` device order after any crash (a restarted service can silently load a different GPU).
**Nothing needs recompiling between kernel versions** — which is the point of §2.

---

## 6. What the port enforces

### 6.1 Advisories

`harness/vk_compat.{hpp,cpp}` turns these rules into code and prints them with severities.

| Rule | Severity | Behaviour |
|---|---|---|
| Intel + kernel below the **per-card** floor | **Fatal** | refuses to run: **6.12** for a B580 (`e20b`), **6.14** for an Arc Pro B70 (`e223`) — the rule keys on `VkPhysicalDeviceProperties.deviceID`, so check yours with `lspci -nn | grep -i vga` |
| Intel + floor ≤ kernel < 7.0 | Warn | the xe compute-load crash range: bound the submissions, smoke-test first |
| Intel + 7.1 | Note | its Battlemage performance reports conflict (§4.2) |
| Intel + 7.2 | Info | the vRAM memory-pressure work is present |
| Intel + ≥ 7.3 | Note | TTM eviction is more aggressive — re-verify the reserve, do not assume 7.2 behaviour |
| Intel + beyond 7.3 | Info | the migration-queue work. **Cannot fire today: no such kernel exists**; it is in place so the note appears the day one does |
| Intel, any kernel | Note | the GuC 640 ms preemption timeout (§4.3) |
| Intel + Mesa < 26.2 + no driver figure | Warn | names the Mesa release that added the budget support on Intel |
| Any kernel marked `-rc` | Note | a prerelease is not released behaviour |
| Unparseable kernel | Warn | the rules cannot be applied, so every caveat stays live |

### 6.2 The one refusal that is not about a version: the ledger rule

If the free-memory figure comes from the heap total rather than the driver (`VK_EXT_memory_budget` absent —
on Intel that means Mesa < 26.2), **and** the card is discrete, **and** no explicit ceiling was given, then
**nothing may be allocated**: `usable_bytes()` is 0 and the first allocation exits 3 with the reason and the
remedy. That is the shape that filled an RX 6800 driving a desktop (`docs/AMD_HIP.md`, #380/#377). The remedy is
`STRATA_VK_MAX_BUDGET_MIB=<MiB>`, which is also the honest way to run on an older-Mesa Intel host.

### 6.3 How this is verified

`gates/run_gate.sh` reports **36 passed, 0 failed, 0 skipped** on the primary implementation and **34** on the
software one (`lvp`) at the time of writing; the gate prints its own totals, so a number here that disagrees with
a run is a stale document, not a result. Implementations present here (RADV and
llvmpipe), and the Linux-compatibility parts are covered by:

* a **15-case advisory table** with a **three-way boundary discriminator** (below the floor / at the floor on a
  B580 / at the floor on an Arc Pro B70), each case carrying an explicit *must-not-contain* so a case cannot pass
  on prose it merely recognises;
* the live kernel-release and Mesa-version parse, asserted against the real strings;
* three child-process cases for the ledger rule — over-budget refused, ledger+discrete+no-ceiling refused, and an
  explicit ceiling unblocking it.

Each check was **proven able to fail**: moving the per-card floor makes 2 table cases fail, and disabling the
ledger rule makes the child report **24 GiB as usable** — the bug itself, caught by the check.

---

## 7. The unresolved item, which is not a compatibility question

**Battlemage CCS engine resets under sustained compute are unresolved across every kernel in scope.** The
evidence spans the range: openSUSE on 6.18.2 and Arch on 6.18.32 (May 2026) with `engine_class=ccs` resets and
coredumps; `intel/compute-runtime#944` (timed-out job on the CCS engine, three resets in 20 minutes); a
reproducible B580 hang on GuC 70.65.0; a games repro; and the forum question about the 640 ms preemption timeout
under tensor-parallel inference. The 7.2 fix for "Battlemage G21 cold-boot black screens" is a *display* fix.

No compatibility code changes this. **Cold reset recovery in the post-7.3 cycle** is what would change the
*consequence* — a reset the driver recovers from instead of a wedge needing a power cycle — and it is the first
thing to re-test on real hardware. Until then: bounded submissions, no device spin-wait, smoke-test before
enabling a service, re-check device enumeration after a crash.

---

## 8. Verification status

**Verified on this machine:** the port has no kernel-facing interface (inspection + grep); its only enabled
extension is `VK_EXT_memory_budget`; `DRM_IOCTL_XE_MADVISE` exists in this box's 7.0 headers; this box is
7.0.0-34 on Ubuntu 24.04, Mesa 25.2.8, `CONFIG_DRM_XE=m` and `CONFIG_DRM_XE_GPUSVM=y`, and
`CONFIG_DRM_XE_PREEMPT_TIMEOUT=640000`; the gate passes its full suite on the AMD and software implementations
(once the card is swapped, on the Intel one - see `SWAP-PREFLIGHT.md`).

**From external sources (release notes, forum and press reports), not reproduced here:** release dates and
per-cycle DRM change lists; Mesa 26.2 release notes; the engine-reset reports; the 6.12 Xe2 claim.

**Unverified:** the exact 7.1 addition to the xe memory-pressure uAPI; whether ANV reports driver-backed budget
figures on xe at a given Mesa version (needs Arc hardware); whether the CCS reset reproduces with this port's
workload specifically; every performance claim for this workload, since no Arc benchmark exists here.

**Sources:** kernel.org release feed (version facts); Phoronix (7.1 graphics, 7.2 release, 7.3 DRM, 7.3-rc5,
Xe pull requests, TTM-7.3); Mesa 26.2.0 release notes; kernelnewbies 7.2; Wikipedia kernel version history;
openSUSE/Arch/Intel-community threads for the reset reports.

---

## 9. Change log

Corrections kept here rather than inline, so the body of the document reads as the current state.

| Date | Correction |
|---|---|
| 2026-10-03 | **7.4 does not exist.** An earlier revision listed it as a ladder row and put it first in the performance recommendation, on the strength of press coverage of pull requests *targeting* the 7.4 cycle. A patch labelled "for 7.4" is queued for a cycle that has not opened. Recommendations are now restricted to installable kernels, and §1.3 gives the command to check. |
| 2026-10-03 | **The Battlemage floor is 6.12, per card**, not 6.14 for all Intel. 6.12 is the first mainline kernel with Xe2 out of the box (B580); 6.14 is what our notes record for the Arc Pro B70 (BMG-G31). The rule now keys on the PCI device ID. |
| 2026-10-03 | **Rules made per-generation (3.6).** Intel's own KMD tables were used to add Alchemist explicitly. The previous version applied the Battlemage full-support line (6.12) to every Intel device, which would have refused a working Arc A770 on kernel 6.6, and raised the Xe2 CCS-reset warning for a generation that has none of those reports. Floors: Alchemist desktop/Pro 6.0 (full 6.2), Alchemist mobile 5.19, Battlemage B580-class 6.11 (full 6.12). Case table 17 -> 30 rows, including the cross-generation discriminator. |
| 2026-10-03 | **Extended from the kernel to the whole stack** (3.5): the loader, ICD files and their library resolution, libdrm, GPU firmware blobs, the Level-Zero/OpenCL path (labelled as *not* this port), the session type, and the build toolchain's capability probes. The document was renamed to match. The ICD check's first version was wrong in a way worth recording: it treated `library_path` as a file path and declared all nine working ICDs broken. |
| 2026-10-03 | **The reserve floor is 512 MiB**, not 256, after a field report of a two-display KDE/Wayland session holding 354 MB in `kwin_wayland` alone — the old floor was below the compositor it exists to protect. |
