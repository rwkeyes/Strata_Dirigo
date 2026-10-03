# Linux kernel compatibility for the Arc/Vulkan port, through 7.3

**Audience:** whoever ships or runs the Vulkan backend on Intel Arc.
**Written:** 2026-10-03. **Port revision:** branch `vulkan-arc-port` (`82bc2b2`).
**Question answered:** how hard is it to write compatibility for each kernel version *relative to the one
immediately before* (7.3 from 7.2, 7.2 from 7.1, 7.1 from 7.0).

---

## 1. The honest framing: this port is a userspace Vulkan client

Verified by inspection of the port itself, not assumed:

- The only Vulkan API-version constant in the tree is `VK_API_VERSION_1_2` (instance), and the only device
  extension it enables is `VK_EXT_memory_budget`. Everything else it uses
  (`storageBuffer16BitAccess`, `shaderInt16`, subgroup ops, `gl_NumSubgroups`) is core 1.1+ or a SPIR-V
  capability.
- `grep -rn "/dev/dri|ioctl|drm_|DRM_IOCTL|/sys/class/drm" harness/ gates/` returns **nothing**. The port opens
  no device node, issues no ioctl, parses no kernel header, and links no libdrm.

So there is no kernel ABI for it to track, no out-of-tree module, and no `#if LINUX_VERSION_CODE`. Everything
the kernel does for this port arrives through **Mesa** (ANV), which is where kernel uAPIs are consumed.

**Consequence, stated before the per-version detail:** the difficulty of "writing compatibility" for this port
is **zero lines of code for every transition from 7.0 to 7.3**. What actually varies per version is (a) whether
the driver/kernel pair provides the features the port relies on, (b) whether behaviour under memory pressure
changed in a way that invalidates an assumption, and (c) whether the long-running-compute stability problem is
any better. Those are verification and gating questions, not porting questions.

---

## 2. The ladder, with dates and sources

| Kernel | Status on 2026-10-03 | DRM/Xe-relevant changes |
|---|---|---|
| 6.18 | stable, in the wild | Battlemage CCS engine resets reported on 6.18.2 and 6.18.32 (openSUSE, Arch; May 2026) |
| 7.0 | stable (this box: `7.0.0-34-generic`, Ubuntu 24.04 HWE) | Xe SR-IOV + **Multi-Device SVM** (Battlematrix); VFIO Xe driver merged |
| **7.1** | **stable, 2026-06-14** (DRM merged 2026-04-20) | Xe: **memory-pressure / out-of-memory behaviour for vRAM** (new userspace API); Nova Lake P + Xe3P_LPG initial enablement; DRM RAS over Netlink |
| **7.2** | **stable, 2026-08-16** | Xe: **Battlemage G21 cold-boot black-screen fix**; Panther Lake Xe3 Arc B390 performance; Crescent Island improvements; cache-aware scheduling |
| **7.3** | **NOT STABLE — rc5 (2026-09-27), stable expected 2026-10-18** | Xe: Nova Lake-S graphics enabled **by default**, Xe3P considered stable; **TTM eviction more aggressive**; rc5 DRM fixes (AI bug fixes, Crescent Island power brake) |
| 7.4 | merge window open, first Xe PR landed | Xe: **cold reset recovery**; vRAM health check + degraded-memory handling |

Sources: Phoronix 7.1-graphics / 7.3-DRM / 7.3-rc5 / Intel-Xe-Linux-7.4-First-PR articles; Wikipedia kernel
version history (7.2 = 16 Aug 2026); ostechnix RC schedule (7.3-rc1 30 Aug → rc5 27 Sep); kernel community
forums for the engine-reset reports.

**7.3 is not released.** If a deliverable says "supports through 7.3", it currently means "supports 7.3-rc5".

---

## 3. Difficulty per transition (what was asked)

Ratings are for *writing compatibility in this port* — code, not verification.

| Transition | What changed that could matter to this port | Code required | Verification required | Difficulty |
|---|---|---|---|---|
| **7.0 → 7.1** | Xe gains better vRAM memory-pressure/OOM handling; Nova Lake/Xe3P enablement; DRM RAS via Netlink | **none** | re-run the gate on the new kernel; re-check the desktop-reserve refusal path, because memory-pressure behaviour is exactly what it interacts with | **Very low** (verification only) |
| **7.1 → 7.2** | Battlemage G21 **cold-boot black-screen** fix; Xe3/Panther Lake perf; Crescent Island; cache-aware scheduling | **none** | re-run the gate; the display-related fix is a reason to prefer 7.2 over 7.1 on Arc, not a code change | **Very low** (verification only) |
| **7.2 → 7.3** | Nova Lake-S by default; Xe3P stable; **TTM more aggressive** (eviction); rc5 DRM fixes | **none** | re-run the gate **and** re-verify the desktop reserve, because more aggressive TTM changes what happens when the card is full — the one behaviour this port's display contract depends on | **Very low**, plus one **required** re-verification |
| (7.3 → 7.4) | Xe **cold reset recovery**, vRAM health check/degraded memory | none expected | this is the first cycle likely to change the Battlemage guidance (§5) | future |

**Anything above "very low" in this port is not about the kernel.** Two items are genuinely moderate, and both
are version-gated *features we could adopt*, not compatibility we owe:

1. **Purgeable VRAM for the expert cache (7.1+, Mesa 26.2+).** The kernel exposes
   `DRM_IOCTL_XE_MADVISE` — verified present in this box's **7.0** headers (`/usr/src/linux-headers-7.0.0-34-generic/include/uapi/drm/xe_drm.h`,
   incl. `DRM_XE_VM_BIND_FLAG_MADVISE_AUTORESET`), so the base interface predates 7.1; what 7.1 added is a
   refinement I have **not** pinned to a specific constant (UNVERIFIED). Adopting it would let the kernel
   reclaim engine VRAM instead of failing an allocation — a better answer than a static reserve, but it needs
   Mesa-side support and a fallback for older kernels, so it is a *feature*, not a port.
2. **`VK_EXT_memory_budget` on Intel is gated by MESA, not the kernel.** Mesa **26.2** (2026-08-05) is where
   `anv: add memory heap budget tracking across VkInstance` landed — the same release adds
   `intel: madvise purgeable VMAs in Xe KMD`, `anv: fixup compute queue detection` (we use a compute queue),
   and syncs `xe_drm.h`. So the display-reserve work in §4 depends on the **userspace** version, and a user on
   Mesa < 26.2 must not have the port size itself from the ledger fallback.

---

## 4. What this means for the display-reserve work already in the port

The reserve logic (1024 MiB default, 256 MiB floor, 25% cap, enforced refusal) reads its free figure from
`VK_EXT_memory_budget`. That means:

- **On Intel with Mesa ≥ 26.2**: real driver figures, as intended.
- **On Intel with Mesa < 26.2**: the extension may be missing or report the ledger rather than a measurement.
  The port labels that case loudly, but the safe policy on a *display* card is to require an explicit limit
  rather than trust it — a recommendation to encode when Arc hardware is first tested.
- **Measured floor correction from the field data**: in the B580 report, `kwin_wayland` holds **354 MB** of GPU
  memory on a dual-display desktop. The port's 256 MiB floor is therefore *below* what a real KDE/Wayland
  session was using. Raise the floor to 512 MiB, or default the reserve higher (1024 MiB already covers it).

---

## 5. The item that is not a compatibility question at all

**Battlemage CCS engine resets under sustained compute are unresolved across every version in scope.**
Evidence spanning the range: openSUSE Tumbleweed on **6.18.2** and Arch on **6.18.32** (May 2026) with
`engine_class=ccs` resets and coredumps; `intel/compute-runtime#944` (a timed-out job on the CCS engine, three
resets in 20 minutes); a fresh community report of a **reproducible GPU hang and full system crash to forced
reboot on a B580 with GuC 70.65.0**; and a games report of xe engine resets on Battlemage. The 7.2 Xe fix for
"Battlemage G21 cold-boot black screens" is a *display* fix, not this.

No compatibility code changes this. What 7.4's **cold reset recovery** would change is the *consequence* (a
reset the driver recovers from, instead of a wedge needing a power cycle), and that is the first thing worth
re-testing on real hardware. Until then the guidance in PORT-PLAN §4b stands: bounded submissions, no
device spin-wait, smoke-test before enabling a service, re-check device enumeration after any crash.

---

## 6. Recommended support statement for the port

| Layer | Requirement | Why |
|---|---|---|
| Kernel | **7.2 or 7.3-rc5+ recommended**; 7.0/7.1 work; xe required for Battlemage (i915 for Alchemist) | 7.2 carries the Battlemage display fixes; 7.1 carries the vRAM pressure work |
| Kernel config | `CONFIG_DRM_XE=m` (or `=y`); `CONFIG_DRM_XE_DISPLAY=y` if the card drives a display; `CONFIG_DRM_XE_GPUSVM` only if the SYCL/USM path is also used | verified present in this box's 7.0 config; GPUSVM is for the llama.cpp SYCL path, **not** for this Vulkan port |
| Userspace | **Mesa ≥ 26.2** for a trustworthy free-memory figure on Intel | `anv: memory heap budget tracking` landed in 26.2 |
| Hardware | Resizable BAR as large as the card allows (16 GB observed on a B580, 32 GB on the AMD box here) | it sets the size of the local heap the driver reports |
| Blocked | Battlemage CCS resets under sustained compute | unresolved through 7.3-rc5; 7.4 adds cold reset recovery |

**Test-per-kernel recipe** (cheap, and it is the whole of the per-version work): run
`bash ports/vulkan/gates/run_gate.sh` on the target kernel (it exercises every ICD present, including the Intel
one), then one 60-second sustained-compute smoke test on the card while watching `dmesg` for `Engine reset`,
then re-check `vulkaninfo` device order after any crash. Nothing needs recompiling between kernel versions —
which is the point.

---

## 7. Verified vs not

**Verified here:** the port has no kernel-facing interface (inspection + grep); the port's only extension is
`VK_EXT_memory_budget`; `DRM_IOCTL_XE_MADVISE` exists in the 7.0 headers on this box; this box is 7.0.0-34 on
Ubuntu 24.04 with Mesa 25.2.8 and `CONFIG_DRM_XE=m`, `CONFIG_DRM_XE_GPUSVM=y`.

**From external sources, not reproduced here:** release dates and the per-cycle DRM change lists; Mesa 26.2
release notes; the engine-reset reports.

**UNVERIFIED:** the exact 7.1 addition to the xe memory-pressure uAPI; whether ANV reports driver-backed budget
figures on xe at any given Mesa version (needs Arc hardware); whether the CCS reset occurs with this port's
workload specifically; and everything about 7.3 stable, which does not exist yet.
