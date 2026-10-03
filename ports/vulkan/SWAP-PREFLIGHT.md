# Swapping the 7900 XTX for an Arc Pro B70 — pre-flight

**Scope:** this host (`vega`, Ubuntu 24.04.5 LTS, kernel `7.0.0-34-generic`) has its **Radeon RX 7900 XTX**
(`1002:744c`) removed and an **Intel Arc Pro B70** (`8086:e223`, BMG-G31, Battlemage/Xe2) installed in its place.
The card is not yet fitted; everything below is either verified on the host as it stands, or listed as an action.

**Why this document exists:** the port under `ports/vulkan/` was built and measured *against the AMD card*. A
hardware swap silently invalidates assumptions (a driver floor, a memory figure, a display connector, the GPU the
local model runs on). This is the audit of those assumptions, and the list of things that must change with the
card rather than after it.

## Verdict table

| # | Item | State | Evidence |
|---|---|---|---|
| 1 | Kernel recognises the B70 | **OK** | `modinfo xe` → `alias: pci:v00008086d0000E223...` on 7.0.0-34 |
| 2 | xe driver present and display-capable | **OK** | `CONFIG_DRM_XE=m`, `CONFIG_DRM_XE_DISPLAY=y`; `xe.ko.zst` in this kernel's modules |
| 3 | Driver will bind without `force_probe` | **OK** | Intel's table gives E223 full support from 6.17; this kernel is 7.0 |
| 4 | i915 cannot claim the card (correct) | **OK** | `modinfo i915` has no E223 alias — xe must own it |
| 5 | GPU firmware blobs | **OK** | `xe/bmg_guc_70.bin.zst`, `xe/bmg_huc.bin.zst`, `i915/bmg_dmc.bin.zst` present; `CONFIG_FW_LOADER_COMPRESS_ZSTD=y` loads them |
| 6 | Mesa/ANV knows the device | **OK** (name only) | Mesa 25.2.0 added PCI IDs `0xe220-0xe223` as BMG G31; `(BMG G31)` is in the installed 25.2.8 driver |
| 7 | X11 needs no card-specific config | **OK** | `/etc/X11/xorg.conf.d/` holds only `00-keyboard.conf`; autodetect |
| 8 | **Local model service** | **BREAKS** | `strata-coder.service` runs `backend = hip`, `gpu = 0`, maps `libamdhip64.so.7` — pinned to the card being removed |
| 9 | **VRAM safety rule cannot use the driver figure** | **ACTION** | Mesa 25.2.8 < 26.2 (anv heap budget tracking); noble tops out at 25.2.8 and the configured Intel (kobuk) PPA carries no `mesa-vulkan-drivers` |
| 10 | **Display connector** | **ACTION** | Monitor is on `card2-HDMI-A-2`; B-series Pro cards are 4× DisplayPort (or mini-DP) and have no HDMI |
| 11 | Resizable BAR | **ACTION** | Current card's BAR is 256M — ReBAR off; Intel asks for it on Arc |
| 12 | XMX / cooperative matrix | **PATH PROVEN, TUNING PENDING** | the toolchain *does* emit it (corrected 2026-10-03 - the "absent" verdict was a broken probe shader of mine); a CMA GEMM now exists and is verified on RADV on three shapes, worst relative error 4.5e-4 |
| 13 | Engine sizing | **ACTION** | serve config uses `--resident-budget-gib 20`, tuned for the 24 GiB card; the B70 has 32 GiB |
| 14 | Physical fit | **CHECK** | 2-slot blower, ~267 mm, 230 W reference (160–290 W by partner) — confirm a PCIe power connector and PSU headroom |

## 1. The item that matters most: the local model dies with the card

**DECIDED (2026-10-03): swap now, and accept cloud tokens for subagents until an Arc backend can serve a
model.** The alternatives below are kept as the fallbacks if the gap runs long, but nothing waits on them: the
card goes in when it arrives, and local delegation is expected to be down until the Vulkan backend can serve.

`strata-coder.service` is what Hermes subagents run on (delegation endpoint `:18110`). It is the engine under
development, built for AMD: `backend = hip`, `gpu = 0`, and the process maps `libamdhip64.so.7`. **Removing the
7900 XTX removes the local coder**, and the port that would put it on Arc is not finished — so there is a window
where local inference has no GPU. Options, in the order I would try them:

1. **Serve it from `z820b`**, which already has a 7900 XTX (24 GiB) and ROCm, and point the delegation endpoint at
   it. `ornith-tunnel.service` already exists as the pattern for reaching a model on another box.
2. **Keep using the local endpoint on this host until the Arc backend lands**, i.e. do the swap when the Vulkan
   backend can actually serve, not before.
3. **SYCL/Level-Zero on Arc** — Level-Zero 1.17.39395 and OpenCL are already installed here, and the Intel LLM
   Scaler (vLLM) path is the documented preference for Battlemage. It shares nothing with this port, so it is a
   second project rather than a fallback.

Accepting cloud tokens for subagents in the gap is a cost decision, not a technical one — worth deciding *before*
the card goes in rather than discovering it when the endpoint stops answering.

## 2. Mesa: the card works, but the VRAM-safety rule does not

Mesa **25.2.0** added PCI IDs `0xe220-0xe223` as BMG G31, so the installed **25.2.8** will enumerate the B70. What
it cannot do is report a usable free-memory figure: `VK_EXT_memory_budget`'s heap budget/usage on Intel needs Mesa
**≥ 26.2**. Noble's newest is 25.2.8, and the Intel graphics PPA configured on this host (`kobuk-team/intel-graphics`)
lists no `mesa-vulkan-drivers` at all, so no `apt` upgrade reaches it.

Consequence, by design: with no driver figure, the port's ledger rule treats the free number as unsafe on a
display card, returns **0 usable bytes** and **refuses to allocate**, naming `STRATA_VK_MAX_BUDGET_MIB`. That is
the correct refusal — but it means the card will appear to "not work" until one of these is done:

- set an explicit ceiling (`STRATA_VK_MAX_BUDGET_MIB=<MiB>`), which is supported and is the honest trade: you
  declare the budget instead of the driver reporting it; **or**
- get Mesa ≥ 26.2 (a newer Mesa PPA or a distro release with it), which restores the automatic figure.

## 3. Display: the connector changes with the card

The connected output today is **HDMI** on the discrete card. Arc Pro B-series cards are **4× DisplayPort 2.1**
(full-size on the ASRock/Sparkle B70 boards, mini-DP on some others) with **no HDMI**. Budget for a DP or mini-DP
cable/adapter. Two things soften this: X11 needs no configuration (autodetect, no card-specific `xorg.conf`), and
the Ryzen iGPU (`1002:164e`) can drive a display, so a mis-probe does not leave the box headless — but the cable
would have to move to a motherboard port.

## 4. Performance expectations to set before, not after

- **The matrix-unit path exists and is verified - on AMD, not yet on Intel.** A cooperative-matrix GEMM
  (`C = A·B`, fp16 operands, fp32 accumulator) is in the port and passes on three shapes against a host
  reference, worst relative error 4.5e-4 — the fp16 storage format's own limit, not the kernel's. What is still
  Intel-specific and unmeasured is the same shader on Xe2: its subgroup width (16/32 rather than RADV's 64) and,
  for the real engine, the dequant step, because IQ1_M weights must become fp16 before the matrix units can
  consume them. Two things are not gated on that at all: the 32 GB and the 608 GB/s are available to the
  plain-FMA kernels on day one, which is already more than the card being removed offers.
- **Resizable BAR:** the current card shows a 256M BAR (ReBAR off). Intel asks for ReBAR for Arc performance;
  check the BIOS (Above-4G decoding + ReBAR) before benchmarking, or the numbers will be quietly low.
- **Engine sizing:** `--resident-budget-gib 20` was tuned for 24 GiB. On 32 GiB there is room to raise it, and the
  port's own reserve is capped at 25% of the heap (8 GiB on this card) with a 1024 MiB default.

## 5. What to run the moment the card is in

```bash
# 1. did the kernel bind it, and which driver owns it?
lspci -nn | grep -i 8086:                # expect ... [8086:e223]
lsmod | grep -E '^xe' ; dmesg | grep -iE 'xe .*(bmg|g31|GuC|VRAM|probe)'
# expect "Found bmg/g31", a GuC version, and a VRAM size near 32 GiB. A probe failure here is firmware.

# 2. does Vulkan see it?
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json vulkaninfo --summary | head -30

# 3. the port's own gate, on the Intel implementation (and note the radeon arm reporting "no device")
cd /home/bob/strata-vulkan-wt && bash ports/vulkan/gates/run_gate.sh

# 4. the budget/refusal behaviour on the real card, before trusting the desktop reserve
#    (expect the refusal described in §2 until Mesa >= 26.2 or an explicit ceiling is set)
```

## 6. What this swap changes about the port's existing evidence

- Every measurement recorded on **RADV (7900 XTX, subgroup 64)** becomes historical. The **Intel ICD becomes the
  reference implementation**, and the widths that matter become Battlemage's (16/32) instead of RADV's 64. The
  llvmpipe arm (width 8) stays useful as the width-independence check.
- The **Alchemist** rules and notes stay **unverified**: this card is Battlemage, so fitting it does not test them.
- The gate runs every ICD that reports a device, so it adapts on its own — the radeon arm will report "no device
  on this machine", which is a result, not a failure.

## 7. Assumptions this audit found to be wrong

Recording these because each one would have caused a wrong action:

1. **"This box has a 7900 XTX and a K620."** It does not. `lspci` shows the Navi 31 card and the Ryzen iGPU
   (`1002:164e`) — no NVIDIA card on this host. That claim appeared in `STACK-COMPAT.md` and `STATUS.md` and has
   been corrected. (The Quadro K620 belongs to the Z820 boxes.)
2. **"The firmware is missing."** The port's own check said so, because it probed the uncompressed filename while
   Ubuntu ships `.zst` and the kernel loads that transparently. Fixed in the check, with a case that proves both
   directions (`firmware blobs probed`, `compressed firmware variant found`, `absent firmware still reported
   absent`). Every blob for both generations is in fact present.
3. **"Mesa 25.2.8 may not support the B70 at all."** Wrong, and close to being reported as a blocker: the device
   *name* for the B70 is absent from the driver, but the **PCI ID table** is what decides, and Mesa 25.2.0 added
   `0xe220-0xe223` as BMG G31. Name missing ≠ device unsupported.
4. **"24 GiB"** as the card size throughout the reserve discussion — that was the AMD card. The B70 is **32 GiB**.

5. **"The toolchain cannot emit cooperative matrix, so XMX is unreachable."** Wrong, and wrong *because the check
   was wrong*: the probe shader omitted `GL_KHR_memory_scope_semantics`, called `coopmatMulAdd` instead of
   `coopMatMulAdd`, and used `gl_MatrixLayoutRowMajor` instead of `gl_CooperativeMatrixLayoutRowMajor`. `glslc`
   emits the opcode and the SPIR-V validates. That false verdict reached three documents before the probe was made
   to compile the real kernel; a capability probe that is a fixture rather than the shipping shader can report a
   limitation that does not exist.