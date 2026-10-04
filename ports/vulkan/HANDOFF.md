# Handoff - host `vega`, the Arc swap, and the Strata Vulkan port

Written 2026-10-04 at the end of a long working session.  It assumes you know nothing about that session, so it
states what is true, what was measured, and what is genuinely unknown.  Read this first, then `NEXT.md`'s top
block (the port's own resume point), then `STATUS.md`.

---

## 1. What changed on the machine (and the one thing that broke it)

The host gained an **Intel Arc Pro B70** and lost the **AMD Radeon RX 7900 XTX**.  The monitor stayed on the
**AMD Ryzen iGPU** (Raphael, `gfx1036`), which is where it was before - the discrete card never drove a display
here and still does not: the Arc owns **no connectors at all** (`probe_display=0` below).

| | |
|---|---|
| discrete GPU | `03:00.0` Battlemage G31 [Arc Pro B70] `[8086:e223]`, driver **`xe`**, no `force_probe` needed |
| memory | 32 GiB VRAM (31.89 GiB usable, fully CPU-accessible), ReBAR on (Region 2 = 32G prefetchable) |
| firmware | GuC `bmg_guc_70.bin` 70.44.1, HuC `bmg_huc.bin` 8.2.10 - loaded from the **.zst** blobs |
| display | the **iGPU**: `card2-HDMI-A-1` connected.  Note the connector was `HDMI-A-2` with the Radeon in |
| second implementation | the iGPU itself, via RADV (`gfx1036`, 4 SIMDs) - the gate's radeon arm now picks it |
| third | llvmpipe (`lvp_icd`), as always |

**A boot failure that looks like hardware and is not.**  Six consecutive boots froze at the Plymouth splash with
a dead keyboard.  The cause was a file the previous session wrote to "pin" the display:

```
/var/log/Xorg.0.log:  (EE) open /dev/dri/card0: No such file or directory
                      (EE) No devices detected.  (EE) no screens found
journal:              sddm: Attempt 1..3 starting the Display server on vt 2 failed
journal:              amdgpu 0000:1a:00.0: [drm] REG_WAIT timeout - optc31_disable_crtc line:145
```

The amdgpu X driver resolved an explicit `BusID "PCI:0:26:0"` to `/dev/dri/card0`, which does not exist here (the
iGPU is `card2`, the dGPU was `card1`).  X found no devices, SDDM gave up after three attempts, and the last frame
stayed on screen - which reads exactly like a hard freeze, and happened with **either** card installed because the
config was GPU-independent.  The kernel's `REG_WAIT timeout` is a red herring; it is the symptom of the failed
modeset, not failing hardware.

**Current state:** `/etc/X11/xorg.conf.d/` holds only `00-keyboard.conf`.  Nothing pins the display, and nothing
should: with the monitor on the iGPU and the discrete card having no connected output, X picks the iGPU itself.
**Never pin a display by BusID on this box** - if it is ever genuinely required, prove it on a spare VT and read
`/var/log/Xorg.*.log` for `(EE)` before rebooting into it.  The boot is healthy today (`sddm active`, zero
"Could not start Display server" lines).

Two files survive from the swap preparation, both benign:

- `/etc/modprobe.d/50-xe-compute-only.conf` -> `options xe probe_display=0`.  Inert with no Intel GPU; with the Arc
  it keeps the card out of the display path, which is what we want.  Untested against the card before it was
  written; the card behaves as intended.
- `/etc/default/grub` was edited (by the user, while fighting the wrong cause): `idle=nomwait amdgpu.gpu_recovery=1`
  were added, `pci=realloc` predates it.  All benign.  **`nomodeset` lives only in the recovery entries - do not
  copy it into the defaults**, or normal boots lose the display driver too.

## 2. The engine cannot run on this box, and that is measured

The resident model (`qwen3.8-flash-next-coder-iq1_m`, a 58 GB MoE) was served by Strata's **HIP** engine built for
**gfx1100**.  With the 7900 XTX gone there is no HIP device:

```
$ strata generate --pack ... --tokens ...        # the real thing, not a simulation
strata generate: PCIe probe: 12.9 GB/s host->device ...
strata generate: native pack: .../packs/coder-iq1_m experts ...
strata generate: cudaMemcpy failed for blk.0.ffn_gate_inp_shexp.weight
```

The kernel's own topology is the authority, not the runtime's enumerator:

```
/sys/class/kfd/kfd/topology/nodes/1/properties:  gfx_target_version=100306   (gfx1036, the iGPU)
$ rocm_agent_enumerator                          gfx1100                     (stale - the departed card)
```

Consequences, all current:

- **`strata-coder.service` is stopped AND disabled** (so a reboot does not crash-loop it against a missing GPU).
  Re-enable it only when a HIP-capable GPU is back.
- **Hermes delegation to `:18110` has no backend.**  The alternative tunnel (`ornith-tunnel.service` -> z820a
  Ornith-1.5-9B `@127.0.0.1:18400`) is **inactive** too.  Expect local subagents to fail.
- `/etc/modprobe.d/50-xe-compute-only.conf` and the display config are the only persistent changes from the swap.

## 3. What was measured with the model, and what it is NOT

llama.cpp's Vulkan build (`~/llama-050/build-vulkan`, device 0 = the Arc) with the model's **NVMe** shards, experts
split between the card and CPU threads:

| expert placement | pp64 | tg32 |
|---|---|---|
| all 48 layers' experts on CPU (Arc does attention/shared only) | 15.75 t/s | 10.03 t/s |
| **24/48 layers' experts on the Arc** | **25.72 t/s** | **13.72 t/s** |
| the same with the page cache dropped (cold) | 24.35 t/s | 13.65 t/s |

Storage, measured directly on the shards (`dd`, 4M direct): **NVMe 2.7 GB/s vs USB disk 250 MB/s**.

**Read the caveats as carefully as the numbers.**

- **This is llama.cpp, not Strata.**  Strata's engine cannot run here (§2).
- **It is not comparable to the 62-92 t/s Strata did on the 7900 XTX** (same box, different software, different
  expert placement, different cache design).  Reading 13.72 against 92 as a regression is a category error.
- The cold run **refuted** the expectation that the page cache was hiding the NVMe: ~5% prefill, 0.5% decode.  At a
  64-token prompt too few experts are touched for the disk to matter; the NVMe's speed shows at long contexts and
  large batches.
- `ggml_vulkan` reports **`int dot: 0`** for the Arc, i.e. no integer-dot-product path for the sub-4-bit kernels -
  the likely reason the card's contribution is +37-63% rather than multiples.  Worth investigating: the port's own
  device query *does* see `VK_KHR_shader_integer_dot_product` as an extension, so this may be a detection or
  feature-enablement gap rather than absent hardware.

## 4. The Strata -> Arc Vulkan port: where it stands

Living in `~/strata-vulkan-wt` (worktree of the fork, branch **`vulkan-arc-port`**), the port itself under
`ports/vulkan/`.  Read `NEXT.md`'s top block first; `gates/run_gate.sh` is the only authority on coverage.

**State: the port's kernel suite runs on the Arc.**

```
intel_icd  (Arc Pro B70, BMG G31)   156 passed, 0 failed, 1 skipped
lvp_icd    (llvmpipe)               154 passed, 0 failed, 1 skipped
radeon_icd (AMD iGPU, RADV)         155 passed,  1 failed, 1 skipped
```

- The Intel **skip** is `gemm_coopmat`: this device reports no usable M16N16K16 subgroup-scope f16 -> f32 config.
  The gate exits non-zero on any skip by its own rule ("a skipped case is not a passing one"), so **the Arc's exit
  code is 1 with zero failures** - read the totals, not the exit code, and say which you are quoting.
- The iGPU's single failure is `budget: independent requery agrees`: an integrated GPU's free figure is system RAM
  shared with the OS, so two queries disagree by construction.  Left red on purpose (documented in `STATUS.md`).
- The one **device-specific kernel defect found so far** was `quantize_q8_K` on the Arc, off by one byte in a
  block's scale: the driver FOLDS `1.0f/(-127/mx)` into `mx/-127`.  The case now carries both forms as images,
  demands a byte-exact match to one, and prints which (`Arc -> folded, iGPU -> source`).  Commits `c8d32c4`.

**The Battlemage stability question is answered for the loads tested.**  `gates/smoke-arc.sh` ran 8 concurrent
instances for 8 minutes: **5,744 suite runs (~890k case executions), 0 kernel failures, 0 hangs, 0 xe errors, no
latency creep**, after 502 sequential runs likewise clean.  The card did **not** wedge - the open bug
(`intel/compute-runtime#948`) did not reproduce.  The plan's own bar is "an hour of inference", so the honest claim
is "has not wedged under the loads tested"; an inference-shaped arm is what would close that gap.

## 5. What to do next, in order

1. **Stage 3, the remaining step: `case_recorded_step`.**  The API is written and compiling
   (`Ctx::record_begin` / `record_dispatch` / `record_end_and_submit` / `replay_recorded`, commit `55555ca`), and
   the refactor it rests on is verified by the full existing suite.  The **new calls are exercised by no case**, so
   they are not evidence yet.  The case's design is in `NEXT.md`: three chained copies of the harness's own copy
   kernel; the single-shot `dispatch()` path as the reference; then **write NEW bytes into the source and replay** -
   which no re-recording path can pass.  Also outstanding: the destructor does not destroy `rec_fence_`.
2. **Stage 4** - device-local memory + staging + `VK_EXT_memory_budget` fit accounting (the gate currently uses
   host-visible memory only, which is correct for a gate and wrong for a benchmark).
3. **Stage 5** - the hand-written GEMM / prefill path.  The plan calls this "the only part that is genuine
   engineering rather than translation. Multi-day."  Do not promise a date before it is done.
4. **Stage 6** - engine integration: `STRATA_ENABLE_VULKAN`, the arena, `gpu_arch_problem` for Intel.  **Its
   verification changed**: the plan says to compare tokens against the HIP build, and this box can no longer produce
   one (§2).  Use the engine's CPU oracle on the same inputs, or a token stream captured pre-swap if one exists.

## 6. Traps that cost time in this session (so they do not cost yours)

- **`pkill -f` with a pattern that also appears in your own command line kills your own shell.**  The bracket trick
  (`[s]moke-arc`) does *not* save you if the literal name appears elsewhere in the same command - a log path was
  enough.  Kill by PID, or from a pattern that cannot appear in your own line.
- **A test that creates and removes a file needs a per-process path.**  `case_firmware_variants` used one fixed
  `/tmp/...` name; under 8 concurrent instances one cleanup deleted the file another was reading (2 of 8 jobs),
  invisible in 502 sequential passes.  Fixed with `/tmp/<name>-<pid>/`.
- **Two harness checks are incompatible with a busy card by construction** (`budget: independent requery agrees`,
  `stack: resolvable ICD not flagged`).  `smoke-arc.sh` classifies them; an explicit ceiling does **not** stabilise
  the requery (measured: still 8/8 with 16 GiB set).
- **A recorded step needs one descriptor set per dispatch** - host updates happen at record time, dispatches run at
  submit time, so a shared set leaves every dispatch reading the last binding.  Caught before it produced a wrong
  token; the same class as the grouped-expert wave's wrong-buffer read.
- **`vulkaninfo`'s ICD filename is `intel_icd.json` here, not `intel_icd.x86_64.json`.**  A wrong
  `VK_ICD_FILENAMES` silently yields no device and an empty summary.
- **The Vulkan loader is older than the drivers advertise** (`libvulkan.so.1.3.275` vs ICDs claiming 1.4.318/1.4.329).
  Harmless so far; suspect it first if a documented extension is missing.
- **The kernel wants GuC 70.54.0** for the Arc and this box ships 70.44.1 (`linux-firmware` update would clear it;
  the only consequence seen is SR-IOV PF migration being disabled, which we do not want).
- **Mesa is 25.2.8**, below the 26.2 floor the port's own rules name for trusting a driver-backed free-memory
  figure on Intel - so the explicit-ceiling path (`STRATA_VK_MAX_BUDGET_MIB`) is the one that matters here.

## 7. Loose ends outside the port

- **The USB disk (12.7 TB, `sda`) holds the Strata packs** (`Strata-data/packs` is a symlink into it) and the base
  models' shards.  With it unplugged no model loads at all - the NVMe holds only the coder's GGUF shards.  The pack
  for the coder model is **1406 MiB**, so it could be staged on the NVMe to make that model disk-independent; not
  done.
- **The NVMe staging of the base models is a husk**: `~/strata-gguf-iq3/flat/` holds 88- and 125-byte symlink stubs
  from an aborted HuggingFace download, 4K of real data.  Re-staging needs 44 GB against 27 GB free.
- **`~/start-strata-flash-next.sh`** (home dir) starts the coder model from the NVMe shards; its preflight is what
  caught the missing pack.  It cannot start anything today (§2), and its default is the NVMe checkpoint.
- **The Hermes desktop app runs on this box; never kill or suspend it.**
