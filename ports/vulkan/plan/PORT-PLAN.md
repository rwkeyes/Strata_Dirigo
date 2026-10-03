# Strata on Intel Arc through Vulkan — port plan

**Target:** Intel Arc Pro, Battlemage (Xe2, e.g. B60/B70) first, Alchemist (Xe-HPG, A-series) second.
**Backend chosen:** Vulkan 1.3 compute (Mesa ANV), not SYCL/oneAPI.
**Status:** stage 1 of 6 complete and verified on a real Vulkan GPU (see `STATUS.md`). Nothing here has run on
Intel silicon yet.

---

## 1. What "port to Arc" actually means for this engine

Strata's GPU code is CUDA, compiled twice — once by `nvcc`, once by `hipcc` through `include/strata/hip_compat`
(which supplies `cuda_runtime.h` and friends on top of HIP). There is no backend interface and no second
implementation: the device layer is a single `cudaMalloc` arena (`src/core/device.hpp`), kernels are free
functions taking a `void* stream`, and every launch is an in-file `<<<grid, block, 0, stream>>>`.

Measured size of the surface that has to move:

| Part | Files | Lines |
|---|---|---|
| `src/kernels/cuda/*.cu` | 44 | 17,342 |
| `src/prefill/*` (prompt path, CUDA/HIP) | 8 | 4,964 |
| `src/core/*` (device, pinned, streams, graphs) | 28 | 11,984 |
| `src/program/*` (the per-layer graph) | 2 | 6,745 |
| **Total `.cu`/`.cuh` in the tree** | — | **~21,000** |

CUDA API surface actually used, by call count (this is the worklist, not a guess):

| CUDA call | Uses | Vulkan replacement | Difficulty |
|---|---|---|---|
| `cudaMemcpy` / `…Async` / `…HostToDevice` / `…DeviceToHost` | 483 / 109 / 324 / 216 | staged `vkCmdCopyBuffer`, or host-visible coherent memory | mechanical |
| `cudaMalloc` / `cudaFree` / `cudaMemset` | 386 / 278 / 49 | `DeviceArena` over `VkDeviceMemory` slabs | mechanical |
| stream create/sync/flags | 388 | one `VkQueue` + per-stream `VkCommandBuffer` | mechanical |
| events (`Record`/`ElapsedTime`/`Query`) | 56 / 22 | `vkCmdWriteTimestamp` + a query pool | mechanical |
| **stream capture + graphs** (`GraphLaunch` 41, `BeginCapture` 22, `Instantiate` 20) | ~120 | **one recorded command buffer replayed per token** | design, not translation |
| `cudaHostAlloc` / `HostGetDevicePointer` (mapped pinned memory) | 33 / 21 | `HOST_VISIBLE|HOST_COHERENT` + persistent map | mechanical |
| `cudaMemGetInfo` | 28 | `VK_EXT_memory_budget` (`heap_budget - heap_usage`) | mechanical |
| `cudaDeviceGetAttribute` | 21 | `vkGetPhysicalDeviceProperties` | mechanical |
| `cuBLAS` / `cuBLASLt` (`GemmEx` 7 sites) | 7 sites | **written by hand** (see §3) | real work |

So the shape of the job: about 90 % of the CUDA surface maps to Vulkan almost one-to-one, three things do
not, and the kernel bodies are a grind that is already gated by this repo's own parity tests.

## 2. The three things that do not map

1. **CUDA graphs → recorded command buffers.** This is a *simplification*, not a loss: a `VkCommandBuffer`
   recorded once and re-submitted per token is exactly what the graph was buying. It also means the port
   should record **one command buffer per decode step shape** and re-submit it, never re-record per token.
   (Measured on the Atlas port: a graph-replaying engine makes ~2 driver calls per token, so per-dispatch
   overhead — the usual reason to want command buffers — is not the prize here; re-record cost is.)
2. **`cuBLASLt`.** Seven call sites, no Vulkan equivalent. The prompt path needs a real GEMM. Two ways:
   plain FMA shaders (works everywhere, bandwidth-bound on Arc, fine for a first port) or **cooperative
   matrix** (`VK_KHR_cooperative_matrix`) to reach the Xe XMX units. The second is where Battlemage's
   performance actually lives, and it is blocked on this host's toolchain (§4).
3. **`__threadfence_system` / the host handshake.** `elementwise.cu`'s doorbell design has the HOST spin on a
   device-written counter, and the CUDA kernel orders that write with `__threadfence_system()`.
   **Vulkan has no equivalent.** Device writes become host-visible only through a coherent mapping plus an
   explicit synchronization/signal boundary; a device-scope release does not order writes for the CPU. The
   port must replace the doorbell with a real synchronization primitive (`vkWaitForFences` on a timeline
   semaphore, or a tiny per-step submit) — a host spin on mapped memory is not portable and must be designed
   out, not translated.

## 3. The kernel port: how the 44 files get done

Not by hand-translating all of them at once, and not by asking a model to "port the engine". The working
method (from the Atlas and llama.cpp ports in this tree's history):

* **One kernel per file, one `main()` per file.** A GLSL compute module has exactly one entry point; a CUDA
  file with four kernels has to become four files or the engine's four name lookups cannot resolve.
* **Interface convention** (already used by `harness/vk_compute.hpp`): storage buffers at `set = 0,
  binding = 0..N-1` in CUDA argument order; every scalar in ONE push-constant block, std430, in the CUDA
  scalar order; `local_size_x = 256`; counts are `int` (the port documents element counts < 2^31).
* **Generate structure with the local model, hand-port the numerics.** Measured on this project already:
  the worker's elementwise/reduction/conversion translations came back *bit-exact* on the first pass
  (bf16 and fp16 converters, scale, add); the kernel whose accuracy was wrong was wrong in a *transcendental*
  (softplus's log1p), which is exactly the class a model cannot get right by pattern-matching. Rule: LUTs,
  block scales and transcendental accuracy come from the source and from a measurement.
* **Gate every kernel against the engine's own CPU reference.** This repo already ships 31 parity tests with
  the references inline (`src/kernels/*_parity.cpp`), and the port harness re-uses their arithmetic. A kernel
  is accepted only when a command exits 0 comparing element-by-element (bit-exact for bf16/fp16 and for
  elementwise f32; a *measured* tolerance where a driver transcendental is involved).

Order of work (highest value first, each row is independently gate-able):

| Wave | Kernels | Why first |
|---|---|---|
| 1 ✅ | elementwise glue: scale, add, f32→bf16, f32→f16, gdn_gate, rms_norm, silu | done; they touch every layer |
| 2 | `kv_q8.cu`, `rope.cu` (NeoX), `quantize_act.cu`, `ple.cu` | KV + positions; small, high frequency |
| 3 | `native_bf16.cu`, `s_gemv.cu`, `s2_gemv*.cu`, `bf16_gemv.cu` | the GEMVs; the actual throughput |
| 4 | `native_flash_attn.cu`, `qsa*.cu` (5 files) | attention; reductions, width-sensitive |
| 5 | `native_moe.cu`, `router_top10.cu`, `native_expert*.cu`, `iq_kernels.cu`, `s2_expert_grouped.cu` | MoE + the 7 quant formats; the long tail |
| 6 | `prefill/*` (8 files) | needs the hand-written GEMM |

## 4. Intel-specific findings that change the plan

**Battlemage (Xe2) - the risk that is not in this codebase.** On the xe kernel driver, Arc Battlemage under
sustained *compute* load can wedge permanently (ccs/bcs engine reset, power cycle required) via Level-Zero,
OpenCL **and Vulkan** — it is the driver, not the frontend. It is an open upstream bug
(`intel/compute-runtime#948`), present on every kernel that supports BMG (6.14/6.17/7.0; 6.8 has no BMG
support at all). What *does* run the card stably is Intel's LLM Scaler vLLM container on the validated stack
(Ubuntu 25.04 + `kobuk-team/intel-graphics` PPA, oneAPI/Level-Zero), i.e. a different runtime and a different
frontend.

**Consequence for this port, stated plainly:** a Vulkan backend can be written and *verified numerically*
anywhere, but whether it survives an hour of inference on a B70 is a hardware/driver question the port
cannot answer. That check is a 60-second smoke test on the actual card and it should be the FIRST thing
done on Battlemage hardware — before any scheduler work, and before promising a delivery date. If it wedges,
the honest options are (a) Alchemist first, (b) the sane-driver stack, or (c) not Vulkan on Xe2 at all.

**Subgroup width.** RADV is 64 lanes; Intel Xe is 8/16/32 depending on the kernel and the hardware. Every
CUDA `__shfl_*`/`& 31`/`>> 5` becomes a subgroup builtin with the width read at runtime. The ported
`rms_norm.comp` is the template: `gl_NumSubgroups` and `gl_SubgroupSize`, and the host sizes the dispatch by
`local_size_x / subgroupSize` — the two must agree or the grid is wrong, which is not a compile error.

**Toolchain blocker for Battlemage's real speed.** Xe2's matrix units are reachable from Vulkan only through
cooperative matrix, which needs glslang > 15.1. This host has glslang 14.0 (shaderc) / 15.1, and it cannot
even emit a double-precision `exp` for SPIR-V (`shaders/blocked/`). So the port can reach *correct* on Arc
today and needs a toolchain bump (glslang ≥ 16, or naga/rust-gpu) before it can reach *fast* on Xe2.

**Alchemist (Xe-HPG)** has none of the xe wedge reports, and SYCL and Vulkan are both described as stable
there — so Alchemist is the sensible first *hardware* target even though the code port is the same.

## 4b. The display contract: the Arc card stays the video card

**The requirement:** the card that runs Strata is the card that drives the desktop, and Strata must leave the
desktop enough to composite. Not gaming - a desktop.

**What the engine already promises, which the Vulkan path must match rather than reinvent:**
`--vram-reserve-mib` (default **700 MiB**, `src/program/generate.cpp`) is the engine's own reserve for its
graphs, scratch and head, and `docs/AMD_HIP.md` states the AMD path "leaves 1 GiB of VRAM headroom".
**The reason this is not a formality is in that same file:** on an RX 6800 that drives the desktop,
`hipMemGetInfo` did not subtract what the desktop and other programs held, so `--expert-cache auto` filled the
card and decode fell 41 -> 30 tok/s (#380/#377). A free-memory number that ignores everyone else is the bug.

**What the Vulkan backend does instead:**

1. **Asks the driver, not itself.** `VK_EXT_memory_budget` reports usage for the HEAP, not just this process.
   It adds no entry points (a capability check plus the enabled-extension name is the whole wiring), and where
   it is absent the fallback is the heap total - which is a **ledger, not a measurement**, and the code says so
   out loud rather than degrading silently.
2. **Holds back a desktop reserve**, as a pure function with a floor and a cap: default **1024 MiB**, never
   below **512 MiB**, never more than **25%** of the card. The floor was 256 MiB until field data moved it: a
   two-display KDE/Wayland B580 desktop held **354 MB in kwin_wayland alone**, so 256 MiB was a reserve smaller
   than the compositor it was meant to protect. It composes with the engine's own 700 MiB rather than
   overlapping it, because the engine's planner subtracts that from the figure this layer reports as usable.
3. **Refuses, loudly, rather than over-allocating.** The check runs against the driver's figure plus the
   reserve *before* any allocation, and names the numbers. A backend that cannot fit must say so; allocating
   anyway is how the card gets filled and the desktop stops compositing.

What that yields, from the real code (`STRATA_VK_FORCE_BUDGET_MIB` used to pose each state):

| State | Free (driver) | Reserve | Usable |
|---|---|---|---|
| Arc 32 GiB, desktop default | 32.0 GiB | 1.0 GiB | 31.0 GiB |
| Arc 32 GiB, browser doing GPU compositing | 32.0 GiB | 2.0 GiB | 30.0 GiB |
| Headless box (no display) | 32.0 GiB | 0.5 GiB (floor) | 31.5 GiB |
| Card already busy (e.g. another model resident) | 3.0 GiB | 1.0 GiB | 2.0 GiB |
| Card nearly full | 0.5 GiB | 1.0 GiB | **0 -> refuses everything** |

Measured on this box as a live example of why the driver figure matters: the resident local model holds the
7900 XTX, so RADV reports **0.19 GiB free of 24 GiB** and the harness refused to allocate - the exact behaviour
that would have saved the RX 6800 incident.

**Tunables** (env, engine-side names to follow in stage 6): `STRATA_VK_DESKTOP_RESERVE_MIB` (default 1024),
`STRATA_VK_RESERVE_FLOOR_MIB` (default 512; set 0 only for a small correctness harness that must run beside a
resident model). 512 MiB is the compositor-only floor (measured: 354 MB for a two-display KDE/Wayland session);
1.5-2 GiB is the number if a browser is compositing.

**Display safety beyond memory, which no reserve can buy back:**

* **Do not translate the device spin-wait.** `elementwise.cu`'s doorbell has a kernel spin until the host
  answers. On the display card a hung compute kernel is a KMD timeout at best, and on Battlemage (§4) a wedge
  that needs a power cycle - the desktop dying with it. The Vulkan path must use fences/timeline semaphores
  per step, never a kernel that waits.
* **Bounded submissions.** Submit and wait per step rather than leaving long-running work queued, so a fault
  surfaces as one failed request instead of an unresponsive desktop.
* **Smoke-test before enabling a service**, as the Arc notes already say, and after any hard crash re-check
  device enumeration order before restarting: a crashed-and-restarted service can silently load a different
  GPU.

## 5. Staged plan, with what each stage is worth

| Stage | Content | Verifiable by |
|---|---|---|
| 1 ✅ | Vulkan compute layer + numeric gate for the first kernel wave | `gates/run_gate.sh` → 14/14 on RADV |
| 2 | kernel registry + kernel waves 2-3, gated as they land | same gate, extended |
| 3 | recorded command buffers (the graph replacement) + one captured decode step | replay a captured step and compare tokens to the HIP path |
| 4 | device-local memory + staging + `VK_EXT_memory_budget` fit accounting | the engine's own VRAM plan printed against the driver's numbers |
| 5 | hand-written GEMM (+ cooperative matrix if the toolchain allows) | prompt-path parity vs the CPU reference |
| 6 | engine integration: `STRATA_ENABLE_VULKAN`, the arena, `gpu_arch_problem` for Intel | serve a model and compare output tokens to the HIP build |

**Where it can be measured, and where it cannot.** Stages 1-2 and the numerics of 3-6 are verifiable on any
Vulkan GPU (this one: RADV, 7900 XTX). Fit, speed and *stability* on Arc need the card.
