// ports/vulkan/harness/vk_compute.hpp - the smallest Vulkan COMPUTE layer a ported GGUF engine needs:
// a device, host-visible buffers, and "load this SPIR-V, bind these N buffers, push these bytes, dispatch
// G groups".  Everything the CUDA backend does with cudaMalloc/cudaMemcpy/<<<>>> has an entry point here.
//
// THE DISPLAY CONTRACT (why the memory code below is not optional):
// The card that runs this is the card that drives the desktop.  On CUDA/HIP the engine learns how much it may
// use from `cudaMemGetInfo`/`hipMemGetInfo` and then holds back `--vram-reserve-mib` (default 700 MiB) for its
// own graphs, scratch and head.  Those numbers do NOT subtract what the desktop and other programs hold - the
// AMD/HIP history records the result: `--expert-cache auto` filled an RX 6800 that drives the desktop and
// decode fell 41 -> 30 tok/s (docs/AMD_HIP.md, #380/#377).  Vulkan's `VK_EXT_memory_budget` reports usage for
// the HEAP, not just this process, so it is the correct source; where it is absent the fallback is a heap
// total, which is a LEDGER, not a measurement, and this layer says so out loud rather than degrading silently.
// On top of the engine's internal reserve this layer holds back a DESKTOP reserve so the compositor keeps
// enough to composite - a desktop's need, not a game's.
#pragma once

#include "vk_compat.hpp"
#include "vk_stack.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::vulkan {

// A buffer in this layer comes in two flavours, and which one it is comes from the driver, not from us:
//
//   * THE GATE'S PATH (alloc()) - host-visible and host-coherent on purpose.  This layer exists to answer "does
//     the ported kernel compute the right numbers", and staging plus fences would only add ways for the GATE to
//     be wrong.  A coherence-first allocation is a correctness device, not a performance one.
//   * THE ENGINE'S PATH (alloc_device() + alloc_staging()) - PORT-PLAN stage 4.  The things that live on the
//     card (weights, KV cache, arenas) go in DEVICE_LOCAL memory that is NOT mappable, and every byte in and out
//     goes through a host-visible STAGING buffer over vkCmdCopyBuffer.  write()/read() do that automatically, so
//     a case written for the first path works unchanged on the second - and the gate exercises the second path
//     rather than describing it.
//
// `mapped` is null exactly when the type is not host-visible, and the gate ASSERTS the recorded flags against
// what the memory type says, so a bug in the selection cannot hide behind a flag we set ourselves.
struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;          // null when the type is not host-visible: write/read then STAGE
    // MAP-ON-DEMAND (STRATA_VK_DIRECT_UPLOAD): the allocation's TYPE is host-visible, but the port holds no
    // persistent mapping for it.  `stage_upload` then maps, stores and unmaps per call, so the host still writes at
    // the mapped type's ~5.64 GB/s while `mapped` stays null - which matters far beyond tidiness: `dispatch`
    // flushes the live batch whenever a bound buffer carries a persistent mapping (a host-visible region the
    // engine may legitimately read the instant the call returns - relaxing THAT rule once cost 5 gate failures and
    // is not on the table).  With every device buffer persistently mapped, that rule fired on every dispatch and
    // turned the prefill's 563 batches into 45,635, which is where a 47% prefill regression came from.  Null here
    // keeps the rule aimed at the buffers that actually need it.
    bool map_on_demand = false;
    uint64_t bytes = 0;
    uint32_t mem_type = UINT32_MAX;
    bool device_local = false;       // DEVICE_LOCAL (real VRAM on a discrete card)
    bool host_visible = false;       // ...and mappable, so no staging is needed
    bool vram_account = false;       // which account alloc() charged it to (see the account rule below)
    // A VIEW OFFSET, in bytes, honoured when this handle is bound as a descriptor.  The engine does not pass whole
    // buffers around: it reaches a row slice by pointer arithmetic (`Y + t0 * ldy`, `X + t0 * K`), so the device
    // layer has to be able to bind "this buffer, from here".  A copy of the handle with `offset` set is that view,
    // and zero (the default) keeps every existing call site binding exactly what it bound before.
    //
    // THE DEVICE'S ALIGNMENT RULE APPLIES: a storage-buffer descriptor offset must be a multiple of
    // `minStorageBufferOffsetAlignment`, or the bind is invalid and the read is undefined.  MEASURED on this box:
    // the Arc reports **4 bytes** (llvmpipe and radeon print their own from `case_descriptor_offset`), which is
    // what makes the engine's `X + t0 * K` slices bindable as they stand - on a device with a 64-byte limit that
    // arithmetic would need padding, and the case prints both numbers rather than assuming either.
    uint64_t offset = 0;
};

// A view of `b` starting `off` bytes in: the engine's row-slice pattern, made explicit.
//
// RELATIVE TO `b`, NOT TO THE ARENA.  This was `v.offset = off` (replacing the base handle's offset), which is
// correct only for a handle whose own offset is 0 - e.g. the single arena buffer.  `native_expert_grouped`
// views its SCRATCH (a bump-allocation with a non-zero arena offset) as `view(b_scr, k*fa)`, so the four
// scratch regions were bound at the ARENA BASE instead of inside the scratch: the gate/up/h/hq stages wrote
// over the arena's first ~8 KiB (the first expert's blob), the q8_1 quantiser - the one stage that uses raw
// pointer arithmetic - wrote to the CALLER's scratch, and the down stage read back a region that was neither.
// Measured 2026-10-05 by the gate's launcher arm: zeroing an expert's whole gate/up half left `out` bitwise
// unchanged, and the scratch read back all-zero after a call.  Adding the base offset is the one-line fix;
// every call site with a zero-offset base is unchanged.
inline Buf view(const Buf& b, uint64_t off) {
    Buf v = b;
    v.offset = b.offset + off;
    return v;
}

// What the device OFFERS, printed rather than assumed, because the three implementations in the gate differ
// completely: the Arc has three non-visible DEVICE_LOCAL types plus a separate system heap (so staging is
// real), llvmpipe has ONE type that is host-visible and device-local at once (so it has no staging path at
// all), and RADV marks its system heap's types device-local.  A single "THERE IS DEVICE-LOCAL MEMORY" bool
// would be wrong on two of the three.
struct MemTypeInfo {
    uint32_t index = 0;
    uint32_t heap = 0;
    uint64_t heap_bytes = 0;
    bool heap_device_local = false;
    bool device_local = false, host_visible = false, host_coherent = false, host_cached = false;
};

// ---- THE ENGINE'S VRAM PLAN: fit accounting against the driver's numbers (PORT-PLAN stage 4) --------------
// The engine's own plan is a list of things it intends to hold ON THE CARD - resident weights, the expert
// cache, the KV cache, graph and scratch arenas - and the question this layer must answer is which line does
// not fit, by name, rather than allocating until the desktop stops compositing.  `droppable` is the engine's
// own notion (an expert cache can be given up to fit; a weight cannot), and it is a property of the ITEM, not
// of the layer - so it is a field, not a heuristic here.
struct PlanItem {
    const char* name = "";
    uint64_t bytes = 0;
    bool droppable = false;
};

struct PlanVerdict {
    bool fits = false;          // every non-droppable item fitted (AN EMPTY PLAN NEVER FITS: no weights, no run)
    int first_overflow = -1;    // index of the item that crossed a non-droppable line, -1 if none
    uint64_t budget = 0;        // what the plan was fitted against
    uint64_t resident = 0;      // bytes of the items that fit
    uint64_t dropped = 0;       // bytes of droppable items the plan had to give up
    int dropped_items = 0;
};

// Pure, so it can be tested without a GPU (same reason compute_desktop_reserve is): walk the plan IN ORDER,
// keep what fits, drop what does not fit and is droppable, and fail on the first item that does not fit and is
// not droppable - stopping there, because the engine cannot start at all.
PlanVerdict plan_fit(const PlanItem* items, size_t n, uint64_t budget_bytes);

// What the driver says about this heap.  `from_driver` distinguishes a real answer (VK_EXT_memory_budget:
// usage is for the whole heap, including the desktop and every other process) from the fallback, which is the
// heap's total size and therefore an over-estimate by however much everyone else is using.
struct MemoryBudget {
    bool from_driver = false;
    bool from_explicit_limit = false;   // STRATA_VK_MAX_BUDGET_MIB - honest about what the number is
    uint64_t heap_budget = 0;   // driver: bytes this process may still allocate across DEVICE_LOCAL heaps
    uint64_t heap_usage = 0;    // driver: bytes already allocated in those heaps
    uint64_t heap_total = 0;    // sum of DEVICE_LOCAL heap sizes (always available)
};

// The desktop reserve, as a pure function so it can be tested without a GPU.
struct ReserveDecision {
    uint64_t reserve_bytes = 0;
    bool clamped_by_cap = false;   // the requested reserve exceeded the fraction cap
    bool raised_to_floor = false;  // the requested reserve was below the floor
};
// The GEMM's shape precondition, as a testable predicate rather than a comment in a shader.
//
// WHY IT EXISTS: the cooperative-matrix kernel derives `tiles_m = m / tile_m` and everything returns when the tile
// count is zero.  An M=1 dispatch (single-token decode) therefore computes NOTHING and leaves the output buffer
// untouched - the caller reads stale or uninitialised memory and no layer reports a problem.  That is the exact
// failure this port refuses everywhere else ("refuse, never degrade"), so the precondition is enforced at the
// boundary instead of documented: every dimension must be a non-zero multiple of the kernel's tile.
//
// `tile_m` is a PARAMETER because the M dimension is the device's, not the port's: 16 on RADV/WMMA and 8 on Intel
// XMX (measured - see DeviceInfo::cm_m).  n and k are 16 in both, but they are checked against the tile constants
// the kernel actually declares rather than against 16 by name.
bool gemm_shape_ok(uint32_t m, uint32_t n, uint32_t k, uint32_t tile_m = 16, uint32_t tile_n = 16, uint32_t tile_k = 16);

// The short-step decode attention's precondition, as a testable predicate for the same reason.
//
// WHY IT EXISTS: attn_decode_short computes one score per key and drops the keys past `width`.  A width of 0 makes
// every score -inf, and the kernel then writes its documented degenerate value rather than a wrong token; a width
// beyond the 256-key window the engine's cache provides would read rows that were never written.  The engine states
// the rule itself - "the device step must satisfy pos+1 == n_kv == width in [1, max_context]" with
// "capacity >= 256" - so the port checks it at the boundary instead of discovering it as a wrong number.
bool attn_short_shape_ok(uint32_t width, uint32_t capacity);

ReserveDecision compute_desktop_reserve(uint64_t requested_bytes, uint64_t heap_total_bytes,
                                       uint64_t floor_bytes, uint32_t cap_percent_of_heap);

struct DeviceInfo {
    std::string name;
    uint32_t vendor_id = 0, device_id = 0;
    uint32_t api = 0;
    bool storage_buffer_16bit = false;   // VK_KHR_16bit_storage storageBuffer16BitAccess
    // VK_KHR_cooperative_matrix - the ONLY route to the matrix units (Intel XMX, AMD WMMA).  Two separate facts,
    // because the second is not implied by the first: on RADV the 14 supported configs are all M16 N16 K16 with
    // subgroup scope and NONE of them takes fp32 operands, so an fp32 GEMM compiles and still cannot run.
    bool cooperative_matrix = false;     // extension + feature present
    bool cm_f16_f32 = false;             // ...and a usable M16 N16 K16 subgroup f16/f16 -> f32 config exists
    // ...AND THE TILE IS A PROPERTY OF THE DEVICE, NOT A CONSTANT.  Measured on this box (2026-10-04) with
    // vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR: **Intel BMG G31's floating-point config is M8 N16 K16**
    // (XMX's op is 8 rows wide), while RADV's list is all M16 N16 K16.  A single "is cooperative matrix usable?"
    // bool therefore gets the Arc wrong - it says no - and the pipeline has to be chosen by shape, which is what
    // these three fields are for.  They hold the SELECTED config (M16 preferred where a device offers both,
    // because that is the kernel this port verified on a Radeon); 0/0/0 means no usable config.
    uint32_t cm_m = 0, cm_n = 0, cm_k = 0;
    bool shader_int16 = false;
    // VK_KHR_8bit_storage storageBuffer8BitAccess: kv_q8 stores its codes as int8 in a storage buffer, so this
    // one is required for that kernel and not merely nice to have (the 16-bit flag covers the scales).
    bool storage_buffer_8bit = false;
    bool shader_float64 = false;
    uint32_t subgroup_size = 0;
    // The alignment a storage-buffer descriptor offset must have.  It is what decides whether the engine's row-slice
    // pointer arithmetic is portable at all: `X + t0 * K` is bindable only if `K * sizeof(elem)` is a multiple of
    // this, so the port checks it instead of discovering it as a wrong number on one driver.
    uint32_t min_storage_offset_align = 0;
    // The DEVICE_LOCAL heap total.  **NOT a model-size budget, and on Intel Arc it is misleading**: Arc
    // reports its dedicated VRAM or the resizable-BAR window here, while the GPU allocates from shared system
    // memory.  Sizing comes from MemoryBudget above; this is here to be printed and to clamp the reserve.
    uint64_t heap_device_local_bytes = 0;
};

// ---- A CAPTURED STEP'S SEGMENTS AND BOUNDARIES (the P6 verify handshake seam) ------------------------------
// A CUDA capture records the WHOLE verify window as one graph; the engine's host loop (Verifier::run) raises the
// handshake flags AFTER the launch, and the window's `wait_flag_ge` spins on them on the device.  This backend
// forbids a spinning kernel, so the recording is CUT at every `wait_flag_ge` into SEGMENTS, and the segment that
// follows boundary j is submitted only once `*(volatile uint32_t*)boundary[j].flag >= boundary[j].value` - the
// poll is on the HOST thread (the engine's own cudaStreamQuery/cudaStreamSynchronize calls drive it) BETWEEN the
// split submissions, exactly the shape `sync.hpp` designed for the doorbell.  A boundary-less capture is one
// segment and one thread of the driver's advance loop, so nothing else changes.
struct CaptureSeg {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t dispatches = 0;
    uint32_t copies = 0;
};
struct CaptureBoundary {
    const void* flag = nullptr;    // HOST address of the uint32 handshake word (mapped, coherent)
    uint32_t value = 0;            // submit the NEXT segment once *flag >= value
};

class Ctx {
public:
    // `want_device` < 0 picks the first device with a compute queue and the required features; otherwise it is
    // an index into the enumerated list (STRATA_VK_DEVICE).  `need_16bit` selects only devices that can run
    // the 16-bit-storage kernels.
    explicit Ctx(int want_device = -1, bool need_16bit = false);
    ~Ctx();
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;

    const DeviceInfo& info() const { return info_; }
    int device_index() const { return device_index_; }
    // Exposed so a test can re-derive the budget with its OWN query instead of asking this class for its own
    // answer back (a test that calls the accessor twice compares the code to itself and passes regardless).
    VkPhysicalDevice physical_device() const { return phys_; }

    static std::vector<DeviceInfo> list_devices();

    // ---- the display contract -------------------------------------------------------------------------
    // The reserve is configured once, after the device is up and before anything is allocated.
    //   STRATA_VK_DESKTOP_RESERVE_MIB  what to hold back for the desktop (default 1024)
    //   STRATA_VK_RESERVE_FLOOR_MIB    the floor under it (default 512)
    //   STRATA_VK_MAX_BUDGET_MIB       an EXPLICIT ceiling on what may be allocated.  Required when the driver
    //                                  will not give a real free figure on a discrete card (see below), and
    //                                  useful as a test hook to pose a card state without filling one.
    //   STRATA_VK_NO_MEMORY_BUDGET     take the labelled ledger-only fallback (the compatibility test uses it)
    void configure_display_reserve();
    const MemoryBudget& budget() const { return budget_; }
    // What this box is, and what that means.  Detected from uname + /sys/module + the device's driver
    // properties: the port opens no device node, so this is the whole of its Linux compatibility surface.
    const HostEnv& host_env() const { return env_; }
    // The rest of the stack - loader, ICDs, libdrm, firmware, session, accelerator runtimes - detected from the
    // filesystem with no device access.  Its advisories are appended to the same list as the kernel's.
    const StackReport& stack() const { return stack_; }
    const std::vector<Advisory>& advisories() const { return advisories_; }
    // True when the free figure is a ledger and the card is discrete, i.e. when sizing from it would be the
    // over-allocation that filled an RX 6800 that drives the desktop.  In that state nothing is allocatable
    // until an explicit STRATA_VK_MAX_BUDGET_MIB is given.
    bool ledger_untrusted() const { return ledger_untrusted_; }
    uint64_t reserve_bytes() const { return reserve_bytes_; }
    // What the engine may still allocate: the driver's free figure (or the ledger fallback) minus the desktop
    // reserve.  Never negative, and zero means "refuse everything" rather than "underflow".
    uint64_t usable_bytes() const;
    uint64_t engine_allocated() const { return allocated_; }
    ReserveDecision reserve_decision() const { return reserve_decision_; }

    Buf alloc(uint64_t bytes);
    // ---- DEVICE-LOCAL MEMORY AND STAGING (PORT-PLAN stage 4) -------------------------------------------
    // `alloc_device` is the path the ENGINE's backend uses for what lives on the card.  It prefers a
    // DEVICE_LOCAL memory type that is NOT host-visible (real VRAM on a discrete card) and falls back to a
    // mappable device-local type only where the device offers nothing else - which is llvmpipe's situation, not
    // a preference.  `buf.host_visible` then says which happened, and write/read STAGE automatically when there
    // is no mapping, so a case does not have to know.
    //
    // THE ACCOUNT RULE, stated once because the refusal below depends on it: an allocation is charged to the
    // VRAM account when it is either (a) alloc()/alloc_device(), whatever heap the driver put it in, or (b) a
    // staging buffer on a device whose only heap is device-local.  Case (a) is CONSERVATIVE on purpose: alloc()
    // is the path every case and the engine's arena take, and its rule must not change with which memory type a
    // driver happens to prefer.  It is exact on the Arc (alloc() lands in the device-local heap) and conservative
    // on the AMD iGPU (it lands in the system heap), and either way it refuses rather than over-commits - which
    // is the whole point of the display contract.
    Buf alloc_device(uint64_t bytes);
    // A HOST allocation: what `cudaHostAlloc`/`cudaMallocHost` hands out.  HOST_VISIBLE | HOST_COHERENT and,
    // where the device has one, a type whose HEAP is NOT device-local - i.e. SYSTEM RAM.  The old rule reused
    // `alloc()`'s DEVICE_LOCAL-preferred type, so a "host" buffer came out of VRAM (the Arc's BAR-mapped type):
    // the engine's PCIe probe then timed a BAR READ of VRAM (0.06 GB/s) instead of a link transfer (1.8 GB/s
    // from system RAM), and every host-tier allocation silently spent device memory.  Falls back to `alloc()`'s
    // type only where no non-local host type exists (llvmpipe: one heap, device-local and mappable).
    Buf alloc_host(uint64_t bytes);
    // A TRANSFER buffer: host-visible and coherent, so it can be mapped, memcpy'd and copied from.  NOT model
    // memory, so it is charged to the host account (unless, as above, the device has nowhere else to put it).
    Buf alloc_staging(uint64_t bytes);
    void free(Buf& b);
    // Every memory type the device offers, with its heap and that heap's size.  Printed by the gate, so the
    // numbers behind the selection above are visible rather than inferred from its consequences.
    const std::vector<MemTypeInfo>& memory_types() const { return mem_types_; }
    const MemTypeInfo& type_of(const Buf& b) const { return mem_types_[b.mem_type]; }
    // True when this device REQUIRES staging: the VRAM type chosen below is one with NO mapping (real VRAM).
    // llvmpipe answers false - its single type is host-visible and device-local at once - and that is a property
    // of the device, reported, not a failure of the port.  The flag is set by the SELECTION rather than derived
    // from "a device-local type exists": on llvmpipe one does, and it is mappable, and a case that could not
    // tell those apart reported a failure the port did not have (measured 2026-10-04).
    bool has_nonvisible_device_local() const { return vram_unmappable_; }
    // Test hook (STRATA_VK_FORCE_STAGING=1): route write/read through staging even where a mapping exists, so
    // the staging path is exercised on EVERY implementation instead of only where a driver forces it.
    void set_force_staging(bool on) { force_staging_ = on; }
    bool force_staging() const { return force_staging_; }
    uint64_t allocated_vram_bytes() const { return allocated_device_local_; }
    uint64_t allocated_host_bytes() const { return allocated_host_; }
    // Host -> device and device -> host.  Through the mapping when there is one (HOST_COHERENT, so no
    // flush/invalidate), otherwise through a staging buffer and vkCmdCopyBuffer.
    void write(Buf& b, const void* src, uint64_t bytes, uint64_t offset = 0);
    void read(const Buf& b, void* dst, uint64_t bytes, uint64_t offset = 0);

    // Compile a compute pipeline from a .spv file with `nbufs` storage-buffer bindings at bindings 0..nbufs-1
    // and a push-constant block of `push_bytes` (0 = none).  Cached per (path, nbufs, push_bytes).
    VkPipeline pipeline(const std::string& spv_path, uint32_t nbufs, uint32_t push_bytes);

    // One dispatch.  `groups` is the x-dimension and `groups_y` the y (`local_size_x` comes from the shader's
    // own layout()).  The y dimension exists for ONE family of kernels: the grouped expert pair strides over
    // groups in y because the group COUNT lives on the device (`for (g = blockIdx.y; g < ng; g += gridDim.y)`),
    // so the launch cannot be sized to it - see native_gu_iq2s.comp.  Every other kernel here keeps y at 1.
    void dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                  uint32_t groups, uint32_t groups_y = 1);

    // Device-side completion of everything dispatched so far on this Ctx.  `dispatch()` BATCHES (see the live
    // batch below): it returns after ENCODING, so a caller that needs the work DONE - the doorbell handoff's
    // publish, which `sync.cpp` rests on - calls this.  `read`/`write`/transfers flush implicitly, so this is for
    // the explicit cases, and it costs nothing when nothing is pending.
    void flush();

    // ---- EVENT TIMESTAMPS: real device time for cudaEventRecord / cudaEventElapsedTime ---------------------
    // WHY THIS EXISTS.  The engine's phase table (`STRATA_PREFILL_TIMING=1`) is EVENT-based - see
    // `src/prefill/prefill.cpp:1458`: "Events are recorded on the compute stream in order; the time between two
    // consecutive marks is charged to the phase of the first".  That is device time ONLY if the marks are, and
    // this port's `cudaEvent*` was `steady_clock` - so every phase number it reported was HOST time: the time
    // this (blocking, batching) backend spent inside the phase, which is why its `gdn recurrence` read 225x the
    // SYCL tree's device-time figure for the same work.  These four calls are the honest form: a timestamp
    // written INTO THE LIVE BATCH at the point of the mark, read back after the device has executed it.
    // `ts_read` uses VK_QUERY_RESULT_WITH_AVAILABILITY_BIT and NEVER a WAIT_BIT - a waiting read over this
    // device's doorbell path is a measured DEVICE LOST (r=-4), not a slow read.
    uint32_t ts_alloc();                              // a slot in the event pool, or UINT32_MAX on exhaustion
    void     ts_mark(uint32_t slot);                  // write a timestamp into the live batch AT THIS POINT
    bool     ts_read(uint32_t slot, uint64_t* ticks); // availability-bit read; false = not executed yet
    double   ts_period_ns() const;                    // ns per tick for this device (0 = instrument off)

    // ---- RECORDED STEPS: the CUDA-graph replacement (see NEXT.md's stage-3 note) ------------------------
    // The engine's decode step is a fixed sequence of dispatches re-issued every token, and the live path's
    // batch above is flushed at every observer, so it cannot re-issue a sequence without re-encoding it.  These
    // calls record a sequence into ONE persistent command buffer and then RE-SUBMIT it, which is what a graph
    // was buying.  `record_dispatch` inserts a compute -> compute barrier (in a step one kernel's output is the
    // next one's input); the host-read barrier goes once, at the end.  `replay_recorded()` must not re-record -
    // that is the property a case has to prove.
    void record_begin();
    void record_dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                         uint32_t groups, uint32_t groups_y = 1);
    void record_end();               // close the recording (the same closing host-read barrier); NO submit
    void record_end_and_submit();
    void replay_recorded();
    uint32_t recorded_dispatches() const { return recorded_; }
    bool has_recording() const { return have_recording_; }

    // ---- STREAM CAPTURE: the CUDA-graph API's recording, built ON the recorded step above ------------------
    // A "graph" on this backend IS a recorded step: one persistent command buffer holding the dispatches a
    // capture body issued, re-submitted by every launch and never re-recorded.  Capture adds NO second
    // recording mechanism - it reuses `encode_dispatch` (a chain barrier between dispatches, a fresh descriptor
    // set per dispatch) and the same submit path as `record_dispatch`/`submit_recorded`.  What is new is only
    // the DIVERSION: while a capture is active, `dispatch()` RECORDS instead of submitting and waiting, which
    // is exactly what CUDA's stream capture does to a kernel launch.
    //
    // A command buffer can only hold pure device commands, so a capture can record the two this layer has:
    // dispatches and device->device buffer copies (vkCmdCopyBuffer).  An op the port CANNOT record as a device
    // command - a host<->device transfer (this layer's copies STAGE through host memory) or a memset (it has no
    // memset shader) - makes the capture INVALID rather than silently executing inside a capture; the caller
    // must then refuse (cudaErrorStreamCaptureUnsupported).  See `cuda_compat/cuda_runtime.h`.
    void capture_begin();
    void capture_invalidate() { capture_valid_ = false; }
    bool capturing() const { return capture_; }
    bool capture_valid() const { return capture_valid_; }
    // Record a device->device byte copy into the active capture: one vkCmdCopyBuffer region + a transfer->compute
    // barrier, so the next dispatch reads what it wrote.
    void capture_copy(const Buf& dst, const Buf& src, uint64_t bytes);
    // A HOST BOUNDARY inside the capture (deliverable A of the P6 verify seam).  The engine's verify window is a
    // captured step whose `wait_flag_ge` waits for a HOST-raised handshake word; this backend forbids a spinning
    // kernel, so the recording is CUT here and the driver submits the SEGMENT that follows only once
    // `*(volatile uint32_t*)flag_host >= value`.  `flag_host` is a mapped host address (the shim's
    // cudaHostGetDevicePointer returns the host pointer), so the poll is a plain host read on the host thread
    // BETWEEN SPLIT SUBMISSIONS - no kernel ever waits.
    void capture_boundary(const void* flag_host, uint32_t value);
    // Close the capture WITHOUT submitting.  False when the capture was invalidated or recorded nothing.
    bool capture_end();
    // Hand the closed recording's SEGMENTS (one per boundary, so a boundary-less capture is a one-element list)
    // and its BOUNDARIES to the caller (the shim's cudaGraph_t).
    void take_recording(std::vector<CaptureSeg>& segs, std::vector<CaptureBoundary>& bounds);
    // Abandon an in-progress/closed recording (an invalidated capture).  Frees its command buffers and fences.
    void discard_recording();
    // Submit + wait ONE owned segment (the SAME submission path `submit_recorded` uses), and destroy one.
    void submit_segment(const CaptureSeg& seg);
    void destroy_owned(const CaptureSeg& seg);
    // One destroyed recording: decrements the leak-test counter (there is one recording per graph, however many
    // segments it holds).
    void release_recording();
    // How many recordings the caller (the shim's graphs) currently owns: incremented when a recording is taken,
    // decremented when one is destroyed.  The instrument a leak test needs - a leaked instantiation shows up
    // here even though it never touches the arena.
    uint32_t owned_recordings() const { return owned_recordings_; }
    // IS WORK ISSUED AND NOT YET COMPLETE?  `dispatch` ENCODES into the live batch and returns; the batch is
    // submitted and fenced only at a flush.  So between a dispatch and the next flush the stream holds work that
    // has not run, and `cudaStreamQuery` MUST NOT answer "complete" - that is the answer this accessor lets the
    // shim refuse.  Only the live arm can be pending this way; a recording is not in the stream until submitted.
    bool live_pending() const { return live_open_; }

private:
    // One pipeline and everything that must be created and destroyed with it.  A key list parallel to a value
    // list is two containers that have to stay the same length by hand; one struct cannot drift.
    struct Pipe {
        std::string spv_path;
        uint32_t nbufs = 0, push_bytes = 0;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    DeviceInfo info_{};
    MemoryBudget budget_{};
    ReserveDecision reserve_decision_{};
    uint64_t reserve_bytes_ = 0;
    uint64_t allocated_ = 0;      // this layer's own ledger, reported alongside the driver's number
    bool force_no_budget_ext_ = false;
    uint64_t forced_budget_bytes_ = 0;   // 0 = off (STRATA_VK_MAX_BUDGET_MIB)
    bool ledger_untrusted_ = false;      // ledger fallback + discrete card + no explicit limit
    HostEnv env_{};
    StackReport stack_{};
    std::vector<Advisory> advisories_{};

    int device_index_ = -1;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;      // the CURRENT pool; the set below grows when it fills
    std::vector<VkDescriptorPool> desc_pools_;         // every pool created, so all of them are destroyed
    uint32_t mem_type_ = 0;
    // The two types stage 4 selects, chosen once at device creation (see the constructor): real VRAM
    // (DEVICE_LOCAL, no mapping) where the device has one, and a mappable host-visible type for transfers,
    // preferring a heap that is NOT device-local so staging costs system memory rather than VRAM.
    uint32_t vram_type_ = UINT32_MAX;
    bool vram_unmappable_ = false;   // the chosen VRAM type has NO mapping: cheap to know here, invisible later
    uint32_t staging_type_ = UINT32_MAX;
    // The HOST allocation type: HOST_VISIBLE | HOST_COHERENT in a heap that is NOT device-local where the device
    // offers one (system RAM), else `mem_type_` (llvmpipe).  Used by `alloc_host`/`cudaHostAlloc` only - kernel
    // scratch and the engine arena keep `alloc`/`alloc_device`, so this cannot move a device-side binding.
    uint32_t host_type_ = UINT32_MAX;
    std::vector<MemTypeInfo> mem_types_;
    uint64_t allocated_device_local_ = 0;   // the VRAM account (see the account rule in the public block)
    uint64_t allocated_host_ = 0;           // the host account: staging/transfer allocations
    bool force_staging_ = false;            // STRATA_VK_FORCE_STAGING - exercises the staging path anywhere
    std::vector<Pipe> pipes_;
    // STRATA_VK_TRIVIAL_REC (deliverable 2): the trivial `scale` pipeline and its 4 KiB scratch, created lazily the
    // first time a recorded dispatch is substituted.  MEASUREMENT-ONLY (see g_trivial_rec); always VK_NULL_HANDLE
    // on the shipped path, so nothing here can move a default run.
    VkPipeline trivial_pipe_ = VK_NULL_HANDLE;
    Buf trivial_buf_{};

    // One allocation, shared by all three entry points above so the refusal, the ledger and the printed
    // message cannot drift between them.  `vram_account` decides which account it is charged to.
    Buf alloc_impl(uint64_t bytes, uint32_t type_index, bool vram_account, const char* what);

    // A descriptor set for `layout`, out of a pool that GROWS.  One set is allocated per pipeline (and per
    // dispatch inside a recorded step) and none is recycled, so a fixed `maxSets` is a silent ceiling on how
    // many cases the gate can hold - and it appears as VK_ERROR_OUT_OF_POOL_MEMORY at the END of a long run on
    // whichever implementation has the most to do, which reads like a kernel failure and is not one.  Measured:
    // RADV hit it after four new cases took the gate past 64 sets, while the same binary passed on the Arc.
    VkDescriptorSet set_alloc(VkDescriptorSetLayout layout);
    VkDescriptorPool new_desc_pool();
    // Refuses an unaligned descriptor offset with a message naming the device's limit (see the definition).
    void check_offsets(const std::vector<const Buf*>& bufs) const;

    // ---- staging transfers: a one-shot command buffer and a fence, per transfer -------------------------
    // A backend pools these (stage 5's problem, not a correctness one) - the same call the descriptor sets and
    // the recorded step already made.  The barriers are stated in each direction, because they are the part of
    // a transfer that a gate can get wrong and never see: a copy without a following barrier leaves the
    // device-local buffer's contents unavailable to the next shader that reads it.
    VkCommandBuffer begin_oneshot();
    void end_oneshot_and_wait(VkCommandBuffer cb);
    void stage_upload(Buf& dst, const void* src, uint64_t bytes, uint64_t offset);
    void stage_download(const Buf& src, void* dst, uint64_t bytes, uint64_t offset);

    // ---- THE LIVE DISPATCH BATCH: the prompt path's submit model ----------------------------------------------
    // Every live (non-capture) dispatch used to be its OWN command buffer, its OWN fence, its OWN submit AND its
    // OWN wait.  Measured on the Arc Pro B70 at 249,878 of each for 233,768 dispatches, with the fence `wait`
    // 88.7% of the dispatch layer (54,674 of 61,662 ms) - i.e. the GPU idled while the host returned from the
    // fence, and every phase inflated together.  Now dispatches accumulate into ONE command buffer and are
    // submitted ONCE, so the round trip is paid per BATCH, not per dispatch.  The batch is flushed before
    // anything that must observe the device (a host read, a host write, a transfer, a capture begin, a recorded
    // step's submit, teardown) and when it reaches `kLiveBatchMax` dispatches, which bounds the work in one
    // submission well under the Battlemage GuC preemption timeout (the batch's own kernels are ~us each).
    // One FRESH descriptor set per encoded dispatch is REQUIRED (execution is deferred to submit, so a shared set
    // would leave every dispatch in the batch reading the last binding - the trap the recorded step already
    // documents); the sets come from `live_pool_`, which is RESET at each flush so they are recycled rather than
    // grown (the pool churn the gate prints is the recording's, not this path's).
    static constexpr uint32_t kLiveBatchMax = 128;
    VkCommandBuffer live_cb_ = VK_NULL_HANDLE;
    VkFence live_fence_ = VK_NULL_HANDLE;
    VkDescriptorPool live_pool_ = VK_NULL_HANDLE;
    uint32_t live_n_ = 0;
    bool live_open_ = false;
    void flush_live();
    void live_ensure();
    VkDescriptorSet set_alloc_in(VkDescriptorPool pool, VkDescriptorSetLayout layout);

    // Shared encoding half of a dispatch: the pipes_ lookup, the descriptor update, the binds, the push constants
    // and vkCmdDispatch.  `chain_barrier` adds a compute -> compute barrier, which a recorded STEP needs between
    // its dispatches (one kernel's output is the next one's input); the single-shot path passes false and does
    // its own host barrier afterwards.  `fresh_set` allocates a descriptor set for THIS dispatch - required for a
    // recorded step, where every dispatch must keep its own bindings (the host updates happen at record time, the
    // dispatches run at submit time, so one shared set would leave them all reading the last binding).
    void encode_dispatch(VkCommandBuffer cb, VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push,
                         uint32_t push_bytes, uint32_t groups, uint32_t groups_y, bool chain_barrier,
                         bool fresh_set, VkDescriptorSet forced_set);
    void submit_recorded();   // submit the recorded buffer and wait: used by the first submit AND by every replay

    VkCommandBuffer rec_cb_ = VK_NULL_HANDLE;
    VkFence rec_fence_ = VK_NULL_HANDLE;
    uint32_t recorded_ = 0;         // dispatches in the current/last recording
    uint32_t recorded_copies_ = 0;  // device->device copies in the current/last recording
    bool recording_ = false;        // between record_begin() and record_end_and_submit()
    bool have_recording_ = false;   // a finished recording exists and may be replayed
    bool capture_ = false;          // a stream capture is active: dispatch() records instead of submitting
    bool capture_valid_ = false;    // ...and nothing has invalidated it (see capture_invalidate)
    // The capture's FINISHED segments and the host boundaries between them (see CaptureSeg/CaptureBoundary).
    std::vector<CaptureSeg> cap_segs_;
    std::vector<CaptureBoundary> cap_bounds_;
    uint32_t owned_recordings_ = 0; // recordings handed to the caller that have not been destroyed

    void query_budget();   // called after device creation, so the extension can be enabled
};

}  // namespace strata::vulkan
