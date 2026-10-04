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

namespace portvk {

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
    uint64_t bytes = 0;
    uint32_t mem_type = UINT32_MAX;
    bool device_local = false;       // DEVICE_LOCAL (real VRAM on a discrete card)
    bool host_visible = false;       // ...and mappable, so no staging is needed
    bool vram_account = false;       // which account alloc() charged it to (see the account rule below)
};

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
// WHY IT EXISTS: the cooperative-matrix kernel derives `tiles_m = m / 16` and everything returns when the tile
// count is zero.  An M=1 dispatch (single-token decode) therefore computes NOTHING and leaves the output buffer
// untouched - the caller reads stale or uninitialised memory and no layer reports a problem.  That is the exact
// failure this port refuses everywhere else ("refuse, never degrade"), so the precondition is enforced at the
// boundary instead of documented: all three dimensions must be non-zero multiples of the 16x16 tile.
bool gemm_shape_ok(uint32_t m, uint32_t n, uint32_t k);

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
    bool cm_f16_f32 = false;             // ...and a usable config exists: M16 N16 K16 subgroup, f16/f16 -> f32
    bool shader_int16 = false;
    // VK_KHR_8bit_storage storageBuffer8BitAccess: kv_q8 stores its codes as int8 in a storage buffer, so this
    // one is required for that kernel and not merely nice to have (the 16-bit flag covers the scales).
    bool storage_buffer_8bit = false;
    bool shader_float64 = false;
    uint32_t subgroup_size = 0;
    // The DEVICE_LOCAL heap total.  **NOT a model-size budget, and on Intel Arc it is misleading**: Arc
    // reports its dedicated VRAM or the resizable-BAR window here, while the GPU allocates from shared system
    // memory.  Sizing comes from MemoryBudget above; this is here to be printed and to clamp the reserve.
    uint64_t heap_device_local_bytes = 0;
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

    // ---- RECORDED STEPS: the CUDA-graph replacement (see NEXT.md's stage-3 note) ------------------------
    // The engine's decode step is a fixed sequence of dispatches re-issued every token, and `dispatch()` above
    // submits and waits per call, so it cannot express that.  These calls record a sequence into ONE persistent
    // command buffer and then RE-SUBMIT it, which is what a graph was buying.  `record_dispatch` inserts a
    // compute -> compute barrier (in a step one kernel's output is the next one's input); the host-read barrier
    // goes once, at the end.  `replay_recorded()` must not re-record - that is the property a case has to prove.
    void record_begin();
    void record_dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                         uint32_t groups, uint32_t groups_y = 1);
    void record_end_and_submit();
    void replay_recorded();
    uint32_t recorded_dispatches() const { return recorded_; }
    bool has_recording() const { return have_recording_; }

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
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    uint32_t mem_type_ = 0;
    // The two types stage 4 selects, chosen once at device creation (see the constructor): real VRAM
    // (DEVICE_LOCAL, no mapping) where the device has one, and a mappable host-visible type for transfers,
    // preferring a heap that is NOT device-local so staging costs system memory rather than VRAM.
    uint32_t vram_type_ = UINT32_MAX;
    bool vram_unmappable_ = false;   // the chosen VRAM type has NO mapping: cheap to know here, invisible later
    uint32_t staging_type_ = UINT32_MAX;
    std::vector<MemTypeInfo> mem_types_;
    uint64_t allocated_device_local_ = 0;   // the VRAM account (see the account rule in the public block)
    uint64_t allocated_host_ = 0;           // the host account: staging/transfer allocations
    bool force_staging_ = false;            // STRATA_VK_FORCE_STAGING - exercises the staging path anywhere
    std::vector<Pipe> pipes_;

    // One allocation, shared by all three entry points above so the refusal, the ledger and the printed
    // message cannot drift between them.  `vram_account` decides which account it is charged to.
    Buf alloc_impl(uint64_t bytes, uint32_t type_index, bool vram_account, const char* what);

    // ---- staging transfers: a one-shot command buffer and a fence, per transfer -------------------------
    // A backend pools these (stage 5's problem, not a correctness one) - the same call the descriptor sets and
    // the recorded step already made.  The barriers are stated in each direction, because they are the part of
    // a transfer that a gate can get wrong and never see: a copy without a following barrier leaves the
    // device-local buffer's contents unavailable to the next shader that reads it.
    VkCommandBuffer begin_oneshot();
    void end_oneshot_and_wait(VkCommandBuffer cb);
    void stage_upload(Buf& dst, const void* src, uint64_t bytes, uint64_t offset);
    void stage_download(const Buf& src, void* dst, uint64_t bytes, uint64_t offset);

    // Shared encoding half of a dispatch: the pipes_ lookup, the descriptor update, the binds, the push constants
    // and vkCmdDispatch.  `chain_barrier` adds a compute -> compute barrier, which a recorded STEP needs between
    // its dispatches (one kernel's output is the next one's input); the single-shot path passes false and does
    // its own host barrier afterwards.  `fresh_set` allocates a descriptor set for THIS dispatch - required for a
    // recorded step, where every dispatch must keep its own bindings (the host updates happen at record time, the
    // dispatches run at submit time, so one shared set would leave them all reading the last binding).
    void encode_dispatch(VkCommandBuffer cb, VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push,
                         uint32_t push_bytes, uint32_t groups, uint32_t groups_y, bool chain_barrier,
                         bool fresh_set);
    void submit_recorded();   // submit the recorded buffer and wait: used by the first submit AND by every replay

    VkCommandBuffer rec_cb_ = VK_NULL_HANDLE;
    VkFence rec_fence_ = VK_NULL_HANDLE;
    uint32_t recorded_ = 0;         // dispatches in the current/last recording
    bool recording_ = false;        // between record_begin() and record_end_and_submit()
    bool have_recording_ = false;   // a finished recording exists and may be replayed

    void query_budget();   // called after device creation, so the extension can be enabled
};

}  // namespace portvk
