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

// One buffer, host-visible and host-coherent on purpose: this layer exists to answer "does the ported kernel
// compute the right numbers", and staging plus fences would only add ways for the GATE to be wrong.  The
// engine's own backend must use device-local memory + staging (PORT-PLAN.md stage 4) - a coherence-first
// allocation is a correctness device, not a performance one.
struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    uint64_t bytes = 0;
};

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
    void free(Buf& b);
    // Host -> device and device -> host over the mapping.  HOST_COHERENT, so no flush/invalidate calls and no
    // staging: correct for a gate, wrong for a benchmark (which is why the engine's backend will not do this).
    void write(Buf& b, const void* src, uint64_t bytes, uint64_t offset = 0);
    void read(const Buf& b, void* dst, uint64_t bytes, uint64_t offset = 0);

    // Compile a compute pipeline from a .spv file with `nbufs` storage-buffer bindings at bindings 0..nbufs-1
    // and a push-constant block of `push_bytes` (0 = none).  Cached per (path, nbufs, push_bytes).
    VkPipeline pipeline(const std::string& spv_path, uint32_t nbufs, uint32_t push_bytes);

    // One dispatch.  `groups` is the x-dimension; the y/z dims are 1 and `local_size_x` comes from the
    // shader's own layout() (the engine's kernels are all 1-D grids).  Submits, waits, and leaves the
    // results visible to the host.
    void dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                  uint32_t groups);

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
    std::vector<Pipe> pipes_;

    void query_budget();   // called after device creation, so the extension can be enabled
};

}  // namespace portvk
