// ports/vulkan/harness/vk_compute.hpp - the smallest Vulkan COMPUTE layer a ported GGUF engine needs:
// a device, host-visible buffers, and "load this SPIR-V, bind these N buffers, push these bytes, dispatch
// G groups".  Everything the CUDA backend does with cudaMalloc/cudaMemcpy/<<<>>> has an entry point here.
//
// Deliberately NOT here yet (and listed in ports/vulkan/PORT-PLAN.md): device-local staging, command-buffer
// RECORDING (the CUDA-graph replacement), timeline semaphores, VK_EXT_memory_budget, and the 8-bit-storage
// descriptor set.  Those are stages 3+ of the plan; this is stage 1 and it has to be exactly right first.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace portvk {

// One buffer, host-visible and host-coherent on purpose: this layer exists to answer "does the ported
// kernel compute the right numbers", and staging plus fences would only add ways for the GATE to be wrong.
// The engine's own backend will use device-local memory + staging (PORT-PLAN.md, stage 4).
struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    uint64_t bytes = 0;
};

struct DeviceInfo {
    std::string name;
    uint32_t vendor_id = 0, device_id = 0;
    uint32_t api = 0;
    bool storage_buffer_16bit = false;   // VK_KHR_16bit_storage storageBuffer16BitAccess
    bool shader_int16 = false;
    bool shader_int64 = false;
    bool shader_float64 = false;
    uint32_t subgroup_size = 0;
    uint64_t heap_device_local_bytes = 0;
};

class Ctx {
public:
    // `want_device` < 0 picks the first device that has a compute queue; otherwise it is an index into the
    // enumerated list (STRATA_VK_DEVICE).  Enumerating without picking is a supported use (list_devices).
    explicit Ctx(int want_device = -1, bool need_16bit = false);
    ~Ctx();
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;

    const DeviceInfo& info() const { return info_; }
    int device_index() const { return device_index_; }

    static std::vector<DeviceInfo> list_devices();

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

    const char* last_error() const { return err_.c_str(); }

private:
    struct PipeKey {
        std::string path;
        uint32_t nbufs = 0, push = 0;
        bool operator==(const PipeKey& o) const { return path == o.path && nbufs == o.nbufs && push == o.push; }
    };
    struct PipeVal {
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    DeviceInfo info_{};
    int device_index_ = -1;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    uint32_t mem_type_ = 0;
    std::vector<PipeKey> pkeys_;
    std::vector<PipeVal> pvals_;
    std::string err_;

    void die(const std::string& what);
};

}  // namespace portvk
