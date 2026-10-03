// ports/vulkan/harness/vk_compute.cpp - see the header.  No exceptions thrown out of here: every failure
// prints and exits, because a gate that cannot build its device must not look like a gate that passed.
#include "vk_compute.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

namespace portvk {

#define VK_CHECK(x)                                                                        \
    do {                                                                                   \
        const VkResult _r = (x);                                                           \
        if (_r != VK_SUCCESS) {                                                            \
            std::fprintf(stderr, "Vulkan call failed: %s -> %d at %s:%d\n", #x, (int) _r,   \
                         __FILE__, __LINE__);                                              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> out((size_t) n);
    if (n > 0 && !f.read((char*) out.data(), n)) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::exit(1);
    }
    return out;
}

}  // namespace

// The desktop reserve, as a pure function so the policy can be tested without a GPU.  Rules: at least the
// floor (a compositor needs something, so a caller asking for nothing still gets the floor), at most
// `cap_percent_of_heap` of the card (a small card must stay usable for the engine at all), and a request in
// between is taken as given.
ReserveDecision compute_desktop_reserve(uint64_t requested_bytes, uint64_t heap_total_bytes, uint64_t floor_bytes,
                                       uint32_t cap_percent_of_heap) {
    ReserveDecision d{};
    uint64_t r = requested_bytes;
    if (r < floor_bytes) {
        r = floor_bytes;
        d.raised_to_floor = true;
    }
    if (heap_total_bytes) {
        const uint64_t cap = heap_total_bytes / 100ull * (uint64_t) cap_percent_of_heap;
        if (r > cap) {
            r = cap;
            d.clamped_by_cap = true;
        }
    }
    d.reserve_bytes = r;
    return d;
}

static bool device_has_extension(VkPhysicalDevice pd, const char* want) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> eps(n);
    if (n) vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, eps.data());
    for (const auto& e : eps) {
        if (std::strcmp(e.extensionName, want) == 0) return true;
    }
    return false;
}

static void fill_info(DeviceInfo& di, VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    di.name = props.deviceName;
    di.vendor_id = props.vendorID;
    di.device_id = props.deviceID;
    di.api = props.apiVersion;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDevice16BitStorageFeatures f16{};
    f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    f2.pNext = &f16;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    di.storage_buffer_16bit = f16.storageBuffer16BitAccess;
    di.shader_int16 = f2.features.shaderInt16;
    di.shader_float64 = f2.features.shaderFloat64;

    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    di.subgroup_size = sg.subgroupSize;

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) di.heap_device_local_bytes += mp.memoryHeaps[i].size;
    }
}

std::vector<DeviceInfo> Ctx::list_devices() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return {};
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n) vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<DeviceInfo> out;
    for (auto pd : pds) {
        DeviceInfo di{};
        fill_info(di, pd);
        out.push_back(di);
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

Ctx::Ctx(int want_device, bool need_16bit) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "strata-vulkan-port-gate";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, nullptr));
    if (n == 0) {
        std::fprintf(stderr, "no Vulkan physical devices\n");
        std::exit(1);
    }
    std::vector<VkPhysicalDevice> pds(n);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, pds.data()));

    // Pick the device: an explicit index wins; otherwise the first with a compute queue that has the
    // features the shaders need (so a lavapipe run does not silently win over the real GPU).
    int chosen = -1;
    for (uint32_t i = 0; i < n; ++i) {
        DeviceInfo di{};
        fill_info(di, pds[i]);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, qf.data());
        bool has_compute = false;
        for (auto& q : qf) has_compute = has_compute || (q.queueFlags & VK_QUEUE_COMPUTE_BIT);
        const bool ok16 = !need_16bit || (di.storage_buffer_16bit && di.shader_int16);
        if (want_device >= 0) {
            if ((int) i == want_device) {
                chosen = (int) i;
                break;
            }
            continue;
        }
        if (has_compute && ok16) {
            chosen = (int) i;
            break;
        }
    }
    if (chosen < 0 && want_device < 0) {
        std::fprintf(stderr, "no Vulkan device with a compute queue and the required features\n");
        std::exit(1);
    }
    if (chosen < 0) chosen = want_device;
    phys_ = pds[(size_t) chosen];
    device_index_ = chosen;
    fill_info(info_, phys_);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            found = true;
            break;
        }
    }
    if (!found) {
        std::fprintf(stderr, "device %d has no compute queue family\n", chosen);
        std::exit(1);
    }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    // Ask for exactly what the ported kernels use, and only where the device reports it.  Requesting a
    // feature the device lacks fails vkCreateDevice outright - and `shaderFloat16` was requested here for no
    // reason at all (no ported kernel stores fp16), i.e. a pure compatibility risk on hardware that lacks it.
    // `shaderFloat64` is queried but not requested: the one kernel that wanted it (silu) cannot be expressed
    // through glslang's SPIR-V backend, so nothing in the shipping set needs it.
    VkPhysicalDevice16BitStorageFeatures f16{};
    f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    f16.storageBuffer16BitAccess = info_.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    f2.pNext = &f16;
    f2.features.shaderInt16 = info_.shader_int16 ? VK_TRUE : VK_FALSE;

    // VK_EXT_memory_budget adds NO entry points: a capability check plus the name in the enabled list is the
    // whole wiring, and enabling it is what makes the driver report a budget instead of a raw heap size.
    const char* const kBUDGET_EXT = "VK_EXT_memory_budget";
    const bool want_budget_ext = device_has_extension(phys_, kBUDGET_EXT);
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = want_budget_ext ? 1u : 0u;
    dci.ppEnabledExtensionNames = want_budget_ext ? &kBUDGET_EXT : nullptr;
    VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
    vkGetDeviceQueue(dev_, queue_family_, 0, &queue_);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    VK_CHECK(vkCreateCommandPool(dev_, &pci, nullptr, &cmd_pool_));

    // 8 storage-buffer descriptors per set is the widest kernel a first slice needs (4 in gdn_gate); the
    // general harness keeps N flexible by sizing the pool from the widest pipeline it is asked for.
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 256;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &desc_pool_));

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    mem_type_ = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((mp.memoryTypes[i].propertyFlags & want) != want) continue;
        if (mem_type_ == UINT32_MAX) mem_type_ = i;
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            mem_type_ = i;
            break;
        }
    }
    if (mem_type_ == UINT32_MAX) {
        std::fprintf(stderr, "no host-visible coherent memory type\n");
        std::exit(1);
    }
}

Ctx::~Ctx() {
    if (dev_ != VK_NULL_HANDLE) vkDeviceWaitIdle(dev_);
    for (Pipe& pv : pipes_) {
        if (pv.pipe) vkDestroyPipeline(dev_, pv.pipe, nullptr);
        if (pv.layout) vkDestroyPipelineLayout(dev_, pv.layout, nullptr);
        if (pv.set_layout) vkDestroyDescriptorSetLayout(dev_, pv.set_layout, nullptr);
    }
    if (desc_pool_) vkDestroyDescriptorPool(dev_, desc_pool_, nullptr);
    if (cmd_pool_) vkDestroyCommandPool(dev_, cmd_pool_, nullptr);
    if (dev_) vkDestroyDevice(dev_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Ctx::query_budget() {
    budget_.heap_total = info_.heap_device_local_bytes;   // the cap needs a heap size to take a fraction of
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    std::vector<uint32_t> local_heaps;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local_heaps.push_back(i);
    }

    if (!force_no_budget_ext_) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{};
        bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp2{};
        mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp2.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(phys_, &mp2);
        uint64_t b = 0, u = 0;
        for (uint32_t h : local_heaps) {
            b += bp.heapBudget[h];
            u += bp.heapUsage[h];
        }
        budget_.from_driver = true;
        budget_.heap_budget = b;
        budget_.heap_usage = u;
    } else {
        // The FALLBACK is a heap size, which knows nothing about who else is using the card - including the
        // desktop.  It is labelled, never silent: a caller that treats it as free memory will fill the card.
        budget_.from_driver = false;
        std::fprintf(stderr, "vk_compute: VK_EXT_memory_budget unavailable (or disabled) - free memory is a "
                             "LEDGER (heap total minus this process), not a measurement\n");
    }
    if (forced_budget_bytes_) {
        budget_.from_driver = true;
        budget_.heap_budget = forced_budget_bytes_;
        budget_.heap_usage = 0;
    }
}

void Ctx::configure_display_reserve() {
    const char* noext = std::getenv("STRATA_VK_NO_MEMORY_BUDGET");
    force_no_budget_ext_ = noext != nullptr && *noext && std::strcmp(noext, "0") != 0;

    const char* forced = std::getenv("STRATA_VK_FORCE_BUDGET_MIB");
    forced_budget_bytes_ = forced ? (uint64_t) std::strtoull(forced, nullptr, 10) << 20 : 0;

    // 1024 MiB by default: enough for a compositor plus a browser doing GPU compositing at 4K, and NOT sized
    // for a game (the engine's own `--vram-reserve-mib` 700 MiB covers its graphs/scratch/head separately, so
    // the two compose rather than overlapping).
    uint64_t want_bytes = 1024ull << 20;
    const char* rsv = std::getenv("STRATA_VK_DESKTOP_RESERVE_MIB");
    if (rsv && *rsv) want_bytes = (uint64_t) std::strtoull(rsv, nullptr, 10) << 20;

    // 256 MiB floor: a card with nothing to spare still must not be filled to the last byte.  Lowering it to 0
    // is a TEST hook - the numerical gate needs a few MiB of buffers, and on this box the resident local model
    // already holds the card, so a 256 MiB floor would (correctly) refuse even the gate.
    uint64_t floor_bytes = 256ull << 20;
    const char* flr = std::getenv("STRATA_VK_RESERVE_FLOOR_MIB");
    if (flr && *flr) floor_bytes = (uint64_t) std::strtoull(flr, nullptr, 10) << 20;

    query_budget();
    reserve_decision_ = compute_desktop_reserve(want_bytes, budget_.heap_total, floor_bytes, 25);
    reserve_bytes_ = reserve_decision_.reserve_bytes;
    std::fprintf(stderr,
                 "vk_compute: heap total %.2f GiB | free %s %.2f GiB | desktop reserve %.2f GiB%s%s -> %.2f GiB "
                 "usable\n",
                 (double) budget_.heap_total / 1073741824.0, budget_.from_driver ? "(driver)" : "(LEDGER)",
                 (double) (budget_.from_driver ? (budget_.heap_budget - budget_.heap_usage) : budget_.heap_total) /
                     1073741824.0,
                 (double) reserve_bytes_ / 1073741824.0, reserve_decision_.raised_to_floor ? " (floor)" : "",
                 reserve_decision_.clamped_by_cap ? " (clamped to 25% of the card)" : "",
                 (double) usable_bytes() / 1073741824.0);
}

uint64_t Ctx::usable_bytes() const {
    uint64_t free_b;
    if (budget_.from_driver) {
        free_b = budget_.heap_budget > budget_.heap_usage ? budget_.heap_budget - budget_.heap_usage : 0;
    } else {
        free_b = budget_.heap_total > allocated_ ? budget_.heap_total - allocated_ : 0;
    }
    return free_b > reserve_bytes_ ? free_b - reserve_bytes_ : 0;
}

Buf Ctx::alloc(uint64_t bytes) {
    Buf b;
    b.bytes = bytes ? bytes : 4;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = b.bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev_, b.buffer, &req);

    // THE DISPLAY CONTRACT, ENFORCED.  Checked against the driver's own figure (not this layer's ledger) plus
    // the reserve, before anything is allocated.  A backend that cannot fit must refuse and name the numbers -
    // allocating anyway is exactly how the card gets filled and the desktop stops compositing.
    const uint64_t usable = usable_bytes();
    if (allocated_ + req.size > usable) {
        vkDestroyBuffer(dev_, b.buffer, nullptr);
        std::fprintf(stderr,
                     "vk_compute: REFUSING a %.2f MiB allocation - %.2f MiB already held by this process, "
                     "%.2f MiB usable (free %s %.2f GiB, desktop reserve %.2f GiB).  Raise "
                     "STRATA_VK_DESKTOP_RESERVE_MIB only if the desktop can spare it.\n",
                     (double) req.size / 1048576.0, (double) allocated_ / 1048576.0, (double) usable / 1048576.0,
                     budget_.from_driver ? "(driver)" : "(LEDGER)",
                     (double) (budget_.from_driver ? (budget_.heap_budget - budget_.heap_usage) : budget_.heap_total) /
                         1073741824.0,
                     (double) reserve_bytes_ / 1073741824.0);
        std::exit(3);
    }
    allocated_ += req.size;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_type_;
    VK_CHECK(vkAllocateMemory(dev_, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev_, b.buffer, b.mem, 0));
    VK_CHECK(vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    std::memset(b.mapped, 0, (size_t) b.bytes);
    return b;
}

void Ctx::free(Buf& b) {
    // The ledger follows the driver's own allocation size, so the accounting cannot drift from reality.
    if (b.mem != VK_NULL_HANDLE) {
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, b.buffer, &req);
        allocated_ = allocated_ > req.size ? allocated_ - req.size : 0;
    }
    if (b.mapped) vkUnmapMemory(dev_, b.mem);
    if (b.buffer) vkDestroyBuffer(dev_, b.buffer, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buf{};
}

void Ctx::write(Buf& b, const void* src, uint64_t bytes, uint64_t offset) {
    if (offset + bytes > b.bytes) {
        std::fprintf(stderr, "write past end of buffer (%llu+%llu > %llu)\n", (unsigned long long) offset,
                     (unsigned long long) bytes, (unsigned long long) b.bytes);
        std::exit(1);
    }
    std::memcpy((uint8_t*) b.mapped + offset, src, (size_t) bytes);
}

void Ctx::read(const Buf& b, void* dst, uint64_t bytes, uint64_t offset) {
    if (offset + bytes > b.bytes) {
        std::fprintf(stderr, "read past end of buffer\n");
        std::exit(1);
    }
    std::memcpy(dst, (const uint8_t*) b.mapped + offset, (size_t) bytes);
}

VkPipeline Ctx::pipeline(const std::string& spv_path, uint32_t nbufs, uint32_t push_bytes) {
    for (const Pipe& p : pipes_) {
        if (p.spv_path == spv_path && p.nbufs == nbufs && p.push_bytes == push_bytes) return p.pipe;
    }
    Pipe pv{};
    pv.spv_path = spv_path;
    pv.nbufs = nbufs;
    pv.push_bytes = push_bytes;

    std::vector<VkDescriptorSetLayoutBinding> binds(nbufs);
    for (uint32_t i = 0; i < nbufs; ++i) {
        binds[i] = {};
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = nbufs;
    slci.pBindings = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev_, &slci, nullptr, &pv.set_layout));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_bytes;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &pv.set_layout;
    plci.pushConstantRangeCount = push_bytes ? 1u : 0u;
    plci.pPushConstantRanges = push_bytes ? &pcr : nullptr;
    VK_CHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &pv.layout));

    // A descriptor set per pipeline, allocated up front and never reset: the gate runs each kernel a handful
    // of times, so recycling is a stage-3 optimisation, not a correctness one.
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = desc_pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pv.set_layout;
    VK_CHECK(vkAllocateDescriptorSets(dev_, &dsai, &pv.set));

    const std::vector<uint8_t> code = read_file(spv_path);
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode = (const uint32_t*) code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(dev_, &smci, nullptr, &module));

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = pv.layout;
    VK_CHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pv.pipe));
    vkDestroyShaderModule(dev_, module, nullptr);

    pipes_.push_back(pv);
    return pv.pipe;
}

void Ctx::dispatch(VkPipeline pipe, const std::vector<const Buf*>& bufs, const void* push, uint32_t push_bytes,
                   uint32_t groups) {
    // Find the pipeline layout/set that belongs to this pipeline handle.
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (const Pipe& p : pipes_) {
        if (p.pipe == pipe) {
            layout = p.layout;
            set = p.set;
        }
    }
    if (!layout) {
        std::fprintf(stderr, "dispatch: unknown pipeline\n");
        std::exit(1);
    }

    std::vector<VkDescriptorBufferInfo> info(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        info[i] = {};
        info[i].buffer = bufs[i]->buffer;
        info[i].offset = 0;
        info[i].range = VK_WHOLE_SIZE;
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t) i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(dev_, (uint32_t) writes.size(), writes.data(), 0, nullptr);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &cbai, &cb));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (push_bytes) vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cb, groups, 1, 1);
    // Shader writes -> host reads.  Vulkan requires this barrier; without it a coherent mapping may still
    // show the pre-dispatch contents, which would read as "the kernel wrote nothing".
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence));
    VK_CHECK(vkWaitForFences(dev_, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(dev_, fence, nullptr);
    vkFreeCommandBuffers(dev_, cmd_pool_, 1, &cb);
}

}  // namespace portvk
