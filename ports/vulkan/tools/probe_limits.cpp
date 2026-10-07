// ports/vulkan/tools/probe_limits.cpp - WHAT THE DEVICE ACTUALLY ALLOWS, for the 4 GiB window question.
//
// WHY.  `native_expert_grouped` launches ONE dispatch per 4 GiB window because a storage-buffer binding reaches
// only 4 GiB while the pack's experts live at 1.4-24.8 GiB.  The port's records attribute that cap to the
// toolchain ("glslang 15.1 has no 64-bit buffer index"), but the cap that actually matters is the DRIVER's:
// `maxStorageBufferRange` bounds a descriptor's range whatever the shader's index width is, so if the device
// limit IS ~4 GiB then no index change could ever bind the whole weights region and buffer-device-address is the
// only route.  This prints that limit plus the feature flags the alternative routes rest on.
//
// Build: g++ -std=c++20 -O2 -o /tmp/probe_limits ports/vulkan/tools/probe_limits.cpp -lvulkan
// Run:   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json /tmp/probe_limits
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "strata-probe-limits";
    ai.apiVersion = VK_API_VERSION_1_2;                 // BDA is core in 1.2 (feature flag) / 1.3 (no flag needed)
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) { std::fprintf(stderr, "no instance\n"); return 2; }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(devs[i], &f2);

        const double gib = 1073741824.0;
        std::printf("== %s (api %u.%u.%u, driver %u)\n", p.deviceName, VK_VERSION_MAJOR(p.apiVersion),
                    VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion), p.driverVersion);
        std::printf("   maxStorageBufferRange          : %llu bytes (%.2f GiB)%s\n",
                    (unsigned long long) p.limits.maxStorageBufferRange, p.limits.maxStorageBufferRange / gib,
                    p.limits.maxStorageBufferRange < (1ull << 33) ? "   <-- THIS is the 4 GiB cap" : "");
        std::printf("   maxUniformBufferRange          : %u bytes (%.2f GiB)\n",
                    p.limits.maxUniformBufferRange, p.limits.maxUniformBufferRange / gib);
        std::printf("   maxMemoryAllocationCount       : %u\n", p.limits.maxMemoryAllocationCount);
        std::printf("   shaderInt64                    : %d\n", (int) f2.features.shaderInt64);
        std::printf("   bufferDeviceAddress            : %d   (Vulkan 1.2 feature)\n", (int) f12.bufferDeviceAddress);
        std::printf("   shaderInt8                     : %d\n", (int) f12.shaderInt8);
        std::printf("   descriptorIndexing             : %d\n", (int) f12.descriptorIndexing);
        std::printf("   maxPerStageDescriptorStorageBuffers: %u\n", p.limits.maxPerStageDescriptorStorageBuffers);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
