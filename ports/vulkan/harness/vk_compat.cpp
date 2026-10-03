// ports/vulkan/harness/vk_compat.cpp - see the header.  Every rule below cites the range it comes from, and the
// ranges are in ports/vulkan/KERNEL-COMPAT.md with their sources.
#include "vk_compat.hpp"

#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace portvk {

std::string KernelVersion::str() const {
    char b[64];
    std::snprintf(b, sizeof b, "%d.%d.%d%s", major, minor, patch, prerelease ? " (prerelease)" : "");
    return b;
}

const char* severity_name(Severity s) {
    switch (s) {
        case Severity::Info: return "info";
        case Severity::Note: return "note";
        case Severity::Warn: return "WARN";
        case Severity::Fatal: return "FATAL";
    }
    return "?";
}

KernelVersion parse_kernel_release(const std::string& r) {
    KernelVersion k;
    k.release = r;
    // "7.3.0-rc5", "7.0.0-34-generic", "6.14.8-3-bpo12-pve" - take the first three dot-separated numbers and
    // treat an "rc" anywhere in the remainder as a prerelease.
    int got = std::sscanf(r.c_str(), "%d.%d.%d", &k.major, &k.minor, &k.patch);
    if (got < 2) {
        k.major = k.minor = k.patch = 0;
    } else if (got == 2) {
        k.patch = 0;
    }
    k.prerelease = r.find("-rc") != std::string::npos;
    return k;
}

HostEnv detect_host_env() {
    HostEnv env;
    struct utsname u {};
    if (uname(&u) == 0) env.kernel = parse_kernel_release(u.release);

    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psize = sysconf(_SC_PAGE_SIZE);
    env.host_ram_bytes = (pages > 0 && psize > 0) ? (uint64_t) pages * (uint64_t) psize : 0;

    // Which DRM modules are loaded.  This is a read of /sys/module existence, not a device access: it names the
    // kernel side of the stack (xe vs i915 decides whether Battlemage can be driven at all) without the port ever
    // opening /dev/dri.
    static const char* kModules[] = {"xe", "i915", "amdgpu", "radeon", "nouveau", "nvidia", "virtio_gpu", "vkms"};
    for (const char* m : kModules) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/module/%s", m);
        if (access(path, F_OK) == 0) env.drm_modules.push_back(m);
    }
    return env;
}

void fill_driver_info(VkPhysicalDevice pd, HostEnv& env) {
    VkPhysicalDeviceDriverProperties dp{};
    dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &dp;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    env.driver_name = dp.driverName;
    env.mesa_version = dp.driverInfo;
}

bool parse_mesa_version(const std::string& driver_info, int& major, int& minor, int& patch) {
    major = minor = patch = 0;
    const size_t at = driver_info.find("Mesa ");
    if (at == std::string::npos) return false;
    const int got = std::sscanf(driver_info.c_str() + at + 5, "%d.%d.%d", &major, &minor, &patch);
    if (got < 2) return false;
    if (got == 2) patch = 0;
    return true;
}

bool looks_discrete(uint64_t heap_total_bytes, uint64_t host_ram_bytes) {
    if (heap_total_bytes == 0 || host_ram_bytes == 0) return false;
    // Measured: 24 GiB on a 7900 XTX against 46 GiB of RAM = 52% (discrete); llvmpipe reports a heap equal to
    // system memory (software).  90% separates them with room for a big iGPU carve-out.
    return heap_total_bytes < (host_ram_bytes / 100ull * 90ull);
}

std::vector<Advisory> compat_advisories(const HostEnv& env, uint32_t vendor_id, bool budget_from_driver) {
    std::vector<Advisory> out;
    const bool intel = vendor_id == 0x8086;
    const auto& k = env.kernel;

    if (k.major == 0) {
        out.push_back({Severity::Warn, "could not read the kernel version - the compatibility rules cannot be "
                                       "applied, so treat every kernel-dependent caveat as live"});
        return out;
    }

    bool xe_loaded = false, i915_loaded = false;
    for (const auto& m : env.drm_modules) {
        xe_loaded = xe_loaded || m == "xe";
        i915_loaded = i915_loaded || m == "i915";
    }

    if (intel) {
        // Battlemage (Xe2) needs the xe driver; 6.8 predates BMG support entirely and 6.14 is where it arrives.
        if (k.below(6, 14)) {
            out.push_back({Severity::Fatal,
                           "kernel " + k.str() + " cannot drive Battlemage: the xe driver has no BMG support "
                           "before 6.14 (and 6.8 has none for the generation at all).  Do not downgrade hoping "
                           "for stability."});
        } else if (k.below(7, 0)) {
            out.push_back({Severity::Warn,
                           "kernel " + k.str() + " is in the range with recurring xe compute-load crashes "
                           "(engine_class=ccs resets, reported on 6.14/6.17/6.18).  Keep submissions bounded, "
                           "expect engine resets, and smoke-test before enabling a service."});
        }
        if (k.at_least(7, 1) && k.below(7, 3)) {
            out.push_back({Severity::Info, "kernel " + k.str() + " carries the Xe vRAM memory-pressure work "
                                                               "(7.1), which is the path a full card takes."});
        }
        if (k.at_least(7, 3)) {
            out.push_back({Severity::Note,
                           "kernel " + k.str() + ": TTM eviction became more aggressive in 7.3.  That changes "
                           "what happens when the card is full, so re-verify the desktop reserve holds rather "
                           "than assuming the 7.2 behaviour."});
        }
        if (!xe_loaded && i915_loaded) {
            out.push_back({Severity::Note, "i915 is loaded and xe is not: fine for Alchemist, but Battlemage "
                                           "requires xe - check which module owns the card."});
        }
        bool budget_usable = budget_from_driver;
        int maj = 0, min = 0, pat = 0;
        if (!budget_usable && parse_mesa_version(env.mesa_version, maj, min, pat)) {
            if (maj < 26 || (maj == 26 && min < 2)) {
                out.push_back({Severity::Warn,
                               "Mesa " + std::to_string(maj) + "." + std::to_string(min) +
                                   " predates VK_EXT_memory_budget support on Intel (needs >= 26.2, which added "
                                   "anv memory heap budget tracking).  The driver's free figure is unavailable "
                                   "and the ledger fallback is not safe on a display card."});
            }
        }
    }

    if (k.prerelease) {
        out.push_back({Severity::Note, "kernel " + k.release + " is a prerelease: its behaviour is not the "
                                                          "released behaviour, and the 7.4 cycle already has "
                                                          "Xe changes in flight."});
    }
    if (out.empty()) {
        out.push_back({Severity::Info, "kernel " + k.str() + " and the userspace driver are outside every range "
                                                          "these rules know to be a problem."});
    }
    return out;
}

bool any_fatal(const std::vector<Advisory>& v) {
    for (const auto& a : v) {
        if (a.sev == Severity::Fatal) return true;
    }
    return false;
}

}  // namespace portvk
