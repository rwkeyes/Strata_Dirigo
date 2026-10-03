// ports/vulkan/harness/vk_compat.cpp - see the header.  Every rule below cites the range it comes from, and the
// ranges are in ports/vulkan/STACK-COMPAT.md with their sources.
#include "vk_compat.hpp"

#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace portvk {
namespace {

// Intel's KMD support tables, read 2026-10-03.  "Initial support" is the first kernel where the GPU is available
// EXPERIMENTALLY and may need force_probe=PCI_ID; "full support" is where it is enabled by default and validated.
// `floor_*` is the line this port refuses below - normally Intel's initial-support release, except for the Arc Pro
// B70/B65 where Intel lists no initial release and the floor comes from our own field notes (6.14), with Intel's
// 6.17 recorded in the advisory.
struct IntelDevice {
    uint32_t id;
    const char* name;
    IntelGen gen;
    int floor_maj, floor_min;
    int initial_maj, initial_min;     // 0,0 = Intel lists no initial-support release
    int full_maj, full_min;
};

const IntelDevice kIntelDevices[] = {
    // Alchemist / Xe-HPG, driven by i915.  Desktop and Pro parts: initial 6.0, full 6.2.
    {0x56a0, "Arc A770",        IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    {0x56a1, "Arc A750",        IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    {0x56a2, "Arc A580",        IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    {0x56a5, "Arc A380",        IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    {0x56a6, "Arc A310",        IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    {0x56b1, "Arc Pro A40/A50", IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    {0x56b3, "Arc Pro A60",     IntelGen::Alchemist,  6, 0,  6, 0,  6, 2},
    // The mobile Alchemist parts start one release earlier.
    {0x5690, "Arc A770M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x5691, "Arc A730M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x5692, "Arc A550M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x5693, "Arc A370M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x5694, "Arc A350M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x5696, "Arc A570M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x5697, "Arc A530M",       IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x56b0, "Arc Pro A30M",    IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    {0x56b2, "Arc Pro A60M",    IntelGen::Alchemist,  5, 19, 5, 19, 6, 2},
    // Battlemage / Xe2, driven by xe.
    {0xe20b, "Arc B580",        IntelGen::Battlemage, 6, 11, 6, 11, 6, 12},
    {0xe20c, "Arc B570",        IntelGen::Battlemage, 6, 11, 6, 11, 6, 12},
    {0xe212, "Arc Pro B50",     IntelGen::Battlemage, 6, 11, 6, 11, 6, 14},
    {0xe211, "Arc Pro B60",     IntelGen::Battlemage, 6, 15, 0, 0,  6, 15},
    {0xe222, "Arc Pro B65",     IntelGen::Battlemage, 6, 14, 0, 0,  6, 17},
    {0xe223, "Arc Pro B70",     IntelGen::Battlemage, 6, 14, 0, 0,  6, 17},
};

const IntelDevice* find_intel_device(uint32_t id) {
    for (const IntelDevice& d : kIntelDevices) {
        if (d.id == id) return &d;
    }
    return nullptr;
}

std::string dotted(int maj, int min) { return std::to_string(maj) + "." + std::to_string(min); }

std::string hex4(uint32_t id) {
    char b[8];
    std::snprintf(b, sizeof b, "%04x", id);
    return b;
}

std::string support_str(const IntelDevice* d) {
    const std::string ini = d->initial_maj ? dotted(d->initial_maj, d->initial_min) : std::string("none listed");
    return ini + ", full support " + dotted(d->full_maj, d->full_min);
}

}  // namespace

IntelGen intel_generation(uint32_t device_id) {
    const IntelDevice* d = find_intel_device(device_id);
    return d ? d->gen : IntelGen::Unknown;
}

const char* intel_device_name(uint32_t device_id) {
    const IntelDevice* d = find_intel_device(device_id);
    return d ? d->name : nullptr;
}


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

std::vector<Advisory> compat_advisories(const HostEnv& env, uint32_t vendor_id, uint32_t device_id,
                                        bool budget_from_driver) {
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
        // PER-GENERATION rules, from Intel's own KMD support tables.  Intel's model: "initial support" is the
        // first kernel where the GPU is available experimentally and may need force_probe=PCI_ID; "full support"
        // is where it is enabled by default and validated.  So the severity rule is: Fatal below the initial
        // release (the kernel predates the device), Warn inside the initial band, silent once full support is
        // reached.
        //
        // The FIRST version of this rule applied the Battlemage full-support line (6.12) to EVERY Intel device.
        // That would have refused a working Arc A770 on kernel 6.6 - a common LTS - because Alchemist has been
        // fully supported since 6.2.  The floor is per generation, not per vendor.
        const IntelDevice* dev = find_intel_device(device_id);
        const IntelGen gen = intel_generation(device_id);
        // Alchemist's driver story differs: i915 is the driver it belongs on, and Intel's Xe support table lists
        // no DG2/Alchemist part at all (it begins at Lunar Lake and Battlemage).
        const bool on_xe = (gen == IntelGen::Battlemage) || xe_loaded;
        const std::string who = intel_device_name(device_id) ? std::string(intel_device_name(device_id))
                                                             : ("device 0x" + hex4(device_id));

        if (!dev) {
            out.push_back({Severity::Note,
                           "Intel " + who + " is not in the generation table (Alchemist/DG2, Battlemage/Xe2), so "
                           "no kernel floor is claimed for it.  Check Intel's KMD support tables before pinning a "
                           "kernel; the generation-independent Intel rules below still apply."});
        } else if (k.below(dev->floor_maj, dev->floor_min)) {
            out.push_back({Severity::Fatal,
                           "kernel " + k.str() + " predates the device: " + who + " needs kernel " +
                               dotted(dev->floor_maj, dev->floor_min) + " or newer (Intel's table: initial "
                               "support " + support_str(dev) + ").  Do not downgrade hoping for stability."});
        } else if (k.below(dev->full_maj, dev->full_min)) {
            if (dev->initial_maj) {
                out.push_back({Severity::Warn,
                               "kernel " + k.str() + " is in the initial-support band for " + who + " (Intel's "
                               "table: initial support " + support_str(dev) + ").  The device is available "
                               "experimentally, may need force_probe=0x" + hex4(device_id) + ", and is not fully "
                               "validated - smoke-test before enabling a service."});
            } else {
                out.push_back({Severity::Warn,
                               "kernel " + k.str() + " runs " + who + " but is not vendor-validated for it: "
                               "Intel's table lists no initial-support release and full support at " +
                               dotted(dev->full_maj, dev->full_min) + ".  The floor this port uses (" +
                               dotted(dev->floor_maj, dev->floor_min) + ") comes from our own field notes."});
            }
        }

        // Xe2 only.  The CCS engine-reset reports are Battlemage's; raising them for Alchemist would be a false
        // alarm, which the earlier rule did.
        if (gen == IntelGen::Battlemage && dev && k.at_least(dev->floor_maj, dev->floor_min) && k.below(7, 0)) {
            out.push_back({Severity::Warn,
                           "kernel " + k.str() + " is in the range with recurring xe compute-load crashes "
                           "(engine_class=ccs resets, reported on 6.14/6.17/6.18).  Keep submissions bounded, "
                           "expect engine resets, and smoke-test before enabling a service."});
        }
        if (gen == IntelGen::Alchemist) {
            out.push_back({Severity::Info,
                           "Alchemist supports subgroup sizes 8/16/32, which makes it exactly the generation a "
                           "width-assuming dispatch breaks on.  This port's one-workgroup-per-row dispatch and "
                           "strided stage-2 combine were written for that; the width-8 arm of the gate is the "
                           "closest proxy available until real Alchemist silicon is attached."});
            out.push_back({Severity::Info,
                           "cooperative matrix on Alchemist: ANV has exposed VK_KHR_cooperative_matrix since Mesa "
                           "24.0, but it is a measured PERFORMANCE REGRESSION on pre-Xe2 parts, which is why "
                           "llama.cpp gates it to Xe2 only.  The plain-FMA ceiling therefore stands even with a "
                           "matrix-capable toolchain."});
            if (xe_loaded) {
                out.push_back({Severity::Warn,
                               "xe is driving an Alchemist card.  Intel's Xe support table lists no DG2/Alchemist "
                               "part at all (it begins at Lunar Lake/Battlemage), the xe path on DG2 is "
                               "experimental with open HuC issues, and compute-runtime reports zero "
                               "OpenCL/Level-Zero platforms on DG2 under xe (intel/compute-runtime#905).  i915 is "
                               "the driver Alchemist belongs on."});
            }
        }

        // The notes below describe the Xe driver's behaviour, so they are raised for Battlemage (Xe2 only drives
        // through xe) and for an Alchemist card that happens to be on xe - not for Alchemist on i915.
        if (on_xe) {
            out.push_back({Severity::Note,
                           "every individual submission must complete inside the GuC preemption timeout "
                           "(CONFIG_DRM_XE_PREEMPT_TIMEOUT = 640 ms on the 7.0 kernel inspected).  Exceeding it "
                           "is an engine reset under LLM inference on Battlemage, not a slowdown - keep the work "
                           "per submission bounded."});
            if (k.at_least(7, 1) && k.below(7, 3)) {
                out.push_back({Severity::Info, "kernel " + k.str() + " carries the Xe vRAM memory-pressure work "
                                                                   "(7.1), which is the path a full card takes."});
            }
            if (k.at_least(7, 1) && k.below(7, 2)) {
                out.push_back({Severity::Note,
                               "7.1 has CONFLICTING performance reports on Battlemage: Phoronix measured 7.1 "
                               "improving the Arc B580, while a community report claims a 50-90% OpenGL "
                               "regression beginning with 7.1.  Both exist - measure on the actual card instead "
                               "of trusting either."});
            }
            if (k.at_least(7, 3)) {
                out.push_back({Severity::Note,
                               "kernel " + k.str() + ": TTM eviction became more aggressive in 7.3.  That changes "
                               "what happens when the card is full, so re-verify the desktop reserve holds rather "
                               "than assuming the 7.2 behaviour."});
            }
            if (k.at_least(7, 4)) {
                out.push_back({Severity::Info, "kernel " + k.str() + " is at or beyond the 7.4 cycle, which "
                                                               "carries CPU binds and ULLS on the migration queue "
                                                               "- the first Battlemage-flagged performance "
                                                               "change (7.4 does not exist yet; verified "
                                                               "2026-10-03)."});
            }
        }
        if (!xe_loaded && i915_loaded && gen == IntelGen::Battlemage) {
            out.push_back({Severity::Warn, "i915 is loaded and xe is not, on a Battlemage device: i915 cannot "
                                           "drive Xe2.  Check which module owns the card."});
        } else if (!xe_loaded && i915_loaded) {
            out.push_back({Severity::Info, "i915 is loaded and xe is not: that is the correct driver for "
                                           "Alchemist (Intel's Xe table does not list DG2 at all)."});
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
