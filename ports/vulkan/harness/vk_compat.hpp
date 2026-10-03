// ports/vulkan/harness/vk_compat.hpp - the Linux compatibility layer: what this box is, and what that means
// for the port.
//
// WHY THIS EXISTS AT ALL, given the port has no kernel-facing interface.  The kernel is not in the port's code
// path, but it is squarely in its risk path: the Xe driver's history decides whether the card can run sustained
// compute without a CCS engine reset, and the kernel/userland split decides whether the free-memory figure the
// display reserve depends on is real.  Those are facts about the host, so the port detects them, states them,
// and refuses only where the combination is known to be impossible.
//
// The rules are literature, not folklore: each one names the version range it comes from.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace portvk {
bool looks_discrete(uint64_t heap_total_bytes, uint64_t host_ram_bytes);
}

namespace portvk {

// "7.3.0-rc5" / "7.0.0-34-generic" -> {7,3,0,prerelease=false}
struct KernelVersion {
    int major = 0, minor = 0, patch = 0;
    bool prerelease = false;
    std::string release;   // uname -r verbatim, for the log
    bool at_least(int maj, int min) const { return major > maj || (major == maj && minor >= min); }
    bool below(int maj, int min) const { return !at_least(maj, min); }
    std::string str() const;
};

struct HostEnv {
    KernelVersion kernel;
    uint64_t host_ram_bytes = 0;
    std::vector<std::string> drm_modules;   // DRM modules loaded, from /sys/module
    std::string mesa_version;               // parsed out of VkPhysicalDeviceDriverProperties.driverInfo
    std::string driver_name;                // e.g. "Intel open-source Mesa driver"
};

enum class Severity { Info, Note, Warn, Fatal };

struct Advisory {
    Severity sev = Severity::Info;
    std::string text;
};

const char* severity_name(Severity s);

// Exposed so the parse can be tested against real release strings rather than trusted.
KernelVersion parse_kernel_release(const std::string& release);

// OS-side facts (uname, host RAM, loaded DRM modules).  No Vulkan needed, so it is unit-testable.
HostEnv detect_host_env();
// Adds the userspace driver identity from the device actually chosen.
void fill_driver_info(VkPhysicalDevice pd, HostEnv& env);

// Parses "Mesa 25.2.8", "Mesa 26.2.0-devel", "Mesa 25.2.8 (LLVM 20.1.2)" -> {25,2,8}.  Returns false when the
// string carries no version, which is a case that must be reported rather than assumed.
bool parse_mesa_version(const std::string& driver_info, int& major, int& minor, int& patch);

// The rules.  `vendor_id` and `budget_from_driver` come from the chosen device and the memory query; a
// caller that has not queried yet passes budget_from_driver = false and gets the conservative reading.
std::vector<Advisory> compat_advisories(const HostEnv& env, uint32_t vendor_id, bool budget_from_driver);
bool any_fatal(const std::vector<Advisory>& v);

// True when the device's local heap is small relative to host RAM, i.e. a real discrete card rather than a
// software implementation or a unified-memory chip.  The display reserve and the ledger rule both key on it.
bool looks_discrete(uint64_t heap_total_bytes, uint64_t host_ram_bytes);

}  // namespace portvk
