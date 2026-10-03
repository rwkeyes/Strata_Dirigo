// ports/vulkan/harness/vk_stack.hpp - the rest of the software stack: drivers, libraries, tools-ish files.
//
// vk_compat covers the KERNEL.  This covers what sits between the kernel and this port, because the port's
// behaviour depends on components whose versions move independently of each other - and the port links only ONE
// of them (the loader).  Everything here is detected from the filesystem with no device access:
//
//   * the Vulkan loader      - its version is in the .so name; it is the only stack component this port links
//   * the ICD files          - and whether the library each one NAMES actually resolves.  That is the check
//                              whose first version was wrong: `library_path` is usually a BARE SONAME
//                              ("libvulkan_radeon.so"), resolved by the loader through the system library path,
//                              so a plain existence test flags every working ICD as broken.  The resolution
//                              below handles both a bare name and an absolute path.
//   * libdrm                 - Mesa's dependency, not the port's; its version tracks the kernel uAPI it speaks
//   * GPU firmware blobs     - a missing GuC/HuC blob is a driver that fails to probe, not a slow driver
//   * the accelerator runtimes (Level-Zero, OpenCL) - they serve the SYCL/OpenCL path, NOT this port, and are
//                              reported so nobody debugs the wrong stack
//   * the session type       - X11 vs Wayland changes how much memory the desktop holds, which is what the
//                              display reserve is sized against
//
// TOOLS (glslc, spirv-val, the compiler) are probed by gates/run_gate.sh, which is where they are already run:
// probing them from here would mean duplicating the invocations and could disagree with the ones that matter.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace portvk {

struct Advisory;   // vk_compat.hpp

struct StackFile {
    std::string name;          // as requested by the driver, e.g. "xe/bmg_guc_70.bin"
    std::string found_path;    // "" when absent
    uint64_t size = 0;
    bool present() const { return !found_path.empty(); }
};

struct IcdEntry {
    std::string file;            // /usr/share/vulkan/icd.d/radeon_icd.json
    std::string library_path;    // verbatim from the JSON, bare name or absolute
    std::string resolved;        // "" when it cannot be found anywhere
    std::string api_version;     // the ICD's advertised api_version, verbatim
    bool resolves() const { return !resolved.empty(); }
};

struct StackReport {
    std::string loader_path, loader_version;   // loader_version from the .so name, e.g. "1.3.275"
    std::string libdrm_version;
    std::vector<IcdEntry> icds;
    std::vector<StackFile> firmware;
    std::string session_type;                  // "wayland" / "x11" / "none"
    bool level_zero = false;                   // libze_intel_gpu
    bool opencl = false;                       // libOpenCL
};

StackReport detect_stack();

// ---- the pure parts, so they can be tested without a filesystem ----

// "libvulkan.so.1.3.275" -> {1,3,275}; "libdrm.so.2.125.0" -> {2,125,0}.  False when there is no numeric tail.
bool parse_so_version(const std::string& soname, int& maj, int& min, int& pat);
// "1.4.318" -> {1,4,318}.  Used only to compare the loader against what an ICD advertises.
bool parse_dotted_version(const std::string& text, int& maj, int& min, int& pat);
// "" when `library_path` cannot be found.  An absolute path is checked as given; a bare soname is searched for
// in the standard library directories and $LD_LIBRARY_PATH, accepting ".N" suffixed variants.
std::string resolve_icd_library(const std::string& library_path);

// `device_id` is needed because the firmware blobs are per GENERATION: Battlemage (xe/bmg_*) and Alchemist
// (i915/dg2_*) load different files, so a missing-blob verdict can only be raised for the generation present.
std::vector<Advisory> stack_advisories(const StackReport& s, uint32_t vendor_id, uint32_t device_id);

}  // namespace portvk
