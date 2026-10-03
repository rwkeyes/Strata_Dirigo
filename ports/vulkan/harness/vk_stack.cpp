// ports/vulkan/harness/vk_stack.cpp - see the header.  Detection is filesystem-only: no device node, no shelling
// out, nothing that needs privileges.
#include "vk_stack.hpp"

#include "vk_compat.hpp"   // Advisory, Severity, IntelGen, intel_generation

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace portvk {

namespace {

std::vector<std::string> library_dirs() {
    std::vector<std::string> dirs = {"/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
                                     "/usr/lib64",                 "/usr/lib",
                                     "/lib"};
    if (const char* lp = std::getenv("LD_LIBRARY_PATH")) {
        std::stringstream ss(lp);
        std::string d;
        while (std::getline(ss, d, ':')) {
            if (!d.empty()) dirs.push_back(d);
        }
    }
    return dirs;
}

std::string read_symlink_target(const std::string& path) {
    std::error_code ec;
    if (!fs::is_symlink(path, ec)) {
        // A real file of that name is fine too.
        return fs::exists(path, ec) ? fs::path(path).filename().string() : std::string();
    }
    return fs::read_symlink(path, ec).filename().string();
}

std::string find_library(const std::string& dir, const std::string& base) {
    // Accept `base`, and any `base.N` / `base.N.M` that sits beside it.
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name == base || (name.rfind(base + ".", 0) == 0 && name.size() > base.size() + 1)) {
            const char* tail = name.c_str() + base.size();
            if (name == base || (tail[0] == '.' && std::isdigit((unsigned char) tail[1]))) return e.path().string();
        }
    }
    return {};
}

std::string json_field(const std::string& text, const char* key) {
    // Just enough JSON for an ICD file: "key" : "value".  No nesting, no escapes, no arrays.
    const std::string k = std::string("\"") + key + "\"";
    size_t at = text.find(k);
    if (at == std::string::npos) return {};
    at = text.find(':', at + k.size());
    if (at == std::string::npos) return {};
    const size_t q1 = text.find('"', at);
    if (q1 == std::string::npos) return {};
    const size_t q2 = text.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return text.substr(q1 + 1, q2 - q1 - 1);
}

std::string read_file_or_empty(const std::string& p) {
    std::ifstream f(p);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

StackFile probe_file(const char* name) {
    StackFile f;
    f.name = name;
    // /lib/firmware and /usr/lib/firmware are both used depending on the distro and how it merged /usr.
    for (const char* root : {"/lib/firmware/", "/usr/lib/firmware/"}) {
        if (firmware_present_in(root, name, &f.found_path)) {
            std::error_code ec;
            f.size = (uint64_t) fs::file_size(f.found_path, ec);
            return f;
        }
    }
    return f;
}

}  // namespace

bool parse_so_version(const std::string& soname, int& maj, int& min, int& pat) {
    maj = min = pat = 0;
    // The version is the numeric tail after the last ".so".
    const size_t at = soname.rfind(".so");
    if (at == std::string::npos) return false;
    size_t i = at + 3;
    if (i >= soname.size() || soname[i] != '.') return false;
    ++i;
    const int got = std::sscanf(soname.c_str() + i, "%d.%d.%d", &maj, &min, &pat);
    if (got < 1) {
        maj = min = pat = 0;
        return false;
    }
    return true;
}

bool parse_dotted_version(const std::string& text, int& maj, int& min, int& pat) {
    maj = min = pat = 0;
    const int got = std::sscanf(text.c_str(), "%d.%d.%d", &maj, &min, &pat);
    if (got < 2) {
        maj = min = pat = 0;
        return false;
    }
    return true;
}

// The loader accepts .zst/.xz/.gz variants of a firmware name; the uncompressed name is tried FIRST so a host
// that carries both reports the plain file.
bool firmware_present_in(const std::string& root, const std::string& name, std::string* found_path) {
    static const char* kSuffixes[] = {"", ".zst", ".xz", ".gz"};
    std::error_code ec;
    for (const char* sfx : kSuffixes) {
        const std::string p = root + name + sfx;
        if (fs::exists(p, ec)) {
            if (found_path) *found_path = p;
            return true;
        }
    }
    return false;
}

std::string resolve_icd_library(const std::string& library_path) {
    if (library_path.empty()) return {};
    std::error_code ec;
    if (library_path.find('/') != std::string::npos) {
        // Absolute or relative path with a directory component: take it as given.
        return fs::exists(library_path, ec) ? library_path : std::string();
    }
    // A bare soname.  THIS IS THE NORMAL CASE and the reason a plain existence test is wrong: the loader
    // resolves it through the system library path, so it lives in a standard directory, not next to the JSON.
    for (const std::string& dir : library_dirs()) {
        const std::string direct = dir + "/" + library_path;
        if (fs::exists(direct, ec)) return direct;
        const std::string found = find_library(dir, library_path);
        if (!found.empty()) return found;
    }
    return {};
}

StackReport detect_stack() {
    StackReport s;

    // The loader: only the version matters, and it is the tail of the symlink target.
    for (const char* p : {"/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/lib/x86_64-linux-gnu/libvulkan.so.1",
                          "/usr/lib64/libvulkan.so.1", "/usr/lib/libvulkan.so.1"}) {
        const std::string t = read_symlink_target(p);
        if (!t.empty()) {
            s.loader_path = p;
            s.loader_version = t;
            break;
        }
    }
    for (const char* p : {"/usr/lib/x86_64-linux-gnu/libdrm.so.2", "/lib/x86_64-linux-gnu/libdrm.so.2",
                          "/usr/lib64/libdrm.so.2", "/usr/lib/libdrm.so.2"}) {
        const std::string t = read_symlink_target(p);
        if (!t.empty()) {
            s.libdrm_version = t;
            break;
        }
    }

    // ICD files.  VK_DRIVER_FILES / VK_ICD_FILENAMES REPLACE the default search rather than adding to it (that
    // is what the loader does), so a check that always scanned the system directory would report the system's
    // ICDs as "the" ICDs and would test the wrong file whenever one of those variables is set.
    std::vector<std::string> icd_files;
    const char* env_files = std::getenv("VK_DRIVER_FILES");
    if (!env_files) env_files = std::getenv("VK_ICD_FILENAMES");
    if (env_files && *env_files) {
        std::stringstream ss(env_files);
        std::string p;
        while (std::getline(ss, p, ':')) {
            if (!p.empty()) icd_files.push_back(p);
        }
    } else {
        for (const char* dir : {"/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"}) {
            std::error_code ec;
            if (!fs::exists(dir, ec)) continue;
            for (const auto& e : fs::directory_iterator(dir, ec)) {
                if (e.path().extension() == ".json") icd_files.push_back(e.path().string());
            }
        }
    }
    for (const std::string& f : icd_files) {
        IcdEntry e;
        e.file = f;
        const std::string text = read_file_or_empty(f);
        e.library_path = json_field(text, "library_path");
        e.api_version = json_field(text, "api_version");
        e.resolved = resolve_icd_library(e.library_path);
        s.icds.push_back(e);
    }

    // The firmware blobs the xe driver asks for, and the i915 DMC blob.  A missing GuC blob is a driver that
    // fails to probe - the log says `firmware production part check failure` and `probe ... failed with -71`.
    // Both generations' blobs are probed because the file that is loaded depends on the CARD, and this function
    // has no device yet; the advisory below selects the set that matters.  The Alchemist names are the i915 DG2
    // blobs, and a "missing" verdict is only raised when a generation's WHOLE set is absent - a single drifted
    // filename must not be reported as missing firmware.
    for (const char* f : {"xe/bmg_guc_70.bin", "xe/bmg_huc.bin", "i915/bmg_dmc.bin",
                          "i915/dg2_guc_70.bin", "i915/dg2_huc_gsc.bin", "i915/dg2_dmc_ver2_08.bin"}) {
        s.firmware.push_back(probe_file(f));
    }

    if (const char* t = std::getenv("XDG_SESSION_TYPE")) {
        s.session_type = t;
    }
    if (s.session_type.empty()) {
        if (std::getenv("WAYLAND_DISPLAY")) s.session_type = "wayland";
        else if (std::getenv("DISPLAY")) s.session_type = "x11";
        else s.session_type = "none";
    }

    for (const std::string& dir : library_dirs()) {
        std::error_code ec;
        if (fs::exists(dir + "/libze_intel_gpu.so.1", ec)) s.level_zero = true;
        if (fs::exists(dir + "/libOpenCL.so.1", ec)) s.opencl = true;
    }
    return s;
}

std::vector<Advisory> stack_advisories(const StackReport& s, uint32_t vendor_id, uint32_t device_id) {
    std::vector<Advisory> out;

    // An ICD that names a library nobody can find is a real failure mode: the loader skips it, so the GPU
    // silently disappears from enumeration and every downstream symptom points at the wrong layer.
    for (const IcdEntry& e : s.icds) {
        if (!e.resolves()) {
            out.push_back({Severity::Warn, "ICD " + e.file + " names '" + e.library_path +
                                               "', which does not resolve anywhere in the library path.  The "
                                               "loader will skip it and the GPU it belongs to will be missing "
                                               "from enumeration.  Check for a stale ICD left by an old Mesa "
                                               "(or a wrong VK_ICD_FILENAMES)."});
        }
    }

    // Loader older than what an ICD advertises.  Commonly harmless (the loader is a thin trampoline) but it is
    // the first thing to suspect when a documented extension is absent, so it is reported as a note, not a rule.
    int lmaj = 0, lmin = 0, lpat = 0;
    if (parse_so_version(s.loader_version, lmaj, lmin, lpat)) {
        for (const IcdEntry& e : s.icds) {
            int amaj = 0, amin = 0, apat = 0;
            if (!parse_dotted_version(e.api_version, amaj, amin, apat)) continue;
            const bool loader_older = lmaj * 10000 + lmin * 100 + lpat / 1000 < amaj * 10000 + amin * 100 + apat / 1000;
            if (e.resolves() && loader_older) {
                out.push_back({Severity::Note, "the Vulkan loader (" + s.loader_version + ") is older than what " +
                                                   std::string(e.file) + " advertises (" + e.api_version +
                                                   ").  Usually harmless; suspect it first if a documented "
                                                   "extension is missing."});
            }
        }
    }

    // Firmware: the blobs are per generation, so only the set belonging to the card present is judged.  A
    // missing GuC blob is a driver that FAILS TO PROBE (firmware production part check failure, probe -71), so
    // this only warns when the whole set for that generation is absent; a partly-present set means the firmware
    // package is installed and a name has drifted.
    const IntelGen fw_gen = intel_generation(device_id);
    const char* fw_want = fw_gen == IntelGen::Alchemist ? "dg2" : "bmg";
    int fw_have = 0, fw_want_count = 0;
    for (const StackFile& f : s.firmware) {
        if (f.name.find(fw_want) == std::string::npos) continue;
        ++fw_want_count;
        if (f.present()) ++fw_have;
    }
    if (vendor_id == 0x8086) {
        if (fw_want_count > 0 && fw_have == 0) {
            out.push_back({Severity::Warn, std::string("this host presents an Intel ") +
                                               (fw_gen == IntelGen::Alchemist ? "Alchemist" : "Battlemage") +
                                               " GPU but none of its firmware blobs (" + fw_want +
                                               "*) are present.  A missing GuC blob is a driver that fails to "
                                               "PROBE, not a slow driver - check the firmware package before "
                                               "anything else."});
        } else if (fw_want_count > 0 && fw_have < fw_want_count) {
            out.push_back({Severity::Info, std::string("the firmware package is installed (") +
                                               std::to_string(fw_have) + " of " + std::to_string(fw_want_count) +
                                               " " + fw_want + " blobs present); a missing name is a naming "
                                               "difference, not necessarily a missing package."});
        }
    } else if (fw_have == 0) {
        out.push_back({Severity::Info, "no Intel GPU firmware on this host - expected unless an Intel GPU "
                                       "is fitted."});
    }

    if (s.level_zero || s.opencl) {
        out.push_back({Severity::Info, (s.level_zero ? std::string("Level-Zero ") : std::string()) +
                                           (s.opencl ? std::string("OpenCL ") : std::string()) +
                                           "user-space is installed.  It serves the SYCL/OpenCL path, NOT this "
                                           "Vulkan port - report Vulkan problems as Vulkan problems."});
    }
    if (s.session_type == "wayland" || s.session_type == "x11") {
        out.push_back({Severity::Info, "session type: " + s.session_type +
                                           ".  A Wayland compositor was measured holding 354 MB of GPU memory on "
                                           "a two-display desktop, which is what the 512 MiB reserve floor is "
                                           "sized against; an X11 session's usage differs."});
    }
    if (out.empty()) out.push_back({Severity::Info, "no stack-level findings."});
    return out;
}

}  // namespace portvk
