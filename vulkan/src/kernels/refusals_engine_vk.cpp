// vulkan/src/kernels/refusals_engine_vk.cpp - THE ENGINE-LEVEL LOUD REFUSALS: the `strata::core::` and
// `strata::core::device_code_error` symbols a full `strata` PROGRAM link wants, defined ONLY so the program
// LINKS, and refusing with the flag chain that reaches them where the operation is not implemented.
//
// WHY THIS FILE, AND WHY IT IS SEPARATE FROM refusals_vk.cpp.  `refusals_vk.cpp` answers the kernels-NAMESPACE
// holes of the decode path.  These symbols are engine CLASSES / helpers the executable also references:
//
//   * `strata::core::RemoteExpertOpt` - the REMOTE / PEER expert tier (`src/core/remote_expert_opt.cu`, EXCLUDED
//     from the Vulkan build), reached only with `--expert-cache-remote N` / `--remote-expert-opt` / `--peer-device`
//     (`generate.cpp:3554-3664`) - a multi-GPU configuration this one-device backend does not select.
//
// THE DESTRUCTORS ARE REAL, EMPTY BODIES, NOT REFUSALS.  `RemoteExpertOpt`'s destructor runs at scope exit even
// when the tier was never opened; refusing it would abort a run that never used the object.  Only the
// substantive operations refuse.  A refusal exits with status 2 (the port's convention).
//
// `strata::core::device_code_error` (device.cu:214) is the CUDA-ARCHITECTURE probe: it asks the CUDA runtime
// whether the engine's kernels were built for the cards present (`cudaFuncGetAttributes` on a probe kernel).
// A Vulkan device has no CUDA architecture to mismatch, so the backend's answer is the real one - the empty
// string, "no problem" - not a refusal.  (Refusing it would abort a startup check that the CUDA build passes.)
#if !defined(STRATA_ENABLE_VULKAN)
#error "refusals_engine_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/core/device.hpp"              // device_code_error
#include "strata/core/remote_expert_opt.hpp"   // RemoteExpertOpt (the exact signatures)

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
[[noreturn]] void refuse_engine(const char* sym, const char* chain) {
    std::fprintf(stderr,
                 "%s: NOT PORTED on the Vulkan backend - REFUSING.\n"
                 "  Reached only by: %s\n"
                 "  This definition exists so the `strata` program LINKS; it is a LOUD REFUSAL, never a silent\n"
                 "  fallback and never a fabricated size.\n",
                 sym, chain);
    std::exit(2);
}
}  // namespace

// ---- the REMOTE / PEER expert tier (src/core/remote_expert_opt.cu, excluded) ---------------------------------
namespace strata::core {

// A Vulkan device has no CUDA architecture; there is no mismatch to report.
std::string device_code_error() { return std::string(); }

RemoteExpertOpt::~RemoteExpertOpt() {}   // REAL: destroyed at scope exit whether or not the tier was used

void RemoteExpertOpt::attach(RemoteExperts&) {
    refuse_engine("strata::core::RemoteExpertOpt::attach", "--remote-expert-opt / --expert-cache-remote N / --peer-device (generate.cpp:3554-3664)");
}
bool RemoteExpertOpt::init(std::string&) {
    refuse_engine("strata::core::RemoteExpertOpt::init", "--remote-expert-opt / --expert-cache-remote N / --peer-device");
}
bool RemoteExpertOpt::owns(int64_t, int32_t) const {
    refuse_engine("strata::core::RemoteExpertOpt::owns", "--remote-expert-opt (a residency query only that tier makes)");
}
bool RemoteExpertOpt::adapt(const std::vector<float>&, const std::vector<int32_t>&,
                            const std::vector<std::pair<int32_t, int32_t>>&, int, ExpertSource&) {
    refuse_engine("strata::core::RemoteExpertOpt::adapt", "--remote-expert-opt / --expert-cache-remote N; the remote tier's swap planner");
}
void RemoteExpertOpt::begin(const float*, int, int) {
    refuse_engine("strata::core::RemoteExpertOpt::begin", "--remote-expert-opt / --peer-device");
}
void RemoteExpertOpt::copy_rows(float*, const float*, int, int, const int32_t*, const int32_t*, void*) const {
    refuse_engine("strata::core::RemoteExpertOpt::copy_rows", "--remote-expert-opt / --peer-device; the remote tier's row copy");
}
void RemoteExpertOpt::combine(float*, float*, int, int, const uint32_t*, uint32_t, void*) const {
    refuse_engine("strata::core::RemoteExpertOpt::combine", "--remote-expert-opt / --peer-device; the remote tier's combine");
}
size_t RemoteExpertOpt::metadata_bytes() {
    refuse_engine("strata::core::RemoteExpertOpt::metadata_bytes", "--remote-expert-opt / --peer-device; the remote tier's metadata size (not fabricated)");
}
void RemoteExpertOpt::prepare(const RemoteExperts&, void*) const {
    refuse_engine("strata::core::RemoteExpertOpt::prepare", "--remote-expert-opt / --peer-device");
}
bool RemoteExpertOpt::reduce(RemoteExperts&, const void*, std::string&) {
    refuse_engine("strata::core::RemoteExpertOpt::reduce", "--remote-expert-opt / --peer-device");
}
void RemoteExpertOpt::accumulate(const RemoteExperts&) {
    refuse_engine("strata::core::RemoteExpertOpt::accumulate", "--remote-expert-opt / --peer-device");
}

}  // namespace strata::core
