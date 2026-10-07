/* hipcount.c -- LD_PRELOAD launch counter for the Strata engine's HIP path.
 *
 * WHY.  The HIP control on z820b answered tok/s (74.85 prefill / 24.07 decode with the PLE in RAM) but NOT the
 * brief's headline question, because the engine prints per-round TIME buckets and never a dispatch count.  Without
 * a count, `launch 36.101 ms/round` cannot be turned into us/dispatch, and the port's 66.8 us/dispatch (its own
 * instrument) has nothing to be compared against.  So: count the launches, from outside, with no engine change.
 *
 * ABI NOTE.  `hipLaunchKernel` takes two dim3 STRUCTS (12 bytes each), not six unsigned scalars -- declaring the
 * scalars would misalign the call and corrupt every launch.  `hipModuleLaunchKernel` really does take the scalars,
 * plus sharedMemBytes, stream, params, extra.  Both prototypes below match the HIP runtime exactly; this file
 * includes no HIP header, so the types are spelled out to the same layout (uint32 x3, 4-byte aligned).
 *
 * WHAT IT COUNTS.  Every kernel submission the HIP runtime sees:
 *   - hipLaunchKernel / hipModuleLaunchKernel        (direct launches; hipLaunchKernelGGL and <<< >>> both land here)
 *   - hipGraphAddKernelNode                          (kernels captured INTO a graph)
 *   - hipGraphLaunch / hipGraphInstantiate           (graph replays -- what a decode round actually submits)
 * A captured kernel is counted once at capture AND once per replay, which is the honest cost model.
 *
 * HOW TO READ IT.  One timestamped line per event to $HIPCOUNT_LOG (default /tmp/hipcount.log) so the launch
 * timeline can be correlated against the engine's own phase boundaries, plus a summary on stderr at exit.
 *
 * BUILD:  gcc -O2 -fPIC -shared -o hipcount.so hipcount.c -ldl
 * USE:    LD_PRELOAD=$PWD/hipcount.so HIPCOUNT_LOG=/tmp/hipcount.log ./build-hip/strata ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

typedef struct { uint32_t x, y, z; } dim3_like;   /* must match HIP's dim3: 12 bytes, 4-byte aligned */

static FILE *g_log;
static unsigned long g_launch, g_modlaunch, g_node, g_glaunch, g_ginst;
static struct timespec g_t0;
static int g_started;

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec - g_t0.tv_sec) * 1e6 + (ts.tv_nsec - g_t0.tv_nsec) / 1e3;
}
static void open_log(void) {
    if (g_started) return;
    g_started = 1;
    clock_gettime(CLOCK_MONOTONIC, &g_t0);
    const char *p = getenv("HIPCOUNT_LOG");
    if (!p || !*p) p = "/tmp/hipcount.log";
    g_log = fopen(p, "w");
}
static void note(const char *kind) {
    open_log();
    if (g_log) { fprintf(g_log, "%.1f\t%s\n", now_us(), kind); fflush(g_log); }
}

/* ---- direct launches ---- */
typedef int (*fn_launch)(const void *, dim3_like, dim3_like, void **, size_t, void *);
int hipLaunchKernel(const void *func, dim3_like grid, dim3_like block, void **args, size_t shmem, void *stream) {
    static fn_launch real;
    if (!real) { open_log(); real = (fn_launch)dlsym(RTLD_NEXT, "hipLaunchKernel"); }
    ++g_launch; note("launch");
    if (!real) { fprintf(stderr, "hipcount: real hipLaunchKernel missing\n"); return 1; }
    return real(func, grid, block, args, shmem, stream);
}

typedef int (*fn_modlaunch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, void *, void **, void **);
int hipModuleLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                          unsigned shmem, void *stream, void **params, void **extra) {
    static fn_modlaunch real;
    if (!real) { open_log(); real = (fn_modlaunch)dlsym(RTLD_NEXT, "hipModuleLaunchKernel"); }
    ++g_modlaunch; note("modlaunch");
    if (!real) return 1;
    return real(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
}

/* ---- capture: kernels recorded into a graph ---- */
typedef int (*fn_addnode)(void *, void **, const void *, const void *, unsigned, const void **);
int hipGraphAddKernelNode(void *graph, void **node, const void *params, const void *deps,
                          unsigned ndeps, const void **dep_edges) {
    static fn_addnode real;
    if (!real) { open_log(); real = (fn_addnode)dlsym(RTLD_NEXT, "hipGraphAddKernelNode"); }
    ++g_node; note("node");
    if (!real) return 1;
    return real(graph, node, params, deps, ndeps, dep_edges);
}

typedef int (*fn_glaunch)(void *, void *);
int hipGraphLaunch(void *exec, void *stream) {
    static fn_glaunch real;
    if (!real) { open_log(); real = (fn_glaunch)dlsym(RTLD_NEXT, "hipGraphLaunch"); }
    ++g_glaunch; note("graphlaunch");
    if (!real) return 1;
    return real(exec, stream);
}
typedef int (*fn_ginst)(void **, void *, unsigned long);
int hipGraphInstantiate(void **exec, void *graph, unsigned long flags) {
    static fn_ginst real;
    if (!real) { open_log(); real = (fn_ginst)dlsym(RTLD_NEXT, "hipGraphInstantiate"); }
    ++g_ginst; note("graphinst");
    if (!real) return 1;
    return real(exec, graph, flags);
}

__attribute__((destructor)) static void hipcount_report(void) {
    if (g_log) fclose(g_log);
    unsigned long subs = g_launch + g_modlaunch + g_node + g_glaunch;
    fprintf(stderr,
            "\n== hipcount: launch %lu  modlaunch %lu  nodes %lu  graphlaunch %lu  graphinst %lu  (kernel submissions %lu)\n",
            g_launch, g_modlaunch, g_node, g_glaunch, g_ginst, subs);
}
