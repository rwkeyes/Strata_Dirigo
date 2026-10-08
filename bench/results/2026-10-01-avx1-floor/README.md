# The AVX1 floor's cost — RX 7900 XTX (gfx1100), 2026-10-01

**Question:** what does it cost to run Strata's CPU expert kernels on the AVX1 floor (an AVX-only Xeon, no
AVX2/FMA/F16C) instead of the AVX-512 path?

**Design:** *one binary*, two arms, so the difference cannot come from a rebuild, an alignment change or a
different code revision:

- `control` — `STRATA_FORCE_AVX1=1 STRATA_FORCE_AVX2=1`, i.e. exactly the tier an AVX-only Xeon executes
- `optimized` — the same binary taking this host's own AVX-512 path

Pack `coder-iq1_m`, identical template and settings, `tools/hip/bench_prefill.py`, one fresh server per arm.

## Result — a null result, stated as such

| arm | prefill tps (min / median / max) | decode tps (min – max) |
|---|---|---|
| native AVX-512 | 202 / 696 / 706 | 8.8 – 43.6 |
| forced AVX1 | 191 / 751 / 927 | 10.9 – 58.8 |

**This harness cannot resolve the floor's cost on this workload.** The trial-to-trial spread is ~5× on
prefill and ~6× on decode within a *single* arm — first-touch reads of a 55 GB coder pack from a rotational
drive dominate everything else. Any real delta is far below that noise, so neither arm winning is meaningful.

**The better number**, from a tighter controlled test on the same pack and binary (same prompts, small context,
sampling held constant): decode **21.8 → 19.1 tok/s**, i.e. **~12%** for the AVX1 path — with byte-identical
model output (both arms produced the same 627 chars).

## Honest caveats

- The arms ran **sequentially**, not interleaved (A then B). Interleaving would cancel slow drifts in page
  cache and drive behaviour; that is the fix for anyone re-running this.
- Prefill on this pack is memory/expert-streaming bound, so a CPU dot-product difference is expected to be
  small here. A CPU-only (`-ngl 0`) run, or a pack small enough to sit in RAM, would isolate the kernel cost
  much better.

## Files

- `raw-native.json`, `raw-avx1-floor.json` — verbatim `bench_prefill.py` output for both arms.
- `docs/benchmarks/2026-10-01-gfx1100-avx1-floor.json` — curated entry, candidate binary
  `build-hip/strata` (sha256 `aebf1213716f132e52b5a0fb09dc95059a75fab41470ee355ea5e9f18347527f`).
