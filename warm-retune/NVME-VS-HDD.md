# NVMe vs HDD on vega (RX 7900 XTX): the same checkpoint, same config, one variable

Measured 2026-10-03 with the fork's engine, `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M` (58.4 GB, 2 shards), identical
config (16k context, `--kv int8`, `--kv-resident 32768`, `--resident-budget-gib 20`, `--spec 4`, the same pack,
tokenizer, expert profile and MTP head) on **both** arms.  The only difference is where the GGUF bytes live:

| arm | model files | port | start |
|---|---|---|---|
| **HDD** | `/media/bob/3d65…/public/strata-gguf/IQ1_M/` (spinning disk) | 18110 | page cache dropped first (cold) |
| **NVMe** | `/home/bob/strata-models/IQ1_M/` (OEM NVMe, staged + sha256-verified against the pinned HF revision) | 18113 | page cache dropped first (cold) |

`~/bin/strata-arm-run.sh hdd|nvme --label …` runs an arm; `~/bin/strata-arm-diff.py <hdd-label> <nvme-label>` prints
the table below from the two JSONs in `~/forktest/logs/`.

## Cold start (health + first-token path)

| | HDD | NVMe | |
|---|---:|---:|---|
| ready after | **311 s** | **64 s** | 4.9× |

## The bench (`tools/hip/bench_prefill.py`: matched fresh/follow-up pairs, 128 tokens out)

| trial | prefill t/s HDD | prefill t/s NVMe | decode t/s HDD | decode t/s NVMe | fresh/reused (both) |
|---|---:|---:|---:|---:|---|
| warmup (15-token prompt) | 41.1 | 78.3 | 12.1 | 38.4 | 15/0 |
| fresh 1 (4210 tok) | 232.7 | 942.3 | **15.5** | **58.9** | 4210/0 |
| follow-up 1 | 120.3 | 269.9 | 19.1 | 68.0 | 248/4203 |
| fresh 2 (8830 tok) | 664.4 | 997.3 | 26.8 | 79.2 | 8830/0 |
| follow-up 2 | 141.5 | 170.3 | 26.0 | 72.2 | 113/8958 |
| fresh 3 (4210 tok) | 944.0 | 993.1 | 43.2 | 83.5 | 4210/0 |
| follow-up 3 | 259.6 | 170.3 | 25.4 | 78.1 | 248/4203 |
| fresh 4 (8830 tok) | 965.5 | 986.9 | 34.1 | 82.9 | 8830/0 |
| follow-up 4 | 153.9 | 170.1 | 28.2 | 72.9 | 114/8957 |
| **decode min/median/max** | 12.1 / **26.0** / 43.2 | 38.4 / **72.9** / 83.5 | | | 9 trials each |

Decode is **2.8× faster** on the NVMe (median), and the *first* request after a cold start is where the spinning
disk hurts most: 15.5 → 58.9 t/s (3.8×).  Prefill settles near **990 t/s** on the NVMe from the first big prompt,
while the HDD needs three trials to reach the same range (664 → 944 → 965) — its prefill rate is bounded by how
fast the expert blobs come off the disk.  Expert-cache hit rates were 98.7–99.5% on both arms, so this is the
storage tier's cost, not a difference in how the cache was populated; the engine's own accounting shows the HDD arm reading 54 GB from
the GGUF during the run.

## What this does not say

* **Not a comparison against the fork's published tables.** `docs/DETAILS.md`'s "Speed (measured)" numbers are an
  RTX 5070 12 GB on Windows at 4K–262K contexts — a different machine, GPU, driver and context, so they are not a
  baseline for this box.  The comparable prior here is the same box's earlier **8k-context HDD run**: at 4210 fresh
  tokens it read 15.9 t/s decode against this run's 15.5, and 19.4 t/s on the follow-up against 19.1 — i.e. moving
  the context from 8192 to 16384 changed nothing measurable at the same prompt length, which is what makes the
  HDD/NVMe pair above a fair one.
* **Not an end-to-end speed claim.** The bench is one prompt family (generated `ruleN` JavaScript) at a fixed
  output length, `temperature 0, top_k 1`, and decode speed on this model also moves several percent with the text
  itself (accepted draft tokens).  The 2.8× is a storage-tier effect of that size, not a promise about every
  workload.
* **Not a recommendation to keep both copies.** With the NVMe staged copy verified against the pinned revision,
  the HDD copy is a spare; the NVMe arm is the one to serve from.
