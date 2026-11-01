# Intel Arc Pro B70: v0.2.6 -> v0.2.9 prompt throughput, interleaved (not a standard result)

This is **not** a `standard-v1` result and must not be read as one: it uses `tools/hip/bench_prefill.py` plus a
long-prompt probe on a tuned configuration, not the protocol's frozen corpus and twelve sanity checks. The
standard result from the same machine is
[`2026-11-01-standard-ryzen-7-7700x-8-core-processor-7d0d7841`](../2026-11-01-standard-ryzen-7-7700x-8-core-processor-7d0d7841/README.md)
(read that folder's `HARDWARE.md`: the GPU is an **Intel Arc Pro B70 32 GB**, not the Ryzen APU its title names).

## What was measured

Same card, same pack, same tuned configuration, same harness; only the engine version changes. Four arms,
**interleaved** `0.2.6 / 0.2.9 / 0.2.6 / 0.2.9`, each arm a fresh engine load, so page-cache and order effects
show up as a spread inside one release rather than as a release delta.

- Engine: v0.2.6 = engine 0.1.51, v0.2.9 = engine 0.1.55 (both source builds, SYCL AOT `bmg-g31`, oneAPI DPC++ 2026.2.0)
- Model: Coder IQ1_M native pack, `--native` shard 1 / `--ple-gguf` shard 2, 131,072-token context
- Config: `--kv int8 --kv-resident 32768 --spec 4 --spec-min-p 0.5 --prefill auto --expert-cache auto --stream-experts --vram-reserve-mib 700 --no-prefill-borrow`
- Harness: `tools/hip/bench_prefill.py` (unmodified) and a streaming long-prompt probe
- Fit, identical in both releases' middle arms: **11 615 of 11 615 expert slots, 673 experts read over PCIe**

## Result — steady state (second pass of each lane, mean of the two arms per release)

| lane | 0.2.6 prompt tok/s | 0.2.9 prompt tok/s | delta | within-release spread |
| --- | ---: | ---: | ---: | --- |
| fresh, 30 178 tokens | 1 269.5 | **1 414.3** | **+11.4 %** | 0.4 % / 1.9 % |
| fresh, 7 855 tokens (needle) | 1 284.2 | **1 428.2** | **+11.2 %** | 0.4 % / 2.1 % |
| fresh, 8 830 tokens | 1 310.9 | **1 486.2** | **+13.4 %** | 0.6 % / 2.5 % |
| fresh, 4 210 tokens | 1 216.8 | **1 395.0** | **+14.6 %** | 4.4 % / 6.4 % |

Decode is level in the same arms: +1.0 % (30 K), +2.0 % (needle), +0.1 % (8.8 K), −1.3 % (4.2 K).

The cached-prefix **follow-up** lanes are **not quotable** here: the within-release spread reached 13–44 %, and
one lane read −28.5 %. They are in the raw files; do not read the delta as a finding.

Functional parity on the card: every arm returned the same needle answer, the same 296-character code answer
in all four arms, and the same clean HTTP 400 past the window ("requests are never truncated").

## Why this pair

v0.2.8 turned on **ESIMD int8-DPAS prompt attention for Battlemage** by default (`STRATA_PROMPT_ATTN_ESIMD=0`
turns it off; the path self-gates on `intel_gpu_gen == Battlemage` and needs the int8 KV pools, `--kv int8`) and
budgeted the prompt-attention batch; v0.2.9 adds `STRATA_KPROF` per-kernel timing. Both are prompt-side, and both
release notes list prefill-class work only, which is what the numbers show. The engine log carries
`strata: prompt attention on the int8 DPAS (ESIMD)` in the 0.2.9 arms.

An on/off pair of the same binary (`STRATA_PROMPT_ATTN_ESIMD=1` vs `=0`) was not run, so this submission
attributes the gain to the release, not to that single switch.

## Files

`raw/bench-*.json` — the harness's own per-trial output, four files (`ab026-a`, `ab029-b`, `ab026-c`,
`ab029-d`); `raw/longprobe-*.json` — prompt/decode for the code, 30 K, needle and over-window legs;
`raw/engine-*.txt` — the engine's own start/slots/PCIe lines per arm; `raw/battery.log` — the arm log with the
request bodies stripped. Host paths are redacted.
