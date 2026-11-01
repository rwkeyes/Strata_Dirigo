# Strata standard benchmark: AMD Ryzen 7 7700X 8-Core Processor

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261101T220815Z-7d0d7841`, started 2026-11-01T22:08:15Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Ryzen 7 7700X 8-Core Processor
- CPU: AMD Ryzen 7 7700X 8-Core Processor; RAM: 61.9 GiB
- Storage of the model files: nvme SAMSUNG MZVL4512HBLU-00BL7 (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-38-generic; drivers: gpu-driver 7.0.0-38-generic; runtime: rocm 6.2.1-112
- Strata commit: `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; engine 0.1.55; source build
- Background work: lact (root) holds the Arc render node and the Hermes desktop Electron is resident (~431 MiB VRAM before the engine loads)

## Model and configuration

- Model: qwen3.8-flash-next-coder-iq1_m; quantization IQ1_M; revision bba88856f8c3c0c9e40e45f82d05e228b814ac43; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` `e11083ba855e7666...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (v0.2.9 source build (SYCL AOT, bmg-g31), tuned serving config on this box rather than setup defaults: --kv int8 --kv-resident 32768 --spec 4 --spec-min-p 0.5 --prefill auto --expert-cache auto --stream-experts --vram-reserve-mib 700 --no-prefill-borrow; every value chosen by measurement on this card)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 49 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 34.5 (30.5-41.0) n=3 | 0.51 (0.43-0.64) n=3 | 2.83 (2.75-7.91) n=3 | 11.3 (8.2-15.3) n=3 | 1/3 |
| prompt-4k | 1491.8 (587.4-1494.6) n=3 | 36.5 (32.8-43.6) n=3 | 2.77 (2.76-6.99) n=3 | 13.60 (9.40-17.80) n=2 | 16.8 (14.5-22.6) n=3 | 3/3 |
| prompt-16k | 1501.4 (1477.3-1508.2) n=3 | 40.9 (33.0-41.9) n=3 | 10.94 (10.90-11.12) n=3 | 17.88 (16.61-19.15) n=2 | 23.5 (23.1-26.6) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
