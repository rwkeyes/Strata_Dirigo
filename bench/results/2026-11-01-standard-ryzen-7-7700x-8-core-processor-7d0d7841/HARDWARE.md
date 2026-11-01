# Hardware this run actually used, and three fields the report reads wrong

The `README.md` and `results.json` in this folder are the runner's output, unmodified. Over `--url` the runner
derives the machine from the local host, and on this box three fields came out wrong. They are corrected here
rather than edited in place, so the artifacts stay exactly what `strata bench run` wrote.

| field | runner says | actual | how it was verified |
| --- | --- | --- | --- |
| GPU | `AMD Ryzen 7 7700X 8-Core Processor` | **Intel Arc Pro B70, 32 GB (Battlemage, BMG-G31)** | `lspci -nn`: `03:00.0 VGA ... Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223]`; the engine's own log line: `GPU 0: Intel(R) Arc(TM) Pro B70 Graphics, compute capability 20.2` |
| driver / runtime | `gpu-driver 7.0.0-38-generic`; `rocm 6.2.1-112` | SYCL / oneAPI, **DPC++ 2026.2.0**, in-tree `xe` KMD on 7.0.0-38-generic (no ROCm involved in this engine) | `dpkg -l intel-oneapi-compiler-dpcpp-cpp` → 2026.2.0-863; `lsmod` → `xe`; build `STRATA_SYCL_AOT=bmg-g31` |
| storage of the model files | `nvme SAMSUNG MZVL4512HBLU-00BL7` | a **rotational 12.9 TB HDD** (`/dev/sda1`, `ROTA=1`), reached through symlinks; the engine warns about it at start | `df`/`lsblk` on both `.gguf` targets; engine log: `the n-gram table (...) is on a rotational disk (sda)` |

## Why each one is wrong

- **GPU.** `gpu_info_local()` probes `nvidia-smi`, then falls back to `rocm-smi`. There is no Intel path, and on
  an AMD APU `rocm-smi --showproductname` prints the *CPU* name: `Card Series: AMD Ryzen 7 7700X 8-Core
  Processor` (`Card Model: 0x164e`, `RAPHAEL` — the integrated Raphael GPU). So on a hybrid Intel-discrete +
  AMD-APU box the report names the APU as the GPU, and its VRAM (`536870912` bytes read from card2's
  `mem_info_vram_total`) as the card's. The server's `/props` did not carry `hardware_static.gpu_name` for this
  `--url` run, so the "the engine names its own cards" branch in `machine()` never fired.
- **Runtime.** The same `rocm-smi` fallback supplies `rocm_driver`, so a ROCm version is reported for a SYCL
  run. `cuda` is correctly null.
- **Storage.** `storage_info()` inspects the filesystem of the working directory, which is `/` on the NVMe. The
  model files are symlinks onto the data HDD, so the number describes the wrong disk. This one matters: the
  PLE/ngram table is read at random from that spindle (the engine says so), which is why load took 96 s.

Everything else in the report is right: CPU, RAM, OS, Strata commit `c62d5e3a`, engine 0.1.55, IQ1_M, the
model-file SHA-256s, the 131,072-token context, `track: custom`, and all 21 requests.

The tuned configuration in `track: custom` is the engine's own released binary with per-flag choices measured on
this card (`--kv int8`, `--kv-resident 32768`, `--spec 4`, `--spec-min-p 0.5`, `--prefill auto`,
`--expert-cache auto`, `--stream-experts`, `--vram-reserve-mib 700`, `--no-prefill-borrow`) — not setup
defaults, which is why it is not filed under `track: default`.
