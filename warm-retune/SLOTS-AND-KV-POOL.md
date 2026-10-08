# Controlling slots and the KV pool on a running server

This is the **warm-retune** feature for the vendored `llama-server`: changing how it is set up — how many
slots exist, how much context each may use, how big the pool behind them is, and how the KV itself is stored —
**while the model stays loaded**. The weights are never reloaded and no request in flight is dropped silently.

## The two things you are moving

* **A slot** is one request's workspace: it holds the conversation's KV (the attention keys and values) plus the
  sampler state. One slot serves one request at a time.
* **The pool** (with `--kv-unified`) is the single KV budget that every slot draws from. `-c` is its total size,
  and `-np N` with a per-slot context is a *reservation* drawn against it.

That last sentence is the whole reason this feature exists: **`-np 4 -c 131072` is not "4 slots, 131072 tokens
available" — it is four 32,768-token shares.** Three idle slots are then holding 96k tokens of KV that a single
long-context request could be using, and the usual fix is to restart the server with different flags. A restart
means reading the whole model back in — a minute at best on an NVMe, several on a spinning disk — and everything
in flight dies with it.

## Why you would want to change it while it runs

1. **What you need keeps changing, and no single setup suits all of it.** A coding agent that fans out four
   subagents wants four slots. The moment you hand it a 100k-token repository, you want one big slot instead.
   That pattern repeats all day.
2. **The alternative is a model reload.** On this box a 58 GB checkpoint takes ~65 s cold off an NVMe (311 s off
   the HDD) and drops every open conversation. Dividing the pool differently is measured in **microseconds**; rebuilding
   the pool is measured in **under a second** — because none of that touches the weight buffers.
3. **You can size for the demand you have now.** With a shared pool you do not have to decide the worst case at
   load time. Start with the sizes the work in front of you needs, and change them when it changes.
4. **You can also change how the KV itself is stored.** 8-bit KV to fit a longer pool into the same VRAM, or the fused
   attention kernels on/off, rebuild the same way, with the same loaded weights.
5. **It is a capacity lever, not just a count.** Growing the *total* pool can use VRAM that was free at load
   time without paying for a restart; shrinking it can hand VRAM back to something else (another process, a
   desktop that started misbehaving).

## The commands

Serve with a unified pool and the write path enabled:

```sh
llama-server -m model.gguf -ngl 99 -fa on --kv-unified --props -np 4 -c 131072 -a my-model
#             └ the weights   └ offload  └ attention kernels  └ one shared pool  └ write path  └ slots └ pool
```

`--kv-unified` is what makes the slot count a **server-side policy** instead of a load-time property, and
`--props` is what enables the write path. Ask what it is set to now:

```sh
curl -s localhost:8080/props | jq '{total_slots, slots_max, kv_pool_n_ctx, kv_unified}'
```

Then move it — **the same pool, divided differently with no restart (measured 0–0.4 ms, weights stay loaded):**

```sh
# four parallel short-context sessions
curl -s -X POST localhost:8080/props -d '{"parallel": 4, "ctx_per_slot": 32768}'

# now one long-context request needs most of the pool: hand it over
curl -s -X POST localhost:8080/props -d '{"parallel": 1, "ctx_per_slot": 131072}'

# back to four, each with more room than the old shares (this one grows the total, with no restart)
curl -s -X POST localhost:8080/props -d '{"parallel": 4, "ctx_per_slot": 65536}'
```

**Grow the pool itself, without touching the weights:**

```sh
curl -s -X POST localhost:8080/props -d '{"ctx": 262144}'        # a 262k pool, ~0.8 s measured 32k -> 262k
curl -s -X POST localhost:8080/props -d '{"parallel_max": 8}'     # raise the ceiling the reservation may reach
```

**Change how the KV itself is stored (same rebuild path):**

```sh
curl -s -X POST localhost:8080/props -d '{"cache_type_k": "q8_0"}'   # 8-bit KV: a longer pool in the same VRAM
curl -s -X POST localhost:8080/props -d '{"flash_attn": "on"}'        # fused attention kernels
```

**Rent out more slots than the pool backs** — every slot shares one budget, so the sizes are no longer
guaranteed, and a slot that overruns is refused rather than silently truncated:

```sh
curl -s -X POST localhost:8080/props -d '{"allow_oversubscribe": true}'
```

## What it refuses to do

A request that would take a slot **out of service while it is decoding** is answered

```
HTTP 400  slot N is busy, cannot take it out of service
```

— a refusal, never a hang and never a silent drop. That rule is why the change is safe to call from a script
that does not know what else is running: it will not pull the rug out from under a live conversation. Plan for
it by re-slicing while the pool is idle, or by reserving `parallel_max` ahead of the load that needs it.

## What it costs — measured

RX 7900 XTX 24 GB, a 35B-A3B MoE, Vulkan backend, q8_0 KV:

| operation | cost |
|---|---|
| dividing the pool differently (`parallel` / `ctx_per_slot`) | **0–0.4 ms** |
| the same change *while other slots were still answering* | **0.35 ms** |
| pool rebuild (`ctx` 32k → 262k), weights never reloaded | **~0.8 s** |
| `parallel_max` rebuild with 4 slots reserved | **814 ms** |

What the pool itself costs on that model — which is why a *reservation* is worth moving rather than paying for:

| pool | VRAM used | headroom | rebuild |
|---|---:|---:|---:|
| 32768 | 18,418 MiB | 6,142 MiB | — |
| 65536 | 18,882 MiB | 5,678 MiB | 0.82 s |
| 131072 | 19,818 MiB | 4,742 MiB | 1.01 s |
| 196608 | 20,754 MiB | 3,806 MiB | 0.93 s |
| 262144 | 21,690 MiB | 2,870 MiB | 0.86 s |

That model's KV is ~14.6 B/token (q8_0, few KV heads), so the whole 262k pool costs ~3.7 GB — cheap enough that
the number of slots and their sizes, not the total pool, is usually what you want to change. On a model with more KV heads the pool is the
expensive part, and the same calls are how you get VRAM back.

## Getting a server that can do this

Upstream's vendored llama.cpp answers `POST /props` with a stub (`{"success": true}` and nothing else). This
repository's patch — [`retune-llama-server-3cf0325.patch`](retune-llama-server-3cf0325.patch) — is what makes the
write path real, and `setup.py` applies it automatically when the vendored tree exists
([`apply.sh`](apply.sh) applies/undoes/checks it by hand). Build `llama-server` from that tree, start it with
`--kv-unified --props`, and everything above works.

The same idea inside Strata's *own* engine (which does not build `tools/server`) is documented separately in
[`RETUNE-CANDIDATES.md`](RETUNE-CANDIDATES.md): the `TUNE` line on the engine's stdin protocol and
`POST /props {"strata_tune": {...}}` for the settings read per request.
