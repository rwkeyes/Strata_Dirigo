# The CUDA graph API over the port's OWN recorded step

**Batch:** the CUDA graph API — the last structural blocker between a layer body that runs (M-B, `vega`, Arc B70)
and an engine program that can record and replay its step.
**Where the code is:** `vulkan/include/cuda_compat/cuda_runtime.h` (declarations + the stated semantics),
`vulkan/src/compat/cuda_runtime.cpp` (the graphs), `vulkan/src/device/vk_compute.{hpp,cpp}` (the capture
primitives over the recorded step), the gate case `case_cuda_graph_entry` in `ports/vulkan/harness/vk_gate.cpp`.

---

## A. HOW THE ENGINE ACTUALLY USES CAPTURE (read from the engine's own code)

### Which calls, in what order

The engine's recorder is `src/core/session.cpp` (and the same shape in `src/core/mtp.cpp` and
`src/core/verify.cpp`); `include/strata/core/graph.hpp`'s `GraphRegistry` exists but is REFERENCED BY NO ENGINE
HOST CODE (only `graph.cpp` itself and the `sycl/` copy name it) — the real recorder is `session.cpp`.

`session_capture` (per layer):

```cpp
// src/core/session.cpp:221-258
if (cudaStreamCreate(&cs) != cudaSuccess) { ... }
if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { ... }
const bool ok = post ? block_layer_post(..., (void*) cs, err)
                     : block_layer_pre (..., (void*) cs, err, ...);
const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
cudaStreamDestroy(cs);
if (cudaGraphInstantiate(out, graph, 0) != cudaSuccess) { ... }
cudaGraphDestroy(graph);
```

and `session_replay` (per token, per layer):

```cpp
// src/core/session.cpp:282-289
stage_token(g, pos, pos_base, s);                              // host writes the pinned per-token staging
for (int64_t l = 0; l < g.n_layers; ++l) cudaGraphLaunch(gr.execs[l], cs);
```

`mtp.cpp`'s `finish_capture` adds `cudaGraphUpload(exec, cs)` + `cudaStreamSynchronize(cs)`; `verify.cpp`
additionally calls `cudaGraphGetNodes` / `cudaGraphNodeGetType` / `cudaGraphKernelNodeGetParams` (a diagnostic
under `STRATA_VERIFY_NODES`). `session_graphs_free` / the `mtp` and `verify` destructors call
`cudaGraphExecDestroy`.

### Recording, not graph-level semantics

Every capture is a **Recording** (`begin` → launch a body of the layer's ops → `end`) with the SAME stream
captured once and replayed many times. There is **no** use of:

* node dependencies expressed in the graph (the engine never adds nodes by hand),
* cross-stream capture (every `cudaStreamBeginCapture` is on one stream; no `cudaGraphAddDependencies`),
* event/wait nodes inside a capture — `cudaEventRecord` appears only BETWEEN two `cudaGraphLaunch` calls
  (`session.hpp`: "An event recorded BETWEEN two `cudaGraphLaunch` calls IS valid"), and
  `cudaStreamWaitEvent` only in the `verify`/`overlap` split paths,
* `cudaGraphExecUpdate` (not referenced anywhere),
* memory-node aliasing.

The graph-level semantics the engine DOES rely on are the two the header documents:

1. **A replay re-reads its input BUFFERS but not its kernel ARGUMENTS.** `graph.hpp` NOTE 2: "eight replays
   with a different pinned input each time all tracked correctly, so the data path is live; but a kernel
   argument is copied into the node at capture and is never re-read. Anything that changes per step - a
   position, a page-table base, a token id - must be DATA in a device buffer, never an argument."
2. **Fixed addresses are the caller's obligation.** `session.hpp`: "the graphs captured the SOURCE POINTER, not
   the value, and that is exactly why the buffers are pinned and fixed."

### How many nodes a step records, and what the host mutates between replays

A whole block is ~43 nodes (`session.hpp`: "Replaying the 43-node block graph measured 1.585 ms against 2.393 ms
for direct launches"); two graphs per layer (`pre`, `post`). Between replays the host calls `stage_token`
(`src/core/session.cpp:177`), which rewrites the **pinned staging buffers** `q.host_step` / `q.host_pos`; the
recorded body reads them through `copy_i32_from_mapped` into the device buffers `st.step` / `st.pos_dev`
(`layer.cpp:911-914`, guarded by `g_publish_kernel`, **default true**, `layer.cpp:42`). It does NOT change any
pointer: the addresses are allocated before capture and never reallocated.

### What a replay-only implementation would break

"Record the command sequence, replay it" is the right model for this engine — with two conditions the port must
meet, and one it cannot:

* **The capture must RECORD, not RUN.** On CUDA a capture does not execute the body. The port's
  `Ctx::dispatch` submits AND waits, so if a capture let it execute, `session_capture` would advance every
  layer's recurrent state (GDN `state`) and the residual (`gr_write`) once before the first token — a silently
  wrong first token. The port therefore diverts `dispatch()` to `record_dispatch()` while capturing.
* **The replay must re-read DATA.** The port's recording bakes `(VkBuffer, offset)`, not contents, so a replay
  reads the live buffer — matching CUDA.
* **A host<->device copy inside a capture cannot be recorded.** The port stages every host transfer through
  host memory (the arena is DEVICE_LOCAL, unmappable), so it has no pure device command. CUDA records a memcpy
  node whose host source is re-read at replay; the port cannot, and REFUSES (below). In the shipped default this
  is not reached: `g_publish_kernel` is true, so the per-token step state arrives through the
  `copy_i32_from_mapped` KERNEL, not an H2D copy.

---

## B. THE DESIGN: a graph IS the port's recorded step

The port already records a step: `Ctx::record_begin` / `record_dispatch` / `record_end_and_submit` /
`replay_recorded` (`vk_compute.hpp`), one persistent primary command buffer with a compute→compute barrier
between dispatches and one closing host-read barrier, re-submitted by every replay. The CUDA graph API is
mapped ONTO that — the same encoder (`encode_dispatch`), the same submission (`vkQueueSubmit` + fence wait).
There is **no second mechanism**; the only new primitive is the DIVERSION that makes a capture record.

| CUDA | here |
|---|---|
| `cudaStreamCreate` / `WithFlags` | the shim's current stream (one device; the flag is accepted, stated as meaningless on one queue). `cudaStreamDestroy` releases the handle, never the device. |
| `cudaStreamBeginCapture(stream, mode)` | `Ctx::capture_begin()`: begin a fresh recording and set `capture_`. The mode is accepted and all three behave alike (one queue). |
| a kernel launch in the body | `Ctx::dispatch` sees `capture_` and calls `record_dispatch` — **records, does not submit**. |
| `cudaMemcpyAsync(D2D)` in the body | recorded as one `vkCmdCopyBuffer` region + a transfer→compute barrier (`Ctx::capture_copy`). |
| `cudaStreamEndCapture(stream, &graph)` | `Ctx::capture_end()` closes the recording (the same closing host-read barrier, NO submit) and the shim takes ownership of the command buffer + fence into a `cudaGraph_t`. A capture that recorded nothing, or that hit an unrecordable op, is `cudaErrorStreamCaptureUnsupported`. |
| `cudaGraphInstantiate(&exec, graph, 0)` | the `cudaGraphExec_t` takes the recording (graph becomes a defunct handle). |
| `cudaGraphLaunch(exec, stream)` | `Ctx::submit_owned(cb, fence)` — **the port's own recorded-step submit**, which submits and WAITS the fence, so a call that returns is a call whose device work has completed. |
| `cudaGraphDestroy` / `cudaGraphExecDestroy` | free the command buffer + fence (`Ctx::destroy_owned`). |
| `cudaGraphUpload` | a no-op (the recording is host-side command buffers). |
| `cudaGraphGetNodes` / `cudaGraphNodeGetType` | the recording's dispatch + copy counts and kinds. |
| `cudaGraphKernelNodeGetParams` | `cudaErrorNotSupported` (no CUDA kernel params are kept). |

### What is PRESERVED

* Record once, replay many; a replay **re-reads the contents** of the buffers bound at capture.
* A **capture does not run** the body.
* **Bitwise identity** between a replayed recording and direct execution of the same dispatches (measured, §C).
* `SessionGraphs`-shaped use: N per-layer graphs, `cudaGraphLaunch` per layer, events between launches.

### What is NOT preserved (stated in the shim header, not hidden)

* **One queue.** The engine's multi-stream concurrency (cross-stream capture, events ordering two streams) is
  not expressible; `cudaStreamCreate` returns the one device.
* **A host↔device transfer or a memset inside a capture is REFUSED** (`cudaErrorStreamCaptureUnsupported`),
  because neither is a pure device command here. The shipped default does not reach either (`g_publish_kernel`
  true; the memset sites are in session init).
* **Changed pointers are not seen**, exactly as CUDA: the recording names the buffer it captured. The engine's
  contract is fixed addresses; the port cannot detect a violation of it.
* **`cudaGraphInstantiate` flags, `cudaGraphExecUpdate`, node dependencies, upload** are not modelled.
* The `cudaGraphInstantiate` 3-arg and 5-arg toolkit signatures both exist (C++ overloads).
