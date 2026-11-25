# Apple Silicon Inference Optimisation Lever Playbook

Test levers in rough order of expected impact and low risk. Change one lever at a time and verify against interleaved controls above your A/A noise floor.

## Layer 1: macOS and Metal System Levers

### 1.1 Wired GPU Memory Limit (`iogpu.wired_limit_mb`)

- **Why it matters**: macOS caps wired GPU memory conservatively by default (often ~66-75% of RAM). Large models or long contexts hit this wall before physical RAM is full, causing Metal OOM or silent fallback/swap cliffs.
- **How to set (non-persistent across reboots)**:

```bash
# Example on a 64 GB Mac (75% = 49152 MB, leaving 16 GB for macOS)
sudo sysctl iogpu.wired_limit_mb=49152
```

- **Safe ceiling**: Keep at or below 75-80% of physical RAM on 32-64 GB Macs, or leave at least 16-32 GB unwired on 128-512 GB Mac Studios. Over-allocating starves macOS kernel and WindowServer, causing UI freezes and worse tail latency.
- **In MLX Python processes**: Pair with `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` (or `mlx_lm.wired_limit(...)`) so the Metal allocator pins active weights cleanly.

### 1.2 Power, Sleep, and Scheduling Hygiene

- Run long server benchmarks under `caffeinate -dims` to block App Nap and idle sleep.
- Disable Low Power Mode (`sudo pmset -a lowpowermode 0` on AC) before measuring.
- On `llama-server`, test `--prio 2` (high scheduling priority) to reduce jitter from background macOS tasks.

---

## Layer 2: MLX Ecosystem Levers (`mlx-lm`, `vllm-mlx`, `oMLX`)

### 2.1 Engine Choice First

Before tuning micro-flags, screen 2-3 engines with `bench.py screen`. Real Apple Silicon measurements (e.g., 30-core M2 Max 64 GB running Inkling-2-Mini) show stock engines spread from 31.1 to 35.7 tok/s out of the box, with fused Metal kernels pushing decode to 37.0 tok/s solo, 59.2 tok/s at 4 concurrent users, and 71.2 tok/s on code edits.

### 2.2 Prompt / Prefix Caching

- **Impact**: Massive TTFT reduction (5x-25x, e.g., 20.5 s down to 0.71 s at 21K context on M2 Max) for agentic workflows, coding assistants, and multi-turn chat sharing a long system prompt or repository context.
- **How**: Use `mlx_lm` prompt caches (`make_prompt_cache`, `save_prompt_cache`, `load_prompt_cache`) or server-level prefix caching in `vllm-mlx` / `oMLX`. Verify cache hits do not leak into raw prefill benchmarks.

### 2.3 Allocator Cache Limit (`mx.set_cache_limit`)

- **Impact**: Prevents MLX's Metal buffer cache from retaining gigabytes of freed scratch buffers across long-running concurrent server sessions, avoiding creeping memory pressure without slowing steady-state decode.
- **How**: Monitor `mx.get_cache_memory()` alongside `mx.get_active_memory()`. If cache memory ratchets up under concurrency, cap it via `mx.set_cache_limit(<bytes>)` or call `mx.clear_cache()` between heavy sessions.

### 2.4 KV Cache Quantization (Apply KV-Crossover Rule)

- **Impact**: Cuts KV cache memory by ~2x (8-bit) to ~4x (4-bit), unlocking longer contexts and higher concurrency.
- **When it speeds up decode vs. hurts decode**: Check `kv_crossover_ctx` from `scripts/ceiling.py`. When active context is >= 25% of `kv_crossover_ctx`, KV quantization reduces memory traffic enough to accelerate decode. When active context is < 10% of `kv_crossover_ctx`, dequantization overhead can slow decode down.
- **Flags (`mlx_lm`)**: `--kv-bits 8` (safe quality default) or `--kv-bits 4`, with `--kv-group-size 64` and `--quantized-kv-start 512` (keeps short prompts in FP16 to avoid dequant overhead when KV is tiny).
- **Gate**: Always A/B test decode tok/s and pass `bench.py needle`.

### 2.5 Compilation and Retracing Hygiene (`mx.compile`)

- **Impact**: Fuses elementwise ops and reduces Metal command-buffer dispatch overhead on small or MoE models.
- **Key rule**: `mx.compile` retraces whenever input dtype, rank, arity, or shape changes. Use `shapeless=True` only on shape-agnostic blocks (verify correctness first, as dynamic slicing or broadcasts can break under `shapeless=True`).
- **Async pipelining**: In custom decode loops, overlap token sampling/CPU readback with the next GPU step using `mx.async_eval(...)` instead of calling blocking `mx.eval(...)` synchronously on every scalar token.

### 2.6 MLX Kernel Escalation Ladder

Follow this strict order when profiling reveals an op-level bottleneck in MLX:

1. **Stock MLX ops** with clean broadcasting and no accidental host syncs.
2. **Built-in fused Metal kernels (`mx.fast.*`)**: Prefer `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`, `mx.fast.layer_norm`, and `mx.fast.rope` over unfused compositions.
3. **Custom Python-level Metal kernel (`mx.fast.metal_kernel`)**: Use for Apple-only fused hot paths when stock ops are memory-bound by intermediate reads/writes. Watch `ensure_row_contiguous=True`, which can insert hidden copy kernels if inputs are transposed.
4. **C++ MLX Primitive extension**: Only when Python-level `metal_kernel` launch overhead or control flow is insufficient.
5. **GPU trace profiling**: Capture a `.gputrace` with `mx.metal.start_capture(path)` / `mx.metal.stop_capture()` and inspect shader occupancy, ALU vs. memory stall ratios, and command-buffer gaps in Xcode Metal Debugger.

---

## Layer 3: `llama.cpp` Metal Levers (`llama-server`)

### 3.1 The 7 High-Impact Apple Silicon Flags

Most `llama-server` flags come from x86/CUDA setups and do nothing on Apple Silicon. Focus your A/B tests on these seven:

1. **`-ngl 99` (`--n-gpu-layers 99`)**: Offloads all layers to Metal. On unified memory, partial CPU offloading almost always hurts because CPU and GPU share the same RAM pool anyway.
2. **`-fa on` (`--flash-attn on`)**: Enables the Metal Flash Attention kernel. Essential at 16K-32K+ context for both prefill speed and KV memory savings.
3. **`-c <ctx>` (`--ctx-size <ctx>`)**: Set explicitly to your real workload budget. Avoid `-c 0`, which allocates the model's full theoretical context (e.g., 128K-256K) up front and blows unified memory headroom.
4. **`-b 2048 -ub 2048` (`--batch-size`, `--ubatch-size`)**: Controls logical and physical prompt-processing batching (default is 512). Raising `-ub` to 1024-2048 often accelerates prefill TTFT on Max/Ultra GPUs; watch peak scratch memory and re-test on M5+ where Neural Accelerators shift the compute/memory balance.
5. **`--cache-type-k q8_0 --cache-type-v q8_0`** (or `-ctk q8_0 -ctv q8_0`): Cuts KV memory ~50%. Test `q4_0` only when RAM-constrained and gate strictly with `bench.py needle`. Apply the KV-crossover rule and A/B test against `f16` on short contexts or low-KV-layer architectures.
6. **`--mlock`**: Pins model weights and KV in physical RAM, preventing macOS from paging weights out during long sessions. Use only when you have comfortable memory headroom (otherwise it worsens OS pressure).
7. **`--prio 2`**: Raises scheduling priority to reduce background macOS jitter.

### 3.2 Situational `llama-server` Flags

- **`--no-mmap`**: Do not enable preemptively. Use only if you hit the known macOS hang around 75% model load.
- **`--jinja`**: Required for accurate tool calling and thinking/reasoning templates on modern models (Qwen 3+, GLM 4+).
- **`--keep N`**: Pins the first N system-prompt tokens during context shift for long-running agents.
- **`--parallel N` (`-np N`)**: Splits `-c` across `N` concurrent slots. On M4 Pro (48 GB), pinning 2 parallel slots boosted aggregate decode +60% (18.1 -> 29.5 tok/s), whereas 4 slots added no further gain.
- **Draftless / n-gram speculative decoding**: Useful on repetitive code-editing workloads where context-history lookup adds speed without loading a second model into unified memory.

### 3.3 Placebo or Counterproductive Flags on Apple Silicon (Strip These)

- **`--cpu-mask`, `--cpu-strict`, `--poll`**: Linux CPU affinity controls ignored or counterproductive under the macOS Grand Central Dispatch / P+E core scheduler.
- **`--no-kv-offload`**: Forces KV onto CPU-friendly layouts on unified memory, crippling Metal attention speed.
- **`-t N` exceeding P-core count**: Spills work onto E-cores that stall P-cores; on full Metal offload (`-ngl 99`), low thread counts (e.g., P-core count or fewer) are optimal.
- **CUDA/ROCm/Vulkan flags**: Ignored or unavailable on Metal builds.

---

## Layer 4: Speculative Decoding and Multi-Token Prediction (MTP)

Speculative decoding and native MTP are **both hardware-tier and workload-type dependent** on Apple Silicon:

- **Base M1-M4 chips (68-120 GB/s)**: Often **regress** with deep draft lengths (`k=3+`), because verifying multiple candidate tokens increases memory reads per step faster than token acceptance recovers (e.g., M4 dropping from ~6.35 to ~3.25 tok/s with 3 MTP drafts).
- **High-bandwidth Pro / Max / Ultra chips (273-1,228 GB/s)**: Strong wins when draft model size is small (<10% of target). Example measured on Apple M4 Max (`llama.cpp`, Qwen3.8-27B Q8_0 + Q4_0 draft, 65K ctx): depth-3 MTP jumped short decode from **15.29 -> 33.13 tok/s** and cut edit wall time from 10.67 s -> 6.31 s.
- **Draft depth depends on output type**:
  - **Code edits / copy-heavy apply-patch**: Deeper drafts (`k=4-5`) win +15-22% over `k=3`.
  - **Replayed real coding-agent turns**: `k=3` is near optimal (`k=4-5` lands within noise).
  - **Fresh creative prose**: Deep drafts (`k=4-5`) often regress; keep `k=2-3` or disable.
- **Memory rule**: On unified memory, `target_weights + draft_weights + dual_KV` must still stay safely inside `iogpu.wired_limit_mb`. Always A/B test on prompts matching your real output mix.
