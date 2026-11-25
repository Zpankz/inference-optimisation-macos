# Apple Silicon Inference Gotchas and Failure Modes

Diagnose unexpected slowdowns, crashes, or noisy benchmarks against these known macOS, Metal, MLX, and llama.cpp failure modes.

## 1. Coalesced Single-Chunk SSE Responses Faking 100k+ tok/s

- **Symptom**: A broken flag (for example, `activation_fp16_enabled: true` in `oMLX` on an M1 Max) causes the model to degenerate to `!!!!` x 64 and deliver the entire response in a single SSE chunk (`t_end - t_first < 1 ms`), reporting 100k-863k tok/s and beating healthy 20 tok/s controls.
- **Root cause**: Dividing `(completion_tokens - 1) / gen_s` when `gen_s` is microseconds explodes; word-level loop detectors miss single-token punctuation floods.
- **Fix**: `bench.py` automatically flags and excludes `COALESCED` (`gen_s < 50 ms`), `DEGENERATE` (`< 5%` unique characters or tight 4-gram loops), and `IMPLAUSIBLE` (`> 2500 tok/s`) samples from medians. Never disable these guards to make a run look faster.

## 2. Silent `iogpu.wired_limit_mb` Wall and Reboot Reset

- **Symptom**: Model theoretically fits in unified RAM (e.g., 42 GB model on a 64 GB Mac), but Metal fails allocation, crashes with `kIOGPUCommandBufferCallbackErrorOutOfMemory`, or suddenly drops from 30 tok/s to 2 tok/s.
- **Root cause**: macOS caps wired GPU memory below physical RAM by default, and any `sudo sysctl iogpu.wired_limit_mb=...` setting resets to default after a reboot.
- **Fix**: Inspect `sysctl iogpu.wired_limit_mb` first. Raise it to 75-80% of physical RAM (leaving macOS headroom), and if running a dedicated server Mac, persist it via a login script or LaunchDaemon.

## 3. Over-Allocating Wired Memory Triggers Swap/Thermal Cliffs

- **Symptom**: Setting `iogpu.wired_limit_mb` to 90-95% of physical RAM lets a huge model load, but system latency spikes, SSH/UI freezes, or sustained decode degrades rapidly.
- **Root cause**: macOS kernel, WindowServer, and filesystem buffers cannot use wired GPU memory; starving them triggers aggressive memory compression and SSD swap thrashing.
- **Fix**: Never starve macOS. Leave at least 20-25% unwired on 32-64 GB Macs and 16-32 GB unwired on 128-512 GB Macs.

## 4. Headless Mac Studio GPU Clock Stuck at Idle Frequencies

- **Symptom**: A Mac Studio running an MLX or oMLX server via a system LaunchDaemon (with FileVault / SSH and no GUI login) suddenly decodes 10x-25x slower than normal (e.g., 5 tok/s instead of 120 tok/s). Restarting the inference server does not fix it; only a reboot or waking the display session restores full GPU clocks (e.g., ~1,600 MHz instead of 338-620 MHz).
- **Root cause**: macOS GPU power management can latch into a low-frequency state on headless daemon sessions.
- **Fix**: Verify GPU clock/power draw with `macmon` or `sudo powermetrics --samplers gpu_power -n 1`. Mitigate by running under an active user session with `caffeinate -dims` (or HDMI dummy plug) and always checking clock frequency if baseline decode falls below 30% of ceiling.

## 5. Battery Mode, Low Power Mode, and Fanless Thermal Drift

- **Symptom**: Identical configurations vary by 8-15% between morning and afternoon, or the candidate run always looks slower than the earlier control run.
- **Root cause**: Unplugged battery profiles, Low Power Mode, or thermal soak (especially on fanless MacBook Airs or compact Mac minis under sustained prefill).
- **Fix**: Plug into AC, disable Low Power Mode, check `pmset -g therm`, calibrate your A/A noise floor (`bench.py aa`), and always use interleaved `control -> candidate -> control` trials (or `bench.py compare` for sequential triads).

## 6. Looped MoE Models Overstating Ceilings by ~2x

- **Symptom**: A looped transformer (e.g., 24 physical layers whose 12-layer core runs 3 passes = 48 layer visits) looks stuck at ~35-45% of its theoretical ceiling even when well-tuned.
- **Root cause**: Standard `active_params` counts each physical weight once, missing repeated weight reads across loop iterations.
- **Fix**: Pass `--loop-multiplier` (for example, `2.0` for a 48-visit / 24-layer looped model) to `scripts/ceiling.py`.

## 7. MLX Lazy Evaluation and Unsynchronized Timers

- **Symptom**: Custom MLX benchmarks report impossible speeds (thousands of tok/s or sub-millisecond forward passes), or memory grows linearly until OOM in a loop.
- **Root cause**: MLX builds a lazy computation graph and only executes on `mx.eval(...)` or `mx.synchronize(...)`. Stopping a Python timer before `mx.eval()` measures graph queueing, not GPU work; failing to evaluate periodically accumulates huge graphs in memory.
- **Fix**: Call `mx.eval(model.parameters())` after model load, call `mx.eval(output)` or `mx.synchronize()` before stopping timers, and use `mx.get_active_memory()`, `mx.get_peak_memory()`, and `mx.get_cache_memory()` (not deprecated `mx.metal.*` aliases) for memory profiling.

## 8. `mx.compile` Silent Retracing and `ensure_row_contiguous` Copies

- **Symptom**: Adding `@mx.compile` makes decode slower or produces periodic 100-500 ms hiccups.
- **Root cause**: `mx.compile` retraces whenever input shapes, dtypes, ranks, or argument counts change (such as a growing KV cache sequence length). Similarly, `mx.fast.metal_kernel(..., ensure_row_contiguous=True)` silently inserts memory-copy kernels on transposed tensors.
- **Fix**: Pad/bucket sequence shapes or test `shapeless=True` carefully where safe, inspect compile cache hits, and verify tensor contiguity before entering custom Metal kernels.

## 9. KV Cache Quantization Regressions

- **Symptom**: Enabling `--cache-type-k q8_0 --cache-type-v q8_0` in `llama-server` or `--kv-bits 8` in `mlx-lm` slows down decode (for example, dropping 45 tok/s to 27 tok/s) or degrades long-context needle recall.
- **Root cause**: On models with few KV layers (e.g., hybrid/linear-attention or aggressive GQA models) or short contexts below 10% of `kv_crossover_ctx`, KV bandwidth is tiny compared to weights, so dequantization compute overhead outweighs bandwidth savings. Aggressive 4-bit KV can also break precise needle retrieval.
- **Fix**: Treat KV quantization as a hypothesis, not an automatic default. Always A/B test decode speed and run `bench.py needle` at 50% and 90% of target context.

## 10. Speculative Decoding / MTP Bandwidth Trap

- **Symptom**: Enabling 3+ MTP drafts or a draft model makes single-stream decode slower on an M1-M4 base Mac, or `k=5` slows down fresh prose on a Max/Ultra Mac.
- **Root cause**: Base M-series chips (68-120 GB/s) are tightly memory-bandwidth bound; even on high-bandwidth chips, deep drafts punish low-acceptance fresh prose.
- **Fix**: Sweep draft count `k=0, 1, 2, 3, 4, 5` with interleaved A/B trials on prompts matching your real output mix.

## 11. `llama-server` macOS Load Hang (`--no-mmap`) and `-c 0` OOM

- **Symptom**: `llama-server` hangs around 75% model loading on macOS, or immediately OOMs on startup with `-c 0`.
- **Root cause**: Certain macOS versions/model sizes hit an `mmap` paging stall during load; `-c 0` allocates the model's full theoretical context window (often 128K-256K) in KV cache immediately.
- **Fix**: Always pass an explicit `-c <ctx>`. Add `--no-mmap` only if you observe the load hang.

## 12. Reasoning Models Truncated in Short Needle Gates

- **Symptom**: A thinking/reasoning model (Qwen 3+, GLM 4+) fails `bench.py needle` even when retrieval is intact.
- **Root cause**: Setting `--max-tokens 64` cuts off the model mid-`<think>` block before it prints the final verification code.
- **Fix**: Run `bench.py needle` with `--max-tokens 512` (the default) or higher for reasoning models, and verify whether the code appears in the completed answer.
