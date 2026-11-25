# Apple Silicon Benchmarking Methodology and Quality Gates

Performance tuning without strict measurement discipline produces phantom wins caused by thermal drift, macOS background daemons, prompt-cache leakage, or silent quality collapse. Follow this protocol on every hill-climb.

## 1. macOS Noise Hygiene Checklist

Before running any benchmark or A/B comparison, verify all five conditions via `scripts/probe_apple_silicon.sh`:

- **AC Power and Power Mode**: Plug the Mac into wall power. Battery mode or Low Power Mode (`pmset -g | grep lowpowermode` returning `1`) causes 8-15% throughput swings and aggressive GPU clock capping.
- **Thermal Equilibrium**: Check `pmset -g therm`. `CPU_Speed_Limit` must be `100` with no thermal warning. On fanless MacBook Airs or thin laptops, run a 60-second warm-up or allow 2-3 minutes of cooldown between runs so both control and candidate see the same thermal envelope.
- **Prevent Sleep and App Nap**: Wrap long benchmark suites with `caffeinate -dims` so macOS does not throttle backgrounded processes or enter dark wake.
- **Background Load**: Ensure Spotlight (`mds_stores`), Time Machine (`backupd`), Xcode indexing, and browser GPU tabs are idle. Close heavy Metal apps.
- **Single-GPU Queueing**: Always run benchmarks through `scripts/queue.sh` so two agents or shells never execute concurrently against the same unified-memory GPU.

## 2. Interleaved Control / Candidate / Control Protocol

Never compare a candidate run taken at 14:30 against a baseline taken at 14:00. Thermal soak, OS page-cache warmth, and background activity drift over time.

Choose the mode that matches your unified memory headroom:

- **Dual-server interleaved (`bench.py ab`)**: Only when `2 * (weights + KV)` fits comfortably under 70% of RAM with zero swap. Runs `Cycle 1..N: Control -> Candidate -> Control` automatically.
- **Sequential single-server triad (`bench.py baseline` + `bench.py compare`)**: Preferred on unified memory when running large models (30B-500B). Run `control-pre` -> restart server with candidate -> run `candidate` -> restart server with control -> run `control-post`, then score the triad with `bench.py compare` so drift and verdict (`KEEP` / `REJECT` / `NOISY`) are computed identically.

Rules for valid comparison:

- **Warm-up**: Discard the first 1-2 requests after server startup so Metal shader compilation (`MTLLibrary` pipeline cache), weight paging, and `mx.compile` tracing are excluded from steady-state measurements (track cold startup separately if startup latency is a target metric).
- **Prevent Prompt-Cache Cheating**: Vary a short random prefix nonce on every prompt when measuring raw prefill TTFT, unless you are explicitly benchmarking prefix-cache hit rate (`--allow-cache`).
- **Server-Reported Token Counts**: Read `usage.completion_tokens`, `usage.prompt_tokens`, and llama.cpp `timings.predicted_per_second` / `timings.prompt_per_second` from the OpenAI-compatible response. Do not estimate tokens from string length.
- **Metric Direction**: Use `--metric decode_tps` or `--metric prefill_tps` (higher is better) for throughput levers, and `--metric ttft_s` (lower is better) for latency/prefill levers.
- **Robust Statistics**: Report medians and MAD across runs, not fragile single-run means.
- **Promotion Bar**: A candidate wins only if:
  1. Median improvement exceeds control-to-control drift AND is >= 3% on the target metric (or reduces memory footprint materially at equal speed).
  2. Peak unified memory stays under the memory cap with zero swap growth.
  3. Quality and stability gates pass.

## 3. macOS Unified Memory Telemetry

Linux tools (`/proc/<pid>/fdinfo`, `nvidia-smi`, cgroup anon counters) do not exist on macOS. Track five signals together using `scripts/mem_macos.py` (or pass `--mem-pid <pid>` directly to `scripts/bench.py`):

- **Wired GPU limit (`sysctl iogpu.wired_limit_mb`)**: The kernel cap on non-pageable GPU allocations.
- **System memory free percentage (`memory_pressure -Q`)**: Cleanest macOS indicator of impending swap/compression cliffs.
- **Process RSS**: Process resident memory via `ps`.
- **System compressed memory and swapouts (`vm_stat` / `sysctl vm.swapusage`)**: Any sustained rise in `Pages occupied by compressor` or `swap_used` during inference signals you have crossed the safe unified-memory envelope.
- **MLX allocator telemetry (when using MLX)**: `mx.get_active_memory()`, `mx.get_peak_memory()`, and `mx.get_cache_memory()`. Always inspect `cache_memory` separately from `active_memory` because MLX caches freed Metal buffers until `mx.clear_cache()` or `mx.set_cache_limit(...)` eviction.

## 4. Experiment Tracking (`PLAN.md` and `ledger.jsonl`)

Maintain two lightweight files in the optimization working directory:

- **`PLAN.md`**: Records the hardware profile, theoretical ceiling, memory cap, baseline numbers, ranked hypothesis list, and a short table of kept vs. rejected levers with reasons.
- **`ledger.jsonl`**: Append-only JSON Lines log written automatically by `scripts/bench.py` containing timestamp, mode, label, TTFT, prefill tok/s, decode tok/s, ITL p50/p95, pre/post memory and swap deltas, and gate pass/fail status.

Never delete rejected runs from `ledger.jsonl`. Recording dead ends prevents future sessions from re-testing failed flags.

## 5. Mandatory Anti-Cheat Qualification Gates

Speedups that silently corrupt output or crash under load are invalid. Before finalizing any configuration that changes quantization, KV cache precision, attention kernels, speculative decoding, or concurrency, pass all three gates:

1. **Long-Context Needle Retrieval Gate (`bench.py needle`)**:
   - Embed a unique synthetic key-value fact at 10%, 50%, and 90% depth inside non-repeating filler paragraphs sized to 50% and 90% of the target context window.
   - Use `--max-tokens 512` (or higher for reasoning/thinking models that emit `<think>` tokens before the final key) and require 100% exact retrieval at temperature `0.0`.
   - Catches silent KV quantization corruption (`q4_0` KV breakage on sensitive models), broken RoPE scaling, and unintended context shifting.

2. **Concurrency and KV Soak Gate (`bench.py concurrency --mem-pid <pid>`)**:
   - Sweep concurrency (`1, 2, 4` or target max slots) with non-trivial prompt lengths and output lengths.
   - Verify zero HTTP errors and check pre/post `mem_delta` in the ledger to confirm swap does not grow and memory stabilizes.

3. **Sustained Thermal Gate (`bench.py soak --duration-s 180 --mem-pid <pid>`)**:
   - Run 3-5 minutes of continuous generation and compare final-minute median decode tok/s against first-minute median decode tok/s.
   - Flag if throughput drops > 5% or `pmset -g therm` reports throttling. If thermal throttling occurs on a laptop, record both burst and sustained speeds honestly.
