# Apple Silicon Benchmarking Methodology and Quality Gates

Performance tuning without strict measurement discipline produces phantom wins caused by thermal drift, macOS background daemons, coalesced SSE chunks, prompt-cache leakage, or silent quality collapse. Follow this protocol on every hill-climb.

## 1. macOS Noise Hygiene Checklist

Before running any benchmark or A/B comparison, verify all five conditions via `scripts/probe_apple_silicon.sh`:

- **AC Power and Power Mode**: Plug the Mac into wall power. Battery mode or Low Power Mode (`pmset -g | grep lowpowermode` returning `1`) causes 8-15% throughput swings and aggressive GPU clock capping.
- **Thermal Equilibrium**: Check `pmset -g therm`. `CPU_Speed_Limit` must be `100` with no thermal warning. On fanless MacBook Airs or thin laptops, run a 60-second warm-up or allow 2-3 minutes of cooldown between runs so both control and candidate see the same thermal envelope.
- **Prevent Sleep and App Nap**: Wrap long benchmark suites with `caffeinate -dims` so macOS does not throttle backgrounded processes or enter dark wake.
- **Background Load**: Ensure Spotlight (`mds_stores`), Time Machine (`backupd`), Xcode indexing, and browser GPU tabs are idle. Close heavy Metal apps.
- **Single-GPU Queueing**: Always run benchmarks through `scripts/queue.sh` (portable `fcntl.flock` in a private `0700` per-user directory) so two agents or shells never execute concurrently against the same unified-memory GPU.

## 2. Calibrate Your A/A Noise Floor Before Trusting "Beats Both Controls"

With nothing changed, the middle run of any three beats both neighbors about 1 in 3 times purely by random noise (verified on an Apple M4 Max: 15 of 43 unchanged triads, up to 1.54% apparent gain).

Run 10+ unchanged baseline runs first, then run:

```bash
python3 "$SKILL_DIR/scripts/bench.py" aa --ledger ledger.jsonl --label "ctl-pre-1"
```

Require every promoted candidate to beat the control by MORE than `recommended_min_gain_pct` (and at least 3% default).

## 3. Interleaved Control / Candidate / Control Protocol

Never compare a candidate run taken at 14:30 against a baseline taken at 14:00. Thermal soak, OS page-cache warmth, and background activity drift over time.

Choose the mode that matches your unified memory headroom:

- **Dual-server interleaved (`bench.py ab`)**: Only when `2 * (weights + KV)` fits comfortably under 70% of RAM with zero swap. Runs `Cycle 1..N: Control -> Candidate -> Control` automatically.
- **Sequential single-server triad (`bench.py baseline` + `bench.py compare`)**: Preferred on unified memory when running large models (30B-500B). Run `control-pre` -> restart server with candidate -> run `candidate` -> restart server with control -> run `control-post`, then score the triad with `bench.py compare`.

Rules for valid comparison:

- **Warm-up**: Discard the first 1-2 requests after server startup so Metal shader compilation (`MTLLibrary` pipeline cache), weight paging, and `mx.compile` tracing are excluded from steady-state measurements.
- **Tokenizer-Calibrated Prompt Sizing**: Pass `--calibrate-tokenizer` (on by default in `needle`) so `bench.py` measures actual `chars_per_token` from the server's tokenizer rather than under-filling prompts by 20-30%.
- **Prevent Prompt-Cache Cheating**: Vary a short random prefix nonce on every prompt when measuring raw prefill TTFT, unless you are explicitly benchmarking prefix-cache hit rate (`--allow-cache`).
- **Sample Validity Guards**: `bench.py` automatically flags and excludes `COALESCED` (<50 ms decode window), `DEGENERATE` (char-flood `< 5%` unique chars or tight n-gram loops), and `IMPLAUSIBLE` (`> 2500 tok/s`) samples from medians so broken configs cannot win a sweep.
- **Metric Direction**: Use `--metric decode_tps` or `--metric prefill_tps` (higher is better) for throughput levers, and `--metric ttft_s` (lower is better) for latency/prefill levers.

## 4. macOS Unified Memory Telemetry

Linux tools (`/proc/<pid>/fdinfo`, `nvidia-smi`, cgroup anon counters) do not exist on macOS. Track six passwordless signals together using `scripts/mem_macos.py` (or pass `--mem-pid <pid>` directly to `scripts/bench.py`):

- **Live Metal GPU memory (`ioreg -r -c IOAccelerator`)**: Passwordless `"In use system memory"` and `"Alloc system memory"` directly from the Apple GPU driver.
- **Wired GPU limit (`sysctl iogpu.wired_limit_mb`)**: The kernel cap on non-pageable GPU allocations.
- **System memory free percentage (`memory_pressure -Q`)**: Cleanest macOS indicator of impending swap/compression cliffs.
- **Process RSS**: Process resident memory via `ps`.
- **System compressed memory and swapouts (`vm_stat` / `sysctl vm.swapusage`)**: Any sustained rise in `Pages occupied by compressor` or `swap_used` signals you crossed the safe unified-memory envelope.
- **MLX allocator telemetry (when using MLX)**: `mx.get_active_memory()`, `mx.get_peak_memory()`, and `mx.get_cache_memory()`.

## 5. Experiment Tracking (`PLAN.md`, `DEAD-ENDS.md`, and `ledger.jsonl`)

Maintain three lightweight files in the optimization working directory:

- **`PLAN.md`**: Hardware profile, measured Metal BW, theoretical ceiling, memory cap, baseline numbers, and ranked hypothesis queue.
- **`DEAD-ENDS.md`**: Short, reusable log of rejected levers and why they failed so future sessions never re-test dead ends.
- **`ledger.jsonl`**: Append-only JSON Lines log written automatically by `scripts/bench.py` with timestamps, metrics, sample flags, pre/post memory and GPU residency deltas, and gate verdicts.

## 6. Mandatory Anti-Cheat Qualification Gates

Before finalizing any configuration that changes quantization, KV cache precision, attention kernels, speculative decoding, or concurrency, pass all three gates:

1. **Long-Context Needle Retrieval Gate (`bench.py needle`)**:
   - Embed a unique synthetic key-value fact at 10%, 50%, and 90% depth inside tokenizer-calibrated, non-repeating filler paragraphs sized to 50% and 90% of the target context window.
   - Use `--max-tokens 512` (or higher for reasoning/thinking models that emit `<think>` tokens before the final key) and require 100% exact retrieval at temperature `0.0`.

2. **Concurrency and KV Soak Gate (`bench.py concurrency --mem-pid <pid>`)**:
   - Sweep concurrency (`1, 2, 4` or target max slots) with non-trivial prompt lengths and output lengths.
   - Verify zero HTTP errors, zero flagged samples, and zero swap growth in `mem_delta`.

3. **Sustained Thermal Gate (`bench.py soak --duration-s 180 --mem-pid <pid>`)**:
   - Run 3-5 minutes of continuous generation and compare late-window median decode tok/s against early-window median decode tok/s.
   - Flag if throughput drops > 5% or `pmset -g therm` reports throttling.
