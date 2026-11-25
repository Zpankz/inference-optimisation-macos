---
name: inference-optimisation-macos
description: Ceiling-first, evidence-gated local LLM inference optimisation for macOS Apple Silicon (M1 through M6 Base, Pro, Max, and Ultra) across MLX, mlx-lm, vllm-mlx, oMLX, and llama.cpp Metal. Use whenever tuning local Mac LLM serving speed, decode tok/s, prefill TTFT, KV cache footprint, speculative decoding or MTP, unified memory wired limits (iogpu.wired_limit_mb), Metal flash attention, mx.compile, custom Metal kernels, or diagnosing slow or throttling Mac inference servers.
---

# inference-optimisation-macos

Hill-climb local LLM serving speed on a single Apple Silicon Mac with mathematical ceilings, macOS-native unified-memory telemetry, interleaved A/B benchmarks, and anti-cheat quality gates. Every change is tested one lever at a time and kept only when it beats interleaved controls without breaking memory headroom, thermals, or long-context recall.

Note: Invoke bundled scripts using their path relative to this skill directory (for example, set `SKILL_DIR` to the folder containing this `SKILL.md`).

## When to Use

- Optimising decode throughput (tok/s), time to first token (TTFT), prefill speed, concurrency, or startup latency on an Apple Silicon Mac.
- Tuning `mlx_lm.server`, `vllm-mlx`, `oMLX`, `llama-server` (llama.cpp Metal), or custom MLX inference loops.
- Diagnosing why a Mac LLM server is slower than expected, hitting swap cliffs, stalling on Metal allocations, or throttling GPU clocks.
- Evaluating quantization, KV cache quantization, prompt caching, speculative decoding, MTP, or custom `mx.fast.metal_kernel` changes under controlled A/B tests.
- Building a reproducible benchmark ledger and qualification gate before deploying a local Mac model server.

## Core Principles

- **Ceilings before tuning**: Compute the theoretical memory-bandwidth decode ceiling, KV-crossover context, and wired-memory headroom first. If single-stream decode is already at 75-85% of bandwidth (or 55-72% for sparse MoE), switch from decode flags to prefill, prompt caching, concurrency, or model/quantization levers.
- **One lever at a time**: Never stack unmeasured flags. Change a single knob, run an interleaved `control -> candidate -> control` trial, and record both wins and rejections in `ledger.jsonl`.
- **Single-GPU serialization**: An Apple Silicon Mac has one unified-memory GPU. Always serialize benchmark runs through `scripts/queue.sh` so overlapping processes never corrupt memory bandwidth or residency readings.
- **Server-reported tokens only**: Measure decode and prefill rates from server-reported `usage` / `timings` fields alongside wall-clock TTFT, never client-side character or word heuristics.
- **No silent quality regressions**: Every candidate that touches quantization, KV cache precision, attention kernels, or speculative decoding must pass the long-context needle retrieval check, concurrent soak, and thermal stability gate before promotion.

## Steps

### 1. Probe Hardware, Thermals, and Runtimes

Run the environment probe before touching any server flags so you know the exact SoC, GPU core bin, RAM, wired GPU limit, power source, thermal state, memory pressure, and installed engines.

```bash
bash "$SKILL_DIR/scripts/probe_apple_silicon.sh"
```

Verify all `[PASS]` hygiene checks before benchmarking (full checklist in `references/methodology.md`):

- Plug the Mac into AC power and confirm Low Power Mode is off (`pmset -g | grep lowpowermode`).
- Confirm nominal thermal state (`pmset -g therm` reports `CPU_Speed_Limit = 100` and no thermal warning).
- Pause heavy background disk or GPU tasks (Spotlight indexing, Time Machine backups, video encoding).
- On headless Mac Studios running via LaunchDaemon, verify GPU clocks ramp under load rather than sticking at idle frequencies (see `references/gotchas.md`).

### 2. Calculate Hardware Ceilings and Memory Headroom

Compute the theoretical single-stream decode ceiling, KV-crossover context, KV cache footprint, and safe wired-memory budget using `scripts/ceiling.py` and `references/ceilings.md`.

```bash
python3 "$SKILL_DIR/scripts/ceiling.py" \
  --chip "M4 Max 40-core" \
  --ram-gb 128 \
  --active-params-b 3.5 \
  --total-params-b 35 \
  --bits 4.5 \
  --ctx 32768 \
  --kv-bytes-per-token 131072 \
  --concurrency 2
```

Use the ceiling output to classify your bottleneck before picking levers:

- **Memory-bandwidth bound (decode < 70% of ceiling for dense, < 55% for MoE)**: Focus on wired memory limits, Metal Flash Attention, weight/KV layout, `mx.compile` retracing, or redundant host-GPU syncs.
- **Already near bandwidth ceiling**: Stop chasing micro-flags for single-stream decode. Target active bytes per token (quantization, MoE routing), speculative decoding/MTP (only if bandwidth permits), prompt caching, or concurrency.
- **Compute bound (slow TTFT / prefill)**: Tune batch/micro-batch sizes (`-b`, `-ub`), Metal Flash Attention, prompt prefix caching, and chunked prefill.
- **Memory-cap bound (peak footprint > 75% of RAM or > `iogpu.wired_limit_mb`)**: Raise `iogpu.wired_limit_mb` safely (up to 75-80% of physical RAM on 32 GB+ Macs, leaving OS headroom), cap `--ctx-size`, or test KV cache quantization.

### 3. Capture a Verified Baseline

Create a `PLAN.md` file in your working directory stating the target model, hardware ceiling, memory cap, quality gate, and ranked hypothesis queue. Capture unified-memory telemetry alongside a baseline benchmark serialized via `scripts/queue.sh`:

```bash
bash "$SKILL_DIR/scripts/queue.sh" baseline -- \
  python3 "$SKILL_DIR/scripts/bench.py" baseline \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --label "baseline" \
    --runs 5 \
    --prompt-tokens 1024 \
    --max-tokens 256 \
    --mem-pid <server-pid> \
    --ledger ledger.jsonl
```

Compare baseline decode tok/s against `ceiling.py --observed-tps <tok_s>` to establish `% of theoretical ceiling`.

### 4. Hill-Climb One Lever at a Time

Consult `references/levers.md` for the ordered Apple Silicon lever playbook matching your runtime:

- **Layer 1 (macOS / Metal system)**: `sysctl iogpu.wired_limit_mb`, `caffeinate -dims`, process priority, power mode.
- **Layer 2 (MLX / mlx-lm / vllm-mlx / oMLX)**: `mx.set_wired_limit`, `mx.set_cache_limit`, prompt cache reuse, KV quantization (`--kv-bits`, `--quantized-kv-start`), `mx.compile` retracing audit, `mx.fast.*` and `mx.fast.metal_kernel` escalation ladder.
- **Layer 3 (llama.cpp Metal / llama-server)**: The 7 high-impact Apple Silicon flags (`-ngl 99`, `-fa on`, explicit `-c`, `-b`/`-ub`, `--cache-type-k/v`, `--mlock`, `--prio 2`) while stripping Linux/CUDA placebo flags.

Choose the interleaved A/B mode that fits your unified memory budget:

- **Dual-server interleaved A/B** (only when 2x model + KV fits comfortably under 70% RAM):

```bash
bash "$SKILL_DIR/scripts/queue.sh" ab-fa-on -- \
  python3 "$SKILL_DIR/scripts/bench.py" ab \
    --control-url http://127.0.0.1:8080/v1 \
    --candidate-url http://127.0.0.1:8081/v1 \
    --model "<model-id>" \
    --label "candidate-flash-attn-on" \
    --metric decode_tps \
    --cycles 3 \
    --max-tokens 256 \
    --ledger ledger.jsonl
```

- **Sequential single-server A/B** (standard for large models on unified memory): Run `bench.py baseline` in `control-pre -> candidate -> control-post` order (restarting cleanly between runs and verifying `mem_macos.py` shows swap and thermal pressure returned to baseline), then score the paired triad automatically:

```bash
python3 "$SKILL_DIR/scripts/bench.py" compare \
  --ledger ledger.jsonl \
  --control-pre-label "ctl-pre-1" \
  --candidate-label "cand-fa-on" \
  --control-post-label "ctl-post-1" \
  --metric decode_tps
```

Pass `--metric ttft_s` (lower is better) or `--metric prefill_tps` when tuning prefill/TTFT levers instead of decode throughput.

Promote a lever only if:

- Median improvement exceeds both measurement noise (control-vs-control drift) and at least 3% on the target metric.
- Peak unified memory stays inside the configured memory cap with zero swap growth.
- No secondary metric (TTFT, tail ITL, or concurrent throughput) regresses unacceptably.

Log every accepted and rejected change with exact numbers in `PLAN.md` and `ledger.jsonl`.

### 5. Run Qualification and Anti-Cheat Gates

Before declaring a tuned configuration production-ready, pass all three gates (details in `references/methodology.md`):

- **Long-context needle gate**: Verify exact key retrieval across 10%, 50%, and 90% depth inside non-repeating filler sized to your target context window.

```bash
bash "$SKILL_DIR/scripts/queue.sh" needle-gate -- \
  python3 "$SKILL_DIR/scripts/bench.py" needle \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --target-tokens 16384 \
    --depths 0.1 0.5 0.9 \
    --max-tokens 512 \
    --ledger ledger.jsonl
```

- **Concurrency and KV soak gate**: Sweep concurrency (`bench.py concurrency --levels 1 2 4 --mem-pid <pid>`) to confirm memory stabilizes with zero swap thrashing or Metal OOM.
- **Sustained thermal stability gate**: Run continuous generation (`bench.py soak --duration-s 180 --mem-pid <pid>`) to confirm final-minute decode tok/s stays within 5% of first-minute tok/s and `pmset -g therm` remains nominal.

### 6. Summarize and Hand Off

Produce a concise final report directly from `ledger.jsonl` containing:

- Hardware profile (chip, GPU core bin, RAM, bandwidth ceiling, `iogpu.wired_limit_mb`).
- Baseline vs. final metrics (decode tok/s, `% of ceiling`, prefill tok/s, TTFT, concurrent tok/s, peak unified memory, thermal droop %).
- Exact reproducible launch command and sysctl prerequisites.
- Table of tested levers (kept vs. rejected and measured delta) so future sessions never re-test dead ends.

## Gotchas

- **Default `iogpu.wired_limit_mb` causes silent swap or Metal OOM**: macOS caps wired GPU memory conservatively by default and resets it on reboot. Check `sysctl iogpu.wired_limit_mb` before blaming the engine, and never starve macOS (leave at least 20-25% RAM on 32-64 GB Macs, or 16-32 GB on 128-512 GB Macs).
- **Speculative decoding and MTP can cut decode speed in half on base M-series chips**: On bandwidth-limited chips (M1-M4 base, 68-120 GB/s), verifying multiple draft tokens per step can read more memory than acceptance saves. Always A/B test draft count (`k=1, 2, 3`) against `k=0`.
- **KV cache quantization is context-dependent**: `q8_0` or 4-bit KV helps decode speed only once KV read bytes become a meaningful fraction of active weight bytes (check `kv_crossover_ctx` in `ceiling.py`). Below that threshold, dequantization compute can slow decode down.
- **Linux scripts break on macOS**: `flock` is not installed by default on macOS, and `/proc/<pid>/fdinfo` does not exist. Always use `scripts/queue.sh` and `scripts/mem_macos.py`.
- **MLX lazy evaluation fakes benchmark numbers**: Timing MLX code without `mx.eval(...)` or `mx.synchronize(...)` measures graph construction, not GPU execution. Always synchronize before stopping timers and use `mx.get_*_memory()` rather than deprecated `mx.metal.*` aliases.
- **Read `references/gotchas.md`** when encountering Metal allocation failures, `mx.compile` retracing stalls, headless Mac Studio low-clock states, or `llama-server` mmap hangs.
