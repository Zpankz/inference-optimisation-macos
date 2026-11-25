---
name: inference-optimisation-macos
description: Ceiling-first, evidence-gated local LLM inference optimisation for macOS Apple Silicon (M1 through M6 Base, Pro, Max, and Ultra) across MLX, mlx-lm, vllm-mlx, oMLX, and llama.cpp Metal. Use whenever tuning local Mac LLM serving speed, decode tok/s, prefill TTFT, KV cache footprint, speculative decoding or MTP, unified memory wired limits (iogpu.wired_limit_mb), Metal flash attention, mx.compile, custom Metal kernels, or diagnosing slow or throttling Mac inference servers.
---

# inference-optimisation-macos

Hill-climb local LLM serving speed on a single Apple Silicon Mac with mathematical ceilings, native Metal bandwidth measurement, macOS unified-memory telemetry, interleaved A/B benchmarks, and anti-cheat quality gates. Every change is tested one lever at a time and kept only when it beats interleaved controls above the empirical A/A noise floor without breaking memory headroom, thermals, or long-context recall.

Note: Invoke bundled scripts using their path relative to this skill directory (for example, set `SKILL_DIR` to the folder containing this `SKILL.md`).

## When to Use

- Optimising decode throughput (tok/s), time to first token (TTFT), prefill speed, concurrency, or startup latency on an Apple Silicon Mac.
- Choosing the fastest engine on a new Mac (`mlx_lm.server` vs `vllm-mlx` vs `oMLX` vs `llama-server`) before investing in flag tuning.
- Diagnosing why a Mac LLM server is slower than expected, hitting swap cliffs, stalling on Metal allocations, or throttling GPU clocks.
- Evaluating quantization, KV cache quantization, prompt caching, speculative decoding, MTP, or custom `mx.fast.metal_kernel` changes under controlled A/B tests.
- Building a reproducible benchmark ledger and qualification gate before deploying a local Mac model server.

## Core Principles

- **Best engine before flag tuning**: On Apple Silicon, picking the right runtime (`mlx_lm.server`, `vllm-mlx`, `oMLX`, or `llama-server`) often beats the first ten flag tweaks. Use `bench.py screen` and `references/onboarding.md` on a new machine first.
- **Ceilings before tuning**: Measure real sustained unified-memory bandwidth (`bash scripts/measure_bw_metal.sh`) and compute the theoretical decode ceiling (`scripts/ceiling.py`) first. If single-stream decode is already at 75-85% of bandwidth (or 55-72% for sparse MoE), switch from decode flags to prefill, prompt caching, concurrency, or active-byte levers.
- **One lever at a time, above the A/A noise floor**: Never stack unmeasured flags. Run an A/A baseline check (`bench.py aa`) to measure host noise, then run interleaved `control -> candidate -> control` trials and log both wins and dead ends in `PLAN.md` / `DEAD-ENDS.md` and `ledger.jsonl`.
- **Single-GPU serialization**: Always serialize benchmark runs through `scripts/queue.sh` (portable `fcntl.flock` in a private per-user `0700` directory) so overlapping jobs never share unified memory bandwidth.
- **Reject broken speedups automatically**: `bench.py` flags and excludes `COALESCED` (<50 ms decode window), `DEGENERATE` (char-flood / n-gram loops), and `IMPLAUSIBLE` samples so broken configs can never win a sweep.

## Steps

### 1. Probe Hardware, Thermals, and Measured Metal Bandwidth

Run the environment probe and (optionally) the native Swift/Metal bandwidth micro-benchmark so your ceiling uses measured GB/s rather than spec-sheet guesses:

```bash
bash "$SKILL_DIR/scripts/probe_apple_silicon.sh"
bash "$SKILL_DIR/scripts/measure_bw_metal.sh"
```

Verify all `[PASS]` hygiene checks before benchmarking (full checklist in `references/methodology.md`):

- Plug the Mac into AC power and confirm Low Power Mode is off (`pmset -g | grep lowpowermode`).
- Confirm nominal thermal state (`pmset -g therm` reports `CPU_Speed_Limit = 100` and no thermal warning).
- Pause heavy background disk or GPU tasks (Spotlight indexing, Time Machine backups, video encoding).
- On headless Mac Studios running via LaunchDaemon, verify GPU clocks ramp under load rather than sticking at idle frequencies (see `references/gotchas.md`).

### 2. Calculate Hardware Ceilings and Memory Headroom

Compute the theoretical single-stream decode ceiling, aggregate concurrent ceiling, KV-crossover context, and safe wired-memory budget using `scripts/ceiling.py` and `references/ceilings.md`. For looped MoE models (e.g., 3-pass loop cores), pass `--loop-multiplier` so layer visits are not undercounted.

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

### 3. Screen Candidate Engines (New Machine) and Capture Baseline

If you have not picked an engine yet, follow `references/onboarding.md` and run a ~60-second `bench.py screen` bake-off across 2-3 engines (checking prose decode, code decode, two-turn prefix-cache speedup, and greedy determinism):

```bash
bash "$SKILL_DIR/scripts/queue.sh" screen-mlx -- \
  python3 "$SKILL_DIR/scripts/bench.py" screen \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --label "screen-mlx" \
    --mem-pid <server-pid> \
    --ledger ledger.jsonl
```

Once the winning engine is selected, create `PLAN.md` and `DEAD-ENDS.md`, then capture a baseline with `bench.py baseline` and compute your empirical A/A noise floor:

```bash
bash "$SKILL_DIR/scripts/queue.sh" baseline -- \
  python3 "$SKILL_DIR/scripts/bench.py" baseline \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --label "ctl-pre-1" \
    --runs 10 \
    --prompt-tokens 1024 \
    --max-tokens 256 \
    --calibrate-tokenizer \
    --mem-pid <server-pid> \
    --ledger ledger.jsonl

python3 "$SKILL_DIR/scripts/bench.py" aa --ledger ledger.jsonl --label "ctl-pre-1"
```

### 4. Hill-Climb One Lever at a Time

Consult `references/levers.md` for the ordered Apple Silicon lever playbook across macOS/Metal, MLX (`mlx-lm`, `vllm-mlx`, `oMLX`), and `llama.cpp` Metal (`llama-server`).

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

- **Sequential single-server A/B** (standard for large models on unified memory): Run `bench.py baseline` in `control-pre -> candidate -> control-post` order, then score the paired triad automatically:

```bash
python3 "$SKILL_DIR/scripts/bench.py" compare \
  --ledger ledger.jsonl \
  --control-pre-label "ctl-pre-1" \
  --candidate-label "cand-fa-on" \
  --control-post-label "ctl-post-1" \
  --metric decode_tps
```

Pass `--metric ttft_s` (lower is better) or `--metric prefill_tps` when tuning prefill/TTFT levers.

Promote a lever only if:

- Median improvement exceeds both control-vs-control drift AND the A/A noise floor (at least 3% default).
- Zero `COALESCED`, `DEGENERATE`, or `IMPLAUSIBLE` sample flags fired.
- Peak unified memory (including `IOAccelerator` GPU residency) stays inside the memory cap with zero swap growth.

### 5. Run Qualification and Anti-Cheat Gates

Before declaring a tuned configuration production-ready, pass all three gates (details in `references/methodology.md`):

- **Long-context needle gate**: Tokenizer-calibrated non-repeating filler at 10%, 50%, and 90% depth (`--max-tokens 512` for reasoning models).

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

- **Concurrency and KV soak gate**: Sweep concurrency (`bench.py concurrency --levels 1 2 4 --mem-pid <pid>`) with zero errors and stable memory.
- **Sustained thermal stability gate**: Run continuous generation (`bench.py soak --duration-s 180 --mem-pid <pid>`) confirming <= 5% thermal droop and nominal `pmset -g therm`.

### 6. Summarize and Hand Off

Produce a concise final report directly from `ledger.jsonl` containing hardware profile (chip, GPU core bin, measured Metal BW, RAM, `iogpu.wired_limit_mb`), baseline vs. final metrics (`% of ceiling`, prefill, TTFT, concurrency, peak GPU/unified memory, thermal droop), exact reproducible launch command, and kept vs. rejected levers.

## Gotchas

- **Single-chunk coalesced SSE responses fake massive tok/s**: A degenerate model emitting `!!!!` in one chunk can report 100k+ tok/s. Never override `COALESCED` or `DEGENERATE` flags.
- **Default `iogpu.wired_limit_mb` causes silent swap or Metal OOM**: Check `sysctl iogpu.wired_limit_mb` first and remember it resets on reboot.
- **MTP / speculative draft depth depends on BOTH chip bandwidth and workload type**: Code edits reward deeper drafts (`k=3-5` on Max/Ultra), while fresh prose and base M-series chips (68-120 GB/s) often regress.
- **Looped MoE models overstate ceilings by ~2x**: Always pass `--loop-multiplier` to `ceiling.py` when core layers execute multiple passes per token.
- **Read `references/gotchas.md`** for headless Mac Studio stuck GPU clocks, `mx.compile` retracing, KV quantization regressions, and `--no-mmap` load hangs.
