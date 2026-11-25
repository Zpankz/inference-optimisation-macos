# inference-optimisation-macos

Ceiling-first, evidence-gated local LLM inference tuning for Apple Silicon Macs (M1 through M6 Base, Pro, Max, and Ultra). Built as a portable agent skill (`SKILL.md` + stdlib-only scripts + an optional native Swift/Metal bandwidth probe) for `mlx_lm.server`, `vllm-mlx`, `oMLX`, and `llama-server` (`llama.cpp` Metal).

## Why this exists

[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill) nailed the discipline for hill-climbing local LLM servers: calculate hardware ceilings first, change one knob at a time, run interleaved `control -> candidate -> control` trials, log dead ends, and gate promotions on long-context recall and soak stability.

On macOS, though, its original harness assumed Linux + CUDA/XPU (`flock(1)`, `/proc/<pid>/fdinfo`, vLLM/Triton levers). This repo ports that workflow natively to Apple Silicon unified memory and folds in community fixes from real Mac hill-climbs (coalesced SSE guards, A/A noise-floor calibration, `IOAccelerator` GPU memory probing, looped-model rooflines, workload-specific MTP depth, and a 15-minute engine bake-off).

## Why Python + native Swift/Metal (and not Rust or Go)

- **Harness (`bench.py`, `ceiling.py`, `mem_macos.py`, `queue.sh`) stays stdlib Python + Bash**: Against a local LLM generating 10–200 tok/s, client SSE parsing in Python takes ~0.05–0.2 ms per token (<0.5% of inter-token latency). Rewriting the harness in Rust or Go would add toolchain/binary friction and break instant agent editability for no measurable gain.
- **Hardware bandwidth probe (`measure_bw_metal.sh`) uses native Swift + Metal**: Measuring sustained unified-memory bandwidth and querying `MTLCreateSystemDefaultDevice()` (`recommendedMaxWorkingSetSize`, `currentAllocatedSize`) actually benefits from native Metal APIs—and `swift` ships with macOS Xcode Command Line Tools, so there is still zero Cargo, Go, or Homebrew setup required.

## What is in here

- `skills/inference-optimisation-macos/SKILL.md` — the core 6-step loop (probe + Metal BW, ceiling math, engine screen + A/A baseline, single-lever A/B, quality/thermal gates, handoff).
- `skills/inference-optimisation-macos/references/` — on-demand engineering notes:
  - `onboarding.md` — 15-minute Mac onboarding and best-engine-first bake-off workflow.
  - `ceilings.md` — M1–M6 memory bandwidth table (including M3/M4/M5 Max GPU-bin bus splits), measured Metal copy BW, looped-model multipliers, dense vs. MoE efficiency bands, KV-crossover math, and M5+ Neural Accelerator prefill shifts.
  - `methodology.md` — macOS noise hygiene, empirical A/A noise-floor calibration, dual-server vs. sequential single-server A/B triads, sample validity guards, unified memory telemetry, and anti-cheat gates.
  - `levers.md` — ordered tuning playbooks for macOS/Metal (`iogpu.wired_limit_mb`), MLX (`mx.set_wired_limit`, `mx.set_cache_limit`, KV quant, `mx.compile`, `mx.fast.*` -> `mx.fast.metal_kernel`), `llama-server` (7 high-impact flags, x86 placebos stripped), and workload-aware MTP draft depth.
  - `gotchas.md` — field-observed macOS failure modes (coalesced single-chunk fake 863k tok/s responses, reboot-reset wired limits, headless Mac Studio 338–620 MHz stuck GPU clocks, looped-model ceiling overstatement, `mx.compile` retracing, MTP bandwidth regressions, `--no-mmap` load hangs, reasoning-model needle truncation).
- `skills/inference-optimisation-macos/scripts/` — zero-dependency harness:
  - `probe_apple_silicon.sh` — detects SoC, P/E/GPU cores, RAM, `iogpu.wired_limit_mb`, power/thermal state, installed runtimes, and prints `[PASS]`/`[WARN]` hygiene gates.
  - `measure_bw_metal.sh` — native Swift + Metal probe querying `MTLDevice` and timing `MTLBlitCommandEncoder` shared-buffer copies to measure real sustained GB/s.
  - `ceiling.py` — computes single-stream and non-double-counted concurrent ceilings, `--loop-multiplier`, realistic dense/MoE bands, `kv_crossover_ctx`, and GiB unified-memory headroom.
  - `queue.sh` — portable single-GPU lock serializer using Python's `fcntl.flock` inside a private per-user `0700` directory with `O_NOFOLLOW` symlink checks.
  - `mem_macos.py` — passwordless `ioreg -r -c IOAccelerator` GPU residency alongside `sysctl`, `vm_stat`, `memory_pressure -Q`, `pmset -g therm`, `ps`, and optional `mlx.core` allocator stats.
  - `bench.py` — OpenAI-compatible SSE benchmark harness supporting `screen`, `baseline` (with `--calibrate-tokenizer`), `aa` (A/A noise floor), `ab` (dual-server), `compare` (sequential single-server triad), `concurrency`, `soak` (thermal droop gate), and `needle` (non-repeating filler), with `COALESCED`/`DEGENERATE`/`IMPLAUSIBLE` guards.
- `PROMPT.md` — standalone task prompt for agents or models that do not load `SKILL.md` files natively.
- `tests/test_harness.py` — stdlib `unittest` suite covering syntax, markdown structure, parsers, lock contention, sample guards, and end-to-end mock SSE server runs.

## Quick start

### Install as an agent skill (Claude Code / compatible harnesses)

```bash
git clone https://github.com/Zpankz/inference-optimisation-macos
mkdir -p ~/.claude/skills
cp -r inference-optimisation-macos/skills/inference-optimisation-macos ~/.claude/skills/
```

If your agent does not support `SKILL.md` directories, paste `PROMPT.md` into the session and point `SKILL_DIR` at `skills/inference-optimisation-macos`.

### Run the harness directly

```bash
export SKILL_DIR="./skills/inference-optimisation-macos"

# 1. Check hardware/thermal hygiene and measure real Metal memory bandwidth
bash "$SKILL_DIR/scripts/probe_apple_silicon.sh"
bash "$SKILL_DIR/scripts/measure_bw_metal.sh"

# 2. Compute your bandwidth ceiling and KV-crossover context
python3 "$SKILL_DIR/scripts/ceiling.py" \
  --chip "M4 Max 40-core" \
  --ram-gb 128 \
  --active-params-b 3.5 \
  --total-params-b 35 \
  --bits 4.5 \
  --ctx 32768 \
  --concurrency 2

# 3. Screen candidate engines (~60s smoke test: prose, code, prefix cache, greedy check)
bash "$SKILL_DIR/scripts/queue.sh" screen-mlx -- \
  python3 "$SKILL_DIR/scripts/bench.py" screen \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --label "screen-mlx" \
    --mem-pid <server-pid> \
    --ledger ledger.jsonl

# 4. Calibrate your A/A noise floor from 10 unchanged runs
bash "$SKILL_DIR/scripts/queue.sh" ctl-pre -- \
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

## Caveats

- `mem_macos.py` stays `sudo`-free and stdlib-only. If baseline decode is below 30% of ceiling on a headless Mac Studio, check GPU clocks separately with `macmon` or `sudo powermetrics --samplers gpu_power -n 1`.
- `bench.py` targets HTTP OpenAI-compatible servers (`/v1/chat/completions`), not raw in-process MLX Python loops (use `mlx_lm.benchmark` or explicit `mx.eval()` / `mx.synchronize()` timing for those).

## Running tests

```bash
python3 -m unittest discover -s tests -v
```

CI runs the full test suite on both `macos-latest` (Apple Silicon) and `ubuntu-latest`.

## Credits and prior art

This project builds directly on the engineering work of several repositories, pull requests, and practitioners:

- **[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)** (0xSero / Sybil Solutions, MIT) — core ceiling-first hill-climbing workflow, interleaved A/B protocol, `PLAN.md` + `ledger.jsonl` discipline, and anti-cheat verification gates; plus community PRs [#24](https://github.com/sybil-solutions/inference-optimisation-skill/pull/24) (portable `fcntl.flock` queue, `screen` bake-off, Mac onboarding), [#9](https://github.com/sybil-solutions/inference-optimisation-skill/pull/9) (coalesced/degenerate SSE sample guards), [#11](https://github.com/sybil-solutions/inference-optimisation-skill/pull/11) (A/A noise-floor calibration & tokenizer-calibrated prompt sizing), [#14](https://github.com/sybil-solutions/inference-optimisation-skill/pull/14) (`IOAccelerator` Metal memory probe), [#19](https://github.com/sybil-solutions/inference-optimisation-skill/pull/19) (MTP draft depth by output type), [#22](https://github.com/sybil-solutions/inference-optimisation-skill/pull/22) (looped-model ceilings), and Apple Silicon empirical results in [#7](https://github.com/sybil-solutions/inference-optimisation-skill/pull/7), [#8](https://github.com/sybil-solutions/inference-optimisation-skill/pull/8), and [#23](https://github.com/sybil-solutions/inference-optimisation-skill/pull/23).
- **[cavit99/mlx-LLM-cheatsheet](https://github.com/cavit99/mlx-LLM-cheatsheet)** — MLX kernel escalation ladder, `mx.compile` retracing rules, lazy evaluation timing pitfalls, top-level `mx.get_*_memory()` telemetry, and Xcode `.gputrace` capture hooks.
- **[KittenML/inferencekit](https://github.com/KittenML/inferencekit)** — Apple M4 MLX hill-climbing case study and roofline/anti-cheat patterns.
- **[Amal-David/mlx-porting-skill](https://github.com/Amal-David/mlx-porting-skill)** — evidence-gated MLX mechanism and quantization evaluation on Apple Silicon.
- **[LLMCheck Apple Silicon Index](https://llmcheck.net/compare/apple-silicon-chips-for-ai/)** — M1–M6 Base/Pro/Max/Ultra memory bandwidth and RAM specifications.
- **[Michael Hannecke](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)** and **[adamteale](https://gist.github.com/adamteale/a41343fdaa5dea1e88931277d6d17908)** — Apple Silicon `llama-server` flag pruning, `iogpu.wired_limit_mb` practices, and macOS `--no-mmap` workarounds.

## License

MIT — see [LICENSE](LICENSE).
