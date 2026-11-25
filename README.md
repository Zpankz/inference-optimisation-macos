# inference-optimisation-macos

Ceiling-first, evidence-gated local LLM inference tuning for Apple Silicon Macs (M1 through M6 Base, Pro, Max, and Ultra). Built as a portable agent skill (`SKILL.md` + stdlib-only scripts) for `mlx_lm.server`, `vllm-mlx`, `oMLX`, and `llama-server` (`llama.cpp` Metal).

## Why this exists

[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill) nailed the discipline for hill-climbing local LLM servers: calculate hardware ceilings first, change one knob at a time, run interleaved `control -> candidate -> control` trials, log dead ends, and gate promotions on long-context recall and soak stability.

On macOS, though, its harness and lever playbooks assume Linux + CUDA/XPU: `queue.sh` calls Linux `flock`, `mem.sh` reads `/proc/<pid>/fdinfo`, and most levers target vLLM/SGLang/Triton rather than Metal or MLX. This repo ports the same measurement discipline natively to Apple Silicon unified memory.

## What is in here

- `skills/inference-optimisation-macos/SKILL.md` — the core 6-step loop (probe, ceiling math, baseline, single-lever A/B, quality/thermal gates, handoff).
- `skills/inference-optimisation-macos/references/` — on-demand engineering notes:
  - `ceilings.md` — M1–M6 memory bandwidth table (including M3/M4/M5 Max GPU-bin bus splits), dense vs. MoE decode efficiency bands, KV-crossover context math, and M5+ Neural Accelerator prefill shifts.
  - `methodology.md` — macOS noise hygiene, dual-server vs. sequential single-server A/B triads, unified memory telemetry, and anti-cheat gates.
  - `levers.md` — ordered tuning playbooks for macOS/Metal (`iogpu.wired_limit_mb`), MLX (`mx.set_wired_limit`, `mx.set_cache_limit`, KV quant, `mx.compile`, `mx.fast.*` -> `mx.fast.metal_kernel`), and `llama-server` (the 7 flags that move the needle, plus x86/CUDA placebos to drop).
  - `gotchas.md` — field-observed macOS failure modes (reboot-reset wired limits, headless Mac Studio 338–620 MHz stuck GPU clocks, `mx.compile` silent retracing, MTP bandwidth regressions on base M-series chips, `--no-mmap` load hangs, reasoning-model needle truncation).
- `skills/inference-optimisation-macos/scripts/` — zero-dependency Python 3 + Bash harness:
  - `probe_apple_silicon.sh` — detects SoC, P/E/GPU cores, RAM, `iogpu.wired_limit_mb`, power/thermal state, installed runtimes, and prints `[PASS]`/`[WARN]` hygiene gates.
  - `ceiling.py` — computes theoretical decode tok/s, realistic dense/MoE bands, `kv_crossover_ctx`, and GiB unified-memory headroom.
  - `queue.sh` — macOS-safe single-GPU lock serializer using POSIX atomic `mkdir` and atomic `mv` stale-lock reaping (no `flock`).
  - `mem_macos.py` — reads `sysctl`, `vm_stat`, `memory_pressure -Q`, `pmset -g therm`, `ps`, and optional `mlx.core` allocator stats.
  - `bench.py` — OpenAI-compatible SSE benchmark harness supporting `baseline`, `ab` (dual-server), `compare` (sequential single-server triad from `ledger.jsonl`), `concurrency`, `soak` (thermal droop gate), and `needle` (non-repeating filler, reasoning-safe token budget).
- `PROMPT.md` — standalone task prompt for agents or models that do not load `SKILL.md` files natively.
- `tests/test_harness.py` — stdlib `unittest` suite covering syntax, markdown structure, parsers, lock contention, and end-to-end mock SSE server runs.

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

# 1. Check hardware, wired memory limit, power, and thermal hygiene
bash "$SKILL_DIR/scripts/probe_apple_silicon.sh"

# 2. Compute your bandwidth ceiling and KV-crossover context
python3 "$SKILL_DIR/scripts/ceiling.py" \
  --chip "M4 Max 40-core" \
  --ram-gb 128 \
  --active-params-b 3.5 \
  --total-params-b 35 \
  --bits 4.5 \
  --ctx 32768 \
  --concurrency 2

# 3. Record a serialized baseline against any OpenAI-compatible local server
bash "$SKILL_DIR/scripts/queue.sh" baseline -- \
  python3 "$SKILL_DIR/scripts/bench.py" baseline \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --label "ctl-pre-1" \
    --runs 5 \
    --prompt-tokens 1024 \
    --max-tokens 256 \
    --mem-pid <server-pid> \
    --ledger ledger.jsonl

# 4. Score a sequential single-server triad (when 2x model won't fit in unified RAM)
python3 "$SKILL_DIR/scripts/bench.py" compare \
  --ledger ledger.jsonl \
  --control-pre-label "ctl-pre-1" \
  --candidate-label "cand-fa-on" \
  --control-post-label "ctl-post-1" \
  --metric decode_tps
```

## Caveats

- `mem_macos.py` stays `sudo`-free and stdlib-only. If baseline decode is below 30% of ceiling on a headless Mac Studio, check GPU clocks separately with `macmon` or `sudo powermetrics --samplers gpu_power -n 1`.
- `bench.py` targets HTTP OpenAI-compatible servers (`/v1/chat/completions`), not raw in-process MLX Python loops (use `mlx_lm.benchmark` or explicit `mx.eval()` / `mx.synchronize()` timing for those).
- Prompt lengths in `bench.py` are approximated at ~4 chars/token; always read server-reported `prompt_tokens` when pushing right up against a hard context cap.

## Running tests

```bash
python3 -m unittest discover -s tests -v
```

CI runs the full test suite on both `macos-latest` (Apple Silicon) and `ubuntu-latest`.

## Credits and prior art

This project builds directly on the engineering work of several repositories and practitioners:

- **[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)** (0xSero / Sybil Solutions, MIT) — core ceiling-first hill-climbing workflow, interleaved A/B protocol, `PLAN.md` + `ledger.jsonl` discipline, and anti-cheat verification gates.
- **[cavit99/mlx-LLM-cheatsheet](https://github.com/cavit99/mlx-LLM-cheatsheet)** — MLX kernel escalation ladder, `mx.compile` retracing rules, lazy evaluation timing pitfalls, top-level `mx.get_*_memory()` telemetry, and Xcode `.gputrace` capture hooks.
- **[KittenML/inferencekit](https://github.com/KittenML/inferencekit)** — Apple M4 MLX hill-climbing case study and roofline/anti-cheat patterns.
- **[Amal-David/mlx-porting-skill](https://github.com/Amal-David/mlx-porting-skill)** — evidence-gated MLX mechanism and quantization evaluation on Apple Silicon.
- **[LLMCheck Apple Silicon Index](https://llmcheck.net/compare/apple-silicon-chips-for-ai/)** — M1–M6 Base/Pro/Max/Ultra memory bandwidth and RAM specifications.
- **[Michael Hannecke](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)** and **[adamteale](https://gist.github.com/adamteale/a41343fdaa5dea1e88931277d6d17908)** — Apple Silicon `llama-server` flag pruning, `iogpu.wired_limit_mb` practices, and macOS `--no-mmap` workarounds.

## License

MIT — see [LICENSE](LICENSE).
