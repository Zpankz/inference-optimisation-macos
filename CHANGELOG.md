# Changelog

## 1.0.0 — 2026-11-26

- Initial public release of `inference-optimisation-macos`.
- Added ceiling-first Apple Silicon roofline calculator (`scripts/ceiling.py`) covering M1 through M6 Base, Pro, Max (split GPU core bins), and Ultra, dense vs. MoE efficiency bands, and `kv_crossover_ctx`.
- Added macOS-native unified memory and thermal telemetry (`scripts/mem_macos.py`) and hardware/hygiene probe (`scripts/probe_apple_silicon.sh`).
- Added POSIX atomic `mkdir` GPU lock serializer (`scripts/queue.sh`) with atomic `mv` stale-lock reaping.
- Added OpenAI-compatible SSE benchmark harness (`scripts/bench.py`) supporting `baseline`, dual-server `ab`, sequential single-server `compare`, `concurrency`, thermal `soak`, and heterogeneous-filler `needle` gates with `ledger.jsonl` logging.
- Added MLX and `llama.cpp` Metal engineering references (`ceilings.md`, `methodology.md`, `levers.md`, `gotchas.md`) and standalone `PROMPT.md`.
