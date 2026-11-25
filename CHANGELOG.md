# Changelog

## 1.1.0 — 2026-11-26

- Added native Swift + Metal unified-memory bandwidth and `MTLDevice` probe (`scripts/measure_bw_metal.swift`).
- Hardened `scripts/bench.py` with `COALESCED`, `DEGENERATE`, and `IMPLAUSIBLE` sample guards (preventing single-chunk degenerate responses from reporting fake 100k+ tok/s).
- Added `bench.py screen` (60-second engine bake-off: prose, code, two-turn prefix cache, greedy determinism), `bench.py aa` (empirical A/A noise-floor calibration), and `--calibrate-tokenizer` prompt sizing.
- Upgraded `scripts/queue.sh` to portable Python `fcntl.flock` inside a private `0700` per-user directory with `O_NOFOLLOW` symlink checks.
- Added passwordless `ioreg -r -c IOAccelerator` live Metal GPU memory residency to `scripts/mem_macos.py`.
- Added `--loop-multiplier` and non-double-counted aggregate concurrency ceiling to `scripts/ceiling.py`.
- Added `references/onboarding.md` and updated `SKILL.md`, `ceilings.md`, `methodology.md`, `levers.md`, and `gotchas.md` with Apple Silicon community findings.

## 1.0.0 — 2026-11-26

- Initial public release of `inference-optimisation-macos`.
