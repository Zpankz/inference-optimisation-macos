# Standalone Agent Prompt (`inference-optimisation-macos`)

Use this prompt when running an agent or model that does not load `SKILL.md` files natively. Set `SKILL_DIR` to the local path of `skills/inference-optimisation-macos`.

---

You are tuning a local LLM server on an Apple Silicon Mac (`mlx_lm.server`, `vllm-mlx`, `oMLX`, or `llama-server`). Follow this strict evidence-gated hill-climbing loop:

1. **Probe first**: Run `bash "$SKILL_DIR/scripts/probe_apple_silicon.sh"`. Fix any `[WARN]` hygiene issues (AC power, Low Power Mode off, nominal thermals, `iogpu.wired_limit_mb`) before measuring.
2. **Compute ceilings**: Run `python3 "$SKILL_DIR/scripts/ceiling.py"` for the exact Mac SoC, GPU core bin, RAM, model active/total parameters, quantization bits, context length, and concurrency. Note `theoretical_decode_tps_short_ctx`, `kv_crossover_ctx`, and `estimated_footprint_gib`.
3. **Establish a baseline**: Create `PLAN.md` with hardware profile, ceiling, memory cap, and ranked hypotheses. Run `bench.py baseline --mem-pid <pid> --ledger ledger.jsonl` wrapped in `scripts/queue.sh`. Compare observed tok/s against `ceiling.py --observed-tps`.
4. **Hill-climb one lever at a time**: Consult `$SKILL_DIR/references/levers.md` and `$SKILL_DIR/references/gotchas.md`. Never stack untested flags.
   - If 2x model + KV fits under 70% RAM, run `bench.py ab` (`control -> candidate -> control`).
   - Otherwise, run sequential `bench.py baseline` triads (`ctl-pre -> cand -> ctl-post`) and score them with `bench.py compare --ledger ledger.jsonl`.
   - Use `--metric decode_tps`, `--metric prefill_tps`, or `--metric ttft_s` to match the lever.
   - Keep a change only if gain >= 3%, gain > control drift, peak memory stays within cap with zero swap growth, and no secondary metric regresses unacceptably. Log every kept and rejected run in `PLAN.md` and `ledger.jsonl`.
5. **Run qualification gates before stopping**:
   - `bench.py needle` across depths `0.1 0.5 0.9` at target context (`--max-tokens 512` for reasoning models).
   - `bench.py concurrency --levels 1 2 4 --mem-pid <pid>` with zero errors and stable memory.
   - `bench.py soak --duration-s 180 --mem-pid <pid>` with <= 5% thermal droop and nominal `pmset -g therm`.
6. **Hand off**: Report baseline vs. final metrics, `% of ceiling`, exact launch command + `sysctl` prerequisites, and the table of kept vs. rejected levers directly from `ledger.jsonl`.
