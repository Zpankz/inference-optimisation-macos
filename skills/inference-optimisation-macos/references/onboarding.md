# Mac Onboarding and the 15-Minute Best-Engine-First Check

On a new Apple Silicon Mac, choosing the right inference engine (`mlx_lm.server`, `vllm-mlx`, `oMLX`, or `llama-server`) usually moves throughput more than the first ten flag tweaks. Run this short bake-off before starting a deep hill-climb.

## 1. Probe Hardware and Measure Real Memory Bandwidth

Run the read-only host probe and native Metal copy benchmark:

```bash
bash "$SKILL_DIR/scripts/probe_apple_silicon.sh"
bash "$SKILL_DIR/scripts/measure_bw_metal.sh"
```

Fix any `[WARN]` lines before measuring:

- Plug into AC power and disable Low Power Mode (`sudo pmset -a lowpowermode 0`).
- If `iogpu.wired_limit_mb` is `0` on a 32 GB+ Mac and your target model + KV exceeds 65% of RAM, raise it safely (75-80% of physical RAM, leaving OS headroom).
- Note the measured `measured_bw_median_gbs` from `measure_bw_metal.sh` and pass it to `ceiling.py --bw-gbs`.

## 2. Screen 2-3 Candidate Engines (~60s Each)

Start one engine at a time on localhost, wait for it to load, and run `bench.py screen` through `queue.sh`. `screen` tests prose decode, code decode, two-turn prefix-cache reuse, and greedy determinism while enforcing the `COALESCED` / `DEGENERATE` / `IMPLAUSIBLE` guards:

```bash
bash "$SKILL_DIR/scripts/queue.sh" screen-engine-a -- \
  python3 "$SKILL_DIR/scripts/bench.py" screen \
    --base-url http://127.0.0.1:8080/v1 \
    --model "<model-id>" \
    --label "screen-mlx-lm" \
    --mem-pid <server-pid> \
    --ledger ledger.jsonl
```

Stop Engine A cleanly, confirm `mem_macos.py` shows `gpu_in_use_gib` and swap returned to baseline, then screen Engine B (for example, `oMLX`, `vllm-mlx`, or `llama-server` Metal).

## 3. Pick the Winner and Calibrate Your Noise Floor

Compare the `screen` rows in `ledger.jsonl` across four criteria:

- `passed == true` (zero degenerate/coalesced flags and deterministic greedy output).
- Highest prose + code decode tok/s as `% of ceiling`.
- `prefix_cache_speedup` (critical for coding agents and multi-turn chat).
- Clean memory delta (`swap_delta_gib == 0`).

Initialize `PLAN.md` and `DEAD-ENDS.md` for the winning engine, run a 10-sample baseline with `--calibrate-tokenizer`, and compute your host's empirical noise floor with `bench.py aa --ledger ledger.jsonl --label <baseline-label>` before changing your first lever.
