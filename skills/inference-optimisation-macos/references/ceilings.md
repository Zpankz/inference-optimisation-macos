# Apple Silicon Roofline and Memory Ceilings

Calculate hardware ceilings before tuning any server flag. On Apple Silicon unified memory, single-stream token generation (decode) is almost always memory-bandwidth bound, while prompt processing (prefill) and high-batch serving transition toward GPU compute (and M5+ Neural Accelerator) limits.

## Measuring Real Bandwidth vs. Spec-Sheet Bandwidth

Apple publishes theoretical peak bus bandwidth in decimal GB/s (`1e9 bytes/s`). Sustained GPU copy bandwidth is typically 80-92% of spec peak. Run the bundled native Swift/Metal probe on your Mac to measure real sustained unified-memory bandwidth directly:

```bash
swift "$SKILL_DIR/scripts/measure_bw_metal.swift"
```

Pass the reported `measured_bw_median_gbs` to `scripts/ceiling.py --bw-gbs` for the most accurate ceiling, or use the M1-M6 spec-sheet presets below as a starting point.

## Apple Silicon Memory Bandwidth Reference (M1 through M6)

| Chip | GPU Cores | Peak Memory Bandwidth | Max Unified RAM | Notes |
| :--- | :--- | :--- | :--- | :--- |
| M1 | 7-8 | 68 GB/s | 16 GB | Tight bandwidth; small models only |
| M1 Pro | 14-16 | 200 GB/s | 32 GB | Solid 7B-14B decode |
| M1 Max | 24-32 | 400 GB/s | 64 GB | Strong 30B-70B decode |
| M1 Ultra | 48-64 | 800 GB/s | 128 GB | Dual-die UltraFusion |
| M2 | 8-10 | 100 GB/s | 24 GB | +47% bandwidth over M1 |
| M2 Pro | 16-19 | 200 GB/s | 32 GB | Same bandwidth as M1 Pro |
| M2 Max | 30-38 | 400 GB/s | 96 GB | Same bandwidth as M1 Max |
| M2 Ultra | 60-76 | 800 GB/s | 192 GB | High-RAM 70B-120B host |
| M3 | 8-10 | 100 GB/s | 24 GB | Dynamic Caching GPU |
| M3 Pro | 14-18 | 150 GB/s | 36 GB | Lower bandwidth than M2 Pro (150 vs 200 GB/s) |
| M3 Max (30-core) | 30 | 300 GB/s | 96 GB | Reduced memory bus vs 40-core |
| M3 Max (40-core) | 40 | 400 GB/s | 128 GB | Full 400 GB/s bus |
| M3 Ultra | 60-80 | 819 GB/s | 512 GB | Up to 512 GB unified RAM |
| M4 | 10 | 120 GB/s | 32 GB | Bandwidth-limited for deep speculation |
| M4 Pro | 16-20 | 273 GB/s | 64 GB | Big jump over M3 Pro |
| M4 Max (32-core) | 32 | 410 GB/s | 36 GB | Reduced bus on 32-core bin |
| M4 Max (40-core) | 40 | 546 GB/s | 128 GB | +37% bandwidth over M3 Max |
| M5 | 8-10 | 153 GB/s | 32 GB | Neural Accelerators in GPU cores (up to 4x prefill) |
| M5 Pro | 16-20 | 307 GB/s | 64 GB | Strong value tier; M5 prefill jump |
| M5 Max (32-core) | 32 | 460 GB/s | 36 GB | Reduced bus on 32-core bin |
| M5 Max (40-core) | 40 | 614 GB/s | 128 GB | Fastest laptop bandwidth |
| M5 Ultra | 64-80 | 1,228 GB/s | 512 GB | Up to 4.4x faster TTFT vs M3 Ultra |
| M6 (16 GB) | 10-12 | 153 GB/s | 16 GB | Entry M6 memory bus |
| M6 (24-32 GB) | 10-12 | 170 GB/s | 32 GB | Higher memory bus at 24-32 GB |

Important nuance: Always verify the exact GPU core bin (`system_profiler SPDisplaysDataType`) on M3 Max, M4 Max, and M5 Max Macs, because the lower GPU core bin ships with a narrower memory bus (300 vs 400, 410 vs 546, and 460 vs 614 GB/s respectively).

## Decode Ceiling Math (Single-Stream and Concurrent)

During autoregressive decode at batch size 1, generating one token streams every active weight (multiplied by looped layer visits, if applicable) plus the active KV cache from unified memory into the GPU. At concurrency `N`, shared dense weights and draft-model weights are read **once per batched engine step**, while KV cache is read `N` times (once per stream).

```text
active_weight_bytes = active_params * loop_multiplier * (effective_bits_per_param / 8)
kv_read_bytes_per_step = kv_bytes_per_token * current_context_tokens
total_bytes_per_token = active_weight_bytes + kv_read_bytes_per_step

theoretical_decode_tps     = peak_bandwidth_bytes_per_sec / total_bytes_per_token
theoretical_agg_tps_conc_N = (peak_bandwidth_bytes_per_sec / (active_weight_bytes + N * kv_read_bytes_per_step)) * N
realistic_dense_tps        = 0.70 to 0.85 * theoretical_decode_tps
realistic_moe_tps          = 0.55 to 0.72 * theoretical_decode_tps
```

Key implications:

- **Looped models (count layer visits, not physical layers)**: A looped architecture like Inkling-2-Mini (24 physical layers whose 12-layer core executes 3 times = 48 layer visits) reads dense and expert weights roughly 2x per token. Pass `--loop-multiplier 2.0` to `ceiling.py` or your theoretical ceiling will overstate headroom by 2x.
- **MoE models**: Use `active_params` (for example, 3B active out of 35B total) for decode bandwidth math, and `total_params` for unified-memory residency math. Sparse MoE models incur higher per-token dispatch overhead (router, norms, expert gather/scatter), so realized bandwidth efficiency is typically 55-72% rather than 70-85%.
- **Effective bits per parameter**: A nominal "4-bit" quant is rarely 4.00 bits/param once scales, zero-points, and unquantized embeddings/lm_head are included. Prefer actual weight file size on disk (`--file-gb`) when available, or estimate 4.5-4.8 bits for Q4_K_M / MLX 4-bit-g64, 5.5-5.8 bits for Q5_K_M, and 8.5 bits for Q8_0 / MLX 8-bit.
- **KV-crossover context**: Compute `kv_crossover_ctx = active_weight_bytes / kv_bytes_per_token`. Above ~25% of `kv_crossover_ctx`, KV reads meaningfully drag down decode speed and KV quantization becomes a throughput lever (not just a RAM-saving lever). Below ~10% of `kv_crossover_ctx`, KV quantization rarely speeds up decode and can regress it due to dequantization overhead.

## Prefill and Compute Regime (M1-M4 vs M5+)

- On M1 through M4, long-prompt prefill is heavily GPU matrix-compute bound, so increasing `llama-server` micro-batch size (`-ub 1024` or `2048`) and enabling Flash Attention (`-fa on`) delivers large TTFT reductions.
- On M5 and M6 chips with GPU Neural Accelerators, prompt processing is up to 4x faster, meaning prefill transitions toward memory-bandwidth and attention-kernel efficiency earlier. Always A/B test `-ub` and chunked prefill rather than assuming M1-M4 defaults carry over unchanged.

## KV Cache Memory Math

For a standard Grouped-Query Attention (GQA) transformer:

```text
kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

Where `bytes_per_element` is:

- `2.0` for FP16 / BF16 KV cache
- `~1.06` for `q8_0` / 8-bit quantized KV (including group scales)
- `~0.56` for `q4_0` / 4-bit quantized KV (including group scales)

Total KV pool memory for concurrent serving:

```text
total_kv_bytes = kv_bytes_per_token * max_context_tokens * concurrency
```

## Unified Memory and Wired-Limit Headroom

Unlike discrete GPUs with dedicated VRAM, Apple Silicon shares one physical pool between macOS, user apps, Metal wired GPU buffers, and page cache.

```text
total_model_footprint = total_weight_bytes + total_kv_bytes + scratch_activations_bytes
```

Practical safety thresholds:

- **macOS default wired limit**: Typically ~66-75% of physical RAM depending on memory tier.
- **Safe raised `iogpu.wired_limit_mb`**: Up to 75-80% of physical RAM on 32-64 GB Macs, or leaving 16-32 GB unwired for macOS on 128-512 GB Macs.
- **Danger zone**: When `total_model_footprint` approaches or exceeds `iogpu.wired_limit_mb`, Metal either fails allocation or forces macOS into heavy memory compression and SSD swap, collapsing decode throughput by 5x-50x.
