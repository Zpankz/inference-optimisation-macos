#!/usr/bin/env python3
"""Stdlib-only Apple Silicon roofline, KV-crossover, and unified-memory ceiling calculator."""

import argparse
import json
import sys

# Peak memory bandwidth (decimal GB/s) and max unified RAM (GiB) across M1-M6 family.
CHIP_PRESETS = {
    "m1": (68.0, 16),
    "m1 pro": (200.0, 32),
    "m1 max": (400.0, 64),
    "m1 ultra": (800.0, 128),
    "m2": (100.0, 24),
    "m2 pro": (200.0, 32),
    "m2 max": (400.0, 96),
    "m2 ultra": (800.0, 192),
    "m3": (100.0, 24),
    "m3 pro": (150.0, 36),
    "m3 max 30-core": (300.0, 96),
    "m3 max 40-core": (400.0, 128),
    "m3 max": (400.0, 128),
    "m3 ultra": (819.0, 512),
    "m4": (120.0, 32),
    "m4 pro": (273.0, 64),
    "m4 max 32-core": (410.0, 36),
    "m4 max 40-core": (546.0, 128),
    "m4 max": (546.0, 128),
    "m5": (153.0, 32),
    "m5 pro": (307.0, 64),
    "m5 max 32-core": (460.0, 36),
    "m5 max 40-core": (614.0, 128),
    "m5 max": (614.0, 128),
    "m5 ultra": (1228.0, 512),
    "m6 16gb": (153.0, 16),
    "m6": (170.0, 32),
}


def resolve_bandwidth(chip: str | None, bw_gbs: float | None) -> tuple[str, float]:
    if bw_gbs is not None:
        return (chip or "custom", bw_gbs)
    if not chip:
        raise ValueError("Provide either --chip or --bw-gbs.")
    key = chip.strip().lower()
    if key not in CHIP_PRESETS:
        known = ", ".join(sorted(CHIP_PRESETS.keys()))
        raise ValueError(f"Unknown chip '{chip}'. Known presets: {known}")
    return (chip.strip(), CHIP_PRESETS[key][0])


def diagnose(
    efficiency_pct: float | None,
    mem_ratio: float,
    is_moe: bool,
    ctx: int,
    kv_crossover_ctx: int,
) -> list[str]:
    notes: list[str] = []
    if mem_ratio > 0.85:
        notes.append(
            "CRITICAL MEMORY RISK: Estimated footprint exceeds 85% of unified RAM. "
            "Expect swap cliffs or Metal OOM; cut --ctx, quantize KV, or use smaller quant."
        )
    elif mem_ratio > 0.72:
        notes.append(
            "WIRED LIMIT CHECK: Footprint is 72-85% of RAM. Verify sysctl iogpu.wired_limit_mb "
            "is raised safely above footprint while leaving OS headroom."
        )
    else:
        notes.append("MEMORY HEADROOM: Footprint fits comfortably within unified RAM.")

    if kv_crossover_ctx > 0:
        kv_frac = ctx / kv_crossover_ctx
        if kv_frac >= 0.25:
            notes.append(
                f"KV-SIGNIFICANT REGIME: Target ctx ({ctx}) is {kv_frac * 100:.0f}% of KV crossover "
                f"({kv_crossover_ctx} tok). KV quantization (q8_0 / 8-bit) can improve both RAM and decode tok/s."
            )
        elif kv_frac < 0.10:
            notes.append(
                f"WEIGHT-DOMINATED REGIME: Target ctx ({ctx}) is {kv_frac * 100:.1f}% of KV crossover "
                f"({kv_crossover_ctx} tok). KV quantization may regress decode tok/s via dequant overhead; test carefully."
            )

    if efficiency_pct is not None:
        sat_floor = 55.0 if is_moe else 70.0
        sat_high = 72.0 if is_moe else 85.0
        arch_label = "sparse MoE" if is_moe else "dense"
        if efficiency_pct > 95.0:
            notes.append(
                "AUDIT MEASUREMENT: Observed tok/s exceeds 95% of theoretical peak bandwidth. "
                "Check for prompt-cache hit cheating, wrong active_params, or token-count bugs."
            )
        elif efficiency_pct >= sat_floor:
            notes.append(
                f"NEAR BANDWIDTH CEILING ({efficiency_pct:.1f}% vs {sat_floor:.0f}-{sat_high:.0f}% {arch_label} band): "
                "Single-stream decode is well-saturated. Shift focus to TTFT/prefill, prompt caching, concurrency, or active bytes."
            )
        elif efficiency_pct >= 45.0:
            notes.append(
                f"MODERATE HEADROOM ({efficiency_pct:.1f}% for {arch_label}): Tunable gains remain via Flash Attention, "
                "MLX vs llama.cpp runtime selection, mx.compile hygiene, or fused kernels."
            )
        else:
            notes.append(
                "MAJOR BOTTLENECK (<45%): Check for CPU fallback, swap/compression, "
                "Low Power Mode, thermal throttling, stuck headless GPU clock, or mx.compile retracing."
            )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Apple Silicon LLM decode ceilings, KV crossover, and unified-memory headroom."
    )
    parser.add_argument("--list-chips", action="store_true", help="List built-in chip presets and exit")
    parser.add_argument("--chip", help="Apple Silicon chip preset (e.g., 'M4 Max 40-core', 'M5 Pro')")
    parser.add_argument("--bw-gbs", type=float, help="Override peak memory bandwidth in decimal GB/s")
    parser.add_argument("--ram-gb", type=float, help="Total unified RAM in GiB (e.g., 32, 64, 128)")
    parser.add_argument("--wired-limit-gb", type=float, help="Current iogpu.wired_limit_mb in GiB (optional)")
    parser.add_argument("--active-params-b", type=float, help="Active parameters in billions (equals total for dense models)")
    parser.add_argument("--total-params-b", type=float, help="Total parameters in billions (for MoE residency; defaults to active-params-b)")
    parser.add_argument("--bits", type=float, default=4.5, help="Effective bits per weight including scales/embeddings (default: 4.5)")
    parser.add_argument("--file-gb", type=float, help="Exact model weight file size in decimal GB (overrides total-params * bits)")
    parser.add_argument("--ctx", type=int, default=8192, help="Target context window in tokens (default: 8192)")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent slots / streams (default: 1)")
    parser.add_argument("--kv-bytes-per-token", type=float, help="KV cache bytes per token (overrides layer/head math)")
    parser.add_argument("--layers", type=int, help="Number of transformer layers (for KV math)")
    parser.add_argument("--kv-heads", type=int, help="Number of KV heads (for KV math)")
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension (default: 128)")
    parser.add_argument("--kv-bytes-per-elem", type=float, default=2.0, help="Bytes per KV element: 2.0=fp16, 1.06=q8_0, 0.56=q4_0")
    parser.add_argument("--scratch-gb", type=float, default=1.5, help="Estimated activation/Metal scratch buffer in GiB (default: 1.5)")
    parser.add_argument("--observed-tps", type=float, help="Measured single-stream decode tok/s to compute efficiency %%")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    if args.list_chips:
        for name, (bw, max_ram) in sorted(CHIP_PRESETS.items()):
            print(f"{name:18s}  {bw:7.1f} GB/s  (max {max_ram} GiB)")
        return 0

    if args.ram_gb is None or args.active_params_b is None:
        parser.error("--ram-gb and --active-params-b are required unless --list-chips is set.")

    chip_label, bw_gbs = resolve_bandwidth(args.chip, args.bw_gbs)
    total_params_b = args.total_params_b if args.total_params_b is not None else args.active_params_b
    is_moe = total_params_b > args.active_params_b * 1.05

    # Weight bytes in decimal bytes (matches Apple's decimal GB/s memory bandwidth spec).
    if args.file_gb is not None:
        total_weight_bytes = args.file_gb * 1e9
        effective_bits = (total_weight_bytes * 8.0) / (total_params_b * 1e9)
    else:
        effective_bits = args.bits
        total_weight_bytes = total_params_b * 1e9 * (effective_bits / 8.0)

    active_weight_bytes = args.active_params_b * 1e9 * (effective_bits / 8.0)

    if args.kv_bytes_per_token is not None:
        kv_bpt = args.kv_bytes_per_token
    elif args.layers is not None and args.kv_heads is not None:
        kv_bpt = 2.0 * args.layers * args.kv_heads * args.head_dim * args.kv_bytes_per_elem
    else:
        kv_bpt = 131072.0 * (args.kv_bytes_per_elem / 2.0)

    kv_single_ctx_bytes = kv_bpt * args.ctx
    kv_total_bytes = kv_single_ctx_bytes * args.concurrency
    kv_crossover_ctx = int(active_weight_bytes / max(kv_bpt, 1.0))

    # Convert residency into GiB (1024^3) so it aligns directly with macOS hw.memsize & iogpu.wired_limit_mb.
    gib = float(1024**3)
    total_weight_gib = total_weight_bytes / gib
    active_weight_gb = active_weight_bytes / 1e9
    kv_total_gib = kv_total_bytes / gib
    total_footprint_gib = total_weight_gib + kv_total_gib + args.scratch_gb

    bw_bytes_per_s = bw_gbs * 1e9
    theo_tps_short = bw_bytes_per_s / active_weight_bytes
    theo_tps_full_ctx = bw_bytes_per_s / (active_weight_bytes + kv_single_ctx_bytes)
    band_low, band_high = (0.55, 0.72) if is_moe else (0.70, 0.85)
    realistic_low = theo_tps_short * band_low
    realistic_high = theo_tps_short * band_high

    wired_limit_gib = args.wired_limit_gb if args.wired_limit_gb is not None else (args.ram_gb * 0.75)
    mem_ratio = total_footprint_gib / args.ram_gb
    efficiency_pct = (args.observed_tps / theo_tps_short * 100.0) if args.observed_tps is not None else None
    notes = diagnose(efficiency_pct, mem_ratio, is_moe, args.ctx, kv_crossover_ctx)

    result = {
        "chip": chip_label,
        "architecture": "moe" if is_moe else "dense",
        "peak_bandwidth_gbs": round(bw_gbs, 2),
        "ram_gib": args.ram_gb,
        "wired_limit_gib": round(wired_limit_gib, 2),
        "effective_bits_per_param": round(effective_bits, 3),
        "active_weight_gb": round(active_weight_gb, 3),
        "total_weight_gib": round(total_weight_gib, 3),
        "kv_bytes_per_token": round(kv_bpt, 1),
        "kv_crossover_ctx": kv_crossover_ctx,
        "kv_total_gib": round(kv_total_gib, 3),
        "scratch_gib": args.scratch_gb,
        "estimated_footprint_gib": round(total_footprint_gib, 3),
        "footprint_pct_of_ram": round(mem_ratio * 100.0, 1),
        "fits_wired_limit": total_footprint_gib <= wired_limit_gib,
        "theoretical_decode_tps_short_ctx": round(theo_tps_short, 2),
        "realistic_decode_tps_band": [round(realistic_low, 2), round(realistic_high, 2)],
        "theoretical_decode_tps_full_ctx": round(theo_tps_full_ctx, 2),
        "observed_tps": args.observed_tps,
        "observed_efficiency_pct": round(efficiency_pct, 1) if efficiency_pct is not None else None,
        "diagnostics": notes,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    band_pct = "55-72% MoE" if is_moe else "70-85% dense"
    print(f"=== Apple Silicon Roofline: {chip_label} ({bw_gbs:.1f} GB/s, {args.ram_gb:.0f} GiB RAM) ===")
    print(f"Active weights:       {active_weight_gb:6.2f} GB   (total residency: {total_weight_gib:.2f} GiB @ {effective_bits:.2f} bpp)")
    print(f"KV cache ({args.ctx} tok x {args.concurrency}): {kv_total_gib:6.2f} GiB  ({kv_bpt / 1024:.1f} KiB/tok; crossover: {kv_crossover_ctx} tok)")
    print(f"Estimated footprint:  {total_footprint_gib:6.2f} GiB  ({mem_ratio * 100:.1f}% of RAM; wired cap: {wired_limit_gib:.1f} GiB)")
    print(f"Decode ceiling (short ctx): {theo_tps_short:6.2f} tok/s  (realistic {band_pct}: {realistic_low:.1f}-{realistic_high:.1f} tok/s)")
    print(f"Decode ceiling (full ctx):  {theo_tps_full_ctx:6.2f} tok/s")
    if efficiency_pct is not None:
        print(f"Observed decode:      {args.observed_tps:6.2f} tok/s  ({efficiency_pct:.1f}% of theoretical ceiling)")
    print("")
    for note in notes:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
