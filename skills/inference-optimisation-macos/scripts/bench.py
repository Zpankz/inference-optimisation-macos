#!/usr/bin/env python3
"""Stdlib-only OpenAI-compatible inference benchmark harness for Apple Silicon.

Supports:
  - baseline:    Warm-up + N streaming runs measuring TTFT, prefill tok/s, decode tok/s, ITL p50/p95
  - screen:      Fast ~60s multi-workload smoke test (prose, code, two-turn prefix cache, greedy determinism)
  - ab:          Dual-server interleaved control -> candidate -> control A/B cycles with drift check
  - compare:     Score a sequential single-server control-pre / candidate / control-post triad from ledger.jsonl
  - aa:          Compute empirical A/A noise floor (false-win rate & max noise %) from unchanged baseline runs
  - concurrency: Multi-client throughput sweep across concurrency levels
  - soak:        Continuous generation thermal stability gate (first-window vs final-window droop)
  - needle:      Long-context non-repeating needle-in-a-haystack gate with tokenizer-calibrated sizing

Includes anti-cheat sample guards (PR #9): flags and excludes COALESCED (<50ms decode window),
DEGENERATE (char-flood or n-gram loop), and IMPLAUSIBLE (>max_plausible_tps) samples from medians.
"""

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Import sibling mem_macos telemetry helper if available
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import mem_macos  # type: ignore
except Exception:
    mem_macos = None


MIN_DECODE_WINDOW_S = 0.05
DEFAULT_MAX_PLAUSIBLE_TPS = 2500.0

METRIC_SPECS = {
    "decode_tps": ("decode_tps_median", "higher"),
    "prefill_tps": ("prefill_tps_median", "higher"),
    "ttft_s": ("ttft_s_median", "lower"),
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def is_degenerate(text: str) -> bool:
    """Detect char-flood degeneration (e.g., 64x '!') and tight word n-gram looping."""
    stripped = "".join(ch for ch in text if not ch.isspace())
    if len(stripped) >= 16:
        if (len(set(stripped)) / len(stripped)) < 0.05:
            return True
    words = text.split()
    if len(words) >= 24:
        ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        if ngrams and (len(set(ngrams)) / len(ngrams)) < 0.25:
            return True
    return False


def make_prompt(approx_tokens: int, allow_cache: bool, chars_per_token: float = 4.0) -> str:
    nonce = "" if allow_cache else f"[nonce-{random.randint(100000, 999999)}] "
    base = (
        "Summarize the architectural trade-offs between unified memory bandwidth, "
        "KV cache precision, and prompt batching on Apple Silicon hardware. "
    )
    target_chars = max(64, int(approx_tokens * chars_per_token))
    reps = max(1, (target_chars // len(base)) + 1)
    body = (base * reps)[:target_chars]
    return f"{nonce}{body}\nProvide a detailed technical explanation."


def make_heterogeneous_filler(target_chars: int, seed: int) -> list[str]:
    """Generate non-repeating domain sentences so n-gram lookup or KV tricks cannot cheat."""
    rng = random.Random(seed)
    subsystems = [
        "Metal command encoder", "IOGPU allocator", "Unified Memory controller",
        "MLX graph scheduler", "GQA attention kernel", "SIMDgroup matmul tile",
        "KV page pool", "Neural Accelerator pipeline", "Thermal SMC sensor",
        "Grand Central Dispatch queue",
    ]
    events = [
        "completed buffer residency audit", "flushed staged activation cache",
        "rebalanced active expert routing", "verified tensor alignment boundary",
        "measured sustained bus throughput", "logged nominal clock frequency",
        "validated prompt prefix checksum", "retired asynchronous copy fence",
    ]
    sentences: list[str] = []
    total = 0
    seq = 1
    while total < target_chars:
        sub = rng.choice(subsystems)
        ev = rng.choice(events)
        val = rng.randint(10, 9999)
        s = f"Entry {seq:04d}: The {sub} {ev} at sample offset {val} with nominal status."
        sentences.append(s)
        total += len(s) + 1
        seq += 1
    return sentences


def stream_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    api_key: str = "EMPTY",
    timeout: float = 300.0,
    max_plausible_tps: float = DEFAULT_MAX_PLAUSIBLE_TPS,
    min_decode_window_s: float = MIN_DECODE_WINDOW_S,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    t_start = time.perf_counter()
    t_first: float | None = None
    chunk_times: list[float] = []
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue

            if "usage" in obj and isinstance(obj["usage"], dict):
                usage = obj["usage"]
            if "timings" in obj and isinstance(obj["timings"], dict):
                timings = obj["timings"]

            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    chunk_times.append(now)
                    text_parts.append(piece)

    t_end = time.perf_counter()
    total_s = max(t_end - t_start, 1e-9)
    ttft_s = (t_first - t_start) if t_first is not None else total_s
    gen_s = max(t_end - (t_first if t_first is not None else t_start), 0.0)
    output_text = "".join(text_parts)

    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    token_source = "server_usage"

    if completion_tokens is None and "predicted_n" in timings:
        completion_tokens = timings["predicted_n"]
        token_source = "server_timings"
    if prompt_tokens is None and "prompt_n" in timings:
        prompt_tokens = timings["prompt_n"]
    if completion_tokens is None:
        completion_tokens = len(chunk_times)
        token_source = "sse_chunks_fallback"

    flags: list[str] = []
    decode_tps: float | None = None

    if "predicted_per_second" in timings and timings["predicted_per_second"]:
        decode_tps = float(timings["predicted_per_second"])
    else:
        # Guard against single-chunk coalesced responses producing fake 100k+ tok/s (PR #9)
        if completion_tokens > 1 and gen_s < min_decode_window_s:
            flags.append("COALESCED")
            decode_tps = None
        elif completion_tokens > 0 and gen_s > 0:
            denom_tokens = max(completion_tokens - 1, 1) if completion_tokens > 1 else completion_tokens
            decode_tps = denom_tokens / gen_s
        else:
            decode_tps = 0.0

    if is_degenerate(output_text):
        flags.append("DEGENERATE")

    if decode_tps is not None and decode_tps > max_plausible_tps:
        flags.append("IMPLAUSIBLE")

    if "prompt_per_second" in timings and timings["prompt_per_second"]:
        prefill_tps = float(timings["prompt_per_second"])
    elif prompt_tokens:
        prefill_tps = float(prompt_tokens) / max(ttft_s, 1e-9)
    else:
        prefill_tps = 0.0

    itl_ms = [
        (chunk_times[i] - chunk_times[i - 1]) * 1000.0
        for i in range(1, len(chunk_times))
    ]

    valid = len(flags) == 0 and decode_tps is not None
    return {
        "ttft_s": round(ttft_s, 4),
        "total_s": round(total_s, 4),
        "prefill_tps": round(prefill_tps, 2),
        "decode_tps": round(decode_tps, 3) if decode_tps is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_source": token_source,
        "itl_p50_ms": round(percentile(itl_ms, 50), 2),
        "itl_p95_ms": round(percentile(itl_ms, 95), 2),
        "flags": flags,
        "valid": valid,
        "output_text": output_text,
    }


def calibrate_chars_per_token(
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> float:
    """Estimate actual chars/token from the server's own tokenizer using 2 short probes (PR #11)."""
    samples: list[float] = []
    for target_chars in (400, 1200):
        prompt = " ".join(make_heterogeneous_filler(target_chars, seed=target_chars))
        res = stream_chat_completion(
            base_url, model, prompt, max_tokens=2, temperature=0.0, api_key=api_key, timeout=timeout,
            min_decode_window_s=0.0,
        )
        pt = res.get("prompt_tokens")
        if pt and pt > 10:
            samples.append(len(prompt) / float(pt))
    if not samples:
        return 4.0
    return max(1.5, min(8.0, statistics.median(samples)))


def append_ledger(ledger_path: str | None, record: dict[str, Any]) -> None:
    if not ledger_path:
        return
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    valid_runs = [r for r in runs if r.get("valid", True) and r.get("decode_tps") is not None]
    flagged_count = len(runs) - len(valid_runs)
    use = valid_runs if valid_runs else runs

    tps = [r["decode_tps"] for r in valid_runs if r["decode_tps"] is not None]
    pf = [r["prefill_tps"] for r in use]
    ttft = [r["ttft_s"] for r in use]
    itl95 = [r["itl_p95_ms"] for r in use]

    med_tps = statistics.median(tps) if tps else 0.0
    mad_tps = statistics.median([abs(x - med_tps) for x in tps]) if len(tps) > 1 else 0.0
    all_flags = sorted({fl for r in runs for fl in r.get("flags", [])})

    return {
        "n": len(runs),
        "valid_n": len(valid_runs),
        "flagged_n": flagged_count,
        "flags": all_flags,
        "decode_tps_median": round(med_tps, 3),
        "decode_tps_mad": round(mad_tps, 3),
        "prefill_tps_median": round(statistics.median(pf), 2) if pf else 0.0,
        "ttft_s_median": round(statistics.median(ttft), 4) if ttft else 0.0,
        "itl_p95_ms_median": round(statistics.median(itl95), 2) if itl95 else 0.0,
        "token_sources": sorted({r["token_source"] for r in runs}),
    }


def score_triad(
    pre_val: float,
    cand_val: float,
    post_val: float,
    direction: str,
    min_gain_pct: float,
) -> dict[str, Any]:
    ctl_mid = (pre_val + post_val) / 2.0
    drift_pct = abs(post_val - pre_val) / max(pre_val, 1e-9) * 100.0
    if direction == "higher":
        gain_pct = (cand_val - ctl_mid) / max(ctl_mid, 1e-9) * 100.0
    else:
        gain_pct = (ctl_mid - cand_val) / max(ctl_mid, 1e-9) * 100.0

    if drift_pct > max(abs(gain_pct), 3.0):
        verdict = "NOISY"
    elif gain_pct >= min_gain_pct:
        verdict = "KEEP"
    else:
        verdict = "REJECT"

    return {
        "control_pre": round(pre_val, 4),
        "candidate": round(cand_val, 4),
        "control_post": round(post_val, 4),
        "control_drift_pct": round(drift_pct, 2),
        "gain_pct": round(gain_pct, 2),
        "verdict": verdict,
    }


def capture_mem(pid: int | None) -> dict[str, Any] | None:
    if pid is None or mem_macos is None:
        return None
    return mem_macos.collect_snapshot(pid)


def attach_mem_delta(record: dict[str, Any], pre: dict[str, Any] | None, post: dict[str, Any] | None) -> None:
    if pre is not None and post is not None and mem_macos is not None:
        record["mem_delta"] = mem_macos.diff_snapshots(pre, post)


def cmd_baseline(args: argparse.Namespace) -> int:
    for _ in range(args.warmup):
        stream_chat_completion(
            args.base_url, args.model, make_prompt(128, False), min(32, args.max_tokens),
            api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )

    cpt = (
        calibrate_chars_per_token(args.base_url, args.model, args.api_key, args.timeout)
        if args.calibrate_tokenizer
        else 4.0
    )

    mem_pre = capture_mem(args.mem_pid)
    runs = []
    for _ in range(args.runs):
        prompt = make_prompt(args.prompt_tokens, args.allow_cache, chars_per_token=cpt)
        res = stream_chat_completion(
            args.base_url, args.model, prompt, args.max_tokens,
            api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )
        res.pop("output_text", None)
        runs.append(res)
    mem_post = capture_mem(args.mem_pid)

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "baseline",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "prompt_tokens_target": args.prompt_tokens,
        "chars_per_token": round(cpt, 3),
        "max_tokens": args.max_tokens,
        "summary": summarize_runs(runs),
        "runs": runs,
    }
    attach_mem_delta(record, mem_pre, mem_post)
    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    """Fast ~60s onboarding smoke test across prose, code, prefix-cache reuse, and greedy determinism (PR #24)."""
    mem_pre = capture_mem(args.mem_pid)

    prose = stream_chat_completion(
        args.base_url, args.model, make_prompt(256, False), args.max_tokens,
        temperature=0.0, api_key=args.api_key, timeout=args.timeout,
        max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
    )
    code_prompt = (
        f"[nonce-{random.randint(100000, 999999)}] Write a clean Python function that parses "
        "a JSONL benchmark ledger and computes median and MAD per experiment label."
    )
    code = stream_chat_completion(
        args.base_url, args.model, code_prompt, args.max_tokens,
        temperature=0.0, api_key=args.api_key, timeout=args.timeout,
        max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
    )

    shared_prefix = " ".join(make_heterogeneous_filler(2000, seed=42))
    turn1 = stream_chat_completion(
        args.base_url, args.model, shared_prefix + "\nQuestion: Summarize in one sentence.", 48,
        temperature=0.0, api_key=args.api_key, timeout=args.timeout,
        max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
    )
    turn2 = stream_chat_completion(
        args.base_url, args.model, shared_prefix + "\nQuestion: List two subsystems mentioned.", 48,
        temperature=0.0, api_key=args.api_key, timeout=args.timeout,
        max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
    )

    greedy_prompt = "Reply with the exact first 5 prime numbers separated by commas."
    g1 = stream_chat_completion(
        args.base_url, args.model, greedy_prompt, 32, temperature=0.0,
        api_key=args.api_key, timeout=args.timeout, min_decode_window_s=0.0,
    )
    g2 = stream_chat_completion(
        args.base_url, args.model, greedy_prompt, 32, temperature=0.0,
        api_key=args.api_key, timeout=args.timeout, min_decode_window_s=0.0,
    )
    greedy_deterministic = (g1["output_text"] == g2["output_text"]) and bool(g1["output_text"].strip())
    mem_post = capture_mem(args.mem_pid)

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "screen",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "prose_decode_tps": prose["decode_tps"],
        "prose_flags": prose["flags"],
        "code_decode_tps": code["decode_tps"],
        "code_flags": code["flags"],
        "prefix_turn1_ttft_s": turn1["ttft_s"],
        "prefix_turn2_ttft_s": turn2["ttft_s"],
        "prefix_cache_speedup": round(turn1["ttft_s"] / max(turn2["ttft_s"], 1e-6), 2),
        "greedy_deterministic": greedy_deterministic,
        "passed": (prose["valid"] and code["valid"] and greedy_deterministic),
    }
    attach_mem_delta(record, mem_pre, mem_post)
    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0 if record["passed"] else 1


def cmd_ab(args: argparse.Namespace) -> int:
    for url in (args.control_url, args.candidate_url):
        stream_chat_completion(
            url, args.model, make_prompt(128, False), min(32, args.max_tokens),
            api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )

    mem_pre = capture_mem(args.mem_pid)
    ctl_pre: list[dict[str, Any]] = []
    cand: list[dict[str, Any]] = []
    ctl_post: list[dict[str, Any]] = []

    for _ in range(args.cycles):
        r1 = stream_chat_completion(
            args.control_url, args.model, make_prompt(args.prompt_tokens, args.allow_cache),
            args.max_tokens, api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )
        r2 = stream_chat_completion(
            args.candidate_url, args.model, make_prompt(args.prompt_tokens, args.allow_cache),
            args.max_tokens, api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )
        r3 = stream_chat_completion(
            args.control_url, args.model, make_prompt(args.prompt_tokens, args.allow_cache),
            args.max_tokens, api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )
        for r in (r1, r2, r3):
            r.pop("output_text", None)
        ctl_pre.append(r1)
        cand.append(r2)
        ctl_post.append(r3)
    mem_post = capture_mem(args.mem_pid)

    _, direction = METRIC_SPECS[args.metric]
    v_pre = [r[args.metric] for r in ctl_pre if r["valid"] and r[args.metric] is not None]
    v_cand = [r[args.metric] for r in cand if r["valid"] and r[args.metric] is not None]
    v_post = [r[args.metric] for r in ctl_post if r["valid"] and r[args.metric] is not None]

    if not (v_pre and v_cand and v_post):
        scored = {"verdict": "INVALID_SAMPLES", "control_drift_pct": None, "gain_pct": None}
    else:
        scored = score_triad(
            statistics.median(v_pre),
            statistics.median(v_cand),
            statistics.median(v_post),
            direction,
            args.min_gain_pct,
        )

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "ab",
        "label": args.label,
        "model": args.model,
        "metric": args.metric,
        "cycles": args.cycles,
        "control_summary": summarize_runs(ctl_pre + ctl_post),
        "candidate_summary": summarize_runs(cand),
        **scored,
    }
    attach_mem_delta(record, mem_pre, mem_post)
    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Score a sequential single-server control-pre / candidate / control-post triad from ledger.jsonl."""
    p = Path(args.ledger)
    if not p.exists():
        print(f"Error: ledger not found: {p}", file=sys.stderr)
        return 2

    latest_by_label: dict[str, dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("mode") == "baseline" and "label" in obj:
            latest_by_label[obj["label"]] = obj

    missing = [
        lbl for lbl in (args.control_pre_label, args.candidate_label, args.control_post_label)
        if lbl not in latest_by_label
    ]
    if missing:
        print(f"Error: missing baseline labels in ledger: {missing}", file=sys.stderr)
        return 2

    summary_key, direction = METRIC_SPECS[args.metric]
    pre_rec = latest_by_label[args.control_pre_label]
    cand_rec = latest_by_label[args.candidate_label]
    post_rec = latest_by_label[args.control_post_label]

    if cand_rec["summary"].get("valid_n", 1) == 0:
        scored = {"verdict": "INVALID_SAMPLES", "control_drift_pct": None, "gain_pct": None}
    else:
        scored = score_triad(
            float(pre_rec["summary"][summary_key]),
            float(cand_rec["summary"][summary_key]),
            float(post_rec["summary"][summary_key]),
            direction,
            args.min_gain_pct,
        )
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "compare",
        "label": args.label or f"compare-{args.candidate_label}",
        "metric": args.metric,
        "control_pre_label": args.control_pre_label,
        "candidate_label": args.candidate_label,
        "control_post_label": args.control_post_label,
        **scored,
    }
    if not args.no_write:
        append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0


def cmd_aa(args: argparse.Namespace) -> int:
    """Measure empirical A/A noise floor across back-to-back unchanged runs (PR #11)."""
    p = Path(args.ledger)
    if not p.exists():
        print(f"Error: ledger not found: {p}", file=sys.stderr)
        return 2

    vals: list[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("mode") != "baseline":
            continue
        if args.label and obj.get("label") != args.label:
            continue
        for r in obj.get("runs", []):
            if r.get("valid", True) and r.get(args.metric) is not None:
                vals.append(float(r[args.metric]))

    if len(vals) < 3:
        print(f"Error: need at least 3 valid samples for A/A noise floor (found {len(vals)}).", file=sys.stderr)
        return 2

    _, direction = METRIC_SPECS[args.metric]
    triads = len(vals) - 2
    false_wins = 0
    max_false_win_pct = 0.0

    for i in range(1, len(vals) - 1):
        prev_v, mid_v, next_v = vals[i - 1], vals[i], vals[i + 1]
        if direction == "higher":
            better_nbr = max(prev_v, next_v)
            if mid_v > better_nbr:
                false_wins += 1
                pct = (mid_v - better_nbr) / max(better_nbr, 1e-9) * 100.0
                max_false_win_pct = max(max_false_win_pct, pct)
        else:
            better_nbr = min(prev_v, next_v)
            if mid_v < better_nbr:
                false_wins += 1
                pct = (better_nbr - mid_v) / max(better_nbr, 1e-9) * 100.0
                max_false_win_pct = max(max_false_win_pct, pct)

    suggested_floor = round(max(2.0, max_false_win_pct * 1.25), 2)
    out = {
        "mode": "aa_noise_floor",
        "metric": args.metric,
        "samples": len(vals),
        "triads_checked": triads,
        "false_wins": false_wins,
        "false_win_rate_pct": round(false_wins / triads * 100.0, 1),
        "max_no_change_win_pct": round(max_false_win_pct, 2),
        "recommended_min_gain_pct": suggested_floor,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_concurrency(args: argparse.Namespace) -> int:
    mem_pre = capture_mem(args.mem_pid)
    results = []
    for level in args.levels:
        t0 = time.perf_counter()
        worker_runs: list[dict[str, Any]] = []
        errors = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
            futs = [
                pool.submit(
                    stream_chat_completion,
                    args.base_url,
                    args.model,
                    make_prompt(args.prompt_tokens, args.allow_cache),
                    args.max_tokens,
                    0.0,
                    args.api_key,
                    args.timeout,
                    args.max_plausible_tps,
                    args.min_decode_window_s,
                )
                for _ in range(level)
            ]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    r = fut.result()
                    r.pop("output_text", None)
                    worker_runs.append(r)
                except Exception:
                    errors += 1
        wall_s = max(time.perf_counter() - t0, 1e-9)
        valid_runs = [r for r in worker_runs if r["valid"] and r["decode_tps"] is not None]
        total_out_tokens = sum(r["completion_tokens"] or 0 for r in valid_runs)
        results.append(
            {
                "concurrency": level,
                "errors": errors,
                "flagged_samples": len(worker_runs) - len(valid_runs),
                "wall_s": round(wall_s, 3),
                "aggregate_decode_tps": round(total_out_tokens / wall_s, 3),
                "per_stream_median_tps": round(
                    statistics.median([r["decode_tps"] for r in valid_runs]), 3
                ) if valid_runs else 0.0,
                "ttft_p95_s": round(
                    percentile([r["ttft_s"] for r in worker_runs], 95), 4
                ) if worker_runs else 0.0,
            }
        )
    mem_post = capture_mem(args.mem_pid)

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "concurrency",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "levels": results,
    }
    attach_mem_delta(record, mem_pre, mem_post)
    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0


def cmd_soak(args: argparse.Namespace) -> int:
    """Continuous generation thermal stability gate comparing early vs late decode speed."""
    mem_pre = capture_mem(args.mem_pid)
    t_start = time.perf_counter()
    samples: list[dict[str, Any]] = []

    while (time.perf_counter() - t_start) < args.duration_s or len(samples) < args.min_runs:
        res = stream_chat_completion(
            args.base_url, args.model, make_prompt(args.prompt_tokens, False),
            args.max_tokens, api_key=args.api_key, timeout=args.timeout,
            max_plausible_tps=args.max_plausible_tps, min_decode_window_s=args.min_decode_window_s,
        )
        if res["valid"] and res["decode_tps"] is not None:
            samples.append(
                {
                    "elapsed_s": round(time.perf_counter() - t_start, 2),
                    "decode_tps": res["decode_tps"],
                    "ttft_s": res["ttft_s"],
                }
            )
        if (time.perf_counter() - t_start) >= args.duration_s and len(samples) >= args.min_runs:
            break

    mem_post = capture_mem(args.mem_pid)
    win = max(1, len(samples) // 3)
    early_med = statistics.median([s["decode_tps"] for s in samples[:win]])
    late_med = statistics.median([s["decode_tps"] for s in samples[-win:]])
    droop_pct = max(0.0, (early_med - late_med) / max(early_med, 1e-9) * 100.0)

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "soak",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "duration_s": round(time.perf_counter() - t_start, 2),
        "runs": len(samples),
        "early_decode_tps_median": round(early_med, 3),
        "late_decode_tps_median": round(late_med, 3),
        "thermal_droop_pct": round(droop_pct, 2),
    }
    attach_mem_delta(record, mem_pre, mem_post)
    throttled = bool(record.get("mem_delta", {}).get("thermal_throttled", False))
    passed = (droop_pct <= args.max_droop_pct) and (not throttled)
    record["passed"] = passed

    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0 if passed else 1


def cmd_needle(args: argparse.Namespace) -> int:
    cpt = (
        4.0
        if args.no_calibrate_tokenizer
        else calibrate_chars_per_token(args.base_url, args.model, args.api_key, args.timeout)
    )
    target_chars = max(512, int(args.target_tokens * cpt))
    checks = []
    all_passed = True
    mem_pre = capture_mem(args.mem_pid)

    for i, depth in enumerate(args.depths):
        code = "KEY-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        needle = f"IMPORTANT RECORD: The verified calibration code is {code}. Remember {code}."
        sentences = make_heterogeneous_filler(target_chars, seed=1000 + i)
        idx = max(0, min(len(sentences), int(len(sentences) * depth)))
        sentences.insert(idx, needle)
        doc = " ".join(sentences)
        prompt = (
            f"{doc}\n\nQuestion: What is the exact verified calibration code mentioned in the text above? "
            "Reply with the exact code only."
        )
        res = stream_chat_completion(
            args.base_url, args.model, prompt, max_tokens=args.max_tokens, temperature=0.0,
            api_key=args.api_key, timeout=args.timeout, min_decode_window_s=0.0,
        )
        passed = code in res["output_text"]
        if not passed:
            all_passed = False
        checks.append(
            {
                "depth": depth,
                "expected": code,
                "passed": passed,
                "prompt_tokens": res["prompt_tokens"],
                "ttft_s": res["ttft_s"],
                "prefill_tps": res["prefill_tps"],
                "response_excerpt": res["output_text"][:160],
            }
        )
    mem_post = capture_mem(args.mem_pid)

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "needle",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "target_tokens": args.target_tokens,
        "chars_per_token": round(cpt, 3),
        "all_passed": all_passed,
        "checks": checks,
    }
    attach_mem_delta(record, mem_pre, mem_post)
    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2))
    return 0 if all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Apple Silicon OpenAI-compatible benchmark harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, need_model: bool = True) -> None:
        if need_model:
            p.add_argument("--model", required=True, help="Model ID served by endpoint")
        p.add_argument("--label", default="run", help="Experiment label for ledger")
        p.add_argument("--api-key", default="EMPTY", help="Bearer API key if required")
        p.add_argument("--timeout", type=float, default=300.0, help="HTTP request timeout in seconds")
        p.add_argument("--mem-pid", type=int, help="Optional server PID for pre/post macOS memory & thermal telemetry")
        p.add_argument("--max-plausible-tps", type=float, default=DEFAULT_MAX_PLAUSIBLE_TPS)
        p.add_argument("--min-decode-window-s", type=float, default=MIN_DECODE_WINDOW_S)
        p.add_argument("--ledger", help="Optional path to append-only ledger.jsonl")

    p_base = sub.add_parser("baseline", help="Measure single-server baseline")
    add_common(p_base)
    p_base.add_argument("--base-url", required=True)
    p_base.add_argument("--runs", type=int, default=5)
    p_base.add_argument("--warmup", type=int, default=1)
    p_base.add_argument("--prompt-tokens", type=int, default=512)
    p_base.add_argument("--max-tokens", type=int, default=256)
    p_base.add_argument("--allow-cache", action="store_true")
    p_base.add_argument("--calibrate-tokenizer", action="store_true")
    p_base.set_defaults(func=cmd_baseline)

    p_scr = sub.add_parser("screen", help="Fast ~60s onboarding smoke test for engine bake-offs")
    add_common(p_scr)
    p_scr.add_argument("--base-url", required=True)
    p_scr.add_argument("--max-tokens", type=int, default=128)
    p_scr.set_defaults(func=cmd_screen)

    p_ab = sub.add_parser("ab", help="Dual-server interleaved control -> candidate -> control A/B test")
    add_common(p_ab)
    p_ab.add_argument("--control-url", required=True)
    p_ab.add_argument("--candidate-url", required=True)
    p_ab.add_argument("--metric", choices=list(METRIC_SPECS.keys()), default="decode_tps")
    p_ab.add_argument("--cycles", type=int, default=3)
    p_ab.add_argument("--prompt-tokens", type=int, default=512)
    p_ab.add_argument("--max-tokens", type=int, default=256)
    p_ab.add_argument("--min-gain-pct", type=float, default=3.0)
    p_ab.add_argument("--allow-cache", action="store_true")
    p_ab.set_defaults(func=cmd_ab)

    p_cmp = sub.add_parser("compare", help="Score sequential single-server control-pre / candidate / control-post triad")
    p_cmp.add_argument("--ledger", required=True, help="Path to ledger.jsonl containing the three baseline runs")
    p_cmp.add_argument("--control-pre-label", required=True)
    p_cmp.add_argument("--candidate-label", required=True)
    p_cmp.add_argument("--control-post-label", required=True)
    p_cmp.add_argument("--metric", choices=list(METRIC_SPECS.keys()), default="decode_tps")
    p_cmp.add_argument("--min-gain-pct", type=float, default=3.0)
    p_cmp.add_argument("--label", default="")
    p_cmp.add_argument("--no-write", action="store_true", help="Do not append comparison result back to ledger")
    p_cmp.set_defaults(func=cmd_compare)

    p_aa = sub.add_parser("aa", help="Compute empirical A/A noise floor from unchanged baseline runs")
    p_aa.add_argument("--ledger", required=True)
    p_aa.add_argument("--label", default="", help="Optional baseline label filter")
    p_aa.add_argument("--metric", choices=list(METRIC_SPECS.keys()), default="decode_tps")
    p_aa.set_defaults(func=cmd_aa)

    p_conc = sub.add_parser("concurrency", help="Sweep concurrent client load")
    add_common(p_conc)
    p_conc.add_argument("--base-url", required=True)
    p_conc.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    p_conc.add_argument("--prompt-tokens", type=int, default=512)
    p_conc.add_argument("--max-tokens", type=int, default=128)
    p_conc.add_argument("--allow-cache", action="store_true")
    p_conc.set_defaults(func=cmd_concurrency)

    p_soak = sub.add_parser("soak", help="Continuous generation thermal stability gate")
    add_common(p_soak)
    p_soak.add_argument("--base-url", required=True)
    p_soak.add_argument("--duration-s", type=float, default=180.0)
    p_soak.add_argument("--min-runs", type=int, default=6)
    p_soak.add_argument("--prompt-tokens", type=int, default=512)
    p_soak.add_argument("--max-tokens", type=int, default=256)
    p_soak.add_argument("--max-droop-pct", type=float, default=5.0)
    p_soak.set_defaults(func=cmd_soak)

    p_needle = sub.add_parser("needle", help="Long-context needle retrieval quality gate")
    add_common(p_needle)
    p_needle.add_argument("--base-url", required=True)
    p_needle.add_argument("--target-tokens", type=int, default=8192)
    p_needle.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p_needle.add_argument("--max-tokens", type=int, default=512)
    p_needle.add_argument("--no-calibrate-tokenizer", action="store_true")
    p_needle.set_defaults(func=cmd_needle)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as exc:
        print(f"HTTP connection error: {exc}", file=sys.stderr)
        sys.exit(1)
