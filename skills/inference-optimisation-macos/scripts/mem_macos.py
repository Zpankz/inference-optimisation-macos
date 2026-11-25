#!/usr/bin/env python3
"""macOS-native unified memory, Metal wired-limit, memory pressure, swap, and thermal telemetry."""

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from typing import Any


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except Exception:
        return ""


def parse_vm_stat(text: str) -> dict[str, Any]:
    page_size = 16384
    m = re.search(r"page size of (\d+) bytes", text)
    if m:
        page_size = int(m.group(1))

    metrics: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val_clean = val.strip().rstrip(".")
        if val_clean.isdigit():
            metrics[key.strip()] = int(val_clean)

    def to_gib(name: str) -> float:
        return round(metrics.get(name, 0) * page_size / (1024**3), 3)

    return {
        "page_size_bytes": page_size,
        "wired_gib": to_gib("Pages wired down"),
        "active_gib": to_gib("Pages active"),
        "inactive_gib": to_gib("Pages inactive"),
        "compressed_gib": to_gib("Pages occupied by compressor"),
        "free_gib": to_gib("Pages free"),
        "pageouts_count": metrics.get("Pageouts", 0),
        "swapouts_count": metrics.get("Swapouts", 0),
    }


def parse_swapusage(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("total", "used", "free"):
        m = re.search(rf"{key}\s*=\s*([\d.]+)([MG])", text)
        if m:
            val = float(m.group(1))
            if m.group(2) == "M":
                val /= 1024.0
            out[f"swap_{key}_gib"] = round(val, 3)
    return out


def parse_memory_pressure(text: str) -> dict[str, Any]:
    m = re.search(r"free percentage:\s*(\d+)%", text, re.IGNORECASE)
    free_pct = int(m.group(1)) if m else None
    return {"system_free_pct": free_pct}


def parse_thermal(text: str) -> dict[str, Any]:
    cpu_limit = None
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    if m:
        cpu_limit = int(m.group(1))

    therm_level = None
    ml = re.search(r"Thermal_Level\s*=\s*(\d+)", text)
    if ml:
        therm_level = int(ml.group(1))

    lower = text.lower()
    explicit_warning = ("thermal warning" in lower) and ("no thermal warning" not in lower)
    throttled = (
        (cpu_limit is not None and cpu_limit < 100)
        or (therm_level is not None and therm_level > 0)
        or explicit_warning
    )
    return {
        "cpu_speed_limit_pct": cpu_limit,
        "thermal_level": therm_level,
        "thermal_warning": throttled if text else None,
        "raw": text,
    }


def sample_process(pid: int) -> dict[str, Any]:
    ps_out = run_cmd(["ps", "-o", "rss=,vsz=,pcpu=,comm=", "-p", str(pid)])
    if not ps_out:
        return {"pid": pid, "alive": False}
    parts = ps_out.split(None, 3)
    rss_kb = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    vsz_kb = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    pcpu = float(parts[2]) if len(parts) > 2 else 0.0
    comm = parts[3] if len(parts) > 3 else ""
    return {
        "pid": pid,
        "alive": True,
        "rss_gib": round(rss_kb / (1024**2), 3),
        "vsz_gib": round(vsz_kb / (1024**2), 3),
        "cpu_pct": pcpu,
        "command": comm,
    }


def sample_mlx() -> dict[str, Any] | None:
    try:
        import mlx.core as mx  # type: ignore

        info = mx.device_info()
        return {
            "active_gib": round(mx.get_active_memory() / (1024**3), 4),
            "peak_gib": round(mx.get_peak_memory() / (1024**3), 4),
            "cache_gib": round(mx.get_cache_memory() / (1024**3), 4),
            "max_recommended_working_set_gib": round(
                info.get("max_recommended_working_set_size", 0) / (1024**3), 3
            ),
        }
    except Exception:
        return None


def collect_snapshot(pid: int | None = None) -> dict[str, Any]:
    sys_name = platform.system()
    snap: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys_name,
        "arch": platform.machine(),
    }

    if sys_name == "Darwin":
        mem_bytes_str = run_cmd(["sysctl", "-n", "hw.memsize"])
        mem_gib = round(int(mem_bytes_str) / (1024**3), 2) if mem_bytes_str.isdigit() else None
        wired_mb_str = run_cmd(["sysctl", "-n", "iogpu.wired_limit_mb"])
        wired_mb = int(wired_mb_str) if wired_mb_str.isdigit() else None

        snap["ram_gib"] = mem_gib
        snap["iogpu_wired_limit_mb"] = wired_mb
        snap["vm_stat"] = parse_vm_stat(run_cmd(["vm_stat"]))
        snap["swap"] = parse_swapusage(run_cmd(["sysctl", "-n", "vm.swapusage"]))
        snap["memory_pressure"] = parse_memory_pressure(run_cmd(["memory_pressure", "-Q"]))
        snap["thermal"] = parse_thermal(run_cmd(["pmset", "-g", "therm"]))
    else:
        snap["note"] = "Non-macOS host; macOS sysctl/vm_stat/pmset metrics unavailable."

    if pid is not None:
        snap["process"] = sample_process(pid)

    mlx_info = sample_mlx()
    if mlx_info is not None:
        snap["mlx"] = mlx_info

    return snap


def diff_snapshots(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Compute concise memory and thermal delta between pre-run and post-run snapshots."""
    delta: dict[str, Any] = {}
    if "vm_stat" in pre and "vm_stat" in post:
        delta["wired_delta_gib"] = round(
            post["vm_stat"]["wired_gib"] - pre["vm_stat"]["wired_gib"], 3
        )
        delta["compressed_delta_gib"] = round(
            post["vm_stat"]["compressed_gib"] - pre["vm_stat"]["compressed_gib"], 3
        )
    if "swap" in pre and "swap" in post:
        delta["swap_delta_gib"] = round(
            post["swap"].get("swap_used_gib", 0.0) - pre["swap"].get("swap_used_gib", 0.0), 3
        )
    if "memory_pressure" in post:
        delta["system_free_pct_post"] = post["memory_pressure"].get("system_free_pct")
    if "thermal" in post:
        delta["thermal_throttled"] = bool(post["thermal"].get("thermal_warning"))
    if "process" in post:
        delta["pid_alive_post"] = post["process"].get("alive")
        delta["rss_post_gib"] = post["process"].get("rss_gib")
    return delta


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture macOS unified memory and thermal telemetry.")
    parser.add_argument("--pid", type=int, help="Optional inference server PID to track")
    parser.add_argument("--watch", type=float, default=0.0, help="Poll interval in seconds (0 = single snapshot)")
    parser.add_argument("--count", type=int, default=1, help="Number of samples when --watch > 0")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    samples = args.count if args.watch > 0 else 1
    for idx in range(samples):
        snap = collect_snapshot(args.pid)
        if args.json:
            print(json.dumps(snap))
        else:
            print(
                f"[{snap['timestamp']}] platform={snap['platform']} "
                f"ram={snap.get('ram_gib')}GiB wired_cap={snap.get('iogpu_wired_limit_mb')}MB"
            )
            if "vm_stat" in snap:
                vs = snap["vm_stat"]
                sw = snap.get("swap", {})
                mp = snap.get("memory_pressure", {})
                print(
                    f"  vm_stat: wired={vs['wired_gib']}GiB active={vs['active_gib']}GiB "
                    f"compressed={vs['compressed_gib']}GiB swap_used={sw.get('swap_used_gib', 0)}GiB "
                    f"free_pct={mp.get('system_free_pct')}%"
                )
            if "process" in snap:
                pr = snap["process"]
                print(f"  pid={pr['pid']} alive={pr['alive']} rss={pr.get('rss_gib')}GiB cpu={pr.get('cpu_pct')}%")
            if "thermal" in snap:
                th = snap["thermal"]
                print(f"  thermal: cpu_speed_limit={th.get('cpu_speed_limit_pct')}% warning={th.get('thermal_warning')}")
            if "mlx" in snap:
                mx = snap["mlx"]
                print(f"  mlx: active={mx['active_gib']}GiB peak={mx['peak_gib']}GiB cache={mx['cache_gib']}GiB")
            if "note" in snap:
                print(f"  note: {snap['note']}")
        if idx + 1 < samples and args.watch > 0:
            time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
