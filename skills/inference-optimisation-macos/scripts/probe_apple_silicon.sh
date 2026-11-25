#!/usr/bin/env bash
# Probe Apple Silicon hardware, unified memory wired limits, power/thermal state,
# and installed local LLM inference runtimes on macOS, with PASS/WARN hygiene gates.
set -euo pipefail

OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

echo "=== Apple Silicon Inference Environment Probe ==="
echo "OS: $OS_NAME ($ARCH)"

if [[ "$OS_NAME" != "Darwin" ]]; then
  echo "WARNING: Non-macOS host detected ($OS_NAME). Hardware sysctl/pmset checks skipped."
  exit 0
fi

echo "macOS Version: $(sw_vers -productVersion 2>/dev/null || echo unknown) ($(sw_vers -buildVersion 2>/dev/null || echo unknown))"
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo "Chip: $CHIP"

P_CORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo unknown)"
E_CORES="$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo unknown)"
TOTAL_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || echo unknown)"
echo "CPU Cores: total=$TOTAL_CORES (P-cores=$P_CORES, E-cores=$E_CORES)"

GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ {print $2; exit}' || true)"
echo "GPU Cores: ${GPU_CORES:-unknown}"

MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
MEM_GIB="$(awk -v b="$MEM_BYTES" 'BEGIN {printf "%.1f", b / 1073741824}')"
WIRED_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
if [[ "$WIRED_MB" == "0" ]]; then
  WIRED_DESC="0 (macOS default ~66-75% of RAM)"
else
  WIRED_GIB="$(awk -v m="$WIRED_MB" 'BEGIN {printf "%.1f", m / 1024}')"
  WIRED_DESC="${WIRED_MB} MB (${WIRED_GIB} GiB)"
fi
echo "Unified RAM: ${MEM_GIB} GiB"
echo "iogpu.wired_limit_mb: ${WIRED_DESC}"

echo ""
echo "--- Power & Thermal State ---"
BATT_OUT="$(pmset -g batt 2>/dev/null || true)"
echo "$BATT_OUT" | head -n 2
LOW_POWER="$(pmset -g 2>/dev/null | awk '/lowpowermode/ {print $2; exit}' || echo unknown)"
echo "Low Power Mode: ${LOW_POWER:-unknown} (must be 0 for benchmarking)"
THERM_OUT="$(pmset -g therm 2>/dev/null || true)"
echo "$THERM_OUT"

echo ""
echo "--- Memory Pressure & Swap ---"
SWAP_OUT="$(sysctl -n vm.swapusage 2>/dev/null || true)"
echo "$SWAP_OUT"
MP_OUT="$(memory_pressure -Q 2>/dev/null | head -n 3 || true)"
echo "$MP_OUT"

echo ""
echo "--- Benchmark Hygiene Gate Summary ---"
if echo "$BATT_OUT" | grep -qi "Battery Power"; then
  echo "  [WARN] Running on Battery Power (plug into AC to avoid 8-15% swings)"
else
  echo "  [PASS] Power source: AC or desktop"
fi

if [[ "$LOW_POWER" == "1" ]]; then
  echo "  [WARN] Low Power Mode is ENABLED (disable with: sudo pmset -a lowpowermode 0)"
else
  echo "  [PASS] Low Power Mode: disabled ($LOW_POWER)"
fi

CPU_LIMIT="$(echo "$THERM_OUT" | awk -F'= ' '/CPU_Speed_Limit/ {print $2; exit}' || true)"
if [[ -n "$CPU_LIMIT" ]] && [[ "$CPU_LIMIT" -lt 100 ]]; then
  echo "  [WARN] Thermal throttling active (CPU_Speed_Limit=${CPU_LIMIT}%)"
else
  echo "  [PASS] Thermal state nominal (CPU_Speed_Limit=${CPU_LIMIT:-100}%)"
fi

MEM_GIB_INT="${MEM_GIB%.*}"
if [[ "$WIRED_MB" == "0" ]] && [[ "${MEM_GIB_INT:-0}" -ge 32 ]]; then
  echo "  [WARN] iogpu.wired_limit_mb is at macOS default (0); consider raising to 75-80% of RAM for large models"
else
  echo "  [PASS] iogpu.wired_limit_mb: ${WIRED_DESC}"
fi

echo ""
echo "--- Installed Inference Runtimes & Telemetry Tools ---"
for bin in llama-server llama-cli mlx_lm.server vllm-mlx omlx ollama macmon asitop; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "  [OK] $bin -> $(command -v "$bin")"
  else
    echo "  [--] $bin (not in PATH)"
  fi
done

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY' 2>/dev/null || true
import importlib.util
for pkg in ("mlx", "mlx_lm", "vllm_mlx"):
    found = importlib.util.find_spec(pkg) is not None
    ver = ""
    if found:
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", "")
        except Exception:
            ver = "installed"
    status = f"[OK] python:{pkg} {ver}".strip() if found else f"[--] python:{pkg}"
    print(f"  {status}")
try:
    import mlx.core as mx
    info = mx.device_info()
    max_ws = info.get("max_recommended_working_set_size", 0)
    print(f"  [MLX] device_info max_recommended_working_set_size = {max_ws / (1024**3):.2f} GiB")
except Exception:
    pass
PY
fi
