#!/usr/bin/env swift
// Zero-dependency native Metal unified-memory bandwidth & MTLDevice probe for macOS Apple Silicon.
// Run directly via: swift scripts/measure_bw_metal.swift [--mb 512] [--iters 20] [--json]
// Uses MTLBlitCommandEncoder device-to-device copy (reads + writes = 2x bytes moved) to measure
// real sustained unified-memory bandwidth in decimal GB/s for `scripts/ceiling.py --bw-gbs`.

import Foundation
import Metal

var sizeMB = 512
var iters = 20
var emitJSON = false

var idx = 1
while idx < CommandLine.arguments.count {
    let arg = CommandLine.arguments[idx]
    if arg == "--mb", idx + 1 < CommandLine.arguments.count, let v = Int(CommandLine.arguments[idx + 1]) {
        sizeMB = max(64, v)
        idx += 2
    } else if arg == "--iters", idx + 1 < CommandLine.arguments.count, let v = Int(CommandLine.arguments[idx + 1]) {
        iters = max(3, v)
        idx += 2
    } else if arg == "--json" {
        emitJSON = true
        idx += 1
    } else {
        idx += 1
    }
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("Error: No Metal device found.\n", stderr)
    exit(1)
}
guard let queue = device.makeCommandQueue() else {
    fputs("Error: Could not create Metal command queue.\n", stderr)
    exit(1)
}

let byteCount = sizeMB * 1024 * 1024
guard let src = device.makeBuffer(length: byteCount, options: .storageModeShared),
      let dst = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
    fputs("Error: Failed to allocate \(sizeMB) MiB Metal shared buffers.\n", stderr)
    exit(1)
}

// Touch pages so both buffers are faulted in before timing.
memset(src.contents(), 0x5A, byteCount)
memset(dst.contents(), 0x00, byteCount)

func runCopy() -> Double {
    guard let cmdBuf = queue.makeCommandBuffer(),
          let blit = cmdBuf.makeBlitCommandEncoder() else {
        fputs("Error: Could not encode Metal blit.\n", stderr)
        exit(1)
    }
    blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: byteCount)
    blit.endEncoding()
    let t0 = CFAbsoluteTimeGetCurrent()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()
    let t1 = CFAbsoluteTimeGetCurrent()
    return max(t1 - t0, 1e-9)
}

// Warm-up passes
_ = runCopy()
_ = runCopy()

var bwSamples: [Double] = []
for _ in 0..<iters {
    let dt = runCopy()
    // Blit copy reads `byteCount` and writes `byteCount` => 2 * byteCount total bus traffic (decimal GB/s).
    let gbs = (Double(byteCount) * 2.0) / dt / 1e9
    bwSamples.append(gbs)
}

bwSamples.sort()
let medianGBs = bwSamples[bwSamples.count / 2]
let peakGBs = bwSamples.last ?? medianGBs
let maxWSGiB = Double(device.recommendedMaxWorkingSetSize) / Double(1 << 30)
let currentAllocGiB = Double(device.currentAllocatedSize) / Double(1 << 30)

if emitJSON {
    let payload: [String: Any] = [
        "device_name": device.name,
        "has_unified_memory": device.hasUnifiedMemory,
        "recommended_max_working_set_gib": round(maxWSGiB * 100) / 100,
        "current_allocated_gib": round(currentAllocGiB * 100) / 100,
        "buffer_mib": sizeMB,
        "iterations": iters,
        "measured_bw_median_gbs": round(medianGBs * 100) / 100,
        "measured_bw_peak_gbs": round(peakGBs * 100) / 100,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]),
       let str = String(data: data, encoding: .utf8) {
        print(str)
    }
} else {
    print("=== Native Metal Unified-Memory Bandwidth Probe ===")
    print("Device:                       \(device.name) (unified=\(device.hasUnifiedMemory))")
    print(String(format: "Recommended max working set:  %.2f GiB", maxWSGiB))
    print(String(format: "Measured copy BW (median):    %.2f GB/s  (peak: %.2f GB/s across %d x %d MiB blits)", medianGBs, peakGBs, iters, sizeMB))
    print("Tip: Pass --bw-gbs \(String(format: "%.1f", medianGBs)) to scripts/ceiling.py for an empirically grounded ceiling.")
}
