---
name: Measured lever / benchmark report
about: Submit a verified Apple Silicon lever result or gotcha from a real hill-climb
title: "[bench] "
labels: benchmark
---

## Hardware & runtime
- Chip (include GPU core bin) & RAM:
- macOS version:
- `iogpu.wired_limit_mb`:
- Runtime & version:
- Model & quantization:

## Lever tested
What single knob changed between control and candidate?

## Measured result (paste `ledger.jsonl` summary)
```json
```

## Quality & stability gates
- [ ] Interleaved A/B or sequential `compare` triad run
- [ ] `bench.py needle` passed at target context
- [ ] `bench.py soak` passed with <= 5% thermal droop
- [ ] Zero swap growth in `mem_macos.py`
