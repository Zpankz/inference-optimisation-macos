---
name: Bug report
about: Report a parser issue, script failure, or macOS compatibility bug
title: "[bug] "
labels: bug
---

## Environment
- Mac model & chip (include GPU core count):
- macOS version (`sw_vers`):
- Python version (`python3 --version`):
- Runtime & version (`mlx-lm` / `vllm-mlx` / `oMLX` / `llama-server`):

## What happened
Paste the command you ran and the full error output.

## Raw macOS command output (if a parser bug)
If `probe_apple_silicon.sh` or `mem_macos.py` misparsed a field, include the raw output of the failing command (`pmset -g therm`, `vm_stat`, `memory_pressure -Q`, or `sysctl iogpu.wired_limit_mb`).
