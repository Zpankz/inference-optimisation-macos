# Contributing

Contributions that add measured Apple Silicon data, fix macOS parsing edge cases, or tighten the benchmark harness are welcome.

## Ground rules

1. **Evidence over folklore**: New entries in `references/levers.md`, `references/gotchas.md`, or `references/ceilings.md` should include the exact chip (including GPU core bin), macOS version, runtime version, model/quant, and before/after numbers from `ledger.jsonl`.
2. **Zero external dependencies in `scripts/`**: Keep Python scripts stdlib-only (Python 3.10+) and shell scripts POSIX/macOS BSD compatible. Do not introduce Linux-only utilities (`flock`, `/proc`, GNU-only `sed`/`awk` flags).
3. **Keep `SKILL.md` lean**: `SKILL.md` stays under 500 lines and holds the happy-path workflow. Deep tables, flag lists, and edge-case debugging belong in `references/`.
4. **Tests pass on macOS and Linux**: Run `python3 -m unittest discover -s tests -v` before opening a PR.

## Local validation

```bash
bash -n skills/inference-optimisation-macos/scripts/*.sh
python3 -m py_compile skills/inference-optimisation-macos/scripts/*.py
python3 -m unittest discover -s tests -v
```
