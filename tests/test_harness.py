"""Stdlib unittest suite for inference-optimisation-macos."""

import http.server
import importlib.util
import json
import os
import re
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "inference-optimisation-macos"
SCRIPTS = SKILL_DIR / "scripts"
REFS = SKILL_DIR / "references"


class TestSkillStructure(unittest.TestCase):
    def test_skill_frontmatter_and_references(self) -> None:
        skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        lines = skill_md.splitlines()
        self.assertLess(len(lines), 500)

        m = re.match(r"^---\n(.*?)\n---\n", skill_md, re.S)
        self.assertIsNotNone(m)
        fm = m.group(1)
        name = re.search(r"^name:\s*(.+)$", fm, re.M).group(1).strip()
        desc = re.search(r"^description:\s*(.+)$", fm, re.M).group(1).strip()
        self.assertEqual(name, "inference-optimisation-macos")
        self.assertRegex(name, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
        self.assertGreater(len(desc), 20)
        self.assertLess(len(desc), 1024)

        for ref_path in re.findall(r"(?:references|scripts)/[A-Za-z0-9_.-]+", skill_md):
            self.assertTrue((SKILL_DIR / ref_path).exists(), f"Missing {ref_path}")

    def test_no_em_or_en_dashes_in_skill_docs(self) -> None:
        for md_file in [SKILL_DIR / "SKILL.md", *REFS.glob("*.md")]:
            text = md_file.read_text(encoding="utf-8")
            self.assertNotIn("\u2014", text, f"Em-dash in {md_file.name}")
            self.assertNotIn("\u2013", text, f"En-dash in {md_file.name}")


class TestCeilingCalculator(unittest.TestCase):
    def test_list_chips_without_required_flags(self) -> None:
        out = subprocess.check_output(
            ["python3", str(SCRIPTS / "ceiling.py"), "--list-chips"], text=True
        )
        self.assertIn("m5 ultra", out)
        self.assertIn("1228.0 GB/s", out)

    def test_moe_looped_roofline_and_kv_crossover(self) -> None:
        base_out = json.loads(
            subprocess.check_output(
                [
                    "python3", str(SCRIPTS / "ceiling.py"),
                    "--chip", "M4 Max 40-core", "--ram-gb", "128",
                    "--active-params-b", "3.5", "--total-params-b", "35",
                    "--bits", "4.5", "--ctx", "32768", "--concurrency", "2",
                    "--observed-tps", "180", "--json",
                ],
                text=True,
            )
        )
        self.assertEqual(base_out["architecture"], "moe")
        self.assertEqual(base_out["peak_bandwidth_gbs"], 546.0)
        self.assertTrue(base_out["fits_wired_limit"])
        self.assertGreater(base_out["kv_crossover_ctx"], 10000)

        loop_out = json.loads(
            subprocess.check_output(
                [
                    "python3", str(SCRIPTS / "ceiling.py"),
                    "--chip", "M4 Max 40-core", "--ram-gb", "128",
                    "--active-params-b", "3.5", "--total-params-b", "35",
                    "--loop-multiplier", "2.0", "--bits", "4.5", "--json",
                ],
                text=True,
            )
        )
        self.assertAlmostEqual(
            loop_out["theoretical_decode_tps_short_ctx"] * 2.0,
            base_out["theoretical_decode_tps_short_ctx"],
            delta=0.2,
        )


class TestMacOSMemoryParsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("mem_macos", str(SCRIPTS / "mem_macos.py"))
        cls.mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mm)

    def test_parse_vm_stat_swap_and_ioaccelerator(self) -> None:
        sample = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                              100000.\n"
            "Pages active:                           1200000.\n"
            "Pages wired down:                       2000000.\n"
            "Pages occupied by compressor:            150000.\n"
            "Swapouts:                                      7.\n"
        )
        parsed = self.mm.parse_vm_stat(sample)
        self.assertEqual(parsed["page_size_bytes"], 16384)
        self.assertGreater(parsed["wired_gib"], 30.0)
        self.assertEqual(parsed["swapouts_count"], 7)

        sw = self.mm.parse_swapusage("total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)")
        self.assertAlmostEqual(sw["swap_used_gib"], 0.5, places=2)

        io_sample = '"PerformanceStatistics" = {"Alloc system memory"=25769803776,"In use system memory"=21474836480}'
        io_parsed = self.mm.parse_ioaccelerator(io_sample)
        self.assertAlmostEqual(io_parsed["gpu_in_use_gib"], 20.0, places=2)
        self.assertAlmostEqual(io_parsed["gpu_alloc_gib"], 24.0, places=2)

    def test_parse_thermal_no_false_positive(self) -> None:
        nominal = self.mm.parse_thermal(
            "CPU_Scheduler_Limit = 100\nCPU_Available_CPUs = 16\nCPU_Speed_Limit = 100"
        )
        self.assertFalse(nominal["thermal_warning"])

        throttled = self.mm.parse_thermal("CPU_Speed_Limit = 80")
        self.assertTrue(throttled["thermal_warning"])

    def test_parse_memory_pressure(self) -> None:
        mp = self.mm.parse_memory_pressure(
            "The system has 137438953472 bytes.\nSystem-wide memory free percentage: 64%"
        )
        self.assertEqual(mp["system_free_pct"], 64)


class TestQueueSerializer(unittest.TestCase):
    def test_flock_serialization_and_exit_code_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_file = os.path.join(td, "gpu.lock")
            env = os.environ.copy()
            env["QUEUE_LOCK"] = lock_file

            # Propagates non-zero exit code cleanly
            rc = subprocess.call(
                ["bash", str(SCRIPTS / "queue.sh"), "fail-job", "--", "false"],
                env=env,
                stderr=subprocess.DEVNULL,
            )
            self.assertNotEqual(rc, 0)

            # Strict serialization under contention
            log_file = Path(td, "order.txt")
            p1 = subprocess.Popen(
                [
                    "bash", str(SCRIPTS / "queue.sh"), "j1", "--",
                    "bash", "-c", f"echo s1 >> {log_file}; sleep 0.2; echo e1 >> {log_file}",
                ],
                env=env,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.05)
            p2 = subprocess.Popen(
                [
                    "bash", str(SCRIPTS / "queue.sh"), "j2", "--",
                    "bash", "-c", f"echo s2 >> {log_file}; echo e2 >> {log_file}",
                ],
                env=env,
                stderr=subprocess.DEVNULL,
            )
            self.assertEqual(p1.wait(), 0)
            self.assertEqual(p2.wait(), 0)
            self.assertEqual(log_file.read_text(encoding="utf-8").splitlines(), ["s1", "e1", "s2", "e2"])


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def _make_mock_handler(ttft_delay: float, tok_delay: float, degenerate_coalesced: bool = False):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = body["messages"][0]["content"]
            max_toks = min(int(body.get("max_tokens", 12)), 12)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            if degenerate_coalesced:
                # Reproduces PR #9 M1 Max oMLX bug: single coalesced chunk of 64 '!' -> fake 863k tok/s
                chunk = {"choices": [{"delta": {"content": "!" * 64}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                usage = {"choices": [], "usage": {"prompt_tokens": 64, "completion_tokens": 64}}
                self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            time.sleep(ttft_delay)
            m_code = re.search(r"KEY-[A-Z0-9]{8}", prompt)
            if m_code:
                tokens = [("reasoning_content", "Thinking... "), ("content", m_code.group(0))]
            elif "first 5 prime numbers" in prompt:
                tokens = [("content", "2, 3, 5, 7, 11")]
            else:
                tokens = [("content", f"word{i} ") for i in range(max_toks)]

            for field, tok in tokens:
                time.sleep(tok_delay)
                chunk = {"choices": [{"delta": {field: tok}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

            est_prompt_toks = max(16, len(prompt) // 4)
            usage = {"choices": [], "usage": {"prompt_tokens": est_prompt_toks, "completion_tokens": len(tokens)}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class TestBenchHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.srv_ctl = _ThreadedTCPServer(("127.0.0.1", 0), _make_mock_handler(0.03, 0.010))
        cls.srv_cand = _ThreadedTCPServer(("127.0.0.1", 0), _make_mock_handler(0.01, 0.006))
        cls.srv_broken = _ThreadedTCPServer(("127.0.0.1", 0), _make_mock_handler(0.0, 0.0, degenerate_coalesced=True))
        cls.port_ctl = cls.srv_ctl.server_address[1]
        cls.port_cand = cls.srv_cand.server_address[1]
        cls.port_broken = cls.srv_broken.server_address[1]
        for s in (cls.srv_ctl, cls.srv_cand, cls.srv_broken):
            threading.Thread(target=s.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        for s in (cls.srv_ctl, cls.srv_cand, cls.srv_broken):
            s.shutdown()
            s.server_close()

    def test_coalesced_degenerate_sample_rejected(self) -> None:
        rec = json.loads(
            subprocess.check_output(
                [
                    "python3", str(SCRIPTS / "bench.py"), "baseline",
                    "--base-url", f"http://127.0.0.1:{self.port_broken}/v1",
                    "--model", "broken", "--label", "broken",
                    "--runs", "2", "--warmup", "0", "--max-tokens", "12",
                ],
                text=True,
            )
        )
        self.assertEqual(rec["summary"]["valid_n"], 0)
        self.assertIn("COALESCED", rec["summary"]["flags"])
        self.assertIn("DEGENERATE", rec["summary"]["flags"])
        self.assertEqual(rec["summary"]["decode_tps_median"], 0.0)

    def test_end_to_end_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = os.path.join(td, "ledger.jsonl")
            pid = str(os.getpid())

            # screen smoke test
            scr = json.loads(
                subprocess.check_output(
                    [
                        "python3", str(SCRIPTS / "bench.py"), "screen",
                        "--base-url", f"http://127.0.0.1:{self.port_cand}/v1",
                        "--model", "mock", "--label", "screen-cand",
                        "--max-tokens", "12", "--mem-pid", pid, "--ledger", ledger,
                    ],
                    text=True,
                )
            )
            self.assertTrue(scr["passed"])
            self.assertTrue(scr["greedy_deterministic"])

            # sequential baseline triad + A/A noise floor + compare
            for label, port, runs in [("ctl-pre", self.port_ctl, "4"), ("cand", self.port_cand, "3"), ("ctl-post", self.port_ctl, "3")]:
                rec = json.loads(
                    subprocess.check_output(
                        [
                            "python3", str(SCRIPTS / "bench.py"), "baseline",
                            "--base-url", f"http://127.0.0.1:{port}/v1",
                            "--model", "mock", "--label", label,
                            "--runs", runs, "--warmup", "1",
                            "--prompt-tokens", "64", "--max-tokens", "12",
                            "--calibrate-tokenizer", "--mem-pid", pid, "--ledger", ledger,
                        ],
                        text=True,
                    )
                )
                self.assertGreater(rec["summary"]["prefill_tps_median"], 0)
                self.assertGreater(rec["summary"]["valid_n"], 0)

            aa_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3", str(SCRIPTS / "bench.py"), "aa",
                        "--ledger", ledger, "--label", "ctl-pre", "--metric", "decode_tps",
                    ],
                    text=True,
                )
            )
            self.assertEqual(aa_rec["samples"], 4)
            self.assertGreaterEqual(aa_rec["recommended_min_gain_pct"], 2.0)

            cmp_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3", str(SCRIPTS / "bench.py"), "compare",
                        "--ledger", ledger,
                        "--control-pre-label", "ctl-pre",
                        "--candidate-label", "cand",
                        "--control-post-label", "ctl-post",
                        "--metric", "decode_tps",
                    ],
                    text=True,
                )
            )
            self.assertEqual(cmp_rec["verdict"], "KEEP")

            soak_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3", str(SCRIPTS / "bench.py"), "soak",
                        "--base-url", f"http://127.0.0.1:{self.port_cand}/v1",
                        "--model", "mock", "--label", "soak",
                        "--duration-s", "0.15", "--min-runs", "6", "--max-tokens", "12",
                        "--ledger", ledger,
                    ],
                    text=True,
                )
            )
            self.assertTrue(soak_rec["passed"])

            needle_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3", str(SCRIPTS / "bench.py"), "needle",
                        "--base-url", f"http://127.0.0.1:{self.port_cand}/v1",
                        "--model", "mock", "--label", "needle",
                        "--target-tokens", "512", "--depths", "0.1", "0.5", "0.9",
                        "--ledger", ledger,
                    ],
                    text=True,
                )
            )
            self.assertTrue(needle_rec["all_passed"])


if __name__ == "__main__":
    unittest.main()
