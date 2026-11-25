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

    def test_moe_roofline_and_kv_crossover(self) -> None:
        out = subprocess.check_output(
            [
                "python3",
                str(SCRIPTS / "ceiling.py"),
                "--chip",
                "M4 Max 40-core",
                "--ram-gb",
                "128",
                "--active-params-b",
                "3.5",
                "--total-params-b",
                "35",
                "--bits",
                "4.5",
                "--ctx",
                "32768",
                "--concurrency",
                "2",
                "--observed-tps",
                "180",
                "--json",
            ],
            text=True,
        )
        data = json.loads(out)
        self.assertEqual(data["architecture"], "moe")
        self.assertEqual(data["peak_bandwidth_gbs"], 546.0)
        self.assertTrue(data["fits_wired_limit"])
        self.assertGreater(data["kv_crossover_ctx"], 10000)
        self.assertTrue(any("KV-SIGNIFICANT" in d for d in data["diagnostics"]))


class TestMacOSMemoryParsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("mem_macos", str(SCRIPTS / "mem_macos.py"))
        cls.mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mm)

    def test_parse_vm_stat_and_swap(self) -> None:
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
    def test_stale_lock_reap_and_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_dir = os.path.join(td, "gpu.lock")
            os.mkdir(lock_dir)
            Path(lock_dir, "pid").write_text("999999", encoding="utf-8")

            env = os.environ.copy()
            env["QUEUE_LOCK_DIR"] = lock_dir
            env["QUEUE_POLL_INTERVAL"] = "0.05"

            subprocess.run(
                ["bash", str(SCRIPTS / "queue.sh"), "stale", "--", "true"],
                env=env,
                check=True,
                stderr=subprocess.DEVNULL,
            )
            self.assertFalse(os.path.exists(lock_dir))

            log_file = Path(td, "order.txt")
            p1 = subprocess.Popen(
                [
                    "bash",
                    str(SCRIPTS / "queue.sh"),
                    "j1",
                    "--",
                    "bash",
                    "-c",
                    f"echo s1 >> {log_file}; sleep 0.2; echo e1 >> {log_file}",
                ],
                env=env,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.05)
            p2 = subprocess.Popen(
                [
                    "bash",
                    str(SCRIPTS / "queue.sh"),
                    "j2",
                    "--",
                    "bash",
                    "-c",
                    f"echo s2 >> {log_file}; echo e2 >> {log_file}",
                ],
                env=env,
                stderr=subprocess.DEVNULL,
            )
            self.assertEqual(p1.wait(), 0)
            self.assertEqual(p2.wait(), 0)
            self.assertEqual(log_file.read_text(encoding="utf-8").splitlines(), ["s1", "e1", "s2", "e2"])


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def _make_mock_handler(ttft_delay: float, tok_delay: float):
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

            time.sleep(ttft_delay)
            m_code = re.search(r"KEY-[A-Z0-9]{8}", prompt)
            if m_code:
                tokens = [("reasoning_content", "Thinking... "), ("content", m_code.group(0))]
            else:
                tokens = [("content", f"t{i} ") for i in range(max_toks)]

            for field, tok in tokens:
                time.sleep(tok_delay)
                chunk = {"choices": [{"delta": {field: tok}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

            usage = {"choices": [], "usage": {"prompt_tokens": 256, "completion_tokens": len(tokens)}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class TestBenchHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.srv_ctl = _ThreadedTCPServer(("127.0.0.1", 0), _make_mock_handler(0.03, 0.010))
        cls.srv_cand = _ThreadedTCPServer(("127.0.0.1", 0), _make_mock_handler(0.008, 0.003))
        cls.port_ctl = cls.srv_ctl.server_address[1]
        cls.port_cand = cls.srv_cand.server_address[1]
        threading.Thread(target=cls.srv_ctl.serve_forever, daemon=True).start()
        threading.Thread(target=cls.srv_cand.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.srv_ctl.shutdown()
        cls.srv_cand.shutdown()
        cls.srv_ctl.server_close()
        cls.srv_cand.server_close()

    def test_end_to_end_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = os.path.join(td, "ledger.jsonl")
            pid = str(os.getpid())

            for label, port in [("ctl-pre", self.port_ctl), ("cand", self.port_cand), ("ctl-post", self.port_ctl)]:
                rec = json.loads(
                    subprocess.check_output(
                        [
                            "python3",
                            str(SCRIPTS / "bench.py"),
                            "baseline",
                            "--base-url",
                            f"http://127.0.0.1:{port}/v1",
                            "--model",
                            "mock",
                            "--label",
                            label,
                            "--runs",
                            "2",
                            "--warmup",
                            "1",
                            "--prompt-tokens",
                            "64",
                            "--max-tokens",
                            "8",
                            "--mem-pid",
                            pid,
                            "--ledger",
                            ledger,
                        ],
                        text=True,
                    )
                )
                self.assertGreater(rec["summary"]["prefill_tps_median"], 0)
                self.assertIn("mem_delta", rec)

            cmp_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3",
                        str(SCRIPTS / "bench.py"),
                        "compare",
                        "--ledger",
                        ledger,
                        "--control-pre-label",
                        "ctl-pre",
                        "--candidate-label",
                        "cand",
                        "--control-post-label",
                        "ctl-post",
                        "--metric",
                        "decode_tps",
                    ],
                    text=True,
                )
            )
            self.assertEqual(cmp_rec["verdict"], "KEEP")

            ab_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3",
                        str(SCRIPTS / "bench.py"),
                        "ab",
                        "--control-url",
                        f"http://127.0.0.1:{self.port_ctl}/v1",
                        "--candidate-url",
                        f"http://127.0.0.1:{self.port_cand}/v1",
                        "--model",
                        "mock",
                        "--label",
                        "ab-ttft",
                        "--metric",
                        "ttft_s",
                        "--cycles",
                        "2",
                        "--max-tokens",
                        "8",
                        "--ledger",
                        ledger,
                    ],
                    text=True,
                )
            )
            self.assertEqual(ab_rec["verdict"], "KEEP")

            soak_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3",
                        str(SCRIPTS / "bench.py"),
                        "soak",
                        "--base-url",
                        f"http://127.0.0.1:{self.port_cand}/v1",
                        "--model",
                        "mock",
                        "--label",
                        "soak",
                        "--duration-s",
                        "0.15",
                        "--min-runs",
                        "6",
                        "--max-tokens",
                        "6",
                        "--ledger",
                        ledger,
                    ],
                    text=True,
                )
            )
            self.assertTrue(soak_rec["passed"])

            needle_rec = json.loads(
                subprocess.check_output(
                    [
                        "python3",
                        str(SCRIPTS / "bench.py"),
                        "needle",
                        "--base-url",
                        f"http://127.0.0.1:{self.port_cand}/v1",
                        "--model",
                        "mock",
                        "--label",
                        "needle",
                        "--target-tokens",
                        "512",
                        "--depths",
                        "0.1",
                        "0.5",
                        "0.9",
                        "--ledger",
                        ledger,
                    ],
                    text=True,
                )
            )
            self.assertTrue(needle_rec["all_passed"])


if __name__ == "__main__":
    unittest.main()
