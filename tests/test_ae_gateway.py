import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ae" / "gateway.py"
SPEC = importlib.util.spec_from_file_location("ae_gateway", MODULE_PATH)
assert SPEC and SPEC.loader
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


class GatewayTests(unittest.TestCase):
    def test_allowed_commands(self):
        self.assertEqual(gateway.parse_command("run offline"), ("run", "offline"))
        self.assertEqual(gateway.parse_command("help"), ("help", None))
        run_id = "a" * 32
        self.assertEqual(gateway.parse_command(f"status {run_id}"), ("status", run_id))
        self.assertEqual(gateway.parse_command(f"fetch {run_id}"), ("fetch", run_id))

    def test_rejects_arbitrary_shell_and_paths(self):
        for command in ("", "sh", "run hardware", "run offline; id", "status ../../etc/passwd", "fetch /etc/passwd"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                gateway.parse_command(command)

    def test_report_path_is_confined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(gateway.report_path(root, "f" * 32), root / ("f" * 32) / "result.json")
            with self.assertRaises(ValueError):
                gateway.report_path(root, "../escape")

    def test_config_requires_absolute_paths_and_frozen_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "repo": "relative/repo", "python": "/usr/bin/python3",
                "results": "/tmp/results", "expected_commit": "a" * 40,
            }))
            with self.assertRaises(ValueError):
                gateway.load_config(path)

    def test_run_records_frozen_revision_and_rejects_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            results = root / "results"
            results.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "README.md").write_text("fixture\n")
            subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
            subprocess.run([
                "git", "-C", str(repo), "-c", "user.name=AE test",
                "-c", "user.email=ae@example.invalid", "commit", "-qm", "fixture",
            ], check=True)
            commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            config = {"repo": repo, "python": Path("/usr/bin/true"),
                      "results": results, "expected_commit": commit}
            report = gateway.run_offline(config)
            self.assertEqual(report["commit"], commit)
            self.assertTrue(report["passed"])
            self.assertTrue(gateway.report_path(results, report["run_id"]).is_file())
            (repo / "UNTRACKED").write_text("dirty\n")
            with self.assertRaises(ValueError):
                gateway.run_offline(config)


if __name__ == "__main__":
    unittest.main()
