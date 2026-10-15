import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cloud
from workflow_plan import budget


class PortableProtocolTests(unittest.TestCase):
    def gate(self):
        return dict(passed=True, package=cloud.fingerprint(), baseline_choice=dict(lr=1))

    def test_thirteen_screen_cells(self):
        jobs = cloud.jobs_for("screen")
        self.assertEqual(len(jobs), 13)
        self.assertEqual(len({j["id"] for j in jobs}), 13)
        self.assertTrue(all(j["seed"] == 2026 and j["users_per_region"] == 100 for j in jobs))
        self.assertEqual(sum(j["lr"] == 1 for j in jobs), 1)

    def test_formal_requires_gate(self):
        with self.assertRaises(ValueError):
            cloud.jobs_for("formal_training")

    def test_formal_selection_only_affects_baseline(self):
        jobs = cloud.jobs_for("formal_training", self.gate())
        self.assertEqual(len(jobs), 12)
        for job in jobs:
            self.assertEqual(job["lr"], int(job["method"] == "MARATD3-Adapted"))

    def test_gate_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gate.json"
            gate = self.gate()
            cloud.checks.save(path, gate)
            cloud.verify_gate(path)
            gate["package"]["config"] = "changed"
            cloud.checks.save(path, gate)
            with self.assertRaises(ValueError):
                cloud.verify_gate(path)

    def test_failed_gate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gate.json"
            gate = self.gate()
            gate["passed"] = False
            cloud.checks.save(path, gate)
            with self.assertRaises(ValueError):
                cloud.verify_gate(path)

    def test_prepare_is_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            stage, config, jobs = cloud.prepare(out, "screen")
            self.assertEqual(config["training"]["updates"], 30)
            self.assertEqual(config["training"]["rollout_steps"], 128)
            cloud.prepare(out, "screen")
            config["training"]["updates"] = 29
            cloud.checks.save(stage / "config.json", config)
            with self.assertRaises(ValueError):
                cloud.prepare(out, "screen")

    def test_formal_budget_and_reserved_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            _, config, _ = cloud.prepare(Path(temp), "full_evaluation", self.gate())
            self.assertEqual(config["training"]["updates"], 2000)
            self.assertEqual(config["training"]["formal_measure_s"], 300)
            self.assertEqual(config["training"]["test_seeds"], [91005, 91003, 91008, 91020, 91001])

    def test_noamp_does_not_change_physics(self):
        cfg = cloud.configuration()
        self.assertEqual(cfg["environment"]["delta_req"], 1e-6)
        self.assertEqual(cfg["training"]["seed"], 2026)
        self.assertEqual(cfg["experiments"]["learning_rates"][0], [1e-4, 3e-4])

    def test_dispatch_budget_includes_all_matrix_timeouts(self):
        for minutes in (300, 600, 2000):
            for count in (12, 13):
                result = budget("screen", count, minutes)
                self.assertLessEqual(result["maximum_minutes"], minutes)
                self.assertLess(result["seconds"], result["timeout"]*60)

    def test_paid_unbounded_allowance_rejected(self):
        with self.assertRaises(ValueError):
            budget("formal_training", 12, 100000)

    def test_memory_query_works(self):
        self.assertGreater(cloud.available_ram_gib(), 0)

    def test_native_train_pause_resume_and_audit(self):
        config = copy.deepcopy(cloud.configuration())
        config["environment"].update(users_per_region=3, arrival_per_user_s=8)
        config["training"].update(hidden=16, epochs=1, minibatch=32, torch_threads=1,
            rollout_steps=8, episode_steps=80, pilot_warmup_s=.02, pilot_measure_s=.2)
        candidate = cloud.checks.job("TMAPO", 1., 0, 2026)
        candidate["users_per_region"], candidate["id"] = 3, "execution_smoke"
        with tempfile.TemporaryDirectory() as temp, patch.object(cloud, "configuration", return_value=config), \
             patch.object(cloud, "jobs_for", return_value=[candidate]), \
             patch.object(cloud, "available_ram_gib", return_value=8):
            out = Path(temp)
            self.assertFalse(cloud.run_cell(out, "screen", candidate["id"], "cpu", .0001))
            target = out / "screen" / candidate["id"]
            self.assertEqual(cloud.checks.read(target / "paused.json")["update"], 1)
            self.assertFalse((target / "RUNNING.lock").exists())
            self.assertTrue(cloud.run_cell(out, "screen", candidate["id"], "cpu", 600))
            self.assertEqual(cloud.checks.read(target / "complete.json")["updates"], 30)
            self.assertEqual([int(r["update"]) for r in cloud.checks.rows(target / "train_raw.csv")], list(range(1, 31)))
            from audit_comparison import audit
            audit(out / "screen")


if __name__ == "__main__":
    unittest.main()
