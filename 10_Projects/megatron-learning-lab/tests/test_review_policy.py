"""Review 约束的离线回归测试，不证明任何实际训练环境兼容。"""
import copy
import json
import unittest

from tools.verify_handoff import ROOT, validate_run_manifest, validate_compatibility_report


def report_fixture():
    # These strings identify unit-test fixtures, not executed GPU checks.
    required = ["callable_interfaces", "data_semantics", "behavior_smoke", "runtime_source_mapping"]
    return {"status": "supported", "required_capabilities": required,
            "checks": {key: {"status": "supported", "evidence": "unit_test_fixture_only"} for key in required}}


def run_fixture():
    return {"provenance": "observed_bridge", "execution_status": "completed",
            "source_lane": "local-existing-environment", "runtime_profile": "bridge_sft",
            "runtime_versions": {"bridge": "test-different-from-reference"},
            "module_import_paths": {"bridge": "/test/bridge/__init__.py"},
            "config_sha256": "a" * 64, "input_sha256": "b" * 64,
            "model_revision": "c" * 40, "tokenizer_revision": "d" * 40,
            "weights_origin": "unit_test_fixture_only", "architecture_origin": "unit_test_fixture_only",
            "backend": "test", "dtype": "test", "parallelism": {"tp": 1},
            "started_at": "test", "ended_at": "test", "command": "not_executed_fixture",
            "result_files": ["unit_test_fixture_only"], "compatibility": report_fixture()}


class ReviewPolicyTests(unittest.TestCase):
    def test_reference_lock_is_not_install_lock(self):
        lock = json.loads((ROOT / "source.lock.json").read_text())
        self.assertFalse(lock["runtime_policy"]["require_exact_reference_versions"])
        contract = json.loads((ROOT / "profiles/compatibility-contract.json").read_text())
        self.assertFalse(contract["reference_lock_is_install_requirement"])
        self.assertFalse(contract["exact_upstream_versions_required"])

    def test_custom_source_lane_and_different_version_accepted_as_metadata(self):
        validate_run_manifest(run_fixture())

    def test_same_version_but_failed_semantic_report_rejected(self):
        run = run_fixture()
        run["runtime_versions"] = {"bridge": "0.6.2"}
        run["compatibility"]["checks"]["data_semantics"]["status"] = "unsupported"
        with self.assertRaises(ValueError):
            validate_run_manifest(run)

    def test_not_checked_required_capability_rejected(self):
        report = report_fixture()
        report["checks"]["behavior_smoke"]["status"] = "not_checked"
        with self.assertRaises(ValueError):
            validate_compatibility_report(report)

    def test_unused_optional_capability_does_not_block(self):
        report = report_fixture()
        report["checks"]["fp8"] = {"status": "not_applicable"}
        validate_compatibility_report(report)

    def test_missing_evidence_rejected(self):
        report = report_fixture()
        del report["checks"]["behavior_smoke"]["evidence"]
        with self.assertRaises(ValueError):
            validate_compatibility_report(report)

    def test_import_only_is_not_compatibility(self):
        report = report_fixture()
        report["required_capabilities"] = ["callable_interfaces"]
        with self.assertRaises(ValueError):
            validate_compatibility_report(report)

    def test_rl_profile_not_accepted_as_sft(self):
        run = run_fixture()
        run["runtime_profile"] = "rl_grpo"
        with self.assertRaises(ValueError):
            validate_run_manifest(run)

    def test_audience_and_depth_in_entry_instructions(self):
        for name in ("PLAN.md", "CURRICULUM.md", "AGENTS.md", "CODEX_START.md"):
            text = (ROOT / name).read_text()
            self.assertIn("有一定基础的初学者", text, name)
            self.assertIn("可跳过", text, name)
            self.assertIn("精讲", text, name)

    def test_first_case_has_compact_intro_before_core(self):
        text = (ROOT / "content/cases/01_qwen3_complete_sft.md").read_text()
        intro = text.split("## 1. 基础速览", 1)[1].split("## 2. 核心精讲", 1)[0]
        self.assertLess(len(intro), 500)
        self.assertIn("数学符号", text)
        self.assertIn("只做一次 next-token 移位", text)


if __name__ == "__main__":
    unittest.main()
