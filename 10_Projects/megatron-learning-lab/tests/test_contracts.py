"""离线模型元数据与工具行为测试；不运行任何大模型框架。"""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.build_case_data import ROOT, build, derive_case
from tools.verify_handoff import validate_run_manifest, verify
from tools.audit_sources import audit_entry, symbols_in


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {m["id"]: m for m in json.loads((ROOT / "content/models.json").read_text())["models"]}
        cls.cases = {c["id"]: c for c in build()["cases"]}

    def test_qwen3_explicit_head_dimension(self):
        c = self.cases["qwen3-06b"]["attention"]
        self.assertEqual(c["q_projection_width"], 2048)
        self.assertEqual(c["fused_qkv_width"], 4096)
        self.assertEqual(c["output_projection_weight_shape"], [1024, 2048])

    def test_qwen3_complete_layers_and_swiglu(self):
        c = self.cases["qwen3-06b"]
        self.assertEqual(len(c["layers"]), 28)
        self.assertEqual(c["layers"][0]["branch_fused_gate_up_width"], 6144)
        self.assertEqual(c["pipeline"][-1]["id"], "update")

    def test_qwen25_attention_difference(self):
        a, b = (self.cases[k]["attention"] for k in ("qwen25-05b", "qwen3-06b"))
        self.assertTrue(a["qkv_bias"])
        self.assertFalse(b["qkv_bias"])
        self.assertFalse(a["qk_norm"])
        self.assertTrue(b["qk_norm"])

    def test_qwen3_moe_no_shared_expert(self):
        c = self.cases["qwen3-30ba3b"]
        self.assertEqual(len(c["layers"]), 48)
        self.assertTrue(all(l["feed_forward"] == "moe" for l in c["layers"]))
        self.assertEqual(c["layers"][0]["active_routed_experts"], 8)
        self.assertEqual(c["layers"][0]["shared_experts"], 0)

    def test_deepseek_v3_dense_then_moe(self):
        layers = self.cases["deepseek-v3"]["layers"]
        self.assertEqual(len(layers), 61)
        self.assertEqual(sum(l["feed_forward"] == "dense_swiglu" for l in layers), 3)
        self.assertEqual(layers[3]["shared_experts"], 1)
        self.assertEqual(layers[3]["routed_experts"], 256)

    def test_mtp_is_explicit_optional_branch(self):
        mtp = self.cases["deepseek-v3"]["optional_mtp"]
        self.assertEqual(mtp["configured_layers"], 1)
        self.assertFalse(mtp["included_in_base_walkthrough"])

    def test_v2_lite_direct_query_path(self):
        a = self.cases["deepseek-v2-lite"]["attention"]
        self.assertIsNone(a["q_lora_rank"])
        self.assertEqual(a["q_path"], "direct_projection")
        self.assertEqual(a["kv_down_combined_width"], 576)

    def test_distilled_qwen_is_not_mla(self):
        distilled = next(c for c in self.cases.values() if "Distill-Qwen" in c["hf_id"])
        self.assertEqual(distilled["attention"]["type"], "gqa")
        self.assertTrue(all(l["feed_forward"] == "dense_swiglu" for l in distilled["layers"]))

    def test_incomplete_revision_is_explicit(self):
        self.assertIsNotNone(self.models["qwen3-06b"]["revision"])
        self.assertIsNone(self.models["deepseek-v3"]["revision"])

    def test_invalid_gqa_rejected(self):
        model = copy.deepcopy(self.models["qwen3-06b"])
        model["kv_heads"] = 3
        with self.assertRaises(ValueError):
            derive_case(model)

    def test_invalid_expert_topk_rejected(self):
        model = copy.deepcopy(self.models["qwen3-30ba3b"])
        model["experts_per_token"] = 129
        with self.assertRaises(ValueError):
            derive_case(model)

    def test_no_fake_measurements_in_derived(self):
        case = copy.deepcopy(self.cases["qwen3-06b"])
        case["gpu_elapsed_ms"] = 5
        with self.assertRaises(ValueError):
            validate_run_manifest(case)

    def test_observed_missing_metadata_fails(self):
        with self.assertRaises(ValueError):
            validate_run_manifest({"provenance": "observed_rl", "execution_status": "completed"})

    def test_handoff_integrity(self):
        self.assertEqual(verify()["status"], "passed")


class SourceAuditToolTests(unittest.TestCase):
    def test_ast_qualified_symbols_and_decorators(self):
        result = symbols_in("class A:\n    @staticmethod\n    def f():\n        return 1\n")
        self.assertEqual(result["A.f"], [{"start_line": 2, "end_line": 4}])

    def test_source_audit_is_locked_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
            git("init")
            git("config", "user.name", "Local Test")
            git("config", "user.email", "test@example.invalid")
            path = root / "sample.py"
            path.write_text("class A:\n    def f(self):\n        return 1\n")
            git("add", "sample.py")
            git("commit", "-m", "fixture")
            commit = git("rev-parse", "HEAD")
            blob = git("rev-parse", "HEAD:sample.py")
            path.write_text("# uncommitted content must remain untouched\n")
            before = git("status", "--porcelain")
            entry = {"id": "TEST", "path": "sample.py", "git_blob_sha": blob, "symbols": ["A.f"]}
            result = audit_entry(root, commit, entry)
            self.assertEqual(result["status"], "locked_source_match")
            self.assertEqual(git("status", "--porcelain"), before)
            self.assertIn("uncommitted", path.read_text())
            bad = dict(entry, git_blob_sha="0" * 40)
            with self.assertRaises(ValueError):
                audit_entry(root, commit, bad)
            bad = dict(entry, symbols=["A.missing"])
            with self.assertRaises(ValueError):
                audit_entry(root, commit, bad)


if __name__ == "__main__":
    unittest.main()
