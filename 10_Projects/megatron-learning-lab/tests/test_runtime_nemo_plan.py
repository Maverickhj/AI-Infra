"""Stdlib-only frozen NeMo profiles; no runtime or model is imported."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.runtime.nemo_config import read_bound_config, response_rows, validate_bound_config
from experiments.runtime.plan import check_grant, digest, dry_run, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def example(algorithm):
    plan = json.loads((ROOT/f"profiles/runtime-{algorithm}-plan.example.json").read_text())
    config = json.loads((ROOT/f"profiles/runtime-{algorithm}-config.example.json").read_text())
    return plan, config


def assign(config, path, value):
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


class NeMoBoundPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bound(self, algorithm):
        plan, config = example(algorithm)
        plan["execution"]["output_path"] = str(self.root/plan["run_id"])
        config["logger"]["log_dir"] = plan["execution"]["output_path"] + "/logs"
        config["checkpointing"]["checkpoint_dir"] = plan["execution"]["output_path"] + "/checkpoints"
        data = self.root/"input.jsonl"
        data.write_bytes((ROOT/"profiles/runtime-rl-data.example.jsonl").read_bytes())
        plan["data"]["path"] = str(data)
        config["data"]["train"]["data_path"] = str(data)
        path = self.root/(algorithm+".json")
        path.write_text(json.dumps(config))
        plan["training"]["nemo_config"] = str(path)
        plan["training"]["nemo_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return plan, config

    def grant(self, plan):
        return dict(grants={"rl_training": dict(authorized=True,
            authorization_ref="synthetic unit test only", plan_sha256=digest(plan))},
            execution=dict(location="local", devices=["0"], max_gpu_count=1,
                max_wall_seconds=600, max_steps=2, model_snapshot=plan["model"]["snapshot"],
                tokenizer_snapshot=plan["tokenizer"]["snapshot"], data_path=plan["data"]["path"],
                output_path=plan["execution"]["output_path"], allow_model_download=False,
                allow_remote_jobs=False))

    def test_both_profiles_bind_actual_frozen_bytes_and_finite_iteration_counts(self):
        for algorithm in ("grpo", "ppo"):
            example_plan, _ = example(algorithm)
            self.assertEqual(example_plan["training"]["nemo_config_sha256"],
                             hashlib.sha256((ROOT/f"profiles/runtime-{algorithm}-config.example.json").read_bytes()).hexdigest())
            self.assertEqual(example_plan["data"]["sha256"],
                             hashlib.sha256((ROOT/"profiles/runtime-rl-data.example.jsonl").read_bytes()).hexdigest())
            plan, config = self.bound(algorithm)
            validate_plan(plan)
            actual, record, mapping = read_bound_config(plan)
            self.assertEqual(actual, config)
            self.assertEqual(record["sha256"], plan["training"]["nemo_config_sha256"])
            self.assertEqual(mapping["iterations"], 2)
            self.assertEqual(mapping["actor_updates"], 2)
            self.assertEqual(mapping["critic_updates"], 2 if algorithm == "ppo" else 0)
            self.assertEqual(mapping["max_actor_records"], 4)
            self.assertEqual(mapping["runtime_compatibility"], "not_checked")
        self.assertFalse(any(name.split(".")[0] in ("torch", "ray", "nemo_rl", "transformers")
                             for name in sys.modules))

    def test_resource_or_loss_branch_changes_fail_before_allocation(self):
        plan, original = self.bound("grpo")
        changes = [
            ("cluster.num_nodes", 2), ("cluster.gpus_per_node", 2),
            ("cluster.gpus_per_node", True), ("grpo.max_num_steps", 3),
            ("grpo.num_generations_per_prompt", 4), ("grpo.async_grpo.enabled", True),
            ("grpo.val_at_end", True), ("grpo.use_dynamic_sampling", True),
            ("data_plane.enabled", True), ("policy.dtensor_cfg.enabled", True),
            ("policy.sequence_packing.enabled", True), ("policy.megatron_cfg.context_parallel_size", 2),
            ("policy.model_name", "/different/checkpoint"),
            ("policy.tokenizer.name", "/different/tokenizer"),
            ("policy.tokenizer.chat_template_kwargs", {"enable_thinking": 0}),
            ("policy.generation.colocated.enabled", False),
            ("policy.generation.vllm_cfg.async_engine", True),
            ("policy.generation.vllm_cfg.tensor_parallel_size", 2),
            ("policy.generation.max_new_tokens", 128),
            ("policy.generation.temperature", .7), ("policy.generation.top_p", .9),
            ("policy.generation.top_k", 5),
            ("policy.quant_cfg", {"enabled": True}),
            ("policy.generation.vllm_kwargs", {"model": "different/model"}),
            ("data.train.data_path", "/different/data.jsonl"),
            ("data.train.dataset_name", "OpenMathInstruct-2"),
            ("data.default", {"prompt_file": "/different/prompt.txt"}),
            ("logger.log_dir", "/different/logs"), ("logger.wandb_enabled", True),
            ("checkpointing.checkpoint_dir", "/different/checkpoints"),
            ("loss_fn.ratio_clip_max", .3), ("loss_fn.ratio_clip_c", 10),
            ("loss_fn.use_importance_sampling_correction", True),
            ("loss_fn.use_on_policy_kl_approximation", True),
        ]
        for path, value in changes:
            config = copy.deepcopy(original)
            assign(config, path, value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_bound_config(config, plan)

    def test_config_changed_after_authorization_is_not_read_as_granted(self):
        plan, config = self.bound("grpo")
        grant = self.grant(plan)
        check_grant(plan, grant)
        path = Path(plan["training"]["nemo_config"])
        path.write_text(path.read_text()+" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_bound_config(plan)
        # Even a whitespace-only change needs a new plan-bound grant.
        plan["training"]["nemo_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "different plan"):
            check_grant(plan, grant)

    def test_ppo_critic_requires_separate_permission_and_same_bound_snapshot(self):
        plan, config = self.bound("ppo")
        grant = self.grant(plan)
        with self.assertRaisesRegex(ValueError, "critic_snapshot"):
            check_grant(plan, grant)
        grant["execution"]["critic_snapshot"] = plan["critic"]["snapshot"]
        with self.assertRaisesRegex(ValueError, "critic training"):
            check_grant(plan, grant)
        grant["execution"]["allow_critic_training"] = True
        self.assertEqual(check_grant(plan, grant)["status"], "authorized_manifest")
        config["value"]["model_name"] = "/unapproved/critic"
        with self.assertRaisesRegex(ValueError, "value.model_name"):
            validate_bound_config(config, plan)
        plan, config = self.bound("ppo")
        config["value"]["tokenizer"]["name"] = "remote/other-tokenizer"
        with self.assertRaisesRegex(ValueError, "value.tokenizer.name"):
            validate_bound_config(config, plan)
        plan, config = self.bound("ppo")
        config["ppo"]["ppo_epochs"] = 4
        with self.assertRaisesRegex(ValueError, "ppo_epochs"):
            validate_bound_config(config, plan)

    def test_dry_run_records_config_and_cli_inspection_without_runtime_imports(self):
        plan, _ = self.bound("grpo")
        p = self.root/"plan.json"; p.write_text(json.dumps(plan))
        report = dry_run(p)
        self.assertEqual(report["execution"], "not_run")
        self.assertEqual(report["inspection"]["runtime_config"]["algorithm"], "grpo")
        self.assertIn("inspect-nemo-config", report["configuration_command"])
        child = subprocess.run([sys.executable, "-S", "tools/runtime_cli.py", "inspect-nemo-config",
                                "--plan", str(p)], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(child.stdout)["status"], "configuration_bound")
        self.assertFalse(Path(plan["execution"]["output_path"]).exists())

    def test_jsonl_shape_and_dataset_length_are_independently_checked(self):
        plan, _ = self.bound("grpo")
        path = Path(plan["data"]["path"])
        self.assertEqual(len(response_rows(path)), 2)
        for text in ('', '{"input":"x","output":"y","messages":[]}\n',
                     '{"input":"x","input":"y","output":"z"}\n',
                     '{"input":"","output":"z"}\n'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                response_rows(path)
        path.write_text('{"input":"x","output":"z"}\n')
        plan["data"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        p = self.root/"plan.json"; p.write_text(json.dumps(plan))
        self.assertTrue(any("cannot supply" in x for x in dry_run(p)["inspection"]["issues"]))

    def test_unfrozen_overrides_capture_shortfall_and_extra_critic_are_refused(self):
        for value in (None, [], "not a plan"):
            with self.assertRaisesRegex(ValueError, "must be an object"):
                validate_plan(value)
        plan, config = self.bound("grpo")
        plan["training"]["nemo_overrides"] = ["cluster.num_nodes=9"]
        with self.assertRaisesRegex(ValueError, "freeze"):
            validate_plan(plan)
        plan["training"]["nemo_overrides"] = []
        plan["capture"]["max_records"] = 3
        with self.assertRaisesRegex(ValueError, "capture bound"):
            validate_bound_config(config, plan)
        plan["critic"] = copy.deepcopy(plan["model"])
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            validate_plan(plan)
