"""Pre-import checks and explicitly synthetic launch fixtures; no runtime executes."""
import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.runtime.launch import launch, verify_worker
from experiments.runtime.nemo_metadata import inspect_metadata, manifest
from experiments.runtime.plan import dry_run
from experiments.runtime.source_probe import inspect_nemo_runtime
from tests import test_runtime_nemo_plan as plans


def synthetic_checkout(root):
    """Authored signatures only, not official implementation or a compatibility claim."""
    files = {
        "nemo_rl/models/policy/lm_policy.py": """class Policy:
 def __init__(self,cluster,config,tokenizer,worker_extension_cls_fqn=None): pass
 def train(self,data,loss_fn,timer=None): pass
 def get_logprobs(self,data): pass
""",
        "nemo_rl/models/policy/workers/megatron_policy_worker.py": """class MegatronPolicyWorkerImpl:
 def __init__(self,config,tokenizer,init_optimizer=True,init_reference_model=True): pass
 def train(self,data,loss_fn): pass
""",
        "nemo_rl/models/megatron/setup.py": """def validate_model_paths(config): pass
def handle_model_import(config,hf_model_name,pretrained_path): pass
def setup_model_and_optimizer(policy_cfg,megatron_cfg,load_optimizer): pass
def setup_reference_model_state(config,megatron_cfg,pretrained_path): pass
""",
        "nemo_rl/experience/rollouts.py": """def run_multi_turn_rollout(policy_generation,input_batch,tokenizer,task_to_env,max_seq_len): pass
""",
        "nemo_rl/algorithms/utils.py": "def get_tokenizer(tokenizer_config): pass\n",
        "nemo_rl/distributed/ray_actor_environment_registry.py": "def get_actor_python_env(actor_class_fqn): pass\n",
    }
    for algorithm in ("grpo", "ppo"):
        files[f"examples/run_{algorithm}.py"] = """raise RuntimeError("synthetic source must never execute")
def parse_args():
 parser = argparse.ArgumentParser()
 parser.add_argument("--config", type=str)
 return parser.parse_args()
def main():
 load_config()
 parse_hydra_overrides()
 init_ray()
 setup()
"""
        extra = ",policy_factory=None" if algorithm == "grpo" else ""
        critic = ",value_model,value_loss_fn" if algorithm == "ppo" else ""
        files[f"nemo_rl/algorithms/{algorithm}.py"] = (
            f"def setup(master_config,tokenizer,dataset,val_dataset{extra}): pass\n"
            f"def {algorithm}_train(policy,policy_generation,tokenizer,loss_fn,master_config{critic}): pass\n")
    files["nemo_rl/algorithms/grpo.py"] += (
        "def refit_policy_generation(policy,policy_generation,colocated_inference): pass\n")
    for name, body in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)


def local_profile(sample, algorithm):
    plan, config = sample.bound(algorithm)
    model = sample.root/"model"
    model.mkdir(exist_ok=True)
    (model/"config.json").write_text(json.dumps({"model_type":"qwen3","vocab_size":16}))
    (model/"model.safetensors").write_bytes(b"synthetic bytes; never loaded")
    (model/"tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class":"Qwen2Tokenizer", "chat_template":"synthetic template"}))
    (model/"tokenizer.json").write_text('{"synthetic":true}')
    for key in ("model","tokenizer") + (("critic",) if algorithm=="ppo" else ()):
        plan[key]["snapshot"] = str(model)
    for key in ("policy",) + (("value",) if algorithm=="ppo" else ()):
        config[key]["model_name"] = str(model)
        config[key]["tokenizer"]["name"] = str(model)
    source = sample.root/"nemo"
    synthetic_checkout(source)
    plan["sources"]["nemo_rl"] = str(source)
    frozen = Path(plan["training"]["nemo_config"])
    frozen.write_text(json.dumps(config))
    plan["training"]["nemo_config_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    path = sample.root/"plan.json"
    path.write_text(json.dumps(plan))
    return plan, path


class SyntheticTokenizer:
    is_fast = True

    def __init__(self, root):
        self.name_or_path = str(root)
        self.apply_chat_template = partial(self.apply, enable_thinking=False)

    def apply(self, messages, *, enable_thinking):
        raise AssertionError("metadata must not tokenize")

    def get_chat_template(self):
        return "synthetic template"


class NeMoStartupTests(unittest.TestCase):
    def setUp(self):
        self.sample = plans.NeMoBoundPlanTests()
        self.sample.setUp()
        self.addCleanup(self.sample.doCleanups)

    def test_remote_code_unknown_family_and_template_reject_before_import(self):
        plan, _ = local_profile(self.sample, "ppo")
        root = Path(plan["model"]["snapshot"])
        for filename, original, key, value in (
            ("config.json", {"model_type":"qwen3","vocab_size":16}, "auto_map", {"AutoModel":"x.Code"}),
            ("config.json", {"model_type":"qwen3","vocab_size":16}, "model_type", "uninspected"),
            ("config.json", {"model_type":"qwen3","vocab_size":16}, "vocab_size", True),
            ("tokenizer_config.json", {"tokenizer_class":"Qwen2Tokenizer","chat_template":"t"}, "auto_map", {"AutoTokenizer":["x.Code",None]}),
            ("tokenizer_config.json", {"tokenizer_class":"Qwen2Tokenizer","chat_template":"t"}, "tokenizer_class", "UnknownTokenizer"),
            ("tokenizer_config.json", {"tokenizer_class":"Qwen2Tokenizer","chat_template":"t"}, "chat_template", None),
        ):
            path = root/filename
            previous = path.read_bytes()
            with self.subTest(field=key):
                path.write_text(json.dumps(dict(original, **{key:value})))
                with self.assertRaises(ValueError):
                    inspect_metadata(plan)
                path.write_bytes(previous)
        self.assertFalse(any(x in sys.modules for x in ("torch","ray","nemo_rl","transformers")))

    def test_metadata_hashes_actual_bytes_without_loading_or_tokenizing(self):
        plan, _ = local_profile(self.sample, "ppo")
        model = Path(plan["model"]["snapshot"])
        tokenizer = SyntheticTokenizer(model)
        result, vocab = manifest(plan, tokenizer, sources=[], frozen={"synthetic":True})
        self.assertEqual(vocab,16)
        self.assertEqual(result["tokenizer"]["chat_template_sha256"],
                         hashlib.sha256(b"synthetic template").hexdigest())
        for role, field in (("model","model"),("critic","critic"),("tokenizer","tokenizer")):
            for record in result[field]["loaded_files"]:
                self.assertEqual(record["sha256"],hashlib.sha256((model/record["name"]).read_bytes()).hexdigest())
        self.assertIn("apply",result["runtime_sources"][0]["symbol"])
        tokenizer.is_fast=False
        with self.assertRaisesRegex(ValueError,"not fast"):
            manifest(plan,tokenizer,sources=[],frozen={})
        tokenizer.is_fast=True;tokenizer.name_or_path=str(model/"other")
        with self.assertRaisesRegex(ValueError,"differs"):
            manifest(plan,tokenizer,sources=[],frozen={})
        # The production metadata builder is inspected in memory only. No
        # synthetic checkpoint is loaded or written as an observed trace.
        self.assertFalse(Path(plan["execution"]["output_path"]).exists())

    def test_static_interfaces_and_helper_changes_are_bound_without_execution(self):
        plan, path = local_profile(self.sample,"grpo")
        source = Path(plan["sources"]["nemo_rl"])
        report = dry_run(path)
        self.assertEqual(report["inspection"]["status"],"configuration_ready",report["inspection"]["issues"])
        self.assertEqual(report["execution"],"not_run")
        self.assertGreater(len(report["inspection"]["runtime_sources"]),10)
        before=inspect_nemo_runtime(source,"grpo")
        helper=source/"nemo_rl/unimported_helper.py"
        helper.write_text('raise AssertionError("must not import")\n')
        after=inspect_nemo_runtime(source,"grpo")
        self.assertNotEqual(before["package_snapshot"]["sha256"],after["package_snapshot"]["sha256"])
        policy=source/"nemo_rl/models/policy/lm_policy.py"
        policy.write_text(policy.read_text().replace("worker_extension_cls_fqn","removed_extension"))
        with self.assertRaisesRegex(ValueError,"signature mismatch"):
            inspect_nemo_runtime(source,"grpo")
        self.assertFalse(Path(plan["execution"]["output_path"]).exists())

    def test_both_rl_launch_branches_bind_environment_and_reject_fake_success(self):
        for algorithm in ("grpo","ppo"):
            for completion in (None,"synthetic_nemo_contract","completed_nemo_entry"):
                with self.subTest(algorithm=algorithm,completion=completion):
                    sample=plans.NeMoBoundPlanTests();sample.setUp()
                    self.addCleanup(sample.doCleanups)
                    plan,path=local_profile(sample,algorithm)
                    grant=sample.grant(plan)
                    if algorithm=="ppo":
                        grant["execution"].update(critic_snapshot=plan["critic"]["snapshot"],allow_critic_training=True)
                    resource=sample.root/"resources.json";resource.write_text(json.dumps(grant))
                    output=Path(plan["execution"]["output_path"])
                    def synthetic_supervisor(argv, **kwargs):
                        env=kwargs["env"]
                        for key in ("NEMO_RL_PY_EXECUTABLES_SYSTEM","UV_OFFLINE","HF_HUB_OFFLINE"):
                            self.assertEqual(env[key],"1")
                        self.assertEqual(env["MEGATRON_LAB_OUTPUT_PATH"],str(output))
                        self.assertEqual(env["NRL_MEGATRON_CHECKPOINT_DIR"],str(output/"model-import"))
                        self.assertNotIn("RAY_ADDRESS",env)
                        self.assertNotIn("PYTHONPATH",env)
                        with patch.dict(os.environ,env,clear=True):
                            checked,actual_output=verify_worker(path,resource,output/"receipt.json")
                            self.assertEqual(checked,plan);self.assertEqual(actual_output,output)
                            helper=Path(plan["sources"]["nemo_rl"])/"nemo_rl/new_helper.py"
                            helper.write_text("# changed after receipt\n")
                            with self.assertRaisesRegex(ValueError,"source changed"):
                                verify_worker(path,resource,output/"receipt.json")
                            helper.unlink()
                        if completion:
                            (output/"result.json").write_text(json.dumps({"status":completion}))
                        return dict(status="passed",returncode=0,stop_reason=None)
                    with patch("experiments.runtime.launch.supervised_process",synthetic_supervisor):
                        result=launch(path,resource)
                    self.assertEqual(result["execution"],"passed" if completion=="completed_nemo_entry" else "failed")
                    if completion!="completed_nemo_entry":
                        self.assertEqual(result["result"]["stop_reason"],"missing_or_invalid_result")
        self.assertFalse(any(x in sys.modules for x in ("torch","ray","nemo_rl","transformers")))

    def test_ungranted_rl_does_not_inspect_sources_or_create_output(self):
        plan,path=local_profile(self.sample,"grpo")
        resource=self.sample.root/"resources.json"
        resource.write_text('{"grants":{"rl_training":{"authorized":false}}}')
        with patch("experiments.runtime.launch.inspect_plan",side_effect=AssertionError("unexpected inspect")):
            with self.assertRaisesRegex(ValueError,"not authorized"):
                launch(path,resource)
        self.assertFalse(Path(plan["execution"]["output_path"]).exists())

    def test_supervisor_failure_closes_receipt_before_worker_can_retry(self):
        plan,path=local_profile(self.sample,"grpo")
        resource=self.sample.root/"resources.json"
        resource.write_text(json.dumps(self.sample.grant(plan)))
        with patch("experiments.runtime.launch.supervised_process",side_effect=ValueError("synthetic guardian failure")):
            result=launch(path,resource)
        self.assertEqual(result["execution"],"failed")
        self.assertEqual(result["result"]["stop_reason"],"supervisor_error")
        with self.assertRaisesRegex(ValueError,"unconsumed"):
            verify_worker(path,resource,Path(plan["execution"]["output_path"])/"receipt.json")
