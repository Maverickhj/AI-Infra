"""No framework/model imports: hand resource bounds and dry-run refusal cases."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.runtime.plan import check_grant, digest, dry_run, inspect_plan, validate_plan

ROOT = Path(__file__).resolve().parents[1]


class RuntimePlanTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.plan = json.loads((ROOT/"profiles/runtime-plan.example.json").read_text())
        self.plan["profile"] = "hf_reference"
        self.plan["model"]["snapshot"] = str(self.root/"model")
        self.plan["tokenizer"]["snapshot"] = str(self.root/"model")
        self.plan["data"]["path"] = str(self.root/"data.json")
        self.plan["execution"]["output_path"] = str(self.root/self.plan["run_id"])
        self.plan["sources"] = {"bridge": None, "nemo_rl": None}
        model = self.root/"model"
        model.mkdir()
        (model/"config.json").write_text('{"model_type":"synthetic"}')
        (model/"tokenizer_config.json").write_text("{}")
        (model/"tokenizer.json").write_text("synthetic metadata only")
        (model/"model.safetensors").write_bytes(b"synthetic file existence only, never loaded")
        raw=b'{"messages": []}'
        (self.root/"data.json").write_bytes(raw)
        self.plan["data"]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.path=self.root/"plan.json"
        self.path.write_text(json.dumps(self.plan))

    def grant(self):
        p=self.plan
        return dict(grants={"bridge_sft":dict(authorized=True, authorization_ref="synthetic unit-test consent",
                                           plan_sha256=digest(p))},
                    execution=dict(location="local",devices=["0"],max_gpu_count=1,max_wall_seconds=600,
                                   max_steps=2,model_snapshot=p["model"]["snapshot"],
                                   tokenizer_snapshot=p["tokenizer"]["snapshot"],data_path=p["data"]["path"],
                                   output_path=p["execution"]["output_path"],
                                   allow_model_download=False,allow_remote_jobs=False))

    def test_grant_binds_exact_plan_and_bounds_without_mutation(self):
        before=copy.deepcopy(self.plan)
        result=check_grant(self.plan,self.grant())
        self.assertEqual(result["status"],"authorized_manifest")
        self.assertEqual(before,self.plan)
        grant=self.grant()
        changed=copy.deepcopy(self.plan);changed["training"]["learning_rate"]=.1
        with self.assertRaisesRegex(ValueError,"different plan"):check_grant(changed,grant)

    def test_absent_truthy_and_blank_authorization_cannot_enable_execution(self):
        for key,value in (("authorized",False),("authorized","true"),("authorized",1),("authorization_ref"," ")):
            grant=self.grant();grant["grants"]["bridge_sft"][key]=value
            with self.assertRaises(ValueError):check_grant(self.plan,grant)
        with self.assertRaises(ValueError):check_grant(self.plan,{})

    def test_resource_limit_and_path_changes_are_rejected(self):
        for key,value in (("devices",["1"]),("max_gpu_count",0),("max_wall_seconds",1),
                          ("max_steps",1),("data_path",str(self.root/"other")),
                          ("output_path",str(self.root/"other")),
                          ("allow_model_download",True),("allow_remote_jobs",True)):
            grant=self.grant();grant["execution"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check_grant(self.plan,grant)

    def test_unknown_fields_versions_modes_and_capture_bounds_fail(self):
        mutations=[
            lambda p:p.update(command=["sh","-c","touch injected"]),
            lambda p:p.update(schema_version=True),
            lambda p:p["data"].update(mapping="guess"),
            lambda p:p["training"].update(steps=True),
            lambda p:p["training"].update(sequence_length=512),
            lambda p:p["execution"].update(devices=["0","1"]),
            lambda p:p["capture"].update(max_records=65),
            lambda p:p["model"].update(revision="main"),
            lambda p:p["execution"].update(output_path=str(self.root/"../escape")),
        ]
        for mutate in mutations:
            p=copy.deepcopy(self.plan);mutate(p)
            with self.assertRaises(ValueError):validate_plan(p)

    def test_dry_run_reads_metadata_only_and_never_loads_fake_weights(self):
        report=dry_run(self.path)
        self.assertEqual(report["inspection"]["status"],"configuration_ready")
        self.assertEqual(report["inspection"]["behavior"],"not_run")
        self.assertEqual(report["resources"]["status"],"not_authorized")
        self.assertEqual(report["execution"],"not_run")
        self.assertFalse(Path(self.plan["execution"]["output_path"]).exists())
        self.assertTrue(all(not f["path"].endswith(".safetensors") for f in report["inspection"]["files"]))
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("ray",sys.modules)
        self.assertNotIn("transformers",sys.modules)

    def test_hash_missing_shard_and_unknown_cli_are_reported_without_execution(self):
        (self.root/"data.json").write_text("changed")
        model=self.root/"model"
        (model/"model.safetensors").unlink()
        (model/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"weight":"missing.safetensors"}}))
        got=inspect_plan(self.plan)
        self.assertEqual(got["status"],"not_ready")
        self.assertTrue(any("SHA256" in x for x in got["issues"]))
        self.assertTrue(any("missing weight shard" in x for x in got["issues"]))
        self.plan["profile"]="rl_grpo"
        self.plan["sources"]["nemo_rl"]=str(self.root/"nemo")
        self.plan["training"]["nemo_config"]=str(self.root/"config.yaml")
        self.plan["training"]["nemo_config_sha256"]="0"*64
        self.plan["training"]["global_batch_size"]=2
        self.plan["data"]["mapping"]="nemo_response_jsonl_v1"
        launcher=self.root/"nemo/examples/run_grpo.py"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("raise RuntimeError('this launcher must not execute')")
        self.assertTrue(any("parse_args" in x for x in inspect_plan(self.plan)["issues"]))

    def test_cli_plan_read_is_subprocess_clean_and_bound_to_actual_input(self):
        result=subprocess.run([sys.executable,"-S","tools/runtime_cli.py","dry-run","--plan",str(self.path)],
                              cwd=ROOT,text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout)["plan_sha256"],digest(self.plan))
