"""G08 trace/CLI contracts. Fixture-only probes are explicitly synthetic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.runtime.contracts import digest_text, read_trace, validate_trace
from experiments.runtime.source_probe import anchor, parser_options, parse_inspected_cli

ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        traces = json.loads((ROOT/"content/fixtures/runtime-reference.json").read_text())["traces"]
        self.sft, self.rl = copy.deepcopy(traces)
    def change_json(self, trace, key, fn):
        value = json.loads(trace[key+"_json"])
        fn(value)
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        trace[key+"_json"] = text
        trace[key+"_sha256"] = digest_text(text)

    def test_real_cpu_reference_fixtures_are_valid_but_not_runtime_attestation(self):
        for trace in (self.sft, self.rl):
            result = read_trace(json.dumps(trace))
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["trust"], "imported_claim")
            self.assertEqual(result["provenance"], "reference")
            self.assertGreater(result["token_count"], 0)

    def test_synthetic_cannot_be_promoted_to_observed(self):
        trace = self.sft
        trace["provenance"] = "observed_bridge"
        trace["manifest"]["backend"] = "megatron-bridge"
        trace["manifest"]["adapter"] = "bridge_bsh_v1"
        trace["manifest"]["evidence_kind"] = "synthetic_contract"
        with self.assertRaises(ValueError):
            validate_trace(trace)
        trace = self.rl
        trace["provenance"] = "observed_rl"
        with self.assertRaises(ValueError):
            validate_trace(trace)

    def test_revision_unknown_mapping_dtype_and_backend_are_rejected(self):
        mutations = [
            lambda t: t["manifest"]["tokenizer"].update(revision=""),
            lambda t: t["manifest"].update(adapter="unreviewed_adapter_v9"),
            lambda t: t["manifest"].update(dtype="guess"),
            lambda t: t["manifest"].update(backend=""),
            lambda t: t["manifest"].update(layout="SB"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                trace = copy.deepcopy(self.sft);mutate(trace)
                with self.assertRaises(ValueError):validate_trace(trace)

    def test_input_and_config_hashes_check_exact_stored_bytes(self):
        for key in ("input", "config"):
            trace = copy.deepcopy(self.sft);trace[key+"_json"] += " "
            with self.assertRaisesRegex(ValueError, "SHA256"):validate_trace(trace)

    def test_masks_double_shift_and_document_leakage_are_rejected(self):
        for change in (
            lambda d: d.pop("loss_mask"),
            lambda d: d.update(loss_mask=[[0]*len(d["input_ids"][0])]),
            lambda d: d["labels"][0].__setitem__(next(i for i,v in enumerate(d["loss_mask"][0]) if v), 26),
            lambda d: d["document_ids"][0].__setitem__(next(i for i,v in enumerate(d["loss_mask"][0]) if v)+1, 1),
        ):
            trace = copy.deepcopy(self.sft);self.change_json(trace,"input",change)
            with self.assertRaises(ValueError):validate_trace(trace)

    def test_loss_sum_count_and_mean_are_independently_recomputed(self):
        for key in ("loss_sum","token_count","loss_mean"):
            trace=copy.deepcopy(self.sft);trace["measurements"][key]+=1
            with self.assertRaisesRegex(ValueError,"mismatch"):validate_trace(trace)

    def test_reference_disabled_and_critic_fields_have_explicit_meaning(self):
        trace=copy.deepcopy(self.rl)
        self.change_json(trace,"config",lambda c:c.update(kl_beta=0))
        with self.assertRaisesRegex(ValueError,"disabled reference"):validate_trace(trace)
        trace=copy.deepcopy(self.rl);trace["measurements"].pop("returns")
        with self.assertRaisesRegex(ValueError,"returns"):validate_trace(trace)
        trace=copy.deepcopy(self.rl)
        self.change_json(trace,"config",lambda c:c.update(algorithm="grpo"))
        with self.assertRaisesRegex(ValueError,"GRPO"):validate_trace(trace)

    def test_ppo_ratio_kl_value_and_reduction_mismatches_are_rejected(self):
        cases = [
            lambda m:m["ratio"][0].__setitem__(8,9.0),
            lambda m:m["pg_token"][0].__setitem__(8,9.0),
            lambda m:m.update(actor_loss=m["actor_loss"]+1),
            lambda m:m.update(kl_loss=m["kl_loss"]+.01),
            lambda m:m.update(value_loss=m["value_loss"]+.01),
        ]
        for mutate in cases:
            trace=copy.deepcopy(self.rl);mutate(trace["measurements"])
            with self.assertRaisesRegex(ValueError,"mismatch"):validate_trace(trace)

    def test_refit_call_acknowledgment_cannot_claim_weight_hash_verification(self):
        self.rl["manifest"]["evidence_kind"]="synthetic_contract"
        self.rl["manifest"]["execution"].update(status="synthetic",synthetic=True)
        valid=dict(status="acknowledged",completed=True,generation_version=2,
                   evidence="official_delegate_return",weight_hash_verified=False,event=9)
        self.rl["measurements"]["refit"]=valid
        validate_trace(self.rl)
        for change in (dict(weight_hash_verified=True),dict(export_hash="a"*64),
                       dict(ack_hash="a"*64),dict(generation_version=1),
                       dict(evidence="guessed"),dict(event=True),dict(status="synchronized")):
            trace=copy.deepcopy(self.rl);trace["measurements"]["refit"].update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,"refit"):
                validate_trace(trace)

    def test_refit_requires_completed_ack_hash_and_matching_policy_version(self):
        valid=dict(status="synchronized",completed=True,generation_version=2,export_hash="a"*64,ack_hash="a"*64)
        self.rl["measurements"]["refit"]=valid.copy()
        self.assertEqual(validate_trace(self.rl)["status"],"valid")
        for change in (dict(completed=False),dict(generation_version=1),dict(ack_hash="b"*64)):
            trace=copy.deepcopy(self.rl);trace["measurements"]["refit"].update(change)
            with self.assertRaisesRegex(ValueError,"refit"):validate_trace(trace)

    def test_untrusted_json_is_bounded_and_never_executed(self):
        for text in ('{"x":1,"x":2}', '{"__proto__":{}}', '{"x":NaN}', "["*40+"0"+"]"*40, " "*1_048_577):
            with self.assertRaises(ValueError):read_trace(text)
        trace=copy.deepcopy(self.sft)
        trace["manifest"]["execution"]["command"]=["python","-c","raise RuntimeError('must not run')"]
        self.assertEqual(read_trace(json.dumps(trace))["status"],"valid")
        for field in ("provenance","task"):
            trace=copy.deepcopy(self.sft);trace[field]={}
            with self.assertRaises(ValueError):validate_trace(trace)

    def test_derived_cannot_present_reference_numbers_as_observations(self):
        trace=copy.deepcopy(self.sft);trace["provenance"]="derived"
        trace["manifest"]["execution"]["status"]="not_run"
        trace["manifest"]["evidence_kind"]="formula"
        with self.assertRaisesRegex(ValueError,"measured numbers"):validate_trace(trace)
        trace["measurements"]=None
        self.assertEqual(validate_trace(trace)["status"],"valid")

    def test_cli_ast_probe_does_not_run_imports_or_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"launcher.py"
            # Deliberately unimportable synthetic source: AST observation must
            # inspect the declaration without executing this exception/import.
            path.write_text("raise RuntimeError('must not execute')\nimport unavailable_framework\n"
                "def parse_args():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--config', type=str, default=None)\n"
                "    return parser.parse_known_args()\n")
            probe=parser_options(path)
            got=parse_inspected_cli(probe,["--config","offline.json","train.steps=2"])
            self.assertEqual(got["arguments"]["config"],"offline.json")
            self.assertEqual(got["overrides"],["train.steps=2"])
            with self.assertRaisesRegex(ValueError,"unknown CLI"):parse_inspected_cli(probe,["--guessed-flag","2"])
            self.assertEqual(anchor(path,"parse_args")["evidence"],"static_runtime_source_read")
            with self.assertRaisesRegex(ValueError,"missing"):anchor(path,"missing")

    def test_matching_version_cannot_override_bad_semantics_and_version_change_alone_is_not_rejection(self):
        trace=copy.deepcopy(self.sft)
        trace["manifest"]["software"]["torch"]="different-version-for-synthetic-contract-test"
        trace["manifest"]["evidence_kind"]="synthetic_contract"
        trace["manifest"]["execution"]["status"]="synthetic"
        trace["manifest"]["execution"]["synthetic"]=True
        self.assertEqual(validate_trace(trace)["status"],"valid")
        self.assertEqual(validate_trace(trace)["trust"],"imported_claim")
        # Restoring the original package version cannot rescue a wrong loss.
        trace["manifest"]["software"]=self.sft["manifest"]["software"].copy()
        trace["measurements"]["loss_mean"]+=.5
        with self.assertRaisesRegex(ValueError,"reduction"):validate_trace(trace)


    def test_inner_effective_config_checks_hash_and_keeps_outer_object_bound(self):
        inner=json.dumps({"model":{f"field_{i}":i for i in range(308)},
                          "threshold":{"float_sentinel":"+inf"}},separators=(",",":"))
        self.sft["manifest"]["evidence_kind"]="synthetic_contract"
        self.sft["manifest"]["execution"].update(status="synthetic",synthetic=True)
        self.change_json(self.sft,"config",lambda c:c.update(
            effective_config_json=inner,effective_config_sha256=digest_text(inner)))
        self.assertEqual(validate_trace(self.sft)["status"],"valid")
        bad=copy.deepcopy(self.sft)
        self.change_json(bad,"config",lambda c:c.update(effective_config_sha256="0"*64))
        with self.assertRaisesRegex(ValueError,"effective config SHA256"):
            validate_trace(bad)
        bad=copy.deepcopy(self.sft)
        bad["measurements"]["worker_loading"]={
            "effective_config_json":inner,"effective_config_sha256":"0"*64}
        with self.assertRaisesRegex(ValueError,"effective config SHA256"):
            validate_trace(bad)
        bad=copy.deepcopy(self.sft)
        bad["manifest"]["unbounded"]={f"key_{i}":i for i in range(257)}
        with self.assertRaisesRegex(ValueError,"object exceeds"):
            validate_trace(bad)

    def test_inner_config_json_rejects_duplicate_unsafe_nonfinite_and_excessive_data(self):
        texts=['{"x":1,"x":2}','{"constructor":{}}','{"x":NaN}',
               '[]','{"float_sentinel":"NaN"}',
               json.dumps({f"key_{i}":i for i in range(513)}),
               '{"x":'+('['*34)+'0'+(']'*34)+'}',
               json.dumps({"x":[0]*65537})]
        for inner in texts:
            trace=copy.deepcopy(self.sft)
            self.change_json(trace,"config",lambda c:c.update(
                effective_config_json=inner,effective_config_sha256=digest_text(inner)))
            with self.subTest(prefix=inner[:40]),self.assertRaises(ValueError):
                validate_trace(trace)
        for pair in ({"effective_config_json":"{}"},
                     {"effective_config_sha256":digest_text("{}")}):
            trace=copy.deepcopy(self.sft)
            self.change_json(trace,"config",lambda c:c.update(pair))
            with self.assertRaises(ValueError):validate_trace(trace)


    def test_selected_gradient_mapping_does_not_guess_coordinates_or_numeric_strings(self):
        from experiments.runtime.contracts import selected_update
        result=selected_update(self.rl)
        self.assertEqual(result["parameter"],"head[10,0]")
        self.assertEqual(result["gradient"],self.rl["measurements"]["update"]["gradient"])
        self.assertIsNone(selected_update(self.sft))
        for key,value in (("gradient","0.1"),("parameter","head[unknown]"),("before",None)):
            trace=copy.deepcopy(self.rl);trace["measurements"]["update"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_trace(trace)

    def test_gradient_geometry_records_dimensions_without_claiming_a_collective(self):
        trace=copy.deepcopy(self.rl)
        geometry=json.loads(trace["config_json"])["gradient_geometry"]
        self.assertEqual(geometry["tensors"],[
            dict(parameter="head",shape=[27,8],dtype="float64"),
            dict(parameter="critic",shape=[8],dtype="float64")])
        self.assertEqual(validate_trace(trace)["status"],"valid")
        for field,value in (("shape",[]),("shape",[True]),("shape",[10000000,10000000]),("dtype","unknown")):
            bad=copy.deepcopy(trace)
            self.change_json(bad,"config",lambda c:c["gradient_geometry"]["tensors"][0].update({field:value}))
            with self.assertRaises(ValueError):validate_trace(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
