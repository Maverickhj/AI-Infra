"""Read-only evidence storage and actual callable/config source checks."""
import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import tempfile
import unittest
from experiments.runtime.capture import TraceCollector, callable_source, effective_config, make_trace

ROOT=Path(__file__).resolve().parents[1]
class CaptureChecks(unittest.TestCase):
    def setUp(self):
        self.fixture=json.loads((ROOT/"content/fixtures/runtime-reference.json").read_text())["traces"][0]

    def test_trace_write_validates_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"capture"
            collector=TraceCollector(output)
            invalid=copy.deepcopy(self.fixture);invalid["measurements"]["loss_mean"]+=1
            with self.assertRaisesRegex(ValueError,"semantic mismatch"):collector.write(invalid)
            self.assertFalse(output.exists())
            record=collector.write(self.fixture)
            self.assertEqual(json.loads(Path(record["path"]).read_text()),self.fixture)
            self.assertEqual(record["trust"],"imported_claim")

    def test_untrusted_run_id_cannot_choose_path_and_existing_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture["run_id"]="../../outside.json"
            collector=TraceCollector(folder,max_records=1)
            path=Path(collector.write(self.fixture)["path"])
            self.assertEqual(path.name,"trace-0000.json")
            with self.assertRaisesRegex(ValueError,"limit"):collector.write(self.fixture)
            another=TraceCollector(folder)
            with self.assertRaises(FileExistsError):another.write(self.fixture)
            self.assertEqual(json.loads(path.read_text())["run_id"],"../../outside.json")

    def test_constructor_keeps_exact_config_input_bytes_and_checks_provenance(self):
        r=self.fixture
        got=make_trace(r["run_id"],r["task"],r["provenance"],r["manifest"],
                       json.loads(r["config_json"]),json.loads(r["input_json"]),r["measurements"])
        self.assertEqual(got["input_sha256"],r["input_sha256"])
        self.assertEqual(got["config_sha256"],r["config_sha256"])
        with self.assertRaises(ValueError):
            make_trace(r["run_id"],"sft","observed_bridge",r["manifest"],
                       json.loads(r["config_json"]),json.loads(r["input_json"]),r["measurements"])

    def test_actual_callable_source_and_hash_locate_this_test(self):
        source=callable_source(self.test_actual_callable_source_and_hash_locate_this_test,"contract_test")
        import hashlib
        self.assertEqual(source["sha256"],hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.assertIn("test_actual_callable_source",source["symbol"])
        self.assertGreater(source["end_line"],source["line"])
        self.assertEqual(source["evidence"],"loaded_python_callable")

    def test_effective_config_uses_actual_values_not_static_defaults(self):
        class Mode(Enum):BF16="bf16"
        @dataclass
        class SyntheticConfig:
            steps:int
            dtype:Mode
            output:Path
        data,sha=effective_config(SyntheticConfig(2,Mode.BF16,Path("output")))
        self.assertEqual(data["steps"],2)
        self.assertEqual(data["dtype"]["value"],"bf16")
        self.assertEqual(data["output"],"output")
        self.assertEqual(len(sha),64)
        self.assertNotEqual(sha,effective_config(SyntheticConfig(3,Mode.BF16,Path("output")))[1])
        class Resolved:
            def to_dict(self):return {"steps":7}
        self.assertEqual(effective_config(Resolved())[0],{"steps":7})

    def test_unknown_config_object_does_not_execute_repr(self):
        class Trap:
            def __repr__(self):raise AssertionError("do not invoke")
        with self.assertRaisesRegex(ValueError,"unsupported effective config"):effective_config({"unknown":Trap()})
        with self.assertRaises(ValueError):effective_config({"metric":float("nan")})

    def test_config_infinity_sentinels_preserve_sign_and_remain_json(self):
        data, sha=effective_config({"upper":float("inf"),"lower":float("-inf")})
        self.assertEqual(data,{"upper":{"float_sentinel":"+inf"},"lower":{"float_sentinel":"-inf"}})
        self.assertEqual(json.loads(json.dumps(data,allow_nan=False)),data)
        self.assertNotEqual(sha,effective_config({"upper":0,"lower":0})[1])
        for value in (float("inf"),float("-inf"),float("nan")):
            invalid=copy.deepcopy(self.fixture)
            invalid["measurements"]["loss_mean"]=value
            with self.assertRaises(ValueError):
                make_trace(invalid["run_id"],invalid["task"],invalid["provenance"],invalid["manifest"],
                           json.loads(invalid["config_json"]),json.loads(invalid["input_json"]),
                           invalid["measurements"])
