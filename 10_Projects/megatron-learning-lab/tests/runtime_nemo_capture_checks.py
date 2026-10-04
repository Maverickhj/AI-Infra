"""Actual CPU execution of verbatim pinned loss bodies with synthetic interfaces.

No NeMo trainer, GPU, Ray cluster, model, or rollout is initialized. Independent
scalar diagnostics are checked against the source loss, with unequal masks.
"""
import ast
import copy
from enum import Enum
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType
from typing import Any, Optional, TypedDict
import unittest
from unittest.mock import patch

import torch
from pydantic import BaseModel
from experiments.runtime.adapters import LossTap
from experiments.runtime.nemo_worker_observation import TrainObservationMixin
from experiments.runtime.nemo_capture import (LossRecordSink, actor_terms, concatenate,
    compare_official_loss, read_records, value_term)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_groups():
    return dict(tp=1, pp=1, cp=1, dp=1, ep=1, world_size=1, groups_origin="in_memory_reference")


class SyntheticLossType(Enum):
    TOKEN_LEVEL = "token"
    SEQUENCE_LEVEL = "sequence"


class SyntheticInputType(Enum):
    LOGPROB = "logprob"
    LOGIT = "logit"


class RecordingDelegate:
    def __init__(self, official_body):
        self.original, self.calls, self.last = official_body, 0, None

    def __call__(self, *args):
        self.calls += 1
        self.last = self.original(*args)
        return self.last


class SyntheticOptimizer:
    """The explicit optimizer interface backed by an actual CPU SGD step."""
    def __init__(self, model):
        self.parameters = list(model.parameters())
        self.sgd = torch.optim.SGD(self.parameters, lr=.1)

    def get_parameters(self):
        return self.parameters


class SyntheticWorker:
    """Only the orchestration is synthetic; loss/backward/SGD use real Torch."""
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight.fill_(1.)
        self.optimizer = SyntheticOptimizer(self.model)
        self.calls, self.result = 0, {"scope": "synthetic CPU worker"}

    def train(self, data, loss_fn, *, fail=False):
        self.calls += 1
        self.optimizer.sgd.zero_grad()
        output = torch.tensor([[-1., -2., -3.]], dtype=torch.float64)*self.model.weight[0, 0]
        loss, _ = loss_fn(output, data, torch.tensor(1.), torch.tensor(2.))
        loss.backward()
        if fail:
            raise RuntimeError("synthetic failure after backward")
        self.optimizer.sgd.step()
        return self.result


class ObservedSyntheticWorker(TrainObservationMixin, SyntheticWorker):
    pass


class NeMoCaptureCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        manifest = json.loads((ROOT/"tests/fixtures/nemo-loss/manifest.json").read_text())
        cls.module = ModuleType("lab_extracted_nemo_cpu_loss")
        cls.module.__dict__.update(torch=torch, BaseModel=BaseModel, LossFunction=object,
            LossType=SyntheticLossType, LossInputType=SyntheticInputType,
            Tensor=torch.Tensor, BatchedDataDict=dict, Any=Any, Optional=Optional, TypedDict=TypedDict)
        sys.modules[cls.module.__name__] = cls.module
        cls.addClassCleanup(sys.modules.pop, cls.module.__name__, None)
        for source in reversed(manifest["sources"]):
            # Blank lines retain original line numbers; only verbatim declared
            # bodies are executed. Imported framework interfaces are synthetic.
            lines = ["\n"]*max(block["end_line"] for block in source["blocks"])
            for block in source["blocks"]:
                raw = block["content"].encode()
                if hashlib.sha256(raw).hexdigest() != block["sha256"]:
                    raise ValueError("changed pinned loss body")
                lines[block["start_line"]-1:block["end_line"]] = block["content"].splitlines(keepends=True)
            path = Path(cls.folder.name)/(source["source_id"]+".py")
            path.write_text("".join(lines))
            exec(compile(path.read_text(), str(path), "exec"), cls.module.__dict__)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.versions = dict(generation=0, previous=0, current=0, after=1)
        self.config = dict(vocab_size=16, algorithm="grpo", loss_variant="clipped_pg_k3",
            offpolicy_correction=False, reduction="token", ratio_clip=.2, kl_beta=.01,
            force_on_policy=False, kl_input_clamp=20., kl_output_clamp=10.,
            kl_sampling="non_is_score_gradient")
        self.data = dict(
            input_ids=torch.tensor([[1,2,3,0],[1,4,0,0]]),
            token_mask=torch.tensor([[0.,1.,1.,0.],[0.,1.,0.,0.]], dtype=torch.float64),
            sample_mask=torch.ones(2, dtype=torch.float64),
            generation_logprobs=torch.tensor([[0.,-1.2,-2.2,0.],[0.,-.3,0.,0.]], dtype=torch.float64),
            prev_logprobs=torch.tensor([[0.,-1.1,-2.1,0.],[0.,-.4,0.,0.]], dtype=torch.float64),
            reference_policy_logprobs=torch.tensor([[0.,-1.3,-2.3,0.],[0.,-.6,0.,0.]], dtype=torch.float64),
            advantages=torch.tensor([[0.,1.,-1.,0.],[0.,.5,0.,0.]], dtype=torch.float64))
        self.current = torch.tensor([[-1.,-2.,-3.],[-.5,-1.,-2.]], dtype=torch.float64)
        self.sequences, self.tokens = torch.tensor(2.), torch.tensor(3.)

    def delegate(self, reduction="token", force=False):
        return self.module.ClippedPGLossFn(self.module.ClippedPGLossConfig(
            token_level_loss=reduction=="token", force_on_policy_ratio=force))

    def actor_records(self, config):
        original = self.delegate(config["reduction"], config["force_on_policy"])
        delegate = RecordingDelegate(original)
        sink = LossRecordSink(self.path, role="actor", config=config, versions=self.versions,
            delegate=delegate, max_records=2, group_reader=synthetic_groups)
        tap = LossTap(delegate, sink)
        gradients = []
        for i in range(2):
            micro = {k:v[i:i+1] for k,v in self.data.items()}
            leaf = self.current[i:i+1].clone().requires_grad_()
            baseline_leaf = leaf.detach().clone().requires_grad_()
            baseline = original(baseline_leaf, micro, self.sequences, self.tokens)
            result = tap(leaf, micro, self.sequences, self.tokens)
            self.assertIs(result, delegate.last)
            self.assertEqual(delegate.calls, i+1)
            result[0].backward(); baseline[0].backward()
            torch.testing.assert_close(leaf.grad, baseline_leaf.grad, rtol=0, atol=0)
            gradients.append(leaf.grad)
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        return read_records(self.path, 2)

    def test_token_global_normalization_and_original_backward_match_source_loss(self):
        records = self.actor_records(self.config)
        data, measured = concatenate(records)
        measured.update(actor_terms(data, measured, self.config))
        compared = compare_official_loss(records, measured["policy_loss"])
        self.assertLess(abs(compared["value"]-compared["diagnostic_value"]), 1e-8)
        self.assertEqual(sum(map(sum, data["response_mask"])), 3)
        self.assertEqual([sum(r["data"]["response_mask"][0]) for r in records], [2,1])
        # A mean of microbatch means is the wrong global token reduction.
        wrong = sum(r["official_loss"]*3/sum(r["data"]["response_mask"][0]) for r in records)/2
        self.assertGreater(abs(wrong-compared["value"]), .01)
        with self.assertRaisesRegex(ValueError, "official globally"):
            compare_official_loss(records, wrong)

    def test_sequence_and_force_branches_keep_their_real_gradients(self):
        self.config.update(reduction="sequence", force_on_policy=True)
        records = self.actor_records(self.config)
        data, measured = concatenate(records)
        terms = actor_terms(data, measured, self.config)
        self.assertEqual(terms["ratio"], [[1.]*4,[1.]*4])
        compare_official_loss(records, terms["policy_loss"])

    def test_value_head_capture_matches_clipped_official_loss_and_gradient(self):
        self.config.update(algorithm="ppo", value_clip=.2)
        data = {**self.data, "values":torch.tensor([[0.,.2,.4,0.],[0.,.1,0.,0.]],dtype=torch.float64),
            "returns":torch.tensor([[0.,1.,-.2,0.],[0.,.5,0.,0.]],dtype=torch.float64)}
        original = self.module.MseValueLossFn(self.module.MseValueLossConfig(scale=1.,cliprange=.2))
        delegate = RecordingDelegate(original)
        sink = LossRecordSink(self.path, role="critic", config=self.config, versions=self.versions,
            delegate=delegate, max_records=2, group_reader=synthetic_groups)
        tap = LossTap(delegate, sink)
        for i in range(2):
            leaf = (data["values"][i:i+1]+.3).unsqueeze(-1).requires_grad_()
            result = tap(leaf, {k:v[i:i+1] for k,v in data.items()}, self.sequences, self.tokens)
            self.assertIs(result, delegate.last)
            result[0].backward()
            self.assertIsNotNone(leaf.grad)
        records = read_records(self.path, 2)
        canonical, measured = concatenate(records)
        expected = value_term(canonical, measured, self.config)
        compare_official_loss(records, expected)
        self.assertEqual(measured["old_values"], data["values"].tolist())
        self.assertGreater(expected, 0)

    def test_wrong_normalizer_or_missing_microbatch_does_not_form_a_trace(self):
        records = self.actor_records(self.config)
        records[0]["global_valid_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "normalizers"):
            concatenate(records)
        with self.assertRaisesRegex(ValueError, "loss captures"):
            read_records(self.path, 3)

    def test_fractional_mask_filtered_sample_and_extra_calls_are_rejected(self):
        delegate = RecordingDelegate(self.delegate())
        sink = LossRecordSink(self.path, role="actor", config=self.config, versions=self.versions,
            delegate=delegate, max_records=1, group_reader=synthetic_groups)
        data = {k:v[:1].clone() for k,v in self.data.items()}
        leaf = self.current[:1].requires_grad_()
        result = delegate(leaf, data, self.sequences, self.tokens)
        for key, index, value in (("token_mask",(0,1),.5),("sample_mask",(0,),0)):
            changed = copy.deepcopy(data); changed[key][index] = value
            with self.assertRaises(ValueError):
                sink((leaf,changed,self.sequences,self.tokens),{},result)
        sink((leaf,data,self.sequences,self.tokens),{},result)
        with self.assertRaisesRegex(ValueError, "rerun"):
            sink((leaf,data,self.sequences,self.tokens),{},result)

    def worker_inputs(self):
        directory = self.path/"actor"
        directory.mkdir()
        delegate = self.delegate()
        sink = LossRecordSink(directory, role="actor", config=self.config, versions=self.versions,
            delegate=delegate, max_records=1, group_reader=synthetic_groups,
            parameter={"name": "weight", "index": [0, 0]})
        return {k:v[:1] for k,v in self.data.items()}, LossTap(delegate, sink)

    def test_worker_observes_actual_update_without_changing_delegate_result_or_gradient(self):
        data, tap = self.worker_inputs()
        actual, baseline = ObservedSyntheticWorker(), SyntheticWorker()
        baseline.train(data, self.delegate())
        with patch("experiments.runtime.nemo_worker_observation.unwrap_training_model", lambda model: model):
            result = actual.train(data, tap)
        self.assertIs(result, actual.result)
        self.assertEqual(actual.calls, 1)
        torch.testing.assert_close(actual.model.weight, baseline.model.weight, atol=0, rtol=0)
        torch.testing.assert_close(actual.model.weight.grad, baseline.model.weight.grad, atol=0, rtol=0)
        record = json.loads((self.path/"actor-update.json").read_text())
        self.assertEqual(record["parameter"]["backward_calls"], 1)
        self.assertNotEqual(record["optimizer_parameter"]["before"], record["optimizer_parameter"]["after"])
        self.assertNotEqual(record["optimizer_parameter"]["gradient"], 0)
        self.assertFalse(actual.model.weight._backward_hooks)

    def test_worker_failure_removes_hook_and_never_reports_an_optimizer_update(self):
        data, tap = self.worker_inputs()
        worker = ObservedSyntheticWorker()
        with patch("experiments.runtime.nemo_worker_observation.unwrap_training_model", lambda model: model):
            with self.assertRaisesRegex(RuntimeError, "after backward"):
                worker.train(data, tap, fail=True)
        self.assertEqual(worker.calls, 1)
        self.assertEqual(worker.model.weight.item(), 1.)
        self.assertFalse(worker.model.weight._backward_hooks)
        self.assertFalse((self.path/"actor-update.json").exists())

    def test_existing_update_record_rejects_before_another_optimizer_step(self):
        data, tap = self.worker_inputs()
        worker = ObservedSyntheticWorker()
        path = self.path/"actor-update.json"
        path.write_text("existing")
        with patch("experiments.runtime.nemo_worker_observation.unwrap_training_model", lambda model: model):
            with self.assertRaisesRegex(ValueError, "already exists"):
                worker.train(data, tap)
        self.assertEqual(worker.calls, 0)
        self.assertEqual(path.read_text(), "existing")
        self.assertFalse(worker.model.weight._backward_hooks)

    def test_sink_transport_preserves_exclusive_records_and_never_initializes_cuda(self):
        delegate = RecordingDelegate(self.delegate())
        sink = LossRecordSink(self.path, role="actor", config=self.config, versions=self.versions,
            delegate=delegate, max_records=2, group_reader=synthetic_groups)
        copied = pickle.loads(pickle.dumps(LossTap(delegate, sink)))
        leaf = self.current[:1].clone().requires_grad_()
        data = {k:v[:1] for k,v in self.data.items()}
        copied(leaf, data, self.sequences, self.tokens)
        with self.assertRaises(FileExistsError):
            LossTap(delegate, sink)(leaf, data, self.sequences, self.tokens)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":
    unittest.main(verbosity=2)
