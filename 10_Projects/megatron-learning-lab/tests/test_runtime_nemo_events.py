"""Synthetic event ordering/action contracts; no model, Torch or Ray execution."""
import copy
import tempfile
from pathlib import Path
import unittest
from experiments.runtime.nemo_events import SyncEvents

SOURCE = dict(symbol="synthetic_refit", sha256="a"*64)
INPUT = dict(input_ids=[[1],[1]], input_lengths=[1,1])
OUTPUT = dict(output_ids=[[1,2,3],[1,4,0]], generation_lengths=[2,1],
              unpadded_sequence_lengths=[3,2], logprobs=[[0.,-.2,-.3],[0.,-.4,0.]])
TRAIN = dict(input_ids=OUTPUT["output_ids"], token_mask=[[0,1,1],[0,1,0]],
             generation_logprobs=OUTPUT["logprobs"], sample_mask=[1,1], rewards=[1.,0.])
UPDATE = dict(optimizer_parameter=dict(before=1.,after=.9,gradient=1.),
              parameter=dict(backward_calls=2,optimizer_executed=True))


class NeMoEventTests(unittest.TestCase):
    def ledger(self):
        return SyncEvents(steps=2,batch_size=2,sequence_length=8,vocab_size=16)

    def rollout(self, events):
        events.before_generation()
        events.generation_completed(INPUT, OUTPUT)
        events.rollout_completed([1.,0.], [[1,2,3],[1,4]])
        return events.bind_training(TRAIN)

    def finished(self):
        events = self.ledger()
        for step in range(2):
            events.refit_completed(SOURCE)
            versions = self.rollout(events)
            self.assertEqual(versions, dict(generation=step,previous=step,current=step,after=step+1))
            events.update_completed(UPDATE)
        events.generation_completed(INPUT,OUTPUT,probe="before_refit")
        events.refit_completed(SOURCE)
        events.generation_completed(INPUT,OUTPUT,probe="after_refit")
        return events

    def test_two_iterations_and_final_fixed_input_refit_keep_distinct_versions(self):
        events = self.finished()
        result = events.finish()
        self.assertEqual(result["iterations"], 2)
        self.assertFalse(result["refit_weight_hash_verified"])
        self.assertEqual(events.probes["before_refit"]["generation_version"],1)
        self.assertEqual(events.probes["after_refit"]["generation_version"],2)
        status = events.refit_status(2)
        self.assertEqual(status["status"],"acknowledged")
        self.assertNotIn("ack_hash",status)
        self.assertFalse(status["weight_hash_verified"])

    def test_stale_generation_early_training_and_duplicate_refit_are_rejected(self):
        events = self.ledger()
        with self.assertRaisesRegex(ValueError,"before an acknowledged"):
            events.before_generation()
        with self.assertRaisesRegex(ValueError,"before captured"):
            events.bind_training(TRAIN)
        events.refit_completed(SOURCE)
        with self.assertRaisesRegex(ValueError,"duplicate refit"):
            events.refit_completed(SOURCE)
        self.rollout(events)
        with self.assertRaisesRegex(ValueError,"unfinished rollout"):
            events.before_refit()
        events.update_completed(UPDATE)
        with self.assertRaisesRegex(ValueError,"stale"):
            events.before_generation()
        with self.assertRaisesRegex(ValueError,"incomplete"):
            events.finish()

    def test_generated_tokens_lengths_and_logprobs_are_not_repaired_or_shifted(self):
        for field, value in (("output_ids",[[9,2,3],[1,4,0]]),
                             ("generation_lengths",[1,1]),
                             ("logprobs",[[0.,.2,-.3],[0.,-.4,0.]])):
            events = self.ledger(); events.refit_completed(SOURCE)
            changed = copy.deepcopy(OUTPUT); changed[field] = value
            with self.assertRaises(ValueError):
                events.generation_completed(INPUT,changed)
            self.assertIsNone(events.generated)
            self.assertEqual(len(events.events),1)

    def test_training_must_preserve_exact_actions_masks_probabilities_and_rewards(self):
        events = self.ledger(); events.refit_completed(SOURCE)
        self.rollout(events)
        for field, value in (
            ("input_ids",[[1,4,0],[1,2,3]]),
            ("token_mask",[[0,1,0],[0,1,0]]),
            ("generation_logprobs",[[0.,-.3,-.2],[0.,-.4,0.]]),
            ("sample_mask",[1,0]), ("rewards",[0.,1.]),
        ):
            changed=copy.deepcopy(TRAIN);changed[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                events.bind_training(changed)
        self.assertEqual(events.bind_training(TRAIN)["after"],1)

    def test_zero_or_missing_update_evidence_never_advances_policy_version(self):
        events=self.ledger();events.refit_completed(SOURCE);self.rollout(events)
        for key,value in (("after",1.),("gradient",0.),("gradient",float("nan"))):
            changed=copy.deepcopy(UPDATE);changed["optimizer_parameter"][key]=value
            with self.assertRaisesRegex(ValueError,"nonzero update"):
                events.update_completed(changed)
            self.assertEqual(events.policy_version,0)
        changed=copy.deepcopy(UPDATE);changed["parameter"]["optimizer_executed"]=False
        with self.assertRaisesRegex(ValueError,"call evidence"):
            events.update_completed(changed)

    def test_fixed_probe_change_and_duplicate_output_are_rejected(self):
        events=self.finished()
        with self.assertRaisesRegex(ValueError,"duplicate fixed-input"):
            events.generation_completed(INPUT,OUTPUT,probe="after_refit")
        with self.assertRaisesRegex(ValueError,"unplanned"):
            events.before_generation()
        unfinished=self.ledger()
        for _ in range(2):
            unfinished.refit_completed(SOURCE)
            self.rollout(unfinished)
            unfinished.update_completed(UPDATE)
        changed_input=copy.deepcopy(INPUT);changed_input["input_ids"]=[[5],[5]]
        changed_output=copy.deepcopy(OUTPUT)
        changed_output["output_ids"][0][0]=changed_output["output_ids"][1][0]=5
        with self.assertRaisesRegex(ValueError,"changed its input"):
            unfinished.generation_completed(changed_input,changed_output,probe="before_refit")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"events.json"
            events.write(path);before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                events.write(path)
            self.assertEqual(path.read_bytes(),before)

    def test_recorded_values_are_detached_from_mutable_caller_containers(self):
        events=self.ledger();events.refit_completed(SOURCE)
        copied=copy.deepcopy(OUTPUT)
        returned=events.generation_completed(INPUT,copied)
        copied["output_ids"][0][1]=8
        returned["output"]["output_ids"][0][1]=9
        self.assertEqual(events.generated["output"]["output_ids"][0][1],2)
        events.rollout_completed([1.,0.], [[1,2,3],[1,4]])
        self.assertEqual(events.bind_training(TRAIN)["current"],0)
