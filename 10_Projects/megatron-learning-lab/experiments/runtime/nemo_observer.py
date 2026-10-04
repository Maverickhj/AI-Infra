"""Thin observation around official synchronous NeMo setup/train/generate/refit.

All numerical work remains in the original policy, generation engine, loss, and
critic. The caller provides a provenance builder: production records actual
runtime metadata; tests explicitly provide synthetic-contract metadata.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .adapters import LossTap
from .capture import TraceCollector, callable_source, effective_config, make_trace
from .contracts import MAX_BYTES, number, require, strict_json
from .nemo_capture import LossRecordSink, actor_terms, compare_official_loss, concatenate, read_records, scalar, value_term
from .nemo_events import SyncEvents


def copied(value):
    import torch
    require(isinstance(value, torch.Tensor), "observed NeMo field must be a real tensor")
    return value.detach().cpu().tolist()


class NeMoObserver:
    def __init__(self, plan, output, mapping, *, extensions, manifest_builder, batch_factory):
        self.plan, self.output, self.mapping = plan, Path(output), mapping
        self.extensions, self.manifest_builder, self.batch_factory = extensions, manifest_builder, batch_factory
        self.restores, self.pending = [], []
        self.policy = self.generation = self.critic = None
        self.probe = None
        self.critic_capture = None
        self.critic_steps = 0
        self.started = datetime.now(timezone.utc).isoformat()
        self.collector = TraceCollector(self.output/"traces", max_records=plan["capture"]["max_records"])

    def configure(self, actual, sha, tokenizer):
        self.base, vocab_size = self.manifest_builder(self.plan, tokenizer)
        kind = self.base.get("evidence_kind")
        require(kind in ("rl_runtime", "synthetic_contract"), "unknown observation provenance")
        self.provenance = "observed_rl" if kind == "rl_runtime" else "reference"
        self.config = dict(self.mapping["loss"], vocab_size=vocab_size,
                           tolerances=self.plan["tolerances"])
        self.events = SyncEvents(steps=self.plan["training"]["steps"],
            batch_size=self.plan["training"]["global_batch_size"],
            sequence_length=self.plan["training"]["sequence_length"], vocab_size=vocab_size)
        self._configuration(actual, sha)
        self.extensions.open()

    def _configuration(self, actual, sha):
        encoded, actual_sha = effective_config(actual)
        require(sha == actual_sha, "effective configuration hash differs")
        self.config.update(effective_config_json=json.dumps(encoded,ensure_ascii=False,separators=(",",":"),allow_nan=False),
                           effective_config_sha256=sha)

    def setup(self, original, *args, **kwargs):
        return self.extensions.setup(original, *args, **kwargs)

    def _patch(self, owner, name, replacement):
        namespace = owner.__dict__
        own = name in namespace
        previous = namespace.get(name)
        original = getattr(owner, name)
        self.restores.append((owner, name, own, previous))
        setattr(owner, name, replacement)
        return original

    def _source(self, function, component):
        source = callable_source(function, component)
        if self.provenance == "observed_rl":
            require(Path(source["path"]).is_relative_to(Path(self.plan["sources"]["nemo_rl"]).resolve()),
                    "loaded NeMo callable is outside the inspected source checkout")
        self.base["runtime_sources"].append(source)
        return source

    def attach(self, result, launcher):
        algorithm = self.mapping["algorithm"]
        require(isinstance(result, tuple) and len(result) == (13 if algorithm == "grpo" else 12),
                "unknown official setup result")
        self.policy, self.generation = result[:2]
        require(self.policy is not self.generation and self.generation is not None,
                "capture requires a separate colocated generation engine")
        self.actor_loss = result[6]
        if algorithm == "ppo":
            self.critic, self.critic_loss = result[2], result[7]
        master = result[10 if algorithm == "grpo" else 11]
        from .nemo_entry import runtime_config
        actual = runtime_config(master, self.plan)
        encoded, sha = effective_config(actual)
        self._configuration(actual, sha)
        (self.output/"effective-config-after-setup.json").write_text(json.dumps(encoded,indent=2)+"\n")
        self.config["effective_config_stage"] = "after_official_setup"
        train = getattr(launcher, algorithm+"_train")
        namespace = sys.modules[train.__module__]
        refit = namespace.refit_policy_generation
        rollout = namespace.run_multi_turn_rollout
        refit_source = self._source(refit, "nemo_refit")
        self._source(rollout, "nemo_rollout")
        self._source(self.generation.generate, "nemo_generation_driver")
        self._source(self.policy.train, "nemo_policy_driver")
        generate = self.generation.generate
        actor_train = self.policy.train

        def observe_generation(data, *, greedy=False):
            require(greedy is (self.probe is not None), "generation sampling mode differs from capture phase")
            self.events.before_generation(probe=self.probe)
            inputs = {key:copied(data[key]) for key in ("input_ids","input_lengths")}
            original = generate(data, greedy=greedy)
            outputs = {key:copied(original[key]) for key in
                       ("output_ids","logprobs","generation_lengths","unpadded_sequence_lengths")}
            self.events.generation_completed(inputs, outputs, probe=self.probe)
            return original

        def observe_rollout(*args, **kwargs):
            original = rollout(*args, **kwargs)
            require(isinstance(original, tuple) and len(original) == 2, "unknown rollout return")
            batch = original[0]
            rows = [[token for message in messages for token in copied(message["token_ids"])]
                    for messages in batch["message_log"]]
            self.events.rollout_completed(copied(batch["total_reward"]), rows)
            return original

        def observe_refit(policy, generation, colocated, *args, **kwargs):
            require(policy is self.policy and generation is self.generation and colocated is True,
                    "refit objects differ from the observed official setup")
            self.events.before_refit()
            original = refit(policy, generation, colocated, *args, **kwargs)
            self.events.refit_completed(refit_source)
            return original

        self._patch(self.generation, "generate", observe_generation)
        self._patch(namespace, "run_multi_turn_rollout", observe_rollout)
        self._patch(namespace, "refit_policy_generation", observe_refit)
        self._patch(self.policy, "train",
                    lambda data, loss, **kwargs:self._train("actor",actor_train,data,loss,kwargs))
        if self.critic is not None:
            critic_train = self.critic.train
            self._source(critic_train, "nemo_critic_driver")
            self._patch(self.critic, "train",
                        lambda data, loss, **kwargs:self._train("critic",critic_train,data,loss,kwargs))
        self.refit = observe_refit
        return result

    def _train(self, role, original, batch, loss, kwargs):
        require(set(kwargs) <= {"timer"}, "uninspected training call options")
        require(loss is (self.actor_loss if role == "actor" else self.critic_loss),
                "official loss object changed after setup")
        fields = ("input_ids","token_mask","generation_logprobs","sample_mask")
        supplied = {key:copied(batch[key]) for key in fields}
        if "rewards" in batch:
            supplied["rewards"] = copied(batch["rewards"])
        versions = self.events.bind_training(supplied)
        iteration = versions["after"]
        directory = self.output/"capture"/f"iteration-{iteration:04d}"/role
        directory.mkdir(parents=True,exist_ok=False)
        sink = LossRecordSink(directory,role=role,config=self.config,versions=versions,
            delegate=loss,max_records=self.plan["training"]["global_batch_size"],
            parameter={"name":self.plan["capture"]["parameter_name"],
                       "index":self.plan["capture"]["parameter_index"]})
        # The worker determines actual distributed groups. A synthetic test may
        # explicitly supply its own group reader through the test sink factory.
        if self.provenance == "reference":
            sink.group_reader = self.synthetic_groups
        result = original(batch, LossTap(loss,sink), **kwargs)
        records = read_records(directory,self.plan["training"]["global_batch_size"])
        data, measured = concatenate(records)
        grad_norm = scalar(result["grad_norm"], role+" gradient norm")
        require(grad_norm > 0, "official gradient norm does not show a nonzero "+role+" backward")
        if role == "critic":
            measured["value_loss"] = value_term(data,measured,self.config)
            official = compare_official_loss(records, measured["value_loss"])
            self._compare_reported_loss(result,official)
            self.critic_capture = dict(data=data,measured=measured,official=official,records=records,grad_norm=grad_norm)
            self.critic_steps += 1
            return result
        if self.mapping["algorithm"] == "ppo":
            critic = self.critic_capture
            require(critic is not None and critic["data"] == data, "actor/critic action identity differs")
            measured.update(critic["measured"])
            measured["official_value_loss"] = critic["official"]
            measured["critic_gradient_norm"] = critic["grad_norm"]
            self.critic_capture = None
        else:
            measured.update(values=None,old_values=None,returns=None,value_loss=None)
        measured.update(actor_terms(data,measured,self.config))
        official = compare_official_loss(records,measured["policy_loss"])
        self._compare_reported_loss(result,official)
        path = directory.parent/"actor-update.json"
        require(path.stat().st_size <= MAX_BYTES, "parameter observation exceeds bound")
        observation = strict_json(path.read_text())
        if self.provenance == "observed_rl":
            loading = observation.get("loading")
            require(isinstance(loading,dict) and loading.get("status") == "official_conversion_and_load_completed",
                    "actual initial checkpoint loading was not observed")
            for row in loading["imports"]+loading["loads"]:
                self.base["runtime_sources"].append(row["source"])
        measured.update(official_policy_loss=official,gradient_norm=grad_norm,
            rewards=copy.deepcopy(self.events.rollout["rewards"]),update=observation,
            actions_origin="model_generation" if self.provenance == "observed_rl" else "synthetic_contract")
        self.events.update_completed(observation)
        self.pending.append(dict(data=data,measurements=measured,parallel=records[0]["parallel"],
                                 sources=[row["source"] for row in records]+[observation["delegated_train"]]))
        return result

    @staticmethod
    def synthetic_groups():
        return dict(tp=1,pp=1,cp=1,dp=1,ep=1,world_size=1,groups_origin="in_memory_reference")

    @staticmethod
    def _compare_reported_loss(result, captured):
        reported = scalar(result["loss"], "official driver loss")
        expected = captured["value"]
        require(abs(reported-expected) <= 1e-5+1e-5*abs(expected),
                "driver loss differs from captured official loss calls")

    def _batch(self, ids, lengths):
        import torch
        return self.batch_factory(dict(input_ids=torch.tensor(ids,dtype=torch.long),
                                       input_lengths=torch.tensor(lengths,dtype=torch.long)))

    def _probe(self, phase):
        self.probe = phase
        try:
            prompt = self.events.fixed_input
            require(prompt is not None, "no actual rollout prompt available for fixed-input verification")
            self.generation.generate(self._batch(prompt["input_ids"],prompt["input_lengths"]),greedy=True)
        finally:
            self.probe = None

    def _fixed_input_verification(self):
        # Two bounded diagnostic generations are separate from training rollouts.
        # A greedy answer may stay unchanged after a small update.
        if self.critic is not None:
            self.critic.finish_training()
        self.policy.offload_after_refit()
        self.generation.prepare_for_generation()
        self._probe("before_refit")
        self.generation.finish_generation()
        self.policy.prepare_for_lp_inference()
        self.refit(self.policy,self.generation,True)
        self._probe("after_refit")
        self.generation.finish_generation()
        self.policy.prepare_for_lp_inference()
        event = self.events.probes["after_refit"]
        output = event["output"]
        recomputed = self.policy.get_logprobs(self._batch(output["output_ids"],output["unpadded_sequence_lengths"]))
        actual = copied(recomputed["logprobs"])
        require(len(actual) == self.events.batch and all(len(row)==len(output["output_ids"][0]) for row in actual),
                "fixed-input policy logprob shape differs")
        atol, rtol = self.plan["tolerances"]["logprob_atol"], self.plan["tolerances"]["logprob_rtol"]
        differences = []
        for i, row in enumerate(actual):
            for j in range(event["input"]["input_lengths"][i],output["unpadded_sequence_lengths"][i]):
                generated = output["logprobs"][i][j]
                require(number(row[j]) and row[j] <= 1e-6, "invalid recomputed fixed-input logprob")
                difference = abs(row[j]-generated)
                require(difference <= atol+rtol*abs(row[j]), "refit fixed-input logprob alignment failed")
                differences.append(difference)
        self.policy.offload_after_refit()
        return dict(status="passed",generation_calls=2,policy_version=self.events.policy_version,
                    generation_version=self.events.generation_version,
                    selected_tokens=len(differences),max_abs_difference=max(differences),
                    atol=atol,rtol=rtol,
                    meaning="same prompt before/after refit; after-refit generated token logprobs vs current policy",
                    weight_hash_verified=False)

    def finish(self):
        require(len(self.pending) == self.plan["training"]["steps"], "planned actor iterations are incomplete")
        require(self.critic_steps == (len(self.pending) if self.critic is not None else 0)
                and self.critic_capture is None, "planned critic iterations are incomplete")
        verification = self._fixed_input_verification()
        self.events.write(self.output/"events.json")
        completed = datetime.now(timezone.utc).isoformat()
        for row in self.pending:
            measured = row["measurements"]
            measured["refit"] = self.events.refit_status(row["data"]["policy_versions"]["after"])
            manifest = copy.deepcopy(self.base)
            manifest["parallel"] = row["parallel"]
            manifest["runtime_sources"] += row["sources"]
            manifest["execution"].update(started_at=self.started,completed_at=completed)
            trace = make_trace(self.plan["run_id"]+"-"+str(row["data"]["policy_versions"]["after"]),
                "rl",self.provenance,manifest,self.config,row["data"],measured)
            self.collector.write(trace)
        result = dict(status="completed_nemo_entry" if self.provenance == "observed_rl" else "synthetic_nemo_contract",
            iterations=len(self.pending),critic_iterations=self.critic_steps,
            traces=self.collector.records,fixed_input_verification=verification,
            refit_weight_hash_verified=False)
        with (self.output/"result.json").open("x") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        return result

    def close(self):
        for owner,name,own,previous in reversed(self.restores):
            if own:
                setattr(owner,name,previous)
            else:
                delattr(owner,name)
        self.restores.clear()
        self.extensions.close()
