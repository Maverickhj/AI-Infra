"""Bounded synchronous rollout/update/refit evidence, independent of a trainer.

Callers supply copied values only after the original framework operation returns.
This ledger checks ordering and action identity; it does not attest that imported
metadata is true, recompute rewards, or manufacture optimizer/weight evidence.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .contracts import MAX_BYTES, finite_tree, integer, matrix, number, require


class SyncEvents:
    def __init__(self, *, steps, batch_size, sequence_length, vocab_size):
        require(integer(steps) and 1 <= steps <= 100, "invalid event step bound")
        require(integer(batch_size) and 2 <= batch_size <= 8, "invalid event batch bound")
        require(integer(sequence_length) and 8 <= sequence_length <= 511, "invalid event sequence bound")
        require(integer(vocab_size) and vocab_size > 1, "invalid event vocabulary")
        self.steps, self.batch = steps, batch_size
        self.sequence, self.vocab = sequence_length, vocab_size
        self.policy_version, self.generation_version = 0, None
        self.events, self.refits, self.probes = [], {}, {}
        self.generated, self.rollout, self.fixed_input = None, None, None
        self.training_bound = False

    def _event(self, kind, values):
        record = dict(event=len(self.events), kind=kind, **copy.deepcopy(values))
        finite_tree(record)
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        require(len(encoded.encode()) <= MAX_BYTES, "runtime event exceeds 1 MiB")
        require(len(self.events) < self.steps*4+5, "runtime event count exceeded")
        self.events.append(record)
        return copy.deepcopy(record)

    def before_refit(self):
        require(self.generated is None and self.rollout is None, "refit during an unfinished rollout")
        require(self.generation_version != self.policy_version, "duplicate refit without a policy update")

    def refit_completed(self, source):
        self.before_refit()
        require(isinstance(source, dict) and source.get("sha256") and source.get("symbol"),
                "actual refit callable source missing")
        record = self._event("refit", dict(policy_version=self.policy_version,
            generation_version=self.policy_version, completed=True,
            evidence="official_delegate_return", weight_hash_verified=False, source=source))
        self.generation_version = self.policy_version
        self.refits[self.policy_version] = copy.deepcopy(record)
        return record

    def before_generation(self, *, probe=None):
        require(self.generation_version is not None, "generation before an acknowledged refit")
        require(self.generated is None and self.rollout is None, "generation before previous update completed")
        if probe is None:
            require(self.policy_version < self.steps, "unplanned training rollout")
            require(self.generation_version == self.policy_version, "training generation policy is stale")
        else:
            require(self.policy_version == self.steps and probe in ("before_refit", "after_refit"),
                    "fixed-input probe outside the final verification")
            require(probe not in self.probes, "duplicate fixed-input probe")
            if probe == "before_refit":
                require(not self.probes and self.generation_version == self.policy_version-1,
                        "before-refit probe requires the preceding generation version")
            else:
                require("before_refit" in self.probes and self.generation_version == self.policy_version,
                        "after-refit probe requires completed final refit")

    def generation_completed(self, inputs, outputs, *, probe=None):
        self.before_generation(probe=probe)
        ids = inputs.get("input_ids")
        require(isinstance(ids, list) and len(ids) == self.batch, "unexpected generation batch")
        width = len(ids[0])
        require(1 <= width <= self.sequence, "generation prompt exceeds bound")
        valid_token = lambda value: integer(value) and 0 <= value < self.vocab
        matrix(ids, self.batch, width, "generation prompt", valid_token)
        lengths = inputs.get("input_lengths")
        require(isinstance(lengths, list) and len(lengths) == self.batch
                and all(integer(n) and 1 <= n <= width for n in lengths), "invalid prompt lengths")
        output_ids = outputs.get("output_ids")
        require(isinstance(output_ids, list) and len(output_ids) == self.batch, "unexpected generated batch")
        output_width = len(output_ids[0])
        require(2 <= output_width <= self.sequence, "generated sequence exceeds bound")
        matrix(output_ids, self.batch, output_width, "generated tokens", valid_token)
        matrix(outputs.get("logprobs"), self.batch, output_width, "generation logprobs",
               lambda value: number(value) and value <= 1e-6)
        full, new = outputs.get("unpadded_sequence_lengths"), outputs.get("generation_lengths")
        require(isinstance(full, list) and isinstance(new, list)
                and len(full) == len(new) == self.batch, "generation lengths missing")
        for row, length, total, count, prompt in zip(output_ids, lengths, full, new, ids):
            require(integer(total) and integer(count) and 0 < count
                    and length+count == total <= output_width, "inconsistent generation lengths")
            require(row[:length] == prompt[:length], "generation changed the prompt tokens")
        copied_input = dict(input_ids=copy.deepcopy(ids), input_lengths=list(lengths))
        if probe is not None:
            require(copied_input == self.fixed_input, "fixed-input probe changed its input")
        record = self._event("probe" if probe else "generation",
            dict(policy_version=self.policy_version, generation_version=self.generation_version,
                 input=copied_input, output=outputs, probe=probe, greedy=probe is not None))
        if probe is not None:
            self.probes[probe] = copy.deepcopy(record)
        else:
            self.generated = copy.deepcopy(record)
            if self.fixed_input is None:
                self.fixed_input = copied_input
        return record

    def rollout_completed(self, rewards, message_token_rows):
        require(self.generated is not None and self.rollout is None, "rollout without one captured generation")
        require(isinstance(rewards, list) and len(rewards) == self.batch
                and all(number(value) for value in rewards), "invalid actual rollout rewards")
        require(isinstance(message_token_rows, list) and len(message_token_rows) == self.batch,
                "rollout token batch differs from generation")
        output = self.generated["output"]
        for row, generated, total in zip(message_token_rows, output["output_ids"], output["unpadded_sequence_lengths"]):
            require(isinstance(row, list) and total <= len(row) <= self.sequence
                    and all(integer(v) and 0 <= v < self.vocab for v in row),
                    "invalid rollout message tokens")
            require(row[:total] == generated[:total], "rollout messages changed generated actions")
        record = self._event("rollout", dict(policy_version=self.policy_version,
            generation_event=self.generated["event"], rewards=rewards, message_token_rows=message_token_rows))
        self.rollout = copy.deepcopy(record)
        return record

    def bind_training(self, data):
        require(self.generated is not None and self.rollout is not None, "training before captured rollout")
        ids, mask = data.get("input_ids"), data.get("token_mask")
        require(isinstance(ids, list) and len(ids) == self.batch, "training batch differs from rollout")
        width = len(ids[0])
        require(2 <= width <= self.sequence, "training capture exceeds sequence bound")
        matrix(ids, self.batch, width, "training tokens", lambda v: integer(v) and 0 <= v < self.vocab)
        matrix(mask, self.batch, width, "training mask", lambda v: type(v) in (int, float) and v in (0, 1))
        matrix(data.get("generation_logprobs"), self.batch, width, "training generation logprobs")
        require(data.get("sample_mask") == [1]*self.batch, "filtered training samples are unsupported")
        output = self.generated["output"]
        for i, row in enumerate(ids):
            prompt = self.generated["input"]["input_lengths"][i]
            total = output["unpadded_sequence_lengths"][i]
            messages = self.rollout["message_token_rows"][i]
            require(len(messages) <= width and row[:len(messages)] == messages,
                    "training changed or reordered rollout tokens")
            require(mask[i] == [int(prompt <= j < total) for j in range(width)],
                    "training action mask differs from actual generated span")
            for j in range(prompt, total):
                require(data["generation_logprobs"][i][j] == output["logprobs"][i][j],
                        "training generation logprobs changed or shifted")
        if "rewards" in data:
            require(data["rewards"] == self.rollout["rewards"], "training rewards differ from actual rollout")
        self.training_bound = True
        return dict(generation=self.generation_version, previous=self.policy_version,
                    current=self.policy_version, after=self.policy_version+1)

    def update_completed(self, observation):
        require(self.rollout is not None and self.generated is not None and self.training_bound,
                "update before captured and aligned training batch")
        require(isinstance(observation, dict), "optimizer observation missing")
        master, parameter = observation.get("optimizer_parameter"), observation.get("parameter")
        require(isinstance(master, dict) and isinstance(parameter, dict), "actual parameter observation missing")
        before, after, gradient = (master.get(key) for key in ("before", "after", "gradient"))
        require(all(number(v) for v in (before, after, gradient)) and gradient != 0 and before != after,
                "selected parameter does not prove a nonzero update")
        require(integer(parameter.get("backward_calls")) and parameter["backward_calls"] > 0
                and parameter.get("optimizer_executed") is True, "backward/optimizer call evidence missing")
        record = self._event("update", dict(before_version=self.policy_version,
            after_version=self.policy_version+1, observation=observation))
        self.policy_version += 1
        self.generated, self.rollout = None, None
        self.training_bound = False
        return record

    def refit_status(self, version):
        require(version in self.refits, "requested generation version has no completed refit")
        return dict(status="acknowledged", completed=True, generation_version=version,
                    evidence="official_delegate_return", weight_hash_verified=False,
                    event=self.refits[version]["event"],
                    reason="official refit returned successfully; no end-to-end weight hash acknowledgment")

    def finish(self):
        require(self.policy_version == self.steps and self.generated is None and self.rollout is None,
                "planned rollout/update iterations are incomplete")
        require(self.generation_version == self.steps and set(self.probes) == {"before_refit", "after_refit"},
                "final refit and fixed-input probes are incomplete")
        return dict(policy_version=self.policy_version, generation_version=self.generation_version,
                    iterations=self.steps, events=copy.deepcopy(self.events),
                    refit_weight_hash_verified=False)

    def write(self, path):
        result = self.finish()
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
        require(len(text.encode()) <= MAX_BYTES, "event journal exceeds 1 MiB")
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(text)
        return result
