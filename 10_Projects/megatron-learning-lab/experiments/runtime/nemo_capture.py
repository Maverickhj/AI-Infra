"""Bounded observation of official NeMo loss calls; no trainer or optimizer.

The original loss is executed by LossTap exactly once. Diagnostics are recomputed
from copied token logprobs and compared with the actual globally normalized loss.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from .adapters import bounded_shape, copy_rows, nemo_action_slice, tensor
from .capture import callable_source
from .contracts import MAX_BYTES, finite_tree, matrix, number, require, strict_json


def runtime_groups():
    import torch
    from megatron.core import parallel_state as ps
    require(torch.distributed.is_initialized(), "worker distributed group is not initialized")
    groups = dict(tp=ps.get_tensor_model_parallel_world_size(),
        pp=ps.get_pipeline_model_parallel_world_size(), cp=ps.get_context_parallel_world_size(),
        dp=ps.get_data_parallel_world_size(), ep=ps.get_expert_model_parallel_world_size(),
        world_size=torch.distributed.get_world_size(), groups_origin="runtime")
    require(all(groups[k] == 1 for k in ("tp", "pp", "cp", "dp", "ep", "world_size")),
            "unknown distributed loss capture mapping")
    return groups


def scalar(value, name):
    import torch
    require(isinstance(value, torch.Tensor) and value.numel() == 1, name + " must be one real tensor scalar")
    result = float(value.detach().item())
    require(number(result), name + " is nonfinite")
    return result


def identity_rows(ids, mask, iteration, offset):
    prompts = []
    for tokens, row in zip(ids, mask):
        require(1 in row, "empty generated response")
        prefix = tokens[:row.index(1)]
        prompts.append(hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest())
    return dict(trajectory_ids=[f"iteration-{iteration}-capture-{offset+i}" for i in range(len(ids))],
                prompt_ids=prompts, group_ids=prompts)


class LossRecordSink:
    """Serializable, one-rank file sink; refuses reruns and unexpected batch counts."""
    def __init__(self, directory, *, role, config, versions, delegate, max_records,
                 group_reader=runtime_groups, parameter=None):
        require(role in ("actor", "critic"), "unknown loss capture role")
        self.directory = str(directory)
        self.role, self.config, self.versions = role, config, versions
        self.delegate, self.max_records, self.group_reader = delegate, max_records, group_reader
        self.sequence = 0
        self.parameter = parameter

    def __call__(self, args, kwargs, result):
        import torch
        require(not kwargs and len(args) == 4, "unknown loss call signature")
        output, data, valid_seqs, valid_toks = args
        b, s = bounded_shape(data.get("input_ids"))
        require(b == 1 and self.sequence < self.max_records, "unexpected microbatch or capture rerun")
        require(isinstance(result, tuple) and len(result) == 2, "unknown loss return")
        ids = copy_rows(data["input_ids"])
        raw_mask = tensor(data.get("token_mask"), (b, s), "token mask")
        require(bool(((raw_mask == 0) | (raw_mask == 1)).all()), "nonbinary token mask")
        mask = copy_rows(raw_mask.to(torch.int64))
        sample = tensor(data.get("sample_mask"), (b,), "sample mask")
        require(bool((sample == 1).all()), "filtered samples need a separate mapping")
        for row in mask:
            require(row[0] == 0 and all(x in (0, 1) for x in row), "invalid action mask")
        ids_meta = identity_rows(ids, mask, self.versions["after"], self.sequence)
        if self.role == "actor":
            canonical, measured = nemo_action_slice(output, data, config=self.config, mapping="nemo_full_token_v1",
                versions=self.versions, identities=ids_meta)
        else:
            canonical = dict(input_ids=ids, response_mask=mask, sample_mask=[1]*b,
                             alignment="action_position", policy_versions=self.versions, **ids_meta)
            if output.ndim == 3 and output.shape[-1] == 1:
                output = output.squeeze(-1)
            measured = dict(values=copy_rows(tensor(output, (b, s), "value prediction")),
                old_values=copy_rows(tensor(data.get("values"), (b, s), "old values")),
                returns=copy_rows(tensor(data.get("returns"), (b, s), "returns")))
        record = dict(role=self.role, data=canonical, measurements=measured,
            official_loss=scalar(result[0], "official loss"),
            global_valid_sequences=scalar(valid_seqs, "global sequence normalizer"),
            global_valid_tokens=scalar(valid_toks, "global token normalizer"),
            parallel=self.group_reader(),
            source=callable_source(self.delegate.__call__, "nemo_"+self.role+"_loss"),
            differentiable_output=bool(result[0].requires_grad),
            identity_origin="capture-local occurrence IDs; prompt group is a token-prefix SHA256")
        require(record["global_valid_sequences"] > 0 and record["global_valid_tokens"] > 0,
                "nonpositive global normalizer")
        finite_tree(record)
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        require(len(encoded.encode()) <= MAX_BYTES, "loss capture exceeds 1 MiB")
        path = Path(self.directory)/f"{self.sequence:03d}.json"
        with path.open("x") as stream:
            stream.write(encoded)
        self.sequence += 1


def read_records(directory, expected):
    paths = sorted(Path(directory).glob("*.json"))
    require(len(paths) == expected, "missing, duplicate, or rerun loss captures")
    records = []
    for i, path in enumerate(paths):
        require(path.name == f"{i:03d}.json" and path.stat().st_size <= MAX_BYTES,
                "unexpected capture file")
        records.append(strict_json(path.read_text()))
    return records


def concatenate(records):
    require(records, "no loss records")
    first = records[0]
    require(all(r["role"] == first["role"] and r["parallel"] == first["parallel"]
                and r["data"]["policy_versions"] == first["data"]["policy_versions"]
                for r in records), "loss workers disagree on role/groups/version")
    data = dict(first["data"])
    for key in ("input_ids", "response_mask", "sample_mask", "trajectory_ids", "prompt_ids", "group_ids"):
        data[key] = [row for r in records for row in r["data"][key]]
    measured = {}
    for key, value in first["measurements"].items():
        measured[key] = (None if value is None else
                         [row for r in records for row in r["measurements"][key]])
    b, s = len(data["input_ids"]), len(data["input_ids"][0])
    require(1 <= b <= 8 and 2 <= s <= 512 and all(len(row) == s for row in data["input_ids"]),
            "unknown ragged/global batch capture")
    count = sum(map(sum, data["response_mask"]))
    require(all(r["global_valid_sequences"] == b and r["global_valid_tokens"] == count for r in records),
            "local masks disagree with the official global normalizers")
    return data, measured


def reduced(rows, mask, reduction):
    sums = [sum(x*m for x, m in zip(row, active)) for row, active in zip(rows, mask)]
    if reduction == "token":
        return sum(sums)/sum(map(sum, mask))
    return sum(v/sum(active) for v, active in zip(sums, mask))/len(mask)


def actor_terms(data, measured, config):
    """Diagnostic arithmetic only; these values are never fed to the trainer."""
    mask, ratio, pg, penalties = data["response_mask"], [], [], []
    previous, current = measured["previous_logprobs"], measured["current_logprobs"]
    for i, row in enumerate(current):
        ratios, losses, kl = [], [], []
        for j, lp in enumerate(row):
            delta = lp-previous[i][j]
            require(abs(delta) <= 50, "invalid policy ratio range")
            r = 1.0 if config["force_on_policy"] else math.exp(delta)
            a = measured["advantages"][i][j]
            ratios.append(r)
            losses.append(max(-a*r, -a*min(1+config["ratio_clip"], max(1-config["ratio_clip"], r))))
            d = 0.0 if not config["kl_beta"] else measured["reference_logprobs"][i][j]-lp
            d = min(config["kl_input_clamp"], max(-config["kl_input_clamp"], d))
            kl.append(config["kl_beta"]*min(config["kl_output_clamp"], math.exp(d)-1-d))
        ratio.append(ratios); pg.append(losses); penalties.append(kl)
    actor, kl = reduced(pg, mask, config["reduction"]), reduced(penalties, mask, config["reduction"])
    return dict(ratio=ratio, pg_token=pg, actor_loss=actor, kl_loss=kl, policy_loss=actor+kl)


def value_term(data, measured, config):
    mask = data["response_mask"]
    b, s = len(mask), len(mask[0])
    for key in ("values", "old_values", "returns"):
        matrix(measured.get(key), b, s, key)
    total = 0.0
    for i, row in enumerate(mask):
        for j, active in enumerate(row):
            value, old, target = (measured[k][i][j] for k in ("values", "old_values", "returns"))
            bounded = min(old+config["value_clip"], max(old-config["value_clip"], value))
            total += active * .5 * max((value-target)**2, (bounded-target)**2)
    return total/sum(map(sum, mask))


def compare_official_loss(records, expected):
    actual = sum(r["official_loss"] for r in records)
    # The upstream masked_mean adds 1e-8 to its global divisor, and GPU loss
    # arithmetic is FP32. Keep this explicit; never reuse BF16 logprob tolerances.
    require(number(actual) and abs(actual-expected) <= 1e-5 + 1e-5*abs(expected),
            "captured loss differs from the official globally normalized result")
    return dict(value=actual, diagnostic_value=expected, atol=1e-5, rtol=1e-5,
                meaning="sum of actual loss outputs before backward; diagnostics recomputed from captured values")
