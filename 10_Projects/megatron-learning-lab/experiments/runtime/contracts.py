"""Bounded, read-only trace contracts. JSON metadata is a claim, not attestation.

No framework imports, shell execution, dynamic imports, URL loading or pickle.
The browser independently validates this envelope before rendering it.
"""
from __future__ import annotations
import hashlib
import json
import math
import re

SCHEMA = "megatron-learning-lab.trace"
MAX_BYTES = 1_048_576
MAX_BATCH = 8
MAX_SEQUENCE = 512
PROVENANCE = {"derived", "reference", "observed_bridge", "observed_rl"}
ADAPTERS = {"authored_cpu_v1", "hf_next_token_v1", "bridge_bsh_v1", "bridge_sbh_v1", "nemo_full_token_v1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    return type(value) is int


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finite_tree(value, depth=0):
    require(depth <= 32, "JSON nesting exceeds limit")
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) in (int, float):
        require(math.isfinite(value), "nonfinite numeric value")
    elif isinstance(value, list):
        require(len(value) <= 65536, "array exceeds limit")
        for item in value:
            finite_tree(item, depth + 1)
    elif isinstance(value, dict):
        require(len(value) <= 256, "object exceeds limit")
        for key, item in value.items():
            require(isinstance(key, str) and key not in ("__proto__", "prototype", "constructor"), "unsafe JSON key")
            finite_tree(item, depth + 1)
    else:
        raise ValueError("unsupported JSON value")


def strict_json(text):
    require(isinstance(text, str) and len(text.encode("utf-8")) <= MAX_BYTES, "trace exceeds 1 MiB or is not text")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid JSON") from exc
    finite_tree(value)
    return value


def hashed_json(trace, key):
    text = trace.get(key + "_json")
    value = strict_json(text)
    require(trace.get(key + "_sha256") == digest_text(text), key + " SHA256 mismatch")
    require(isinstance(value, dict), key + " must encode an object")
    return value


def matrix(value, b, s, name, predicate=number):
    require(isinstance(value, list) and len(value) == b, name + " batch shape mismatch")
    for row in value:
        require(isinstance(row, list) and len(row) == s, name + " sequence shape mismatch")
        require(all(predicate(v) for v in row), name + " invalid element")
    return value


def close(a, b, name, atol=1e-7, rtol=1e-6):
    require(number(a) and number(b) and abs(a-b) <= atol + rtol*abs(b), name + " semantic mismatch")


def validate_identity(manifest, provenance):
    require(isinstance(manifest, dict), "manifest missing")
    require(manifest.get("adapter") in ADAPTERS, "unknown adapter/version mapping")
    require(manifest.get("dtype") in ("float64", "float32", "bfloat16", "float16"), "unknown dtype")
    require(manifest.get("layout") == "BS", "trace must use canonical BS layout")
    require(isinstance(manifest.get("backend"), str) and manifest["backend"], "backend missing")
    for key in ("model", "tokenizer"):
        identity = manifest.get(key)
        require(isinstance(identity, dict) and isinstance(identity.get("id"), str) and identity["id"], key + " identity missing")
        revision = identity.get("revision")
        require(isinstance(revision, str) and revision, key + " revision missing")
        if provenance.startswith("observed") or manifest.get("evidence_kind") == "hf_runtime":
            require(re.fullmatch(r"[a-f0-9]{40,64}", revision) is not None, key + " immutable revision required")
    require(re.fullmatch(r"[a-f0-9]{64}", manifest["tokenizer"].get("chat_template_sha256", "")) is not None,
            "chat template hash missing")
    require(manifest["model"].get("weights_origin") in ("authored", "hf_checkpoint", "training_checkpoint", "random_initialized"),
            "weights origin missing")
    require(manifest["model"].get("architecture_origin") in ("authored_scaled", "hf_config"), "architecture origin missing")
    group = manifest.get("parallel")
    require(isinstance(group, dict), "parallel groups missing")
    for key in ("tp", "pp", "dp", "cp", "ep", "world_size"):
        require(integer(group.get(key)) and group[key] > 0, "invalid parallel dimension: " + key)
    require(group["tp"]*group["pp"]*group["dp"]*group["cp"] == group["world_size"], "parallel world size mismatch")
    require(group["world_size"] % group["ep"] == 0, "EP does not divide world size")
    require(group.get("groups_origin") in ("runtime", "in_memory_reference"), "parallel group origin missing")
    require(isinstance(manifest.get("software"), dict) and manifest["software"].get("python"), "software versions missing")
    sources = manifest.get("runtime_sources")
    require(isinstance(sources, list) and sources, "runtime source mapping missing")
    for item in sources:
        require(isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k] for k in ("component", "path", "symbol")), "invalid runtime source")
        require(re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")) is not None, "runtime source hash missing")
    capture = manifest.get("capture")
    require(isinstance(capture, dict) and capture.get("timing") == "not_measured", "this trace schema does not accept unqualified timing")
    require(capture.get("scope") == "selected_slices", "capture scope must be explicit and bounded")
    require(isinstance(capture.get("unobserved"), list), "unobserved fields missing")
    execution = manifest.get("execution")
    require(isinstance(execution, dict), "execution metadata missing")
    evidence = manifest.get("evidence_kind")
    if provenance == "derived":
        require(execution.get("status") == "not_run" and evidence == "formula", "derived cannot claim execution")
    elif provenance == "reference":
        require(evidence in ("authored_cpu", "synthetic_contract", "hf_runtime"), "wrong reference provenance")
        require(execution.get("status") == ("synthetic" if evidence == "synthetic_contract" else "executed"), "reference execution status mismatch")
    else:
        backend, kind, adapter_prefix = (
            ("megatron-bridge", "bridge_runtime", "bridge_") if provenance == "observed_bridge"
            else ("nemo-rl", "rl_runtime", "nemo_"))
        require(manifest["backend"] == backend and evidence == kind and manifest["adapter"].startswith(adapter_prefix),
                "observed provenance/backend/adapter mismatch")
        require(execution.get("status") == "executed" and execution.get("synthetic") is False, "synthetic or unexecuted data cannot be observed")
        require(group["groups_origin"] == "runtime", "observed requires actual groups")
        require(manifest["model"]["weights_origin"] in ("hf_checkpoint", "training_checkpoint"), "observed profile requires real checkpoint")
        require(isinstance(execution.get("command"), list) and execution["command"], "actual argv missing")
        require(execution.get("started_at") and execution.get("completed_at"), "actual run timestamps missing")
    require(isinstance(manifest.get("limitations"), list), "limitations missing")


def validate_input(data, config, task):
    ids = data.get("input_ids")
    require(isinstance(ids, list) and 1 <= len(ids) <= MAX_BATCH, "invalid input batch")
    b = len(ids)
    require(isinstance(ids[0], list) and 2 <= len(ids[0]) <= MAX_SEQUENCE, "invalid input length")
    s = len(ids[0])
    vocab = config.get("vocab_size")
    require(integer(vocab) and vocab > 1, "vocab size missing")
    matrix(ids, b, s, "input_ids", lambda v: integer(v) and 0 <= v < vocab)
    mask = matrix(data.get("loss_mask" if task == "sft" else "response_mask"), b, s, "mask", lambda v: integer(v) and v in (0, 1))
    require(sum(map(sum, mask)) > 0, "global mask is empty")
    if task == "sft":
        require(data.get("alignment") == "next_token", "unknown SFT shift/mapping")
        labels = matrix(data.get("labels"), b, s, "labels", lambda v: integer(v) and (v == -100 or 0 <= v < vocab))
        matrix(data.get("position_ids"), b, s, "position_ids", lambda v: integer(v) and v >= 0)
        docs = matrix(data.get("document_ids"), b, s, "document_ids", lambda v: integer(v) and v >= -1)
        for i in range(b):
            require(mask[i][-1] == 0, "last input has no next-token target")
            for j in range(s-1):
                if mask[i][j]:
                    require(labels[i][j] == ids[i][j+1], "double/missing shift or wrong label")
                    require(docs[i][j] == docs[i][j+1] and docs[i][j] >= 0, "cross-document target leakage")
    else:
        require(data.get("alignment") == "action_position" and all(row[0] == 0 for row in mask), "RL action-position alignment mismatch")
        for key in ("trajectory_ids", "prompt_ids", "group_ids"):
            require(isinstance(data.get(key), list) and len(data[key]) == b and all(isinstance(x, str) and x for x in data[key]), key + " missing")
        require(len(set(data["trajectory_ids"])) == b, "duplicate trajectory id")
        require(all(sum(row) > 0 for row in mask), "empty RL response")
        sample = data.get("sample_mask")
        require(sample == [1]*b and all(integer(v) for v in sample), "this profile supports only retained samples; filtered batches need explicit adapter")
        versions = data.get("policy_versions")
        require(isinstance(versions, dict) and all(integer(versions.get(k)) and versions[k] >= 0 for k in ("generation", "previous", "current", "after")), "policy versions missing")
        require(versions["after"] == versions["current"] + 1, "one-step policy version mismatch")
    return b, s, mask


def _validate_trace(trace):
    require(isinstance(trace, dict) and trace.get("schema") == SCHEMA and trace.get("schema_version") == 1,
            "unknown trace schema/version")
    finite_tree(trace)
    require(isinstance(trace.get("run_id"), str) and 1 <= len(trace["run_id"]) <= 128, "run id missing")
    task = trace.get("task")
    require(task in ("sft", "rl"), "unsupported trace task")
    provenance = trace.get("provenance")
    require(provenance in PROVENANCE, "unknown provenance")
    require((provenance != "observed_bridge" or task == "sft") and (provenance != "observed_rl" or task == "rl"), "provenance/task mismatch")
    validate_identity(trace.get("manifest"), provenance)
    config, data = hashed_json(trace, "config"), hashed_json(trace, "input")
    b, s, mask = validate_input(data, config, task)
    m = trace.get("measurements")
    if provenance == "derived":
        require(m is None, "derived traces cannot carry measured numbers")
        return dict(status="valid", trust="imported_claim", task=task, provenance=provenance, config=config, data=data)
    require(isinstance(m, dict), "measurements missing")
    count = sum(map(sum, mask))
    if task == "sft":
        lp = matrix(m.get("token_logprobs"), b, s, "token_logprobs", lambda v: number(v) and v <= 1e-6)
        total = sum(-v*mask[i][j] for i, row in enumerate(lp) for j, v in enumerate(row))
        require(integer(m.get("token_count")) and m["token_count"] == count, "loss token count mismatch")
        close(m.get("loss_sum"), total, "loss sum")
        close(m.get("loss_mean"), total/count, "loss reduction")
    else:
        algorithm = config.get("algorithm")
        require(algorithm in ("grpo", "ppo"), "unsupported RL algorithm")
        require(config.get("loss_variant") == "clipped_pg_k3" and config.get("offpolicy_correction") is False,
                "unknown RL loss mapping")
        reduction = config.get("reduction")
        require(reduction in ("token", "sequence"), "unknown RL reduction")
        clip = config.get("ratio_clip")
        beta = config.get("kl_beta")
        require(number(clip) and 0 < clip < 1 and number(beta) and beta >= 0, "invalid RL loss parameters")
        prev = matrix(m.get("previous_logprobs"), b, s, "previous_logprobs", lambda v: number(v) and v <= 1e-6)
        curr = matrix(m.get("current_logprobs"), b, s, "current_logprobs", lambda v: number(v) and v <= 1e-6)
        matrix(m.get("generation_logprobs"), b, s, "generation_logprobs", lambda v: number(v) and v <= 1e-6)
        adv = matrix(m.get("advantages"), b, s, "advantages")
        ratio = matrix(m.get("ratio"), b, s, "ratio")
        pg = matrix(m.get("pg_token"), b, s, "pg_token")
        if beta:
            ref = matrix(m.get("reference_logprobs"), b, s, "reference_logprobs", lambda v: number(v) and v <= 1e-6)
        else:
            require(m.get("reference_logprobs") is None, "disabled reference must be explicitly absent")
        force = config.get("force_on_policy")
        require(type(force) is bool, "force-on-policy flag missing")
        if force:
            require(data["policy_versions"]["generation"] == data["policy_versions"]["previous"] == data["policy_versions"]["current"],
                    "force-on-policy batch version mismatch")
        for i in range(b):
            for j in range(s):
                require(abs(curr[i][j]-prev[i][j]) <= 50, "logprob ratio overflow/invalid mapping")
                expected = 1.0 if force else math.exp(curr[i][j]-prev[i][j])
                close(ratio[i][j], expected, "policy ratio")
                close(pg[i][j], max(-adv[i][j]*expected, -adv[i][j]*min(1+clip, max(1-clip, expected))), "PG clipping")
                if not mask[i][j]:
                    close(adv[i][j], 0, "masked advantage")
        sums = [sum(v*mask[i][j] for j,v in enumerate(row)) for i,row in enumerate(pg)]
        loss = sum(sums)/count if reduction == "token" else sum(v/sum(mask[i]) for i,v in enumerate(sums))/b
        close(m.get("actor_loss"), loss, "actor reduction")
        require(number(m.get("kl_loss")) and m["kl_loss"] >= 0, "invalid KL loss")
        if beta == 0:
            close(m["kl_loss"], 0, "disabled KL")
        else:
            bound, cap = config.get("kl_input_clamp"), config.get("kl_output_clamp")
            require(number(bound) and 0 < bound <= 50 and number(cap) and cap > 0, "KL clamp configuration missing")
            require(config.get("kl_sampling") == "non_is_score_gradient", "unknown KL sampling mapping")
            penalties = []
            for i in range(b):
                row = []
                for j in range(s):
                    d = min(bound, max(-bound, ref[i][j]-curr[i][j]))
                    row.append(min(cap, math.exp(d)-1-d) * mask[i][j])
                penalties.append(sum(row))
            expected_kl = sum(penalties)/count if reduction == "token" else sum(v/sum(mask[i]) for i,v in enumerate(penalties))/b
            close(m["kl_loss"], beta*expected_kl, "KL penalty")
        close(m.get("policy_loss"), loss+m["kl_loss"], "policy loss")
        if algorithm == "ppo":
            values = matrix(m.get("values"), b, s, "values")
            old = matrix(m.get("old_values"), b, s, "old_values")
            returns = matrix(m.get("returns"), b, s, "returns")
            value_clip = config.get("value_clip")
            require(number(value_clip) and value_clip > 0, "value clip missing")
            total = 0.0
            for i in range(b):
                for j in range(s):
                    v = values[i][j]
                    limited = min(old[i][j]+value_clip, max(old[i][j]-value_clip, v))
                    total += 0.5*mask[i][j]*max((v-returns[i][j])**2, (limited-returns[i][j])**2)
            close(m.get("value_loss"), total/count, "value loss")

        else:
            require(m.get("values") is None and m.get("returns") is None and m.get("value_loss") is None, "GRPO must not claim critic measurements")
        refit = m.get("refit")
        require(isinstance(refit, dict) and refit.get("status") in ("not_run", "synchronized"), "refit state missing")
        if refit["status"] == "synchronized":
            require(refit.get("completed") is True and refit.get("generation_version") == data["policy_versions"]["after"], "refit version/ack mismatch")
            require(re.fullmatch(r"[a-f0-9]{64}", refit.get("export_hash", "")) is not None and refit["export_hash"] == refit.get("ack_hash"),
                    "refit weight acknowledgment mismatch")
    return dict(status="valid", trust="imported_claim", task=task, provenance=provenance,
                config=config, data=data, token_count=count)


def validate_trace(trace):
    try:
        return _validate_trace(trace)
    except (TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError("malformed trace fields") from exc


def read_trace(text):
    return validate_trace(strict_json(text))
