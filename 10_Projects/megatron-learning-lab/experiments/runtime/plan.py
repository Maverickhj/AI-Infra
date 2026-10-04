"""Read-only runtime plans and explicit resource grants; no framework imports.

A plan is data, never an executable command. Source inspection is not behavioral
compatibility evidence. Authorizing a JSON file does not replace user consent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

from .contracts import MAX_BYTES, finite_tree, integer, require, strict_json
from .source_probe import anchor, inspect_nemo_cli

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {"hf_reference", "bridge_sft", "rl_grpo", "rl_ppo"}
REVISION = re.compile(r"[a-f0-9]{40,64}")
DEVICE = re.compile(r"(?:[0-9]+|GPU-[a-fA-F0-9-]+)")


def canonical(value):
    finite_tree(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_document(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_BYTES, "plan/resource document exceeds 1 MiB")
    value = strict_json(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "plan/resource document must be an object")
    return value


def absolute_path(value, name):
    require(isinstance(value, str) and value and "\x00" not in value,
            name + " must be a path")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts,
            name + " must be an absolute path without traversal")
    return path.resolve()


def _keys(value, allowed, name):
    require(isinstance(value, dict), name + " must be an object")
    require(set(value) == set(allowed), name + " missing or unknown fields")


def validate_plan(plan):
    """Validate the narrow, local, one-rank first runtime profile."""
    require(isinstance(plan, dict), "plan must be an object")
    extra = {"critic"} if plan.get("profile") == "rl_ppo" else set()
    _keys(plan, {"schema", "schema_version", "profile", "run_id", "model",
                 "tokenizer", "data", "execution", "sources", "training",
                 "capture", "tolerances"} | extra, "plan")
    require(plan["schema"] == "megatron-learning-lab.runtime-plan"
            and type(plan["schema_version"]) is int and plan["schema_version"] == 1,
            "unknown runtime plan version")
    require(plan["profile"] in PROFILES, "unknown runtime profile")
    require(isinstance(plan["run_id"], str)
            and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", plan["run_id"]),
            "invalid run_id")
    for name in ("model", "tokenizer") + (("critic",) if extra else ()):
        obj = plan[name]
        _keys(obj, {"id", "revision", "snapshot"}, name)
        require(isinstance(obj["id"], str) and obj["id"], name + " id missing")
        require(isinstance(obj["revision"], str) and REVISION.fullmatch(obj["revision"]),
                name + " immutable revision missing")
        absolute_path(obj["snapshot"], name + ".snapshot")
    _keys(plan["data"], {"path", "sha256", "mask_mode", "mapping"}, "data")
    absolute_path(plan["data"]["path"], "data.path")
    require(isinstance(plan["data"]["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", plan["data"]["sha256"]),
            "input file SHA256 missing")
    require(plan["data"]["mask_mode"] in ("assistant", "last_turn", "full"),
            "unknown mask mode")
    expected_mapping = "nemo_response_jsonl_v1" if plan["profile"].startswith("rl_") else "qwen_im_chat_v1"
    require(plan["data"]["mapping"] == expected_mapping, "unknown tokenizer mapping")
    if plan["profile"].startswith("rl_"):
        require(plan["data"]["mask_mode"] == "assistant", "RL masks come from generated assistant actions")
    execution = plan["execution"]
    _keys(execution, {"location", "devices", "max_wall_seconds", "output_path",
                     "allow_model_download", "allow_remote_jobs"}, "execution")
    require(execution["location"] == "local", "remote execution is not implemented")
    devices = execution["devices"]
    require(isinstance(devices, list) and len(devices) == 1
            and isinstance(devices[0], str) and DEVICE.fullmatch(devices[0]),
            "this runtime profile requires exactly one explicit GPU")
    require(integer(execution["max_wall_seconds"])
            and 1 <= execution["max_wall_seconds"] <= 86400, "invalid wall time bound")
    output = absolute_path(execution["output_path"], "execution.output_path")
    require(output.name == plan["run_id"], "output directory must end in run_id")
    require(execution["allow_model_download"] is False
            and execution["allow_remote_jobs"] is False,
            "this profile runs local snapshots only; downloads/jobs require a separate action")
    _keys(plan["sources"], {"bridge", "nemo_rl"}, "sources")
    for name in ("bridge", "nemo_rl"):
        if plan["sources"][name] is not None:
            absolute_path(plan["sources"][name], "sources." + name)
    training = plan["training"]
    _keys(training, {"steps", "sequence_length", "micro_batch_size", "global_batch_size",
                     "dtype", "learning_rate", "seed", "resume_from",
                     "nemo_config", "nemo_overrides"} | ({"nemo_config_sha256"} if plan["profile"].startswith("rl_") else set()), "training")
    require(integer(training["steps"]) and 1 <= training["steps"] <= 100, "invalid steps")
    require(integer(training["sequence_length"])
            and 8 <= training["sequence_length"] <= 511, "invalid sequence capture bound")
    require(type(training["micro_batch_size"]) is int and training["micro_batch_size"] == 1,
            "initial capture requires micro batch size one")
    global_batch = training["global_batch_size"]
    require(integer(global_batch) and (2 <= global_batch <= 8 if plan["profile"].startswith("rl_") else global_batch == 1),
            "invalid global batch bound for profile")
    require(training["dtype"] in ("float32", "bfloat16"), "unsupported runtime dtype")
    require(type(training["learning_rate"]) in (int, float)
            and 0 < training["learning_rate"] <= 1, "invalid learning rate")
    require(integer(training["seed"]) and 0 <= training["seed"] < 2**31, "invalid seed")
    if training["resume_from"] is not None:
        absolute_path(training["resume_from"], "resume_from")
    if plan["profile"].startswith("rl_"):
        require(training["nemo_config"] is not None and plan["sources"]["nemo_rl"] is not None,
                "RL source and effective config are required")
        absolute_path(training["nemo_config"], "nemo_config")
        require(training["nemo_overrides"] == [], "freeze all NeMo overrides before authorizing the plan")
        require(isinstance(training["nemo_config_sha256"], str)
                and re.fullmatch(r"[a-f0-9]{64}", training["nemo_config_sha256"]),
                "frozen NeMo config SHA256 missing")
    else:
        require(training["nemo_config"] is None and training["nemo_overrides"] == [],
                "NeMo settings are not applicable to this profile")
    if plan["profile"] == "bridge_sft":
        require(plan["sources"]["bridge"] is not None, "Bridge source path required")
    _keys(plan["capture"], {"max_records", "parameter_name", "parameter_index"}, "capture")
    capture = plan["capture"]
    require(integer(capture["max_records"]) and 1 <= capture["max_records"] <= 64,
            "invalid capture record bound")
    require(isinstance(capture["parameter_name"], str) and capture["parameter_name"],
            "explicit parameter_name required")
    require(isinstance(capture["parameter_index"], list)
            and capture["parameter_index"] and all(integer(x) and x >= 0 for x in capture["parameter_index"]),
            "invalid parameter slice")
    _keys(plan["tolerances"], {"logprob_atol", "logprob_rtol", "activation_atol",
                              "activation_rtol", "resume_atol", "resume_rtol"}, "tolerances")
    for name, value in plan["tolerances"].items():
        require(type(value) in (int, float) and 0 <= value < 1,
                "invalid predeclared tolerance: " + name)
    if plan["profile"] == "bridge_sft":
        require(capture["max_records"] >= training["steps"], "capture limit cannot cover requested steps")
    finite_tree(plan)
    return plan


def inspect_plan(plan):
    """Read only explicitly named small files and source symbols."""
    validate_plan(plan)
    issues, files, sources = [], [], []
    runtime_config = None
    def small_file(path, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                issues.append("missing file: " + str(path))
            return None
        if path.stat().st_size > MAX_BYTES:
            issues.append("metadata file exceeds 1 MiB: " + str(path))
            return None
        raw = path.read_bytes()
        files.append(dict(path=str(path.resolve()), bytes=len(raw),
                          sha256=hashlib.sha256(raw).hexdigest()))
        return raw
    data_path = Path(plan["data"]["path"])
    raw = small_file(data_path)
    if raw is not None and hashlib.sha256(raw).hexdigest() != plan["data"]["sha256"]:
        issues.append("input file SHA256 mismatch")
    model_root = Path(plan["model"]["snapshot"])
    small_file(model_root / "config.json")
    tokenizer_root = Path(plan["tokenizer"]["snapshot"])
    small_file(tokenizer_root / "tokenizer_config.json")
    # tokenizer.json/weights may be large. They are never read in a dry-run.
    if not (tokenizer_root / "tokenizer.json").is_file():
        issues.append("missing fast tokenizer: " + str(tokenizer_root / "tokenizer.json"))
    template = tokenizer_root / "chat_template.jinja"
    if template.is_file():
        small_file(template)
    has_weights = (model_root / "model.safetensors").is_file()
    index = model_root / "model.safetensors.index.json"
    if index.is_file():
        has_weights = True
        raw = small_file(index)
        if raw is not None:
            try:
                def unique_pairs(pairs):
                    value={}
                    for key,item in pairs:
                        require(key not in value,"duplicate weight index key")
                        value[key]=item
                    return value
                # A real full-model index can have more than 256 parameter
                # keys. This metadata reader stays byte-bounded and validates
                # shard basenames; it does not relax the trace object limit.
                mapping = json.loads(raw.decode("utf-8"),object_pairs_hook=unique_pairs)["weight_map"]
                require(isinstance(mapping, dict), "invalid safetensors index")
                for name in set(mapping.values()):
                    require(isinstance(name, str) and Path(name).name == name
                            and name.endswith(".safetensors"), "invalid shard path")
                    if not (model_root / name).is_file():
                        issues.append("missing weight shard: " + name)
            except (ValueError, KeyError, TypeError) as exc:
                issues.append("invalid model index: " + str(exc))
    if not has_weights:
        issues.append("local safetensors weights missing")
    if plan["profile"] == "bridge_sft":
        base = Path(plan["sources"]["bridge"]) / "src/megatron/bridge"
        symbols = [
            (base/"models/conversion/auto_bridge.py", "AutoBridge.from_hf_config", {"config"}),
            (base/"models/conversion/auto_bridge.py", "AutoBridge.to_megatron_provider", {"load_weights", "hf_path"}),
            (base/"training/finetune.py", "finetune", {"config", "forward_step_func", "callbacks"}),
            (base/"training/gpt_step.py", "forward_step", {"state", "data_iterator", "model"}),
            (base/"training/config.py", "DatasetProvider.build_datasets", {"context"}),
        ]
        for path, symbol, expected in symbols:
            try:
                result = anchor(path, symbol)
                require(expected <= set(result.get("parameters", [])), "signature mismatch: " + symbol)
                sources.append(result)
            except (ValueError, OSError, SyntaxError) as exc:
                issues.append(str(exc))
    elif plan["profile"].startswith("rl_"):
        try:
            observed = inspect_nemo_cli(plan["sources"]["nemo_rl"], plan["profile"][3:])
            sources.append(observed["source"])
        except (ValueError, OSError, SyntaxError) as exc:
            issues.append(str(exc))
    if plan["profile"].startswith("rl_"):
        try:
            from .nemo_config import read_bound_config, response_rows
            _, record, runtime_config = read_bound_config(plan)
            files.append(record)
            require(len(response_rows(data_path)) >= runtime_config["min_dataset_rows"],
                    "local dataset cannot supply all planned iterations")
        except (ValueError, OSError) as exc:
            issues.append(str(exc))
        if plan["profile"] == "rl_ppo":
            critic = Path(plan["critic"]["snapshot"])
            small_file(critic / "config.json")
            # The first critic profile deliberately accepts only one local
            # safetensors file; unknown sharded critics fail instead of guessing.
            if not (critic / "model.safetensors").is_file():
                issues.append("local single-file critic checkpoint missing")
    resume = plan["training"]["resume_from"]
    if resume is not None and not Path(resume).is_dir():
        issues.append("resume checkpoint directory missing")
    return dict(status="not_ready" if issues else "configuration_ready",
                issues=issues, files=files, runtime_sources=sources, runtime_config=runtime_config,
                compatibility="not_checked", behavior="not_run")


def check_grant(plan, resources):
    """Require a user-supplied, exact-plan-bound grant before any runtime work."""
    validate_plan(plan)
    require(isinstance(resources, dict), "resource grant missing")
    key = "rl_training" if plan["profile"].startswith("rl_") else "bridge_sft"
    require(isinstance(resources.get("grants"), dict), "resource grants missing")
    grant = resources["grants"].get(key, {})
    require(isinstance(grant, dict), "invalid resource grant")
    require(grant.get("authorized") is True, "resource grant not authorized: " + key)
    require(isinstance(grant.get("authorization_ref"), str) and grant["authorization_ref"].strip(),
            "user authorization reference missing")
    require(grant.get("plan_sha256") == digest(plan), "resource grant belongs to a different plan")
    execution = resources.get("execution", {})
    require(isinstance(execution, dict), "invalid resource execution limits")
    require(execution.get("location") == plan["execution"]["location"], "location exceeds grant")
    devices = execution.get("devices")
    require(isinstance(devices, list) and all(x in devices for x in plan["execution"]["devices"]),
            "device exceeds grant")
    require(integer(execution.get("max_gpu_count")) and execution["max_gpu_count"] >= 1,
            "GPU count exceeds grant")
    for field, used in (("max_wall_seconds", plan["execution"]["max_wall_seconds"]),
                        ("max_steps", plan["training"]["steps"])):
        require(integer(execution.get(field)) and used <= execution[field],
                field + " exceeds grant")
    for field, used in (("model_snapshot", plan["model"]["snapshot"]),
                       ("tokenizer_snapshot", plan["tokenizer"]["snapshot"]),
                       ("data_path", plan["data"]["path"]),
                       ("output_path", plan["execution"]["output_path"])):
        require(absolute_path(execution.get(field), field) == absolute_path(used, field),
                field + " differs from grant")
    if plan["profile"] == "rl_ppo":
        require(absolute_path(execution.get("critic_snapshot"), "critic_snapshot")
                == absolute_path(plan["critic"]["snapshot"], "critic.snapshot"),
                "critic snapshot differs from grant")
        require(execution.get("allow_critic_training") is True, "separate critic training authorization missing")
    require(execution.get("allow_model_download") is False
            and execution.get("allow_remote_jobs") is False, "grant must bind the local-only profile")
    # The declared plan remains untouched; this check cannot grant permission.
    return dict(status="authorized_manifest", grant=key,
                authorization_ref=grant["authorization_ref"], plan_sha256=digest(plan),
                limitation="metadata binding, not independent verification of human consent")


def dry_run(plan_path, resources_path=None):
    path = Path(plan_path).resolve()
    plan = validate_plan(read_document(path))
    observed = inspect_plan(plan)
    command = [sys.executable, str(ROOT/"tools/runtime_cli.py"), "run",
               "--plan", str(path), "--resources",
               str(Path(resources_path).resolve()) if resources_path else "/absolute/path/to/resources.json"]
    grant = dict(status="not_authorized")
    if resources_path is not None:
        try:
            grant = check_grant(plan, read_document(resources_path))
        except (ValueError, OSError) as exc:
            grant = dict(status="not_authorized", reason=str(exc))
    configuration_command = None
    if plan["profile"] == "bridge_sft":
        configuration_command = [sys.executable, str(ROOT/"tools/runtime_cli.py"),
                                 "resolve-bridge", "--plan", str(path)]
    elif plan["profile"].startswith("rl_"):
        configuration_command = [sys.executable, str(ROOT/"tools/runtime_cli.py"),
            "inspect-nemo-config", "--plan", str(path)]
    return dict(schema="megatron-learning-lab.runtime-dry-run", plan_sha256=digest(plan),
                plan=plan, inspection=observed, resources=grant, command=command,
                configuration_command=configuration_command,
                command_text=shlex.join(command), execution="not_run",
                boundary="No launcher/framework import, download, model construction, GPU or cluster initialization")
