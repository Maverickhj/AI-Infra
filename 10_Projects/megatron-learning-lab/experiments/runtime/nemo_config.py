"""Finite, local NeMo execution profiles bound to resolved configuration bytes.

Configuration inspection imports no Torch, Ray, NeMo, model, or launcher.
This is a software prerequisite, never evidence that a runtime is compatible.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .contracts import MAX_BYTES, finite_tree, integer, number, require, strict_json


def same_path(actual, expected, label):
    require(isinstance(actual, str) and Path(actual).is_absolute(),
            label + " must be an explicit local absolute path")
    require(Path(actual).resolve() == Path(expected).resolve(), label + " differs from plan")


def at(config, path):
    result = config
    for key in path.split("."):
        require(isinstance(result, dict) and key in result, "missing NeMo config field: " + path)
        result = result[key]
    return result


def equal(config, path, expected):
    value = at(config, path)
    # Pydantic may normalize a declared numeric field from 1 to 1.0.
    # Booleans/strings are never accepted as numbers; nested structures retain
    # their JSON types (e.g. enable_thinking=0 cannot masquerade as false).
    same = (number(value) and value == expected) if type(expected) in (int, float) else (
        type(value) is type(expected) and json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True))
    require(same, "unsupported or unbound NeMo config: " + path)


def loss_mapping(config, algorithm):
    loss = at(config, "loss_fn")
    for key in ("disable_ppo_ratio", "sequence_level_importance_ratios",
                "use_importance_sampling_correction", "use_on_policy_kl_approximation",
                "use_kl_in_reward"):
        equal(loss, key, False)
    # Optional fields have these defaults in the inspected ClippedPGLossConfig.
    require(loss.get("use_cispo", False) is False, "CISPO mapping is not implemented")
    require(loss.get("positive_example_nll_weight", 0.0) == 0.0, "NLL auxiliary loss is unsupported")
    for key in ("ratio_clip_c", "truncated_importance_sampling_type",
                "truncated_importance_sampling_ratio", "truncated_importance_sampling_ratio_min"):
        equal(loss, key, None)
    equal(loss, "reference_policy_kl_type", "k3")
    clip, beta = at(loss, "ratio_clip_min"), at(loss, "reference_policy_kl_penalty")
    require(number(clip) and 0 < clip < 1 and at(loss, "ratio_clip_max") == clip,
            "trace mapping requires symmetric policy clipping")
    require(number(beta) and beta >= 0, "invalid KL penalty")
    for key in ("token_level_loss", "force_on_policy_ratio"):
        require(type(at(loss, key)) is bool, "loss flag must be boolean: " + key)
    bound, cap = at(loss, "kl_input_clamp_value"), at(loss, "kl_output_clamp_value")
    require(number(bound) and 0 < bound <= 50 and number(cap) and cap > 0, "invalid KL clamp")
    result = dict(algorithm=algorithm, loss_variant="clipped_pg_k3",
        offpolicy_correction=False, reduction="token" if loss["token_level_loss"] else "sequence",
        ratio_clip=clip, kl_beta=beta, force_on_policy=loss["force_on_policy_ratio"],
        kl_input_clamp=bound, kl_output_clamp=cap, kl_sampling="non_is_score_gradient")
    if algorithm == "ppo":
        equal(config, "value_loss_fn.scale", 1.0)
        value_clip = at(config, "value_loss_fn.cliprange")
        require(number(value_clip) and value_clip > 0, "PPO value clipping is required")
        result["value_clip"] = value_clip
    return result


def model_bounds(config, section, plan, snapshot):
    model = at(config, section)
    require(model.get("pretrained_checkpoint") is None, "unbound pretrained checkpoint source")
    require(model.get("megatron_cfg", {}).get("force_reconvert_from_hf", False) is False,
            "forced cache replacement is not part of the initial capture profile")
    train = plan["training"]
    same_path(at(model, "model_name"), snapshot, section + ".model_name")
    same_path(at(model, "tokenizer.name"), plan["tokenizer"]["snapshot"], section + ".tokenizer.name")
    equal(model, "tokenizer.chat_template_kwargs", {"enable_thinking": False})
    require(model["tokenizer"].get("chat_template", "default") == "default",
            "runtime tokenizer must preserve its checkpoint chat template")
    equal(model, "precision", train["dtype"])
    equal(model, "train_micro_batch_size", train["micro_batch_size"])
    equal(model, "train_global_batch_size", train["global_batch_size"])
    equal(model, "max_total_sequence_length", train["sequence_length"])
    equal(model, "dtensor_cfg.enabled", False)
    equal(model, "megatron_cfg.enabled", True)
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                "context_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size"):
        equal(model, "megatron_cfg." + key, 1)
    equal(model, "megatron_cfg.sequence_parallel", False)
    equal(model, "sequence_packing.enabled", False)
    equal(model, "dynamic_batching.enabled", False)
    equal(model, "megatron_cfg.fp8_cfg.enabled", False)
    equal(model, "megatron_cfg.env_vars", None)
    require(model.get("quant_cfg") is None, "quantized worker mapping is unsupported")
    require(model.get("hf_config_overrides", {}) == {}, "model config overrides require a separate mapping")
    require(model["megatron_cfg"].get("checkpoint") in (None, {}), "extra checkpoint paths are unsupported")
    equal(model, "megatron_cfg.optimizer.lr", train["learning_rate"])
    equal(model, "megatron_cfg.optimizer.optimizer", "adam")
    equal(model, "megatron_cfg.optimizer.bf16", train["dtype"] == "bfloat16")
    equal(model, "megatron_cfg.optimizer.fp16", False)
    equal(model, "megatron_cfg.scheduler.lr_warmup_iters", 0)
    equal(model, "megatron_cfg.scheduler.lr_decay_style", "constant")
    require(integer(at(model, "megatron_cfg.scheduler.lr_decay_iters"))
            and model["megatron_cfg"]["scheduler"]["lr_decay_iters"] >= train["steps"],
            "scheduler horizon is shorter than the planned iterations")
    if section == "policy":
        equal(model, "megatron_cfg.peft.enabled", False)
        equal(model, "megatron_cfg.mtp_num_layers", 0)
        equal(model, "draft.enabled", False)
        equal(model, "make_sequence_length_divisible_by", 1)
        require(not model.get("reward_model_cfg", {}).get("enabled", False), "policy cannot use a value head")
    else:
        equal(model, "reward_model_cfg.enabled", True)
        equal(model, "reward_model_cfg.reward_model_type", "regression")


def validate_bound_config(config, plan):
    """Refuse unsupported branches before an official setup can allocate resources."""
    finite_tree(config)
    algorithm = plan["profile"].removeprefix("rl_")
    require(algorithm in ("grpo", "ppo"), "not an RL plan")
    train, output = plan["training"], Path(plan["execution"]["output_path"])
    require(train["resume_from"] is None, "first RL profile does not support resume")
    equal(config, "cluster.num_nodes", 1)
    equal(config, "cluster.gpus_per_node", len(plan["execution"]["devices"]))
    model_bounds(config, "policy", plan, plan["model"]["snapshot"])
    generation = at(config, "policy.generation")
    equal(generation, "backend", "vllm")
    equal(generation, "colocated.enabled", True)
    equal(generation, "vllm_cfg.async_engine", False)
    require(generation.get("use_async_rollouts", False) is False, "async rollout mapping is unsupported")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size"):
        equal(generation, "vllm_cfg." + name, 1)
    equal(generation, "vllm_cfg.max_model_len", train["sequence_length"])
    equal(generation, "vllm_cfg.precision", train["dtype"])
    equal(generation, "vllm_cfg.enforce_eager", True)
    equal(generation, "vllm_kwargs", {})
    require(integer(at(generation, "max_new_tokens")) and 1 <= generation["max_new_tokens"] < train["sequence_length"],
            "invalid generation length bound")
    equal(generation, "temperature", 1.0)
    equal(generation, "top_p", 1.0)
    require(generation.get("top_k") is None or (type(generation.get("top_k")) is int and generation["top_k"] == -1),
            "filtered logprobs need an explicit unfiltered KL mapping")
    require(generation.get("model_name", plan["model"]["snapshot"]) == plan["model"]["snapshot"],
            "generation checkpoint differs from policy")
    algo = at(config, algorithm)
    equal(algo, "max_num_steps", train["steps"])
    equal(algo, "max_num_epochs", 1)
    equal(algo, "seed", train["seed"])
    equal(algo, "max_rollout_turns", 1)
    equal(algo, "batch_multiplier", 1)
    equal(algo, "use_dynamic_sampling", False)
    equal(algo, "overlong_filtering", False)
    equal(algo, "val_at_start", False)
    equal(algo, "val_at_end", False)
    equal(algo, "val_period", 0)
    equal(algo, "reward_shaping.enabled", False)
    equal(algo, "reward_scaling.enabled", False)
    prompts, generations = at(algo, "num_prompts_per_step"), at(algo, "num_generations_per_prompt")
    require(integer(prompts) and integer(generations) and prompts >= 1 and 2 <= generations <= 8,
            "RL capture needs explicit prompt groups of 2–8 generations")
    require(prompts * generations == train["global_batch_size"] <= 8,
            "one bounded global batch and optimizer update per rollout is required")
    require(plan["capture"]["max_records"] >= train["steps"] * train["global_batch_size"],
            "capture bound cannot cover all actor microbatches")
    if algorithm == "grpo":
        equal(algo, "async_grpo.enabled", False)
        equal(algo, "adv_estimator.name", "grpo")
        for key in ("advantage_clip_low", "advantage_clip_high", "seq_logprob_error_threshold",
                    "invalid_tool_call_advantage", "malformed_thinking_advantage"):
            require(algo.get(key) is None, "unsupported GRPO branch: " + key)
        require(config.get("data_plane", {}).get("enabled", False) is False, "TransferQueue mapping is unsupported")
        require(not config.get("teacher"), "distillation requires a separate resource and loss mapping")
    else:
        equal(algo, "ppo_epochs", 1)
        equal(algo, "policy_training_start_step", 0)
        equal(algo, "adv_estimator.name", "gae")
        for key in ("gae_lambda_value", "gae_lambda_policy"):
            require(algo["adv_estimator"].get(key) is None, "decoupled GAE mapping is unsupported")
        require(algo["adv_estimator"].get("length_adaptive_alpha", 0.0) == 0, "length-adaptive GAE is unsupported")
        model_bounds(config, "value", plan, plan["critic"]["snapshot"])
    mapping = loss_mapping(config, algorithm)
    if algorithm == "ppo" and mapping["kl_beta"]:
        equal(algo, "skip_reference_policy_logprobs_calculation", False)
    data = at(config, "data")
    equal(data, "use_multiple_dataloader", False)
    equal(data, "num_workers", 0)
    equal(data, "shuffle", False)
    equal(data, "validation", None)
    require(data.get("default") in (None, {}), "freeze dataset defaults into the train section")
    require(integer(at(data, "max_input_seq_length"))
            and 0 < data["max_input_seq_length"] <= train["sequence_length"] - generation["max_new_tokens"],
            "prompt bound must leave room for the entire generation bound")
    dataset = at(data, "train")
    allowed = {"dataset_name", "data_path", "input_key", "output_key", "split_validation_size",
               "seed", "processor", "env_name", "prompt_file", "system_prompt_file"}
    require(isinstance(dataset, dict) and set(dataset) == allowed, "unknown ResponseDataset fields")
    for key, expected in dict(dataset_name="ResponseDataset", input_key="input", output_key="output",
        split_validation_size=0, seed=train["seed"], processor="math_hf_data_processor", env_name="math",
        prompt_file=None, system_prompt_file=None).items():
        equal(dataset, key, expected)
    same_path(dataset["data_path"], plan["data"]["path"], "data.train.data_path")
    equal(config, "env", {"math": {"num_workers": 1, "math_verify_impl": "hf_math_verify"}})
    same_path(at(config, "logger.log_dir"), output/"logs", "logger.log_dir")
    for key in ("wandb_enabled", "tensorboard_enabled", "mlflow_enabled", "swanlab_enabled", "monitor_gpus"):
        equal(config, "logger." + key, False)
    same_path(at(config, "checkpointing.checkpoint_dir"), output/"checkpoints", "checkpointing.checkpoint_dir")
    equal(config, "checkpointing.enabled", True)
    equal(config, "checkpointing.save_period", 1)
    equal(config, "checkpointing.save_optimizer", True)
    equal(config, "checkpointing.checkpoint_must_save_by", None)
    require(at(config, "checkpointing.metric_name") in (None, "train:reward"), "validation checkpoint metric is unavailable")
    return dict(mapping="nemo_sync_megatron_v1", algorithm=algorithm, loss=mapping,
                iterations=train["steps"], actor_updates=train["steps"],
                critic_updates=train["steps"] if algorithm == "ppo" else 0,
                max_actor_records=train["steps"] * train["global_batch_size"],
                min_dataset_rows=train["steps"] * prompts,
                fixed_input_generation_calls=2, fixed_input_logprob_calls=1,
                worker_python="current_environment_only", conversion_cache="run_output/model-import",
                runtime_compatibility="not_checked")


def read_bound_config(plan):
    path = Path(plan["training"]["nemo_config"])
    require(path.stat().st_size <= MAX_BYTES, "frozen NeMo config exceeds 1 MiB")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    require(sha == plan["training"]["nemo_config_sha256"], "frozen NeMo config SHA256 mismatch")
    config = strict_json(raw.decode("utf-8"))
    require(isinstance(config, dict), "frozen NeMo config must be a JSON object")
    mapping = validate_bound_config(config, plan)
    return config, dict(path=str(path.resolve()), sha256=sha, bytes=len(raw)), mapping


def response_rows(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= MAX_BYTES, "ResponseDataset exceeds the bounded local profile")
    rows = [strict_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    require(1 <= len(rows) <= 1000, "expected 1–1000 local response rows")
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"input", "output"}
                and all(isinstance(v, str) and v for v in row.values()), "invalid local response row")
    return rows


def freeze_config(path, *, source_root, overrides, output):
    """Resolve once with the inspected helper contract; never import a launcher."""
    from .read_only_config import resolve_nemo_config
    resolved = resolve_nemo_config(path, source_root=source_root, overrides=overrides)
    encoded = (json.dumps(resolved["config"], ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    require(len(encoded) <= MAX_BYTES, "frozen NeMo config exceeds 1 MiB")
    with Path(output).open("xb") as stream:
        stream.write(encoded)
    return dict(status="configuration_frozen", path=str(Path(output).resolve()),
                sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded),
                source_files=resolved["files"], overrides=list(overrides),
                execution="not_run", compatibility="not_checked")
