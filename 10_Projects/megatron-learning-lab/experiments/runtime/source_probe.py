"""Read explicit source paths without importing frameworks or executing launchers.

Interface observations are NOT runtime compatibility approval. Package versions
are metadata; only inspected contracts select adapters.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return path, raw, ast.parse(raw, filename=str(path))


def anchor(path, symbol):
    path, raw, tree = read_source(path)
    parts = symbol.split(".")
    nodes = [tree]
    for part in parts:
        nodes = [child for node in nodes for child in getattr(node, "body", [])
                 if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == part]
    if len(nodes) != 1:
        raise ValueError("source symbol missing or ambiguous: " + symbol)
    node = nodes[0]
    result = dict(path=str(path), symbol=symbol, sha256=sha256(raw),
                  line=node.lineno, end_line=node.end_lineno,
                  evidence="static_runtime_source_read")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        result["parameters"] = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
        result["var_keyword"] = node.args.kwarg.arg if node.args.kwarg else None
    return result


def parser_options(path):
    """Inspect literal argparse options. Do not run even parse_args from a repo."""
    path, raw, tree = read_source(path)
    parser = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"]
    if len(parser) != 1:
        raise ValueError("unknown CLI mapping: parse_args missing or ambiguous")
    options = {}
    for node in ast.walk(parser[0]):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        flags = [ast.literal_eval(arg) for arg in node.args]
        if not flags or any(not isinstance(flag, str) or not flag.startswith("-") for flag in flags):
            raise ValueError("unsupported CLI positional/dynamic argument")
        spec = {}
        for kw in node.keywords:
            if kw.arg in ("required", "default", "action", "dest"):
                try:
                    spec[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    raise ValueError("unsupported dynamic CLI default: " + flags[0])
            elif kw.arg == "type":
                if not isinstance(kw.value, ast.Name) or kw.value.id not in ("str", "int", "float"):
                    raise ValueError("unsupported CLI type")
                spec["type"] = kw.value.id
            elif kw.arg == "choices":
                # Dynamic registry is recorded as unresolved, never silently approved.
                try:
                    spec["choices"] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    spec["choices_expression"] = ast.unparse(kw.value)
        options[flags[0]] = dict(flags=flags, **spec)
    if not options:
        raise ValueError("unknown empty CLI mapping")
    return dict(path=str(path), sha256=sha256(raw), options=options,
                evidence="AST only; launcher not imported or executed")


def literal_registry(path, name):
    _, _, tree = read_source(path)
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            matches.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            matches.append(node.value)
    if len(matches) != 1:
        raise ValueError("unknown registry: " + name)
    node = matches[0]
    if isinstance(node, ast.Dict):
        return [ast.literal_eval(key) for key in node.keys]
    value = ast.literal_eval(node)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("registry must be literal string list")
    return value


def parse_inspected_cli(observation, argv, registries=None):
    """Rebuild the *declared* parser using only inspected literals, not eval."""
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    registries = registries or {}
    for spec in observation["options"].values():
        kwargs = {k: spec[k] for k in ("required", "default", "action", "dest", "choices") if k in spec}
        if "type" in spec:
            kwargs["type"] = {"str": str, "int": int, "float": float}[spec["type"]]
        if "choices_expression" in spec:
            expression = spec["choices_expression"]
            if expression not in registries:
                raise ValueError("unresolved CLI choices: " + expression)
            kwargs["choices"] = registries[expression]
        parser.add_argument(*spec["flags"], **kwargs)
    try:
        args, overrides = parser.parse_known_args(argv)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError("CLI argument mismatch") from exc
    if any("=" not in value or value.startswith("-") for value in overrides):
        raise ValueError("unknown CLI option or malformed config override")
    return dict(arguments=vars(args), overrides=overrides,
                scope="declared parser only; override semantics/runtime unchecked")


def inspect_bridge(root, recipe="qwen3_600m_sft_config"):
    root = Path(root).resolve()
    launcher = root / "scripts/training/run_recipe.py"
    cli = parser_options(launcher)
    required = {"--recipe", "--dataset", "--seq_length", "--step_func", "--hf_path"}
    if not required <= set(cli["options"]) or "--mode" in cli["options"]:
        raise ValueError("unknown Bridge CLI mapping; require explicit adapter")
    registry_path = root / "src/megatron/bridge/recipes/utils/dataset_utils.py"
    datasets = literal_registry(registry_path, "DATASET_TYPES")
    if "llm-finetune-preloaded" not in datasets:
        raise ValueError("preloaded finetune dataset unavailable")
    steps = literal_registry(launcher, "STEP_FUNCTIONS")
    if "gpt_step" not in steps:
        raise ValueError("gpt_step unavailable")
    recipe_path = root / "src/megatron/bridge/recipes/qwen/qwen3.py"
    recipe_anchor = anchor(recipe_path, recipe)
    fn = anchor(root / "src/megatron/bridge/training/finetune.py", "finetune")
    if not {"config", "forward_step_func", "callbacks"} <= set(fn["parameters"]):
        raise ValueError("unknown finetune/callback signature")
    callbacks = root / "src/megatron/bridge/training/callbacks.py"
    callback_anchors = [anchor(callbacks, "Callback." + name) for name in
                        ("on_train_start", "on_train_step_start", "on_train_step_end", "on_checkpoint_save")]
    # hf_path is an offline guarantee only if the selected factory actually accepts it.
    accepts_hf_path = "hf_path" in recipe_anchor["parameters"]
    return dict(adapter="bridge_dataset_cli_v1", cli=cli,
                registries={"DATASET_TYPES": datasets, "sorted(STEP_FUNCTIONS.keys())": sorted(steps)},
                recipe=recipe_anchor, finetune=fn, callbacks=callback_anchors,
                accepts_hf_path=accepts_hf_path,
                compatibility="not_checked", behavior="not_run",
                limitation="AST interface only; model/data/config semantics require behavioral probes")


def inspect_nemo_cli(root, algorithm):
    if algorithm not in ("grpo", "ppo"):
        raise ValueError("unsupported RL algorithm")
    path = Path(root).resolve() / "examples" / ("run_" + algorithm + ".py")
    cli = parser_options(path)
    if set(cli["options"]) != {"--config"}:
        raise ValueError("unknown NeMo RL CLI mapping; require explicit adapter")
    _, _, tree = read_source(path)
    main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    calls = {ast.unparse(node.func) for node in ast.walk(main[0]) if isinstance(node, ast.Call)} if len(main) == 1 else set()
    if not {"load_config", "parse_hydra_overrides", "init_ray", "setup"} <= calls:
        raise ValueError("unknown NeMo RL configuration/setup path")
    return dict(adapter="nemo_config_cli_v1", algorithm=algorithm, cli=cli,
                source=anchor(path, "main"), compatibility="not_checked", behavior="not_run",
                limitation="launcher calls init_ray; dry-run must not import/call main")


# Arguments consumed by the lab's synchronous adapter. This is an interface
# observation only; numerical and distributed compatibility require R02.
NEMO_INTERFACES = (
    ("nemo_rl/models/policy/lm_policy.py", "Policy.__init__", {"cluster", "config", "tokenizer", "worker_extension_cls_fqn"}),
    ("nemo_rl/models/policy/lm_policy.py", "Policy.train", {"data", "loss_fn", "timer"}),
    ("nemo_rl/models/policy/lm_policy.py", "Policy.get_logprobs", {"data"}),
    ("nemo_rl/models/policy/workers/megatron_policy_worker.py", "MegatronPolicyWorkerImpl.__init__", {"config", "tokenizer", "init_optimizer", "init_reference_model"}),
    ("nemo_rl/models/policy/workers/megatron_policy_worker.py", "MegatronPolicyWorkerImpl.train", {"data", "loss_fn"}),
    ("nemo_rl/models/megatron/setup.py", "validate_model_paths", {"config"}),
    ("nemo_rl/models/megatron/setup.py", "handle_model_import", {"config", "hf_model_name", "pretrained_path"}),
    ("nemo_rl/models/megatron/setup.py", "setup_model_and_optimizer", {"policy_cfg", "megatron_cfg", "load_optimizer"}),
    ("nemo_rl/models/megatron/setup.py", "setup_reference_model_state", {"config", "megatron_cfg", "pretrained_path"}),
    ("nemo_rl/experience/rollouts.py", "run_multi_turn_rollout", {"policy_generation", "input_batch", "tokenizer", "task_to_env", "max_seq_len"}),
    ("nemo_rl/algorithms/grpo.py", "refit_policy_generation", {"policy", "policy_generation", "colocated_inference"}),
    ("nemo_rl/algorithms/utils.py", "get_tokenizer", {"tokenizer_config"}),
    ("nemo_rl/distributed/ray_actor_environment_registry.py", "get_actor_python_env", {"actor_class_fqn"}),
)


def inspect_nemo_runtime(root, algorithm):
    observed = inspect_nemo_cli(root, algorithm)
    root = Path(root).resolve()
    specifications = list(NEMO_INTERFACES)
    setup_args = {"master_config", "tokenizer", "dataset", "val_dataset"}
    if algorithm == "grpo":
        setup_args.add("policy_factory")
    specifications += [
        (f"nemo_rl/algorithms/{algorithm}.py", "setup", setup_args),
        (f"nemo_rl/algorithms/{algorithm}.py", algorithm+"_train",
         {"policy", "policy_generation", "tokenizer", "loss_fn", "master_config"}
         | ({"value_model", "value_loss_fn"} if algorithm == "ppo" else set())),
    ]
    sources = [observed["source"]]
    for relative, symbol, expected in specifications:
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("NeMo source escaped the inspected checkout: " + relative)
        located = anchor(path, symbol)
        if not expected <= set(located.get("parameters", [])):
            raise ValueError("NeMo signature mismatch: " + symbol)
        sources.append(located)
    # Bind imported package helpers too, without importing the package. This
    # checksum is change detection, not a claim to have reviewed every function.
    files = sorted((root/"nemo_rl").rglob("*.py"))
    if not files or len(files) > 4096:
        raise ValueError("NeMo Python source inventory exceeds the inspected bound")
    digest = hashlib.sha256()
    total = 0
    for path in files:
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 4*1024*1024:
            raise ValueError("NeMo Python source is outside bounds: " + str(path))
        raw = path.read_bytes()
        total += len(raw)
        if total > 32*1024*1024:
            raise ValueError("NeMo Python source inventory exceeds 32 MiB")
        digest.update(str(path.relative_to(root)).encode()+b"\0"+raw+b"\0")
    observed.update(runtime_sources=sources,
        package_snapshot=dict(root=str(root/"nemo_rl"), python_files=len(files),
                              bytes=total, sha256=digest.hexdigest(),
                              evidence="read_only_source_change_detection"))
    return observed
