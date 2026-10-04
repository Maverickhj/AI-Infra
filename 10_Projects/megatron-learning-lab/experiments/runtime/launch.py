"""Bounded local subprocess supervisor for already authorized runtime plans."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .contracts import require
from .plan import ROOT, check_grant, digest, inspect_plan, read_document, validate_plan


def now():
    return datetime.now(timezone.utc).isoformat()


def supervised_process(argv, *, cwd, env, log_path, max_seconds, max_output_bytes=16*1024*1024):
    """Delegate to a private Linux reaper, including detached descendants."""
    require(max_seconds > 0 and max_output_bytes > 0, "invalid subprocess bounds")
    guard = Path(__file__).with_name("process_guard.py")
    result = subprocess.run([sys.executable, "-S", str(guard),
        "--log", str(Path(log_path).resolve()), "--seconds", str(max_seconds),
        "--bytes", str(max_output_bytes), "--parent", str(os.getpid()), "--", *argv],
        cwd=cwd, env=env, text=True, capture_output=True)
    require(result.returncode == 0, "owned process guard failed: " + result.stderr[-4096:])
    record = json.loads(result.stdout)
    require(record.get("ownership") == "dedicated_linux_subreaper_pidfd"
            and record.get("descendant_cleanup", {}).get("status") == "completed",
            "owned runtime descendants were not fully cleaned up")
    return record


def launch(plan_path, resources_path):
    plan_path, resources_path = Path(plan_path).resolve(), Path(resources_path).resolve()
    plan = validate_plan(read_document(plan_path))
    resources = read_document(resources_path)
    permission = check_grant(plan, resources)  # Must happen before inspection/imports/outputs.
    inspected = inspect_plan(plan)
    require(not inspected["issues"], "runtime prerequisites missing: " + "; ".join(inspected["issues"]))
    output = Path(plan["execution"]["output_path"]).resolve()
    require(output.parent.is_dir(), "create the approved output parent before execution")
    require(not output.exists(), "output already exists; use a new run_id/output path")
    output.mkdir(exist_ok=False)
    receipt = dict(schema="megatron-learning-lab.runtime-receipt", plan_sha256=digest(plan),
                   resources_sha256=hashlib.sha256(resources_path.read_bytes()).hexdigest(),
                   plan_path=str(plan_path), resources_path=str(resources_path),
                   started_at=now(), permission=permission, inspection=inspected,
                   execution="starting", environment={
                       "CUDA_VISIBLE_DEVICES": ",".join(plan["execution"]["devices"]),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true"})
    if plan["profile"].startswith("rl_"):
        receipt["environment"].update(
            NEMO_RL_PY_EXECUTABLES_SYSTEM="1", UV_OFFLINE="1",
            MEGATRON_LAB_OUTPUT_PATH=str(output),
            NRL_MEGATRON_CHECKPOINT_DIR=str(output/"model-import"))
    receipt_path = output/"receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    if plan["profile"] == "bridge_sft":
        prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                  "--nnodes=1", "--nproc_per_node=1", "--max_restarts=0",
                  "--module", "experiments.runtime.worker"]
    else:
        prefix = [sys.executable, "-m", "experiments.runtime.worker"]
    argv = prefix + ["--plan", str(plan_path),
            "--resources", str(resources_path), "--receipt", str(receipt_path)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("RAY_ADDRESS", None)
    env.update(receipt["environment"])
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    try:
        result = supervised_process(argv, cwd=ROOT, env=env, log_path=output/"runtime.log",
                                    max_seconds=plan["execution"]["max_wall_seconds"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = dict(status="failed", returncode=None, stop_reason="supervisor_error", error=str(exc))
    if result["status"] == "passed":
        try:
            final = read_document(output/"result.json")
            expected = {"hf_reference":"completed_reference","bridge_sft":"completed_bridge_entry",
                        "rl_grpo":"completed_nemo_entry","rl_ppo":"completed_nemo_entry"}[plan["profile"]]
            require(final.get("status") == expected, "worker did not finish its required result")
        except (ValueError, OSError) as exc:
            result.update(status="failed", stop_reason="missing_or_invalid_result", error=str(exc))
    receipt.update(execution=result["status"], result=result, completed_at=now(), command=argv)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    return receipt


def verify_worker(plan_path, resources_path, receipt_path):
    """The child rechecks the exact grant and environment before importing Torch."""
    plan = validate_plan(read_document(plan_path))
    resources = read_document(resources_path)
    check_grant(plan, resources)
    receipt = read_document(receipt_path)
    output = Path(plan["execution"]["output_path"]).resolve()
    require(Path(receipt_path).resolve() == output/"receipt.json", "receipt output mismatch")
    require(receipt["plan_sha256"] == digest(plan), "plan changed after authorization")
    require(receipt["resources_sha256"] == hashlib.sha256(Path(resources_path).read_bytes()).hexdigest(),
            "resource manifest changed after launch")
    require(receipt["execution"] == "starting", "receipt is not an unconsumed launch")
    for name, expected in receipt["environment"].items():
        require(os.environ.get(name) == expected, "worker environment mismatch: " + name)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(plan["execution"]["devices"]),
            "worker device differs from plan")
    require(inspect_plan(plan) == receipt["inspection"], "inspected inputs/source changed after launch")
    # Exact field validation prevents caller-supplied shell/module selection.
    return plan, output
