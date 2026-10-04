"""Bounded local subprocess supervisor for already authorized runtime plans."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from .contracts import require
from .plan import ROOT, check_grant, digest, inspect_plan, read_document, validate_plan


def now():
    return datetime.now(timezone.utc).isoformat()


def supervised_process(argv, *, cwd, env, log_path, max_seconds, max_output_bytes=16*1024*1024):
    """Own a fresh process group; terminate only this invocation on a limit."""
    start = time.monotonic()
    reason = None
    total = 0
    with Path(log_path).open("xb") as output:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map() or proc.poll() is None:
                if time.monotonic() - start >= max_seconds:
                    reason = "wall_time_limit"
                    break
                for key, _ in selector.select(timeout=min(.1, max_seconds)):
                    raw = os.read(key.fileobj.fileno(), 65536)
                    if not raw:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(raw)
                    output.write(raw[:max(0, max_output_bytes-(total-len(raw)))])
                    if total > max_output_bytes:
                        reason = "output_limit"
                        break
                if reason:
                    break
            if reason:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # A child can outlive its exited group leader. Always finish
                # this owned group after the cleanup grace period.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            returncode = proc.wait(timeout=5)
        finally:
            selector.close()
            proc.stdout.close()
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
    return dict(status="passed" if returncode == 0 and reason is None else "failed",
                returncode=returncode, stop_reason=reason, output_bytes=total,
                elapsed_wall_seconds=time.monotonic()-start)


def launch(plan_path, resources_path):
    plan_path, resources_path = Path(plan_path).resolve(), Path(resources_path).resolve()
    plan = validate_plan(read_document(plan_path))
    resources = read_document(resources_path)
    permission = check_grant(plan, resources)  # Must happen before inspection/imports/outputs.
    inspected = inspect_plan(plan)
    require(not inspected["issues"], "runtime prerequisites missing: " + "; ".join(inspected["issues"]))
    require(plan["profile"] in ("hf_reference", "bridge_sft"),
            "NeMo runtime supervisor wiring is not complete yet")
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
    result = supervised_process(argv, cwd=ROOT, env=env, log_path=output/"runtime.log",
                                max_seconds=plan["execution"]["max_wall_seconds"])
    if result["status"] == "passed":
        try:
            final = read_document(output/"result.json")
            expected = "completed_reference" if plan["profile"] == "hf_reference" else "completed_bridge_entry"
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
