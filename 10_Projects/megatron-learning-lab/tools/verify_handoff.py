#!/usr/bin/env python3
"""离线校验交付包的引用、模型字段及结果来源，不运行 Megatron。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-f]{40}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_run_manifest(run: dict[str, Any]) -> None:
    """实测来源的最低元数据门槛；通过不代表数值正确或结果真实。"""
    allowed = {"derived", "reference", "observed_bridge", "observed_rl"}
    require(run.get("provenance") in allowed, "未知 provenance")
    if run["provenance"] == "derived":
        require(run.get("execution_status") == "not_run", "派生数据不能标成已执行")
        for key in ("gpu_elapsed_ms", "peak_memory_bytes", "measured_loss"):
            require(key not in run, "派生档案不能混入实测字段")
        return
    if run["provenance"].startswith("observed_"):
        required = ("source_lane", "runtime_versions", "module_import_paths", "config_sha256", "input_sha256",
                    "model_revision", "tokenizer_revision", "weights_origin", "architecture_origin",
                    "backend", "dtype", "parallelism", "started_at", "ended_at", "command", "result_files")
        for key in required:
            require(bool(run.get(key)), f"实测档案缺少 {key}")
        require(run.get("execution_status") == "completed", "实测结果必须记录已完成执行")
        for key in ("config_sha256", "input_sha256"):
            require(bool(re.fullmatch(r"[0-9a-f]{64}", run[key])), f"{key} 不是 SHA256")
        # 本课程观察官方 HF checkpoint 时以完整 commit 锁定模型和 tokenizer。
        for key in ("model_revision", "tokenizer_revision"):
            require(bool(SHA.fullmatch(run[key])), f"{key} 不是完整 revision")
        require(isinstance(run["source_lane"], str), "source_lane 应记录实际来源标识")
        profile = run.get("runtime_profile")
        profiles = {"bridge_sft", "parallel_sft", "moe_mla"} if run["provenance"] == "observed_bridge" else {"rl_grpo", "rl_ppo"}
        require(profile in profiles, "runtime_profile 与实测场景不匹配")
        validate_compatibility_report(run.get("compatibility"))


def validate_compatibility_report(report: Any) -> None:
    """只验报告结构；不运行探针，不推断包版本或实际语义兼容。"""
    require(isinstance(report, dict), "缺少兼容检查报告")
    require(report.get("status") == "supported", "所需接口/语义尚未通过验收")
    required = report.get("required_capabilities")
    require(isinstance(required, list) and bool(required), "必须明确必需检查集合")
    require(all(isinstance(key, str) and bool(key) for key in required), "无效的必需检查 ID")
    require(len(set(required)) == len(required), "重复的必需检查 ID")
    base = {"callable_interfaces", "data_semantics", "behavior_smoke", "runtime_source_mapping"}
    require(base <= set(required), "不能只检查签名而跳过语义/行为/实际源码")
    checks = report.get("checks")
    require(isinstance(checks, dict), "缺少逐项检查记录")
    for key in required:
        item = checks.get(key)
        require(isinstance(item, dict), f"缺少检查 {key}")
        require(item.get("status") == "supported", f"必需能力未通过: {key}")
        require(isinstance(item.get("evidence"), str) and bool(item["evidence"].strip()), f"缺少检查证据: {key}")


def verify(root: Path = ROOT) -> dict[str, Any]:
    lock = json.loads((root / "source.lock.json").read_text(encoding="utf-8"))
    require(lock.get("purpose") == "reference_source_evidence_not_runtime_install_lock", "源码锁不能成为安装锁")
    require(lock["runtime_policy"].get("require_exact_reference_versions") is False, "不得强制参考版本相等")
    contract = json.loads((root / "profiles/compatibility-contract.json").read_text(encoding="utf-8"))
    require(contract.get("exact_upstream_versions_required") is False, "不得强制复制上游依赖")
    evidence = json.loads((root / "research/source-evidence.json").read_text(encoding="utf-8"))["entries"]
    models = json.loads((root / "content/models.json").read_text(encoding="utf-8"))["models"]
    cases = json.loads((root / "content/generated/model-cases.json").read_text(encoding="utf-8"))["cases"]
    ids = {e["id"] for e in evidence}
    require(len(ids) == len(evidence), "重复 source ID")
    for repo in lock["repositories"].values():
        require(bool(SHA.fullmatch(repo["commit"])), "源码未锁定完整 SHA")
    for entry in evidence:
        repo = lock["repositories"][entry["repo_key"]]
        require(bool(SHA.fullmatch(entry["git_blob_sha"])), "无效 blob SHA")
        require(repo["commit"] in entry["url"], "源码 URL 未锁到对应 commit")
        require(entry["evidence_level"] == "static_source_read", "静态证据不应标为实测")
    require(len({m["id"] for m in models}) == len(models), "重复 model ID")
    require({m["id"] for m in models} == {c["id"] for c in cases}, "模型与整模案例不对应")
    for model in models:
        require(set(model["sources"]) <= ids, "模型存在未定义 source ID")
        require(model["revision"] is None or bool(SHA.fullmatch(model["revision"])), "无效 HF revision")
    for case in cases:
        validate_run_manifest(case)
        require(case["pipeline"][0]["id"] == "input" and case["pipeline"][-1]["id"] == "update", "不完整的整模链路")
        require(set(case["attention"]["source_ids"]) <= ids, "attention 引用不存在")
        for stage in case["pipeline"]:
            require(set(stage["source_ids"]) <= ids, "stage 引用不存在")
    for path in (root / "content/cases").glob("*.md"):
        require("数学符号" in path.read_text(encoding="utf-8"), f"{path.name} 缺少独立数学符号表")
    return {"status": "passed", "source_entries": len(evidence), "model_profiles": len(models),
            "derived_whole_model_cases": len(cases), "checks_are": "metadata_integrity_only",
            "gpu_or_framework_execution": False, "runtime_compatibility_verified": False}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
