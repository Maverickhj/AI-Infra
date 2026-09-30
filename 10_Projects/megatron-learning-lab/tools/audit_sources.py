#!/usr/bin/env python3
"""从显式指定的本地 Git 仓库只读生成源码锚点。无联网、checkout 或 import。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> bytes:
    command = ["git", "-C", str(repo), *args]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def symbols_in(source: str) -> dict[str, list[dict[str, int]]]:
    result: dict[str, list[dict[str, int]]] = {}

    def walk(nodes: list[ast.stmt], parents: list[str]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join(parents + [node.name])
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                result.setdefault(name, []).append({"start_line": start, "end_line": node.end_lineno or node.lineno})
                walk(node.body, parents + [node.name])
            elif hasattr(node, "body") and isinstance(node.body, list):
                walk(node.body, parents)
                if hasattr(node, "orelse"):
                    walk(node.orelse, parents)
    walk(ast.parse(source).body, [])
    return result


def audit_entry(repo: Path, commit: str, entry: dict[str, Any]) -> dict[str, Any]:
    path = entry["path"]
    if path.startswith("/") or ".." in Path(path).parts:
        raise ValueError("不允许非仓库内相对路径")
    object_name = f"{commit}:{path}"
    blob = git(repo, "rev-parse", object_name).decode().strip()
    if blob != entry["git_blob_sha"]:
        raise ValueError(f"{entry['id']} blob 与证据清单不符")
    source = git(repo, "show", object_name).decode("utf-8")
    known = symbols_in(source) if path.endswith(".py") else {}
    anchors = {}
    for symbol in entry.get("symbols", []):
        candidates = known.get(symbol, [])
        if len(candidates) != 1:
            raise ValueError(f"{entry['id']} 符号 {symbol} 匹配 {len(candidates)} 次，不能猜测")
        anchors[symbol] = candidates[0]
    return {"id": entry["id"], "status": "locked_source_match", "commit": commit,
            "git_blob_sha": blob, "text_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "anchors": anchors, "runtime_verified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[], metavar="KEY=/path/to/repo")
    parser.add_argument("--output", type=Path, default=ROOT / "research/local-source-audit.json")
    args = parser.parse_args()
    repos = {}
    for item in args.repo:
        key, separator, value = item.partition("=")
        if not separator or not value:
            parser.error("--repo 格式应为 KEY=/path/to/repo")
        repos[key] = Path(value).expanduser().resolve()
    lock = json.loads((ROOT / "source.lock.json").read_text())
    unknown = set(repos) - set(lock["repositories"])
    if unknown:
        parser.error(f"未知 repo key: {sorted(unknown)}")
    entries = json.loads((ROOT / "research/source-evidence.json").read_text())["entries"]
    results = []
    for entry in entries:
        key = entry["repo_key"]
        if key not in repos:
            results.append({"id": entry["id"], "status": "not_checked", "reason": "没有指定本地仓库"})
            continue
        try:
            results.append(audit_entry(repos[key], lock["repositories"][key]["commit"], entry))
        except (ValueError, OSError, subprocess.TimeoutExpired, SyntaxError) as exc:
            results.append({"id": entry["id"], "status": "failed", "reason": str(exc)})
    report = {"runtime_verified": False, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    checked = sum(r["status"] == "locked_source_match" for r in results)
    print(f"{checked}/{len(results)} 个本地源码条目匹配；未检查不等于通过。")
    raise SystemExit(1 if any(r["status"] == "failed" for r in results) else 0)


if __name__ == "__main__":
    main()
