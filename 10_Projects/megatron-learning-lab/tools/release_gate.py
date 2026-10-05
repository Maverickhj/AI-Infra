#!/usr/bin/env python3
"""Execute the complete software release gate; never run or authorize GPU work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from goal_gate import ROOT, browser_result, digest, execute, inside, snapshot as baseline_snapshot
from program_gate import snapshot as program_snapshot
from program_plan import report_issues, validate

CHECKS = ("test:gqa", "test:sft-data", "test:decoder", "test:tp-dp",
          "test:sequence", "test:moe-mla", "test:rl-reference", "test:runtime-contracts")
REVIEW = "goals/PROGRAM-V1/G09-REVIEW.md"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def source_snapshot():
    entries = {r["path"]: r for r in program_snapshot()["files"]}
    for name in ("README.md", REVIEW):
        entries[name] = dict(path=name, sha256=digest(ROOT/name))
    rows = sorted(entries.values(), key=lambda row: row["path"])
    return dict(algorithm="sha256(canonical JSON file hash list)", files=rows,
                sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest())


def tree_snapshot(folder):
    rows = [dict(path=str(p.relative_to(ROOT)),sha256=digest(p))
            for p in sorted(folder.rglob("*")) if p.is_file() and not p.is_symlink()]
    if not rows:
        raise ValueError("required built artifact tree is empty")
    return dict(files=rows,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest())


def dependencies():
    # Query the same interpreter with normal site initialization as CPU runners.
    # -S changes venv prefix/search paths and can select shadowed system metadata.
    program = """import importlib.metadata as m,json,sys
names=('torch','transformers','megatron-core','megatron-bridge','ray','nemo-rl','vllm','numpy','omegaconf')
versions={}
for name in names:
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
print(json.dumps(dict(versions=versions,prefix=sys.prefix,search_path=sys.path,
    frameworks_imported=[n for n in ('torch','ray','transformers','nemo_rl') if n in sys.modules])))
"""
    env=dict(os.environ)
    env.pop("PYTHONPATH",None)
    observed=json.loads(subprocess.check_output([sys.executable,"-c",program],cwd=ROOT,env=env,text=True,timeout=30))
    if observed["frameworks_imported"]:
        raise ValueError("metadata inspection unexpectedly imported a runtime framework")
    package = json.loads((ROOT/"package.json").read_text())
    npm = {}
    for name in sorted(set(package["dependencies"]) | set(package["devDependencies"])):
        npm[name] = json.loads((ROOT/"node_modules"/name/"package.json").read_text())["version"]
    return dict(python=sys.version, executable=sys.executable,
                python_packages=observed["versions"], python_prefix=observed["prefix"],
                python_search_path=observed["search_path"], npm_packages=npm,
                node=subprocess.check_output(["node","--version"],text=True).strip(),
                npm=subprocess.check_output(["npm","--version"],text=True).strip(),
                package_lock_sha256=digest(ROOT/"package-lock.json"),
                environment={key:os.environ.get(key) for key in
                             ("CUDA_VISIBLE_DEVICES","PLAYWRIGHT_BROWSERS_PATH")},
                scope="installed metadata only; not a GPU/ABI compatibility claim")


def validate_baseline(path):
    data=json.loads(path.read_text())
    expected={"data","python-tests","handoff","snippets","build","browser","diff"}
    commands=data.get("commands",[])
    if (data.get("status")!="baseline_passed" or data.get("mode")!="baseline"
            or data.get("issues") or len(commands)!=len(expected)
            or {c["id"] for c in commands}!=expected):
        raise ValueError("current complete baseline did not pass")
    contract=json.loads((ROOT/"goals/G01-GQA/acceptance.json").read_text())
    if data.get("source_fingerprint")!=baseline_snapshot(ROOT,contract):
        raise ValueError("baseline source snapshot changed")
    for row in commands:
        if row.get("status")!="passed" or row.get("returncode")!=0:
            raise ValueError("baseline contains failed or unexecuted command")
        if digest(inside(ROOT,row["log"]))!=row["log_sha256"]:
            raise ValueError("baseline log changed")
    py=next(c for c in commands if c["id"]=="python-tests")
    text=inside(ROOT,py["log"]).read_text()
    count=re.search(r"Ran (\d+) tests",text)
    if not count or int(count[1])<=0 or re.search(r"OK \(.*skipped=",text):
        raise ValueError("Python baseline must execute nonzero tests without skips")
    row=next(c for c in commands if c["id"]=="browser")
    browser=browser_result(inside(ROOT,row["log"]).read_text(),
        contract["browser_markers_existing"]+contract["browser_markers_goal"]+
        [f"[G{i:02d}]" for i in range(2,10)])
    return data, browser, int(count[1])


def command_evidence(command, folder):
    """Retain a real receipt when a successful command legitimately prints nothing."""
    log=inside(ROOT,command["log"])
    if digest(log)!=command["log_sha256"]:
        raise ValueError("command log hash changed")
    if log.stat().st_size:
        return log
    path=folder/("empty-output-"+command["id"]+".json")
    write(path,dict(kind="executed_command_with_empty_output",execution=command,
        output_bytes=0,note="Original empty log is retained and bound by its actual SHA256; no output was fabricated."))
    return path


def release_status(stages):
    runtime=[stages[key] for key in ("R01","R02")]
    if all(row["status"]=="validated" for row in runtime):
        return "full_project_ready_for_review"
    if all(row["status"] in {"validated","blocked_external"} for row in runtime):
        if any(row["status"]=="blocked_external" and not row.get("blocker") for row in runtime):
            raise ValueError("runtime blocker must be concrete")
        return "software_ready_runtime_blocked"
    raise ValueError("runtime status must be explicitly validated or blocked")


def regression(folder):
    folder.mkdir(parents=True,exist_ok=False)
    results=[]
    for name in CHECKS:
        row=execute(ROOT,["npm","run",name],folder/(name.replace(":","-")+".log"),900)
        row["id"]=name
        results.append(row)
        write(folder/"report.json",dict(checks=results,status="running"))
        print(name+": "+row["status"],flush=True)
        if row["status"]!="passed":
            break
    passed=len(results)==len(CHECKS) and all(r["status"]=="passed" for r in results)
    write(folder/"report.json",dict(checks=results,status="passed" if passed else "failed"))
    return 0 if passed else 1


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("release gate requires CUDA_VISIBLE_DEVICES= for CPU-only validation")
    plan=json.loads((ROOT/"goals/PROGRAM-V1/plan.json").read_text())
    state=json.loads((ROOT/"goals/PROGRAM-V1/STATE.json").read_text())
    stages=validate(plan,state)
    for key, stage in stages.items():
        if key=="G09" or state["stages"][key]["status"]=="blocked_external":
            continue
        if state["stages"][key]["status"]!="validated":
            raise ValueError("dependency has no validated report: "+key)
        issues=report_issues(ROOT,plan,stage,state["stages"][key])
        if issues:
            raise ValueError(key+": "+"; ".join(issues))
    for key in stages["G09"]["depends_on"]:
        if state["stages"][key]["status"]!="validated":
            raise ValueError("software dependency is blocked: "+key)
    outcome=release_status(state["stages"])
    now=datetime.now(timezone.utc)
    folder=ROOT/"runs/program-v1/G09"/now.strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True,exist_ok=False)
    source=source_snapshot()
    environment=dependencies()
    configs=[r for r in source["files"] if r["path"].startswith("profiles/") or r["path"] in
             ("source.lock.json","tsconfig.json","playwright.config.ts","tests/playwright.release.config.ts","package.json","package-lock.json")]
    write(folder/"source-manifest.json",source)
    write(folder/"dependencies.json",environment)
    write(folder/"configuration-manifest.json",dict(files=configs,
        sha256=hashlib.sha256(json.dumps(configs,sort_keys=True,separators=(",",":")).encode()).hexdigest()))
    inspections=[]
    for name in ("runtime", "runtime-grpo", "runtime-ppo"):
        command=[sys.executable,"-S","tools/runtime_cli.py","dry-run","--plan",f"profiles/{name}-plan.example.json"]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,check=True)
        data=json.loads(result.stdout)
        if data["execution"]!="not_run":
            raise ValueError("read-only startup inspection changed execution semantics")
        path=folder/(name+"-dry-run.json")
        path.write_text(result.stdout)
        inspections.append(dict(profile=name,command=command,returncode=result.returncode,
            report=str(path.relative_to(ROOT)),inspection_status=data["inspection"]["status"],
            issues=data["inspection"]["issues"],resources=data["resources"],execution=data["execution"]))
    write(folder/"runtime-readiness.json",dict(sample_plan_inspections=inspections,
        resource_file_exists=(ROOT/"runs/program-v1/resources.json").exists(),
        scope="read-only example-plan inspection; does not discover other resources or authorize training"))
    write(folder/"stage-index.json",dict(program_id=plan["program_id"],historical_stages={
        k:v for k,v in state["stages"].items() if k!="G09"},
        current_release_report=str((folder/"report.json").relative_to(ROOT)),
        note="historical reports cover their own snapshots; current regression independently reruns every software check"))
    report=dict(program_id=plan["program_id"],stage_id="G09",status="running",
        release_status=None,source_fingerprint=source["sha256"],
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_at=now.isoformat(),checks=[],artifacts=[],reviewed=False,issues=[],
        stages_at_start=state["stages"],
        limitations=["CPU reference, explicit synthetic contracts and Chromium only",
            "Firefox/WebKit not executed; no GPU/NCCL/performance claims",
            "Production preview is local; no public deployment",
            "Import validation does not authenticate observed trace provenance"])
    def artifact(path):
        relative=str(path.relative_to(ROOT))
        if relative not in {a["path"] for a in report["artifacts"]}:
            report["artifacts"].append(dict(path=relative,sha256=digest(path)))
    for name in ("source-manifest.json","dependencies.json","configuration-manifest.json","stage-index.json"):
        artifact(folder/name)
    for name in ("runtime", "runtime-grpo", "runtime-ppo"):
        artifact(folder/(name+"-dry-run.json"))
    artifact(folder/"runtime-readiness.json")
    output=folder/"report.json"
    py=[sys.executable,"-S"]
    steps=[
        ("baseline",py+["tools/goal_gate.py","baseline"]),
        ("integrated-regression",py+["tools/release_gate.py","regression","--output",str((folder/"regressions").relative_to(ROOT))]),
        ("production-smoke",["npm","exec","--","playwright","test","--config","tests/playwright.release.config.ts","--reporter=json"]),
        ("source",py+["tools/verify_source_snippets.py","--cache","runs/source-cache"]),
        ("review",["cat",REVIEW]),
    ]
    dist=None
    for key, argv in steps:
        row=execute(ROOT,argv,folder/(key+".log"),1800 if key=="integrated-regression" else 900)
        row["id"]=key
        report["checks"].append(row)
        try:
            if row["status"]!="passed":
                raise ValueError(key+" did not pass")
            if key=="baseline":
                match=re.search(r"baseline_passed: (runs/goal-g01/[^\s]+/report.json)",inside(ROOT,row["log"]).read_text())
                if not match:
                    raise ValueError("no current baseline report")
                path=inside(ROOT,match[1])
                data,browser,python_count=validate_baseline(path)
                report.update(browser=browser,python_tests=python_count,baseline_report=match[1])
                browser_row=next(c for c in data["commands"] if c["id"]=="browser")
                report["checks"].append({**browser_row,"id":"browser"})
                artifact(path)
                for command in data["commands"]:
                    artifact(command_evidence(command,folder))
                dist=tree_snapshot(ROOT/"dist")
                write(folder/"dist-manifest.json",dist)
                artifact(folder/"dist-manifest.json")
            elif key=="integrated-regression":
                path=folder/"regressions/report.json"
                data=json.loads(path.read_text())
                if (data.get("status")!="passed" or len(data["checks"])!=len(CHECKS)
                        or [c["id"] for c in data["checks"]]!=list(CHECKS)):
                    raise ValueError("missing current integrated checks")
                report["integrated_checks"]=data["checks"]
                artifact(path)
                for command in data["checks"]:
                    if command["status"]!="passed" or command["returncode"]!=0 or digest(inside(ROOT,command["log"]))!=command["log_sha256"]:
                        raise ValueError("failed or changed integrated check")
                    artifact(inside(ROOT,command["log"]))
            elif key=="production-smoke":
                report["production_browser"]=browser_result(inside(ROOT,row["log"]).read_text(),[
                    "[G09] integrated sample", "[G09] direct nested", "[G09] narrow production"])
                if tree_snapshot(ROOT/"dist")!=dist:
                    raise ValueError("production bundle changed during validation")
                archive=folder/"production-artifacts"
                shutil.copytree(ROOT/"runs/production-results",archive)
                for path in archive.rglob("*"):
                    if path.is_file():
                        artifact(path)
                # Browser screenshots/logs are evidence to inspect, not a replacement for review.
            elif key=="review":
                artifact(ROOT/REVIEW)
        except (ValueError,KeyError,OSError,TypeError,StopIteration) as exc:
            report["issues"].append(str(exc))
            row["status"]="failed"
        write(output,report)
        print(key+": "+row["status"],flush=True)
        if row["status"]!="passed":
            break
    if source_snapshot()!=source:
        report["issues"].append("source or delivery documentation changed during release gate")
    if dependencies()!=environment:
        report["issues"].append("installed dependencies changed during release gate")
    required=set(stages["G09"]["required_checks"])
    passed=(required<={c["id"] for c in report["checks"]}
            and all(c["status"]=="passed" for c in report["checks"]) and not report["issues"])
    report.update(status="passed" if passed else "failed",
                  release_status=outcome if passed else "not_ready",
                  completed_at=datetime.now(timezone.utc).isoformat())
    write(output,report)
    if passed:
        issues=report_issues(ROOT,plan,stages["G09"],dict(
            report=str(output.relative_to(ROOT)),report_sha256=digest(output)))
        if issues:
            report["issues"].extend(issues)
            report.update(status="failed",release_status="not_ready")
            write(output,report)
            passed=False
    print(report["status"]+": "+str(output.relative_to(ROOT)),flush=True)
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["run","regression"])
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.mode=="regression":
        if not args.output or not args.output.startswith("runs/"):
            parser.error("regression output must be a fresh project runs directory")
        return regression(inside(ROOT,args.output))
    return run()


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print("RELEASE BLOCKED: "+str(exc),file=sys.stderr)
        raise SystemExit(2)
