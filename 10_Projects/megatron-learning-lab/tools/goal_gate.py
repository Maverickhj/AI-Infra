#!/usr/bin/env python3
"""G01 的显式质量门槛。只运行检查，不启动 Codex、不安装依赖、不修改 Git。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
GOAL = Path('goals/G01-GQA')
SKIP = {'node_modules', '__pycache__', '.git', 'generated', 'runs', 'dist'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, relative: str) -> Path:
    """拒绝越界、绝对路径及指向根外的符号链接。"""
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'不是项目内相对路径: {relative}')
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'路径超出项目: {relative}')
    return result


def snapshot(root: Path, contract: dict) -> str:
    files: set[Path] = set()
    for name in contract['source_roots']:
        folder = inside(root, name)
        if folder.is_dir():
            for parent, dirs, names in os.walk(folder, followlinks=False):
                dirs[:] = [d for d in dirs if d not in SKIP and not (Path(parent)/d).is_symlink()]
                for name2 in names:
                    file = Path(parent)/name2
                    if file.suffix != '.pyc' and not file.is_symlink():
                        files.add(file)
    for name in contract['source_files']:
        file = inside(root, name)
        if file.is_file():
            files.add(file)
    # 目标和验收标准也属于这次报告的指纹。进度和证据单独核对。
    files.update(inside(root, str(GOAL/name)) for name in ('GOAL.md', 'acceptance.json'))
    h = hashlib.sha256()
    for file in sorted(files):
        h.update(str(file.relative_to(root)).replace(os.sep, '/').encode())
        h.update(b'\0' + file.read_bytes() + b'\0')
    return h.hexdigest()


def check_evidence(root: Path, contract: dict) -> list[str]:
    problems = []
    for name in contract['required_artifacts']:
        file = inside(root, name)
        if not file.is_file() or not file.stat().st_size:
            problems.append(f'缺少交付物: {name}')
    evidence = json.loads(inside(root, str(GOAL/'evidence.json')).read_text(encoding='utf-8'))
    if evidence.get('goal_id') != contract['goal_id']:
        problems.append('evidence goal_id 不匹配')
    items = evidence.get('criteria', [])
    expected = {c['id'] for c in contract['criteria']}
    if len(items) != len(expected) or {c.get('id') for c in items} != expected:
        problems.append('验收项集合缺失、重复或被替换')
    for item in items:
        key = item.get('id', '?')
        if item.get('status') != 'verified' or not item.get('summary', '').strip():
            problems.append(f'{key}: 尚未核验或缺少说明')
        artifacts = item.get('artifacts', [])
        if not artifacts:
            problems.append(f'{key}: 缺少证据文件')
        for artifact in artifacts:
            file = inside(root, artifact.get('path', ''))
            if not file.is_file() or not file.stat().st_size or digest(file) != artifact.get('sha256'):
                problems.append(f'{key}: 证据缺失、为空或 SHA256 不符: {artifact.get("path")}')
    return problems


def browser_result(text: str, markers: list[str]) -> dict:
    """只解析本次进程 stdout 的 Playwright JSON，拒绝旧文件、跳过和 flaky。"""
    data = None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
            if isinstance(candidate, dict) and 'suites' in candidate and 'stats' in candidate:
                data = candidate
                break
        except ValueError:
            pass
    if data is None:
        raise ValueError('本次运行没有可解析的 Playwright JSON')
    stats = data['stats']
    if stats.get('expected', 0) <= 0 or any(stats.get(k, 0) for k in ('unexpected', 'flaky', 'skipped')) or data.get('errors'):
        raise ValueError(f'浏览器测试未全部通过（不接受 skipped/flaky/零用例）: {stats}')
    titles = []
    def walk(suite):
        for spec in suite.get('specs', []):
            tests = spec.get('tests', [])
            if tests and all(t.get('status') == 'expected' for t in tests):
                titles.append(spec.get('title', ''))
        for child in suite.get('suites', []):
            walk(child)
    for suite in data['suites']:
        walk(suite)
    missing = [m for m in markers if not any(m in title for title in titles)]
    if missing:
        raise ValueError(f'必要浏览器场景未执行通过: {missing}')
    return {'stats': stats, 'passed_titles': titles}


def execute(root: Path, argv: list[str], log: Path, timeout: int) -> dict:
    """保存真实退出状态。超时和不可执行不转换为成功。"""
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONUTF8'] = '1'
    env['CI'] = '1'  # 现有配置因此不会复用无法确认来源的 dev server。
    command = list(argv)
    command[0] = shutil.which(command[0]) or command[0]
    try:
        with log.open('wb') as output:
            proc = subprocess.Popen(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=(os.name == 'posix'))
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only terminate the process group created by this check.
                if os.name == 'posix':
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=5)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()
                raise
        status = 'passed' if rc == 0 else 'failed'
    except (OSError, subprocess.TimeoutExpired) as exc:
        with log.open('ab') as output:
            output.write(('\nGATE BLOCKED: ' + str(exc)).encode('utf-8'))
        rc, status = None, 'blocked'
    return {'command': argv, 'returncode': rc, 'status': status,
            'log': str(log.relative_to(root)), 'log_sha256': digest(log)}


def verify_report(root: Path, report_path: Path, contract: dict) -> dict:
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('mode') != 'final' or report.get('status') != 'ready_for_review':
        raise ValueError('不是通过的 final 报告；baseline 不能代表完成')
    if report.get('source_fingerprint') != snapshot(root, contract):
        raise ValueError('源码或目标已改变，必须重新 final')
    if report.get('evidence_sha256') != digest(root/GOAL/'evidence.json'):
        raise ValueError('证据清单已改变，必须重新 final')
    expected = {'data','python-tests','handoff','snippets','build','browser','diff','gqa-numeric','upstream-snippets'}
    commands = report.get('commands', [])
    if len(commands) != len(expected) or {c.get('id') for c in commands} != expected:
        raise ValueError('门槛命令缺失或重复')
    for result in commands:
        if result.get('status') != 'passed' or result.get('returncode') != 0:
            raise ValueError('报告包含未通过命令')
        log = inside(root, result['log'])
        if not log.is_file() or digest(log) != result['log_sha256']:
            raise ValueError('日志缺失或被修改')
    browser = next(c for c in commands if c['id'] == 'browser')
    browser_result(inside(root,browser['log']).read_text(encoding='utf-8',errors='replace'), contract['browser_markers_existing']+contract['browser_markers_goal'])
    problems = check_evidence(root, contract)
    if problems:
        raise ValueError('; '.join(problems))
    return {'status':'ready_for_review','scope':'G01 only; not a proof of training or pedagogical correctness'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight','baseline','final','verify'])
    parser.add_argument('report', nargs='?')
    parser.add_argument('--timeout', type=int, default=900, help='单条检查进程超时秒数')
    args = parser.parse_args()
    contract = json.loads((ROOT/GOAL/'acceptance.json').read_text(encoding='utf-8'))
    if args.mode == 'verify':
        if not args.report:
            parser.error('verify 需要本项目内的报告路径')
        print(json.dumps(verify_report(ROOT,inside(ROOT,args.report),contract),ensure_ascii=False,indent=2))
        return 0
    required = ['package.json','package-lock.json','web/main.tsx','tools/build_case_data.py','playwright.config.ts']
    missing = [name for name in required if not (ROOT/name).is_file()]
    tools = {name:shutil.which(name) for name in ('node','npm','git','uv')}
    # uv 当前由仓库 package scripts 调用；不是所有 E0 项目的永久依赖。
    package = json.loads((ROOT/'package.json').read_text(encoding='utf-8')) if (ROOT/'package.json').is_file() else {}
    uv_needed = any('uv run' in cmd for cmd in package.get('scripts',{}).values())
    missing.extend(name for name,path in tools.items() if path is None and (name != 'uv' or uv_needed))
    if not (ROOT/'node_modules').is_dir():
        missing.append('node_modules（由用户按 package-lock 准备，本脚本不安装）')
    preflight = {'status':'blocked' if missing else 'available_for_checks','missing':missing,'tools':tools,
                 'python':sys.version.split()[0], 'note':'不证明构建、Chromium、GPU 或 Codex Goal 已可运行'}
    if args.mode == 'preflight' or missing:
        print(json.dumps(preflight,ensure_ascii=False,indent=2))
        return 2 if missing else 0
    if args.timeout <= 0:
        parser.error('--timeout 必须为正数')
    folder = ROOT/'runs/goal-g01'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    before = snapshot(ROOT, contract)
    py = [sys.executable,'-S']
    steps = [
        ('data', py+['tools/build_case_data.py']),
        ('python-tests',py+['-m','unittest','discover','-s','tests','-v']),
        ('handoff',py+['tools/verify_handoff.py']),
        ('snippets',py+['tools/verify_source_snippets.py']),
        ('build',['npm','run','build']),
        ('browser',['npm','run','test:e2e','--','--reporter=json']),
        ('diff',['git','diff','--check']),
    ]
    if args.mode == 'final':
        steps += [('gqa-numeric',['npm','run','test:gqa']),
                  ('upstream-snippets',py+['tools/verify_source_snippets.py','--cache','runs/source-cache'])]
    identity = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=False)
    if identity.returncode:
        raise ValueError('无法记录当前 Git HEAD，停止生成运行报告')
    report = {'goal_id':contract['goal_id'],'git_head':identity.stdout.strip(),'mode':args.mode,'status':'running',
              'started_at':datetime.now(timezone.utc).isoformat(),'commands':[], 'issues':[]}
    output = folder/'report.json'
    for name, argv in steps:
        result = execute(ROOT,argv,folder/f'{name}.log',args.timeout)
        result['id'] = name
        if name == 'browser' and result['status'] == 'passed':
            try:
                markers = contract['browser_markers_existing'] + (contract['browser_markers_goal'] if args.mode == 'final' else [])
                result['browser'] = browser_result((folder/f'{name}.log').read_text(encoding='utf-8',errors='replace'),markers)
            except ValueError as exc:
                result['status'] = 'failed'
                report['issues'].append(str(exc))
        report['commands'].append(result)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'{name}: {result["status"]}',flush=True)
        if result['status'] != 'passed':
            break
    after = snapshot(ROOT, contract)
    report['source_fingerprint'] = after
    if before != after:
        report['issues'].append('执行期间源码发生变化，不能复用本轮结果')
    if args.mode == 'final':
        report['issues'].extend(check_evidence(ROOT,contract))
        report['evidence_sha256'] = digest(ROOT/GOAL/'evidence.json')
    passed = len(report['commands']) == len(steps) and all(c['status'] == 'passed' for c in report['commands']) and not report['issues']
    report['status'] = ('ready_for_review' if args.mode == 'final' else 'baseline_passed') if passed else 'not_ready'
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['limitation'] = '真实进程结果与文件完整性；不自动证明教学论证正确，不启动或完成原生 Codex Goal。'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{report["status"]}: {output.relative_to(ROOT)}')
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f'GATE BLOCKED: {exc}',file=sys.stderr)
        raise SystemExit(2)
