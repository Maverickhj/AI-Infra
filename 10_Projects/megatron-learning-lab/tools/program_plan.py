#!/usr/bin/env python3
"""只读检查 PROGRAM-V1 并选择下一阶段；不运行报告中的命令或启动 Codex。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path('goals/PROGRAM-V1')
STATUSES = {'pending', 'in_progress', 'validated', 'revalidate', 'failed', 'blocked_external'}
HEX64 = re.compile(r'[0-9a-f]{64}')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local(root: Path, name: str) -> Path:
    require(isinstance(name, str) and bool(name), '路径为空')
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, '拒绝项目外路径')
    target = (root / path).resolve()
    require(target.is_relative_to(root.resolve()), '拒绝指向项目外的符号链接')
    return target


def read(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'重复 JSON key: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(plan: dict, state: dict) -> dict[str, dict]:
    require(plan.get('program_id') == state.get('program_id'), 'program_id 不匹配')
    rows = plan.get('stages')
    require(isinstance(rows, list) and bool(rows), '阶段列表为空')
    ids = [row['id'] for row in rows]
    require(len(set(ids)) == len(ids), '阶段 ID 重复')
    stages = {row['id']: row for row in rows}
    records = state.get('stages', {})
    require(set(records) == set(stages), 'STATE 阶段集合不匹配')
    for key, row in stages.items():
        require(row.get('track') in {'software', 'runtime'}, f'{key}: 未知 track')
        deps = row.get('depends_on')
        require(isinstance(deps, list) and len(set(deps)) == len(deps), f'{key}: 依赖无效')
        require(set(deps) <= set(stages), f'{key}: 未知依赖')
        checks = row.get('required_checks')
        require(isinstance(checks, list) and bool(checks) and len(set(checks)) == len(checks), f'{key}: 验收 ID 无效')
        require(records[key].get('status') in STATUSES, f'{key}: 未知状态')
        if records[key]['status'] == 'blocked_external':
            require(bool(records[key].get('blocker')), f'{key}: 阻塞原因为空')
        if row['track'] == 'runtime':
            require(bool(row.get('requires_grant')), f'{key}: 真实运行缺资源门槛')
    visited, active = set(), set()
    def visit(key):
        require(key not in active, '阶段依赖存在环')
        if key in visited:
            return
        active.add(key)
        for dep in stages[key]['depends_on']:
            visit(dep)
        active.remove(key)
        visited.add(key)
    for key in stages:
        visit(key)
    return stages


def artifact(root: Path, item: dict, path_key: str = 'path', hash_key: str = 'sha256') -> Path:
    path = local(root, item.get(path_key, ''))
    expected = item.get(hash_key)
    require(isinstance(expected, str) and HEX64.fullmatch(expected) is not None, '缺少 SHA256')
    require(path.is_file() and path.stat().st_size > 0, f'证据不存在或为空: {path}')
    require(sha(path) == expected, f'证据 SHA256 不匹配: {path}')
    return path


def report_issues(root: Path, plan: dict, stage: dict, item: dict) -> list[str]:
    """只检查报告与文件的结构/完整性；不能证明检查真实执行或语义正确。"""
    try:
        path = artifact(root, item, 'report', 'report_sha256')
        report = read(path)
        require(report.get('program_id') == plan['program_id'] and report.get('stage_id') == stage['id'], '报告身份不匹配')
        require(report.get('status') == 'passed', '报告未通过')
        require(isinstance(report.get('source_fingerprint'), str) and HEX64.fullmatch(report['source_fingerprint']) is not None, '未记录源码指纹')
        checks = report.get('checks', [])
        ids = [check['id'] for check in checks]
        require(len(ids) == len(set(ids)) and set(stage['required_checks']) <= set(ids), '必要检查缺失或重复')
        for check in checks:
            require(check.get('status') == 'passed' and type(check.get('returncode')) is int and check['returncode'] == 0, '存在失败或未执行检查')
            cmd = check.get('command')
            require(isinstance(cmd, list) and bool(cmd) and all(isinstance(x, str) and x for x in cmd), '缺少实际命令记录')
            artifact(root, check, 'log', 'log_sha256')
        files = report.get('artifacts')
        require(isinstance(files, list) and bool(files), '没有交付证据')
        for value in files:
            artifact(root, value)
        return []
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return [str(exc)]


def select_next(root: Path, plan: dict, state: dict, resources: dict | None = None) -> dict:
    stages = validate(plan, state)
    records = state['stages']
    validated, issues = set(), {}
    for key, row in stages.items():
        if records[key]['status'] == 'validated':
            errors = report_issues(root, plan, row, records[key])
            if errors:
                issues[key] = errors
            else:
                validated.add(key)
    # A valid receipt does not make an unvalidated dependency disappear.
    while True:
        invalid = {key for key in validated if not set(stages[key]['depends_on']) <= validated}
        if not invalid:
            break
        for key in invalid:
            validated.remove(key)
            issues[key] = ['依赖尚未验证，不能继承此阶段成功状态']
    grants = (resources or {}).get('grants', {})
    waiting = {}
    for key, row in stages.items():
        if key in validated:
            continue
        deps = [d for d in row['depends_on'] if d not in validated]
        if deps:
            waiting[key] = {'dependencies': deps}
            continue
        grant = row.get('requires_grant')
        record = grants.get(grant, {}) if grant else {}
        if grant and not (record.get('authorized') is True and isinstance(record.get('authorization_ref'), str) and record['authorization_ref'].strip()):
            waiting[key] = {'resource_grant': grant, 'status': 'not_authorized'}
            continue
        status = records[key]['status']
        if key in issues or status in {'pending', 'in_progress', 'revalidate'}:
            return {'action': 'revalidate' if key in issues or status == 'revalidate' else 'continue',
                    'stage': key, 'title': row['title'], 'integrity_issues': issues,
                    'note': '只读计划建议；执行前仍须检查权限、资源和真实验收。'}
        waiting[key] = {'status': status, 'blocker': records[key].get('blocker'),
                        'next': '有新证据或解除阻塞后再将状态改为 revalidate'}
    software = {key for key, row in stages.items() if row['track'] == 'software'}
    summary = ('all_stage_reports_present' if len(validated) == len(stages) else
               'software_reports_present_runtime_blocked' if software <= validated else 'blocked')
    return {'action': 'review_final' if len(validated) == len(stages) else 'blocked',
            'status': summary, 'validated_reports': sorted(validated), 'waiting': waiting,
            'integrity_issues': issues,
            'note': '报告存在不等于当前最终代码已验收；G09 仍须在最新快照重跑全部检查。'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'status', 'next'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--resources', default='runs/program-v1/resources.json')
    args = parser.parse_args()
    root = args.root.resolve()
    plan, state = read(root / DIRECTORY / 'plan.json'), read(root / DIRECTORY / 'STATE.json')
    stages = validate(plan, state)
    resource_file = local(root, args.resources)
    resources = read(resource_file) if resource_file.exists() else {}
    result = select_next(root, plan, state, resources)
    if args.command == 'check':
        result = {'plan': 'valid', 'stages': len(stages), 'next': result,
                  'scope': 'planning_and_file_integrity_only'}
    elif args.command == 'status':
        result = {'stages': {k: v['status'] for k, v in state['stages'].items()}, 'next': result}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'PLAN INVALID: {exc}')
