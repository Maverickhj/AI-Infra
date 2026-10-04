#!/usr/bin/env python3
"""Execute PROGRAM-V1 software checks; preserve logs and an immutable source manifest.

This records evidence, not pedagogical truth or runtime authorization. STATE is
updated separately only after inspecting the results and the actual UI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from goal_gate import ROOT, SKIP, browser_result, digest, execute, inside


def snapshot() -> dict:
    files = set()
    for name in ('web', 'content', 'research', 'tests', 'tools', 'experiments', 'profiles'):
        for parent, dirs, names in os.walk(ROOT / name):
            dirs[:] = [d for d in dirs if d not in SKIP and not (Path(parent)/d).is_symlink()]
            files.update(Path(parent)/n for n in names if not n.endswith('.pyc') and not (Path(parent)/n).is_symlink())
    for name in ('package.json', 'package-lock.json', 'tsconfig.json', 'playwright.config.ts',
                 'source.lock.json', 'goals/PROGRAM-V1/plan.json', 'goals/PROGRAM-V1/MASTER_GOAL.md',
                 'goals/PROGRAM-V1/STAGES.md', 'goals/PROGRAM-V1/VALIDATION.md', 'goals/PROGRAM-V1/RESOURCES.md'):
        files.add(ROOT / name)
    entries = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(files)]
    return {'algorithm': 'sha256(canonical JSON file hash list)', 'files': entries,
            'sha256': hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage')
    parser.add_argument('--review', required=True, help='Already written, specific human/agent self-review file')
    args = parser.parse_args()
    plan = json.loads((ROOT/'goals/PROGRAM-V1/plan.json').read_text())
    row = next((s for s in plan['stages'] if s['id'] == args.stage), None)
    if row is None or row['track'] != 'software' or args.stage == 'G09':
        parser.error('Use only M00/G02–G08; runtime and release require their additional gates')
    review = inside(ROOT, args.review)
    if not review.is_file() or review.stat().st_size < 100:
        parser.error('Write the actual self-review before executing this gate')
    started = datetime.now(timezone.utc)
    folder = ROOT/'runs/program-v1'/args.stage/started.strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    manifest = snapshot()
    (folder/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    py = [sys.executable, '-S']
    steps = [('baseline', py+['tools/goal_gate.py', 'baseline']),
             ('gqa-regression', ['npm', 'run', 'test:gqa'])]
    if row['npm_check']:
        steps.append(('stage-specific', ['npm', 'run', row['npm_check']]))
    steps += [('source', py+['tools/verify_source_snippets.py', '--cache', 'runs/source-cache']),
              ('review', ['cat', args.review])]
    report = {'program_id': plan['program_id'], 'stage_id': args.stage, 'status': 'running',
              'source_fingerprint': manifest['sha256'],
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'started_at': started.isoformat(), 'checks': [], 'artifacts': [], 'reviewed': False,
              'limitations': ['CPU reference and static source only; no observed HF/Bridge/RL run',
                              'Chromium only; Firefox/WebKit not executed'], 'issues': []}
    def artifact(path):
        report['artifacts'].append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    artifact(folder/'source-manifest.json')
    output = folder/'report.json'
    for key, command in steps:
        result = execute(ROOT, command, folder/f'{key}.log', 900)
        result['id'] = key
        report['checks'].append(result)
        if key == 'baseline' and result['status'] == 'passed':
            try:
                line = (folder/'baseline.log').read_text().split('baseline_passed: ')[-1].strip()
                baseline = inside(ROOT, line)
                data = json.loads(baseline.read_text())
                if data['status'] != 'baseline_passed':
                    raise ValueError('Nested baseline not passed')
                artifact(baseline)
                browser = next(c for c in data['commands'] if c['id'] == 'browser')
                contract = json.loads((ROOT/'goals/G01-GQA/acceptance.json').read_text())
                markers = contract['browser_markers_existing']+contract['browser_markers_goal']
                if row['browser_marker']:
                    markers.append(row['browser_marker'])
                report['browser'] = browser_result(inside(ROOT, browser['log']).read_text(), markers)
                if row['browser_marker']:
                    report['checks'].append({**browser, 'id': 'browser'})
            except (ValueError, KeyError, OSError, StopIteration) as exc:
                result['status'] = 'failed'
                report['issues'].append(str(exc))
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f'{key}: {result["status"]}', flush=True)
        if result['status'] != 'passed':
            break
    if snapshot() != manifest:
        report['issues'].append('Source changed during checks; rerun against a stable snapshot')
    required = set(row['required_checks'])
    passed = not report['issues'] and required <= {c['id'] for c in report['checks']} and all(c['status'] == 'passed' for c in report['checks'])
    report['status'] = 'passed' if passed else 'failed'
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'{report["status"]}: {output.relative_to(ROOT)}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
