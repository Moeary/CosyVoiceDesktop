"""对比两组评测报告：基线（可由多份报告按用例合并）vs 候选。

    pixi run python -m evals.role_assign.compare --base reports/A reports/B --candidate reports/C

汇总从各报告的原始 trials / judgments 重新计算，因此旧报告也能得到后来新增的指标。
同一用例出现在多份基线报告里时，以后列出的报告为准。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import metrics
from .pipeline import summarize
from .run import REPORT_DIR, _num, _pct

# (key, 名称, 格式, 越大越好?)
OVERALL_ROWS = [
    ('label_acc', '片段准确率', _pct, True),
    ('speaker_acc', '对白/心理说话人准确率', _pct, True),
    ('category_acc', '类别准确率', _pct, True),
    ('label_acc_std', '准确率组内标准差（越小越稳）', _pct, False),
    ('all_agree_rate', '3 次全一致率', _pct, True),
    ('pairwise_agreement', '两两一致率', _pct, True),
    ('naming_consistency', '规范名一致率', _pct, True),
    ('roster_jaccard', '角色表 Jaccard（归一后）', _pct, True),
    ('raw_roster_jaccard', '角色表 Jaccard（原始名）', _pct, True),
    ('roster_recall', '角色表召回', _pct, True),
    ('roster_precision', '角色表精确率', _pct, True),
    ('casting_acc', '选角准确率', _pct, True),
    ('review_trigger_rate', 'ReAct 触发率', _pct, None),
    ('review_fixed', 'ReAct 改对', lambda v: _num(v, 0), True),
    ('review_broken', 'ReAct 改错', lambda v: _num(v, 0), False),
    ('failed_runs', '失败运行', lambda v: _num(v, 0), False),
    ('structured_fallbacks', '结构化降级次数', lambda v: _num(v, 0), False),
    ('cost_per_run', '每次运行 $', lambda v: _num(v, 4), False),
    ('cost_per_kchar', '每千字 $', lambda v: _num(v, 4), False),
    ('tokens_per_segment', '每片段 token', lambda v: _num(v, 0), False),
    ('reasoning_share', '推理 token 占比', _pct, False),
    ('cost_cv', '花费变异系数', _pct, False),
    ('latency_s', '平均耗时 s', lambda v: _num(v, 1), False),
    ('latency_cv', '耗时变异系数', _pct, False),
    ('judge_overall', '裁判总分', lambda v: _num(v, 1), True),
]


def report_path(path) -> Path:
    path = Path(path)
    if not path.exists():
        path = REPORT_DIR / path.name  # 允许只写报告目录名
    return path / 'report.json' if path.is_dir() else path


def load_report(path) -> dict:
    return json.loads(report_path(path).read_text(encoding='utf-8'))


RESCORED = ('label_acc', 'category_acc', 'speaker_acc', 'narration_false_positive', 'wrong',
            'roster', 'casting_acc', 'casting_wrong', 'gold_unmatched', 'gold_ambiguous')


def rescore(trial: dict, case: dict) -> dict:
    """按当前金标准重算与 gold 相关的字段，让新旧报告用同一把尺子。

    校对前准确率、ReAct 改对/改错依赖未落盘的校对前标签，保留运行时的原值。
    """
    if 'score' not in trial:
        return trial
    state = dict(trial['state'])
    state['labels'] = {int(k): v for k, v in state['labels'].items()}
    fresh = metrics.score_trial(case, state)
    return {**trial, 'score': {**trial['score'], **{k: fresh[k] for k in RESCORED}}}


def merge(reports: List[dict]) -> dict:
    """按用例合并多份报告的 trials / judgments（后者覆盖前者），并按当前金标准重算。"""
    cases = {c['id']: c for c in metrics.load_cases()}
    by_case = {}
    for report in reports:
        for case_id in {t['case_id'] for t in report['trials']} & set(cases):
            by_case[case_id] = {
                'trials': [rescore(t, cases[case_id]) for t in report['trials'] if t['case_id'] == case_id],
                'judgments': [j for j in report.get('judgments', []) if j['case_id'] == case_id],
                'skill': report['config']['skill'],
                'trials_per_case': report['config']['trials'],
            }
    return by_case


def resummarize(by_case: dict, case_ids: List[str]) -> dict:
    cases = [c for c in metrics.load_cases(only=case_ids)]
    trials = [t for cid in case_ids for t in by_case[cid]['trials']]
    judgments = [j for cid in case_ids for j in by_case[cid]['judgments']]
    expected = max(by_case[cid]['trials_per_case'] for cid in case_ids)
    return summarize(cases, trials, judgments, expected)


def _delta(a, b, fmt, higher_better) -> str:
    if a is None or b is None:
        return ''
    diff = b - a
    if abs(diff) < 1e-9:
        return '='
    text = (f"{diff:+.1%}" if fmt is _pct else f"{diff:+.4g}")
    if higher_better is None:
        return text
    return f"{text} {'✅' if (diff > 0) == higher_better else '⚠️'}"


def _skill_labels(by_case: dict, case_ids: List[str]) -> str:
    return '、'.join(sorted({f"v{by_case[c]['skill']['version']}#{by_case[c]['skill']['digest']}" for c in case_ids}))


def render(base: dict, cand: dict, base_by: dict, cand_by: dict, case_ids: List[str], groups: dict) -> str:
    lines = [
        '# Skill 版本对比',
        '',
        f"- 基线：{_skill_labels(base_by, case_ids)}",
        f"- 候选：{_skill_labels(cand_by, case_ids)}",
        f"- 用例：{len(case_ids)} 个（{'、'.join(case_ids)}）",
        '',
    ]
    for title, (b, c) in {'全部用例': (base, cand), **groups}.items():
        lines += [f"## 总体 · {title}", '', '| 指标 | 基线 | 候选 | 变化 |', '|---|---|---|---|']
        for key, name, fmt, better in OVERALL_ROWS:
            a, z = b['overall'].get(key), c['overall'].get(key)
            lines.append(f"| {name} | {fmt(a)} | {fmt(z)} | {_delta(a, z, fmt, better)} |")
        lines.append('')

    lines += [
        '## 分用例',
        '',
        '| 用例 | 片段 | 准确率 基线→候选 | 说话人 | 全一致 | 规范名一致 | 选角 | 每次 $ | token/片段 | 裁判 |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]

    def ms(stat):
        return '-' if not stat or stat.get('mean') is None else f"{stat['mean']:.1%}±{stat['std']:.1%}"

    def pair(a, b, fmt=_pct):
        return f"{fmt(a)} → {fmt(b)}"

    for cid in case_ids:
        x, y = base['cases'][cid], cand['cases'][cid]
        if not x.get('runs') or not y.get('runs'):
            lines.append(f"| {cid} | - | 基线 {x.get('runs', 0)} 次 / 候选 {y.get('runs', 0)} 次成功 | | | | | | | |")
            continue
        lines.append(
            f"| {cid} | {y['segments']} | {ms(x['label_acc'])} → {ms(y['label_acc'])} "
            f"| {pair(x['speaker_acc']['mean'], y['speaker_acc']['mean'])} "
            f"| {pair(x['stability'].get('all_agree_rate'), y['stability'].get('all_agree_rate'))} "
            f"| {pair(x['stability'].get('naming_consistency'), y['stability'].get('naming_consistency'))} "
            f"| {pair(x['casting_acc'], y['casting_acc'])} "
            f"| {pair(x['efficiency']['cost']['mean'], y['efficiency']['cost']['mean'], lambda v: _num(v, 4))} "
            f"| {pair(x['efficiency']['tokens_per_segment'], y['efficiency']['tokens_per_segment'], lambda v: _num(v, 0))} "
            f"| {pair(x['judge'].get('overall'), y['judge'].get('overall'), lambda v: _num(v, 1))} |"
        )
    lines.append('')

    lines += ['## 候选版本仍存在的问题', '']
    for cid in case_ids:
        y = cand['cases'][cid]
        issues = list(y.get('top_wrong', []))
        drift = y.get('stability', {}).get('name_variants') or {}
        if drift:
            issues.append('规范名漂移：' + '；'.join(f"{k} ← {'/'.join(v)}" for k, v in drift.items()))
        if y.get('casting_wrong'):
            issues.append('选角不符：' + '；'.join(y['casting_wrong']))
        if not issues:
            continue
        lines.append(f"### {cid}")
        lines += [f"- {i}" for i in issues]
        lines.append('')
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='对比 Skill 评测报告')
    parser.add_argument('--base', nargs='+', required=True, help='基线报告目录或 report.json，可多份按用例合并')
    parser.add_argument('--candidate', nargs='+', required=True, help='候选报告目录或 report.json')
    parser.add_argument('--group', action='append', default=[], metavar='名称=用例1,用例2',
                        help='额外输出一组用例的总体对比，可重复')
    parser.add_argument('--out', help='输出 Markdown 路径，默认写到第一份候选报告目录下的 compare.md')
    args = parser.parse_args(argv)

    base_by = merge([load_report(p) for p in args.base])
    cand_by = merge([load_report(p) for p in args.candidate])
    case_ids = [c['id'] for c in metrics.load_cases() if c['id'] in base_by and c['id'] in cand_by]
    if not case_ids:
        print('基线与候选没有共同用例', file=sys.stderr)
        return 1

    base, cand = resummarize(base_by, case_ids), resummarize(cand_by, case_ids)
    groups = {}
    for spec in args.group:
        name, _, ids = spec.partition('=')
        ids = [i for i in ids.split(',') if i in case_ids]
        if ids:
            groups[name] = (resummarize(base_by, ids), resummarize(cand_by, ids))

    text = render(base, cand, base_by, cand_by, case_ids, groups)
    out = Path(args.out) if args.out else report_path(args.candidate[0]).parent / 'compare.md'
    out.write_text(text, encoding='utf-8')
    (out.with_suffix('.json')).write_text(json.dumps(
        {'cases': case_ids, 'base': base, 'candidate': cand,
         'groups': {k: {'base': v[0], 'candidate': v[1]} for k, v in groups.items()}},
        ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(text)
    print(f"\n已写入 {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
