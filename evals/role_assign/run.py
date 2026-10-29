"""分角 Skill 评测入口。

用法（项目根目录）：
    pixi run eval-roles --dry-run                       # 只估算花费，不调用模型
    pixi run eval-roles                                 # 5 段文本 × 3 次，默认预算 $0.50
    pixi run eval-roles --cases dorm,system --trials 2
    pixi run eval-roles --skill path/to/SKILL.md        # 评测修改后的 Skill
    pixi run eval-roles --param low_confidence=0.99     # 临时覆盖 Skill 参数（例如强制触发校对）

评分只看人工金标准（dataset/*.yaml 的 gold / roster / casting），不使用 LLM 裁判。
报告写到 evals/role_assign/reports/<时间>_<skill 哈希>/report.md 与 report.json。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

from core.role_agent.llm import OPENROUTER_BASE_URL, LLMConfigError, resolve_llm_settings
from core.role_agent.segmenter import chunk_segments, split_segments
from core.role_agent.skill import load_skill

from . import metrics
from .pipeline import CostLedger, EvalPipeline

DEFAULT_MODEL = 'deepseek/deepseek-v4.1-flash'
DEFAULT_ANALYST = 'openai/gpt-6-luna'
REPORT_DIR = Path(__file__).resolve().parent / 'reports'
# token 数按公式估得准，但 OpenRouter 实际回传的花费约为标价的 1.3~2.8 倍（路由到的供应商单价不同），取均值校准
COST_CALIBRATION = 1.9

# 指标看板的目标线
TARGETS = {
    'label_acc': ('片段准确率', 0.90, '≥'),
    'speaker_acc': ('对白/心理说话人准确率', 0.90, '≥'),
    'all_agree_rate': ('3 次运行全一致率', 0.90, '≥'),
    'roster_recall': ('角色表召回', 0.95, '≥'),
    'casting_acc': ('选角准确率', 0.80, '≥'),
    'cost_per_kchar': ('每千字花费 $', 0.01, '≤'),
}

# 校验疑点类型（与 core/role_agent/graph.py validate 一致）
KIND_NAMES = {
    'missing': '模型漏标',
    'unknown_speaker': '说话人未识别',
    'quote_as_narration': '引号内被标旁白',
    'not_in_roster': '说话人不在角色表',
    'cue_mismatch': '提示语指向不一致',
    'group_singular': '群体说单人台词',
    'low_confidence': '置信度低',
}


class _EnvConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def fetch_prices(models) -> dict:
    """从 OpenRouter 公开模型列表取单价（美元/token），失败时返回空。"""
    try:
        data = requests.get(f"{OPENROUTER_BASE_URL}/models", timeout=30).json()['data']
    except Exception:
        return {}
    return {
        item['id']: {'input': float(item['pricing']['prompt']), 'output': float(item['pricing']['completion'])}
        for item in data if item['id'] in models
    }


def key_usage(api_key: str):
    """当前 key 的累计花费（美元）。OpenRouter 记账可能有几秒延迟。"""
    try:
        data = requests.get(f"{OPENROUTER_BASE_URL}/key", headers={'Authorization': f'Bearer {api_key}'}, timeout=30).json()['data']
        return {'usage': data.get('usage'), 'limit_remaining': data.get('limit_remaining')}
    except Exception:
        return None


def estimate_costs(cases, skill, prices, model, trials, analyst=True) -> dict:
    """粗估每次运行的花费：按字数、片段数、批次数估 token。输出 token 按 1.5 倍计入推理开销。"""
    p_agent = prices.get(model) or {'input': 1e-6, 'output': 5e-6}
    estimates = {}
    for case in cases:
        segments = split_segments(case['text'], skill.params['speech_lead_verbs'])
        chunks = chunk_segments(segments, skill.params['chunk_max_items'], skill.params['chunk_max_chars'])
        chars = len(case['text'])
        tokens_in = 900 + chars * 1.2 + len(chunks) * 1100 + len(segments) * 30 + 900
        tokens_out = (600 + len(segments) * 45 + 300) * 1.5
        estimates[case['id']] = (tokens_in * p_agent['input'] + tokens_out * p_agent['output']) * COST_CALIBRATION
    total = sum(estimates[c['id']] for c in cases) * trials + (0.01 if analyst else 0)
    return {'per_item': estimates, 'total': total}


# ---------- 报告 ----------

def _pct(value) -> str:
    return '-' if value is None else f"{value:.1%}"


def _num(value, digits=2) -> str:
    return '-' if value is None else f"{value:.{digits}f}"


def _ms(stat, fmt=_pct) -> str:
    if not stat or stat.get('mean') is None:
        return '-'
    return f"{fmt(stat['mean'])} ± {fmt(stat['std'])}"


def render_markdown(config: dict, summary: dict, analysis: dict) -> str:
    o = summary['overall']
    cost = summary['cost']
    lines = [
        f"# 分角 Skill 评测报告",
        '',
        f"- 时间：{config['started']}　耗时 {config['elapsed_s']:.0f} 秒",
        f"- Skill：`{config['skill']['name']}` v{config['skill']['version']} #{config['skill']['digest']}　参数覆盖：{config['overrides'] or '无'}",
        f"- 分角模型：`{config['model']}`　分析模型：`{config.get('analyst_model', '-')}`　评分：金标准（无 LLM 裁判）",
        f"- 用例 {len(summary['cases'])} 个 × {config['trials']} 次，成功 {o['runs']}/{o['expected_runs']}，因预算跳过 {summary['skipped_for_budget']}",
        '',
        '## 花费',
        '',
        '| 项目 | 美元 |',
        '|---|---|',
        *[f"| {name} | {value:.4f} |" for name, value in cost.items() if name != 'total'],
        f"| **合计（按每次调用回传的花费）** | **{cost['total']:.4f}** |",
    ]
    if config.get('key_delta') is not None:
        lines.append(f"| OpenRouter 账户实际扣费（前后差额） | {config['key_delta']:.4f} |")
    if config.get('key_after'):
        lines.append(f"\n账户剩余额度：${config['key_after'].get('limit_remaining', 0):.2f}")

    lines += ['', '## 指标看板', '', '| 指标 | 结果 | 目标 | 结论 |', '|---|---|---|---|']
    for key, (name, target, op) in TARGETS.items():
        value = o.get(key)
        ok = value is not None and (value >= target if op == '≥' else value <= target)
        verdict = '无数据' if value is None else ('达标' if ok else '未达标')
        shown = _num(value, 4) if key == 'cost_per_kchar' else _pct(value)
        goal = f"{op} {target}" if key == 'cost_per_kchar' else f"{op} {target:.0%}"
        lines.append(f"| {name} | {shown} | {goal} | {verdict} |")
    lines += [
        '',
        f"其他：类别准确率 {_pct(o['category_acc'])}，角色表精确率 {_pct(o['roster_precision'])}，"
        f"两两一致率 {_pct(o['pairwise_agreement'])}，角色表一致性 (Jaccard) {_pct(o['roster_jaccard'])}，"
        f"规范名一致率 {_pct(o.get('naming_consistency'))}，失败 {o.get('failed_runs', 0)} 次，结构化降级 {o.get('structured_fallbacks', 0)} 次，"
        f"每段 {_num(o['tokens_per_segment'], 0)} token，推理 token 占输出 {_pct(o['reasoning_share'])}，"
        f"平均每次 ${_num(o['cost_per_run'], 4)}、{_num(o['latency_s'], 1)} 秒。",
        '',
        '## 各用例',
        '',
        '| 用例 | 段/批 | 准确率 | 说话人 | 校对前 | 全一致率 | 角色表 召回/精确 | 选角 | 每次花费 $ | token/段 | 耗时 s |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for case_id, c in summary['cases'].items():
        if not c.get('runs'):
            lines.append(f"| {case_id} | 全部失败：{'; '.join(c.get('errors', []))[:80]} ||||||||||")
            continue
        e = c['efficiency']
        lines.append(
            f"| {case_id} | {c['segments']}/{c['chunks']} | {_ms(c['label_acc'])} | {_ms(c['speaker_acc'])} | "
            f"{_pct(c['pre_review_label_acc'])} | {_pct(c['stability'].get('all_agree_rate'))} | "
            f"{_pct(c['roster']['recall'])} / {_pct(c['roster']['precision'])} | {_pct(c['casting_acc'])} | "
            f"{_ms(e['cost'], lambda v: f'{v:.4f}')} | {_num(e['tokens_per_segment'], 0)} | "
            f"{_ms(e['latency_s'], lambda v: f'{v:.0f}')} |"
        )

    lines += ['', '## ReAct 自我修复', '',
              f"- 校对触发率 {_pct(o['review_trigger_rate'])}；累计修正 {o['review_fixed']} 处、改坏 {o['review_broken']} 处",
              f"- 校验规则：精确率 {_pct(o['validator_precision'])}（被标疑点里真错的比例），召回率 {_pct(o['validator_recall'])}（真错里被标出的比例）",
              '', '| 疑点类型 | 标出 | 其中真错 | 附修正建议 | 建议正确 |', '|---|---|---|---|---|']
    for kind, counts in sorted((o.get('validator_by_kind') or {}).items(), key=lambda kv: -kv[1]['flagged']):
        lines.append(f"| {KIND_NAMES.get(kind, kind)} | {counts['flagged']} | {counts['wrong']} | "
                     f"{counts['suggested']} | {counts['suggest_ok']} |")
    lines += ['', '| 用例 | 触发率 | 平均疑点 | 修正 | 改坏 | 校验精确率 | 校验召回率 | 校对净收益 |', '|---|---|---|---|---|---|---|---|']
    for case_id, c in summary['cases'].items():
        if not c.get('runs'):
            continue
        r, v = c['review'], c['validator']
        lines.append(f"| {case_id} | {_pct(r['trigger_rate'])} | {_num(v['flagged'], 1)} | {r['fixed']} | {r['broken']} | "
                     f"{_pct(v['precision'])} | {_pct(v['recall'])} | {_pct(r['net_gain'])} |")

    lines += ['', '## Token 效率', '', '| 用例 | 输入 | 输出 | 推理占比 | 缓存命中 | 结构化降级 | 节点 token 占比 |', '|---|---|---|---|---|---|---|']
    for case_id, c in summary['cases'].items():
        if not c.get('runs'):
            continue
        e = c['efficiency']
        shares = '，'.join(f"{n} {_pct(v)}" for n, v in sorted(e['node_token_share'].items(), key=lambda kv: -(kv[1] or 0)))
        lines.append(f"| {case_id} | {_num(e['input_tokens'], 0)} | {_num(e['output_tokens'], 0)} | {_pct(e['reasoning_share'])} | "
                     f"{_pct(e['cache_hit_share'])} | {e['structured_fallbacks']} | {shares} |")

    lines += ['', '## 主要错误（对照金标准）', '']
    for case_id, c in summary['cases'].items():
        if not c.get('runs'):
            continue
        lines.append(f"**{case_id} · {c['title']}**")
        lines += [f"- {w}" for w in c['top_wrong']] or ['- 无']
        if c['stability'].get('unstable_segments'):
            lines.append('- 不稳定片段：' + '；'.join(
                f"#{u['id']} {' vs '.join(u['variants'])}" for u in c['stability']['unstable_segments'][:6]))
        if c['stability'].get('name_variants'):
            lines.append('- 规范名漂移：' + '；'.join(f"{g}：{' / '.join(v)}" for g, v in c['stability']['name_variants'].items()))
        if c['roster']['split'] or c['roster']['extra']:
            lines.append(f"- 角色表：拆分 {c['roster']['split'] or '无'}，多余 {c['roster']['extra'] or '无'}")
        if c['casting_wrong']:
            lines.append(f"- 选角不符：{'、'.join(c['casting_wrong'])}")
        lines.append('')

    lines += ['## 分析智能体结论', '']
    if not analysis:
        lines.append('未运行。')
    elif analysis.get('error'):
        lines.append(f"失败：{analysis['error']}")
    else:
        lines += [f"**总体**：{analysis['verdict']}", '', f"**编排**：{analysis['orchestration']}", '',
                  f"**Token 效率**：{analysis['token_efficiency']}", '', '**问题**：']
        lines += [f"- {w}" for w in analysis['weaknesses']]
        lines += ['', '**Skill 修改建议**：', '', '| 位置 | 修改 | 理由 | 预期效果 |', '|---|---|---|---|']
        lines += [f"| {e['section']} | {e['change']} | {e['reason']} | {e['expected_effect']} |" for e in analysis['skill_edits']]
    return '\n'.join(lines) + '\n'


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(description='分角 Skill 评测：准确率、稳定性、ReAct 收益与 token 效率')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--analyst-model', default=DEFAULT_ANALYST)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--cases', default='', help='逗号分隔的用例 id，默认全部')
    parser.add_argument('--skill', default='', help='SKILL.md 路径，默认使用内置 Skill')
    parser.add_argument('--param', action='append', default=[], help='覆盖 Skill 参数，如 low_confidence=0.99')
    parser.add_argument('--max-cost', type=float, default=0.5, help='本次评测的美元上限')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--no-analyst', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='只估算花费')
    parser.add_argument('--out', default='', help='报告目录')
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.param:
        key, _, value = item.partition('=')
        overrides[key.strip()] = yaml.safe_load(value)
    skill = load_skill(args.skill or None, overrides)
    cases = metrics.load_cases(only=[c for c in args.cases.split(',') if c] or None)
    if not cases:
        sys.exit('没有匹配的用例')
    voices = metrics.load_voices()

    prices = fetch_prices({args.model, args.analyst_model})
    estimate = estimate_costs(cases, skill, prices, args.model, args.trials, not args.no_analyst)
    print(f"Skill {skill.label}；{len(cases)} 个用例 × {args.trials} 次；预估花费 ${estimate['total']:.3f}（上限 ${args.max_cost:.2f}）")
    for model in (args.model, args.analyst_model):
        price = prices.get(model)
        print(f"  {model}: " + (f"${price['input'] * 1e6:.2f} / ${price['output'] * 1e6:.2f} 每百万 token" if price else '未取到单价'))
    if args.dry_run:
        return
    if estimate['total'] > args.max_cost:
        print('预估超过上限，超出部分的任务会被跳过。')

    try:
        settings = resolve_llm_settings(_EnvConfig())
    except LLMConfigError as e:
        sys.exit(str(e))

    key_before = key_usage(settings.api_key) if settings.is_openrouter else None
    started = datetime.now()
    t0 = time.perf_counter()
    ledger = CostLedger(args.max_cost)
    pipeline = EvalPipeline(
        settings, skill, voices, args.model, args.analyst_model, args.trials, ledger, prices,
        estimate['per_item'], analyst=not args.no_analyst,
    )
    result = pipeline.run(cases, workers=args.workers)
    elapsed = time.perf_counter() - t0

    key_after = None
    if key_before:
        # OpenRouter 记账有延迟：轮询到差额接近按调用累计的花费，最多等 60 秒
        for _ in range(12):
            time.sleep(5)
            key_after = key_usage(settings.api_key)
            if key_after and key_after['usage'] - key_before['usage'] >= ledger.total * 0.95:
                break
    config = {
        'started': started.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_s': elapsed,
        'skill': skill.info(),
        'overrides': overrides,
        'model': args.model,
        'analyst_model': args.analyst_model,
        'trials': args.trials,
        'max_cost': args.max_cost,
        'estimate': estimate['total'],
        'prices_per_mtok': {m: {k: v * 1e6 for k, v in p.items()} for m, p in prices.items()},
        'key_before': key_before,
        'key_after': key_after,
        'key_delta': (key_after['usage'] - key_before['usage']) if key_before and key_after else None,
    }
    summary = result['summary']
    analysis = result.get('analysis', {})

    out_dir = Path(args.out) if args.out else REPORT_DIR / f"{started:%Y%m%d-%H%M%S}_{skill.digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.json').write_text(json.dumps(_jsonable({
        'config': config, 'summary': summary, 'analysis': analysis,
        'trials': result['trials'],
    }), ensure_ascii=False, indent=2), encoding='utf-8')
    markdown = render_markdown(config, summary, analysis)
    (out_dir / 'report.md').write_text(markdown, encoding='utf-8')

    o = summary['overall']
    print(f"\n准确率 {_pct(o['label_acc'])}，说话人 {_pct(o['speaker_acc'])}，全一致率 {_pct(o['all_agree_rate'])}，"
          f"选角 {_pct(o['casting_acc'])}，校对修正 {o['review_fixed']} / 改坏 {o['review_broken']}")
    print(f"花费 ${summary['cost']['total']:.4f}（{', '.join(f'{k} ${v:.4f}' for k, v in summary['cost'].items() if k != 'total')}）"
          + (f"；账户实际扣费 ${config['key_delta']:.4f}" if config['key_delta'] is not None else ''))
    print(f"报告：{out_dir / 'report.md'}")


if __name__ == '__main__':
    main()
